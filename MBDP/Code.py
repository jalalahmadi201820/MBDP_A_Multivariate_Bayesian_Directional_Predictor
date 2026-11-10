#!/usr/bin/env python3
"""Run the manuscript's synchronized MBDP and independent trading baselines.

Single-file implementation; only the two bundled market CSV files are needed.
Example: python Code.py --verify-against reference_results.csv
All numerical returns in the canonical columns are fractions, not percentages.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time

# Keep each worker single-threaded; beta candidates can run in parallel.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
import numba
from numba import njit
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BETAS = np.array([0.800, 0.850, 0.900, 0.950, 1.000])
GAMMAS = np.arange(0.0, 1.0, 0.0025)
METHODS = ("Logistic", "LDA", "Random forest", "Momentum")
PROBABILITY_CLIP = 1e-6
WINDOWS = {
    "DRL_Ensemble": ("2015-06-01", "2016-01-01", "2020-05-08"),
    "FineRL": ("2019-07-01", "2020-07-01", "2022-03-31"),
    "BTCUSD_Daily": ("2020-01-01", "2021-07-01", "2023-03-01"),
}


def locate(filename):
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required market file not found: {path}")
    return path


def prepare(name):
    begin, test, end = WINDOWS[name]
    if name == 'BTCUSD_Daily':
        df = pd.read_csv(locate('BTCUSD.csv'), parse_dates=['date'])
        panel = df.set_index('date')[['close']].rename(columns={'close': 'BTCUSD'})
        bpy = 365.25
    else:
        df = pd.read_csv(locate('djia30_full_2009_2022.csv'), parse_dates=['Date'])
        if df.duplicated(['Date', 'Ticker']).any():
            raise ValueError('Duplicate Date/Ticker')
        panel = df.pivot(index='Date', columns='Ticker', values='Adj Close')
        bpy = 252.0
    panel = panel.sort_index().astype(float)
    if panel.index.has_duplicates:
        raise ValueError('Duplicate dates')
    train = panel.loc[(panel.index >= begin) & (panel.index < test)]
    if train.empty:
        raise ValueError('Empty training period')
    assets = train.columns[train.notna().all() & (train > 0).all()]
    prices = panel.loc[begin:end, assets]
    if prices.empty or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Empty, missing or nonpositive prices')
    start = int(prices.index.searchsorted(test))
    if not 1 <= start < len(prices):
        raise ValueError('Need training and test observations')
    y = np.log(prices).diff().fillna(0.0).to_numpy()
    return prices, y, np.expm1(y), start, bpy


@njit(cache=True)
def forecast_components(y, beta, bpy):
    rows, assets = y.shape
    mu = np.empty_like(y)
    sd = np.empty_like(y)
    for j in range(assets):
        mean, variance, degrees, scale = 0.0, 0.01, 5.0, 0.04 / bpy
        for t in range(rows):
            prior = variance
            predictive = prior + scale
            mu[t, j] = mean
            sd[t, j] = np.sqrt(predictive * degrees / (degrees - 2.0))
            error = y[t, j] - mean
            gain = prior / predictive
            mean += gain * max(-4 * np.sqrt(predictive), min(4 * np.sqrt(predictive), error))
            degrees = beta * degrees + 1.0
            nxt = max(scale + scale / degrees * (error * error / predictive - 1), 1e-10)
            variance = max(nxt / scale * (prior - gain * gain * predictive), 1e-10)
            scale = nxt
    return mu, sd


@njit(cache=True)
def choose(held, side, mu, sd, gamma, long_gate, short_gate, allow_long):
    # As in the original: gate controls entries, negative utility controls exits.
    best, direction = held, side
    cur = side * mu[held] - gamma * sd[held] if side else 0.0
    if cur < 0:
        best, direction, cur = -1, 0, 0.0
    for j in range(len(mu)):
        if allow_long and long_gate[j]:
            utility = mu[j] - gamma * sd[j]
            if utility > cur:
                best, direction, cur = j, 1, utility
        if short_gate[j]:
            utility = -mu[j] - gamma * sd[j]
            if utility > cur:
                best, direction, cur = j, -1, utility
    return best, direction


def schedule(index, start, step, name, lookback):
    origin = pd.Timestamp(WINDOWS[name][1])
    dates, k = [], 0
    while origin + pd.DateOffset(months=k * step) <= index[-1]:
        date = origin + pd.DateOffset(months=k * step)
        s = max(start, int(index.searchsorted(date)))
        a = max(1, int(index.searchsorted(date - pd.DateOffset(months=lookback))))
        if not dates or dates[-1][1] != s:
            dates.append((str(date.date()), s, a))
        k += 1
    return [(date, s, a, dates[i+1][1] if i+1 < len(dates) else len(index))
            for i, (date, s, a) in enumerate(dates)]


@njit(cache=True)
def baseline_forecast_components(y, beta, bpy):
    """Causal local-level filter. Row t holds the prior predictive state for t."""
    rows, assets = y.shape
    f = np.empty_like(y)
    cv = np.empty_like(y)
    noise = np.empty_like(y)
    scales = np.empty(rows)
    for j in range(assets):
        mean, variance, degrees, scale = 0.0, 0.01, 5.0, 0.04 / bpy
        for t in range(rows):
            prior = variance / 1.0
            predictive = prior + scale
            f[t, j] = mean
            cv[t, j] = prior
            noise[t, j] = scale
            scales[t] = degrees / (degrees - 2.0)
            error = y[t, j] - mean
            gain = prior / predictive
            mean += gain * max(-4.0 * np.sqrt(predictive), min(4.0 * np.sqrt(predictive), error))
            degrees = beta * degrees + 1.0
            next_scale = max(scale + scale / degrees * (error * error / predictive - 1.0), 1e-10)
            variance = max(next_scale / scale * (prior - gain * gain * predictive), 1e-10)
            scale = next_scale
    return f, cv, noise, scales


@njit(cache=True)
def at_horizon(f, cv, noise, scales, horizon):
    mu = horizon * f
    variance = (
        horizon * horizon * cv
        + horizon * noise
        + (1.0 - 1.0) * cv * ((horizon - 1) * horizon * (2 * horizon - 1) / 6.0)
    )
    return mu, np.sqrt(variance * scales[:, None])


def build_signals(prices, log_returns, returns, annualization, method, beta=.8):
    mean, scale = at_horizon(*baseline_forecast_components(log_returns, beta, annualization), 1)
    state = mean / scale
    information_rows = np.maximum(0, np.arange(len(prices)) - 1)
    risk_scale = scale[information_rows]
    score = np.zeros_like(returns)
    probability = np.full_like(returns, np.nan)
    fits = []
    if method == 'Momentum':
        average = pd.DataFrame(log_returns).rolling(20, min_periods=20).mean().to_numpy()
        score = np.nan_to_num(average[np.maximum(0, np.arange(len(prices)) - 2)], nan=0.)
        return dict(score=score, scale=risk_scale, probability=probability, first_valid=22,
                    fits=pd.DataFrame(), method=method)
    labels = np.zeros_like(returns, dtype=int)
    labels[:-1] = returns[1:] > 0
    quarters = prices.index.to_period('M')
    boundaries = [42] + [position for position in range(43, len(prices))
                         if quarters[position] != quarters[position - 1]]
    for number, start in enumerate(boundaries):
        stop = boundaries[number + 1] if number + 1 < len(boundaries) else len(prices)
        history = max(1, int(prices.index.searchsorted(prices.index[start] - pd.DateOffset(months=12))))
        if start - 1 - history < 40:
            raise ValueError('Classifier history has fewer than 40 observed labels')
        for asset, asset_name in enumerate(prices.columns):
            features = state[history:start - 1, asset, None]
            target = labels[history:start - 1, asset]
            if len(np.unique(target)) < 2:
                predicted = np.full(stop - start, (target.sum() + 1) / (len(target) + 2))
            else:
                if method == 'Logistic':
                    classifier = LogisticRegression(C=1., solver='lbfgs', max_iter=200)
                elif method == 'LDA':
                    classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                elif method == 'Random forest':
                    classifier = RandomForestClassifier(n_estimators=100, max_depth=3,
                                                        min_samples_leaf=10, random_state=42, n_jobs=1)
                else:
                    raise ValueError(method)
                classifier.fit(features, target)
                predicted = classifier.predict_proba(state[information_rows[start:stop], asset, None])[:, 1]
            probability[start:stop, asset] = predicted
            clipped = np.clip(predicted, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
            score[start:stop, asset] = risk_scale[start:stop, asset] * np.log(clipped / (1 - clipped))
            fits.append(dict(method=method, asset=str(asset_name), history_start=str(prices.index[history].date()),
                             fit_last_feature=str(prices.index[start - 2].date()),
                             last_label_date=str(prices.index[start - 1].date()),
                             prediction_start=str(prices.index[start].date()),
                             prediction_end=str(prices.index[stop - 1].date()), n_fit=len(target),
                             up_labels=int(target.sum()), down_labels=int(len(target) - target.sum())))
    return dict(score=score, scale=risk_scale, probability=probability, first_valid=42,
                fits=pd.DataFrame(fits), method=method)


@njit(cache=True)
def transact(cash, exposure, held, side, new, direction, fee):
    if held == new and side == direction:
        return cash, exposure, held, side, 0, 0., 0.
    trades, notional = 0, 0.
    if side:
        notional += abs(exposure)
        cash += exposure - fee * abs(exposure)
        trades += 1
        exposure = 0.
    if cash <= 0:
        return cash, 0., -1, 0, trades, notional, fee * notional
    if direction:
        entry = cash / (1 + fee)
        notional += entry
        exposure = direction * entry
        cash -= exposure + fee * entry
        trades += 1
    return cash, exposure, new, direction, trades, notional, fee * notional


@njit(cache=True)
def replay(returns, mean, scale, a, s, gamma, horizon, fee, offset=1):
    cash, exposure, held, side = 1., 0., -1, 0
    gate = np.ones(returns.shape[1], dtype=np.bool_)
    for t in range(a, s):
        if (t - a) % horizon == 0:
            new, direction = choose(held, side, mean[t-offset], scale[t-offset],
                                      gamma, gate, gate, True)
            cash, exposure, held, side, _, _, _ = transact(
                cash, exposure, held, side, new, direction, fee)
        if side:
            exposure *= 1 + returns[t, held]
        if cash + exposure - fee * abs(exposure) <= 0:
            break
    return cash + exposure - fee * abs(exposure) - 1


def select(returns, candidates, a, s, fee, offset=1, horizon=30):
    values = np.array([[replay(returns, mean, scale, a, s, g, horizon, fee, offset)
                        for g in GAMMAS] for _, mean, scale in candidates])
    ib, ig = np.unravel_index(np.argmax(values), values.shape)
    return candidates[ib], float(GAMMAS[ig]), float(values[ib, ig]), values


def fit(x, labels, ridge=1e-6, pool=True, orient=True):
    up, down = x[labels == 1], x[labels == 0]
    prior = (len(up) + 1.) / (len(x) + 2.)
    nu, nd = len(up), len(down)
    up = up if nu else x
    down = down if nd else x
    means = [float(up.mean()), float(down.mean())]
    variances = [float(np.var(c, ddof=1)) if len(c) > 1 else 0. for c in [up, down]]
    variances = [v + ridge for v in variances]
    if pool:
        variances = [sum(variances)/2] * 2
    flipped = orient and means[0] < means[1]
    if flipped:
        means.reverse()
        variances.reverse()
        prior = 1 - prior
    return dict(means=means, variances=variances, prior=prior, flipped=flipped,
                up_labels=nu, down_labels=nd, degenerate=nu == 0 or nd == 0)


def logodds(x, p, nu, dist='normal', moment=False):
    result = np.log(p['prior']) - np.log1p(-p['prior']) + np.zeros_like(np.asarray(x))
    for sign, mean, v in zip([1., -1.], p['means'], p['variances']):
        if dist == 'normal' or np.isinf(nu):
            ll = -.5 * (np.log(v) + (x - mean)**2/v)
        else:
            if moment:
                v *= (nu-2)/nu
            ll = -.5*np.log(v) - (nu+1)/2*np.log1p((x-mean)**2/(nu*v))
        result += sign * ll
    return result


def simulate(prices, y, returns, start, annualization, name, fee=.002,
             candidates=None, ridge=1e-6, pool=True, orient=True,
             dist='normal', nu_override=None, moment=False, signals=None,
             fixed_actions=None, horizon=30, borrow_rate=0.):
    if candidates is None:
        candidates = [(b, *forecast_components(y, b, annualization)) for b in BETAS]
    if signals is not None:
        if 'candidates' in signals:
            candidates = signals['candidates']
        else:
            candidates = [(np.nan, signals['score'], signals['scale'])]
    n = len(y)-start
    curve, active, turnover = np.ones(n), np.zeros(n), np.zeros(n)
    legs, holdings, sides = np.zeros(n, int), np.full(n, -1, int), np.zeros(n, int)
    cash_path, exposure_path, wealth_before = np.zeros(n), np.zeros(n), np.zeros(n)
    probabilities = np.full((n, returns.shape[1]), np.nan)
    events, blocks, fits, selections = [], [], [], []
    cash, exposure, held, side, bankrupt = 1., 0., -1, 0, False
    actions = {} if fixed_actions is None else fixed_actions
    for date, s, a, e in schedule(prices.index, start, 1, name, 12):
        if signals is not None:
            a = max(a, signals['first_valid'])
        if fixed_actions is None:
            (beta, mean, scale), gamma, train, values = select(
                returns, candidates, a, s, fee, offset=0 if signals is not None else 1,
                horizon=horizon)
        else:
            beta, mean, scale = candidates[0]
            gamma, train = np.nan, np.nan
        initial = cash + exposure
        if signals is None and fixed_actions is None:
            x = mean / scale
            nu = (np.inf if beta == 1 else 1/(1-beta)) if nu_override is None else nu_override
            local = [fit(x[a:s-1, j], (returns[a+1:s, j] > 0).astype(int), ridge, pool, orient)
                     for j in range(returns.shape[1])]
            upper = np.array([logodds(gamma, p, nu, dist, moment) for p in local])
            lower = np.array([logodds(-gamma, p, nu, dist, moment) for p in local])
            for j, p in enumerate(local):
                probabilities[s-start:e-start, j] = expit(logodds(x[s-1:e-1, j], p, nu, dist, moment))
                fits.append(dict(boundary=date, asset=str(prices.columns[j]), beta=beta,
                    gamma=gamma, fit_start=str(prices.index[a].date()),
                    fit_last_feature=str(prices.index[s-2].date()),
                    last_label_date=str(prices.index[s-1].date()),
                    n_fit=s-1-a, prior_up=p['prior'], mean_up=p['means'][0], mean_down=p['means'][1],
                    scale_up=p['variances'][0], scale_down=p['variances'][1],
                    tau=float(expit(upper[j])), tau_short=float(expit(lower[j])),
                    flipped=p['flipped'], degenerate=p['degenerate'],
                    up_labels=p['up_labels'], down_labels=p['down_labels']))
        gate = np.ones(returns.shape[1], dtype=np.bool_)
        for t in range(s, e):
            k = t-start
            wealth_before[k] = cash + exposure
            decision = (t-s) % horizon == 0
            if fixed_actions is not None:
                decision = t in actions
            if decision:
                old, oldside = held, side
                if fixed_actions is not None:
                    new, direction = actions[t]
                elif signals is not None:
                    new, direction = choose(held, side, mean[t], scale[t], gamma, gate, gate, True)
                else:
                    lo = np.array([logodds(x[t-1, j], p, nu, dist, moment) for j, p in enumerate(local)])
                    new, direction = choose(held, side, mean[t-1], scale[t-1], gamma,
                                              lo > upper, lo < lower, True)
                before, old_exp, old_cash = cash+exposure, exposure, cash
                cash, exposure, held, side, fills, notional, charge = transact(
                    cash, exposure, held, side, new, direction, fee)
                if fills:
                    legs[k] += fills
                    turnover[k] += notional / before
                    events.append(dict(date=str(prices.index[t].date()), row=t, old=old,
                        old_side=oldside, new=held, new_side=side, cash_before=old_cash,
                        exposure_before=old_exp, cash_after=cash, exposure_after=exposure,
                        fee_paid=charge, traded_notional=notional, turnover=notional/before,
                        legs=fills, terminal=False))
                actions[t] = (held, side)
            holdings[k], sides[k], active[k] = held, side, side != 0
            if side:
                exposure *= 1 + returns[t, held]
            if side < 0:
                cash -= borrow_rate / annualization * abs(exposure)
            if cash + exposure - fee*abs(exposure) <= 0:
                bankrupt = True
                raise ValueError(f'Insolvent strategy: {name}, {fee}, {dist}')
            curve[k], cash_path[k], exposure_path[k] = cash+exposure, cash, exposure
        blocks.append(dict(boundary=date, beta=beta, gamma=gamma, train_score=train,
                           start=str(prices.index[s].date()), end=str(prices.index[e-1].date()),
                           replay_start=str(prices.index[a].date()),
                           replay_end=str(prices.index[s-1].date()),
                           return_=curve[e-start-1]/initial-1))
    if side:
        before, old_cash, old_exp, old, oldside = cash+exposure, cash, exposure, held, side
        cash, exposure, held, side, count, notional, charge = transact(cash, exposure, held, side, -1, 0, fee)
        curve[-1], cash_path[-1], exposure_path[-1] = cash, cash, 0.
        legs[-1] += count
        turnover[-1] += notional/before
        events.append(dict(date=str(prices.index[-1].date()), row=len(y)-1, old=old, old_side=oldside,
            new=-1, new_side=0, cash_before=old_cash, exposure_before=old_exp,
            cash_after=cash, exposure_after=0., fee_paid=charge, traded_notional=notional,
            turnover=notional/before, legs=count, terminal=True))
        # Include the final liquidation fee in the last forward-block return.
        blocks[-1]['return_'] = (1+blocks[-1]['return_']) * cash/before - 1
    outcome = dict(curve=curve, returns=curve/np.r_[1., curve[:-1]]-1, exposure=active,
        turnover=turnover, legs=legs, holdings=holdings, sides=sides, cash=cash_path,
        signed_notional=exposure_path, wealth_before=wealth_before, actions=actions,
        events=pd.DataFrame(events), blocks=pd.DataFrame(blocks), fit=pd.DataFrame(fits),
        candidates=pd.DataFrame(selections), probs=probabilities, bankrupt=bankrupt)
    validate(outcome, prices, returns, start, fee, borrow_rate, annualization)
    return outcome


def validate(z, prices, returns, start, fee, borrow_rate=0., annualization=252.):
    """Independently reconstruct with actual fixed units and price levels."""
    cash, units, held, cursor = 1., 0., -1, 0
    events = z['events'].to_dict('records')
    reconstructed = []
    for t in range(start, len(prices)):
        while cursor < len(events) and events[cursor]['row'] == t and not events[cursor]['terminal']:
            event = events[cursor]
            if units:
                value = units * prices.iloc[t-1, held]
                cash += value - fee*abs(value)
            units, held = 0., int(event['new'])
            if event['new_side']:
                value = event['new_side']*cash/(1+fee)
                units = value/prices.iloc[t-1, held]
                cash -= value + fee*abs(value)
            cursor += 1
        value = units*prices.iloc[t, held] if units else 0.
        if units < 0:
            cash -= borrow_rate / annualization * abs(value)
        if cursor < len(events) and events[cursor]['row'] == t and events[cursor]['terminal']:
            cash += value-fee*abs(value)
            value, units = 0., 0.
            cursor += 1
        reconstructed.append(cash+value)
    error = float(np.max(np.abs(np.array(reconstructed)-z['curve'])))
    if error > 1e-9:
        raise AssertionError(f'Fixed-unit reconstruction failed: {error}')
    if z['events'].shape[0] and not np.isclose(z['events'].fee_paid.sum(),
                                             fee*z['events'].traded_notional.sum()):
        raise AssertionError('Fee ledger mismatch')
    z['reconstruction_error'] = error


def bh(prices,start,fee):
    q=prices.to_numpy()[start:]/prices.to_numpy()[start-1]
    vals=q.mean(axis=1)/(1+fee);vals[-1]*=1-fee
    returns=vals/np.r_[1.,vals[:-1]]-1
    turn=np.zeros(len(vals));turn[0]=1/(1+fee);turn[-1]+=1.
    legs=np.zeros(len(vals),int);legs[0]=legs[-1]=prices.shape[1]
    return dict(curve=vals,returns=returns,exposure=np.ones(len(vals)),turnover=turn,legs=legs)


def metrics(z,bpy):
    rr=z['returns'];vol=np.std(rr,ddof=1)*np.sqrt(bpy)
    peak=np.maximum.accumulate(np.r_[1.,z['curve']])[1:]
    return dict(total_return=z['curve'][-1]-1,cagr=z['curve'][-1]**(bpy/len(rr))-1,
        sharpe=np.mean(rr)/np.std(rr,ddof=1)*np.sqrt(bpy) if vol else 0.,volatility=vol,
        mdd=-np.min(z['curve']/peak-1),turnover=float(z['turnover'].sum()),trades=int(z['legs'].sum()),exposure=z['exposure'].mean())


def build_beta_candidate(payload):
    """Fit one baseline's causal monthly signal history for one beta."""
    prices, y, returns, annualization, method, beta = payload
    signal = build_signals(prices, y, returns, annualization, method, beta)
    return beta, signal


def baseline_signals(prices, y, returns, annualization, method, executor=None):
    tasks = [(prices, y, returns, annualization, method, float(beta)) for beta in BETAS]
    outcomes = map(build_beta_candidate, tasks) if executor is None else executor.map(build_beta_candidate, tasks)
    candidates, records, starts = [], [], []
    for beta, signal in outcomes:
        candidates.append((beta, signal["score"], signal["scale"]))
        starts.append(signal["first_valid"])
        if not signal["fits"].empty:
            records.append(signal["fits"].assign(beta=beta))
        print(f"  {method}: beta={beta:.2f} causal fits ready", flush=True)
    return {
        "candidates": candidates,
        "first_valid": max(starts),
        "method": method,
        "fits": pd.concat(records, ignore_index=True) if records else pd.DataFrame(),
    }


def result_row(panel, method, fee, outcome, annualization):
    row = dict(panel=panel, h=0 if method == "Buy and hold" else 30,
               fee=fee, method=method, **metrics(outcome, annualization))
    row.update(
        fee_bps=fee * 10000,
        cumulative_return_pct=100 * row["total_return"],
        annual_return_pct=100 * row["cagr"],
        max_drawdown_pct=100 * row["mdd"],
        annual_volatility_pct=100 * row["volatility"],
        exposure_pct=100 * row["exposure"],
    )
    if "sides" in outcome:
        events = outcome["events"]
        row.update(long_days=int((outcome["sides"] == 1).sum()),
                   short_days=int((outcome["sides"] == -1).sum()),
                   cash_days=int((outcome["sides"] == 0).sum()),
                   long_entries=int(((events.new_side == 1) & (~events.terminal)).sum()) if not events.empty else 0,
                   short_entries=int(((events.new_side == -1) & (~events.terminal)).sum()) if not events.empty else 0,
                   ledger_max_error=outcome["reconstruction_error"])
    return row


def save_results(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def save_audit(directory, panel, method, fee, prices, start, outcome):
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{panel}_{method.replace(' ', '_')}_{fee * 10000:g}bp"
    daily = pd.DataFrame({"date": prices.index[start:].strftime("%Y-%m-%d"),
                          "wealth": outcome["curve"], "return": outcome["returns"]})
    for source, target in [("holdings", "asset_index"), ("sides", "side"),
                           ("cash", "cash"), ("signed_notional", "signed_notional")]:
        if source in outcome:
            daily[target] = outcome[source]
    daily.to_csv(directory / (stem + "_daily.csv"), index=False)
    for key in ("blocks", "events", "fit"):
        if key in outcome and not outcome[key].empty:
            outcome[key].to_csv(directory / (stem + "_" + key + ".csv"), index=False)


def verify_results(actual, path):
    """Compare freshly computed metrics with the packaged manuscript export."""
    reference = pd.read_csv(path)
    keys = ["panel", "method", "fee"]
    columns = ["total_return", "cagr", "sharpe", "volatility", "mdd", "turnover", "trades", "exposure"]
    if reference.duplicated(keys).any() or actual.duplicated(keys).any():
        raise AssertionError("Duplicate strategy/market/fee result")
    joined = actual.merge(reference, on=keys, suffixes=("_actual", "_reference"), validate="one_to_one")
    if len(joined) != len(actual):
        raise AssertionError("Some computed rows have no matching reference result")
    maximum = 0.0
    for column in columns:
        a = joined[column + "_actual"].to_numpy()
        b = joined[column + "_reference"].to_numpy()
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, err_msg=column)
        maximum = max(maximum, float(np.max(np.abs(a-b))))
    print(f"Verified {len(joined)} rows against {path.name}; maximum metric error {maximum:.3g}", flush=True)
    return {"rows": len(joined), "maximum_metric_error": maximum, "tolerance": 1e-9}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing the two bundled market CSV files")
    parser.add_argument("--output", type=Path, default=ROOT / "results.csv",
                        help="Consolidated results CSV, updated after each completed run")
    parser.add_argument("--markets", nargs="+", choices=list(WINDOWS), default=list(WINDOWS),
                        help="Market panels to recompute (default: all three)")
    parser.add_argument("--fees-bps", nargs="+", type=float, default=[5, 10, 20],
                        help="Per-leg transaction fees in basis points (default: 5 10 20)")
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1),
                        help="Parallel beta-fit workers; use 1 for serial execution")
    parser.add_argument("--audit-dir", type=Path,
                        help="Optional daily wealth, trade and selected-parameter CSV directory")
    parser.add_argument("--verify-against", type=Path,
                        help="Optional reference CSV; used only after recomputing results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if any(not np.isfinite(fee) or fee < 0 or fee >= 10000 for fee in args.fees_bps):
        parser.error("Fees must be finite and in the range [0, 10000) basis points")
    if len(set(args.markets)) != len(args.markets) or len(set(args.fees_bps)) != len(args.fees_bps):
        parser.error("Market and fee lists must not contain duplicates")
    if args.verify_against is not None and args.output.resolve() == args.verify_against.resolve():
        parser.error("The output and reference CSV must be different files")
    return args


def main():
    global DATA_DIR
    args = parse_args()
    DATA_DIR = args.data_dir.resolve()
    began = time.perf_counter()
    rows, datasets = [], []
    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers,
                                       mp_context=multiprocessing.get_context("spawn"))
    try:
        for panel in args.markets:
            prices, y, returns, start, annualization = prepare(panel)
            datasets.append(dict(panel=panel, history_start=str(prices.index[0].date()),
                                 test_start=str(prices.index[start].date()),
                                 test_end=str(prices.index[-1].date()), assets=len(prices.columns),
                                 test_bars=len(prices)-start, bars_per_year=annualization))
            print(f"\n{panel}: {len(prices.columns)} assets, {len(prices)-start} test bars", flush=True)
            states = [(float(beta), *forecast_components(y, beta, annualization)) for beta in BETAS]
            for beta, mean, scale in states:
                prefix_mean, prefix_scale = forecast_components(y[:start], beta, annualization)
                np.testing.assert_array_equal(mean[:start], prefix_mean)
                np.testing.assert_array_equal(scale[:start], prefix_scale)
            for method in ("MBDP", "Buy and hold", *METHODS):
                signals = None
                if method in METHODS:
                    print(f"Fitting {method} independently for all five betas...", flush=True)
                    signals = baseline_signals(prices, y, returns, annualization, method, executor)
                    if args.audit_dir is not None and not signals["fits"].empty:
                        args.audit_dir.mkdir(parents=True, exist_ok=True)
                        signals["fits"].to_csv(args.audit_dir / f"{panel}_{method.replace(' ', '_')}_all_beta_fits.csv", index=False)
                for fee_bps in args.fees_bps:
                    fee = fee_bps / 10000
                    if method == "Buy and hold":
                        outcome = bh(prices, start, fee)
                    else:
                        outcome = simulate(prices, y, returns, start, annualization, panel,
                                           fee=fee, candidates=states, signals=signals)
                    row = result_row(panel, method, fee, outcome, annualization)
                    rows.append(row)
                    save_results(rows, args.output)
                    save_audit(args.audit_dir, panel, method, fee, prices, start, outcome)
                    print(f"  {method:15s} fee={fee_bps:g} bp  return={row['cumulative_return_pct']:8.2f}%  Sharpe={row['sharpe']:7.3f}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    checks = None
    if args.verify_against is not None:
        checks = verify_results(pd.DataFrame(rows), args.verify_against)
    manifest = dict(
        implementation="Single-file synchronized MBDP and independent trading baselines",
        beta_grid=BETAS.tolist(), gamma_start=0.0, gamma_step=0.0025, gamma_stop_exclusive=1.0,
        candidate_pairs=len(BETAS)*len(GAMMAS), lookback_months=12, step_months=1,
        rebalance_bars=30, rebalance_reset="each monthly forward boundary",
        historical_replay="30 bars from trailing-window origin, no interior monthly resets",
        selection_criterion="net terminal utility-only historical replay return",
        fee_bps=args.fees_bps, datasets=datasets, rows=len(rows),
        python=platform.python_version(),
        versions={"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                  "scikit-learn": sklearn.__version__, "numba": numba.__version__},
        sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [Path(__file__).resolve(), locate("BTCUSD.csv"), locate("djia30_full_2009_2022.csv")]},
        elapsed_seconds=time.perf_counter()-began, reference_check=checks,
    )
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(rows)} results to {args.output.resolve()}", flush=True)
    print(f"Elapsed: {manifest['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
