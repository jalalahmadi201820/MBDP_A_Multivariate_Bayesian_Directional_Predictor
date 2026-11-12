from __future__ import annotations

# Must precede numpy/scipy/numba imports: one numerical thread per Pool worker.
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
              "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import random
import sys
import time
import shutil
import traceback
from math import lgamma
from pathlib import Path
import warnings

try:
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.stats import wilcoxon, ttest_rel, skew, rankdata
    import numba
    from numba import njit
    from threadpoolctl import threadpool_limits
except ImportError as exc:
    raise SystemExit("Missing dependency. Install: python -m pip install numpy pandas scipy numba threadpoolctl\n" + str(exc))

STATISTICAL_METHODS = ("wilcoxon", "ttest")
REFERENCE_POOL = "same-asset/same-panel/same-variant leave-one-candidate-out equal-weight mean on common train dates"
SCENARIO_FIELDS = ("estimator", "method", "block_length", "p_adjustment", "penalty_mode", "lambda_value")

PROFILES = {
    "DRL_Ensemble": dict(data_path="djia30_full_2009_2022.csv", price_col="Adj Close",
        train_start="2015-06-01", train_end="2015-12-31", test_start="2016-01-01", test_end="2020-05-08",
        window_size=250, max_lags=2, min_lag_value=1, max_lag_value=30, initial_capital=10000., annualization=252),
    "Fine_rl": dict(data_path="djia30_full_2009_2022.csv", price_col="Adj Close",
        train_start="2019-07-01", train_end="2020-07-01", test_start="2020-07-01", test_end="2022-03-31",
        window_size=250, max_lags=5, min_lag_value=5, max_lag_value=50, initial_capital=10000., annualization=252),
    "XGboost_GRU": dict(data_path="BTCUSD.pkl", price_col="close", asset="BTCUSD",
        train_start="2020-01-01", train_end="2021-07-01", test_start="2021-07-01", test_end="2023-03-01",
        window_size=80, max_lags=2, min_lag_value=1, max_lag_value=30, initial_capital=1., annualization=2190),
}


@njit(cache=True, nogil=True)
def _features(prices, lags, horizon, add_skew):
    largest = np.max(lags)
    size = max(0, len(prices) - horizon - largest)
    X = np.empty((size, len(lags) + int(add_skew)), np.float64)
    indices = np.empty(size, np.int64)
    y = np.empty(size, np.int64)
    used = 0
    for t in range(largest, len(prices) - horizon):
        p0, p1 = prices[t], prices[t + horizon]
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0):
            continue
        ok = True
        for c in range(len(lags)):
            previous = prices[t - lags[c]]
            if not np.isfinite(previous) or previous <= 0:
                ok = False
                break
            X[used, c] = p0 / previous
        if not ok:
            continue
        if add_skew:
            count = largest + 1
            average = 0.
            for k in range(t - largest, t + 1):
                average += prices[k]
            average /= count
            m2, m3 = 0., 0.
            for k in range(t - largest, t + 1):
                delta = prices[k] - average
                m2 += delta * delta
                m3 += delta * delta * delta
            m2 /= count
            m3 /= count
            s = 0.
            # Match scipy.stats.skew(bias=False), including near-constant data.
            if count >= 3 and m2 > (np.finfo(np.float64).eps * average) ** 2:
                s = np.sqrt(count * (count - 1.)) / (count - 2.) * m3 / m2 ** 1.5
            X[used, len(lags)] = s if np.isfinite(s) else 0.
        indices[used] = t
        y[used] = int(p1 > p0)
        used += 1
    return X[:used].copy(), y[:used].copy(), indices[:used].copy()


def build_samples(prices, dates, lags, predict_day=1, include_maxlag_skew=False):
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    lags = np.asarray(sorted(set(lags)), dtype=np.int64)
    X, y, indices = _features(prices, lags, predict_day, include_maxlag_skew)
    result = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
    result.insert(0, "label", y)
    result.insert(0, "future_price", prices[indices + predict_day])
    result.insert(0, "today_price", prices[indices])
    result.insert(0, "date", np.asarray(dates)[indices])
    return result


@njit(cache=True, nogil=True)
def _moment_update(counts, means, scatter, x, cls, direction):
    """Welford add/remove; stable central moments, not E[x*x]-E[x]**2."""
    old = counts[cls]
    new = old + direction
    d = len(x)
    if new <= 0:
        counts[cls] = 0
        means[cls, :] = 0.
        scatter[cls, :, :] = 0.
        return
    delta = x - means[cls]
    means[cls] += direction * delta / new
    factor = direction * old / new
    for a in range(d):
        for b in range(d):
            scatter[cls, a, b] += factor * delta[a] * delta[b]
    counts[cls] = new


@njit(cache=True, nogil=True)
def _density(x, mean, cov, student, nu):
    d = len(x)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            cov = cov + 1e-5 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
        delta = x - mean
        # Explicit tolerance matches numpy.linalg.pinv in the supplied scripts.
        quad = delta @ np.linalg.pinv(cov, 1e-15) @ delta
        if student:
            return (lgamma((nu + d) / 2.) - lgamma(nu / 2.)
                    - .5 * d * np.log(nu * np.pi) - .5 * logdet
                    - .5 * (nu + d) * np.log(1. + quad / nu))
        return -.5 * (d * np.log(2. * np.pi) + logdet + quad)
    except Exception:
        return -1e12


@njit(cache=True, nogil=True)
def _rolling(X, y, prices, window, horizon, student, nu, reg, threshold, commission, rebase_every):
    n, d = X.shape
    probs = np.full(n, np.nan)
    signals = np.zeros(n, np.int8)
    counts = np.zeros(2, np.int64)
    means = np.zeros((2, d), np.float64)
    scatter = np.zeros((2, d, d), np.float64)
    first = window + horizon
    # Loader and feature wrapper guarantee finite features; no pandas inside loop.
    for j in range(first, n):
        end = j - horizon
        start = end - window
        if (j - first) % rebase_every == 0:
            counts[:] = 0
            means[:, :] = 0.
            scatter[:, :, :] = 0.
            for k in range(start, end):
                _moment_update(counts, means, scatter, X[k], y[k], 1)
        else:
            _moment_update(counts, means, scatter, X[start - 1], y[start - 1], -1)
            _moment_update(counts, means, scatter, X[end - 1], y[end - 1], 1)
        if window < max(10, d + 2):
            continue
        logs = np.empty(2, np.float64)
        for cls in range(2):
            source = cls if counts[cls] else 1 - cls
            count = counts[source]
            cov = np.zeros((d, d), np.float64)
            if count > 1:
                cov = scatter[source] / (count - 1.)
            for a in range(d):
                cov[a, a] += reg
            logs[cls] = np.log((counts[cls] + 1.) / (window + 2.)) + _density(X[j], means[source], cov, student, nu)
        maximum = max(logs[0], logs[1])
        eu, ed = np.exp(logs[1] - maximum), np.exp(logs[0] - maximum)
        total = eu + ed
        pu = eu / total if total > 0 and np.isfinite(total) else .5
        pdn = ed / total if total > 0 and np.isfinite(total) else .5
        probs[j] = pu
        signals[j] = 1 if pu > threshold else -1 if pdn > threshold else 0
    pnl = np.zeros(n, np.float64)
    positions = np.zeros(n, np.int64)
    position = 0
    for j in range(1, n):
        today, previous = prices[j], prices[j - 1]
        new = signals[j - 1] if signals[j - 1] else position
        bar = position * ((today - previous) / previous)
        if new != position:
            bar -= (1 if position == 0 or new == 0 else 2) * commission
        pnl[j], positions[j], position = bar, new, new
    return probs, signals, positions, pnl


def rolling_trade(samples, n_features, window_size=250, predict_day=1, dist="student",
                  student_df=10., reg=1e-6, prob_threshold=.9, commission=0., rebase_every=128):
    X = np.ascontiguousarray(samples[[f"x_{i}" for i in range(n_features)]].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite derived features (possible price-ratio overflow)")
    y = np.ascontiguousarray(samples.label.to_numpy(), dtype=np.int64)
    prices = np.ascontiguousarray(samples.today_price.to_numpy(), dtype=np.float64)
    probs, signals, positions, pnl = _rolling(X, y, prices, window_size, predict_day,
        dist == "student", student_df, reg, prob_threshold, commission, rebase_every)
    out = samples.copy()
    out["prob_up"] = probs
    out["signal"] = np.where(signals == 1, "BUY", np.where(signals == -1, "SELL", "HOLD"))
    out["position"], out["bar_pnl"], out["cumulative_pnl"] = positions, pnl, np.cumsum(pnl)
    return out


@njit(cache=True, nogil=True)
def _bootstrap_batch(d, bins, origins, block_length, nbins):
    """Exact tied signed ranks via counts of pre-sorted absolute-value bins.

    O(n + unique_abs_values) per draw instead of sorting n values per draw.
    RNG remains numpy Generator in the caller, preserving original resamples.
    """
    n = len(d)
    means = np.empty(len(origins), np.float64)
    ranks = np.empty(len(origins), np.float64)
    for row in range(len(origins)):
        counts = np.zeros(nbins, np.int64)
        signed_counts = np.zeros(nbins, np.int64)
        used, nonzero, total = 0, 0, 0.
        for block in range(origins.shape[1]):
            for offset in range(block_length):
                if used == n:
                    break
                index = (origins[row, block] + offset) % n
                value = d[index]
                total += value
                if value != 0.:
                    category = bins[index]
                    counts[category] += 1
                    signed_counts[category] += 1 if value > 0 else -1
                    nonzero += 1
                used += 1
        cumulative, weighted = 0, 0.
        for category in range(nbins):
            count = counts[category]
            rank = cumulative + (count + 1.) / 2.
            weighted += rank * signed_counts[category]
            cumulative += count
        means[row] = total / n
        ranks[row] = weighted / (nonzero * (nonzero + 1.) / 2.) if nonzero else 0.
    return ranks, means


def block_bootstrap_tests(d, block_length, reps, seed, alternative="two-sided"):
    d = np.ascontiguousarray(d, dtype=np.float64)
    n = len(d)
    if not 1 <= block_length <= n // 2:
        raise ValueError("Block length must be <= half the common train sample")
    if np.all(d == 0):
        return {"wilcoxon": 1., "ttest": 1.}
    unique, bins = np.unique(np.abs(d), return_inverse=True)
    bins = np.ascontiguousarray(bins, dtype=np.int64)
    observed = (float(signed_rank_stat(d)), float(d.mean()))
    exceed = [0, 0]
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block_length))
    for start in range(0, reps, 128):
        size = min(128, reps - start)
        origins = rng.integers(0, n, size=(size, nblocks))
        boot = _bootstrap_batch(d, bins, origins, block_length, len(unique))
        for i, obs in enumerate(observed):
            centered = boot[i] - obs
            tol = 1e-12 * max(abs(obs), 1e-12)
            if alternative == "greater":
                exceed[i] += int(np.sum(centered >= obs - tol))
            else:
                exceed[i] += int(np.sum(np.abs(centered) >= abs(obs) - tol))
    return dict(zip(STATISTICAL_METHODS, [(v + 1) / (reps + 1) for v in exceed]))

def split_df(df, train_start, train_end, test_start, test_end):
    """Assign samples to train, test, or unused date ranges."""
    df = df.copy()
    df["date"] = pd.to_datetime(df.date)
    df["split"] = "unused"
    df.loc[(df.date >= pd.to_datetime(train_start)) & (df.date < pd.to_datetime(train_end)), "split"] = "train"
    df.loc[(df.date >= pd.to_datetime(test_start)) & (df.date < pd.to_datetime(test_end)), "split"] = "test"
    return df

def generate_candidates(args):
    """One fixed, unique candidate universe, reused for every sensitivity setting."""
    from math import comb
    lo, hi = args.min_lag_value, args.max_lag_value
    if not (1 <= lo <= hi and 1 <= args.min_lags <= args.max_lags <= hi-lo+1):
        raise ValueError("Invalid lag bounds")
    capacity = sum(comb(hi-lo+1, k) for k in range(args.min_lags, args.max_lags+1))
    if not 2 <= args.n_lag_sets <= capacity:
        raise ValueError(f"n-lag-sets must be between 2 and {capacity}")
    rng = random.Random(args.random_seed)
    found = set()
    while len(found) < args.n_lag_sets:
        k = rng.randint(args.min_lags, args.max_lags)
        found.add(tuple(sorted(rng.sample(range(lo, hi+1), k))))
    return sorted(found)

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")

def holm_adjust(p):
    """Family-wise adjustment over ALL candidate lags within one comparison family."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    out = np.empty_like(p)
    out[order] = adjusted
    return out

def signed_rank_stat(x):
    """Normalized signed-rank statistic; zero differences contribute zero (Wilcox)."""
    x = np.asarray(x, dtype=float)
    ranks = rankdata(np.abs(x), axis=-1, method="average")
    zeros = np.sum(x == 0, axis=-1, keepdims=True)
    ranks = np.where(x == 0, 0., ranks-zeros)
    nonzero = x.shape[-1]-zeros[..., 0]
    denominator = nonzero*(nonzero+1)/2
    return np.divide(np.sum(np.sign(x)*ranks, axis=-1), denominator,
                     out=np.zeros_like(denominator, dtype=float), where=denominator > 0)

def iid_tests(d, alternative):
    if np.all(d == 0):
        return {"wilcoxon": 1., "ttest": 1.}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w = wilcoxon(d, zero_method="wilcox", alternative=alternative, method="auto").pvalue
        t = ttest_rel(d, np.zeros(len(d)), alternative=alternative).pvalue
    return {"wilcoxon": float(w) if np.isfinite(w) else 1.,
            "ttest": float(t) if np.isfinite(t) else 1.}

def simulate(raw, lags, include_skew, cfg):
    """Original rolling Bayesian model and execution timing, without changing strategy."""
    samples = build_samples(raw.price.to_numpy(), raw.date.to_numpy(), lags,
                            cfg["predict_day"], include_skew)
    if samples.empty:
        raise ValueError(f"No samples for {lags}")
    samples = split_df(samples, cfg["train_start"], cfg["train_end"],
                       cfg["test_start"], cfg["test_end"])
    return rolling_trade(samples, len(lags)+int(include_skew), cfg["window_size"],
                         cfg["predict_day"], cfg["dist"], cfg["student_df"],
                         cfg["reg"], cfg["prob_threshold"], cfg["commission"], cfg.get("rebase_every", 128))

def train_series(raw_train, lags, include_skew, cfg):
    result = simulate(raw_train, lags, include_skew, cfg)
    # Original execution earns with the previous position, itself from a lagged
    # signal: require two prior prediction slots to eliminate cold-start bars.
    ready = result.prob_up.shift(2).notna()
    part = result.loc[(result.split == "train") & ready, ["date", "bar_pnl"]]
    return part.set_index("date").bar_pnl.rename(str(lags))

def make_selection(matrix, lag_sets, cfg, identity):
    """No test data accepted here. Select afresh for every lambda/p-value setting."""
    n, k = matrix.shape
    x = matrix.to_numpy(float)
    pnl = x.sum(axis=0)
    sd = x.std(axis=0, ddof=1)
    sharpe = np.divide(x.mean(axis=0)*np.sqrt(cfg["annualization"]), sd,
                       out=np.full(k, -np.inf), where=sd > 0)
    # A predeclared TRAIN-only characteristic PnL magnitude, not chosen on test.
    scale = max(float(np.median(np.abs(pnl))), 1e-12)
    diff = x-(x.sum(axis=1, keepdims=True)-x)/(k-1)
    families = {}
    iid = [iid_tests(diff[:, j], cfg["alternative"]) for j in range(k)]
    for method in STATISTICAL_METHODS:
        families[("iid", method, 0)] = np.array([p[method] for p in iid])
    for block in cfg["block_lengths"]:
        values = [block_bootstrap_tests(diff[:, j], block, cfg["bootstrap_reps"],
                  stable_seed(cfg["random_seed"], identity, lag_sets[j], block),
                  cfg["alternative"]) for j in range(k)]
        for method in STATISTICAL_METHODS:
            families[("block_bootstrap", method, block)] = np.array([p[method] for p in values])

    selected, scores = [], []
    common = dict(n_train=n, n_candidates=k, train_first=str(matrix.index[0]),
                  train_last=str(matrix.index[-1]), pnl_scale=scale,
                  reference_pool=REFERENCE_POOL, alternative=cfg["alternative"])

    def choose(meta, value, raw_p, adjusted_p, used_p, effective_lambda):
        best = int(np.argmax(value))
        scenario = "|".join(str(meta[name]) for name in SCENARIO_FIELDS)
        records = []
        for j, lags in enumerate(lag_sets):
            records.append({**common, **meta, "scenario": scenario, "lags": str(lags),
                "train_pnl": float(pnl[j]), "train_compound_return": float(np.prod(1+x[:, j])-1),
                "train_sharpe": float(sharpe[j]), "reference_mean_difference": float(diff[:, j].mean()),
                "p_raw": float(raw_p[j]), "p_holm": float(adjusted_p[j]),
                "p_used": float(used_p[j]), "lambda_effective": effective_lambda,
                "penalty": float(effective_lambda*used_p[j]), "score": float(value[j]),
                "is_selected": j == best})
        scores.extend(records)
        selected.append(dict(records[best]))

    zeros = np.zeros(k)
    for method, value in [("raw_pnl", pnl), ("raw_sharpe", sharpe)]:
        choose(dict(estimator="none", method=method, block_length=0, p_adjustment="none",
                    penalty_mode="none", lambda_value=0.), value, zeros, zeros, zeros, 0.)
    for (estimator, method, block), raw in families.items():
        adjusted = holm_adjust(raw)
        for adjustment in cfg["p_adjustments"]:
            used = raw if adjustment == "raw" else adjusted
            for mode in cfg["penalty_modes"]:
                for lam in cfg["lambdas"]:
                    # absolute reproduces the original units; scaled is a separate
                    # relative penalty: 100 -> one characteristic train-PnL scale.
                    effective = lam if mode == "absolute" else (lam/100.)*scale
                    value = pnl-effective*used
                    choose(dict(estimator=estimator, method=method, block_length=block,
                                p_adjustment=adjustment, penalty_mode=mode, lambda_value=lam),
                           value, raw, adjusted, used, effective)
    return selected, scores, families

def return_metrics(returns, initial_capital, annualization):
    """Rebase to pre-first-bar capital: do NOT lose the first test return."""
    r = np.asarray(returns, float)
    if len(r) == 0 or not np.all(np.isfinite(r)) or np.any(r <= -1):
        raise ValueError("Empty/invalid returns or bankrupt strategy (return <= -100%)")
    curve = initial_capital*np.r_[1., np.cumprod(1+r)]
    total = curve[-1]/initial_capital-1
    annual = np.expm1(np.log1p(r).sum()*annualization/len(r))
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    dd = float((curve/np.maximum.accumulate(curve)-1).min())
    return dict(bars=len(r), equity_start=initial_capital, equity_end=curve[-1],
        return_pct=100*total, annual_return_pct=100*annual,
        annual_volatility_pct=100*sd*np.sqrt(annualization),
        sharpe_ratio=r.mean()/sd*np.sqrt(annualization) if sd > 0 else np.nan,
        max_drawdown_pct=100*dd, calmar_ratio=annual/abs(dd) if dd < 0 else np.nan,
        positive_bar_fraction=float(np.mean(r > 0)), total_pnl=float(r.sum()))

def process_asset(task):
    raw, asset, panel, include_skew, lag_sets, cfg = task
    variant = "maxlag_skew" if include_skew else "regular"
    identity = (panel, asset, variant)
    labels = dict(panel=panel, asset=asset, variant=variant)
    # HARD barrier: candidate selection cannot even read test or gap-period rows.
    raw_train = raw.loc[raw.date < pd.Timestamp(cfg["train_end"])].copy()
    series = [train_series(raw_train, lags, include_skew, cfg) for lags in lag_sets]
    matrix = pd.concat(series, axis=1, join="inner").sort_index()
    if not np.all(np.isfinite(matrix.to_numpy(float))):
        raise ValueError(f"Non-finite training returns: {identity}")
    if len(matrix) < max(cfg["min_train_bars"], 2*max(cfg["block_lengths"])):
        raise ValueError(f"{identity}: only {len(matrix)} common active train bars; supply more history or smaller blocks")
    if np.any(matrix.to_numpy() <= -1):
        raise ValueError(f"{identity}: bankrupt training strategy")
    selected, scores, _ = make_selection(matrix, lag_sets, cfg, identity)
    for row in selected + scores:
        row.update(labels)
    lag_lookup = {str(lags): lags for lags in lag_sets}
    test_cache, asset_metrics = {}, []
    # Test is reached ONLY after every choice is locked. Run once per unique
    # selected lag; reusing an identical deterministic trajectory is equivalent
    # to rerunning it for every lambda, not selecting the lag once for all lambdas.
    for key in sorted({row["lags"] for row in selected}):
        result = simulate(raw.loc[raw.date < pd.Timestamp(cfg["test_end"])], lag_lookup[key], include_skew, cfg)
        observed_train = result.set_index("date").bar_pnl.reindex(matrix.index)
        if not np.allclose(observed_train, matrix[key], atol=1e-14, rtol=1e-12):
            raise AssertionError("Train results changed when test rows were appended")
        test = result.loc[result.split == "test"].copy()
        if test.empty or test.prob_up.isna().any():
            raise ValueError(f"{identity}: empty/unready test for {key}")
        # Audit count of executed position changes, not count of BUY/SELL predictions.
        prev_position = result.position.shift(1).fillna(0)
        turnover = (result.position-prev_position).abs()
        test_cache[key] = dict(returns=test.set_index("date").bar_pnl,
                              trade_units=int(turnover.loc[test.index].sum()))
    common_test = None
    for entry in test_cache.values():
        idx = entry["returns"].index
        if common_test is None:
            common_test = idx
        elif not common_test.equals(idx):
            raise ValueError("Selected candidates have unequal test calendars; comparison refused")
    for row in selected:
        entry = test_cache[row["lags"]]
        asset_metrics.append({**row, **return_metrics(entry["returns"], cfg["initial_capital"], cfg["annualization"]),
                              "trade_units": entry["trade_units"]})
    diagnostics = dict(**labels, n_train_common=len(matrix), n_test=len(common_test),
                       n_candidates=len(lag_sets), pnl_min=float(matrix.sum().min()),
                       pnl_median=float(matrix.sum().median()), pnl_max=float(matrix.sum().max()),
                       median_abs_pnl=max(float(matrix.sum().abs().median()), 1e-12),
                       train_first=str(matrix.index[0]), train_last=str(matrix.index[-1]),
                       n_unique_selected=len(test_cache))
    return selected, scores, asset_metrics, test_cache, diagnostics

def csv_append(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)

def resolve_input_path(value):
    """Accept a CWD path, or a file next to this script (also its parent)."""
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        here = Path(__file__).resolve().parent
        for base in (here, *here.parents):
            for candidate in (base / "data" / path, base / path):
                if candidate.is_file():
                    return candidate.resolve()
    raise FileNotFoundError(f"Input file not found: {value!r}. Use --data-path /full/path/to/file")

def resolve_column(columns, requested, required=True):
    """Case-insensitive matching; never silently substitute Close for Adj Close."""
    if requested in columns:
        return requested
    matches = [c for c in columns if c.casefold() == requested.strip().casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous column {requested!r}: {matches}")
    if required:
        raise ValueError(f"Missing column {requested!r}; available columns: {list(columns)}")
    return None

def load_data(args):
    path = resolve_input_path(args.data_path)
    args.data_path = str(path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        raw = pd.read_pickle(path)  # only load your own trusted pickle files
    else:
        raw = pd.read_csv(path)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Input must contain a DataFrame")
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.columns.duplicated().any():
        raise ValueError("Duplicate input column names")
    date_col = resolve_column(raw.columns, args.date_col, required=False)
    price_col = resolve_column(raw.columns, args.price_col)
    ticker_col = resolve_column(raw.columns, args.ticker_col, required=False)
    if date_col is not None:
        dates = pd.to_datetime(raw[date_col], errors="raise", utc=True).dt.tz_localize(None).to_numpy()
    elif isinstance(raw.index, pd.DatetimeIndex):
        dates = pd.to_datetime(raw.index, utc=True).tz_localize(None).to_numpy()
    else:
        raise ValueError(f"Missing date column {args.date_col!r} or DatetimeIndex")
    if ticker_col is not None and (raw[ticker_col].isna().any() or raw[ticker_col].astype(str).str.strip().eq("").any()):
        raise ValueError("Missing ticker values")
    assets = raw[ticker_col].astype(str).str.strip().to_numpy() if ticker_col is not None else np.repeat(args.asset, len(raw))
    data = pd.DataFrame(dict(date=dates, price=pd.to_numeric(raw[price_col], errors="raise").to_numpy(), asset=assets))
    if data.date.isna().any():
        raise ValueError("Missing dates")
    if data.duplicated(["asset", "date"]).any():
        raise ValueError("Duplicate asset/date observations")
    invalid = ~np.isfinite(data.price) | (data.price <= 0)
    args.data_cleaning_report = []
    for asset, frame in data.groupby("asset", sort=True):
        bad = int(invalid.loc[frame.index].sum())
        args.data_cleaning_report.append(dict(asset=str(asset), input_rows=len(frame),
            invalid_price_rows=bad, retained_rows=len(frame)-bad,
            status="no_valid_prices" if bad == len(frame) else "retained"))
    if invalid.any():
        message = f"{int(invalid.sum())} missing/non-positive/non-finite price rows in {path.name}"
        if args.invalid_price_policy == "error":
            raise ValueError(message + "; inspect input or use --invalid-price-policy drop")
        warnings.warn(message + "; dropped without forward/back-filling. See data_cleaning_report.csv.", UserWarning)
        data = data.loc[~invalid].copy()
    if data.empty:
        raise ValueError("No valid price observations remain")
    return {str(asset): frame[["date", "price"]].sort_values("date").reset_index(drop=True)
            for asset, frame in data.groupby("asset", sort=True)}

def training_eligibility(raw, lag_sets, cfg):
    """Preflight uses only pre-train-end observations, never test performance."""
    history = raw.loc[raw.date < pd.Timestamp(cfg["train_end"])]
    # With strictly positive, finite prices, the largest lag is the latest
    # warm-up start, and its calendar is the intersection of candidate calendars.
    first = max(max(lags) for lags in lag_sets) + cfg["window_size"] + cfg["predict_day"] + 2
    stop = len(history) - cfg["predict_day"]
    active_dates = history.date.iloc[first:max(first, stop)]
    count = int((active_dates >= pd.Timestamp(cfg["train_start"])).sum())
    required = max(cfg["min_train_bars"], 2*max(cfg["block_lengths"]))
    return count, required

def validate_args(args):
    for name in ("lambdas", "block_lengths", "p_adjustments", "penalty_modes"):
        if not getattr(args, name):
            raise ValueError(f"{name} cannot be empty")
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            raise ValueError(f"Duplicate values in {name}")
    if any(not np.isfinite(x) or x < 0 for x in args.lambdas):
        raise ValueError("Lambdas must be finite and nonnegative")
    if args.bootstrap_reps < 99 or min(args.block_lengths) < 1:
        raise ValueError("Need >=99 bootstrap reps and positive block lengths")
    if args.window_size < 10 or args.predict_day < 1 or args.annualization <= 0:
        raise ValueError("Invalid window, horizon or annualization")
    if args.initial_capital <= 0 or args.portfolio_capital <= 0 or args.commission < 0:
        raise ValueError("Invalid capital/commission")
    for name in ("initial_capital", "portfolio_capital", "commission", "student_df", "reg", "prob_threshold", "annualization"):
        if not np.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if args.student_df <= 0 or args.reg <= 0:
        raise ValueError("student-df and reg must be positive")
    if args.window_size < max(10, args.max_lags + 3):
        raise ValueError("window-size must be at least max(10, max-lags + 3), including skew variant")
    if not 0.5 <= args.prob_threshold <= 1 or args.min_train_bars < 5 or args.n_workers < 1:
        raise ValueError("Invalid probability threshold/sample size/workers")

def safe_log(x, eps=1e-12):
    """Return a numerically safe natural logarithm."""
    return np.log(np.maximum(x, eps))

def regularized_covariance(X, reg=1e-6):
    """Return a finite covariance matrix with diagonal regularization."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]
    if X.shape[0] <= 1:
        return np.eye(d) * reg
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return cov + reg * np.eye(d)

def logpdf_multivariate_normal(x, mean, cov):
    """Evaluate a multivariate normal log-density robustly."""
    x, mean, cov = map(lambda a: np.asarray(a, dtype=np.float64), [x, mean, cov])
    d = len(x)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov += 1e-5 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
        diff = x - mean
        quad = diff @ np.linalg.pinv(cov) @ diff
        return float(-.5 * (d * np.log(2 * np.pi) + logdet + quad))
    except Exception:
        return -1e12

def logpdf_multivariate_student_t(x, mean, cov, df=5.):
    """Evaluate a multivariate Student-t log-density robustly."""
    x, mean, cov = map(lambda a: np.asarray(a, dtype=np.float64), [x, mean, cov])
    d = len(x)
    nu = float(df)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov += 1e-5 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
        diff = x - mean
        delta = diff @ np.linalg.pinv(cov) @ diff
        return float(
            lgamma((nu + d) / 2) - lgamma(nu / 2) - .5 * d * np.log(nu * np.pi) - .5 * logdet - .5 * (nu + d) * np.log(
                1 + delta / nu))
    except Exception:
        return -1e12

def fit_params(X_w, y_w, reg=1e-6):
    """Fit class priors, means, and regularized covariance matrices."""
    n = len(y_w)
    up = (y_w == 1).sum()
    p = {"prior_up": (up + 1) / (n + 2), "prior_down": (n - up + 1) / (n + 2), "valid": True}
    for cls, key in [(1, "up"), (0, "down")]:
        Xc = X_w[y_w == cls] if (y_w == cls).sum() > 0 else X_w
        p[f"mean_{key}"] = Xc.mean(axis=0)
        p[f"cov_{key}"] = regularized_covariance(Xc, reg)
    for k in ["mean_up", "mean_down", "cov_up", "cov_down"]:
        if not np.all(np.isfinite(p[k])):
            p["valid"] = False
    return p

def predict_proba(x, params, dist="normal", df=5.):
    """Return the predicted up and down probabilities for one sample."""
    if not params["valid"]:
        return .5, .5
    fn = logpdf_multivariate_normal if dist == "normal" else lambda a, b, c: logpdf_multivariate_student_t(a, b, c, df)
    lu = safe_log(params["prior_up"]) + fn(x, params["mean_up"], params["cov_up"])
    ld = safe_log(params["prior_down"]) + fn(x, params["mean_down"], params["cov_down"])
    m = max(lu, ld)
    eu, ed = np.exp(lu - m), np.exp(ld - m)
    s = eu + ed
    return (.5, .5) if s <= 0 or not np.isfinite(s) else (float(eu / s), float(ed / s))

def reference_build_samples(prices, dates, lags, predict_day=1, include_maxlag_skew=False):
    """Build lag features and one-step direction labels."""
    rows = []
    max_lag = max(lags)
    for t in range(max_lag, len(prices) - predict_day):
        p0, p1 = prices[t], prices[t + predict_day]
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0):
            continue
        vec = []
        ok = True
        for m in lags:
            pp = prices[t - m]
            if not (np.isfinite(pp) and pp > 0):
                ok = False
                break
            vec.append(p0 / pp)

        if ok:
            if include_maxlag_skew:
                window_prices = prices[t - max_lag: t + 1]
                if len(window_prices) >= 3:
                    sk = skew(window_prices, bias=False)
                    sk = sk if np.isfinite(sk) else 0.0
                else:
                    sk = 0.0
                vec.append(float(sk))

            row = {"date": dates[t], "today_price": p0, "future_price": p1, "label": int(p1 > p0)}
            row.update({f"x_{j}": v for j, v in enumerate(vec)})
            rows.append(row)
    return pd.DataFrame(rows)

def reference_rolling_trade(df, n_features, window_size=120, predict_day=1, dist="normal", df_student=5., reg=1e-6,
                  prob_threshold=.5, commission=.001):
    """Run the rolling classifier and return signals, PnL, and positions."""
    cols = [f"x_{i}" for i in range(n_features)]
    X = df[cols].values.astype(float)
    y = df.label.values.astype(int)
    prices = df.today_price.values.astype(float)
    n = len(df)
    probs = np.full(n, np.nan)
    signals = ["HOLD"] * n

    for j in range(n):
        end = j - predict_day
        start = end - window_size
        if start < 0 or end <= 0:
            continue
        Xw, yw = X[start:end], y[start:end]
        mask = np.all(np.isfinite(Xw), axis=1)
        Xw, yw = Xw[mask], yw[mask]
        if len(yw) < max(10, n_features + 2) or not np.all(np.isfinite(X[j])):
            continue
        pu, pd_ = predict_proba(X[j], fit_params(Xw, yw, reg), dist, df_student)
        probs[j] = pu
        signals[j] = "BUY" if pu > prob_threshold else "SELL" if pd_ > prob_threshold else "HOLD"

    pnl = np.zeros(n)
    positions = np.zeros(n, dtype=int)
    position = 0
    for j in range(1, n):
        sig = signals[j - 1]
        today, prev = prices[j], prices[j - 1]
        if not (np.isfinite(today) and np.isfinite(prev) and prev > 0):
            positions[j] = position
            continue
        ret = (today - prev) / prev
        new = 1 if sig == "BUY" else -1 if sig == "SELL" else position
        bar = ret if position == 1 else -ret if position == -1 else 0.
        if new != position:
            bar -= (1 if position == 0 or new == 0 else 2) * commission
        pnl[j] = bar
        positions[j] = new
        position = new

    out = df.copy()
    out["signal"], out["prob_up"], out["position"] = signals, probs, positions
    out["bar_pnl"], out["cumulative_pnl"] = pnl, np.cumsum(pnl)
    return out

def reference_bootstrap_tests(d, block_length, reps, seed, alternative="two-sided"):
    """Circular moving-block bootstrap of paired differences in chronological order.

    Bootstrap the signed-rank and mean statistics, recenter T* by observed T,
    and compare to H0: T=0. This is NOT iid resampling or scipy on a shuffled
    sample. Signed-rank inference requires a symmetric marginal difference
    distribution for a location interpretation; stationarity/weak dependence
    are needed for either block bootstrap. No claim of exact finite-sample p.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if not 1 <= block_length <= n//2:
        raise ValueError("Block length must be <= half the common train sample")
    if np.all(d == 0):
        return {"wilcoxon": 1., "ttest": 1.}
    observed = {"wilcoxon": float(signed_rank_stat(d)), "ttest": float(d.mean())}
    exceed = {key: 0 for key in observed}
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block_length))
    for start in range(0, reps, 128):
        size = min(128, reps-start)
        origins = rng.integers(0, n, size=(size, nblocks))
        indices = ((origins[..., None]+np.arange(block_length)) % n).reshape(size, -1)[:, :n]
        draws = d[indices]
        boot = {"wilcoxon": signed_rank_stat(draws), "ttest": draws.mean(axis=1)}
        for method, obs in observed.items():
            centered = boot[method]-obs
            tol = 1e-12*max(abs(obs), 1e-12)
            if alternative == "greater":
                exceed[method] += int(np.sum(centered >= obs-tol))
            else:
                exceed[method] += int(np.sum(np.abs(centered) >= abs(obs)-tol))
    return {key: (count+1)/(reps+1) for key, count in exceed.items()}

def audit_self_test():
    """Fast numerical invariants; not a substitute for empirical data validation."""
    np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])
    assert signed_rank_stat(np.array([0., 1., 2.])) == 1.
    assert signed_rank_stat(np.zeros(8)) == 0.
    d = np.random.default_rng(42).normal(size=120)
    p = reference_bootstrap_tests(d, 5, 199, 123)
    assert p == reference_bootstrap_tests(d, 5, 199, 123)
    assert all(1/200 <= v <= 1 for v in p.values())
    assert reference_bootstrap_tests(np.zeros(120), 5, 199, 1) == {"wilcoxon": 1., "ttest": 1.}
    m = return_metrics([-.1, .1], 100., 252)
    np.testing.assert_allclose(m["return_pct"], -1.)
    np.testing.assert_allclose(m["max_drawdown_pct"], -10.)
    cfg = dict(annualization=252, alternative="two-sided", block_lengths=[5], bootstrap_reps=99,
        random_seed=42, p_adjustments=["raw", "holm"], penalty_modes=["absolute", "scaled"], lambdas=[0., 100.])
    x = pd.DataFrame(np.random.default_rng(4).normal(0, .01, (120, 3)),
                     index=pd.date_range("2020-01-01", periods=120), columns=["(1, 2)", "(2, 3)", "(3, 4)"])
    selected, _, _ = make_selection(x, [(1,2), (2,3), (3,4)], cfg, ("test", "asset", "regular"))
    baseline = next(row["lags"] for row in selected if row["method"] == "raw_pnl")
    assert all(row["lags"] == baseline for row in selected if row["lambda_value"] == 0 and row["method"] != "raw_sharpe")
    print("Self-test passed: Holm, signed rank, bootstrap seed/bounds/zeros, first-bar metrics, lambda=0 selection.")

def run_experiment(args, pool, data_cache):
    validate_args(args)
    args.data_path = str(resolve_input_path(args.data_path))
    cache_key = (args.data_path, args.price_col, args.date_col, args.ticker_col, args.asset, args.invalid_price_policy)
    if cache_key not in data_cache:
        data_cache[cache_key] = (load_data(args), args.data_cleaning_report)
    data, args.data_cleaning_report = data_cache[cache_key]
    if args.assets:
        unknown = set(args.assets) - set(data)
        if unknown:
            raise ValueError(f"Unknown assets for {args.panel}: {sorted(unknown)}")
        data = {key: value for key, value in data.items() if key in args.assets}
    lag_sets = generate_candidates(args)
    if args.panels_file:
        args.panels_file = str(resolve_input_path(args.panels_file))
        panels = json.loads(Path(args.panels_file).read_text(encoding="utf-8"))
    else:
        panels = [dict(panel=args.panel, **{key: getattr(args, key) for key in
                   ("train_start", "train_end", "test_start", "test_end")})]
    if not isinstance(panels, list) or not panels:
        raise ValueError("panels-file must be a nonempty JSON list")
    seen, tasks = set(), []
    eligible_by_panel, eligibility_report = {}, []
    for panel in panels:
        if not isinstance(panel, dict) or not all(key in panel for key in
                ("panel", "train_start", "train_end", "test_start", "test_end")):
            raise ValueError("Each panel needs panel, train_start, train_end, test_start, test_end")
        name = str(panel["panel"])
        if name in seen:
            raise ValueError("Panel names must be unique")
        seen.add(name)
        cfg = vars(args).copy()
        cfg.update({key: panel[key] for key in ("train_start", "train_end", "test_start", "test_end")})
        ts, te, vs, ve = [pd.Timestamp(cfg[key]) for key in ("train_start", "train_end", "test_start", "test_end")]
        if any(pd.isna(date) or date.tzinfo is not None for date in (ts, te, vs, ve)):
            raise ValueError("Panel boundaries must be valid timezone-naive dates (input timestamps are normalized to UTC)")
        if not ts < te <= vs < ve:
            raise ValueError(f"{name}: need train_start < train_end <= test_start < test_end")
        eligible_by_panel[name] = []
        for asset, raw in data.items():
            count, required = training_eligibility(raw, lag_sets, cfg)
            eligible = count >= required
            eligibility_report.append(dict(panel=name, asset=asset, n_train_common=count,
                required_train_bars=required, status="included" if eligible else "insufficient_training_history"))
            if not eligible:
                message = f"{name}/{asset}: {count} common active train bars, need {required}"
                if args.insufficient_history == "error":
                    raise ValueError(message)
                warnings.warn(message + "; excluded from BOTH variants and all scenarios in this panel.", UserWarning)
                continue
            eligible_by_panel[name].append(asset)
            for skew_variant in (False, True):
                task_cfg = dict(cfg, _score_header=not tasks,
                    _score_path=str(Path(args.output_dir).resolve()/"_score_shards"/f"{len(tasks):05d}.csv.gz"))
                tasks.append((raw, asset, name, skew_variant, lag_sets, task_cfg))
        if not eligible_by_panel[name]:
            raise ValueError(f"{name}: no eligible assets; supply more training history or reduce window/lag/block sizes")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    managed = ["candidate_scores.csv.gz", "candidate_scores.csv", "selected_lags.csv", "asset_test_metrics.csv", "train_diagnostics.csv",
               "portfolio_test_metrics.csv", "portfolio_weights.csv", "portfolio_equity.csv", "lag_stability.csv",
               "run_manifest.json", "RUN_COMPLETE.txt", "ERROR.txt", "data_cleaning_report.csv", "asset_eligibility.csv"]
    if any((output/name).exists() for name in managed) and not args.overwrite:
        raise FileExistsError("Output already contains a run; use a new --output-dir or --overwrite")
    for name in managed:
        (output/name).unlink(missing_ok=True)
    spool = output/"_score_shards"
    if spool.exists():
        if not args.overwrite:
            raise FileExistsError(f"Previous score shards exist: {spool}; use --overwrite")
        shutil.rmtree(spool)
    spool.mkdir()
    pd.DataFrame(args.data_cleaning_report).to_csv(output/"data_cleaning_report.csv", index=False)
    pd.DataFrame(eligibility_report).to_csv(output/"asset_eligibility.csv", index=False)
    print(f"Starting {len(tasks)} asset/variant tasks, {len(lag_sets)} lag candidates; output: {output.resolve()}", flush=True)
    manifest = dict(eligible_assets_by_panel=eligible_by_panel,
        script=Path(__file__).name, strategy="original rolling Bayesian classifier (not file-name algorithm)",
        settings=vars(args), panels=panels, candidate_lags=lag_sets, reference_pool=REFERENCE_POOL,
        pnl_definition="sum of net fractional bar returns on common active TRAIN dates; not dollars or compounded return",
        scale_definition="max(median(abs(candidate train PnL)), 1e-12), separately per asset/panel/variant",
        scaled_formula="score = PnL - (lambda/100)*train_PnL_scale*p_used",
        bootstrap_definition="circular moving blocks of paired differences; centered bootstrap signed-rank and mean statistics",
        bootstrap_ttest_note="bootstrap mean-difference test, not studentized bootstrap-t",
        inference_limit="candidate-level p-values, not a significance test of the selected winner; Holm does not cure model selection or nonstationarity",
        test_protocol="fixed lag; original causal rolling refits may use earlier matured test labels; existing signal delay and positions carried across boundaries",
        intervals="[start, end), exclusive end; selection input truncated before train_end, dropping final predict_day labels",
        missing_asset_dates="portfolio sleeves carried forward (cash before first observation); weights fixed using aligned train compounded returns",
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, numba=numba.__version__),
        input_sha256=hashlib.sha256(Path(args.data_path).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), status="running")
    (output/"run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    groups, all_selected = {}, []
    try:
        iterator = map(score_task, tasks) if pool is None else pool.imap(score_task, tasks, chunksize=1)
        for count, (selected, scores, metrics, cache, diagnostics) in enumerate(iterator, 1):
            # Concatenate independent gzip members without decompressing/recompressing.
            with (output/"candidate_scores.csv.gz").open("ab") as dest, Path(scores).open("rb") as src:
                shutil.copyfileobj(src, dest)
            Path(scores).unlink()
            csv_append(output/"selected_lags.csv", selected)
            csv_append(output/"asset_test_metrics.csv", metrics)
            csv_append(output/"train_diagnostics.csv", [diagnostics])
            all_selected.extend(selected)
            for row in selected:
                key = (row["panel"], row["variant"], row["scenario"])
                groups.setdefault(key, []).append((row, cache[row["lags"]]["returns"]))
            print(f"[{count}/{len(tasks)}] {diagnostics['panel']} / {diagnostics['asset']} / {diagnostics['variant']}", flush=True)
        portfolio_rows, weight_rows = [], []
        portfolio_cache = {}
        for (panel, variant, scenario), members in sorted(groups.items()):
            if len(members) != len(eligible_by_panel[panel]):
                raise RuntimeError("Unbalanced asset universe across scenarios")
            meta = {key: members[0][0][key] for key in SCENARIO_FIELDS}
            meta.update(panel=panel, variant=variant, scenario=scenario)
            fingerprint = (panel, variant, tuple((row["asset"], row["lags"]) for row, _ in members))
            if fingerprint not in portfolio_cache:
                training_returns = np.array([row["train_compound_return"] for row, _ in members])
                w = np.maximum(training_returns, 0.)
                w = w/w.sum() if w.sum() > 0 else np.full(len(w), 1/len(w))
                sleeves = pd.concat([(1+r).cumprod().rename(row["asset"]) for row, r in members], axis=1).sort_index().ffill().fillna(1.)
                eq = args.portfolio_capital*(sleeves.to_numpy() @ w)
                ret = eq/np.r_[args.portfolio_capital, eq[:-1]]-1
                portfolio_cache[fingerprint] = (w, sleeves.index, eq,
                    return_metrics(ret, args.portfolio_capital, args.annualization))
            w, equity_dates, eq, portfolio_metrics = portfolio_cache[fingerprint]
            portfolio_rows.append({**meta, "n_assets": len(members), **portfolio_metrics})
            for weight, (row, _) in zip(w, members):
                weight_rows.append({**meta, "asset": row["asset"], "lags": row["lags"],
                    "train_compound_return": row["train_compound_return"], "weight": weight})
            if args.save_equity:
                csv_append(output/"portfolio_equity.csv", [{**meta, "date": str(date), "equity": value}
                           for date, value in zip(equity_dates, eq)])
        pd.DataFrame(portfolio_rows).to_csv(output/"portfolio_test_metrics.csv", index=False)
        pd.DataFrame(weight_rows).to_csv(output/"portfolio_weights.csv", index=False)
        selection = pd.DataFrame(all_selected)
        # Baseline is max train PnL, so this works even if user omits lambda=0.
        baseline = selection.loc[selection.method == "raw_pnl", ["panel", "asset", "variant", "lags"]].rename(columns={"lags": "raw_pnl_lags"})
        stability = selection.merge(baseline, on=["panel", "asset", "variant"], validate="many_to_one")
        stability["changed_vs_raw_pnl"] = stability.lags != stability.raw_pnl_lags
        cols = ["panel", "variant", "scenario", *SCENARIO_FIELDS]
        stability.groupby(cols, dropna=False).agg(n_assets=("asset", "size"),
            n_changed=("changed_vs_raw_pnl", "sum"), changed_fraction=("changed_vs_raw_pnl", "mean")).reset_index().to_csv(output/"lag_stability.csv", index=False)
        spool.rmdir()
        manifest["status"] = "complete"
        (output/"run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (output/"RUN_COMPLETE.txt").write_text("All scenarios completed. Inspect all prespecified settings; do not tune lambda on test.\n", encoding="utf-8")
        print(f"Completed: {output.resolve()}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        (output/"run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (output/"ERROR.txt").write_text(repr(exc), encoding="utf-8")
        raise

def argument_parser():
    p = argparse.ArgumentParser(description="Run all three supplied Bayesian strategy profiles with one fast shared Pool")
    p.add_argument("--data-path", default=None)
    p.add_argument("--output-dir", default="unified_results")
    p.add_argument("--panels-file", help="JSON list: panel, train_start, train_end, test_start, test_end")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--price-col", default=None)
    p.add_argument("--ticker-col", default="Ticker")
    p.add_argument("--asset", default=None)
    for flag in ("train_start", "train_end", "test_start", "test_end"):
        p.add_argument("--"+flag.replace("_", "-"), default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--predict-day", type=int, default=1, help="Forecast horizon in BARS, not calendar days")
    p.add_argument("--dist", choices=["normal", "student"], default="student")
    p.add_argument("--student-df", type=float, default=10.)
    p.add_argument("--reg", type=float, default=1e-6)
    p.add_argument("--prob-threshold", type=float, default=.9)
    p.add_argument("--commission", type=float, default=0.)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--portfolio-capital", type=float, default=1_000_000.)
    p.add_argument("--annualization", type=int, default=None, help="Bars per year, NOT days unless input is daily")
    p.add_argument("--n-lag-sets", type=int, default=100)
    p.add_argument("--min-lags", type=int, default=2)
    p.add_argument("--max-lags", type=int, default=None)
    p.add_argument("--min-lag-value", type=int, default=None)
    p.add_argument("--max-lag-value", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0., 10., 50., 100., 200.])
    p.add_argument("--penalty-modes", choices=["absolute", "scaled"], nargs="+", default=["absolute", "scaled"])
    p.add_argument("--block-lengths", type=int, nargs="+", default=[5, 10, 20], help="Block lengths in observation bars")
    p.add_argument("--bootstrap-reps", type=int, default=1999)
    p.add_argument("--p-adjustments", choices=["raw", "holm"], nargs="+", default=["raw", "holm"])
    p.add_argument("--alternative", choices=["two-sided", "greater"], default="two-sided",
                   help="two-sided matches original; greater is a separately declared superiority analysis")
    p.add_argument("--min-train-bars", type=int, default=40)
    p.add_argument("--invalid-price-policy", choices=["drop", "error"], default="drop",
                   help="Drop invalid price rows with an audit report; never impute prices")
    p.add_argument("--insufficient-history", choices=["skip", "error"], default="skip",
                   help="Exclude assets lacking TRAIN history, consistently across all scenarios")
    p.add_argument("--n-workers", type=int, default=available_cpus(),
                   help="Shared worker count; default respects CPU affinity/cgroup quota")
    p.add_argument("--save-equity", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Replace managed outputs only, preserving other files")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="End-to-end execution check: 3 lag sets and 99 bootstrap reps; NOT research-grade inference")
    p.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--assets", nargs="+", help="Optional exact asset filter (most useful with one profile)")
    p.add_argument("--rebase-every", type=int, default=128, help="Rebuild rolling central moments every N bars")
    p.add_argument("--benchmark", action="store_true", help="Numerical equivalence tests plus warmed-kernel timing")
    p.add_argument("--start-method", choices=mp.get_all_start_methods(), default="spawn")
    return p


def available_cpus():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(np.ceil(int(quota) / int(period)))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def warmup_worker():
    threadpool_limits(limits=1)
    prices = np.exp(np.linspace(0., .1, 50))
    X, y, idx = _features(prices, np.array([1, 3], dtype=np.int64), 1, True)
    _rolling(X, y, prices[idx], 12, 1, True, 10., 1e-6, .9, 0., 128)
    block_bootstrap_tests(np.array([0., 1., -1., .5]), 1, 1, 42)


def score_task(task):
    selected, scores, metrics, cache, diagnostics = process_asset(task)
    cfg = task[-1]
    path = cfg["_score_path"]
    pd.DataFrame(scores).to_csv(path, index=False, header=cfg["_score_header"], compression="gzip")
    return selected, path, metrics, cache, diagnostics


def numerical_self_test(benchmark=False):
    audit_self_test()
    rng = np.random.default_rng(732)
    dates = pd.date_range("2010-01-01", periods=700).to_numpy()
    cases = [100*np.exp(np.cumsum(rng.normal(0, .012, len(dates)))),
             np.full(len(dates), 100.), 100*np.exp(np.arange(len(dates))*.001)]
    max_error = 0.
    checks = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for prices in cases:
            for lags, horizon, add_skew in [([1, 9], 1, False), ([5, 17, 30], 3, True), ([1], 1, True)]:
                ref = reference_build_samples(prices, dates, lags, horizon, add_skew)
                fast = build_samples(prices, dates, lags, horizon, add_skew)
                pd.testing.assert_frame_equal(fast, ref, check_dtype=False, atol=2e-10, rtol=2e-10)
                for dist in ("normal", "student"):
                    opts = dict(window_size=60, predict_day=horizon, dist=dist, reg=1e-6,
                                prob_threshold=.9, commission=.001)
                    expected = reference_rolling_trade(ref, len(lags)+int(add_skew), df_student=10., **opts)
                    actual = rolling_trade(fast, len(lags)+int(add_skew), student_df=10., **opts)
                    np.testing.assert_allclose(actual.prob_up, expected.prob_up, atol=2e-9, rtol=2e-9)
                    np.testing.assert_array_equal(actual.signal, expected.signal)
                    np.testing.assert_array_equal(actual.position, expected.position)
                    np.testing.assert_allclose(actual.bar_pnl, expected.bar_pnl, atol=1e-14, rtol=1e-12)
                    max_error = max(max_error, float(np.nanmax(np.abs(actual.prob_up-expected.prob_up))))
                    checks += 1
    for d in (rng.normal(.01, .3, 311), rng.choice([0., -.5, .5, 1.], 311), np.zeros(311)):
        for alternative in ("two-sided", "greater"):
            for block in (1, 5, 20):
                expected = reference_bootstrap_tests(d, block, 257, 42, alternative)
                actual = block_bootstrap_tests(d, block, 257, 42, alternative)
                assert expected == actual, (expected, actual)
    print(f"Numerical tests passed: {checks} rolling cases; 18 bootstrap cases; max probability error={max_error:.3g}", flush=True)
    if benchmark:
        prices = 100*np.exp(np.cumsum(rng.normal(0, .01, 2500)))
        dates = pd.date_range("2010-01-01", periods=len(prices)).to_numpy()
        samples = build_samples(prices, dates, [5, 17, 30], 1, True)
        def timing(fn, *args, **kwargs):
            values = []
            for _ in range(3):
                start = time.perf_counter()
                fn(*args, **kwargs)
                values.append(time.perf_counter()-start)
            return float(np.median(values))
        opts = dict(window_size=80, dist="student", prob_threshold=.9, commission=.001)
        rolling_trade(samples, 4, student_df=10., **opts)
        a = timing(reference_rolling_trade, samples, 4, df_student=10., **opts)
        b = timing(rolling_trade, samples, 4, student_df=10., **opts)
        print(f"Warmed rolling median (2500 bars): reference={a:.4f}s fast={b:.4f}s speedup={a/b:.2f}x")
        d = rng.normal(0, .01, 3000)
        block_bootstrap_tests(d, 10, 1999, 42)
        a = timing(reference_bootstrap_tests, d, 10, 1999, 42)
        b = timing(block_bootstrap_tests, d, 10, 1999, 42)
        print(f"Warmed bootstrap median (3000 bars, 1999 reps): reference={a:.4f}s fast={b:.4f}s speedup={a/b:.2f}x")


def main(argv=None):
    parser = argument_parser()
    base = parser.parse_args(argv)
    if base.self_test or base.benchmark:
        numerical_self_test(base.benchmark)
        return
    if base.rebase_every < 1:
        parser.error("--rebase-every must be positive")
    if len(set(base.profiles)) != len(base.profiles):
        parser.error("--profiles must not contain duplicates")
    jobs = []
    for profile in base.profiles:
        args = argparse.Namespace(**vars(base))
        for key, value in PROFILES[profile].items():
            if getattr(args, key, None) is None:
                setattr(args, key, str(Path(base.data_dir)/value) if key == "data_path" else value)
        if args.asset is None:
            args.asset = "SINGLE_ASSET"
        args.panel = profile
        args.output_dir = str(Path(base.output_dir)/profile)
        if args.smoke_test:
            args.n_lag_sets, args.bootstrap_reps = 3, 99
        validate_args(args)
        generate_candidates(args)
        jobs.append(args)
    started = time.perf_counter()
    # Compile once in parent before spawning, so workers read the disk cache.
    warmup_worker()
    print(f"Shared execution: {base.n_workers} worker(s), profiles={base.profiles}; warmup={time.perf_counter()-started:.2f}s", flush=True)
    cache = {}
    if base.n_workers == 1:
        for args in jobs:
            run_experiment(args, None, cache)
    else:
        # Exactly ONE Pool for all profiles; workers cap BLAS/Numba threads at 1.
        with mp.get_context(base.start_method).Pool(base.n_workers, initializer=warmup_worker) as pool:
            for args in jobs:
                run_experiment(args, pool, cache)
    print(f"All profiles complete in {time.perf_counter()-started:.2f}s (including warmup).", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
