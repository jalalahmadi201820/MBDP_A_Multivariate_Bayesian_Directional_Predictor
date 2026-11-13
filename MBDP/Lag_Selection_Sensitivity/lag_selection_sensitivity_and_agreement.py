"""Controlled selector audit

Single-file distribution of the completed run_audit.py experiment.
The numerical routines are included verbatim from run_162_scenario_lag_grid.py;
only packaging and input/output path configuration have changed.
This packaged version has not been used to rerun the full experiment.

Install:
    python -m pip install numpy pandas scipy numba threadpoolctl
Run:
    python controlled_selector_audit.py --data-dir ./Data --output-dir ./selector_audit_results

Required input files in --data-dir:
    djia30_full_2009_2022.csv
    BTCUSD.pkl (use only a trusted pickle file)

Completed original replay: 114 cases; same raw-selector lags in 97 cases;
exactly identical test-return paths in 104 cases. These counts are observations,
not hard-coded outputs. This is a controlled replay, not an exact reconstruction
of every archived run. All lag choices precede test-return computation.

Outputs: manifest.json, selections.csv, metrics.csv, raw_identity.csv,
and per-case training returns, scores, selections and test returns.
Use a fresh output directory for each independent run: existing per-case
metrics.csv files are treated as cached results, as in the original runner.
The original experiment used two workers; this is also the default here.
"""
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


# Original controlled-replay runner (portable paths).
import sys, json, time, hashlib, warnings
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(os.environ.get('SELECTOR_AUDIT_OUTPUT', './selector_audit_results'))
DATA_DIR=Path(os.environ.get('SELECTOR_AUDIT_DATA', './Data'))
WORKERS=int(os.environ.get('SELECTOR_AUDIT_WORKERS', '2'))

def task(t):
    panel,asset,skew,raw,lags,cfg=t
    variant='maxlag_skew' if skew else 'regular'
    dest=ROOT/panel/asset/variant; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metrics.csv').exists(): return str(dest)+' cached'
    t0=time.time()
    hist=raw.loc[raw.date < pd.Timestamp(cfg['train_end'])]
    matrix=pd.concat([train_series(hist,l,skew,cfg) for l in lags],axis=1,join='inner').sort_index()
    matrix.to_csv(dest/'training_returns.csv.gz')
    x=matrix.to_numpy(); n,k=x.shape
    assert np.isfinite(x).all() and n>=40
    pnl=x.sum(axis=0); sd=x.std(axis=0,ddof=1)
    sharpe=np.divide(x.mean(axis=0)*np.sqrt(cfg['annualization']),sd,out=np.full(k,-np.inf),where=sd>0)
    diff=x-(x.sum(axis=1,keepdims=True)-x)/(k-1)
    rows=[]; scores=[]
    def choose(name,score,p=None,lam=0):
        j=int(np.argmax(score)); p=np.zeros(k) if p is None else p
        rows.append(dict(panel=panel,asset=asset,variant=variant,selector=name,index=j,lags=str(lags[j]),n_train=n,train_pnl=pnl[j],train_sharpe=sharpe[j],p_value=p[j],score=score[j],reference_mean_difference=diff[:,j].mean()))
        for a in range(k):
            scores.append(dict(selector=name,lags=str(lags[a]),train_pnl=pnl[a],train_sharpe=sharpe[a],p_value=p[a],score=score[a],selected=a==j,reference_mean_difference=diff[:,a].mean()))
    choose('raw_pnl',pnl); choose('raw_sharpe',sharpe)
    # Identical dates/support in every arm: concat-truncate reduces to the first
    # other candidate. This isolates reference construction, not cold-start rules.
    for reverse in [False,True]:
        order=list(range(k))[::(-1 if reverse else 1)]
        ps=[iid_tests(x[:,j]-x[:,next(a for a in order if a!=j)],'two-sided') for j in range(k)]
        for method in STATISTICAL_METHODS:
            p=np.array([v[method] for v in ps]); choose(('concat_reverse_' if reverse else 'concat_')+method,pnl-100*p,p,100)
    ps=[iid_tests(diff[:,j],'two-sided') for j in range(k)]
    for method in STATISTICAL_METHODS:
        p=np.array([v[method] for v in ps])
        for lam in [50,100]: choose(f'aligned_iid_{method}_{lam}',pnl-lam*p,p,lam)
    for block in [5,10,20]:
        ps=[block_bootstrap_tests(diff[:,j],block,1999,stable_seed(42,(panel,asset,variant),lags[j],block),'two-sided') for j in range(k)]
        for method in STATISTICAL_METHODS:
            p=np.array([v[method] for v in ps])
            for lam in [50,100]: choose(f'aligned_boot_{method}_L{block}_{lam}',pnl-lam*p,p,lam)
    # Lock all choices before any test return is computed.
    pd.DataFrame(rows).to_csv(dest/'selections.csv',index=False)
    pd.DataFrame(scores).to_csv(dest/'scores.csv.gz',index=False)
    paths={}; mets={}
    for j in sorted({r['index'] for r in rows}):
        out=simulate(raw.loc[raw.date < pd.Timestamp(cfg['test_end'])],lags[j],skew,cfg)
        observed=out.set_index('date').bar_pnl.reindex(matrix.index)
        assert np.allclose(observed,matrix.iloc[:,j],atol=1e-14,rtol=1e-12)
        test=out.loc[out.split=='test'].set_index('date')
        assert len(test) and test.prob_up.notna().all()
        paths[str(lags[j])]=test.bar_pnl
        mets[j]=return_metrics(test.bar_pnl,cfg['initial_capital'],cfg['annualization'])
    pd.DataFrame(paths).to_csv(dest/'test_returns.csv.gz')
    for r in rows: r.update(mets[r['index']])
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
    a,b=rows[:2]
    pd.DataFrame([dict(panel=panel,asset=asset,variant=variant,same_lags=a['lags']==b['lags'],same_test_returns=np.array_equal(paths[a['lags']],paths[b['lags']]),max_abs_test_difference=float(np.max(np.abs(paths[a['lags']]-paths[b['lags']]))))]).to_csv(dest/'raw_identity.csv',index=False)
    return f'{panel} {asset} {variant}: {n} bars, {time.time()-t0:.1f}s'

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks=[]; skips=[]; manifests={}
    for panel,prof in PROFILES.items():
        cfg=dict(prof,predict_day=1,dist='student',student_df=10.,reg=1e-6,prob_threshold=.9,commission=0.,rebase_every=128,min_train_bars=40,block_lengths=[5,10,20])
        args=SimpleNamespace(**dict(prof,n_lag_sets=100,min_lags=2,random_seed=42,date_col='Date',ticker_col='Ticker',invalid_price_policy='drop'))
        args.data_path=str(DATA_DIR/prof['data_path'])
        if not hasattr(args,'asset'): args.asset=None
        data=load_data(args); lags=generate_candidates(args)
        manifests[panel]=dict(config=cfg,lags=lags,data_sha256=hashlib.sha256(Path(args.data_path).read_bytes()).hexdigest(),cleaning=args.data_cleaning_report)
        for asset,raw in data.items():
            count,required=training_eligibility(raw,lags,cfg)
            if count<required:
                skips.append(dict(panel=panel,asset=asset,n_train=count,required=required)); continue
            for skew in [False,True]: tasks.append((panel,asset,skew,raw,lags,cfg))
    (ROOT/'manifest.json').write_text(json.dumps(dict(profiles=manifests,bootstrap_reps=1999,block_lengths=[5,10,20],alternative='two-sided',adjustment='raw',penalties=[50,100],seed=42,skipped=skips,notes='NEW CONTROLLED REPLAY, not exact reconstruction of archived grid. All arms use identical common active training timestamps. Concat arms implement original reference rule on this common support. No change to supplied execution timing. Test outcomes are diagnostic, not configuration selection.'),indent=2))
    print('TASKS',len(tasks),'SKIPS',skips,flush=True)
    # BTC first so a complete single-asset audit is available promptly.
    tasks.sort(key=lambda t:t[0]!='XGboost_GRU')
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(task,tasks): print(result,flush=True)
    for filename in ['metrics.csv','selections.csv','raw_identity.csv']:
        pd.concat([pd.read_csv(p) for p in ROOT.glob('*/*/*/'+filename)],ignore_index=True).to_csv(ROOT/filename,index=False)
    print('COMPLETE',flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Controlled selector audit')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=ROOT,
                        help='Use a fresh directory; existing per-case results are cached.')
    parser.add_argument('--workers', type=int, default=WORKERS)
    cli = parser.parse_args()
    if cli.workers < 1:
        parser.error('--workers must be positive')
    DATA_DIR = cli.data_dir.resolve()
    ROOT = cli.output_dir.resolve()
    WORKERS = cli.workers
    for profile in PROFILES.values():
        if not (DATA_DIR / profile['data_path']).is_file():
            parser.error('Missing input: ' + str(DATA_DIR / profile['data_path']))
    # Propagate configured paths to spawned workers on all supported platforms.
    os.environ['SELECTOR_AUDIT_DATA'] = str(DATA_DIR)
    os.environ['SELECTOR_AUDIT_OUTPUT'] = str(ROOT)
    os.environ['SELECTOR_AUDIT_WORKERS'] = str(WORKERS)
    main()
