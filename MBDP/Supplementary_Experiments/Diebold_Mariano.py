#!/usr/bin/env python3
from __future__ import annotations

import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_key] = "1"

import argparse
import math
import multiprocessing as mp
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from numba import njit
from scipy.stats import skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent

POLICIES = ("MBDP", "LDA", "LogisticRegression")
COMMISSION = 0.001
BOOT_SEED = 20261116


@dataclass(frozen=True)
class Profile:
    name: str
    asset: str
    window: int
    train_bars: int
    test_bars: int
    annualization: int
    min_lag: int
    max_lag: int
    max_count: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


PROFILES = {
    "DRL": Profile(
        "DRL", "stocks", 250, 504, 126, 252, 1, 30, 2,
        "2015-06-01", "2015-12-31", "2016-01-01", "2020-05-08",
    ),
    "FineRL": Profile(
        "FineRL", "stocks", 250, 504, 126, 252, 5, 50, 5,
        "2019-07-01", "2020-07-01", "2020-07-01", "2022-03-31",
    ),
    "BTC": Profile(
        "BTC", "btc", 80, 2190, 540, 2190, 1, 30, 2,
        "2020-01-01", "2021-07-01", "2021-07-01", "2023-03-01",
    ),
}


def load_panels(stock_path, btc_path):
    raw = pd.read_csv(stock_path)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    panel = raw.pivot(
        index="Date", columns="Ticker", values="Adj Close"
    ).sort_index()

    valid = panel.notna().all() & (panel > 0).all() & np.isfinite(panel).all()
    included = panel.columns[valid].tolist()
    complete = panel[included]
    stocks = {str(c): complete[c].to_numpy(dtype=float) for c in included}

    btc = pd.read_pickle(btc_path)
    p = btc["close"].to_numpy(dtype=float)

    return {
        "stocks": (complete.index.to_numpy(), stocks),
        "btc": (btc.index.to_numpy(), {"BTCUSD": p}),
    }


def lag_candidates(profile, count=100):
    rng = random.Random(42)
    found = set()
    while len(found) < count:
        k = rng.randint(2, profile.max_count)
        lags = rng.sample(range(profile.min_lag, profile.max_lag + 1), k)
        found.add(tuple(sorted(lags)))
    return sorted(found)


def lag_ratios(prices, lags):
    prices = np.asarray(prices, dtype=float)
    out = np.full((len(prices), len(lags)), np.nan)
    for j, lag in enumerate(lags):
        out[lag:, j] = prices[lag:] / prices[:-lag]
    return np.ascontiguousarray(out)


def build_features(prices, lags):
    x = lag_ratios(prices, lags)
    max_lag = max(lags)
    windows = np.lib.stride_tricks.sliding_window_view(prices, max_lag + 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sk = skew(windows, axis=1, bias=False)

    sk[~np.isfinite(sk)] = 0.0
    mb = np.full((len(prices), len(lags) + 1), np.nan)
    mb[:, :-1] = x
    mb[max_lag:, -1] = sk

    y = np.full(len(prices), -1, dtype=np.int64)
    y[:-1] = (prices[1:] > prices[:-1]).astype(np.int64)
    return x, np.ascontiguousarray(mb), y


@njit(cache=True)
def _class_log_density(X, y, start, end, x, cls, reg, student, nu):
    d = X.shape[1]
    count = 0
    total = 0

    for i in range(start, end):
        if np.isfinite(X[i]).all():
            total += 1
            if y[i] == cls:
                count += 1

    fallback = count == 0
    used = total if fallback else count
    mean = np.zeros(d)

    for i in range(start, end):
        if np.isfinite(X[i]).all() and (fallback or y[i] == cls):
            for a in range(d):
                mean[a] += X[i, a]
    mean /= used

    cov = np.zeros((d, d))
    if used > 1:
        for i in range(start, end):
            if np.isfinite(X[i]).all() and (fallback or y[i] == cls):
                for a in range(d):
                    for b in range(d):
                        cov[a, b] += (
                            (X[i, a] - mean[a]) * (X[i, b] - mean[b])
                        )
        cov /= used - 1

    for a in range(d):
        cov[a, a] += reg

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        for a in range(d):
            cov[a, a] += 1e-5
        sign, logdet = np.linalg.slogdet(cov)

    diff = x - mean
    quad = diff @ np.linalg.pinv(cov, rcond=1e-15) @ diff

    if student:
        density = (
            math.lgamma((nu + d) / 2)
            - math.lgamma(nu / 2)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log(1 + quad / nu)
        )
    else:
        density = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    return np.log(max((count + 1.0) / (total + 2.0), 1e-12)) + density


@njit(cache=True)
def mbdp_probabilities(x, y, decision_indices, window):
    out = np.empty(len(decision_indices))

    for i in range(len(decision_indices)):
        t = decision_indices[i]
        end = t - 1
        start = end - window

        lu = _class_log_density(x, y, start, end, x[t], 1, 1e-6, True, 20.0)
        ld = _class_log_density(x, y, start, end, x[t], 0, 1e-6, True, 20.0)

        m = max(lu, ld)
        eu, ed = np.exp(lu - m), np.exp(ld - m)
        total = eu + ed
        out[i] = eu / total if total > 0 and np.isfinite(total) else 0.5

    return out


def baseline_probabilities(x, y, decision_indices, window):
    out = np.empty((len(decision_indices), 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, t in enumerate(decision_indices):
            end = t - 1
            xw, yw = x[end - window:end], y[end - window:end]
            counts = np.bincount(yw, minlength=2)
            fallback = float((yw.sum() + 1) / (len(yw) + 2))

            scaler = StandardScaler().fit(xw)
            z = scaler.transform(xw)
            query = scaler.transform(x[t:t + 1])

            if counts.min() < 2:
                out[i, 0] = fallback
            else:
                lda = LinearDiscriminantAnalysis()
                lda.fit(z, yw)
                out[i, 0] = lda.predict_proba(query)[0, list(lda.classes_).index(1)]

            if counts.min() == 0:
                out[i, 1] = fallback
            else:
                lr = LogisticRegression(random_state=42)
                lr.fit(z, yw)
                out[i, 1] = lr.predict_proba(query)[0, list(lr.classes_).index(1)]

    return out


def predict_indices(prices, lags, decision_indices, window):
    decision_indices = np.ascontiguousarray(decision_indices, dtype=np.int64)
    x, mb, y = build_features(prices, lags)

    pp = np.empty((len(decision_indices), len(POLICIES)))
    pp[:, 0] = mbdp_probabilities(mb, y, decision_indices, window)
    pp[:, 1:] = baseline_probabilities(x, y, decision_indices, window)
    return pp


def folds_for(profile, protocol, dates):
    d = np.asarray(dates, dtype="datetime64[ns]")
    n = len(d)
    warmup = profile.max_lag + profile.window + 2
    start = warmup + profile.train_bars
    folds = []

    for origin in range(start, n, profile.test_bars):
        test = np.arange(origin, min(origin + profile.test_bars, n))
        if len(test) < 2:
            break
        train = np.arange(origin - profile.train_bars - 1, origin - 1)
        folds.append(dict(fold=len(folds), train=train, test=test))

    return folds


def candidate_signals(prices, candidates, folds, window):
    """Cache rolling signals; each selector scores only its own train bars."""
    dates = np.unique(np.concatenate([f["train"] for f in folds]))
    history = prices[:dates[-1] + 2]
    signals = np.zeros((len(candidates), len(history)), dtype=np.int8)

    for j, lags in enumerate(candidates):
        _, features, labels = build_features(history, lags)
        prob = mbdp_probabilities(features, labels, dates, window)
        signals[j, dates] = probability_signals(prob, "MBDP")

    return signals


def wilcoxon_selector(prices, train, candidates, signals):
    r = prices[train + 1] / prices[train] - 1
    paths = [execute_signals(r, s[train])[0] for s in signals]
    pnl = np.array([path.sum() for path in paths])
    p = np.ones(len(paths))

    for j, path in enumerate(paths):
        delta = path - paths[1 if j == 0 else 0]
        if np.any(delta != 0):
            p[j] = st.wilcoxon(delta, alternative="two-sided").pvalue

    scores = pnl - 100.0 * p
    j = int(np.argmax(scores))
    return candidates[j], scores[j], pnl[j], p[j]


def ticker_job(profile_dict, protocol, ticker, prices, dates, output_file):
    profile = Profile(**profile_dict)
    folds = folds_for(profile, protocol, dates)
    candidates = lag_candidates(profile)
    rows = []
    started = time.monotonic()
    audit = []

    with threadpool_limits(limits=1):
        signals = candidate_signals(prices, candidates, folds, profile.window)

        for split in folds:
            train, test = split["train"], split["test"]
            chosen, score, pnl, p = wilcoxon_selector(
                prices, train, candidates, signals
            )
            audit.append(dict(
                selector="Wilcoxon_MaxLagSkew",
                ticker=ticker,
                fold=split["fold"],
                lags=",".join(map(str, chosen)),
                score=score,
                train_pnl=pnl,
                wilcoxon_p=p,
            ))

            pp = predict_indices(
                prices[:test[-1] + 1], chosen, test - 1, profile.window
            )

            row = pd.DataFrame({
                "realization_date": pd.to_datetime(dates[test]),
                "decision_date": pd.to_datetime(dates[test - 1]),
                "ticker": ticker,
                "fold": split["fold"],
                "asset_return": prices[test] / prices[test - 1] - 1,
                "label": (prices[test] > prices[test - 1]).astype(int),
                "p_MBDP": pp[:, 0],
                "p_LDA": pp[:, 1],
                "p_LogisticRegression": pp[:, 2],
            })
            rows.append(row)

    pd.concat(rows, ignore_index=True).to_csv(
        output_file, index=False, float_format="%.17g"
    )
    pd.DataFrame(audit).to_csv(
        output_file.parent.parent / "selected_lags" / output_file.name,
        index=False,
    )

    return dict(
        ticker=ticker,
        seconds=time.monotonic() - started,
        rows=sum(len(r) for r in rows),
        folds=len(folds),
    )


def probability_signals(p, model_name):
    threshold = 0.90 if model_name == "MBDP" else 0.50
    p = np.asarray(p)
    return np.where(
        p > threshold, 1, np.where((1 - p) > threshold, -1, 0)
    ).astype(np.int8)


def execute_signals(
    asset_returns, signals, commission=COMMISSION, initial_position=0, liquidate=True
):
    asset_returns = np.asarray(asset_returns, dtype=float)
    signals = np.asarray(signals, dtype=np.int8)
    positions = np.empty(len(signals), dtype=np.int8)
    turnover = np.zeros(len(signals))
    p = int(initial_position)

    for i, signal in enumerate(signals):
        new = int(signal) if signal else p
        turnover[i] = abs(new - p)
        positions[i] = new
        p = new

    if liquidate and len(signals):
        turnover[-1] += abs(p)

    net = positions * asset_returns - commission * turnover
    return net, positions, turnover


def aggregate_sleeves(sleeve_returns):
    sleeve_returns = np.asarray(sleeve_returns)
    sleeve_returns = np.clip(sleeve_returns, -0.9999, 10.0)
    wealth = np.cumprod(1 + sleeve_returns, axis=0)
    equity = wealth.mean(axis=1)
    returns = np.diff(np.r_[1.0, equity]) / np.maximum(
        np.r_[1.0, equity[:-1]], 1e-12
    )
    return returns, equity, wealth


def calc_risk_metrics(returns, annualization):
    mean_ret = np.mean(returns) * annualization
    vol = np.std(returns) * math.sqrt(annualization)
    sharpe = mean_ret / vol if vol > 1e-12 else 0.0

    wealth = np.cumprod(1.0 + np.clip(returns, -0.9999, 10.0))
    peak = np.maximum.accumulate(wealth)
    mdd = np.min((wealth - peak) / np.maximum(peak, 1e-12))
    return mean_ret, vol, sharpe, mdd


def diebold_mariano_test(r1, r2, h=1):
    d = r1 - r2
    n = len(d)
    mean_d = np.mean(d)
    gamma = np.zeros(h)

    for lag in range(h):
        if lag == 0:
            gamma[lag] = np.sum((d - mean_d) ** 2) / n
        else:
            gamma[lag] = np.sum((d[lag:] - mean_d) * (d[:-lag] - mean_d)) / n

    var_d = gamma[0] + 2 * np.sum(gamma[1:])
    if var_d <= 0:
        return 0.5

    stat = mean_d / math.sqrt(var_d / n)
    return 1.0 - st.norm.cdf(stat)


def assemble_run(run_dir, profile):
    files = sorted((run_dir / "predictions").glob("*.csv"))
    frames = [
        pd.read_csv(p, parse_dates=["realization_date", "decision_date"])
        for p in files
    ]
    dates = frames[0].realization_date
    nr, na, ns = len(dates), len(frames), len(POLICIES)

    sleeve = np.empty((nr, na, ns))
    turnover = np.empty((nr, na, ns))

    for a, frame in enumerate(frames):
        for s, model in enumerate(POLICIES):
            signal = probability_signals(frame[f"p_{model}"], model)
            net, _, trn = execute_signals(frame.asset_return, signal)
            sleeve[:, a, s] = net
            turnover[:, a, s] = trn

    portfolio_returns = np.empty((nr, ns))
    for s in range(ns):
        rp, _, _ = aggregate_sleeves(sleeve[:, :, s])
        portfolio_returns[:, s] = rp

    pd.DataFrame(
        portfolio_returns, index=dates, columns=POLICIES
    ).rename_axis("realization_date").to_csv(
        run_dir / "portfolio_returns.csv", float_format="%.17g"
    )

    avg_trn = turnover.mean(axis=(0, 1))
    pd.Series(avg_trn, index=POLICIES).to_csv(run_dir / "turnover.csv")


@njit(cache=True)
def bootstrap_kernel_metrics(returns, annualization, replicates, expected_block, seed):
    np.random.seed(seed)
    n, k = returns.shape
    out_mean = np.empty((replicates, k))
    out_sharpe = np.empty((replicates, k))

    for b in range(replicates):
        sums = np.zeros(k)
        sums_sq = np.zeros(k)
        index = np.random.randint(n)

        for i in range(n):
            if i > 0:
                if np.random.random() < 1.0 / expected_block:
                    index = np.random.randint(n)
                else:
                    index = (index + 1) % n

            for j in range(k):
                val = returns[index, j]
                sums[j] += val
                sums_sq[j] += val * val

        for j in range(k):
            m = sums[j] / n
            var = sums_sq[j] / n - m * m
            vol = np.sqrt(max(var, 0.0)) * np.sqrt(annualization)
            ann_mean = m * annualization
            out_mean[b, j] = ann_mean
            out_sharpe[b, j] = ann_mean / vol if vol > 1e-8 else 0.0

    return out_mean, out_sharpe


def infer_run(run_dir, profile, replicates=5000):
    rf = pd.read_csv(run_dir / "portfolio_returns.csv", index_col=0)
    returns = np.ascontiguousarray(rf.loc[:, list(POLICIES)].to_numpy())
    n = len(rf)
    primary_block = max(2, int(np.ceil(n ** (1 / 3))))

    boot_mean, boot_sharpe = bootstrap_kernel_metrics(
        returns, profile.annualization, replicates, primary_block, BOOT_SEED
    )

    point_mean, point_vol, point_sharpe, point_mdd = [], [], [], []
    for j in range(len(POLICIES)):
        m, v, s, mdd = calc_risk_metrics(returns[:, j], profile.annualization)
        point_mean.append(m)
        point_vol.append(v)
        point_sharpe.append(s)
        point_mdd.append(mdd)

    turnover_df = pd.read_csv(run_dir / "turnover.csv", index_col=0)
    turnovers = turnover_df.loc[list(POLICIES)].iloc[:, 0].to_numpy()

    contrast_rows = []
    for j, baseline in enumerate(POLICIES[1:], 1):
        diff_mean_boot = boot_mean[:, 0] - boot_mean[:, j]
        diff_mean = point_mean[0] - point_mean[j]
        q_mean = np.quantile(diff_mean_boot, [0.025, 0.975])

        diff_sharpe_boot = boot_sharpe[:, 0] - boot_sharpe[:, j]
        diff_sharpe = point_sharpe[0] - point_sharpe[j]
        q_sharpe = np.quantile(diff_sharpe_boot, [0.025, 0.975])

        dm_h = int(np.ceil(n ** (1 / 3)))
        dm_p = diebold_mariano_test(returns[:, 0], returns[:, j], h=dm_h)

        contrast_rows.append(dict(
            baseline=baseline,
            mbdp_mean=point_mean[0],
            base_mean=point_mean[j],
            diff_mean=diff_mean,
            ci_low_mean=q_mean[0],
            ci_high_mean=q_mean[1],
            mbdp_sharpe=point_sharpe[0],
            base_sharpe=point_sharpe[j],
            diff_sharpe=diff_sharpe,
            ci_low_sharpe=q_sharpe[0],
            ci_high_sharpe=q_sharpe[1],
            mbdp_vol=point_vol[0],
            base_vol=point_vol[j],
            mbdp_mdd=point_mdd[0],
            base_mdd=point_mdd[j],
            mbdp_trn=turnovers[0],
            base_trn=turnovers[j],
            dm_p_value=dm_p,
        ))

    return pd.DataFrame(contrast_rows)


def render_screenshot_table(output_dir):
    rows = []

    for profile_name, display in [("BTC", "Bitcoin"), ("DRL", "DRL"), ("FineRL", "FineRL")]:
        run_name = f"walkforward_{profile_name}"
        df = pd.read_csv(output_dir / run_name / "contrasts.csv")

        for baseline, model in [("LDA", "LDA"), ("LogisticRegression", "Logistic")]:
            r = df[df.baseline == baseline].iloc[0]
            rows.append(dict(
                profile=display,
                baseline=model,
                ret_mb=f"{r.mbdp_mean * 100:.1f}/{r.base_mean * 100:.1f}",
                ret_ci=f"[{r.ci_low_mean * 100:+.1f}, {r.ci_high_mean * 100:+.1f}]",
                sharpe_mb=f"{r.mbdp_sharpe:.2f}/{r.base_sharpe:.2f}",
                sharpe_ci=f"[{r.ci_low_sharpe:+.2f}, {r.ci_high_sharpe:+.2f}]",
                vol_mb=f"{r.mbdp_vol * 100:.1f}/{r.base_vol * 100:.1f}",
                mdd_mb=f"{r.mbdp_mdd * 100:.1f}/{r.base_mdd * 100:.1f}",
                trn_mb=f"{r.mbdp_trn:.2f}/{r.base_trn:.2f}",
                dm_p=f"{r.dm_p_value:.4f}",
            ))

    table = pd.DataFrame(rows)
    lines = [
        "| Profile | Baseline | Ret(M/B %) | Ret Diff CI | Shp(M/B) | Shp Diff CI | Vol(M/B %) | MDD(M/B %) | TO(M/B) | DM p-val |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in table.itertuples():
        lines.append(
            f"| {r.profile} | {r.baseline} | {r.ret_mb} | {r.ret_ci} | "
            f"{r.sharpe_mb} | {r.sharpe_ci} | {r.vol_mb} | {r.mdd_mb} | "
            f"{r.trn_mb} | {r.dm_p} |"
        )

    text = "\n".join(lines)
    (output_dir / "TABLE.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return table


def pool_ticker_job(job):
    return ticker_job(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-path", type=Path, default=HERE / "djia30_full_2009_2022.csv")
    ap.add_argument("--btc-path", type=Path, default=HERE / "BTCUSD.pkl")
    ap.add_argument("--output", type=Path, default=HERE / "output_wilcoxon")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--replicates", type=int, default=5000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    panels = load_panels(args.stock_path, args.btc_path)
    selector = "Wilcoxon_MaxLagSkew"
    full = []
    tasks = []
    root = args.output

    for name, profile in PROFILES.items():
        run_dir = root / f"walkforward_{name}"
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "selected_lags").mkdir(exist_ok=True)
        dates, tickers = panels[profile.asset]

        for ticker, prices in tickers.items():
            dest = run_dir / "predictions" / f"{ticker}.csv"
            tasks.append((asdict(profile), "walkforward", ticker, prices, dates, dest))

    print(f"Running {selector}: {len(tasks)} ticker/profile jobs", flush=True)
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(pool_ticker_job, tasks, chunksize=1)
        for i, status in enumerate(results, 1):
            print(f"{selector} {i}/{len(tasks)}: {status['ticker']}", flush=True)

    for name, profile in PROFILES.items():
        run_dir = root / f"walkforward_{name}"
        assemble_run(run_dir, profile)
        ec = infer_run(run_dir, profile, args.replicates)
        ec.to_csv(run_dir / "contrasts.csv", index=False)
        full.append(ec.assign(selector=selector, profile=name))

    pd.concat(full, ignore_index=True).to_csv(root / "all_contrasts.csv", index=False)
    table = render_screenshot_table(root)
    table.to_csv(root / "comparison.csv", index=False)
    print(f"\nCombined table: {root / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
