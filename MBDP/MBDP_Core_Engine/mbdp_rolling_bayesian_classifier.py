from pathlib import Path
import os
# Set BEFORE importing numerical libraries; process-level parallelism only.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_env] = "1"
import argparse
import hashlib
import json
import math
from math import lgamma
import multiprocessing as mp
import platform
import random
import sys
import time
import traceback
import warnings
from contextlib import nullcontext
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon, ttest_rel, skew
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
try:
    import numba
    from numba import njit
    from threadpoolctl import threadpool_limits
except ImportError as exc:
    raise SystemExit("Install dependencies: python -m pip install numpy pandas scipy scikit-learn numba threadpoolctl") from exc

METHODS = ("raw_pnl", "raw_sharpe", "wilcoxon", "ttest")
STATISTICAL_METHODS = ("wilcoxon", "ttest")
STRATEGIES = ("MBDP", "LDA", "LogisticRegression", "BuyHold", "TSMomentum")
VARIANTS = ("regular", "maxlag_skew")
SUMMARY_COLUMNS = (
    "method", "variant", "lags", "equity_start", "equity_end",
    "return_pct", "annual_return_pct", "annual_volatility_pct",
    "sharpe_ratio", "calmar_ratio", "max_drawdown_pct", "win_rate", "trades",
)

# Reference-compatible reporting, validation and statistical definitions.
def baseline_probability(model, Xw, yw, x, momentum_features):
    if model == "BuyHold":
        return 1.0, 0.0
    if model == "TSMomentum":
        value = float(np.log(x[:momentum_features]).mean())
        return (1.0, 0.0) if value > 0 else (0.0, 1.0) if value < 0 else (0.5, 0.5)
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(yw)) < 2 or (model == "LDA" and np.bincount(yw, minlength=2).min() < 2):
        p = float((yw.sum() + 1) / (len(yw) + 2))
        return p, 1 - p
    scaler = StandardScaler().fit(Xw)
    estimator = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                 if model == "LDA" else LogisticRegression(max_iter=2000, random_state=42))
    estimator.fit(scaler.transform(Xw), yw)
    p = float(estimator.predict_proba(scaler.transform(x.reshape(1, -1)))[0, list(estimator.classes_).index(1)])
    return p, 1 - p

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

def reference_rolling_trade(df, n_features, window_size=120, predict_day=1, dist="normal", df_student=5., reg=1e-6,
                  prob_threshold=.5, commission=.001, model="MBDP", momentum_features=None):
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
        if model == "MBDP":
            pu, pd_ = predict_proba(X[j], fit_params(Xw, yw, reg), dist, df_student)
        else:
            pu, pd_ = baseline_probability(model, Xw, yw, X[j], momentum_features or n_features)
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

def split_df(df, train_start, train_end, test_start, test_end):
    """Assign samples to train, test, or unused date ranges."""
    df = df.copy()
    df["date"] = pd.to_datetime(df.date)
    df["split"] = "unused"
    df.loc[(df.date >= pd.to_datetime(train_start)) & (df.date < pd.to_datetime(train_end)), "split"] = "train"
    df.loc[(df.date >= pd.to_datetime(test_start)) & (df.date < pd.to_datetime(test_end)), "split"] = "test"
    return df

def generate_random_lag_sets(n_sets=20, min_lags=2, max_lags=3, min_lag_val=7, max_lag_val=100):
    """Generate deterministic unique lag combinations."""
    random.seed(42)
    lag_sets = []
    attempts = 0
    max_attempts = n_sets * 10
    while len(lag_sets) < n_sets and attempts < max_attempts:
        attempts += 1
        n_lags = random.randint(min_lags, max_lags)
        lags = sorted(random.sample(range(min_lag_val, max_lag_val + 1), n_lags))
        if lags not in lag_sets:
            lag_sets.append(tuple(lags))
    return lag_sets

def compute_max_drawdown(equity):
    """Return the maximum drawdown as a negative fraction."""
    values = pd.Series(equity, dtype=float).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if values.empty:
        return np.nan
    return float((values / values.cummax() - 1.0).min())

def compute_portfolio_metrics(results_list, portfolio, annualization=252):
    """Return test-equity metrics for one method and one variant."""
    if portfolio is None or portfolio.empty:
        return {}

    equity = pd.Series(portfolio["equity"], dtype=float).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if equity.empty or equity.iloc[0] <= 0:
        return {}

    returns = equity.pct_change().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    periods = max(len(returns), 1)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    annual_return = (
        equity.iloc[-1] / equity.iloc[0]
    ) ** (annualization / periods) - 1.0
    volatility = returns.std(ddof=1) if len(returns) > 1 else np.nan
    annual_volatility = (
        volatility * np.sqrt(annualization)
        if np.isfinite(volatility)
        else np.nan
    )
    sharpe = (
        returns.mean() / volatility * np.sqrt(annualization)
        if np.isfinite(volatility) and volatility > 0
        else np.nan
    )
    max_drawdown = compute_max_drawdown(equity)
    calmar = (
        annual_return / abs(max_drawdown)
        if np.isfinite(max_drawdown) and max_drawdown < 0
        else np.nan
    )

    trades = 0
    wins = 0
    for result in results_list:
        test = result[result["split"] == "test"]
        buys = test[test["signal"] == "BUY"]
        sells = test[test["signal"] == "SELL"]
        trades += len(buys) + len(sells)
        wins += int(
            (buys["future_price"] > buys["today_price"]).sum()
            + (sells["future_price"] < sells["today_price"]).sum()
        )

    return {
        "equity_start": float(equity.iloc[0]),
        "equity_end": float(equity.iloc[-1]),
        "return_pct": float(total_return * 100.0),
        "annual_return_pct": float(annual_return * 100.0),
        "annual_volatility_pct": (
            float(annual_volatility * 100.0)
            if np.isfinite(annual_volatility)
            else np.nan
        ),
        "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else np.nan,
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "win_rate": float(wins / trades) if trades else np.nan,
        "trades": int(trades),
    }

def write_metrics_summary(rows, output_path):
    """Write the consolidated test metrics CSV without console output."""
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(output_path, index=False)

def stationary_indices(rng, n, block_length):
    restart = rng.random(n) < 1.0 / block_length
    restart[0] = True
    anchors = rng.integers(0, n, size=n)
    times = np.arange(n)
    last = np.maximum.accumulate(np.where(restart, times, 0))
    return (anchors[last] + times - last) % n

def reality_check(returns, families, repetitions=5000, block_length=None, seed=20250303):
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError('Expected a nonempty return DataFrame')
    if returns.index.has_duplicates or returns.columns.has_duplicates:
        raise ValueError('Duplicate dates or candidate names')
    if not returns.index.is_monotonic_increasing:
        raise ValueError('Dates must be increasing')
    x = returns.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite or unaligned returns; no silent dropping')
    n, k = x.shape
    if n < 20 or repetitions < 99:
        raise ValueError('Require at least 20 returns and 99 replicates')
    length = max(2, round(n ** (1/3))) if block_length is None else float(block_length)
    if not np.isfinite(length) or not 1 <= length <= n:
        raise ValueError('Expected block length must lie in [1,n]')
    indices = {}
    for name, columns in families.items():
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f'Invalid family {name}')
        indices[name] = [returns.columns.get_loc(c) for c in columns]
    observed = np.sqrt(n) * x.mean(axis=0)
    centered = x - x.mean(axis=0)
    rng = np.random.default_rng(seed)
    maxima = {name: np.empty(repetitions) for name in families}
    for b in range(repetitions):
        sample = stationary_indices(rng, n, length)
        bootstrap_statistics = _bootstrap_statistic(centered, sample)
        for name, cols in indices.items():
            maxima[name][b] = max(0., float(bootstrap_statistics[cols].max()))
    report = dict(test='White Reality Check, unstudentized, stationary bootstrap',
        benchmark='zero-return cash', null='All family members have expected net bar return <= 0',
        conditional_on='Realized training sample and its selected lags',
        n=n, start=str(returns.index[0]), end=str(returns.index[-1]),
        bootstrap_repetitions=repetitions, expected_block_length=length, seed=seed,
        alpha=.05, families={})
    for name, cols in indices.items():
        null = maxima[name]
        statistic = max(0., float(observed[cols].max()))
        p = (1 + np.count_nonzero(null >= statistic)) / (repetitions + 1)
        report['families'][name] = dict(candidates=families[name], size=len(cols),
            global_p_value=p, reject_at_005=bool(p < .05),
            candidate_adjusted_p_values={returns.columns[c]:
                (1 + np.count_nonzero(null >= max(0., observed[c]))) / (repetitions + 1)
                for c in cols})
    return report, pd.DataFrame(maxima)

def validate_return_frame(frame):
    """Never silently drop dates/candidates or replace missing returns by zero."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty return DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Returns must have valid datetime indices")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Duplicate dates or candidate names")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be increasing")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite or unaligned returns; no silent dropping")

def portfolio_return_stream(portfolio, initial_capital):
    """Include the first Test bar relative to allocated initial capital."""
    if portfolio is None or portfolio.empty:
        raise RuntimeError("Missing portfolio; refusing to shrink the RC family")
    equity = portfolio.set_index("date")["equity"].astype(float)
    previous = equity.shift(1)
    previous.iloc[0] = initial_capital
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("RC requires finite positive portfolio equity")
    if not np.isfinite(previous).all() or (previous <= 0).any():
        raise ValueError("Invalid portfolio denominator")
    return equity / previous - 1.0

def export_rc_report(frame, families, output_dir, args, scope):
    """Use the same joint stationary-bootstrap resample for all candidates."""
    validate_return_frame(frame)
    report, bootstrap = reality_check(
        frame, families, args.rc_repetitions, args.rc_block_length, args.rc_seed
    )
    report["conditional_on"] = (
        "Realized training sample, MBDP-selected lags and training-based portfolio weights; "
        "original causal rolling refitting on available past observations retained"
    )
    report["scope"] = scope
    report["return_definition"] = (
        "Net Test portfolio equity return versus zero-return cash, including first Test bar "
        "relative to initial capital. Original commission and execution timing retained."
    )
    report["interpretation"] = {
        "fixed_wilcoxon_skew": "Single pre-fixed strategy only; not a correction for selecting it on Test.",
        "wilcoxon_two_variants": "Regular/skew comparison with model and Wilcoxon fixed independently of Test.",
        "all_eight_pipelines": "Four selectors x two variants, with model fixed independently of Test.",
        "all_models_all_pipelines": "Joint correction across every model/selector/variant executed in this run.",
        "decision": "p < 0.05 rejects the family null of no positive expected net return versus cash. "
                    "Failure to reject is not evidence that the strategy has no value, and rejection "
                    "does not establish absence of bias.",
    }
    report["limitations"] = [
        "This is a data-snooping test, not demographic fairness or proof of absence of look-ahead bias.",
        "Not historical preregistration: prior Test inspection remains relevant.",
        "Only the declared families are corrected; undocumented experiments, alternative dates, "
        "other seeds, hyperparameters, files and universes are not covered.",
        "Train lag selection and weighting are not rerun inside the bootstrap; inference is conditional.",
        "Weak-dependence/stationarity assumptions are required for stationary-bootstrap validity.",
        "Benchmark is cash, not buy-and-hold; legacy execution, rolling refitting and costs are unchanged.",
        "Original portfolio carries each ticker's last equity forward on dates without observations; "
        "this calendar convention is retained, not silently changed by the RC alignment step.",
        "Original metrics_summary.csv excludes the first Test return from equity-change metrics; "
        "RC returns include it using initial capital.",
        "Dataset survivorship, price-adjustment and historical universe biases are not tested here.",
        "Input filtering may exclude whole tickers based on price availability across the entire CSV, "
        "including future dates. This is not a point-in-time universe or a correction for selection bias; "
        "consult the root data_quality.json and data_quality_report.csv for the actual universe.",
    ]
    report["configuration"] = vars(args).copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "rc_test_returns.csv", index_label="date")
    bootstrap.to_csv(out / "rc_bootstrap_maxima.csv", index=False)
    (out / "reality_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    for family, values in report["families"].items():
        for candidate, p in values["candidate_adjusted_p_values"].items():
            rows.append({
                "family": family,
                "candidate": candidate,
                "family_size": values["size"],
                "family_global_p_value": values["global_p_value"],
                "family_reject_at_005": values["reject_at_005"],
                "candidate_adjusted_p_value": p,
                "candidate_reject_at_005": bool(p < .05),
                "mean_test_bar_return": float(frame[candidate].mean()),
            })
    pd.DataFrame(rows).to_csv(out / "rc_summary.csv", index=False)
    return report

def validate_ticker_result(result, ticker, method, variant):
    if result is None or result.empty:
        raise RuntimeError(f"Missing {ticker}/{method}/{variant}")
    for split in ("train", "test"):
        part = result.loc[result["split"].eq(split)]
        if part.empty or part["date"].duplicated().any():
            raise RuntimeError(f"Empty or duplicate {split} dates: {ticker}/{method}/{variant}")
        if not np.isfinite(part[["bar_pnl", "equity"]].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Nonfinite {split} results: {ticker}/{method}/{variant}")
        if (part["equity"] <= 0).any():
            raise RuntimeError(f"Nonpositive {split} equity: {ticker}/{method}/{variant}")

def load_stock_data(args):
    """Validate input and explicitly audit whole-ticker exclusions before modeling."""
    data_path = Path(args.data_path).expanduser()
    if not data_path.is_file() and not data_path.is_absolute():
        # PyCharm may use a working directory different from this script's directory.
        candidate = Path(__file__).resolve().parent / data_path
        if candidate.is_file():
            data_path = candidate
    if not data_path.is_file():
        raise FileNotFoundError(f"CSV not found: {args.data_path}. Use --data-path /absolute/path/file.csv")
    args.data_path = str(data_path.resolve())
    data = pd.read_csv(data_path)
    data.columns = [str(c).strip() for c in data.columns]
    required = {"Date", "Ticker", args.price_col}
    if not required.issubset(data.columns) or data.columns.has_duplicates:
        raise ValueError(f"CSV must have unique columns including {sorted(required)}")
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    if data.empty or data[["Date", "Ticker"]].isna().any().any():
        raise ValueError("Empty CSV or missing Date/Ticker values")
    data["Ticker"] = data["Ticker"].astype(str).str.strip()
    if data.Ticker.eq("").any() or data.duplicated(["Ticker", "Date"]).any():
        raise ValueError("Empty ticker or duplicate ticker/date rows")
    raw_price = data[args.price_col].copy()
    numeric_price = pd.to_numeric(raw_price, errors="coerce")
    price = numeric_price.to_numpy(dtype=float)
    invalid = ~np.isfinite(price) | (price <= 0)
    reasons = np.select(
        [raw_price.isna().to_numpy(), numeric_price.isna().to_numpy(),
         ~np.isfinite(price), price <= 0],
        ["missing", "not_numeric", "nonfinite", "nonpositive"], default=""
    )
    invalid_rows = data.loc[invalid, ["Date", "Ticker"]].copy()
    invalid_rows["price_column"] = args.price_col
    invalid_rows["original_value"] = raw_price.loc[invalid]
    invalid_rows["reason"] = reasons[invalid]
    data[args.price_col] = numeric_price

    audit_rows = []
    for ticker, group in data.groupby("Ticker", sort=True):
        bad = invalid[group.index.to_numpy()]
        valid_dates = group.loc[~bad, "Date"]
        audit_rows.append({
            "Ticker": ticker, "total_rows": len(group),
            "valid_price_rows": int((~bad).sum()), "invalid_price_rows": int(bad.sum()),
            "first_valid_date": valid_dates.min(), "last_valid_date": valid_dates.max(),
            "action": ("excluded" if args.invalid_price_policy == "exclude-ticker" else "error")
                      if bad.any() else "kept",
        })
    audit = pd.DataFrame(audit_rows)
    affected = audit.loc[audit.invalid_price_rows.gt(0), "Ticker"].tolist()
    excluded = affected if args.invalid_price_policy == "exclude-ticker" else []
    cleaned = data.loc[~data.Ticker.isin(excluded)].copy()
    quality = {
        "policy": args.invalid_price_policy, "price_column": args.price_col,
        "input_rows": len(data), "input_ticker_count": int(data.Ticker.nunique()),
        "invalid_price_rows": int(invalid.sum()), "affected_tickers": affected,
        "excluded_tickers": excluded, "excluded_rows": len(data) - len(cleaned),
        "remaining_rows": len(cleaned), "remaining_tickers": sorted(cleaned.Ticker.unique()),
        "note": "Whole-ticker filtering over the entire CSV, not individual row deletion. "
                "No price imputation, backfill, forward-fill, or Close/Adj Close substitution. "
                "This changes the universe and is not a point-in-time eligibility rule; "
                "it does not establish absence of selection or survivorship bias.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "data_quality_report.csv", index=False)
    invalid_rows.to_csv(out / "invalid_price_rows.csv", index=False)
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    if affected:
        counts = audit.loc[audit.invalid_price_rows.gt(0)]
        details = ", ".join(f"{row.Ticker}={row.invalid_price_rows}/{row.total_rows}"
                            for row in counts.itertuples())
        if args.invalid_price_policy == "error":
            raise ValueError(
                f"Invalid prices in '{args.price_col}': {details}. "
                f"See {out / 'data_quality_report.csv'}. Repair the source data or explicitly "
                "use --invalid-price-policy exclude-ticker to remove whole affected tickers."
            )
        print(f"[data] Invalid prices: {details}. Excluding whole tickers: {', '.join(excluded)}.",
              flush=True)
    if cleaned.empty:
        raise ValueError(f"No tickers remain after price validation. See {out / 'data_quality_report.csv'}")
    remaining_price = cleaned[args.price_col].to_numpy(dtype=float)
    if not (np.isfinite(remaining_price) & (remaining_price > 0)).all():
        raise RuntimeError("Price validation failed after whole-ticker filtering")
    print(f"[data] Using {cleaned.Ticker.nunique()} tickers, {len(cleaned)} rows. "
          f"Audit: {out / 'data_quality_report.csv'}", flush=True)
    return cleaned, quality

def generate_btc_lags(
    n_sets=100,
    min_lags=2,
    max_lags=3,
    min_lag_value=1,
    max_lag_value=30,
    seed=42
):
    """Generate deterministic unique lag combinations."""
    random.seed(seed)

    lag_sets = []
    attempts = 0
    max_attempts = max(n_sets * 20, 100)

    while (
        len(lag_sets) < n_sets
        and attempts < max_attempts
    ):
        attempts += 1

        number_of_lags = random.randint(
            min_lags,
            max_lags
        )

        candidate = tuple(
            sorted(
                random.sample(
                    range(
                        min_lag_value,
                        max_lag_value + 1
                    ),
                    number_of_lags
                )
            )
        )

        if candidate not in lag_sets:
            lag_sets.append(candidate)

    return lag_sets

def safe_statistical_test(test_name, first, second):
    """Return a supported paired-test p-value without raising."""
    try:
        if test_name == "wilcoxon":
            if np.allclose(first, second):
                return 1.0

            _, p_value = wilcoxon(
                first,
                second,
                zero_method="wilcox",
                alternative="two-sided"
            )

        elif test_name == "ttest":
            _, p_value = ttest_rel(
                first,
                second,
                nan_policy="omit"
            )

        else:
            p_value = 1.0

        if not np.isfinite(p_value):
            return 1.0

        return float(p_value)

    except Exception:
        return 1.0

def compute_btc_metrics(
    result,
    split_name,
    annualization
):
    """Compute performance metrics for one train or test split."""
    part = result[
        result["split"] == split_name
    ].copy()

    if part.empty:
        return {}

    returns = (
        part["bar_pnl"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    equity = (
        part["equity"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if returns.empty or equity.empty:
        return {}

    total_return = (
        equity.iloc[-1] / equity.iloc[0] - 1.0
    )

    periods = len(returns)

    if periods > 0 and equity.iloc[0] > 0:
        annual_return = (
            (equity.iloc[-1] / equity.iloc[0])
            ** (annualization / periods)
            - 1.0
        )
    else:
        annual_return = np.nan

    if len(returns) > 1:
        volatility = returns.std(ddof=1)
        annual_volatility = (
            volatility * np.sqrt(annualization)
        )
    else:
        volatility = np.nan
        annual_volatility = np.nan

    if np.isfinite(volatility) and volatility > 0:
        sharpe = (
            returns.mean()
            / volatility
            * np.sqrt(annualization)
        )
    else:
        sharpe = np.nan

    max_drawdown = compute_max_drawdown(equity)

    if (
        np.isfinite(max_drawdown)
        and max_drawdown < 0
        and np.isfinite(annual_return)
    ):
        calmar = annual_return / abs(max_drawdown)
    else:
        calmar = np.nan

    signals = part["signal"]

    n_buy = int((signals == "BUY").sum())
    n_sell = int((signals == "SELL").sum())
    n_hold = int((signals == "HOLD").sum())

    buy_rows = part[signals == "BUY"]
    sell_rows = part[signals == "SELL"]

    wins = (
        (buy_rows["future_price"] > buy_rows["today_price"]).sum()
        + (sell_rows["future_price"] < sell_rows["today_price"]).sum()
    )

    total_signals = n_buy + n_sell

    if total_signals > 0:
        win_rate = wins / total_signals
    else:
        win_rate = np.nan

    return {
        "split": split_name,
        "bars": int(len(part)),
        "trades": int(total_signals),
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_hold": n_hold,
        "total_pnl": float(part["bar_pnl"].sum()),
        "final_cumulative_pnl": float(
            part["cumulative_pnl"].iloc[-1]
        ),
        "win_rate": float(win_rate)
        if np.isfinite(win_rate)
        else np.nan,
        "equity_start": float(equity.iloc[0]),
        "equity_end": float(equity.iloc[-1]),
        "return_pct": float(total_return * 100.0),
        "annual_return_pct": float(annual_return * 100.0)
        if np.isfinite(annual_return)
        else np.nan,
        "annual_volatility_pct": float(
            annual_volatility * 100.0
        )
        if np.isfinite(annual_volatility)
        else np.nan,
        "sharpe_ratio": float(sharpe)
        if np.isfinite(sharpe)
        else np.nan,
        "calmar_ratio": float(calmar)
        if np.isfinite(calmar)
        else np.nan,
        "max_drawdown_pct": float(max_drawdown * 100.0),
    }

def load_btc_data(data_path):
    """Load and clean a dated positive price series from a pickle."""
    raw = pd.read_pickle(data_path)

    if not isinstance(raw, pd.DataFrame):
        raise TypeError(
            "The pickle file must contain a pandas DataFrame."
        )

    raw = raw.copy()

    price_candidates = [
        "Close",
        "close",
        "Adj Close",
        "adj close",
        "Price",
        "price",
        "Last",
        "last",
    ]

    price_column = None

    for column in price_candidates:
        if column in raw.columns:
            price_column = column
            break

    if price_column is None:
        numeric_columns = raw.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_columns:
            raise ValueError(
                "No numeric price column was found."
            )

        price_column = numeric_columns[-1]

    date_candidates = [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Time",
        "time",
    ]

    date_column = None

    for column in date_candidates:
        if column in raw.columns:
            date_column = column
            break

    if date_column is not None:
        dates = pd.to_datetime(
            raw[date_column],
            errors="coerce"
        )

    elif isinstance(raw.index, pd.DatetimeIndex):
        dates = pd.to_datetime(
            raw.index,
            errors="coerce"
        )

    else:
        raise ValueError(
            "No date column or DatetimeIndex was found."
        )

    prices = pd.to_numeric(
        raw[price_column],
        errors="coerce"
    )

    data = pd.DataFrame({
        "date": dates,
        "price": prices,
    })

    data = data.dropna(
        subset=["date", "price"]
    ).copy()

    data = data[
        np.isfinite(data["price"])
        & (data["price"] > 0)
    ].copy()

    data = (
        data.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return data

def export_btc_report(result_map, output_dir, args):
    streams = {}
    for method, variants in result_map.items():
        for variant, result in variants.items():
            part = result.loc[result['split'].eq('test')]
            streams[f'{method}__{variant}'] = part.set_index('date')['bar_pnl']
    expected = [f'{m}__{v}' for m in ('raw_pnl','raw_sharpe','wilcoxon','ttest')
                for v in ('regular','maxlag_skew')]
    missing = set(expected) - set(streams)
    if missing:
        raise RuntimeError(f'Missing pipelines; refusing to shrink RC family: {sorted(missing)}')
    frame = pd.DataFrame(streams).sort_index()[expected]
    families = {
        'fixed_wilcoxon_skew': ['wilcoxon__maxlag_skew'],
        'wilcoxon_two_variants': ['wilcoxon__regular','wilcoxon__maxlag_skew'],
        'all_eight_pipelines': expected,
    }
    report, bootstrap = reality_check(frame, families, args.rc_repetitions,
                                     args.rc_block_length, args.rc_seed)
    report['interpretation'] = {
        'fixed_wilcoxon_skew': 'Single-strategy one-sided bootstrap test, not a correction for choosing methods/variants on Test. Applicable to an independently fixed Wilcoxon+skew pipeline.',
        'wilcoxon_two_variants': 'Correction for comparing regular and skew after fixing Wilcoxon independently of Test.',
        'all_eight_pipelines': 'Correction for the eight displayed method/variant pipelines, conditional on their Train-selected lags. Does not cover undocumented historical searches.',
    }
    report['limitations'] = [
        'Not historical preregistration; prior Test inspection remains relevant.',
        'Original training selector, rolling execution and costs retained unchanged.',
        'No claim of accounting for all Train-selection uncertainty across new training samples.',
        'Weak-dependence/stationarity assumptions required for bootstrap validity.',
        'Null is mean return relative to cash, not superiority to buy-and-hold or positive Sharpe after all historical experimentation.',
    ]
    out = Path(output_dir)
    frame.to_csv(out/'rc_test_returns.csv',index_label='date')
    bootstrap.to_csv(out/'rc_bootstrap_maxima.csv',index=False)
    (out/'reality_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows=[]
    for family, values in report['families'].items():
        for candidate, p in values['candidate_adjusted_p_values'].items():
            rows.append(dict(family=family,candidate=candidate,
                family_global_p_value=values['global_p_value'],candidate_adjusted_p_value=p,
                mean_test_bar_return=float(frame[candidate].mean())))
    pd.DataFrame(rows).to_csv(out/'rc_summary.csv',index=False)
    return report


# Compiled numerical kernels. No fastmath, no approximate rolling covariance.
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
                        cov[a, b] += (X[i, a] - mean[a]) * (X[i, b] - mean[b])
        cov /= used - 1
    for a in range(d):
        cov[a, a] += reg
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        for a in range(d):
            cov[a, a] += 1e-5
        sign, logdet = np.linalg.slogdet(cov)
    diff = x - mean
    # Same pseudoinverse cutoff as NumPy's original implementation.
    quad = diff @ np.linalg.pinv(cov, rcond=1e-15) @ diff
    if student:
        density = (math.lgamma((nu + d) / 2) - math.lgamma(nu / 2)
                   - .5 * d * np.log(nu * np.pi) - .5 * logdet
                   - .5 * (nu + d) * np.log(1 + quad / nu))
    else:
        density = -.5 * (d * np.log(2 * np.pi) + logdet + quad)
    return np.log(max((count + 1.) / (total + 2.), 1e-12)) + density


@njit(cache=True)
def _probabilities(X, y, window, horizon, student, nu, reg, kind, momentum):
    n, d = X.shape
    probs = np.full(n, np.nan)
    downs = np.full(n, np.nan)
    minimum = max(10, d + 2)
    for j in range(window + horizon, n):
        end = j - horizon
        start = end - window
        valid = 0
        for i in range(start, end):
            if np.isfinite(X[i]).all():
                valid += 1
        if valid < minimum or not np.isfinite(X[j]).all():
            continue
        if kind == 1:  # BuyHold
            pu, pd_ = 1., 0.
        elif kind == 2:  # TSMomentum: feature ratios only, not skew.
            value = 0.
            for a in range(momentum):
                value += np.log(X[j, a])
            pu = 1. if value > 0 else 0. if value < 0 else .5
            pd_ = 1. - pu
        else:
            lu = _class_log_density(X, y, start, end, X[j], 1, reg, student, nu)
            ld = _class_log_density(X, y, start, end, X[j], 0, reg, student, nu)
            m = max(lu, ld)
            eu, ed = np.exp(lu - m), np.exp(ld - m)
            total = eu + ed
            if total <= 0 or not np.isfinite(total):
                pu, pd_ = .5, .5
            else:
                pu, pd_ = eu / total, ed / total
        probs[j], downs[j] = pu, pd_
    return probs, downs


@njit(cache=True)
def _execute(prices, probs, downs, threshold, commission):
    n = len(prices)
    signals = np.zeros(n, np.int64)
    pnl = np.zeros(n)
    positions = np.zeros(n, np.int64)
    for j in range(n):
        if probs[j] > threshold:
            signals[j] = 1
        elif downs[j] > threshold:
            signals[j] = -1
    position = 0
    for j in range(1, n):
        today, prev = prices[j], prices[j - 1]
        if not (np.isfinite(today) and np.isfinite(prev) and prev > 0):
            positions[j] = position
            continue
        ret = (today - prev) / prev
        new = signals[j - 1] if signals[j - 1] != 0 else position
        bar = ret if position == 1 else -ret if position == -1 else 0.
        if new != position:
            bar -= (1 if position == 0 or new == 0 else 2) * commission
        pnl[j], positions[j] = bar, new
        position = new
    return signals, pnl, positions


@njit(cache=True)
def _bootstrap_statistic(centered, sample):
    n, k = centered.shape
    sums = np.zeros(k)
    for i in range(n):
        for j in range(k):
            sums[j] += centered[sample[i], j]
    return sums / np.sqrt(n)


def build_samples(prices, dates, lags, predict_day=1, include_maxlag_skew=False):
    """Vectorized features; SciPy skew is retained for numerical compatibility."""
    prices = np.asarray(prices, dtype=np.float64)
    dates = np.asarray(dates)
    max_lag = max(lags)
    t = np.arange(max_lag, len(prices) - predict_day)
    if not len(t):
        return pd.DataFrame()
    p0, p1 = prices[t], prices[t + predict_day]
    past = prices[t[:, None] - np.asarray(lags)[None, :]]
    valid = (np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0)
             & np.all(np.isfinite(past) & (past > 0), axis=1))
    X = p0[:, None] / past
    cols = dict(date=dates[t][valid], today_price=p0[valid],
                future_price=p1[valid], label=(p1[valid] > p0[valid]).astype(int))
    for i in range(len(lags)):
        cols[f'x_{i}'] = X[valid, i]
    if include_maxlag_skew:
        windows = np.lib.stride_tricks.sliding_window_view(prices, max_lag + 1)[:len(t)]
        if max_lag + 1 >= 3:
            # Bounded temporary memory even for user-specified large lag windows.
            values = np.empty(len(t))
            chunk = max(1, 1_000_000 // (max_lag + 1))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for start in range(0, len(t), chunk):
                    values[start:start + chunk] = skew(windows[start:start + chunk], axis=1, bias=False)
            values[~np.isfinite(values)] = 0.
        else:
            values = np.zeros(len(t))
        cols[f'x_{len(lags)}'] = values[valid]
    return pd.DataFrame(cols)


def rolling_trade(df, n_features, window_size=120, predict_day=1, dist='normal',
                  df_student=5., reg=1e-6, prob_threshold=.5, commission=.001,
                  model='MBDP', momentum_features=None):
    X = np.ascontiguousarray(df[[f'x_{i}' for i in range(n_features)]], dtype=float)
    y = np.ascontiguousarray(df.label, dtype=np.int64)
    prices = np.ascontiguousarray(df.today_price, dtype=float)
    if model in ('MBDP', 'BuyHold', 'TSMomentum'):
        kind = {'MBDP': 0, 'BuyHold': 1, 'TSMomentum': 2}[model]
        probs, downs = _probabilities(X, y, window_size, predict_day, dist != 'normal',
                                     df_student, reg, kind, momentum_features or n_features)
    else:
        # sklearn estimators stay EXACT: no changed solver, tolerance or fit cadence.
        probs, downs = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for j in range(window_size + predict_day, len(df)):
            end = j - predict_day
            Xw, yw = X[end - window_size:end], y[end - window_size:end]
            mask = np.all(np.isfinite(Xw), axis=1)
            Xw, yw = Xw[mask], yw[mask]
            if len(yw) < max(10, n_features + 2) or not np.isfinite(X[j]).all():
                continue
            probs[j], downs[j] = baseline_probability(model, Xw, yw, X[j], momentum_features or n_features)
    codes, pnl, positions = _execute(prices, probs, downs, prob_threshold, commission)
    out = df.copy()
    out['signal'] = np.where(codes == 1, 'BUY', np.where(codes == -1, 'SELL', 'HOLD'))
    out['prob_up'], out['position'] = probs, positions
    out['bar_pnl'], out['cumulative_pnl'] = pnl, np.cumsum(pnl)
    return out


def build_portfolio_for_method(results_list, initial_capital=1_000_000.):
    """Original allocation/carry-forward convention without date x ticker loops."""
    items, weights = [], []
    for result in results_list:
        train = result.loc[result.split.eq('train')]
        test = result.loc[result.split.eq('test')]
        if train.empty or test.empty or train.equity.iloc[0] <= 0:
            raise RuntimeError('Missing/invalid portfolio constituent')
        weights.append(max(float(train.equity.iloc[-1] / train.equity.iloc[0] - 1), 0.))
        items.append(test.set_index('date').bar_pnl)
    if not items:
        return None
    weights = np.asarray(weights)
    weights = np.full(len(items), 1 / len(items)) if weights.sum() <= 0 else weights / weights.sum()
    allocated = initial_capital * weights
    dates = pd.DatetimeIndex(sorted(set().union(*(s.index for s in items))))
    total = np.zeros(len(dates))
    for series, capital in zip(items, allocated):
        equity = capital * (1 + series).cumprod()
        total += equity.reindex(dates).ffill().fillna(capital).to_numpy()
    return pd.DataFrame({'date': dates, 'equity': total})


# One worker-resident dataset, compact tasks, stable ordered task collection.
_DATA = None
_ARGS = None
_LIMITER = None


def warmup():
    X = np.ascontiguousarray(np.arange(80, dtype=float).reshape(40, 2) / 100 + 1)
    y = np.arange(40, dtype=np.int64) % 2
    p, d = _probabilities(X, y, 15, 1, True, 10., 1e-6, 0, 2)
    _execute(np.ones(40), p, d, .9, .001)
    _bootstrap_statistic(np.ones((20, 2)), np.arange(20, dtype=np.int64))


def init_worker(data, args):
    global _DATA, _ARGS, _LIMITER
    _DATA, _ARGS = data, args
    _LIMITER = threadpool_limits(limits=1)
    warmup()


def evaluate(task):
    """task=(ticker, variant, lag_tuple, model, training_only)."""
    ticker, variant, lags, model, training_only = task
    a = _ARGS
    try:
        raw = _DATA[ticker]
        samples = build_samples(raw.price.to_numpy(), raw.date.to_numpy(), lags,
                                a.predict_day, variant == 'maxlag_skew')
        if samples.empty or len(samples) < a.window_size + max(lags) + a.predict_day + 10:
            raise ValueError('Insufficient history for lag/window configuration')
        samples = split_df(samples, a.train_start, a.train_end, a.test_start, a.test_end)
        if not samples.split.eq('train').any() or not samples.split.eq('test').any():
            raise ValueError('Empty Train or Test; check dates and history')
        # Build labels using full data first. Only omit calculations AFTER cutoff.
        # All pre-Train history and the gap between Train and Test are retained.
        cutoff = pd.Timestamp(a.train_end if training_only else a.test_end)
        samples = samples.loc[samples.date < cutoff].reset_index(drop=True)
        result = rolling_trade(samples, len(lags) + int(variant == 'maxlag_skew'),
                               a.window_size, a.predict_day, a.dist, a.student_df,
                               a.reg, a.prob_threshold, a.commission, model, len(lags))
        if training_only:
            returns = result.loc[result.split.eq('train'), 'bar_pnl'].to_numpy()
            if not len(returns) or not np.isfinite(returns).all():
                raise ValueError('Invalid Train returns')
            return returns
        capital = a.initial_capital if a.profile == 'XGboost_GRU' else 10_000.
        result['equity'] = capital * (1 + result.bar_pnl).cumprod()
        result['Ticker'], result['lags'] = ticker, str(tuple(lags))
        result['include_maxlag_skew'] = variant == 'maxlag_skew'
        validate_ticker_result(result, ticker, model, variant)
        # Full-history cumulative columns calculated before removing unused rows.
        # Keep unused rows so that exact commission repricing does not splice
        # the train/test segments across the inter-segment date gap.
        return result.copy()
    except Exception as exc:
        raise RuntimeError(f'{a.profile}/{ticker}/{variant}/{lags}/{model}: {exc}') from exc


def collect(pool, tasks, label):
    """imap retains generation order: ties/NaNs follow legacy selection order."""
    iterator = map(evaluate, tasks) if pool is None else pool.imap(evaluate, tasks, chunksize=1)
    results = []
    start = time.perf_counter()
    every = max(1, len(tasks) // 20)
    for i, result in enumerate(iterator, 1):
        results.append(result)
        if i == 1 or i == len(tasks) or i % every == 0:
            print(f'[{label}] {i}/{len(tasks)} | {time.perf_counter() - start:.1f}s', flush=True)
    return results


def select_lags(lag_sets, returns_by_lag, btc=False, annualization=252):
    """Legacy selector, including its concatenate-then-truncate comparator.

    This comparator is NOT a new paired-by-date test. It is intentionally
    preserved rather than silently changing the supplied statistical strategy.
    Stock generator's duplicates are kept in this evaluation list.
    """
    evaluations = []
    for lags in lag_sets:
        r = returns_by_lag[tuple(lags)]
        std = np.std(r, ddof=1 if btc else 0) if len(r) > 1 else 0.
        score = np.mean(r) / std * np.sqrt(annualization if btc else 252) if std > 0 else -np.inf
        evaluations.append(dict(lags=tuple(lags), returns=r, total_pnl=float(r.sum()), sharpe=score))
    selected = {m: max(evaluations, key=lambda e: e[key])['lags']
                for m, key in [('raw_pnl', 'total_pnl'), ('raw_sharpe', 'sharpe')]}
    details = []
    for i, current in enumerate(evaluations):
        # Only construct the prefix actually consumed by the original selector.
        pieces, remaining = [], len(current['returns'])
        for j, other in enumerate(evaluations):
            if i != j and remaining > 0:
                piece = other['returns'][:remaining]
                pieces.append(piece)
                remaining -= len(piece)
        second = np.concatenate(pieces) if pieces else np.empty(0)
        first = current['returns'][:len(second)]
        pvals = {m: 1. for m in STATISTICAL_METHODS}
        if len(second) >= 5:
            for method in STATISTICAL_METHODS:
                if btc:
                    pvals[method] = safe_statistical_test(method, first, second)
                else:
                    try:
                        fn = wilcoxon if method == 'wilcoxon' else ttest_rel
                        pvals[method] = float(fn(first, second).pvalue)
                    except Exception:
                        pvals[method] = 1.
        details.append(dict(lags=current['lags'], **{
            method: current['total_pnl'] - 100 * pvals[method] for method in STATISTICAL_METHODS}))
    for method in STATISTICAL_METHODS:
        selected[method] = max(details, key=lambda e: e[method])['lags']
    return selected


def model_report(model, tickers, selections, final_results, args):
    out = Path(args.output_dir) / model
    out.mkdir(parents=True, exist_ok=True)
    rows, selection_rows, streams = [], [], {}
    for method in METHODS:
        for variant in VARIANTS:
            results = []
            for ticker in tickers:
                lags = selections[ticker, variant][method]
                result = final_results[ticker, variant, tuple(lags), model, False]
                results.append(result)
                selection_rows.append(dict(model=model, ticker=ticker, method=method,
                                           variant=variant, lags=str(tuple(lags))))
            portfolio = build_portfolio_for_method(results, args.initial_capital)
            metrics = compute_portfolio_metrics(results, portfolio, args.annualization)
            if not metrics:
                raise RuntimeError(f'Missing metrics: {model}/{method}/{variant}')
            name = f'{method}__{variant}'
            streams[name] = portfolio_return_stream(portfolio, args.initial_capital)
            portfolio.to_csv(out / f'portfolio_{name}.csv', index=False)
            rows.append(dict(method=method, variant=variant,
                             lags='; '.join(sorted({str(r.lags.iloc[0]) for r in results})), **metrics))
    frame = pd.DataFrame(streams).sort_index()
    families = dict(fixed_wilcoxon_skew=['wilcoxon__maxlag_skew'],
                    wilcoxon_two_variants=['wilcoxon__regular', 'wilcoxon__maxlag_skew'],
                    all_eight_pipelines=list(frame.columns))
    export_rc_report(frame, families, out, args, f'Within fixed model: {model}')
    write_metrics_summary(rows, out / 'metrics_summary.csv')
    pd.DataFrame(selection_rows).sort_values(['ticker', 'method', 'variant']).to_csv(out / 'selected_lags.csv', index=False)
    return frame


def btc_report(selections, final_results, args):
    out = Path(args.output_dir)
    result_map = {m: {} for m in METHODS}
    rows = []
    for method in METHODS:
        for variant in VARIANTS:
            lags = selections['BTCUSD', variant][method]
            result = final_results['BTCUSD', variant, tuple(lags), 'MBDP', False]
            result_map[method][variant] = result
            metrics = compute_btc_metrics(result, 'test', args.annualization)
            rows.append(dict(method=method, variant=variant, lags=str(tuple(lags)), **metrics))
            result.loc[result.split.eq('test'), ['date', 'equity', 'bar_pnl', 'position']].to_csv(
                out / f'test_{method}__{variant}.csv', index=False)
    write_metrics_summary(rows, out / 'metrics_summary.csv')
    export_btc_report(result_map, out, args)
    (out / 'train_selected_lags.json').write_text(json.dumps(
        {v: selections['BTCUSD', v] for v in VARIANTS}, indent=2), encoding='utf-8')


PROFILES = {
    'DRL_Ensemble': dict(dates=('2015-06-01', '2015-12-31', '2016-01-01', '2020-05-08'),
                         max_lags=2, min_lag_value=1, max_lag_value=30, window_size=250,
                         commission=.001, initial_capital=1_000_000., annualization=252),
    'FineRL': dict(dates=('2019-07-01', '2020-07-01', '2020-07-01', '2022-03-31'),
                   max_lags=5, min_lag_value=5, max_lag_value=50, window_size=250,
                   commission=.001, initial_capital=1_000_000., annualization=252),
    'XGboost_GRU': dict(dates=('2020-01-01', '2021-07-01', '2021-07-01', '2023-03-01'),
                        max_lags=2, min_lag_value=1, max_lag_value=30, window_size=80,
                        commission=0., initial_capital=1., annualization=365),
}


def profile_args(cli, profile):
    config = PROFILES[profile].copy()
    dates = config.pop('dates')
    config.update(zip(('train_start', 'train_end', 'test_start', 'test_end'), dates))
    config.update(profile=profile, data_path=str(Path(cli.btc_path if profile == 'XGboost_GRU' else cli.stock_path).resolve()),
                  output_dir=str(Path(cli.output_dir).resolve() / profile), models=cli.models,
                  n_workers=cli.workers, n_lag_sets=cli.n_lag_sets, min_lags=2,
                  predict_day=1, dist='student', student_df=10., reg=1e-6,
                  prob_threshold=.9, price_col=cli.price_col,
                  invalid_price_policy=cli.invalid_price_policy, random_seed=42,
                  rc_repetitions=cli.rc_repetitions, rc_block_length=cli.rc_block_length,
                  rc_seed=cli.rc_seed)
    for name in ('train_start', 'train_end', 'test_start', 'test_end', 'window_size',
                 'predict_day', 'dist', 'student_df', 'reg', 'prob_threshold', 'commission',
                 'initial_capital', 'annualization', 'min_lags', 'max_lags', 'min_lag_value', 'max_lag_value'):
        value = getattr(cli, name, None)
        if value is not None:
            config[name] = value
    args = argparse.Namespace(**config)
    dates = [pd.Timestamp(config[k]) for k in ('train_start', 'train_end', 'test_start', 'test_end')]
    if any(pd.isna(d) for d in dates) or not dates[0] < dates[1] <= dates[2] < dates[3]:
        raise ValueError('Require train_start < train_end <= test_start < test_end')
    if min(args.window_size, args.predict_day, args.annualization) < 1:
        raise ValueError('Window, horizon and annualization must be positive')
    if not 1 <= args.min_lags <= args.max_lags <= args.max_lag_value - args.min_lag_value + 1 or args.min_lag_value < 1:
        raise ValueError('Invalid lag search bounds')
    values = [args.student_df, args.reg, args.prob_threshold, args.commission, args.initial_capital]
    if not np.isfinite(values).all() or min(args.student_df, args.reg, args.initial_capital) <= 0 or args.commission < 0 or not .5 <= args.prob_threshold < 1:
        raise ValueError('Invalid numeric strategy parameters')
    return args


def run_profile(args):
    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(profile=args.profile, status='running', configuration=vars(args),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, scipy=scipy.__version__,
                                  sklearn=sklearn.__version__, numba=numba.__version__),
                    acceleration='Numba without fastmath; exact per-bar refits; shared Train selection; persistent spawn Pool',
                    source_note='Profile filenames are labels, not DRL/FinRL/XGBoost/GRU implementations.',
                    inference_note='Conditional White RC, not proof of absence of bias; no cross-profile joint RC (different calendars).',
                    selector_note='Original concatenate-then-truncate paired comparison retained, including stock NaN/tie behavior.',
                    metric_note='Original annualization defaults retained: BTC H4 uses 365, not 6*365; trades counts signals, not fills.')
    manifest_path = out / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        manifest['input_sha256'] = hashlib.sha256(Path(args.data_path).read_bytes()).hexdigest()
        is_btc = args.profile == 'XGboost_GRU'
        if is_btc:
            data = {'BTCUSD': load_btc_data(args.data_path)}
            raw = pd.read_pickle(args.data_path)  # trusted input only, audit original row count
            quality = dict(policy='legacy-drop-invalid-and-duplicate-dates', input_rows=len(raw),
                           retained_rows=len(data['BTCUSD']), removed_rows=len(raw) - len(data['BTCUSD']),
                           note='Original BTC price-column selection/cleaning and H4 annualization=365 retained. Not a new data-quality correction.')
            (out / 'data_quality.json').write_text(json.dumps(quality, indent=2), encoding='utf-8')
            lag_sets = generate_btc_lags(args.n_lag_sets, args.min_lags, args.max_lags,
                                        args.min_lag_value, args.max_lag_value, args.random_seed)
            models = ['MBDP']
        else:
            frame, quality = load_stock_data(args)
            data = {str(t): g.sort_values('Date')[['Date', args.price_col]].rename(
                columns={'Date': 'date', args.price_col: 'price'}).reset_index(drop=True)
                    for t, g in frame.groupby('Ticker', sort=True)}
            lag_sets = generate_random_lag_sets(args.n_lag_sets, args.min_lags, args.max_lags,
                                               args.min_lag_value, args.max_lag_value)
            models = args.models
        if not lag_sets:
            raise RuntimeError('No lag candidates')
        unique_lags = list(dict.fromkeys(map(tuple, lag_sets)))
        manifest.update(data_quality=quality, tickers=sorted(data), models_executed=models,
                        lag_sets_as_generated=lag_sets, unique_lag_set_count=len(unique_lags),
                        lag_generator_note='Legacy stock generator may emit duplicates; evaluation is cached but original selection order/multiplicity is retained.')
        (out / 'lag_candidates.json').write_text(json.dumps(lag_sets, indent=2), encoding='utf-8')
        (out / 'run_arguments.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        tickers = sorted(data)
        tasks = [(t, v, lag, 'MBDP', True) for t in tickers for v in VARIANTS for lag in unique_lags]
        workers = min(args.n_workers, len(tasks))
        manifest['actual_workers'] = workers
        print(f'\n[{args.profile}] tickers={len(tickers)} models={models} lag entries={len(lag_sets)} unique={len(unique_lags)} workers={workers}', flush=True)
        if workers == 1:
            init_worker(data, args)
        context = (mp.get_context('spawn').Pool(workers, initializer=init_worker, initargs=(data, args))
                   if workers > 1 else nullcontext(None))
        with context as pool:
            train_outputs = collect(pool, tasks, args.profile + '/Train')
            lookup = dict(zip(tasks, train_outputs))
            selections = {}
            for ticker in tickers:
                for variant in VARIANTS:
                    by_lag = {lag: lookup[ticker, variant, lag, 'MBDP', True] for lag in unique_lags}
                    selections[ticker, variant] = select_lags(lag_sets, by_lag, is_btc, args.annualization)
            del lookup, train_outputs
            selected_rows = [dict(ticker=t, variant=v, method=m, lags=str(lags))
                             for (t, v), choices in selections.items() for m, lags in choices.items()]
            pd.DataFrame(selected_rows).to_csv(out / 'selected_lags.csv', index=False)
            final_tasks = list(dict.fromkeys((t, v, tuple(lag), model, False)
                               for model in models for (t, v), choices in selections.items() for lag in choices.values()))
            # Report/release one model at a time to bound parent memory.
            frames = []
            for model in models:
                subset = [task for task in final_tasks if task[3] == model]
                outputs = collect(pool, subset, args.profile + '/' + model)
                final_results = dict(zip(subset, outputs))
                if is_btc:
                    btc_report(selections, final_results, args)
                else:
                    frame = model_report(model, tickers, selections, final_results, args)
                    frames.append(frame.add_prefix(f'{model}__'))
                del outputs, final_results
        if not is_btc:
            combined = pd.concat(frames, axis=1).sort_index()
            export_rc_report(combined, {'all_models_all_pipelines': list(combined.columns)},
                             out / 'all_models', args, 'Joint correction across models in this profile')
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest['status'], manifest['error'] = 'failed', str(exc)
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[{args.profile}] DONE in {manifest["elapsed_seconds"]:.1f}s -> {out}', flush=True)
    return manifest


def self_test():
    """Synthetic equivalence checks, NOT a claim of full-data equivalence."""
    threadpool_limits(limits=1)
    warmup()
    rng = np.random.default_rng(123)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, .015, 240)))
    dates = pd.date_range('2019-01-01', periods=len(prices))
    max_probability_error = 0.
    for with_skew in (False, True):
        samples = build_samples(prices, dates, (2, 7), 1, with_skew)
        for dist in ('normal', 'student'):
            for model in STRATEGIES:
                options = dict(window_size=35, dist=dist, df_student=10., reg=1e-6,
                               prob_threshold=.65, commission=.001, model=model, momentum_features=2)
                expected = reference_rolling_trade(samples, 2 + int(with_skew), **options)
                actual = rolling_trade(samples, 2 + int(with_skew), **options)
                np.testing.assert_allclose(actual.prob_up, expected.prob_up, atol=1e-8, rtol=1e-8)
                np.testing.assert_array_equal(actual.signal, expected.signal)
                np.testing.assert_array_equal(actual.position, expected.position)
                np.testing.assert_allclose(actual.bar_pnl, expected.bar_pnl, atol=1e-14)
                error = np.nanmax(np.abs(actual.prob_up - expected.prob_up))
                max_probability_error = max(max_probability_error, float(error))
    # Constant prices and single-class labels exercise covariance fallback.
    for path in (np.full(150, 100.), np.linspace(100, 130, 150)):
        samples = build_samples(path, dates[:150], (2,), 2, True)
        expected = reference_rolling_trade(samples, 2, window_size=25, predict_day=2)
        actual = rolling_trade(samples, 2, window_size=25, predict_day=2)
        np.testing.assert_allclose(actual.prob_up, expected.prob_up, atol=1e-8, rtol=1e-8)
        np.testing.assert_array_equal(actual.signal, expected.signal)
    x = rng.normal(size=(80, 7))
    for _ in range(20):
        idx = stationary_indices(rng, len(x), 5.)
        np.testing.assert_allclose(_bootstrap_statistic(x, idx), x[idx].sum(axis=0) / np.sqrt(len(x)), atol=1e-14)
    # Causal truncation must preserve all earlier PnL.
    samples = build_samples(prices, dates, (2, 7))
    full = rolling_trade(samples, 2, window_size=35)
    prefix = rolling_trade(samples.iloc[:120], 2, window_size=35)
    np.testing.assert_array_equal(full.bar_pnl.iloc[:120], prefix.bar_pnl)
    print(f'SELF-TEST PASSED: all five models, both distributions/variants, constant prices, horizon=2, bootstrap and causal prefix. Max probability error={max_probability_error:.3g}')


def parse_cli(argv=None):
    here = Path(__file__).resolve().parent
    def _repo_data(name):
        """Locate a dataset in the repository data/ folder (or next to the script)."""
        here = Path(__file__).resolve().parent
        for base in (here, *here.parents):
            for candidate in (base / "data" / name, base / name):
                if candidate.is_file():
                    return str(candidate)
        return str(here.parent / "data" / name)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument('--stock-path', default=_repo_data('djia30_full_2009_2022.csv'))
    parser.add_argument('--btc-path', default=_repo_data('BTCUSD.pkl'))
    parser.add_argument('--output-dir', default=str(here / 'bias_results_fast'))
    parser.add_argument('--workers', '--n-workers', dest='workers', type=int,
                        default=max(1, min(8, os.cpu_count() or 1)), help='Default: min(8, CPU count); one BLAS thread per worker')
    parser.add_argument('--models', nargs='+', choices=STRATEGIES, default=list(STRATEGIES),
                        help='Stock models; BTC profile always uses its original MBDP')
    parser.add_argument('--n-lag-sets', type=int, default=100)
    parser.add_argument('--rc-repetitions', type=int, default=5000)
    parser.add_argument('--rc-block-length', type=float)
    parser.add_argument('--rc-seed', type=int, default=20250303)
    parser.add_argument('--price-col', default='Adj Close')
    parser.add_argument('--invalid-price-policy', choices=['exclude-ticker', 'error'], default='exclude-ticker')
    parser.add_argument('--self-test', action='store_true', help='Run numerical checks, then exit without reading inputs')
    parser.add_argument('--continue-on-error', action='store_true', help='Attempt remaining profiles after a failure; still exit nonzero')
    for name in ('train-start', 'train-end', 'test-start', 'test-end'):
        parser.add_argument('--' + name, help='Override for ALL selected profiles (default: each original profile)')
    for name in ('window-size', 'predict-day', 'annualization', 'min-lags', 'max-lags', 'min-lag-value', 'max-lag-value'):
        parser.add_argument('--' + name, type=int)
    for name in ('student-df', 'reg', 'prob-threshold', 'commission', 'initial-capital'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--dist', choices=['normal', 'student'])
    args = parser.parse_args(argv)
    if min(args.workers, args.n_lag_sets) < 1 or args.rc_repetitions < 99 or args.rc_seed < 0:
        parser.error('workers/lag sets must be positive; RC repetitions >= 99; seed >= 0')
    if args.rc_block_length is not None and (not np.isfinite(args.rc_block_length) or args.rc_block_length < 1):
        parser.error('RC block length must be finite and >= 1')
    if len(set(args.models)) != len(args.models) or len(set(args.profiles)) != len(args.profiles):
        parser.error('Duplicate model/profile')
    return args


def main_cli(argv=None):
    cli = parse_cli(argv)
    if cli.self_test:
        self_test()
        return 0
    configs = [profile_args(cli, name) for name in cli.profiles]
    # Check all inputs before starting a potentially long run.
    for args in configs:
        if not Path(args.data_path).is_file():
            raise FileNotFoundError(args.data_path)
    root = Path(cli.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Never mix outputs from different configurations or leave stale "success" files.
    for args in configs:
        directory = Path(args.output_dir)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'Output not empty: {directory}. Use a new --output-dir or move the old directory.')
    threadpool_limits(limits=1)
    print('Warming Numba cache (first execution includes compilation)...', flush=True)
    warmup()
    statuses, all_metrics, all_rc = [], [], []
    failed = False
    for args in configs:
        try:
            report = run_profile(args)
            statuses.append(dict(profile=args.profile, status='completed', seconds=report['elapsed_seconds']))
            directory = Path(args.output_dir)
            for file in sorted(directory.rglob('metrics_summary.csv')):
                df = pd.read_csv(file)
                df.insert(0, 'scope', str(file.parent.relative_to(directory)))
                df.insert(0, 'profile', args.profile)
                all_metrics.append(df)
            for file in sorted(directory.rglob('rc_summary.csv')):
                df = pd.read_csv(file)
                df.insert(0, 'scope', str(file.parent.relative_to(directory)))
                df.insert(0, 'profile', args.profile)
                all_rc.append(df)
        except Exception as exc:
            failed = True
            statuses.append(dict(profile=args.profile, status='failed', error=str(exc)))
            traceback.print_exc()
            if not cli.continue_on_error:
                break
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(root / 'all_metrics.csv', index=False)
    if all_rc:
        pd.concat(all_rc, ignore_index=True).to_csv(root / 'all_rc_summary.csv', index=False)
    summary = dict(status='failed' if failed else 'completed', profiles=statuses,
                   requested_profiles=cli.profiles,
                   not_attempted=[p for p in cli.profiles if p not in {s['profile'] for s in statuses}],
                   note='Summary concatenates reports only; NOT a joint test across incompatible profile calendars.')
    (root / 'suite_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nSuite {summary["status"]}: {root}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main_cli())
