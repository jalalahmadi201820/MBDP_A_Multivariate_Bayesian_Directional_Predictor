"""Silent Bayesian lag-search strategy with one consolidated CSV export."""

from pathlib import Path
from math import lgamma
import argparse
import random
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd

from scipy.stats import (
    skew,
    wilcoxon,
    ttest_rel,
)

warnings.filterwarnings("ignore")

METHODS = ("raw_pnl", "raw_sharpe", "wilcoxon", "ttest")
STATISTICAL_METHODS = ("wilcoxon", "ttest")
SUMMARY_COLUMNS = (
    "method", "variant", "lags", "equity_start", "equity_end",
    "return_pct", "annual_return_pct", "annual_volatility_pct",
    "sharpe_ratio", "calmar_ratio", "max_drawdown_pct", "win_rate",
    "trades",
)


# ============================================================
# Mathematical and Bayesian utilities
# ============================================================

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

    cov = np.asarray(cov, dtype=np.float64)

    if cov.shape != (d, d):
        cov = np.eye(d) * float(np.nanmean(cov))

    cov = np.nan_to_num(
        cov,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return cov + reg * np.eye(d)


def logpdf_multivariate_normal(x, mean, cov):
    """Evaluate a multivariate normal log-density robustly."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    d = len(x)

    try:
        sign, logdet = np.linalg.slogdet(cov)

        if sign <= 0 or not np.isfinite(logdet):
            cov = cov + 1e-5 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)

        diff = x - mean
        quad = diff @ np.linalg.pinv(cov) @ diff

        return float(
            -0.5 * (
                d * np.log(2.0 * np.pi)
                + logdet
                + quad
            )
        )

    except Exception:
        return -1e12


def logpdf_multivariate_student_t(x, mean, cov, df=5.0):
    """Evaluate a multivariate Student-t log-density robustly."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    d = len(x)
    nu = float(df)

    try:
        sign, logdet = np.linalg.slogdet(cov)

        if sign <= 0 or not np.isfinite(logdet):
            cov = cov + 1e-5 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)

        diff = x - mean
        delta = diff @ np.linalg.pinv(cov) @ diff

        return float(
            lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log(1.0 + delta / nu)
        )

    except Exception:
        return -1e12


# ============================================================
# Sample construction
# ============================================================

def build_samples(
    prices,
    dates,
    lags,
    predict_day=1,
    include_maxlag_skew=False
):
    """Build lag features and one-step direction labels."""
    prices = np.asarray(prices, dtype=np.float64)
    dates = np.asarray(dates)

    lags = sorted(set(int(x) for x in lags))
    max_lag = max(lags)

    rows = []

    for t in range(max_lag, len(prices) - predict_day):
        p0 = prices[t]
        p1 = prices[t + predict_day]

        if not (
            np.isfinite(p0)
            and np.isfinite(p1)
            and p0 > 0
            and p1 > 0
        ):
            continue

        feature_vector = []
        valid = True

        for lag in lags:
            historical_price = prices[t - lag]

            if not (
                np.isfinite(historical_price)
                and historical_price > 0
            ):
                valid = False
                break

            feature_vector.append(p0 / historical_price)

        if not valid:
            continue

        if include_maxlag_skew:
            window_prices = prices[t - max_lag:t + 1]

            if len(window_prices) >= 3:
                skew_value = skew(
                    window_prices,
                    bias=False,
                    nan_policy="omit"
                )

                if not np.isfinite(skew_value):
                    skew_value = 0.0
            else:
                skew_value = 0.0

            feature_vector.append(float(skew_value))

        row = {
            "date": dates[t],
            "today_price": float(p0),
            "future_price": float(p1),
            "label": int(p1 > p0),
        }

        for index, value in enumerate(feature_vector):
            row[f"x_{index}"] = float(value)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Bayesian classifier
# ============================================================

def fit_params(X_window, y_window, reg=1e-6):
    """Fit class priors, means, and regularized covariance matrices."""
    X_window = np.asarray(X_window, dtype=np.float64)
    y_window = np.asarray(y_window, dtype=int)

    n = len(y_window)
    n_up = int((y_window == 1).sum())
    n_down = int((y_window == 0).sum())

    params = {
        "prior_up": (n_up + 1.0) / (n + 2.0),
        "prior_down": (n_down + 1.0) / (n + 2.0),
        "valid": True,
    }

    for cls, key in [(1, "up"), (0, "down")]:
        class_mask = y_window == cls

        if class_mask.sum() > 0:
            class_data = X_window[class_mask]
        else:
            class_data = X_window

        params[f"mean_{key}"] = class_data.mean(axis=0)
        params[f"cov_{key}"] = regularized_covariance(
            class_data,
            reg=reg
        )

    for key in [
        "mean_up",
        "mean_down",
        "cov_up",
        "cov_down",
    ]:
        if not np.all(np.isfinite(params[key])):
            params["valid"] = False

    return params


def predict_proba(
    x,
    params,
    dist="student",
    student_df=10.0
):
    """Return the predicted up and down probabilities for one sample."""
    if not params["valid"]:
        return 0.5, 0.5

    if dist == "normal":
        pdf_function = logpdf_multivariate_normal
    else:
        pdf_function = lambda a, b, c: logpdf_multivariate_student_t(
            a,
            b,
            c,
            student_df
        )

    log_prob_up = (
        safe_log(params["prior_up"])
        + pdf_function(
            x,
            params["mean_up"],
            params["cov_up"]
        )
    )

    log_prob_down = (
        safe_log(params["prior_down"])
        + pdf_function(
            x,
            params["mean_down"],
            params["cov_down"]
        )
    )

    maximum = max(log_prob_up, log_prob_down)

    exp_up = np.exp(log_prob_up - maximum)
    exp_down = np.exp(log_prob_down - maximum)
    total = exp_up + exp_down

    if total <= 0 or not np.isfinite(total):
        return 0.5, 0.5

    return (
        float(exp_up / total),
        float(exp_down / total)
    )


# ============================================================
# Rolling trading simulation
# ============================================================

def rolling_trade(
    samples,
    n_features,
    window_size=250,
    predict_day=1,
    dist="student",
    student_df=10.0,
    reg=1e-6,
    prob_threshold=0.9,
    commission=0.0
):
    """Run the rolling classifier and return signals, PnL, and positions."""
    feature_columns = [
        f"x_{index}"
        for index in range(n_features)
    ]

    X = samples[feature_columns].values.astype(np.float64)
    y = samples["label"].values.astype(int)
    prices = samples["today_price"].values.astype(np.float64)

    n = len(samples)

    probabilities = np.full(n, np.nan)
    signals = ["HOLD"] * n

    for j in range(n):
        end = j - predict_day
        start = end - window_size

        if start < 0 or end <= 0:
            continue

        X_window = X[start:end]
        y_window = y[start:end]

        finite_mask = np.all(np.isfinite(X_window), axis=1)

        X_window = X_window[finite_mask]
        y_window = y_window[finite_mask]

        minimum_samples = max(10, n_features + 2)

        if len(y_window) < minimum_samples:
            continue

        if not np.all(np.isfinite(X[j])):
            continue

        params = fit_params(
            X_window,
            y_window,
            reg=reg
        )

        prob_up, prob_down = predict_proba(
            X[j],
            params,
            dist=dist,
            student_df=student_df
        )

        probabilities[j] = prob_up

        if prob_up > prob_threshold:
            signals[j] = "BUY"
        elif prob_down > prob_threshold:
            signals[j] = "SELL"
        else:
            signals[j] = "HOLD"

    pnl = np.zeros(n, dtype=np.float64)
    positions = np.zeros(n, dtype=int)

    current_position = 0

    for j in range(1, n):
        signal = signals[j - 1]

        today_price = prices[j]
        previous_price = prices[j - 1]

        if not (
            np.isfinite(today_price)
            and np.isfinite(previous_price)
            and previous_price > 0
        ):
            positions[j] = current_position
            continue

        price_return = (
            today_price - previous_price
        ) / previous_price

        new_position = current_position

        if signal == "BUY":
            new_position = 1
        elif signal == "SELL":
            new_position = -1

        bar_pnl = 0.0

        if current_position == 1:
            bar_pnl = price_return
        elif current_position == -1:
            bar_pnl = -price_return

        if new_position != current_position:
            if current_position == 0 or new_position == 0:
                number_of_trades = 1
            else:
                number_of_trades = 2

            bar_pnl -= number_of_trades * commission

        pnl[j] = bar_pnl
        positions[j] = new_position
        current_position = new_position

    result = samples.copy()

    result["signal"] = signals
    result["prob_up"] = probabilities
    result["position"] = positions
    result["bar_pnl"] = pnl
    result["cumulative_pnl"] = np.cumsum(pnl)

    return result


# ============================================================
# Date split
# ============================================================

def split_df(
    df,
    train_start,
    train_end,
    test_start,
    test_end
):
    """Assign samples to train, test, or unused date ranges."""
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])

    result["split"] = "unused"

    train_mask = (
        (result["date"] >= pd.to_datetime(train_start))
        & (result["date"] < pd.to_datetime(train_end))
    )

    test_mask = (
        (result["date"] >= pd.to_datetime(test_start))
        & (result["date"] < pd.to_datetime(test_end))
    )

    result.loc[train_mask, "split"] = "train"
    result.loc[test_mask, "split"] = "test"

    return result


# ============================================================
# Random lag generation
# ============================================================

def generate_random_lag_sets(
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


# ============================================================
# Lag search utilities
# ============================================================

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


def run_one_lag_configuration(
    raw_data,
    lags,
    include_maxlag_skew,
    args
):
    """Evaluate one lag configuration and attach its equity series."""
    prices = raw_data["price"].values.astype(np.float64)
    dates = raw_data["date"].values

    samples = build_samples(
        prices=prices,
        dates=dates,
        lags=lags,
        predict_day=args.predict_day,
        include_maxlag_skew=include_maxlag_skew
    )

    if samples.empty:
        return None

    samples = split_df(
        samples,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end
    )

    train_count = int(
        (samples["split"] == "train").sum()
    )

    test_count = int(
        (samples["split"] == "test").sum()
    )

    minimum_required = (
        args.window_size
        + max(lags)
        + args.predict_day
        + 10
    )

    if train_count == 0 or test_count == 0:
        return None

    if len(samples) < minimum_required:
        return None

    n_features = len(lags)

    if include_maxlag_skew:
        n_features += 1

    result = rolling_trade(
        samples=samples,
        n_features=n_features,
        window_size=args.window_size,
        predict_day=args.predict_day,
        dist=args.dist,
        student_df=args.student_df,
        reg=args.reg,
        prob_threshold=args.prob_threshold,
        commission=args.commission
    )

    result["equity"] = (
        args.initial_capital
        * (1.0 + result["bar_pnl"]).cumprod()
    )

    result["lags"] = str(tuple(lags))
    result["include_maxlag_skew"] = bool(
        include_maxlag_skew
    )

    return result


def get_train_returns(result):
    """Return train-period bar returns from a strategy result."""
    train = result[
        result["split"] == "train"
    ].copy()

    if train.empty:
        return np.array([], dtype=np.float64)

    return train["bar_pnl"].values.astype(np.float64)


# ---- برای استفاده در Pool: ارزیابی یک مجموعه لَگ ----
def _evaluate_single_lag(args_tuple):
    """Evaluate one lag set for multiprocessing lag selection."""
    raw_data, lags, include_maxlag_skew, args = args_tuple
    try:
        result = run_one_lag_configuration(
            raw_data=raw_data,
            lags=lags,
            include_maxlag_skew=include_maxlag_skew,
            args=args
        )
        if result is None or result.empty:
            return None

        train_returns = get_train_returns(result)
        if len(train_returns) == 0:
            return None

        total_pnl = float(np.sum(train_returns))

        if len(train_returns) > 1:
            standard_deviation = float(np.std(train_returns, ddof=1))
        else:
            standard_deviation = 0.0

        if standard_deviation > 0:
            sharpe = (
                np.mean(train_returns)
                / standard_deviation
                * np.sqrt(args.annualization)
            )
        else:
            sharpe = -np.inf

        return {
            "lags": tuple(lags),
            "returns": train_returns,
            "total_pnl": total_pnl,
            "sharpe": float(sharpe),
        }
    except Exception:
        return None


def select_best_lags(
    raw_data,
    lag_sets,
    include_maxlag_skew,
    args
):
    """Select lags using raw scores, Wilcoxon, and paired t-test."""
    # استفاده از Pool برای موازی‌سازی روی lag_sets
    worker_args = [
        (raw_data, lags, include_maxlag_skew, args)
        for lags in lag_sets
    ]

    # تعداد هسته‌ها
    n_workers = args.n_workers if getattr(args, "n_workers", None) else mp.cpu_count()

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_evaluate_single_lag, worker_args)

    evaluations = [r for r in results if r is not None]

    if not evaluations:
        return {}

    methods = METHODS

    selected = {}

    selected["raw_pnl"] = max(
        evaluations,
        key=lambda item: item["total_pnl"]
    )["lags"]

    selected["raw_sharpe"] = max(
        evaluations,
        key=lambda item: item["sharpe"]
    )["lags"]

    detailed_scores = []

    for current in evaluations:
        other_returns = []

        for other in evaluations:
            if other is current:
                continue

            other_returns.extend(
                other["returns"].tolist()
            )

        other_returns = np.asarray(
            other_returns,
            dtype=np.float64
        )

        first_returns = current["returns"]

        minimum_length = min(
            len(first_returns),
            len(other_returns)
        )

        if minimum_length >= 5:
            first = first_returns[:minimum_length]
            second = other_returns[:minimum_length]

            p_values = {
                method: safe_statistical_test(
                    method,
                    first,
                    second
                )
                for method in STATISTICAL_METHODS
            }
        else:
            p_values = {
                method: 1.0
                for method in STATISTICAL_METHODS
            }

        scores = {
            method: current["total_pnl"]
            - 100.0 * p_values[method]
            for method in STATISTICAL_METHODS
        }

        detailed_scores.append({
            "lags": current["lags"],
            "total_pnl": current["total_pnl"],
            **scores,
        })

    for method in STATISTICAL_METHODS:
        selected[method] = max(
            detailed_scores,
            key=lambda item: item[method]
        )["lags"]

    return selected


# ============================================================
# Performance metrics
# ============================================================

def compute_max_drawdown(equity):
    """Return the maximum drawdown as a negative fraction."""
    equity = pd.Series(equity).astype(float)

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    return float(drawdown.min())


def compute_metrics(
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


def write_metrics_summary(rows, output_path):
    """Write the consolidated test metrics CSV without console output."""
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(output_path, index=False)


# ============================================================
# Data loading
# ============================================================

def load_price_data(data_path):
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


# ============================================================
# Main
# ============================================================

# برای اجرای موازی روی (variant, method): worker
def _run_configuration_worker(args_tuple):
    """Run one selected variant/method configuration silently."""
    data, variant_name, include_skew, method, lags, args = args_tuple
    try:
        result = run_one_lag_configuration(
            raw_data=data,
            lags=lags,
            include_maxlag_skew=include_skew,
            args=args
        )
        return (variant_name, method, lags, result)
    except Exception:
        return (variant_name, method, lags, None)


def main(args):
    """Run selected configurations silently and export one metrics CSV."""
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    data = load_price_data(data_path)

    lag_sets = generate_random_lag_sets(
        n_sets=args.n_lag_sets,
        min_lags=args.min_lags,
        max_lags=args.max_lags,
        min_lag_value=args.min_lag_value,
        max_lag_value=args.max_lag_value,
        seed=args.random_seed
    )

    if not lag_sets:
        raise RuntimeError(
            "No lag sets were generated."
        )

    methods = METHODS

    result_map = {
        method: {}
        for method in methods
    }

    metrics_rows = []

    variants = [
        ("regular", False),
        ("maxlag_skew", True),
    ]

    # تعداد هسته‌ها برای Pool
    n_workers = args.n_workers if args.n_workers is not None else mp.cpu_count()

    # 1) انتخاب لَگ‌ها برای هر variant (با Pool داخل select_best_lags انجام می‌شود)
    selected_lags_all = {}

    for variant_name, include_skew in variants:
        selected_lags = select_best_lags(
            raw_data=data,
            lag_sets=lag_sets,
            include_maxlag_skew=include_skew,
            args=args
        )

        selected_lags_all[variant_name] = selected_lags

        if not selected_lags:
            continue

        for method in methods:
            lags = selected_lags.get(
                method,
                tuple(args.lags)
            )


    # 2) اجرای نهایی همه‌ی ترکیب‌های (variant, method) با Pool
    worker_tasks = []
    for variant_name, include_skew in variants:
        selected_lags = selected_lags_all.get(variant_name, {})
        if not selected_lags:
            continue
        for method in methods:
            lags = selected_lags.get(
                method,
                tuple(args.lags)
            )
            worker_tasks.append(
                (data, variant_name, include_skew, method, lags, args)
            )

    if worker_tasks:
        with mp.Pool(processes=n_workers) as pool:
            worker_results = pool.map(_run_configuration_worker, worker_tasks)
    else:
        worker_results = []

    # جمع‌آوری خروجی‌ها
    methods_set = set(methods)
    for variant_name, method, lags, result in worker_results:
        if result is None or result.empty or method not in methods_set:
            continue

        result["method"] = method
        result["variant"] = variant_name

        result_map[method][variant_name] = result

        train_metrics = compute_metrics(
            result,
            split_name="train",
            annualization=args.annualization
        )

        test_metrics = compute_metrics(
            result,
            split_name="test",
            annualization=args.annualization
        )

        if test_metrics:
            test_metrics["method"] = method
            test_metrics["variant"] = variant_name
            test_metrics["lags"] = str(tuple(lags))
            metrics_rows.append(test_metrics)

    metrics_path = (
        output_dir
        / "metrics_summary.csv"
    )

    write_metrics_summary(metrics_rows, metrics_path)



# ============================================================
# CLI arguments
# ============================================================

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(
        description=(
            "Bayesian rolling strategy with random "
            "lag selection and max-lag skew comparison "
            "(multiprocessing version)."
        )
    )

    # Same dataset and date ranges as the first code
    parser.add_argument(
        "--data-path",
        type=str,
        default="BTCUSD_H4.pkl"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--train-start",
        type=str,
        default="2020-01-01"
    )

    parser.add_argument(
        "--train-end",
        type=str,
        default="2021-07-01"
    )

    parser.add_argument(
        "--test-start",
        type=str,
        default="2021-07-01"
    )

    parser.add_argument(
        "--test-end",
        type=str,
        default="2023-03-01"
    )

    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 7, 24]
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=80
    )

    parser.add_argument(
        "--predict-day",
        type=int,
        default=1
    )

    parser.add_argument(
        "--dist",
        type=str,
        choices=["normal", "student"],
        default="student"
    )

    parser.add_argument(
        "--student-df",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--reg",
        type=float,
        default=1e-6
    )

    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--commission",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--annualization",
        type=int,
        default=365
    )

    # Random lag search parameters
    parser.add_argument(
        "--n-lag-sets",
        type=int,
        default=100
    )

    parser.add_argument(
        "--min-lags",
        type=int,
        default=2
    )

    parser.add_argument(
        "--max-lags",
        type=int,
        default=2
    )

    parser.add_argument(
        "--min-lag-value",
        type=int,
        default=1
    )

    parser.add_argument(
        "--max-lag-value",
        type=int,
        default=30
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42
    )

    # تعداد پردازنده‌ها برای Pool
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes for multiprocessing Pool (default: cpu_count)"
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(args.data_path).parent
            / "bayesian_lag_search_results_mp"
        )

    if any(lag <= 0 for lag in args.lags):
        raise ValueError(
            "All lags must be positive."
        )

    if args.train_start >= args.train_end:
        raise ValueError(
            "train-start must be earlier than train-end."
        )

    if args.test_start >= args.test_end:
        raise ValueError(
            "test-start must be earlier than test-end."
        )

    if args.train_end > args.test_start:
        raise ValueError(
            "Training and test periods must not overlap."
        )

    main(args)
