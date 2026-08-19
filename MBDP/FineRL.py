"""Silent Bayesian lag-search strategy with consolidated CSV metrics."""

from pathlib import Path
from math import lgamma
import warnings, multiprocessing as mp
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, ttest_rel, skew
import random

warnings.filterwarnings("ignore")

METHODS = ("raw_pnl", "raw_sharpe", "wilcoxon", "ttest")
STATISTICAL_METHODS = ("wilcoxon", "ttest")
SUMMARY_COLUMNS = (
    "method", "variant", "lags", "equity_start", "equity_end",
    "return_pct", "annual_return_pct", "annual_volatility_pct",
    "sharpe_ratio", "calmar_ratio", "max_drawdown_pct", "win_rate",
    "trades",
)

# -------------------------------------------------------------------------
# توابع ریاضی و بیزین
# -------------------------------------------------------------------------
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

def build_samples(prices, dates, lags, predict_day=1, include_maxlag_skew=False):
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

def rolling_trade(df, n_features, window_size=120, predict_day=1, dist="normal", df_student=5., reg=1e-6,
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

def run_one_ticker_with_lag(df_ticker, ticker, price_col="Adj Close", lags=(1, 7, 25), include_maxlag_skew=False,
                            window_size=300, predict_day=1, dist="student", student_df=10., reg=1e-6, prob_threshold=.9,
                            commission=0., initial_capital=10000., train_start="2009-04-01", train_end="2019-07-01",
                            test_start="2021-07-01", test_end="2022-04-01"):
    """Run one ticker configuration and attach its selected-lag metadata."""
    df_ticker = df_ticker.copy()
    df_ticker["Date"] = pd.to_datetime(df_ticker.Date)
    df_ticker = df_ticker.sort_values("Date").reset_index(drop=True)
    prices = pd.to_numeric(df_ticker[price_col], errors="coerce").values.astype(float)
    samples = build_samples(prices, df_ticker.Date.values, list(lags), predict_day, include_maxlag_skew)
    if len(samples) < window_size + max(lags) + predict_day + 10:
        return None
    samples = split_df(samples, train_start, train_end, test_start, test_end)
    n_features = len(lags) + (1 if include_maxlag_skew else 0)
    result = rolling_trade(samples, n_features, window_size, predict_day, dist, student_df, reg, prob_threshold,
                           commission)
    result["equity"] = initial_capital * (1 + result.bar_pnl).cumprod()
    result["Ticker"] = ticker
    result["lags"] = str(tuple(lags))
    result["include_maxlag_skew"] = bool(include_maxlag_skew)
    return result

# -------------------------------------------------------------------------
# ارزیابی ست‌های Lag در داده Train
# -------------------------------------------------------------------------
def run_multi_test_lag_search(df_ticker, ticker, lag_sets, kwargs, include_maxlag_skew=False):
    """Select lag combinations using raw scores and supported paired tests."""
    eval_results = []
    for lags in lag_sets:
        try:
            kw = kwargs.copy()
            kw["lags"] = lags
            kw["include_maxlag_skew"] = include_maxlag_skew
            result = run_one_ticker_with_lag(df_ticker, ticker, **kw)

            if result is not None and not result.empty:
                train_data = result[result.split == "train"].copy()
                if len(train_data) > 0:
                    daily_returns = train_data.bar_pnl.values
                    total_pnl = train_data.bar_pnl.sum()
                    std_dev = np.std(daily_returns) if len(daily_returns) > 1 else 0
                    sharpe = (np.mean(daily_returns) / std_dev * np.sqrt(252)) if std_dev > 0 else -np.inf

                    eval_results.append({
                        "lags": lags,
                        "daily_returns": daily_returns,
                        "total_pnl": total_pnl,
                        "sharpe": sharpe
                    })
        except Exception:
            continue

    if not eval_results:
        return {}

    best_raw_pnl = max(eval_results, key=lambda x: x["total_pnl"])["lags"]
    best_raw_sharpe = max(eval_results, key=lambda x: x["sharpe"])["lags"]

    selected_lags = {
        "raw_pnl": best_raw_pnl,
        "raw_sharpe": best_raw_sharpe
    }

    detailed_metrics = []
    for i, res in enumerate(eval_results):
        other_returns = []
        for j, other in enumerate(eval_results):
            if i != j:
                other_returns.extend(other["daily_returns"])

        min_len = min(len(res["daily_returns"]), len(other_returns))
        p_vals = {}

        if min_len >= 5 and len(other_returns) > 0:
            r1 = res["daily_returns"][:min_len]
            r2 = other_returns[:min_len]

            try:
                _, p_vals["wilcoxon"] = wilcoxon(r1, r2)
            except:
                p_vals["wilcoxon"] = 1.0

            try:
                _, p_vals["ttest"] = ttest_rel(r1, r2)
            except:
                p_vals["ttest"] = 1.0

        else:
            p_vals = {m: 1.0 for m in STATISTICAL_METHODS}

        entry = {"lags": res["lags"], "total_pnl": res["total_pnl"]}
        for m in STATISTICAL_METHODS:
            entry[f"score_{m}"] = res["total_pnl"] - p_vals[m] * 100
        detailed_metrics.append(entry)

    for m in STATISTICAL_METHODS:
        best_item = max(detailed_metrics, key=lambda x: x[f"score_{m}"])
        selected_lags[m] = best_item["lags"]

    return selected_lags

# -------------------------------------------------------------------------
# پردازش‌گر Ticker برای هر دو حالت
# -------------------------------------------------------------------------
def _worker(args):
    """Evaluate regular and max-lag-skew variants for one ticker."""
    df_ticker, ticker, kwargs, lag_sets = args
    output = {
        "ticker": ticker,
        "base_lags": None,
        "base_results": {},
        "skew_lags": None,
        "skew_results": {},
        "error": None
    }
    try:
        # 1. حالت بردار معمولی (بدون Skewness)
        base_lags = run_multi_test_lag_search(df_ticker, ticker, lag_sets, kwargs, include_maxlag_skew=False)
        if not base_lags:
            base_lags = {m: kwargs["lags"] for m in METHODS}
        output["base_lags"] = base_lags
        for m, lags in base_lags.items():
            kw = kwargs.copy()
            kw["lags"] = lags
            kw["include_maxlag_skew"] = False
            res = run_one_ticker_with_lag(df_ticker, ticker, **kw)
            if res is not None:
                output["base_results"][m] = res

        # 2. حالت بردار به همراه Skewness در بازه [t - max(lag), t]
        skew_lags = run_multi_test_lag_search(df_ticker, ticker, lag_sets, kwargs, include_maxlag_skew=True)
        if not skew_lags:
            skew_lags = {m: kwargs["lags"] for m in METHODS}
        output["skew_lags"] = skew_lags
        for m, lags in skew_lags.items():
            kw = kwargs.copy()
            kw["lags"] = lags
            kw["include_maxlag_skew"] = True
            res = run_one_ticker_with_lag(df_ticker, ticker, **kw)
            if res is not None:
                output["skew_results"][m] = res

        return output
    except Exception as e:
        output["error"] = str(e)
        return output

# -------------------------------------------------------------------------
# ساخت پرتفوی واحد از کلیه سهم‌ها
# -------------------------------------------------------------------------
def build_portfolio_for_method(results_list, initial_capital=1_000_000.):
    """Build a date-indexed portfolio equity curve from ticker results."""
    items = []
    for df in results_list:
        train = df[df.split == "train"].copy()
        test = df[df.split == "test"].copy()
        if train.empty or test.empty:
            continue
        ticker = df.Ticker.iloc[0]
        start = float(train.equity.iloc[0])
        end = float(train.equity.iloc[-1])
        if start <= 0:
            continue
        ret = end / start - 1
        items.append({"Ticker": ticker, "train_return": ret, "test_df": test[["date", "bar_pnl"]].copy()})

    if not items:
        return None

    weights = pd.DataFrame([{k: v for k, v in x.items() if k != "test_df"} for x in items])
    weights["weight_base"] = weights.train_return.clip(lower=0)
    weights["weight"] = 1 / len(
        weights) if weights.weight_base.sum() <= 0 else weights.weight_base / weights.weight_base.sum()
    weights["allocated_capital"] = initial_capital * weights.weight

    dates = sorted(set(pd.to_datetime(pd.concat([x["test_df"].date for x in items], ignore_index=True))))
    mapping = {x["Ticker"]: x for x in items}

    for _, row in weights.iterrows():
        ticker = row.Ticker
        alloc = float(row.allocated_capital)
        test = mapping[ticker]["test_df"].copy()
        test.date = pd.to_datetime(test.date)
        test = test.sort_values("date").reset_index(drop=True)
        test["equity_alloc"] = alloc * (1 + test.bar_pnl).cumprod()
        mapping[ticker]["equity_series"] = test[["date", "equity_alloc"]]

    portfolio = []
    for dt in dates:
        total = 0.
        for _, row in weights.iterrows():
            ticker = row.Ticker
            alloc = float(row.allocated_capital)
            eq = mapping[ticker]["equity_series"]
            sub = eq[eq.date <= dt]
            total += alloc if sub.empty else float(sub.equity_alloc.iloc[-1])
        portfolio.append({"date": dt, "equity": total})

    equity_df = pd.DataFrame(portfolio).sort_values("date").reset_index(drop=True)
    return equity_df


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

# -------------------------------------------------------------------------
# تابع اصلی
# -------------------------------------------------------------------------
def main():
    """Run both variants silently and write one test-metrics CSV."""
    data_path = Path(r"C:\Users\Ali\PycharmProjects\PhD_Papers\Second_Paper\djia_data\djia30_full_2009_2022.csv")
    output_dir = Path(r"C:\Users\Ali\PycharmProjects\PhD_Papers\Second_Paper\temp\djia_skew_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_capital = 1_000_000.

    kwargs = {
        "price_col": "Adj Close", "lags": (7, 30), "window_size": 250,
        "predict_day": 1, "dist": "student", "student_df": 10., "reg": 1e-6,
        "prob_threshold": .9, "commission": 0., "initial_capital": 10_000.,
        "train_start": "2019-07-01", "train_end": "2020-07-01",
        "test_start": "2020-07-01", "test_end": "2022-03-31"
    }

    lag_sets = generate_random_lag_sets(n_sets=100, min_lags=2, max_lags=5, min_lag_val=5, max_lag_val=50)

    df = pd.read_csv(data_path)
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df.Date, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    tickers = sorted(df.Ticker.dropna().astype(str).unique())
    tasks = [(df[df.Ticker.astype(str) == t].sort_values("Date").reset_index(drop=True), t, kwargs, lag_sets) for t in
             tickers]

    base_results_map = {m: [] for m in METHODS}
    skew_results_map = {m: [] for m in METHODS}

    cores = mp.cpu_count()

    with mp.Pool(cores) as pool:
        for out in pool.imap_unordered(_worker, tasks):
            if out["error"]:
                continue

            for m in METHODS:
                if m in out["base_results"]:
                    base_results_map[m].append(out["base_results"][m])
                if m in out["skew_results"]:
                    skew_results_map[m].append(out["skew_results"][m])

    # -------------------------------------------------------------------------
    # ترسیم نمودار مقایسه‌ای
    # -------------------------------------------------------------------------
    rows = []
    for method in METHODS:
        for variant, result_list in (
            ("regular", base_results_map[method]),
            ("maxlag_skew", skew_results_map[method]),
        ):
            portfolio = build_portfolio_for_method(result_list, initial_capital)
            metrics = compute_portfolio_metrics(result_list, portfolio)
            if not metrics:
                continue
            rows.append({
                "method": method,
                "variant": variant,
                "lags": "; ".join(sorted({
                    str(lag)
                    for result in result_list
                    for lag in result["lags"].dropna().unique()
                })),
                **metrics,
            })

    write_metrics_summary(rows, output_dir / "metrics_summary.csv")

if __name__ == "__main__":
    mp.freeze_support()
    main()
