from pathlib import Path
from math import lgamma, log, exp, pi
import warnings, multiprocessing as mp
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, ttest_rel, skew
import random
from numba import njit

warnings.filterwarnings("ignore")

METHODS = ("raw_pnl", "raw_sharpe", "wilcoxon", "ttest")
STATISTICAL_METHODS = ("wilcoxon", "ttest")
SUMMARY_COLUMNS = (
    "method", "variant", "lags", "equity_start", "equity_end",
    "return_pct", "annual_return_pct", "annual_volatility_pct",
    "sharpe_ratio", "calmar_ratio", "max_drawdown_pct", "win_rate",
    "trades",
)


# Numba JIT kernels copied from fast_drl.py.
@njit(cache=True)
def _nb_skew_unbiased(w):
    """Unbiased sample skewness matching scipy.stats.skew(w, bias=False)."""
    n = len(w)
    if n < 3:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += w[i]
    mean /= n

    m2 = 0.0
    m3 = 0.0
    for i in range(n):
        d = w[i] - mean
        d2 = d * d
        m2 += d2
        m3 += d2 * d
    m2 /= n
    m3 /= n

    if m2 <= 1e-18:
        return 0.0
    g1 = m3 / (m2 ** 1.5)
    val = (np.sqrt(n * (n - 1.0)) / (n - 2.0)) * g1
    return val if np.isfinite(val) else 0.0


@njit(cache=True)
def _nb_regularized_cov(X, reg):
    """Regularized covariance matrix matching np.cov(X, rowvar=False) + reg * I."""
    n, d = X.shape
    cov = np.zeros((d, d), dtype=np.float64)
    if n <= 1:
        for i in range(d):
            cov[i, i] = reg
        return cov

    means = np.zeros(d, dtype=np.float64)
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i, j]
        means[j] = s / n

    inv_n1 = 1.0 / (n - 1.0)
    for j in range(d):
        for k in range(j, d):
            s = 0.0
            for i in range(n):
                s += (X[i, j] - means[j]) * (X[i, k] - means[k])
            val = s * inv_n1
            cov[j, k] = val
            cov[k, j] = val

    for j in range(d):
        cov[j, j] += reg
    return cov


@njit(cache=True)
def _nb_logpdf_mvn(x, mean, cov):
    """Numba-compiled Multivariate Normal log-density."""
    d = len(x)
    c = cov.copy()
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0.0:
        for i in range(d):
            c[i, i] += 1e-5
        sign, logdet = np.linalg.slogdet(c)
        if sign <= 0.0:
            return -1e12
    diff = x - mean
    pinv_c = np.linalg.pinv(c)
    quad = np.dot(diff, np.dot(pinv_c, diff))
    res = -0.5 * (d * log(2.0 * pi) + logdet + quad)
    return res if np.isfinite(res) else -1e12


@njit(cache=True)
def _nb_logpdf_mvt(x, mean, cov, df):
    """Numba-compiled Multivariate Student-t log-density."""
    d = len(x)
    nu = float(df)
    c = cov.copy()
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0.0:
        for i in range(d):
            c[i, i] += 1e-5
        sign, logdet = np.linalg.slogdet(c)
        if sign <= 0.0:
            return -1e12
    diff = x - mean
    pinv_c = np.linalg.pinv(c)
    delta = np.dot(diff, np.dot(pinv_c, diff))
    if 1.0 + delta / nu <= 0.0:
        return -1e12
    res = (
        lgamma((nu + d) * 0.5)
        - lgamma(nu * 0.5)
        - 0.5 * d * log(nu * pi)
        - 0.5 * logdet
        - 0.5 * (nu + d) * log(1.0 + delta / nu)
    )
    return res if np.isfinite(res) else -1e12


@njit(cache=True)
def _nb_fit_predict(X_w, y_w, x_eval, reg, is_normal, df):
    """Fit Bayesian class parameters and predict probabilities in one compiled step."""
    n, d = X_w.shape
    up = 0
    for i in range(n):
        if y_w[i] == 1:
            up += 1
    down = n - up

    prior_up = (up + 1.0) / (n + 2.0)
    prior_down = (down + 1.0) / (n + 2.0)

    # Class 1 (Up)
    if up > 0:
        X_up = np.empty((up, d), dtype=np.float64)
        idx = 0
        for i in range(n):
            if y_w[i] == 1:
                for j in range(d):
                    X_up[idx, j] = X_w[i, j]
                idx += 1
    else:
        X_up = X_w

    mean_up = np.zeros(d, dtype=np.float64)
    for j in range(d):
        s = 0.0
        for i in range(X_up.shape[0]):
            s += X_up[i, j]
        mean_up[j] = s / X_up.shape[0]
    cov_up = _nb_regularized_cov(X_up, reg)

    # Class 0 (Down)
    if down > 0:
        X_down = np.empty((down, d), dtype=np.float64)
        idx = 0
        for i in range(n):
            if y_w[i] == 0:
                for j in range(d):
                    X_down[idx, j] = X_w[i, j]
                idx += 1
    else:
        X_down = X_w

    mean_down = np.zeros(d, dtype=np.float64)
    for j in range(d):
        s = 0.0
        for i in range(X_down.shape[0]):
            s += X_down[i, j]
        mean_down[j] = s / X_down.shape[0]
    cov_down = _nb_regularized_cov(X_down, reg)

    if not (np.all(np.isfinite(mean_up)) and np.all(np.isfinite(mean_down)) and
            np.all(np.isfinite(cov_up)) and np.all(np.isfinite(cov_down))):
        return 0.5, 0.5, False

    p_up_safe = prior_up if prior_up > 1e-12 else 1e-12
    p_down_safe = prior_down if prior_down > 1e-12 else 1e-12

    if is_normal:
        lu = log(p_up_safe) + _nb_logpdf_mvn(x_eval, mean_up, cov_up)
        ld = log(p_down_safe) + _nb_logpdf_mvn(x_eval, mean_down, cov_down)
    else:
        lu = log(p_up_safe) + _nb_logpdf_mvt(x_eval, mean_up, cov_up, df)
        ld = log(p_down_safe) + _nb_logpdf_mvt(x_eval, mean_down, cov_down, df)

    m = lu if lu > ld else ld
    eu = exp(lu - m)
    ed = exp(ld - m)
    s = eu + ed
    if s <= 0.0 or not np.isfinite(s):
        return 0.5, 0.5, True
    return eu / s, ed / s, True


@njit(cache=True)
def _nb_wf_positions_kernel(prices, lags_arr, include_skew, window_size, reg, is_normal, student_df, prob_threshold):
    """Compiled walk-forward feature extraction, online fitting, and position tracking."""
    n = len(prices)
    n_lags = len(lags_arr)
    d = n_lags + (1 if include_skew else 0)
    X = np.full((n, d), np.nan, dtype=np.float64)

    m = 0
    for i in range(n_lags):
        if lags_arr[i] > m:
            m = lags_arr[i]

    for t in range(m, n):
        p_t = prices[t]
        for k in range(n_lags):
            X[t, k] = p_t / prices[t - lags_arr[k]]
        if include_skew:
            if m >= 2:
                X[t, d - 1] = _nb_skew_unbiased(prices[t - m: t + 1])
            else:
                X[t, d - 1] = 0.0

    labels = np.empty(n - 1, dtype=np.int64)
    for t in range(n - 1):
        labels[t] = 1 if prices[t + 1] > prices[t] else 0

    pos = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    state = 0.0
    error_flag = False

    start_t = m + window_size + 2
    for t in range(start_t, n):
        end = t - 2
        X_w = X[end - window_size: end]
        y_w = labels[end - window_size: end]
        up, down, is_valid = _nb_fit_predict(X_w, y_w, X[t - 1], reg, is_normal, student_df)
        if not is_valid or not (np.isfinite(up) and np.isfinite(down)):
            error_flag = True
            break
        if up > prob_threshold:
            state = 1.0
        elif down > prob_threshold:
            state = -1.0
        pos[t] = state
        valid[t] = True

    return pos, valid, error_flag


@njit(cache=True)
def _nb_rolling_trade_kernel(X, y, prices, n_features, window_size, predict_day, is_normal, df_student, reg, prob_threshold, commission):
    """Compiled rolling trade loop for legacy functions."""
    n = len(prices)
    probs = np.full(n, np.nan, dtype=np.float64)
    sig_codes = np.zeros(n, dtype=np.int64)  # 0: HOLD, 1: BUY, -1: SELL
    min_samples = 10 if 10 > n_features + 2 else n_features + 2

    for j in range(n):
        end = j - predict_day
        start = end - window_size
        if start < 0 or end <= 0:
            continue

        # Check finite X[j]
        xj_ok = True
        for k in range(n_features):
            if not np.isfinite(X[j, k]):
                xj_ok = False
                break
        if not xj_ok:
            continue

        # Filter finite rows in window
        valid_count = 0
        for i in range(start, end):
            row_ok = True
            for k in range(n_features):
                if not np.isfinite(X[i, k]):
                    row_ok = False
                    break
            if row_ok:
                valid_count += 1

        if valid_count < min_samples:
            continue

        Xw = np.empty((valid_count, n_features), dtype=np.float64)
        yw = np.empty(valid_count, dtype=np.int64)
        idx = 0
        for i in range(start, end):
            row_ok = True
            for k in range(n_features):
                if not np.isfinite(X[i, k]):
                    row_ok = False
                    break
            if row_ok:
                for k in range(n_features):
                    Xw[idx, k] = X[i, k]
                yw[idx] = y[i]
                idx += 1

        pu, pd_, _ = _nb_fit_predict(Xw, yw, X[j], reg, is_normal, df_student)
        probs[j] = pu
        if pu > prob_threshold:
            sig_codes[j] = 1
        elif pd_ > prob_threshold:
            sig_codes[j] = -1

    pnl = np.zeros(n, dtype=np.float64)
    positions = np.zeros(n, dtype=np.int64)
    position = 0
    for j in range(1, n):
        sig = sig_codes[j - 1]
        today, prev = prices[j], prices[j - 1]
        if not (np.isfinite(today) and np.isfinite(prev) and prev > 0.0):
            positions[j] = position
            continue
        ret = (today - prev) / prev
        new = 1 if sig == 1 else (-1 if sig == -1 else position)
        bar = ret if position == 1 else (-ret if position == -1 else 0.0)
        if new != position:
            bar -= (1.0 if position == 0 or new == 0 else 2.0) * commission
        pnl[j] = bar
        positions[j] = new
        position = new

    return probs, sig_codes, pnl, positions



# Math/Bayesian wrappers and legacy single-ticker API
def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def regularized_covariance(X, reg=1e-6):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return _nb_regularized_cov(np.ascontiguousarray(X), float(reg))


def logpdf_multivariate_normal(x, mean, cov):
    return float(_nb_logpdf_mvn(np.asarray(x, np.float64), np.asarray(mean, np.float64), np.asarray(cov, np.float64)))


def logpdf_multivariate_student_t(x, mean, cov, df=5.0):
    return float(_nb_logpdf_mvt(np.asarray(x, np.float64), np.asarray(mean, np.float64), np.asarray(cov, np.float64), float(df)))


def build_samples(prices, dates, lags, predict_day=1, include_maxlag_skew=False):
    """Build lag features, delegating skew calculation to the Numba kernel."""
    prices = np.asarray(prices, dtype=np.float64)
    dates = np.asarray(dates)
    lags = sorted(set(int(x) for x in lags))
    if not lags or min(lags) < 1 or predict_day < 1:
        raise ValueError('lags and predict_day must be positive.')
    max_lag = max(lags)
    rows = []
    for t in range(max_lag, len(prices) - predict_day):
        p0, p1 = prices[t], prices[t + predict_day]
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0):
            continue
        values = []
        for lag in lags:
            historical = prices[t-lag]
            if not (np.isfinite(historical) and historical > 0):
                break
            values.append(p0 / historical)
        if len(values) != len(lags):
            continue
        if include_maxlag_skew:
            values.append(float(_nb_skew_unbiased(np.ascontiguousarray(prices[t-max_lag:t+1]))))
        row = {'date': dates[t], 'today_price': float(p0), 'future_price': float(p1), 'label': int(p1 > p0)}
        row.update({f'x_{i}': float(v) for i, v in enumerate(values)})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_params(X_window, y_window, reg=1e-6):
    X = np.ascontiguousarray(X_window, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.ascontiguousarray(y_window, dtype=np.int64)
    n = len(y)
    n_up = int((y == 1).sum())
    params = {'prior_up': (n_up+1.0)/(n+2.0), 'prior_down': (n-n_up+1.0)/(n+2.0), 'valid': True}
    for cls, key in ((1, 'up'), (0, 'down')):
        xc = X[y == cls] if np.any(y == cls) else X
        params[f'mean_{key}'] = xc.mean(axis=0)
        params[f'cov_{key}'] = regularized_covariance(xc, reg)
    for key in ('mean_up','mean_down','cov_up','cov_down'):
        if not np.all(np.isfinite(params[key])): params['valid'] = False
    return params


def predict_proba(x, params, dist='student', student_df=10.0):
    if not params['valid']:
        return 0.5, 0.5
    x = np.asarray(x, dtype=np.float64)
    if dist == 'normal':
        lu = safe_log(params['prior_up']) + _nb_logpdf_mvn(x, params['mean_up'], params['cov_up'])
        ld = safe_log(params['prior_down']) + _nb_logpdf_mvn(x, params['mean_down'], params['cov_down'])
    else:
        lu = safe_log(params['prior_up']) + _nb_logpdf_mvt(x, params['mean_up'], params['cov_up'], float(student_df))
        ld = safe_log(params['prior_down']) + _nb_logpdf_mvt(x, params['mean_down'], params['cov_down'], float(student_df))
    m = max(lu, ld)
    eu, ed = np.exp(lu-m), np.exp(ld-m)
    total = eu+ed
    return (0.5,0.5) if total <= 0 or not np.isfinite(total) else (float(eu/total), float(ed/total))


def rolling_trade(samples, n_features, window_size=250, predict_day=1, dist='student', student_df=10.0,
                  reg=1e-6, prob_threshold=0.9, commission=0.0):
    cols = [f'x_{i}' for i in range(n_features)]
    X = np.ascontiguousarray(samples[cols].to_numpy(), dtype=np.float64)
    y = np.ascontiguousarray(samples['label'].to_numpy(), dtype=np.int64)
    prices = np.ascontiguousarray(samples['today_price'].to_numpy(), dtype=np.float64)
    probs, codes, pnl, positions = _nb_rolling_trade_kernel(
        X, y, prices, int(n_features), int(window_size), int(predict_day),
        dist == 'normal', float(student_df), float(reg), float(prob_threshold), float(commission))
    result = samples.copy()
    result['signal'] = ['BUY' if c == 1 else 'SELL' if c == -1 else 'HOLD' for c in codes]
    result['prob_up'], result['position'] = probs, positions
    result['bar_pnl'], result['cumulative_pnl'] = pnl, np.cumsum(pnl)
    return result


def split_df(df, train_start, train_end, test_start, test_end):
    result = df.copy()
    result['date'] = pd.to_datetime(result['date'])
    result['split'] = 'unused'
    train = (result.date >= pd.to_datetime(train_start)) & (result.date < pd.to_datetime(train_end))
    test = (result.date >= pd.to_datetime(test_start)) & (result.date < pd.to_datetime(test_end))
    result.loc[train, 'split'] = 'train'; result.loc[test, 'split'] = 'test'
    return result


def generate_random_lag_sets(n_sets=100, min_lags=2, max_lags=3, min_lag_value=1, max_lag_value=30, seed=42):
    rng = random.Random(seed); found=[]; attempts=0; limit=max(n_sets*20,100)
    while len(found)<n_sets and attempts<limit:
        attempts += 1
        k=rng.randint(min_lags,max_lags)
        candidate=tuple(sorted(rng.sample(range(min_lag_value,max_lag_value+1),k)))
        if candidate not in found: found.append(candidate)
    return found


def safe_statistical_test(test_name, first, second):
    try:
        if test_name == 'wilcoxon':
            if np.allclose(first,second): return 1.0
            _, pval = wilcoxon(first,second,zero_method='wilcox',alternative='two-sided')
        elif test_name == 'ttest':
            _, pval = ttest_rel(first,second,nan_policy='omit')
        else: return 1.0
        return float(pval) if np.isfinite(pval) else 1.0
    except Exception: return 1.0


def run_one_lag_configuration(raw_data, lags, include_maxlag_skew, args):
    prices=raw_data['price'].to_numpy(dtype=np.float64); dates=raw_data['date'].to_numpy()
    samples=build_samples(prices,dates,lags,args.predict_day,include_maxlag_skew)
    if samples.empty: return None
    samples=split_df(samples,args.train_start,args.train_end,args.test_start,args.test_end)
    train_count=int((samples['split']=='train').sum()); test_count=int((samples['split']=='test').sum())
    minimum=args.window_size+max(lags)+args.predict_day+10
    if train_count==0 or test_count==0 or len(samples)<minimum: return None
    n_features=len(lags)+int(include_maxlag_skew)
    result=rolling_trade(samples,n_features,args.window_size,args.predict_day,args.dist,args.student_df,
                         args.reg,args.prob_threshold,args.commission)
    result['equity']=args.initial_capital*(1.0+result['bar_pnl']).cumprod()
    result['lags']=str(tuple(lags)); result['include_maxlag_skew']=bool(include_maxlag_skew)
    return result


def get_train_returns(result):
    return result.loc[result['split']=='train','bar_pnl'].to_numpy(dtype=np.float64)


def _evaluate_single_lag(args_tuple):
    raw_data,lags,include_maxlag_skew,args=args_tuple
    try:
        result=run_one_lag_configuration(raw_data,lags,include_maxlag_skew,args)
        if result is None or result.empty: return None
        returns=get_train_returns(result)
        if not len(returns): return None
        sd=float(np.std(returns,ddof=1)) if len(returns)>1 else 0.0
        sharpe=float(np.mean(returns)/sd*np.sqrt(args.annualization)) if sd>0 else -np.inf
        return {'lags':tuple(lags),'returns':returns,'total_pnl':float(np.sum(returns)),'sharpe':sharpe}
    except Exception: return None


def select_best_lags(raw_data, lag_sets, include_maxlag_skew, args):
    tasks=[(raw_data,lags,include_maxlag_skew,args) for lags in lag_sets]
    workers=max(1,min(int(getattr(args,'workers',getattr(args,'n_workers',os.cpu_count() or 1))),len(tasks) or 1))
    if workers==1: results=list(map(_evaluate_single_lag,tasks))
    else:
        with mp.Pool(processes=workers) as pool: results=pool.map(_evaluate_single_lag,tasks)
    evaluations=[x for x in results if x is not None]
    if not evaluations: return {}
    selected={'raw_pnl':max(evaluations,key=lambda x:x['total_pnl'])['lags'],
              'raw_sharpe':max(evaluations,key=lambda x:x['sharpe'])['lags']}
    scored=[]
    for current in evaluations:
        others=np.concatenate([x['returns'] for x in evaluations if x is not current]) if len(evaluations)>1 else np.array([])
        n=min(len(current['returns']),len(others))
        pvals={m:safe_statistical_test(m,current['returns'][:n],others[:n]) for m in STATISTICAL_METHODS} if n>=5 else {m:1.0 for m in STATISTICAL_METHODS}
        scored.append({'lags':current['lags'],**{m:current['total_pnl']-100*pvals[m] for m in STATISTICAL_METHODS}})
    for method in STATISTICAL_METHODS: selected[method]=max(scored,key=lambda x:x[method])['lags']
    return selected


def compute_max_drawdown(equity):
    equity=pd.Series(equity).astype(float); return float((equity/equity.cummax()-1).min())


def compute_metrics(result, split_name, annualization):
    part=result[result['split']==split_name].copy()
    if part.empty: return {}
    returns=part['bar_pnl'].replace([np.inf,-np.inf],np.nan).dropna()
    equity=part['equity'].replace([np.inf,-np.inf],np.nan).dropna()
    if returns.empty or equity.empty: return {}
    total_return=equity.iloc[-1]/equity.iloc[0]-1
    periods=len(returns)
    annual_return=(equity.iloc[-1]/equity.iloc[0])**(annualization/periods)-1 if periods>0 and equity.iloc[0]>0 else np.nan
    vol=returns.std(ddof=1) if len(returns)>1 else np.nan
    ann_vol=vol*np.sqrt(annualization) if np.isfinite(vol) else np.nan
    sharpe=returns.mean()/vol*np.sqrt(annualization) if np.isfinite(vol) and vol>0 else np.nan
    dd=compute_max_drawdown(equity); calmar=annual_return/abs(dd) if np.isfinite(dd) and dd<0 and np.isfinite(annual_return) else np.nan
    signals=part['signal']; buys=part[signals=='BUY']; sells=part[signals=='SELL']
    n_buy=int((signals=='BUY').sum()); n_sell=int((signals=='SELL').sum()); n_trades=n_buy+n_sell
    wins=(buys.future_price>buys.today_price).sum()+(sells.future_price<sells.today_price).sum()
    return {'split':split_name,'bars':int(len(part)),'trades':n_trades,'n_buy':n_buy,'n_sell':n_sell,
      'n_hold':int((signals=='HOLD').sum()),'total_pnl':float(part.bar_pnl.sum()),
      'final_cumulative_pnl':float(part.cumulative_pnl.iloc[-1]),'win_rate':float(wins/n_trades) if n_trades else np.nan,
      'equity_start':float(equity.iloc[0]),'equity_end':float(equity.iloc[-1]),'return_pct':float(total_return*100),
      'annual_return_pct':float(annual_return*100) if np.isfinite(annual_return) else np.nan,
      'annual_volatility_pct':float(ann_vol*100) if np.isfinite(ann_vol) else np.nan,
      'sharpe_ratio':float(sharpe) if np.isfinite(sharpe) else np.nan,
      'calmar_ratio':float(calmar) if np.isfinite(calmar) else np.nan,'max_drawdown_pct':float(dd*100)}


def write_metrics_summary(rows, output_path):
    pd.DataFrame(rows,columns=SUMMARY_COLUMNS).to_csv(output_path,index=False)


def load_price_data(data_path):
    raw=pd.read_pickle(data_path)
    if not isinstance(raw,pd.DataFrame): raise TypeError('The pickle file must contain a pandas DataFrame.')
    raw=raw.copy(); price_candidates=('Close','close','Adj Close','adj close','Price','price','Last','last')
    pc=next((c for c in price_candidates if c in raw.columns),None)
    if pc is None:
        numeric=raw.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric: raise ValueError('No numeric price column was found.')
        pc=numeric[-1]
    date_candidates=('Date','date','Datetime','datetime','Time','time')
    dc=next((c for c in date_candidates if c in raw.columns),None)
    if dc is not None: dates=pd.to_datetime(raw[dc],errors='coerce')
    elif isinstance(raw.index,pd.DatetimeIndex): dates=pd.to_datetime(raw.index,errors='coerce')
    else: raise ValueError('No date column or DatetimeIndex was found.')
    prices=pd.to_numeric(raw[pc],errors='coerce')
    data=pd.DataFrame({'date':dates,'price':prices}).dropna(subset=['date','price']).copy()
    data=data[np.isfinite(data.price)&(data.price>0)].copy()
    return data.sort_values('date').drop_duplicates('date').reset_index(drop=True)


def _run_configuration_worker(args_tuple):
    data,variant_name,include_skew,method,lags,args=args_tuple
    try:
        result=run_one_lag_configuration(data,lags,include_skew,args)
        return variant_name,method,lags,result
    except Exception: return variant_name,method,lags,None


def _repo_data(name):
    """Locate a dataset in the repository data/ folder (or next to the script)."""
    here = Path(__file__).resolve().parent
    for base in (here, *here.parents):
        for candidate in (base / "data" / name, base / name):
            if candidate.is_file():
                return str(candidate)
    return str(here.parent / "data" / name)


WF_DEFAULT_DATA = _repo_data('BTCUSD.pkl')
WF_WINDOW = 80
WF_TRAIN = 2190
WF_TEST = 540
WF_MIN_LAGS = 2
WF_MAX_LAGS = 2
WF_MIN_LAG = 1
WF_MAX_LAG = 30
WF_ANNUAL = 2190
WF_REGIME = 120

# Walk-forward-only execution engine. Legacy functions above are not called.
import argparse
import json
import hashlib
from itertools import combinations


def wf_load(path, price_col=None):
    path = Path(path)
    raw = pd.read_pickle(path) if path.suffix.lower() in ('.pkl', '.pickle') else pd.read_csv(path)
    raw = raw.copy()
    raw.columns = raw.columns.map(lambda x: str(x).strip())
    dc = next((c for c in ('Date', 'date', 'Datetime', 'datetime', 'Time', 'time') if c in raw), None)
    if dc is None:
        if not isinstance(raw.index, pd.DatetimeIndex):
            raise ValueError('A date column or DatetimeIndex is required.')
        raw['Date'] = raw.index
        dc = 'Date'
    pc = price_col or next((c for c in ('Adj Close', 'Close', 'close', 'price', 'Price', 'Last') if c in raw), None)
    if pc not in raw:
        raise ValueError('Price column not found; specify --price-col.')
    tc = next((c for c in ('Ticker', 'ticker', 'Symbol', 'symbol') if c in raw), None)
    data = pd.DataFrame({'date': pd.to_datetime(raw[dc], errors='raise').to_numpy(),
                         'ticker': raw[tc].astype(str).to_numpy() if tc else 'ASSET',
                         'price': pd.to_numeric(raw[pc], errors='raise').to_numpy()})
    if data.isna().any().any() or not np.isfinite(data.price).all() or (data.price <= 0).any():
        raise ValueError('Missing/invalid dates or prices; clean input explicitly.')
    if data.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate ticker/date rows are not allowed.')
    panel = data.pivot(index='date', columns='ticker', values='price').sort_index()
    if panel.isna().any().any():
        raise ValueError('Unbalanced price panel: supply a common-calendar panel; no silent filling/dropping.')
    return panel


def wf_candidates(a):
    import math
    count = sum(math.comb(a.max_lag_value-a.min_lag_value+1, k) for k in range(a.min_lags, a.max_lags+1))
    target = min(a.n_lag_sets, count)
    rng = random.Random(a.seed)
    result = set()
    while len(result) < target:
        k = rng.randint(a.min_lags, a.max_lags)
        result.add(tuple(sorted(rng.sample(range(a.min_lag_value, a.max_lag_value+1), k))))
    return sorted(result)


def wf_positions(prices, lags, include_skew, a):
    """Numba-accelerated online position calculation."""
    prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
    lags_arr = np.ascontiguousarray(lags, dtype=np.int64)
    pos, valid, err = _nb_wf_positions_kernel(
        prices_arr,
        lags_arr,
        bool(include_skew),
        int(a.window_size),
        float(a.reg),
        a.dist == 'normal',
        float(a.student_df),
        float(a.prob_threshold),
    )
    if err:
        raise RuntimeError('Non-finite classifier output.')
    return pos, valid


def _wf_candidate_worker(task):
    """Multiprocessing worker for one (ticker, variant) across all candidate lag sets."""
    ticker, variant, prices_arr, lags_list, warmup, window_size, reg, is_normal, student_df, prob_threshold = task
    include_skew = (variant == 'maxlag_skew')
    n_lags = len(lags_list)
    arr = np.empty((len(prices_arr), n_lags), dtype=np.float64)

    for k, lag in enumerate(lags_list):
        lags_arr = np.ascontiguousarray(lag, dtype=np.int64)
        pos, valid, err = _nb_wf_positions_kernel(
            prices_arr, lags_arr, include_skew, window_size, reg, is_normal, student_df, prob_threshold
        )
        if err:
            raise RuntimeError(f'Non-finite classifier output for {ticker} ({variant}).')
        if not valid[warmup:].all():
            raise RuntimeError(f'Incomplete warmup alignment for {ticker} ({variant}).')
        arr[:, k] = pos

    print(f'Completed all {n_lags} candidates for {ticker} ({variant})', flush=True)
    return ticker, variant, arr


def wf_returns(positions, market_returns, commission):
    """Reset interval to flat; charge entry, reversal and terminal liquidation."""
    turnover = np.abs(np.diff(positions, axis=0, prepend=np.zeros_like(positions[:1])))
    turnover[-1] += np.abs(positions[-1])
    return positions * market_returns - commission * turnover


def wf_select(r, methods):
    pnl = r.sum(axis=0)
    sd = r.std(axis=0, ddof=1)
    score = np.divide(r.mean(axis=0), sd, out=np.full(len(sd), -np.inf), where=sd > 0)
    selected = {'raw_pnl': int(np.argmax(pnl)), 'raw_sharpe': int(np.argmax(score))}
    # Paired samples are now aligned on identical dates. Reference is the
    # mean of OTHER candidate strategies, not concatenated unrelated dates.
    for method in ('wilcoxon', 'ttest'):
        values = []
        for k in range(r.shape[1]):
            p = 1.
            if r.shape[1] > 1:
                diff = r[:, k] - (r.sum(axis=1)-r[:, k])/(r.shape[1]-1)
                if np.any(diff):
                    test = wilcoxon(r[:, k], r[:, k]-diff) if method == 'wilcoxon' else ttest_rel(r[:, k], r[:, k]-diff)
                    p = float(test.pvalue) if np.isfinite(test.pvalue) else 1.
            values.append(pnl[k] - 100*p)
        selected[method] = int(np.argmax(values))
    return {m: selected[m] for m in methods}


def wf_metrics(r, annualization):
    r = np.asarray(r, float)
    if not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Non-finite returns or bankrupt portfolio; metrics cannot be compounded.')
    eq = np.r_[1., np.cumprod(1+r)]
    sd = r.std(ddof=1) if len(r)>1 else np.nan
    dd = (eq/np.maximum.accumulate(eq)-1).min()
    ann = eq[-1]**(annualization/len(r))-1
    return {'return_pct': 100*(eq[-1]-1), 'annual_return_pct':100*ann,
            'annual_volatility_pct':100*sd*np.sqrt(annualization),
            'sharpe_ratio': r.mean()/sd*np.sqrt(annualization) if sd>0 else np.nan,
            'max_drawdown_pct':100*dd, 'calmar_ratio':ann/abs(dd) if dd<0 else np.nan,
            'positive_bar_fraction':float((r>0).mean()), 'n_bars':len(r)}


def wf_summary(frame, keys):
    rows = []
    for group, g in frame.groupby(keys, dropna=False):
        if not isinstance(group, tuple): group = (group,)
        for metric in ('return_pct','sharpe_ratio','max_drawdown_pct','annual_volatility_pct'):
            v = g[metric].dropna()
            rows.append(dict(zip(keys, group), metric=metric, n_windows=len(g), n_valid=len(v),
                             mean=v.mean(), std=v.std(), median=v.median(), minimum=v.min(),
                             q25=v.quantile(.25), q75=v.quantile(.75), maximum=v.max(),
                             positive_window_fraction=float((g.return_pct>0).mean())))
    return pd.DataFrame(rows)


def walk_forward_main(argv=None):
    p = argparse.ArgumentParser(description='Comment 9: ONLY chronological walk-forward; no legacy run.')
    p.add_argument('--data-path', default=WF_DEFAULT_DATA)
    p.add_argument('--output-dir', default=str(Path(__file__).with_suffix(''))+'_walk_forward')
    p.add_argument('--price-col', default=None)
    p.add_argument('--window-size', type=int, default=WF_WINDOW)
    p.add_argument('--train-bars', type=int, default=WF_TRAIN, help='Past scored bars (warmup is additional).')
    p.add_argument('--test-bars', type=int, default=WF_TEST)
    p.add_argument('--selection-mode', choices=['rolling','expanding'], default='rolling')
    p.add_argument('--include-partial', action='store_true', help='Include last shorter test window (>=2 bars).')
    p.add_argument('--n-lag-sets', type=int, default=100)
    p.add_argument('--min-lags', type=int, default=WF_MIN_LAGS)
    p.add_argument('--max-lags', type=int, default=WF_MAX_LAGS)
    p.add_argument('--min-lag-value', type=int, default=WF_MIN_LAG)
    p.add_argument('--max-lag-value', type=int, default=WF_MAX_LAG)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--annualization', type=float, default=WF_ANNUAL)
    p.add_argument('--regime-lookback', type=int, default=WF_REGIME)
    p.add_argument('--commission', type=float, default=0.)
    p.add_argument('--initial-capital', type=float, default=1_000_000.)
    p.add_argument('--prob-threshold', type=float, default=.9)
    p.add_argument('--dist', choices=['student','normal'], default='student')
    p.add_argument('--student-df', type=float, default=10.)
    p.add_argument('--reg', type=float, default=1e-6)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1, help='Number of parallel worker processes.')
    a = p.parse_args(argv)
    if not (1 <= a.min_lag_value <= a.max_lag_value and 1 <= a.min_lags <= a.max_lags <= a.max_lag_value-a.min_lag_value+1):
        p.error('Invalid lag range/count.')
    if min(a.train_bars,a.test_bars,a.regime_lookback)<2 or a.window_size<max(10,a.max_lags+3) or a.n_lag_sets<1:
        p.error('Invalid training/test/warmup/candidate settings.')
    if not (0<=a.commission<.5 and .5<=a.prob_threshold<1 and a.annualization>0 and a.initial_capital>0 and a.reg>0 and a.student_df>0):
        p.error('Invalid financial/model parameters.')
    panel = wf_load(a.data_path, a.price_col)
    lags = wf_candidates(a)
    warmup = max(max(x) for x in lags)+a.window_size+2
    first = max(warmup+a.train_bars, 2*a.regime_lookback+1)
    starts = [s for s in range(first, len(panel), a.test_bars)
              if min(a.test_bars, len(panel)-s)>=2 and (s+a.test_bars<=len(panel) or a.include_partial)]
    if len(starts)<2:
        p.error(f'Need >=2 test windows: {len(panel)} rows available, warmup={warmup}, first origin={first}. Adjust train/test bars explicitly.')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error('Output directory must be empty to avoid mixing old and new results.')
    config = vars(a).copy()
    config.update(script=Path(__file__).name, input_sha256=hashlib.sha256(Path(a.data_path).read_bytes()).hexdigest(),
                  n_splits=len(starts), candidate_lags=lags, warmup_bars=warmup,
                  protocol='one-bar close-to-close; online trailing fit; frozen lag/weights per test; flat boundaries',
                  numpy_version=np.__version__, pandas_version=pd.__version__)
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    market = panel.pct_change().fillna(0.)
    bench = market.mean(axis=1)
    vol = bench.rolling(a.regime_lookback).std(ddof=1)*np.sqrt(a.annualization)
    metrics, audits, daily, splits = [], [], [], []
    for split, s in enumerate(starts, 1):
        e = min(s+a.test_bars, len(panel))
        b = warmup if a.selection_mode=='expanding' else s-a.train_bars
        threshold = float(vol.iloc[a.regime_lookback:s].median())
        momentum = float(np.prod(1+bench.iloc[s-a.regime_lookback:s])-1)
        regime = ('up' if momentum>=0 else 'down')+'_'+('high_vol' if vol.iloc[s-1]>threshold else 'low_vol')
        splits.append(dict(split=split, train_start=str(panel.index[b]), train_end=str(panel.index[s-1]),
                           test_start=str(panel.index[s]), test_end=str(panel.index[e-1]),
                           train_bars=s-b,test_bars=e-s, regime=regime, past_market_return=momentum,
                           past_vol=float(vol.iloc[s-1]), past_vol_threshold=threshold, b=b,s=s,e=e))
    # Candidate predictions are causal and shared across splits, so computed only once.
    # Work is distributed by (ticker, variant); each worker evaluates all lag candidates.
    tasks = [
        (
            ticker,
            variant,
            np.ascontiguousarray(panel[ticker].to_numpy(), dtype=np.float64),
            lags,
            warmup,
            int(a.window_size),
            float(a.reg),
            a.dist == 'normal',
            float(a.student_df),
            float(a.prob_threshold),
        )
        for ticker in panel
        for variant in ('regular', 'maxlag_skew')
    ]
    cache = {}
    n_workers = max(1, min(int(a.workers), len(tasks)))
    if n_workers == 1:
        for task in tasks:
            t_name, v_name, arr = _wf_candidate_worker(task)
            cache[t_name, v_name] = arr
    else:
        with mp.Pool(processes=n_workers) as pool:
            for t_name, v_name, arr in pool.imap_unordered(_wf_candidate_worker, tasks):
                cache[t_name, v_name] = arr

    for info in splits:
        b,s,e = info['b'],info['s'],info['e']
        for variant in ('regular','maxlag_skew'):
            chosen = {}
            for ticker in panel:
                r = wf_returns(cache[ticker,variant][b:s], market[ticker].to_numpy()[b:s,None], a.commission)
                chosen[ticker] = (wf_select(r, METHODS), r)
            for method in METHODS:
                past, tests, selections = [], [], []
                for ticker in panel:
                    selected, r = chosen[ticker]
                    k = selected[method]
                    tr = float(np.prod(1+r[:,k])-1)
                    if (r[:,k]<=-1).any(): raise ValueError('Bankrupt training strategy.')
                    past.append(max(tr,0.))
                    tests.append(wf_returns(cache[ticker,variant][s:e,k:k+1], market[ticker].to_numpy()[s:e,None], a.commission)[:,0])
                    selections.append(dict(split=info['split'],method=method,variant=variant,ticker=ticker,
                                           lags=str(lags[k]),train_return=tr, selection_data_end=str(panel.index[s-1])))
                w = np.array(past)
                w = w/w.sum() if w.sum()>0 else np.ones(len(w))/len(w)
                for record, weight in zip(selections,w): audits.append(dict(record,weight=float(weight)))
                # Fixed initial allocations, drifting buy-and-hold sleeve weights (original portfolio convention).
                curves = np.cumprod(1+np.column_stack(tests),axis=0)
                if (np.column_stack(tests)<=-1).any(): raise ValueError('Bankrupt test sleeve.')
                eq = curves@w
                r = eq/np.r_[1.,eq[:-1]]-1
                meta = {k:v for k,v in info.items() if k not in ('b','s','e')}
                metrics.append(dict(meta,method=method,variant=variant,**wf_metrics(r,a.annualization)))
                for date,ret,equity in zip(panel.index[s:e],r,eq):
                    daily.append(dict(split=info['split'],date=date,method=method,variant=variant,
                                      regime=info['regime'],bar_return=ret,split_equity=equity*a.initial_capital))
        print(f'Completed split {info["split"]}/{len(splits)}: {info["regime"]}',flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(out/'window_metrics.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'lag_weights_audit.csv',index=False)
    pd.DataFrame(splits).drop(columns=['b','s','e']).to_csv(out/'splits.csv',index=False)
    d = pd.DataFrame(daily).sort_values(['method','variant','date'])
    d['stitched_equity'] = d.groupby(['method','variant']).bar_return.transform(lambda x: (1+x).cumprod()*a.initial_capital)
    d.to_csv(out/'daily_oos.csv',index=False)
    wf_summary(frame,['method','variant']).to_csv(out/'performance_distribution.csv',index=False)
    wf_summary(frame,['method','variant','regime']).to_csv(out/'regime_distribution.csv',index=False)
    overall = []
    for (method,variant), g in d.groupby(['method','variant']):
        overall.append(dict(method=method,variant=variant,**wf_metrics(g.bar_return,a.annualization)))
    pd.DataFrame(overall).to_csv(out/'overall_oos_metrics.csv',index=False)
    print(frame.groupby(['method','variant']).return_pct.agg(['count','mean','median','min','max']).to_string())
    print(f'Walk-forward ONLY complete. Results: {out.resolve()}')


if __name__ == '__main__':
    walk_forward_main()
