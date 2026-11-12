from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import sys
import warnings

# Avoid BLAS oversubscription inside worker processes.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
              'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
import scipy
from scipy.stats import skew, wilcoxon, ttest_rel
import sklearn
from sklearn.covariance import LedoitWolf

try:
    if '--no-numba' in sys.argv:
        raise ImportError('Numba disabled on command line')
    import numba
    from numba import njit
    NUMBA_ENABLED = True
except ImportError:
    numba = None
    NUMBA_ENABLED = False
    def njit(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate

METHODS = ('raw_pnl', 'raw_sharpe', 'wilcoxon', 'ttest')
VARIANTS = ('regular', 'maxlag_skew')
STAGES = ('empirical', 'fitted', 'fitted_raw_units', 'density')


@dataclass(frozen=True)
class Protocol:
    name: str
    data_kind: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    window: int
    min_lags: int
    max_lags: int
    min_lag: int
    max_lag: int
    annualization: float


PROTOCOLS = {
    'DRL_Ensemble': Protocol('DRL_Ensemble', 'djia', '2015-06-01', '2015-12-31',
                            '2016-01-01', '2020-05-08', 250, 2, 2, 1, 30, 252),
    'FineRL': Protocol('FineRL', 'djia', '2019-07-01', '2020-07-01',
                      '2020-07-01', '2022-03-31', 250, 2, 5, 5, 50, 252),
    'XGboost_GRU': Protocol('XGboost_GRU', 'btc', '2020-01-01', '2021-07-01',
                           '2021-07-01', '2023-03-01', 80, 2, 2, 1, 30, 2190),
}


@dataclass(frozen=True)
class Setting:
    standardize: bool = False
    ledoit_wolf: bool = False
    matched: bool = False

    @property
    def name(self):
        return ('zscore' if self.standardize else 'raw') + '__' + (
            'ledoit_wolf' if self.ledoit_wolf else 'empirical') + '__' + (
            'covariance_matched' if self.matched else 'legacy_scale')


SETTINGS = [Setting(z, lw, matched) for matched in (False, True)
            for z, lw in ((False, False), (True, False), (False, True), (True, True))]
BASELINE = SETTINGS[0].name


@njit(cache=True)
def column_mean(x):
    n, d = x.shape
    result = np.zeros(d)
    for i in range(n):
        for k in range(d):
            result[k] += x[i, k]
    return result / n


@njit(cache=True)
def covariance(x, ridge, use_lw):
    """Empirical ddof=1; LW uses sklearn's MLE (ddof=0) convention."""
    n, d = x.shape
    mean = column_mean(x)
    centered = x - mean
    empirical = np.zeros((d, d))
    if n > 1:
        empirical = centered.T @ centered / (n - 1)
    shrinkage = 0.0
    estimated = empirical.copy()
    if use_lw and n > 1:
        mle = empirical * ((n - 1.0) / n)
        mu = np.trace(mle) / d
        # Same non-blocked Ledoit-Wolf formula as sklearn.covariance.
        fourth = 0.0
        for i in range(n):
            norm2 = 0.0
            for k in range(d):
                norm2 += centered[i, k] ** 2
            fourth += norm2 ** 2
        squared = np.sum(mle * mle)
        beta = (fourth / n - squared) / (d * n)
        delta = (squared - 2 * mu * np.trace(mle) + d * mu ** 2) / d
        if d > 1 and delta > 0:
            shrinkage = max(0.0, min(beta, delta)) / delta
        estimated = (1 - shrinkage) * mle + shrinkage * mu * np.eye(d)
    elif use_lw:
        shrinkage = np.nan
    return mean, empirical, estimated + ridge * np.eye(d), shrinkage


@njit(cache=True)
def matrix_statistics(matrix):
    eigen = np.linalg.eigvalsh(matrix)
    singular = np.linalg.svd(matrix)[1]
    condition = np.inf if singular[-1] <= 0 else singular[0] / singular[-1]
    return condition, eigen[0], eigen[-1]


@njit(cache=True)
def log_density(point, mean, matrix, student_df):
    # Same pseudo-inverse and emergency jitter as the submitted originals.
    d = len(point)
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        matrix = matrix + 1e-5 * np.eye(d)
        sign, logdet = np.linalg.slogdet(matrix)
    difference = point - mean
    quadratic = difference @ np.linalg.pinv(matrix) @ difference
    nu = student_df
    return (math.lgamma((nu + d) / 2) - math.lgamma(nu / 2)
            - .5 * d * np.log(nu * np.pi) - .5 * logdet
            - .5 * (nu + d) * np.log(1 + quadratic / nu))


@njit(cache=True)
def rolling_kernel(x, y, prices, window, ridge, threshold, commission,
                   student_df, standardize, use_lw, matched, diagnostics):
    """Prediction at j uses samples [j-1-window:j-1], exactly as originals."""
    n, d = x.shape
    probabilities = np.full(n, np.nan)
    signals = np.zeros(n, dtype=np.int64)
    # 0..11 = four matrix stages x (condition,min_eigen,max_eigen).
    # 12=count; 13=fallback; 14=shrinkage; 15=effective fit count.
    diag = np.full((n if diagnostics else 0, 2, 16), np.nan)
    factor = (student_df - 2) / student_df if matched else 1.0
    for j in range(window + 1, n):
        start, end = j - 1 - window, j - 1
        xw = x[start:end].copy()
        yw = y[start:end]
        center = np.zeros(d)
        scale = np.ones(d)
        if standardize:
            center = column_mean(xw)
            for k in range(d):
                variance = 0.0
                for i in range(window):
                    variance += (xw[i, k] - center[k]) ** 2
                sd = np.sqrt(variance / window)
                scale[k] = sd if sd > 1e-12 else 1.0
        z = (xw - center) / scale
        current = (x[j] - center) / scale
        logs = np.zeros(2)
        for c in range(2):
            label = 1 - c  # up, down
            count = np.sum(yw == label)
            selected = z[yw == label].copy() if count > 0 else z.copy()
            mean, empirical, fitted, shrinkage = covariance(selected, ridge, use_lw)
            density = factor * fitted
            logs[c] = np.log((count + 1.0) / (window + 2.0)) + log_density(
                current, mean, density, student_df)
            if diagnostics:
                raw_units = fitted * scale.reshape(d, 1) * scale.reshape(1, d)
                matrices = (empirical, fitted, raw_units, density)
                for stage in range(4):
                    stats = matrix_statistics(matrices[stage])
                    for k in range(3):
                        diag[j, c, 3 * stage + k] = stats[k]
                diag[j, c, 12] = count
                diag[j, c, 13] = 1.0 if count == 0 else 0.0
                diag[j, c, 14] = shrinkage
                diag[j, c, 15] = len(selected)
        maximum = max(logs[0], logs[1])
        eu, ed = np.exp(logs[0] - maximum), np.exp(logs[1] - maximum)
        total = eu + ed
        pu = eu / total if total > 0 and np.isfinite(total) else .5
        pdn = ed / total if total > 0 and np.isfinite(total) else .5
        probabilities[j] = pu
        signals[j] = 1 if pu > threshold else -1 if pdn > threshold else 0
    pnl = np.zeros(n)
    positions = np.zeros(n, dtype=np.int64)
    turnover = np.zeros(n, dtype=np.int64)
    position = 0
    for j in range(1, n):
        new = signals[j - 1] if signals[j - 1] else position
        ret = (prices[j] - prices[j - 1]) / prices[j - 1]
        turnover[j] = abs(new - position)
        pnl[j] = position * ret - turnover[j] * commission
        positions[j] = new
        position = new
    return probabilities, signals, pnl, positions, turnover, diag


def generate_lags(protocol, n_sets, seed):
    rng = random.Random(seed)
    result = []
    for _ in range(max(n_sets * 20, 100)):
        number = rng.randint(protocol.min_lags, protocol.max_lags)
        candidate = tuple(sorted(rng.sample(range(protocol.min_lag, protocol.max_lag + 1), number)))
        if candidate not in result:
            result.append(candidate)
        if len(result) == n_sets:
            return result
    raise ValueError('Could not generate requested number of unique lag candidates')


def build_samples(raw, lags, include_skew):
    prices = raw.price.to_numpy(dtype=float)
    maximum = max(lags)
    indices = np.arange(maximum, len(prices) - 1)
    features = [prices[indices] / prices[indices - lag] for lag in lags]
    if include_skew:
        windows = np.lib.stride_tricks.sliding_window_view(prices, maximum + 1)[:-1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = skew(windows, axis=1, bias=False) if maximum >= 2 else np.zeros(len(indices))
        features.append(np.where(np.isfinite(values), values, 0.0))
    samples = pd.DataFrame({'date': raw.date.to_numpy()[indices],
                            'today_price': prices[indices], 'future_price': prices[indices + 1],
                            'label': (prices[indices + 1] > prices[indices]).astype(np.int64)})
    for k, values in enumerate(features):
        samples[f'x_{k}'] = values
    return samples


def run_samples(samples, protocol, setting, args, diagnostics=False, train_only=False):
    if train_only:
        samples = samples.loc[samples.date < pd.Timestamp(protocol.train_end)]
    else:
        samples = samples.loc[samples.date < pd.Timestamp(protocol.test_end)]
    samples = samples.copy().reset_index(drop=True)
    columns = [c for c in samples if c.startswith('x_')]
    if len(samples) <= protocol.window + 1:
        raise ValueError(f'{protocol.name}: insufficient history for rolling window')
    x = np.ascontiguousarray(samples[columns].to_numpy(dtype=float))
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features; refusing to alter the rolling clock by dropping rows')
    result = rolling_kernel(x, samples.label.to_numpy(dtype=np.int64),
                            samples.today_price.to_numpy(dtype=float), protocol.window,
                            args.reg, args.prob_threshold, args.commission, args.student_df,
                            setting.standardize, setting.ledoit_wolf, setting.matched, diagnostics)
    probabilities, signals, pnl, positions, turnover, diag = result
    samples['prob_up'], samples['signal'] = probabilities, signals
    samples['bar_pnl'], samples['position'], samples['turnover_units'] = pnl, positions, turnover
    samples['split'] = 'unused'
    for split in ('train', 'test'):
        start, end = pd.Timestamp(getattr(protocol, split + '_start')), pd.Timestamp(getattr(protocol, split + '_end'))
        samples.loc[(samples.date >= start) & (samples.date < end), 'split'] = split
    if not np.isfinite(pnl).all() or np.any(pnl <= -1):
        raise ValueError('Nonfinite return or wealth <= 0: leveraged compounding is not valid')
    return samples, diag


def select_lags(sample_map, protocol, setting, args, methods):
    evaluations = []
    for lags, samples in sample_map.items():
        result, _ = run_samples(samples, protocol, setting, args, train_only=True)
        part = result.loc[result.split.eq('train')]
        if part.empty or part.prob_up.notna().sum() == 0:
            raise ValueError(f'{protocol.name}: no fitted Train predictions')
        r = part.bar_pnl.to_numpy()
        sd = np.std(r, ddof=1 if protocol.data_kind == 'btc' else 0)
        evaluations.append({'lags': lags, 'returns': r, 'total_pnl': float(r.sum()),
                            'sharpe': float(r.mean() / sd * np.sqrt(protocol.annualization)) if sd > 0 else -np.inf})
    chosen = {}
    if 'raw_pnl' in methods:
        chosen['raw_pnl'] = max(evaluations, key=lambda e: e['total_pnl'])['lags']
    if 'raw_sharpe' in methods:
        chosen['raw_sharpe'] = max(evaluations, key=lambda e: e['sharpe'])['lags']
    for method in ('wilcoxon', 'ttest'):
        if method not in methods:
            continue
        scores = []
        for i, current in enumerate(evaluations):
            others = np.concatenate([e['returns'] for j, e in enumerate(evaluations) if i != j])
            length = min(len(current['returns']), len(others))
            p = 1.0
            if length >= 5:
                first, second = current['returns'][:length], others[:length]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        if method == 'wilcoxon':
                            p = 1.0 if protocol.data_kind == 'btc' and np.allclose(first, second) else float(wilcoxon(first, second).pvalue)
                        else:
                            p = float(ttest_rel(first, second).pvalue)
                except ValueError:
                    p = 1.0
                # Explicitly fix the originals' undefined constant-series NaN edge case.
                if not np.isfinite(p):
                    p = 1.0
            scores.append(current['total_pnl'] - 100 * p)
        chosen[method] = evaluations[int(np.argmax(scores))]['lags']
    return chosen


def return_metrics(returns, annualization):
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError('Expected nonempty, finite, compoundable returns')
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    sd = np.std(r, ddof=1) if len(r) > 1 else np.nan
    annual = float(np.expm1(np.log(wealth[-1]) * annualization / len(r)))
    drawdown = float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
    return dict(bars=len(r), return_pct=100 * (wealth[-1] - 1),
                legacy_return_pct=100 * (wealth[-1] / wealth[1] - 1),
                annualization=annualization, annual_return_pct=100 * annual,
                annual_volatility_pct=100 * sd * np.sqrt(annualization),
                sharpe_ratio=r.mean() / sd * np.sqrt(annualization) if sd > 0 else np.nan,
                max_drawdown_pct=100 * drawdown,
                calmar_ratio=annual / abs(drawdown) if drawdown < 0 else np.nan,
                sum_bar_pnl=float(r.sum()), positive_bar_fraction=float(np.mean(r > 0)),
                equity_start=1.0, equity_end=float(wealth[-1]))


def asset_metrics(result, protocol, metadata):
    rows = []
    for split in ('train', 'test'):
        part = result.loc[result.split.eq(split)]
        if part.empty or not part.prob_up.notna().any():
            raise ValueError(f'No fitted predictions for {metadata} / {split}')
        values = return_metrics(part.bar_pnl, protocol.annualization)
        active = part.signal.ne(0)
        wins = ((part.signal.eq(1) & (part.future_price > part.today_price)) |
                (part.signal.eq(-1) & (part.future_price < part.today_price)))
        values.update(trades=int(part.turnover_units.gt(0).sum()),
                      turnover_units=int(part.turnover_units.sum()), n_signals=int(active.sum()),
                      signal_win_rate=float(wins.sum() / active.sum()) if active.any() else np.nan,
                      fitted_predictions=int(part.prob_up.notna().sum()))
        if protocol.data_kind == 'btc':
            old = return_metrics(part.bar_pnl, 365)
            values['sharpe_legacy_365'] = old['sharpe_ratio']
            values['annual_return_legacy_365_pct'] = old['annual_return_pct']
        rows.append(dict(metadata, split=split, start=str(part.date.iloc[0]), end=str(part.date.iloc[-1]), **values))
    return rows


def summarize_diagnostics(result, diag, metadata):
    rows = []
    for split in ('train', 'test'):
        mask = (result.split.eq(split) & result.prob_up.notna()).to_numpy()
        if not mask.any():
            raise ValueError('No diagnostic windows')
        for c, label in enumerate(('up', 'down')):
            values = diag[mask, c]
            shrink = values[:, 14]
            for stage, name in enumerate(STAGES):
                cond = values[:, 3 * stage]
                finite = cond[np.isfinite(cond)]
                rows.append(dict(metadata, split=split, class_name=label, matrix=name,
                    windows=len(cond), finite_windows=len(finite),
                    nonfinite_fraction=float(np.mean(~np.isfinite(cond))),
                    fraction_condition_gt_1e8=float(np.mean(cond > 1e8)),
                    condition_median=float(np.median(finite)) if len(finite) else np.nan,
                    condition_p95=float(np.quantile(finite, .95)) if len(finite) else np.nan,
                    condition_max=float(np.max(finite)) if len(finite) else np.nan,
                    minimum_eigenvalue=float(values[:, 3 * stage + 1].min()),
                    maximum_eigenvalue=float(values[:, 3 * stage + 2].max()),
                    mean_shrinkage=float(shrink[np.isfinite(shrink)].mean()) if np.isfinite(shrink).any() else np.nan,
                    minimum_class_count=int(values[:, 12].min()),
                    class_fallback_fraction=float(values[:, 13].mean())))
    return rows


def clean_prices(dates, prices, name):
    frame = pd.DataFrame({'date': np.asarray(pd.to_datetime(dates, errors='coerce')),
                          'price': np.asarray(pd.to_numeric(prices, errors='coerce'))})
    bad = frame.date.isna() | ~np.isfinite(frame.price) | frame.price.le(0)
    if bad.any():
        raise ValueError(f'{name}: {bad.sum()} invalid rows. Clean explicitly; no silent row removal.')
    if frame.date.duplicated().any():
        raise ValueError(f'{name}: duplicate dates')
    return frame.sort_values('date').reset_index(drop=True)


def load_data(args):
    loaded, sources = {}, {}
    if any(PROTOCOLS[p].data_kind == 'djia' for p in args.profiles):
        path = Path(args.djia_path).resolve()
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip()
        needed = {'Date', 'Ticker', args.djia_price_column}
        if not needed.issubset(raw):
            raise ValueError(f'{path}: missing columns {needed - set(raw.columns)}')
        if raw.Ticker.isna().any():
            raise ValueError('Missing DJIA ticker identifiers')
        raw['Ticker'] = raw.Ticker.astype(str)
        tickers = sorted(raw.Ticker.unique()) if not args.tickers else args.tickers
        absent = set(tickers) - set(raw.Ticker)
        if absent:
            raise ValueError(f'Unknown tickers: {absent}')
        loaded['djia'] = {}
        quality, invalid_rows, rejected = [], [], []
        for ticker in tickers:
            group = raw.loc[raw.Ticker.eq(ticker)]
            dates = pd.to_datetime(group.Date, errors='coerce')
            prices = pd.to_numeric(group[args.djia_price_column], errors='coerce')
            bad = ~np.isfinite(prices) | prices.le(0)
            invalid_dates = int(dates.isna().sum())
            duplicate_dates = int(dates.duplicated().sum())
            reject = bool(bad.any() or invalid_dates or duplicate_dates)
            quality.append(dict(ticker=ticker, rows=len(group), invalid_prices=int(bad.sum()),
                                invalid_dates=invalid_dates, duplicate_dates=duplicate_dates,
                                action='rejected' if reject else 'kept'))
            if bad.any():
                invalid_rows.append(group.loc[bad].copy())
            if reject:
                rejected.append(ticker)
                continue
            loaded['djia'][ticker] = clean_prices(group.Date, group[args.djia_price_column], ticker)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(quality).to_csv(out / 'data_quality_report.csv', index=False)
        (pd.concat(invalid_rows, ignore_index=True) if invalid_rows else raw.iloc[:0]).to_csv(
            out / 'invalid_price_rows.csv', index=False)
        fatal_dates = any(q['invalid_dates'] or q['duplicate_dates'] for q in quality)
        if fatal_dates or (rejected and args.invalid_price_policy == 'error'):
            raise ValueError(f'DJIA data validation failed for {rejected}; see {out / "data_quality_report.csv"}')
        if not loaded['djia']:
            raise ValueError('No valid DJIA tickers remain; see data_quality_report.csv')
        if rejected:
            print(f'WARNING: Dropping entire invalid-price tickers: {rejected}. '
                  f'Keeping {len(loaded["djia"])} tickers. See data_quality_report.csv. '
                  'Full-file filtering is not a point-in-time universe and does not remove selection/survivorship bias.', flush=True)
        sources['djia'] = path
    if 'XGboost_GRU' in args.profiles:
        path = Path(args.btc_path).resolve()
        # Pickle is executable: use only your own/trusted dataset.
        raw = pd.read_pickle(path)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError('BTC pickle must contain a DataFrame')
        price_col = args.btc_price_column or next((c for c in
            ('Close', 'close', 'Adj Close', 'adj close', 'Price', 'price', 'Last', 'last') if c in raw), None)
        if price_col is None or price_col not in raw:
            raise ValueError('BTC price column not found; specify --btc-price-column')
        date_col = next((c for c in ('Date', 'date', 'Datetime', 'datetime', 'Time', 'time') if c in raw), None)
        if date_col:
            dates = raw[date_col]
        elif isinstance(raw.index, pd.DatetimeIndex):
            dates = raw.index
        else:
            raise ValueError('BTC has no recognized datetime column/index')
        loaded['btc'] = {'BTCUSD': clean_prices(dates, raw[price_col], 'BTCUSD')}
        sources['btc'] = path
    return loaded, sources


def audit_data(protocol, ticker, raw):
    row = dict(profile=protocol.name, ticker=ticker, rows=len(raw),
               first_date=str(raw.date.min()), last_date=str(raw.date.max()),
               median_bar_hours=float(raw.date.diff().dt.total_seconds().median() / 3600))
    for split in ('train', 'test'):
        start, end = pd.Timestamp(getattr(protocol, split + '_start')), pd.Timestamp(getattr(protocol, split + '_end'))
        mask = raw.date.ge(start) & raw.date.lt(end)
        count = int(mask.sum())
        if count < 20 or not (raw.date >= end).any():
            raise ValueError(f'{protocol.name}/{ticker}: insufficient {split} coverage or missing endpoint for forward label')
        row[split + '_rows'] = count
        row[split + '_first'] = str(raw.loc[mask, 'date'].min())
        row[split + '_last'] = str(raw.loc[mask, 'date'].max())
    return row


def asset_worker(task):
    protocol, ticker, raw, candidates, args, task_variant = task
    methods = METHODS if args.selection_method == 'all' else (args.selection_method,)
    designs = ('fixed', 'reselect') if args.lag_design == 'both' else (args.lag_design,)
    metrics, diagnostic_rows, selected_rows, streams = [], [], [], []
    for variant in (task_variant,):
        sample_map = {lags: build_samples(raw, lags, variant == 'maxlag_skew') for lags in candidates}
        anchors = select_lags(sample_map, protocol, SETTINGS[0], args, methods)
        for setting in SETTINGS:
            selected = anchors
            if 'reselect' in designs and setting.name != BASELINE:
                selected = select_lags(sample_map, protocol, setting, args, methods)
            cache = {}
            for design in designs:
                choices = anchors if design == 'fixed' else selected
                for method in methods:
                    lags = choices[method]
                    if lags not in cache:
                        result, diag = run_samples(sample_map[lags], protocol, setting, args, diagnostics=True)
                        metric_template = asset_metrics(result, protocol, {})
                        diagnostic_template = summarize_diagnostics(result, diag, {})
                        compact = result.loc[result.split.ne('unused'), ['date', 'split', 'bar_pnl']].copy()
                        cache[lags] = result, diag, metric_template, diagnostic_template, compact
                    result, diag, metric_template, diagnostic_template, compact = cache[lags]
                    identifier = f'{design}__{method}__{variant}__{setting.name}'
                    metadata = dict(profile=protocol.name, ticker=ticker, id=identifier,
                                    lag_design=design, selection_method=method, variant=variant,
                                    setting=setting.name, lags=str(lags))
                    metrics.extend(dict(metadata, **row) for row in metric_template)
                    diagnostic_rows.extend(dict(metadata, **row) for row in diagnostic_template)
                    selected_rows.append(dict(metadata, baseline_lags=str(anchors[method])))
                    streams.append((metadata, compact))
                    if args.save_paths:
                        path = Path(args.output_dir) / 'asset_paths' / protocol.name / ticker
                        path.mkdir(parents=True, exist_ok=True)
                        detailed = result.copy()
                        detailed['equity_from_history_start'] = np.cumprod(1 + detailed.bar_pnl)
                        for c, label in enumerate(('up', 'down')):
                            for stage, name in enumerate(STAGES):
                                for k, statistic in enumerate(('condition', 'min_eigenvalue', 'max_eigenvalue')):
                                    detailed[f'{name}_{statistic}_{label}'] = diag[:, c, stage * 3 + k]
                        detailed.to_csv(path / f'{identifier}.csv.gz', index=False)
    return metrics, diagnostic_rows, selected_rows, streams


def available_cpus():
    """Respect OS affinity where available; --workers can limit RAM pressure."""
    counter = getattr(os, 'process_cpu_count', os.cpu_count)
    count = counter() or 1
    if hasattr(os, 'sched_getaffinity'):
        count = min(count, len(os.sched_getaffinity(0)))
    return max(1, count)


def pool_initializer():
    # Keep small matrix BLAS work single-threaded in every process.
    global _thread_limiter
    from threadpoolctl import threadpool_limits
    _thread_limiter = threadpool_limits(limits=1)
    if NUMBA_ENABLED:
        numba.set_num_threads(1)


def pool_worker(task):
    result = asset_worker(task)
    return f'{task[0].name}/{task[1]}/{task[5]}', result


def add_baseline_deltas(frame, keys):
    columns = ['return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'annual_volatility_pct']
    baseline = frame.loc[frame.setting.eq(BASELINE), keys + columns].rename(
        columns={c: 'baseline_' + c for c in columns})
    result = frame.merge(baseline, on=keys, how='left', validate='many_to_one')
    for c in columns:
        result['delta_' + c] = result[c] - result['baseline_' + c]
    return result


def build_portfolios(streams, protocols, args):
    grouped = {}
    for metadata, frame in streams:
        grouped.setdefault((metadata['profile'], metadata['id']), []).append((metadata, frame))
    metrics, weights_rows, test_families = [], [], {}
    out = Path(args.output_dir) / 'portfolio_paths'
    out.mkdir(exist_ok=True)
    for (profile, identifier), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item[0]['ticker'])
        tickers = [m['ticker'] for m, _ in items]
        metadata = {k: v for k, v in items[0][0].items() if k not in ('ticker', 'lags')}
        matrices = {}
        for split in ('train', 'test'):
            columns = {m['ticker']: f.loc[f.split.eq(split)].set_index('date').bar_pnl for m, f in items}
            frame = pd.DataFrame(columns).sort_index()
            if frame.index.has_duplicates or not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f'{profile}/{identifier}: unaligned {split} ticker returns; no zero filling')
            matrices[split] = frame[tickers]
        positive = np.array([max(return_metrics(matrices['train'][t], 252)['legacy_return_pct'], 0.0) for t in tickers])
        learned = positive / positive.sum() if positive.sum() > 0 else np.ones(len(tickers)) / len(tickers)
        equal = np.ones(len(tickers)) / len(tickers)
        for policy, split, weights in (('equal_weight', 'train', equal), ('equal_weight', 'test', equal),
                                        ('train_positive', 'test', learned)):
            frame = matrices[split]
            wealth = np.cumprod(1 + frame.to_numpy(), axis=0) @ weights
            r = wealth / np.r_[1., wealth[:-1]] - 1
            values = return_metrics(r, protocols[profile].annualization)
            metrics.append(dict(metadata, allocation=policy, split=split, assets=len(tickers),
                                start=str(frame.index[0]), end=str(frame.index[-1]), **values))
            for ticker, weight in zip(tickers, weights):
                weights_rows.append(dict(metadata, allocation=policy, split=split, ticker=ticker, weight=float(weight)))
            path = pd.DataFrame({'date': frame.index, 'bar_pnl': r, 'equity': wealth})
            path.to_csv(out / f'{profile}__{identifier}__{policy}__{split}.csv.gz', index=False)
            if split == 'test':
                test_families.setdefault(profile, {})[identifier + '__' + policy] = pd.Series(r, index=frame.index)
    return pd.DataFrame(metrics), pd.DataFrame(weights_rows), test_families


def reality_check(frame, repetitions, block_length, seed):
    """Conditional Train-selected family; one-sided max mean vs zero-return cash."""
    x = frame.to_numpy(dtype=float)
    if not np.isfinite(x).all() or len(x) < 20:
        raise ValueError('Reality check needs aligned, finite Test returns, n>=20')
    n, k = x.shape
    length = max(2, round(n ** (1/3))) if block_length is None else block_length
    if not 1 <= length <= n:
        raise ValueError('Expected bootstrap block length outside [1,n]')
    centered = x - x.mean(axis=0)
    observed = np.sqrt(n) * x.mean(axis=0)
    statistic = max(0., float(observed.max()))
    rng = np.random.default_rng(seed)
    maxima = np.empty(repetitions)
    times = np.arange(n)
    for b in range(repetitions):
        restart = rng.random(n) < 1 / length
        restart[0] = True
        anchors = rng.integers(0, n, n)
        last = np.maximum.accumulate(np.where(restart, times, 0))
        indices = (anchors[last] + times - last) % n
        maxima[b] = max(0., float((centered[indices].sum(axis=0) / np.sqrt(n)).max()))
    return dict(test='White Reality Check, unstudentized stationary bootstrap',
                benchmark='zero-return cash', conditional_on='Realized training sample and Train-selected pipelines',
                bars=n, candidates=k, repetitions=repetitions, block_length=length, seed=seed,
                family_p_value=float((1 + np.sum(maxima >= statistic)) / (repetitions + 1)),
                candidate_adjusted_p_values={c: float((1 + np.sum(maxima >= max(0., observed[i]))) / (repetitions + 1))
                                             for i, c in enumerate(frame.columns)},
                limitations=['Not a test of improvement over baseline covariance or buy-and-hold.',
                             'Does not cover historical tuning, prior Test reuse or all new-Train uncertainty.',
                             'Stationarity/weak-dependence assumptions required. DJIA profiles overlap in time.'])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def self_test():
    """Independent sklearn/SciPy reference checks; requires no data or source files."""
    from scipy.stats import multivariate_t
    rng = np.random.default_rng(173)
    for n, d in ((1, 3), (2, 5), (80, 1), (80, 4)):
        x = rng.normal(size=(n, d)) * np.arange(1, d + 1)
        for lw in (False, True):
            mean, empirical, fitted, shrinkage = covariance(x, 1e-6, lw)
            expected_emp = np.atleast_2d(np.cov(x, rowvar=False)) if n > 1 else np.zeros((d, d))
            expected = expected_emp
            if lw and n > 1:
                reference = LedoitWolf().fit(x)
                expected = reference.covariance_
                np.testing.assert_allclose(shrinkage, reference.shrinkage_, atol=1e-12)
            np.testing.assert_allclose(empirical, expected_emp, atol=1e-12)
            np.testing.assert_allclose(fitted, expected + 1e-6 * np.eye(d), atol=1e-12)
            np.testing.assert_allclose(mean, x.mean(axis=0), atol=1e-12)
    x = rng.normal(size=(100, 3))
    x[:, 0] *= .03
    y = rng.integers(0, 2, 100)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, .01, 100)))
    for setting in SETTINGS:
        probability, signal, pnl, position, turnover, diagnostics = rolling_kernel(
            x, y, prices, 30, 1e-6, .9, .001, 10., setting.standardize, setting.ledoit_wolf, setting.matched, True)
        for j in (31, 65, 99):
            window = x[j - 31:j - 1]
            labels = y[j - 31:j - 1]
            center = window.mean(axis=0) if setting.standardize else np.zeros(3)
            scale = window.std(axis=0) if setting.standardize else np.ones(3)
            scale = np.where(scale > 1e-12, scale, 1.)
            z = (window - center) / scale
            logs = []
            for c in (1, 0):
                part = z[labels == c]
                count = len(part)
                if not count:
                    part = z
                matrix = LedoitWolf().fit(part).covariance_ if setting.ledoit_wolf else np.cov(part, rowvar=False)
                matrix = matrix + 1e-6 * np.eye(3)
                if setting.matched:
                    matrix *= .8
                logs.append(np.log((count + 1) / 32) + multivariate_t.logpdf((x[j] - center) / scale,
                    loc=part.mean(axis=0), shape=matrix, df=10))
            expected = 1 / (1 + np.exp(logs[1] - logs[0]))
            np.testing.assert_allclose(probability[j], expected, atol=2e-10)
        np.testing.assert_allclose(pnl[1:], position[:-1] * np.diff(prices) / prices[:-1] - turnover[1:] * .001)
        assert np.isnan(probability[:31]).all()
        altered = x.copy()
        altered[70:] += 100
        check = rolling_kernel(altered, y, prices, 30, 1e-6, .9, .001, 10.,
                               setting.standardize, setting.ledoit_wolf, setting.matched, False)[0]
        np.testing.assert_allclose(probability[:70], check[:70], equal_nan=True)
    # Empty class, singleton class, constant features and nearly singular features.
    for constant in (True, False):
        z = np.ones((50, 3)) if constant else np.column_stack((x[:50, 0], x[:50, 0], x[:50, 1]))
        labels = np.zeros(50, dtype=np.int64)
        labels[20] = 1
        for setting in SETTINGS:
            result = rolling_kernel(z, labels, prices[:50], 10, 1e-6, .9, 0., 10.,
                                    setting.standardize, setting.ledoit_wolf, setting.matched, True)
            assert np.isfinite(result[0][11:]).all()
    metric = return_metrics([-.1, .2], 252)
    np.testing.assert_allclose(metric['return_pct'], 8.)
    np.testing.assert_allclose(metric['max_drawdown_pct'], -10.)
    np.testing.assert_allclose(metric['legacy_return_pct'], 20.)
    print(f'SELF-TEST PASSED (Numba={NUMBA_ENABLED}): covariance, posterior, clock, causality, degeneracy, metrics.')


def parse_args():
    root = Path(__file__).resolve().parent
    def _repo_data(name):
        """Locate a dataset in the repository data/ folder (or next to the script)."""
        here = Path(__file__).resolve().parent
        for base in (here, *here.parents):
            for candidate in (base / "data" / name, base / name):
                if candidate.is_file():
                    return str(candidate)
        return str(here.parent / "data" / name)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--djia-path', default=_repo_data('djia30_full_2009_2022.csv'))
    p.add_argument('--btc-path', default=_repo_data('BTCUSD.pkl'))
    p.add_argument('--output-dir', default='covariance_fast_results')
    p.add_argument('--profiles', nargs='+', choices=list(PROTOCOLS), default=list(PROTOCOLS))
    p.add_argument('--tickers', nargs='+', help='DJIA subset; default is every ticker')
    p.add_argument('--djia-price-column', default='Adj Close')
    p.add_argument('--invalid-price-policy', choices=('drop-tickers', 'error'), default='drop-tickers',
                   help='DJIA only: report and exclude entire invalid-price tickers (default), or fail; dates remain strict')
    p.add_argument('--btc-price-column', default=None)
    p.add_argument('--selection-method', choices=(*METHODS, 'all'), default='wilcoxon')
    p.add_argument('--lag-design', choices=('fixed', 'reselect', 'both'), default='both')
    p.add_argument('--n-lag-sets', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=available_cpus(),
                   help='Pool processes; defaults to available logical CPUs. Lower if RAM is limited.')
    p.add_argument('--student-df', type=float, default=10.)
    p.add_argument('--reg', type=float, default=1e-6)
    p.add_argument('--prob-threshold', type=float, default=.9)
    p.add_argument('--commission', type=float, default=0.)
    p.add_argument('--btc-annualization', type=float, default=2190.)
    p.add_argument('--save-paths', action='store_true', help='Write full per-asset predictions/diagnostic paths (large)')
    p.add_argument('--no-numba', action='store_true', help='Pure NumPy/Python fallback, much slower')
    p.add_argument('--smoke', action='store_true', help='2 candidates, AAPL, fixed design; NOT a full study')
    p.add_argument('--self-test', action='store_true', help='Run independent reference checks and exit')
    p.add_argument('--rc-repetitions', type=int, default=0, help='0 disables; >=99 enables per-profile Test family check')
    p.add_argument('--rc-block-length', type=float, default=None)
    return p.parse_args()


def main(args):
    if args.self_test:
        self_test()
        return
    if args.smoke:
        args.n_lag_sets, args.tickers, args.lag_design = 2, ['AAPL'], 'fixed'
    if len(set(args.profiles)) != len(args.profiles) or (args.tickers and len(set(args.tickers)) != len(args.tickers)):
        raise ValueError('Duplicate profile/ticker requested')
    if args.workers < 1 or args.n_lag_sets < 2:
        raise ValueError('Need workers>=1 and n-lag-sets>=2')
    if not all(np.isfinite(v) for v in (args.reg, args.student_df, args.prob_threshold, args.commission, args.btc_annualization)):
        raise ValueError('Nonfinite numeric parameter')
    if args.reg <= 0 or args.student_df <= 2 or not .5 <= args.prob_threshold < 1 or args.commission < 0 or args.btc_annualization <= 0:
        raise ValueError('Invalid ridge, Student df, threshold, commission or annualization')
    if args.rc_repetitions and args.rc_repetitions < 99:
        raise ValueError('Reality check needs >=99 repetitions')
    if not NUMBA_ENABLED and not args.no_numba:
        raise RuntimeError('Fast mode requires Numba: install numba, or explicitly use --no-numba for debugging')
    if not NUMBA_ENABLED:
        print('WARNING: Numba explicitly disabled; execution will be substantially slower.', flush=True)
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory must be new or empty: {out}')
    args.output_dir = str(out)
    loaded, sources = load_data(args)
    protocols = {name: PROTOCOLS[name] for name in args.profiles}
    if 'XGboost_GRU' in protocols:
        values = asdict(protocols['XGboost_GRU'])
        values['annualization'] = args.btc_annualization
        protocols['XGboost_GRU'] = Protocol(**values)
    candidates = {name: generate_lags(protocol, args.n_lag_sets, args.seed) for name, protocol in protocols.items()}
    audit, tasks = [], []
    for name, protocol in protocols.items():
        for ticker, raw in loaded[protocol.data_kind].items():
            audit.append(audit_data(protocol, ticker, raw))
            # Keep full past history plus the first observation at/after test_end
            # so the last in-range feature row still has its forward label.
            stop = int(np.searchsorted(raw.date.to_numpy(), np.datetime64(protocol.test_end), side='left')) + 1
            for variant in VARIANTS:
                tasks.append((protocol, ticker, raw.iloc[:stop].copy(), candidates[name], args, variant))
    # Long histories first: reduce idle workers at the tail of the run.
    tasks.sort(key=lambda task: len(task[2]) * len(task[3]) * task[0].window, reverse=True)
    args.workers = min(args.workers, len(tasks))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(out / 'data_audit.csv', index=False)
    manifest = dict(arguments=vars(args), protocols={n: asdict(p) for n, p in protocols.items()},
                    settings=[s.name for s in SETTINGS], baseline=BASELINE, lag_candidates=candidates,
                    input_files={k: dict(path=str(v), sha256=sha256(v)) for k, v in sources.items()},
                    script_sha256=sha256(__file__), numba_enabled=NUMBA_ENABLED,
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                  scipy=scipy.__version__, sklearn=sklearn.__version__,
                                  numba=numba.__version__ if numba else None),
                    notes=[
                        'Two physical datasets, three protocols. Source names do not imply actual DRL/GRU implementations.',
                        'End dates are exclusive, exactly as original. Supplied DJIA CSV is trusted, not externally certified.',
                        'DJIA fixed universe/adjustments may contain survivorship bias or synthetic/backfilled history; provenance requires independent audit.',
                        'DJIA invalid-price policy is recorded in arguments; exclusions are in data_quality_report.csv. Full-file filtering is not point-in-time and may introduce selection bias.',
                        'BTC H4 annualization defaults to 2190, not original 365; bar-return calculations are unchanged.',
                        'Window-only scaler; labels in fit end before current observation; no Test lag selection.',
                        'Original delayed position/PnL clock and HOLD rule preserved; no forced liquidation.',
                        'Main return includes first split bar; legacy_return_pct excludes it.',
                        'Empirical ddof=1; Ledoit-Wolf MLE ddof=0. Ridge is retained in all models.',
                        'Student legacy uses covariance estimate as density scale; matched scale=(df-2)/df * covariance estimate.',
                        'Selector concatenation heuristic retained; nonfinite p-values explicitly mapped to 1.',
                        'Fixed lags isolate covariance effects; reselected lags measure complete pipeline sensitivity.',
                        'Train-positive portfolio weights use legacy Train return, then remain fixed on Test (no daily rebalance).',
                        'Train portfolio is equal-weight in-sample only. Test also reports equal-weight comparator.',
                        'No missing ticker/return silently dropped or filled; entire requested run fails on task error.',
                        'Exploratory robustness study, not preregistration; zero default costs mirror originals, not realistic execution.',
                    ])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Warm Numba cache once before spawning; avoids all workers compiling at once.
    if NUMBA_ENABLED:
        print('Preparing Numba kernel (first run may take a few seconds)...', flush=True)
        warm = np.arange(60, dtype=float).reshape(20, 3) / 100
        rolling_kernel(warm, np.arange(20) % 2, np.arange(20, dtype=float) + 100,
                       10, 1e-6, .9, 0., 10., False, False, False, True)
    all_metrics, all_diags, all_selections, all_streams = [], [], [], []
    def collect(result, label):
        m, d, s, r = result
        all_metrics.extend(m)
        all_diags.extend(d)
        all_selections.extend(s)
        all_streams.extend(r)
        print(f'Completed {label}', flush=True)
    print(f'Running {len(tasks)} asset/protocol/variant tasks; {args.workers} processes; '
          f'{args.n_lag_sets} candidates; design={args.lag_design}; Numba={NUMBA_ENABLED}', flush=True)
    if args.workers == 1:
        pool_initializer()
        for task in tasks:
            label, result = pool_worker(task)
            collect(result, label)
    else:
        with mp.get_context('spawn').Pool(processes=args.workers, initializer=pool_initializer) as pool:
            for label, result in pool.imap_unordered(pool_worker, tasks, chunksize=1):
                collect(result, label)
    keys = ['profile', 'ticker', 'lag_design', 'selection_method', 'variant', 'split']
    assets = add_baseline_deltas(pd.DataFrame(all_metrics), keys)
    assets.sort_values(keys + ['setting']).to_csv(out / 'asset_performance.csv', index=False)
    pd.DataFrame(all_diags).sort_values(keys + ['setting', 'class_name', 'matrix']).to_csv(out / 'condition_summary.csv', index=False)
    pd.DataFrame(all_selections).sort_values(['profile', 'ticker', 'id']).to_csv(out / 'selected_lags.csv', index=False)
    portfolio, weights, families = build_portfolios(all_streams, protocols, args)
    keys = ['profile', 'lag_design', 'selection_method', 'variant', 'allocation', 'split']
    portfolio = add_baseline_deltas(portfolio, keys)
    portfolio.sort_values(keys + ['setting']).to_csv(out / 'portfolio_performance.csv', index=False)
    weights.to_csv(out / 'portfolio_weights.csv', index=False)
    if args.rc_repetitions:
        for profile, streams in families.items():
            frame = pd.DataFrame(streams).sort_index()
            report = reality_check(frame, args.rc_repetitions, args.rc_block_length, args.seed)
            (out / f'{profile}__reality_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'COMPLETE.txt').write_text('All requested tasks completed. ' + ('SMOKE TEST ONLY.' if args.smoke else '') + '\n', encoding='utf-8')
    print(f'Finished. Reports: {out}', flush=True)
    shown = portfolio.loc[portfolio.split.eq('test') & portfolio.allocation.eq('train_positive')]
    print(shown[['profile', 'lag_design', 'variant', 'setting', 'return_pct', 'sharpe_ratio']].to_string(index=False))


if __name__ == '__main__':
    mp.freeze_support()
    main(parse_args())
