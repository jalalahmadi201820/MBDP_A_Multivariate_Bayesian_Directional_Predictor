"""Multi-seed lag-selection stability for BTC and the two equity profiles (Reviewer 4 request).

Repeats the 100-candidate protocol with 10 different seeds, keeping everything else
identical to the exhaustive-pair run: training-only raw-Sharpe selection, final
training row purged, reference configuration (ridge 1e-6, df 10, tau 0.9, 10 bps).
Outputs -> outputs/multi_seed_lag_stability/
"""
import os
WORKERS = int(os.environ.get('WORKERS', 0)) or min(os.cpu_count() or 2, 20)
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, time, json, hashlib, itertools
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _sp in (HERE, HERE.parent / 'Covariance_Robustness', HERE.parent / 'MBDP_Core_Engine'):
    if (_sp / 'run_covariance_grid.py').exists():
        sys.path.insert(0, str(_sp))
        break
import run_covariance_grid as e

SEEDS = tuple(int(s) for s in os.environ.get('SEEDS', '42').split(','))
OUT = Path(os.environ.get('MBDP_OUT') or (HERE / 'outputs' / 'multi_seed_lag_stability'))
OUT.mkdir(parents=True, exist_ok=True)
PROFILES = tuple(p.strip() for p in os.environ.get('PROFILES', 'DRL_Ensemble,FineRL,XGboost_GRU').split(',') if p.strip())


def _data_root():
    for p in (os.environ.get('MBDP_DATA'), '../Data', './Data', '/home/ali/Desktop/Test/Data'):
        if p and (Path(p) / 'djia30_full_2009_2022.csv').exists():
            return Path(p)
    raise SystemExit("Error: 'djia30_full_2009_2022.csv' not found; set MBDP_DATA.")


a = SimpleNamespace(profiles=list(e.PROTOCOLS), djia_path=str(_data_root() / 'djia30_full_2009_2022.csv'),
                    btc_path=str(_data_root() / 'BTCUSD.pkl'), djia_price_column='Adj Close',
                    btc_price_column=None, tickers=None, output_dir=str(OUT),
                    invalid_price_policy='drop-tickers', reg=1e-6, student_df=10.,
                    prob_threshold=.9, commission=.001)


def summarize(frame, protocol):
    t = frame[frame.split.eq('test')]
    good = t.prob_up.notna()
    p = t.loc[good, 'prob_up'].to_numpy(dtype=float)
    y = t.loc[good, 'label'].to_numpy(dtype=float)
    m = e.return_metrics(t.bar_pnl.to_numpy(dtype=float), protocol.annualization)
    m.update(brier=float(np.mean((p - y) ** 2)), accuracy=float(np.mean((p > .5) == y)),
             turnover=float(t.turnover_units.sum()), position_changes=int(t.turnover_units.gt(0).sum()),
             signal_records=int(t.signal.ne(0).sum()))
    return m


def task(item):
    profile, ticker, variant, seed, raw = item
    proto = e.PROTOCOLS[profile]
    use_skew = variant == 'maxlag_skew'
    candidates = e.generate_lags(proto, 100, seed)
    args_sel = SimpleNamespace(**vars(a)); args_sel.commission = 0.
    samples = {lags: e.build_samples(raw, lags, use_skew) for lags in candidates}
    train = {lags: s.loc[s.date < pd.Timestamp(proto.train_end)].iloc[:-1] for lags, s in samples.items()}
    selected = e.select_lags(train, proto, e.SETTINGS[0], args_sel, ('raw_sharpe',))
    rows, streams = [], []
    for selector, lags in selected.items():
        args_bt = SimpleNamespace(**vars(a))
        args_bt.reg, args_bt.student_df, args_bt.prob_threshold, args_bt.commission = 1e-6, 10., .9, .001
        res, _ = e.run_samples(samples[lags], proto, e.SETTINGS[0], args_bt)
        meta = dict(profile=profile, ticker=ticker, variant=variant, seed=seed, selector=selector,
                    lags=str(lags), candidates=len(candidates), family='reference')
        rows.append({**meta, **summarize(res, proto)})
        t = res[res.split.eq('test')]
        streams.append((meta, t[['date', 'bar_pnl']]))
    return rows, streams


def main():
    t0 = time.time()
    data, sources = e.load_data(a)
    tasks = [(prof, ticker, variant, seed, raw) for prof in PROFILES for seed in SEEDS
             for variant in ('regular', 'maxlag_skew')
             for ticker, raw in data[e.PROTOCOLS[prof].data_kind].items()]
    print(f'{len(tasks)} tasks on {WORKERS} workers; seeds={SEEDS}', flush=True)
    rows, streams = [], []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for ms, ss in pool.map(task, tasks):
            rows += ms; streams += ss
    pd.DataFrame(rows).to_csv(OUT / 'asset_metrics.csv', index=False)

    grouped = {}
    for meta, frame in streams:
        key = tuple((k, v) for k, v in meta.items() if k not in ('ticker', 'lags'))
        grouped.setdefault(key, []).append(frame)
    summary = []
    for key, frames in grouped.items():
        meta = dict(key); proto = e.PROTOCOLS[meta['profile']]
        joined = pd.concat([f.set_index('date').bar_pnl.rename(str(i)) for i, f in enumerate(frames)], axis=1)
        if joined.isna().any().any():
            raise ValueError('Unsynchronized assets')
        summary.append(dict(meta, **e.return_metrics(joined.mean(axis=1), proto.annualization), assets=len(frames)))
    port = pd.DataFrame(summary)
    port.to_csv(OUT / 'portfolio_summary_by_seed.csv', index=False)

    # Variability of the selected lag structures and of test performance across seeds.
    per_ticker = pd.DataFrame(rows)
    stab = []
    for (prof, variant), grp in per_ticker.groupby(['profile', 'variant']):
        ref = grp[grp.seed.eq(SEEDS[0])].set_index('ticker').lags
        for seed, sub in grp.groupby('seed'):
            cur = sub.set_index('ticker').lags
            common = cur.index.intersection(ref.index)
            jac = float(np.mean([len(set(cur[t].split(', ')) & set(ref[t].split(', '))) /
                                 max(1, len(set(cur[t].split(', ')) | set(ref[t].split(', ')))) for t in common]))
            stab.append(dict(profile=prof, variant=variant, seed=seed, tickers=len(cur),
                             distinct_lag_sets=int(cur.nunique()),
                             agreement_with_seed42=jac))
    pd.DataFrame(stab).to_csv(OUT / 'lag_stability_by_seed.csv', index=False)

    agg = (port.groupby(['profile', 'variant'])
               .agg(seeds=('seed', 'nunique'), sharpe_mean=('sharpe_ratio', 'mean'),
                    sharpe_sd=('sharpe_ratio', 'std'), sharpe_min=('sharpe_ratio', 'min'),
                    sharpe_max=('sharpe_ratio', 'max'), return_mean=('return_pct', 'mean'),
                    return_sd=('return_pct', 'std'), mdd_mean=('max_drawdown_pct', 'mean'))
               .reset_index())
    agg.to_csv(OUT / 'performance_variability.csv', index=False)
    (OUT / 'manifest.json').write_text(json.dumps({
        'experiment': 'multi_seed_lag_stability', 'seeds': list(SEEDS), 'profiles': list(PROFILES),
        'workers': WORKERS, 'candidates_per_seed': 100,
        'protocol': 'Same 100-candidate protocol as the exhaustive-pair run, repeated for 10 seeds; '
                    'training-only raw-Sharpe selection, final training row purged, reference config, 10 bps.',
        'sources': {k: {'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                    for k, p in sources.items()},
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
        'runtime_seconds': round(time.time() - t0, 1)}, indent=2))
    print('TOTAL SECONDS', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
