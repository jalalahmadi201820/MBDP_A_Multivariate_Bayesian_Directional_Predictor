"""Reproducible supplementary robustness runs; not a replacement for archived tables.
Run from /mnt/data: OPENBLAS_NUM_THREADS=1 python revision_experiments.py
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys, json, itertools, random, hashlib, time
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
REPO=ROOT/'Code/MBDP_A_Multivariate_Bayesian_Directional_Predictor-main'
sys.path.insert(0,str(REPO/'MBDP/Covariance_Robustness'))
import run_covariance_grid as e
from experiment_plan import SPECS, pools_for, methods_for, configs_for
EXPERIMENT=sys.argv[1]
if EXPERIMENT not in SPECS: raise ValueError(EXPERIMENT)
OUT=ROOT/'outputs'/EXPERIMENT; OUT.mkdir(parents=True,exist_ok=True)
a=SimpleNamespace(profiles=list(e.PROTOCOLS),djia_path=str(REPO/'Data/djia30_full_2009_2022.csv'),btc_path=str(REPO/'Data/BTCUSD.pkl'),djia_price_column='Adj Close',btc_price_column=None,tickers=None,output_dir=str(OUT),invalid_price_policy='drop-tickers',reg=1e-6,student_df=10.,prob_threshold=.9,commission=.001)

def summarize(frame, protocol):
    t=frame[frame.split.eq('test')]; good=t.prob_up.notna(); p=t.loc[good,'prob_up'].to_numpy(); y=t.loc[good,'label'].to_numpy()
    m=e.return_metrics(t.bar_pnl,protocol.annualization)
    m.update(brier=float(np.mean((p-y)**2)),accuracy=float(np.mean((p>.5)==y)),turnover=float(t.turnover_units.sum()),position_changes=int(t.turnover_units.gt(0).sum()),nonzero_position_segments=int(((t.position!=0)&(t.position!=t.position.shift(1))).sum()),signal_records=int(t.signal.ne(0).sum()))
    return m

def task(item):
    name,ticker,raw=item; proto=e.PROTOCOLS[name]; rows=[]; choices=[]; streams=[]
    pools={'sampled':e.generate_lags(proto,100,42)}
    if name=='XGboost_GRU':
        pools['all_pairs']=list(itertools.combinations(range(1,31),2))
        pools['three_lags']=random.Random(42).sample(list(itertools.combinations(range(1,31),3)),100)
    pools={k:v for k,v in pools.items() if k in pools_for(EXPERIMENT)}
    for variant in ('regular','maxlag_skew'):
        for pool,candidates in pools.items():
            # Candidate evaluation uses training data only, zero commission as in original selection.
            args=SimpleNamespace(**vars(a)); args.commission=0.
            samples={lags:e.build_samples(raw,lags,variant=='maxlag_skew') for lags in candidates}
            methods=('raw_pnl','raw_sharpe','wilcoxon') if pool=='sampled' and name=='XGboost_GRU' else ('raw_sharpe',)
            methods=methods_for(EXPERIMENT)
            # Purge the final training row: its forward label can touch the test boundary.
            training_samples={lags:s.loc[s.date < pd.Timestamp(proto.train_end)].iloc[:-1] for lags,s in samples.items()}
            selected=e.select_lags(training_samples,proto,e.SETTINGS[0],args,methods)
            for method,lags in selected.items():
                choices.append(dict(profile=name,ticker=ticker,variant=variant,pool=pool,selector=method,lags=str(lags),candidates=len(candidates)))
                configs=[('reference',1e-6,10.,.9,.001)]
                if pool=='sampled' and method=='raw_sharpe':
                    configs += [('ridge',v,10.,.9,.001) for v in (1e-8,1e-7,1e-5,1e-4)]
                    configs += [('df',1e-6,v,.9,.001) for v in (3.,4.,5.,6.,7.,8.)]
                    configs += [('threshold',1e-6,10.,v,.001) for v in (.6,.7,.8,.95)]
                    configs += [('cost',1e-6,10.,.9,v) for v in (0.,.0005,.002)]
                if name=='XGboost_GRU' and pool=='sampled' and method=='raw_pnl':
                    configs += [('failure_zero_cost',1e-6,10.,.9,0.)]
                configs=configs_for(EXPERIMENT,configs)
                for family,ridge,df,threshold,cost in configs:
                    args.reg=ridge;args.student_df=df;args.prob_threshold=threshold;args.commission=cost
                    diagnostic=(family in ('reference','ridge','df'))
                    result,diag=e.run_samples(samples[lags],proto,e.SETTINGS[0],args,diagnostics=diagnostic)
                    meta=dict(profile=name,ticker=ticker,variant=variant,pool=pool,selector=method,family=family,ridge=ridge,df=df,threshold=threshold,cost=cost)
                    values=summarize(result,proto)
                    if diagnostic:
                        cond=diag[(result.split.eq('test') & result.prob_up.notna()).to_numpy(),:,3].ravel()
                        values.update(condition_median=float(np.median(cond)),condition_p95=float(np.quantile(cond,.95)),condition_max=float(np.max(cond)))
                    rows.append(dict(meta,**values))
                    t=result[result.split.eq('test')].copy()
                    t['lag_min']=min(lags);t['lag_max']=max(lags)
                    ident=f'{name}_{ticker}_{variant}_{pool}_{method}_{family}_{ridge}_{df}_{threshold}_{cost}'
                    if name=='XGboost_GRU': t.to_csv(OUT/(ident+'.csv.gz'),index=False)
                    streams.append((meta,t[['date','bar_pnl','turnover_units','signal','prob_up','label','position']]))
    pd.DataFrame(rows).to_csv(OUT/f'asset_metrics_{name}_{ticker}.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/f'lags_{name}_{ticker}.csv',index=False)
    print('DONE',name,ticker,flush=True)
    return rows,choices,streams

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    data,sources=e.load_data(a)
    # BTC first, then both equity profiles; full-file invalid-ticker exclusion is disclosed.
    tasks=[(name,t,raw) for name in ('XGboost_GRU','DRL_Ensemble','FineRL') for t,raw in data[e.PROTOCOLS[name].data_kind].items()]
    if SPECS[EXPERIMENT][1]: tasks=[t for t in tasks if t[0]=='XGboost_GRU']
    audits=[e.audit_data(e.PROTOCOLS[n],t,r) for n,t,r in tasks]
    pd.DataFrame(audits).to_csv(OUT/'data_audit.csv',index=False)
    metrics=[];choices=[];streams=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for ms,cs,ss in pool.map(task,tasks): metrics+=ms;choices+=cs;streams+=ss
    pd.DataFrame(metrics).to_csv(OUT/'asset_metrics.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'selected_lags.csv',index=False)
    grouped={}
    for meta,frame in streams:
        meta={k:v for k,v in meta.items() if k!='ticker'}; key=tuple(meta.items());grouped.setdefault(key,[]).append(frame)
    summary=[]
    for key,frames in grouped.items():
        meta=dict(key); proto=e.PROTOCOLS[meta['profile']]
        # Equal-weight return across valid assets, matching synchronized observation dates.
        joined=pd.concat([f.set_index('date').bar_pnl.rename(str(i)) for i,f in enumerate(frames)],axis=1)
        if joined.isna().any().any(): raise ValueError('Unsynchronized assets')
        vals=e.return_metrics(joined.mean(axis=1),proto.annualization)
        allf=pd.concat(frames); fit=allf[allf.prob_up.notna()]
        vals.update(brier=float(np.mean((fit.prob_up-fit.label)**2)),accuracy=float(np.mean((fit.prob_up>.5)==fit.label)),turnover=int(allf.turnover_units.sum()),position_changes=int(allf.turnover_units.gt(0).sum()),signal_records=int(allf.signal.ne(0).sum()),assets=len(frames))
        summary.append(dict(meta,**vals))
    pd.DataFrame(summary).to_csv(OUT/'portfolio_summary.csv',index=False)
    manifest={'experiment':EXPERIMENT,'sources':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in sources.items()},'seed':42,'protocol':'Train-only raw-Sharpe lag selection, fixed lags across scalar sensitivity sweeps; raw empirical Student-t scale; original engine timing; test results exploratory. BTC exhaustive pairs 1..30; 100 sampled triples. Equity candidate pools follow supplied profiles. No test optimization.', 'versions':{'numpy':np.__version__,'pandas':pd.__version__}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
