"""Regenerate the two BTC diagnostics from saved, zero-cost raw-PnL streams."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
O=R/'outputs'/'btc_diagnostic_plots'; O.mkdir(parents=True,exist_ok=True)
frames=[]
for variant in ('regular','maxlag_skew'):
 paths=list((R/'outputs'/'btc_raw_pnl_failure_analysis').glob(f'XGboost_GRU_BTCUSD_{variant}_sampled_raw_pnl_failure_zero_cost_*.csv.gz'))
 assert len(paths)==1
 frames.append(pd.read_csv(paths[0],parse_dates=['date']))
titles=['Regular','Max-Lag Skewness']
fig,ax=plt.subplots(3,2,figsize=(16,12))
for i,(f,title) in enumerate(zip(frames,titles)):
 ax[0,i].plot(f.date,f.lag_min,label='lag min'); ax[0,i].plot(f.date,f.lag_max,label='lag max')
 ax[0,i].set_title(title);ax[0,i].set_ylabel('Selected lag (bars)');ax[0,i].legend()
 ax[1,i].plot(f.date,f.prob_up,lw=.45)
 for y in [.1,.9]:ax[1,i].axhline(y,color='r',ls='--',lw=.7)
 ax[1,i].set_ylim(0,1);ax[1,i].set_ylabel('Up-class posterior')
 ax[2,i].plot(f.date,100*((1+f.bar_pnl).cumprod()-1));ax[2,i].set_ylabel('Cumulative return (%)')
 for j in range(3):ax[j,i].grid(alpha=.2);ax[j,i].tick_params(axis='x',rotation=30)
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(O/f'btc_diagnostic_timeseries.{ext}',dpi=150)
plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(12,4))
for i,(f,title) in enumerate(zip(frames,titles)):
 for y,label in [(0,'Realized down'),(1,'Realized up')]:
  ax[i].hist(f.loc[f.label==y,'prob_up'].dropna(),bins=np.linspace(0,1,31),density=True,alpha=.5,label=label)
 ax[i].set_title(title);ax[i].set_xlabel('Up-class posterior');ax[i].set_ylabel('Density');ax[i].legend()
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(O/f'btc_posterior_distributions.{ext}',dpi=150)
plt.close(fig)
