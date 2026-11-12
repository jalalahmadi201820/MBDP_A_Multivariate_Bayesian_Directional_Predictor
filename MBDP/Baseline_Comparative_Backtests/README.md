# 03 — Baseline Comparative Backtests

Source-protocol backtests for the three evaluation profiles. Each script reproduces the split, feature set and evaluation window of the corresponding source study, and runs the MBDP ablations inside that protocol.

| File | Profile | Market / frequency |
|---|---|---|
| `run_djia_drl_ensemble.py` | DRL-Ensemble | Dow Jones, daily, crash-inclusive window |
| `run_djia_finerl.py` | FineRL | Dow Jones, daily, post-pandemic window |
| `run_bitcoin_xgboost_gru.py` | XGboost_GRU | Bitcoin, H4 bars, contraction window |

## Usage

```bash
python run_djia_drl_ensemble.py
python run_djia_finerl.py
python run_bitcoin_xgboost_gru.py --help
```

The two DJIA scripts take an optional model name (`MBDP` by default) and read their data path from a relative path constant near the top of `main()`; edit it if your CSV lives elsewhere. The Bitcoin script exposes a full command-line interface for paths, window size, lag range, horizon, threshold, commission and output location.

## What is compared

- The MBDP ablation grid: four selectors (`raw_pnl`, `raw_sharpe`, `wilcoxon`, `ttest`) crossed with two variants (`regular`, `maxlag_skew`).
- The source-study baselines of the respective profile, evaluated on their own published windows.

These rows are contextual: they reproduce the source protocol rather than a common one. The protocol-matched comparison is in `Synchronized_Baselines_and_Reality_Check`.

## Outputs

A per-configuration CSV with cumulative return, annualized return and volatility, Sharpe, Calmar, maximum drawdown, win rate and trade count, plus the equity series used by the manuscript figures.
