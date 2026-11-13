# MBDP — Research Code and Supplementary Experiments

This directory contains code associated with the manuscript and referee response. Folders `01`–`08` contain the archived-result workflows; `Supplementary_Experiments` contains the additional exploratory analyses prepared during revision. The supplementary runs are not replacements for, or independent validation of, the archived results.

The original packaging notes describe comment/docstring cleanup, file renaming and import/path adjustments in folders `01`–`08`. Those notes do not establish numerical equivalence of a fresh execution. Archived numerical tables and walk-forward dates have not been independently validated in the supplementary audit.

`SOURCE_MAP.md` records the supplied provenance of the original workflows and the derivation of the supplementary scripts.

## Requirements

The following instructions concern folders `01`–`08`. For the supplementary environment and commands, see [`Supplementary_Experiments/README.md`](Supplementary_Experiments/README.md).

```bash
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `scikit-learn`, `numba`, `threadpoolctl`. Python 3.10+ recommended. All runs are CPU-only; the engine parallelizes across processes.

## Input data

For folders `01`–`08`, the supplied layout places both datasets in the repository-level [`data/`](../data/) folder. The descriptions below are inherited from the supplied documentation, not a new data audit. `Supplementary_Experiments` instead uses the data bundled in its own `Code.zip`; see its README.

| File | Content |
|---|---|
| `data/djia30_full_2009_2022.csv` | Daily OHLCV panel of the Dow Jones constituents, 2009-04-01 to 2022-03-31 (98,220 rows, 3,274 trading days). |
| `data/BTCUSD.pkl` | Bitcoin/USD H4 bars, 2020-01-01 to 2026-05-31 (14,052 bars, 2,342 days). |

Run the scripts from the repository root and pass the paths explicitly, for example `--stock-path data/djia30_full_2009_2022.csv --btc-path data/BTCUSD.pkl`. Two drivers — `07_Bitcoin_Regime_Windows` and `08_Cost_Sensitivity` — also read a small JSON manifest of the recorded run arguments; point them at your copy with the environment variables `MBDP_DATA_ROOT` (folder containing the manifests and the input data) and `MBDP_OUTPUT_DIR` (where results are written), which default to the current directory and `btc_windows` / `cost_sensitivity` respectively.

## Task folders

| Folder | Task | Paper element |
|---|---|---|
| `01_MBDP_Core_Engine` | Shared engine: rolling Bayesian classifier, lag selectors, LDA/logistic/buy-and-hold/momentum references, White Reality Check machinery. | Used by all experiments. |
| `02_Synchronized_Baselines_and_Reality_Check` | Five-model synchronized comparison plus Reality Check and approximate deflated Sharpe ratio. | Synchronized baselines table; Reality Check table. |
| `03_Baseline_Comparative_Backtests` | Source-protocol comparative backtests for the three profiles. | Comparative performance and MBDP ablation table. |
| `04_Lag_Selection_Sensitivity` | 162-scenario lag-selection grid. | Lag-selection sensitivity table. |
| `05_Covariance_Robustness` | Eight-setting covariance and scaling grid. | Covariance robustness table. |
| `06_Walk_Forward_Evaluation` | Rolling-origin evaluation with chronological splits. | Walk-forward table. |
| `07_Bitcoin_Regime_Windows` | Nine Bitcoin regime windows with ensemble Reality Check. | Bitcoin regime-window table. |
| `08_Cost_Sensitivity` | Exact repricing at 0, 5, 10 and 20 basis points per side. | Cost-aware table. |
| `Supplementary_Experiments` | Eight separately runnable exploratory experiments and one diagnostic plotting script, with shared engine and bundled inputs. | Supplementary sensitivity analyses and BTC diagnostics; not validation of archived tables. |

## File map

- `01_MBDP_Core_Engine/mbdp_rolling_bayesian_classifier.py` — shared engine and command-line runner.
- `01_MBDP_Core_Engine/mbdp_unified_trading_engine.py` — unified three-profile engine used by the lag grid.
- `02_Synchronized_Baselines_and_Reality_Check/run_synchronized_baseline_suite.py` — the same engine run with the five synchronized models and Reality Check families.
- `02_Synchronized_Baselines_and_Reality_Check/compute_deflated_sharpe_ratio.py` — standalone approximate DSR calculation (`N = 800`).
- `03_Baseline_Comparative_Backtests/run_djia_drl_ensemble.py` — DRL-Ensemble Dow Jones protocol.
- `03_Baseline_Comparative_Backtests/run_djia_finerl.py` — FineRL Dow Jones protocol.
- `03_Baseline_Comparative_Backtests/run_bitcoin_xgboost_gru.py` — Bitcoin H4 protocol.
- `04_Lag_Selection_Sensitivity/run_162_scenario_lag_grid.py` — lag grid.
- `05_Covariance_Robustness/run_covariance_grid.py` — covariance grid.
- `06_Walk_Forward_Evaluation/prepare_walkforward_dataset.py` — panel preparation used before the walk-forward runs.
- `06_Walk_Forward_Evaluation/run_walkforward_drl_ensemble.py`, `run_walkforward_finerl.py`, `run_walkforward_bitcoin.py` — rolling-origin runs.
- `07_Bitcoin_Regime_Windows/run_bitcoin_regime_windows.py` — nine-window Bitcoin runner; imports the engine from `01_MBDP_Core_Engine`.
- `08_Cost_Sensitivity/compute_cost_sensitivity.py` — commission repricing; imports the engine from `01_MBDP_Core_Engine`.

## Quick checks

```bash
# numerical self-test of the engine, no data required
python 01_MBDP_Core_Engine/mbdp_rolling_bayesian_classifier.py --self-test

# deflated Sharpe ratio from the walk-forward outputs
python 02_Synchronized_Baselines_and_Reality_Check/compute_deflated_sharpe_ratio.py
```

## Protocol scope and verification limits

- The archived workflows document synchronized comparisons and training-based fitting/selection. Consult each script for its actual dates, features, horizon and execution rule; these are not universal guarantees across all folders.
- The archived Reality Check workflow documents a fixed-seed block bootstrap with 5,000 repetitions. The supplementary experiments do not add a new Reality Check or DSR calculation.
- Commissions are quoted per side; a commission sweep is not a full spread, slippage or funding model.
- Supplementary lag selection uses training data, with selected lags held fixed during scalar sensitivity sweeps. Full-file equity filtering still leaves selection/survivorship bias, and ex-post choice of analyses remains a limitation.
- Supplementary BTC annualization uses 2,190 four-hour bars/year. Protocol and annualization differences prevent treating these runs as direct replacements for archived numbers.
- Saved-output/source audits are consistency checks, not independent model refitting, validation of all historic claims, or visual PDF review. Syntax and experiment-routing checks passed for the separated scripts; the complete separated experiment suite has not been rerun.

## Supplementary results

See `Supplementary_Experiments/README.md` for individual experiment commands and output descriptions. If distributed alongside the code, `Complete_Supplementary_Results.csv` contains 118 portfolio summaries, 2,062 asset-metric rows and 122 selected-lag rows from the original combined supplementary run. It excludes archived tables and raw per-bar streams; it is not newly generated evidence from the separated launchers.
