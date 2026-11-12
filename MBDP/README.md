# MBDP — Code for the Reported Results

Python code that produces every numerical result in the manuscript and in the response to the referees. The executable logic is unchanged from the runs that generated the reported numbers; only leading module docstrings, non-English comments, decorative separators and file names were cleaned, and two driver scripts were pointed at the shared engine in `01_MBDP_Core_Engine`.

`SOURCE_MAP.md` maps every packaged file back to the script it was extracted from.

## Requirements

```bash
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `scikit-learn`, `numba`, `threadpoolctl`. Python 3.10+ recommended. All runs are CPU-only; the engine parallelizes across processes.

## Input data

Both datasets are provided in the repository-level [`data/`](../data/) folder:

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

## Conventions shared by all experiments

- Every model in a comparison uses the same dates, features, horizon and execution rule.
- Lag selection, class-conditional fits and portfolio weights use the training partition only.
- The Reality Check uses block bootstrap with a fixed seed and 5,000 repetitions.
- Commissions are quoted per side; spread, slippage and funding are held at zero unless the script says otherwise.
