# Source Map

Each packaged script is a cleaned copy of the script that produced the reported numbers. Cleaning removed leading module docstrings, non-English comments, decorative separator lines and shebang lines, renamed the files to describe the task they perform, and updated two internal imports to the renamed shared engine. **No executable logic, parameter value or numerical constant was changed.**

| Packaged file | Original script (file name only) |
|---|---|
| `01_MBDP_Core_Engine/mbdp_rolling_bayesian_classifier.py` | `bias_tests_fast.py` |
| `01_MBDP_Core_Engine/mbdp_unified_trading_engine.py` | `Unified_Trading_Fast.py` (lag-selection sensitivity suite) |
| `02_Synchronized_Baselines_and_Reality_Check/run_synchronized_baseline_suite.py` | `bias_tests_fast.py` |
| `02_Synchronized_Baselines_and_Reality_Check/compute_deflated_sharpe_ratio.py` | DSR routine extracted from the response-generation script (`dsr_value`) |
| `03_Baseline_Comparative_Backtests/run_djia_drl_ensemble.py` | `DRL_Ensemble.py` |
| `03_Baseline_Comparative_Backtests/run_djia_finerl.py` | `FineRL.py` |
| `03_Baseline_Comparative_Backtests/run_bitcoin_xgboost_gru.py` | `XGboost_GRU.py` |
| `04_Lag_Selection_Sensitivity/run_162_scenario_lag_grid.py` | `Unified_Trading_Fast.py` (lag-selection sensitivity suite) |
| `05_Covariance_Robustness/run_covariance_grid.py` | `covariance_study_all_fast.py` |
| `06_Walk_Forward_Evaluation/prepare_walkforward_dataset.py` | `run.py` (walk-forward suite) |
| `06_Walk_Forward_Evaluation/run_walkforward_drl_ensemble.py` | `fast_drl.py` |
| `06_Walk_Forward_Evaluation/run_walkforward_finerl.py` | `fast_finerl.py` |
| `06_Walk_Forward_Evaluation/run_walkforward_bitcoin.py` | `fast_xgboost.py` |
| `07_Bitcoin_Regime_Windows/run_bitcoin_regime_windows.py` | `gen_btc_windows.py` |
| `08_Cost_Sensitivity/compute_cost_sensitivity.py` | `gen_cost_curve.py` |

## Path configuration

Three scripts read their input and output locations from configuration rather than from constants: `07_Bitcoin_Regime_Windows/run_bitcoin_regime_windows.py` and `08_Cost_Sensitivity/compute_cost_sensitivity.py` use the environment variables `MBDP_DATA_ROOT` and `MBDP_OUTPUT_DIR`, and `02_Synchronized_Baselines_and_Reality_Check/compute_deflated_sharpe_ratio.py` takes `--walkforward-root`. The defaults are relative, so the scripts run from a checkout once the data and the recorded run manifests are in place.
