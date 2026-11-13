# Source Map

This map distinguishes the archived-workflow provenance supplied with folders `01`–`08` from the supplementary experiment code prepared during revision. A source mapping is not proof that a fresh run reproduces every reported number.

## Archived workflows: supplied provenance

The original packaging notes describe removal of leading docstrings, non-English comments, decorative separators and shebangs, file renaming, and import/path adjustments. The mappings below retain that supplied provenance; full source-to-source equivalence and independent numerical reproduction have not been established here.

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

## Path configuration for archived workflows

Three scripts read their input and output locations from configuration rather than from constants: `07_Bitcoin_Regime_Windows/run_bitcoin_regime_windows.py` and `08_Cost_Sensitivity/compute_cost_sensitivity.py` use the environment variables `MBDP_DATA_ROOT` and `MBDP_OUTPUT_DIR`, and `02_Synchronized_Baselines_and_Reality_Check/compute_deflated_sharpe_ratio.py` takes `--walkforward-root`. The defaults are relative; inspect each script's argument handling and provide the required data and recorded run manifests before execution.

## Supplementary experiments: verified derivation

Paths below are relative to `Supplementary_Experiments/`.

| Packaged file | Source / role |
|---|---|
| `original_combined_run.py` | Copy of `revision_experiments.py`, the combined supplementary runner used for the saved exploratory results. |
| `shared_engine.py` | Derived from `revision_experiments.py`; adds experiment-specific routing, output directories and manifest identification. |
| `experiment_plan.py` | New routing definitions selecting profiles, candidate pools, lag selectors and configuration families. |
| `ridge_lambda_sensitivity.py` | New launcher: reference plus Ridge sensitivity configurations. |
| `student_t_df_sensitivity.py` | New launcher: reference plus Student-t degrees-of-freedom configurations. |
| `btc_exhaustive_lag_pairs.py` | New launcher: sampled reference pairs versus all pairs in lags 1–30, BTC only. |
| `btc_sampled_lag_triples.py` | New launcher: sampled reference pairs versus 100 sampled triples, BTC only. |
| `posterior_threshold_sensitivity.py` | New launcher: reference plus posterior-threshold configurations. |
| `transaction_cost_sensitivity.py` | New launcher: reference plus commission configurations. |
| `btc_raw_pnl_failure_analysis.py` | New launcher: raw-PnL selection with reference and zero-commission configurations, BTC only. |
| `btc_lag_selector_comparison.py` | New launcher: raw-PnL, raw-Sharpe and Wilcoxon selectors at reference settings, BTC only. |
| `btc_diagnostic_plots.py` | Derived from `plot_revision.py`; input/output paths point to the separated failure-analysis and plot directories. |
| `Code.zip` | Bundled original repository and data; extract locally to create `Code/`. The supplementary engine imports `Code/MBDP_A_Multivariate_Bayesian_Directional_Predictor-main/MBDP/Covariance_Robustness/run_covariance_grid.py`, not the renamed engine in folder `01`. |
| `requirements.txt` | Dependency list copied from the earlier supplementary delivery. |
| `README_FA.txt` | Persian execution and scope notes from the named-experiment package. |
| `README.md` | English experiment guide added for the `Supplementary_Experiments` folder. |
| `SHA256SUMS.txt` | Integrity hashes from the named-experiment package; the subsequently added `README.md` is not covered by that original hash list. |

The separated launchers reuse the original numerical engine and parameter settings, but routing and output-path logic were intentionally changed. Accordingly, a blanket statement that no executable logic changed would be inaccurate. Syntax and eight experiment-routing checks passed; full numerical equality of freshly rerun separated outputs with the original combined results remains untested.

If `Complete_Supplementary_Results.csv` is placed in this folder, its source is the original run's `portfolio_summary.csv`, `asset_metrics.csv` and `selected_lags.csv`. It was assembled by `combine_supplementary_results.py` with record-type/scope/source labels and exact comparison of the transferred source fields. It does not contain every raw stream or any archived-table validation.
