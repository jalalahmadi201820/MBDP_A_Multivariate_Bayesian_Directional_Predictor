# Supplementary Experiments

Separately runnable exploratory analyses prepared during manuscript revision. These experiments supplement, but do not replace or independently validate, the archived manuscript results.

## Directory layout

Place the contents of `Experiments_By_Name.zip` directly in this folder, alongside this README (do not add another nested experiment folder). The parent directory contains the main `README.md`, `SOURCE_MAP.md` and folders `01`–`08`.

```text
../
├── README.md
├── SOURCE_MAP.md
├── 01_MBDP_Core_Engine/ ... 08_Cost_Sensitivity/
└── Supplementary_Experiments/
    ├── README.md
    ├── README_FA.txt
    ├── requirements.txt
    ├── Code.zip
    ├── shared_engine.py
    ├── experiment_plan.py
    ├── original_combined_run.py
    ├── <experiment scripts listed below>
    ├── SHA256SUMS.txt
    └── Complete_Supplementary_Results.csv  # optional saved-results attachment
```

## Setup

Use Python 3.10+ in an isolated environment. From this folder:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows alternative: .venv\Scripts\activate
pip install -r requirements.txt
python -m zipfile -e Code.zip .
```

Extraction must create `Code/MBDP_A_Multivariate_Bayesian_Directional_Predictor-main/` directly under this folder. The runner uses the bundled `Data/djia30_full_2009_2022.csv` and `Data/BTCUSD.pkl` within that repository, rather than the parent project's `data/` directory. Keep `shared_engine.py` and `experiment_plan.py` beside the launchers. These scripts do not require the renamed parent-folder engine.

## Run an experiment

```bash
python ridge_lambda_sensitivity.py
```

Replace the filename with any launcher below. No command-line parameters are needed. The settings are deliberately fixed to the original supplementary design rather than retuned on test performance.

| Script | Analysis |
|---|---|
| `ridge_lambda_sensitivity.py` | Ridge values 1e-8, 1e-7, 1e-6 (reference), 1e-5 and 1e-4. |
| `student_t_df_sensitivity.py` | Student-t degrees of freedom 3–8 and 10 (reference). |
| `btc_exhaustive_lag_pairs.py` | All 435 pairs from lags 1–30, compared with the sampled-pair reference. |
| `btc_sampled_lag_triples.py` | 100 sampled triples from lags 1–30, compared with the sampled-pair reference; not exhaustive triple search. |
| `posterior_threshold_sensitivity.py` | Probability thresholds 0.6, 0.7, 0.8, 0.9 (reference) and 0.95. |
| `transaction_cost_sensitivity.py` | Commission 0, 5, 10 (reference) and 20 basis points per side. |
| `btc_raw_pnl_failure_analysis.py` | Raw-PnL lag selection in BTC, at zero commission and the 10-basis-point reference; preserves the unfavorable diagnostic case. |
| `btc_lag_selector_comparison.py` | Raw-PnL, raw-Sharpe and Wilcoxon lag selectors in BTC, using reference model settings. |
| `btc_diagnostic_plots.py` | Diagnostic time series and posterior distributions from the failure-analysis streams; run after `btc_raw_pnl_failure_analysis.py`. |

All eight experiments include Regular and Max-Lag Skewness variants. Scalar sensitivity experiments cover the `XGboost_GRU`, `DRL_Ensemble` and `FineRL` profiles; scripts prefixed `btc_` are BTC-only. These profile names identify replay protocols, not fresh training of the named XGBoost/GRU or reinforcement-learning models. Lag selection uses training data, with its final row purged to avoid a forward label crossing the split boundary. Scalar sensitivity sweeps hold selected lags fixed. Random seed: 42.

The engine uses two worker processes and limits numerical-library threads to one. Runs can be computationally expensive. Launchers call the shared engine; they are not standalone copies of its numerical routines.

## Outputs

Each experiment writes to `outputs/<experiment_name>/`, independently of the other experiments:

- `portfolio_summary.csv`: equal-weight portfolio metrics across synchronized valid assets.
- `asset_metrics.csv`: asset-level metrics and configuration labels.
- `selected_lags.csv`: selected lag tuples and candidate counts.
- `data_audit.csv`: input audit records for that experiment.
- `manifest.json`: experiment identity, source hashes, seed, protocol notes and NumPy/pandas versions.
- Per-asset metrics/lag CSVs and BTC test-stream `.csv.gz` files.

The plotting script writes PDF and PNG figures to `outputs/btc_diagnostic_plots/`. Rerunning an experiment overwrites matching outputs; use a clean output directory if changing code/settings. Full per-bar equity streams are not exported by this runner.

`original_combined_run.py` preserves the earlier combined runner. It is not required for individual runs; if invoked, it runs the original full supplementary grid and writes to `revision_results/` instead.

## Saved combined results (optional attachment)

If supplied, `Complete_Supplementary_Results.csv` combines saved outputs from the original combined run:

| `record_type` | Rows |
|---|---:|
| `portfolio_summary` | 118 |
| `asset_metrics` | 2,062 |
| `selected_lags` | 122 |
| **Total** | **2,302** |

`source_file` identifies the originating summary and `result_scope` marks the exploratory scope. Blank cells can indicate a field not applicable to that record type or unavailable in the source. Original source-field strings were preserved and compared after assembly. This CSV excludes raw per-bar streams, data-audit records and archived numerical tables. Reference configurations recur in separate experiments, so concatenating new experiment outputs will not reproduce these original row counts without deduplication.

## Verification status and limitations

- Syntax, eight experiment-routing checks and ZIP integrity passed when the named-experiment package was prepared. The full separated suite has not been rerun; numerical equivalence to the original combined run has not been demonstrated by a new full execution.
- Saved-output/source consistency audits do not independently validate archived tables, walk-forward dates, equity per-bar returns or fresh model fits.
- Full-file equity filtering leaves survivorship/selection bias. Training-only lag selection does not eliminate that limitation or the ex-post selection of analyses.
- Supplementary BTC annualization uses 2,190 four-hour bars/year. Protocol/annualization differences matter when comparing against archived results.
- The Student-t calculation retains the original raw empirical scale. The commission sweep does not implement a complete spread, slippage or funding model.
- These scripts do not perform new Reality Check/DSR tests or justify selecting a new winning model using test results.
- `SHA256SUMS.txt` belongs to the original named-experiment ZIP. This subsequently added README and any separately added results CSV are not covered by that hash list.

See the parent `SOURCE_MAP.md` for code provenance and the parent `README.md` for archived-workflow documentation.
