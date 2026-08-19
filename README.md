# MBDP: A Multivariate Bayesian Directional Predictor

Repository accompanying the paper:

> **MBDP: A Rolling Multivariate Bayesian Generative Framework for Market Direction Prediction**

This repository brings together the benchmark experiments and the proposed
MBDP framework used to study market-direction prediction, probabilistic
decision-making, and backtested trading performance under different market
regimes.

The central idea is to replace a single globally trained black-box predictor
with a transparent rolling probabilistic model. MBDP constructs multiscale
price-ratio features, optionally adds a causal Max-Lag Skewness descriptor,
estimates local class-conditional distributions, and converts Bayesian
posterior scores into directional predictions or trading actions.

The repository contains four experiment families:

1. `CNNPred` — convolutional benchmark for U.S. equity-index direction
   prediction.
2. `DANSMP` — attention and knowledge-graph benchmark for Chinese equities.
3. `SAMBA` — bidirectional Mamba and graph-convolution benchmark for U.S.
   market return prediction.
4. `MBDP` — the proposed rolling Bayesian generative predictor and its
   financial trading experiments.

The benchmark folders provide the experimental context for the paper. The
`MBDP` folder contains the main model implementation and the ablation studies
reported for DJIA and Bitcoin environments.

---

## Repository structure

```text
repository/
├── CNNPred/
│   ├── CNNPred benchmark code and/or experiment files
│   └── README or dataset-specific documentation
│
├── DANSMP/
│   ├── DANSMP benchmark code and/or experiment files
│   └── README or dataset-specific documentation
│
├── MBDP/
│   ├── DRL_Ensemble.py
│   ├── FineRL.py
│   ├── XGboost_GRU.py
│   ├── tests/
│   └── metrics_summary.csv files generated after execution
│
├── SAMBA/
│   ├── SAMBA benchmark code and/or experiment files
│   └── README or dataset-specific documentation
│
├── README.md
└── requirements.txt
```

The exact contents of the benchmark folders may vary depending on whether
the repository is being used for source-code reproduction, result storage, or
dataset-specific experiments. Their methodological roles are described
below.

---

## How the folders relate to the paper

The paper has two connected layers.

### Layer 1: Statistical enhancements to existing models

The first layer investigates whether statistical information can improve
existing neural or hybrid forecasting systems. This layer is represented by
the benchmark folders:

- `CNNPred`
- `DANSMP`
- `SAMBA`

The experiments examine statistical feature augmentation, hypothesis-driven
feature selection, and Bayesian-style output refinement in different model
families.

### Layer 2: The standalone MBDP model

The second layer distills the useful statistical ideas into a dedicated
rolling Bayesian generative predictor. This is the `MBDP` folder.

MBDP is not a deep neural-network training pipeline. It does not require
backpropagation, actor-critic training, replay buffers, or end-to-end
gradient optimization. Its parameters are estimated locally from historical
rolling windows.

---

# 1. `CNNPred`: U.S. Equity-Index Forecasting

## Purpose

`CNNPred` represents the U.S. equity-index directional-prediction benchmark
described in the paper.

The original CNNPred formulation converts a multivariate historical market
window into a two-dimensional or three-dimensional tensor. A convolutional
neural network then processes the tensor as an image-like representation.
This allows the model to learn interactions across:

- time;
- market variables;
- technical indicators;
- macroeconomic variables;
- global market indices;
- exchange rates;
- commodity and futures information.

## Dataset context

The paper describes five major U.S. equity indices:

- S&P 500;
- Dow Jones Industrial Average;
- NASDAQ;
- NYSE Composite;
- Russell 2000.

The reported period covers approximately January 2010 to November 2017.
Each trading day is represented by an 82-dimensional feature vector.

The task is binary directional classification:

```text
Will the index close higher or lower on the following trading day?
```

The chronological split described in the paper is:

```text
60% training
20% validation
20% testing
```

The temporal order is preserved to reduce look-ahead bias.

## Evaluation

The main metrics for this folder are:

- accuracy;
- macro-F1;
- AUC.

MBDP is also evaluated on the CNNPred directional task as a standalone
probabilistic model. In the paper, this provides a direct comparison between
the proposed rolling Bayesian framework and the CNN-based benchmark.

## Role in the repository

`CNNPred` is primarily a benchmark and directional-classification
environment. It is not the folder that contains the main MBDP trading
implementation.

---

# 2. `DANSMP`: Chinese Stock Market with Knowledge Graphs

## Purpose

`DANSMP` represents the Chinese stock-market benchmark with attention
mechanisms and market knowledge graphs.

The model addresses two related problems:

1. selecting or reweighting informative input features;
2. modeling long-range temporal dependencies.

Its architecture contains:

- input attention;
- temporal attention;
- encoder-decoder sequence modeling;
- bi-typed hybrid-relational market knowledge graphs.

The knowledge graph is intended to capture relationships among companies and
market entities that cannot be represented by an isolated price series.

## Dataset context

The paper discusses constituent stocks from:

- CSI100E, with 73 stocks;
- CSI300E, with 185 stocks.

The dataset contains 516 trading days from November 21, 2017 to
December 31, 2019. The chronological split is:

```text
Training:   2017-11-21 to 2019-08-05
Validation: 2019-08-06 to 2019-10-22
Testing:    2019-10-23 to 2019-12-31
```

The model produces continuous return forecasts that are later converted into
directional signals for evaluation.

## Statistical extensions studied in the paper

The paper compares the original DANSMP model with variants that add:

- statistical feature augmentation;
- t-test-based feature selection;
- Bayesian output refinement.

The main metric is AUC because the task is evaluated as a directional
classification problem after mapping the return forecasts to classes.

## Role in the repository

`DANSMP` is a benchmark folder for studying whether statistically motivated
features and inference procedures improve a more complex attention-based
model. It is methodologically related to MBDP, but it is not implemented by
the rolling Bayesian functions in `MBDP`.

---

# 3. `SAMBA`: U.S. Market Forecasting

## Purpose

`SAMBA` represents the U.S. market forecasting benchmark based on a
bidirectional Mamba-style sequence model and an adaptive graph-convolution
component.

The model is intended to capture:

- long-range temporal dependencies;
- interactions among market variables;
- graph-like relationships among market entities;
- sequential structure in a high-dimensional market representation.

## Dataset context

The paper describes U.S. market data involving:

- NASDAQ;
- NYSE;
- DJIA.

The reported period is approximately January 2010 to November 2023. The
daily representation contains 82 market features.

Unlike the CNNPred and DANSMP tasks, SAMBA is evaluated as a return-prediction
problem rather than a direct binary classification problem.

## Evaluation

The main metrics are:

- RMSE;
- Information Coefficient (IC).

The paper retains SAMBA values as contextual benchmark results. Therefore,
the SAMBA rows should not automatically be interpreted as a full
reimplementation inside the MBDP pipeline.

## Role in the repository

`SAMBA` provides a modern sequence-model benchmark for comparison with the
statistical and probabilistic approach. It is included to show how the
proposed statistical ideas relate to a model with substantially greater
architectural complexity.

---

# 4. `MBDP`: Rolling Bayesian Financial Experiments

## Purpose

`MBDP` contains the proposed rolling Bayesian generative framework and the
financial trading experiments.

The current implementation contains three main scripts:

| File | Environment | Main role |
|---|---|---|
| `FineRL.py` | DJIA / FinRL-style post-pandemic environment | Multi-asset Bayesian trading evaluation |
| `DRL_Ensemble.py` | DJIA stability and crash environment | Multi-asset Bayesian trading evaluation |
| `XGboost_GRU.py` | BTC/USD crypto-winter environment | Bitcoin Bayesian trading evaluation |

The filename `XGboost_GRU.py` is retained for compatibility with the project
history. The current code implements the rolling Bayesian strategy and does
not directly train an XGBoost model.

---

## MBDP pipeline

The three MBDP scripts follow the same conceptual pipeline.

```text
Raw price data
      │
      ▼
Chronological cleaning and date split
      │
      ▼
Lag-ratio feature construction
      │
      ├── regular feature set
      │
      └── maxlag_skew feature set
      │
      ▼
Rolling class-conditional estimation
      │
      ▼
Gaussian or Student-t likelihoods
      │
      ▼
Bayesian posterior probabilities
      │
      ▼
BUY / SELL / HOLD signals
      │
      ▼
Position-level PnL and equity
      │
      ▼
Training-only lag selection
      │
      ▼
Test-period portfolio metrics
      │
      ▼
metrics_summary.csv
```

---

## MBDP feature construction

For a price series \(P_t\) and a lag set
\(\mathcal{L}=\{m_1,m_2,\ldots,m_M\}\), the base feature vector is:

\[
\mathbf{x}^{ratio}_t =
\left[
\frac{P_t}{P_{t-m_1}},
\frac{P_t}{P_{t-m_2}},
\ldots,
\frac{P_t}{P_{t-m_M}}
\right]^T.
\]

These ratios provide a scale-aware representation of recent price movement.
They compare the current price with historical prices instead of directly
using the absolute price level.

When `include_maxlag_skew=True`, the implementation appends the
bias-corrected sample skewness of the causal price window:

\[
z^{skew}_t =
\operatorname{skew}
\left(P_{t-\max(\mathcal{L})},\ldots,P_t\right).
\]

This feature describes the local shape and asymmetry of the recent price
trajectory. It is not calculated from future data and is not an unconditional
skewness estimate of the entire return series.

The binary target is:

\[
y_t=\mathbb{1}(P_{t+k}>P_t),
\]

where `k` is the `predict_day` parameter.

Samples are retained only when the required current, historical, and future
prices are finite and strictly positive.

---

## Rolling class-conditional estimation

At prediction index \(j\), the model estimates its parameters from the most
recent historical samples. The current target interval is excluded from the
estimation window.

For each class:

- upward movement;
- downward movement;

the model estimates:

- the class mean vector;
- the covariance matrix;
- the local class prior.

If one class is absent from a window, the complete valid rolling window is
used as a numerical fallback for that class's mean and covariance.

The covariance matrix is regularized as:

\[
\widehat{\Sigma}_{\omega}
=
\widehat{\operatorname{Cov}}_{\omega}
+\lambda I.
\]

This improves numerical stability when:

- the rolling window is small;
- features are highly correlated;
- one class is temporarily underrepresented;
- the covariance matrix is close to singular.

The class priors use Laplace smoothing:

\[
\widehat{P}(\omega)
=
\frac{n_{\omega}+1}{n_{up}+n_{down}+2}.
\]

---

## Bayesian posterior scores

The model supports two likelihood specifications:

1. multivariate Gaussian;
2. multivariate Student-t.

The Student-t option is useful for heavy-tailed local feature distributions.
The likelihood calculation is performed in log space to reduce numerical
underflow.

For the upward and downward classes:

\[
\ell_{\omega}
=
\log \widehat{P}(\omega)
+\log p(\mathbf{x}_j\mid\omega).
\]

The posterior upward probability is obtained with a stabilized log-sum-exp
calculation. The downward probability is the complementary posterior score.

These are model-based posterior scores under the fitted local distributional
assumptions. They should not automatically be interpreted as externally
calibrated probabilities or as a formal posterior over all model parameters.

---

## Trading layer

The posterior scores are converted into three actions:

```text
BUY   if P(up | x)   > threshold
SELL  if P(down | x) > threshold
HOLD  otherwise
```

Positions are represented by:

```text
 1  long
 0  flat
-1  short
```

`HOLD` preserves the previous position. The bar return uses the position
held at the beginning of the interval. Commission is charged when the
position changes:

- one commission for entering or leaving a directional position;
- two commissions for a direct long-to-short or short-to-long reversal.

The equity curve is generated recursively from the net bar returns.

---

## Portfolio construction

For multi-asset experiments, each asset first produces a training-period
return.

The allocation base is:

\[
b_a=\max(R^{train}_a,0).
\]

If at least one asset has a positive allocation base, weights are proportional
to \(b_a\). If all allocation bases are non-positive, equal weights are used.

The resulting allocated asset equity curves are aggregated into one test
portfolio equity curve.

This means the test allocation is determined from the training period rather
than from future test performance.

---

# Market environments in `MBDP`

## `FineRL.py`: post-pandemic DJIA environment

This script evaluates the Bayesian strategy on multiple DJIA constituents in
a FinRL-style environment.

The paper describes this environment as:

- asset class: U.S. equities;
- universe: DJIA 30 constituents;
- features: price, volume, and technical indicators;
- market regime: post-pandemic recovery and bullish market;
- paper test period: July 1, 2020 to March 31, 2022;
- actions: BUY, SELL, HOLD.

The script uses multi-asset portfolio aggregation and compares the selected
lag rules under regular and Max-Lag Skew feature representations.

## `DRL_Ensemble.py`: stability and crash environment

This script evaluates the same general Bayesian pipeline on the DJIA
stability/crash environment associated with the DRL-Ensemble benchmark.

The paper describes this environment as:

- asset class: U.S. equities;
- universe: DJIA 30 constituents;
- features: OHLC data, volume, and technical indicators;
- market regime: stable growth followed by the COVID-19 market disruption;
- paper test period: January 2016 to May 2020;
- actions: BUY, SELL, HOLD.

This environment is useful for studying whether a lag-selection rule that
works in a persistent bullish market remains effective when the market regime
changes.

## `XGboost_GRU.py`: Bitcoin crypto-winter environment

This script evaluates the Bayesian strategy on BTC/USD data in a volatile
crypto-market regime.

The paper describes this environment as:

- asset class: cryptocurrency;
- instrument: BTC/USD;
- features: price action and technical indicators;
- market regime: prolonged crypto-winter decline;
- paper test period: July 1, 2021 to March 1, 2023;
- actions: BUY, SELL, HOLD.

This environment is particularly important for the Max-Lag Skew ablation
because local price-path asymmetry can be much more pronounced in volatile
and non-stationary cryptocurrency data.

---

# Lag-selection methods

The current implementation contains four lag-selection methods:

```text
raw_pnl
raw_sharpe
wilcoxon
ttest
```

`raw_pnl` selects the candidate with the highest training PnL.

`raw_sharpe` selects the candidate with the highest training Sharpe ratio.

`wilcoxon` ranks candidates using the candidate training PnL combined with a
Wilcoxon-based probability score.

`ttest` ranks candidates using the candidate training PnL combined with a
paired t-test probability score.

The statistical ranking score has the form:

\[
S_q(\mathcal{L})
=
\operatorname{PnL}_{train}(\mathcal{L})
-100p_q(\mathcal{L}).
\]

The p-value is used as a continuous ranking component. It is not treated as
proof that a lag set is globally optimal.

## Removed tests

Mann-Whitney and Kolmogorov-Smirnov tests were explored during development,
but they are not part of the final comparison. In the current code they have
been removed from:

- imports;
- lag-selection calculations;
- method lists;
- worker execution;
- output tables;
- documentation of supported methods.

---

# Regular and Max-Lag Skew variants

The two feature variants are:

| Variant | Feature set |
|---|---|
| `regular` | Lag-ratio features only |
| `maxlag_skew` | Lag-ratio features plus causal Max-Lag Skewness |

The paper's reported ablation table retains `regular` as the control for
`raw_pnl` and `raw_sharpe`, while reporting the Max-Lag Skew comparison for
the Wilcoxon and t-test selectors.

The current repository version was modified to produce both variants for all
four supported methods because the requested CSV output requires the equity
and metrics of every method with and without the skewness feature. This is an
intentional execution/output change and should be distinguished from the
paper's original reported ablation table.

---

# Output files

Every main execution path is silent and writes one consolidated file:

```text
metrics_summary.csv
```

The summary contains rows for available method/variant combinations and
includes:

```text
method
variant
lags
equity_start
equity_end
return_pct
annual_return_pct
annual_volatility_pct
sharpe_ratio
calmar_ratio
max_drawdown_pct
win_rate
trades
```

Example:

```text
method,variant,lags,equity_start,equity_end,return_pct,annual_return_pct,annual_volatility_pct,sharpe_ratio,calmar_ratio,max_drawdown_pct,win_rate,trades
wilcoxon,maxlag_skew,"(11, 28)",1000000,1706809,70.68,5.49,25.71,0.33,0.11,-51.37,0.42,117
```

The scripts do not print:

- progress messages;
- worker messages;
- errors from candidate lag evaluation;
- final summaries;
- chart paths;
- logging output.

They also do not generate charts, per-method CSV files, selected-lag CSV
files, or normalized-equity CSV files in the current main execution paths.

---

# Input data

## `FineRL.py` and `DRL_Ensemble.py`

These scripts expect a CSV with at least the following columns:

```text
Ticker
Date
Adj Close
```

The data are grouped by ticker and sorted chronologically before processing.
The default input and output paths are defined inside each script's
`main()` function and should be updated for a different machine or dataset.

## `XGboost_GRU.py`

This script accepts a pandas DataFrame stored in a pickle file.

Accepted price-column candidates include:

```text
Close
close
Adj Close
adj close
Price
price
Last
last
```

Accepted date-column candidates include:

```text
Date
date
Datetime
datetime
Time
time
```

A pandas `DatetimeIndex` may also be used.

---

# Configuration

The paper's main MBDP configuration uses:

```text
100 candidate lag sets
two distinct lags
lag range [1, 30]
random seed 42
rolling window 250
one-period-ahead prediction
Student-t degrees of freedom 10
covariance regularization 1e-6
probability threshold 0.9
zero commission
```

The exact script defaults are not identical in every file. For example,
`XGboost_GRU.py` exposes its settings through command-line arguments, while
the two DJIA scripts define their benchmark-specific settings inside
`main()`. Always check the configuration in the script when reproducing a
specific table or market environment.

---

# Running the MBDP experiments

Install the scientific Python dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available in a local checkout, the core
dependencies are:

```bash
pip install numpy pandas scipy
```

Run the DJIA experiments after updating their data paths:

```bash
cd MBDP
python FineRL.py
python DRL_Ensemble.py
```

Run the Bitcoin experiment:

```bash
cd MBDP
python XGboost_GRU.py \
  --data-path BTCUSD_H4.pkl \
  --output-dir bayesian_lag_search_results_mp
```

Useful `XGboost_GRU.py` options include:

```text
--train-start
--train-end
--test-start
--test-end
--lags
--window-size
--predict-day
--dist
--student-df
--reg
--prob-threshold
--commission
--initial-capital
--annualization
--n-lag-sets
--min-lags
--max-lags
--min-lag-value
--max-lag-value
--random-seed
--n-workers
```

---

# Metrics and interpretation

For directional-classification folders such as `CNNPred` and `DANSMP`, the
paper uses:

- accuracy;
- macro-F1;
- AUC.

For the regression-oriented `SAMBA` benchmark, the paper uses:

- RMSE;
- Information Coefficient.

For discrete trading and portfolio experiments in `MBDP`, the repository
reports:

- cumulative or total return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- Calmar ratio;
- maximum drawdown;
- win rate;
- number of trades;
- start and end equity.

These metrics are backtest statistics. They do not guarantee live-trading
performance, future profitability, probability calibration, or protection
from drawdowns.

---

# Reproducibility notes

- All lag candidates are selected from the training partition only.
- The test partition is not used during lag ranking.
- Dates are processed chronologically.
- The rolling window excludes the current future target interval.
- Max-Lag Skewness is computed from the price window ending at the current
  observation.
- The random lag generator is controlled by a seed.
- Results depend on the selected market regime, dates, data frequency,
  threshold, commission, annualization factor, and lag-selection rule.
- A strong result for one method or variant should not be interpreted as
  universal superiority across all market environments.

---

# Code-to-paper map

| Paper topic | Main functions |
|---|---|
| Mathematical utilities | `safe_log`, `regularized_covariance` |
| Gaussian likelihood | `logpdf_multivariate_normal` |
| Student-t likelihood | `logpdf_multivariate_student_t` |
| Sample and target construction | `build_samples` |
| Local parameter estimation | `fit_params` |
| Posterior calculation | `predict_proba` |
| Rolling strategy execution | `rolling_trade` |
| Train/test chronology | `split_df` |
| Candidate generation | `generate_random_lag_sets` |
| Lag selection | `run_multi_test_lag_search`, `select_best_lags` |
| Asset-level equity | `run_one_ticker_with_lag`, `run_one_lag_configuration` |
| Portfolio aggregation | `build_portfolio_for_method` |
| Risk/performance metrics | `compute_metrics`, `compute_portfolio_metrics` |
| Silent CSV export | `write_metrics_summary` |

---

# Verification

The `MBDP` folder includes a lightweight output-contract test at
`tests/test_output_contract.py`. It checks that:

- only `raw_pnl`, `raw_sharpe`, `wilcoxon`, and `ttest` remain;
- Mann-Whitney and KS code is absent;
- scripts contain no `print()` calls;
- each script writes only `metrics_summary.csv`;
- the CSV contains the required equity and performance columns.

The MBDP scripts also contain docstrings for their functions so that the
implementation can be read alongside the methodology in the paper.

---

# Citation

If you use this repository or the MBDP implementation, cite:

```text
MBDP: A Rolling Multivariate Bayesian Generative Framework for Market Direction Prediction
```

The benchmark descriptions and the MBDP methodology in this README are based
on the accompanying paper, especially the methodology sections on feature
construction, rolling Bayesian inference, training-only lag selection, and
the CNNPred, DANSMP, SAMBA, DJIA, and Bitcoin experimental environments.
