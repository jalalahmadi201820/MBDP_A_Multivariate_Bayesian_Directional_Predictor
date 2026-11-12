# MBDP — A Rolling Multivariate Bayesian Generative Framework for Market Direction Prediction

Code and data for the manuscript *"MBDP: A Rolling Multivariate Bayesian Generative Framework for Market Direction Prediction"*.

MBDP is a rolling, causal, generative framework for next-bar market-direction prediction. Within each chronological window it estimates **class-conditional multivariate Gaussian or Student-*t* densities** with regularized class covariances and Laplace-smoothed local priors, selects a small set of **multiscale price-ratio lags** using training data only, and maps the resulting posterior directional score to a BUY/SELL/HOLD decision. There is no end-to-end gradient training: the model is transparent, cheap to fit, and easy to inspect.

The repository has two parts:

| Folder | Content |
|---|---|
| [`MBDP/`](MBDP/) | The proposed framework and every experiment reported in the paper: core engine, synchronized baselines, Reality Check / deflated Sharpe diagnostics, lag-selection and covariance sensitivity, walk-forward evaluation, Bitcoin regime windows and cost sensitivity. |
| [`Appendix_Tests/`](Appendix_Tests/) | The preliminary/appendix studies that motivate the design. Three third-party architectures — [DANSMP](Appendix_Tests/DANSMP/), [2D-CNNpred](Appendix_Tests/CNNPred/) and [SAMBA](Appendix_Tests/SAMBA/) — each re-implemented with four statistical extensions. |

## Repository layout

```
.
├── MBDP/
│   ├── 01_MBDP_Core_Engine/                  shared engine + CLI
│   ├── 02_Synchronized_Baselines_and_Reality_Check/
│   ├── 03_Baseline_Comparative_Backtests/
│   ├── 04_Lag_Selection_Sensitivity/
│   ├── 05_Covariance_Robustness/
│   ├── 06_Walk_Forward_Evaluation/
│   ├── 07_Bitcoin_Regime_Windows/
│   ├── 08_Cost_Sensitivity/
│   ├── README.md                             detailed guide
│   ├── SOURCE_MAP.md                         file → original script map
│   └── requirements.txt
├── Appendix_Tests/
│   ├── DANSMP/                               graph-based CSI100/CSI300 model
│   ├── CNNPred/                              2D-CNN model on DJI/NASDAQ/NYSE/Russell/S&P
│   └── SAMBA/                                Graph-Mamba model on DJI/IXIC/NYSE
├── data/
│   ├── djia30_full_2009_2022.csv             Dow Jones daily panel
│   └── BTCUSD.pkl                            Bitcoin/USD H4 bars
├── requirements.txt
└── README.md
```

## Method in one paragraph

For each asset and each rolling window, MBDP builds a feature vector from a small set of training-selected lag ratios `P_t / P_{t-m}` plus an optional **Max-Lag Skewness** descriptor (bias-corrected sample skewness of the causal price window `[t - max(L), t]`). It fits class-conditional Gaussian or multivariate Student-*t* densities for the up/down labels, regularizes each covariance with `lambda I`, smooths the class priors, and scores the current observation with Bayes' rule. The score is compared with a fixed threshold `tau = 0.9` to produce a position; HOLD keeps the existing position. Candidate lag sets are ranked on the training partition by a selection score that combines training P&L with a paired-test *p*-value, and the test block is never touched during selection.

## Data

The two datasets used in the paper are included in the [`data/`](data/) folder:

| File | Content | Size |
|---|---|---|
| [`data/djia30_full_2009_2022.csv`](data/djia30_full_2009_2022.csv) | Daily OHLCV panel of the 30 Dow Jones constituents, 2009-04-01 to 2022-03-31:  3,274 trading days. | 10.8 MB |
| [`data/BTCUSD.pkl`](data/BTCUSD.pkl) | Bitcoin/USD  2020-01-01 to 2026-05-31: 2,342 days (six bars per day). | 0.7 MB |

The CSV has the columns `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Adj Close` and `Volume`. The pickle is a `pandas.DataFrame` indexed by `datetime` with the columns `open`, `high`, `low`, `close` and `volume`.

The appendix studies ship their own processed datasets inside `Appendix_Tests/*/Dataset` and `Appendix_Tests/DANSMP/data`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` at the repository root covers the whole project: the numerical stack used by `MBDP/` (`numpy`, `pandas`, `scipy`, `scikit-learn`, `numba`, `threadpoolctl`) and the deep-learning stack needed only by the appendix studies (`torch`, `torchvision`, `einops`, `h5py`, `configparser`, `matplotlib`, `tqdm`). If you only intend to run the MBDP experiments, `pip install -r MBDP/requirements.txt` installs the smaller CPU-only subset.

Python 3.10+ is recommended. All reported runs were executed on CPU; the engine parallelizes across processes, not threads.

## Quick start

```bash
# 1. numerical self-test of the core engine (no data needed)
python MBDP/01_MBDP_Core_Engine/mbdp_rolling_bayesian_classifier.py --self-test

# 2. full synchronized suite on all three profiles
python MBDP/02_Synchronized_Baselines_and_Reality_Check/run_synchronized_baseline_suite.py \
    --stock-path data/djia30_full_2009_2022.csv \
    --btc-path data/BTCUSD.pkl \
    --output-dir results/synchronized

# 3. cost sensitivity at 0 / 5 / 10 / 20 basis points per side
python MBDP/08_Cost_Sensitivity/compute_cost_sensitivity.py
```

## What is where in the paper

| Paper element | Code |
|---|---|
| Comparative backtested performance and MBDP ablations | `MBDP/03_Baseline_Comparative_Backtests/` |
| Synchronized five-model baselines | `MBDP/02_Synchronized_Baselines_and_Reality_Check/run_synchronized_baseline_suite.py` |
| White Reality Check and approximate deflated Sharpe ratio | `MBDP/02_Synchronized_Baselines_and_Reality_Check/` |
| Lag-selection sensitivity (162 scenarios) | `MBDP/04_Lag_Selection_Sensitivity/` |
| Covariance / scaling robustness (8 settings) | `MBDP/05_Covariance_Robustness/` |
| Rolling-origin walk-forward evaluation | `MBDP/06_Walk_Forward_Evaluation/` |
| Bitcoin regime windows | `MBDP/07_Bitcoin_Regime_Windows/` |
| Commission sweep | `MBDP/08_Cost_Sensitivity/` |
| Preliminary statistical modules on DANSMP / CNNpred / SAMBA | `Appendix_Tests/` |

## Reproducibility notes

- Every model in a comparison shares the same dates, features, horizon and execution rule; the synchronized suite exists precisely to remove endpoint and protocol mismatches.
- Lag selection, class-conditional fits and portfolio weights use the training partition only.
- All stochastic components are seeded; the Reality Check uses a fixed bootstrap seed (`20250303`) and 5,000 repetitions.
- The appendix studies select their statistical hyperparameters on the last 10% of the training data.

## Citation

If you use this code, please cite the manuscript:

```bibtex
@article{mbdp2026,
  title  = {MBDP: A Rolling Multivariate Bayesian Generative Framework for Market Direction Prediction},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

No license file is included in this repository. Please contact the corresponding author of the manuscript before redistributing or reusing the code beyond academic review.
