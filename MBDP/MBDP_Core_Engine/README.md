# 01 — MBDP Core Engine

The shared implementation used by every experiment in this repository. Two modules:

| File | Role |
|---|---|
| `mbdp_rolling_bayesian_classifier.py` | Rolling class-conditional classifier, lag selectors, reference models, White Reality Check, and a command-line runner. |
| `mbdp_unified_trading_engine.py` | Unified three-profile trading engine used by the lag-selection grid. |

## What the engine does

For each asset and each rolling window it:

1. builds a feature vector from a small set of lag ratios `P_t / P_{t-m}` and, optionally, a bias-corrected Max-Lag Skewness descriptor over `[t - max(L), t]`;
2. fits class-conditional multivariate Gaussian or Student-*t* densities with covariance regularization `lambda I` and Laplace-smoothed class priors;
3. scores the current observation with Bayes' rule and compares it with the decision threshold (`tau = 0.9` by default) to emit BUY / SELL / HOLD, where HOLD keeps the current position;
4. charges commission per side on position changes.

Candidate lag sets are ranked on the training partition with a selection score that combines training P&L and a paired-test *p*-value. The test block is never used for selection.

## Selectors and variants

- Selectors: `raw_pnl`, `raw_sharpe`, `wilcoxon`, `ttest`.
- Variants: `regular` (lag ratios only) and `maxlag_skew` (lag ratios plus the skewness descriptor).
- Reference models available in the same loop: `MBDP`, `LDA`, `LogisticRegression`, `BuyHold`, `TSMomentum`.

## Command line

```bash
python mbdp_rolling_bayesian_classifier.py --self-test          # numerical checks only
python mbdp_rolling_bayesian_classifier.py \
    --profiles DRL_Ensemble FineRL XGboost_GRU \
    --stock-path data/djia30_full_2009_2022.csv \
    --btc-path data/BTCUSD.pkl \
    --output-dir results
```

Useful options: `--models`, `--n-lag-sets` (default 100), `--workers`, `--rc-repetitions` (default 5000), `--rc-block-length`, `--rc-seed` (default 20250303), `--price-col`, `--dist {normal,student}`, plus per-profile overrides for window size, lag range, horizon, regularization and threshold.

## Outputs

Per profile, the runner writes `run_arguments.json`, a per-configuration summary (`method`, `variant`, `lags`, equity endpoints, return, annualized return and volatility, Sharpe, Calmar, maximum drawdown, win rate, trade count), the selected lags, and the Reality Check bootstrap output.
