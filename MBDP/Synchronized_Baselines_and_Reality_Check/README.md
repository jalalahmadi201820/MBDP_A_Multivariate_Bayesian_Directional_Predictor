# 02 — Synchronized Baselines, Reality Check and Deflated Sharpe Ratio

Two diagnostics that answer the "are the comparisons fair?" and "is the result selection-biased?" questions.

| File | Role |
|---|---|
| `run_synchronized_baseline_suite.py` | Re-estimates five decision models on identical dates, features, horizon and trading rule, then runs the Reality Check families. |
| `compute_deflated_sharpe_ratio.py` | Approximate deflated Sharpe ratio from the walk-forward return series. |

## Synchronized suite

The five models are `MBDP`, `LDA`, `LogisticRegression`, `BuyHold` and `TSMomentum`. They share the same asset universe, calendar, lag inputs and one-step BUY/SELL/HOLD rule within each profile, so differences cannot be attributed to endpoint or protocol mismatches. All panels charge 10 basis points per side by default.

```bash
python run_synchronized_baseline_suite.py \
    --profiles DRL_Ensemble FineRL XGboost_GRU \
    --stock-path data/djia30_full_2009_2022.csv \
    --btc-path data/BTCUSD.pkl \
    --output-dir results/synchronized
```

Options: `--models`, `--n-lag-sets`, `--workers`, `--rc-repetitions`, `--rc-block-length`, `--rc-seed`, `--price-col`, `--invalid-price-policy`.

The Reality Check is reported for two families: the prespecified default configuration alone, and the full eight-pipeline family. Both use the same block-bootstrap machinery as the core engine.

## Deflated Sharpe ratio

```bash
python compute_deflated_sharpe_ratio.py --walkforward-root <folder with the walk-forward outputs>
```

The script reads `daily_oos.csv` from each walk-forward result folder, computes the observed Sharpe ratio, the expected maximum under `N = 800` trials, and the Bailey–López de Prado adjustment, then prints one line per profile. Options: `--walkforward-root` (folder holding the `fast_*_walk_forward` result folders; default `walkforward_results`), `--profile` (omit to run all three), `--trials` (default 800). The profile-to-folder mapping and the per-profile configuration used for the tail test are set at the top of the file.

## Outputs

Per profile: the synchronized metric table (return, annualized return, volatility, Sharpe, maximum drawdown, win rate, trades), the selected lags, and the Reality Check bootstrap maxima.
