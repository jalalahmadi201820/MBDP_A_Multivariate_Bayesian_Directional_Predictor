# 05 — Covariance and Scaling Robustness

An eight-setting grid that separates three modelling choices which are often confounded.

## Grid dimensions

| Dimension | Settings |
|---|---|
| Feature scaling | raw lag ratios, within-window z-scores |
| Covariance estimator | empirical, Ledoit–Wolf shrinkage |
| Student-*t* scale | legacy (sample covariance used directly), covariance-matched (`nu/(nu-2)` correction) |

`2 x 2 x 2 = 8` settings per profile. Each cell reports return, Sharpe, maximum drawdown and the median, 95th-percentile and maximum condition number of the class covariances.

## Usage

```bash
python run_covariance_grid.py \
    --djia-path data/djia30_full_2009_2022.csv \
    --btc-path data/BTCUSD.pkl \
    --output-dir results/covariance
```

Options: `--profiles`, `--tickers`, `--djia-price-column`, `--btc-price-column`, `--invalid-price-policy`, `--selection-method`, `--lag-design {fixed,reselect,both}`, `--n-lag-sets`, `--seed`.

`--lag-design` controls whether the lag set is frozen from the training partition (`fixed`), re-selected inside each window (`reselect`), or both.

## Outputs

A per-setting CSV with the metric set, the condition-number diagnostics and the run manifest. The prespecified `lambda = 1e-6` configuration remains the default; the other cells document how far the results move when the conditioning changes.
