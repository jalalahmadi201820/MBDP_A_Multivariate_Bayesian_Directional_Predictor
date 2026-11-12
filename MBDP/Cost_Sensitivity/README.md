# 08 — Cost Sensitivity

Reprices the selected MBDP configuration of each profile at 0, 5, 10 and 20 basis points per side, holding everything else fixed.

## Usage

```bash
python compute_cost_sensitivity.py
```

Before running, set `MBDP_DATA_ROOT` to the folder containing the input data and the recorded run-argument manifest, and `MBDP_OUTPUT_DIR` to the output folder (defaults: current directory and `cost_sensitivity`). The configuration repriced for each profile is listed in the `CHOICES` dictionary (`selector`, `variant`).

## What is varied

Only the commission level. Positions, lags, weights and the decision threshold are taken from the recorded run, so the four rows differ solely in the per-side charge. Spread, slippage and funding are held at zero.

## Outputs

`cost_sensitivity.csv` with one row per profile and commission level: equity endpoints, cumulative and annualized return, annualized volatility, Sharpe, Calmar, maximum drawdown, win rate, trade count, and the buy/sell/hold counts. The same file feeds the commission figure in the manuscript.
