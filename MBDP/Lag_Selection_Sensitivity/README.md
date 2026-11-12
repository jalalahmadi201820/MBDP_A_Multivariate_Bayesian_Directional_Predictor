# 04 — Lag-Selection Sensitivity (162 Scenarios)

A grid over the lag-selection procedure, to show how much of the reported performance depends on the selector rather than on the model.

## Grid dimensions

The scenarios cross:

| Dimension | Values |
|---|---|
| Selection estimator | `raw_pnl`, `raw_sharpe`, `wilcoxon`, `ttest` |
| Bootstrap block length | 5, 10, 20 bars |
| *p*-value treatment | raw, Holm |
| Penalty mode | absolute, scaled |
| Penalty weight | 0, 10, 50, 100, 200 |

Cells that are not defined for a given estimator are skipped, which is why the resulting grid has 162 scenarios rather than the full Cartesian product. Each cell re-selects lags on the training partition and re-runs the trading rule; the test block is untouched. The grid is run at zero commission so that the comparison isolates the selection step.

## Usage

```bash
python run_162_scenario_lag_grid.py \
    --data-path djia30_full_2009_2022.csv \
    --panels-file panels.json \
    --output-dir results/lag_grid
```

`--panels-file` is a JSON list of `panel, train_start, train_end, test_start, test_end` entries. Other options: `--date-col`, `--price-col`, `--ticker-col`, `--asset`, `--window-size`, `--predict-day`, `--dist {normal,student}`, `--student-df`, `--lambdas`, `--penalty-modes`, `--block-lengths`, `--bootstrap-reps`, `--p-adjustments`, `--alternative`, `--n-workers`, `--profiles`, and per-profile overrides. `--self-test` and `--smoke-test` give quick execution checks.

## Outputs

One row per scenario per panel with the selected lags, the training score components and the full test-block metric set. The prespecified default is reported next to the best cell in the manuscript, so the best cell carries no claim.
