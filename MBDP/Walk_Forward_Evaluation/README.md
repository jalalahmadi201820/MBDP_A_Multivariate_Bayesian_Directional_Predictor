# 06 — Rolling-Origin Walk-Forward Evaluation

Chronological out-of-sample evaluation with repeated origins. Each split trains on a trailing block, freezes the lag set and the weights inside the test block, and rolls forward.

| File | Role |
|---|---|
| `prepare_walkforward_dataset.py` | Builds the clean Dow Jones panel used by the walk-forward runs. |
| `run_walkforward_drl_ensemble.py` | DRL-Ensemble profile. |
| `run_walkforward_finerl.py` | FineRL profile. |
| `run_walkforward_bitcoin.py` | Bitcoin profile (H4 bars). |

## Dataset preparation

```bash
python prepare_walkforward_dataset.py
```

Reads `djia30_full_2009_2022.csv` from the repository `data/` folder (or from the current directory), drops the two tickers with incomplete coverage, validates the panel (no missing dates, no duplicate rows, no non-positive prices, 28 symbols expected) and writes `djia30_full_2009_2022_1.csv` to the current directory. The script refuses to overwrite an existing output file.

## Running a profile

```bash
python run_walkforward_drl_ensemble.py --data-path djia30_full_2009_2022_1.csv
python run_walkforward_finerl.py      --data-path djia30_full_2009_2022_1.csv
python run_walkforward_bitcoin.py     --data-path data/BTCUSD.pkl
```

Options: `--window-size`, `--train-bars`, `--test-bars`, `--selection-mode {rolling,expanding}`, `--include-partial`, `--n-lag-sets`, `--min-lags`, `--max-lags`, `--min-lag-value`, `--price-col`, `--output-dir`.

`--train-bars` counts scored bars only; the warm-up needed by the rolling estimator is added on top.

## Outputs

Per split: the selected lags, the frozen weights, the test-block metrics and `daily_oos.csv`, the concatenated out-of-sample return series. That series is the input to the deflated Sharpe ratio script in `Synchronized_Baselines_and_Reality_Check`.
