# 07 — Bitcoin Regime Windows

Evaluates the equal-weight ensemble of the eight MBDP pipelines across nine Bitcoin windows, so that the 2021–2023 contraction can be read against other market regimes.

## Windows

| ID | Train | Test |
|---|---|---|
| `W1_crash_2021_2023` | 2020-01-01 → 2021-07-01 | 2021-07-01 → 2023-03-01 |
| `W2_recovery_2023_2024` | 2022-03-01 → 2023-03-01 | 2023-03-01 → 2024-09-01 |
| `W3_recent_2024_2026` | 2023-09-01 → 2024-09-01 | 2024-09-01 → 2026-05-01 |
| `W4_bull_2020_2021` | 2020-01-01 → 2020-09-01 | 2020-09-01 → 2021-07-01 |
| `W5_bull_2020_2021a` | 2020-01-01 → 2020-09-01 | 2020-09-01 → 2021-03-01 |
| `W6_range_2023_2024` | 2022-03-01 → 2023-03-01 | 2023-03-01 → 2024-03-01 |
| `W7_range_2024_2025` | 2023-03-01 → 2024-03-01 | 2024-03-01 → 2025-03-01 |
| `W8_crash_2022` | 2021-07-01 → 2022-03-01 | 2022-03-01 → 2023-03-01 |
| `W9_late_2025` | 2024-03-01 → 2025-03-01 | 2025-03-01 → 2026-05-01 |

Some windows overlap by construction; they are descriptive regime slices, not independent samples.

## Usage

```bash
python run_bitcoin_regime_windows.py            # all nine windows
python run_bitcoin_regime_windows.py --only W1_crash_2021_2023
```

Before running, set `MBDP_DATA_ROOT` to the folder containing the input data and the recorded run-argument manifest, and `MBDP_OUTPUT_DIR` to the output folder. Both default to the current directory and `btc_windows`.

## Outputs

Per window: `run_arguments.json`, `metrics_summary.csv`, `rc_test_returns.csv` (the eight pipeline return series), `reality_check.json`, the bootstrap maxima and the selected lags. The ensemble series is the row mean of `rc_test_returns.csv`.
