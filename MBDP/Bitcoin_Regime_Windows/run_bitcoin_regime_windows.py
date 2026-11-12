import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_MBDP_Core_Engine"))
import mbdp_rolling_bayesian_classifier as bt  # noqa: E402

# Base directory holding the recorded run manifests and the input data.
HQ = Path(os.environ.get("MBDP_DATA_ROOT", "."))
# Output directory for the nine window runs.
OUT = Path(os.environ.get("MBDP_OUTPUT_DIR", "btc_windows"))
BASE = json.loads((HQ / "Soogiri" / "bias_results_fast" / "XGboost_GRU" / "run_arguments.json").read_text())

WINDOWS = {
    "W1_crash_2021_2023": ("2020-01-01", "2021-07-01", "2021-07-01", "2023-03-01"),
    "W2_recovery_2023_2024": ("2022-03-01", "2023-03-01", "2023-03-01", "2024-09-01"),
    "W3_recent_2024_2026": ("2023-09-01", "2024-09-01", "2024-09-01", "2026-05-01"),
    "W4_bull_2020_2021": ("2020-01-01", "2020-09-01", "2020-09-01", "2021-07-01"),
    "W5_bull_2020_2021a": ("2020-01-01", "2020-09-01", "2020-09-01", "2021-03-01"),
    "W6_range_2023_2024": ("2022-03-01", "2023-03-01", "2023-03-01", "2024-03-01"),
    "W7_range_2024_2025": ("2023-03-01", "2024-03-01", "2024-03-01", "2025-03-01"),
    "W8_crash_2022": ("2021-07-01", "2022-03-01", "2022-03-01", "2023-03-01"),
    "W9_late_2025": ("2024-03-01", "2025-03-01", "2025-03-01", "2026-05-01"),
}


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (tr_s, tr_e, te_s, te_e) in WINDOWS.items():
        if only and name != only:
            continue
        args = dict(BASE)
        args.update(
            output_dir=str(OUT / name),
            models=["MBDP"],
            n_workers=8,
            n_lag_sets=100,
            commission=0.001,
            train_start=tr_s, train_end=tr_e, test_start=te_s, test_end=te_e,
        )
        print(f"=== {name} train {tr_s}..{tr_e} test {te_s}..{te_e}", flush=True)
        bt.run_profile(argparse.Namespace(**args))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
