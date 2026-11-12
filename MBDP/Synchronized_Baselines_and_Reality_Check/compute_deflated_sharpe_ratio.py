import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


PROFILE_FOLDERS = {
    "DRL_Ensemble": "fast_drl_walk_forward",
    "FineRL": "fast_finerl_walk_forward",
    "XGboost_GRU": "fast_xgboost_walk_forward",
}

TAIL_CONFIG = {
    "DRL_Ensemble": ("wilcoxon", "maxlag_skew"),
    "FineRL": ("wilcoxon", "maxlag_skew"),
    "XGboost_GRU": ("raw_pnl", "regular"),
}

TRIALS = 800


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def read_daily_returns(path):
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["method"], row["variant"])].append(float(row["bar_return"]))
    return grouped


def deflated_sharpe(grouped, profile):
    stats = {}
    for key, values in grouped.items():
        n = len(values)
        mu = statistics.mean(values)
        sd = statistics.stdev(values)
        sr = mu / sd
        skew = sum(((x - mu) / sd) ** 3 for x in values) / n
        kurt = sum(((x - mu) / sd) ** 4 for x in values) / n
        stats[key] = (n, sr, skew, kurt)

    var_sr = statistics.variance([value[1] for value in stats.values()])
    gamma = 0.5772156649
    sr0 = math.sqrt(var_sr) * (
        (1.0 - gamma) * phi(1.0 - 1.0 / TRIALS)
        + gamma * phi(1.0 - 1.0 / (TRIALS * math.e))
    )

    n, sr, skew, kurt = stats[TAIL_CONFIG[profile]]
    denominator = math.sqrt(
        max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
    )
    dsr = phi((sr - sr0) * math.sqrt(n - 1.0) / denominator)
    return dsr, n, sr


def main():
    global TRIALS
    parser = argparse.ArgumentParser(
        description="Compute the approximate deflated Sharpe ratio from walk-forward daily returns."
    )
    parser.add_argument(
        "--walkforward-root",
        default="walkforward_results",
        help="Directory containing the fast_*_walk_forward result folders.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_FOLDERS),
        help="Profile to evaluate; omit to run all three.",
    )
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()

    TRIALS = args.trials
    root = Path(args.walkforward_root)
    profiles = [args.profile] if args.profile else list(PROFILE_FOLDERS)

    for profile in profiles:
        path = root / PROFILE_FOLDERS[profile] / "daily_oos.csv"
        grouped = read_daily_returns(path)
        dsr, n, sr = deflated_sharpe(grouped, profile)
        print(f"{profile}: DSR={dsr:.4f}  n={n}  SR={sr:.6f}")


if __name__ == "__main__":
    main()
