"""Table 16 commission-sensitivity replay for the three reported MBDP profiles.

Two protocols are combined in the reported table:

* Equity profiles (DRL-Ensemble, FineRL) keep the archived
  ``wilcoxon`` + ``maxlag_skew`` lag selections recorded in the run manifests.
* Bitcoin is re-selected with a fresh train-only Wilcoxon search over the
  original 100 seed-42 candidate lag pairs and annualized with
  ``A = 2190`` four-hour bars per year.

For every profile the positions are produced once and then repriced at
0, 5, 10 and 20 basis points per side, so the four rows differ only in the
per-side commission. Spread, slippage and funding stay at zero.

Run from the repository root::

    python MBDP/Cost_Sensitivity/compute_cost_sensitivity.py \
      --data-root /path/to/MBDP_Cache/All_Result \
      --output-dir results/cost_sensitivity \
      --workers 8
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# The shared engine lives next to this folder in the paper repository and beside
# this file in the standalone reproduction package.
for _engine_dir in (HERE.parent / "MBDP_Core_Engine", HERE, HERE.parent):
    if (_engine_dir / "mbdp_rolling_bayesian_classifier.py").is_file():
        sys.path.insert(0, str(_engine_dir))
        break

import mbdp_rolling_bayesian_classifier as bt  # noqa: E402

BPS = (0, 5, 10, 20)
BTC_PROFILE = "XGboost_GRU"
CHOICES = {
    "DRL_Ensemble": ("wilcoxon", "maxlag_skew"),
    "FineRL": ("wilcoxon", "maxlag_skew"),
    "XGboost_GRU": ("wilcoxon", "maxlag_skew"),
}
# Manuscript labels and figure-data file stems for the three profiles.
LABELS = {"DRL_Ensemble": "DRL-Ensemble", "FineRL": "FineRL", "XGboost_GRU": "BTC"}
FIGURE_STEMS = {"DRL_Ensemble": "4-cost_drl", "FineRL": "4-cost_finrl",
                "XGboost_GRU": "4-cost_btc"}

BTC_ANNUALIZATION = 2190

# Columns shared by the per-asset (BTC) and portfolio (equity) metric helpers.
# The BTC-only detail (split, bars, buy/sell/hold records) stays in the BTC file.
COMMON_METRICS = ("equity_start", "equity_end", "return_pct", "annual_return_pct",
                  "annual_volatility_pct", "sharpe_ratio", "calmar_ratio",
                  "max_drawdown_pct", "win_rate", "trades")
# Per-bar/per-signal counters returned by the BTC helper are not reported.
HIDDEN_METRICS = ("bars", "n_buy", "n_sell", "n_hold")

STOCK_DATA_NAMES = ("djia30_full_2009_2022.csv",)
BTC_DATA_NAMES = ("BTCUSD_H4.pkl", "BTCUSD.pkl")
ARCHIVE_SUBPATHS = (
    "Soogiri/bias_results_fast",
    "All_Result/Soogiri/bias_results_fast",
    "MBDP_Cache/All_Result/Soogiri/bias_results_fast",
    "bias_results_fast",
)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reprice the reported MBDP configurations at 0/5/10/20 bps."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("MBDP_DATA_ROOT"),
        help="Folder holding the archived run manifests and the input data. "
             "Defaults to MBDP_DATA_ROOT, then to the repository/working folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("MBDP_OUTPUT_DIR", "cost_sensitivity")),
        help="Directory for the commission-sensitivity table and diagnostics.",
    )
    parser.add_argument(
        "--stock-data",
        type=Path,
        default=os.environ.get("MBDP_STOCK_DATA"),
        help="Override the DJIA-30 CSV used by the two equity profiles.",
    )
    parser.add_argument(
        "--btc-data",
        type=Path,
        default=os.environ.get("MBDP_BTC_DATA"),
        help="Override the BTC four-hour price pickle.",
    )
    parser.add_argument(
        "--btc-lags",
        choices=("fresh", "archived"),
        default="fresh",
        help="'fresh' re-selects the BTC lag pair on the training split "
             "(train-only Wilcoxon, the reported row); 'archived' reuses the "
             "recorded archive selection and is kept for audits only.",
    )
    parser.add_argument("--profiles", nargs="*", default=list(CHOICES))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)

def search_roots(cli_root):
    roots = []
    for candidate in [cli_root, os.environ.get("MBDP_DATA_ROOT"), HERE, *HERE.parents,
                      Path.cwd()]:
        if candidate:
            path = Path(candidate).expanduser()
            if path not in roots:
                roots.append(path)
    return roots

def resolve_file(cli_path, env_name, names, roots):
    for candidate in (cli_path, os.environ.get(env_name)):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path.resolve()
            raise FileNotFoundError(f"Requested file does not exist: {path}")
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path.resolve()
        for extra in ("data", "Data", "All_Result", "MBDP_Cache/All_Result"):
            for name in names:
                path = root / extra / name
                if path.is_file():
                    return path.resolve()
    return None

def resolve_archive(cli_root, profiles, roots):
    for root in roots:
        for relative in ARCHIVE_SUBPATHS:
            archive = root / relative
            if all((archive / profile / "run_arguments.json").is_file()
                   for profile in profiles):
                return archive.resolve()
        if len(root.parts) <= 2:
            # Skip filesystem roots that are too shallow for a bounded glob.
            continue
        nested = list(root.glob("*/*/Soogiri/bias_results_fast"))
        for archive in nested:
            if all((archive / profile / "run_arguments.json").is_file()
                   for profile in profiles):
                return archive.resolve()
    return None

def build_args(profile, output_dir, archive, data_path):
    raw = json.loads(
        (archive / profile / "run_arguments.json").read_text(encoding="utf-8")
    )
    raw.update(
        data_path=str(data_path),
        output_dir=str(output_dir / "audit" / profile),
        commission=0.0,
        models=["MBDP"],
        annualization=BTC_ANNUALIZATION if profile == BTC_PROFILE else raw["annualization"],
    )
    Path(raw["output_dir"]).mkdir(parents=True, exist_ok=True)
    return raw

def load_data(profile, args):
    namespace = SimpleNamespace(**args)
    if profile == BTC_PROFILE:
        return {"BTCUSD": bt.load_btc_data(args["data_path"])}
    frame, _ = bt.load_stock_data(namespace)
    return {
        str(ticker): group.sort_values("Date")[["Date", args["price_col"]]]
        .rename(columns={"Date": "date", args["price_col"]: "price"})
        .reset_index(drop=True)
        for ticker, group in frame.groupby("Ticker", sort=True)
    }

def archived_lags(archive, profile, method, variant):
    for relative in (Path("MBDP") / "selected_lags.csv", Path("selected_lags.csv")):
        path = archive / profile / relative
        if path.is_file():
            selected = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            f"selected_lags.csv not found under {archive / profile}"
        )
    mask = (selected.method == method) & (selected.variant == variant)
    if "model" in selected.columns:
        mask &= selected.model == "MBDP"
    chosen = selected.loc[mask]
    if chosen.empty:
        raise ValueError(
            f"No archived {method}/{variant} selection for {profile}"
        )
    return {row.ticker: ast.literal_eval(row.lags) for row in chosen.itertuples()}

def fresh_btc_lags(data, args, workers, output_dir):
    candidates = bt.generate_btc_lags(
        args["n_lag_sets"],
        args["min_lags"],
        args["max_lags"],
        args["min_lag_value"],
        args["max_lag_value"],
        args["random_seed"],
    )
    unique = list(dict.fromkeys(map(tuple, candidates)))
    tasks = [(ticker, "maxlag_skew", lag, "MBDP", True)
             for ticker in data for lag in unique]
    (output_dir / "btc_lag_candidates.json").write_text(
        json.dumps([list(lag) for lag in candidates], indent=2), encoding="utf-8"
    )
    print(f"BTC train-only search: {len(tasks)} candidates, {workers} workers",
          flush=True)
    with mp.get_context("spawn").Pool(
        workers, initializer=bt.init_worker, initargs=(data, SimpleNamespace(**args))
    ) as pool:
        outputs = bt.collect(pool, tasks, "BTC Wilcoxon (train)")
    lookup = dict(zip(unique, outputs))
    np.savez_compressed(
        output_dir / "btc_training_return_paths.npz",
        **{"_".join(map(str, lag)): path for lag, path in lookup.items()},
    )
    selection = bt.select_lags(candidates, lookup, btc=True,
                               annualization=args["annualization"])
    selected = tuple(selection["wilcoxon"])
    print(f"BTC selected {args['annualization']}-annualized Wilcoxon lags: {selected}",
          flush=True)
    return selected

def reprice(result, bps, capital):
    """Charge commission on position changes while holding signals fixed."""
    position = result["position"].to_numpy(dtype=np.int64)
    price = result["today_price"].to_numpy(dtype=float)
    price_return = np.zeros(len(price))
    price_return[1:] = price[1:] / price[:-1] - 1.0
    held = np.concatenate(([0], position[:-1]))
    changed = position != held
    sides = np.where((held == 0) | (position == 0), 1.0, 2.0)
    out = result.copy()
    out["bar_pnl"] = price_return * held - sides * changed * bps / 1e4
    out["equity"] = capital * np.cumprod(1.0 + out["bar_pnl"].to_numpy())
    out["cumulative_pnl"] = out["bar_pnl"].cumsum()
    return out

def main(argv=None):
    cli = parse_args(argv)
    if cli.workers < 1:
        raise ValueError("--workers must be positive")
    profiles = [p for p in cli.profiles if p in CHOICES]
    if not profiles:
        raise ValueError(f"Nothing to do; choose from {sorted(CHOICES)}")

    roots = search_roots(cli.data_root)
    archive = resolve_archive(cli.data_root, profiles, roots)
    if archive is None:
        raise FileNotFoundError(
            "Could not locate Soogiri/bias_results_fast. Pass --data-root."
        )
    stock_data = resolve_file(cli.stock_data, "MBDP_STOCK_DATA",
                              STOCK_DATA_NAMES, roots)
    btc_data = resolve_file(cli.btc_data, "MBDP_BTC_DATA", BTC_DATA_NAMES, roots)
    if stock_data is None and any(p != BTC_PROFILE for p in profiles):
        raise FileNotFoundError("Could not locate djia30_full_2009_2022.csv")
    if btc_data is None and BTC_PROFILE in profiles:
        raise FileNotFoundError("Could not locate the BTC four-hour price pickle")

    output_dir = Path(cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Archive: {archive}\nStock data: {stock_data}\nBTC data: {btc_data}",
          flush=True)

    rows, per_profile, used_lags = [], {}, {}
    for profile in profiles:
        method, variant = CHOICES[profile]
        data_path = btc_data if profile == BTC_PROFILE else stock_data
        args = build_args(profile, output_dir, archive, data_path)
        data = load_data(profile, args)
        bt.init_worker(data, SimpleNamespace(**args))

        if profile == BTC_PROFILE:
            if cli.btc_lags == "fresh":
                lags = {ticker: fresh_btc_lags(data, args, cli.workers, output_dir)
                        for ticker in data}
            else:
                lags = archived_lags(archive, profile, method, variant)
        else:
            lags = archived_lags(archive, profile, method, variant)
        used_lags[profile] = lags

        results = {ticker: bt.evaluate((ticker, variant, tuple(lag), "MBDP", False))
                   for ticker, lag in sorted(lags.items())}
        capital = 1.0 if profile == BTC_PROFILE else 10_000.0

        profile_rows = []
        for bps in BPS:
            repriced = {ticker: reprice(result, bps, capital)
                        for ticker, result in results.items()}
            if profile == BTC_PROFILE:
                metrics = bt.compute_btc_metrics(
                    repriced["BTCUSD"], "test", args["annualization"]
                )
            else:
                portfolio = bt.build_portfolio_for_method(
                    list(repriced.values()), 1_000_000.0
                )
                metrics = bt.compute_portfolio_metrics(
                    list(repriced.values()), portfolio, args["annualization"]
                )
            row = {
                "profile": LABELS.get(profile, profile),
                "method": method,
                "variant": variant,
                "bps": bps,
                "annualization": args["annualization"],
                **metrics,
            }
            profile_rows.append(row)
            rows.append(row)
            print(
                f"{profile:>13s} {bps:2d} bps | return={row.get('return_pct', float('nan')):9.5f}% "
                f"sharpe={row.get('sharpe_ratio', float('nan')):.4f} "
                f"maxDD={row.get('max_drawdown_pct', float('nan')):8.3f}% "
                f"trades={row.get('trades', '')}",
                flush=True,
            )
        per_profile[profile] = pd.DataFrame(profile_rows)
        per_profile[profile].drop(
            columns=[c for c in HIDDEN_METRICS if c in per_profile[profile].columns]
        ).to_csv(
            output_dir / f"cost_sensitivity_{profile}.csv", index=False
        )

    frame = pd.DataFrame(rows)
    table = frame[
        ["profile", "method", "variant", "bps", "annualization", *COMMON_METRICS]
    ]
    table.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    minimal = frame[["profile", "bps", "return_pct", "sharpe_ratio"]].rename(
        columns={"sharpe_ratio": "sharpe"}
    )
    minimal.to_csv(output_dir / "4-cost_sensitivity.csv", index=False)
    for profile in per_profile:
        stem = FIGURE_STEMS.get(profile, profile)
        subset = minimal.loc[
            minimal.profile.eq(LABELS.get(profile, profile)),
            ["bps", "return_pct", "sharpe"],
        ]
        subset.to_csv(output_dir / f"{stem}.csv", index=False)
    print()
    print(minimal.to_string(index=False), flush=True)
    print(f"Figure data written to {output_dir}", flush=True)

    if BTC_PROFILE in per_profile:
        btc = per_profile[BTC_PROFILE].set_index("bps")
        summary = {
            "profile": LABELS[BTC_PROFILE],
            "btc_mode": cli.btc_lags,
            "annualization": BTC_ANNUALIZATION,
            "lags": [int(v) for v in used_lags[BTC_PROFILE]["BTCUSD"]],
            "bps": [int(v) for v in BPS],
            "return_pct": [float(btc.loc[bps, "return_pct"]) for bps in BPS],
            "sharpe_ratio": [float(btc.loc[bps, "sharpe_ratio"]) for bps in BPS],
        }
        (output_dir / "btc_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"BTC lags {tuple(summary['lags'])}; 20 bps -> "
              f"{summary['return_pct'][-1]:.8f}% / Sharpe "
              f"{summary['sharpe_ratio'][-1]:.10f}", flush=True)

if __name__ == "__main__":
    main()

