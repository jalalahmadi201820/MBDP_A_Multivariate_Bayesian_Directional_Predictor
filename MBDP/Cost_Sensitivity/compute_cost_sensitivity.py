import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Base directory holding the recorded run manifests and the input data.
HQ = Path(os.environ.get("MBDP_DATA_ROOT", "."))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_MBDP_Core_Engine"))

import mbdp_rolling_bayesian_classifier as bt  # noqa: E402

# Output directory for the repriced runs.
OUT = Path(os.environ.get("MBDP_OUTPUT_DIR", "cost_sensitivity"))
BPS = [0, 5, 10, 20]
CHOICES = {
    "DRL_Ensemble": ("wilcoxon", "maxlag_skew"),
    "FineRL": ("wilcoxon", "maxlag_skew"),
    "XGboost_GRU": ("raw_pnl", "maxlag_skew"),
}


def build_args(profile):
    raw = json.loads((HQ / "Soogiri" / "bias_results_fast" / profile / "run_arguments.json").read_text())
    raw = dict(raw)
    raw["output_dir"] = str(OUT / "cost_curve_audit" / profile)
    raw["commission"] = 0.0
    raw["models"] = ["MBDP"]
    Path(raw["output_dir"]).mkdir(parents=True, exist_ok=True)
    return raw


def load_data(profile, args):
    ns = type("A", (), args)()
    if profile == "XGboost_GRU":
        return {"BTCUSD": bt.load_btc_data(ns.data_path)}
    frame, _ = bt.load_stock_data(ns)
    return {
        str(t): g.sort_values("Date")[["Date", args["price_col"]]]
        .rename(columns={"Date": "date", args["price_col"]: "price"})
        .reset_index(drop=True)
        for t, g in frame.groupby("Ticker", sort=True)
    }


def reprice(result, bps, capital):
    pos = result["position"].to_numpy(dtype=np.int64)
    price = result["today_price"].to_numpy(dtype=float)
    ret = np.zeros(len(price))
    ret[1:] = price[1:] / price[:-1] - 1.0
    held = np.concatenate([[0], pos[:-1]])
    gross = ret * held
    change = pos != held
    mult = np.where((held == 0) | (pos == 0), 1.0, 2.0)
    pnl = gross - mult * change * (bps / 1e4)
    out = result.copy()
    out["bar_pnl"] = pnl
    out["equity"] = capital * np.cumprod(1.0 + pnl)
    out["cumulative_pnl"] = np.cumsum(pnl)
    return out


def main():
    rows = []
    for profile, (method, variant) in CHOICES.items():
        args = build_args(profile)
        data = load_data(profile, args)
        bt.init_worker(data, type("A", (), args)())
        sel_path = (HQ / "Soogiri" / "bias_results_fast" / profile / "MBDP" / "selected_lags.csv")
        if not sel_path.exists():
            sel_path = HQ / "Soogiri" / "bias_results_fast" / profile / "selected_lags.csv"
        sel = pd.read_csv(sel_path)
        mask = (sel.method == method) & (sel.variant == variant)
        if "model" in sel.columns:
            mask &= sel.model == "MBDP"
        sel = sel[mask]
        lag_map = {r.ticker: eval(r.lags) for r in sel.itertuples()}
        capital = 1.0 if profile == "XGboost_GRU" else 10_000.0
        results = {}
        for ticker, lags in sorted(lag_map.items()):
            results[ticker] = bt.evaluate((ticker, variant, tuple(lags), "MBDP", False))
        for bps in BPS:
            repriced = {t: reprice(r, bps, capital) for t, r in results.items()}
            if profile == "XGboost_GRU":
                metrics = bt.compute_btc_metrics(repriced["BTCUSD"], "test", args["annualization"])
            else:
                portfolio = bt.build_portfolio_for_method(list(repriced.values()), 1_000_000.0)
                metrics = bt.compute_portfolio_metrics(list(repriced.values()), portfolio,
                                                       args["annualization"])
            rows.append(dict(profile=profile, method=method, variant=variant, bps=bps,
                             annualization=args["annualization"], **metrics))
            print(profile, bps, {k: round(v, 4) for k, v in metrics.items()
                                 if k in ("return_pct", "sharpe_ratio", "max_drawdown_pct", "trades")},
                  flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cost_sensitivity.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
