#!/usr/bin/env python3
"""
Plot mass-constraint ice changes as ice production from segment CSV output.

Input CSV is expected to be from drift_aware_mass_budget.py segment mode
and include at least:
  - time_center
  - dMdt_material_kg_s

Optional columns used when present:
  - dHdt_material_m_s
  - available_count_in_window
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_timeseries(df, rho_i, segment_area_m2):
    grouped = df.groupby("time_center", as_index=True)

    out = grouped["dMdt_material_kg_s"].agg(
        nseg="count",
        sum_dMdt_kg_s="sum",
        mean_dMdt_kg_s="mean",
        median_dMdt_kg_s="median",
    )

    # Convert summed mass tendency to equivalent thickness tendency.
    # Positive means net ice growth; negative means net melt/thinning.
    total_area = out["nseg"] * float(segment_area_m2)
    out["prod_m_s_area_mean"] = out["sum_dMdt_kg_s"] / (float(rho_i) * total_area)
    out["prod_cm_day_area_mean"] = out["prod_m_s_area_mean"] * 100.0 * 86400.0

    if "dHdt_material_m_s" in df.columns:
        hstats = grouped["dHdt_material_m_s"].agg(
            mean_dHdt_m_s="mean",
            median_dHdt_m_s="median",
        )
        out = out.join(hstats)

    if "available_count_in_window" in df.columns:
        cstats = grouped["available_count_in_window"].agg(
            mean_available_count="mean",
            min_available_count="min",
            max_available_count="max",
        )
        out = out.join(cstats)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="segment_window_union_nomask.csv")
    ap.add_argument("--out_png", default="mass_constraint_ice_production.png")
    ap.add_argument("--out_csv", default="mass_constraint_ice_production_timeseries.csv")
    ap.add_argument("--rho_i", type=float, default=915.0)
    ap.add_argument("--segment_area_m2", type=float, default=6250.0 * 6250.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, parse_dates=["time_center"])
    need = {"time_center", "dMdt_material_kg_s"}
    if not need.issubset(df.columns):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")

    ts = build_timeseries(
        df=df,
        rho_i=args.rho_i,
        segment_area_m2=args.segment_area_m2,
    )
    ts.to_csv(args.out_csv, index=True)

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax[0].plot(ts.index, ts["sum_dMdt_kg_s"], marker="o", lw=1.3, ms=3)
    ax[0].axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax[0].set_ylabel("Net material dM/dt (kg s$^{-1}$)")
    ax[0].set_title("Mass-constraint ice changes as ice production")
    ax[0].grid(True, alpha=0.25)

    ax2 = ax[1]
    ax2.plot(ts.index, ts["prod_cm_day_area_mean"], marker="o", lw=1.3, ms=3, color="tab:green")
    ax2.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax2.set_ylabel("Area-mean production (cm day$^{-1}$)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.25)

    # Lightly annotate varying segment support.
    ax2b = ax2.twinx()
    ax2b.plot(ts.index, ts["nseg"], color="tab:gray", lw=0.9, alpha=0.6)
    ax2b.set_ylabel("Segment count")
    ax2b.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200)

    print(f"[INFO] wrote plot: {args.out_png}")
    print(f"[INFO] wrote table: {args.out_csv}")
    print(
        "[INFO] net production stats (kg/s): "
        f"min={ts['sum_dMdt_kg_s'].min():.3g}, "
        f"max={ts['sum_dMdt_kg_s'].max():.3g}, "
        f"median={ts['sum_dMdt_kg_s'].median():.3g}"
    )


if __name__ == "__main__":
    main()
