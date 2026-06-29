#!/usr/bin/env python3
"""
Eulerian mass-constraint decomposition (draft).

Builds on drift_aware_mass_budget.compute_eulerian_polynya_closure with explicit
splitting of the control-area budget into:

  dM*/dt  : storage tendency of drift-aware mass proxy M*(t)
  F_net   : coarse boundary ice-mass transport proxy (kg/s), positive = net export
  P_thermo: residual closing the budget (thermodynamic + closure error)

Algebra (same as drift_aware_mass_budget.py):
  P_thermo = dM*/dt + F_net

The same residual P_thermo is also written as SWOT–IS2 area-mean production
(P_thermo_cm_day_area_mean*, from segment thickness) in the Eulerian CSV; this
script prints those rates after the kg/s decomposition summary.

This script does NOT claim a full Lagrangian parcel budget; boundary flux is a
proxy. Use production_mode strict in the upstream run for conservative covered
diagnostics.

Usage
-----
  (1) Full run (same Eulerian arguments as drift_aware_mass_budget.py --eulerian_closure):
      python eulerian_mass_constraint_decomposition.py --ice_motion_nc ... \\
        --segment_npz ... --amsr_mat ... --out_csv mass_decomp.csv ...

  (2) Post-process an existing Eulerian CSV:
      python eulerian_mass_constraint_decomposition.py --from_csv eulerian_closure.csv \\
        --out_csv mass_decomp.csv

Comparison to RACMO in plot_SWOT_PM_RACMO_SIT.py
------------------------------------------------
This CSV keeps the same columns as the Eulerian closure (including
P_thermo_cm_day_area_mean). Pass your --out_csv path as --swot_prod_csv in the
plot script. Default loading uses whole polynya P_thermo_cm_day_area_mean
(SWOT–IS2 thickness residual per unit polynya area); that matches RACMO regional
ice production best. Do not use --swot_prod_use_covered for that comparison:
covered valid/raw divide by A_thickness_covered* and can diverge from
whole-area when coverage is small (different quantity than RACMO grid means).
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from drift_aware_mass_budget import compute_eulerian_polynya_closure


def _f(name, row, default=np.nan):
    try:
        v = row.get(name, None)
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def extend_mass_decomposition_rows(rows: list[dict]) -> list[dict]:
    """
    Add explicit advection / thermodynamic split and budget check.

    Sign convention (consistent with drift_aware_mass_budget.py):
      F_net positive  -> net export across boundary (proxy)
      F_export = max(F_net, 0), F_import = max(-F_net, 0)
      P_growth = max(P_thermo, 0), P_melt = max(-P_thermo, 0)
    """
    out = []
    for r in rows:
        dmdt = _f("dMdt_kg_s", r)
        # Prefer explicit Fout column names from closure CSV
        fnet = _f("Fout_kg_s", r)
        if not np.isfinite(fnet):
            fnet = _f("Fout_boundary_transport_proxy_kg_s", r)
        pth = _f("P_thermo_kg_s", r)

        r = dict(r)
        r["F_net_kg_s"] = fnet
        r["F_export_kg_s"] = float(max(fnet, 0.0)) if np.isfinite(fnet) else np.nan
        r["F_import_kg_s"] = float(max(-fnet, 0.0)) if np.isfinite(fnet) else np.nan
        r["P_thermo_growth_kg_s"] = float(max(pth, 0.0)) if np.isfinite(pth) else np.nan
        r["P_thermo_melt_kg_s"] = float(max(-pth, 0.0)) if np.isfinite(pth) else np.nan
        # Closure: P - (dMdt + F) should be ~0
        if np.isfinite(dmdt) and np.isfinite(fnet) and np.isfinite(pth):
            r["budget_closure_error_kg_s"] = float(pth - (dmdt + fnet))
        else:
            r["budget_closure_error_kg_s"] = np.nan

        # Shares of absolute magnitude (diagnostic only; not energy partition)
        num = abs(dmdt) + abs(fnet) + abs(pth)
        if np.isfinite(num) and num > 0:
            r["share_abs_dMdt"] = float(abs(dmdt) / num)
            r["share_abs_F_net"] = float(abs(fnet) / num)
            r["share_abs_P_thermo"] = float(abs(pth) / num)
        else:
            r["share_abs_dMdt"] = np.nan
            r["share_abs_F_net"] = np.nan
            r["share_abs_P_thermo"] = np.nan

        out.append(r)
    return out


def _cumulative_trapezoid_kg(times: pd.Series, y_kg_s: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of rate (kg/s) -> kg."""
    t_ns = pd.to_datetime(times).astype("int64").to_numpy(dtype=np.int64)
    y = np.asarray(y_kg_s, dtype=float)
    out = np.zeros(y.shape[0], dtype=float)
    for i in range(1, y.shape[0]):
        if (
            np.isfinite(y[i])
            and np.isfinite(y[i - 1])
            and np.isfinite(t_ns[i])
            and np.isfinite(t_ns[i - 1])
        ):
            dt_sec = (t_ns[i] - t_ns[i - 1]) * 1e-9
            out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * dt_sec
        else:
            out[i] = out[i - 1]
    return out


def add_cumulative_integrals(rows: list[dict]) -> list[dict]:
    """Append cum_* columns (kg) for dMdt, F_net, P_thermo."""
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    if "time_center" not in df.columns:
        return rows
    t = pd.to_datetime(df["time_center"])
    for key, col in [
        ("cum_dMdt_kg", "dMdt_kg_s"),
        ("cum_F_net_kg", "F_net_kg_s"),
        ("cum_P_thermo_kg", "P_thermo_kg_s"),
    ]:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        # Use F_net if already extended; else Fout
        if col == "F_net_kg_s" and not np.any(np.isfinite(y)):
            alt = "Fout_kg_s" if "Fout_kg_s" in df.columns else None
            if alt:
                y = pd.to_numeric(df[alt], errors="coerce").to_numpy(dtype=float)
        df[key] = _cumulative_trapezoid_kg(t, y)
    return df.to_dict("records")


def load_csv_rows(path: str) -> list[dict]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def write_csv_rows(rows: list[dict], path: str) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_swot_is2_production_summary(rows: list[dict]) -> None:
    """
    Print area-mean ice production rates from the SWOT–IS2 segment thickness
    field used in the Eulerian closure (same residual as P_thermo, expressed
    as cm/day over whole polynya or covered area).
    """
    if not rows:
        print("[INFO] No rows for SWOT–IS2 production summary.")
        return

    def _arr(key: str) -> np.ndarray:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key, np.nan)))
            except Exception:
                vals.append(np.nan)
        return np.asarray(vals, dtype=float)

    def _fmt_stats(name: str, arr: np.ndarray) -> None:
        m = np.isfinite(arr)
        n = int(np.count_nonzero(m))
        if n == 0:
            print(f"  {name}: n=0 (all NaN)")
            return
        a = arr[m]
        print(
            f"  {name}: n={n}, mean={float(np.nanmean(a)):.3f}, median={float(np.nanmedian(a)):.3f}, "
            f"range=[{float(np.nanmin(a)):.3f}, {float(np.nanmax(a)):.3f}]"
        )

    keys_cm = [
        ("P_thermo_cm_day_area_mean", "Whole polynya (cm/day)"),
        ("P_thermo_cm_day_area_mean_covered", "Covered valid (cm/day)"),
        ("P_thermo_cm_day_area_mean_covered_raw", "Covered raw (cm/day)"),
    ]
    keys_ms = [
        ("P_thermo_m_s_area_mean", "Whole polynya (m/s)"),
        ("P_thermo_m_s_area_mean_covered", "Covered valid (m/s)"),
        ("P_thermo_m_s_area_mean_covered_raw", "Covered raw (m/s)"),
    ]
    any_present = any(k in rows[0] for k, _ in keys_cm)
    if not any_present:
        print(
            "[INFO] SWOT–IS2 ice production: no P_thermo_cm_day_* columns in "
            "rows (use an Eulerian closure CSV from drift_aware_mass_budget.py)."
        )
        return

    print("[INFO] SWOT–IS2 ice production (Eulerian thickness closure; area-mean rates)")
    for key, label in keys_cm:
        if key not in rows[0]:
            continue
        _fmt_stats(label, _arr(key))
    for key, label in keys_ms:
        if key not in rows[0]:
            continue
        _fmt_stats(label, _arr(key))

    if "P_thermo_cm_day_area_mean" in rows[0]:
        _fmt_stats(
            "Whole polynya (m/month, 30 d)",
            _arr("P_thermo_cm_day_area_mean") * 0.30,
        )
    if "P_thermo_cm_day_area_mean_covered_raw" in rows[0]:
        _fmt_stats(
            "Covered raw (m/month, 30 d)",
            _arr("P_thermo_cm_day_area_mean_covered_raw") * 0.30,
        )
    if "P_thermo_cm_day_area_mean_unc_proxy" in rows[0]:
        _fmt_stats("Whole polynya unc proxy (cm/day)", _arr("P_thermo_cm_day_area_mean_unc_proxy"))
    if "P_thermo_cm_day_area_mean_covered_unc_proxy" in rows[0]:
        _fmt_stats(
            "Covered valid unc proxy (cm/day)",
            _arr("P_thermo_cm_day_area_mean_covered_unc_proxy"),
        )


def print_decomposition_summary(rows: list[dict]) -> None:
    if not rows:
        print("[INFO] No rows for decomposition summary.")
        return
    df = pd.DataFrame(rows)
    print("[INFO] Mass-constraint decomposition summary (kg/s)")
    for col, label in [
        ("dMdt_kg_s", "dM*/dt (kg/s)"),
        ("F_net_kg_s", "F_net (kg/s)"),
        ("P_thermo_kg_s", "P_thermo (kg/s)"),
        ("budget_closure_error_kg_s", "closure error P-(dMdt+F) (kg/s)"),
    ]:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x)
        n = int(np.count_nonzero(m))
        if n == 0:
            print(f"  {label}: n=0")
            continue
        print(
            f"  {label}: n={n}, median={float(np.nanmedian(x)):.3e}, "
            f"range=[{float(np.nanmin(x)):.3e}, {float(np.nanmax(x)):.3e}]"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Eulerian mass constraint: advection vs thermodynamic residual (draft)."
    )
    ap.add_argument(
        "--from_csv",
        default=None,
        help="Existing Eulerian closure CSV from drift_aware_mass_budget.py (skip recomputation).",
    )
    ap.add_argument("--out_csv", default="eulerian_mass_decomposition.csv")
    ap.add_argument("--no_cumulative", action="store_true", help="Skip cumulative integrals (kg).")

    # Pass-through to compute_eulerian_polynya_closure when --from_csv is not used
    ap.add_argument("--segment_npz", default=None)
    ap.add_argument("--amsr_mat", default=None)
    ap.add_argument("--ice_motion_nc", default=None)
    ap.add_argument("--rho_i", type=float, default=915.0)
    ap.add_argument("--polynya_threshold", type=float, default=0.7)
    ap.add_argument("--polynya_type", choices=["open", "coastal", "both"], default="coastal")
    ap.add_argument("--max_dt_days", type=float, default=31.0)
    ap.add_argument("--thickness_reconstruct", choices=["none", "tracked_fusion"], default="none")
    ap.add_argument("--fusion_stat", choices=["median", "mean"], default="median")
    ap.add_argument("--fusion_max_dist_km", type=float, default=100.0)
    ap.add_argument("--center_fill_max_dist_km", type=float, default=None)
    ap.add_argument("--segment_step_hours", type=float, default=6.0)
    ap.add_argument("--ice_motion_subsample", type=int, default=5)
    ap.add_argument("--min_coverage_fraction", type=float, default=0.0)
    ap.add_argument("--min_boundary_flux_coverage_fraction", type=float, default=0.0)
    ap.add_argument("--min_coverage_fraction_for_covered_rate", type=float, default=1e-4)
    ap.add_argument("--region_lon_min", type=float, default=None)
    ap.add_argument("--region_lon_max", type=float, default=None)
    ap.add_argument("--region_lat_min", type=float, default=None)
    ap.add_argument("--region_lat_max", type=float, default=None)
    ap.add_argument("--covered_valid_min_boundary_flux_coverage_fraction", type=float, default=0.015)
    ap.add_argument("--covered_valid_min_coverage_fraction", type=float, default=0.002)
    ap.add_argument("--covered_valid_min_n_window_scenes_used", type=float, default=3.0)
    ap.add_argument("--covered_valid_min_effective_boundary_flux_length_m", type=float, default=8e4)
    ap.add_argument("--production_mode", choices=["standard", "strict"], default="standard")
    ap.add_argument("--backtrack_days", type=float, default=None)
    ap.add_argument("--forwardtrack_days", type=float, default=None)
    ap.add_argument(
        "--no_fill_nan_velocity_zero_on_ocean",
        action="store_true",
        help="Disable NaN drift -> 0 fill over AMSR-valid surface in advection",
    )

    args = ap.parse_args()

    if args.from_csv:
        path = Path(args.from_csv)
        if not path.is_file():
            raise FileNotFoundError(f"--from_csv not found: {path}")
        rows = load_csv_rows(str(path))
        print(f"[INFO] Loaded {len(rows)} rows from {path}")
    else:
        if args.segment_npz is None or args.amsr_mat is None or args.ice_motion_nc is None:
            raise RuntimeError(
                "Without --from_csv, require --segment_npz, --amsr_mat, and --ice_motion_nc."
            )
        fd, tmp_out = tempfile.mkstemp(suffix="_eulerian_closure.csv", prefix="mass_decomp_")
        os.close(fd)
        rows = compute_eulerian_polynya_closure(
            segment_npz=args.segment_npz,
            amsr_mat=args.amsr_mat,
            ice_motion_nc=args.ice_motion_nc,
            out_csv=tmp_out,
            rho_i=args.rho_i,
            polynya_threshold=args.polynya_threshold,
            polynya_type=args.polynya_type,
            max_dt_days=args.max_dt_days,
            thickness_reconstruct=args.thickness_reconstruct,
            fusion_stat=args.fusion_stat,
            fusion_max_dist_km=args.fusion_max_dist_km,
            center_fill_max_dist_km=args.center_fill_max_dist_km,
            step_hours=args.segment_step_hours,
            backtrack_days=args.backtrack_days,
            forwardtrack_days=args.forwardtrack_days,
            fill_nan_velocity_zero_on_ocean=not args.no_fill_nan_velocity_zero_on_ocean,
            ice_motion_subsample=args.ice_motion_subsample,
            min_coverage_fraction=args.min_coverage_fraction,
            min_boundary_flux_coverage_fraction=args.min_boundary_flux_coverage_fraction,
            min_coverage_fraction_for_covered_rate=args.min_coverage_fraction_for_covered_rate,
            region_lon_min=args.region_lon_min,
            region_lon_max=args.region_lon_max,
            region_lat_min=args.region_lat_min,
            region_lat_max=args.region_lat_max,
            covered_valid_min_boundary_flux_coverage_fraction=args.covered_valid_min_boundary_flux_coverage_fraction,
            covered_valid_min_coverage_fraction=args.covered_valid_min_coverage_fraction,
            covered_valid_min_n_window_scenes_used=args.covered_valid_min_n_window_scenes_used,
            covered_valid_min_effective_boundary_flux_length_m=args.covered_valid_min_effective_boundary_flux_length_m,
            production_mode=args.production_mode,
        )
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        print(f"[INFO] Computed {len(rows)} Eulerian closure rows (intermediate)")

    rows = extend_mass_decomposition_rows(rows)
    if not args.no_cumulative:
        rows = add_cumulative_integrals(rows)

    write_csv_rows(rows, args.out_csv)
    print(f"[INFO] Wrote mass decomposition CSV: {args.out_csv}")
    print_decomposition_summary(rows)
    print_swot_is2_production_summary(rows)
    print(
        "[INFO] plot_SWOT_PM_RACMO_SIT.py: use --swot_prod_csv "
        f"{args.out_csv} without --swot_prod_use_covered for RACMO-comparable "
        "SWOT–IS2 production (whole-area P_thermo_cm_day_area_mean → m/month)."
    )


if __name__ == "__main__":
    main()
