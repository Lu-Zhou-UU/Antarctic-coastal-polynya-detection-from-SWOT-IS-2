#!/usr/bin/env python3
"""
Regional-mean IS2 / SWOT–IS2 track thickness with Lagrangian time-window sampling,
then lumped polynya mass budget and decomposition.

For each center time t (each scene in the segment NPZ stack), we estimate a
**regional mean thickness** from all segment SIT samples that lie inside the
AMSR polynya mask (and optional lon/lat bbox) **at time t** after advection:

  - For every scene time t_k with |t_k - t| <= lag_window_days (default: 3 days),
    each segment point (lon, lat, h) observed at t_k is advected to time t using
    the same ice-motion stepping as drift_aware_mass_budget.tracked_fusion.

  - If the advected position falls inside the polynya mask at t, h is pooled.

This increases effective sample size (your t−3 … t+3 Lagrangian sampling idea)
while keeping an Eulerian polynya mask at the center time.

Lumped mass proxy (diagnostic):
  M*(t) = rho_i * h_regional(t) * A_polynya(t)

Boundary flux uses the **scalar** h_regional on the boundary (same lumped
approximation as drift_aware_mass_budget.scene_mass_from_mean_only).

  P_thermo = dM*/dt + F_out   (central difference over consecutive scenes)

Inferred ice production (same residual, explicit names in CSV):
  ice_production_* columns: net area-mean rate (cm/day, m/month), growth/melt split
  (positive vs negative P_thermo), and mass tendency (kg/s).

Output CSV: time_center + ice_production_m_month_area_mean (and P_thermo_cm_day_area_mean).
plot_SWOT_PM_RACMO_SIT.py prefers ice_production_m_month_area_mean for the production overlay
when not using --swot_prod_use_covered. Also compatible with eulerian_mass_constraint_decomposition.extend_*.

Example
-------
  cd .../SWOT
  python3 regional_track_mean_lumped_decomposition.py \\
    --segment_npz results/is2_polynya_sit \\
    --amsr_mat AMSR2_2023.mat \\
    --ice_motion_nc /path/to/icemotion_daily.nc \\
    --lag_window_days 3 \\
    --out_csv regional_lumped_ross.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import timedelta

import numpy as np

from drift_aware_mass_budget import (
    _advect_points_between_times,
    _apply_region_bbox_mask,
    _boundary_flux_and_uncertainty,
    _land_mask_for_date,
    _load_segment_snapshot_npz,
    _neighbors4,
    _load_ice_motion_cached,
    _sample_bool_mask_nearest,
    _velocity_at_points,
    grid_cell_area_m2,
    load_polynya_mask_for_date,
    mask_boundary_4n,
    mean_spacing_m,
)

try:
    from eulerian_mass_constraint_decomposition import (
        add_cumulative_integrals,
        extend_mass_decomposition_rows,
    )
except ImportError:
    extend_mass_decomposition_rows = None
    add_cumulative_integrals = None


def _safe_f(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _pooled_h_stats(all_h: np.ndarray) -> dict:
    """Sample spread of pooled thickness (m) at center time; used for production error proxy."""
    all_h = np.asarray(all_h, dtype=float).ravel()
    all_h = all_h[np.isfinite(all_h)]
    if all_h.size == 0:
        return {}
    out = {
        "h_track_std_m": float(np.nanstd(all_h, ddof=1)) if all_h.size > 1 else 0.0,
        "h_track_p25_m": float(np.nanpercentile(all_h, 25.0)),
        "h_track_p50_m": float(np.nanmedian(all_h)),
        "h_track_p75_m": float(np.nanpercentile(all_h, 75.0)),
    }
    return out


def _indices_in_lag_window(times, i_center: int, lag_days: float) -> list[int]:
    t0 = times[i_center]
    dt_max = timedelta(days=float(lag_days))
    out = []
    for k in range(len(times)):
        if abs(times[k] - t0) <= dt_max:
            out.append(k)
    return out


def _regional_mean_from_lag_window(
    times,
    lon_seg,
    lat_seg,
    h_seg,
    i_center: int,
    lag_days: float,
    amsr_mat: str,
    ice_motion_nc: str,
    step_hours: float,
    polynya_threshold: float,
    polynya_type: str,
    region_lon_min,
    region_lon_max,
    region_lat_min,
    region_lat_max,
    fill_nan_velocity_zero_on_ocean: bool,
    ice_motion_subsample: int,
    regional_stat: str,
    min_track_samples: int,
):
    """
    Pool thickness samples advected into the polynya+region mask at center time.
    Returns (h_regional, n_samples, n_scenes_used, stats_dict) or (nan, 0, 0, None).
    stats_dict contains h_track_std_m, h_track_p25_m, h_track_p50_m, h_track_p75_m when pooled.
    """
    t_c = times[i_center]
    lat_m, lon_m, mask_full = load_polynya_mask_for_date(
        amsr_mat, t_c,
        polynya_threshold=polynya_threshold,
        polynya_type=polynya_type,
    )
    mask_m = _apply_region_bbox_mask(
        mask_full, lon_m, lat_m,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
    )
    win_idx = _indices_in_lag_window(times, i_center, lag_days)
    pooled = []
    n_scenes_with_any = 0
    for k in win_idx:
        lo = np.asarray(lon_seg[k, :], dtype=float).ravel()
        la = np.asarray(lat_seg[k, :], dtype=float).ravel()
        hh = np.asarray(h_seg[k, :], dtype=float).ravel()
        m = np.isfinite(lo) & np.isfinite(la) & np.isfinite(hh)
        if not np.any(m):
            continue
        lo = lo[m]
        la = la[m]
        hh = hh[m]
        lon_a, lat_a = _advect_points_between_times(
            lo, la,
            t_start=times[k],
            t_end=t_c,
            ice_motion_nc=ice_motion_nc,
            step_hours=step_hours,
            amsr_mat=amsr_mat,
            fill_nan_velocity_zero_on_ocean=fill_nan_velocity_zero_on_ocean,
            ice_motion_subsample=ice_motion_subsample,
        )
        inside = _sample_bool_mask_nearest(lon_m, lat_m, mask_m, lon_a, lat_a)
        if np.any(inside):
            n_scenes_with_any += 1
            pooled.append(hh[inside].astype(float))
    if len(pooled) == 0:
        return np.nan, 0, 0, None
    all_h = np.concatenate(pooled, axis=0)
    n = int(all_h.size)
    stats = _pooled_h_stats(all_h)
    if n < int(min_track_samples):
        return np.nan, n, n_scenes_with_any, stats
    if regional_stat == "median":
        h_reg = float(np.nanmedian(all_h))
    else:
        h_reg = float(np.nanmean(all_h))
    return h_reg, n, n_scenes_with_any, stats


def _lumped_fout_at_time(
    amsr_mat: str,
    t,
    rho_i: float,
    h_scalar: float,
    polynya_threshold: float,
    polynya_type: str,
    region_lon_min,
    region_lon_max,
    region_lat_min,
    region_lat_max,
    ice_motion_nc: str,
    ice_motion_subsample: int,
    min_boundary_flux_coverage_fraction: float,
):
    """Scalar-thickness boundary export proxy (kg/s), same structure as Eulerian."""
    if not np.isfinite(h_scalar):
        return np.nan, np.nan, np.nan, np.nan

    lat_m, lon_m, mask_full = load_polynya_mask_for_date(
        amsr_mat, t,
        polynya_threshold=polynya_threshold,
        polynya_type=polynya_type,
    )
    mask_m = _apply_region_bbox_mask(
        mask_full, lon_m, lat_m,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
    )

    bnd_all = mask_boundary_4n(mask_full) & mask_m
    land_m = _land_mask_for_date(amsr_mat, t)
    bnd = bnd_all & _neighbors4(land_m)
    if not np.any(bnd):
        bnd = bnd_all

    ds = mean_spacing_m(lat_m, lon_m, bnd)
    if not np.isfinite(ds):
        ds = 6250.0

    gy, gx = np.gradient(mask_full.astype(float))
    norm = np.sqrt(gx ** 2 + gy ** 2)
    nx = np.divide(gx, norm, out=np.zeros_like(gx), where=norm > 0)
    ny = np.divide(gy, norm, out=np.zeros_like(gy), where=norm > 0)

    idx_bnd = np.where(bnd)
    lon_b = lon_m[idx_bnd]
    lat_b = lat_m[idx_bnd]

    lat_i, lon_i, u_i, v_i, valid_i = _load_ice_motion_cached(
        ice_motion_nc, t, subsample=ice_motion_subsample
    )
    u_b, v_b = _velocity_at_points(
        lon_i, lat_i, u_i, v_i, valid_i, q_lon=lon_b, q_lat=lat_b
    )
    un = u_b * nx[idx_bnd] + v_b * ny[idx_bnd]
    h_b = np.full(un.shape, float(h_scalar), dtype=float)
    m_bflux = np.isfinite(h_b) & np.isfinite(un)
    boundary_flux_coverage_fraction = (
        float(np.count_nonzero(m_bflux) / m_bflux.size) if m_bflux.size > 0 else np.nan
    )
    if (
        np.isfinite(boundary_flux_coverage_fraction)
        and boundary_flux_coverage_fraction < float(min_boundary_flux_coverage_fraction)
    ):
        return np.nan, np.nan, boundary_flux_coverage_fraction, ds

    fout, fout_unc = _boundary_flux_and_uncertainty(h_b, un, ds, rho_i)
    return fout, fout_unc, boundary_flux_coverage_fraction, ds


def compute_regional_track_mean_lumped_closure(
    segment_npz: str,
    amsr_mat: str,
    ice_motion_nc: str,
    rho_i: float = 915.0,
    polynya_threshold: float = 0.7,
    polynya_type: str = "coastal",
    max_dt_days: float = 31.0,
    lag_window_days: float = 3.0,
    step_hours: float = 24.0,
    ice_motion_subsample: int = 5,
    fill_nan_velocity_zero_on_ocean: bool = True,
    region_lon_min=None,
    region_lon_max=None,
    region_lat_min=None,
    region_lat_max=None,
    min_boundary_flux_coverage_fraction: float = 0.0,
    min_track_samples: int = 3,
    regional_stat: str = "mean",
    days_per_month: float = 30.0,
):
    """
    Build per-scene lumped scenes then central-difference closure rows (dicts).
    """
    times, lon_seg, lat_seg, h_seg = _load_segment_snapshot_npz(segment_npz)
    ntime = len(times)
    if ntime < 3:
        raise RuntimeError("Need at least 3 scenes for central differences.")

    scenes = []
    for i in range(ntime):
        lat_m, lon_m, mask_full = load_polynya_mask_for_date(
            amsr_mat, times[i],
            polynya_threshold=polynya_threshold,
            polynya_type=polynya_type,
        )
        mask_m = _apply_region_bbox_mask(
            mask_full, lon_m, lat_m,
            region_lon_min=region_lon_min,
            region_lon_max=region_lon_max,
            region_lat_min=region_lat_min,
            region_lat_max=region_lat_max,
        )
        area_cell = grid_cell_area_m2(lat_m, lon_m)
        idx_mask = np.where(mask_m & np.isfinite(area_cell))
        if idx_mask[0].size < 3:
            continue

        A_poly = float(np.nansum(area_cell[idx_mask]))
        if not np.isfinite(A_poly) or A_poly <= 0:
            continue

        h_reg, n_samp, n_win_scenes, h_stats = _regional_mean_from_lag_window(
            times, lon_seg, lat_seg, h_seg, i,
            lag_days=lag_window_days,
            amsr_mat=amsr_mat,
            ice_motion_nc=ice_motion_nc,
            step_hours=step_hours,
            polynya_threshold=polynya_threshold,
            polynya_type=polynya_type,
            region_lon_min=region_lon_min,
            region_lon_max=region_lon_max,
            region_lat_min=region_lat_min,
            region_lat_max=region_lat_max,
            fill_nan_velocity_zero_on_ocean=fill_nan_velocity_zero_on_ocean,
            ice_motion_subsample=ice_motion_subsample,
            regional_stat=regional_stat,
            min_track_samples=min_track_samples,
        )
        if not np.isfinite(h_reg):
            continue

        fout, fout_unc, bfcov, _ds = _lumped_fout_at_time(
            amsr_mat, times[i], rho_i, h_reg,
            polynya_threshold, polynya_type,
            region_lon_min, region_lon_max, region_lat_min, region_lat_max,
            ice_motion_nc, ice_motion_subsample,
            min_boundary_flux_coverage_fraction,
        )
        if not np.isfinite(fout):
            continue

        mass_kg = float(rho_i * h_reg * A_poly)
        sc = {
            "t": times[i],
            "area_m2": A_poly,
            "h_regional_track_mean_m": h_reg,
            "n_is2_track_samples": n_samp,
            "n_window_scenes_with_samples": n_win_scenes,
            "lag_window_days": float(lag_window_days),
            "regional_stat": regional_stat,
            "mass_kg": mass_kg,
            "fout_kg_s": fout,
            "fout_kg_s_unc_proxy": fout_unc,
            "boundary_flux_coverage_fraction": bfcov,
        }
        if h_stats:
            sc.update(h_stats)
        scenes.append(sc)

    scenes = sorted(scenes, key=lambda s: s["t"])
    if len(scenes) < 3:
        raise RuntimeError(
            "Not enough valid lumped scenes after regional sampling / boundary gates."
        )

    rows = []
    for j in range(1, len(scenes) - 1):
        s0, s1, s2 = scenes[j - 1], scenes[j], scenes[j + 1]
        dt_sec = (s2["t"] - s0["t"]).total_seconds()
        if dt_sec <= 0:
            continue
        if max_dt_days is not None and abs((s2["t"] - s0["t"]).days) > float(max_dt_days):
            continue

        dmdt = (s2["mass_kg"] - s0["mass_kg"]) / dt_sec
        p_thermo = dmdt + s1["fout_kg_s"]
        A1 = s1["area_m2"]
        prod_m_s = p_thermo / (rho_i * A1) if A1 > 0 else np.nan
        cm_day = prod_m_s * 100.0 * 86400.0 if np.isfinite(prod_m_s) else np.nan
        dpm = float(days_per_month)
        m_month = (cm_day / 100.0) * dpm if np.isfinite(cm_day) else np.nan
        growth_cm = max(cm_day, 0.0) if np.isfinite(cm_day) else np.nan
        melt_cm = max(-cm_day, 0.0) if np.isfinite(cm_day) else np.nan
        pgrow_kg = max(p_thermo, 0.0) if np.isfinite(p_thermo) else np.nan
        pmelt_kg = max(-p_thermo, 0.0) if np.isfinite(p_thermo) else np.nan
        ice_vol_m3_s = p_thermo / rho_i if np.isfinite(p_thermo) else np.nan

        # Uncertainty proxy on m/month from spread of pooled track thickness at center time:
        # scale |m_month| by fractional distance of regional h to p25/p75 (asymmetric) or std/h (symmetric).
        h_m = float(s1["h_regional_track_mean_m"])
        p25 = _safe_f(s1.get("h_track_p25_m"))
        p75 = _safe_f(s1.get("h_track_p75_m"))
        hstd = _safe_f(s1.get("h_track_std_m"))
        den = max(abs(h_m), 1e-9)
        if np.isfinite(m_month) and np.isfinite(p25) and np.isfinite(p75):
            err_mmo_lo = abs(m_month) * max(h_m - p25, 0.0) / den
            err_mmo_hi = abs(m_month) * max(p75 - h_m, 0.0) / den
        else:
            err_mmo_lo = err_mmo_hi = np.nan
        if np.isfinite(m_month) and np.isfinite(hstd):
            err_mmo_sym = abs(m_month) * (hstd / den)
        else:
            err_mmo_sym = np.nan
        m_month_growth = max(m_month, 0.0) if np.isfinite(m_month) else np.nan

        rows.append({
            "time_center": s1["t"].isoformat(sep=" "),
            "time_back": s0["t"].isoformat(sep=" "),
            "time_fwd": s2["t"].isoformat(sep=" "),
            "method": "regional_track_mean_lumped_lagrangian_window",
            "lag_window_days": s1["lag_window_days"],
            "regional_stat": s1["regional_stat"],
            "h_regional_track_mean_m": s1["h_regional_track_mean_m"],
            "h_track_std_m": s1.get("h_track_std_m", np.nan),
            "h_track_p25_m": s1.get("h_track_p25_m", np.nan),
            "h_track_p50_m": s1.get("h_track_p50_m", np.nan),
            "h_track_p75_m": s1.get("h_track_p75_m", np.nan),
            "n_is2_track_samples": s1["n_is2_track_samples"],
            "n_window_scenes_with_samples": s1["n_window_scenes_with_samples"],
            "A_polynya_m2": s1["area_m2"],
            "M_polynya_kg": s1["mass_kg"],
            "dMdt_kg_s": dmdt,
            "Fout_kg_s": s1["fout_kg_s"],
            "Fout_boundary_transport_proxy_kg_s": s1["fout_kg_s"],
            "fout_unc_proxy_kg_s": s1["fout_kg_s_unc_proxy"],
            "P_thermo_kg_s": p_thermo,
            "P_thermo_m_s_area_mean": prod_m_s,
            "P_thermo_cm_day_area_mean": cm_day,
            # Inferred ice production = budget residual (same as P_thermo); explicit labels for analysis.
            "ice_production_residual_kg_s": p_thermo,
            "ice_production_volume_tendency_m3_s": ice_vol_m3_s,
            "ice_production_cm_day_area_mean": cm_day,
            "ice_production_m_month_area_mean": m_month,
            "ice_production_m_month_growth_only": m_month_growth,
            "ice_production_m_month_err_lower": err_mmo_lo,
            "ice_production_m_month_err_upper": err_mmo_hi,
            "ice_production_m_month_err_symmetric": err_mmo_sym,
            "ice_production_growth_cm_day": growth_cm,
            "ice_production_melt_cm_day": melt_cm,
            "ice_production_growth_kg_s": pgrow_kg,
            "ice_production_melt_kg_s": pmelt_kg,
            "ice_production_m_month_growth": (growth_cm / 100.0) * dpm if np.isfinite(growth_cm) else np.nan,
            "ice_production_m_month_melt": (melt_cm / 100.0) * dpm if np.isfinite(melt_cm) else np.nan,
            "boundary_flux_coverage_fraction": s1["boundary_flux_coverage_fraction"],
        })
    return rows


def _arr_col(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, np.nan)))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _print_stats_line(label: str, arr: np.ndarray) -> None:
    m = np.isfinite(arr)
    n = int(np.count_nonzero(m))
    if n == 0:
        print(f"  {label}: n=0 (all NaN)")
        return
    a = arr[m]
    print(
        f"  {label}: n={n}, mean={float(np.nanmean(a)):.4f}, median={float(np.nanmedian(a)):.4f}, "
        f"range=[{float(np.nanmin(a)):.4f}, {float(np.nanmax(a)):.4f}]"
    )


def _print_summary(rows: list[dict], days_per_month: float = 30.0) -> None:
    if not rows:
        print("[INFO] No closure rows.")
        return
    p = _arr_col(rows, "P_thermo_cm_day_area_mean")
    h = _arr_col(rows, "h_regional_track_mean_m")
    m = np.isfinite(p)
    print("[INFO] Regional lumped closure summary")
    print(
        f"  h_regional_track_mean (m): n={len(rows)}, "
        f"median={float(np.nanmedian(h)):.4f}, range=[{float(np.nanmin(h)):.4f}, {float(np.nanmax(h)):.4f}]"
    )
    if np.any(m):
        print(
            f"  P_thermo whole-polynya (cm/day): n={int(np.count_nonzero(m))}, "
            f"median={float(np.nanmedian(p[m])):.4f}, "
            f"range=[{float(np.nanmin(p[m])):.4f}, {float(np.nanmax(p[m])):.4f}]"
        )

    print(
        "[INFO] Inferred ice production (budget residual P_thermo = dM*/dt + F_out; "
        "whole-polynya area-mean rate; growth/melt are positive parts of net)"
    )
    _print_stats_line(
        f"net ice production (cm/day)",
        _arr_col(rows, "ice_production_cm_day_area_mean"),
    )
    _print_stats_line(
        f"net ice production (m/month, {days_per_month:.0f} d)",
        _arr_col(rows, "ice_production_m_month_area_mean"),
    )
    _print_stats_line("ice growth component (cm/day)", _arr_col(rows, "ice_production_growth_cm_day"))
    _print_stats_line("ice melt component (cm/day)", _arr_col(rows, "ice_production_melt_cm_day"))
    _print_stats_line("net mass tendency (kg/s)", _arr_col(rows, "ice_production_residual_kg_s"))
    _print_stats_line("volume tendency (m^3/s)", _arr_col(rows, "ice_production_volume_tendency_m3_s"))
    _print_stats_line("growth mass tendency (kg/s)", _arr_col(rows, "ice_production_growth_kg_s"))
    _print_stats_line("melt mass tendency (kg/s)", _arr_col(rows, "ice_production_melt_kg_s"))


def main():
    ap = argparse.ArgumentParser(
        description="Lumped mass budget using regional mean track SIT + Lagrangian ±lag window sampling."
    )
    ap.add_argument("--segment_npz", required=True)
    ap.add_argument("--amsr_mat", required=True)
    ap.add_argument("--ice_motion_nc", required=True)
    ap.add_argument("--out_csv", default="regional_track_mean_lumped.csv")
    ap.add_argument("--rho_i", type=float, default=915.0)
    ap.add_argument("--polynya_threshold", type=float, default=0.7)
    ap.add_argument("--polynya_type", choices=["open", "coastal", "both"], default="coastal")
    ap.add_argument("--max_dt_days", type=float, default=31.0)
    ap.add_argument(
        "--lag_window_days",
        type=float,
        default=3.0,
        help="Pool segment points from scenes with |t_k - t_center| <= this (days); "
             "each point is advected to t_center before polynya test.",
    )
    ap.add_argument("--segment_step_hours", type=float, default=24.0)
    ap.add_argument("--ice_motion_subsample", type=int, default=5)
    ap.add_argument("--min_boundary_flux_coverage_fraction", type=float, default=0.0)
    ap.add_argument("--min_track_samples", type=int, default=3,
                    help="Minimum pooled IS2 thickness samples required to form h_regional.")
    ap.add_argument("--regional_stat", choices=["mean", "median"], default="mean")
    ap.add_argument("--region_lon_min", type=float, default=None)
    ap.add_argument("--region_lon_max", type=float, default=None)
    ap.add_argument("--region_lat_min", type=float, default=None)
    ap.add_argument("--region_lat_max", type=float, default=None)
    ap.add_argument(
        "--no_fill_nan_velocity_zero_on_ocean",
        action="store_true",
        help="Disable NaN drift -> 0 over AMSR ocean mask during advection.",
    )
    ap.add_argument("--no_extend_decomposition", action="store_true",
                    help="Skip F_export / cum_* extension from eulerian_mass_constraint_decomposition.")
    ap.add_argument("--no_cumulative", action="store_true")
    ap.add_argument(
        "--days_per_month",
        type=float,
        default=30.0,
        help="Days per month for ice_production_m_month_* columns and printed m/month (default 30).",
    )

    args = ap.parse_args()

    rows = compute_regional_track_mean_lumped_closure(
        segment_npz=args.segment_npz,
        amsr_mat=args.amsr_mat,
        ice_motion_nc=args.ice_motion_nc,
        rho_i=args.rho_i,
        polynya_threshold=args.polynya_threshold,
        polynya_type=args.polynya_type,
        max_dt_days=args.max_dt_days,
        lag_window_days=args.lag_window_days,
        step_hours=args.segment_step_hours,
        ice_motion_subsample=args.ice_motion_subsample,
        fill_nan_velocity_zero_on_ocean=not args.no_fill_nan_velocity_zero_on_ocean,
        region_lon_min=args.region_lon_min,
        region_lon_max=args.region_lon_max,
        region_lat_min=args.region_lat_min,
        region_lat_max=args.region_lat_max,
        min_boundary_flux_coverage_fraction=args.min_boundary_flux_coverage_fraction,
        min_track_samples=args.min_track_samples,
        regional_stat=args.regional_stat,
        days_per_month=args.days_per_month,
    )

    if not args.no_extend_decomposition and extend_mass_decomposition_rows is not None:
        rows = extend_mass_decomposition_rows(rows)
        if not args.no_cumulative and add_cumulative_integrals is not None:
            rows = add_cumulative_integrals(rows)

    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} rows -> {args.out_csv}")
    _print_summary(rows, days_per_month=args.days_per_month)
    print(
        "[INFO] Intercompare with RACMO: plot_SWOT_PM_RACMO_SIT.py "
        f"--swot_prod_csv {args.out_csv} "
        f"(loader uses ice_production_m_month_area_mean when present; match "
        f"--swot_prod_days_per_month {args.days_per_month:g} if you use cm/day fallback). "
        "Optional: --swot_prod_label 'SWOT-IS2 production (regional track mean)'. "
        "Plot omits SWOT production <=0 by default; use --swot_prod_plot_signed for the signed series. "
        "Optional: --plot_two_panels for thickness vs production/area stacked subplots."
    )


if __name__ == "__main__":
    main()
