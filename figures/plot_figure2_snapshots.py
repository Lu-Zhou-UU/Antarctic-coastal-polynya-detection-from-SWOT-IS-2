#!/usr/bin/env python3
"""
Figure 2: Contrasting snapshot states of polynya mass budget.

This script auto-selects two representative center times from an Eulerian closure CSV:
- retention/accumulation case (high P_thermo)
- export/flushing case (low P_thermo)

For each case it plots:
1) reconstructed thickness field within AMSR2 polynya mask
2) coastal boundary-normal velocity (red outward, blue inward)
"""
import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drift_aware_mass_budget as damb

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover
    ccrs = None
    cfeature = None

def _pick_cases(df):
    """Pick one positive and one negative residual case with decent support."""
    x = df.copy()
    for c in [
        "P_thermo_kg_s",
        "coverage_fraction",
        "boundary_flux_coverage_fraction",
        "n_window_scenes_used",
    ]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    # Strong support-aware filter first.
    m_sup = np.isfinite(x["P_thermo_kg_s"])
    if "boundary_flux_coverage_fraction" in x.columns:
        m_sup &= np.isfinite(x["boundary_flux_coverage_fraction"]) & (x["boundary_flux_coverage_fraction"] >= 0.015)
    if "coverage_fraction" in x.columns:
        m_sup &= np.isfinite(x["coverage_fraction"]) & (x["coverage_fraction"] >= 5e-5)
    if "n_window_scenes_used" in x.columns:
        m_sup &= np.isfinite(x["n_window_scenes_used"]) & (x["n_window_scenes_used"] >= 3)
    xs = x.loc[m_sup].copy()
    # Fallback to median boundary coverage only if strict criteria leave nothing.
    if xs.empty:
        if "boundary_flux_coverage_fraction" in x.columns:
            med_b = np.nanmedian(x["boundary_flux_coverage_fraction"])
            xs = x.loc[
                np.isfinite(x["P_thermo_kg_s"])
                & np.isfinite(x["boundary_flux_coverage_fraction"])
                & (x["boundary_flux_coverage_fraction"] >= med_b)
            ].copy()
        if xs.empty:
            xs = x.loc[np.isfinite(x["P_thermo_kg_s"])].copy()

    p = pd.to_numeric(xs["P_thermo_kg_s"], errors="coerce").to_numpy(dtype=float)
    if p.size == 0 or not np.any(np.isfinite(p)):
        raise RuntimeError("No finite P_thermo_kg_s values found to pick snapshot cases.")
    pos = xs.iloc[int(np.nanargmax(p))]
    neg = xs.iloc[int(np.nanargmin(p))]
    return pos, neg


def _reconstruct_snapshot(
    t_center,
    segment_npz,
    amsr_mat,
    ice_motion_nc,
    thickness_reconstruct,
    fusion_stat,
    fusion_max_dist_km,
    center_fill_max_dist_km,
    backtrack_days,
    forwardtrack_days,
    step_hours,
    ice_motion_subsample,
    polynya_threshold=0.7,
    polynya_type="coastal",
    region_lon_min=None,
    region_lon_max=None,
    region_lat_min=None,
    region_lat_max=None,
):
    """Reconstruct thickness map and boundary normal transport for one center time."""
    times, lon_seg, lat_seg, h_seg = damb._load_segment_snapshot_npz(segment_npz)
    i = damb.nearest_time_index(times, t_center)
    t = times[i]

    lat_m, lon_m, mask_full = damb.load_polynya_mask_for_date(
        amsr_mat, t, polynya_threshold=polynya_threshold, polynya_type=polynya_type
    )
    mask_m = damb._apply_region_bbox_mask(
        mask_full,
        lon_m,
        lat_m,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
    )
    area_cell = damb.grid_cell_area_m2(lat_m, lon_m)
    idx_mask = np.where(mask_m & np.isfinite(area_cell))
    lon_q = lon_m[idx_mask]
    lat_q = lat_m[idx_mask]

    lo = np.asarray(lon_seg[i, :], dtype=float).ravel()
    la = np.asarray(lat_seg[i, :], dtype=float).ravel()
    hh = np.asarray(h_seg[i, :], dtype=float).ravel()
    m = np.isfinite(lo) & np.isfinite(la) & np.isfinite(hh)

    h_center = np.full(lon_q.shape, np.nan, dtype=float)
    if np.count_nonzero(m) >= 3:
        h_center = damb.interpolate_h_to_points(
            lo[m],
            la[m],
            hh[m],
            lon_q,
            lat_q,
            nearest_fill_max_dist_km=center_fill_max_dist_km
            if thickness_reconstruct == "tracked_fusion"
            else None,
        )

    layers = []
    if np.any(np.isfinite(h_center)):
        layers.append(h_center)

    if thickness_reconstruct == "tracked_fusion":
        if backtrack_days is None:
            t_back_target = times[max(i - 1, 0)]
        else:
            t_back_target = t - timedelta(days=float(backtrack_days))
        if forwardtrack_days is None:
            t_fwd_target = times[min(i + 1, len(times) - 1)]
        else:
            t_fwd_target = t + timedelta(days=float(forwardtrack_days))

        win_idx = [
            k
            for k, tk in enumerate(times)
            if (k != i) and (tk >= t_back_target) and (tk <= t_fwd_target)
        ]
        for k in win_idx:
            lon_k, lat_k = damb._advect_points_between_times(
                lon_q,
                lat_q,
                t_start=t,
                t_end=times[k],
                ice_motion_nc=ice_motion_nc,
                step_hours=step_hours,
                amsr_mat=amsr_mat,
                fill_nan_velocity_zero_on_ocean=True,
                ice_motion_subsample=ice_motion_subsample,
            )
            h_k = damb._nearest_sample_points(
                lon_seg[k, :],
                lat_seg[k, :],
                h_seg[k, :],
                lon_k,
                lat_k,
                max_dist_km=fusion_max_dist_km,
            )
            if np.any(np.isfinite(h_k)):
                layers.append(h_k)

    if len(layers) == 0:
        h_use_mask = np.full(lon_q.shape, np.nan, dtype=float)
    else:
        stk = np.stack(layers, axis=0)
        h_mean, h_med, _, _, _, _ = damb._stack_reduce_stats(stk)
        h_use_mask = h_mean if fusion_stat == "mean" else h_med

    h_use = np.full_like(mask_m, np.nan, dtype=float)
    h_use[idx_mask] = h_use_mask

    # Boundary normal transport for map overlay.
    # Use coast-adjacent boundary (not outer sea-ice edge) in selected region.
    bnd_all = damb.mask_boundary_4n(mask_full) & mask_m
    land_m = damb._land_mask_for_date(amsr_mat, t)
    bnd = bnd_all & damb._neighbors4(land_m)
    if not np.any(bnd):
        bnd = bnd_all
    gy, gx = np.gradient(mask_full.astype(float))
    norm = np.sqrt(gx**2 + gy**2)
    nx = np.divide(gx, norm, out=np.zeros_like(gx), where=norm > 0)
    ny = np.divide(gy, norm, out=np.zeros_like(gy), where=norm > 0)
    idx_bnd = np.where(bnd)
    lon_b = lon_m[idx_bnd]
    lat_b = lat_m[idx_bnd]

    lat_i, lon_i, u_i, v_i, valid_i = damb._load_ice_motion_cached(
        ice_motion_nc, t, subsample=ice_motion_subsample
    )
    u_b, v_b = damb._velocity_at_points(
        lon_i, lat_i, u_i, v_i, valid_i, q_lon=lon_b, q_lat=lat_b
    )
    un_b = u_b * nx[idx_bnd] + v_b * ny[idx_bnd]

    return {
        "time_center": t,
        "lon_m": lon_m,
        "lat_m": lat_m,
        "mask_m": mask_m,
        "h_use": h_use,
        "bnd_flux": bnd,
        "lon_b": lon_b,
        "lat_b": lat_b,
        "un_b": un_b,
    }


def _fmt_e(x):
    x = pd.to_numeric(x, errors="coerce")
    return "nan" if not np.isfinite(x) else f"{float(x):.2e}"


def _shared_limits_thickness(snaps, qlo=2.0, qhi=98.0):
    vals = []
    for s in snaps:
        x = np.asarray(s["h_use"], dtype=float)
        m = np.asarray(s["mask_m"], dtype=bool) & np.isfinite(x)
        if np.any(m):
            vals.append(x[m])
    if len(vals) == 0:
        return 0.0, 1.0
    z = np.concatenate(vals)
    vmin = float(np.nanpercentile(z, qlo))
    vmax = float(np.nanpercentile(z, qhi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, max(float(np.nanmax(z)), 0.1)
    return vmin, vmax


def _shared_limits_un(snaps, q=95.0):
    vals = []
    for s in snaps:
        x = np.asarray(s["un_b"], dtype=float)
        m = np.isfinite(x)
        if np.any(m):
            vals.append(np.abs(x[m]))
    if len(vals) == 0:
        return 1.0
    z = np.concatenate(vals)
    vmax = float(np.nanpercentile(z, q))
    return max(vmax, 1e-6)


def _plot_case(
    ax_h,
    ax_u,
    snap,
    row_label,
    row_stats,
    h_limits,
    un_vmax,
    region_extent=None,
    panel_labels=("a", "b"),
):
    """Plot one row (thickness + coastal boundary-normal velocity)."""
    if ccrs is None:
        raise RuntimeError(
            "Cartopy is required for polar projection plotting. "
            "Install with: pip install cartopy"
        )

    lon = snap["lon_m"]
    lat = snap["lat_m"]
    h = snap["h_use"]
    mask = snap["mask_m"]

    src_crs = ccrs.PlateCarree()
    # Robust rendering on curvilinear grids: scatter only valid mask points.
    mm = mask & np.isfinite(h) & np.isfinite(lon) & np.isfinite(lat)
    vmin_h, vmax_h = h_limits

    if np.any(mm):
        pc = ax_h.scatter(
            lon[mm], lat[mm], c=h[mm], s=28, cmap="viridis",
            vmin=vmin_h, vmax=vmax_h,
            linewidths=0.0, edgecolors="none", alpha=0.96,
            rasterized=True, transform=src_crs, zorder=3,
        )
    else:
        pc = ax_h.scatter([], [], c=[], cmap="viridis")
    # Coast-adjacent flux boundary: high-contrast outline (not tiny black dots).
    bnd = np.asarray(snap.get("bnd_flux", damb.mask_boundary_4n(mask)), dtype=bool)
    mb = bnd & np.isfinite(lon) & np.isfinite(lat)
    if np.any(mb):
        ax_h.scatter(
            lon[mb], lat[mb], s=14, facecolors="none", edgecolors="orangered",
            linewidths=1.1, rasterized=True, transform=src_crs, zorder=5,
        )
    ax_h.set_title(f"({panel_labels[0]}) {row_label}: thickness", fontsize=15, pad=6)
    try:
        ax_h.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    except Exception:
        pass
    ax_h.coastlines(linewidth=0.9, edgecolor="0.35", zorder=2)
    ax_h.add_feature(cfeature.LAND, facecolor="0.88", edgecolor="0.45", linewidth=0.4, zorder=1)
    gl = ax_h.gridlines(
        crs=src_crs, draw_labels=False,
        linewidth=0.45, color="0.45", alpha=0.55, linestyle=":",
    )
    _ = gl
    if region_extent is not None:
        ax_h.set_extent(region_extent, crs=src_crs)
    else:
        ax_h.set_extent([0, 360, -85, -40], crs=src_crs)

    un = snap["un_b"]
    vmax = float(un_vmax)
    sc = ax_u.scatter(
        snap["lon_b"],
        snap["lat_b"],
        c=un,
        s=36,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.2,
        edgecolors="0.2",
        alpha=0.95,
        transform=src_crs,
        zorder=4,
    )
    if np.any(mb):
        ax_u.scatter(
            lon[mb], lat[mb], s=12, facecolors="none", edgecolors="orangered",
            linewidths=1.0, rasterized=True, transform=src_crs, zorder=5,
        )
    ax_u.set_title(
        f"({panel_labels[1]}) {row_label}: coastal boundary-normal velocity",
        fontsize=15,
        pad=6,
    )
    try:
        ax_u.add_feature(cfeature.OCEAN, facecolor="#d6e8f5", zorder=0)
    except Exception:
        pass
    ax_u.coastlines(linewidth=0.9, edgecolor="0.35", zorder=2)
    ax_u.add_feature(cfeature.LAND, facecolor="0.88", edgecolor="0.45", linewidth=0.4, zorder=1)
    gl2 = ax_u.gridlines(
        crs=src_crs, draw_labels=False,
        linewidth=0.45, color="0.45", alpha=0.55, linestyle=":",
    )
    _ = gl2
    if region_extent is not None:
        ax_u.set_extent(region_extent, crs=src_crs)
    else:
        ax_u.set_extent([0, 360, -85, -40], crs=src_crs)

    ttxt = pd.to_datetime(row_stats["time_center"]).strftime("%Y-%m-%d %H:%M")
    txt = (
        f"{ttxt}\n"
        f"dM/dt={_fmt_e(row_stats.get('dMdt_kg_s'))}\n"
        f"Fout={_fmt_e(row_stats.get('Fout_kg_s'))}\n"
        f"Pth={_fmt_e(row_stats.get('P_thermo_kg_s'))}"
    )
    ax_u.text(
        0.02,
        0.98,
        txt,
        transform=ax_u.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.7", linewidth=0.5),
    )
    return pc, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Eulerian closure CSV")
    ap.add_argument("--segment_npz", required=True, help="Segment snapshots NPZ or directory")
    ap.add_argument("--amsr_mat", required=True, help="AMSR2 MAT file")
    ap.add_argument("--ice_motion_nc", required=True, help="Ice motion NetCDF")
    ap.add_argument("--out", default="figure2_snapshot_contrast.png", help="Output figure path")
    ap.add_argument("--thickness_reconstruct", choices=["none", "tracked_fusion"], default="tracked_fusion")
    ap.add_argument("--fusion_stat", choices=["median", "mean"], default="median")
    ap.add_argument("--fusion_max_dist_km", type=float, default=100.0)
    ap.add_argument("--center_fill_max_dist_km", type=float, default=None)
    ap.add_argument("--backtrack_days", type=float, default=3.0)
    ap.add_argument("--forwardtrack_days", type=float, default=3.0)
    ap.add_argument("--segment_step_hours", type=float, default=24.0)
    ap.add_argument("--ice_motion_subsample", type=int, default=10)
    ap.add_argument("--polynya_threshold", type=float, default=0.7)
    ap.add_argument("--polynya_type", choices=["open", "coastal", "both"], default="coastal")
    ap.add_argument("--region_lon_min", type=float, default=None)
    ap.add_argument("--region_lon_max", type=float, default=None)
    ap.add_argument("--region_lat_min", type=float, default=None)
    ap.add_argument("--region_lat_max", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["time_center"]).sort_values("time_center")
    for c in ["P_thermo_kg_s", "dMdt_kg_s", "Fout_kg_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pos, neg = _pick_cases(df)
    snap_pos = _reconstruct_snapshot(
        pd.to_datetime(pos["time_center"]).to_pydatetime(),
        segment_npz=args.segment_npz,
        amsr_mat=args.amsr_mat,
        ice_motion_nc=args.ice_motion_nc,
        thickness_reconstruct=args.thickness_reconstruct,
        fusion_stat=args.fusion_stat,
        fusion_max_dist_km=args.fusion_max_dist_km,
        center_fill_max_dist_km=args.center_fill_max_dist_km,
        backtrack_days=args.backtrack_days,
        forwardtrack_days=args.forwardtrack_days,
        step_hours=args.segment_step_hours,
        ice_motion_subsample=args.ice_motion_subsample,
        polynya_threshold=args.polynya_threshold,
        polynya_type=args.polynya_type,
        region_lon_min=args.region_lon_min,
        region_lon_max=args.region_lon_max,
        region_lat_min=args.region_lat_min,
        region_lat_max=args.region_lat_max,
    )
    snap_neg = _reconstruct_snapshot(
        pd.to_datetime(neg["time_center"]).to_pydatetime(),
        segment_npz=args.segment_npz,
        amsr_mat=args.amsr_mat,
        ice_motion_nc=args.ice_motion_nc,
        thickness_reconstruct=args.thickness_reconstruct,
        fusion_stat=args.fusion_stat,
        fusion_max_dist_km=args.fusion_max_dist_km,
        center_fill_max_dist_km=args.center_fill_max_dist_km,
        backtrack_days=args.backtrack_days,
        forwardtrack_days=args.forwardtrack_days,
        step_hours=args.segment_step_hours,
        ice_motion_subsample=args.ice_motion_subsample,
        polynya_threshold=args.polynya_threshold,
        polynya_type=args.polynya_type,
        region_lon_min=args.region_lon_min,
        region_lon_max=args.region_lon_max,
        region_lat_min=args.region_lat_min,
        region_lat_max=args.region_lat_max,
    )
    # Default to tighter Ross Sea crop when user does not pass region bounds.
    if (
        args.region_lon_min is None
        and args.region_lon_max is None
        and args.region_lat_min is None
        and args.region_lat_max is None
    ):
        args.region_lon_min = 155.0
        args.region_lon_max = 235.0
        args.region_lat_min = -79.8
        args.region_lat_max = -70.0
    region_extent = [
        float(args.region_lon_min),
        float(args.region_lon_max),
        float(args.region_lat_min),
        float(args.region_lat_max),
    ]

    if ccrs is None:
        raise RuntimeError(
            "Cartopy not available. Install with `pip install cartopy` to make polar Figure 2."
        )
    # Center stereographic view on the regional box (reduces distortion vs default lon=0).
    if region_extent is not None:
        clon = 0.5 * (float(region_extent[0]) + float(region_extent[1])) % 360.0
    else:
        clon = 180.0
    proj = ccrs.SouthPolarStereo(central_longitude=float(clon))
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.titleweight": "semibold",
    })
    fig, axs = plt.subplots(
        2, 2, figsize=(14.0, 10.5), constrained_layout=True,
        subplot_kw={"projection": proj}
    )
    h_limits = _shared_limits_thickness([snap_pos, snap_neg])
    un_vmax = _shared_limits_un([snap_pos, snap_neg])
    pc1, sc1 = _plot_case(
        axs[0, 0], axs[0, 1], snap_pos, "Accumulation/retention case", pos,
        h_limits=h_limits, un_vmax=un_vmax,
        region_extent=region_extent, panel_labels=("a", "b"),
    )
    pc2, sc2 = _plot_case(
        axs[1, 0], axs[1, 1], snap_neg, "Export/flushing case", neg,
        h_limits=h_limits, un_vmax=un_vmax,
        region_extent=region_extent, panel_labels=("c", "d"),
    )
    cbar_h = fig.colorbar(pc1, ax=[axs[0, 0], axs[1, 0]], fraction=0.03, pad=0.02, shrink=0.98)
    cbar_h.set_label("SIT (m)", fontsize=15)
    cbar_h.ax.tick_params(labelsize=9)
    cbar_u = fig.colorbar(sc1, ax=[axs[0, 1], axs[1, 1]], fraction=0.03, pad=0.02, shrink=0.98)
    cbar_u.set_label(r"$u_n$ (m s$^{-1}$)", fontsize=15)
    cbar_u.ax.tick_params(labelsize=9)
    # fig.suptitle(
    #     "Figure 2: Contrasting snapshot states (coastal polynya; orange ring = coastal flux boundary)",
    #     fontsize=13,
    #     y=1.01,
    # )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[INFO] Wrote figure: {out_path}")


if __name__ == "__main__":
    main()

