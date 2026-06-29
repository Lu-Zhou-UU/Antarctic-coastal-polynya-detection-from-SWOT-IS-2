#!/usr/bin/env python3
"""
plot_zoom_tracking_clean.py

A cleaner, publication-quality zoom-in plot of ICESat-2 (IS-2) measurement
locations BEFORE and AFTER ice-drift tracking.

It reuses all the loaders/helpers already defined in `plot_esacci_sit_map.py`,
so just keep this file next to that script and run it with the same NPZ /
AMSR / BedMachine / ice-motion inputs.

What it draws (per the requested style):
  - BEFORE positions (IS-2 measured)  -> blue circles
  - AFTER  positions (after tracking) -> red  triangles
  - TRUE-SCALE displacement arrows connecting each before -> after point
    (no exaggeration; a scale bar + median-offset note make them readable)
  - Optional context: coastal polynya dots, BedMachine ice shelves,
    ice-motion vectors, coastline/land, labelled gridlines.

Example
-------
python plot_zoom_tracking_clean.py \
  --tracking_npz trajectories_backward.npz \
  --amsr_mat AMSR2_2023.mat \
  --date 20231028 \
  --bedmachine_nc NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc \
  --show_ice_shelves \
  --ice_motion_nc /Volumes/Yotta_1/Antarctic/Ice_Drift/icemotion_daily_sh_25km_20230101_20231231_v4.1.nc \
  --ice_motion_subsample 2 \
  --out_png sit_map_20231028_zoom.png \
  --debug
"""

import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Reuse everything from your main script (must be in the same folder / PYTHONPATH)
from plot_esacci_sit_map import (
    load_tracking_positions,
    normalize_longitude_0_360,
    wrap_lon180,
    load_mat_any,
    polynya_output_amsr2,
    load_bedmachine_mask,
    load_ice_motion,
    displace_lonlat,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
})


def _nice_round(x):
    """Round x down to a 'nice' 1/2/5 * 10^n number (for the scale bar)."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = 10 ** exp
    for m in (5, 2, 1):
        if x >= m * base:
            return m * base
    return base


def _median_offset_km(lon0, lat0, lon1, lat1):
    """Median great-circle-ish offset (km) between two sets of lon/lat points."""
    latm = np.deg2rad((lat0 + lat1) / 2.0)
    dx = (lon1 - lon0) * 111320.0 * np.cos(latm)
    dy = (lat1 - lat0) * 111320.0
    return float(np.nanmedian(np.hypot(dx, dy)) / 1000.0)


def plot_zoom_tracking_clean(
    tracking_npz,
    out_png,
    lat_range=(-78, -76),
    lon_range=(165, 190),
    target_points=60,
    # context layers
    amsr_mat=None,
    target_date=None,
    polynya_threshold=0.7,
    polynya_dot_size=85,
    polynya_dot_stride=1,
    polynya_color="#EBB6AE",     # soft muted coral
    bedmachine_nc=None,
    show_ice_shelves=True,
    ice_motion_nc=None,
    ice_motion_subsample=1,
    ice_motion_arrow_gain=3.0,
    ice_motion_ref_speed=0.10,   # reference-vector speed (m/s); None -> use median
    # before/after styling (colorblind-friendly blue vs vermillion)
    before_color="#0072B2",   # blue   open square  = start (before tracking)
    after_color="#D55E00",    # vermillion diamond  = end   (after tracking)
    arrow_color="#333333",
    before_label="Start positions",
    after_label="End positions",
    point_size=130,
    displacement_gain=25.0,   # exaggerate start->end arrows (1.0 = true scale)
    rotate_180=True,
    debug=False,
):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.lines as mlines

    if not os.path.isfile(tracking_npz):
        raise FileNotFoundError(f"Tracking NPZ not found: {tracking_npz}")

    # ---- load + region-filter + subsample tracking positions ----
    td = load_tracking_positions(tracking_npz, debug=debug)
    lat_b = td["lat_start"]        # before tracking
    lon_b = td["lon_start"]
    lat_a = td["lat_final"]        # after tracking
    lon_a = td["lon_final"]
    valid = td["valid"]

    lon_b_plot = wrap_lon180(normalize_longitude_0_360(lon_b))
    lon_a_plot = wrap_lon180(normalize_longitude_0_360(lon_a))
    lon_b_0360 = normalize_longitude_0_360(lon_b)
    lon_a_0360 = normalize_longitude_0_360(lon_a)

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    region = (
        (lat_b >= lat_min) & (lat_b <= lat_max) &
        (lon_b_0360 >= lon_min) & (lon_b_0360 <= lon_max) &
        (lat_a >= lat_min) & (lat_a <= lat_max) &
        (lon_a_0360 >= lon_min) & (lon_a_0360 <= lon_max)
    )
    m = valid & region & np.isfinite(lat_b) & np.isfinite(lon_b_plot) & \
        np.isfinite(lat_a) & np.isfinite(lon_a_plot)

    n_valid = int(np.sum(m))
    if n_valid == 0:
        raise RuntimeError(
            f"No valid points in region lat={lat_range}, lon={lon_range}"
        )

    # even subsample so arrows don't overplot
    if n_valid > target_points:
        stride = max(1, n_valid // target_points)
        idx = np.where(m)[0][::stride]
        m = np.zeros_like(m, dtype=bool)
        m[idx] = True
        if debug:
            print(f"[ZOOM] Subsample {n_valid} -> {int(np.sum(m))} (stride={stride})")

    lon_bs, lat_bs = lon_b_plot[m], lat_b[m]
    lon_as, lat_as = lon_a_plot[m], lat_a[m]

    median_km = _median_offset_km(lon_bs, lat_bs, lon_as, lat_as)
    if debug:
        print(f"[ZOOM] Plotting {lon_bs.size} before/after pairs; "
              f"median offset ~ {median_km:.2f} km")

    # ---- projection + figure ----
    proj = ccrs.SouthPolarStereo(central_longitude=0.0)
    pc = ccrs.PlateCarree()

    fig = plt.figure(figsize=(11, 9), facecolor="white")
    ax = plt.axes(projection=proj)

    # extent from the before/after points (project, pad 12%)
    lon_all = np.concatenate([lon_bs, lon_as])
    lat_all = np.concatenate([lat_bs, lat_as])
    pts = proj.transform_points(
        pc,
        np.array([lon_all.min(), lon_all.max(), lon_all.min(), lon_all.max()]),
        np.array([lat_all.min(), lat_all.min(), lat_all.max(), lat_all.max()]),
    )
    x, y = pts[:, 0], pts[:, 1]
    xspan = max(np.ptp(x), 1.0) * 1.12
    yspan = max(np.ptp(y), 1.0) * 1.12
    span = max(xspan, yspan)  # keep square so true-scale arrows aren't distorted
    xc, yc = np.mean(x), np.mean(y)
    extent = (xc - span / 2, xc + span / 2, yc - span / 2, yc + span / 2)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")

    # ---- base map ----
    ax.add_feature(cfeature.OCEAN, facecolor="#eef4fb", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.45",
                   linewidth=0.4, zorder=1)
    ax.coastlines(linewidth=0.7, color="0.35", zorder=2)
    gl = ax.gridlines(draw_labels=True, alpha=0.4, linestyle="--",
                      linewidth=0.6, color="0.5", zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}

    # ---- context: BedMachine ice shelves ----
    if show_ice_shelves and bedmachine_nc and os.path.isfile(bedmachine_nc):
        try:
            bm = load_bedmachine_mask(
                bedmachine_nc, lon_data=lon_all, lat_data=lat_all,
                buffer_km=200, debug=debug,
            )
            bm_lon = wrap_lon180(bm["lon"])
            bm_lat = bm["lat"]
            bm_0360 = normalize_longitude_0_360(bm["lon"])
            reg = ((bm_lat >= lat_min) & (bm_lat <= lat_max) &
                   (bm_0360 >= lon_min) & (bm_0360 <= lon_max))
            shelf = bm["ice_shelf_mask"] & reg
            if np.any(shelf):
                ma = np.ma.masked_array(shelf.astype(float), mask=~shelf)
                ax.pcolormesh(bm_lon, bm_lat, ma, transform=pc, shading="auto",
                              cmap="gray", alpha=0.55, vmin=0, vmax=3,
                              zorder=3, edgecolors="none")
        except Exception as e:
            if debug:
                print(f"[ZOOM] ice shelves skipped: {e}")

    # ---- context: coastal polynya dots ----
    polynya_plotted = False
    if amsr_mat and os.path.isfile(amsr_mat):
        try:
            amsr = load_mat_any(amsr_mat)
            sic = np.array(amsr["sic"], dtype=float)
            lat_amsr = np.array(amsr["lat"], dtype=float)
            lon_amsr = np.array(amsr["lon"], dtype=float)
            if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
                sic = np.moveaxis(sic, -1, 0)
            if lat_amsr.shape != sic.shape[1:] and lat_amsr.T.shape == sic.shape[1:]:
                lat_amsr, lon_amsr = lat_amsr.T, lon_amsr.T
            if lat_amsr.ndim == 1 and lon_amsr.ndim == 1 and sic.ndim == 2:
                lon_amsr, lat_amsr = np.meshgrid(lon_amsr, lat_amsr)
            lon_amsr = normalize_longitude_0_360(lon_amsr)
            lon_amsr_plot = wrap_lon180(lon_amsr)

            td_date = target_date or datetime(2023, 10, 28)
            doy = min(td_date.timetuple().tm_yday, sic.shape[0]) - 1
            sic_daily = sic[doy] if sic.ndim == 3 else sic
            sic_frac = sic_daily / 100.0 if np.nanmax(sic_daily) > 1.0 else sic_daily

            _, coastal = polynya_output_amsr2(sic_frac, polynya_threshold)
            cmask = ~np.isnan(coastal) if coastal is not None else (sic_frac < polynya_threshold)
            cmask = cmask & (lat_amsr >= lat_min) & (lat_amsr <= lat_max) & \
                    (lon_amsr >= lon_min) & (lon_amsr <= lon_max)
            stride = max(1, int(polynya_dot_stride))
            if stride > 1:
                samp = np.zeros_like(cmask, dtype=bool)
                samp[::stride, ::stride] = True
                cmask = cmask & samp
            yy, xx = np.where(cmask)
            ld, ad = lon_amsr_plot[yy, xx], lat_amsr[yy, xx]
            good = np.isfinite(ld) & np.isfinite(ad)
            if np.any(good):
                ax.scatter(ld[good], ad[good], s=polynya_dot_size, c=polynya_color,
                           marker="o", linewidths=0, alpha=0.65, transform=pc,
                           zorder=4, label="Coastal polynya")
                polynya_plotted = True
        except Exception as e:
            if debug:
                print(f"[ZOOM] polynya skipped: {e}")

    # ---- context: ice-motion vectors (dense) + reference vector ----
    if ice_motion_nc and os.path.isfile(ice_motion_nc) and target_date is not None:
        try:
            lat_i, lon_i, ue, vn, vi = load_ice_motion(
                ice_motion_nc, target_date, subsample=ice_motion_subsample, debug=debug)
            lon_i0360 = normalize_longitude_0_360(lon_i)
            reg = ((lat_i >= lat_min) & (lat_i <= lat_max) &
                   (lon_i0360 >= lon_min) & (lon_i0360 <= lon_max))
            vi = vi & reg
            if np.any(vi):
                lo = wrap_lon180(lon_i[vi]); la = lat_i[vi]
                u0 = ue[vi]; v0 = vn[vi]
                speed0 = np.hypot(u0, v0)
                if np.nanmedian(speed0) > 2.0:  # cm/s guard
                    u0, v0 = u0 / 100.0, v0 / 100.0
                    speed0 = np.hypot(u0, v0)
                dt_sec = 86400.0
                lo1, la1 = displace_lonlat(lo, la, u0, v0, dt_sec)
                p0 = proj.transform_points(pc, lo, la)
                p1 = proj.transform_points(pc, lo1, la1)
                x0, y0 = p0[:, 0], p0[:, 1]
                dx = (p1[:, 0] - x0) * ice_motion_arrow_gain
                dy = (p1[:, 1] - y0) * ice_motion_arrow_gain
                g = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(dx) & np.isfinite(dy)
                if np.any(g):
                    q = ax.quiver(x0[g], y0[g], dx[g], dy[g], transform=proj,
                                  angles="xy", scale_units="xy", scale=1.0,
                                  width=0.0058, headwidth=4.2, headlength=4.6,
                                  headaxislength=4.0, color="0.5", alpha=0.7, zorder=5)
                    # reference vector (e.g. "0.10 m/s")
                    rs = ice_motion_ref_speed
                    if rs is None or not np.isfinite(rs) or rs <= 0:
                        rs = float(np.nanmedian(speed0[g]))
                        rs = round(max(rs, 0.05), 2) if np.isfinite(rs) else 0.10
                    ref_disp = rs * dt_sec * ice_motion_arrow_gain
                    ax.quiverkey(q, 0.12, 0.045, ref_disp, f"{rs:.2f} m/s",
                                 labelpos="E", coordinates="axes",
                                 fontproperties={"size": 13}, color="k")
        except Exception as e:
            if debug:
                print(f"[ZOOM] ice motion skipped: {e}")

    # ---- start -> end displacement arrows (exaggerated for visibility) ----
    pb = proj.transform_points(pc, lon_bs, lat_bs)
    pa = proj.transform_points(pc, lon_as, lat_as)
    xb, yb = pb[:, 0], pb[:, 1]
    dxa, dya = pa[:, 0] - xb, pa[:, 1] - yb
    g = np.isfinite(xb) & np.isfinite(yb) & np.isfinite(dxa) & np.isfinite(dya)

    gain_d = float(displacement_gain) if displacement_gain and displacement_gain > 0 else 1.0
    xe = xb + dxa * gain_d   # exaggerated end (projection metres)
    ye = yb + dya * gain_d

    if np.any(g):
        ax.quiver(
            xb[g], yb[g], (xe - xb)[g], (ye - yb)[g],
            transform=proj, angles="xy", scale_units="xy", scale=1.0,
            width=0.0030, headwidth=4.2, headlength=4.6, headaxislength=4.0,
            color="#6d4c91", alpha=0.85, zorder=7,
        )

    # markers: open blue square at TRUE start; filled vermillion diamond at the
    # (exaggerated) end -- both angular so they don't read as polynya dots.
    ax.scatter(lon_bs, lat_bs, s=point_size, marker="s",
               facecolors="none", edgecolors=before_color, linewidths=1.8,
               alpha=0.95, transform=pc, zorder=10, label=before_label)
    ax.scatter(xe[g], ye[g], s=point_size * 0.62, marker="D",
               facecolors=after_color, edgecolors="white", linewidths=0.6,
               alpha=0.95, transform=proj, zorder=12, label=after_label)

    # ---- optional 180-degree view flip (keeps your original orientation) ----
    if rotate_180:
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])

    # ---- scale bar (true scale reference) ----
    bar_km = _nice_round(span / 1000.0 / 5.0)  # ~1/5 of view, rounded nice
    try:
        sb = AnchoredSizeBar(
            ax.transData, bar_km * 1000.0, f"{bar_km:g} km",
            loc="lower right", pad=0.5, borderpad=0.8, sep=5,
            frameon=True, size_vertical=span * 0.004, color="black",
        )
        sb.patch.set_alpha(0.85)
        ax.add_artist(sb)
    except Exception as e:
        if debug:
            print(f"[ZOOM] scale bar skipped: {e}")

    # ---- legend (Coastal polynya / Start / End -- all markers same size so
    #      they line up; no displacement entry) ----
    ms = 13
    handles = []
    if polynya_plotted:
        handles.append(
            mlines.Line2D([], [], marker="o", linestyle="None",
                          markerfacecolor=polynya_color, markeredgecolor="none",
                          markersize=ms, label="Coastal polynya"))
    handles += [
        mlines.Line2D([], [], marker="s", linestyle="None",
                      markerfacecolor="none", markeredgecolor=before_color,
                      markeredgewidth=1.8, markersize=ms, label=before_label),
        mlines.Line2D([], [], marker="D", linestyle="None",
                      markerfacecolor=after_color, markeredgecolor="white",
                      markersize=ms - 3, label=after_label),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.92, fontsize=13,
              handletextpad=0.6, labelspacing=0.8, borderpad=0.7)

    # caption: exaggeration factor + true median offset
    if gain_d > 1.0:
        ax.text(
            0.985, 0.985,
            f"Start→End arrows exaggerated ×{gain_d:g}\n"
            f"(true median offset ≈ {median_km:.2f} km)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.88, edgecolor="0.7"), zorder=20,
        )

    plt.savefig(out_png, dpi=300, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.05)
    print(f"[ZOOM] Saved: {out_png}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Clean zoom plot of IS-2 before/after tracking.")
    ap.add_argument("--tracking_npz", required=True)
    ap.add_argument("--out_png", default="zoom_tracking_clean.png")
    ap.add_argument("--date", default=None, help="YYYYMMDD (for polynya/ice-motion day)")
    ap.add_argument("--lat_min", type=float, default=-78.0)
    ap.add_argument("--lat_max", type=float, default=-76.0)
    ap.add_argument("--lon_min", type=float, default=165.0)
    ap.add_argument("--lon_max", type=float, default=190.0)
    ap.add_argument("--target_points", type=int, default=60)
    ap.add_argument("--amsr_mat", default=None)
    ap.add_argument("--polynya_threshold", type=float, default=0.7)
    ap.add_argument("--polynya_dot_stride", type=int, default=1)
    ap.add_argument("--polynya_dot_size", type=float, default=85)
    ap.add_argument("--bedmachine_nc", default=None)
    ap.add_argument("--show_ice_shelves", action="store_true")
    ap.add_argument("--ice_motion_nc", default=None)
    ap.add_argument("--ice_motion_subsample", type=int, default=1)
    ap.add_argument("--ice_motion_arrow_gain", type=float, default=3.0)
    ap.add_argument("--ice_motion_ref_speed", type=float, default=0.10,
                    help="Reference-vector speed in m/s (default 0.10; <=0 -> use median)")
    ap.add_argument("--point_size", type=float, default=110, help="Start/End marker size")
    ap.add_argument("--displacement_gain", type=float, default=25.0,
                    help="Exaggerate start->end arrows (1.0 = true scale; default 25)")
    ap.add_argument("--no_rotate", action="store_true", help="Disable 180-degree view flip")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y%m%d")

    plot_zoom_tracking_clean(
        tracking_npz=args.tracking_npz,
        out_png=args.out_png,
        lat_range=(args.lat_min, args.lat_max),
        lon_range=(args.lon_min, args.lon_max),
        target_points=args.target_points,
        amsr_mat=args.amsr_mat,
        target_date=target_date,
        polynya_threshold=args.polynya_threshold,
        polynya_dot_stride=args.polynya_dot_stride,
        polynya_dot_size=args.polynya_dot_size,
        bedmachine_nc=args.bedmachine_nc,
        show_ice_shelves=args.show_ice_shelves,
        ice_motion_nc=args.ice_motion_nc,
        ice_motion_subsample=args.ice_motion_subsample,
        ice_motion_arrow_gain=args.ice_motion_arrow_gain,
        ice_motion_ref_speed=(args.ice_motion_ref_speed if args.ice_motion_ref_speed > 0 else None),
        point_size=args.point_size,
        displacement_gain=args.displacement_gain,
        rotate_180=not args.no_rotate,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
