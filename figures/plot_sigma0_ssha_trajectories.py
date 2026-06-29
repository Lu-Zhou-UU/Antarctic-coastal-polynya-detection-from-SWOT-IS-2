#!/usr/bin/env python3
"""
Read sigma0_ssha_trajectories.nc and plot (smoothed):
  - Top: combined mean sigma0 in dB (left y) and mean SSHA (right y) vs latitude
  - Bottom: approx. std sigma0 in dB (left y) and std SSHA (right y) vs latitude

Usage:
  python plot_sigma0_ssha_trajectories.py sigma0_ssha_trajectories.nc [--out figure.png] [--window 31]
"""

import argparse
import re
from datetime import datetime

import numpy as np

try:
    import netCDF4
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False


def smooth(y, window, poly=3):
    """Smooth 1D array: Savitzky-Golay if scipy available, else rolling mean. Preserve NaN."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    valid = np.isfinite(y)
    if not np.any(valid) or window < 2:
        return y
    n = int(np.sum(valid))
    if HAS_SAVGOL and n >= poly + 2:
        w = min(window, n)
        if w % 2 == 0:
            w -= 1
        if w < poly + 2:
            w = poly + 2 if (poly + 2) % 2 else poly + 3
        w = min(w, n)
        if w % 2 == 0:
            w -= 1
        if w >= 3:
            out[valid] = savgol_filter(y[valid], window_length=w, polyorder=min(poly, w - 1))
            return out
    # Fallback: rolling mean (nan-safe)
    half = window // 2
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        chunk = y[lo:hi]
        if np.any(np.isfinite(chunk)):
            out[i] = np.nanmean(chunk)
    return out


def haversine_distance_m(lon1_deg, lat1_deg, lon2_deg, lat2_deg):
    """Great-circle distance in meters. Inputs in degrees; can be arrays."""
    R = 6371000.0  # m
    lat1 = np.deg2rad(np.atleast_1d(lat1_deg))
    lon1 = np.deg2rad(np.atleast_1d(lon1_deg))
    lat2 = np.deg2rad(np.atleast_1d(lat2_deg))
    lon2 = np.deg2rad(np.atleast_1d(lon2_deg))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.minimum(np.sqrt(np.maximum(a, 0)), 1.0))
    return R * c


def filename_to_timeslot_label(basename):
    """
    Parse SWOT filename like *_20231028T215103_20231028T224230_* into a short label.
    Returns e.g. "28 Oct 21:51–22:42 UTC" or the pass number if parsing fails.
    """
    if not basename:
        return ""
    m = re.search(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_", basename)
    if not m:
        # Fallback: try to get cycle_pass e.g. 005_455 or 005_452
        m2 = re.search(r"_005_(\d+)_", basename)
        return m2.group(1) if m2 else basename[:20]
    try:
        t1 = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        t2 = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S")
        return "{} {:02d}:{:02d}–{:02d}:{:02d} UTC".format(
            t1.strftime("%d %b"), t1.hour, t1.minute, t2.hour, t2.minute
        )
    except Exception:
        return basename[:30]


def linear_sigma0_to_db(s0_linear):
    """Convert SWOT linear sigma0 (dimensionless) to decibels; non-positive -> NaN."""
    s0 = np.asarray(s0_linear, dtype=float)
    return 10.0 * np.log10(np.where(s0 > 0, s0, np.nan))


def linear_sigma0_relative_std_to_db(s0_mean_linear, s0_std_linear):
    """Approximate within-radius spread in dB from linear mean and std (sigma/mu)."""
    mean = np.asarray(s0_mean_linear, dtype=float)
    std = np.asarray(s0_std_linear, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where((mean > 0) & np.isfinite(std), std / mean, np.nan)
    return (10.0 / np.log(10.0)) * rel


def load_nc(nc_path):
    """Load trajectory sigma0/SSHA NetCDF. Return dict of 1D arrays (means and stds) and source file names."""
    if not HAS_NETCDF:
        raise RuntimeError("netCDF4 is required.")
    nc = netCDF4.Dataset(nc_path, "r")
    out = {}
    for key in ["lat_start", "lon_start", "lat_final", "lon_final",
                "sigma0_start", "ssha_start", "sigma0_end", "ssha_end",
                "sigma0_std_start", "ssha_std_start", "sigma0_std_end", "ssha_std_end"]:
        if key in nc.variables:
            out[key] = np.asarray(nc.variables[key][:]).ravel()
        else:
            out[key] = None
    out["_source_start_file"] = getattr(nc, "source_start_file", "")
    out["_source_end_file"] = getattr(nc, "source_end_file", "")
    nc.close()
    return out


def compute_trajectory_stats(data, lat_min=None, lat_max=None):
    """
    Compute statistics (unsmoothed) for discussion/paper: mean and std of difference
    between earlier pass (end) and later pass (start) for sigma0 and SSHA.

    sigma0 pass differences are reported in dB (10 log10 sigma0_end - 10 log10 sigma0_start
    per trajectory). Linear means are kept for reference/plot context only.

    Optional lat_min/lat_max restrict to trajectories whose later-pass latitude (lat_start)
    falls in [lat_min, lat_max] (useful for polynya-focused reporting).
    """
    if data.get("sigma0_start") is None or data.get("sigma0_end") is None:
        return {}
    sigma0_start = np.asarray(data["sigma0_start"], dtype=float).ravel()
    sigma0_end = np.asarray(data["sigma0_end"], dtype=float).ravel()
    lat_start = np.asarray(data["lat_start"], dtype=float).ravel()
    ssha_start = np.asarray(data["ssha_start"] if data.get("ssha_start") is not None else np.nan, dtype=float).ravel()
    ssha_end = np.asarray(data["ssha_end"] if data.get("ssha_end") is not None else np.nan, dtype=float).ravel()
    if sigma0_start.size != sigma0_end.size:
        return {}
    if ssha_start.size != sigma0_start.size:
        ssha_start = np.full_like(sigma0_start, np.nan)
        ssha_end = np.full_like(sigma0_end, np.nan)

    lat_mask = np.ones_like(lat_start, dtype=bool)
    if lat_min is not None:
        lat_mask &= lat_start >= lat_min
    if lat_max is not None:
        lat_mask &= lat_start <= lat_max

    valid_s0 = np.isfinite(sigma0_start) & np.isfinite(sigma0_end) & lat_mask
    valid_ssha = np.isfinite(ssha_start) & np.isfinite(ssha_end) & lat_mask
    n_valid_s0 = int(np.sum(valid_s0))
    n_valid_ssha = int(np.sum(valid_ssha))
    stats = {
        "n_valid": max(n_valid_s0, n_valid_ssha),
        "lat_min": lat_min,
        "lat_max": lat_max,
    }
    if n_valid_s0 == 0:
        return stats

    # sigma0: end = earlier pass (452), start = later pass (455); stored in linear units
    s0_start = sigma0_start[valid_s0]
    s0_end = sigma0_end[valid_s0]
    diff_s0_lin = s0_end - s0_start
    stats["mean_sigma0_start_lin"] = float(np.nanmean(s0_start))
    stats["mean_sigma0_end_lin"] = float(np.nanmean(s0_end))
    stats["mean_sigma0_diff_lin"] = float(np.nanmean(diff_s0_lin))
    stats["std_sigma0_diff_lin"] = float(np.nanstd(diff_s0_lin))

    s0_start_db = linear_sigma0_to_db(s0_start)
    s0_end_db = linear_sigma0_to_db(s0_end)
    valid_db = np.isfinite(s0_start_db) & np.isfinite(s0_end_db)
    diff_s0_db = s0_end_db[valid_db] - s0_start_db[valid_db]
    stats["mean_sigma0_end_db"] = float(np.nanmean(s0_end_db))
    stats["mean_sigma0_start_db"] = float(np.nanmean(s0_start_db))
    stats["mean_sigma0_diff_db"] = float(np.nanmean(diff_s0_db))
    stats["std_sigma0_diff_db"] = float(np.nanstd(diff_s0_db))
    stats["rms_sigma0_diff_db"] = float(np.sqrt(np.nanmean(diff_s0_db ** 2)))
    stats["median_sigma0_diff_db"] = float(np.nanmedian(diff_s0_db))
    stats["n_valid_sigma0"] = int(np.sum(valid_db))

    # SSHA in meters (SWOT L3 ssha_unfiltered units)
    if n_valid_ssha > 0:
        ssha_start_m = ssha_start[valid_ssha]
        ssha_end_m = ssha_end[valid_ssha]
        diff_ssha_m = ssha_end_m - ssha_start_m
        stats["mean_ssha_start_m"] = float(np.nanmean(ssha_start_m))
        stats["mean_ssha_end_m"] = float(np.nanmean(ssha_end_m))
        stats["mean_ssha_diff_m"] = float(np.nanmean(diff_ssha_m))
        stats["std_ssha_diff_m"] = float(np.nanstd(diff_ssha_m))
        stats["n_valid_ssha"] = n_valid_ssha
    return stats


def print_stats_for_latex(stats, verbose=True):
    """Print statistics to console for copy-paste into discussion. If verbose, print human-readable too."""
    if not stats or stats.get("n_valid", 0) == 0:
        print("No valid trajectory pairs for statistics.")
        return
    n = stats["n_valid"]
    has_ssha = "mean_ssha_diff_m" in stats
    lat_note = ""
    if stats.get("lat_min") is not None or stats.get("lat_max") is not None:
        lat_note = "  latitude filter: {:.2f} to {:.2f}\n".format(
            stats.get("lat_min", -90.0), stats.get("lat_max", 90.0))
    if verbose:
        print("\n--- Statistics (unsmoothed) for discussion ---")
        if lat_note:
            print(lat_note.rstrip())
        print("  sigma0 (linear): earlier pass (452) mean = {:.4f}   later pass (455) mean = {:.4f}".format(
            stats["mean_sigma0_end_lin"], stats["mean_sigma0_start_lin"]))
        print("                   linear difference (earlier - later) = {:.4f} (std {:.4f})".format(
            stats["mean_sigma0_diff_lin"], stats["std_sigma0_diff_lin"]))
        print("  sigma0 (dB):     earlier pass mean = {:.2f} dB   later pass mean = {:.2f} dB".format(
            stats["mean_sigma0_end_db"], stats["mean_sigma0_start_db"]))
        print("                   pass difference (earlier - later) = {:.2f} dB (std {:.2f} dB; RMS {:.2f} dB; median {:.2f} dB)".format(
            stats["mean_sigma0_diff_db"], stats["std_sigma0_diff_db"],
            stats["rms_sigma0_diff_db"], stats["median_sigma0_diff_db"]))
        if has_ssha:
            print("  SSHA (m):        earlier pass mean = {:.4f}   later pass mean = {:.4f}".format(
                stats["mean_ssha_end_m"], stats["mean_ssha_start_m"]))
            print("                   difference (earlier - later) = {:.4f} m (std {:.4f} m)".format(
                stats["mean_ssha_diff_m"], stats["std_ssha_diff_m"]))
        print("  n_trajectories (sigma0 dB) = {}".format(stats.get("n_valid_sigma0", n)))
    print("\n--- LaTeX snippet (copy into discussion) ---")
    if has_ssha:
        print(r"  $\sigma_0$: mean pass difference {:.2f}~dB (std {:.2f}~dB); SSHA: mean difference {:.2f}~m (std {:.2f}~m); $n={}$ trajectories.".format(
            stats["mean_sigma0_diff_db"], stats["std_sigma0_diff_db"],
            stats["mean_ssha_diff_m"], stats["std_ssha_diff_m"], stats.get("n_valid_sigma0", n)))
    else:
        print(r"  $\sigma_0$: mean pass difference {:.2f}~dB (std {:.2f}~dB); $n={}$ trajectories.".format(
            stats["mean_sigma0_diff_db"], stats["std_sigma0_diff_db"], stats.get("n_valid_sigma0", n)))
    print()


def plot(nc_path, out_path=None, window=31, dpi=300, print_stats=True, lat_min=None, lat_max=None):
    """Plot: top = combined mean sigma0 + mean SSHA vs lat; bottom = std sigma0 + std SSHA vs lat. Smoothed."""
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required.")
    data = load_nc(nc_path)
    if print_stats:
        stats = compute_trajectory_stats(data, lat_min=lat_min, lat_max=lat_max)
        print_stats_for_latex(stats, verbose=True)
    lat_start = data["lat_start"]
    lon_start = data["lon_start"]
    lat_final = data["lat_final"]
    lon_final = data["lon_final"]
    sigma0_start = data["sigma0_start"]
    ssha_start = data["ssha_start"]
    sigma0_end = data["sigma0_end"]
    ssha_end = data["ssha_end"]
    def _get(key, fallback):
        v = data.get(key)
        return np.asarray(v).ravel() if v is not None else fallback
    sigma0_std_start = _get("sigma0_std_start", np.full_like(lat_start, np.nan))
    ssha_std_start = _get("ssha_std_start", np.full_like(lat_start, np.nan))
    sigma0_std_end = _get("sigma0_std_end", np.full_like(lat_start, np.nan))
    ssha_std_end = _get("ssha_std_end", np.full_like(lat_start, np.nan))

    # Distance between start and end per trajectory (m)
    distance_m = haversine_distance_m(lon_start, lat_start, lon_final, lat_final)

    # Sort by latitude
    idx = np.argsort(lat_start)
    x = lat_start[idx]
    sigma0_start = sigma0_start[idx]
    ssha_start = ssha_start[idx]
    sigma0_end = sigma0_end[idx]
    ssha_end = ssha_end[idx]
    sigma0_std_start = sigma0_std_start[idx]
    ssha_std_start = ssha_std_start[idx]
    sigma0_std_end = sigma0_std_end[idx]
    ssha_std_end = ssha_std_end[idx]
    distance_m = distance_m[idx]

    # Convert sigma0 to dB before smoothing (do not smooth linear then log)
    s0_start_lin = np.asarray(sigma0_start, dtype=float)
    s0_end_lin = np.asarray(sigma0_end, dtype=float)
    sigma0_start = linear_sigma0_to_db(s0_start_lin)
    sigma0_end = linear_sigma0_to_db(s0_end_lin)
    sigma0_std_start = linear_sigma0_relative_std_to_db(s0_start_lin, sigma0_std_start)
    sigma0_std_end = linear_sigma0_relative_std_to_db(s0_end_lin, sigma0_std_end)

    # Smooth (use odd window)
    w = int(window) if window % 2 else int(window) + 1
    sigma0_start = smooth(sigma0_start, w)
    ssha_start = smooth(ssha_start, w)
    sigma0_end = smooth(sigma0_end, w)
    ssha_end = smooth(ssha_end, w)
    sigma0_std_start = smooth(sigma0_std_start, w)
    ssha_std_start = smooth(ssha_std_start, w)
    sigma0_std_end = smooth(sigma0_std_end, w)
    ssha_std_end = smooth(ssha_std_end, w)
    distance_m = smooth(distance_m, w)

    # Softer colors: same category, sigma0 darker / SSHA lighter
    # 452: orange family — sigma0 = darker orange, SSHA = light orange
    # 455: green family — sigma0 = soft teal-green, SSHA = light sage
    # Distance: soft blue
    color_452_s0 = "#D97B2B"   # darker orange
    color_452_ssha = "#F0B87A"  # light orange
    color_455_s0 = "#4A9B7A"   # soft teal-green
    color_455_ssha = "#7FBD9B"  # light sage
    color_dist = "#6B9AC4"      # soft blue

    # Labels from source filenames (start = later pass 455, end = earlier pass 452)
    label_start = filename_to_timeslot_label(data.get("_source_start_file", ""))
    label_end = filename_to_timeslot_label(data.get("_source_end_file", ""))
    if not label_start:
        label_start = "455 (start)"
    if not label_end:
        label_end = "452 (end)"

    fig, (ax_mean, ax_std) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # --- Top: combined mean sigma0 (left), mean SSHA (first right), distance (second right) ---
    ax_mean.set_xlim(-77.5, -76.5)
    ax_mean.set_ylabel(r"Mean $\sigma_0$ (dB)", color=color_452_s0)
    ax_mean.tick_params(axis="y", labelcolor=color_452_s0)
    ax_mean.spines["left"].set_color(color_452_s0)
    ax_mean.plot(x, sigma0_end, color=color_452_s0, linestyle="-", linewidth=2, label="{} sigma0".format(label_end))
    ax_mean.plot(x, sigma0_start, color=color_455_s0, linestyle="-", linewidth=2, label="{} sigma0".format(label_start))

    ax_mean_ssha = ax_mean.twinx()
    ax_mean_ssha.set_ylabel("Mean SSHA (m)", color=color_455_s0)
    ax_mean_ssha.tick_params(axis="y", labelcolor=color_455_s0)
    ax_mean_ssha.spines["right"].set_color(color_455_s0)
    ax_mean_ssha.plot(x, ssha_end, color=color_452_ssha, linestyle="-", linewidth=1.5, alpha=0.95, label="{} SSHA".format(label_end))
    ax_mean_ssha.plot(x, ssha_start, color=color_455_ssha, linestyle="-", linewidth=1.5, alpha=0.95, label="{} SSHA".format(label_start))

    ax_mean_dist = ax_mean.twinx()
    ax_mean_dist.spines["right"].set_position(("outward", 55))
    ax_mean_dist.set_ylabel("Distance start–end (m)", color=color_dist)
    ax_mean_dist.tick_params(axis="y", labelcolor=color_dist)
    ax_mean_dist.spines["right"].set_color(color_dist)
    ax_mean_dist.set_ylim(0, 800)
    ax_mean_dist.plot(x, distance_m, color=color_dist, linestyle=":", linewidth=1.5, alpha=0.9, label="Distance")

    ax_mean.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax_mean_ssha.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    # ax_mean.set_title("Mean sigma0 and SSHA", fontsize=10, loc="left")
    ax_mean.grid(True, alpha=0.3)
    lines1, labels1 = ax_mean.get_legend_handles_labels()
    lines2, labels2 = ax_mean_ssha.get_legend_handles_labels()
    lines3, labels3 = ax_mean_dist.get_legend_handles_labels()
    ax_mean.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper left", fontsize=8, ncol=2)

    # --- Bottom: std sigma0 (left), std SSHA (right), distance (second right) ---
    ax_std.set_xlabel("Latitude (°)")
    ax_std.set_ylabel(r"Std $\sigma_0$ (dB)", color=color_452_s0)
    ax_std.tick_params(axis="y", labelcolor=color_452_s0)
    ax_std.spines["left"].set_color(color_452_s0)
    ax_std.plot(x, sigma0_std_end, color=color_452_s0, linestyle="-", linewidth=2, label="{} std sigma0".format(label_end))
    ax_std.plot(x, sigma0_std_start, color=color_455_s0, linestyle="-", linewidth=2, label="{} std sigma0".format(label_start))

    ax_std_ssha = ax_std.twinx()
    ax_std_ssha.set_ylabel("Std SSHA (m)", color=color_455_s0)
    ax_std_ssha.tick_params(axis="y", labelcolor=color_455_s0)
    ax_std_ssha.spines["right"].set_color(color_455_s0)
    ax_std_ssha.plot(x, ssha_std_end, color=color_452_ssha, linestyle="-", linewidth=1.5, alpha=0.95, label="{} std SSHA".format(label_end))
    ax_std_ssha.plot(x, ssha_std_start, color=color_455_ssha, linestyle="-", linewidth=1.5, alpha=0.95, label="{} std SSHA".format(label_start))

    ax_std_dist = ax_std.twinx()
    ax_std_dist.spines["right"].set_position(("outward", 55))
    ax_std_dist.set_ylabel("Distance start–end (m)", color=color_dist)
    ax_std_dist.tick_params(axis="y", labelcolor=color_dist)
    ax_std_dist.spines["right"].set_color(color_dist)
    ax_std_dist.set_ylim(0, 800)
    ax_std_dist.plot(x, distance_m, color=color_dist, linestyle=":", linewidth=1.5, alpha=0.9, label="Distance")

    # ax_std.set_title("Std sigma0 and SSHA", fontsize=10, loc="left")
    ax_std.grid(True, alpha=0.3)
    lines1, labels1 = ax_std.get_legend_handles_labels()
    lines2, labels2 = ax_std_ssha.get_legend_handles_labels()
    lines3, labels3 = ax_std_dist.get_legend_handles_labels()
    # ax_std.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    if out_path:
        # High-resolution: use dpi for raster (png, jpg); PDF/SVG are vector
        fmt = "pdf" if out_path.lower().endswith(".pdf") else "svg" if out_path.lower().endswith(".svg") else None
        save_kw = {"bbox_inches": "tight"}
        if fmt:
            plt.savefig(out_path, format=fmt, **save_kw)
        else:
            save_kw["dpi"] = dpi
            plt.savefig(out_path, **save_kw)
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot sigma0/SSHA trajectories from NetCDF (smoothed).")
    ap.add_argument("nc_file", help="sigma0_ssha_trajectories.nc")
    ap.add_argument("--out", "-o", type=str, default=None, help="Output figure path (e.g. figure.png, figure.pdf, figure.svg)")
    ap.add_argument("--window", "-w", type=int, default=31, help="Smoothing window size (odd, default 31)")
    ap.add_argument("--dpi", type=int, default=600, help="DPI for raster output (png/jpg); default 300 for high-res. Use .pdf or .svg for vector.")
    ap.add_argument("--no-stats", action="store_true", help="Do not print trajectory statistics for discussion.")
    ap.add_argument("--stats-tex", type=str, default=None, help="Write LaTeX macros for stats to this file (e.g. discussion_stats.tex).")
    ap.add_argument("--lat-min", type=float, default=None, help="Min later-pass latitude (deg) for stats filter (optional)")
    ap.add_argument("--lat-max", type=float, default=None, help="Max later-pass latitude (deg) for stats filter (optional)")
    args = ap.parse_args()
    stats = None
    if not args.no_stats:
        data = load_nc(args.nc_file)
        stats = compute_trajectory_stats(data, lat_min=args.lat_min, lat_max=args.lat_max)
        print_stats_for_latex(stats, verbose=True)
        if args.stats_tex and stats and stats.get("n_valid", 0) > 0:
            with open(args.stats_tex, "w") as f:
                f.write("% Auto-generated by plot_sigma0_ssha_trajectories.py --stats-tex\n")
                f.write("% sigma0 pass differences are in dB (per-trajectory 10log10 ratio); plot axes remain linear.\n")
                f.write("\\newcommand{\\statSigmaMeanDiffDb}{" + "{:.2f}".format(stats["mean_sigma0_diff_db"]) + "}\n")
                f.write("\\newcommand{\\statSigmaStdDiffDb}{" + "{:.2f}".format(stats["std_sigma0_diff_db"]) + "}\n")
                f.write("\\newcommand{\\statSigmaRmsDiffDb}{" + "{:.2f}".format(stats["rms_sigma0_diff_db"]) + "}\n")
                f.write("\\newcommand{\\statSigmaMeanDiffLin}{" + "{:.4f}".format(stats["mean_sigma0_diff_lin"]) + "}\n")
                if "mean_ssha_diff_m" in stats:
                    f.write("\\newcommand{\\statSSHAMeanDiffM}{" + "{:.2f}".format(stats["mean_ssha_diff_m"]) + "}\n")
                    f.write("\\newcommand{\\statSSHAStdDiffM}{" + "{:.2f}".format(stats["std_ssha_diff_m"]) + "}\n")
                f.write("\\newcommand{\\statNTraj}{" + "{}".format(stats.get("n_valid_sigma0", stats["n_valid"])) + "}\n")
            print("Wrote stats macros to {}".format(args.stats_tex))
    plot(args.nc_file, out_path=args.out, window=args.window, dpi=args.dpi, print_stats=False,
         lat_min=args.lat_min, lat_max=args.lat_max)


if __name__ == "__main__":
    main()
