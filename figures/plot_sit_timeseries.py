#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from datetime import datetime, date as _date, timedelta
import csv
import calendar

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional deps
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    import mat73
except Exception:
    mat73 = None

try:
    from scipy.ndimage import label, binary_dilation
    HAS_NDIMAGE = True
except Exception:
    HAS_NDIMAGE = False


# --------------------------
# Utilities
# --------------------------
def _parse_datetime_from_text(s: str):
    """Find first occurrence of YYYYMMDDTHHMMSS or YYYYMMDDHHMMSS or YYYYMMDD."""
    s = str(s)
    m = re.search(r"(?<!\d)(\d{8})T(\d{6})(?!\d)", s)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    m = re.search(r"(?<!\d)(\d{14})(?!\d)", s)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    m = re.search(r"(?<!\d)(\d{8})(?!\d)", s)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    return None


def _wrap_lon180(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _lon360(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return (lon % 360.0 + 360.0) % 360.0


def _load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _pick_mask(d: dict, preference: str, sit: np.ndarray, filename: str) -> np.ndarray:
    mask = None
    if preference in d:
        mask = np.asarray(d[preference], dtype=bool).ravel()
    else:
        if "in_scope" in d:
            mask = np.asarray(d["in_scope"], dtype=bool).ravel()
        elif "in_polynya" in d:
            mask = np.asarray(d["in_polynya"], dtype=bool).ravel()

    if mask is None or mask.size != sit.size:
        print(f"[WARN] Mask missing/mismatched in {filename}. Using finite SIT only.")
        mask = np.ones_like(sit, dtype=bool)
    return mask


def _count_segments(d: dict, ok: np.ndarray) -> int:
    """Count unique seg_joint bins used by ok points. Fallback to point count if missing."""
    if "seg_joint" not in d:
        return int(np.sum(ok))
    seg = np.asarray(d["seg_joint"]).ravel()
    if seg.size != ok.size:
        return int(np.sum(ok))
    seg_ok = seg[ok]
    seg_ok = seg_ok[np.isfinite(seg_ok)]
    seg_ok = seg_ok[seg_ok >= 0]
    if seg_ok.size == 0:
        return int(np.sum(ok))
    return int(np.unique(seg_ok.astype(np.int64)).size)


def _load_sit_series_csv(path: str, value_field: str = "mean_sit", date_field: str = "date") -> dict:
    """
    Load a daily SIT time series from CSV into a {YYYY-MM-DD: value} dict.

    Assumes:
      - CSV has at least `date_field` (YYYY-MM-DD) and `value_field` columns.
      - One row per day (later rows overwrite earlier ones if duplicated).
    """
    series = {}
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ymd = row.get(date_field)
            if not ymd:
                continue
            try:
                val = float(row.get(value_field, "nan"))
            except Exception:
                continue
            if not np.isfinite(val):
                continue
            series[str(ymd)] = val
    return series


def _load_mat_any(mat_path: str) -> dict:
    p = Path(mat_path)
    if not p.exists():
        raise FileNotFoundError(mat_path)

    if loadmat is not None:
        try:
            return loadmat(str(p), squeeze_me=True, struct_as_record=False)
        except Exception:
            pass

    if mat73 is not None:
        return mat73.loadmat(str(p))

    raise RuntimeError("Cannot read .mat: install scipy or mat73")


def _matlab_day_index2(ymd: str) -> int:
    """
    MATLAB:
      day_index2 = days(datetime(YYYY,MM,DD) - datetime(2022,12,31))
    For 2023-01-01 -> 1. So python index is (that - 1).
    """
    d = datetime.strptime(ymd, "%Y-%m-%d").date()
    base = _date(2022, 12, 31)
    return (d - base).days


def _daterange(d0: _date, d1: _date):
    d = d0
    while d <= d1:
        yield d
        d = d + timedelta(days=1)


def _month_bounds(d: _date):
    first = d.replace(day=1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = d.replace(day=last_day)
    return first, last


def _parse_ymd(s: str) -> _date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# --------------------------
# Area on sphere (no pyproj)
# --------------------------
def _compute_cell_area_m2_spherical(latt: np.ndarray, lonn: np.ndarray) -> np.ndarray:
    """
    Approx per-pixel area (m^2) using spherical formula:
      area ~= R^2 * |dlat| * |dlon| * cos(lat)
    with gradients along array indices.
    """
    R = 6371000.0  # m
    lat = np.deg2rad(np.asarray(latt, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lonn, dtype=np.float64))
    lon_u = np.unwrap(lon, axis=1)
    dlat = np.gradient(lat, axis=0)
    dlon = np.gradient(lon_u, axis=1)
    area = (R**2) * np.abs(dlat) * np.abs(dlon) * np.cos(lat)
    bad = ~np.isfinite(latt) | ~np.isfinite(lonn) | ~np.isfinite(area)
    area[bad] = np.nan
    return area


# --------------------------
# AMSR SIC loading (preload once + robust scaling)
# --------------------------
def amsr_preload(
    amsr_mat_path: str,
    *,
    lat_name="lat",
    lon_name="lon",
    sic_name="sic",
    out_shape=(1264, 1328),
):
    m = _load_mat_any(amsr_mat_path)
    if lat_name not in m or lon_name not in m or sic_name not in m:
        raise KeyError(f"Missing {lat_name}/{lon_name}/{sic_name} in {amsr_mat_path}")

    lat = np.asarray(m[lat_name], dtype=np.float64)
    lon = np.asarray(m[lon_name], dtype=np.float64)
    sic = np.asarray(m[sic_name], dtype=np.float64)

    # MATLAB: latt = lat'; lonn = lon'
    latt = lat.T if lat.ndim == 2 else lat
    lonn = lon.T if lon.ndim == 2 else lon

    # Make 2D grids if 1D
    if latt.ndim == 1 and lonn.ndim == 1:
        Lon, Lat = np.meshgrid(lonn, latt)
        lonn2, latt2 = Lon, Lat
    else:
        latt2, lonn2 = np.asarray(latt), np.asarray(lonn)

    # Compute cell areas once
    area_m2 = _compute_cell_area_m2_spherical(latt2, lonn2)

    return {
        "latt": latt2,
        "lonn": lonn2,
        "sic": sic,
        "out_shape": tuple(out_shape),
        "area_m2": area_m2,
    }


def amsr_get_sic_day(pre, ymd: str, *, autoscale=True) -> np.ndarray:
    sic = pre["sic"]
    out_shape = pre["out_shape"]
    di2 = _matlab_day_index2(ymd)
    idx0 = di2 - 1
    if idx0 < 0 or idx0 >= sic.shape[0]:
        raise IndexError(f"Date {ymd} -> sic index {idx0} out of range (0..{sic.shape[0]-1})")

    sl = np.asarray(sic[idx0, ...]).squeeze()
    n = int(np.prod(out_shape))
    if sl.size != n:
        raise ValueError(f"SIC slice size={sl.size}, expected {n} for out_shape={out_shape}")

    sic_day = sl.reshape(out_shape, order="F")

    if autoscale and np.isfinite(sic_day).any():
        mx = np.nanpercentile(sic_day[np.isfinite(sic_day)], 99)
        mn = np.nanpercentile(sic_day[np.isfinite(sic_day)], 1)
        # Heuristic: if looks like percent (0..100) or small negative coast codes scaled by 100
        if (mx > 1.5) or (mn < -0.2):
            sic_day = sic_day / 100.0

    return sic_day


# --------------------------
# AMSR polynya detection (robust)
# --------------------------
def detect_polynya_masks_from_amsr(
    sic_day: np.ndarray,
    *,
    threshold=0.7,
    coastal_code_range=(-0.035, -0.015),
):
    if not HAS_NDIMAGE:
        raise RuntimeError("Need scipy.ndimage for AMSR polynya masks (pip install scipy)")

    sic = np.asarray(sic_day, dtype=np.float64)

    low = np.isfinite(sic) & (sic >= 0.0) & (sic < threshold)

    # Coast band (if present), otherwise fallback to negatives/NaNs
    c0, c1 = coastal_code_range
    coast = np.isfinite(sic) & (sic >= c0) & (sic <= c1)
    if np.sum(coast) == 0:
        coast = np.isfinite(sic) & (sic < 0.0)
    if np.sum(coast) == 0:
        coast = ~np.isfinite(sic)

    # Remove open ocean via boundary-connected low-SIC components
    water = low.copy()
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    lbl, n = label(water, structure=structure)

    if n == 0:
        z = np.zeros_like(low, dtype=bool)
        return z, z.copy(), z.copy()

    boundary = np.zeros_like(water, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    boundary_labels = np.unique(lbl[boundary & water])
    boundary_labels = boundary_labels[boundary_labels != 0]
    open_ocean = np.isin(lbl, boundary_labels)

    total_polynya = low & (~open_ocean)

    # Fallback if removal kills everything
    if np.sum(total_polynya) == 0 and np.sum(low) > 0:
        total_polynya = low.copy()

    # Split into coastal/open by coast adjacency
    coast_dil = binary_dilation(coast, structure=structure, iterations=1)
    pol_lbl, pn = label(total_polynya, structure=structure)
    if pn == 0:
        z = np.zeros_like(low, dtype=bool)
        return z, z.copy(), z.copy()

    touching = np.unique(pol_lbl[coast_dil & total_polynya])
    touching = touching[touching != 0]

    if touching.size == 0 and np.sum(total_polynya) > 0:
        coastal_polynya = total_polynya.copy()
        open_polynya = np.zeros_like(total_polynya, dtype=bool)
    else:
        coastal_polynya = total_polynya & np.isin(pol_lbl, touching)
        open_polynya = total_polynya & (~coastal_polynya)

    return total_polynya, coastal_polynya, open_polynya


def compute_amsr_polynya_area_km2(
    latt: np.ndarray,
    lonn: np.ndarray,
    sic_day: np.ndarray,
    area_m2: np.ndarray,
    *,
    threshold=0.7,
    area_mode="total",            # total | coastal | open
    area_weighting="openwater",   # openwater | geom
    roi_lat_min=None,
    roi_lat_max=None,
    roi_lon_min=None,
    roi_lon_max=None,
):
    total, coastal, openp = detect_polynya_masks_from_amsr(sic_day, threshold=threshold)

    if area_mode == "coastal":
        pol = coastal
    elif area_mode == "open":
        pol = openp
    else:
        pol = total

    # ROI mask
    roi = np.ones_like(pol, dtype=bool)
    if any(v is not None for v in [roi_lat_min, roi_lat_max, roi_lon_min, roi_lon_max]):
        lon360 = _lon360(lonn)
        if roi_lat_min is not None:
            roi &= (latt >= float(roi_lat_min))
        if roi_lat_max is not None:
            roi &= (latt <= float(roi_lat_max))
        if roi_lon_min is not None:
            roi &= (lon360 >= float(roi_lon_min))
        if roi_lon_max is not None:
            roi &= (lon360 <= float(roi_lon_max))

    mask = pol & roi
    n_pix = int(np.sum(mask))

    if n_pix == 0:
        return 0.0, 0, {"n_total_pix": int(np.sum(total & roi)),
                        "n_coastal_pix": int(np.sum(coastal & roi)),
                        "n_open_pix": int(np.sum(openp & roi))}

    if area_weighting == "geom":
        area_km2 = float(np.nansum(area_m2[mask]) / 1e6)
    else:
        sic_clip = np.clip(np.asarray(sic_day, dtype=np.float64), 0.0, 1.0)
        w = 1.0 - sic_clip
        w[~np.isfinite(w)] = 0.0
        area_km2 = float(np.nansum(area_m2[mask] * w[mask]) / 1e6)

    meta = {
        "n_total_pix": int(np.sum(total & roi)),
        "n_coastal_pix": int(np.sum(coastal & roi)),
        "n_open_pix": int(np.sum(openp & roi)),
    }
    return area_km2, n_pix, meta


# --------------------------
# Plot helpers for reuse with extra data
# --------------------------
def load_plot_elements(npz_path: str) -> dict:
    """Load saved plot elements from NPZ file."""
    z = np.load(npz_path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    
    # Convert strings back to datetime
    if "sit_times_str" in data:
        data["sit_times"] = [datetime.fromisoformat(str(t)) for t in data["sit_times_str"]]
    if "amsr_days_str" in data:
        data["amsr_days"] = [datetime.fromisoformat(str(d)) for d in data["amsr_days_str"]]
    
    return data


def plot_with_extra_data(
    plot_elements_path: str,
    extra_data_dict: dict = None,
    out_png: str = "sit_timeseries_extra_data.png",
):
    """
    Recreate plot from saved elements and add extra data.
    
    extra_data_dict: {
        "name": "Series Name",
        "times": [...],  # datetime objects or strings
        "values": [...],
        "errors": [...] or None,  # optional
        "color": "#XXXXXX",
        "marker": "o",
        "linestyle": "-",
        "axis": "left" or "right",  # which Y-axis to use
    }
    
    For multiple series, pass a list of dicts.
    """
    
    pe = load_plot_elements(plot_elements_path)
    
    sit_times = pe["sit_times"]
    sit_mean = pe["sit_mean"]
    sit_std = pe["sit_std"]
    sit_nseg = pe["sit_nseg"]
    sit_sizes = pe["sit_sizes"]
    
    amsr_days = pe["amsr_days"]
    amsr_area = pe["amsr_area"]
    
    color_sit = "#d62728"      # Blue for SIT
    color_amsr = "#1f77b4"     # Red for AMSR
    
    # Setup plot
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # SIT errorbars
    ax.errorbar(
        sit_times, sit_mean, yerr=sit_std,
        fmt="none", capsize=4, elinewidth=2.0, alpha=0.5,
        color=color_sit, label="SIT mean ± std", zorder=2
    )

    # SIT scatter
    ax.scatter(
        sit_times, sit_mean,
        s=sit_sizes, alpha=0.7, color=color_sit,
        edgecolors="white", linewidth=0.8,
        label="SIT (daily samples)", zorder=3
    )

    ax.set_xlabel("Date", fontweight="bold", fontsize=15)
    ax.set_ylabel("Sea Ice Thickness (m)", fontweight="bold", fontsize=15, color=color_sit)
    ax.tick_params(axis="y", labelcolor=color_sit)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)

    # Date ticks
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Right axis: AMSR
    ax2 = ax.twinx()
    ax2.plot(
        amsr_days, amsr_area,
        color=color_amsr, linewidth=2.3, marker="o", markersize=4.5,
        alpha=0.8, label="AMSR polynya area (daily)", zorder=4
    )
    ax2.set_ylabel("AMSR polynya area (km²)\n[coastal]",
                   fontweight="bold", fontsize=15, color=color_amsr)
    ax2.tick_params(axis="y", labelcolor=color_amsr)
    
    ax2.spines['right'].set_color(color_amsr)
    ax2.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_color(color_sit)
    ax.spines['left'].set_linewidth(2.0)

    # Add extra data if provided
    if extra_data_dict:
        extra_list = extra_data_dict if isinstance(extra_data_dict, list) else [extra_data_dict]
        for extra in extra_list:
            times = extra.get("times", [])
            values = extra.get("values", [])
            errors = extra.get("errors", None)
            color = extra.get("color", "#000000")
            marker = extra.get("marker", "s")
            linestyle = extra.get("linestyle", "-")
            axis = extra.get("axis", "left")
            name = extra.get("name", "Extra Data")
            
            # Convert string times to datetime if needed
            times = [datetime.fromisoformat(str(t)) if isinstance(t, str) else t for t in times]
            
            target_ax = ax if axis == "left" else ax2
            
            # Plot line if requested
            if linestyle and linestyle != "none":
                target_ax.plot(times, values, linestyle=linestyle, color=color,
                             linewidth=2.0, alpha=0.7, zorder=5)
            
            # Plot scatter
            if errors is not None:
                target_ax.errorbar(times, values, yerr=errors, fmt="none",
                                 capsize=3, elinewidth=1.5, alpha=0.5,
                                 color=color, zorder=5)
            
            target_ax.scatter(times, values, s=80, alpha=0.8, color=color,
                            edgecolors="white", linewidth=0.8, marker=marker,
                            label=name, zorder=6)

    # title2 = "SIT marker size ∝ # IS2 segments | SIT: scatter only | AMSR: daily measurements"
    # ax.set_title("Satellite Data Comparison: IS2 Sea Ice Thickness vs. AMSR Polynya Area\n" + title2,
    #              fontweight="bold", fontsize=13, pad=20)

    # Legends
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=15)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved figure with extra data: {out_png}")
    plt.show()


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot SIT mean/std (scatter only) and AMSR daily polynya area (right axis)."
    )

    ap.add_argument("--root", default=".", help="Root directory for NPZ search.")
    ap.add_argument("--glob", default="**/*_is2_polynya_sit_outputs.npz",
                    help="Glob pattern (quoted) to find NPZ files. Default recursive.")
    ap.add_argument("--mask_preference", choices=["in_scope", "in_polynya"], default="in_scope",
                    help="Which mask to use for SIT stats.")

    ap.add_argument("--sit_min", type=float, default=0.0)
    ap.add_argument("--sit_max", type=float, default=10.0)

    ap.add_argument("--out_png", default="sit_timeseries_with_amsr_area.png")
    ap.add_argument("--out_csv", default="sit_timeseries_with_amsr_area.csv")

    # Marker sizing for SIT
    ap.add_argument("--size_min", type=float, default=25.0)
    ap.add_argument("--size_scale", type=float, default=12.0)

    # AMSR inputs
    ap.add_argument("--amsr_mat", required=True,
                    help="Path to AMSR2_2023.mat (must contain lat, lon, sic).")
    ap.add_argument("--amsr_thresh", type=float, default=0.7)
    ap.add_argument("--amsr_area_mode", choices=["total", "coastal", "open"], default="coastal")
    ap.add_argument("--amsr_area_weighting", choices=["openwater", "geom"], default="openwater")
    ap.add_argument("--amsr_shape_rows", type=int, default=1264)
    ap.add_argument("--amsr_shape_cols", type=int, default=1328)
    ap.add_argument("--amsr_lat_name", default="lat")
    ap.add_argument("--amsr_lon_name", default="lon")
    ap.add_argument("--amsr_sic_name", default="sic")

    # Optional ROI in AMSR grid (0..360 lon recommended)
    ap.add_argument("--roi_lat_min", type=float, default=None)
    ap.add_argument("--roi_lat_max", type=float, default=None)
    ap.add_argument("--roi_lon_min", type=float, default=None)
    ap.add_argument("--roi_lon_max", type=float, default=None)

    # Force AMSR plotting range (default = full months spanning IS2 dates)
    ap.add_argument("--amsr_start", default=None, help="YYYY-MM-DD (optional).")
    ap.add_argument("--amsr_end", default=None, help="YYYY-MM-DD (optional).")

    # Optional external SIT time series for comparison (daily means over same ROI)
    ap.add_argument("--pmw_csv", default=None,
                    help="Optional CSV with PMW daily SIT (columns: date, mean_sit).")
    ap.add_argument("--racmo_csv", default=None,
                    help="Optional CSV with RACMO-derived daily SIT (columns: date, mean_sit).")
    ap.add_argument("--esacci_csv", default=None,
                    help="Optional CSV with ESA CCI daily SIT (columns: date, mean_sit).")

    args = ap.parse_args()

    if not HAS_NDIMAGE:
        raise RuntimeError("scipy is required for AMSR polynya masks: pip install scipy")

    root = Path(args.root)
    files = sorted(root.glob(args.glob))
    if not files:
        raise SystemExit(f"No NPZ files matched: root={root} glob={args.glob}")

    # ---------------- SIT records ----------------
    records = []
    for f in files:
        d = _load_npz(f)

        meta_tag = ""
        if "meta_tag" in d:
            try:
                meta_tag = str(np.asarray(d["meta_tag"]).ravel()[0])
            except Exception:
                meta_tag = ""

        t = _parse_datetime_from_text(meta_tag) if meta_tag else None
        if t is None:
            t = _parse_datetime_from_text(f.name)
        if t is None:
            print(f"[WARN] Could not parse time for {f} -> skipping")
            continue

        sit = np.asarray(d.get("sit", []), dtype=np.float64).ravel()
        if sit.size == 0:
            print(f"[WARN] No 'sit' in {f} -> skipping")
            continue

        mask = _pick_mask(d, args.mask_preference, sit, str(f))
        ok = mask & np.isfinite(sit) & (sit > args.sit_min) & (sit < args.sit_max)

        n_pts = int(np.sum(ok))
        if n_pts == 0:
            print(f"[WARN] No valid SIT samples after masking in {f} -> skipping")
            continue

        mean = float(np.nanmean(sit[ok]))
        std = float(np.nanstd(sit[ok]))
        n_seg = _count_segments(d, ok)

        ymd = t.strftime("%Y-%m-%d")
        records.append({
            "time": t,
            "date": ymd,
            "mean_sit": mean,
            "std_sit": std,
            "n_points": n_pts,
            "n_segments": int(n_seg),
            "file": str(f),
            "meta_tag": meta_tag,
        })

    if not records:
        raise SystemExit("No valid SIT records produced.")

    records = sorted(records, key=lambda r: r["time"])

    # Save CSV (SIT-only + AMSR daily will be saved separately below)
    with open(args.out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"[INFO] Wrote SIT CSV: {args.out_csv}")

    # SIT arrays (SWOT--IS-2)
    sit_times = [r["time"] for r in records]
    sit_mean = np.array([r["mean_sit"] for r in records], dtype=float)
    sit_std  = np.array([r["std_sit"] for r in records], dtype=float)
    sit_nseg = np.array([r["n_segments"] for r in records], dtype=float)
    sit_sizes = args.size_min + args.size_scale * np.sqrt(np.maximum(sit_nseg, 1.0))

    # Pre-compute list of date strings for overlap matching
    sit_dates = [r["time"].date() for r in records]
    sit_dates_str = [d.strftime("%Y-%m-%d") for d in sit_dates]

    # Optional external SIT series (PMW, RACMO, ESA CCI) aligned to SWOT--IS-2 dates
    extra_series = []  # list of dicts for plotting helper and statistics

    def _align_external_series(csv_path: str, name: str, color: str, marker: str = "s"):
        series_map = _load_sit_series_csv(csv_path)
        times = []
        values = []
        for r in records:
            ymd = r["date"]
            if ymd in series_map:
                times.append(r["time"])
                values.append(series_map[ymd])
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            print(f"[WARN] No overlapping days for {name} in {csv_path}")
            return
        mean_val = float(np.nanmean(values))
        print(f"[INFO] {name} mean SIT over {values.size} overlapping SWOT--IS-2 days: "
              f"{mean_val:.3f} m")
        extra_series.append({
            "name": name,
            "times": times,
            "values": values,
            "errors": None,
            "color": color,
            "marker": marker,
            "linestyle": "-",
            "axis": "left",
        })

    # Load and align optional products
    if args.pmw_csv:
        _align_external_series(args.pmw_csv, "PMW SIT", "#1b9e77", marker="^")
    if args.racmo_csv:
        _align_external_series(args.racmo_csv, "RACMO SIT", "#7570b3", marker="v")
    if args.esacci_csv:
        _align_external_series(args.esacci_csv, "ESA CCI SIT", "#d95f02", marker="D")

    # Determine AMSR daily plotting range
    dmin = min(sit_dates)
    dmax = max(sit_dates)

    if args.amsr_start is not None:
        amsr_start = _parse_ymd(args.amsr_start)
    else:
        amsr_start, _ = _month_bounds(dmin)

    if args.amsr_end is not None:
        amsr_end = _parse_ymd(args.amsr_end)
    else:
        _, amsr_end = _month_bounds(dmax)

    # ---------------- AMSR daily series ----------------
    pre = amsr_preload(
        args.amsr_mat,
        lat_name=args.amsr_lat_name,
        lon_name=args.amsr_lon_name,
        sic_name=args.amsr_sic_name,
        out_shape=(args.amsr_shape_rows, args.amsr_shape_cols),
    )

    latt = pre["latt"]
    lonn = pre["lonn"]
    area_m2 = pre["area_m2"]

    amsr_days = []
    amsr_area = []

    for d in _daterange(amsr_start, amsr_end):
        ymd = d.strftime("%Y-%m-%d")
        try:
            sic_day = amsr_get_sic_day(pre, ymd, autoscale=True)
            area_km2, _, _ = compute_amsr_polynya_area_km2(
                latt, lonn, sic_day, area_m2,
                threshold=float(args.amsr_thresh),
                area_mode=args.amsr_area_mode,
                area_weighting=args.amsr_area_weighting,
                roi_lat_min=args.roi_lat_min,
                roi_lat_max=args.roi_lat_max,
                roi_lon_min=args.roi_lon_min,
                roi_lon_max=args.roi_lon_max,
            )
        except Exception:
            area_km2 = np.nan

        amsr_days.append(datetime(d.year, d.month, d.day))
        amsr_area.append(area_km2)

    amsr_area = np.asarray(amsr_area, dtype=float)

    # ---- Save plotting elements to NPZ for later use with extra data ----
    plot_npz_path = args.out_png.replace(".png", "_plot_elements.npz")
    np.savez(
        plot_npz_path,
        # SIT data
        sit_times_str=[str(t) for t in sit_times],  # Convert to strings for NPZ
        sit_mean=sit_mean,
        sit_std=sit_std,
        sit_nseg=sit_nseg,
        sit_sizes=sit_sizes,
        # AMSR data
        amsr_days_str=[str(d) for d in amsr_days],  # Convert to strings for NPZ
        amsr_area=amsr_area,
        # Configuration
        size_min=np.array([args.size_min]),
        size_scale=np.array([args.size_scale]),
        amsr_area_mode=args.amsr_area_mode,
        amsr_area_weighting=args.amsr_area_weighting,
    )
    print(f"[INFO] Saved plot elements: {plot_npz_path}")

    # ---------------- Plot ----------------
    # Color palette: professional and publication-ready
    color_sit = "#d62728"      # SIT (SWOT--IS-2)
    color_amsr = "#1f77b4"     # AMSR area
    
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # SIT errorbars (light, semi-transparent)
    ax.errorbar(
        sit_times, sit_mean, yerr=sit_std,
        fmt="none", capsize=4, elinewidth=2.0, alpha=0.5,
        color=color_sit, label="SIT mean ± std", zorder=2
    )

    # SIT scatter (no connecting line)
    sc = ax.scatter(
        sit_times, sit_mean,
        s=sit_sizes, alpha=0.7, color=color_sit,
        edgecolors="white", linewidth=0.8,
        label="SIT (daily samples)", zorder=3
    )

    ax.set_xlabel("Date", fontweight="bold", fontsize=15)
    ax.set_ylabel("Sea Ice Thickness (m)", fontweight="bold", fontsize=15, color=color_sit)
    ax.tick_params(axis="y", labelcolor=color_sit)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)

    # Nicer date ticks
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    
    # Rotate x-labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Right axis: AMSR daily area (continuous Sep–Oct)
    ax2 = ax.twinx()
    line = ax2.plot(
        amsr_days, amsr_area,
        color=color_amsr, linewidth=2.3, marker="o", markersize=4.5,
        alpha=0.8, label="AMSR polynya area (daily)", zorder=4
    )
    
    ylabel = "AMSR polynya area (km²)"
    if args.amsr_area_weighting == "openwater":
        ylabel = "AMSR polynya open-water-equivalent area (km²)"
    ax2.set_ylabel(f"{ylabel}\n[{args.amsr_area_mode}]", 
                   fontweight="bold", fontsize=15, color=color_amsr)
    ax2.tick_params(axis="y", labelcolor=color_amsr)
    
    # Color the right spine to match the AMSR color
    ax2.spines['right'].set_color(color_amsr)
    ax2.spines['right'].set_linewidth(2.0)
    
    # Color left spine for SIT
    ax.spines['left'].set_color(color_sit)
    ax.spines['left'].set_linewidth(2.0)

    # title2 = "SIT marker size ∝ # IS2 segments | SIT: scatter only | AMSR: daily measurements"
    # ax.set_title("Satellite Data Comparison: IS2 Sea Ice Thickness vs. AMSR Polynya Area\n" + title2,
    #              fontweight="bold", fontsize=13, pad=20)

    # ---- Legends ----
    # Main legend for series (including any extra SIT products)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, 
              loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=15)

    # Marker-size legend (3 representative sizes)
    # pick three representative nseg levels
    nseg_vals = sit_nseg[np.isfinite(sit_nseg)]
    if nseg_vals.size > 0:
        q = np.quantile(nseg_vals, [0.25, 0.50, 0.75])
        q = np.unique(np.round(q).astype(int))
        if q.size == 1:
            q = np.array([max(1, q[0]//2), q[0], q[0]*2], dtype=int)

        def _size_from_nseg(n):
            return args.size_min + args.size_scale * np.sqrt(max(float(n), 1.0))

        size_handles = []
        size_labels = []
        for n in q[:3]:
            size_handles.append(
                ax.scatter([], [], s=_size_from_nseg(n), alpha=0.7, 
                          color=color_sit, edgecolors="white", linewidth=0.8)
            )
            size_labels.append(f"{int(n)} segments")

        leg2 = ax.legend(
            size_handles, size_labels,
            loc="upper left",
            frameon=True, fancybox=True, shadow=True, fontsize=13
        )
        ax.add_artist(leg2)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=220, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved figure: {args.out_png}")
    plt.show()


if __name__ == "__main__":
    main()