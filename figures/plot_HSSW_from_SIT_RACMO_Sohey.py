#!/usr/bin/env python3
"""
Plot HSSW production rates from three ice-production / thin-ice-thickness estimates.

This script is designed to plug into your existing workflow:
- The primary input is the same *plot-elements* NPZ you already produce for SIT+AMSR.
- Optional inputs:
  - RACMO MAT (daily series over the polynya)
  - Sohey netCDF (HI field) + AMSR MAT for the MATLAB-like coastal mask sampling

HSSW conversion follows the chain used in your MATLAB snippet:
  ice-production rate Pi  -> salt flux Ps -> HSSW transport PHSSW (Sv)

Key assumptions (all configurable by CLI):
- frazil-ice salinity si = frazil_frac * sw
- sw = S_LSSW

Notes
-----
1) Using *sea-ice thickness* as *sea-ice production* is a strong assumption.
   This script will do it (because you asked), but interpret results cautiously.
2) Unit handling is the main source of "too high" outputs.
   Defaults are set to m/month as you requested.

Observational benchmarks (for plot context)
------------------------------------------
A) Terra Nova Bay (TNB) mooring salinity time series (Miller et al., Nat. Commun., 2024):
   - Annual-mean HSSW production: 0.43 Sv with 95% CI [0.34, 0.55]
   - Event/instantaneous mean across brine-rejection events: 3.76 Sv (95% CI [2.89, 4.75])
   We plot the annual-mean band (default: 0.43 ± 0.105 Sv).

B) Ross Ice Shelf polynya (RISp) in situ floats (Falco et al., Nat. Commun., 2024):
   - HSSW formation estimate: 0.1–0.4 Sv (depending on assumed polynya area)
   We plot a band spanning 0.1–0.4 Sv by default.

C) Western Ross Sea export benchmark (Cape Adare slope/benthic current; Gordon et al., 2009):
   - Annual-average HSSW export: ~0.2 Sv (if export is ~6-month seasonal event)
   - Summer snapshot HSSW component: ~0.4 Sv
   Cape Adare reference lines are not shown by default; use --show_cape_adare (and optionally
   --show_cape_adare_summer for the 0.4 Sv line) to add them.

Example (single region)
----------------------
python plot_HSSW_PM_RACMO_SIT.py sit_with_amsr_area_nice_plot_elements.npz \
  --amsr_mat AMSR2_2023.mat \
  --sohey_nc HI-2023-latlon_Sohey.nc --sohey_expand_daily \
  --racmo_mat RACMO_20230501_1101_all_hi.mat --racmo_var Ice_mean \
  --start 2023-05-01 --end 2023-11-01 \
  --out_png hssw_rates.png

Example (three polynyas: RSP, TNBP, MSP)
----------------------------------------
python plot_HSSW_from_SIT_RACMO_Sohey.py sit_with_amsr_area_nice_plot_elements.npz \
  --amsr_mat AMSR2_2023.mat \
  --sohey_nc HI-2023-latlon_Sohey.nc --sohey_expand_daily \
  --racmo_mat RACMO_20230501_1101_all_hi_by_polynya.mat --racmo_var Ice_mean \
  --start 2023-09-01 --end 2023-11-01 \
  --out_png hssw_three_methods.png

When the RACMO MAT contains Ice_mean_RSP, Ice_mean_TNBP, Ice_mean_MSP (and optionally
Mean_all_RSP, etc.), the script auto-detects and plots HSSW for each polynya separately.

To show Cape Adare reference lines:
  --show_cape_adare
  --show_cape_adare_summer   # also add summer (0.4 Sv) line
"""

# Polynya region suffixes in per-polynya RACMO MAT files (Ross Sea, Terra Nova Bay, McMurdo Sound)
POLYNYA_SUFFIXES = ("RSP", "TNBP", "MSP")
POLYNYA_LABELS = {
    "RSP": "Ross Sea Polynya (RSP)",
    "TNBP": "Terra Nova Bay Polynya (TNBP)",
    "MSP": "McMurdo Sound Polynya (MSP)",
}

import argparse
import calendar
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional deps
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    from scipy.interpolate import LinearNDInterpolator
    HAS_SCIPY_INTERP = True
except Exception:
    HAS_SCIPY_INTERP = False

try:
    import netCDF4
    HAS_NETCDF4 = True
except Exception:
    HAS_NETCDF4 = False

try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False


# --------------------------
# MAT helpers (v7.3-safe)
# --------------------------

def _std_from_mean_all_cell(mean_all):
    """Compute sample std for each cell entry in a MATLAB cell array."""
    arr = np.array(mean_all).squeeze()
    if arr.ndim == 0:
        arr = np.array([arr], dtype=object)
    arr = arr.ravel()

    std = np.full(arr.size, np.nan, dtype=float)
    for i, item in enumerate(arr):
        if item is None:
            continue
        try:
            vals = np.array(item, dtype=float).ravel()
        except Exception:
            vals = np.array(item).astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            std[i] = float(np.std(vals, ddof=1))
        elif vals.size == 1:
            std[i] = 0.0
    return std


def _h5_read_dataset(ds):
    arr = np.array(ds)
    if arr.shape == ():
        return arr
    if arr.ndim >= 2:
        arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
    return arr


def load_mat_any(path: str):
    """Load MAT files for both <=v7.2 (scipy) and v7.3 (HDF5 via h5py)."""
    path = str(path)

    if loadmat is not None:
        try:
            return loadmat(path)
        except NotImplementedError:
            pass
        except Exception:
            pass

    if not HAS_H5PY:
        raise RuntimeError(
            "MATLAB v7.3 file detected. Install h5py: `conda install h5py` or `pip install h5py`."
        )

    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                key = name.split("/")[-1]
                if key in out:
                    out[name] = _h5_read_dataset(obj)
                else:
                    out[key] = _h5_read_dataset(obj)
        f.visititems(visit)

    return out


# --------------------------
# Date/time helpers
# --------------------------

def _parse_dt(s: str) -> datetime:
    s = str(s).strip().replace("T", " ")
    if "." in s:
        base, frac = s.split(".", 1)
        m = re.match(r"(\d+)", frac)
        frac_digits = (m.group(1) if m else "")
        frac_digits = (frac_digits + "000000")[:6]
        s = f"{base}.{frac_digits}"
    return datetime.fromisoformat(s)


def _parse_dt_list(str_list):
    return [_parse_dt(s) for s in str_list]


def _matlab_datenum_to_datetime(dn):
    dn = float(dn)
    days = int(dn)
    frac = dn - days
    return datetime.fromordinal(days) + timedelta(days=frac) - timedelta(days=366)


def _maybe_read_mat_dates(mat_obj):
    for k in ["date_daily", "dates", "date", "time", "t"]:
        if k in mat_obj:
            arr = np.array(mat_obj[k]).squeeze()
            if arr.size == 0:
                continue
            try:
                return [_matlab_datenum_to_datetime(v) for v in arr.ravel()]
            except Exception:
                pass
    return None


def _crop_series(dates, values, start_dt=None, end_dt=None):
    if start_dt is None and end_dt is None:
        return list(dates), np.array(values)
    dd = np.array(dates)
    vv = np.array(values)
    mask = np.ones(dd.shape, dtype=bool)
    if start_dt is not None:
        mask &= (dd >= start_dt)
    if end_dt is not None:
        mask &= (dd <= end_dt)
    return dd[mask].tolist(), vv[mask]


# --------------------------
# Sohey helper (MATLAB-like)
# --------------------------

def _nanmean_safe(x, axis=None):
    with np.errstate(invalid="ignore"):
        return np.nanmean(x, axis=axis)


def _as_2d_latlon(lat, lon, target_shape):
    lat = np.array(lat, dtype=float)
    lon = np.array(lon, dtype=float)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    if lat2d.shape != target_shape:
        if lat2d.T.shape == target_shape:
            lat2d = lat2d.T
            lon2d = lon2d.T
        else:
            raise ValueError(
                f"lat/lon shape {lat2d.shape} does not match target spatial shape {target_shape}"
            )
    return lat2d, lon2d


def _compute_monthly_mean_sic(sic_daily):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Y, X = sic_daily.shape[1], sic_daily.shape[2]
    out = np.full((Y, X, 12), np.nan, dtype=float)
    idx = 0
    for m, nd in enumerate(days_in_month):
        out[:, :, m] = _nanmean_safe(sic_daily[idx: idx + nd, :, :], axis=0)
        idx += nd
    return out


def _harmonize_lon(lon_a, lon_b):
    lon_a = np.array(lon_a, dtype=float)
    lon_b = np.array(lon_b, dtype=float)
    a_min, a_max = np.nanmin(lon_a), np.nanmax(lon_a)
    b_min, b_max = np.nanmin(lon_b), np.nanmax(lon_b)

    a_0360 = (a_min >= 0) and (a_max > 180)
    b_0360 = (b_min >= 0) and (b_max > 180)

    if a_0360 and (b_min < 0):
        lon_b = np.mod(lon_b, 360.0)
    elif b_0360 and (a_min < 0):
        lon_a = np.mod(lon_a, 360.0)

    return lon_a, lon_b


def _cut_to_sohey_bounds(lat_amsr, lon_amsr, lat_sohey, lon_sohey):
    lat_min, lat_max = np.nanmin(lat_sohey), np.nanmax(lat_sohey)
    lon_min, lon_max = np.nanmin(lon_sohey), np.nanmax(lon_sohey)

    in_box = (
        (lat_amsr >= lat_min) & (lat_amsr <= lat_max) &
        (lon_amsr >= lon_min) & (lon_amsr <= lon_max)
    )
    ii, jj = np.where(in_box)
    if ii.size == 0 or jj.size == 0:
        raise RuntimeError("No overlap found between AMSR grid and Sohey bounds. Check lon convention.")
    return ii.min(), ii.max(), jj.min(), jj.max()


def _read_sohey_hi_nc(sohey_nc, sohey_var="HI"):
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available; cannot read netcdf.")
    nc = netCDF4.Dataset(sohey_nc, "r")

    if sohey_var not in nc.variables:
        raise KeyError(
            f"Variable '{sohey_var}' not in {sohey_nc}. Available: {list(nc.variables.keys())}"
        )

    v = nc.variables[sohey_var]
    hi_raw = v[:]
    if np.ma.isMaskedArray(hi_raw):
        hi = hi_raw.filled(np.nan).astype(float)
    else:
        hi = np.array(hi_raw, dtype=float)

    fill = getattr(v, "_FillValue", None)
    if fill is not None:
        hi[hi == float(fill)] = np.nan

    lat = np.array(nc.variables["latitude"][:], dtype=float) if "latitude" in nc.variables else None
    lon = np.array(nc.variables["longitude"][:], dtype=float) if "longitude" in nc.variables else None
    nc.close()

    if hi.ndim != 3:
        raise ValueError(f"Expected HI 3D array; got {hi.shape}")

    # Put time on last axis by smallest-dim heuristic
    t_axis = int(np.argmin(list(hi.shape)))
    hi_yxt = np.moveaxis(hi, t_axis, -1)

    if lat is None or lon is None:
        raise RuntimeError("Sohey netcdf missing latitude/longitude variables.")

    lat2d, lon2d = _as_2d_latlon(lat, lon, hi_yxt.shape[:2])
    return hi_yxt, lat2d, lon2d


def _compute_sohey_mean_std_like_matlab(
    sohey_nc,
    amsr_mat,
    sohey_var="HI",
    thr=0.7,
    coastal_mat=None,
    coastal_var="Coastal",
    coastal_month_offset=0,
    debug=False,
):
    """Interpolate Sohey HI to AMSR cut grid and compute mean/std over Coastal mask."""
    if not HAS_SCIPY_INTERP:
        raise RuntimeError("scipy.interpolate not available (need LinearNDInterpolator).")

    hi_yxt, lat_sohey, lon_sohey = _read_sohey_hi_nc(sohey_nc, sohey_var=sohey_var)
    T = hi_yxt.shape[2]

    amsr = load_mat_any(amsr_mat)
    if "sic" not in amsr or "lat" not in amsr or "lon" not in amsr:
        raise KeyError("AMSR MAT must contain sic, lat, lon")

    sic = np.array(amsr["sic"], dtype=float)
    latA = np.array(amsr["lat"], dtype=float)
    lonA = np.array(amsr["lon"], dtype=float)

    # sic: accept [Y,X,365] -> [365,Y,X]
    if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
        sic = np.moveaxis(sic, -1, 0)

    # MATLAB does lat=lat'; lon=lon'
    if latA.shape != sic.shape[1:]:
        if latA.T.shape == sic.shape[1:]:
            latA = latA.T
            lonA = lonA.T
        else:
            raise ValueError(f"AMSR lat shape {latA.shape} does not match sic spatial {sic.shape[1:]}")

    lonA, lon_sohey = _harmonize_lon(lonA, lon_sohey)

    sic_monthly = _compute_monthly_mean_sic(sic)
    sic_monthly_frac = sic_monthly / 100.0

    # Coastal mask
    Coastal = None
    if coastal_mat:
        cm = load_mat_any(coastal_mat)
        if coastal_var not in cm:
            raise KeyError(f"'{coastal_var}' not in {coastal_mat}. Keys: {sorted(cm.keys())}")
        Coastal = np.array(cm[coastal_var], dtype=float)
    elif coastal_var in amsr:
        Coastal = np.array(amsr[coastal_var], dtype=float)

    if Coastal is None:
        Coastal = np.where(sic_monthly_frac < thr, 1.0, np.nan)

    # Ensure Coastal [Y,X,12]
    if Coastal.shape != sic_monthly.shape:
        if Coastal.shape == (sic_monthly.shape[2], sic_monthly.shape[0], sic_monthly.shape[1]):
            Coastal = np.moveaxis(Coastal, 0, 2)
        else:
            raise ValueError(f"Coastal shape {Coastal.shape} not compatible with sic_monthly {sic_monthly.shape}")

    # Cut AMSR domain to Sohey bounds
    i0, i1, j0, j1 = _cut_to_sohey_bounds(latA, lonA, lat_sohey, lon_sohey)
    lat_cut = latA[i0:i1 + 1, j0:j1 + 1]
    lon_cut = lonA[i0:i1 + 1, j0:j1 + 1]
    coastal_cut = Coastal[i0:i1 + 1, j0:j1 + 1, :]

    lat_flat = lat_sohey.ravel()
    lon_flat = lon_sohey.ravel()

    out_mean = np.full((T,), np.nan, dtype=float)
    out_std = np.full((T,), np.nan, dtype=float)

    if debug:
        print(f"[SOHEY] HI shape={hi_yxt.shape} (Y,X,T)")
        print(f"[SOHEY] AMSR lon range: {np.nanmin(lonA):.2f}..{np.nanmax(lonA):.2f}")
        print(f"[SOHEY] Sohey lon range: {np.nanmin(lon_sohey):.2f}..{np.nanmax(lon_sohey):.2f}")
        print(f"[SOHEY] coastal_month_offset={coastal_month_offset}")

    for t in range(T):
        hi_flat = hi_yxt[:, :, t].ravel()
        valid = np.isfinite(lat_flat) & np.isfinite(lon_flat) & np.isfinite(hi_flat)
        if np.count_nonzero(valid) < 50:
            continue

        F = LinearNDInterpolator(
            np.column_stack([lon_flat[valid], lat_flat[valid]]),
            hi_flat[valid],
            fill_value=np.nan,
        )
        hi_interp = F(lon_cut, lat_cut)

        m_idx = int(t + coastal_month_offset)
        m_idx = max(0, min(m_idx, coastal_cut.shape[2] - 1))

        msk = np.isfinite(coastal_cut[:, :, m_idx])
        vals = hi_interp[msk]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out_mean[t] = float(np.mean(vals))
            out_std[t] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

        if debug and (t < 5 or t == T - 1):
            print(f"[SOHEY] t={t} m_idx={m_idx} n={vals.size} mean={out_mean[t]:.4f} std={out_std[t]:.4f}")

    return out_mean, out_std


# --------------------------
# Physics conversions
# --------------------------

def _days_in_month(dt: datetime) -> int:
    return int(calendar.monthrange(dt.year, dt.month)[1])


def thickness_to_rate_mps(thickness, dates, units: str) -> np.ndarray:
    """Convert thickness (m/<period>) into production rate Pi (m/s)."""
    units = (units or "").lower().strip()
    x = np.array(thickness, dtype=float)

    if units in ["m_s", "mps", "m/s"]:
        return x

    if units in ["m_day", "m/d", "m_per_day", "m/day"]:
        return x / 86400.0

    if units in ["m_month", "m/mo", "m_per_month", "m/month"]:
        secs = np.array([_days_in_month(d) * 86400.0 for d in dates], dtype=float)
        return x / secs

    if units in ["m_year", "m/yr", "m_per_year"]:
        return x / (365.0 * 86400.0)

    raise ValueError(f"Unknown thickness units: {units}")


def area_to_m2(area, area_units: str) -> np.ndarray:
    """Convert area to m^2. Supports km2, m2, and auto."""
    a = np.array(area, dtype=float)
    u = (area_units or "").lower().strip()

    if u == "auto":
        # crude but robust for polynya areas
        med = np.nanmedian(a)
        if np.isfinite(med) and med > 1e8:
            u = "m2"
        else:
            u = "km2"

    if u in ["km2", "km^2", "kmsq"]:
        return a * 1e6
    if u in ["m2", "m^2", "msq"]:
        return a

    raise ValueError(f"Unknown area_units: {area_units}")


def align_area_to_dates(series_dates, area_dates, area_values, max_gap_days=10):
    """Nearest-neighbour match from area_dates -> series_dates.

    If the nearest area observation is more than max_gap_days away,
    returns NaN for that date.
    """
    sd = np.array(series_dates)
    ad = np.array(area_dates)
    av = np.array(area_values, dtype=float)

    out = np.full(sd.shape, np.nan, dtype=float)
    if sd.size == 0 or ad.size == 0:
        return out

    ad_ts = np.array([d.timestamp() for d in ad], dtype=float)
    sd_ts = np.array([d.timestamp() for d in sd], dtype=float)

    for i, t in enumerate(sd_ts):
        j = int(np.argmin(np.abs(ad_ts - t)))
        dt_days = abs(ad_ts[j] - t) / 86400.0
        if dt_days <= max_gap_days:
            out[i] = av[j]

    return out


def hssw_from_production(
    pi_mps,
    area_m2,
    s_hssw=34.85,
    s_lssw=34.79,
    delta_s=None,
    rho_i=950.0,
    rho_hssw=1028.0,
    frazil_frac=0.31,
    clip_negative=True,
):
    """Compute HSSW transport rate (Sv) from ice production rate Pi (m/s).

    Ps = rho_i * Pi * A * (sw - si) * 1e-3, with si = frazil_frac * sw
    PHSSW = Ps / (rho_hssw * (S_HSSW - S_LSSW) * 1e-3) * 1e-6
    """
    pi = np.array(pi_mps, dtype=float)
    A = np.array(area_m2, dtype=float)

    sw = float(s_lssw)
    si = float(frazil_frac) * sw
    deltaS = float(delta_s) if (delta_s is not None) else (float(s_hssw) - float(s_lssw))

    if deltaS <= 0:
        raise ValueError("Need s_hssw > s_lssw")

    salt_flux = rho_i * pi * A * (sw - si) * 1e-3  # kg/s
    hssw = salt_flux / (rho_hssw * deltaS * 1e-3) * 1e-6  # Sv

    if clip_negative:
        hssw = np.where(np.isfinite(hssw), np.maximum(hssw, 0.0), np.nan)

    return hssw


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to plot-elements NPZ (SIT+AMSR).")
    ap.add_argument("--out_png", default="hssw_rates.png")
    ap.add_argument("--title", default="High Salinity Shelf Water Production Rates")

    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)

    # Area
    ap.add_argument("--area_units", default="auto", choices=["km2", "m2", "auto"],
                    help="Units of AMSR polynya area stored in the NPZ. 'auto' guesses from magnitude (safer).")
    ap.add_argument("--max_area_gap_days", type=float, default=10.0,
                    help="Max allowed time gap (days) when matching AMSR area to series dates.")

    # Production thickness units (defaults requested: m/month)
    ap.add_argument("--sit_units", default="m_month", choices=["m_month", "m_day", "m_s", "m_year"],
                    help="Units for SIT-derived thickness/production in NPZ (assumed production thickness).")
    ap.add_argument("--racmo_units", default="m_month", choices=["m_month", "m_day", "m_s", "m_year"],
                    help="Units for RACMO series (Ice_mean) that represents ice production/thickness.")
    ap.add_argument("--sohey_units", default="m_month", choices=["m_month", "m_day", "m_s", "m_year"],
                    help="Units for Sohey HI sampling (mean series).")

    # RACMO
    ap.add_argument("--racmo_mat", default=None)
    ap.add_argument("--racmo_var", default="Ice_mean")
    ap.add_argument("--racmo_start", default="2023-05-01")
    ap.add_argument("--racmo_std_var", default=None)

    # Sohey
    ap.add_argument("--sohey_nc", default=None)
    ap.add_argument("--sohey_var", default="HI")
    ap.add_argument("--sohey_year", type=int, default=2023)
    ap.add_argument("--sohey_expand_daily", action="store_true",
                    help="If Sohey time axis is monthly (6 or 8 slices), repeat values to daily May–Oct.")
    ap.add_argument("--sohey_thr", type=float, default=0.7)
    ap.add_argument("--amsr_mat", default=None)
    ap.add_argument("--coastal_mat", default=None)
    ap.add_argument("--coastal_var", default="Coastal")
    ap.add_argument("--sohey_coastal_month_offset", type=int, default=0)
    ap.add_argument("--debug_sohey", action="store_true")

    # HSSW parameters
    ap.add_argument("--s_hssw", type=float, default=34.81)
    ap.add_argument("--s_lssw", type=float, default=34.79)
    ap.add_argument("--delta_s", type=float, default=None,
                    help="Override ΔS = S_HSSW - S_LSSW (psu). If omitted, computed from --s_hssw and --s_lssw.")
    ap.add_argument("--rho_i", type=float, default=950.0)
    ap.add_argument("--rho_hssw", type=float, default=1028.0)
    ap.add_argument("--frazil_frac", type=float, default=0.31)

    # Observational context overlays
    # TNB mooring annual mean (default = 0.43 ± 0.105 Sv)
    ap.add_argument("--mooring_mean", type=float, default=0.43,
                    help="TNB mooring annual-mean reference (Sv). Default: 0.43.")
    ap.add_argument("--mooring_std", type=float, default=0.105,
                    help="Half-width of TNB band (Sv). Default approximates 95% CI half-range: (0.55-0.34)/2=0.105.")
    ap.add_argument("--no_tnb_band", action="store_true",
                    help="Disable the TNB mooring annual-mean reference band.")

    # RIS polynya formation band
    ap.add_argument("--risp_low", type=float, default=0.10,
                    help="Lower bound (Sv) for RIS polynya HSSW formation estimate.")
    ap.add_argument("--risp_high", type=float, default=0.40,
                    help="Upper bound (Sv) for RIS polynya HSSW formation estimate.")
    ap.add_argument("--no_risp_band", action="store_true",
                    help="Disable the RIS polynya formation reference band.")

    # Cape Adare export lines (export benchmark)
    ap.add_argument("--cape_adare_export", type=float, default=0.20,
                    help="Cape Adare HSSW export benchmark (Sv), annual-average.")
    ap.add_argument("--cape_adare_summer", type=float, default=0.40,
                    help="Cape Adare HSSW export benchmark (Sv), summer snapshot.")
    ap.add_argument("--show_cape_adare_summer", action="store_true",
                    help="If set (and --show_cape_adare), plot the Cape Adare summer snapshot line as well.")
    ap.add_argument("--show_cape_adare", action="store_true",
                    help="Show Cape Adare HSSW export reference line(s). Default: do not show.")

    # Plot
    ap.add_argument("--ylim", type=float, default=1.7)
    ap.add_argument("--save_npz_out", default=None,
                    help="Optional output NPZ containing computed HSSW series.")

    args = ap.parse_args()

    start_dt = _parse_dt(args.start) if args.start else None
    end_dt = _parse_dt(args.end) if args.end else None

    # ---- Load base NPZ (SIT+AMSR) ----
    d = np.load(args.npz, allow_pickle=True)

    sit_times = _parse_dt_list(d["sit_times_str"])
    sit_mean = np.array(d["sit_mean"], dtype=float)
    sit_std = np.array(d["sit_std"], dtype=float)
    sit_sizes = np.array(d["sit_sizes"], dtype=float) if "sit_sizes" in d else None

    amsr_days = _parse_dt_list(d["amsr_days_str"])
    amsr_area = np.array(d["amsr_area"], dtype=float)

    # Crop base
    sit_times_c, sit_mean_c = _crop_series(sit_times, sit_mean, start_dt, end_dt)
    _, sit_std_c = _crop_series(sit_times, sit_std, start_dt, end_dt)
    amsr_days_c, amsr_area_c = _crop_series(amsr_days, amsr_area, start_dt, end_dt)

    # Convert area
    amsr_area_m2_c = area_to_m2(amsr_area_c, args.area_units)

    # ---- RACMO (single region or per-polynya: RSP, TNBP, MSP) ----
    racmo_times_c = None
    racmo_prod_c = None
    racmo_std_prod_c = None
    racmo_by_polynya = False  # True when MAT has Ice_mean_RSP, Ice_mean_TNBP, Ice_mean_MSP
    racmo_prod_by_region = {}   # suffix -> 1d array (cropped)
    racmo_std_by_region = {}    # suffix -> 1d array or None

    if args.racmo_mat:
        mat = load_mat_any(args.racmo_mat)
        base_var = args.racmo_var

        # Auto-detect per-polynya MAT: expect {base_var}_RSP, _TNBP, _MSP
        var_rsp = f"{base_var}_RSP"
        if var_rsp in mat:
            racmo_by_polynya = True
            print(f"[INFO] RACMO per-polynya MAT detected: using {var_rsp}, {base_var}_TNBP, {base_var}_MSP")

        if racmo_by_polynya:
            racmo_dates = None
            time_len = None
            for suf in POLYNYA_SUFFIXES:
                vname = f"{base_var}_{suf}"
                if vname not in mat:
                    raise KeyError(
                        f"Per-polynya MAT expected variable '{vname}' in {args.racmo_mat}. Keys: {sorted(k for k in mat.keys() if not k.startswith('__'))}"
                    )
                vals = np.array(mat[vname]).squeeze().astype(float)
                if time_len is None:
                    time_len = vals.size
                    racmo_dates = _maybe_read_mat_dates(mat)
                    if racmo_dates is None:
                        start_r = _parse_dt(args.racmo_start)
                        racmo_dates = [start_r + timedelta(days=i) for i in range(time_len)]
                times_c, prod_c = _crop_series(racmo_dates, vals, start_dt, end_dt)
                racmo_prod_by_region[suf] = prod_c
                if racmo_times_c is None:
                    racmo_times_c = times_c

                # std from Mean_all_{suf} if present
                mean_all_key = f"Mean_all_{suf}"
                std_vals = None
                if args.racmo_std_var:
                    std_var_suf = f"{args.racmo_std_var}_{suf}"
                    if std_var_suf in mat:
                        cand = np.array(mat[std_var_suf]).squeeze()
                        if cand.ndim == 1 and cand.size == time_len:
                            std_vals = cand.astype(float)
                if std_vals is None and mean_all_key in mat:
                    try:
                        std_vals = _std_from_mean_all_cell(mat[mean_all_key])
                        if std_vals.size != time_len:
                            std_vals = None
                    except Exception:
                        std_vals = None
                if std_vals is not None:
                    _, std_c = _crop_series(racmo_dates, std_vals, start_dt, end_dt)
                    racmo_std_by_region[suf] = std_c
                else:
                    racmo_std_by_region[suf] = None
        else:
            if base_var not in mat:
                keys = sorted([k for k in mat.keys() if not k.startswith("__")])
                raise KeyError(f"Variable '{base_var}' not in {args.racmo_mat}. Keys: {keys}")

            racmo_vals = np.array(mat[base_var]).squeeze().astype(float)
            time_len = racmo_vals.size

            racmo_dates = _maybe_read_mat_dates(mat)
            if racmo_dates is None:
                start_r = _parse_dt(args.racmo_start)
                racmo_dates = [start_r + timedelta(days=i) for i in range(time_len)]

            racmo_times_c, racmo_prod_c = _crop_series(racmo_dates, racmo_vals, start_dt, end_dt)

            # std: prefer explicit var; else compute from Mean_all
            racmo_std = None
            if args.racmo_std_var and args.racmo_std_var in mat:
                cand = np.array(mat[args.racmo_std_var]).squeeze()
                if cand.ndim == 1 and cand.size == time_len:
                    racmo_std = cand.astype(float)
                    print(f"[INFO] Using RACMO std variable: {args.racmo_std_var}")

            if racmo_std is None and "Mean_all" in mat:
                try:
                    racmo_std = _std_from_mean_all_cell(mat["Mean_all"])
                    if racmo_std.size == time_len:
                        print("[INFO] Computed RACMO std from Mean_all cell.")
                    else:
                        print(f"[WARN] Mean_all std length {racmo_std.size} != mean length {time_len}")
                        racmo_std = None
                except Exception as e:
                    print(f"[WARN] Failed to compute RACMO std from Mean_all: {e}")
                    racmo_std = None

            if racmo_std is not None:
                _, racmo_std_prod_c = _crop_series(racmo_dates, racmo_std, start_dt, end_dt)
            else:
                racmo_std_prod_c = None

    # ---- Sohey ----
    sohey_times_c, sohey_prod_c, sohey_std_prod_c = None, None, None
    if args.sohey_nc:
        if args.amsr_mat is None:
            raise RuntimeError("To follow MATLAB for Sohey you must provide --amsr_mat AMSR2_2023.mat")

        sohey_mean, sohey_std = _compute_sohey_mean_std_like_matlab(
            sohey_nc=args.sohey_nc,
            amsr_mat=args.amsr_mat,
            sohey_var=args.sohey_var,
            thr=args.sohey_thr,
            coastal_mat=args.coastal_mat,
            coastal_var=args.coastal_var,
            coastal_month_offset=args.sohey_coastal_month_offset,
            debug=args.debug_sohey,
        )

        # Build dates for Sohey series
        T = int(np.array(sohey_mean).size)

        # Heuristic: if 6 or 8 slices -> treat as monthly means; else treat as daily starting May 1
        if args.sohey_expand_daily and T in (6, 8):
            if T == 6:
                months = list(range(5, 11))  # May..Oct
                days_in_block = [31, 30, 31, 31, 30, 31]
            else:
                months = list(range(3, 11))  # Mar..Oct
                days_in_block = [31, 30, 31, 30, 31, 31, 30, 31]

            vals = np.repeat(sohey_mean[:T], days_in_block).astype(float)
            sigs = np.repeat(sohey_std[:T], days_in_block).astype(float)
            base = datetime(args.sohey_year, months[0], 1)
            dates = [base + timedelta(days=i) for i in range(len(vals))]
            sohey_dates = dates
            sohey_vals = vals
            sohey_sigs = sigs
        else:
            if T >= 90:
                base = datetime(args.sohey_year, 5, 1)
                sohey_dates = [base + timedelta(days=i) for i in range(T)]
            else:
                if T == 6:
                    months = list(range(5, 11))
                elif T == 8:
                    months = list(range(3, 11))
                else:
                    months = list(range(1, T + 1))
                sohey_dates = [datetime(args.sohey_year, m, 15) for m in months]
            sohey_vals = np.array(sohey_mean, dtype=float)
            sohey_sigs = np.array(sohey_std, dtype=float)

        sohey_times_c, sohey_prod_c = _crop_series(sohey_dates, sohey_vals, start_dt, end_dt)
        _, sohey_std_prod_c = _crop_series(sohey_dates, sohey_sigs, start_dt, end_dt)

    # --------------------------
    # Compute HSSW (Sv)
    # --------------------------

    # SIT
    area_for_sit = align_area_to_dates(
        sit_times_c, amsr_days_c, amsr_area_m2_c, max_gap_days=args.max_area_gap_days
    )
    pi_sit = thickness_to_rate_mps(sit_mean_c, sit_times_c, args.sit_units)
    hssw_sit = hssw_from_production(
        pi_sit, area_for_sit,
        s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
        rho_i=args.rho_i, rho_hssw=args.rho_hssw,
        frazil_frac=args.frazil_frac,
    )

    # Propagate SIT std
    hssw_sit_lo = hssw_sit_hi = None
    if sit_std_c is not None and len(sit_std_c) == len(sit_mean_c):
        pi_lo = thickness_to_rate_mps(np.array(sit_mean_c) - np.array(sit_std_c), sit_times_c, args.sit_units)
        pi_hi = thickness_to_rate_mps(np.array(sit_mean_c) + np.array(sit_std_c), sit_times_c, args.sit_units)
        hssw_sit_lo = hssw_from_production(
            pi_lo, area_for_sit,
            s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
            rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
        )
        hssw_sit_hi = hssw_from_production(
            pi_hi, area_for_sit,
            s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
            rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
        )

    # RACMO (single region or per-polynya)
    hssw_racmo = hssw_racmo_lo = hssw_racmo_hi = None
    hssw_racmo_by_region = {}   # suffix -> array; only set when racmo_by_polynya
    hssw_racmo_lo_by_region = {}
    hssw_racmo_hi_by_region = {}

    if racmo_times_c is not None and len(racmo_times_c) > 0:
        area_for_racmo = align_area_to_dates(
            racmo_times_c, amsr_days_c, amsr_area_m2_c, max_gap_days=args.max_area_gap_days
        )

        if racmo_by_polynya:
            for suf in POLYNYA_SUFFIXES:
                prod_c = racmo_prod_by_region[suf]
                std_c = racmo_std_by_region.get(suf)
                pi_r = thickness_to_rate_mps(prod_c, racmo_times_c, args.racmo_units)
                hssw_racmo_by_region[suf] = hssw_from_production(
                    pi_r, area_for_racmo,
                    s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                    rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
                )
                if std_c is not None and len(std_c) == len(prod_c):
                    pi_lo = thickness_to_rate_mps(np.array(prod_c) - np.array(std_c), racmo_times_c, args.racmo_units)
                    pi_hi = thickness_to_rate_mps(np.array(prod_c) + np.array(std_c), racmo_times_c, args.racmo_units)
                    hssw_racmo_lo_by_region[suf] = hssw_from_production(
                        pi_lo, area_for_racmo,
                        s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                        rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
                    )
                    hssw_racmo_hi_by_region[suf] = hssw_from_production(
                        pi_hi, area_for_racmo,
                        s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                        rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
                    )
                else:
                    hssw_racmo_lo_by_region[suf] = None
                    hssw_racmo_hi_by_region[suf] = None
        else:
            if racmo_prod_c is not None:
                pi_racmo = thickness_to_rate_mps(racmo_prod_c, racmo_times_c, args.racmo_units)
                hssw_racmo = hssw_from_production(
                    pi_racmo, area_for_racmo,
                    s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                    rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
                )
                if racmo_std_prod_c is not None and len(racmo_std_prod_c) == len(racmo_prod_c):
                    pi_lo = thickness_to_rate_mps(np.array(racmo_prod_c) - np.array(racmo_std_prod_c), racmo_times_c, args.racmo_units)
                    pi_hi = thickness_to_rate_mps(np.array(racmo_prod_c) + np.array(racmo_std_prod_c), racmo_times_c, args.racmo_units)
                    hssw_racmo_lo = hssw_from_production(
                        pi_lo, area_for_racmo,
                        s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                        rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
                    )
                    hssw_racmo_hi = hssw_from_production(
                        pi_hi, area_for_racmo,
                        s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                        rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
                    )

    # Sohey
    hssw_sohey = hssw_sohey_lo = hssw_sohey_hi = None
    if sohey_times_c is not None and sohey_prod_c is not None and len(sohey_times_c) > 0:
        area_for_sohey = align_area_to_dates(
            sohey_times_c, amsr_days_c, amsr_area_m2_c, max_gap_days=args.max_area_gap_days
        )
        pi_sohey = thickness_to_rate_mps(sohey_prod_c, sohey_times_c, args.sohey_units)
        hssw_sohey = hssw_from_production(
            pi_sohey, area_for_sohey,
            s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
            rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
        )
        if sohey_std_prod_c is not None and len(sohey_std_prod_c) == len(sohey_prod_c):
            pi_lo = thickness_to_rate_mps(np.array(sohey_prod_c) - np.array(sohey_std_prod_c), sohey_times_c, args.sohey_units)
            pi_hi = thickness_to_rate_mps(np.array(sohey_prod_c) + np.array(sohey_std_prod_c), sohey_times_c, args.sohey_units)
            hssw_sohey_lo = hssw_from_production(
                pi_lo, area_for_sohey,
                s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
            )
            hssw_sohey_hi = hssw_from_production(
                pi_hi, area_for_sohey,
                s_hssw=args.s_hssw, s_lssw=args.s_lssw, delta_s=args.delta_s,
                rho_i=args.rho_i, rho_hssw=args.rho_hssw, frazil_frac=args.frazil_frac
            )

    # --------------------------
    # Plot
    # --------------------------

    color_sit = "#d62728"    # red
    color_sohey = "#2ca02c"  # green
    color_racmo = "k"        # black (single-region RACMO)
    # Per-polynya RACMO: same color, different markers
    color_racmo_polynya = "k"
    markers_racmo_by_region = {"RSP": "o", "TNBP": "s", "MSP": "^"}  # circle, square, triangle

    # Observational overlays colors
    color_tnb = (0.70, 0.60, 0.90)     # lavender band/line
    color_risp = (0.95, 0.75, 0.45)    # warm light band
    color_export = (77/255., 77/255., 255/255.)  # dark gray

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor((0.96, 0.96, 0.96))

    # --- Observational context overlays ---
    # TNB mooring annual-mean band
    if not args.no_tnb_band:
        y0 = args.mooring_mean - args.mooring_std
        y1 = args.mooring_mean + args.mooring_std
        ax.axhspan(y0, y1, color=color_tnb, alpha=0.22, zorder=0)
        ax.axhline(args.mooring_mean, linestyle=(0, (4, 4)), color=color_tnb, linewidth=3,
                   label=f"TNB mooring (annual) {args.mooring_mean:.2f}±{args.mooring_std:.2f} Sv", zorder=1)

    # RIS polynya formation band
    if not args.no_risp_band:
        lo = float(min(args.risp_low, args.risp_high))
        hi = float(max(args.risp_low, args.risp_high))
        ax.axhspan(lo, hi, color=color_risp, alpha=0.18, zorder=0,
                   label=f"RIS polynya formation {lo:.1f}–{hi:.1f} Sv")

    # Cape Adare export lines (only if --show_cape_adare)
    if args.show_cape_adare:
        ax.axhline(args.cape_adare_export, linestyle="--", color=color_export, linewidth=3, zorder=3,
                   label=f"Cape Adare HSSW export (annual) ~{args.cape_adare_export:.2f} Sv")
        if args.show_cape_adare_summer:
            ax.axhline(args.cape_adare_summer, linestyle=":", color=color_export, linewidth=3, zorder=3,
                       label=f"Cape Adare HSSW export (summer) ~{args.cape_adare_summer:.2f} Sv")

    # RACMO: either three polynyas (RSP, TNBP, MSP) or single region
    if racmo_by_polynya and hssw_racmo_by_region:
        for suf in POLYNYA_SUFFIXES:
            hh = hssw_racmo_by_region.get(suf)
            if hh is None:
                continue
            mk = markers_racmo_by_region.get(suf, "o")
            lab = f"RACMO {POLYNYA_LABELS.get(suf, suf)}"
            lo = hssw_racmo_lo_by_region.get(suf)
            hi = hssw_racmo_hi_by_region.get(suf)
            if lo is not None and hi is not None:
                ax.fill_between(
                    racmo_times_c, lo, hi,
                    color=color_racmo_polynya, alpha=0.10, linewidth=0.0, zorder=1
                )
            ax.plot(racmo_times_c, hh, color=color_racmo_polynya, linewidth=2.0, alpha=0.2, zorder=2)
            ax.scatter(
                racmo_times_c, hh, s=120, marker=mk,
                facecolors=color_racmo_polynya, edgecolors="none", alpha=0.5,
                label=lab, zorder=3
            )
    elif hssw_racmo is not None:
        if hssw_racmo_lo is not None and hssw_racmo_hi is not None:
            ax.fill_between(
                racmo_times_c, hssw_racmo_lo, hssw_racmo_hi,
                color=color_racmo, alpha=0.10, linewidth=0.0, zorder=1
            )
        ax.plot(racmo_times_c, hssw_racmo, color=color_racmo, linewidth=2.5, alpha=0.20, zorder=2)
        ax.scatter(
            racmo_times_c, hssw_racmo, s=150, marker="o",
            facecolors=color_racmo, edgecolors="none", alpha=0.45,
            label="RACMO2.4 flux", zorder=3
        )

    # Sohey (green, circles)
    if hssw_sohey is not None:
        if hssw_sohey_lo is not None and hssw_sohey_hi is not None:
            ax.fill_between(
                sohey_times_c, hssw_sohey_lo, hssw_sohey_hi,
                color=color_sohey, alpha=0.12, linewidth=0.0, zorder=1
            )
        ax.plot(sohey_times_c, hssw_sohey, color=color_sohey, linewidth=2.5, alpha=0.20, zorder=2)
        ax.scatter(
            sohey_times_c, hssw_sohey, s=150, marker="o",
            facecolors=color_sohey, edgecolors="none", alpha=0.45,
            label="PMW (AMSR2)", zorder=3
        )

    # # SIT (red, hexagons + optional vertical uncertainty)
    # if hssw_sit is not None:
    #     if hssw_sit_lo is not None and hssw_sit_hi is not None:
    #         ax.vlines(
    #             sit_times_c, hssw_sit_lo, hssw_sit_hi,
    #             color=color_sit, alpha=0.25, linewidth=3, zorder=2
    #         )
    #     ax.scatter(
    #         sit_times_c, hssw_sit, s=180, marker="h",
    #         facecolors=color_sit, edgecolors="none", alpha=0.45,
    #         label="IS2+SWOT", zorder=4
    #     )

    ax.set_ylabel("HSSW Production / Transport Rate (Sv)")
    ax.set_xlabel("Date")
    ax.set_ylim(0, args.ylim)

    # Grid: white on gray background
    ax.grid(True)
    ax.grid(color="white", linestyle="-", linewidth=1.0, alpha=1.0)
    ax.set_axisbelow(False)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(direction="out", length=5, width=0.8)

    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"[INFO] Saved figure: {args.out_png}")

    # --------------------------
    # Optional save
    # --------------------------
    if args.save_npz_out:
        out = {
            "amsr_days_str": np.array([str(t) for t in amsr_days_c], dtype=object),
            "amsr_area": np.array(amsr_area_c, dtype=float),
            "area_units": np.array([args.area_units], dtype=object),
            "hssw_params": np.array([
                f"s_hssw={args.s_hssw}",
                f"s_lssw={args.s_lssw}",
                f"delta_s={args.delta_s}",
                f"rho_i={args.rho_i}",
                f"rho_hssw={args.rho_hssw}",
                f"frazil_frac={args.frazil_frac}",
            ], dtype=object),
            "obs_overlays": np.array([
                f"tnb_mean={args.mooring_mean}",
                f"tnb_halfwidth={args.mooring_std}",
                f"risp_low={args.risp_low}",
                f"risp_high={args.risp_high}",
                f"cape_adare_export={args.cape_adare_export}",
                f"cape_adare_summer={args.cape_adare_summer}",
                f"show_cape_adare={args.show_cape_adare}",
                f"show_cape_adare_summer={args.show_cape_adare_summer}",
            ], dtype=object),
        }

        out["sit_times_str"] = np.array([str(t) for t in sit_times_c], dtype=object)
        out["hssw_sit"] = np.array(hssw_sit, dtype=float)
        if hssw_sit_lo is not None and hssw_sit_hi is not None:
            out["hssw_sit_lo"] = np.array(hssw_sit_lo, dtype=float)
            out["hssw_sit_hi"] = np.array(hssw_sit_hi, dtype=float)

        if racmo_times_c is not None:
            out["racmo_times_str"] = np.array([str(t) for t in racmo_times_c], dtype=object)
        if racmo_by_polynya and hssw_racmo_by_region:
            for suf in POLYNYA_SUFFIXES:
                if suf in hssw_racmo_by_region:
                    out[f"hssw_racmo_{suf}"] = np.array(hssw_racmo_by_region[suf], dtype=float)
                    if hssw_racmo_lo_by_region.get(suf) is not None and hssw_racmo_hi_by_region.get(suf) is not None:
                        out[f"hssw_racmo_{suf}_lo"] = np.array(hssw_racmo_lo_by_region[suf], dtype=float)
                        out[f"hssw_racmo_{suf}_hi"] = np.array(hssw_racmo_hi_by_region[suf], dtype=float)
        elif hssw_racmo is not None:
            out["hssw_racmo"] = np.array(hssw_racmo, dtype=float)
            if hssw_racmo_lo is not None and hssw_racmo_hi is not None:
                out["hssw_racmo_lo"] = np.array(hssw_racmo_lo, dtype=float)
                out["hssw_racmo_hi"] = np.array(hssw_racmo_hi, dtype=float)

        if hssw_sohey is not None:
            out["sohey_times_str"] = np.array([str(t) for t in sohey_times_c], dtype=object)
            out["hssw_sohey"] = np.array(hssw_sohey, dtype=float)
            if hssw_sohey_lo is not None and hssw_sohey_hi is not None:
                out["hssw_sohey_lo"] = np.array(hssw_sohey_lo, dtype=float)
                out["hssw_sohey_hi"] = np.array(hssw_sohey_hi, dtype=float)

        np.savez(args.save_npz_out, **out)
        print(f"[INFO] Saved HSSW series NPZ: {args.save_npz_out}")


if __name__ == "__main__":
    main()
