#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from datetime import datetime, timedelta

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

# Optional deps
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False


def _std_from_mean_all_cell(mean_all):
    """
    mean_all: MATLAB cell array loaded via scipy loadmat -> numpy object array
              shape typically (1,185) or (185,1)
    Returns: std_series (185,)
    """
    arr = np.array(mean_all).squeeze()

    # Ensure 1D list-like
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
            # sometimes item itself is an object wrapper
            vals = np.array(item).astype(float).ravel()

        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            std[i] = float(np.std(vals, ddof=1))  # MATLAB-like sample std
        elif vals.size == 1:
            std[i] = 0.0

    return std


# ----------------------------
# ESACCI Sea Ice Thickness Reading
# ----------------------------

def read_esacci_sit_nc(nc_path, sit_var=None, filter_quality=True, max_uncertainty=None):
    """
    Read ESACCI sea ice thickness netCDF file (1D swath/trajectory data).
    
    Parameters
    ----------
    nc_path : str
        Path to ESACCI netCDF file
    sit_var : str, optional
        Variable name for sea ice thickness. If None, will try to auto-detect.
        Common names: "sea_ice_thickness", "sit", "SIT", "thickness"
    filter_quality : bool
        If True, filter out measurements with MIZ bias flags (flag_miz == 2)
    max_uncertainty : float, optional
        Maximum allowed uncertainty (m). If provided, filters out measurements
        with uncertainty > max_uncertainty
    
    Returns
    -------
    sit : array
        Sea ice thickness (m), 1D array
    lat : array
        Latitude (degrees), 1D array
    lon : array
        Longitude (degrees), 1D array
    time : datetime
        Representative observation time (from filename or first measurement)
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available; cannot read ESACCI files.")
    
    nc = netCDF4.Dataset(nc_path, "r")
    available_vars = list(nc.variables.keys())
    
    # Auto-detect sea ice thickness variable if not provided
    if sit_var is None or sit_var not in nc.variables:
        # Try common variable names
        candidates = ["sea_ice_thickness", "sit", "SIT", "thickness", "ice_thickness", 
                     "sea_ice_thickness_corrected", "sea_ice_thickness_uncorrected"]
        sit_var = None
        for cand in candidates:
            if cand in nc.variables:
                sit_var = cand
                break
        
        if sit_var is None:
            # Look for variables containing "thickness" or "sit"
            for var_name in available_vars:
                var_lower = var_name.lower()
                if "thickness" in var_lower or (var_lower == "sit"):
                    sit_var = var_name
                    break
        
        if sit_var is None:
            nc.close()
            raise KeyError(
                f"Could not find sea ice thickness variable in {nc_path}. "
                f"Available variables: {available_vars}"
            )
    
    # Read sea ice thickness
    sit_var_obj = nc.variables[sit_var]
    sit_raw = sit_var_obj[:]
    
    if np.ma.isMaskedArray(sit_raw):
        sit = sit_raw.filled(np.nan).astype(float)
    else:
        sit = np.array(sit_raw, dtype=float)
    
    # Handle fill values
    fill_value = getattr(sit_var_obj, "_FillValue", None)
    if fill_value is not None:
        sit[sit == float(fill_value)] = np.nan
    
    # Read coordinates (1D swath data)
    lat = None
    lon = None
    for lat_name in ["latitude", "lat", "Latitude", "Lat"]:
        if lat_name in nc.variables:
            lat_raw = nc.variables[lat_name][:]
            if np.ma.isMaskedArray(lat_raw):
                lat = lat_raw.filled(np.nan).astype(float)
            else:
                lat = np.array(lat_raw, dtype=float)
            break
    
    for lon_name in ["longitude", "lon", "Longitude", "Lon"]:
        if lon_name in nc.variables:
            lon_raw = nc.variables[lon_name][:]
            if np.ma.isMaskedArray(lon_raw):
                lon = lon_raw.filled(np.nan).astype(float)
            else:
                lon = np.array(lon_raw, dtype=float)
            break
    
    # Read time variable (seconds since 1970-01-01)
    time = None
    if "time" in nc.variables:
        time_var = nc.variables["time"]
        time_units = getattr(time_var, "units", None)
        if time_units:
            try:
                # ESACCI uses "seconds since 1970-01-01"
                # Get first time value as representative
                time_vals = time_var[:]
                if len(time_vals) > 0:
                    time_val = netCDF4.num2date(time_vals[0], time_units)
                    if isinstance(time_val, datetime):
                        time = time_val
                    else:
                        time = datetime.fromisoformat(str(time_val).replace("Z", "+00:00"))
            except Exception:
                pass
    
    # Fallback: try to parse from filename (YYYYMMDD format)
    if time is None:
        match = re.search(r"(\d{8})", os.path.basename(nc_path))
        if match:
            date_str = match.group(1)
            try:
                time = datetime.strptime(date_str, "%Y%m%d")
            except Exception:
                pass
    
    # Quality filtering
    if filter_quality and "flag_miz" in nc.variables:
        flag_miz = np.array(nc.variables["flag_miz"][:], dtype=int)
        # flag_miz: 0=not_in_miz, 1=miz_no_bias_detected, 2=miz_bias_detected
        # Filter out measurements with bias detected (flag == 2)
        valid_mask = (flag_miz != 2)
        sit = sit[valid_mask]
        lat = lat[valid_mask]
        lon = lon[valid_mask]
    
    # Uncertainty filtering
    if max_uncertainty is not None:
        uncertainty_var = None
        for unc_name in ["sea_ice_thickness_uncertainty", "sit_uncertainty", 
                        "thickness_uncertainty"]:
            if unc_name in nc.variables:
                uncertainty_var = unc_name
                break
        
        if uncertainty_var is not None:
            unc_raw = nc.variables[uncertainty_var][:]
            if np.ma.isMaskedArray(unc_raw):
                uncertainty = unc_raw.filled(np.nan).astype(float)
            else:
                uncertainty = np.array(unc_raw, dtype=float)
            
            if filter_quality and "flag_miz" in nc.variables:
                uncertainty = uncertainty[valid_mask]
            
            valid_unc = np.isfinite(uncertainty) & (uncertainty <= max_uncertainty)
            sit = sit[valid_unc]
            lat = lat[valid_unc]
            lon = lon[valid_unc]
    
    nc.close()
    
    if lat is None or lon is None:
        raise RuntimeError(f"Could not find lat/lon in {nc_path}")
    
    # Ensure 1D arrays (ESACCI is swath/trajectory data)
    sit = np.array(sit, dtype=float).ravel()
    lat = np.array(lat, dtype=float).ravel()
    lon = np.array(lon, dtype=float).ravel()
    
    # Ensure same length
    min_len = min(len(sit), len(lat), len(lon))
    if min_len < len(sit):
        sit = sit[:min_len]
    if min_len < len(lat):
        lat = lat[:min_len]
    if min_len < len(lon):
        lon = lon[:min_len]
    
    return sit, lat, lon, time


def compute_esacci_daily_polynya_mean(
    esacci_files, amsr_mat, polynya_threshold=0.7, 
    filter_quality=True, max_uncertainty=None, debug=False
):
    """
    Compute daily average sea ice thickness from ESACCI files within polynya region.
    
    Parameters
    ----------
    esacci_files : list of str
        List of paths to ESACCI netCDF files (can include wildcards or patterns)
    amsr_mat : str
        Path to AMSR MAT file containing polynya mask
    polynya_threshold : float
        SIC threshold for polynya mask (< threshold = polynya)
    filter_quality : bool
        If True, filter out measurements with MIZ bias flags (flag_miz == 2)
    max_uncertainty : float, optional
        Maximum allowed uncertainty (m) for filtering
    debug : bool
        Print debug information
    
    Returns
    -------
    dates : list of datetime
        Dates for each daily average
    sit_mean : array
        Mean sea ice thickness (m) within polynya for each day
    sit_std : array
        Standard deviation of sea ice thickness (m) within polynya for each day
    sit_count : array
        Number of valid observations for each day
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available.")
    if not HAS_SCIPY_INTERP:
        raise RuntimeError("scipy.interpolate not available (need LinearNDInterpolator).")
    
    # Load AMSR polynya mask
    amsr = load_mat_any(amsr_mat)
    if "sic" not in amsr or "lat" not in amsr or "lon" not in amsr:
        raise KeyError("AMSR MAT must contain sic, lat, lon")
    
    sic = np.array(amsr["sic"], dtype=float)
    lat_amsr = np.array(amsr["lat"], dtype=float)
    lon_amsr = np.array(amsr["lon"], dtype=float)
    
    # Fix sic dimension if needed: accept [Y,X,365] -> [365,Y,X]
    if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
        sic = np.moveaxis(sic, -1, 0)
    
    # MATLAB does lat=lat'; lon=lon'
    if lat_amsr.shape != sic.shape[1:]:
        if lat_amsr.T.shape == sic.shape[1:]:
            lat_amsr = lat_amsr.T
            lon_amsr = lon_amsr.T
        else:
            raise ValueError(f"AMSR lat shape {lat_amsr.shape} does not match sic spatial {sic.shape[1:]}")
    
    # Create daily polynya mask (SIC < threshold)
    sic_frac = sic / 100.0  # Convert percentage to fraction
    polynya_mask_daily = sic_frac < polynya_threshold  # [365, Y, X]
    
    # Group ESACCI files by date
    import glob
    import os
    from collections import defaultdict
    
    # Expand wildcards
    all_files = []
    for pattern in esacci_files:
        expanded = glob.glob(pattern)
        if expanded:
            all_files.extend(expanded)
        else:
            all_files.append(pattern)  # Assume it's a specific file
    
    # Group by date
    files_by_date = defaultdict(list)
    for fpath in all_files:
        # Extract date from filename
        match = re.search(r"(\d{8})", os.path.basename(fpath))
        if match:
            date_str = match.group(1)
            try:
                date = datetime.strptime(date_str, "%Y%m%d")
                files_by_date[date].append(fpath)
            except Exception:
                if debug:
                    print(f"[WARN] Could not parse date from {fpath}")
    
    if len(files_by_date) == 0:
        raise RuntimeError(f"No valid ESACCI files found from patterns: {esacci_files}")
    
    dates = sorted(files_by_date.keys())
    sit_mean = []
    sit_std = []
    sit_count = []
    
    for date in dates:
        files = files_by_date[date]
        
        # Get day of year (1-365, non-leap)
        day_of_year = date.timetuple().tm_yday
        if day_of_year > 365:
            day_of_year = 365  # Handle leap years
        
        day_idx = day_of_year - 1  # 0-indexed
        
        # Get polynya mask for this day
        if day_idx < polynya_mask_daily.shape[0]:
            polynya_mask = polynya_mask_daily[day_idx, :, :]  # [Y, X]
        else:
            polynya_mask = polynya_mask_daily[-1, :, :]  # Use last day if out of range
        
        # Collect all valid thickness values within polynya from all files for this date
        all_sit_values = []
        
        for fpath in files:
            try:
                # Read ESACCI swath data (1D trajectory)
                sit, lat_esacci, lon_esacci, time = read_esacci_sit_nc(
                    fpath, 
                    filter_quality=filter_quality,
                    max_uncertainty=max_uncertainty
                )
                
                # Ensure 1D arrays (ESACCI is swath/trajectory data)
                sit = np.array(sit, dtype=float).ravel()
                lat_esacci = np.array(lat_esacci, dtype=float).ravel()
                lon_esacci = np.array(lon_esacci, dtype=float).ravel()
                
                # Check we have valid data
                if len(sit) == 0 or len(lat_esacci) == 0 or len(lon_esacci) == 0:
                    if debug:
                        print(f"[WARN] Empty data in {os.path.basename(fpath)}")
                    continue
                
                # Harmonize longitude conventions
                lon_amsr_harmonized, lon_esacci_harmonized = _harmonize_lon(
                    lon_amsr, lon_esacci
                )
                
                # Filter valid points
                valid = np.isfinite(lat_esacci) & np.isfinite(lon_esacci_harmonized) & np.isfinite(sit)
                if np.count_nonzero(valid) < 10:
                    if debug:
                        print(f"[WARN] Too few valid points ({np.count_nonzero(valid)}) in {os.path.basename(fpath)}")
                    continue
                
                # Extract valid swath points
                lat_valid = lat_esacci[valid]
                lon_valid = lon_esacci_harmonized[valid]
                sit_valid = sit[valid]
                
                # Interpolate swath data to AMSR grid using LinearNDInterpolator
                F = LinearNDInterpolator(
                    np.column_stack([lon_valid, lat_valid]),
                    sit_valid,
                    fill_value=np.nan
                )
                
                # Interpolate to AMSR grid points
                sit_interp = F(lon_amsr_harmonized, lat_amsr)
                
                # Mask to polynya region (SIC < threshold)
                sit_polynya = sit_interp[polynya_mask & np.isfinite(sit_interp)]
                sit_polynya = sit_polynya[np.isfinite(sit_polynya)]
                
                if sit_polynya.size > 0:
                    all_sit_values.extend(sit_polynya.tolist())
                    if debug:
                        print(f"[ESACCI] {os.path.basename(fpath)}: {sit_polynya.size} polynya points, "
                              f"mean={np.mean(sit_polynya):.3f} m")
                
            except Exception as e:
                if debug:
                    print(f"[WARN] Error reading {fpath}: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # Compute statistics for this day
        all_sit_values = np.array(all_sit_values, dtype=float)
        all_sit_values = all_sit_values[np.isfinite(all_sit_values)]
        
        if all_sit_values.size > 0:
            sit_mean.append(float(np.mean(all_sit_values)))
            sit_std.append(float(np.std(all_sit_values, ddof=1)) if all_sit_values.size > 1 else 0.0)
            sit_count.append(int(all_sit_values.size))
        else:
            sit_mean.append(np.nan)
            sit_std.append(np.nan)
            sit_count.append(0)
        
        if debug:
            print(f"[ESACCI] {date.strftime('%Y-%m-%d')}: n={sit_count[-1]}, "
                  f"mean={sit_mean[-1]:.3f} m, std={sit_std[-1]:.3f} m")
    
    return dates, np.array(sit_mean, dtype=float), np.array(sit_std, dtype=float), np.array(sit_count, dtype=int)


def find_esacci_files(base_dir, year=None, debug=False):
    """
    Automatically find ESACCI sea ice thickness files in a directory.
    
    Searches recursively for three ESACCI sources:
    - CryoSat-2: ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-*.nc
    - Sentinel-3A: ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3A-SH-*.nc
    - Sentinel-3B: ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3B-SH-*.nc
    
    Parameters
    ----------
    base_dir : str
        Base directory to search for ESACCI files (searches recursively)
    year : int, optional
        If provided, only search for files from this year (YYYY format in filename)
    debug : bool
        Print debug information
    
    Returns
    -------
    files : list of str
        List of all found ESACCI file paths
    """
    import glob
    import os
    
    if not os.path.isdir(base_dir):
        if debug:
            print(f"[WARN] Directory does not exist: {base_dir}")
        return []
    
    # ESACCI file patterns (search recursively)
    patterns = [
        "**/ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-*.nc",
        "**/ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3A-SH-*.nc",
        "**/ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3B-SH-*.nc",
    ]
    
    all_files = []
    
    # Search recursively in base directory
    for pattern in patterns:
        # Use recursive glob (requires Python 3.5+)
        files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        all_files.extend(files)
        
        # Fallback: also search in base directory directly (non-recursive)
        pattern_base = os.path.basename(pattern.replace("**/", ""))
        files_base = glob.glob(os.path.join(base_dir, pattern_base))
        all_files.extend(files_base)
    
    # Filter by year if specified
    if year is not None:
        year_str = str(year)
        filtered_files = []
        for fpath in all_files:
            # Extract date from filename (YYYYMMDD format)
            match = re.search(r"(\d{8})", os.path.basename(fpath))
            if match:
                date_str = match.group(1)
                if date_str.startswith(year_str):
                    filtered_files.append(fpath)
        all_files = filtered_files
    
    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))
    
    if debug:
        print(f"[ESACCI] Found {len(all_files)} files in {base_dir}")
        if len(all_files) > 0:
            # Count by source
            cryosat2 = sum(1 for f in all_files if "CRYOSAT2" in f or "SIRAL_CRYOSAT2" in f)
            s3a = sum(1 for f in all_files if "SENTINEL3A" in f or "SRAL_SENTINEL3A" in f)
            s3b = sum(1 for f in all_files if "SENTINEL3B" in f or "SRAL_SENTINEL3B" in f)
            print(f"[ESACCI]   CryoSat-2: {cryosat2}, Sentinel-3A: {s3a}, Sentinel-3B: {s3b}")
            if len(all_files) <= 10:
                print(f"[ESACCI]   Sample files: {[os.path.basename(f) for f in all_files[:5]]}")
    
    return all_files


def _h5_read_dataset(ds):
    """
    MATLAB v7.3 stores arrays in HDF5; when read by h5py, dimensions are typically reversed
    relative to MATLAB display. Convert by reversing axes (equivalent to MATLAB ordering).
    """
    arr = np.array(ds)

    # scalars
    if arr.shape == ():
        return arr

    # reverse axes to match MATLAB
    if arr.ndim >= 2:
        arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
    return arr


def load_mat_any(path):
    path = str(path)

    # Try scipy first (v7.2 and earlier)
    if loadmat is not None:
        try:
            return loadmat(path)
        except NotImplementedError:
            pass
        except Exception:
            pass

    # v7.3 fallback
    try:
        import h5py
    except Exception:
        raise RuntimeError(
            "MATLAB v7.3 file detected. Install h5py: `conda install h5py` or `pip install h5py`."
        )

    def _h5_read_dataset(ds):
        arr = np.array(ds)
        if arr.shape == ():
            return arr
        if arr.ndim >= 2:
            arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
        return arr

    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                key = name.split("/")[-1]  # keep basename like MATLAB var
                # avoid overwriting; if duplicate basename, keep full path
                if key in out:
                    out[name] = _h5_read_dataset(obj)
                else:
                    out[key] = _h5_read_dataset(obj)

        f.visititems(visit)

    return out


def _guess_std_key(mat, mean_key, target_len):
    """
    Find a likely std variable in a MAT dict:
    - name contains std/sigma/stdev
    - length matches mean series length
    """
    import re
    cand = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if not re.search(r"(std|sigma|stdev)", k, re.IGNORECASE):
            continue
        try:
            arr = np.array(v).squeeze()
            if arr.ndim == 1 and arr.size == target_len:
                cand.append(k)
        except Exception:
            continue
    return cand[0] if cand else None



# ----------------------------
# Helpers
# ----------------------------
def _to_str_scalar(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return str(x.item())
    return str(x)


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


def _crop_series(dates, values, start_dt=None, end_dt=None, *more):
    """Crop by date range. Optional extra array(s) (same length as dates) use the same mask."""
    if start_dt is None and end_dt is None:
        if not more:
            return dates, values
        return (dates, values) + more
    dd = np.array(dates)
    vv = np.array(values)
    mask = np.ones(dd.shape, dtype=bool)
    if start_dt is not None:
        mask &= (dd >= start_dt)
    if end_dt is not None:
        mask &= (dd <= end_dt)
    out = [dd[mask].tolist(), vv[mask]]
    for m in more:
        out.append(np.asarray(m)[mask])
    return tuple(out)


def load_swot_production_from_eulerian_csv(
    csv_path,
    prefer_covered=False,
    prefer_covered_raw=False,
    allow_whole_fallback_when_prefer_covered=False,
    days_per_month=30.0,
    debug=False,
    closure_p_thermo_only=False,
):
    """
    Load SWOT-IS2 residual thermodynamic production from Eulerian closure or
    eulerian_mass_constraint_decomposition CSV (same column names).

    Default (prefer_covered=False): if the CSV contains
    ice_production_m_month_area_mean (e.g. regional_track_mean_lumped_decomposition.py),
    that column is used directly as m/month. Otherwise uses P_thermo_cm_day_area_mean
    (whole polynya) and converts cm/day -> m/month. Covered columns renormalize by
    thickness-covered area and are not interchangeable with whole-area or RACMO means.

    If closure_p_thermo_only=True, never use ice_production_m_month_area_mean; always
    resolve P_thermo from closure columns (aligned with plot_closure_budget_decomposition).

    Returns
    -------
    times : list[datetime]
    prod_m_month : np.ndarray
        Production in m/month (converted from cm/day or m/s columns).
    err_lo, err_hi, err_sym : np.ndarray
        Optional uncertainty on m/month (regional lumped CSV). NaN if absent.
        Asymmetric: lower/upper from pooled thickness p25/p75; symmetric from sample std.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"SWOT production CSV not found: {csv_path}")

    times = []
    vals = []
    err_lo = []
    err_hi = []
    err_sym = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col_cm_day_whole = "P_thermo_cm_day_area_mean"
        col_cm_day_cov = "P_thermo_cm_day_area_mean_covered"
        col_cm_day_cov_raw = "P_thermo_cm_day_area_mean_covered_raw"
        col_ms_whole = "P_thermo_m_s_area_mean"
        col_ms_cov = "P_thermo_m_s_area_mean_covered"
        col_ms_cov_raw = "P_thermo_m_s_area_mean_covered_raw"
        col_m_month_export = "ice_production_m_month_area_mean"

        for row in reader:
            t_str = row.get("time_center", None)
            if t_str is None:
                continue
            try:
                tt = _parse_dt(t_str)
            except Exception:
                continue

            # Precomputed m/month (regional lumped script); only when not asking for covered columns
            if (
                (not closure_p_thermo_only)
                and (not prefer_covered)
                and (not prefer_covered_raw)
                and (col_m_month_export in cols)
            ):
                try:
                    prod_mm = float(row.get(col_m_month_export, "nan"))
                except Exception:
                    prod_mm = np.nan
                if np.isfinite(prod_mm):
                    times.append(tt)
                    vals.append(prod_mm)
                    try:
                        el = float(row.get("ice_production_m_month_err_lower", "nan"))
                    except Exception:
                        el = np.nan
                    try:
                        eu = float(row.get("ice_production_m_month_err_upper", "nan"))
                    except Exception:
                        eu = np.nan
                    try:
                        es = float(row.get("ice_production_m_month_err_symmetric", "nan"))
                    except Exception:
                        es = np.nan
                    err_lo.append(el)
                    err_hi.append(eu)
                    err_sym.append(es)
                    if debug:
                        print(
                            f"[SWOT_PROD] {tt.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"source=ice_production_m_month_area_mean prod={prod_mm}"
                        )
                    continue

            # Preferred source order:
            # - strict covered mode: covered only (optionally fallback to whole if explicitly enabled)
            # - default mode: whole first
            v = np.nan
            used_source = "none"
            row_valid_flag = row.get("covered_product_valid", None)
            row_valid_cov = np.nan
            if row_valid_flag is not None:
                try:
                    row_valid_cov = int(float(row_valid_flag))
                except Exception:
                    row_valid_cov = np.nan

            if prefer_covered and (col_cm_day_cov in cols):
                try:
                    v = float(row.get(col_cm_day_cov, "nan"))
                    used_source = "covered_cm_day"
                except Exception:
                    v = np.nan
            if (not np.isfinite(v)) and prefer_covered_raw and (col_cm_day_cov_raw in cols):
                try:
                    v = float(row.get(col_cm_day_cov_raw, "nan"))
                    used_source = "covered_cm_day_raw"
                except Exception:
                    v = np.nan
            if (not np.isfinite(v)) and (not prefer_covered) and (col_cm_day_whole in cols):
                try:
                    v = float(row.get(col_cm_day_whole, "nan"))
                    used_source = "whole_cm_day"
                except Exception:
                    v = np.nan
            if (not np.isfinite(v)) and prefer_covered and allow_whole_fallback_when_prefer_covered and (col_cm_day_whole in cols):
                try:
                    v = float(row.get(col_cm_day_whole, "nan"))
                    used_source = "whole_cm_day_fallback"
                except Exception:
                    v = np.nan
            if np.isfinite(v):
                # cm/day -> m/month
                prod = (v / 100.0) * float(days_per_month)
            else:
                v_ms = np.nan
                if prefer_covered and (col_ms_cov in cols):
                    try:
                        v_ms = float(row.get(col_ms_cov, "nan"))
                        used_source = "covered_m_s"
                    except Exception:
                        v_ms = np.nan
                if (not np.isfinite(v_ms)) and prefer_covered_raw and (col_ms_cov_raw in cols):
                    try:
                        v_ms = float(row.get(col_ms_cov_raw, "nan"))
                        used_source = "covered_m_s_raw"
                    except Exception:
                        v_ms = np.nan
                if (not np.isfinite(v_ms)) and (not prefer_covered) and (col_ms_whole in cols):
                    try:
                        v_ms = float(row.get(col_ms_whole, "nan"))
                        used_source = "whole_m_s"
                    except Exception:
                        v_ms = np.nan
                if (not np.isfinite(v_ms)) and prefer_covered and allow_whole_fallback_when_prefer_covered and (col_ms_whole in cols):
                    try:
                        v_ms = float(row.get(col_ms_whole, "nan"))
                        used_source = "whole_m_s_fallback"
                    except Exception:
                        v_ms = np.nan
                prod = v_ms * 86400.0 * float(days_per_month) if np.isfinite(v_ms) else np.nan

            times.append(tt)
            vals.append(prod)
            err_lo.append(np.nan)
            err_hi.append(np.nan)
            err_sym.append(np.nan)
            if debug:
                print(f"[SWOT_PROD] {tt.strftime('%Y-%m-%d %H:%M:%S')} source={used_source} covered_valid={row_valid_cov} prod={prod}")

    if len(times) == 0:
        raise RuntimeError(f"No valid rows parsed from SWOT production CSV: {csv_path}")

    order = np.argsort(np.array(times))
    times = [times[i] for i in order]
    vals = np.array([vals[i] for i in order], dtype=float)
    err_lo = np.array([err_lo[i] for i in order], dtype=float)
    err_hi = np.array([err_hi[i] for i in order], dtype=float)
    err_sym = np.array([err_sym[i] for i in order], dtype=float)

    if debug:
        n_fin = int(np.sum(np.isfinite(vals)))
        print(f"[SWOT_PROD] Loaded {len(vals)} rows, finite={n_fin}, days_per_month={days_per_month}")
    return times, vals, err_lo, err_hi, err_sym


def load_swot_storage_only_from_csv(
    csv_path,
    rho_i=915.0,
    days_per_month=30.0,
    debug=False,
):
    """
    Load storage-only production proxy from closure CSV:
      P_storage = dMdt / (rho_i * A_polynya)
    returned in m/month at time_center.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"SWOT storage CSV not found: {csv_path}")

    times = []
    vals = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_str = row.get("time_center", None)
            if t_str is None:
                continue
            try:
                tt = _parse_dt(t_str)
            except Exception:
                continue
            try:
                dmdt = float(row.get("dMdt_kg_s", "nan"))
                area = float(row.get("A_polynya_m2", "nan"))
            except Exception:
                dmdt, area = np.nan, np.nan
            if np.isfinite(dmdt) and np.isfinite(area) and area > 0:
                v = (dmdt / (float(rho_i) * area)) * 86400.0 * float(days_per_month)
            else:
                v = np.nan
            times.append(tt)
            vals.append(v)
            if debug:
                print(
                    f"[SWOT_STORAGE] {tt.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"dMdt={dmdt:.6g} area={area:.6g} v_m_month={v:.6g}"
                )

    if len(times) == 0:
        raise RuntimeError(f"No valid rows parsed for storage-only from CSV: {csv_path}")
    order = np.argsort(np.array(times))
    times = [times[i] for i in order]
    vals = np.array([vals[i] for i in order], dtype=float)
    return times, vals


def load_swot_simple_apparent_from_csv(
    csv_path,
    rho_i=915.0,
    days_per_month=30.0,
    area_norm="mean",
    debug=False,
):
    """
    Simple apparent production proxy from adjacent closure rows:
      p_simple = (M_i - M_{i-1}) / (rho_i * A_ref * dt)
    where A_ref is one of:
      - mean: 0.5*(A_i + A_{i-1}) [default, stable/symmetric]
      - a2:   A_i
      - a1:   A_{i-1}

    Returns values at current row time_center (i).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"SWOT simple CSV not found: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in CSV: {csv_path}")

    parsed = []
    for r in rows:
        try:
            tt = _parse_dt(r.get("time_center", ""))
        except Exception:
            continue
        try:
            m = float(r.get("M_polynya_kg", "nan"))
        except Exception:
            m = np.nan
        try:
            a = float(r.get("A_polynya_m2", "nan"))
        except Exception:
            a = np.nan
        parsed.append((tt, m, a))
    if len(parsed) < 2:
        raise RuntimeError("Need at least 2 valid rows for simple apparent production.")
    parsed.sort(key=lambda x: x[0])

    times = []
    vals = []
    for i in range(1, len(parsed)):
        t1, m1, a1 = parsed[i - 1]
        t2, m2, a2 = parsed[i]
        dt = (t2 - t1).total_seconds()
        if dt <= 0:
            continue
        if area_norm == "a2":
            aref = a2
        elif area_norm == "a1":
            aref = a1
        else:
            aref = 0.5 * (a1 + a2)
        if np.isfinite(m1) and np.isfinite(m2) and np.isfinite(aref) and aref > 0:
            v = ((m2 - m1) / (float(rho_i) * aref * dt)) * 86400.0 * float(days_per_month)
        else:
            v = np.nan
        times.append(t2)
        vals.append(v)
        if debug:
            print(
                f"[SWOT_SIMPLE] {t2.strftime('%Y-%m-%d %H:%M:%S')} "
                f"dm={m2-m1:.6g} aref={aref:.6g} dt={dt:.6g}s v_m_month={v:.6g}"
            )
    if len(times) == 0:
        raise RuntimeError("No finite adjacent intervals for simple apparent production.")
    return times, np.asarray(vals, dtype=float)


def load_swot_fout_kg_s_from_csv(csv_path):
    """Load (time_center, Fout_kg_s) from Eulerian closure CSV."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"SWOT CSV not found: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in CSV: {csv_path}")
    out = []
    for r in rows:
        try:
            tt = _parse_dt(r.get("time_center", ""))
        except Exception:
            continue
        v = np.nan
        for k in ("Fout_kg_s", "Fout_boundary_transport_proxy_kg_s"):
            try:
                vv = float(r.get(k, "nan"))
            except Exception:
                vv = np.nan
            if np.isfinite(vv):
                v = vv
                break
        out.append((tt, v))
    if len(out) == 0:
        raise RuntimeError("No valid time rows for Fout in CSV.")
    out.sort(key=lambda x: x[0])
    times = [t for t, _ in out]
    vals = np.asarray([v for _, v in out], dtype=float)
    return times, vals


def _nearest_series_within_hours(query_times, src_times, src_vals, max_hours=36.0):
    """Nearest-neighbor sampling from src onto query, only if within max_hours."""
    qn = len(query_times)
    out = np.full(qn, np.nan, dtype=float)
    max_sec = float(max_hours) * 3600.0
    sv = np.asarray(src_vals, dtype=float)
    for i, qt in enumerate(query_times):
        best_j = None
        best_dt = None
        for j, st in enumerate(src_times):
            dt = _time_delta_seconds(qt, st)
            if dt <= max_sec and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_j = j
        if best_j is not None and best_j < sv.size:
            out[i] = sv[best_j]
    return out


def _interp_amsr_area_km2_at_times(q_times, amsr_days, amsr_area_km2, extrapolate_edges=True):
    """
    Linear interpolation of AMSR polynya area (km²) onto query times.

    If extrapolate_edges is True (default), times before the first AMSR sample or after the last
    use the nearest edge value instead of NaN. That avoids wiping M = ρ h A on the first/last
    calendar days of the plot window when the daily grid extends slightly beyond AMSR dates —
    a common cause of sparse pooled-simple production curves.
    """
    amsr_days = list(amsr_days)
    ya = np.asarray(amsr_area_km2, dtype=float)
    if len(amsr_days) < 2:
        return np.full(len(q_times), np.nan, dtype=float)
    ts = mdates.date2num(np.asarray(amsr_days))
    idx = np.argsort(ts)
    ts = ts[idx]
    ya = ya[idx]
    tq = mdates.date2num(np.asarray(q_times))
    if extrapolate_edges:
        left = float(ya[0]) if np.isfinite(ya[0]) else np.nan
        right = float(ya[-1]) if np.isfinite(ya[-1]) else np.nan
        return np.interp(tq, ts, ya, left=left, right=right)
    return np.interp(tq, ts, ya, left=np.nan, right=np.nan)


def _try_read_sit_lon_lat_from_npz(z, n_expect=None):
    """
    Read per-segment longitude/latitude from an NPZ dict (same order as sit_times_str).
    Tries common key pairs; returns (lon, lat) or (None, None).
    """
    files = getattr(z, "files", None) or list(z.keys())
    pairs = (
        ("sit_lon", "sit_lat"),
        ("seg_lon", "seg_lat"),
        ("lon_sit", "lat_sit"),
    )
    for lk, ak in pairs:
        if lk in files and ak in files:
            lon = np.asarray(z[lk], dtype=float).ravel()
            lat = np.asarray(z[ak], dtype=float).ravel()
            if n_expect is not None and (lon.size != n_expect or lat.size != n_expect):
                continue
            if lon.size == lat.size:
                return lon, lat
    if "lon" in files and "lat" in files:
        lon = np.asarray(z["lon"], dtype=float).ravel()
        lat = np.asarray(z["lat"], dtype=float).ravel()
        if n_expect is not None and (lon.size != n_expect or lat.size != n_expect):
            return None, None
        if lon.size == lat.size:
            return lon, lat
    return None, None


def _parse_datetime_from_segment_filename(basename):
    """
    Parse a timestamp from an IS2 scene NPZ filename (same patterns as drift_aware_mass_budget).
    """
    s = str(basename)
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


def _segment_lon_lat_centroid_from_is2_scene_npz(path):
    """
    One representative (lon, lat) per scene NPZ, mirroring drift_aware_mass_budget._segment_means_from_scene_npz.
    """
    try:
        d = np.load(path, allow_pickle=True)
    except Exception:
        return None, None
    if not all(k in d for k in ("lat", "lon", "sit")):
        return None, None
    lat = np.asarray(d["lat"], dtype=float).ravel()
    lon = np.asarray(d["lon"], dtype=float).ravel()
    sit = np.asarray(d["sit"], dtype=float).ravel()
    if "in_scope" in d:
        mscope = np.asarray(d["in_scope"], dtype=bool).ravel()
    elif "in_polynya" in d:
        mscope = np.asarray(d["in_polynya"], dtype=bool).ravel()
    else:
        mscope = np.ones_like(sit, dtype=bool)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(sit) & mscope
    if not np.any(valid):
        return None, None
    lat = lat[valid]
    lon = lon[valid]
    if "seg_joint" in d:
        seg = np.asarray(d["seg_joint"], dtype=float).ravel()
        seg = seg[valid]
        good_seg = np.isfinite(seg) & (seg >= 0)
        if np.any(good_seg):
            seg = seg[good_seg].astype(np.int64)
            lat = lat[good_seg]
            lon = lon[good_seg]
            u = np.unique(seg)
            out_lat = np.full(u.size, np.nan, dtype=float)
            out_lon = np.full(u.size, np.nan, dtype=float)
            for i, sid in enumerate(u):
                m = seg == sid
                out_lon[i] = np.nanmean(lon[m])
                out_lat[i] = np.nanmean(lat[m])
            lon = out_lon
            lat = out_lat
    return float(np.nanmean(lon)), float(np.nanmean(lat))


def _load_sit_lon_lat_from_is2_scene_dir(root_dir, sit_times, glob_pat="*_is2_polynya_sit_outputs.npz", max_match_hours=72.0):
    """
    Build per-segment lon/lat aligned to plot NPZ sit_times by matching each segment time to the
    nearest scene NPZ under root_dir (same convention as drift_aware --segment_npz directory).
    """
    files = sorted(glob.glob(os.path.join(root_dir, glob_pat)))
    if not files:
        files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
    if not files:
        print(
            f"[WARN] Lagrangian: no NPZ files in {root_dir} (tried glob {glob_pat!r} and *.npz). "
            "drift_aware expects names like *_is2_polynya_sit_outputs.npz"
        )
        return None, None
    scenes = []
    for f in files:
        tn = _parse_datetime_from_segment_filename(os.path.basename(f))
        if tn is not None:
            scenes.append((tn, f))
    if not scenes:
        print(f"[WARN] Lagrangian: no scene NPZ filenames with parseable dates under {root_dir}")
        return None, None
    scenes.sort(key=lambda x: x[0])
    max_sec = float(max_match_hours) * 3600.0
    n = len(sit_times)
    out_lon = np.full(n, np.nan, dtype=float)
    out_lat = np.full(n, np.nan, dtype=float)
    for i, t in enumerate(sit_times):
        tt = t if isinstance(t, datetime) else _parse_dt(str(t))
        best_d = None
        best_f = None
        for tn, fp in scenes:
            ds = abs((tn - tt).total_seconds())
            if best_d is None or ds < best_d:
                best_d = ds
                best_f = fp
        if best_d is not None and best_d <= max_sec:
            lo, la = _segment_lon_lat_centroid_from_is2_scene_npz(best_f)
            if lo is not None and la is not None:
                out_lon[i], out_lat[i] = lo, la
    nh = int(np.sum(np.isfinite(out_lon)))
    print(
        f"[INFO] Lagrangian IS2 scene directory: matched lon/lat for {nh}/{n} segment times "
        f"(nearest scene NPZ within {max_match_hours:g} h; {len(scenes)} scenes found)."
    )
    if nh == 0:
        return None, None
    return out_lon, out_lat


def _segment_time_merge_key(t):
    """Stable string key for matching segment NPZ sit_times_str to main sit_times."""
    if isinstance(t, datetime):
        dt = t
    else:
        dt = _parse_dt(str(t))
    dt = dt.replace(microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _load_sit_lon_lat_from_segment_npz_files(paths, sit_times):
    """
    Load per-segment lon/lat from one or more NPZ files.

    Modes (do not mix plain and time-keyed files in one run — plain files are skipped if any
    time-keyed file is present):

    1) **Concatenation:** every file has only sit_lon/sit_lat (no sit_times_str). Arrays are
       concatenated in the order paths are given; total length must equal ``len(sit_times)``.

    2) **Time merge:** any file contains ``sit_times_str`` with the same length as lon/lat.
       All such rows are merged into a lookup (later files overwrite duplicate times), then aligned
       to main ``sit_times`` by matching ``_segment_time_merge_key``. Segments with no match get
       NaN lon/lat (Lagrangian step skips them).

    paths : list of str
        Existing filesystem paths to .npz files.
    sit_times : list
        Main plot NPZ segment times (datetime-like), length n_seg.
    """
    n_seg = len(sit_times)
    if not paths or n_seg == 0:
        return None, None

    print("[INFO] Lagrangian segment geometry: reading " + ", ".join(paths))

    by_key = {}
    concat_lons = []
    concat_lats = []
    files_with_ts = []
    files_plain = []

    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Could not load segment NPZ {p}: {e}")
            continue
        files = getattr(z, "files", None) or list(z.keys())
        lon, lat = _try_read_sit_lon_lat_from_npz(z, n_expect=None)
        if lon is None or lat is None:
            print(f"[WARN] No sit_lon/sit_lat (or seg_lon/seg_lat / lon/lat) in {p}; skipping.")
            continue
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        if lon.size != lat.size:
            print(f"[WARN] lon/lat length mismatch in {p}; skipping.")
            continue
        if "sit_times_str" in files:
            ts_raw = z["sit_times_str"]
            if len(ts_raw) != len(lon):
                print(f"[WARN] sit_times_str length != lon/lat in {p}; skipping.")
                continue
            for i in range(len(lon)):
                k = _segment_time_merge_key(_parse_dt(str(ts_raw[i])))
                by_key[k] = (float(lon[i]), float(lat[i]))
            files_with_ts.append(p)
        else:
            concat_lons.append(lon)
            concat_lats.append(lat)
            files_plain.append(p)

    if files_with_ts and files_plain:
        print(
            "[WARN] Mixture of NPZ files with and without sit_times_str: using only time-keyed files. "
            "Skipped plain lon/lat files: " + ", ".join(files_plain)
        )

    if files_with_ts:
        out_lon = np.full(n_seg, np.nan, dtype=float)
        out_lat = np.full(n_seg, np.nan, dtype=float)
        for i, t in enumerate(sit_times):
            k = _segment_time_merge_key(t)
            if k in by_key:
                out_lon[i], out_lat[i] = by_key[k]
        n_hit = int(np.sum(np.isfinite(out_lon)))
        if n_hit == 0:
            print("[WARN] No segment lon/lat matched main sit_times from time-keyed NPZ file(s).")
            return None, None
        print(
            f"[INFO] Matched segment lon/lat for {n_hit}/{n_seg} segments from time-keyed NPZ file(s)."
        )
        return out_lon, out_lat

    if concat_lons:
        lon = np.concatenate(concat_lons)
        lat = np.concatenate(concat_lats)
        if lon.size != n_seg or lat.size != n_seg:
            print(
                f"[WARN] Concatenated segment NPZ lon/lat length {lon.size} != main sit_times length {n_seg}."
            )
            return None, None
        print(
            f"[INFO] Loaded segment lon/lat by concatenating {len(files_plain)} NPZ file(s) (plain arrays)."
        )
        return lon, lat

    return None, None


def _pooled_sit_lagrangian_on_query_times(
    query_times,
    sample_times,
    h,
    lon,
    lat,
    lag_days=7.0,
    stat="mean",
    band=False,
    ice_motion_nc=None,
    amsr_mat=None,
    step_hours=24.0,
    ice_motion_subsample=5,
):
    """
    Pool SIT onto query_times using a Lagrangian rule: for each sample at (t_j, lon_j, lat_j, h_j),
    advect the parcel position from t_j to each query time t_q with |t_j - t_q| <= lag_days,
    and aggregate thickness h_j at t_q (thickness follows the parcel — no re-interpolation in space).

    Requires drift_aware_mass_budget._advect_points_between_times and ice_motion_nc.
    """
    try:
        import drift_aware_mass_budget as _dam
    except Exception as e:
        raise RuntimeError(
            "Lagrangian pooled SIT needs drift_aware_mass_budget (same folder as this script). "
            f"Import failed: {e}"
        ) from e

    advect = _dam._advect_points_between_times
    h = np.asarray(h, dtype=float)
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    if lon.shape != lat.shape or lon.shape != h.shape:
        raise ValueError("lon, lat, h must have the same length for Lagrangian pooling.")

    n = len(query_times)
    out_c = np.full(n, np.nan, dtype=float)
    out_lo = np.full(n, np.nan, dtype=float)
    out_hi = np.full(n, np.nan, dtype=float)
    half = timedelta(days=float(lag_days))

    def _as_dt(x):
        if isinstance(x, datetime):
            return x
        return _parse_dt(str(x))

    qlist = [_as_dt(t) for t in query_times]
    slist = [_as_dt(t) for t in sample_times]

    for i in range(n):
        tq = qlist[i]
        vals = []
        for j in range(len(slist)):
            if not (-half <= (slist[j] - tq) <= half):
                continue
            if not np.isfinite(h[j]) or not np.isfinite(lon[j]) or not np.isfinite(lat[j]):
                continue
            try:
                lon_e, lat_e = advect(
                    float(lon[j]),
                    float(lat[j]),
                    slist[j],
                    tq,
                    ice_motion_nc,
                    step_hours=float(step_hours),
                    amsr_mat=amsr_mat,
                    fill_nan_velocity_zero_on_ocean=True,
                    ice_motion_subsample=int(ice_motion_subsample),
                )
                if not (np.isfinite(lon_e) and np.isfinite(lat_e)):
                    continue
            except Exception:
                continue
            vals.append(float(h[j]))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        if stat == "median":
            out_c[i] = float(np.nanmedian(arr))
        else:
            out_c[i] = float(np.nanmean(arr))
        if band:
            out_lo[i] = float(np.nanpercentile(arr, 25))
            out_hi[i] = float(np.nanpercentile(arr, 75))
    return out_c, out_lo, out_hi


def compute_swot_simple_from_pooled_sit_npz(
    sit_times,
    sit_mean,
    amsr_days,
    amsr_area_km2,
    lag_days,
    pooled_stat,
    rho_i,
    days_per_month,
    area_norm="mean",
    min_dt_hours=24.0,
    diff_step=1,
    query_times=None,
    debug=False,
    sit_pooled_lagrangian=False,
    sit_lon=None,
    sit_lat=None,
    ice_motion_nc=None,
    amsr_mat_for_drift=None,
    lagrangian_step_hours=24.0,
    lagrangian_ice_motion_subsample=5,
    min_area_km2=None,
    report_diagnostics=True,
    extrapolate_amsr_area_edges=True,
):
    """
    Simple apparent production from NPZ only:
      M_i = rho_i * h_pooled_i * A_i   with h_pooled from ±lag window over segments,
      A_i = AMSR polynya area (km²) interpolated to each segment time, converted to m².
      Rate between times separated by diff_step samples (sorted):
      (M_i - M_{i-diff_step}) / (rho_i * A_ref * dt) -> m/month.
      Pairs with dt < min_dt_hours are skipped to avoid unrealistically large rates from very short
      segment-time gaps.

    This matches the user's (A2*h2 - A1*h1)/A_ref/dt idea when M = rho * h * A.

    When sit_pooled_lagrangian is True and sit_lon, sit_lat, ice_motion_nc are provided,
    h_pooled uses ice-motion advection (parcel thickness at calendar day t) instead of
    Eulerian temporal pooling only.

    If min_area_km2 is set, AMSR area (km²) is set to NaN below that floor before forming M = ρ h A,
    matching the storage/residual pooled-daily path.
    """
    if query_times is None:
        query_times = sit_times
    if (
        sit_pooled_lagrangian
        and sit_lon is not None
        and sit_lat is not None
        and ice_motion_nc is not None
        and str(ice_motion_nc).strip()
    ):
        hp, _, _ = _pooled_sit_lagrangian_on_query_times(
            query_times,
            sit_times,
            sit_mean,
            sit_lon,
            sit_lat,
            lag_days=float(lag_days),
            stat=pooled_stat,
            band=False,
            ice_motion_nc=ice_motion_nc,
            amsr_mat=amsr_mat_for_drift,
            step_hours=float(lagrangian_step_hours),
            ice_motion_subsample=int(lagrangian_ice_motion_subsample),
        )
    else:
        hp, _, _ = _pooled_sit_on_query_times(
            query_times, sit_times, sit_mean, lag_days=float(lag_days), stat=pooled_stat, band=False
        )
    Akm = _interp_amsr_area_km2_at_times(
        query_times,
        amsr_days,
        amsr_area_km2,
        extrapolate_edges=extrapolate_amsr_area_edges,
    )
    if min_area_km2 is not None and float(min_area_km2) > 0:
        floor = float(min_area_km2)
        Akm = np.where(np.isfinite(Akm) & (Akm >= floor), Akm, np.nan)
    A_m2 = Akm * 1e6
    M_kg = np.asarray(rho_i, dtype=float) * np.asarray(hp, dtype=float) * A_m2
    idx = np.argsort(mdates.date2num(np.asarray(query_times)))
    t_list = [query_times[i] for i in idx]
    M = M_kg[idx]
    A2 = A_m2[idx]
    times_out = []
    vals_out = []
    n_skip_short = 0
    n_skip_bad_dt = 0
    n_nan_mass = 0
    n_nan_aref = 0
    min_dt_sec = float(min_dt_hours) * 3600.0
    step = max(1, int(diff_step))
    hp_arr = np.asarray(hp, dtype=float)
    Akm_arr = np.asarray(Akm, dtype=float)
    for i in range(step, len(t_list)):
        t1, t2 = t_list[i - step], t_list[i]
        m1, m2 = M[i - step], M[i]
        a1, a2 = A2[i - step], A2[i]
        dt = (t2 - t1).total_seconds()
        if dt <= 0:
            n_skip_bad_dt += 1
            continue
        if dt < min_dt_sec:
            n_skip_short += 1
            continue
        if area_norm == "a2":
            aref = a2
        elif area_norm == "a1":
            aref = a1
        else:
            aref = 0.5 * (a1 + a2)
        if np.isfinite(m1) and np.isfinite(m2) and np.isfinite(aref) and aref > 0:
            v = ((m2 - m1) / (float(rho_i) * aref * dt)) * 86400.0 * float(days_per_month)
        else:
            v = np.nan
            if not (np.isfinite(m1) and np.isfinite(m2)):
                n_nan_mass += 1
            else:
                n_nan_aref += 1
        times_out.append(t2)
        vals_out.append(v)
        if debug:
            print(
                f"[SWOT_SIMPLE_POOLED] {t2} dm={m2-m1:.6g} aref={aref:.6g} v_m_month={v:.6g}"
            )
    if not times_out:
        raise RuntimeError(
            f"No finite intervals for pooled-sit simple production (all pairs had dt < {min_dt_hours:g} h or invalid values)."
        )
    if debug and n_skip_short > 0:
        print(
            f"[SWOT_SIMPLE_POOLED] skipped {n_skip_short} adjacent pairs with dt < {min_dt_hours:g} h."
        )
    vals_arr = np.asarray(vals_out, dtype=float)
    if report_diagnostics:
        n_q = len(t_list)
        n_hp = int(np.sum(np.isfinite(hp_arr)))
        n_akm = int(np.sum(np.isfinite(Akm_arr)))
        n_fin_v = int(np.sum(np.isfinite(vals_arr)))
        n_pairs = len(vals_arr)
        floor_note = ""
        if min_area_km2 is not None and float(min_area_km2) > 0:
            floor_note = f" AMSR area floor {float(min_area_km2):g} km² applied before M=ρhA."
        edge_note = " AMSR area: edge extrapolation on." if extrapolate_amsr_area_edges else " AMSR area: strict edges (NaN outside span)."
        print(
            "[INFO] SWOT pooled-simple (red) derivative diagnostics — "
            f"daily grid: {n_q} days; finite pooled h: {n_hp}/{n_q}; finite AMSR area (after floor): {n_akm}/{n_q};"
            f"{floor_note}{edge_note} "
            f"diff_step={step} day(s) → {n_pairs} interval end-times t2; "
            f"finite d(Ah)/dt rate: {n_fin_v}/{n_pairs}. "
            f"Skipped: dt≤0 or dt<min_dt: {n_skip_short + n_skip_bad_dt} "
            f"(min_dt={min_dt_hours:g} h); NaN rate from missing M: {n_nan_mass}; from bad A_ref: {n_nan_aref}. "
            "Gaps in the line = NaN rates (mass/area/interpolation); "
            "use --swot_prod_plot_signed if ≤0 rates are masked; "
            "try smaller --swot_pooled_diff_days or lower --swot_pooled_min_area_km2 for more points."
        )
    return times_out, vals_arr


def _to_date_str(t):
    """Normalize datetime/date to YYYY-MM-DD string."""
    s = str(t).split()[0].split("T")[0]
    return s[:10] if len(s) >= 10 else s


def _time_delta_seconds(a, b):
    """Absolute time difference in seconds (robust to datetime / numpy datetime64)."""
    try:
        import pandas as pd

        return abs(float((pd.Timestamp(a) - pd.Timestamp(b)).total_seconds()))
    except Exception:
        pass
    if isinstance(a, datetime) and isinstance(b, datetime):
        return abs((a - b).total_seconds())
    da = np.datetime64(a, "s")
    db = np.datetime64(b, "s")
    return float(np.abs(da - db) / np.timedelta64(1, "s"))


def _align_swot_production_to_sit_times(
    sit_times,
    swot_dates,
    vals,
    elo,
    ehi,
    esym,
    max_hours=72.0,
):
    """
    For each IS2+SWOT scatter time, copy the nearest closure production (and errors)
    if within max_hours. Plots production at the same x positions as SIT (still m/month on y).
    """
    n = len(sit_times)
    out_v = np.full(n, np.nan, dtype=float)
    out_lo = np.full(n, np.nan, dtype=float)
    out_hi = np.full(n, np.nan, dtype=float)
    out_sym = np.full(n, np.nan, dtype=float)
    if swot_dates is None or len(swot_dates) == 0:
        return out_v, out_lo, out_hi, out_sym
    vals = np.asarray(vals, dtype=float)
    elo = np.asarray(elo, dtype=float) if elo is not None else None
    ehi = np.asarray(ehi, dtype=float) if ehi is not None else None
    esym = np.asarray(esym, dtype=float) if esym is not None else None
    max_sec = float(max_hours) * 3600.0
    for i, st in enumerate(sit_times):
        best_j = None
        best_dt = None
        for j, sd in enumerate(swot_dates):
            dt = _time_delta_seconds(st, sd)
            if dt <= max_sec and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_j = j
        if best_j is None:
            continue
        out_v[i] = vals[best_j]
        if elo is not None and best_j < elo.size:
            out_lo[i] = elo[best_j]
        if ehi is not None and best_j < ehi.size:
            out_hi[i] = ehi[best_j]
        if esym is not None and best_j < esym.size:
            out_sym[i] = esym[best_j]
    return out_v, out_lo, out_hi, out_sym


def _pooled_sit_track_thickness(times, h, lag_days=3.0, stat="mean", band=False):
    """
    For each segment time, pool all IS2+SWOT thickness samples with |t_j - t_i| <= lag_days.
    Returns (pooled_center, p25, p75) same length as inputs; NaN where no finite samples in window.
    """
    h = np.asarray(h, dtype=float)
    n = len(times)
    out_c = np.full(n, np.nan, dtype=float)
    out_lo = np.full(n, np.nan, dtype=float)
    out_hi = np.full(n, np.nan, dtype=float)
    half = timedelta(days=float(lag_days))

    def _as_dt(x):
        if isinstance(x, datetime):
            return x
        return _parse_dt(str(x))

    tlist = [_as_dt(t) for t in times]

    for i in range(n):
        ti = tlist[i]
        vals = []
        for j in range(n):
            if -half <= (tlist[j] - ti) <= half and np.isfinite(h[j]):
                vals.append(float(h[j]))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        if stat == "median":
            out_c[i] = float(np.nanmedian(arr))
        else:
            out_c[i] = float(np.nanmean(arr))
        if band:
            out_lo[i] = float(np.nanpercentile(arr, 25))
            out_hi[i] = float(np.nanpercentile(arr, 75))
    return out_c, out_lo, out_hi


def _pooled_sit_on_query_times(query_times, sample_times, h, lag_days=3.0, stat="mean", band=False):
    """
    Pool SIT from sample_times onto arbitrary query_times using |t_sample - t_query| <= lag_days.
    Returns (pooled_center, p25, p75) with same length as query_times.
    """
    h = np.asarray(h, dtype=float)
    n = len(query_times)
    out_c = np.full(n, np.nan, dtype=float)
    out_lo = np.full(n, np.nan, dtype=float)
    out_hi = np.full(n, np.nan, dtype=float)
    half = timedelta(days=float(lag_days))

    def _as_dt(x):
        if isinstance(x, datetime):
            return x
        return _parse_dt(str(x))

    qlist = [_as_dt(t) for t in query_times]
    slist = [_as_dt(t) for t in sample_times]

    for i in range(n):
        tq = qlist[i]
        vals = []
        for j in range(len(slist)):
            if -half <= (slist[j] - tq) <= half and np.isfinite(h[j]):
                vals.append(float(h[j]))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        if stat == "median":
            out_c[i] = float(np.nanmedian(arr))
        else:
            out_c[i] = float(np.nanmean(arr))
        if band:
            out_lo[i] = float(np.nanpercentile(arr, 25))
            out_hi[i] = float(np.nanpercentile(arr, 75))
    return out_c, out_lo, out_hi


def _daily_time_grid(start_dt, end_dt):
    """Daily datetime grid at 00:00 from start_dt to end_dt inclusive."""
    s = datetime(start_dt.year, start_dt.month, start_dt.day)
    e = datetime(end_dt.year, end_dt.month, end_dt.day)
    out = []
    t = s
    while t <= e:
        out.append(t)
        t += timedelta(days=1)
    return out


def _pooled_sit_window_sample_counts(times, h, lag_days=3.0):
    """
    For each segment time i, count how many segment samples j have finite h[j] with
    |t_j - t_i| <= lag_days (same rule as _pooled_sit_track_thickness).

    Returns
    -------
    n_in_window : np.ndarray, shape (n,)
        Integer count per scatter time; 0 where no finite neighbors in window.
    """
    h = np.asarray(h, dtype=float)
    n = len(times)
    out_n = np.zeros(n, dtype=int)
    half = timedelta(days=float(lag_days))

    def _as_dt(x):
        if isinstance(x, datetime):
            return x
        return _parse_dt(str(x))

    tlist = [_as_dt(t) for t in times]
    for i in range(n):
        ti = tlist[i]
        c = 0
        for j in range(n):
            if -half <= (tlist[j] - ti) <= half and np.isfinite(h[j]):
                c += 1
        out_n[i] = c
    return out_n


def _print_pooled_sit_sample_count_info(sit_times_c, sit_mean_c, lag_days, pooled_stat):
    """Log how many raw segments are finite vs how many enter each ±lag_days pool."""
    if sit_times_c is None or len(sit_times_c) == 0:
        return
    span_days = 2.0 * float(lag_days)
    n_seg = len(sit_times_c)
    n_fin_raw = int(np.sum(np.isfinite(np.asarray(sit_mean_c, dtype=float))))
    n_win = _pooled_sit_window_sample_counts(sit_times_c, sit_mean_c, lag_days=lag_days)
    n_def = int(np.sum(n_win > 0))
    if n_def > 0:
        nz = n_win[n_win > 0]
        nmin, nmed, nmax = int(np.min(nz)), float(np.median(nz)), int(np.max(nz))
        pool_stats = f"min={nmin}, median={nmed:.1f}, max={nmax}"
    else:
        pool_stats = "n/a"
    print(
        f"[INFO] Pooled SIT windows: ±{lag_days:g} d half-width (≈{span_days:g} d span per scatter time); "
        f"aggregator={pooled_stat}."
    )
    print(
        f"[INFO] Pooled SIT counts — raw: {n_fin_raw}/{n_seg} segment rows with finite thickness; "
        f"scatter times with ≥1 finite sample in window: {n_def}/{n_seg}."
    )
    print(
        f"[INFO] Pool size (finite segment samples counted per window), where non-empty: {pool_stats}."
    )


def _mask_swot_prod_nonpos(v, elo, ehi, esym, plot_signed):
    """
    For figures, omit SWOT-IS2 production where rate <= 0 unless plot_signed.
    Zeros out matching error bars. Returns arrays (copies when masked).
    """
    v = np.asarray(v, dtype=float)
    if plot_signed:
        return v, elo, ehi, esym
    v2 = np.where(np.isfinite(v) & (v > 0), v, np.nan)
    if elo is not None:
        elo = np.asarray(elo, dtype=float)
        elo = np.where(np.isfinite(v2), elo, np.nan)
    if ehi is not None:
        ehi = np.asarray(ehi, dtype=float)
        ehi = np.where(np.isfinite(v2), ehi, np.nan)
    if esym is not None:
        esym = np.asarray(esym, dtype=float)
        esym = np.where(np.isfinite(v2), esym, np.nan)
    return v2, elo, ehi, esym


def _value_on_date(dates_list, values_list, target_dt):
    """
    Return value from (dates_list, values_list) on the same calendar day as target_dt.
    If no exact date match, return np.nan.
    """
    if dates_list is None or values_list is None or len(dates_list) == 0:
        return np.nan
    target_str = _to_date_str(target_dt)
    for i, d in enumerate(dates_list):
        if _to_date_str(d) == target_str:
            v = values_list[i] if i < len(values_list) else np.nan
            try:
                return float(v) if np.isfinite(v) else np.nan
            except Exception:
                return np.nan
    return np.nan


def _align_series_by_date(times_a, vals_a, times_b, vals_b):
    """Align two (time, value) series by date (YYYY-MM-DD). Returns (a_alg, b_alg) same length."""
    times_a = np.asarray(times_a).ravel()
    times_b = np.asarray(times_b).ravel()
    vals_a = np.asarray(vals_a, dtype=float).ravel()
    vals_b = np.asarray(vals_b, dtype=float).ravel()

    def to_date_str(t):
        s = str(t).split()[0].split("T")[0]
        return s[:10] if len(s) >= 10 else s

    dates_a = [to_date_str(t) for t in times_a]
    dates_b = [to_date_str(t) for t in times_b]
    set_b = set(dates_b)
    a_alg, b_alg = [], []
    for i, da in enumerate(dates_a):
        if da in set_b:
            j = dates_b.index(da)
            a_alg.append(vals_a[i])
            b_alg.append(vals_b[j])
    return np.array(a_alg), np.array(b_alg)


def _corr_and_ci(x, y, n_bootstrap=2000):
    """Pearson r and 95% CI via bootstrap. x, y must be same length."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size != y.size:
        return np.nan, (np.nan, np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return np.nan, (np.nan, np.nan)
    r = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng()
    r_bs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, x.size, size=x.size)
        r_bs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    r_bs = np.array(r_bs)
    ci = (float(np.percentile(r_bs, 2.5)), float(np.percentile(r_bs, 97.5)))
    return r, ci


def _print_series_stats(name, vals, unit="m"):
    """Print mean, s.d., n, min, max for a 1D series."""
    v = np.asarray(vals, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"  {name}: no valid values")
        return
    n = v.size
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    mn, mx = float(np.min(v)), float(np.max(v))
    print(f"  {name}: mean = {mean:.4f} {unit}, s.d. = {std:.4f}, n = {n}")
    print(f"         min = {mn:.4f}, max = {mx:.4f} {unit}")


def _format_log_time(t):
    """Log string for a plot time (include clock time so same-calendar-day SIT samples are distinct)."""
    if isinstance(t, datetime):
        return t.strftime("%Y-%m-%d %H:%M")
    s = str(t).replace("T", " ")
    return s[:16] if len(s) >= 16 else _to_date_str(t)


def _parse_regime_span(spec):
    """Parse 'START,END,LABEL' (label may contain commas if quoted)."""
    spec = str(spec).strip()
    if not spec:
        raise ValueError("empty regime span")
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 3:
        raise ValueError(f"regime span needs START,END,LABEL: {spec!r}")
    start_s, end_s = parts[0], parts[1]
    label = ",".join(parts[2:]).strip().strip("'\"")
    return _parse_dt(start_s), _parse_dt(end_s), label


def _resolve_fig4_regime_spans(args):
    """Regime shading is opt-in via --regime_span only (describe regimes in manuscript text)."""
    if not args.regime_span:
        return []
    return [_parse_regime_span(spec) for spec in args.regime_span]


def _add_fig4_time_annotations(
    *,
    regime_axes=None,
    snapshot_axes=None,
    regime_spans=(),
    snapshot_times=None,
    regime_colors=("#b3d9ff", "#ffd9b3"),
    regime_alpha=0.28,
    snapshot_color="#7a7a7a",
    snapshot_ls="--",
    snapshot_lw=0.75,
    snapshot_alpha=0.32,
    label_y=0.97,
):
    """Optional regime shading + faint SWOT–IS2 snapshot vertical markers."""
    from matplotlib.lines import Line2D

    regime_axes = [ax for ax in (regime_axes or []) if ax is not None]
    snapshot_axes = [ax for ax in (snapshot_axes or []) if ax is not None]
    if not regime_axes and not snapshot_axes:
        return None

    if regime_axes and regime_spans:
        x_lo = min(ax.get_xlim()[0] for ax in regime_axes)
        x_hi = max(ax.get_xlim()[1] for ax in regime_axes)
        for i, (t0, t1, lab) in enumerate(regime_spans):
            t0n = mdates.date2num(t0)
            t1n = mdates.date2num(t1)
            if t1n <= t0n or t1n < x_lo or t0n > x_hi:
                continue
            color = regime_colors[i % len(regime_colors)]
            for ax in regime_axes:
                ax.axvspan(
                    t0,
                    t1,
                    color=color,
                    alpha=regime_alpha,
                    linewidth=0.0,
                    zorder=0,
                )
                t_mid = t0 + (t1 - t0) / 2
                ax.text(
                    t_mid,
                    label_y,
                    lab,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="#333333",
                    alpha=0.92,
                    zorder=6,
                    clip_on=True,
                )

    snapshot_handle = None
    if snapshot_axes and snapshot_times is not None and len(snapshot_times) > 0:
        x_lo = min(ax.get_xlim()[0] for ax in snapshot_axes)
        x_hi = max(ax.get_xlim()[1] for ax in snapshot_axes)
        seen = set()
        snap_unique = []
        for t in snapshot_times:
            if not isinstance(t, datetime):
                continue
            key = t.strftime("%Y-%m-%d %H:%M")
            if key in seen:
                continue
            seen.add(key)
            snap_unique.append(t)
        if snap_unique:
            for ax in snapshot_axes:
                for t in snap_unique:
                    tn = mdates.date2num(t)
                    if tn < x_lo or tn > x_hi:
                        continue
                    ax.axvline(
                        t,
                        color=snapshot_color,
                        linestyle=snapshot_ls,
                        linewidth=snapshot_lw,
                        alpha=snapshot_alpha,
                        zorder=1,
                    )
            snapshot_handle = Line2D(
                [0],
                [0],
                color=snapshot_color,
                linestyle=snapshot_ls,
                linewidth=snapshot_lw,
                alpha=snapshot_alpha,
                label="SWOT–IS-2 snapshot",
            )

    return snapshot_handle


def _print_swot_production_rows(times, vals, label="SWOT-IS2 production", times_note=None):
    """Print per-row SWOT production values before plotting."""
    if times is None or vals is None or len(times) == 0:
        print(f"[{label}] no rows to print")
        return
    v = np.asarray(vals, dtype=float)
    if times_note:
        print(f"[{label}] values used for plotting (m/month); {times_note}")
    else:
        print(f"[{label}] values used for plotting (m/month):")
    for t, x in zip(times, v):
        t_str = _format_log_time(t)
        if np.isfinite(x):
            print(f"  {t_str}: {x:.6f}")
        else:
            print(f"  {t_str}: NaN")
    m = np.isfinite(v)
    if np.any(m):
        print(
            f"[{label}] finite summary: n={int(np.count_nonzero(m))}, "
            f"mean={float(np.nanmean(v)):.6f}, median={float(np.nanmedian(v)):.6f}, "
            f"range=[{float(np.nanmin(v)):.6f}, {float(np.nanmax(v)):.6f}]"
        )
    else:
        print(f"[{label}] finite summary: n=0")


def print_sit_racmo_sohey_statistics(
    sit_times, sit_mean, sit_std,
    racmo_times, racmo_vals, racmo_std,
    sohey_times, sohey_mean, sohey_std,
    esacci_times, esacci_mean, esacci_std,
    n_bootstrap=2000,
):
    """Print thickness/production statistics and pairwise correlations for the plot series."""
    print()
    print("=" * 60)
    print("SIT / RACMO / Sohey / ESACCI statistics (for manuscript)")
    print("=" * 60)

    if sit_times is not None and sit_mean is not None and len(sit_times) > 0:
        _print_series_stats("IS2+SWOT (SIT)", sit_mean, "m")
    if racmo_times is not None and racmo_vals is not None and len(racmo_times) > 0:
        _print_series_stats("RACMO (ice production)", racmo_vals, "m/month")
    if sohey_times is not None and sohey_mean is not None and len(sohey_times) > 0:
        _print_series_stats("Sohey PMW (HI), ice production", sohey_mean, "m/month")
    if esacci_times is not None and esacci_mean is not None and len(esacci_times) > 0:
        _print_series_stats("ESACCI", esacci_mean, "m")

    print()
    print("Pairwise correlations (r, 95% CI), aligned by date where lengths differ:")

    series = []
    if sit_times is not None and sit_mean is not None and len(sit_times) > 0:
        series.append(("SIT", sit_times, np.asarray(sit_mean, dtype=float)))
    if racmo_times is not None and racmo_vals is not None and len(racmo_times) > 0:
        series.append(("RACMO", racmo_times, np.asarray(racmo_vals, dtype=float)))
    if sohey_times is not None and sohey_mean is not None and len(sohey_times) > 0:
        series.append(("Sohey", sohey_times, np.asarray(sohey_mean, dtype=float)))
    if esacci_times is not None and esacci_mean is not None and len(esacci_times) > 0:
        series.append(("ESACCI", esacci_times, np.asarray(esacci_mean, dtype=float)))

    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            name_a, times_a, vals_a = series[i]
            name_b, times_b, vals_b = series[j]
            if len(vals_a) != len(vals_b):
                va, vb = _align_series_by_date(times_a, vals_a, times_b, vals_b)
                if len(va) < 3:
                    continue
                r, ci = _corr_and_ci(va, vb, n_bootstrap)
                print(f"  {name_a} vs {name_b}: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})  (n = {len(va)} aligned dates)")
            else:
                r, ci = _corr_and_ci(vals_a, vals_b, n_bootstrap)
                print(f"  {name_a} vs {name_b}: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})")
    print()


def _nanmean_safe(x, axis=None):
    x = np.asarray(x)
    with np.errstate(invalid="ignore"):
        if axis is None:
            if x.size == 0 or not np.any(np.isfinite(x)):
                return np.nan
            return np.nanmean(x)
        # axis=0 etc.: avoid "Mean of empty slice" when a slice has no finite values
        return np.apply_along_axis(
            lambda s: np.nan if s.size == 0 or not np.any(np.isfinite(s)) else float(np.nanmean(s)),
            axis,
            x,
        )


def _as_2d_latlon(lat, lon, target_shape):
    """
    Ensure lat/lon are 2D and match target_shape (Y,X). Try transpose if needed.
    """
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
            raise ValueError(f"lat/lon shape {lat2d.shape} does not match target spatial shape {target_shape}")
    return lat2d, lon2d


def _compute_monthly_mean_sic(sic_daily):
    """
    MATLAB: sic is [365, Y, X] (non-leap).
    Output: sic_monthly [Y, X, 12]
    """
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Y, X = sic_daily.shape[1], sic_daily.shape[2]
    out = np.full((Y, X, 12), np.nan, dtype=float)
    idx = 0
    for m, nd in enumerate(days_in_month):
        out[:, :, m] = _nanmean_safe(sic_daily[idx:idx + nd, :, :], axis=0)
        idx += nd
    return out


def _harmonize_lon(lon_a, lon_b):
    """
    Make lon_a and lon_b compatible:
    - If one is [-180,180] and the other is [0,360], convert the negative one to 0..360.
    """
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
    """
    MATLAB:
      [ii,jj] = find(lat between min/max lat_sohey AND lon between min/max lon_sohey)
      then cut min(ii):max(ii), min(jj):max(jj)
    """
    lat_min, lat_max = np.nanmin(lat_sohey), np.nanmax(lat_sohey)
    lon_min, lon_max = np.nanmin(lon_sohey), np.nanmax(lon_sohey)

    in_box = (lat_amsr >= lat_min) & (lat_amsr <= lat_max) & (lon_amsr >= lon_min) & (lon_amsr <= lon_max)
    ii, jj = np.where(in_box)
    if ii.size == 0 or jj.size == 0:
        raise RuntimeError("No overlap found between AMSR grid and Sohey bounds. Check lon convention.")
    return ii.min(), ii.max(), jj.min(), jj.max()


def _read_sohey_hi_nc(sohey_nc, sohey_var="HI"):
    """
    Read Sohey HI and lat/lon. Does NOT force lon to 0..360; we harmonize later with AMSR.
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available; cannot read netcdf.")
    nc = netCDF4.Dataset(sohey_nc, "r")

    if sohey_var not in nc.variables:
        raise KeyError(f"Variable '{sohey_var}' not in {sohey_nc}. Available: {list(nc.variables.keys())}")

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

    # for debugging
    units = getattr(v, "units", None)
    scale_factor = getattr(v, "scale_factor", None)
    add_offset = getattr(v, "add_offset", None)

    nc.close()

    if hi.ndim != 3:
        raise ValueError(f"Expected HI 3D array; got {hi.shape}")

    # Move time to last axis by smallest-dim heuristic (Sohey time is usually 6 or 8 or 12)
    t_axis = int(np.argmin(list(hi.shape)))
    hi_yxt = np.moveaxis(hi, t_axis, -1)

    if lat is None or lon is None:
        raise RuntimeError("Sohey netcdf missing latitude/longitude variables.")

    lat2d, lon2d = _as_2d_latlon(lat, lon, hi_yxt.shape[:2])
    return hi_yxt, lat2d, lon2d, units, scale_factor, add_offset


def _compute_sohey_monthly_mean_like_matlab(
    sohey_nc,
    amsr_mat,
    sohey_var="HI",
    thr=0.7,
    coastal_mat=None,
    coastal_var="Coastal",
    coastal_month_offset=0,
    debug=False,
):
    """
    MATLAB-equivalent mean + std:
      - interpolate Sohey HI (ice production, m/month) onto AMSR cut grid
      - compute mean/std over pixels where coastal_cut is NOT NaN
    Returns:
      mean: (T,) , std: (T,)
    """
    if loadmat is None:
        raise RuntimeError("scipy.io.loadmat not available.")
    if not HAS_SCIPY_INTERP:
        raise RuntimeError("scipy.interpolate not available (need LinearNDInterpolator).")

    # --- Sohey ---
    hi_yxt, lat_sohey, lon_sohey, units, scale_factor, add_offset = _read_sohey_hi_nc(sohey_nc, sohey_var=sohey_var)
    T = hi_yxt.shape[2]

    # --- AMSR ---
    amsr = load_mat_any(amsr_mat)  # <-- use your v7.3-safe loader
    if "sic" not in amsr:
        raise KeyError(f"'sic' not found in {amsr_mat}. Keys: {sorted(amsr.keys())}")
    if "lat" not in amsr or "lon" not in amsr:
        raise KeyError(f"'lat'/'lon' not found in {amsr_mat}. Keys: {sorted(amsr.keys())}")

    sic = np.array(amsr["sic"], dtype=float)
    latA = np.array(amsr["lat"], dtype=float)
    lonA = np.array(amsr["lon"], dtype=float)

    # Fix sic dimension if needed: accept [Y,X,365] -> [365,Y,X]
    if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
        sic = np.moveaxis(sic, -1, 0)

    # MATLAB does lat=lat'; lon=lon'
    if latA.shape != sic.shape[1:]:
        if latA.T.shape == sic.shape[1:]:
            latA = latA.T
            lonA = lonA.T
        else:
            raise ValueError(f"AMSR lat shape {latA.shape} does not match sic spatial {sic.shape[1:]}")

    # Harmonize lon conventions between AMSR and Sohey (don’t blindly mod everything)
    lonA, lon_sohey = _harmonize_lon(lonA, lon_sohey)

    # --- SIC monthly ---
    sic_monthly = _compute_monthly_mean_sic(sic)  # [Y,X,12]
    sic_monthly_frac = sic_monthly / 100.0

    # --- Coastal monthly mask ---
    Coastal = None
    if coastal_mat:
        cm = load_mat_any(coastal_mat)
        if coastal_var not in cm:
            raise KeyError(f"'{coastal_var}' not found in {coastal_mat}. Keys: {sorted(cm.keys())}")
        Coastal = np.array(cm[coastal_var], dtype=float)
    elif coastal_var in amsr:
        Coastal = np.array(amsr[coastal_var], dtype=float)

    if Coastal is None:
        # fallback (not identical to polynya_output_AMSR2 coastal)
        Coastal = np.where(sic_monthly_frac < thr, 1.0, np.nan)

    # Ensure Coastal matches [Y,X,12]
    if Coastal.shape != sic_monthly.shape:
        if Coastal.shape == (sic_monthly.shape[2], sic_monthly.shape[0], sic_monthly.shape[1]):
            Coastal = np.moveaxis(Coastal, 0, 2)
        else:
            raise ValueError(f"Coastal shape {Coastal.shape} not compatible with sic_monthly {sic_monthly.shape}")

    # --- Cut AMSR domain to Sohey bounds ---
    i0, i1, j0, j1 = _cut_to_sohey_bounds(latA, lonA, lat_sohey, lon_sohey)
    lat_cut = latA[i0:i1 + 1, j0:j1 + 1]
    lon_cut = lonA[i0:i1 + 1, j0:j1 + 1]
    coastal_cut = Coastal[i0:i1 + 1, j0:j1 + 1, :]

    lat_flat = lat_sohey.ravel()
    lon_flat = lon_sohey.ravel()

    out_mean = np.full((T,), np.nan, dtype=float)
    out_std = np.full((T,), np.nan, dtype=float)

    if debug:
        print(f"[SOHEY] units={units} scale_factor={scale_factor} add_offset={add_offset}")
        print(f"[SOHEY] HI nanmin/nanmax: {np.nanmin(hi_yxt):.3f} / {np.nanmax(hi_yxt):.3f}")
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
            fill_value=np.nan
        )
        hi_interp = F(lon_cut, lat_cut)

        m_idx = int(t + coastal_month_offset)
        m_idx = max(0, min(m_idx, coastal_cut.shape[2] - 1))

        msk = np.isfinite(coastal_cut[:, :, m_idx])
        vals = hi_interp[msk]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out_mean[t] = float(np.mean(vals))
            # MATLAB-like sample std: ddof=1 if possible
            out_std[t] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

        if debug:
            print(f"[SOHEY] t={t} m_idx={m_idx} n={vals.size} mean={out_mean[t]:.3f} std={out_std[t]:.3f}")

    return out_mean, out_std


def _expand_sohey_daily_matlab(hi_monthly, year=2023):
    """
    MATLAB:
      days_in_month = [31,31,30,31,31,31];  % sums 185
      date_daily = datetime(2023,5,1) + days(0:length-1)
    """
    hi_monthly = np.array(hi_monthly, dtype=float).ravel()
    if hi_monthly.size < 6:
        raise ValueError("Need at least 6 monthly values (May–Oct) for MATLAB-like daily expansion.")
    days_in_block = [31, 31, 30, 31, 31, 31]
    vals = np.repeat(hi_monthly[:6], days_in_block).astype(float)
    base = datetime(year, 5, 1)
    dates = [base + timedelta(days=i) for i in range(len(vals))]
    return dates, vals


# ----------------------------
# Map Plotting
# ----------------------------
def plot_sit_map(date_str, esacci_dir, amsr_mat, polynya_threshold=0.7, 
                 out_png="sit_map.png", vmin=0, vmax=3, debug=False):
    """
    Plot sea ice thickness from three ESACCI sources on a map with polynya overlay.
    
    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format (e.g., "20231028")
    esacci_dir : str
        Directory containing ESACCI files
    amsr_mat : str
        Path to AMSR MAT file
    polynya_threshold : float
        SIC threshold for polynya mask
    out_png : str
        Output PNG filename
    vmin, vmax : float
        Color scale limits for sea ice thickness (m)
    debug : bool
        Print debug information
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available.")
    if not HAS_SCIPY_INTERP:
        raise RuntimeError("scipy.interpolate not available.")
    
    import glob
    
    # Parse date
    try:
        target_date = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYYMMDD format.")
    
    # Find ESACCI files for this date
    patterns = [
        f"**/ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{date_str}-*.nc",
        f"**/ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3A-SH-{date_str}-*.nc",
        f"**/ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3B-SH-{date_str}-*.nc",
    ]
    
    esacci_files = {}
    for pattern in patterns:
        files = glob.glob(os.path.join(esacci_dir, pattern), recursive=True)
        if files:
            if "CRYOSAT2" in pattern:
                esacci_files["CryoSat-2"] = files[0] if len(files) == 1 else files
            elif "SENTINEL3A" in pattern:
                esacci_files["Sentinel-3A"] = files[0] if len(files) == 1 else files
            elif "SENTINEL3B" in pattern:
                esacci_files["Sentinel-3B"] = files[0] if len(files) == 1 else files
    
    if not esacci_files:
        raise RuntimeError(f"No ESACCI files found for date {date_str} in {esacci_dir}")
    
    if debug:
        print(f"[MAP] Found ESACCI files for {date_str}: {list(esacci_files.keys())}")
    
    # Load AMSR polynya mask
    amsr = load_mat_any(amsr_mat)
    if "sic" not in amsr or "lat" not in amsr or "lon" not in amsr:
        raise KeyError("AMSR MAT must contain sic, lat, lon")
    
    sic = np.array(amsr["sic"], dtype=float)
    lat_amsr = np.array(amsr["lat"], dtype=float)
    lon_amsr = np.array(amsr["lon"], dtype=float)
    
    # Fix sic dimension if needed
    if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
        sic = np.moveaxis(sic, -1, 0)
    
    # Fix lat/lon orientation
    if lat_amsr.shape != sic.shape[1:]:
        if lat_amsr.T.shape == sic.shape[1:]:
            lat_amsr = lat_amsr.T
            lon_amsr = lon_amsr.T
    
    # Get day of year
    day_of_year = target_date.timetuple().tm_yday
    if day_of_year > 365:
        day_of_year = 365
    day_idx = day_of_year - 1
    
    # Get polynya mask for this day
    if day_idx < sic.shape[0]:
        sic_daily = sic[day_idx, :, :]
    else:
        sic_daily = sic[-1, :, :]
    
    sic_frac = sic_daily / 100.0
    polynya_mask = sic_frac < polynya_threshold
    
    # Create figure with subplots for each source
    n_sources = len(esacci_files)
    fig, axes = plt.subplots(1, n_sources, figsize=(6*n_sources, 6), facecolor="white")
    if n_sources == 1:
        axes = [axes]
    
    colors = {"CryoSat-2": "darkorange", "Sentinel-3A": "darkorange", "Sentinel-3B": "darkorange"}
    
    for idx, (source_name, file_paths) in enumerate(esacci_files.items()):
        ax = axes[idx]
        
        # Read ESACCI data
        if isinstance(file_paths, list):
            # Combine data from multiple files if needed
            all_sit = []
            all_lat = []
            all_lon = []
            for fpath in file_paths:
                try:
                    sit, lat, lon, _ = read_esacci_sit_nc(fpath)
                    all_sit.append(sit)
                    all_lat.append(lat)
                    all_lon.append(lon)
                except Exception as e:
                    if debug:
                        print(f"[WARN] Error reading {fpath}: {e}")
                    continue
            
            if not all_sit:
                ax.text(0.5, 0.5, f"No data for {source_name}", 
                       ha="center", va="center", transform=ax.transAxes)
                continue
            
            # Combine arrays
            sit = np.concatenate([s.ravel() for s in all_sit])
            lat = np.concatenate([l.ravel() for l in all_lat])
            lon = np.concatenate([l.ravel() for l in all_lon])
        else:
            try:
                sit, lat, lon, _ = read_esacci_sit_nc(file_paths)
                sit = sit.ravel()
                lat = lat.ravel()
                lon = lon.ravel()
            except Exception as e:
                if debug:
                    print(f"[WARN] Error reading {file_paths}: {e}")
                ax.text(0.5, 0.5, f"No data for {source_name}", 
                       ha="center", va="center", transform=ax.transAxes)
                continue
        
        # Filter valid points
        valid = np.isfinite(sit) & np.isfinite(lat) & np.isfinite(lon)
        sit = sit[valid]
        lat = lat[valid]
        lon = lon[valid]
        
        if len(sit) == 0:
            ax.text(0.5, 0.5, f"No valid data for {source_name}", 
                   ha="center", va="center", transform=ax.transAxes)
            continue
        
        # Harmonize longitude
        lon_amsr_harmonized, lon_esacci_harmonized = _harmonize_lon(lon_amsr, lon)
        
        # Create scatter plot
        scatter = ax.scatter(lon_esacci_harmonized, lat, c=sit, 
                            s=10, cmap="viridis", vmin=vmin, vmax=vmax,
                            edgecolors="none", alpha=0.6)
        
        # Overlay polynya mask as contour
        try:
            contour = ax.contour(lon_amsr_harmonized, lat_amsr, polynya_mask.astype(float),
                               levels=[0.5], colors="red", linewidths=2.5, linestyles="--",
                               alpha=0.8)
            # Add label for legend (only on first subplot)
            if idx == 0:
                ax.plot([], [], color="red", linestyle="--", linewidth=2.5, 
                       label="Polynya boundary (SIC < {:.1f})".format(polynya_threshold))
        except Exception as e:
            if debug:
                print(f"[WARN] Could not plot polynya contour: {e}")
        
        # Set labels and title
        ax.set_xlabel("Longitude (°E)", fontsize=11)
        ax.set_ylabel("Latitude (°N)", fontsize=11)
        ax.set_title(f"{source_name}\n{target_date.strftime('%Y-%m-%d')}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio for map
        ax.set_aspect("equal", adjustable="box")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label="Sea ice thickness (m)", pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        
        # Add legend (only on first subplot)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    # Add overall title
    fig.suptitle(f"Sea Ice Thickness from ESACCI - {target_date.strftime('%Y-%m-%d')}", 
                 fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"[INFO] Saved map: {out_png}")
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=None, 
                    help="Path to plot-elements NPZ (SIT+AMSR). Optional if --plot_map is used.")
    ap.add_argument("--out_png", default="sit_amsr_racmo_sohey_nice.png")
    
    # Map plotting mode
    ap.add_argument("--plot_map", action="store_true",
                    help="Plot sea ice thickness map for a specific date")
    ap.add_argument("--map_date", default="20231028",
                    help="Date for map plotting in YYYYMMDD format (default: 20231028)")
    ap.add_argument("--map_out_png", default="sit_map.png",
                    help="Output PNG filename for map plot (default: sit_map.png)")
    ap.add_argument("--map_vmin", type=float, default=0,
                    help="Minimum value for color scale (default: 0)")
    ap.add_argument("--map_vmax", type=float, default=3,
                    help="Maximum value for color scale (default: 3)")
    ap.add_argument("--title", default=None)

    ap.add_argument("--start", default=None, help="Crop start date, e.g. 2023-05-01")
    ap.add_argument("--end", default=None, help="Crop end date, e.g. 2023-11-01")

    # RACMO
    ap.add_argument("--racmo_mat", default=None)
    ap.add_argument("--racmo_var", default="Ice_mean")
    ap.add_argument("--racmo_start", default="2023-05-01")

    # Sohey (MATLAB-like); HI is treated as ice production (m/month) for axes/legend (PMW AMSR2 path).
    ap.add_argument(
        "--sohey_nc",
        default=None,
        help="Sohey netCDF with HI (ice production, m/month) for PMW/coastal comparison to RACMO.",
    )
    ap.add_argument("--sohey_var", default="HI")
    ap.add_argument("--sohey_year", type=int, default=2023)
    ap.add_argument("--sohey_expand_daily", action="store_true")
    ap.add_argument(
        "--sohey_divide_by_30",
        action="store_true",
        help="Divide Sohey HI by 30 (MATLAB daily-rate style). Default: HI plotted as m/month ice production.",
    )
    ap.add_argument("--sohey_thr", type=float, default=0.7,
                    help="Fallback threshold if Coastal not provided (NOT same as polynya_output_AMSR2 coastal)")
    ap.add_argument("--amsr_mat", default=None,
                    help="Path to AMSR2_2023.mat (must contain sic, lat, lon)")
    ap.add_argument("--coastal_mat", default=None,
                    help="Optional MAT containing Coastal mask (monthly). If omitted, try inside amsr_mat.")
    ap.add_argument("--coastal_var", default="Coastal",
                    help="Variable name for Coastal mask inside MAT")
    ap.add_argument("--sohey_coastal_month_offset", type=int, default=0,
                    help="Month index offset between HI slices (t=0..) and Coastal months (0..11). "
                         "MATLAB code effectively uses 0; if HI starts in Mar, try 2; if starts in May, try 4.")
    ap.add_argument("--debug_sohey", action="store_true",
                    help="Print debug stats for Sohey interpolation/mask/means")
        # RACMO std (optional)
    ap.add_argument("--racmo_std_var", default=None,
                    help="Optional std variable name in RACMO mat (e.g., Ice_std). "
                         "If not provided, try common names automatically.")
    # SWOT-IS2 residual thermodynamic production from Eulerian closure CSV
    ap.add_argument(
        "--swot_prod_csv",
        default="eulerian_closure_tracked_fusion_ross_lagr_mask.csv",
        help="CSV with SWOT-IS2 ice production for RACMO intercomparison: Eulerian closure "
             "(P_thermo_cm_day_area_mean), or regional_track_mean_lumped_decomposition.py "
             "(prefers ice_production_m_month_area_mean when present). Same y-axis units as RACMO (m/month). "
             "Default matches drift_aware_mass_budget.py step-2 output with --sit_lagrangian_pooled_mask "
             "(align with plot_closure_budget_decomposition.py). For older runs without Lagrangian mask, "
             "pass e.g. eulerian_closure_tracked_fusion_ross.csv explicitly. Run from the directory "
             "containing the CSV or pass an absolute path.",
    )
    ap.add_argument(
        "--swot_prod_label",
        default=r"SWOT–IS-2 $P_{\mathrm{app}}$",
        help="Legend label for the SWOT-IS2 apparent mass tendency line.",
    )
    ap.add_argument(
        "--hide_swot_prod_line",
        action="store_true",
        help="Do not plot the red SWOT-IS2 residual production line (still allows loading/using CSV for other terms).",
    )
    ap.add_argument(
        "--swot_lower_only_pooled_simple",
        action="store_true",
        help="Lower panel: plot only SWOT-IS2 simple (pooled h × AMSR A); hide residual, storage (CSV/pooled), and CSV ΔAh.",
    )
    ap.add_argument(
        "--swot_prod_use_covered",
        action="store_true",
        help="Prefer P_thermo_cm_day_area_mean_covered (gated). Not the same definition as "
             "whole-area; can be orders of magnitude larger when covered area is small.",
    )
    ap.add_argument(
        "--swot_prod_use_covered_raw",
        action="store_true",
        help="Use P_thermo_cm_day_area_mean_covered_raw (diagnostic; unstable tails at low coverage).",
    )
    ap.add_argument("--swot_prod_days_per_month", type=float, default=30.0,
                    help="Conversion factor from day-rate to m/month (default: 30).")
    ap.add_argument("--swot_prod_allow_whole_fallback", action="store_true",
                    help="When --swot_prod_use_covered is set, allow fallback to whole-area values if covered is invalid/missing.")
    ap.add_argument(
        "--swot_prod_plot_signed",
        action="store_true",
        help="Plot SWOT-IS2 production including values <= 0 (signed thermodynamic residual). "
             "Default: draw only positive rates (omit <= 0 so the line does not dive negative).",
    )
    ap.add_argument(
        "--swot_plot_closure_p_thermo",
        action="store_true",
        help="Overlay Eulerian-closure P_thermo as ice production (m/month) from --swot_prod_csv: "
             "P_thermo = dM*/dt + F_out, same columns as plot_closure_budget_decomposition (forces closure "
             "P_thermo columns; never uses ice_production_m_month_area_mean). Respects "
             "--swot_prod_use_covered / _raw / --swot_prod_allow_whole_fallback and "
             "--swot_prod_align_to_sit_times. Compare to red pooled-simple triangles explicitly.",
    )
    ap.add_argument(
        "--swot_closure_p_thermo_label",
        default=None,
        help="Legend label for --swot_plot_closure_p_thermo (default: SWOT–IS2 P_thermo (budget closure)).",
    )
    ap.add_argument(
        "--swot_closure_p_thermo_plot_signed",
        action="store_true",
        help="Plot closure P_thermo overlay including values <= 0 (default: mask non-positive like main SWOT prod).",
    )
    ap.add_argument(
        "--swot_prod_err_mode",
        choices=["asymmetric", "symmetric", "none"],
        default="asymmetric",
        help="Error bars from regional CSV: asymmetric (p25/p75 thickness spread), symmetric (std), or none.",
    )
    ap.add_argument(
        "--swot_prod_align_to_sit_times",
        action="store_true",
        help="Place each production point at the same x time as the nearest IS2+SWOT scatter point "
             "(within --swot_prod_align_max_hours). Thickness (m) and production (m/month) stay different y scales.",
    )
    ap.add_argument(
        "--swot_prod_align_max_hours",
        type=float,
        default=72.0,
        help="Max |Δt| for snapping production to an IS2+SWOT scatter time when --swot_prod_align_to_sit_times.",
    )
    ap.add_argument(
        "--swot_storage_from_dmdt",
        action="store_true",
        help="Overlay storage-only SWOT estimate from dMdt_kg_s/(rho_i*A_polynya_m2) in m/month.",
    )
    ap.add_argument(
        "--swot_storage_label",
        default="SWOT-IS2 storage-only (dMdt)",
        help="Legend label for storage-only SWOT line.",
    )
    ap.add_argument(
        "--hide_swot_storage_dmdt_line",
        action="store_true",
        help="Do not draw the purple CSV storage-only line (dM/dt from closure CSV), even if --swot_storage_from_dmdt loaded it.",
    )
    ap.add_argument(
        "--hide_swot_storage_pooled_line",
        action="store_true",
        help="Do not draw the olive dashed line (storage from pooled SIT / daily pooled d(Ah)/dt), "
             "even if --swot_storage_residual_from_pooled_daily filled it.",
    )
    ap.add_argument(
        "--swot_storage_rho_i",
        type=float,
        default=915.0,
        help="Ice density (kg/m^3) for storage-only conversion from dMdt to m/month (default 915).",
    )
    ap.add_argument(
        "--swot_simple_from_ah",
        action="store_true",
        help="Overlay simple apparent production from adjacent Δ(Ah)/Δt using CSV M_polynya_kg and A_polynya_m2.",
    )
    ap.add_argument(
        "--swot_simple_label",
        default="SWOT-IS2 simple apparent (ΔAh)",
        help="Legend label for simple apparent production line.",
    )
    ap.add_argument(
        "--swot_simple_area_norm",
        choices=["mean", "a2", "a1"],
        default="mean",
        help="Area normalization for simple apparent production: mean=(A1+A2)/2 (default), a2=A2, a1=A1.",
    )
    ap.add_argument(
        "--swot_simple_pooled_min_dt_hours",
        type=float,
        default=24.0,
        help="For --swot_simple_from_pooled_sit: skip adjacent pairs with time gap below this threshold "
             "to avoid inflated Δ(Ah)/Δt from near-duplicate segment times (default 24 h).",
    )
    ap.add_argument(
        "--swot_pooled_diff_days",
        type=int,
        default=7,
        help="For pooled-SIT production series, use finite-difference over this many days/samples "
             "(default 7; helps suppress unrealistic day-to-day spikes).",
    )
    ap.add_argument(
        "--single_yaxis_thickness_and_production",
        action="store_true",
        help="Legacy: plot thickness (m) and ice production (m/month) on the same axis. "
             "Default: separate axes — left thickness (m), inner-right ice production (m/month), "
             "outer-right polynya area (km²).",
    )
    ap.add_argument(
        "--production_axis_outward_pts",
        type=float,
        default=78.0,
        help="When using separate production axis, offset the polynya-area right spine outward "
             "(points). Default 78.",
    )
    ap.add_argument(
        "--plot_two_panels",
        action="store_true",
        help="Two stacked panels: (top) sea ice thickness — IS2+SWOT + ESACCI; "
             "(bottom) ice production (RACMO, PMW/Sohey, SWOT-IS2) + AMSR polynya area. "
             "ESACCI needs --amsr_mat + ESACCI paths; RACMO needs --racmo_mat; PMW needs --sohey_nc + --amsr_mat. "
             "Use --swot_lower_only_pooled_simple to show only pooled-simple SWOT on the bottom axis. "
             "Ignored with --single_yaxis_thickness_and_production.",
    )
    ap.add_argument(
        "--figure4_annotations",
        action="store_true",
        help="Faint vertical lines at SWOT–IS2 snapshot times on the lower panel (two-panel mode). "
             "Default ON when --plot_two_panels unless --no_figure4_annotations.",
    )
    ap.add_argument(
        "--no_figure4_annotations",
        action="store_true",
        help="Disable SWOT–IS2 snapshot vertical markers.",
    )
    ap.add_argument(
        "--regime_span",
        action="append",
        default=None,
        metavar="START,END,LABEL",
        help="Optional shaded interval on both panels (repeatable). Not drawn by default; "
             "prefer describing area- vs thickness-driven intervals in manuscript text.",
    )
    ap.add_argument(
        "--mark_swot_snapshots",
        action="store_true",
        default=True,
        help="Draw snapshot vertical lines when --figure4_annotations (default: on).",
    )
    ap.add_argument(
        "--no_mark_swot_snapshots",
        action="store_false",
        dest="mark_swot_snapshots",
        help="Do not draw SWOT–IS2 snapshot vertical markers.",
    )
    ap.add_argument(
        "--swot_snapshot_alpha",
        type=float,
        default=0.32,
        help="Alpha for SWOT–IS2 snapshot vertical lines (default 0.32).",
    )
    ap.add_argument(
        "--sit_pooled_track_line",
        action="store_true",
        help="On thickness axis: draw a DAILY line of IS2+SWOT thickness pooled over ±sit_pooled_lag_days "
             "(mean/median of all segment samples in each daily window; polynya-context samples from NPZ).",
    )
    ap.add_argument(
        "--sit_pooled_lag_days",
        type=float,
        default=3.0,
        help="Half-width in days for pooling segment thickness around each scatter time (default 3).",
    )
    ap.add_argument(
        "--sit_pooled_stat",
        choices=["mean", "median"],
        default="mean",
        help="Aggregator across samples in the ±lag window (default mean).",
    )
    ap.add_argument(
        "--sit_pooled_label",
        default=None,
        help="Legend label for pooled track line (default: auto as daily pooled from lag and stat).",
    )
    ap.add_argument(
        "--sit_pooled_band",
        action="store_true",
        help="Shade p25–p75 of thickness samples within each ±lag window (thickness axis).",
    )
    ap.add_argument(
        "--sit_pooled_lagrangian",
        action="store_true",
        help="Pool segment thickness onto each calendar day t by advecting each sample's (lon,lat) from "
             "its observation time to t (thickness follows the parcel) within |t_sample−t|≤sit_pooled_lag_days. "
             "Requires --ice_motion_nc and per-segment sit_lon/sit_lat in the NPZ (or --sit_lagrangian_segment_npz ...). "
             "Uses AMSR (--amsr_mat) for NaN drift fill where implemented in drift_aware_mass_budget. "
             "Falls back to Eulerian ±lag pooling if geometry or motion is missing.",
    )
    ap.add_argument(
        "--ice_motion_nc",
        default=None,
        help="NetCDF with daily ice drift (used when --sit_pooled_lagrangian). Same convention as drift_aware_mass_budget.",
    )
    ap.add_argument(
        "--sit_lagrangian_segment_npz",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Optional: (1) A directory of IS2 scene NPZs — same as drift_aware --segment_npz: files matching "
             "--sit_lagrangian_scene_glob (default *_is2_polynya_sit_outputs.npz); each segment time in the plot "
             "NPZ is matched to the nearest scene file by timestamp in the filename (within "
             "--sit_lagrangian_scene_match_hours). "
             "(2) Or one or more explicit .npz files with sit_lon/sit_lat (or seg_lon/seg_lat) or sit_times_str. "
             "Example: --sit_lagrangian_segment_npz /path/to/is2_polynya_sit",
    )
    ap.add_argument(
        "--sit_lagrangian_scene_glob",
        default="*_is2_polynya_sit_outputs.npz",
        help="When --sit_lagrangian_segment_npz is a directory, glob for scene NPZ files (drift_aware default).",
    )
    ap.add_argument(
        "--sit_lagrangian_scene_match_hours",
        type=float,
        default=72.0,
        help="Max |Δt| (hours) between plot segment time and scene NPZ filename time when using a directory.",
    )
    ap.add_argument(
        "--sit_lagrangian_step_hours",
        type=float,
        default=24.0,
        help="Advection sub-step length for Lagrangian pooling (default 24 h).",
    )
    ap.add_argument(
        "--sit_lagrangian_ice_motion_subsample",
        type=int,
        default=5,
        help="Ice-motion grid subsampling for interpolation (passed to drift_aware; default 5).",
    )
    ap.add_argument(
        "--sit_pooled_counts_only",
        action="store_true",
        help="Print pooled-SIT valid-count diagnostics and exit before loading RACMO/CSV/ESACCI or plotting.",
    )
    ap.add_argument(
        "--swot_simple_from_pooled_sit",
        action="store_true",
        help="Red triangles: apparent column rate (M2−M1)/(rho·Aref·Δt) with M=rho*h_pooled*A_AMSR, daily query times, "
             "same lag/stat as --sit_pooled_*; by default masks ≤0 unless --swot_prod_plot_signed. "
             "Large negative spikes can occur when polynya area drops over Δt (same formula as ‘storage’ line but production-style mask).",
    )
    ap.add_argument(
        "--swot_simple_pooled_label",
        default=r"SWOT–IS-2 $P_{\mathrm{app}}$",
        help="Legend label for red triangles: pooled Δ(ρhA)/Δt apparent mass tendency.",
    )
    ap.add_argument(
        "--swot_storage_residual_from_pooled_daily",
        action="store_true",
        help="Use DAILY pooled SIT (from NPZ) to derive storage-only d(Ah)/dt and residual "
             "d(Ah)/dt + Fout/(rho*A). Requires --swot_prod_csv for Fout.",
    )
    ap.add_argument(
        "--swot_pooled_min_area_km2",
        type=float,
        default=100.0,
        help="For pooled-daily storage/residual and pooled-simple SWOT: set AMSR polynya area to NaN "
             "below this threshold (km^2) before M=ρhA (default 100) to avoid blow-up at tiny area.",
    )
    ap.add_argument(
        "--swot_pooled_fout_max_hours",
        type=float,
        default=36.0,
        help="For pooled-daily residual, only use Fout values if a CSV time_center exists within this "
             "time distance from each daily point (default 36 h).",
    )
    ap.add_argument(
        "--swot_pooled_residual_cap_m_month",
        type=float,
        default=15.0,
        help="For pooled-daily residual, mask points where |residual| exceeds this cap (m/month). "
             "Default 15 to suppress implausible outliers.",
    )
    ap.add_argument(
        "--swot_pooled_storage_display",
        choices=["signed", "mask_nonpos", "abs"],
        default="signed",
        help="Olive 'SWOT-IS2 storage from pooled SIT' line only: "
             "'signed' (default, true d(Ah)/dt column rate); "
             "'mask_nonpos' hide v<=0 (omit negatives from the plot only); "
             "'abs' plot |v| for exploratory comparison—does not change the underlying budget.",
    )
    ap.add_argument(
        "--swot_amsr_area_strict_edges",
        action="store_true",
        help="For pooled-simple and pooled-daily storage: interpolate AMSR polynya area onto daily times with "
             "NaN outside the AMSR date span (strict). Default is to hold the nearest edge value so window edges "
             "do not get all-NaN area (which fragments the red SWOT-simple line).",
    )

    # ESACCI Sea Ice Thickness
    ap.add_argument("--esacci_dir", default=None,
                    help="Base directory containing ESACCI files (e.g., '/Volumes/Yotta_1/SWOT/Antarctic_hi'). "
                         "Will automatically search recursively for CryoSat-2, Sentinel-3A, and Sentinel-3B files. "
                         "If not specified, will try default path '/Volumes/Yotta_1/SWOT/Antarctic_hi' if it exists.")
    ap.add_argument("--esacci_year", type=int, default=None,
                    help="Filter ESACCI files by year (e.g., 2023). If not specified, uses all years found.")
    ap.add_argument("--esacci_cryosat2", default=None,
                    help="Path or pattern to ESACCI CryoSat-2 SIT files (alternative to --esacci_dir)")
    ap.add_argument("--esacci_s3a", default=None,
                    help="Path or pattern to ESACCI Sentinel-3A SIT files (alternative to --esacci_dir)")
    ap.add_argument("--esacci_s3b", default=None,
                    help="Path or pattern to ESACCI Sentinel-3B SIT files (alternative to --esacci_dir)")
    ap.add_argument("--esacci_polynya_threshold", type=float, default=0.7,
                    help="SIC threshold for polynya mask (< threshold = polynya). Default: 0.7")
    ap.add_argument("--esacci_filter_quality", action="store_true", default=True,
                    help="Filter out ESACCI measurements with MIZ bias flags (flag_miz == 2). Default: True")
    ap.add_argument("--esacci_no_quality_filter", action="store_false", dest="esacci_filter_quality",
                    help="Disable quality filtering for ESACCI data")
    ap.add_argument("--esacci_max_uncertainty", type=float, default=None,
                    help="Maximum allowed uncertainty (m) for ESACCI measurements. If not specified, no uncertainty filtering.")
    ap.add_argument("--debug_esacci", action="store_true",
                    help="Print debug information for ESACCI processing")


    # Save merged NPZ
    ap.add_argument("--save_npz_out", default=None)
    # Statistics
    ap.add_argument("--print_stats", action="store_true", default=True,
                    help="Print mean/s.d./n and pairwise correlations for SIT, RACMO, Sohey, ESACCI (default: True)")
    ap.add_argument("--no_print_stats", action="store_false", dest="print_stats",
                    help="Do not print statistics")

    args = ap.parse_args()

    if args.swot_lower_only_pooled_simple:
        args.hide_swot_prod_line = True
        if args.swot_prod_csv and (
            args.swot_storage_from_dmdt
            or args.swot_simple_from_ah
            or args.swot_storage_residual_from_pooled_daily
        ):
            print(
                "[INFO] --swot_lower_only_pooled_simple: skipping CSV-based SWOT overlays "
                "(residual, dMdt storage, CSV ΔAh, pooled-daily residual); only pooled-simple SWOT is used."
            )

    # ---- Map plotting mode ----
    if args.plot_map:
        if not args.esacci_dir:
            # Try default path
            default_dir = "/Volumes/Yotta_1/SWOT/Antarctic_hi"
            if os.path.isdir(default_dir):
                args.esacci_dir = default_dir
            else:
                raise RuntimeError("--esacci_dir must be specified for map plotting, or use default path")
        
        if not args.amsr_mat:
            raise RuntimeError("--amsr_mat must be specified for map plotting")
        
        plot_sit_map(
            date_str=args.map_date,
            esacci_dir=args.esacci_dir,
            amsr_mat=args.amsr_mat,
            polynya_threshold=args.esacci_polynya_threshold,
            out_png=args.map_out_png,
            vmin=args.map_vmin,
            vmax=args.map_vmax,
            debug=args.debug_esacci
        )
        return

    # ---- Time series plotting mode ----
    if args.npz is None:
        raise RuntimeError("npz file must be provided for time series plotting, or use --plot_map for map plotting")

    start_dt = _parse_dt(args.start) if args.start else None
    end_dt = _parse_dt(args.end) if args.end else None

    # ---- Load base NPZ (SIT+AMSR) ----
    d = np.load(args.npz, allow_pickle=True)

    sit_times = _parse_dt_list(d["sit_times_str"])
    sit_mean = np.array(d["sit_mean"], dtype=float)
    sit_std = np.array(d["sit_std"], dtype=float)
    sit_nseg = np.array(d["sit_nseg"], dtype=float)
    sit_sizes = np.array(d["sit_sizes"], dtype=float)

    amsr_days = _parse_dt_list(d["amsr_days_str"])
    amsr_area = np.array(d["amsr_area"], dtype=float)

    size_min = float(np.array(d["size_min"]).ravel()[0])
    size_scale = float(np.array(d["size_scale"]).ravel()[0])
    amsr_area_mode = _to_str_scalar(d["amsr_area_mode"])
    amsr_area_weighting = _to_str_scalar(d["amsr_area_weighting"])

    def _npz_existing_file(key: str):
        if key not in d:
            return None
        try:
            s = str(_to_str_scalar(d[key])).strip()
        except Exception:
            return None
        if s and os.path.isfile(s):
            return s
        return None

    def _npz_existing_dir(key: str):
        if key not in d:
            return None
        try:
            s = str(_to_str_scalar(d[key])).strip()
        except Exception:
            return None
        if s and os.path.isdir(s):
            return s
        return None

    if not args.amsr_mat:
        for key in ("amsr_mat_path", "amsr_mat_str", "amsr_mat"):
            p = _npz_existing_file(key)
            if p:
                args.amsr_mat = p
                print(f"[INFO] Using AMSR MAT from NPZ key '{key}': {p}")
                break

    if not args.esacci_dir or not str(args.esacci_dir).strip():
        for key in ("esacci_dir", "esacci_dir_str"):
            pdir = _npz_existing_dir(key)
            if pdir:
                args.esacci_dir = pdir
                print(f"[INFO] Using ESACCI directory from NPZ key '{key}': {pdir}")
                break

    n_seg = len(sit_times)
    sit_lon_full, sit_lat_full = _try_read_sit_lon_lat_from_npz(d, n_expect=n_seg)
    seg_raw = list(args.sit_lagrangian_segment_npz) if args.sit_lagrangian_segment_npz else []
    seg_raw = [p for p in seg_raw if p and str(p).strip()]

    if sit_lon_full is None and seg_raw:
        # Single directory: IS2 scene NPZs (drift_aware convention)
        if len(seg_raw) == 1 and os.path.isdir(seg_raw[0]):
            sit_lon_full, sit_lat_full = _load_sit_lon_lat_from_is2_scene_dir(
                seg_raw[0],
                sit_times,
                glob_pat=args.sit_lagrangian_scene_glob,
                max_match_hours=args.sit_lagrangian_scene_match_hours,
            )
        else:
            seg_npz_paths = []
            for p in seg_raw:
                if os.path.isfile(p) and p.lower().endswith(".npz"):
                    seg_npz_paths.append(p)
                elif os.path.isdir(p):
                    found = sorted(glob.glob(os.path.join(p, args.sit_lagrangian_scene_glob)))
                    if not found:
                        found = sorted(glob.glob(os.path.join(p, "*.npz")))
                    print(f"[INFO] Lagrangian: expanded directory {p} -> {len(found)} NPZ file(s).")
                    seg_npz_paths.extend(found)
                else:
                    print(f"[WARN] --sit_lagrangian_segment_npz not found or not a .npz/dir: {p}")
            seg_npz_paths = [p for p in seg_npz_paths if os.path.isfile(p)]
            if seg_npz_paths:
                sit_lon_full, sit_lat_full = _load_sit_lon_lat_from_segment_npz_files(seg_npz_paths, sit_times)

    # Crop base series (include lon/lat when present so indices stay aligned)
    sit_lon_c = sit_lat_c = None
    if sit_lon_full is not None:
        sit_times_c, sit_mean_c, sit_std_c, sit_sizes_c, sit_nseg_c, sit_lon_c, sit_lat_c = _crop_series(
            sit_times, sit_mean, start_dt, end_dt, sit_std, sit_sizes, sit_nseg, sit_lon_full, sit_lat_full
        )
    else:
        sit_times_c, sit_mean_c = _crop_series(sit_times, sit_mean, start_dt, end_dt)
        _, sit_std_c = _crop_series(sit_times, sit_std, start_dt, end_dt)
        _, sit_sizes_c = _crop_series(sit_times, sit_sizes, start_dt, end_dt)
        _, sit_nseg_c = _crop_series(sit_times, sit_nseg, start_dt, end_dt)
    amsr_days_c, amsr_area_c = _crop_series(amsr_days, amsr_area, start_dt, end_dt)
    pooled_daily_times_c = []
    pooled_daily_center_c = None
    pooled_daily_lo_c = None
    pooled_daily_hi_c = None

    use_lagrangian_pooled = False
    if args.sit_pooled_lagrangian:
        if sit_lon_c is None or sit_lat_c is None:
            print(
                "[WARN] --sit_pooled_lagrangian needs per-segment sit_lon/sit_lat in the NPZ "
                "(or --sit_lagrangian_segment_npz with matching length). Using Eulerian ±lag pooling."
            )
        elif not args.ice_motion_nc or not str(args.ice_motion_nc).strip():
            print(
                "[WARN] --sit_pooled_lagrangian needs --ice_motion_nc. Using Eulerian ±lag pooling."
            )
        else:
            use_lagrangian_pooled = True
            if not args.amsr_mat:
                print(
                    "[WARN] --sit_pooled_lagrangian without --amsr_mat: NaN drift velocities may not be "
                    "filled over ocean; consider supplying AMSR for consistency with drift_aware_mass_budget."
                )
            print(
                "[INFO] Daily pooled SIT: Lagrangian mode (advect segment positions to each calendar day t "
                f"within ±{args.sit_pooled_lag_days:g} d; step={args.sit_lagrangian_step_hours:g} h)."
            )

    if len(sit_times_c) > 0:
        sgrid = start_dt if start_dt is not None else min(sit_times_c)
        egrid = end_dt if end_dt is not None else max(sit_times_c)
        pooled_daily_times_c = _daily_time_grid(sgrid, egrid)
        if use_lagrangian_pooled:
            try:
                pdc, pdl, pdh = _pooled_sit_lagrangian_on_query_times(
                    pooled_daily_times_c,
                    sit_times_c,
                    sit_mean_c,
                    sit_lon_c,
                    sit_lat_c,
                    lag_days=args.sit_pooled_lag_days,
                    stat=args.sit_pooled_stat,
                    band=args.sit_pooled_band,
                    ice_motion_nc=args.ice_motion_nc,
                    amsr_mat=args.amsr_mat,
                    step_hours=args.sit_lagrangian_step_hours,
                    ice_motion_subsample=args.sit_lagrangian_ice_motion_subsample,
                )
            except Exception as e:
                print(f"[WARN] Lagrangian pooled SIT failed ({e}); falling back to Eulerian pooling.")
                pdc, pdl, pdh = _pooled_sit_on_query_times(
                    pooled_daily_times_c,
                    sit_times_c,
                    sit_mean_c,
                    lag_days=args.sit_pooled_lag_days,
                    stat=args.sit_pooled_stat,
                    band=args.sit_pooled_band,
                )
        else:
            pdc, pdl, pdh = _pooled_sit_on_query_times(
                pooled_daily_times_c,
                sit_times_c,
                sit_mean_c,
                lag_days=args.sit_pooled_lag_days,
                stat=args.sit_pooled_stat,
                band=args.sit_pooled_band,
            )
        pooled_daily_center_c = np.asarray(pdc, dtype=float)
        pooled_daily_lo_c = np.asarray(pdl, dtype=float)
        pooled_daily_hi_c = np.asarray(pdh, dtype=float)

    if len(sit_times_c) > 0 and (
        args.sit_pooled_track_line or args.swot_simple_from_pooled_sit
    ):
        _print_pooled_sit_sample_count_info(
            sit_times_c,
            sit_mean_c,
            args.sit_pooled_lag_days,
            args.sit_pooled_stat,
        )
        n_daily_def = int(np.sum(np.isfinite(pooled_daily_center_c))) if pooled_daily_center_c is not None else 0
        print(
            f"[INFO] Daily pooled SIT coverage: {n_daily_def}/{len(pooled_daily_times_c)} days "
            f"have finite pooled thickness in the selected window."
        )
    if sit_times_c is not None and len(sit_times_c) > 0:
        n_sit_fin = int(np.sum(np.isfinite(np.asarray(sit_mean_c, dtype=float))))
        print(
            f"[INFO] IS2+SWOT scatter (upper panel): {n_sit_fin}/{len(sit_times_c)} segment times "
            f"with finite mean thickness in the date window."
        )
    if args.sit_pooled_counts_only:
        print("[INFO] --sit_pooled_counts_only set: exiting after pooled-SIT count diagnostics.")
        return

    # ---- RACMO (optional) ----
    racmo_dates_c = None
    racmo_vals_c = None
    racmo_std_c = None
    swot_prod_dates_c = None
    swot_prod_vals_c = None
    swot_prod_err_lo_c = None
    swot_prod_err_hi_c = None
    swot_prod_err_sym_c = None
    swot_storage_dates_c = None
    swot_storage_vals_c = None
    swot_storage_pooled_dates_c = None
    swot_storage_pooled_vals_c = None
    swot_simple_dates_c = None
    swot_simple_vals_c = None
    swot_simple_pooled_dates_c = None
    swot_simple_pooled_vals_c = None
    swot_pthermo_dates_c = None
    swot_pthermo_vals_c = None

    if args.racmo_mat:
        mat = load_mat_any(args.racmo_mat)

        if args.racmo_var not in mat:
            keys = sorted([k for k in mat.keys() if not k.startswith("__")])
            raise KeyError(f"Variable '{args.racmo_var}' not in {args.racmo_mat}. Keys: {keys}")

        racmo_vals = np.array(mat[args.racmo_var]).squeeze().astype(float)
        time_len = racmo_vals.size

        # dates
        racmo_dates = _maybe_read_mat_dates(mat)
        if racmo_dates is None:
            start_r = _parse_dt(args.racmo_start)
            racmo_dates = [start_r + timedelta(days=i) for i in range(time_len)]

        racmo_dates_c, racmo_vals_c = _crop_series(racmo_dates, racmo_vals, start_dt, end_dt)

        # ---- RACMO std from Mean_all cell ----
        racmo_std = None

        # 1) try user-provided std var (if it actually exists)
        if args.racmo_std_var and args.racmo_std_var in mat:
            cand = np.array(mat[args.racmo_std_var]).squeeze()
            if cand.ndim == 1 and cand.size == time_len:
                racmo_std = cand.astype(float)
                print(f"[INFO] Using RACMO std variable: {args.racmo_std_var}")
            else:
                print(f"[WARN] '{args.racmo_std_var}' exists but shape {cand.shape} doesn't match mean length {time_len}")

        # 2) compute from Mean_all (your case)
        if racmo_std is None and "Mean_all" in mat:
            try:
                racmo_std = _std_from_mean_all_cell(mat["Mean_all"])
                if racmo_std.size != time_len:
                    print(f"[WARN] Mean_all std length {racmo_std.size} != Ice_mean length {time_len}")
                else:
                    print("[INFO] Computed RACMO std from Mean_all cell.")
            except Exception as e:
                print(f"[WARN] Failed to compute RACMO std from Mean_all: {e}")
                racmo_std = None

        if racmo_std is not None:
            _, racmo_std_c = _crop_series(racmo_dates, racmo_std, start_dt, end_dt)
        else:
            racmo_std_c = None
            keys = sorted([k for k in mat.keys() if not k.startswith('__')])
            print("[WARN] RACMO std not found (no std var and Mean_all not usable).")
            print(f"[WARN] Available keys: {keys}")

    # ---- SWOT-IS2 production from Eulerian closure CSV (optional) ----
    if args.swot_prod_csv and (not args.swot_lower_only_pooled_simple):
        try:
            swot_prod_dates, swot_prod_vals, swot_elo, swot_ehi, swot_esym = load_swot_production_from_eulerian_csv(
                args.swot_prod_csv,
                prefer_covered=args.swot_prod_use_covered,
                prefer_covered_raw=args.swot_prod_use_covered_raw,
                allow_whole_fallback_when_prefer_covered=args.swot_prod_allow_whole_fallback,
                days_per_month=args.swot_prod_days_per_month,
                debug=args.debug_esacci,
            )
            swot_prod_dates_c, swot_prod_vals_c, swot_prod_err_lo_c, swot_prod_err_hi_c, swot_prod_err_sym_c = _crop_series(
                swot_prod_dates, swot_prod_vals, start_dt, end_dt,
                swot_elo, swot_ehi, swot_esym,
            )
            if args.swot_prod_align_to_sit_times and sit_times_c is not None and len(sit_times_c) > 0:
                av, alo, ahi, asym = _align_swot_production_to_sit_times(
                    sit_times_c,
                    swot_prod_dates_c,
                    swot_prod_vals_c,
                    swot_prod_err_lo_c,
                    swot_prod_err_hi_c,
                    swot_prod_err_sym_c,
                    max_hours=args.swot_prod_align_max_hours,
                )
                swot_prod_dates_c = list(sit_times_c)
                swot_prod_vals_c = av
                swot_prod_err_lo_c = alo
                swot_prod_err_hi_c = ahi
                swot_prod_err_sym_c = asym
                n_fin = int(np.sum(np.isfinite(av)))
                print(
                    f"[INFO] SWOT-IS2 production x-times aligned to IS2+SWOT scatter ({n_fin}/{len(av)} finite within "
                    f"{args.swot_prod_align_max_hours:g} h). y-axis remains ice production (m/month), not thickness (m)."
                )
            # If strict covered was requested but produced no finite values, warn clearly.
            if args.swot_prod_use_covered and (not args.swot_prod_use_covered_raw):
                vv = np.asarray(swot_prod_vals_c, dtype=float)
                if not np.any(np.isfinite(vv)):
                    print(
                        "[WARN] --swot_prod_use_covered produced no finite values in this date window. "
                        "Use whole-area (omit --swot_prod_use_covered), or add --swot_prod_use_covered_raw "
                        "for diagnostic-only covered rates."
                    )
            swot_prod_vals_c, swot_prod_err_lo_c, swot_prod_err_hi_c, swot_prod_err_sym_c = _mask_swot_prod_nonpos(
                swot_prod_vals_c,
                swot_prod_err_lo_c,
                swot_prod_err_hi_c,
                swot_prod_err_sym_c,
                args.swot_prod_plot_signed,
            )
            if not args.swot_prod_plot_signed:
                print(
                    "[INFO] SWOT-IS2 production: omitting rates <= 0 from plot and table below "
                    "(use --swot_prod_plot_signed for full signed residual)."
                )
            print(f"[INFO] Loaded {len(swot_prod_dates_c)} SWOT-IS2 production points from {args.swot_prod_csv}")
            _print_swot_production_rows(
                swot_prod_dates_c,
                swot_prod_vals_c,
                times_note=(
                    "times are SIT scatter times when aligned"
                    if args.swot_prod_align_to_sit_times
                    else "times are closure time_center from the CSV"
                ),
            )
            try:
                sit_d = {_to_date_str(t) for t in sit_times_c}
                sw_d = {_to_date_str(t) for t in swot_prod_dates_c}
                if args.swot_prod_align_to_sit_times:
                    print(
                        "[INFO] Production markers share x with SIT scatter (nearest closure within "
                        f"{args.swot_prod_align_max_hours:g} h); NaN y where no closure matched. "
                        "Thickness (m) and ice production rate (m/month) are different quantities — they do not match numerically."
                    )
                else:
                    print(
                        "[INFO] Why some IS2+SWOT points have no SWOT production diamond: closure uses time_center; "
                        f"SIT uses segment times ({len(sit_d)} distinct SIT days vs {len(sw_d)} production days). "
                        "Use --swot_prod_align_to_sit_times to snap production x to each scatter time. "
                        "Thickness (m) and production rate (m/month) are not equal numerically."
                    )
            except Exception:
                pass
            if args.swot_prod_use_covered and (not args.swot_prod_use_covered_raw):
                vv = np.asarray(swot_prod_vals_c, dtype=float)
                if np.any(np.isfinite(vv)):
                    mx = float(np.nanmax(vv))
                    med = float(np.nanmedian(vv[np.isfinite(vv)]))
                    if mx > 2.0 or med > 0.5:
                        print(
                            "[WARN] Covered-area SWOT-IS2 production (m/month) is large vs typical RACMO/SIT: "
                            "it uses P_thermo / (rho * A_covered), not whole polynya area — not comparable "
                            "to RACMO regional means and it can stretch the left y-axis. For this figure, "
                            "omit --swot_prod_use_covered (whole-area P_thermo_cm_day_area_mean)."
                        )
        except Exception as e:
            print(f"[WARN] Failed to load SWOT-IS2 production CSV: {e}")

    # ---- Explicit P_thermo (closure budget) for RACMO panel — aligns with plot_closure_budget_decomposition ----
    if (
        args.swot_plot_closure_p_thermo
        and args.swot_prod_csv
        and (not args.swot_lower_only_pooled_simple)
    ):
        try:
            pt_dates, pt_vals, _, _, _ = load_swot_production_from_eulerian_csv(
                args.swot_prod_csv,
                prefer_covered=args.swot_prod_use_covered,
                prefer_covered_raw=args.swot_prod_use_covered_raw,
                allow_whole_fallback_when_prefer_covered=args.swot_prod_allow_whole_fallback,
                days_per_month=args.swot_prod_days_per_month,
                debug=args.debug_esacci,
                closure_p_thermo_only=True,
            )
            swot_pthermo_dates_c, swot_pthermo_vals_c = _crop_series(
                pt_dates, pt_vals, start_dt, end_dt
            )
            if args.swot_prod_align_to_sit_times and sit_times_c is not None and len(sit_times_c) > 0:
                pt_aligned, _, _, _ = _align_swot_production_to_sit_times(
                    sit_times_c,
                    swot_pthermo_dates_c,
                    swot_pthermo_vals_c,
                    None,
                    None,
                    None,
                    max_hours=args.swot_prod_align_max_hours,
                )
                swot_pthermo_dates_c = list(sit_times_c)
                swot_pthermo_vals_c = pt_aligned
                n_pt = int(np.sum(np.isfinite(pt_aligned)))
                print(
                    f"[INFO] Closure P_thermo x-times aligned to IS2+SWOT scatter "
                    f"({n_pt}/{len(pt_aligned)} finite within {args.swot_prod_align_max_hours:g} h)."
                )
            swot_pthermo_vals_c, _, _, _ = _mask_swot_prod_nonpos(
                swot_pthermo_vals_c,
                None,
                None,
                None,
                args.swot_closure_p_thermo_plot_signed,
            )
            if not args.swot_closure_p_thermo_plot_signed:
                print(
                    "[INFO] Closure P_thermo overlay: omitting rates <= 0 "
                    "(use --swot_closure_p_thermo_plot_signed for signed)."
                )
            print(
                f"[INFO] Loaded closure P_thermo (m/month) for overlay: {len(swot_pthermo_dates_c)} points from "
                f"{args.swot_prod_csv} (closure columns only; same definition as decomposition figure)."
            )
        except Exception as e:
            print(f"[WARN] Failed to load closure P_thermo overlay from CSV: {e}")

    # ---- Optional storage-only SWOT estimate from dMdt/(rho*A) ----
    if args.swot_storage_from_dmdt and (not args.swot_lower_only_pooled_simple):
        if not args.swot_prod_csv:
            print("[WARN] --swot_storage_from_dmdt requested but --swot_prod_csv is missing; skipping storage-only overlay.")
        else:
            try:
                st_dates, st_vals = load_swot_storage_only_from_csv(
                    args.swot_prod_csv,
                    rho_i=args.swot_storage_rho_i,
                    days_per_month=args.swot_prod_days_per_month,
                    debug=args.debug_esacci,
                )
                st_dates_c, st_vals_c = _crop_series(st_dates, st_vals, start_dt, end_dt)
                if args.swot_prod_align_to_sit_times and sit_times_c is not None and len(sit_times_c) > 0:
                    st_aligned, _, _, _ = _align_swot_production_to_sit_times(
                        sit_times_c,
                        st_dates_c,
                        st_vals_c,
                        None,
                        None,
                        None,
                        max_hours=args.swot_prod_align_max_hours,
                    )
                    st_dates_c = list(sit_times_c)
                    st_vals_c = st_aligned
                    n_fin_st = int(np.sum(np.isfinite(st_aligned)))
                    print(
                        f"[INFO] SWOT storage-only x-times aligned to IS2+SWOT scatter "
                        f"({n_fin_st}/{len(st_aligned)} finite within {args.swot_prod_align_max_hours:g} h)."
                    )
                swot_storage_dates_c = st_dates_c
                swot_storage_vals_c = np.asarray(st_vals_c, dtype=float)
                _print_swot_production_rows(
                    swot_storage_dates_c,
                    swot_storage_vals_c,
                    label="SWOT-IS2 storage-only",
                    times_note=(
                        "times are SIT scatter times when aligned"
                        if args.swot_prod_align_to_sit_times
                        else "times are closure time_center from the CSV"
                    ),
                )
            except Exception as e:
                print(f"[WARN] Failed to load SWOT storage-only curve from CSV: {e}")

    # ---- Optional simple apparent production from adjacent Δ(Ah)/Δt ----
    if args.swot_simple_from_ah and (not args.swot_lower_only_pooled_simple):
        if not args.swot_prod_csv:
            print("[WARN] --swot_simple_from_ah requested but --swot_prod_csv is missing; skipping simple overlay.")
        else:
            try:
                sp_dates, sp_vals = load_swot_simple_apparent_from_csv(
                    args.swot_prod_csv,
                    rho_i=args.swot_storage_rho_i,
                    days_per_month=args.swot_prod_days_per_month,
                    area_norm=args.swot_simple_area_norm,
                    debug=args.debug_esacci,
                )
                sp_dates_c, sp_vals_c = _crop_series(sp_dates, sp_vals, start_dt, end_dt)
                if args.swot_prod_align_to_sit_times and sit_times_c is not None and len(sit_times_c) > 0:
                    sp_aligned, _, _, _ = _align_swot_production_to_sit_times(
                        sit_times_c,
                        sp_dates_c,
                        sp_vals_c,
                        None,
                        None,
                        None,
                        max_hours=args.swot_prod_align_max_hours,
                    )
                    sp_dates_c = list(sit_times_c)
                    sp_vals_c = sp_aligned
                    n_fin_sp = int(np.sum(np.isfinite(sp_aligned)))
                    print(
                        f"[INFO] SWOT simple apparent x-times aligned to IS2+SWOT scatter "
                        f"({n_fin_sp}/{len(sp_aligned)} finite within {args.swot_prod_align_max_hours:g} h)."
                    )
                swot_simple_dates_c = sp_dates_c
                swot_simple_vals_c = np.asarray(sp_vals_c, dtype=float)
                _print_swot_production_rows(
                    swot_simple_dates_c,
                    swot_simple_vals_c,
                    label="SWOT-IS2 simple apparent",
                    times_note=(
                        f"times are {'SIT scatter times when aligned' if args.swot_prod_align_to_sit_times else 'closure time_center from the CSV'}; "
                        f"area_norm={args.swot_simple_area_norm}"
                    ),
                )
            except Exception as e:
                print(f"[WARN] Failed to load SWOT simple apparent curve from CSV: {e}")

    # ---- Optional simple production from pooled IS2+SWOT h × AMSR A (NPZ only) ----
    if args.swot_simple_from_pooled_sit:
        try:
            if sit_times_c is None or len(sit_times_c) < 2:
                print(
                    "[WARN] --swot_simple_from_pooled_sit needs at least 2 IS2+SWOT segment times in the date window; skipping."
                )
            elif amsr_days_c is None or len(amsr_days_c) < 2:
                print(
                    "[WARN] --swot_simple_from_pooled_sit needs AMSR polynya area on at least 2 days in the window; skipping."
                )
            else:
                pp_dates, pp_vals = compute_swot_simple_from_pooled_sit_npz(
                    sit_times_c,
                    sit_mean_c,
                    amsr_days_c,
                    amsr_area_c,
                    lag_days=args.sit_pooled_lag_days,
                    pooled_stat=args.sit_pooled_stat,
                    rho_i=args.swot_storage_rho_i,
                    days_per_month=args.swot_prod_days_per_month,
                    area_norm=args.swot_simple_area_norm,
                    min_dt_hours=args.swot_simple_pooled_min_dt_hours,
                    diff_step=args.swot_pooled_diff_days,
                    query_times=pooled_daily_times_c,
                    debug=args.debug_esacci,
                    sit_pooled_lagrangian=use_lagrangian_pooled,
                    sit_lon=sit_lon_c,
                    sit_lat=sit_lat_c,
                    ice_motion_nc=args.ice_motion_nc,
                    amsr_mat_for_drift=args.amsr_mat,
                    lagrangian_step_hours=args.sit_lagrangian_step_hours,
                    lagrangian_ice_motion_subsample=args.sit_lagrangian_ice_motion_subsample,
                    min_area_km2=args.swot_pooled_min_area_km2,
                    report_diagnostics=True,
                    extrapolate_amsr_area_edges=not args.swot_amsr_area_strict_edges,
                )
                pp_dates_c, pp_vals_c = pp_dates, pp_vals
                if args.swot_prod_align_to_sit_times and sit_times_c is not None and len(sit_times_c) > 0:
                    pp_aligned, _, _, _ = _align_swot_production_to_sit_times(
                        sit_times_c,
                        pp_dates_c,
                        pp_vals_c,
                        None,
                        None,
                        None,
                        max_hours=args.swot_prod_align_max_hours,
                    )
                    pp_dates_c = list(sit_times_c)
                    pp_vals_c = pp_aligned
                    n_fin_pp = int(np.sum(np.isfinite(pp_aligned)))
                    print(
                        f"[INFO] SWOT pooled-h simple x-times aligned to IS2+SWOT scatter "
                        f"({n_fin_pp}/{len(pp_aligned)} finite within {args.swot_prod_align_max_hours:g} h)."
                    )
                pp_vals_c = np.asarray(pp_vals_c, dtype=float)
                n_fin_pre_mask = int(np.sum(np.isfinite(pp_vals_c)))
                pp_vals_c, _, _, _ = _mask_swot_prod_nonpos(
                    pp_vals_c, None, None, None, args.swot_prod_plot_signed
                )
                n_fin_post_mask = int(np.sum(np.isfinite(pp_vals_c)))
                if (not args.swot_prod_plot_signed) and n_fin_post_mask < n_fin_pre_mask:
                    print(
                        f"[INFO] SWOT pooled-simple: masked {n_fin_pre_mask - n_fin_post_mask} non-positive "
                        f"rates (finite before mask: {n_fin_pre_mask}); {n_fin_post_mask} remain for plot. "
                        "Use --swot_prod_plot_signed to plot signed rates."
                    )
                swot_simple_pooled_dates_c = pp_dates_c
                swot_simple_pooled_vals_c = pp_vals_c
                _print_swot_production_rows(
                    swot_simple_pooled_dates_c,
                    swot_simple_pooled_vals_c,
                    label=args.swot_simple_pooled_label,
                    times_note=(
                        f"pooled SIT: {'Lagrangian (advect to day t) ' if use_lagrangian_pooled else ''}"
                        f"±{args.sit_pooled_lag_days:g} d, {args.sit_pooled_stat}; "
                        f"AMSR area interpolated to pooled query times (daily unless aligned); area_norm={args.swot_simple_area_norm}; "
                        f"min_dt={args.swot_simple_pooled_min_dt_hours:g} h; diff_step={args.swot_pooled_diff_days}"
                    ),
                )
                if not args.swot_prod_plot_signed:
                    print(
                        "[INFO] SWOT pooled-h simple: omitting rates <= 0 from plot and table "
                        "(use --swot_prod_plot_signed for full signed series)."
                    )
        except Exception as e:
            print(f"[WARN] Failed to compute SWOT simple production from pooled SIT × AMSR area: {e}")

    if args.swot_simple_from_pooled_sit and args.swot_storage_residual_from_pooled_daily:
        print(
            "[INFO] Lower-panel curves from pooled NPZ: red (SWOT-IS2 pooled-simple) and yellow “storage from pooled SIT” "
            "both use (M2−M1)/(rho·Aref·Δt) with M=rho*h_pooled*A; red applies the production mask (--swot_prod_plot_signed); "
            "yellow shows signed d(Ah)/dt. Large negatives are often from ΔA over the differencing interval (polynya shrinking), "
            "not necessarily an implementation error."
        )

    # ---- Optional: derive storage/residual from DAILY pooled SIT + AMSR area (+Fout for residual) ----
    if args.swot_storage_residual_from_pooled_daily and (not args.swot_lower_only_pooled_simple):
        if not args.swot_prod_csv:
            print(
                "[WARN] --swot_storage_residual_from_pooled_daily requested but --swot_prod_csv is missing "
                "(need Fout_kg_s); skipping pooled-daily storage/residual override."
            )
        else:
            try:
                st_dates_pd, st_vals_pd = compute_swot_simple_from_pooled_sit_npz(
                    sit_times_c,
                    sit_mean_c,
                    amsr_days_c,
                    amsr_area_c,
                    lag_days=args.sit_pooled_lag_days,
                    pooled_stat=args.sit_pooled_stat,
                    rho_i=args.swot_storage_rho_i,
                    days_per_month=args.swot_prod_days_per_month,
                    area_norm=args.swot_simple_area_norm,
                    min_dt_hours=args.swot_simple_pooled_min_dt_hours,
                    diff_step=args.swot_pooled_diff_days,
                    query_times=pooled_daily_times_c,
                    debug=args.debug_esacci,
                    sit_pooled_lagrangian=use_lagrangian_pooled,
                    sit_lon=sit_lon_c,
                    sit_lat=sit_lat_c,
                    ice_motion_nc=args.ice_motion_nc,
                    amsr_mat_for_drift=args.amsr_mat,
                    lagrangian_step_hours=args.sit_lagrangian_step_hours,
                    lagrangian_ice_motion_subsample=args.sit_lagrangian_ice_motion_subsample,
                    min_area_km2=args.swot_pooled_min_area_km2,
                    report_diagnostics=False,
                    extrapolate_amsr_area_edges=not args.swot_amsr_area_strict_edges,
                )
                st_vals_pd_raw = np.asarray(st_vals_pd, dtype=float)
                A_km2_i = _interp_amsr_area_km2_at_times(
                    st_dates_pd,
                    amsr_days_c,
                    amsr_area_c,
                    extrapolate_edges=not args.swot_amsr_area_strict_edges,
                )
                A_m2_i = np.asarray(A_km2_i, dtype=float) * 1e6
                ok_area = np.isfinite(A_km2_i) & (A_km2_i >= float(args.swot_pooled_min_area_km2))

                st_vals_pd_plot = np.where(ok_area, st_vals_pd_raw, np.nan)
                # d(Ah)/dt storage can be negative (ice growth / thickening); do not apply production-only v>0 mask.
                st_vals_pd_plot, _, _, _ = _mask_swot_prod_nonpos(
                    st_vals_pd_plot, None, None, None, plot_signed=True
                )
                # Keep true dMdt storage from CSV visible when requested; plot pooled-storage separately.
                if (
                    args.swot_storage_from_dmdt
                    and swot_storage_dates_c is not None
                    and swot_storage_vals_c is not None
                    and len(swot_storage_dates_c) > 0
                ):
                    swot_storage_pooled_dates_c = st_dates_pd
                    swot_storage_pooled_vals_c = st_vals_pd_plot
                else:
                    swot_storage_dates_c = st_dates_pd
                    swot_storage_vals_c = st_vals_pd_plot
                _print_swot_production_rows(
                    st_dates_pd,
                    st_vals_pd_plot,
                    label=f"{args.swot_storage_label} [daily pooled SIT]",
                    times_note=(
                        f"storage=d(Ah)/dt from daily pooled SIT "
                        f"{'(Lagrangian advect) ' if use_lagrangian_pooled else ''}"
                        f"±{args.sit_pooled_lag_days:g} d ({args.sit_pooled_stat}), "
                        f"area_norm={args.swot_simple_area_norm}, min_dt={args.swot_simple_pooled_min_dt_hours:g} h, "
                        f"diff_step={args.swot_pooled_diff_days}, area_min={args.swot_pooled_min_area_km2:g} km^2"
                    ),
                )

                fout_t, fout_v = load_swot_fout_kg_s_from_csv(args.swot_prod_csv)
                fout_i = _nearest_series_within_hours(
                    st_dates_pd,
                    fout_t,
                    fout_v,
                    max_hours=args.swot_pooled_fout_max_hours,
                )
                export_term = np.full_like(st_vals_pd, np.nan, dtype=float)
                ok = np.isfinite(fout_i) & np.isfinite(A_m2_i) & (A_m2_i > 0) & ok_area
                export_term[ok] = (
                    (fout_i[ok] / (float(args.swot_storage_rho_i) * A_m2_i[ok]))
                    * 86400.0
                    * float(args.swot_prod_days_per_month)
                )
                res_vals = st_vals_pd_raw + export_term
                cap = abs(float(args.swot_pooled_residual_cap_m_month))
                if cap > 0:
                    res_vals = np.where(np.isfinite(res_vals) & (np.abs(res_vals) <= cap), res_vals, np.nan)
                # Residual can be negative; do not hide with production-only v>0 mask.
                res_vals, _, _, _ = _mask_swot_prod_nonpos(
                    res_vals, None, None, None, plot_signed=True
                )
                swot_prod_dates_c = st_dates_pd
                swot_prod_vals_c = res_vals
                swot_prod_err_lo_c = None
                swot_prod_err_hi_c = None
                swot_prod_err_sym_c = None
                _print_swot_production_rows(
                    swot_prod_dates_c,
                    swot_prod_vals_c,
                    label=f"{args.swot_prod_label} [daily pooled SIT + Fout]",
                    times_note=(
                        "residual = d(Ah)/dt from daily pooled SIT + Fout/(rho*A); "
                        f"Fout nearest within {args.swot_pooled_fout_max_hours:g} h; "
                        f"min_dt={args.swot_simple_pooled_min_dt_hours:g} h; diff_step={args.swot_pooled_diff_days}; "
                        f"area_min={args.swot_pooled_min_area_km2:g} km^2; cap=±{cap:g} m/month"
                    ),
                )
                print(
                    "[INFO] Overrode storage-only and residual SWOT curves using daily pooled SIT "
                    "(residual adds Fout/(rho*A))."
                )
            except Exception as e:
                print(f"[WARN] Failed pooled-daily storage/residual override: {e}")

    # ---- Sohey (MATLAB-like mean+std) ----
    sohey_dates_c = None
    sohey_mean_c = None
    sohey_std_c = None

    if args.sohey_nc:
        if args.amsr_mat is None:
            raise RuntimeError("To follow MATLAB for Sohey you must provide --amsr_mat AMSR2_2023.mat")

        # you can expose this as CLI if needed; keep 0 by default
        coastal_month_offset = 0

        sohey_m, sohey_s = _compute_sohey_monthly_mean_like_matlab(
            sohey_nc=args.sohey_nc,
            amsr_mat=args.amsr_mat,
            sohey_var=args.sohey_var,
            thr=args.sohey_thr,
            coastal_mat=args.coastal_mat,
            coastal_var=args.coastal_var,
            coastal_month_offset=coastal_month_offset,
            debug=args.debug_sohey,
        )

        if args.sohey_expand_daily:
            # MATLAB daily expansion May–Oct (6 values)
            days_in_block = [31, 31, 30, 31, 31, 31]
            sohey_mean = np.repeat(sohey_m[:6], days_in_block).astype(float)
            sohey_std = np.repeat(sohey_s[:6], days_in_block).astype(float)
            base = datetime(args.sohey_year, 5, 1)
            sohey_dates = [base + timedelta(days=i) for i in range(len(sohey_mean))]
        else:
            # monthly midpoints (best-effort)
            if sohey_m.size == 6:
                months = list(range(5, 11))
            elif sohey_m.size == 8:
                months = list(range(3, 11))
            else:
                months = list(range(1, sohey_m.size + 1))
            sohey_dates = [datetime(args.sohey_year, m, 15) for m in months]
            sohey_mean = sohey_m
            sohey_std = sohey_s

        # Optional daily-rate proxy
        if args.sohey_divide_by_30:
            sohey_mean = sohey_mean / 30.0
            sohey_std = sohey_std / 30.0

        sohey_dates_c, sohey_mean_c = _crop_series(sohey_dates, sohey_mean, start_dt, end_dt)
        _, sohey_std_c = _crop_series(sohey_dates, sohey_std, start_dt, end_dt)

    # ---- ESACCI Sea Ice Thickness (optional) ----
    esacci_dates_c = None
    esacci_mean_c = None
    esacci_std_c = None
    
    esacci_files = []

    # Automatic file finding from directory (module-level import os)
    esacci_dir_to_use = args.esacci_dir

    # If still unset: look for Antarctic_hi next to (or above) the plot NPZ — common layout.
    if not esacci_dir_to_use or not str(esacci_dir_to_use).strip():
        if args.npz:
            npz_abs = os.path.abspath(args.npz)
            nd = os.path.dirname(npz_abs)
            for cand in (os.path.join(nd, "Antarctic_hi"), os.path.join(os.path.dirname(nd), "Antarctic_hi")):
                if cand and os.path.isdir(cand):
                    esacci_dir_to_use = cand
                    print(f"[INFO] Using ESACCI directory (next to plot NPZ): {cand}")
                    break

    # Try default volume path if not specified
    if not esacci_dir_to_use or not str(esacci_dir_to_use).strip():
        default_dir = "/Volumes/Yotta_1/SWOT/Antarctic_hi"
        if os.path.isdir(default_dir):
            esacci_dir_to_use = default_dir
            print(f"[INFO] Using default ESACCI directory: {default_dir}")
    
    if esacci_dir_to_use and esacci_dir_to_use.strip():
        if os.path.isdir(esacci_dir_to_use):
            auto_files = find_esacci_files(
                esacci_dir_to_use, 
                year=args.esacci_year,
                debug=args.debug_esacci
            )
            if auto_files:
                esacci_files.extend(auto_files)
                print(f"[INFO] Automatically found {len(auto_files)} ESACCI files in {esacci_dir_to_use}")
            else:
                if args.debug_esacci:
                    print(f"[WARN] No ESACCI files found in {esacci_dir_to_use}")
        else:
            if args.debug_esacci:
                print(f"[WARN] ESACCI directory does not exist: {esacci_dir_to_use}")
    
    # Manual file patterns (alternative to automatic finding)
    if args.esacci_cryosat2:
        esacci_files.append(args.esacci_cryosat2)
    if args.esacci_s3a:
        esacci_files.append(args.esacci_s3a)
    if args.esacci_s3b:
        esacci_files.append(args.esacci_s3b)
    
    if esacci_files and args.amsr_mat:
        try:
            esacci_dates, esacci_mean, esacci_std, esacci_count = compute_esacci_daily_polynya_mean(
                esacci_files, args.amsr_mat,
                polynya_threshold=args.esacci_polynya_threshold,
                filter_quality=args.esacci_filter_quality,
                max_uncertainty=args.esacci_max_uncertainty,
                debug=args.debug_esacci
            )
            esacci_dates_c, esacci_mean_c = _crop_series(esacci_dates, esacci_mean, start_dt, end_dt)
            _, esacci_std_c = _crop_series(esacci_dates, esacci_std, start_dt, end_dt)
            n_es_fin = int(np.sum(np.isfinite(np.asarray(esacci_mean_c, dtype=float))))
            print(
                f"[INFO] ESACCI polynya-mean thickness (upper panel): loaded {len(esacci_dates_c)} daily values; "
                f"{n_es_fin}/{len(esacci_dates_c)} days finite after date crop."
            )
        except Exception as e:
            print(f"[WARN] Failed to load ESACCI data: {e}")
            if args.debug_esacci:
                import traceback
                traceback.print_exc()

    if (
        bool(args.plot_two_panels)
        and (not bool(args.single_yaxis_thickness_and_production))
        and (esacci_dates_c is None or len(esacci_dates_c) == 0)
    ):
        print(
            "[WARN] Upper panel: ESACCI thickness was not drawn. It needs (1) a readable AMSR .mat path "
            "via --amsr_mat or NPZ key amsr_mat_path, and (2) ESACCI L2P NetCDF files found from "
            "--esacci_dir (recursive glob ESACCI-SEAICE-L2P-SITHICK-*.nc) or --esacci_cryosat2 / s3a / s3b. "
            "The script also looks for a folder Antarctic_hi next to your plot NPZ or under /Volumes/Yotta_1/SWOT/."
        )
        if not args.amsr_mat:
            print("[HINT] --amsr_mat is missing: ESACCI polynya-mean SIT is not computed (AMSR masks the polynya).")
        if not esacci_files:
            print(
                f"[HINT] No ESACCI file list was built (check --esacci_dir / on-disk paths; "
                f"last directory tried for search: {esacci_dir_to_use!r})."
            )
    elif esacci_dates_c is not None and len(esacci_dates_c) > 0:
        _em = np.asarray(esacci_mean_c, dtype=float)
        if not np.any(np.isfinite(_em)):
            print(
                "[WARN] ESACCI dates were loaded but all daily means are NaN in the plot window "
                "(no polynya ESACCI points after filters / threshold). Upper panel will show no ESACCI curve."
            )

    if bool(args.plot_two_panels) and (not bool(args.single_yaxis_thickness_and_production)):
        tips = []
        if esacci_dates_c is None or len(esacci_dates_c) == 0:
            tips.append(
                "ESACCI (top): pass --amsr_mat and ESACCI paths (--esacci_dir or --esacci_cryosat2 / s3a / s3b), "
                "or store amsr_mat_path + esacci_dir in the NPZ"
            )
        if racmo_dates_c is None or len(racmo_dates_c) == 0:
            tips.append(
                "RACMO (bottom): pass --racmo_mat (and --racmo_var / --racmo_std_var as needed)"
            )
        if sohey_dates_c is None or len(sohey_dates_c) == 0:
            tips.append(
                "PMW (bottom): pass --sohey_nc, --amsr_mat, and usually --sohey_expand_daily"
            )
        if tips:
            print("[INFO] Two-panel reference-style figure: also provide:")
            for t in tips:
                print(f"       • {t}")

    # ---------------- Statistics ----------------
    if args.print_stats:
        print_sit_racmo_sohey_statistics(
            sit_times_c, sit_mean_c, sit_std_c,
            racmo_dates_c, racmo_vals_c, racmo_std_c,
            sohey_dates_c, sohey_mean_c, sohey_std_c,
            esacci_dates_c, esacci_mean_c, esacci_std_c,
            n_bootstrap=2000,
        )

    # ---------------- Plot ----------------
    # --------------------------
    # Plot panel (a): match HSSW figure style
    # --------------------------
    use_two_panels = bool(args.plot_two_panels) and (not bool(args.single_yaxis_thickness_and_production))
    if bool(args.plot_two_panels) and bool(args.single_yaxis_thickness_and_production):
        print("[WARN] --plot_two_panels is ignored when --single_yaxis_thickness_and_production is set.")

    if args.no_figure4_annotations:
        use_fig4_annotations = False
    elif args.figure4_annotations:
        use_fig4_annotations = True
    else:
        use_fig4_annotations = use_two_panels
    fig4_regime_spans = _resolve_fig4_regime_spans(args) if use_fig4_annotations else []

    if not args.single_yaxis_thickness_and_production:
        if use_two_panels:
            print(
                "[INFO] Plot layout: two panels — (top) sea ice thickness (m), IS2+SWOT & ESACCI; "
                "(bottom) ice production (m/month) + AMSR polynya area (km²)."
            )
            if use_fig4_annotations:
                print(
                    "[INFO] Fig. 4 snapshot markers ON: faint vertical lines on lower panel only "
                    "(disable with --no_figure4_annotations)."
                )
        else:
            print(
                "[INFO] Plot layout: sea ice thickness (m) left axis; ice production (m/month) inner-right "
                "(RACMO, PMW/Sohey HI, SWOT-IS2 if provided); polynya area (km²) outer-right. "
                "Use --plot_two_panels for stacked thickness vs production/area. "
                "Use --single_yaxis_thickness_and_production for legacy single-axis."
            )

    color_sit  = "#d62728"  # red #d62728
    color_amsr = "#1f77b4"  # blue
    color_sohey = "#2ca02c" # green
    color_racmo = "k"       # black

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,   # re-enable on twinx axes (production / polynya)
    })

    ax_prod = None
    ax_bot = None
    if use_two_panels:
        fig, (ax_sit_plot, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12, 9.5),
            facecolor="white",
            gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.06},
        )
        ax_sit_plot.set_facecolor((0.96, 0.96, 0.96))
        ax_bot.set_facecolor((0.96, 0.96, 0.96))
        poly_ax = ax_bot.twinx()
        # Keep area axis close to lower panel in two-panel mode for readability.
        poly_ax.spines["right"].set_position(("outward", 8.0))
        poly_ax.patch.set_visible(False)
        ax_bot.set_zorder(1)
        poly_ax.set_zorder(2)
        racmo_ax = ax_bot
        swot_ax = ax_bot
        sohey_ax = ax_bot
        split_prod_axis = True
    else:
        fig, ax_sit_plot = plt.subplots(figsize=(12, 7), facecolor="white")
        ax_sit_plot.set_facecolor((0.96, 0.96, 0.96))
        split_prod_axis = not bool(args.single_yaxis_thickness_and_production)
        if split_prod_axis:
            ax_prod = ax_sit_plot.twinx()
            poly_ax = ax_sit_plot.twinx()
            poly_ax.spines["right"].set_position(("outward", float(args.production_axis_outward_pts)))
            ax_prod.patch.set_visible(False)
            poly_ax.patch.set_visible(False)
            ax_prod.set_zorder(ax_sit_plot.get_zorder() + 1)
            poly_ax.set_zorder(ax_sit_plot.get_zorder() + 2)
            ax_sit_plot.set_zorder(ax_sit_plot.get_zorder())
            racmo_ax = ax_prod
            swot_ax = ax_prod
            sohey_ax = ax_prod
        else:
            poly_ax = ax_sit_plot.twinx()
            poly_ax.set_facecolor("none")
            racmo_ax = ax_sit_plot
            swot_ax = ax_sit_plot
            sohey_ax = ax_sit_plot

    # --- RACMO: band + faint mean line (background-like)
    if racmo_dates_c is not None and racmo_vals_c is not None and len(racmo_dates_c) > 0:
        racmo_mean = np.array(racmo_vals_c, dtype=float)

        if racmo_std_c is not None and len(racmo_std_c) == len(racmo_vals_c):
            racmo_std = np.array(racmo_std_c, dtype=float)
            racmo_ax.fill_between(
                racmo_dates_c, racmo_mean - racmo_std, racmo_mean + racmo_std,
                color=color_racmo, alpha=0.10, linewidth=0.0,
                label="RACMO", zorder=1
            )

        racmo_ax.plot(
            racmo_dates_c, racmo_mean,
            color=color_racmo, linewidth=2.5, alpha=0.20,
            label="", zorder=2  # No label - already labeled by fill_between
        )

    # --- Sohey / PMW: ice production (m/month), same axis as RACMO when split
    if (sohey_dates_c is not None and sohey_mean_c is not None and
        sohey_std_c is not None and len(sohey_dates_c) > 0):

        sohey_label = "Sohey HI (ice production, m/month)"
        if args.sohey_expand_daily:
            sohey_label = "PMW (AMSR2) (m/month)"

        sohey_mean_arr = np.array(sohey_mean_c, dtype=float)
        sohey_std_arr  = np.array(sohey_std_c, dtype=float)

        sohey_ax.fill_between(
            sohey_dates_c, sohey_mean_arr - sohey_std_arr, sohey_mean_arr + sohey_std_arr,
            alpha=0.12, color=color_sohey, linewidth=0.0,
            label=sohey_label, zorder=1
        )
        sohey_ax.plot(
            sohey_dates_c, sohey_mean_arr,
            linewidth=2.5, alpha=0.20, color=color_sohey,
            label="", zorder=2  # No label - already labeled by fill_between
        )

    # --- ESACCI (draw before IS2+SWOT so thinner orange line sits *under* red scatter/curve; zorder kept low)
    if esacci_dates_c is not None and esacci_mean_c is not None and len(esacci_dates_c) > 0:
        esacci_mean_arr = np.array(esacci_mean_c, dtype=float)
        esacci_std_arr = np.array(esacci_std_c, dtype=float) if esacci_std_c is not None else None

        color_esacci_line = "#f28c38"
        color_esacci_fill = "#ffd4a8"

        if esacci_std_arr is not None and len(esacci_std_arr) == len(esacci_mean_arr):
            ax_sit_plot.fill_between(
                esacci_dates_c,
                esacci_mean_arr - esacci_std_arr,
                esacci_mean_arr + esacci_std_arr,
                color=color_esacci_fill,
                alpha=0.55,
                linewidth=0.0,
                label="ESACCI (polynya mean ±1σ)",
                zorder=2,
            )
        else:
            ax_sit_plot.plot(
                [], [],
                color=color_esacci_line,
                linewidth=0.0,
                label="ESACCI (polynya mean)",
                zorder=2,
            )

        ax_sit_plot.plot(
            esacci_dates_c,
            esacci_mean_arr,
            color=color_esacci_line,
            linewidth=2.2,
            linestyle="-",
            alpha=0.95,
            label="" if esacci_std_arr is not None else "ESACCI (polynya mean)",
            zorder=3,
        )

    # --- SIT: errorbars + scatter (above ESACCI)
    ax_sit_plot.errorbar(
        sit_times_c, sit_mean_c, yerr=sit_std_c,
        fmt="none", capsize=4, elinewidth=2.5, alpha=0.25,
        color=color_sit, label="", zorder=12
    )
    ax_sit_plot.scatter(
        sit_times_c, sit_mean_c,
        s=sit_sizes_c, alpha=0.45, color=color_sit,
        edgecolors="none", linewidth=0.0,
        label="SWOT-IS-2", zorder=13
    )
    # --- Optional: ±lag-day pooled IS2+SWOT thickness (all segment samples in window)
    if args.sit_pooled_track_line and sit_times_c is not None and len(sit_times_c) > 0:
        plab = args.sit_pooled_label
        if not plab:
            if use_lagrangian_pooled:
                plab = (
                    f"SWOT-IS-2 daily Lagrangian pooled (±{args.sit_pooled_lag_days:g} d, {args.sit_pooled_stat})"
                )
            else:
                plab = f"IS2+SWOT daily pooled (±{args.sit_pooled_lag_days:g} d, {args.sit_pooled_stat})"
        t_line = pooled_daily_times_c
        y_line = pooled_daily_center_c
        plo = pooled_daily_lo_c
        phi = pooled_daily_hi_c
        if args.sit_pooled_band and plo is not None and np.any(np.isfinite(plo)):
            ax_sit_plot.fill_between(
                t_line,
                plo,
                phi,
                color=color_sit,
                alpha=0.14,
                linewidth=0.0,
                label="",
                zorder=11,
            )
        if np.any(np.isfinite(y_line)):
            ax_sit_plot.plot(
                t_line,
                y_line,
                color="#7a0f12",
                linewidth=2.4,
                linestyle="-",
                alpha=0.95,
                label=plab,
                zorder=14,
            )
            print(
                f"[INFO] IS2+SWOT daily {'Lagrangian ' if use_lagrangian_pooled else ''}pooled thickness line drawn "
                f"({args.sit_pooled_stat} of samples in ±{args.sit_pooled_lag_days:g} d windows; "
                f"see pooled count lines above)."
            )
    # --- SWOT-IS2 production (budget residual), same color family as IS2+SWOT
    if (
        (not args.swot_lower_only_pooled_simple)
        and (not args.hide_swot_prod_line)
        and swot_prod_dates_c is not None
        and swot_prod_vals_c is not None
        and len(swot_prod_dates_c) > 0
    ):
        v = np.asarray(swot_prod_vals_c, dtype=float)
        elo = (
            np.asarray(swot_prod_err_lo_c, dtype=float)
            if swot_prod_err_lo_c is not None
            else None
        )
        ehi = (
            np.asarray(swot_prod_err_hi_c, dtype=float)
            if swot_prod_err_hi_c is not None
            else None
        )
        esym = (
            np.asarray(swot_prod_err_sym_c, dtype=float)
            if swot_prod_err_sym_c is not None
            else None
        )
        if np.any(np.isfinite(v)):
            emode = args.swot_prod_err_mode
            if emode != "none":
                if emode == "symmetric" and esym is not None and np.any(np.isfinite(esym)):
                    swot_ax.errorbar(
                        swot_prod_dates_c,
                        v,
                        yerr=esym,
                        fmt="none",
                        ecolor=color_sit,
                        elinewidth=1.2,
                        capsize=3,
                        alpha=0.75,
                        zorder=3,
                    )
                elif emode == "asymmetric" and elo is not None and ehi is not None and np.any(
                    np.isfinite(elo) | np.isfinite(ehi)
                ):
                    swot_ax.errorbar(
                        swot_prod_dates_c,
                        v,
                        yerr=np.vstack([elo, ehi]),
                        fmt="none",
                        ecolor=color_sit,
                        elinewidth=1.2,
                        capsize=3,
                        alpha=0.75,
                        zorder=3,
                    )
            swot_ax.plot(
                swot_prod_dates_c,
                v,
                color=color_sit,
                linestyle="--",
                linewidth=2.2,
                marker="D",
                markersize=4,
                alpha=0.85,
                label=args.swot_prod_label,
                zorder=4,
            )
    # --- Eulerian closure P_thermo (green squares); same m/month as decomposition figure ---
    if (
        (not args.swot_lower_only_pooled_simple)
        and swot_pthermo_dates_c is not None
        and swot_pthermo_vals_c is not None
        and len(swot_pthermo_dates_c) > 0
    ):
        pv = np.asarray(swot_pthermo_vals_c, dtype=float)
        if np.any(np.isfinite(pv)):
            _pth_lab = (
                args.swot_closure_p_thermo_label
                if args.swot_closure_p_thermo_label
                else r"SWOT–IS2 $P_{\mathrm{thermo}}$ (budget closure)"
            )
            swot_ax.plot(
                swot_pthermo_dates_c,
                pv,
                color=color_sohey,
                linestyle="-",
                linewidth=2.0,
                marker="s",
                markersize=4,
                alpha=0.92,
                label=_pth_lab,
                zorder=5,
            )
    # --- Optional storage-only SWOT estimate from dMdt/(rho*A) in CSV (purple); omit with --hide_swot_storage_dmdt_line
    if (
        (not args.swot_lower_only_pooled_simple)
        and (not args.hide_swot_storage_dmdt_line)
        and swot_storage_dates_c is not None
        and swot_storage_vals_c is not None
        and len(swot_storage_dates_c) > 0
    ):
        sv = np.asarray(swot_storage_vals_c, dtype=float)
        if np.any(np.isfinite(sv)):
            swot_ax.plot(
                swot_storage_dates_c,
                sv,
                color="tab:purple",
                linestyle="-.",
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                alpha=0.85,
                label=args.swot_storage_label,
                zorder=4,
            )
    # Olive: pooled-daily storage d(Ah)/dt; omit with --hide_swot_storage_pooled_line
    if (
        (not args.swot_lower_only_pooled_simple)
        and (not args.hide_swot_storage_pooled_line)
        and swot_storage_pooled_dates_c is not None
        and swot_storage_pooled_vals_c is not None
        and len(swot_storage_pooled_dates_c) > 0
    ):
        spv = np.asarray(swot_storage_pooled_vals_c, dtype=float)
        if np.any(np.isfinite(spv)):
            _stor_lab = "SWOT-IS2 storage from pooled SIT"
            if use_lagrangian_pooled:
                _stor_lab += " (Lagrangian h)"
            _disp = args.swot_pooled_storage_display
            if _disp == "mask_nonpos":
                spv = np.where(np.isfinite(spv) & (spv > 0), spv, np.nan)
                _stor_lab += " (v>0 only)"
            elif _disp == "abs":
                spv = np.where(np.isfinite(spv), np.abs(spv), np.nan)
                _stor_lab += " (|column rate|)"
            if np.any(np.isfinite(spv)):
                swot_ax.plot(
                    swot_storage_pooled_dates_c,
                    spv,
                    color="tab:olive",
                    linestyle="--",
                    linewidth=1.8,
                    marker=".",
                    markersize=2.5,
                    alpha=0.9,
                    label=_stor_lab,
                    zorder=4,
                )
    if (
        (not args.swot_lower_only_pooled_simple)
        and swot_simple_dates_c is not None
        and swot_simple_vals_c is not None
        and len(swot_simple_dates_c) > 0
    ):
        qv = np.asarray(swot_simple_vals_c, dtype=float)
        if np.any(np.isfinite(qv)):
            swot_ax.plot(
                swot_simple_dates_c,
                qv,
                color="tab:brown",
                linestyle=":",
                linewidth=2.0,
                marker="s",
                markersize=3.0,
                alpha=0.85,
                label=args.swot_simple_label,
                zorder=4,
            )
    if (
        swot_simple_pooled_dates_c is not None
        and swot_simple_pooled_vals_c is not None
        and len(swot_simple_pooled_dates_c) > 0
    ):
        pv = np.asarray(swot_simple_pooled_vals_c, dtype=float)
        if np.any(np.isfinite(pv)):
            swot_ax.plot(
                swot_simple_pooled_dates_c,
                pv,
                color=color_sit,
                linestyle="-",
                linewidth=2.0,
                marker="^",
                markersize=6,
                alpha=0.9,
                label=args.swot_simple_pooled_label,
                zorder=4,
            )

    # --- Axes labels & style
    time_ax = ax_bot if use_two_panels else ax_sit_plot
    if use_two_panels:
        ax_sit_plot.set_xlabel("")
        ax_bot.set_xlabel("Date")
    else:
        ax_sit_plot.set_xlabel("Date")

    if use_two_panels:
        ax_sit_plot.set_ylabel("Sea ice thickness (m)", color=color_sit)
        ax_bot.set_ylabel(
            r"Apparent mass tendency, $P_{\mathrm{app}}$ (m/month)",
            color=color_racmo,
        )
        ax_bot.tick_params(axis="y", labelcolor=color_racmo, direction="out", length=5, width=0.8)
        ax_bot.spines["left"].set_visible(True)
        ax_bot.spines["left"].set_color(color_racmo)
        ax_bot.spines["left"].set_linewidth(1.8)
    elif split_prod_axis:
        ax_sit_plot.set_ylabel("Sea ice thickness (m)", color=color_sit)
        ax_prod.set_ylabel(
            r"Apparent mass tendency, $P_{\mathrm{app}}$ (m/month)",
            color=color_racmo,
        )
        ax_prod.tick_params(axis="y", labelcolor=color_racmo, direction="out", length=5, width=0.8)
        ax_prod.spines["right"].set_visible(True)
        ax_prod.spines["right"].set_color(color_racmo)
        ax_prod.spines["right"].set_linewidth(1.8)
    else:
        ax_sit_plot.set_ylabel(
            "Sea ice thickness (m) / ice production incl. PMW & RACMO (m/month)",
            color=color_sit,
        )

    ax_sit_plot.tick_params(axis="y", labelcolor=color_sit, direction="out", length=5, width=0.8)

    # Grid: white, solid, drawn on top (like HSSW style)
    grid_axes = [ax_sit_plot, ax_bot] if use_two_panels else [ax_sit_plot]
    for _gax in grid_axes:
        _gax.grid(True)
        _gax.grid(color="white", linestyle="-", linewidth=1.0, alpha=1.0)
        _gax.set_axisbelow(False)
        _gax.tick_params(direction="out", length=5, width=0.8)

    # Date formatting (shared x when two panels)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    time_ax.xaxis.set_major_locator(locator)
    time_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(time_ax.xaxis.get_majorticklabels(), ha="right")
    if use_two_panels:
        plt.setp(ax_sit_plot.get_xticklabels(), visible=False)

    # Color left spine (optional, matches your old style but still “clean”)
    ax_sit_plot.spines["left"].set_color(color_sit)
    ax_sit_plot.spines["left"].set_linewidth(1.8)
    ax_sit_plot.spines["bottom"].set_linewidth(0.8)
    if use_two_panels:
        ax_bot.spines["bottom"].set_linewidth(0.8)

    # --- AMSR polynya area on right axis (outer spine when split_prod_axis)
    poly_ax.set_facecolor("none")

    poly_ax.plot(
        amsr_days_c, amsr_area_c,
        color=color_amsr, linewidth=2.3, marker="h", markersize=4.5,
        alpha=0.80, label="Polynya area", zorder=5
    )

    ylabel_poly = "AMSR polynya area (km²)"
    if amsr_area_weighting == "openwater":
        ylabel_poly = "AMSR coastal polynya area (km²)"
    poly_ax.set_ylabel(ylabel_poly, color=color_amsr)
    poly_ax.tick_params(axis="y", labelcolor=color_amsr, direction="out", length=5, width=0.8)

    poly_ax.spines["right"].set_visible(True)
    poly_ax.spines["right"].set_color(color_amsr)
    poly_ax.spines["right"].set_linewidth(1.8)

    # --- Title
    if args.title:
        ax_sit_plot.set_title(args.title, fontweight="normal", pad=10)

    # --- Fig. 4 snapshot markers (lower panel only in two-panel mode; no default regime shading)
    snapshot_legend_handle = None
    if use_fig4_annotations:
        regime_axes = (
            ([ax_sit_plot, ax_bot] if use_two_panels else [ax_sit_plot])
            if fig4_regime_spans
            else []
        )
        snapshot_axes = [ax_bot] if use_two_panels else [ax_sit_plot]
        snap_times = sit_times_c if args.mark_swot_snapshots else None
        snapshot_legend_handle = _add_fig4_time_annotations(
            regime_axes=regime_axes,
            snapshot_axes=snapshot_axes,
            regime_spans=fig4_regime_spans,
            snapshot_times=snap_times,
            snapshot_alpha=float(args.swot_snapshot_alpha),
        )
        if fig4_regime_spans:
            print(
                "[INFO] Optional regime shading — "
                + "; ".join(
                    f"{t0.date()}–{t1.date()} ({lab})" for t0, t1, lab in fig4_regime_spans
                )
            )
        if args.mark_swot_snapshots and sit_times_c is not None and len(sit_times_c) > 0:
            where = "lower panel" if use_two_panels else "main panel"
            print(
                f"[INFO] SWOT–IS2 snapshot markers: {len(sit_times_c)} vertical lines on {where} "
                f"(alpha={args.swot_snapshot_alpha:g})."
            )

    # --- Legend(s)
    if use_two_panels:
        leg_top_h, leg_top_l = [], []
        h, lab = ax_sit_plot.get_legend_handles_labels()
        leg_top_h.extend(h)
        leg_top_l.extend(lab)
        leg1_top = ax_sit_plot.legend(
            leg_top_h, leg_top_l,
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.0,
            edgecolor="black",
            ncol=2,
            bbox_to_anchor=(0.01, 0.995),
        )
        ax_sit_plot.add_artist(leg1_top)

        leg_bot_h, leg_bot_l = [], []
        for _ax in (ax_bot, poly_ax):
            h, lab = _ax.get_legend_handles_labels()
            leg_bot_h.extend(h)
            leg_bot_l.extend(lab)
        if snapshot_legend_handle is not None:
            leg_bot_h.append(snapshot_legend_handle)
            leg_bot_l.append(snapshot_legend_handle.get_label())
        leg1_bot = ax_bot.legend(
            leg_bot_h, leg_bot_l,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.995),
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.0,
            edgecolor="black",
            ncol=2,
        )
        ax_bot.add_artist(leg1_bot)
    else:
        leg_handles, leg_labels = [], []
        for _ax in (ax_sit_plot, ax_prod, poly_ax):
            if _ax is None:
                continue
            h, lab = _ax.get_legend_handles_labels()
            leg_handles.extend(h)
            leg_labels.extend(lab)
        if snapshot_legend_handle is not None:
            leg_handles.append(snapshot_legend_handle)
            leg_labels.append(snapshot_legend_handle.get_label())
        leg1 = ax_sit_plot.legend(
            leg_handles, leg_labels,
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.0,
            edgecolor="black",
            ncol=3,
        )
        ax_sit_plot.add_artist(leg1)

    # --- Marker-size legend for SIT (top / thickness axis)
    nseg_vals = np.array(sit_nseg_c, dtype=float)
    nseg_vals = nseg_vals[np.isfinite(nseg_vals)]
    if nseg_vals.size > 0:
        q = np.quantile(nseg_vals, [0.25, 0.50, 0.75])
        q = np.unique(np.round(q).astype(int))
        if q.size == 1:
            q = np.array([max(1, q[0] // 2), q[0], q[0] * 2], dtype=int)

        def _size_from_nseg(n):
            return size_min + size_scale * np.sqrt(max(float(n), 1.0))

        size_handles, size_labels = [], []
        for n in q[:3]:
            size_handles.append(
                ax_sit_plot.scatter([], [], s=_size_from_nseg(n), alpha=0.45,
                                    color=color_sit, edgecolors="none")
            )
            size_labels.append(f"{int(n)} segments")

        leg2 = ax_sit_plot.legend(
            size_handles, size_labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.81),
            frameon=False,
            fontsize=12,
            ncol=1,
        )
        ax_sit_plot.add_artist(leg2)
        if use_two_panels:
            ax_sit_plot.add_artist(leg1_top)
        else:
            ax_sit_plot.add_artist(leg1)

    if use_two_panels:
        fig.align_ylabels([ax_sit_plot, ax_bot])

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight", facecolor="white")
    print(
        f"[INFO] Saved figure: {args.out_png} | "
        f"Lagrangian pooled SIT: {'ON' if use_lagrangian_pooled else 'OFF'} "
        f"(pooled storage/residual curves use signed d(Ah)/dt; red pooled-simple still obeys --swot_prod_plot_signed)."
    )
    plt.show()


    # ---- Optionally save merged NPZ ----
    if args.save_npz_out:
        out = dict(d)
        if racmo_dates_c is not None and racmo_vals_c is not None:
            out["racmo_times_str"] = np.array([str(t) for t in racmo_dates_c], dtype=object)
            out["racmo_values"] = np.array(racmo_vals_c, dtype=float)
            out["racmo_var"] = np.array([args.racmo_var], dtype=object)
        if sohey_dates_c is not None and sohey_mean_c is not None:
            out["sohey_times_str"] = np.array([str(t) for t in sohey_dates_c], dtype=object)
            out["sohey_values"] = np.array(sohey_mean_c, dtype=float)
            out["sohey_divide_by_30"] = np.array([bool(args.sohey_divide_by_30)])
            out["sohey_expand_daily"] = np.array([bool(args.sohey_expand_daily)])
            out["sohey_coastal_month_offset"] = np.array([int(args.sohey_coastal_month_offset)])
        if esacci_dates_c is not None and esacci_mean_c is not None:
            out["esacci_times_str"] = np.array([str(t) for t in esacci_dates_c], dtype=object)
            out["esacci_values"] = np.array(esacci_mean_c, dtype=float)
            if esacci_std_c is not None:
                out["esacci_std"] = np.array(esacci_std_c, dtype=float)
            out["esacci_polynya_threshold"] = np.array([float(args.esacci_polynya_threshold)])
        np.savez(args.save_npz_out, **out)
        print(f"[INFO] Saved merged plot-elements NPZ: {args.save_npz_out}")


if __name__ == "__main__":
    main()
