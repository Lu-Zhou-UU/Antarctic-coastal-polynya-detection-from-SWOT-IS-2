#!/usr/bin/env python3
"""
Read RACMO Ross Sea MAT (future or historical), compute net heat flux and daily
sea ice production, mask coastal polynya (RACMO_coastal_polynya_detection.m logic),
and export daily ice production and HSSW formation to NPZ (coastal polynya only).

Grid alignment:
  - read_RACMO_future.m: Ross Sea subset i_min=200, i_max=726, j_min=1, j_max=300 -> 527 x 300.
  - read_RACMO_mask.m / PXANT11_RossSea_mask.mat: same box; LSM mask(200:end, 1:300) -> lsm_ross.

Supports:
  - Future: RACMO_future_2015_2099_RossSea_CESM_ssp370.mat (year_start/year_end in MAT)
  - Historical: RACMO_historical_RossSea_CESM.mat (1985-2014)

Variable names: tries *_ross first (e.g. str_ross), then *_hist (e.g. str_hist).

MATLAB logic (RACMO_coastal_polynya_detection):
  flux = str + ssr + hfls + hfss; ice_product = -flux / (rho_i * L_f) * 86400 [m/day]
  polynya = SIC < threshold (default 0.7); pack flood from (400,1); open-ocean flood from (floor(ny/2), 200).
  coastal = polynyas_all with open-ocean component set to NaN (ice/HSSW only in coastal polynya).

Usage:
  python racmo_future_ice_production_hssw.py RACMO_future_2015_2099_RossSea_CESM_ssp370.mat --out_npz future_ice_hssw.npz
  python racmo_future_ice_production_hssw.py RACMO_historical_RossSea_CESM.mat --out_npz historical_ice_hssw.npz --lsm_mat PXANT11_RossSea_mask.mat

If the MAT holds the full RACMO grid, apply the same subset as MATLAB (ice(200:end,1:300)):
  --row_start 200 --col_end 300
"""

import argparse
import calendar
from datetime import datetime, timedelta

import numpy as np

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None
try:
    from scipy.ndimage import label as ndi_label
    HAS_SCIPY_NDI = True
except Exception:
    HAS_SCIPY_NDI = False

try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

# Constants (match MATLAB)
RHO_I = 950.0      # kg/m^3
L_F = 3.34e5       # J/kg
SEC_PER_DAY = 86400.0
SIC_THRESHOLD = 0.7   # polynya where SIC < 70% (match RACMO_coastal_polynya_detection.m)
# HSSW
S_LSSW = 34.79
S_HSSW = 34.85
FRAZIL_FRAC = 0.31
RHO_HSSW = 1028.0
R_EARTH = 6.371e6  # m

# Variable names to load from RACMO MAT (v7.3); only these are read to avoid MemoryError
RACMO_MAT_KEYS = [
    "str_ross", "str_hist", "str",
    "ssr_ross", "ssr_hist", "ssr",
    "hfls_ross", "hfls_hist", "hfls",
    "hfss_ross", "hfss_hist", "hfss",
    "siconca_ross", "siconca_hist", "siconca",
    "lat_ross", "lat_hist", "lat",
    "lon_ross", "lon_hist", "lon",
    "years", "year_start", "year_end",
    "lsm_ross", "LSM_ross", "lsm", "LSM",
]

LSM_MAT_KEYS = ["lsm_ross", "LSM_ross", "lsm", "LSM"]

# Don't load 3D arrays larger than this (bytes); read by slice instead to avoid MemoryError
LARGE_ARRAY_BYTES = 100 * 1024 * 1024  # 100 MB


def _h5_read_dataset(ds):
    arr = np.array(ds)
    if arr.shape == ():
        return arr
    if arr.ndim >= 2:
        arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
    return arr


def _h5_read_slice(ds, t):
    """Read time slice t from an h5py Dataset (MAT v7.3). Returns (ny, nx) with MATLAB order undone."""
    sh = ds.shape
    if len(sh) != 3:
        raise ValueError("Expected 3D dataset, got shape {}".format(sh))
    # Time dimension is usually the largest (e.g. 10950) or the first in HDF5 (MATLAB col-major -> nt first)
    time_dim = int(np.argmax(sh))
    if time_dim == 0:
        slab = np.array(ds[t, :, :])
        return np.transpose(slab)  # (nx, ny) -> (ny, nx)
    elif time_dim == 2:
        slab = np.array(ds[:, :, t])
        return slab if slab.shape[0] > slab.shape[1] else np.transpose(slab)
    else:
        slab = np.array(ds[:, t, :])
        return np.transpose(slab)


def load_mat_any(path, keys=None, lazy_large=True):
    """
    Load MAT file (v7 or v7.3). For v7.3 with keys= and lazy_large=True, large 3D arrays
    are not loaded; instead their h5py Dataset refs are returned and the file is left open.
    Returns (data_dict, file_handle). Caller must close file_handle when done (or pass None).
    """
    path = str(path)
    if loadmat is not None:
        try:
            data = loadmat(path)
            if keys is not None:
                data = {k: data[k] for k in keys if k in data}
            return (data, None)
        except NotImplementedError:
            pass
        except Exception:
            pass
    if not HAS_H5PY:
        raise RuntimeError("MAT v7.3 detected. Install h5py.")
    f = h5py.File(path, "r")
    out = {}
    need_handle = False

    def visit(name, obj):
        nonlocal need_handle
        if not isinstance(obj, h5py.Dataset):
            return
        key = name.split("/")[-1]
        if keys is not None and key not in keys:
            return
        nbytes = obj.size * (obj.dtype.itemsize if hasattr(obj.dtype, "itemsize") else 8)
        if lazy_large and keys is not None and obj.ndim == 3 and nbytes > LARGE_ARRAY_BYTES:
            out[key] = obj
            need_handle = True
        else:
            out[key] = _h5_read_dataset(obj)

    f.visititems(visit)
    if not need_handle:
        f.close()
        f = None
    return (out, f)


def _get_mat_var(mat, *candidate_keys):
    """Return first existing array from mat for candidate keys. Raises KeyError if none found."""
    for k in candidate_keys:
        if k in mat:
            return np.array(mat[k], dtype=float).squeeze()
    raise KeyError("None of {} found in MAT. Keys: {}".format(candidate_keys, sorted(k for k in mat if not k.startswith("_"))))


def _grid_cell_areas_m2(lat_2d, lon_2d):
    """Approximate grid cell area (m^2) for a lat/lon grid. lat, lon in degrees."""
    lat = np.array(lat_2d, dtype=float)
    lon = np.array(lon_2d, dtype=float)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    # dlat, dlon in radians: diff has shape (ny-1,nx) / (ny,nx-1); fill edges from neighbor
    dlat = np.zeros_like(lat)
    dlon = np.zeros_like(lon)
    dlat[:-1, :] = np.diff(lat_rad, axis=0)
    dlat[-1, :] = dlat[-2, :]
    dlon[:, :-1] = np.diff(lon_rad, axis=1)
    dlon[:, -1] = dlon[:, -2]
    area = (R_EARTH ** 2) * np.cos(lat_rad) * np.abs(dlat * dlon)
    return np.maximum(area, 1.0)  # avoid zero


def _coastal_polynya_mask_flood(sic_frac_2d, lsm_2d=None, coastal_edges="south_west"):
    """
    Coastal polynya mask matching MATLAB RACMO_coastal_polynya_detection(ice_test, threshold):

      1) ice_test(mask==1) = NaN  (land -> NaN)
      2) pack: SIC < thr -> 0, SIC >= thr -> 1, NaN -> 0; aice1 = flood(1, 400, aice1, 100).
         Seed at (row 400, col 1) 1-based -> 0-based (399, 0).
      3) aaice = ice_con; aaice(aice1==100)=NaN; aaice(aaice>=thr)=NaN -> polynyas_all.
      4) aaice binary 1/0; aaice = flood(floor(ny/2), 200, aaice, 100).
         Seed at (row floor(ny/2), col 200) 1-based -> 0-based (floor(ny/2)-1, 199).
      5) coastal = polynyas_all with aaice==100 set to NaN (MATLAB: coastal(find(aaice==100))=NaN).
         So coastal polynya = polynya cells NOT in the open-ocean flood component.
    Returns: boolean mask True only in coastal polynya (excludes open-ocean polynya).
    """
    if not HAS_SCIPY_NDI:
        return np.isfinite(sic_frac_2d) & (sic_frac_2d < SIC_THRESHOLD)

    sic = np.array(sic_frac_2d, dtype=float)
    ny, nx = sic.shape
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    if lsm_2d is None:
        return np.isfinite(sic) & (sic < SIC_THRESHOLD)

    # Step 1: land -> NaN (MATLAB: ice_test(find(mask==1)) = NaN)
    lsm = np.array(lsm_2d, dtype=float)
    land = np.isfinite(lsm) & (lsm > 0.5)
    ice_test = sic.copy()
    ice_test[land] = np.nan

    # Step 2: pack binary (MATLAB: ice_test < thr -> 0, >= thr -> 1, NaN -> 0; aice1 = flood(1, 400, aice1, 100))
    pack = np.zeros_like(ice_test, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        pack[np.isfinite(ice_test) & (ice_test >= SIC_THRESHOLD)] = 1

    # Step 3: pack component at seed (row 400, col 1) 1-based -> 0-based (399, 0)
    seed_north_row = min(399, ny - 1)
    seed_north_col = 0
    labeled_pack, n_pack = ndi_label(pack, structure=structure)
    open_pack_mask = np.zeros_like(pack, dtype=bool)
    if n_pack > 0 and 0 <= seed_north_row < ny and 0 <= seed_north_col < nx and labeled_pack[seed_north_row, seed_north_col] > 0:
        open_pack_mask = labeled_pack == labeled_pack[seed_north_row, seed_north_col]

    # Step 4: polynya candidates = polynyas_all (not in pack, SIC < threshold)
    aaice = ice_test.copy()
    aaice[open_pack_mask] = np.nan
    with np.errstate(invalid="ignore"):
        aaice[aaice >= SIC_THRESHOLD] = np.nan
    low_sic = np.isfinite(aaice)
    if not np.any(low_sic):
        return np.zeros_like(low_sic, dtype=bool)

    # Step 5: open-ocean flood seed at (floor(ny/2), 200) 1-based -> 0-based (floor(ny/2)-1, 199)
    seed_open_row = min(max(0, ny // 2 - 1), ny - 1)
    seed_open_col = min(199, nx - 1)

    labeled_low, n_low = ndi_label(low_sic.astype(np.int8), structure=structure)
    if n_low == 0:
        return np.zeros_like(low_sic, dtype=bool)
    # main_label = open-ocean polynya component (the one containing seed)
    if not (0 <= seed_open_row < ny and 0 <= seed_open_col < nx and low_sic[seed_open_row, seed_open_col]):
        return low_sic  # no open-ocean component found; all polynya candidates are coastal
    main_label = labeled_low[seed_open_row, seed_open_col]
    # coastal = polynyas_all with open-ocean (aaice==100) set to NaN -> mask = polynya AND label != main_label
    return (labeled_low > 0) & (labeled_low != main_label)


def build_dates_from_years(year_start, year_end, n_days=None):
    """Build list of daily datetime from year_start to year_end (365/366 per year). If n_days given, truncate to first n_days."""
    dates = []
    for y in range(int(year_start), int(year_end) + 1):
        nd = 366 if calendar.isleap(y) else 365
        base = datetime(y, 1, 1)
        for d in range(nd):
            dates.append(base + timedelta(days=d))
    if n_days is not None:
        dates = dates[:n_days]
    return dates


def main():
    ap = argparse.ArgumentParser(description="RACMO Ross Sea MAT (future or historical): flux -> ice production -> HSSW, export NPZ")
    ap.add_argument("mat", help="Path to RACMO MAT (e.g. RACMO_future_...mat or RACMO_historical_RossSea_CESM.mat)")
    ap.add_argument("--out_npz", default="racmo_future_ice_hssw.npz", help="Output NPZ path")
    ap.add_argument("--year_start", type=int, default=None, help="Override year range start (e.g. 1985 for historical)")
    ap.add_argument("--year_end", type=int, default=None, help="Override year range end (e.g. 2014 for historical)")
    ap.add_argument("--lsm_mat", default=None, help="Optional MAT file containing lsm_ross on Ross Sea grid")
    ap.add_argument("--row_start", type=int, default=None, help="Subset rows from this 1-based index (MATLAB 200:end -> --row_start 200)")
    ap.add_argument("--col_end", type=int, default=None, help="Subset cols up to this 1-based index (MATLAB 1:300 -> --col_end 300)")
    ap.add_argument("--sic_threshold", type=float, default=0.7, help="Polynya SIC threshold (fraction; pack >= this, polynya < this; default 0.7 to match RACMO_coastal_polynya_detection.m)")
    ap.add_argument("--no_flood", action="store_true", help="Do not use flooding; use simple SIC < threshold mask")
    ap.add_argument("--s_lssw", type=float, default=S_LSSW)
    ap.add_argument("--s_hssw", type=float, default=S_HSSW)
    ap.add_argument("--frazil_frac", type=float, default=FRAZIL_FRAC)
    ap.add_argument("--rho_hssw", type=float, default=RHO_HSSW)
    args = ap.parse_args()

    global SIC_THRESHOLD
    SIC_THRESHOLD = args.sic_threshold

    # Load MAT (support both future *_ross and historical *_hist / plain names).
    # v7.3: only requested keys; large 3D arrays are kept as Dataset refs and read by slice.
    mat, mat_file = load_mat_any(args.mat, keys=RACMO_MAT_KEYS)

    def _get_ds(*cands):
        for k in cands:
            v = mat.get(k)
            if v is not None and HAS_H5PY and isinstance(v, h5py.Dataset):
                return v
        return None

    str_ds = _get_ds("str_ross", "str_hist", "str")
    ssr_ds = _get_ds("ssr_ross", "ssr_hist", "ssr")
    hfls_ds = _get_ds("hfls_ross", "hfls_hist", "hfls")
    hfss_ds = _get_ds("hfss_ross", "hfss_hist", "hfss")
    siconca_ds = _get_ds("siconca_ross", "siconca_hist", "siconca")
    lazy = str_ds is not None

    if lazy:
        if not (ssr_ds is not None and hfls_ds is not None and hfss_ds is not None and siconca_ds is not None):
            raise KeyError("Lazy load: need all flux/sic datasets as refs.")
        sh = str_ds.shape
        nt = int(max(sh))
        time_dim = np.argmax(sh)
        if time_dim == 0:
            ny_full, nx_full = sh[2], sh[1]
        elif time_dim == 2:
            ny_full, nx_full = sh[0], sh[1]
        else:
            ny_full, nx_full = sh[0], sh[2]
        # Ensure (ny, nx) = (rows, cols)
        if ny_full < nx_full and time_dim == 0:
            ny_full, nx_full = sh[1], sh[2]
        str_ross = ssr_ross = hfls_ross = hfss_ross = siconca_ross = None
    else:
        str_ross = _get_mat_var(mat, "str_ross", "str_hist", "str")
        ssr_ross = _get_mat_var(mat, "ssr_ross", "ssr_hist", "ssr")
        hfls_ross = _get_mat_var(mat, "hfls_ross", "hfls_hist", "hfls")
        hfss_ross = _get_mat_var(mat, "hfss_ross", "hfss_hist", "hfss")
        siconca_ross = _get_mat_var(mat, "siconca_ross", "siconca_hist", "siconca")
        fill = -9999.0
        for arr in (str_ross, ssr_ross, hfls_ross, hfss_ross, siconca_ross):
            arr[arr == fill] = np.nan
        if str_ross.ndim != 3:
            raise ValueError("Expected str_ross 3D (Y,X,T), got {}".format(str_ross.shape))
        ny_full, nx_full, nt = str_ross.shape
        flux = str_ross + ssr_ross + hfls_ross + hfss_ross
        ice_production_daily = -flux / (RHO_I * L_F) * SEC_PER_DAY
        ice_production_daily = np.where(np.isfinite(ice_production_daily), np.maximum(ice_production_daily, 0.0), np.nan)
        sic_frac = siconca_ross / 100.0 if np.nanmax(siconca_ross) > 2.0 else siconca_ross.copy()

    # Small vars: always from mat (loaded)
    lat_ross = _get_mat_var(mat, "lat_ross", "lat_hist", "lat")
    lon_ross = _get_mat_var(mat, "lon_ross", "lon_hist", "lon")

    r0 = (args.row_start - 1) if args.row_start is not None else 0
    c1 = args.col_end if args.col_end is not None else nx_full
    if r0 >= ny_full or c1 > nx_full or r0 < 0 or c1 <= 0:
        r0, c1 = 0, nx_full
        import sys
        if args.row_start is not None or args.col_end is not None:
            print("Warning: subset row_start={} col_end={} not applied (grid is {}x{}).".format(
                getattr(args, "row_start", None), getattr(args, "col_end", None), ny_full, nx_full), file=sys.stderr)
    ny, nx = ny_full - r0, min(c1, nx_full)

    if not lazy:
        if args.row_start is not None or args.col_end is not None:
            str_ross = str_ross[r0:, :c1].copy()
            ssr_ross = ssr_ross[r0:, :c1].copy()
            hfls_ross = hfls_ross[r0:, :c1].copy()
            hfss_ross = hfss_ross[r0:, :c1].copy()
            siconca_ross = siconca_ross[r0:, :c1].copy()
            lat_ross = lat_ross[r0:, :c1].copy()
            lon_ross = lon_ross[r0:, :c1].copy()
            ice_production_daily = ice_production_daily[r0:, :c1, :].copy()
            sic_frac = sic_frac[r0:, :c1, :].copy()

    if lat_ross.ndim == 2:
        if lat_ross.shape != (ny, nx):
            if lat_ross.shape == (nx, ny):
                lat_ross = lat_ross.T
                lon_ross = lon_ross.T if lon_ross.ndim == 2 else lon_ross
            if lat_ross.shape != (ny, nx):
                lat_ross = lat_ross[r0:, :c1] if lat_ross.shape[0] == ny_full else lat_ross
                lon_ross = lon_ross[r0:, :c1] if lon_ross.shape[0] == ny_full else lon_ross
        if lat_ross.shape != (ny, nx):
            raise ValueError("lat_ross shape {} not compatible with flux grid {}x{}".format(lat_ross.shape, ny, nx))

    lsm_ross = None
    if getattr(args, "lsm_mat", None):
        lsm_data, _ = load_mat_any(args.lsm_mat, keys=LSM_MAT_KEYS)
        for key in ("lsm_ross", "LSM_ross", "lsm", "LSM"):
            if key in lsm_data:
                lsm_ross = np.array(lsm_data[key], dtype=float).squeeze()
                break
    if lsm_ross is None:
        for key in ("lsm_ross", "LSM_ross", "lsm", "LSM"):
            if key in mat:
                v = mat[key]
                if not (HAS_H5PY and isinstance(v, h5py.Dataset)):
                    lsm_ross = np.array(v, dtype=float).squeeze()
                    break

    if lsm_ross is not None and lsm_ross.ndim == 2:
        if lsm_ross.shape == (ny, nx):
            pass
        elif lsm_ross.shape == (nx, ny):
            lsm_ross = lsm_ross.T
        elif lsm_ross.shape == (ny_full, nx_full):
            lsm_ross = lsm_ross[r0:, :c1]
        elif lsm_ross.shape == (nx_full, ny_full):
            lsm_ross = lsm_ross.T[r0:, :c1]
        else:
            raise ValueError("lsm_ross shape {} not compatible with flux grid {}x{}".format(lsm_ross.shape, ny, nx))

    lon_ross = np.where(lon_ross < 0, lon_ross + 360.0, lon_ross)

    area_m2 = _grid_cell_areas_m2(lat_ross, lon_ross)
    if area_m2.ndim == 2 and area_m2.shape != (ny, nx):
        if area_m2.shape == (nx, ny):
            area_m2 = area_m2.T
        else:
            area_m2 = np.broadcast_to(area_m2, (ny, nx)).copy()

    year_start = 1985
    year_end = 2014
    if "year_start" in mat:
        v = mat["year_start"]
        if not (HAS_H5PY and isinstance(v, h5py.Dataset)):
            year_start = int(np.squeeze(v))
    if "year_end" in mat:
        v = mat["year_end"]
        if not (HAS_H5PY and isinstance(v, h5py.Dataset)):
            year_end = int(np.squeeze(v))
    if "years" in mat:
        v = mat["years"]
        if not (HAS_H5PY and isinstance(v, h5py.Dataset)):
            years_arr = np.array(v).ravel()
            if years_arr.size >= 2:
                year_start, year_end = int(years_arr[0]), int(years_arr[-1])
    if args.year_start is not None:
        year_start = args.year_start
    if args.year_end is not None:
        year_end = args.year_end
    day_list = build_dates_from_years(year_start, year_end, n_days=nt)

    ice_mean_daily = np.full(nt, np.nan, dtype=float)
    ice_total_daily = np.full(nt, np.nan, dtype=float)
    polynya_area_km2_daily = np.full(nt, np.nan, dtype=float)
    hssw_sv_daily = np.full(nt, np.nan, dtype=float)
    fill = -9999.0
    sw = args.s_lssw
    si = args.frazil_frac * sw
    delta_s = args.s_hssw - args.s_lssw

    try:
        for t in range(nt):
            if lazy:
                str_t = _h5_read_slice(str_ds, t)
                ssr_t = _h5_read_slice(ssr_ds, t)
                hfls_t = _h5_read_slice(hfls_ds, t)
                hfss_t = _h5_read_slice(hfss_ds, t)
                sic_t = _h5_read_slice(siconca_ds, t)
                if r0 or c1 < nx_full:
                    str_t = str_t[r0:, :c1]
                    ssr_t = ssr_t[r0:, :c1]
                    hfls_t = hfls_t[r0:, :c1]
                    hfss_t = hfss_t[r0:, :c1]
                    sic_t = sic_t[r0:, :c1]
                str_t[str_t == fill] = np.nan
                ssr_t[ssr_t == fill] = np.nan
                hfls_t[hfls_t == fill] = np.nan
                hfss_t[hfss_t == fill] = np.nan
                sic_t[sic_t == fill] = np.nan
                flux_t = str_t + ssr_t + hfls_t + hfss_t
                ice_t = -flux_t / (RHO_I * L_F) * SEC_PER_DAY
                ice_t = np.where(np.isfinite(ice_t), np.maximum(ice_t, 0.0), np.nan)
                sic_frac_t = sic_t / 100.0 if np.nanmax(sic_t) > 2.0 else sic_t
            else:
                sic_frac_t = sic_frac[:, :, t]
                ice_t = ice_production_daily[:, :, t]
            if args.no_flood:
                mask_t = np.isfinite(sic_frac_t) & (sic_frac_t < SIC_THRESHOLD)
            else:
                mask_t = _coastal_polynya_mask_flood(sic_frac_t, lsm_ross, coastal_edges="south_west")
            valid = mask_t & np.isfinite(ice_t) & (ice_t > 0)
            if not np.any(valid):
                continue
            area_valid = area_m2[valid]
            pi_m_day = ice_t[valid]
            pi_m_s = pi_m_day / SEC_PER_DAY
            ice_mean_daily[t] = np.mean(pi_m_day)
            ice_total_daily[t] = np.sum(pi_m_day * area_valid)
            polynya_area_km2_daily[t] = np.sum(area_valid) / 1e6
            salt_flux_total = np.sum(RHO_I * pi_m_s * area_valid * (sw - si) * 1e-3)
            hssw_sv_daily[t] = salt_flux_total / (args.rho_hssw * delta_s * 1e-3) * 1e-6
    finally:
        if mat_file is not None:
            mat_file.close()

    # Save NPZ
    date_strs = np.array([d.strftime("%Y-%m-%d") for d in day_list], dtype=object)
    np.savez(
        args.out_npz,
        date_str=date_strs,
        ice_mean_m_per_day=ice_mean_daily,
        ice_total_m3_per_day=ice_total_daily,
        polynya_area_km2=polynya_area_km2_daily,
        hssw_sv=hssw_sv_daily,
        year_start=np.array([year_start]),
        year_end=np.array([year_end]),
        sic_threshold=np.array([args.sic_threshold]),
        lat_ross=lat_ross,
        lon_ross=lon_ross,
        allow_pickle=True,
    )
    print("Saved {} (n_days={}, year_start={}, year_end={})".format(args.out_npz, nt, year_start, year_end))
    print("  ice_mean (m/day): min={:.4f}, max={:.4f}".format(np.nanmin(ice_mean_daily), np.nanmax(ice_mean_daily)))
    print("  HSSW (Sv): min={:.4f}, max={:.4f}".format(np.nanmin(hssw_sv_daily), np.nanmax(hssw_sv_daily)))


if __name__ == "__main__":
    main()
