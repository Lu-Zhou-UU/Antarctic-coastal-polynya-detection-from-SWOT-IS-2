#!/usr/bin/env python3
"""
From trajectories_backward.npz (backtrack: start = later SWOT 005_455, end = earlier SWOT 005_452),
extract sigma0 and ssha_unfiltered per trajectory from the correct SWOT file and export to NetCDF.

Usage:
  python swot_means_near_trajectories.py \\
    trajectories_backward.npz \\
    SWOT_L3_LR_SSH_Unsmoothed_005_452_20231028T191642_20231028T200727_v2.0.1.nc \\
    SWOT_L3_LR_SSH_Unsmoothed_005_455_20231028T215103_20231028T224230_v2.0.1.nc \\
    --radius_m 2000 --output_nc sigma0_ssha_trajectories.nc
"""

import argparse
import os
import re

import numpy as np

try:
    import netCDF4
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Earth radius (m) for chord-radius conversion
EARTH_R_M = 6371000.0


def _lon_to_180(lon):
    """Normalize longitude to [-180, 180] (handles 0-360 or already -180..180)."""
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180, lon - 360, np.where(lon < -180, lon + 360, lon))


def _latlon_deg_to_xyz(lat_deg, lon_deg):
    """Convert (lat, lon) in degrees to unit-sphere Cartesian (x, y, z)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])


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


def load_tracking_positions(npz_path):
    """Return lat_start, lon_start, lat_final, lon_final, valid (1D arrays)."""
    data = np.load(npz_path)
    lat_start = np.asarray(data["lat_start"], dtype=float).ravel()
    lon_start = np.asarray(data["lon_start"], dtype=float).ravel()
    lat_final = np.asarray(data["lat_final"], dtype=float).ravel()
    lon_final = np.asarray(data["lon_final"], dtype=float).ravel()
    valid = data.get("valid")
    if valid is None:
        valid = (
            np.isfinite(lat_start) & np.isfinite(lon_start)
            & np.isfinite(lat_final) & np.isfinite(lon_final)
        )
    else:
        valid = np.asarray(valid, dtype=bool).ravel()
    return lat_start, lon_start, lat_final, lon_final, valid


def _get_nc_var(nc, candidates, default=None):
    for name in candidates:
        if name in nc.variables:
            return nc.variables[name]
    return default


def _identify_start_end_files(nc_paths):
    """
    Backtrack: start = later pass (005_455), end = earlier pass (005_452).
    Return (path_start, path_end) by parsing cycle_pass from filename (e.g. 005_455, 005_452).
    """
    found = {}
    for p in nc_paths:
        base = os.path.basename(p)
        m = re.search(r"(\d+)_(\d+)", base)
        if m:
            cycle, pass_ = int(m.group(1)), int(m.group(2))
            key = (cycle, pass_)
            found[key] = p
    if len(found) < 2:
        return None, None
    keys = sorted(found.keys(), key=lambda k: (k[0], k[1]))
    # Earlier pass = end (backtrack end), later pass = start (backtrack start)
    path_end = found[keys[0]]
    path_start = found[keys[-1]]
    if keys[0] == keys[-1]:
        return None, None
    return path_start, path_end


def _get_scale_offset(var):
    """Get scale_factor and add_offset from a NetCDF variable."""
    scale, offset = 1.0, 0.0
    for sname in ("scale_factor", "Scale_Factor", "scaleFactor"):
        val = getattr(var, sname, None)
        if val is not None:
            try:
                scale = float(np.asarray(val).ravel()[0] if np.ndim(val) else val)
                break
            except Exception:
                pass
    if scale == 1.0 and hasattr(var, "ncattrs"):
        for name in var.ncattrs():
            if name.lower().replace("-", "_") == "scale_factor":
                try:
                    scale = float(np.asarray(var.getncattr(name)).ravel()[0])
                    break
                except Exception:
                    pass
    for oname in ("add_offset", "Add_Offset", "addOffset"):
        val = getattr(var, oname, None)
        if val is not None:
            try:
                offset = float(np.asarray(val).ravel()[0] if np.ndim(val) else val)
                break
            except Exception:
                pass
    if offset == 0.0 and hasattr(var, "ncattrs"):
        for name in var.ncattrs():
            if name.lower().replace("-", "_") == "add_offset":
                try:
                    offset = float(np.asarray(var.getncattr(name)).ravel()[0])
                    break
                except Exception:
                    pass
    return scale, offset


def _read_scaled(var, fill_to_nan=True):
    """Read a NetCDF variable with scale_factor/add_offset applied exactly once."""
    var.set_auto_scale(False)
    raw = var[:]
    if np.ma.isMaskedArray(raw):
        data = np.ma.asarray(raw, dtype=float).filled(np.nan)
    else:
        data = np.asarray(raw, dtype=float)
    data = data.ravel()
    scale, offset = _get_scale_offset(var)
    data = data * scale + offset
    if fill_to_nan:
        fv = getattr(var, "_FillValue", None)
        if fv is not None and np.issubdtype(type(fv), np.number):
            data[data == float(fv)] = np.nan
    return data


def extract_sigma0_ssha_per_trajectory(nc_path, ref_lat, ref_lon, radius_m=2000.0, debug=False):
    """
    Load one SWOT file and, for each (ref_lat[i], ref_lon[i]), compute mean sigma0 and mean
    ssha_unfiltered over SWOT pixels within radius_m. Backtrack: ref = start -> use 005_455 file;
    ref = end -> use 005_452 file.

    Returns
    -------
    sigma0_mean, ssha_mean, sigma0_std, ssha_std : 1D arrays (n_traj), np.nan where no pixels within radius.
    """
    if not HAS_NETCDF or not os.path.isfile(nc_path):
        return None, None, None, None
    ref_lat = np.asarray(ref_lat, dtype=float).ravel()
    ref_lon = np.asarray(ref_lon, dtype=float).ravel()
    ref_lon = _lon_to_180(ref_lon)
    n_traj = len(ref_lat)
    sigma0_mean = np.full(n_traj, np.nan, dtype=np.float64)
    ssha_mean = np.full(n_traj, np.nan, dtype=np.float64)
    sigma0_std = np.full(n_traj, np.nan, dtype=np.float64)
    ssha_std = np.full(n_traj, np.nan, dtype=np.float64)

    nc = netCDF4.Dataset(nc_path, "r")
    lat_var = _get_nc_var(nc, ["latitude", "lat", "Latitude"])
    lon_var = _get_nc_var(nc, ["longitude", "lon", "Longitude"])
    if lat_var is None or lon_var is None:
        nc.close()
        return sigma0_mean, ssha_mean, sigma0_std, ssha_std

    raw = lat_var[:]
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    lat_swot = np.asarray(raw, dtype=np.float32).ravel()
    raw = lon_var[:]
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    lon_swot = np.asarray(raw, dtype=np.float32).ravel()
    lon_swot = _lon_to_180(lon_swot)
    if lat_swot.size != lon_swot.size:
        nc.close()
        return sigma0_mean, ssha_mean, sigma0_std, ssha_std

    sigma0_var = _get_nc_var(nc, ["sigma0", "sig0", "sigma_0"])
    ssha_var = _get_nc_var(nc, ["ssha_unfiltered", "ssh_unfiltered", "ssha", "sla"])
    sigma0 = _read_scaled(sigma0_var) if sigma0_var is not None else None
    ssha = _read_scaled(ssha_var) if ssha_var is not None else None
    if sigma0 is not None and sigma0.size != lat_swot.size:
        sigma0 = None
    if ssha is not None and ssha.size != lat_swot.size:
        ssha = None
    nc.close()

    ok = np.isfinite(lat_swot) & np.isfinite(lon_swot)
    if sigma0 is not None:
        ok = ok & np.isfinite(sigma0)
    if ssha is not None:
        ok = ok & np.isfinite(ssha)
    lat_swot = lat_swot[ok]
    lon_swot = lon_swot[ok]
    if sigma0 is not None:
        sigma0 = sigma0[ok]
    if ssha is not None:
        ssha = ssha[ok]
    n_swot = len(lat_swot)
    if n_swot == 0:
        return sigma0_mean, ssha_mean, sigma0_std, ssha_std

    margin_deg = max(0.5, (radius_m / 111320.0) * 2.0)
    lat_lo, lat_hi = np.min(ref_lat) - margin_deg, np.max(ref_lat) + margin_deg
    lon_lo, lon_hi = np.min(ref_lon) - margin_deg, np.max(ref_lon) + margin_deg
    in_box = (
        (lat_swot >= lat_lo) & (lat_swot <= lat_hi) &
        (lon_swot >= lon_lo) & (lon_swot <= lon_hi)
    )
    n_cand = int(np.sum(in_box))
    if n_cand == 0:
        if debug:
            print(f"[extract] No SWOT pixels in box for {os.path.basename(nc_path)}")
        return sigma0_mean, ssha_mean, sigma0_std, ssha_std

    lat_c = lat_swot[in_box]
    lon_c = lon_swot[in_box]
    sigma0_c = sigma0[in_box] if sigma0 is not None else None
    ssha_c = ssha[in_box] if ssha is not None else None
    radius_chord = 2.0 * np.sin(radius_m / (2.0 * EARTH_R_M))
    swot_xyz = _latlon_deg_to_xyz(lat_c, lon_c)
    tree_swot = cKDTree(swot_xyz)
    ref_xyz = _latlon_deg_to_xyz(ref_lat, ref_lon)
    for i in range(n_traj):
        hits = tree_swot.query_ball_point(ref_xyz[i], radius_chord)
        if len(hits) == 0:
            continue
        if sigma0_c is not None:
            vals = sigma0_c[hits]
            sigma0_mean[i] = np.nanmean(vals)
            sigma0_std[i] = np.nanstd(vals)
        if ssha_c is not None:
            vals = ssha_c[hits]
            ssha_mean[i] = np.nanmean(vals)
            ssha_std[i] = np.nanstd(vals)
    if debug:
        n_ok = np.sum(np.isfinite(sigma0_mean)) if sigma0_mean is not None else 0
        print(f"[extract] {os.path.basename(nc_path)}: {n_traj} trajectories, {n_ok} with sigma0/ssha within {radius_m} m")
    return sigma0_mean, ssha_mean, sigma0_std, ssha_std


def compute_means_near_trajectories(
    npz_path,
    nc_paths,
    radius_m=2000.0,
    debug=False,
):
    """
    For each SWOT NetCDF, select pixels within radius_m of any trajectory
    start/end point, then compute mean of sigma0 and ssha_unfiltered.

    Returns
    -------
    list of dict
        One dict per NetCDF with keys: file, n_pixels, mean_sigma0, mean_ssha_unfiltered,
        std_sigma0, std_ssha_unfiltered (or None if variable missing).
    """
    if not HAS_NETCDF:
        raise RuntimeError("netCDF4 is required.")

    lat_start, lon_start, lat_final, lon_final, valid = load_tracking_positions(npz_path)
    # Use both start and end positions; only valid trajectories
    lat_ref = np.concatenate([lat_start[valid], lat_final[valid]])
    lon_ref = np.concatenate([lon_start[valid], lon_final[valid]])
    lon_ref = _lon_to_180(lon_ref)  # normalize so box/tree match SWOT (often 0-360)
    n_ref = len(lat_ref)
    if debug:
        print(f"[REF] Loaded {n_ref} reference points (start+end, valid only) from {npz_path}")
        print(f"[REF] Extent: lat [{np.min(lat_ref):.2f}, {np.max(lat_ref):.2f}]  lon [{np.min(lon_ref):.2f}, {np.max(lon_ref):.2f}]")

    # Build k-d tree on ref points (unit-sphere 3D) for fast "within radius" queries.
    # Chord radius on unit sphere: arc_m = radius_m / R, chord = 2*sin(arc_m/2).
    ref_tree = None
    radius_chord = None
    if HAS_SCIPY and n_ref > 0:
        ref_xyz = _latlon_deg_to_xyz(lat_ref, lon_ref)
        ref_tree = cKDTree(ref_xyz)
        radius_chord = 2.0 * np.sin(radius_m / (2.0 * EARTH_R_M))
        if debug:
            print(f"[REF] Built k-d tree; query radius (chord) = {radius_chord:.6g}")

    results = []
    for nc_path in nc_paths:
        if not os.path.isfile(nc_path):
            print(f"[WARN] File not found: {nc_path}")
            results.append({
                "file": nc_path,
                "n_pixels": 0,
                "mean_sigma0": None,
                "mean_ssha_unfiltered": None,
                "std_sigma0": None,
                "std_ssha_unfiltered": None,
            })
            continue

        nc = netCDF4.Dataset(nc_path, "r")

        # Discover coordinates (often latitude, longitude or lat, lon)
        lat_var = _get_nc_var(nc, ["latitude", "lat", "Latitude"])
        lon_var = _get_nc_var(nc, ["longitude", "lon", "Longitude"])
        if lat_var is None or lon_var is None:
            if debug:
                print(f"[SWOT] Variables in {nc_path}: {list(nc.variables.keys())}")
            nc.close()
            results.append({
                "file": nc_path,
                "n_pixels": 0,
                "mean_sigma0": None,
                "mean_ssha_unfiltered": None,
                "std_sigma0": None,
                "std_ssha_unfiltered": None,
            })
            continue

        # Lat/lon: use as-read (no scale_factor/add_offset applied)
        raw = lat_var[:]
        if np.ma.isMaskedArray(raw):
            raw = raw.filled(np.nan)
        lat_swot = np.asarray(raw, dtype=np.float32).ravel()
        raw = lon_var[:]
        if np.ma.isMaskedArray(raw):
            raw = raw.filled(np.nan)
        lon_swot = np.asarray(raw, dtype=np.float32).ravel()
        if lat_swot.size != lon_swot.size:
            nc.close()
            results.append({
                "file": nc_path,
                "n_pixels": 0,
                "mean_sigma0": None,
                "mean_ssha_unfiltered": None,
                "std_sigma0": None,
                "std_ssha_unfiltered": None,
            })
            continue

        # sigma0 and ssha_unfiltered (scale_factor 1e-4, _FillValue -2147483647 in SWOT)
        sigma0_var = _get_nc_var(nc, ["sigma0", "sig0", "sigma_0"])
        ssha_var = _get_nc_var(nc, ["ssha_unfiltered", "ssh_unfiltered", "ssha", "sla"])

        if sigma0_var is not None:
            sigma0 = _read_scaled(sigma0_var)
            if sigma0.size != lat_swot.size:
                sigma0 = None
        else:
            sigma0 = None
        if ssha_var is not None:
            ssha_unfiltered = _read_scaled(ssha_var)
            if ssha_unfiltered.size != lat_swot.size:
                ssha_unfiltered = None
        else:
            ssha_unfiltered = None

        nc.close()

        # Mask invalid coordinates
        ok = np.isfinite(lat_swot) & np.isfinite(lon_swot)
        if sigma0 is not None:
            ok = ok & np.isfinite(sigma0)
        if ssha_unfiltered is not None:
            ok = ok & np.isfinite(ssha_unfiltered)

        lat_swot = lat_swot[ok]
        lon_swot = lon_swot[ok]
        lon_swot = _lon_to_180(lon_swot)  # match ref convention for box overlap
        if sigma0 is not None:
            sigma0 = sigma0[ok]
        if ssha_unfiltered is not None:
            ssha_unfiltered = ssha_unfiltered[ok]

        n_swot = len(lat_swot)
        if debug and n_swot > 0:
            print(f"[SWOT] {os.path.basename(nc_path)} extent: lat [{np.min(lat_swot):.2f}, {np.max(lat_swot):.2f}]  lon [{np.min(lon_swot):.2f}, {np.max(lon_swot):.2f}]  n_valid={n_swot}")
        if n_swot == 0:
            results.append({
                "file": nc_path,
                "n_pixels": 0,
                "mean_sigma0": None,
                "mean_ssha_unfiltered": None,
                "std_sigma0": None,
                "std_ssha_unfiltered": None,
            })
            continue

        # Bounding-box pre-filter: only pixels that can be within radius_m of any ref point.
        # Margin ~ radius in degrees (1 deg lat ~111 km); use 2x for safety and lon at mid-lat.
        margin_deg = max(0.5, (radius_m / 111320.0) * 2.0)
        lat_lo = np.min(lat_ref) - margin_deg
        lat_hi = np.max(lat_ref) + margin_deg
        lon_lo = np.min(lon_ref) - margin_deg
        lon_hi = np.max(lon_ref) + margin_deg
        in_box = (
            (lat_swot >= lat_lo) & (lat_swot <= lat_hi) &
            (lon_swot >= lon_lo) & (lon_swot <= lon_hi)
        )
        n_candidates = int(np.sum(in_box))
        if debug:
            print(f"[BOX] ref box: lat [{lat_lo:.2f}, {lat_hi:.2f}]  lon [{lon_lo:.2f}, {lon_hi:.2f}]  -> {n_candidates} SWOT pixels in box")
        if n_candidates == 0:
            if debug:
                print(f"[WARN] No SWOT pixels inside ref box for {os.path.basename(nc_path)} (ref and SWOT extents do not overlap)")
            results.append({
                "file": os.path.basename(nc_path),
                "n_pixels": 0,
                "mean_sigma0": None,
                "mean_ssha_unfiltered": None,
                "std_sigma0": None,
                "std_ssha_unfiltered": None,
            })
            continue

        lat_cand = lat_swot[in_box]
        lon_cand = lon_swot[in_box]
        sigma0_cand = sigma0[in_box] if sigma0 is not None else None
        ssha_cand = ssha_unfiltered[in_box] if ssha_unfiltered is not None else None

        # Fast path: k-d tree query (O(n_candidates * log(n_ref)) vs O(n_candidates * n_ref)).
        if ref_tree is not None:
            within = np.zeros(n_candidates, dtype=bool)
            chunk_size = 100000  # query in chunks to limit memory from list-of-lists return
            for start in range(0, n_candidates, chunk_size):
                end = min(start + chunk_size, n_candidates)
                xyz = _latlon_deg_to_xyz(lat_cand[start:end], lon_cand[start:end])
                hits = ref_tree.query_ball_point(xyz, radius_chord)
                within[start:end] = np.array([len(h) > 0 for h in hits], dtype=bool)
        else:
            # Fallback: chunked haversine (no scipy or tree build failed).
            swot_chunk_size = 50000
            ref_chunk_size = 2000
            min_dist = np.full(n_candidates, np.inf, dtype=np.float32)
            for start in range(0, n_candidates, swot_chunk_size):
                end = min(start + swot_chunk_size, n_candidates)
                lon_c = lon_cand[start:end]
                lat_c = lat_cand[start:end]
                chunk_min = np.full(end - start, np.inf, dtype=np.float32)
                for j in range(0, n_ref, ref_chunk_size):
                    j_end = min(j + ref_chunk_size, n_ref)
                    dist = haversine_distance_m(
                        lon_c[:, None], lat_c[:, None],
                        lon_ref[j:j_end], lat_ref[j:j_end],
                    )
                    if dist.ndim == 2:
                        chunk_min = np.minimum(chunk_min, np.nanmin(dist, axis=1))
                    else:
                        chunk_min = np.minimum(chunk_min, dist)
                min_dist[start:end] = chunk_min
            within = min_dist <= radius_m
        n_pixels = int(np.sum(within))

        mean_sigma0 = float(np.nanmean(sigma0_cand[within])) if sigma0_cand is not None and n_pixels else None
        mean_ssha = float(np.nanmean(ssha_cand[within])) if ssha_cand is not None and n_pixels else None
        std_sigma0 = float(np.nanstd(sigma0_cand[within])) if sigma0_cand is not None and n_pixels else None
        std_ssha = float(np.nanstd(ssha_cand[within])) if ssha_cand is not None and n_pixels else None

        results.append({
            "file": os.path.basename(nc_path),
            "n_pixels": n_pixels,
            "mean_sigma0": mean_sigma0,
            "mean_ssha_unfiltered": mean_ssha,
            "std_sigma0": std_sigma0,
            "std_ssha_unfiltered": std_ssha,
        })

        if debug:
            print(f"[SWOT] {os.path.basename(nc_path)}: {n_swot} valid -> {n_candidates} in box -> {n_pixels} within {radius_m} m")

    return results


def write_trajectory_sigma0_ssha_nc(
    out_path,
    lat_start, lon_start, lat_final, lon_final,
    sigma0_start, ssha_start, sigma0_end, ssha_end,
    sigma0_std_start, ssha_std_start, sigma0_std_end, ssha_std_end,
    file_start, file_end, radius_m,
):
    """Write one NetCDF with per-trajectory sigma0 and SSHA (mean and std) for start and end dates."""
    n = len(lat_start)
    nc = netCDF4.Dataset(out_path, "w", format="NETCDF4")
    nc.createDimension("n_trajectories", n)
    for name, data, attrs in [
        ("lat_start", lat_start, {"long_name": "latitude of trajectory start (backtrack)", "units": "degrees_north"}),
        ("lon_start", lon_start, {"long_name": "longitude of trajectory start (backtrack)", "units": "degrees_east"}),
        ("lat_final", lat_final, {"long_name": "latitude of trajectory end (backtrack)", "units": "degrees_north"}),
        ("lon_final", lon_final, {"long_name": "longitude of trajectory end (backtrack)", "units": "degrees_east"}),
        ("sigma0_start", sigma0_start, {"long_name": "mean sigma0 near start (SWOT start-date file)", "units": "1"}),
        ("ssha_start", ssha_start, {"long_name": "mean ssha_unfiltered near start (m)", "units": "m"}),
        ("sigma0_std_start", sigma0_std_start, {"long_name": "std sigma0 near start", "units": "1"}),
        ("ssha_std_start", ssha_std_start, {"long_name": "std ssha_unfiltered near start (m)", "units": "m"}),
        ("sigma0_end", sigma0_end, {"long_name": "mean sigma0 near end (SWOT end-date file)", "units": "1"}),
        ("ssha_end", ssha_end, {"long_name": "mean ssha_unfiltered near end (m)", "units": "m"}),
        ("sigma0_std_end", sigma0_std_end, {"long_name": "std sigma0 near end", "units": "1"}),
        ("ssha_std_end", ssha_std_end, {"long_name": "std ssha_unfiltered near end (m)", "units": "m"}),
    ]:
        v = nc.createVariable(name, "f8", ("n_trajectories",), fill_value=np.nan)
        v[:] = np.asarray(data, dtype=np.float64)
        for k, val in attrs.items():
            v.setncattr(k, val)
    nc.setncattr("source_start_file", os.path.basename(file_start) if file_start else "")
    nc.setncattr("source_end_file", os.path.basename(file_end) if file_end else "")
    nc.setncattr("radius_m", radius_m)
    nc.setncattr("conventions", "backtrack: start = later SWOT pass (005_455), end = earlier SWOT pass (005_452)")
    nc.close()


def main():
    ap = argparse.ArgumentParser(
        description="Extract sigma0 and ssha_unfiltered per trajectory from SWOT files (backtrack: start=005_455, end=005_452) and optionally export to NetCDF."
    )
    ap.add_argument("npz", help="trajectories_backward.npz")
    ap.add_argument("nc_files", nargs="+", help="SWOT L3 LR SSH Unsmoothed NetCDF files (005_452 and 005_455)")
    ap.add_argument("--radius_m", type=float, default=2000.0, help="Radius in meters (default 2000)")
    ap.add_argument("--output_nc", type=str, default=None, help="Output NetCDF path for sigma0/SSHA per trajectory")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    lat_start, lon_start, lat_final, lon_final, valid = load_tracking_positions(args.npz)
    n_valid = int(np.sum(valid))
    path_start, path_end = _identify_start_end_files(args.nc_files)

    if args.output_nc and path_start and path_end and n_valid > 0:
        ref_lat_start = lat_start[valid]
        ref_lon_start = _lon_to_180(lon_start[valid])
        ref_lat_end = lat_final[valid]
        ref_lon_end = _lon_to_180(lon_final[valid])
        if args.debug:
            print(f"[BACKTRACK] start (later pass) -> {os.path.basename(path_start)}  end (earlier pass) -> {os.path.basename(path_end)}")
        sigma0_start, ssha_start, sigma0_std_start, ssha_std_start = extract_sigma0_ssha_per_trajectory(
            path_start, ref_lat_start, ref_lon_start, radius_m=args.radius_m, debug=args.debug
        )
        sigma0_end, ssha_end, sigma0_std_end, ssha_std_end = extract_sigma0_ssha_per_trajectory(
            path_end, ref_lat_end, ref_lon_end, radius_m=args.radius_m, debug=args.debug
        )
        write_trajectory_sigma0_ssha_nc(
            args.output_nc,
            ref_lat_start, ref_lon_start, ref_lat_end, ref_lon_end,
            sigma0_start, ssha_start, sigma0_end, ssha_end,
            sigma0_std_start, ssha_std_start, sigma0_std_end, ssha_std_end,
            path_start, path_end, args.radius_m,
        )
        print(f"Wrote {args.output_nc}  (n_trajectories={n_valid})")
    elif args.output_nc and (not path_start or not path_end):
        print("[WARN] Could not identify start (005_455) and end (005_452) files; need exactly two SWOT files with different cycle_pass. Skipping --output_nc.")

    results = compute_means_near_trajectories(
        args.npz,
        args.nc_files,
        radius_m=args.radius_m,
        debug=args.debug,
    )

    print("\nResults (radius = {} m):\n".format(args.radius_m))
    for r in results:
        print("File: {}".format(r["file"]))
        print("  Pixels within radius: {}".format(r["n_pixels"]))
        if r["mean_sigma0"] is not None:
            print("  sigma0:           mean = {:.6g}  std = {:.6g}".format(r["mean_sigma0"], r["std_sigma0"] or np.nan))
        else:
            print("  sigma0:           (not found or no valid pixels)")
        if r["mean_ssha_unfiltered"] is not None:
            print("  ssha_unfiltered:  mean = {:.6g}  std = {:.6g}".format(r["mean_ssha_unfiltered"], r["std_ssha_unfiltered"] or np.nan))
        else:
            print("  ssha_unfiltered:  (not found or no valid pixels)")
        print()


if __name__ == "__main__":
    main()
