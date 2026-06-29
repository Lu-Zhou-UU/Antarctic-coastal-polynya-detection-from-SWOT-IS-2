#!/usr/bin/env python3
"""
drift_aware_mass_budget.py

Draft code for a drift-aware polynya ice-mass budget diagnostic:
  M(t) = rho_i * hbar(t) * A(t)
  dM/dt from finite differences
  F_out from boundary normal ice transport (coarse)
  residual = dM/dt + F_out   (if F_out is defined positive outward)

This is a consistency check, not an absolute production retrieval.
"""

import os
import csv
import argparse
import glob
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

# Reuse your existing utilities
# from plot_esacci_sit_map.py
from plot_esacci_sit_map import (
    load_mat_any,
    polynya_output_amsr2,
    normalize_longitude_0_360,
    load_ice_motion,
)

# from plot_SWOT_PM_RACMO_SIT.py
from plot_SWOT_PM_RACMO_SIT import _parse_dt_list


# -----------------------------
# Helpers
# -----------------------------
R_EARTH = 6371000.0  # m

# In-memory caching to avoid re-loading AMSR2 .mat repeatedly.
# The Eulerian/tracked_fusion path calls `load_polynya_mask_for_date` many times.
_AMSR_BASE_CACHE = {}  # key: amsr_mat -> {'sic':..., 'lat':..., 'lon':...}
_AMSR_MASK_CACHE = {}  # key: (amsr_mat, day_idx, threshold, type) -> mask(bool 2D)

# In-memory caching for ice motion to avoid repeatedly reading/interpolating
# the drift field inside the advection step loop.
_ICE_MOTION_CACHE = {}  # key: (ice_motion_nc, date_iso, subsample) -> (lat, lon, u, v, valid)
_MASK_TREE_CACHE = {}  # key: (id(lon2d), id(lat2d)) -> cKDTree over mask grid points


def _load_ice_motion_cached(ice_motion_nc, dt, subsample=5):
    date_iso = dt.date().isoformat()
    key = (ice_motion_nc, date_iso, float(subsample))
    cached = _ICE_MOTION_CACHE.get(key)
    if cached is not None:
        return cached
    lat_i, lon_i, u_i, v_i, valid_i = load_ice_motion(
        ice_motion_nc, dt, subsample=subsample, debug=False
    )
    _ICE_MOTION_CACHE[key] = (lat_i, lon_i, u_i, v_i, valid_i)
    return lat_i, lon_i, u_i, v_i, valid_i


def to_date(d):
    if isinstance(d, datetime):
        return d.date()
    return datetime.fromisoformat(str(d).replace("Z", "")).date()


def day_of_year_idx(dt):
    # 0-based for non-leap style AMSR matrices
    return min(dt.timetuple().tm_yday, 365) - 1


def grid_cell_area_m2(lat2d, lon2d):
    """
    Approximate cell area from lat/lon 2D grid using local dlat/dlon.
    Works reasonably for AMSR regular-ish grids.
    """
    lat = np.deg2rad(lat2d)
    lon = np.deg2rad(lon2d)

    # central differences
    dlat = np.zeros_like(lat)
    dlon = np.zeros_like(lon)

    dlat[1:-1, :] = 0.5 * (lat[2:, :] - lat[:-2, :])
    dlat[0, :] = lat[1, :] - lat[0, :]
    dlat[-1, :] = lat[-1, :] - lat[-2, :]

    dlon[:, 1:-1] = 0.5 * (lon[:, 2:] - lon[:, :-2])
    dlon[:, 0] = lon[:, 1] - lon[:, 0]
    dlon[:, -1] = lon[:, -1] - lon[:, -2]

    dlon = np.abs(dlon)
    dlat = np.abs(dlat)

    area = (R_EARTH ** 2) * np.cos(lat) * dlat * dlon
    area[~np.isfinite(area)] = np.nan
    return area


def nearest_time_index(time_list, target_dt):
    td = np.array([abs((t - target_dt).total_seconds()) for t in time_list], dtype=float)
    return int(np.nanargmin(td))


def wrap_lon180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def _apply_region_bbox_mask(mask, lon2d, lat2d, region_lon_min=None, region_lon_max=None, region_lat_min=None, region_lat_max=None):
    """Apply optional lon/lat bbox clip to a boolean mask."""
    m = np.asarray(mask, dtype=bool).copy()
    lon = np.asarray(lon2d, dtype=float)
    lat = np.asarray(lat2d, dtype=float)

    if region_lon_min is not None and region_lon_max is not None:
        lo = float(region_lon_min) % 360.0
        hi = float(region_lon_max) % 360.0
        lon0360 = np.mod(lon, 360.0)
        if lo <= hi:
            m &= (lon0360 >= lo) & (lon0360 <= hi)
        else:
            # Wrap-around interval, e.g. [330, 40]
            m &= ((lon0360 >= lo) | (lon0360 <= hi))
    if region_lat_min is not None:
        m &= lat >= float(region_lat_min)
    if region_lat_max is not None:
        m &= lat <= float(region_lat_max)
    return m


def mask_boundary_4n(mask):
    """
    Boundary pixels by 4-neighborhood: True where mask pixel has at least one outside neighbor.
    """
    m = mask.astype(bool)
    up = np.zeros_like(m, dtype=bool); up[1:, :] = m[:-1, :]
    dn = np.zeros_like(m, dtype=bool); dn[:-1, :] = m[1:, :]
    lf = np.zeros_like(m, dtype=bool); lf[:, 1:] = m[:, :-1]
    rg = np.zeros_like(m, dtype=bool); rg[:, :-1] = m[:, 1:]
    interior4 = m & up & dn & lf & rg
    return m & (~interior4)


def _neighbors4(mask):
    """Return pixels having at least one 4-neighbor True in `mask`."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m, dtype=bool)
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _land_mask_for_date(amsr_mat, dt):
    """
    Build a land/invalid-SIC proxy mask for this day from AMSR SIC.
    Mirrors the validity logic used in polynya_output_amsr2.
    """
    day_idx = day_of_year_idx(dt)
    base = _AMSR_BASE_CACHE.get(amsr_mat)
    if base is None:
        # Populate cache through the same loader path used elsewhere.
        _ = load_polynya_mask_for_date(amsr_mat, dt, polynya_threshold=0.7, polynya_type="coastal")
        base = _AMSR_BASE_CACHE[amsr_mat]
    sic = base["sic"]
    sic_daily = sic[day_idx] if day_idx < sic.shape[0] else sic[-1]
    sic_frac = sic_daily / 100.0 if np.nanmax(sic_daily) > 1.0 else sic_daily
    valid = np.isfinite(sic_frac) & (sic_frac >= 0.0) & (sic_frac <= 1.2)
    return ~valid


def mean_spacing_m(lat2d, lon2d, mask):
    """
    Rough boundary segment length scale from local grid spacing.
    """
    lat = np.deg2rad(lat2d)
    lon = np.deg2rad(lon2d)

    # x-spacing
    dlon = np.abs(np.roll(lon, -1, axis=1) - lon)
    dx = R_EARTH * np.cos(lat) * dlon
    # y-spacing
    dlat = np.abs(np.roll(lat, -1, axis=0) - lat)
    dy = R_EARTH * dlat

    # Use boundary region stats
    mm = mask & np.isfinite(dx) & np.isfinite(dy)
    if np.any(mm):
        ds = np.nanmean(0.5 * (dx[mm] + dy[mm]))
    else:
        ds = np.nan
    return ds


def interpolate_h_to_grid(points_lon, points_lat, points_h, lon2d, lat2d):
    """
    Interpolate sparse SIT points to AMSR grid.
    Uses scipy if available; otherwise nearest-neighbor fallback.
    """
    try:
        from scipy.interpolate import LinearNDInterpolator
        F = LinearNDInterpolator(
            np.column_stack([points_lon, points_lat]),
            points_h,
            fill_value=np.nan
        )
        return F(lon2d, lat2d)
    except Exception:
        # nearest-neighbor fallback
        h_grid = np.full_like(lon2d, np.nan, dtype=float)
        p = np.column_stack([points_lon, points_lat])
        q = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        # brute force (slow but robust for draft)
        for i in range(q.shape[0]):
            d2 = np.sum((p - q[i]) ** 2, axis=1)
            j = np.nanargmin(d2)
            h_grid.ravel()[i] = points_h[j]
        return h_grid


def interpolate_h_to_points(points_lon, points_lat, points_h, q_lon, q_lat, nearest_fill_max_dist_km=None):
    """
    Interpolate sparse SIT points to query lon/lat point arrays (1D or any shape).
    Returns an array with the same shape as q_lon/q_lat.

    If `nearest_fill_max_dist_km` is provided and the primary interpolation returns NaNs
    for some query points (e.g. outside the convex hull in LinearNDInterpolator),
    those NaNs are filled using nearest-neighbor within the provided distance limit.
    """
    points_lon = np.asarray(points_lon, dtype=float).ravel()
    points_lat = np.asarray(points_lat, dtype=float).ravel()
    points_h = np.asarray(points_h, dtype=float).ravel()
    q_lon = np.asarray(q_lon, dtype=float)
    q_lat = np.asarray(q_lat, dtype=float)
    q_lon_flat = q_lon.ravel()
    q_lat_flat = q_lat.ravel()
    m = np.isfinite(points_lon) & np.isfinite(points_lat) & np.isfinite(points_h)
    if np.count_nonzero(m) < 3:
        return np.full_like(q_lon, np.nan, dtype=float)

    try:
        from scipy.interpolate import LinearNDInterpolator
        F = LinearNDInterpolator(
            np.column_stack([points_lon[m], points_lat[m]]),
            points_h[m],
            fill_value=np.nan,
        )
        out = F(q_lon_flat, q_lat_flat)
        out_arr = np.asarray(out, dtype=float).reshape(q_lon.shape)
        if nearest_fill_max_dist_km is not None:
            mnan = ~np.isfinite(out_arr)
            if np.any(mnan):
                out_arr[mnan] = _nearest_sample_points(
                    points_lon[m], points_lat[m], points_h[m],
                    q_lon[mnan], q_lat[mnan],
                    max_dist_km=nearest_fill_max_dist_km,
                )
        return out_arr
    except Exception:
        # Faster than brute force loops: nearest-neighbor via cKDTree (inside _nearest_sample_points).
        return _nearest_sample_points(
            points_lon[m], points_lat[m], points_h[m],
            q_lon, q_lat,
            max_dist_km=nearest_fill_max_dist_km,
        )


def _nearest_sample_points(src_lon, src_lat, src_val, q_lon, q_lat, max_dist_km=None):
    """
    Sample src_val at query points using nearest-neighbor in lon/lat.
    Uses cKDTree when n_query * n_src is large (e.g. full AMSR grid vs segments).
    
    If max_dist_km is provided, nearest samples farther than this (approximate km
    distance in lon/lat) are set to NaN.
    """
    src_lon = np.asarray(src_lon, dtype=float).ravel()
    src_lat = np.asarray(src_lat, dtype=float).ravel()
    src_val = np.asarray(src_val, dtype=float).ravel()
    q_lon = np.asarray(q_lon, dtype=float).ravel()
    q_lat = np.asarray(q_lat, dtype=float).ravel()

    out = np.full(q_lon.shape, np.nan, dtype=float)
    msrc = np.isfinite(src_lon) & np.isfinite(src_lat) & np.isfinite(src_val)
    mq = np.isfinite(q_lon) & np.isfinite(q_lat)
    if not np.any(msrc) or not np.any(mq):
        return out

    p = np.column_stack([src_lon[msrc], src_lat[msrc]])
    v = src_val[msrc]
    q = np.column_stack([q_lon[mq], q_lat[mq]])
    q_flat_idx = np.where(mq)[0]
    nq = q.shape[0]
    nsrc = p.shape[0]
    # Brute loop is O(nq * nsrc) vectorized inner; use KD-tree for large jobs.
    if nq * nsrc > 2_000_000:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(p)
            try:
                _, jj = tree.query(q, workers=-1)
            except TypeError:
                _, jj = tree.query(q)
            jj = np.asarray(jj, dtype=int)
            nearest_vals = v[jj]
            if max_dist_km is not None:
                nearest_lon = p[jj, 0]
                nearest_lat = p[jj, 1]
                # Approximate distance: local km/deg scaling.
                dx = wrap_lon180(nearest_lon - q[:, 0]) * np.cos(np.deg2rad(q[:, 1])) * 111.0
                dy = (nearest_lat - q[:, 1]) * 111.0
                dist_km = np.sqrt(dx ** 2 + dy ** 2)
                nearest_vals = np.where(dist_km <= float(max_dist_km), nearest_vals, np.nan)
            out[q_flat_idx] = nearest_vals
            return out
        except Exception:
            pass
    for i in range(q.shape[0]):
        d2 = np.sum((p - q[i]) ** 2, axis=1)
        j = int(np.nanargmin(d2))
        if max_dist_km is not None:
            nearest_lon = p[j, 0]
            nearest_lat = p[j, 1]
            dx = wrap_lon180(nearest_lon - q[i, 0]) * np.cos(np.deg2rad(q[i, 1])) * 111.0
            dy = (nearest_lat - q[i, 1]) * 111.0
            dist_km = float(np.sqrt(dx ** 2 + dy ** 2))
            if dist_km > float(max_dist_km):
                continue
        out[q_flat_idx[i]] = v[j]
    return out


def _velocity_at_points(lon_grid, lat_grid, u_grid, v_grid, valid_mask, q_lon, q_lat):
    """
    Bilinear u/v sampling at query lon/lat points (linear on unstructured lon/lat),
    with nearest-neighbor fallback.
    """
    p = np.column_stack([lon_grid[valid_mask].ravel(), lat_grid[valid_mask].ravel()])
    uu = u_grid[valid_mask].ravel()
    vv = v_grid[valid_mask].ravel()
    q = np.column_stack([q_lon.ravel(), q_lat.ravel()])
    uq = np.full(q.shape[0], np.nan, dtype=float)
    vq = np.full(q.shape[0], np.nan, dtype=float)
    if p.size == 0:
        return uq.reshape(q_lon.shape), vq.reshape(q_lon.shape)

    # bilinear/linear interpolation in lon-lat space
    try:
        from scipy.interpolate import griddata
        uq = griddata(p, uu, q, method="linear", fill_value=np.nan)
        vq = griddata(p, vv, q, method="linear", fill_value=np.nan)
    except Exception:
        pass

    # nearest fallback where linear is NaN
    mfill = ~(np.isfinite(uq) & np.isfinite(vq))
    if np.any(mfill):
        qm = q[mfill]
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(p)
            try:
                _, jj = tree.query(qm, workers=-1)
            except TypeError:
                _, jj = tree.query(qm)
            jj = np.asarray(jj, dtype=int)
            fill_idx = np.where(mfill)[0]
            uq[fill_idx] = uu[jj]
            vq[fill_idx] = vv[jj]
        except Exception:
            for i in range(qm.shape[0]):
                if not np.isfinite(qm[i, 0]) or not np.isfinite(qm[i, 1]):
                    continue
                d2 = np.sum((p - qm[i]) ** 2, axis=1)
                j = int(np.nanargmin(d2))
                uq[np.where(mfill)[0][i]] = uu[j]
                vq[np.where(mfill)[0][i]] = vv[j]
    return uq.reshape(q_lon.shape), vq.reshape(q_lon.shape)


def _displace_lonlat(lon_deg, lat_deg, u_east_ms, v_north_ms, dt_seconds):
    """
    Small-angle spherical displacement.
    """
    R = 6371000.0
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    u = np.asarray(u_east_ms, dtype=float)
    v = np.asarray(v_north_ms, dtype=float)

    lat_rad = np.deg2rad(lat)
    dlat = (v * dt_seconds) / R
    denom = R * np.cos(lat_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlon = (u * dt_seconds) / denom
    lat2 = lat + np.rad2deg(dlat)
    lon2 = normalize_longitude_0_360(lon + np.rad2deg(dlon))
    return lon2, lat2


def _load_segment_snapshot_npz(npz_path):
    """
    Expected keys:
      - times: 'seg_times_str' (preferred) or 'times_str' or 'sit_times_str'
      - per-scene lon: 'seg_lon' or 'lon'
      - per-scene lat: 'seg_lat' or 'lat'
      - per-scene h:   'seg_h' or 'h' or 'sit'
    Each variable should be shape (ntime, nseg) or broadcastable to it.
    """
    if os.path.isdir(npz_path):
        return _load_segment_snapshots_from_dir(npz_path)

    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())

    tkey = None
    for k in ("seg_times_str", "times_str", "sit_times_str"):
        if k in d:
            tkey = k
            break
    if tkey is None:
        raise KeyError("segment NPZ must contain one of: seg_times_str, times_str, sit_times_str")
    times = _parse_dt_list(d[tkey])

    def _get_any(keys):
        for k in keys:
            if k in d:
                return np.array(d[k], dtype=float)
        return None

    lon = _get_any(("seg_lon", "lon"))
    lat = _get_any(("seg_lat", "lat"))
    h = _get_any(("seg_h", "h", "sit"))
    if lon is None or lat is None or h is None:
        summary_keys = {"sit_times_str", "sit_mean", "sit_std", "sit_nseg", "amsr_area"}
        looks_like_summary = len(summary_keys.intersection(set(keys))) >= 3
        if looks_like_summary:
            raise KeyError(
                "Input NPZ looks like scene-summary plot elements (sit_mean/sit_std/sit_nseg), "
                "not per-segment geometry. Segment mode requires per-time per-segment lon/lat/h arrays "
                "under keys seg_lon/seg_lat/seg_h (or lon/lat/h). "
                f"Available keys: {keys}"
            )
        raise KeyError(
            "segment NPZ must contain lon/lat/h fields "
            "(seg_lon/seg_lat/seg_h or lon/lat/h). "
            f"Available keys: {keys}"
        )

    if lon.ndim == 1:
        lon = np.tile(lon.reshape(1, -1), (len(times), 1))
    if lat.ndim == 1:
        lat = np.tile(lat.reshape(1, -1), (len(times), 1))
    if h.ndim == 1:
        h = np.tile(h.reshape(1, -1), (len(times), 1))

    n = len(times)
    if lon.shape[0] != n or lat.shape[0] != n or h.shape[0] != n:
        raise ValueError("segment NPZ arrays must have first dimension equal to number of times.")

    return times, normalize_longitude_0_360(lon), lat, h


def _parse_datetime_from_text(s):
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


def _segment_means_from_scene_npz(path):
    d = np.load(path, allow_pickle=True)
    if not all(k in d for k in ("lat", "lon", "sit")):
        return None
    lat = np.asarray(d["lat"], dtype=float).ravel()
    lon = normalize_longitude_0_360(np.asarray(d["lon"], dtype=float).ravel())
    sit = np.asarray(d["sit"], dtype=float).ravel()

    if "in_scope" in d:
        mscope = np.asarray(d["in_scope"], dtype=bool).ravel()
    elif "in_polynya" in d:
        mscope = np.asarray(d["in_polynya"], dtype=bool).ravel()
    else:
        mscope = np.ones_like(sit, dtype=bool)

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(sit) & mscope
    if not np.any(valid):
        return None
    lat = lat[valid]
    lon = lon[valid]
    sit = sit[valid]

    if "seg_joint" in d:
        seg = np.asarray(d["seg_joint"]).ravel()
        seg = seg[valid]
        good_seg = np.isfinite(seg) & (seg >= 0)
        if np.any(good_seg):
            seg = seg[good_seg].astype(np.int64)
            lat = lat[good_seg]
            lon = lon[good_seg]
            sit = sit[good_seg]
            u = np.unique(seg)
            out_lat = np.full(u.size, np.nan, dtype=float)
            out_lon = np.full(u.size, np.nan, dtype=float)
            out_h = np.full(u.size, np.nan, dtype=float)
            for i, sid in enumerate(u):
                m = seg == sid
                out_lat[i] = np.nanmean(lat[m])
                out_lon[i] = np.nanmean(lon[m])
                out_h[i] = np.nanmean(sit[m])
            return out_lon, out_lat, out_h

    return lon, lat, sit


def _load_segment_snapshots_from_dir(root_dir, file_glob="*_is2_polynya_sit_outputs.npz"):
    files = sorted(glob.glob(os.path.join(root_dir, file_glob)))
    if len(files) == 0:
        raise RuntimeError(f"No segment scene NPZ files found in {root_dir} with glob {file_glob}")

    rec = []
    for f in files:
        t = _parse_datetime_from_text(os.path.basename(f))
        if t is None:
            continue
        seg = _segment_means_from_scene_npz(f)
        if seg is None:
            continue
        lon, lat, h = seg
        if lon.size == 0:
            continue
        rec.append((t, lon, lat, h))
    if len(rec) == 0:
        raise RuntimeError("No valid segment scenes were parsed from directory NPZ files.")

    rec = sorted(rec, key=lambda x: x[0])
    times = [r[0] for r in rec]
    maxn = max(r[1].size for r in rec)
    lon2 = np.full((len(rec), maxn), np.nan, dtype=float)
    lat2 = np.full((len(rec), maxn), np.nan, dtype=float)
    h2 = np.full((len(rec), maxn), np.nan, dtype=float)
    for i, (_, lo, la, hh) in enumerate(rec):
        n = lo.size
        lon2[i, :n] = lo
        lat2[i, :n] = la
        h2[i, :n] = hh
    return times, lon2, lat2, h2


def _advect_points_between_times(
    lon0,
    lat0,
    t_start,
    t_end,
    ice_motion_nc,
    step_hours=6.0,
    amsr_mat=None,
    fill_nan_velocity_zero_on_ocean=True,
    ice_motion_subsample=5,
):
    """
    Advect points from t_start to t_end using piecewise-constant daily velocity.
    Backward advection is handled automatically when t_end < t_start.
    If fill_nan_velocity_zero_on_ocean=True and amsr_mat is provided, NaN velocities
    are replaced by zero where AMSR SIC is valid (non-land surface proxy).
    """
    lon = np.array(lon0, dtype=float, copy=True)
    lat = np.array(lat0, dtype=float, copy=True)
    if t_end == t_start:
        return lon, lat

    sign = 1.0 if t_end > t_start else -1.0
    dt_step = abs(float(step_hours)) * 3600.0
    if dt_step <= 0:
        raise ValueError("step_hours must be > 0")

    nsteps = int(np.ceil(abs((t_end - t_start).total_seconds()) / dt_step))
    t = t_start
    for _ in range(nsteps):
        rem = abs((t_end - t).total_seconds())
        if rem <= 0:
            break
        this_dt = min(dt_step, rem)
        dt_signed = sign * this_dt
        t_vel = t if sign > 0 else (t - timedelta(seconds=this_dt))
        lat_i, lon_i, u_i, v_i, valid_i = _load_ice_motion_cached(
            ice_motion_nc, t_vel, subsample=ice_motion_subsample
        )
        uq, vq = _velocity_at_points(
            lon_i, lat_i, u_i, v_i, valid_i,
            q_lon=lon, q_lat=lat
        )

        if fill_nan_velocity_zero_on_ocean and (amsr_mat is not None):
            lat_m, lon_m, mask_m = load_polynya_mask_for_date(
                amsr_mat, t_vel, polynya_threshold=1.01, polynya_type="both"
            )
            # With threshold >1, mask_m acts as "valid SIC surface" (non-land proxy).
            # Replace NaN drift by 0 only on this valid surface.
            on_surface = _sample_bool_mask_nearest(lon_m, lat_m, mask_m, lon, lat)
            bad_u = ~np.isfinite(uq)
            bad_v = ~np.isfinite(vq)
            uq = np.where(bad_u & on_surface, 0.0, uq)
            vq = np.where(bad_v & on_surface, 0.0, vq)

        lon, lat = _displace_lonlat(lon, lat, uq, vq, dt_signed)
        t = t + timedelta(seconds=dt_signed)
    return lon, lat


def _sample_bool_mask_nearest(mask_lon2d, mask_lat2d, mask_bool2d, q_lon, q_lat):
    """
    Nearest-neighbor sampling of a boolean mask at query points.
    Returns an array with the same shape as q_lon (must match q_lat).
    """
    q_lon_arr = np.asarray(q_lon, dtype=float)
    q_lat_arr = np.asarray(q_lat, dtype=float)
    if q_lon_arr.shape != q_lat_arr.shape:
        raise ValueError("q_lon and q_lat must have the same shape")
    out_shape = q_lon_arr.shape
    mask_lon2d = np.asarray(mask_lon2d, dtype=float)
    mask_lat2d = np.asarray(mask_lat2d, dtype=float)
    mask_bool2d = np.asarray(mask_bool2d, dtype=bool)
    # Same grid as the mask: no search (Eulerian advection step 0 uses this often).
    if (q_lon_arr.shape == mask_bool2d.shape and
            np.allclose(q_lon_arr, mask_lon2d, rtol=1e-9, atol=1e-9, equal_nan=True) and
            np.allclose(q_lat_arr, mask_lat2d, rtol=1e-9, atol=1e-9, equal_nan=True)):
        return mask_bool2d.copy()

    src_lon = mask_lon2d.ravel()
    src_lat = mask_lat2d.ravel()
    src_val = mask_bool2d.ravel()
    q_lon = q_lon_arr.ravel()
    q_lat = q_lat_arr.ravel()
    out = np.zeros(q_lon.shape, dtype=bool)

    msrc = np.isfinite(src_lon) & np.isfinite(src_lat)
    mq = np.isfinite(q_lon) & np.isfinite(q_lat)
    if not np.any(msrc) or not np.any(mq):
        return out.reshape(out_shape)

    p = np.column_stack([src_lon[msrc], src_lat[msrc]])
    v = src_val[msrc]
    q = np.column_stack([q_lon[mq], q_lat[mq]])
    q_idx = np.where(mq)[0]
    nq = q.shape[0]
    nsrc = p.shape[0]
    # Python loop is O(nq * nsrc) per query; use KD-tree for full AMSR grids.
    if nq * nsrc > 2_000_000:
        try:
            from scipy.spatial import cKDTree
            key = (id(mask_lon2d), id(mask_lat2d))
            tree = _MASK_TREE_CACHE.get(key)
            if tree is None:
                tree = cKDTree(p)
                _MASK_TREE_CACHE[key] = tree
            try:
                _, jj = tree.query(q, workers=-1)
            except TypeError:
                _, jj = tree.query(q)
            out[q_idx] = v[np.asarray(jj, dtype=int)]
            return out.reshape(out_shape)
        except Exception:
            pass
    for i in range(q.shape[0]):
        d2 = np.sum((p - q[i]) ** 2, axis=1)
        j = int(np.nanargmin(d2))
        out[q_idx[i]] = bool(v[j])
    return out.reshape(out_shape)


def _stack_reduce_stats(stk):
    """
    Robust reducers for a stack of thickness reconstructions with shape (nlayer, npix).
    Returns:
      h_mean, h_median, h_std, h_q25, h_q75, nsrc
    Spread terms are proxy reconstruction/sampling spread, not a full error budget.
    """
    stk = np.asarray(stk, dtype=float)
    if stk.ndim != 2:
        raise ValueError("stk must be 2D with shape (nlayer, npix)")
    nsrc = np.sum(np.isfinite(stk), axis=0).astype(float)
    any_valid = nsrc > 0

    h_mean = np.full(stk.shape[1], np.nan, dtype=float)
    h_median = np.full(stk.shape[1], np.nan, dtype=float)
    h_std = np.full(stk.shape[1], np.nan, dtype=float)
    h_q25 = np.full(stk.shape[1], np.nan, dtype=float)
    h_q75 = np.full(stk.shape[1], np.nan, dtype=float)
    if not np.any(any_valid):
        return h_mean, h_median, h_std, h_q25, h_q75, nsrc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        h_mean[any_valid] = np.nanmean(stk[:, any_valid], axis=0)
        h_median[any_valid] = np.nanmedian(stk[:, any_valid], axis=0)
        h_std[any_valid] = np.nanstd(stk[:, any_valid], axis=0)
        h_q25[any_valid] = np.nanpercentile(stk[:, any_valid], 25.0, axis=0)
        h_q75[any_valid] = np.nanpercentile(stk[:, any_valid], 75.0, axis=0)
    return h_mean, h_median, h_std, h_q25, h_q75, nsrc


def _area_weighted_mean(x, area):
    x = np.asarray(x, dtype=float)
    area = np.asarray(area, dtype=float)
    m = np.isfinite(x) & np.isfinite(area)
    if not np.any(m):
        return np.nan
    denom = np.nansum(area[m])
    if denom <= 0:
        return np.nan
    return float(np.nansum(x[m] * area[m]) / denom)


def _area_weighted_rms(x, area):
    x = np.asarray(x, dtype=float)
    area = np.asarray(area, dtype=float)
    m = np.isfinite(x) & np.isfinite(area)
    if not np.any(m):
        return np.nan
    denom = np.nansum(area[m])
    if denom <= 0:
        return np.nan
    return float(np.sqrt(np.nansum((x[m] ** 2) * area[m]) / denom))


def _integrated_mass_and_uncertainty(h, area, rho_i, h_std=None):
    h = np.asarray(h, dtype=float)
    area = np.asarray(area, dtype=float)
    m = np.isfinite(h) & np.isfinite(area)
    if not np.any(m):
        return np.nan, np.nan
    mass = float(rho_i * np.nansum(h[m] * area[m]))
    unc = np.nan
    if h_std is not None:
        h_std = np.asarray(h_std, dtype=float)
        mu = m & np.isfinite(h_std)
        if np.any(mu):
            unc = float(rho_i * np.sqrt(np.nansum((h_std[mu] * area[mu]) ** 2)))
    return mass, unc


def _boundary_flux_and_uncertainty(h_b, un_b, ds, rho_i, hstd_b=None):
    h_b = np.asarray(h_b, dtype=float)
    un_b = np.asarray(un_b, dtype=float)
    m = np.isfinite(h_b) & np.isfinite(un_b)
    if not np.any(m):
        return np.nan, np.nan
    flux = float(rho_i * np.nansum(h_b[m] * un_b[m] * ds))
    unc = np.nan
    if hstd_b is not None:
        hstd_b = np.asarray(hstd_b, dtype=float)
        mu = m & np.isfinite(hstd_b)
        if np.any(mu):
            unc = float(rho_i * np.sqrt(np.nansum((hstd_b[mu] * un_b[mu] * ds) ** 2)))
    return flux, unc


@dataclass
class SceneMass:
    t: datetime
    area_m2: float
    hbar_m: float
    mass_kg: float
    fout_kg_s: float  # outward positive


# -----------------------------
# Core workflow
# -----------------------------
def load_polynya_mask_for_date(amsr_mat, dt, polynya_threshold=0.7, polynya_type="coastal"):
    day_idx = day_of_year_idx(dt)
    key_mask = (amsr_mat, int(day_idx), float(polynya_threshold), str(polynya_type))
    cached = _AMSR_MASK_CACHE.get(key_mask)
    if cached is not None:
        base = _AMSR_BASE_CACHE[amsr_mat]
        return base["lat"], base["lon"], cached

    base = _AMSR_BASE_CACHE.get(amsr_mat)
    if base is None:
        amsr = load_mat_any(amsr_mat)
        sic = np.array(amsr["sic"], dtype=float)
        lat = np.array(amsr["lat"], dtype=float)
        lon = np.array(amsr["lon"], dtype=float)

        if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
            sic = np.moveaxis(sic, -1, 0)

        if lat.shape != sic.shape[1:] and lat.T.shape == sic.shape[1:]:
            lat = lat.T
            lon = lon.T

        if lat.ndim == 1 and lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        lon = normalize_longitude_0_360(lon)

        base = {"sic": sic, "lat": lat, "lon": lon}
        _AMSR_BASE_CACHE[amsr_mat] = base

    sic = base["sic"]
    lat = base["lat"]
    lon = base["lon"]

    sic_daily = sic[day_idx] if day_idx < sic.shape[0] else sic[-1]
    sic_frac = sic_daily / 100.0 if np.nanmax(sic_daily) > 1.0 else sic_daily
    open_pol, coastal_pol = polynya_output_amsr2(sic_frac, polynya_threshold)

    if polynya_type == "open":
        mask = np.isfinite(open_pol)
    elif polynya_type == "both":
        mask = np.isfinite(open_pol) | np.isfinite(coastal_pol)
    else:
        mask = np.isfinite(coastal_pol)

    _AMSR_MASK_CACHE[key_mask] = mask
    return lat, lon, mask


def load_sit_snapshot_points(npz_path):
    """
    Expects your SWOT-IS2 plot-elements npz structure:
      sit_times_str, sit_mean, sit_std ...
    For spatial mass budget we also need per-scene mapped points.
    If absent, adapt to your scene file(s). This function is a placeholder:
    """
    d = np.load(npz_path, allow_pickle=True)
    times = _parse_dt_list(d["sit_times_str"])
    # If your NPZ has only mean SIT, cannot do boundary flux robustly.
    # You can still do M* = rho * hbar * A using means.
    hbar = np.array(d["sit_mean"], dtype=float)
    return times, hbar


def scene_mass_from_mean_only(dt, hbar, amsr_mat, ice_motion_nc, rho_i=915.0,
                              polynya_threshold=0.7, polynya_type="coastal",
                              ice_motion_subsample=5):
    """
    Minimal version using only mean SIT + area:
      M = rho * hbar * A
    and a coarse F_out from boundary velocity + hbar.
    """
    lat, lon, mask = load_polynya_mask_for_date(
        amsr_mat, dt, polynya_threshold=polynya_threshold, polynya_type=polynya_type
    )
    area_cell = grid_cell_area_m2(lat, lon)
    A = np.nansum(area_cell[mask])

    # load drift (same date)
    lat_i, lon_i, u, v, valid = load_ice_motion(
        ice_motion_nc, dt, subsample=ice_motion_subsample, debug=False
    )

    # Regrid u,v to AMSR grid (simple nearest for draft)
    u_grid = np.full_like(lat, np.nan, dtype=float)
    v_grid = np.full_like(lat, np.nan, dtype=float)
    p = np.column_stack([lon_i[valid].ravel(), lat_i[valid].ravel()])
    uu = u[valid].ravel()
    vv = v[valid].ravel()
    q = np.column_stack([lon.ravel(), lat.ravel()])
    for k in range(q.shape[0]):
        d2 = np.sum((p - q[k]) ** 2, axis=1)
        j = np.nanargmin(d2)
        u_grid.ravel()[k] = uu[j]
        v_grid.ravel()[k] = vv[j]

    # boundary-normal flux approximation
    bnd = mask_boundary_4n(mask)
    ds = mean_spacing_m(lat, lon, bnd)
    if not np.isfinite(ds):
        ds = 6250.0  # rough AMSR 6.25 km fallback

    # approximate outward normal using mask gradient
    gy, gx = np.gradient(mask.astype(float))
    norm = np.sqrt(gx**2 + gy**2)
    nx = np.where(norm > 0, gx / norm, 0.0)
    ny = np.where(norm > 0, gy / norm, 0.0)

    # convert u/v (east,north) to local grid-normal approx:
    # this is coarse; acceptable for consistency diagnostic
    un = u_grid * nx + v_grid * ny
    un_b = np.where(bnd, un, np.nan)

    # outward positive flux:
    # F_out = rho * h * integral(un dl)
    F_out = rho_i * hbar * np.nansum(un_b) * ds

    M = rho_i * hbar * A
    return SceneMass(t=dt, area_m2=A, hbar_m=hbar, mass_kg=M, fout_kg_s=F_out)


def compute_mass_budget(times, hbar, amsr_mat, ice_motion_nc, rho_i=915.0,
                        polynya_threshold=0.7, polynya_type="coastal",
                        max_dt_days=10):
    scenes = []
    for t, h in zip(times, hbar):
        if not np.isfinite(h):
            continue
        scenes.append(scene_mass_from_mean_only(
            t, float(h), amsr_mat, ice_motion_nc, rho_i=rho_i,
            polynya_threshold=polynya_threshold, polynya_type=polynya_type
        ))

    scenes = sorted(scenes, key=lambda s: s.t)
    n = len(scenes)
    if n < 3:
        raise RuntimeError("Need at least 3 scenes for central dM/dt.")

    out_rows = []
    for i in range(1, n - 1):
        s0, s1, s2 = scenes[i - 1], scenes[i], scenes[i + 1]
        dt_sec = (s2.t - s0.t).total_seconds()
        if dt_sec <= 0:
            continue
        if abs((s2.t - s0.t).days) > max_dt_days:
            # skip if window too wide
            continue

        dMdt = (s2.mass_kg - s0.mass_kg) / dt_sec  # kg/s
        # with F_out positive outward, budget is:
        # dM/dt = -F_out + thermo + eps
        # -> residual thermo+eps = dM/dt + F_out
        residual = dMdt + s1.fout_kg_s

        out_rows.append({
            "time": s1.t.isoformat(sep=" "),
            "A_m2": s1.area_m2,
            "hbar_m": s1.hbar_m,
            "M_kg": s1.mass_kg,
            "dMdt_kg_s": dMdt,
            "Fout_kg_s": s1.fout_kg_s,
            "residual_kg_s": residual,
        })
    return out_rows


def _h_lagrangian_pooled_on_mask(
    t_center,
    lon_q,
    lat_q,
    times,
    lon_seg,
    lat_seg,
    h_seg,
    lag_days,
    ice_motion_nc,
    amsr_mat,
    step_hours,
    ice_motion_subsample,
    fill_nan_velocity_zero_on_ocean,
    nearest_fill_max_dist_km=None,
):
    """
    Thickness on AMSR mask points by pooling segment samples in a calendar lag window:
    for each scene k with |t_k - t_center| <= lag_days, advect segment (lon, lat) from
    t_k to t_center, then interpolate scattered (lon, lat, h) to lon_q/lat_q. Same spirit
    as plot_SWOT_PM_RACMO_SIT._pooled_sit_lagrangian_on_query_times, but as a 2D field
    on the polynya mask (not a single regional mean).
    """
    half = timedelta(days=float(lag_days))
    ntime = len(times)
    lo_acc = []
    la_acc = []
    h_acc = []
    for k in range(ntime):
        if abs((times[k] - t_center).total_seconds()) > half.total_seconds():
            continue
        mk = (
            np.isfinite(lon_seg[k, :])
            & np.isfinite(lat_seg[k, :])
            & np.isfinite(h_seg[k, :])
        )
        if not np.any(mk):
            continue
        lon0 = np.asarray(lon_seg[k, mk], dtype=float)
        lat0 = np.asarray(lat_seg[k, mk], dtype=float)
        hk = np.asarray(h_seg[k, mk], dtype=float)
        if abs((times[k] - t_center).total_seconds()) < 1.0:
            lon_a, lat_a = lon0, lat0
        else:
            lon_a, lat_a = _advect_points_between_times(
                lon0,
                lat0,
                times[k],
                t_center,
                ice_motion_nc=ice_motion_nc,
                step_hours=step_hours,
                amsr_mat=amsr_mat,
                fill_nan_velocity_zero_on_ocean=fill_nan_velocity_zero_on_ocean,
                ice_motion_subsample=ice_motion_subsample,
            )
        lo_acc.append(np.asarray(lon_a, dtype=float).ravel())
        la_acc.append(np.asarray(lat_a, dtype=float).ravel())
        h_acc.append(hk.ravel())
    if not lo_acc:
        return np.full(np.asarray(lon_q).shape, np.nan, dtype=float)
    points_lon = np.concatenate(lo_acc)
    points_lat = np.concatenate(la_acc)
    points_h = np.concatenate(h_acc)
    return interpolate_h_to_points(
        points_lon,
        points_lat,
        points_h,
        lon_q,
        lat_q,
        nearest_fill_max_dist_km=nearest_fill_max_dist_km,
    )


def compute_eulerian_polynya_closure(
    segment_npz,
    amsr_mat,
    ice_motion_nc,
    out_csv,
    rho_i=915.0,
    polynya_threshold=0.7,
    polynya_type="coastal",
    max_dt_days=31.0,
    thickness_reconstruct="none",
    fusion_stat="median",
    fusion_max_dist_km=100.0,
    center_fill_max_dist_km=None,
    step_hours=6.0,
    backtrack_days=None,
    forwardtrack_days=None,
    fill_nan_velocity_zero_on_ocean=True,
    ice_motion_subsample=5,
    min_coverage_fraction=0.0,
    min_boundary_flux_coverage_fraction=0.0,
    min_coverage_fraction_for_covered_rate=1e-4,
    region_lon_min=None,
    region_lon_max=None,
    region_lat_min=None,
    region_lat_max=None,
    covered_valid_min_boundary_flux_coverage_fraction=0.015,
    covered_valid_min_coverage_fraction=0.002,
    covered_valid_min_n_window_scenes_used=3,
    covered_valid_min_effective_boundary_flux_length_m=8e4,
    production_mode="standard",
    sit_temporal_pool=False,
    sit_pool_lag_days=7.0,
    sit_pool_stat="median",
    sit_lagrangian_pooled_mask=False,
):
    """
    Eulerian mass closure over the AMSR polynya mask:
      M(t) = rho_i * sum_{mask(t)}[ h(x,t) * A_cell ]
      dM/dt from central difference
      F_out > 0 (export proxy from boundary transport, coarse normal projection)
      P_thermo = dM/dt + F_out

    Notes:
      - This is a diagnostic closure; boundary flux is coarse.
      - h(x,t) is from SWOT/IS2 segments interpolated to AMSR locations each scene.
      - Optional hybrid reconstruction:
          thickness_reconstruct='tracked_fusion'
        uses center + backward/forward drift-aware thickness samples to form h_fused.
        This is not a rigorous Lagrangian parcel reconstruction; it is a
        drift-aware multi-temporal thickness fusion (gap-filling).
      - Uncertainty terms are proxy measures of reconstruction/sampling spread
        across center/backward/forward layers, not a full formal error budget.
      - center_fill_max_dist_km controls nearest-fill for center-day interpolation
        (outside linear convex hull); set None for no distance cap on this fill.
      - min_boundary_flux_coverage_fraction can reject scenes where too little
        boundary has finite thickness+velocity support for F_out proxy.
      - min_coverage_fraction_for_covered_rate avoids unstable covered-area
        rates when covered area is extremely tiny.
      - covered_product_valid combines support criteria used for reporting
        covered-area production rates more conservatively.
      - covered_valid_min_coverage_fraction guards against unstable covered-area
        rates when only a tiny fraction of the control area has valid thickness.
      - production_mode='strict' applies conservative built-in validity thresholds
        for covered production intended for publication-style interpretation.
      - sit_temporal_pool=True pools SIT in time for each center scene by aggregating
        per-scene interpolated SIT fields within +/- sit_pool_lag_days.
      - sit_lagrangian_pooled_mask=True builds the base thickness on the polynya mask by
        advecting segment samples from all scenes with |t_k - t_center| <= sit_pool_lag_days
        to the center time (same spirit as plot_SWOT_PM_RACMO_SIT Lagrangian pooled SIT), then
        interpolating to mask points. When True, Eulerian sit_temporal_pool stacking is skipped.
    """
    if sit_lagrangian_pooled_mask:
        if bool(sit_temporal_pool):
            print(
                "[INFO] --sit_lagrangian_pooled_mask: Lagrangian pooled thickness on mask "
                f"(±{float(sit_pool_lag_days):g} d); skipping Eulerian --sit_temporal_pool."
            )
        else:
            print(
                "[INFO] --sit_lagrangian_pooled_mask: thickness on mask from advected segment "
                f"samples within ±{float(sit_pool_lag_days):g} d."
            )

    # Optional strict profile for covered production validity.
    if str(production_mode).lower() == "strict":
        min_coverage_fraction_for_covered_rate = max(float(min_coverage_fraction_for_covered_rate), 0.01)
        covered_valid_min_coverage_fraction = max(float(covered_valid_min_coverage_fraction), 0.02)
        covered_valid_min_boundary_flux_coverage_fraction = max(
            float(covered_valid_min_boundary_flux_coverage_fraction), 0.50
        )
        covered_valid_min_n_window_scenes_used = max(float(covered_valid_min_n_window_scenes_used), 4.0)
        covered_valid_min_effective_boundary_flux_length_m = max(
            float(covered_valid_min_effective_boundary_flux_length_m), 1.2e5
        )

    times, lon_seg, lat_seg, h_seg = _load_segment_snapshot_npz(segment_npz)
    ntime = len(times)
    if ntime < 3:
        raise RuntimeError("Need at least 3 scenes for Eulerian central differences.")

    scenes = []
    for i, t in enumerate(times):
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
        area_cell = grid_cell_area_m2(lat_m, lon_m)

        # Only work on polynya pixels to keep this diagnostic practical.
        idx_mask = np.where(mask_m & np.isfinite(area_cell))
        if idx_mask[0].size < 3:
            continue

        # Thickness on polynya pixels: Eulerian center / temporal pool, or Lagrangian pooled mask.
        lo = np.asarray(lon_seg[i, :], dtype=float).ravel()
        la = np.asarray(lat_seg[i, :], dtype=float).ravel()
        hh = np.asarray(h_seg[i, :], dtype=float).ravel()
        m = np.isfinite(lo) & np.isfinite(la) & np.isfinite(hh)
        lon_q = lon_m[idx_mask]
        lat_q = lat_m[idx_mask]
        h_center_mask = np.full(lon_q.shape, np.nan, dtype=float)
        nearest_fill_max_dist_km = (
            center_fill_max_dist_km if thickness_reconstruct == "tracked_fusion" else None
        )

        if bool(sit_lagrangian_pooled_mask):
            h_center_mask = _h_lagrangian_pooled_on_mask(
                t,
                lon_q,
                lat_q,
                times,
                lon_seg,
                lat_seg,
                h_seg,
                float(sit_pool_lag_days),
                ice_motion_nc,
                amsr_mat,
                step_hours=step_hours,
                ice_motion_subsample=ice_motion_subsample,
                fill_nan_velocity_zero_on_ocean=fill_nan_velocity_zero_on_ocean,
                nearest_fill_max_dist_km=nearest_fill_max_dist_km,
            )
            if (not np.any(np.isfinite(h_center_mask))) and thickness_reconstruct != "tracked_fusion":
                continue
        else:
            # Interpolate center scene segment thickness to polynya pixels.
            # In tracked_fusion, center can be missing and window scenes may still contribute.
            if np.count_nonzero(m) >= 3:
                h_center_mask = interpolate_h_to_points(
                    lo[m], la[m], hh[m],
                    lon_q, lat_q,
                    nearest_fill_max_dist_km=nearest_fill_max_dist_km,
                )
            elif thickness_reconstruct != "tracked_fusion":
                continue

            # Optional temporal pooling of SIT at center time:
            # aggregate interpolated SIT maps from scenes within +/- sit_pool_lag_days.
            if bool(sit_temporal_pool):
                half = timedelta(days=float(sit_pool_lag_days))
                pooled_layers = []
                for k, tk in enumerate(times):
                    if abs((tk - t).total_seconds()) > half.total_seconds():
                        continue
                    lok = np.asarray(lon_seg[k, :], dtype=float).ravel()
                    lak = np.asarray(lat_seg[k, :], dtype=float).ravel()
                    hhk = np.asarray(h_seg[k, :], dtype=float).ravel()
                    mk = np.isfinite(lok) & np.isfinite(lak) & np.isfinite(hhk)
                    if np.count_nonzero(mk) < 3:
                        continue
                    hk = interpolate_h_to_points(
                        lok[mk], lak[mk], hhk[mk],
                        lon_q, lat_q,
                        nearest_fill_max_dist_km=center_fill_max_dist_km,
                    )
                    if np.any(np.isfinite(hk)):
                        pooled_layers.append(hk)
                if len(pooled_layers) > 0:
                    stk_pool = np.stack(pooled_layers, axis=0)
                    if str(sit_pool_stat).lower() == "mean":
                        h_center_mask = np.nanmean(stk_pool, axis=0)
                    else:
                        h_center_mask = np.nanmedian(stk_pool, axis=0)

        # Drift-aware multi-temporal fusion statistics at center time
        layers = []
        if np.any(np.isfinite(h_center_mask)):
            layers.append(h_center_mask)
        h_use_mask = h_center_mask
        n_fused_sources_mean = np.nan
        n_fused_sources_std = np.nan
        n_window_scenes_used = 0
        if thickness_reconstruct == "tracked_fusion":
            if backtrack_days is None:
                t_back_target = times[max(i - 1, 0)]
            else:
                t_back_target = t - timedelta(days=float(backtrack_days))
            if forwardtrack_days is None:
                t_fwd_target = times[min(i + 1, ntime - 1)]
            else:
                t_fwd_target = t + timedelta(days=float(forwardtrack_days))

            # Use all available scenes within [t_back_target, t_fwd_target] (excluding center).
            win_idx = [
                k for k, tk in enumerate(times)
                if (k != i) and (tk >= t_back_target) and (tk <= t_fwd_target)
            ]
            for k in win_idx:
                lon_k, lat_k = _advect_points_between_times(
                    lon_q, lat_q,
                    t_start=t, t_end=times[k],
                    ice_motion_nc=ice_motion_nc,
                    step_hours=step_hours,
                    amsr_mat=amsr_mat,
                    fill_nan_velocity_zero_on_ocean=fill_nan_velocity_zero_on_ocean,
                    ice_motion_subsample=ice_motion_subsample,
                )
                h_k = _nearest_sample_points(
                    lon_seg[k, :], lat_seg[k, :], h_seg[k, :],
                    lon_k, lat_k,
                    max_dist_km=fusion_max_dist_km
                )
                if np.any(np.isfinite(h_k)):
                    layers.append(h_k)
                    n_window_scenes_used += 1

        if len(layers) == 0:
            continue
        stk = np.stack(layers, axis=0)
        h_mean_mask, h_median_mask, h_std_mask, h_q25_mask, h_q75_mask, nsrc = _stack_reduce_stats(stk)
        # Report fusion-source support over covered pixels only; averaging over all
        # polynya pixels can be misleadingly tiny when most pixels are uncovered.
        msrc_cov = nsrc > 0
        n_fused_sources_mean = float(np.nanmean(nsrc[msrc_cov])) if np.any(msrc_cov) else np.nan
        n_fused_sources_std = float(np.nanstd(nsrc[msrc_cov])) if np.any(msrc_cov) else np.nan
        h_use_mask = h_mean_mask if fusion_stat == "mean" else h_median_mask

        # Coverage-aware thickness field on polynya pixels
        h_use = np.full_like(mask_m, np.nan, dtype=float)
        h_use[idx_mask] = h_use_mask

        A_polynya_m2 = float(np.nansum(area_cell[idx_mask]))
        h_finite = np.isfinite(h_use_mask)
        A_thickness_covered_m2 = float(np.nansum(area_cell[idx_mask][h_finite]))
        effective_supported_area_m2 = A_thickness_covered_m2
        coverage_fraction = (A_thickness_covered_m2 / A_polynya_m2) if A_polynya_m2 > 0 else np.nan

        # Center-only reconstructed coverage (reconstruction support, not observational coverage).
        h_center_finite = np.isfinite(h_center_mask)
        A_center_thickness_covered_m2 = float(np.nansum(area_cell[idx_mask][h_center_finite]))
        coverage_fraction_center_only = (
            (A_center_thickness_covered_m2 / A_polynya_m2) if A_polynya_m2 > 0 else np.nan
        )

        area_mask = area_cell[idx_mask]
        hbar_covered_m = _area_weighted_mean(h_use_mask, area_mask)
        hbar_mean_m = _area_weighted_mean(h_mean_mask, area_mask)
        hbar_median_m = _area_weighted_mean(h_median_mask, area_mask)
        hbar_q25_m = _area_weighted_mean(h_q25_mask, area_mask)
        hbar_q75_m = _area_weighted_mean(h_q75_mask, area_mask)
        hbar_iqr_m = (hbar_q75_m - hbar_q25_m) if (np.isfinite(hbar_q75_m) and np.isfinite(hbar_q25_m)) else np.nan
        hfusion_std_area_m = _area_weighted_rms(h_std_mask, area_mask)

        # Optional scene filtering by coverage
        if np.isfinite(coverage_fraction) and coverage_fraction < float(min_coverage_fraction):
            continue

        # Mass diagnostics: selected reducer + sensitivity/uncertainty proxies.
        mass_kg, mass_unc_proxy_kg = _integrated_mass_and_uncertainty(
            h_use_mask, area_mask, rho_i, h_std=h_std_mask
        )
        mass_mean_kg, _ = _integrated_mass_and_uncertainty(h_mean_mask, area_mask, rho_i)
        mass_median_kg, _ = _integrated_mass_and_uncertainty(h_median_mask, area_mask, rho_i)

        # Boundary flux: export proxy from coarse normal transport
        # Use coast-adjacent boundary inside the selected region.
        # This avoids treating the outer sea-ice edge as the control boundary.
        bnd_all = mask_boundary_4n(mask_full) & mask_m
        land_m = _land_mask_for_date(amsr_mat, t)
        bnd = bnd_all & _neighbors4(land_m)
        if not np.any(bnd):
            # Fallback to full physical polynya boundary if coast-adjacent set is empty.
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
        h_tmp_mean = np.full_like(mask_m, np.nan, dtype=float)
        h_tmp_median = np.full_like(mask_m, np.nan, dtype=float)
        h_tmp_std = np.full_like(mask_m, np.nan, dtype=float)
        h_tmp_mean[idx_mask] = h_mean_mask
        h_tmp_median[idx_mask] = h_median_mask
        h_tmp_std[idx_mask] = h_std_mask

        h_b = h_use[idx_bnd]
        h_b_mean = h_tmp_mean[idx_bnd]
        h_b_median = h_tmp_median[idx_bnd]
        h_b_std = h_tmp_std[idx_bnd]
        m_bflux = np.isfinite(h_b) & np.isfinite(un)
        boundary_flux_coverage_fraction = (
            float(np.count_nonzero(m_bflux) / m_bflux.size) if m_bflux.size > 0 else np.nan
        )
        n_boundary_points_total = int(m_bflux.size)
        n_boundary_points_used = int(np.count_nonzero(m_bflux))
        effective_boundary_flux_length_m = float(n_boundary_points_used * ds) if np.isfinite(ds) else np.nan
        if (
            np.isfinite(boundary_flux_coverage_fraction)
            and boundary_flux_coverage_fraction < float(min_boundary_flux_coverage_fraction)
        ):
            continue
        fout_kg_s, fout_unc_proxy_kg_s = _boundary_flux_and_uncertainty(
            h_b, un, ds, rho_i, hstd_b=h_b_std
        )
        fout_mean_kg_s, _ = _boundary_flux_and_uncertainty(h_b_mean, un, ds, rho_i)
        fout_median_kg_s, _ = _boundary_flux_and_uncertainty(h_b_median, un, ds, rho_i)

        scenes.append({
            "t": t,
            "area_m2": A_polynya_m2,
            "A_thickness_covered_m2": A_thickness_covered_m2,
            "effective_supported_area_m2": effective_supported_area_m2,
            "coverage_fraction": coverage_fraction,
            "coverage_fraction_center_only": coverage_fraction_center_only,
            "boundary_flux_coverage_fraction": boundary_flux_coverage_fraction,
            "n_boundary_points_total": n_boundary_points_total,
            "n_boundary_points_used": n_boundary_points_used,
            "effective_boundary_flux_length_m": effective_boundary_flux_length_m,
            "hbar_covered_m": hbar_covered_m,
            "hbar_mean_m": hbar_mean_m,
            "hbar_median_m": hbar_median_m,
            "hbar_q25_m": hbar_q25_m,
            "hbar_q75_m": hbar_q75_m,
            "hbar_iqr_m": hbar_iqr_m,
            "hfusion_std_area_m": hfusion_std_area_m,
            "mass_kg": mass_kg,
            "mass_mean_kg": mass_mean_kg,
            "mass_median_kg": mass_median_kg,
            "mass_unc_proxy_kg": mass_unc_proxy_kg,
            "fout_kg_s": fout_kg_s,
            "fout_mean_kg_s": fout_mean_kg_s,
            "fout_median_kg_s": fout_median_kg_s,
            "fout_unc_proxy_kg_s": fout_unc_proxy_kg_s,
            "n_window_scenes_used": n_window_scenes_used,
            "n_fused_sources_mean": n_fused_sources_mean,
            "n_fused_sources_std": n_fused_sources_std,
            "thickness_on_mask_mode": (
                "lagrangian_pooled_mask"
                if sit_lagrangian_pooled_mask
                else ("temporal_pool_eulerian" if sit_temporal_pool else "center_eulerian")
            ),
        })

    scenes = sorted(scenes, key=lambda x: x["t"])
    if len(scenes) < 3:
        raise RuntimeError("Not enough valid scenes for Eulerian closure after interpolation.")

    rows = []
    for i in range(1, len(scenes) - 1):
        s0 = scenes[i - 1]
        s1 = scenes[i]
        s2 = scenes[i + 1]
        dt_sec = (s2["t"] - s0["t"]).total_seconds()
        if dt_sec <= 0:
            continue
        if (max_dt_days is not None) and (abs((s2["t"] - s0["t"]).days) > float(max_dt_days)):
            continue

        dMdt = (s2["mass_kg"] - s0["mass_kg"]) / dt_sec
        dMdt_mean = (s2["mass_mean_kg"] - s0["mass_mean_kg"]) / dt_sec if (
            np.isfinite(s2["mass_mean_kg"]) and np.isfinite(s0["mass_mean_kg"])
        ) else np.nan
        dMdt_median = (s2["mass_median_kg"] - s0["mass_median_kg"]) / dt_sec if (
            np.isfinite(s2["mass_median_kg"]) and np.isfinite(s0["mass_median_kg"])
        ) else np.nan
        dMdt_unc_proxy = np.nan
        if np.isfinite(s2["mass_unc_proxy_kg"]) and np.isfinite(s0["mass_unc_proxy_kg"]):
            dMdt_unc_proxy = float(np.sqrt(s2["mass_unc_proxy_kg"] ** 2 + s0["mass_unc_proxy_kg"] ** 2) / dt_sec)

        p_thermo = dMdt + s1["fout_kg_s"]
        p_thermo_mean = dMdt_mean + s1["fout_mean_kg_s"] if (
            np.isfinite(dMdt_mean) and np.isfinite(s1["fout_mean_kg_s"])
        ) else np.nan
        p_thermo_median = dMdt_median + s1["fout_median_kg_s"] if (
            np.isfinite(dMdt_median) and np.isfinite(s1["fout_median_kg_s"])
        ) else np.nan
        p_thermo_unc_proxy = np.nan
        if np.isfinite(dMdt_unc_proxy) and np.isfinite(s1["fout_unc_proxy_kg_s"]):
            p_thermo_unc_proxy = float(np.sqrt(dMdt_unc_proxy ** 2 + s1["fout_unc_proxy_kg_s"] ** 2))
        # This is a positive residual (clip of sign of p_thermo). It is not
        # physically isolated production unless you assume melt is negligible.
        p_positive_residual = max(p_thermo, 0.0)
        p_melt_only = max(-p_thermo, 0.0)

        A_polynya = s1["area_m2"]
        # Use a triplet-consistent covered area for covered-rate normalization because
        # P_thermo is derived from a 3-scene tendency (s0,s1,s2). Using only center
        # coverage can amplify spikes when support changes rapidly between scenes.
        A_cov0 = s0["A_thickness_covered_m2"]
        A_cov1 = s1["A_thickness_covered_m2"]
        A_cov2 = s2["A_thickness_covered_m2"]
        A_cov_triplet = float(np.nanmedian([A_cov0, A_cov1, A_cov2]))
        cov0 = s0["coverage_fraction"]
        cov1 = s1["coverage_fraction"]
        cov2 = s2["coverage_fraction"]
        cov_frac_triplet_min = float(np.nanmin([cov0, cov1, cov2]))
        cov_frac = cov1
        prod_m_s_polynya = p_thermo / (rho_i * A_polynya) if A_polynya > 0 else np.nan
        covered_ok_base = (
            np.isfinite(A_cov_triplet)
            and (A_cov_triplet > 0)
            and np.isfinite(cov_frac_triplet_min)
            and (cov_frac_triplet_min >= float(min_coverage_fraction_for_covered_rate))
        )
        covered_product_valid = bool(
            covered_ok_base
            and np.isfinite(cov_frac_triplet_min)
            and (cov_frac_triplet_min >= float(covered_valid_min_coverage_fraction))
            and np.isfinite(s1["boundary_flux_coverage_fraction"])
            and (s1["boundary_flux_coverage_fraction"] >= float(covered_valid_min_boundary_flux_coverage_fraction))
            and (float(s1["n_window_scenes_used"]) >= float(covered_valid_min_n_window_scenes_used))
            and np.isfinite(s1["effective_boundary_flux_length_m"])
            and (s1["effective_boundary_flux_length_m"] >= float(covered_valid_min_effective_boundary_flux_length_m))
        )
        # Raw covered-area diagnostic (no validity gate), useful to inspect expected magnitude.
        prod_m_s_covered_raw = p_thermo / (rho_i * A_cov_triplet) if A_cov_triplet > 0 else np.nan
        prod_m_s_covered = prod_m_s_covered_raw if covered_product_valid else np.nan
        prod_m_s_polynya_unc_proxy = (
            p_thermo_unc_proxy / (rho_i * A_polynya) if (A_polynya > 0 and np.isfinite(p_thermo_unc_proxy)) else np.nan
        )
        prod_m_s_covered_unc_proxy = (
            p_thermo_unc_proxy / (rho_i * A_cov_triplet) if (covered_product_valid and np.isfinite(p_thermo_unc_proxy)) else np.nan
        )

        rows.append({
            "time_center": s1["t"].isoformat(sep=" "),
            "time_back": s0["t"].isoformat(sep=" "),
            "time_fwd": s2["t"].isoformat(sep=" "),
            "thickness_reconstruct": thickness_reconstruct,
            "sit_lagrangian_pooled_mask": int(bool(sit_lagrangian_pooled_mask)),
            "thickness_on_mask_mode": s1["thickness_on_mask_mode"],
            "sit_temporal_pool": int(bool(sit_temporal_pool)),
            "sit_pool_lag_days": float(sit_pool_lag_days),
            "sit_pool_stat": str(sit_pool_stat),
            "fusion_stat": fusion_stat,
            "fusion_max_dist_km": fusion_max_dist_km,
            "center_fill_max_dist_km": center_fill_max_dist_km,
            "min_boundary_flux_coverage_fraction": min_boundary_flux_coverage_fraction,
            "min_coverage_fraction_for_covered_rate": min_coverage_fraction_for_covered_rate,
            "production_mode": str(production_mode),
            "covered_valid_min_coverage_fraction": covered_valid_min_coverage_fraction,
            "covered_valid_min_boundary_flux_coverage_fraction": covered_valid_min_boundary_flux_coverage_fraction,
            "covered_valid_min_n_window_scenes_used": covered_valid_min_n_window_scenes_used,
            "covered_valid_min_effective_boundary_flux_length_m": covered_valid_min_effective_boundary_flux_length_m,
            "A_polynya_m2": s1["area_m2"],
            "A_thickness_covered_m2": s1["A_thickness_covered_m2"],
            "A_thickness_covered_triplet_m2": A_cov_triplet,
            "effective_supported_area_m2": s1["effective_supported_area_m2"],
            "coverage_fraction": s1["coverage_fraction"],
            "coverage_fraction_triplet_min": cov_frac_triplet_min,
            "coverage_fraction_center_only": s1["coverage_fraction_center_only"],
            "boundary_flux_coverage_fraction": s1["boundary_flux_coverage_fraction"],
            "n_boundary_points_total": s1["n_boundary_points_total"],
            "n_boundary_points_used": s1["n_boundary_points_used"],
            "effective_boundary_flux_length_m": s1["effective_boundary_flux_length_m"],
            "covered_product_valid": int(covered_product_valid),
            "hbar_covered_m": s1["hbar_covered_m"],
            "hbar_mean_m": s1["hbar_mean_m"],
            "hbar_median_m": s1["hbar_median_m"],
            "hbar_q25_m": s1["hbar_q25_m"],
            "hbar_q75_m": s1["hbar_q75_m"],
            "hbar_iqr_m": s1["hbar_iqr_m"],
            "hfusion_std_area_m": s1["hfusion_std_area_m"],
            "M_polynya_kg": s1["mass_kg"],
            "M_polynya_kg_mean": s1["mass_mean_kg"],
            "M_polynya_kg_median": s1["mass_median_kg"],
            "M_polynya_kg_unc_proxy": s1["mass_unc_proxy_kg"],
            "n_window_scenes_used": s1["n_window_scenes_used"],
            "n_fused_sources_mean": s1["n_fused_sources_mean"],
            "n_fused_sources_std": s1["n_fused_sources_std"],
            "dMdt_kg_s": dMdt,
            "dMdt_kg_s_mean": dMdt_mean,
            "dMdt_kg_s_median": dMdt_median,
            "dMdt_kg_s_unc_proxy": dMdt_unc_proxy,
            "Fout_kg_s": s1["fout_kg_s"],
            # Same value, clearer interpretation: coarse export proxy.
            "Fout_boundary_transport_proxy_kg_s": s1["fout_kg_s"],
            "Fout_kg_s_mean": s1["fout_mean_kg_s"],
            "Fout_kg_s_median": s1["fout_median_kg_s"],
            "Fout_kg_s_unc_proxy": s1["fout_unc_proxy_kg_s"],
            "P_thermo_kg_s": p_thermo,
            "P_thermo_kg_s_mean": p_thermo_mean,
            "P_thermo_kg_s_median": p_thermo_median,
            "P_thermo_kg_s_unc_proxy": p_thermo_unc_proxy,
            # Backward-compatible name; see P_positive_residual_kg_s for wording.
            "P_new_only_kg_s": p_positive_residual,
            "P_positive_residual_kg_s": p_positive_residual,
            "P_negative_residual_kg_s": p_melt_only,
            "P_melt_only_kg_s": p_melt_only,
            "P_thermo_m_s_area_mean": prod_m_s_polynya,
            "P_thermo_cm_day_area_mean": prod_m_s_polynya * 100.0 * 86400.0 if np.isfinite(prod_m_s_polynya) else np.nan,
            "P_thermo_m_s_area_mean_unc_proxy": prod_m_s_polynya_unc_proxy,
            "P_thermo_cm_day_area_mean_unc_proxy": prod_m_s_polynya_unc_proxy * 100.0 * 86400.0 if np.isfinite(prod_m_s_polynya_unc_proxy) else np.nan,
            "P_thermo_m_s_area_mean_covered": prod_m_s_covered,
            "P_thermo_cm_day_area_mean_covered": prod_m_s_covered * 100.0 * 86400.0 if np.isfinite(prod_m_s_covered) else np.nan,
            "P_thermo_m_s_area_mean_covered_raw": prod_m_s_covered_raw,
            "P_thermo_cm_day_area_mean_covered_raw": prod_m_s_covered_raw * 100.0 * 86400.0 if np.isfinite(prod_m_s_covered_raw) else np.nan,
            "P_thermo_m_s_area_mean_covered_unc_proxy": prod_m_s_covered_unc_proxy,
            "P_thermo_cm_day_area_mean_covered_unc_proxy": prod_m_s_covered_unc_proxy * 100.0 * 86400.0 if np.isfinite(prod_m_s_covered_unc_proxy) else np.nan,
        })

    if len(rows) == 0:
        raise RuntimeError("Eulerian closure produced no output rows.")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def compute_segment_trajectory_budget(
    segment_npz,
    ice_motion_nc,
    out_csv,
    amsr_mat=None,
    rho_i=915.0,
    segment_area_m2=6250.0 * 6250.0,
    step_hours=6.0,
    max_dt_days=31,
    polynya_threshold=0.7,
    polynya_type="coastal",
    amsr_mask_mode="center",
    fill_nan_velocity_zero_on_ocean=True,
    ice_motion_subsample=5,
    backtrack_days=None,
    forwardtrack_days=None,
    segment_selection_mode="center",
    debug=False,
):
    """
    Segment-level backward/forward drifting mass diagnostic.
    For each interior scene t_i and each segment j:
      - backtrack (lon_i,lat_i) to t_{i-1}
      - forward-track (lon_i,lat_i) to t_{i+1}
      - select segments by mode:
          * center: only segments finite at center time
          * window_union: segments finite at any snapshot within [t_i-back, t_i+fwd]
      - optionally require AMSR SIC-based polynya membership
      - sample h_{i-1}, h_i, h_{i+1} at fixed and advected locations
      - estimate:
          dHdt_local    = [h_{i+1}(x_i)-h_{i-1}(x_i)] / dt
          dHdt_material = [h_{i+1}(x_fwd)-h_{i-1}(x_back)] / dt
          advective_est = dHdt_local - dHdt_material
      - convert material tendency to mass tendency using segment_area_m2
        (bookkeeping approximation unless your segments were constructed
         to represent equal-area support).
    """
    times, lon, lat, h = _load_segment_snapshot_npz(segment_npz)
    rows = []
    ntime, nseg = h.shape
    if ntime < 3:
        raise RuntimeError("Need at least 3 scenes for central differences.")

    for i in range(1, ntime - 1):
        t0, t1, t2 = times[i - 1], times[i], times[i + 1]
        if backtrack_days is None:
            t_back_target = t0
        else:
            t_back_target = t1 - timedelta(days=float(backtrack_days))
        if forwardtrack_days is None:
            t_fwd_target = t2
        else:
            t_fwd_target = t1 + timedelta(days=float(forwardtrack_days))

        dt_sec = (t_fwd_target - t_back_target).total_seconds()
        if dt_sec <= 0:
            continue
        if (max_dt_days is not None) and (abs((t_fwd_target - t_back_target).days) > max_dt_days):
            if debug:
                print(
                    f"[DEBUG] {t1}: skip window {abs((t_fwd_target - t_back_target).days)} days "
                    f"> max_dt_days={max_dt_days}"
                )
            continue

        lon_i = lon[i, :]
        lat_i = lat[i, :]
        h_i = h[i, :]
        valid_center_geo = np.isfinite(lon_i) & np.isfinite(lat_i)
        valid_center_full = valid_center_geo & np.isfinite(h_i)
        if not np.any(valid_center_geo):
            continue

        # Segment availability selector
        tmask = np.array([(tt >= t_back_target) and (tt <= t_fwd_target) for tt in times], dtype=bool)
        if np.any(tmask):
            avail_in_window_matrix = (
                np.isfinite(lon[tmask, :]) &
                np.isfinite(lat[tmask, :]) &
                np.isfinite(h[tmask, :])
            )
            available_count_in_window = np.sum(avail_in_window_matrix, axis=0).astype(int)
        else:
            available_count_in_window = np.zeros(nseg, dtype=int)

        if segment_selection_mode == "window_union":
            any_in_window = available_count_in_window > 0
            selected = valid_center_geo & any_in_window
        else:
            selected = valid_center_full
        if not np.any(selected):
            if debug:
                print(f"[DEBUG] {t1}: no selected segments after selection mode '{segment_selection_mode}'")
            continue

        center_in_polynya = np.ones_like(selected, dtype=bool)
        if amsr_mat is not None and amsr_mask_mode in ("center", "all"):
            lat_m, lon_m, mask_m = load_polynya_mask_for_date(
                amsr_mat, t1,
                polynya_threshold=polynya_threshold,
                polynya_type=polynya_type,
            )
            center_in_polynya = _sample_bool_mask_nearest(
                lon_m, lat_m, mask_m, lon_i, lat_i
            )

        lon_back, lat_back = _advect_points_between_times(
            lon_i,
            lat_i,
            t_start=t1,
            t_end=t_back_target,
            ice_motion_nc=ice_motion_nc,
            step_hours=step_hours,
            amsr_mat=amsr_mat,
            fill_nan_velocity_zero_on_ocean=fill_nan_velocity_zero_on_ocean,
            ice_motion_subsample=ice_motion_subsample,
        )
        lon_fwd, lat_fwd = _advect_points_between_times(
            lon_i, lat_i, t_start=t1, t_end=t_fwd_target, ice_motion_nc=ice_motion_nc, step_hours=step_hours,
            amsr_mat=amsr_mat, fill_nan_velocity_zero_on_ocean=fill_nan_velocity_zero_on_ocean,
            ice_motion_subsample=ice_motion_subsample
        )

        back_in_polynya = np.ones_like(selected, dtype=bool)
        fwd_in_polynya = np.ones_like(selected, dtype=bool)
        if amsr_mat is not None and amsr_mask_mode == "all":
            lat_bm, lon_bm, mask_bm = load_polynya_mask_for_date(
                amsr_mat, t_back_target,
                polynya_threshold=polynya_threshold,
                polynya_type=polynya_type,
            )
            back_in_polynya = _sample_bool_mask_nearest(
                lon_bm, lat_bm, mask_bm, lon_back, lat_back
            )
            lat_fm, lon_fm, mask_fm = load_polynya_mask_for_date(
                amsr_mat, t_fwd_target,
                polynya_threshold=polynya_threshold,
                polynya_type=polynya_type,
            )
            fwd_in_polynya = _sample_bool_mask_nearest(
                lon_fm, lat_fm, mask_fm, lon_fwd, lat_fwd
            )

        h0_fixed = _nearest_sample_points(lon[i - 1, :], lat[i - 1, :], h[i - 1, :], lon_i, lat_i)
        h2_fixed = _nearest_sample_points(lon[i + 1, :], lat[i + 1, :], h[i + 1, :], lon_i, lat_i)
        h0_back = _nearest_sample_points(lon[i - 1, :], lat[i - 1, :], h[i - 1, :], lon_back, lat_back)
        h2_fwd = _nearest_sample_points(lon[i + 1, :], lat[i + 1, :], h[i + 1, :], lon_fwd, lat_fwd)

        dHdt_local = (h2_fixed - h0_fixed) / dt_sec
        dHdt_material = (h2_fwd - h0_back) / dt_sec
        adv_est = dHdt_local - dHdt_material
        dMdt_material = rho_i * segment_area_m2 * dHdt_material

        # Drift distances (km)
        dx_back = (wrap_lon180(lon_back - lon_i)) * np.cos(np.deg2rad(lat_i)) * 111.0
        dy_back = (lat_back - lat_i) * 111.0
        d_back_km = np.sqrt(dx_back ** 2 + dy_back ** 2)
        dx_fwd = (wrap_lon180(lon_fwd - lon_i)) * np.cos(np.deg2rad(lat_i)) * 111.0
        dy_fwd = (lat_fwd - lat_i) * 111.0
        d_fwd_km = np.sqrt(dx_fwd ** 2 + dy_fwd ** 2)

        for j in range(nseg):
            keep = bool(selected[j] and center_in_polynya[j])
            if amsr_mat is not None and amsr_mask_mode == "all":
                keep = keep and bool(back_in_polynya[j] and fwd_in_polynya[j])
            if not keep:
                continue
            rows.append({
                "time_center": t1.isoformat(sep=" "),
                "time_back_target": t_back_target.isoformat(sep=" "),
                "time_fwd_target": t_fwd_target.isoformat(sep=" "),
                "segment_id": j,
                "segment_area_m2_used": float(segment_area_m2),
                "segment_selected_mode": segment_selection_mode,
                "selected_in_window_union": int(selected[j]),
                "available_count_in_window": int(available_count_in_window[j]),
                "lon_center_0360": float(lon_i[j]),
                "lat_center_deg": float(lat_i[j]),
                "h_center_m": float(h_i[j]),
                "in_polynya_center": int(center_in_polynya[j]),
                "in_polynya_back": int(back_in_polynya[j]),
                "in_polynya_fwd": int(fwd_in_polynya[j]),
                "lon_back_0360": float(lon_back[j]) if np.isfinite(lon_back[j]) else np.nan,
                "lat_back_deg": float(lat_back[j]) if np.isfinite(lat_back[j]) else np.nan,
                "lon_fwd_0360": float(lon_fwd[j]) if np.isfinite(lon_fwd[j]) else np.nan,
                "lat_fwd_deg": float(lat_fwd[j]) if np.isfinite(lat_fwd[j]) else np.nan,
                "drift_back_km": float(d_back_km[j]) if np.isfinite(d_back_km[j]) else np.nan,
                "drift_fwd_km": float(d_fwd_km[j]) if np.isfinite(d_fwd_km[j]) else np.nan,
                "h_backsample_m": float(h0_back[j]) if np.isfinite(h0_back[j]) else np.nan,
                "h_fwdsample_m": float(h2_fwd[j]) if np.isfinite(h2_fwd[j]) else np.nan,
                "dHdt_local_m_s": float(dHdt_local[j]) if np.isfinite(dHdt_local[j]) else np.nan,
                "dHdt_material_m_s": float(dHdt_material[j]) if np.isfinite(dHdt_material[j]) else np.nan,
                "advective_est_m_s": float(adv_est[j]) if np.isfinite(adv_est[j]) else np.nan,
                "dMdt_material_kg_s": float(dMdt_material[j]) if np.isfinite(dMdt_material[j]) else np.nan,
            })
        if debug:
            n_sel = int(np.sum(selected))
            n_c = int(np.sum(center_in_polynya & selected))
            if amsr_mask_mode == "all":
                n_all = int(np.sum(selected & center_in_polynya & back_in_polynya & fwd_in_polynya))
                print(f"[DEBUG] {t1}: selected={n_sel} center_polynya={n_c} all_masks={n_all}")
            else:
                print(f"[DEBUG] {t1}: selected={n_sel} center_polynya={n_c}")

    if len(rows) == 0:
        raise RuntimeError("No valid segment rows were produced.")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def _print_eulerian_production_summary(rows):
    """Print compact production diagnostics for a quick pre-check."""
    if not rows:
        print("[INFO] No Eulerian rows available for production summary.")
        return

    def _arr(key):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key, np.nan)))
            except Exception:
                vals.append(np.nan)
        return np.asarray(vals, dtype=float)

    def _fmt_stats(name, arr):
        m = np.isfinite(arr)
        n = int(np.count_nonzero(m))
        if n == 0:
            print(f"  {name}: n=0 (all NaN)")
            return
        a = arr[m]
        print(
            f"  {name}: n={n}, mean={np.nanmean(a):.3f}, median={np.nanmedian(a):.3f}, "
            f"range=[{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]"
        )

    print("[INFO] Eulerian production pre-check")
    _fmt_stats("Whole-area (cm/day)", _arr("P_thermo_cm_day_area_mean"))
    _fmt_stats("Whole-area (m/month, 30d)", _arr("P_thermo_cm_day_area_mean") * 0.30)
    _fmt_stats("Covered valid (cm/day)", _arr("P_thermo_cm_day_area_mean_covered"))
    _fmt_stats("Covered raw (cm/day)", _arr("P_thermo_cm_day_area_mean_covered_raw"))
    _fmt_stats("Covered raw (m/month, 30d)", _arr("P_thermo_cm_day_area_mean_covered_raw") * 0.30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sit_npz", default=None, help="plot-elements npz containing sit_times_str and sit_mean")
    ap.add_argument("--amsr_mat", default=None, help="AMSR2_2023.mat with sic/lat/lon")
    ap.add_argument("--ice_motion_nc", required=True, help="daily ice motion netcdf")
    ap.add_argument("--segment_npz", default=None,
                    help="Segment snapshots NPZ OR directory of *_is2_polynya_sit_outputs.npz scenes")
    ap.add_argument("--eulerian_closure", action="store_true",
                    help="Compute Eulerian polynya mass closure (scene-level) from segment snapshots")
    ap.add_argument("--thickness_reconstruct", choices=["none", "tracked_fusion"], default="none",
                    help="Eulerian thickness field option: 'none' = center-only; 'tracked_fusion' = drift-aware multi-temporal thickness fusion (center/back/fwd), not a rigorous Lagrangian parcel reconstruction")
    ap.add_argument("--sit_temporal_pool", action="store_true",
                    help="In Eulerian closure, temporally pool SIT at each center scene using scenes within +/- --sit_pool_lag_days before mass/flux diagnostics.")
    ap.add_argument("--sit_pool_lag_days", type=float, default=7.0,
                    help="Half-window (days) for --sit_temporal_pool (default 7).")
    ap.add_argument("--sit_pool_stat", choices=["median", "mean"], default="median",
                    help="Reducer for --sit_temporal_pool (default median).")
    ap.add_argument("--sit_lagrangian_pooled_mask", action="store_true",
                    help="Eulerian closure: thickness on the AMSR polynya mask from segment samples in "
                         "|t_k-t|<=sit_pool_lag_days, advected to center time (aligned with "
                         "plot_SWOT_PM_RACMO_SIT Lagrangian pooled SIT), then interpolated to mask points. "
                         "When set, Eulerian --sit_temporal_pool is not used (info printed).")
    ap.add_argument("--fusion_stat", choices=["median", "mean"], default="median",
                    help="Reducer used in 'tracked_fusion' to combine center/back/fwd thickness estimates")
    ap.add_argument("--fusion_max_dist_km", type=float, default=100.0,
                    help="Approx max lon/lat distance (km) allowed for nearest-neighbor back/forward thickness fusion in tracked_fusion")
    ap.add_argument("--center_fill_max_dist_km", type=float, default=None,
                    help="Approx max lon/lat distance (km) for nearest-fill of center-day interpolation outside linear hull; default None (no cap)")
    ap.add_argument("--out_csv", default="drift_aware_mass_budget.csv")
    ap.add_argument("--rho_i", type=float, default=915.0)
    ap.add_argument("--polynya_threshold", type=float, default=0.7)
    ap.add_argument("--polynya_type", choices=["open", "coastal", "both"], default="coastal")
    ap.add_argument("--segment_area_m2", type=float, default=6250.0 * 6250.0,
                    help="Area represented by one segment point, for dM/dt conversion")
    ap.add_argument("--segment_step_hours", type=float, default=6.0,
                    help="Advection integration step (hours) for segment tracking")
    ap.add_argument("--ice_motion_subsample", type=int, default=5,
                    help="Subsample passed to load_ice_motion (smaller can be more accurate but slower)")
    ap.add_argument("--max_dt_days", type=float, default=31.0,
                    help="Maximum allowed (back+forward) window length in days; set larger for long windows")
    ap.add_argument("--min_coverage_fraction", type=float, default=0.0,
                    help="Eulerian mode: reject scenes with thickness coverage below this fraction")
    ap.add_argument("--min_boundary_flux_coverage_fraction", type=float, default=0.0,
                    help="Eulerian mode: reject scenes with low finite boundary support for Fout proxy")
    ap.add_argument("--min_coverage_fraction_for_covered_rate", type=float, default=1e-4,
                    help="Eulerian mode: only compute covered-area rates when coverage_fraction >= this value")
    ap.add_argument("--region_lon_min", type=float, default=None,
                    help="Optional regional clip (degE 0..360): minimum longitude")
    ap.add_argument("--region_lon_max", type=float, default=None,
                    help="Optional regional clip (degE 0..360): maximum longitude")
    ap.add_argument("--region_lat_min", type=float, default=None,
                    help="Optional regional clip (deg): minimum latitude")
    ap.add_argument("--region_lat_max", type=float, default=None,
                    help="Optional regional clip (deg): maximum latitude")
    ap.add_argument("--covered_valid_min_boundary_flux_coverage_fraction", type=float, default=0.015,
                    help="Covered-product validity: minimum boundary flux coverage fraction")
    ap.add_argument("--covered_valid_min_coverage_fraction", type=float, default=0.002,
                    help="Covered-product validity: minimum thickness coverage fraction")
    ap.add_argument("--covered_valid_min_n_window_scenes_used", type=float, default=3.0,
                    help="Covered-product validity: minimum number of window scenes contributing")
    ap.add_argument("--covered_valid_min_effective_boundary_flux_length_m", type=float, default=8e4,
                    help="Covered-product validity: minimum effective boundary flux length (m)")
    ap.add_argument("--production_mode", choices=["standard", "strict"], default="standard",
                    help="Covered-production validity profile: standard or strict")
    ap.add_argument("--backtrack_days", type=float, default=None,
                    help="Explicit backward tracking duration in days from each center time")
    ap.add_argument("--forwardtrack_days", type=float, default=None,
                    help="Explicit forward tracking duration in days from each center time")
    ap.add_argument("--amsr_mask_mode", choices=["none", "center", "all"], default="center",
                    help="Segment mode AMSR mask requirement: none, center only, or center+back+forward")
    ap.add_argument("--segment_selection_mode", choices=["center", "window_union"], default="center",
                    help="Segment inclusion rule: finite at center only, or union across [t-back, t+fwd] window")
    ap.add_argument("--no_fill_nan_velocity_zero_on_ocean", action="store_true",
                    help="Disable NaN drift -> 0 fill over AMSR-valid (non-land proxy) surface")
    ap.add_argument("--debug", action="store_true", help="Print diagnostic counts per center time")
    args = ap.parse_args()

    if args.eulerian_closure:
        if args.segment_npz is None:
            raise RuntimeError("Eulerian closure mode requires --segment_npz (file or directory).")
        if args.amsr_mat is None:
            raise RuntimeError("Eulerian closure mode requires --amsr_mat.")
        rows = compute_eulerian_polynya_closure(
            segment_npz=args.segment_npz,
            amsr_mat=args.amsr_mat,
            ice_motion_nc=args.ice_motion_nc,
            out_csv=args.out_csv,
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
            sit_temporal_pool=args.sit_temporal_pool,
            sit_pool_lag_days=args.sit_pool_lag_days,
            sit_pool_stat=args.sit_pool_stat,
            sit_lagrangian_pooled_mask=args.sit_lagrangian_pooled_mask,
        )
        print(f"[INFO] Wrote {len(rows)} Eulerian closure rows: {args.out_csv}")
        _print_eulerian_production_summary(rows)
        return

    if args.segment_npz is not None:
        if args.amsr_mat is None:
            raise RuntimeError("Segment mode now requires --amsr_mat for SIC/polynya masking.")
        rows = compute_segment_trajectory_budget(
            segment_npz=args.segment_npz,
            ice_motion_nc=args.ice_motion_nc,
            out_csv=args.out_csv,
            amsr_mat=args.amsr_mat,
            rho_i=args.rho_i,
            segment_area_m2=args.segment_area_m2,
            step_hours=args.segment_step_hours,
            ice_motion_subsample=args.ice_motion_subsample,
            polynya_threshold=args.polynya_threshold,
            polynya_type=args.polynya_type,
            amsr_mask_mode=args.amsr_mask_mode,
            fill_nan_velocity_zero_on_ocean=not args.no_fill_nan_velocity_zero_on_ocean,
            backtrack_days=args.backtrack_days,
            forwardtrack_days=args.forwardtrack_days,
            max_dt_days=args.max_dt_days,
            segment_selection_mode=args.segment_selection_mode,
            debug=args.debug,
        )
        print(f"[INFO] Wrote {len(rows)} segment rows: {args.out_csv}")
        return

    if args.sit_npz is None or args.amsr_mat is None:
        raise RuntimeError("Scene-mean mode requires --sit_npz and --amsr_mat")

    times, hbar = load_sit_snapshot_points(args.sit_npz)
    rows = compute_mass_budget(
        times=times, hbar=hbar,
        amsr_mat=args.amsr_mat, ice_motion_nc=args.ice_motion_nc,
        rho_i=args.rho_i,
        polynya_threshold=args.polynya_threshold,
        polynya_type=args.polynya_type
    )

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} scene rows: {args.out_csv}")


if __name__ == "__main__":
    main()