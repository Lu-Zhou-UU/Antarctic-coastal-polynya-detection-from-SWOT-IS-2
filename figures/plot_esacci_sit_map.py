#!/usr/bin/env python3
"""
plot_esacci_sit_map.py

Plot sea ice thickness from ESACCI sources (CryoSat-2, Sentinel-3A, Sentinel-3B)
on a polar projection map with:
  (i) coastal polynya outline from AMSR2 SIC
  (ii) ice drift vectors overlay from ice motion product
  (iii) ice shelves and grounding line from BedMachine (optional)

Robust ice drift overlay:
- Do NOT plot native x/y from the ice-motion file (can mismatch your map central_lon/projection).
- Instead, use lon/lat + u/v to compute endpoints, project both, and quiver in projection meters.

Usage:
  python plot_esacci_sit_map.py --date 20231028 \
      --esacci_dir /Volumes/Yotta_1/SWOT/Antarctic_hi \
      --amsr_mat AMSR2_2023.mat \
      --ice_motion_nc /path/to/icemotion_daily_sh_25km_20230101_20231231_v4.1.nc \
      --bedmachine_nc BedMachineAntarctica_2020-07-15_v02.nc \
      --show_ice_shelves --show_grounding_line \
      --constrain_ross_sea \
      --out_png sit_map_20231028.png
"""

import argparse
import os
import re
import glob
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

# Publication-quality settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 1.0,
})

# Optional dependencies
try:
    import netCDF4
    HAS_NETCDF4 = True
except Exception:
    HAS_NETCDF4 = False

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

def _neighbors4_mask(mask):
    """Return boolean mask of pixels that have at least one 4-neighbor True in `mask`."""
    m = mask.astype(bool)
    out = np.zeros_like(m, dtype=bool)

    out[1:, :]  |= m[:-1, :]   # from north
    out[:-1, :] |= m[1:, :]    # from south
    out[:, 1:]  |= m[:, :-1]   # from west
    out[:, :-1] |= m[:, 1:]    # from east
    return out


def _bfs_grow(candidate, seeds):
    """Flood-fill over `candidate` starting from `seeds` (both bool 2D)."""
    from collections import deque

    cand = candidate.astype(bool)
    seen = np.zeros_like(cand, dtype=bool)

    ys, xs = np.where(seeds & cand)
    if ys.size == 0:
        return seen

    q = deque(zip(ys.tolist(), xs.tolist()))
    seen[ys, xs] = True

    ny, nx = cand.shape
    while q:
        y, x = q.popleft()
        # 4-neighbors
        if y > 0 and cand[y-1, x] and not seen[y-1, x]:
            seen[y-1, x] = True; q.append((y-1, x))
        if y < ny-1 and cand[y+1, x] and not seen[y+1, x]:
            seen[y+1, x] = True; q.append((y+1, x))
        if x > 0 and cand[y, x-1] and not seen[y, x-1]:
            seen[y, x-1] = True; q.append((y, x-1))
        if x < nx-1 and cand[y, x+1] and not seen[y, x+1]:
            seen[y, x+1] = True; q.append((y, x+1))

    return seen
# ---------------------------
# MATLAB file loader helpers
# ---------------------------

def _h5_read_dataset(ds):
    """Read HDF5 dataset and reverse axes to match MATLAB ordering."""
    arr = np.array(ds)
    if arr.shape == ():
        return arr
    if arr.ndim >= 2:
        arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
    return arr

def transform_grid_to_geographic(u_grid, v_grid, lon_deg, lat_deg=None):
    """
    Transform EASE-Grid South velocity components to geographic (eastward, northward).
    
    For EASE-Grid South (Lambert Azimuthal Equal Area centered at South Pole):
    - The grid x-axis points eastward at longitude 0°
    - The grid y-axis points northward at longitude 0°
    - At other longitudes, the grid rotates relative to geographic coordinates
    
    Parameters
    ----------
    u_grid : array
        Along-x component (EASE-Grid coordinates)
    v_grid : array
        Along-y component (EASE-Grid coordinates)
    lon_deg : array
        Longitude in degrees (for rotation angle)
    lat_deg : array, optional
        Latitude in degrees (not used in transformation, kept for compatibility)
    
    Returns
    -------
    u_east : array
        Eastward velocity component
    v_north : array
        Northward velocity component
    
    Notes
    -----
    The transformation formula is:
        u_east = u_grid * sin(lon) + v_grid * cos(lon)
        v_north = u_grid * cos(lon) - v_grid * sin(lon)
    
    This accounts for the rotation of the EASE-Grid coordinate system relative
    to geographic coordinates as we move around the South Pole.
    """
    lon_rad = np.deg2rad(lon_deg)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    # Transform from grid coordinates to geographic coordinates
    # For EASE-Grid South (Lambert Azimuthal Equal Area):
    # Note: This transformation matches the working code that produces correct vectors
    # The formula rotates grid-aligned (x,y) to geographic (east, north)
    v_north = u_grid * sin_lon + v_grid * cos_lon
    u_east = u_grid * cos_lon - v_grid * sin_lon
    
    return u_east, v_north


def load_mat_any(path):
    """Load MATLAB file (v7.2 or v7.3)."""
    path = str(path)

    if HAS_SCIPY:
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


def _unwrap_mat_scalar(x):
    """Unwrap common MATLAB loadmat wrappers (0-d arrays, 1-element object arrays)."""
    if x is None:
        return None
    # h5py datasets come as numpy arrays already
    if isinstance(x, np.ndarray):
        if x.shape == () and x.dtype != object:
            return x.item()
        # 1-element object arrays
        if x.dtype == object and x.size == 1:
            return _unwrap_mat_scalar(x.flat[0])
        # 0-d object array
        if x.dtype == object and x.shape == ():
            return _unwrap_mat_scalar(x.item())
    return x


def _mat_get_field(obj, field, default=None):
    """
    Robustly get a field from:
      - scipy.io.loadmat mat_struct (struct_as_record=False)
      - dict
      - numpy structured/record arrays
      - numpy object arrays containing one struct
    """
    obj = _unwrap_mat_scalar(obj)

    if obj is None:
        return default

    # dict-like
    if isinstance(obj, dict):
        return obj.get(field, default)

    # scipy mat_struct (struct_as_record=False)
    if hasattr(obj, "_fieldnames"):
        if field in obj._fieldnames:
            return getattr(obj, field)
        return default

    # numpy structured array / record
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        if field in obj.dtype.names:
            return obj[field]
        return default

    # numpy void (a single record)
    if isinstance(obj, np.void) and obj.dtype.names is not None:
        if field in obj.dtype.names:
            return obj[field]
        return default

    # fall back: attribute access
    if hasattr(obj, field):
        return getattr(obj, field)

    return default


def _to_1d_float(a):
    """Convert lat/lon vectors to 1D float array."""
    a = _unwrap_mat_scalar(a)
    if a is None:
        return None
    a = np.array(a)
    a = np.squeeze(a)
    if a.ndim == 0:
        a = a.reshape(1)
    return a.astype(float).ravel()


def load_is2_tracks(is2_mat, beams=None, target_date=None, debug=False):
    """
    Load ICESat-2 tracks from a MATLAB .mat file that contains:
        beam_data.gt1l.lat, beam_data.gt1l.lon, ... (and similarly for gt1r, gt2l, gt2r, gt3l, gt3r)

    Parameters
    ----------
    is2_mat : str
        Path to .mat file
    beams : list[str] or str or None
        Which beams to load. If None -> load all 6: gt1l,gt1r,gt2l,gt2r,gt3l,gt3r
        If str -> comma-separated list.
    target_date : datetime or None
        Currently not used (file typically already matched), kept for compatibility.
    debug : bool

    Returns
    -------
    dict:
      {
        "beams": [ ... ],
        "tracks": { beam: {"lat": 1D array, "lon": 1D array}, ... }
      }
    """
    default_beams = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

    # normalize beams input
    if beams is None:
        beams = default_beams
    elif isinstance(beams, str):
        beams = [b.strip() for b in beams.split(",") if b.strip()]
    else:
        beams = [str(b).strip() for b in beams if str(b).strip()]

    tracks = {}
    used = []

    # ----------------------------
    # Case A: MAT v7.3 via h5py
    # ----------------------------
    if HAS_H5PY:
        try:
            with h5py.File(is2_mat, "r") as f:
                if "beam_data" in f:
                    g = f["beam_data"]
                    for b in beams:
                        if b not in g:
                            continue
                        gb = g[b]
                        if ("lat" not in gb) or ("lon" not in gb):
                            continue
                        lat = np.array(gb["lat"]).squeeze()
                        lon = np.array(gb["lon"]).squeeze()

                        # Ensure 1D
                        lat = np.ravel(lat).astype(float)
                        lon = np.ravel(lon).astype(float)

                        m = np.isfinite(lat) & np.isfinite(lon)
                        lat = lat[m]
                        lon = lon[m]

                        if lat.size == 0:
                            continue

                        tracks[b] = {"lat": lat, "lon": lon}
                        used.append(b)

                    if used:
                        if debug:
                            print(f"[IS2] Loaded beams (h5py): {used}")
                        return {"beams": used, "tracks": tracks}
        except Exception as e:
            if debug:
                print(f"[IS2] h5py read failed (maybe not v7.3): {e}")

    # ------------------------------------
    # Case B: MAT v7.2 via scipy.io.loadmat
    # ------------------------------------
    if not HAS_SCIPY:
        raise RuntimeError("Need scipy or h5py to read ICESat-2 MAT file.")

    md = loadmat(is2_mat, squeeze_me=True, struct_as_record=False)

    if "beam_data" not in md:
        raise KeyError("IS2 MAT file does not contain 'beam_data' variable.")

    beam_data = md["beam_data"]

    def _get_field(obj, name):
        """Get MATLAB struct field robustly across scipy loadmat outputs."""
        if obj is None:
            return None
        # mat_struct style
        if hasattr(obj, name):
            return getattr(obj, name)
        # numpy void / structured
        try:
            if isinstance(obj, np.void) and name in obj.dtype.names:
                return obj[name]
        except Exception:
            pass
        # object array holding struct
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            # common shapes: (1,1) or scalar squeeze
            try:
                oo = obj.item()
                return _get_field(oo, name)
            except Exception:
                pass
        return None

    for b in beams:
        bd = _get_field(beam_data, b)
        if bd is None:
            continue
        lat = _get_field(bd, "lat")
        lon = _get_field(bd, "lon")
        if lat is None or lon is None:
            continue

        lat = np.ravel(np.array(lat, dtype=float))
        lon = np.ravel(np.array(lon, dtype=float))

        m = np.isfinite(lat) & np.isfinite(lon)
        lat = lat[m]
        lon = lon[m]
        if lat.size == 0:
            continue

        tracks[b] = {"lat": lat, "lon": lon}
        used.append(b)

    if debug:
        print(f"[IS2] Loaded beams (scipy): {used}")

    return {"beams": used, "tracks": tracks}


def backtrack_is2_to_swot(
    is2_lat, is2_lon, is2_time,
    ice_motion_nc, swot_start_time, swot_end_time,
    dt_hours=1.0, debug=False
):
    """
    Backtrack IS2 locations using ice drift to find where ice was during SWOT observation window.
    
    Parameters
    ----------
    is2_lat : array
        IS2 latitude points (degrees)
    is2_lon : array
        IS2 longitude points (degrees, 0-360 or -180-180)
    is2_time : datetime
        IS2 observation time (assumed same for all points, or can be array)
    ice_motion_nc : str
        Path to ice motion NetCDF file
    swot_start_time : datetime
        Start of SWOT observation window
    swot_end_time : datetime
        End of SWOT observation window
    dt_hours : float
        Time step for backtracking integration (hours, default: 1.0)
    debug : bool
        Print debug information
    
    Returns
    -------
    dict:
        {
            "lat_start": array,  # Backtracked locations at SWOT start time
            "lon_start": array,
            "lat_end": array,    # Backtracked locations at SWOT end time
            "lon_end": array,
            "trajectories": list, # List of (lat, lon) arrays for each point's trajectory
            "valid": array       # Boolean mask of valid backtracked points
        }
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is required for backtracking")
    
    try:
        from scipy.interpolate import griddata
    except ImportError:
        raise RuntimeError("scipy.interpolate is required for backtracking")
    
    # Normalize inputs
    is2_lat = np.asarray(is2_lat, dtype=float)
    is2_lon = np.asarray(is2_lon, dtype=float)
    is2_lon0360 = normalize_longitude_0_360(is2_lon)
    
    n_points = is2_lat.size
    if debug:
        print(f"[BACKTRACK] Starting backtracking for {n_points} IS2 points")
        print(f"[BACKTRACK] IS2 time: {is2_time}")
        print(f"[BACKTRACK] SWOT window: {swot_start_time} to {swot_end_time}")
    
    # Normalize IS2 time to datetime
    if isinstance(is2_time, datetime):
        t_is2 = is2_time
    else:
        # If not datetime, assume end of day
        t_is2 = datetime(is2_time.year, is2_time.month, is2_time.day, 23, 59, 59)
    
    # Calculate time range needed
    time_start = min(swot_start_time, swot_end_time)
    time_end = max(t_is2, swot_start_time, swot_end_time)
    
    # Generate time steps for backtracking (backwards from IS2 time to SWOT times)
    t_current = t_is2
    
    # Create time steps backwards
    time_steps = []
    t = t_current
    while t >= time_start:
        time_steps.append(t)
        t -= timedelta(hours=dt_hours)
    time_steps.reverse()  # Now in forward chronological order
    
    if debug:
        print(f"[BACKTRACK] Using {len(time_steps)} time steps for integration")
    
    # Initialize backtracked positions (start from IS2 locations)
    lat_back = is2_lat.copy()
    lon_back = is2_lon0360.copy()
    
    # Store trajectories
    trajectories = []
    for i in range(n_points):
        trajectories.append([(is2_lat[i], is2_lon0360[i])])
    
    # Load ice motion data for each time step and integrate backwards
    nc = netCDF4.Dataset(ice_motion_nc, "r")
    
    # Get time variable info
    time_var = nc.variables["time"]
    time_units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    time_vals = np.array(time_var[:], dtype=float)
    
    # Get lat/lon grid
    if "latitude" in nc.variables:
        lat_grid = np.array(nc.variables["latitude"][:], dtype=float)
    elif "lat" in nc.variables:
        lat_grid = np.array(nc.variables["lat"][:], dtype=float)
    else:
        nc.close()
        raise RuntimeError("Ice motion file must contain latitude")
    
    if "longitude" in nc.variables:
        lon_grid = np.array(nc.variables["longitude"][:], dtype=float)
    elif "lon" in nc.variables:
        lon_grid = np.array(nc.variables["lon"][:], dtype=float)
    else:
        nc.close()
        raise RuntimeError("Ice motion file must contain longitude")
    
    # Make 2D if needed
    if lat_grid.ndim == 1 and lon_grid.ndim == 1:
        lon_grid_2d, lat_grid_2d = np.meshgrid(lon_grid, lat_grid)
    else:
        lon_grid_2d = lon_grid
        lat_grid_2d = lat_grid
    
    lon_grid_2d_0360 = normalize_longitude_0_360(lon_grid_2d)
    
    # Flatten grid for interpolation
    lat_grid_flat = lat_grid_2d.ravel()
    lon_grid_flat_0360 = lon_grid_2d_0360.ravel()
    
    # Track valid points
    valid = np.ones(n_points, dtype=bool)
    
    # Integrate backwards through time steps
    for step_idx, t_step in enumerate(time_steps):
        if step_idx == 0:
            continue  # Skip first step (IS2 time)
        
        # Find closest time in ice motion file
        target_num = netCDF4.date2num(t_step, units=time_units, calendar=calendar)
        time_idx = int(np.nanargmin(np.abs(time_vals - target_num)))
        actual_time = netCDF4.num2date(time_vals[time_idx], units=time_units, calendar=calendar)
        
        if debug and step_idx % 10 == 0:
            print(f"[BACKTRACK] Step {step_idx}/{len(time_steps)-1}: {t_step} (using ice motion from {actual_time})")
        
        # Load u/v for this time
        u_var = nc.variables["u"]
        v_var = nc.variables["v"]
        u = np.array(u_var[time_idx, :, :], dtype=float)
        v = np.array(v_var[time_idx, :, :], dtype=float)
        
        # Handle fill values
        u_fill = getattr(u_var, "_FillValue", None)
        v_fill = getattr(v_var, "_FillValue", None)
        if u_fill is not None:
            u[u == float(u_fill)] = np.nan
        if v_fill is not None:
            v[v == float(v_fill)] = np.nan
        
        # Convert to m/s
        fu = _units_to_mps(getattr(u_var, "units", ""))
        fv = _units_to_mps(getattr(v_var, "units", ""))
        u_ms = u * fu
        v_ms = v * fv
        
        # Transform to geographic if needed
        u_name = (getattr(u_var, "standard_name", "") + " " + getattr(u_var, "long_name", "")).lower()
        v_name = (getattr(v_var, "standard_name", "") + " " + getattr(v_var, "long_name", "")).lower()
        already_geo = (("east" in u_name) or ("zonal" in u_name)) and (("north" in v_name) or ("merid" in v_name))
        
        if not already_geo:
            u_ms, v_ms = transform_grid_to_geographic(u_ms, v_ms, lon_deg=lon_grid_2d_0360)
        
        # Flatten velocity fields
        u_flat = u_ms.ravel()
        v_flat = v_ms.ravel()
        
        # Interpolate velocities at current backtracked positions
        u_interp = griddata(
            (lon_grid_flat_0360, lat_grid_flat),
            u_flat,
            (lon_back, lat_back),
            method='linear',
            fill_value=np.nan
        )
        v_interp = griddata(
            (lon_grid_flat_0360, lat_grid_flat),
            v_flat,
            (lon_back, lat_back),
            method='linear',
            fill_value=np.nan
        )
        
        # Update valid mask
        valid = valid & np.isfinite(u_interp) & np.isfinite(v_interp) & np.isfinite(lat_back) & np.isfinite(lon_back)
        
        # Calculate time step (backwards, so negative)
        dt_sec = -dt_hours * 3600.0  # Negative for backwards integration
        
        # Update positions (backwards integration)
        # Convert lat/lon displacement to degrees
        # Approximate: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
        lat_back[valid] += np.rad2deg(v_interp[valid] * dt_sec / 111000.0)
        lon_back[valid] += np.rad2deg(u_interp[valid] * dt_sec / (111000.0 * np.cos(np.deg2rad(lat_back[valid]))))
        
        # Normalize longitude
        lon_back = normalize_longitude_0_360(lon_back)
        
        # Store trajectory
        for i in range(n_points):
            if valid[i]:
                trajectories[i].append((lat_back[i], lon_back[i]))
    
    nc.close()
    
    # Extract positions at SWOT start and end times
    # Find indices in trajectories closest to SWOT times
    lat_start = np.full(n_points, np.nan)
    lon_start = np.full(n_points, np.nan)
    lat_end = np.full(n_points, np.nan)
    lon_end = np.full(n_points, np.nan)
    
    # For now, use the last positions (earliest time) for start and positions at middle for end
    # This is approximate - could be refined with time interpolation
    for i in range(n_points):
        if valid[i] and len(trajectories[i]) > 0:
            # Start time: first position in trajectory (earliest time)
            lat_start[i], lon_start[i] = trajectories[i][0]
            # End time: last position before IS2 (or middle of trajectory)
            if len(trajectories[i]) > 1:
                mid_idx = len(trajectories[i]) // 2
                lat_end[i], lon_end[i] = trajectories[i][mid_idx]
            else:
                lat_end[i], lon_end[i] = trajectories[i][0]
    
    if debug:
        n_valid = np.sum(valid)
        print(f"[BACKTRACK] Completed: {n_valid}/{n_points} points successfully backtracked")
    
    return {
        "lat_start": lat_start,
        "lon_start": lon_start,
        "lat_end": lat_end,
        "lon_end": lon_end,
        "trajectories": trajectories,
        "valid": valid
    }


# ---------------------------
# Longitude helpers
# ---------------------------

def normalize_longitude_0_360(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return (lon + 360.0) % 360.0


def wrap_lon180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


# ---------------------------
# Small-angle displacement on sphere
# ---------------------------

def displace_lonlat(lon_deg, lat_deg, u_east_ms, v_north_ms, dt_seconds):
    """
    Move (lon,lat) by (u,v) over dt using small-angle spherical approximation.
    lon/lat in degrees, u/v in m/s.
    """
    R = 6371000.0
    lat_rad = np.deg2rad(lat_deg)

    dlat = (v_north_ms * dt_seconds) / R
    dlon = (u_east_ms * dt_seconds) / (R * np.cos(lat_rad))

    lat2 = lat_deg + np.rad2deg(dlat)
    lon2 = lon_deg + np.rad2deg(dlon)
    lon2 = wrap_lon180(lon2)
    return lon2, lat2


# ---------------------------
# Ross Sea constrain (scatter)
# ---------------------------

def constrain_scatter_to_ross_sea(lon0360, lat, sit, lon_range=(165, 190), lat_range=(-78, -74)):
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    m = (
        np.isfinite(lon0360) & np.isfinite(lat) & np.isfinite(sit) &
        (lon0360 >= lon_min) & (lon0360 <= lon_max) &
        (lat >= lat_min) & (lat <= lat_max)
    )
    return lon0360[m], lat[m], sit[m]


# ---------------------------
# Polynya detection (MATLAB-like)
# ---------------------------

def flood_fill(img, x, y, value):
    from collections import deque
    img = np.array(img, copy=True)
    dims = img.shape
    if y < 0 or y >= dims[0] or x < 0 or x >= dims[1]:
        return img
    initial = img[y, x]
    if value == initial:
        return img
    q = deque()
    q.append((y, x))
    img[y, x] = value
    while q:
        y_pt, x_pt = q.popleft()
        if y_pt < dims[0] - 1 and img[y_pt + 1, x_pt] == initial:
            img[y_pt + 1, x_pt] = value
            q.append((y_pt + 1, x_pt))
        if y_pt > 0 and img[y_pt - 1, x_pt] == initial:
            img[y_pt - 1, x_pt] = value
            q.append((y_pt - 1, x_pt))
        if x_pt < dims[1] - 1 and img[y_pt, x_pt + 1] == initial:
            img[y_pt, x_pt + 1] = value
            q.append((y_pt, x_pt + 1))
        if x_pt > 0 and img[y_pt, x_pt - 1] == initial:
            img[y_pt, x_pt - 1] = value
            q.append((y_pt, x_pt - 1))
    return img


def polynya_output_amsr2(sic_frac, threshold, coastal_buffer_pixels=2, debug=False):
    """
    Robust coastal/open polynya split:
      candidate = (0 <= SIC <= 1) & (SIC < threshold)
      land      = invalid SIC (NaN or outside [0,1.2])
      coastal   = candidate components connected to land-adjacent candidate pixels
      open      = candidate but not coastal

    Returns:
      open_polynya, coastal_polynya  (arrays with NaN outside)
    """
    sic = np.array(sic_frac, dtype=float, copy=True)

    valid = np.isfinite(sic) & (sic >= 0.0) & (sic <= 1.2)
    land = ~valid

    candidate = valid & (sic < float(threshold))

    # Nothing to do
    if not np.any(candidate):
        if debug:
            print("[POLYNYA] No candidate pixels (SIC<threshold) found.")
        nan = np.full_like(sic, np.nan)
        return nan, nan

    # Build a "coastal seed" = candidate pixels adjacent to land (optionally buffered)
    try:
        # Prefer scipy if available for dilation + labeling (fast and clean)
        from scipy.ndimage import binary_dilation, label

        land_buf = binary_dilation(land, iterations=int(max(1, coastal_buffer_pixels)))
        coastal_seed = candidate & land_buf

        lbl, nlab = label(candidate)
        if nlab == 0:
            nan = np.full_like(sic, np.nan)
            return nan, nan

        seed_labels = np.unique(lbl[coastal_seed])
        seed_labels = seed_labels[seed_labels > 0]

        coastal_mask = np.isin(lbl, seed_labels) if seed_labels.size else np.zeros_like(candidate, bool)

    except Exception:
        # No scipy: approximate buffering using neighbor expansion, then BFS grow
        land_buf = land.copy()
        for _ in range(int(max(1, coastal_buffer_pixels))):
            land_buf = land_buf | _neighbors4_mask(land_buf)

        coastal_seed = candidate & _neighbors4_mask(land_buf)
        coastal_mask = _bfs_grow(candidate, coastal_seed)

    open_mask = candidate & ~coastal_mask

    open_pol = np.full_like(sic, np.nan)
    coastal_pol = np.full_like(sic, np.nan)
    open_pol[open_mask] = sic[open_mask]
    coastal_pol[coastal_mask] = sic[coastal_mask]

    if debug:
        print(f"[POLYNYA] candidate={int(candidate.sum())} coastal={int(coastal_mask.sum())} open={int(open_mask.sum())}")

    return open_pol, coastal_pol



def outline_mask(mask, closing_iters=1):
    try:
        from scipy.ndimage import binary_closing, binary_erosion
        m = mask.astype(bool)
        if closing_iters and closing_iters > 0:
            m = binary_closing(m, iterations=int(closing_iters))
        er = binary_erosion(m, iterations=1)
        return m & ~er
    except Exception:
        m = mask.astype(bool)
        er = m.copy()
        if m.shape[0] > 2 and m.shape[1] > 2:
            er[1:-1, 1:-1] = (
                m[1:-1, 1:-1] &
                m[:-2, 1:-1] & m[2:, 1:-1] &
                m[1:-1, :-2] & m[1:-1, 2:]
            )
        return m & ~er


# ---------------------------
# Map extent helper
# ---------------------------

def compute_projection_extent(lon0360, lat, proj, pc, buffer_factor=1.2, min_span_km=None):
    valid = np.isfinite(lon0360) & np.isfinite(lat)
    if not np.any(valid):
        return None

    lon = lon0360[valid]
    lat = lat[valid]
    lon180 = np.where(lon > 180, lon - 360, lon)

    pts = proj.transform_points(pc, lon180, lat)
    x = pts[:, 0]
    y = pts[:, 1]
    v2 = np.isfinite(x) & np.isfinite(y)
    if not np.any(v2):
        return None
    x = x[v2]
    y = y[v2]

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    x_range = float(np.ptp(x))
    y_range = float(np.ptp(y))

    # --- key change: enforce square span so the map fills the axes ---
    span = max(x_range, y_range)
    if min_span_km is not None:
        span = max(span, float(min_span_km) * 1000.0)

    span *= float(buffer_factor)

    x_min = x_mean - span / 2.0
    x_max = x_mean + span / 2.0
    y_min = y_mean - span / 2.0
    y_max = y_mean + span / 2.0

    return (x_min, x_max, y_min, y_max)


# ---------------------------
# ESACCI reader
# ---------------------------

def read_esacci_sit_nc(nc_path, filter_quality=True, max_uncertainty=None):
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available; cannot read ESACCI files.")

    nc = netCDF4.Dataset(nc_path, "r")
    available_vars = list(nc.variables.keys())

    sit_var = None
    candidates = ["sea_ice_thickness", "sit", "SIT", "thickness", "ice_thickness"]
    for cand in candidates:
        if cand in nc.variables:
            sit_var = cand
            break
    if sit_var is None:
        for vn in available_vars:
            vl = vn.lower()
            if "thickness" in vl or vl == "sit":
                sit_var = vn
                break
    if sit_var is None:
        nc.close()
        raise KeyError(f"Could not find SIT variable in {nc_path}. Available: {available_vars}")

    sit_obj = nc.variables[sit_var]
    sit_raw = sit_obj[:]
    sit = sit_raw.filled(np.nan).astype(float) if np.ma.isMaskedArray(sit_raw) else np.array(sit_raw, dtype=float)

    fv = getattr(sit_obj, "_FillValue", None)
    if fv is not None:
        sit[sit == float(fv)] = np.nan

    lat = None
    lon = None
    for lat_name in ["latitude", "lat", "Latitude", "Lat"]:
        if lat_name in nc.variables:
            tmp = nc.variables[lat_name][:]
            lat = tmp.filled(np.nan).astype(float) if np.ma.isMaskedArray(tmp) else np.array(tmp, dtype=float)
            break
    for lon_name in ["longitude", "lon", "Longitude", "Lon"]:
        if lon_name in nc.variables:
            tmp = nc.variables[lon_name][:]
            lon = tmp.filled(np.nan).astype(float) if np.ma.isMaskedArray(tmp) else np.array(tmp, dtype=float)
            break
    if lat is None or lon is None:
        nc.close()
        raise RuntimeError(f"Could not find lat/lon in {nc_path}")

    # Quality filter
    valid_mask = None
    if filter_quality and "flag_miz" in nc.variables:
        flag_miz = np.array(nc.variables["flag_miz"][:], dtype=int)
        valid_mask = (flag_miz != 2)
        sit = sit[valid_mask]
        lat = lat[valid_mask]
        lon = lon[valid_mask]

    # Uncertainty filter
    if max_uncertainty is not None:
        unc_var = None
        for unc_name in ["sea_ice_thickness_uncertainty", "sit_uncertainty", "thickness_uncertainty"]:
            if unc_name in nc.variables:
                unc_var = unc_name
                break
        if unc_var is not None:
            unc_raw = nc.variables[unc_var][:]
            unc = unc_raw.filled(np.nan).astype(float) if np.ma.isMaskedArray(unc_raw) else np.array(unc_raw, dtype=float)
            if valid_mask is not None:
                unc = unc[valid_mask]
            ok = np.isfinite(unc) & (unc <= max_uncertainty)
            sit = sit[ok]
            lat = lat[ok]
            lon = lon[ok]

    nc.close()

    sit = np.array(sit, dtype=float).ravel()
    lat = np.array(lat, dtype=float).ravel()
    lon = np.array(lon, dtype=float).ravel()

    n = min(len(sit), len(lat), len(lon))
    return sit[:n], lat[:n], lon[:n], None


# ---------------------------
# Ice motion loader (robust)
# ---------------------------

def _units_to_mps(units):
    """Convert common velocity units to m/s."""
    if not units:
        return 0.01  # assume cm/s
    u = units.strip().lower().replace(" ", "")
    if "cm/s" in u or "cms-1" in u:
        return 0.01
    if "m/s" in u or "ms-1" in u:
        return 1.0
    if "km/day" in u:
        return 1000.0 / 86400.0
    return 0.01


def load_ice_motion(ice_motion_nc, target_date, subsample=5, debug=False):
    """
    Load ice motion for target_date.

    Returns:
      lat2D (deg), lon2D (0..360), u_east_ms, v_north_ms, valid_mask

    Notes:
    - Many products store u/v in grid-x/grid-y. If metadata suggests east/north, use directly.
      Otherwise apply a simple longitude-based rotation using transform_grid_to_geographic().
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available.")

    nc = netCDF4.Dataset(ice_motion_nc, "r")

    # --- time handling ---
    if "time" not in nc.variables:
        nc.close()
        raise RuntimeError("Ice motion file has no 'time' variable.")
    time_var = nc.variables["time"]
    time_units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    if time_units is None:
        nc.close()
        raise RuntimeError("Ice motion time variable has no 'units'.")

    time_vals = np.array(time_var[:], dtype=float)
    target_num = netCDF4.date2num(target_date, units=time_units, calendar=calendar)
    time_idx = int(np.nanargmin(np.abs(time_vals - target_num)))

    if debug:
        actual_time = netCDF4.num2date(time_vals[time_idx], units=time_units, calendar=calendar)
        print(f"[ICE_MOTION] Target date: {target_date:%Y-%m-%d}")
        print(f"[ICE_MOTION] Closest time index: {time_idx}, date: {actual_time}")

    # --- u/v ---
    if "u" not in nc.variables or "v" not in nc.variables:
        nc.close()
        raise RuntimeError("Ice motion file must contain 'u' and 'v'.")

    u_var = nc.variables["u"]
    v_var = nc.variables["v"]

    u = np.array(u_var[time_idx, :, :], dtype=float)
    v = np.array(v_var[time_idx, :, :], dtype=float)

    # Fill values
    u_fill = getattr(u_var, "_FillValue", None)
    v_fill = getattr(v_var, "_FillValue", None)
    if u_fill is not None:
        u[u == float(u_fill)] = np.nan
    if v_fill is not None:
        v[v == float(v_fill)] = np.nan

    # --- lat/lon ---
    if "latitude" in nc.variables:
        lat = np.array(nc.variables["latitude"][:], dtype=float)
    elif "lat" in nc.variables:
        lat = np.array(nc.variables["lat"][:], dtype=float)
    else:
        nc.close()
        raise RuntimeError("Ice motion file must contain latitude (latitude/lat).")

    if "longitude" in nc.variables:
        lon = np.array(nc.variables["longitude"][:], dtype=float)
    elif "lon" in nc.variables:
        lon = np.array(nc.variables["lon"][:], dtype=float)
    else:
        nc.close()
        raise RuntimeError("Ice motion file must contain longitude (longitude/lon).")

    nc.close()

    # Make lat/lon 2D if they are 1D
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    lon0360 = normalize_longitude_0_360(lon)

    # --- convert units to m/s ---
    fu = _units_to_mps(getattr(u_var, "units", ""))
    fv = _units_to_mps(getattr(v_var, "units", ""))
    u_ms = u * fu
    v_ms = v * fv

    # --- decide whether already geographic (east/north) ---
    # For EASE-Grid South ice motion data, u/v are ALWAYS grid-aligned (along-x, along-y)
    # The metadata may say "x_velocity" and "y_velocity" but these are grid coordinates, not geographic
    # So we should ALWAYS apply the transformation for EASE-Grid data
    u_name = (getattr(u_var, "standard_name", "") + " " + getattr(u_var, "long_name", "")).lower()
    v_name = (getattr(v_var, "standard_name", "") + " " + getattr(v_var, "long_name", "")).lower()
    
    # Check if this is EASE-Grid data (which always needs transformation)
    is_ease_grid = ("ease" in str(getattr(nc, "crs", "")).lower() or 
                    "lambert_azimuthal" in str(getattr(nc.variables.get("crs", ""), "grid_mapping_name", "")).lower() or
                    "x_velocity" in u_name or "y_velocity" in v_name or
                    "along-x" in u_name or "along-y" in v_name)
    
    # Only skip transformation if explicitly geographic AND not EASE-Grid
    already_geo = (("east" in u_name) or ("zonal" in u_name)) and (("north" in v_name) or ("merid" in v_name)) and not is_ease_grid

    if already_geo:
        u_east_ms = u_ms
        v_north_ms = v_ms
        if debug:
            print("[ICE_MOTION] u/v appear to be geographic (east/north); skipping grid->geo rotation.")
    else:
        # Rotate from grid-aligned to geographic using lon (degrees).
        # lon0360 is fine; trig only needs radians.
        # For EASE-Grid South, this transformation is always needed
        u_east_ms, v_north_ms = transform_grid_to_geographic(u_ms, v_ms, lon_deg=lon0360)
        if debug:
            print(f"[ICE_MOTION] u/v treated as grid-aligned; applied transform_grid_to_geographic().")
            print(f"[ICE_MOTION] u_name='{u_name}', v_name='{v_name}', is_ease_grid={is_ease_grid}")
            # Sample a few transformed values for verification
            if np.any(np.isfinite(u_east_ms)) and np.any(np.isfinite(v_north_ms)):
                sample_idx = np.where(np.isfinite(u_east_ms) & np.isfinite(v_north_ms))[0][:3]
                for idx in sample_idx:
                    i, j = np.unravel_index(idx, u_east_ms.shape)
                    print(f"[ICE_MOTION] Sample [{i},{j}]: lon={lon0360[i,j]:.2f}°, "
                          f"u_grid={u_ms[i,j]*100:.2f} cm/s, v_grid={v_ms[i,j]*100:.2f} cm/s, "
                          f"u_east={u_east_ms[i,j]*100:.2f} cm/s, v_north={v_north_ms[i,j]*100:.2f} cm/s")

    # --- subsample (IMPORTANT: use the correct variable names) ---
    if subsample and subsample > 1:
        lat = lat[::subsample, ::subsample]
        lon0360 = lon0360[::subsample, ::subsample]
        u_east_ms = u_east_ms[::subsample, ::subsample]
        v_north_ms = v_north_ms[::subsample, ::subsample]

    # --- valid mask ---
    valid = (
        np.isfinite(lat) & np.isfinite(lon0360) &
        np.isfinite(u_east_ms) & np.isfinite(v_north_ms)
    )

    # Filter small speeds (avoid noise)
    speed = np.sqrt(u_east_ms**2 + v_north_ms**2)
    valid = valid & (speed > 0.001)  # > 0.5 cm/s

    if debug:
        nv = int(np.sum(valid))
        print(f"[ICE_MOTION] Valid vectors: {nv} (subsample={subsample})")
        if nv > 0:
            sp = speed[valid]
            print(f"[ICE_MOTION] Speed (m/s): min={np.nanmin(sp):.3f} med={np.nanmedian(sp):.3f} max={np.nanmax(sp):.3f}")

    return lat, lon0360, u_east_ms, v_north_ms, valid


# ---------------------------
# Tracking positions loader
# ---------------------------

def load_tracking_positions(npz_file, debug=False):
    """
    Load start and end positions from ice tracking NPZ file.
    
    Parameters
    ----------
    npz_file : str
        Path to NPZ file containing tracking results from ice_tracking.py
    debug : bool
        Print debug information
    
    Returns
    -------
    dict:
        {
            'lat_start': array,  # Start positions (at tracking start time)
            'lon_start': array,
            'lat_final': array,  # Final positions (at tracking end time)
            'lon_final': array,
            'valid': array,      # Boolean mask of valid points
            'lat_init': array,   # Initial positions (optional)
            'lon_init': array
        }
    """
    try:
        data = np.load(npz_file)
        
        # Extract positions
        lat_start = data.get('lat_start', None)
        lon_start = data.get('lon_start', None)
        lat_final = data.get('lat_final', None)
        lon_final = data.get('lon_final', None)
        valid = data.get('valid', None)
        lat_init = data.get('lat_init', None)
        lon_init = data.get('lon_init', None)
        
        if lat_start is None or lon_start is None or lat_final is None or lon_final is None:
            raise ValueError("NPZ file must contain lat_start, lon_start, lat_final, lon_final")
        
        # Convert to arrays
        lat_start = np.array(lat_start, dtype=float)
        lon_start = np.array(lon_start, dtype=float)
        lat_final = np.array(lat_final, dtype=float)
        lon_final = np.array(lon_final, dtype=float)
        
        if valid is not None:
            valid = np.array(valid, dtype=bool)
        else:
            # Create valid mask from finite values
            valid = np.isfinite(lat_start) & np.isfinite(lon_start) & \
                   np.isfinite(lat_final) & np.isfinite(lon_final)
        
        if debug:
            n_valid = np.sum(valid)
            print(f"[TRACKING] Loaded positions from {npz_file}")
            print(f"[TRACKING] Valid points: {n_valid}/{len(valid)}")
            if n_valid > 0:
                print(f"[TRACKING] Start positions: lon=[{np.nanmin(lon_start[valid]):.2f}, {np.nanmax(lon_start[valid]):.2f}], "
                      f"lat=[{np.nanmin(lat_start[valid]):.2f}, {np.nanmax(lat_start[valid]):.2f}]")
                print(f"[TRACKING] Final positions: lon=[{np.nanmin(lon_final[valid]):.2f}, {np.nanmax(lon_final[valid]):.2f}], "
                      f"lat=[{np.nanmin(lat_final[valid]):.2f}, {np.nanmax(lat_final[valid]):.2f}]")
        
        result = {
            'lat_start': lat_start,
            'lon_start': lon_start,
            'lat_final': lat_final,
            'lon_final': lon_final,
            'valid': valid
        }
        
        if lat_init is not None and lon_init is not None:
            result['lat_init'] = np.array(lat_init, dtype=float)
            result['lon_init'] = np.array(lon_init, dtype=float)
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Failed to load tracking positions from {npz_file}: {e}")


# ---------------------------
# BedMachine loader
# ---------------------------

def load_bedmachine_mask(
    bedmachine_path,
    lon_data,
    lat_data,
    subsample=5,
    debug=False,
    buffer_km=200.0,
):
    """
    Load BedMachine (NSIDC-0756 v4.x) masks and return lon/lat grids + masks.

    Works even when PROJ database (proj.db) is missing by using PROJ4 strings
    instead of EPSG:3031 / EPSG:4326.

    BedMachine mask codes (common for NSIDC-0756):
      0=ocean
      1=ice-free land
      2=grounded ice
      3=floating ice / ice shelf
      4=lake

    Returns dict:
      lon, lat : 2D arrays (deg, lon in [-180,180])
      ice_shelf_mask : bool 2D
      grounding_line_mask : bool 2D
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available.")

    try:
        import pyproj
        from pyproj import CRS, Transformer
    except Exception as e:
        raise RuntimeError(f"pyproj is required for BedMachine transform: {e}")

    # --- Open file ---
    nc = netCDF4.Dataset(bedmachine_path, "r")
    if debug:
        print(f"[BEDMACHINE] Loading from: {bedmachine_path}")
        print(f"[BEDMACHINE] Available variables: {list(nc.variables.keys())}")

    if "x" not in nc.variables or "y" not in nc.variables or "mask" not in nc.variables:
        nc.close()
        raise RuntimeError("BedMachine file must contain variables: x, y, mask")

    x_full = np.array(nc.variables["x"][:], dtype=float)  # 1D
    y_full = np.array(nc.variables["y"][:], dtype=float)  # 1D
    mask_var = nc.variables["mask"]

    # --- Build PROJ4 for BedMachine polar stereographic (EPSG:3031-like) without EPSG ---
    # Default params for Antarctic PS (as used by BedMachine/3031)
    lat_ts = -71.0
    lon0 = 0.0
    x0 = 0.0
    y0 = 0.0

    # If file has CF mapping variable, try to read parameters (no EPSG needed)
    if "mapping" in nc.variables:
        mp = nc.variables["mapping"]
        # CF names vary; try common ones
        lat_ts = float(getattr(mp, "standard_parallel", lat_ts))
        lon0 = float(
            getattr(mp, "straight_vertical_longitude_from_pole",
                    getattr(mp, "longitude_of_projection_origin", lon0))
        )
        x0 = float(getattr(mp, "false_easting", x0))
        y0 = float(getattr(mp, "false_northing", y0))

    # Avoid datum lookups (which may touch proj.db): specify ellipsoid explicitly
    a = 6378137.0
    rf = 298.257223563

    src_proj4 = (
        f"+proj=stere +lat_0=-90 +lat_ts={lat_ts}_toggle +lon_0={lon0} "
        f"+x_0={x0} +y_0={y0} +a={a} +rf={rf} +units=m +no_defs"
    )
    # (tiny hack: remove accidental token if present)
    src_proj4 = src_proj4.replace("_toggle", "")

    dst_proj4 = f"+proj=longlat +a={a} +rf={rf} +no_defs"

    try:
        crs_src = CRS.from_proj4(src_proj4)
        crs_dst = CRS.from_proj4(dst_proj4)
        to_ll = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
        to_xy = Transformer.from_crs(crs_dst, crs_src, always_xy=True)
    except Exception as e:
        nc.close()
        raise RuntimeError(
            f"[BEDMACHINE] Could not build PROJ transformers from PROJ4 strings: {e}\n"
            "If this still fails, you likely have a broken PROJ install.\n"
            "Conda fix: conda install -c conda-forge proj pyproj proj-data"
        )

    # --- Subset ROI in native x/y using your plotted data lon/lat ---
    # lon_data likely in [-180,180] already; lat_data in degrees
    buffer_m = float(buffer_km) * 1000.0

    # If lon_data/lat_data not provided, use full domain
    i0, i1 = 0, x_full.size
    j0, j1 = 0, y_full.size

    lon_in = np.asarray(lon_data, dtype=float) if lon_data is not None else None
    lat_in = np.asarray(lat_data, dtype=float) if lat_data is not None else None

    if lon_in is not None and lat_in is not None:
        vv = np.isfinite(lon_in) & np.isfinite(lat_in)
        if np.any(vv):
            # sample to reduce cost
            lon_s = lon_in[vv][::max(1, lon_in[vv].size // 20000)]
            lat_s = lat_in[vv][::max(1, lat_in[vv].size // 20000)]

            try:
                x_s, y_s = to_xy.transform(lon_s, lat_s)
                xmin, xmax = np.nanmin(x_s) - buffer_m, np.nanmax(x_s) + buffer_m
                ymin, ymax = np.nanmin(y_s) - buffer_m, np.nanmax(y_s) + buffer_m

                # x_full and y_full are 1D axes; find index ranges
                ix = np.where((x_full >= xmin) & (x_full <= xmax))[0]
                iy = np.where((y_full >= ymin) & (y_full <= ymax))[0]
                if ix.size and iy.size:
                    i0, i1 = int(ix.min()), int(ix.max()) + 1
                    j0, j1 = int(iy.min()), int(iy.max()) + 1

                if debug:
                    print(f"[BEDMACHINE] ROI indices x[{i0}:{i1}] y[{j0}:{j1}] (buffer {buffer_km:.0f} km)")
            except Exception as e:
                if debug:
                    print(f"[BEDMACHINE][WARN] ROI projection failed, using full domain: {e}")

    # --- Apply subsample and read small region only ---
    s = int(max(1, subsample))
    x = x_full[i0:i1:s]
    y = y_full[j0:j1:s]

    # Read mask on same slice/stride (netCDF slicing is efficient)
    mask = np.array(mask_var[j0:j1:s, i0:i1:s], dtype=np.int16)
    nc.close()

    if debug:
        u = np.unique(mask)
        print(f"[BEDMACHINE] mask unique values (ROI): {u}")

    # --- Transform x/y grid -> lon/lat (ROI only) ---
    X, Y = np.meshgrid(x, y)
    lon_bm, lat_bm = to_ll.transform(X, Y)
    lon_bm = wrap_lon180(np.array(lon_bm, dtype=float))
    lat_bm = np.array(lat_bm, dtype=float)

    # --- Build masks ---
    ice_shelf_mask = (mask == 3)
    grounded_mask = (mask == 2)

    # Grounding line: grounded boundary adjacent to shelf
    try:
        from scipy.ndimage import binary_erosion, binary_dilation

        grounded_edge = grounded_mask & ~binary_erosion(grounded_mask)
        shelf_near = binary_dilation(ice_shelf_mask, iterations=1)
        grounding_line_mask = grounded_edge & shelf_near
    except Exception:
        # no scipy: cheap edge + adjacency
        grounded_edge = grounded_mask & outline_mask(grounded_mask, closing_iters=0)
        shelf_near = _neighbors4_mask(ice_shelf_mask)
        grounding_line_mask = grounded_edge & shelf_near

    if debug:
        print(f"[BEDMACHINE] ice_shelf pixels: {int(ice_shelf_mask.sum())}")
        print(f"[BEDMACHINE] grounding_line pixels: {int(grounding_line_mask.sum())}")
        print(f"[BEDMACHINE] Lon range: [{np.nanmin(lon_bm):.1f}, {np.nanmax(lon_bm):.1f}]")
        print(f"[BEDMACHINE] Lat range: [{np.nanmin(lat_bm):.1f}, {np.nanmax(lat_bm):.1f}]")

    return {
        "lon": lon_bm,
        "lat": lat_bm,
        "ice_shelf_mask": ice_shelf_mask,
        "grounding_line_mask": grounding_line_mask,
    }


# ---------------------------
# Main plotting
# ---------------------------

def load_tracking_positions(npz_file, debug=False):
    """
    Load start and end positions from ice tracking NPZ file.
    
    Parameters
    ----------
    npz_file : str
        Path to NPZ file containing tracking results
    debug : bool
        Print debug information
    
    Returns
    -------
    dict:
        {
            'lat_start': array,  # Start positions (at tracking start time)
            'lon_start': array,
            'lat_final': array,  # Final positions (at tracking end time)
            'lon_final': array,
            'valid': array,      # Boolean mask of valid points
            'lat_init': array,   # Initial positions (optional)
            'lon_init': array
        }
    """
    try:
        data = np.load(npz_file)
        
        # Extract positions
        lat_start = data.get('lat_start', None)
        lon_start = data.get('lon_start', None)
        lat_final = data.get('lat_final', None)
        lon_final = data.get('lon_final', None)
        valid = data.get('valid', None)
        lat_init = data.get('lat_init', None)
        lon_init = data.get('lon_init', None)
        
        if lat_start is None or lon_start is None or lat_final is None or lon_final is None:
            raise ValueError("NPZ file must contain lat_start, lon_start, lat_final, lon_final")
        
        # Convert to arrays
        lat_start = np.array(lat_start, dtype=float)
        lon_start = np.array(lon_start, dtype=float)
        lat_final = np.array(lat_final, dtype=float)
        lon_final = np.array(lon_final, dtype=float)
        
        if valid is not None:
            valid = np.array(valid, dtype=bool)
        else:
            # Create valid mask from finite values
            valid = np.isfinite(lat_start) & np.isfinite(lon_start) & \
                   np.isfinite(lat_final) & np.isfinite(lon_final)
        
        if debug:
            n_valid = np.sum(valid)
            print(f"[TRACKING] Loaded positions from {npz_file}")
            print(f"[TRACKING] Valid points: {n_valid}/{len(valid)}")
            if n_valid > 0:
                print(f"[TRACKING] Start positions: lon=[{np.nanmin(lon_start[valid]):.2f}, {np.nanmax(lon_start[valid]):.2f}], "
                      f"lat=[{np.nanmin(lat_start[valid]):.2f}, {np.nanmax(lat_start[valid]):.2f}]")
                print(f"[TRACKING] Final positions: lon=[{np.nanmin(lon_final[valid]):.2f}, {np.nanmax(lon_final[valid]):.2f}], "
                      f"lat=[{np.nanmin(lat_final[valid]):.2f}, {np.nanmax(lat_final[valid]):.2f}]")
        
        result = {
            'lat_start': lat_start,
            'lon_start': lon_start,
            'lat_final': lat_final,
            'lon_final': lon_final,
            'valid': valid
        }
        
        if lat_init is not None and lon_init is not None:
            result['lat_init'] = np.array(lat_init, dtype=float)
            result['lon_init'] = np.array(lon_init, dtype=float)
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Failed to load tracking positions from {npz_file}: {e}")


def plot_sit_map(
    date_str,
    esacci_dir,
    amsr_mat,
    polynya_threshold=0.7,
    polynya_type="coastal",
    out_png="sit_map.png",
    vmin=0.0,
    vmax=3.0,
    central_lon=180.0,
    buffer_factor=1.2,
    global_view=False,
    filter_quality=True,
    max_uncertainty=None,
    ice_motion_nc=None,
    ice_motion_subsample=5,
    ice_motion_dt_hours=24.0,
    ice_motion_arrow_gain=1.0,
    constrain_ross_sea=False,
    bedmachine_nc=None,
    show_ice_shelves=False,
    show_grounding_line=False,
    bedmachine_subsample=5,
    # NEW:
    show_coastal_polynya_dots=True,
    polynya_dot_stride=2,      # plot every Nth dot to avoid overplotting
    polynya_dot_size=15,
    polynya_plot_edge_only=True,  # dotted outline (recommended)
    is2_mat=None,              # Path to IS2-SWOT collocation MATLAB file
    show_is2_tracks=False,     # Overlay IS2 tracks
    is2_track_color="grey",  # Color for IS2 tracks (if not using colormap)
    is2_track_alpha=0.01,       # Alpha for IS2 tracks
    is2_track_linewidth=0.5,   # Line width for IS2 tracks (if using lines)
    is2_use_scatter=True,      # Use scatter plot (circles) instead of lines
    is2_scatter_size=10,       # Size of scatter circles
    is2_colormap="k",     # Colormap for scatter (light to dark: "gray_r", "viridis_r", etc.)
    is2_color_by="height",     # Color by: "height", "index", or None (single color)
    backtrack_is2=False,  # Backtrack IS2 locations using ice drift
    swot_start_time_str=None,  # SWOT observation start time (e.g., "20231028T191642")
    swot_end_time_str=None,    # SWOT observation end time (e.g., "20231028T215103")
    backtrack_dt_hours=1.0,    # Time step for backtracking integration (hours)
    tracking_npz=None,         # Path to NPZ file with tracking positions (from ice_tracking.py)
    show_tracking_start=True,  # Show start positions from tracking
    show_tracking_end=True,     # Show end positions from tracking
    tracking_start_color="lime",  # Color for start positions (brighter green)
    tracking_end_color="red",     # Color for end positions
    tracking_marker_size=12,    # Size of tracking position markers (increased for visibility)
    tracking_alpha=0.1,         # Alpha for tracking positions (more opaque)
    tracking_subsample=5,      # Subsample tracking points (plot every Nth point, default: 1 = all points)
    debug=False,
):
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is not available.")
    if not (HAS_SCIPY or HAS_H5PY):
        raise RuntimeError("Need scipy (loadmat) or h5py (v7.3 MAT) to read AMSR.")

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        raise ImportError("cartopy is required. Install: conda install cartopy")

    # Parse date
    target_date = datetime.strptime(date_str, "%Y%m%d")
    target_date_minus_1 = target_date - timedelta(days=1)
    # Find ESACCI files
    patterns = [
        f"**/ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{date_str}-*.nc",
        f"**/ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3A-SH-{date_str}-*.nc",
        f"**/ESACCI-SEAICE-L2P-SITHICK-SRAL_SENTINEL3B-SH-{date_str}-*.nc",
    ]

    esacci_files = []
    for pat in patterns:
        esacci_files += glob.glob(os.path.join(esacci_dir, pat), recursive=True)

    if not esacci_files:
        raise RuntimeError(f"No ESACCI files found for date {date_str} in {esacci_dir}")

    if debug:
        print(f"[MAP] Found {len(esacci_files)} ESACCI files")

    # Load SIT
    all_sit, all_lat, all_lon = [], [], []
    for fpath in esacci_files:
        try:
            sit, lat, lon, _ = read_esacci_sit_nc(
                fpath,
                filter_quality=filter_quality,
                max_uncertainty=max_uncertainty,
            )
            all_sit.append(sit)
            all_lat.append(lat)
            all_lon.append(lon)
        except Exception as e:
            if debug:
                print(f"[WARN] Failed reading {fpath}: {e}")
            continue

    if not all_sit:
        raise RuntimeError("No valid ESACCI data found after reading all files.")

    sit = np.concatenate([a.ravel() for a in all_sit])
    lat = np.concatenate([a.ravel() for a in all_lat])
    lon = np.concatenate([a.ravel() for a in all_lon])
    lon0360 = normalize_longitude_0_360(lon)

    ok = np.isfinite(sit) & np.isfinite(lat) & np.isfinite(lon0360)
    sit = sit[ok]
    lat = lat[ok]
    lon0360 = lon0360[ok]

    if constrain_ross_sea:
        lon0360, lat, sit = constrain_scatter_to_ross_sea(lon0360, lat, sit)

    if sit.size == 0:
        raise RuntimeError("No valid SIT points after filtering / region constraint.")

    sit_mean = float(np.nanmean(sit))
    sit_std = float(np.nanstd(sit))

    if debug:
        print(f"[MAP] SIT mean={sit_mean:.3f} std={sit_std:.3f} n={sit.size}")

    # AMSR load
    amsr = load_mat_any(amsr_mat)
    if "sic" not in amsr or "lat" not in amsr or "lon" not in amsr:
        raise KeyError("AMSR MAT must contain: sic, lat, lon")

    sic = np.array(amsr["sic"], dtype=float)
    lat_amsr = np.array(amsr["lat"], dtype=float)
    lon_amsr = np.array(amsr["lon"], dtype=float)

    # Ensure sic time dimension first
    if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
        sic = np.moveaxis(sic, -1, 0)

    # Fix lat/lon orientation if needed
    if lat_amsr.shape != sic.shape[1:]:
        if lat_amsr.T.shape == sic.shape[1:]:
            lat_amsr = lat_amsr.T
            lon_amsr = lon_amsr.T
    
    # If AMSR lat/lon are 1D vectors, expand to 2D to match SIC grid
    if lat_amsr.ndim == 1 and lon_amsr.ndim == 1 and sic.ndim == 2:
        lon_amsr, lat_amsr = np.meshgrid(lon_amsr, lat_amsr)

    lon_amsr = normalize_longitude_0_360(lon_amsr)

    # Day index
    doy = target_date.timetuple().tm_yday
    doy = min(doy, sic.shape[0])
    day_idx = doy - 1
    sic_daily = sic[day_idx, :, :] if day_idx < sic.shape[0] else sic[-1, :, :]

    # Fraction (0..1)
    sic_frac = sic_daily / 100.0 if np.nanmax(sic_daily) > 1.0 else sic_daily.copy()

    # Polynya detection
    try:
        open_polynya, coastal_polynya = polynya_output_amsr2(sic_frac, polynya_threshold)
        if debug:
            open_count = np.sum(~np.isnan(open_polynya)) if open_polynya is not None else 0
            coastal_count = np.sum(~np.isnan(coastal_polynya)) if coastal_polynya is not None else 0
            print(f"[POLYNYA] Detection results: open={open_count} pixels, coastal={coastal_count} pixels")
        if polynya_type.lower() == "open":
            polynya_mask = ~np.isnan(open_polynya)
        elif polynya_type.lower() == "both":
            polynya_mask = (~np.isnan(open_polynya)) | (~np.isnan(coastal_polynya))
        else:
            polynya_mask = ~np.isnan(coastal_polynya)
    except Exception as e:
        if debug:
            print(f"[POLYNYA] Detection failed: {e}, using fallback (SIC < {polynya_threshold})")
        polynya_mask = sic_frac < polynya_threshold
        coastal_polynya = None

    # Projection
    proj = ccrs.SouthPolarStereo(central_longitude=float(central_lon))
    pc = ccrs.PlateCarree()

    # Extent
    extent_proj = None
    if not global_view:
        # If tracking positions are provided, use their actual bounds for extent
        tracking_lons = None
        tracking_lats = None
        if tracking_npz and os.path.isfile(tracking_npz):
            try:
                tracking_data = load_tracking_positions(tracking_npz, debug=False)
                lat_start = tracking_data['lat_start']
                lon_start = tracking_data['lon_start']
                lat_final = tracking_data['lat_final']
                lon_final = tracking_data['lon_final']
                valid = tracking_data['valid']
                
                # Combine start and end positions
                all_lats = np.concatenate([lat_start[valid], lat_final[valid]])
                all_lons = np.concatenate([lon_start[valid], lon_final[valid]])
                
                # Filter to Ross Sea region (for extent calculation, use full bounds if constrain_ross_sea)
                lon_0360 = normalize_longitude_0_360(all_lons)
                if constrain_ross_sea:
                    # Use full Ross Sea bounds for extent
                    ross_mask = (
                        (all_lats >= -78) & (all_lats <= -76.0) &
                        (lon_0360 >= 165.0) & (lon_0360 <= 190.0)
                    )
                else:
                    # Use actual data bounds
                    ross_mask = (
                        (all_lats >= -78) & (all_lats <= -76.0) &
                        (lon_0360 >= 165.0) & (lon_0360 <= 190.0)
                    )
                
                if np.any(ross_mask):
                    tracking_lats = all_lats[ross_mask]
                    tracking_lons = all_lons[ross_mask]
                    if debug:
                        print(f"[MAP] Using tracking positions for extent: {len(tracking_lats)} points")
            except Exception as e:
                if debug:
                    print(f"[MAP] Could not load tracking positions for extent: {e}")
        
        if tracking_lons is not None and tracking_lats is not None and len(tracking_lats) > 0:
            # If constrain_ross_sea is True, use full Ross Sea bounds instead of just tracking bounds
            if constrain_ross_sea:
                # Use full Ross Sea region: 165-200°E, -78 to -74°S
                lon_min = 165.0
                lon_max = 190.0
                lat_min = -78.0
                lat_max = -74.0
            else:
                # Use actual tracking position bounds
                lon_min = np.nanmin(tracking_lons)
                lon_max = np.nanmax(tracking_lons)
                lat_min = np.nanmin(tracking_lats)
                lat_max = np.nanmax(tracking_lats)
                
                # Add small buffer (10% on each side)
                lon_range = lon_max - lon_min
                lat_range = lat_max - lat_min
                lon_min = lon_min - 0.1 * lon_range
                lon_max = lon_max + 0.1 * lon_range
                lat_min = lat_min - 0.1 * lat_range
                lat_max = lat_max + 0.1 * lat_range
            
            # Convert to projection coordinates
            lon_corners = np.array([lon_min, lon_max, lon_min, lon_max])
            lat_corners = np.array([lat_min, lat_min, lat_max, lat_max])
            
            pts = proj.transform_points(pc, lon_corners, lat_corners)
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            
            # Add buffer
            x_range = np.ptp(x_vals) * buffer_factor
            y_range = np.ptp(y_vals) * buffer_factor
            x_center = np.mean(x_vals)
            y_center = np.mean(y_vals)
            
            # Enforce maximum aspect ratio to prevent too-tall figures
            # If aspect ratio > 1.5, expand the x_range to make it wider
            aspect_ratio = y_range / x_range if x_range > 0 else 1.0
            max_aspect = 1.5  # Maximum aspect ratio (height/width)
            
            if aspect_ratio > max_aspect:
                # Expand x_range to achieve desired aspect ratio
                x_range = y_range / max_aspect
                if debug:
                    print(f"[MAP] Adjusted extent: original aspect={aspect_ratio:.2f}, new aspect={max_aspect:.2f}")
            
            extent_proj = (
                x_center - x_range / 2.0,
                x_center + x_range / 2.0,
                y_center - y_range / 2.0,
                y_center + y_range / 2.0
            )
            
            if debug:
                final_aspect = (extent_proj[3] - extent_proj[2]) / (extent_proj[1] - extent_proj[0])
                print(f"[MAP] Using tracking position bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")
                print(f"[MAP] Projection extent: x=[{extent_proj[0]/1000:.1f}, {extent_proj[1]/1000:.1f}] km, "
                      f"y=[{extent_proj[2]/1000:.1f}, {extent_proj[3]/1000:.1f}] km, aspect={final_aspect:.2f}")
        elif tracking_npz or constrain_ross_sea:
            # Use focused Ross Sea region: 165-200°E, -78 to -74°S
            ross_sea_extent_deg = [165, 190, -79, -76]  # [lon_min, lon_max, lat_min, lat_max]
            
            # Convert to projection coordinates
            lon_corners = np.array([ross_sea_extent_deg[0], ross_sea_extent_deg[1], 
                                   ross_sea_extent_deg[0], ross_sea_extent_deg[1]])
            lat_corners = np.array([ross_sea_extent_deg[2], ross_sea_extent_deg[2],
                                   ross_sea_extent_deg[3], ross_sea_extent_deg[3]])
            
            pts = proj.transform_points(pc, lon_corners, lat_corners)
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            
            # Add buffer
            x_range = np.ptp(x_vals) * buffer_factor
            y_range = np.ptp(y_vals) * buffer_factor
            x_center = np.mean(x_vals)
            y_center = np.mean(y_vals)
            
            extent_proj = (
                x_center - x_range / 2.0,
                x_center + x_range / 2.0,
                y_center - y_range / 2.0,
                y_center + y_range / 2.0
            )
            
            if debug:
                print(f"[MAP] Using focused Ross Sea extent: {ross_sea_extent_deg}")
                print(f"[MAP] Projection extent: x=[{extent_proj[0]/1000:.1f}, {extent_proj[1]/1000:.1f}] km, "
                      f"y=[{extent_proj[2]/1000:.1f}, {extent_proj[3]/1000:.1f}] km")
        else:
            extent_proj = compute_projection_extent(lon0360, lat, proj, pc, buffer_factor=buffer_factor)
            
            # Ensure minimum extent size to prevent too-narrow maps
            if extent_proj is not None:
                x_min, x_max, y_min, y_max = extent_proj
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Minimum range in projection meters (roughly 200 km)
                min_range = 200000.0
                if x_range < min_range:
                    center_x = (x_min + x_max) / 2.0
                    x_min = center_x - min_range / 2.0
                    x_max = center_x + min_range / 2.0
                if y_range < min_range:
                    center_y = (y_min + y_max) / 2.0
                    y_min = center_y - min_range / 2.0
                    y_max = center_y + min_range / 2.0
                
                extent_proj = (x_min, x_max, y_min, y_max)

    # Figure - adjust size based on extent aspect ratio
    # For cartopy maps, we want the figure to match the data extent aspect ratio
    if extent_proj is not None:
        x_range = extent_proj[1] - extent_proj[0]
        y_range = extent_proj[3] - extent_proj[2]
        aspect_ratio = abs(y_range / x_range) if x_range > 0 else 1.0
        
        # Use a base width and adjust height to match aspect ratio
        base_width = 10.0
        fig_height = base_width * aspect_ratio
        
        # Limit extreme aspect ratios to keep figure reasonable
        # But allow more flexibility for narrow regions
        fig_height = max(7.0, min(14.0, fig_height))
        figsize = (base_width, fig_height)
        
        if debug:
            print(f"[MAP] Extent: x_range={x_range/1000:.1f} km, y_range={y_range/1000:.1f} km, aspect={aspect_ratio:.2f}")
            print(f"[MAP] Figure size: {figsize}")
    else:
        figsize = (10, 10)
    
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.subplot(1, 1, 1, projection=proj)
    
    # Explicitly set axes position to fill most of the figure
    # Leave space on right for colorbar
    ax.set_position([0.02, 0.02, 0.86, 0.96])

    if extent_proj is not None:
        ax.set_xlim(extent_proj[0], extent_proj[1])
        ax.set_ylim(extent_proj[2], extent_proj[3])
        # Set aspect to equal to prevent distortion
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.4", linewidth=0.3, zorder=1)
    ax.coastlines(linewidth=0.5, color="0.3", zorder=2)

    # SIT scatter
    lon_plot = wrap_lon180(lon0360)
    sc = ax.scatter(
        lon_plot, lat, c=sit,
        s=20, cmap="viridis", vmin=vmin, vmax=vmax,
        edgecolors="none", alpha=0.7,
        transform=pc, zorder=3
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, aspect=30)
    cbar.set_label("ESACCI Sea ice thickness (m)", fontsize=15)

    # # Mean annotation
    # lon_center = float(np.nanmean(lon_plot))
    # lat_center = float(np.nanmean(lat))
    # ax.text(
    #     lon_center, lat_center,
    #     f"Mean SIT: {sit_mean:.2f} m\n(n={sit.size})",
    #     transform=pc, fontsize=14, ha="center", va="center",
    #     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="black"),
    #     zorder=10
    # )

    # --- Coastal polynya filled with red dots ---
    lon_amsr_plot = wrap_lon180(lon_amsr)

    # coastal mask (MATLAB: coastal is NaN outside)
    if coastal_polynya is not None:
        coastal_mask = ~np.isnan(coastal_polynya)
    else:
        coastal_mask = polynya_mask.astype(bool)

    # Optional Ross Sea constraint for dots too
    if constrain_ross_sea:
        ross_dot = (
            (lat_amsr >= -78.0) & (lat_amsr <= -74.0) &
            (lon_amsr >= 165.0) & (lon_amsr <= 190.0)
        )
        coastal_mask = coastal_mask & ross_dot

    if show_coastal_polynya_dots and np.any(coastal_mask):
        # FILL mode (not boundary): use the whole coastal_mask
        dots_mask = coastal_mask.copy()

        # Thin dots uniformly on the 2D grid (much nicer than yy[::stride],xx[::stride])
        stride = int(max(1, polynya_dot_stride))
        if stride > 1:
            sampler = np.zeros_like(dots_mask, dtype=bool)
            sampler[::stride, ::stride] = True
            dots_mask = dots_mask & sampler

        yy, xx = np.where(dots_mask)
        lon_d = lon_amsr_plot[yy, xx]
        lat_d = lat_amsr[yy, xx]

        valid_coords = np.isfinite(lon_d) & np.isfinite(lat_d)
        lon_d = lon_d[valid_coords]
        lat_d = lat_d[valid_coords]

        if lon_d.size > 0:
            ax.scatter(
                lon_d, lat_d,
                s=float(polynya_dot_size),
                c="lightcoral", marker="o", linewidths=0,
                alpha=0.85,
                transform=pc, zorder=6,
                label="Coastal polynya"
            )
            if debug:
                print(f"[POLYNYA] Filled dots plotted: {lon_d.size} (stride={stride})")
    else:
        if debug:
            print("[POLYNYA] No coastal polynya dots to plot (mask empty or disabled).")


    # --- BedMachine ice shelves and grounding line ---
    bedmachine_data = None
    if bedmachine_nc and os.path.isfile(bedmachine_nc):
        try:
            bedmachine_data = load_bedmachine_mask(
                bedmachine_nc, lon_plot, lat, subsample=bedmachine_subsample, debug=debug
            )
            
            bm_lon_plot = bedmachine_data['lon']
            bm_lat = bedmachine_data['lat']
            
            if show_ice_shelves:
                ice_shelf_mask = bedmachine_data['ice_shelf_mask']
                if np.any(ice_shelf_mask):
                    # Use pcolormesh for filled ice shelf regions
                    ice_shelf_ma = np.ma.masked_where(
                        ~ice_shelf_mask,
                        np.ones_like(ice_shelf_mask, dtype=float)
                    )
                    im = ax.pcolormesh(
                        bm_lon_plot, bm_lat, ice_shelf_ma,
                        transform=pc, shading='auto',
                        cmap='gray', alpha=0.6,
                        vmin=0, vmax=3, zorder=3, edgecolors='none'
                    )
                    if debug:
                        print(f"[BEDMACHINE] Plotted ice shelves: {np.sum(ice_shelf_mask)} pixels")
                else:
                    if debug:
                        print("[WARN] No ice shelf pixels found in BedMachine data")
            
            # if show_grounding_line:
            #     grounding_line_mask = bedmachine_data['grounding_line_mask']
            #     if np.any(grounding_line_mask):
            #         # Use contour for grounding line
            #         try:
            #             ax.contour(
            #                 bm_lon_plot, bm_lat, grounding_line_mask.astype(float),
            #                 levels=[0.5], colors="darkred", linewidths=1.5,
            #                 transform=pc, zorder=5
            #             )
            #             if debug:
            #                 print(f"[BEDMACHINE] Plotted grounding line: {np.sum(grounding_line_mask)} pixels")
            #         except Exception as e:
            #             if debug:
            #                 print(f"[WARN] Could not plot grounding line as contour: {e}")
            #             # Fallback to scatter
            #             gl_lon = bm_lon_plot[grounding_line_mask]
            #             gl_lat = bm_lat[grounding_line_mask]
            #             if len(gl_lon) > 0:
            #                 ax.scatter(gl_lon, gl_lat, s=1, c="darkred", marker=".", 
            #                          alpha=0.8, transform=pc, zorder=5)
            #     else:
            #         if debug:
            #             print("[WARN] No grounding line pixels found in BedMachine data")
        except Exception as e:
            if debug:
                print(f"[WARN] Could not load BedMachine data: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"[WARN] Could not load BedMachine data: {e}")

    # --- Ice motion vectors (robust: displacement in projection meters) ---
    if ice_motion_nc and os.path.isfile(ice_motion_nc):
        if debug:
            print(f"[ICE_MOTION] Loading: {ice_motion_nc}")

        lat_i, lon0360_i, u_e_ms, v_n_ms, valid_i = load_ice_motion(
            ice_motion_nc, target_date, subsample=ice_motion_subsample, debug=debug
        )

        # Optional Ross Sea constraint
        if constrain_ross_sea:
            ross = (
                (lat_i >= -78.0) & (lat_i <= -74.0) &
                (lon0360_i >= 165.0) & (lon0360_i <= 190.0)
            )
            valid_i = valid_i & ross
            if debug:
                print(f"[ICE_MOTION] After Ross Sea constraint (165-200°E, -78 to -74°S): {int(np.sum(valid_i))} vectors")

        if np.any(valid_i):
            # Extract valid points in lon/lat
            lon0 = wrap_lon180(lon0360_i[valid_i]).astype(float)
            lat0 = lat_i[valid_i].astype(float)
            u0 = u_e_ms[valid_i].astype(float)   # m/s
            v0 = v_n_ms[valid_i].astype(float)   # m/s

            # Safety: if speeds look like cm/s mistakenly (median > ~2 m/s is unrealistic), convert
            speed0 = np.sqrt(u0**2 + v0**2)
            med_speed = float(np.nanmedian(speed0)) if speed0.size else np.nan
            if np.isfinite(med_speed) and med_speed > 2.0:
                if debug:
                    print(f"[ICE_MOTION][WARN] Median speed={med_speed:.2f} m/s looks too large; assuming cm/s -> m/s conversion missing.")
                u0 = u0 / 100.0
                v0 = v0 / 100.0
                speed0 = np.sqrt(u0**2 + v0**2)
                med_speed = float(np.nanmedian(speed0)) if speed0.size else np.nan

            # Endpoints after dt (in geographic)
            dt_seconds = float(ice_motion_dt_hours) * 3600.0
            lon1, lat1 = displace_lonlat(lon0, lat0, u0, v0, dt_seconds)

            # Project both start and end to map projection coords, then take dx/dy (meters)
            p0 = proj.transform_points(pc, lon0, lat0)
            p1 = proj.transform_points(pc, lon1, lat1)
            x0, y0 = p0[:, 0], p0[:, 1]
            dx, dy = (p1[:, 0] - p0[:, 0]), (p1[:, 1] - p0[:, 1])

            # Remove any bad transforms
            good = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(dx) & np.isfinite(dy)
            x0, y0, dx, dy, speed0 = x0[good], y0[good], dx[good], dy[good], speed0[good]

            if x0.size == 0:
                if debug:
                    print("[ICE_MOTION] No finite vectors after projection transform.")
            else:
                # Gain handling:
                # - if user sets ice_motion_arrow_gain <= 0, auto-scale to ~3% of map span
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xspan = float(np.ptp(xlim))
                yspan = float(np.ptp(ylim))
                span = min(xspan, yspan)

                gain = float(ice_motion_arrow_gain)
                med_disp = float(np.nanmedian(np.sqrt(dx**2 + dy**2))) if dx.size else np.nan

                if gain <= 0 and np.isfinite(med_disp) and med_disp > 0 and np.isfinite(span) and span > 0:
                    desired = 0.03 * span  # target median arrow length ~3% of map width
                    gain = desired / med_disp
                    gain = float(np.clip(gain, 0.5, 30.0))
                    if debug:
                        print(f"[ICE_MOTION] Auto gain={gain:.2f} (median disp={med_disp/1000:.2f} km over {ice_motion_dt_hours} h, span={span/1000:.0f} km)")
                else:
                    if debug and np.isfinite(med_disp):
                        print(f"[ICE_MOTION] Using gain={gain:.2f} (median disp={med_disp/1000:.2f} km over {ice_motion_dt_hours} h)")

                dx *= gain
                dy *= gain

                # Clip to current view extent so you only draw vectors actually on the map
                on = (x0 >= min(xlim)) & (x0 <= max(xlim)) & (y0 >= min(ylim)) & (y0 <= max(ylim))
                x0, y0, dx, dy, speed0 = x0[on], y0[on], dx[on], dy[on], speed0[on]

                if debug:
                    print(f"[ICE_MOTION] Vectors on-map after clipping: {x0.size}")

                if x0.size > 0:
                    # Slightly thicker arrows for visibility
                    q = ax.quiver(
                        x0, y0, dx, dy,
                        transform=proj,
                        angles="xy", scale_units="xy", scale=1.0,
                        width=0.0032, headwidth=3.0, headlength=3, headaxislength=3.5,
                        color="k", alpha=0.5, zorder=8
                    )

                    # Key: use median speed and display km/day
                    ref_speed = float(np.nanmedian(speed0)) if np.isfinite(np.nanmedian(speed0)) else 0.10
                    if ref_speed < 0.05:
                        ref_speed = 0.05
                    elif ref_speed < 0.10:
                        ref_speed = 0.10
                    elif ref_speed < 0.20:
                        ref_speed = 0.20
                    else:
                        ref_speed = round(ref_speed, 2)

                    ref_disp = ref_speed * dt_seconds * gain   # meters over dt, times gain
                    km_per_day = (ref_speed * 86400.0) / 1000.0

                    ax.quiverkey(
                        q, 0.05, 0.03, ref_disp,
                        f"{ref_speed:.2f} m/s",
                        labelpos="E", coordinates="axes",
                        fontproperties={"size": 12}, color="k"
                    )
                else:
                    if debug:
                        print("[ICE_MOTION] All vectors fell outside map extent (check region/extent).")
        else:
            if debug:
                print("[ICE_MOTION] No valid vectors after filtering/masking.")
    else:
        if debug:
            print("[ICE_MOTION] No ice motion file provided/found (use --ice_motion_nc).")

    # --- Tracking positions from NPZ file (from ice_tracking.py) ---
    if tracking_npz and os.path.isfile(tracking_npz):
        if debug:
            print(f"[TRACKING] Loading tracking positions from: {tracking_npz}")
        
        try:
            tracking_data = load_tracking_positions(tracking_npz, debug=debug)
            
            lat_start = tracking_data['lat_start']
            lon_start = tracking_data['lon_start']
            lat_final = tracking_data['lat_final']
            lon_final = tracking_data['lon_final']
            valid = tracking_data['valid']
            
            # Convert to -180-180 longitude for plotting
            lon_start_plot = wrap_lon180(normalize_longitude_0_360(lon_start))
            lon_final_plot = wrap_lon180(normalize_longitude_0_360(lon_final))
            
            # Always apply Ross Sea constraint when tracking positions are loaded
            # Use full Ross Sea region: 165-200°E, -78 to -74°S
            lon_start_0360 = normalize_longitude_0_360(lon_start)
            lon_final_0360 = normalize_longitude_0_360(lon_final)
            ross_mask = (
                (lat_start >= -78) & (lat_start <= -76.0) &
                (lon_start_0360 >= 165.0) & (lon_start_0360 <= 190.0)
            ) & (
                (lat_final >= -78) & (lat_final <= -76.0) &
                (lon_final_0360 >= 165.0) & (lon_final_0360 <= 190.0)
            )
            valid = valid & ross_mask
            if debug:
                print(f"[TRACKING] After Ross Sea constraint (165-200°E, -78 to -74°S): {int(np.sum(valid))} points")
            
            # Filter to valid points
            valid_mask = valid & np.isfinite(lat_start) & np.isfinite(lon_start_plot) & \
                        np.isfinite(lat_final) & np.isfinite(lon_final_plot)
            
            # Subsample tracking points if requested
            if tracking_subsample > 1 and np.any(valid_mask):
                n_valid = np.sum(valid_mask)
                indices = np.arange(n_valid)[::tracking_subsample]
                valid_indices = np.where(valid_mask)[0][indices]
                valid_mask_sub = np.zeros_like(valid_mask, dtype=bool)
                valid_mask_sub[valid_indices] = True
                valid_mask = valid_mask_sub
                if debug:
                    print(f"[TRACKING] Subsample: {n_valid} -> {np.sum(valid_mask)} points (stride={tracking_subsample})")
            
            if np.any(valid_mask):
                # Plot start positions
                if show_tracking_start:
                    ax.scatter(
                        lon_start_plot[valid_mask], lat_start[valid_mask],
                        s=tracking_marker_size, c='darkgreen', marker='o',
                        alpha=tracking_alpha, edgecolors='darkgreen', linewidths=0.1,
                        transform=pc, zorder=8,
                        label='Tracking start positions'
                    )
                    if debug:
                        print(f"[TRACKING] Plotted {np.sum(valid_mask)} start positions")
                
                # Plot end positions
                if show_tracking_end:
                    ax.scatter(
                        lon_final_plot[valid_mask], lat_final[valid_mask],
                        s=tracking_marker_size, c='darkred', marker='o',
                        alpha=tracking_alpha, edgecolors='darkred', linewidths=0.1,
                        transform=pc, zorder=8,
                        label='Tracking end positions'
                    )
                    if debug:
                        print(f"[TRACKING] Plotted {np.sum(valid_mask)} end positions")
            else:
                if debug:
                    print("[TRACKING] No valid tracking positions to plot")
                    
        except Exception as e:
            if debug:
                print(f"[WARN] Could not load/plot tracking positions: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"[WARN] Could not load/plot tracking positions: {e}")
    elif tracking_npz:
        if debug:
            print(f"[TRACKING] Tracking NPZ file not found: {tracking_npz}")

    # --- IS2 (ICESat-2) tracks overlay (DISABLED by default) ---
    if show_is2_tracks and is2_mat and os.path.isfile(is2_mat):
        if debug:
            print(f"[IS2] Loading IS2 tracks from: {is2_mat}")

        try:
            # Force the 6 tracks you want
            is2_data = load_is2_tracks(
                is2_mat,
                beams=["gt1l","gt1r","gt2l","gt2r","gt3l","gt3r"],
                target_date=target_date,
                debug=debug
            )


            if len(is2_data["beams"]) == 0:
                if debug:
                    print("[IS2] No beams loaded — check MAT structure/variable names.")
            else:
                # color cycle (one per beam)
                color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["k"])
                beam_colors = {b: color_cycle[i % len(color_cycle)] for i, b in enumerate(is2_data["beams"])}

                for i, beam in enumerate(is2_data["beams"]):
                    track = is2_data["tracks"][beam]
                    lat_is2 = track["lat"]
                    lon_is2 = track["lon"]
                    height_is2 = track.get("height", None)

                    # Convert longitude to -180..180 for plotting
                    lon_is2_plot = wrap_lon180(lon_is2)

                    # Optional Ross Sea constraint
                    if constrain_ross_sea:
                        lon_is2_0360 = normalize_longitude_0_360(lon_is2)
                        ross_mask = (
                            (lat_is2 >= -78.0) & (lat_is2 <= -74.0) &
                            (lon_is2_0360 >= 165.0) & (lon_is2_0360 <= 190.0)
                        )
                        if not np.any(ross_mask):
                            if debug:
                                print(f"[IS2] Beam {beam}: no points in Ross Sea window")
                            continue
                        lat_is2 = lat_is2[ross_mask]
                        lon_is2_plot = lon_is2_plot[ross_mask]
                        if height_is2 is not None and height_is2.size == ross_mask.size:
                            height_is2 = height_is2[ross_mask]

                    if lat_is2.size == 0:
                        continue

                    this_color = beam_colors[beam]

                    if is2_use_scatter:
                        ax.scatter(
                            lon_is2_plot, lat_is2,
                            s=is2_scatter_size,
                            color="k",
                            alpha=is2_track_alpha,
                            edgecolors="none",
                            transform=pc,
                            zorder=7,
                            label=f"IS2 {beam}",
                        )
                    else:
                        # line track
                        ax.plot(
                            lon_is2_plot, lat_is2,
                            transform=pc,
                            color="k",
                            alpha=is2_track_alpha,
                            linewidth=is2_track_linewidth,
                            zorder=7,
                            label=f"IS2 {beam}",
                        )

                    if debug:
                        print(f"[IS2] Plotted {beam}: {lat_is2.size} points")

        except Exception as e:
            if debug:
                print(f"[WARN] Could not overlay IS2 tracks: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"[WARN] Could not overlay IS2 tracks: {e}")

    # --- Backtrack IS2 to SWOT observation window ---
    if backtrack_is2 and is2_mat and os.path.isfile(is2_mat) and ice_motion_nc and os.path.isfile(ice_motion_nc):
        if swot_start_time_str is None or swot_end_time_str is None:
            if debug:
                print("[BACKTRACK] SWOT start/end times not provided, skipping backtracking")
        else:
            try:
                # Parse SWOT times
                swot_start_time = datetime.strptime(swot_start_time_str, "%Y%m%dT%H%M%S")
                swot_end_time = datetime.strptime(swot_end_time_str, "%Y%m%dT%H%M%S")
                
                if debug:
                    print(f"[BACKTRACK] Starting backtracking from IS2 to SWOT window")
                    print(f"[BACKTRACK] SWOT window: {swot_start_time} to {swot_end_time}")
                
                # Load IS2 data if not already loaded
                if not show_is2_tracks:
                    is2_data = load_is2_tracks(
                        is2_mat,
                        beams=["gt1l","gt1r","gt2l","gt2r","gt3l","gt3r"],
                        target_date=target_date,
                        debug=debug
                    )
                
                if len(is2_data["beams"]) > 0:
                    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["k"])
                    
                    for beam_idx, beam in enumerate(is2_data["beams"]):
                        track = is2_data["tracks"][beam]
                        lat_is2 = track["lat"]
                        lon_is2 = track["lon"]
                        
                        # Convert to 0-360 longitude
                        lon_is2_0360 = normalize_longitude_0_360(lon_is2)
                        
                        # Optional Ross Sea constraint
                        if constrain_ross_sea:
                            lon_is2_plot = wrap_lon180(lon_is2)
                            ross_mask = (
                                (lat_is2 >= -78.0) & (lat_is2 <= -74.0) &
                                (lon_is2_0360 >= 165.0) & (lon_is2_0360 <= 190.0)
                            )
                            if not np.any(ross_mask):
                                continue
                            lat_is2 = lat_is2[ross_mask]
                            lon_is2_0360 = lon_is2_0360[ross_mask]
                        
                        if lat_is2.size == 0:
                            continue
                        
                        # Perform backtracking
                        # Assume IS2 observation time is end of day (or could be extracted from file)
                        is2_obs_time = target_date.replace(hour=23, minute=59, second=59)
                        
                        if debug:
                            print(f"[BACKTRACK] Backtracking {beam}: {lat_is2.size} points")
                        
                        backtrack_result = backtrack_is2_to_swot(
                            lat_is2, lon_is2_0360, is2_obs_time,
                            ice_motion_nc, swot_start_time, swot_end_time,
                            dt_hours=backtrack_dt_hours, debug=debug
                        )
                        
                        # Plot backtracked trajectories
                        valid = backtrack_result["valid"]
                        trajectories = backtrack_result["trajectories"]
                        
                        if np.any(valid):
                            beam_color = color_cycle[beam_idx % len(color_cycle)]
                            
                            # Plot trajectories (lines connecting backtracked points)
                            for traj_idx, traj in enumerate(trajectories):
                                if valid[traj_idx] and len(traj) > 1:
                                    traj_lat = np.array([p[0] for p in traj])
                                    traj_lon = np.array([p[1] for p in traj])
                                    traj_lon_plot = wrap_lon180(traj_lon)
                                    
                                    ax.plot(
                                        traj_lon_plot, traj_lat,
                                        transform=pc,
                                        color=beam_color,
                                        alpha=0.6,
                                        linewidth=1.0,
                                        linestyle='--',
                                        zorder=8,
                                        label=f"Backtrack {beam}" if traj_idx == 0 else ""
                                    )
                            
                            # Plot start positions (SWOT start time)
                            lat_start = backtrack_result["lat_start"][valid]
                            lon_start = backtrack_result["lon_start"][valid]
                            lon_start_plot = wrap_lon180(lon_start)
                            
                            ax.scatter(
                                lon_start_plot, lat_start,
                                s=30, color=beam_color, marker='o',
                                alpha=0.8, edgecolors='k', linewidths=0.5,
                                transform=pc, zorder=9,
                                label=f"Backtrack start {beam}" if beam_idx == 0 else ""
                            )
                            
                            # Plot end positions (SWOT end time)
                            lat_end = backtrack_result["lat_end"][valid]
                            lon_end = backtrack_result["lon_end"][valid]
                            lon_end_plot = wrap_lon180(lon_end)
                            
                            ax.scatter(
                                lon_end_plot, lat_end,
                                s=30, color=beam_color, marker='s',
                                alpha=0.8, edgecolors='k', linewidths=0.5,
                                transform=pc, zorder=9,
                                label=f"Backtrack end {beam}" if beam_idx == 0 else ""
                            )
                            
                            if debug:
                                print(f"[BACKTRACK] Plotted {beam}: {np.sum(valid)} valid trajectories")
                
            except Exception as e:
                if debug:
                    print(f"[WARN] Could not perform backtracking: {e}")
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"[WARN] Could not perform backtracking: {e}")

    # # Title
    # ax.set_title(
    #     f"ESACCI Mean Sea Ice Thickness: {sit_mean:.2f} m\n{target_date_minus_1:%Y-%m-%d}",
    #     fontsize=16, fontweight="bold", pad=15
    # )
    ax.spines["geo"].set_linewidth(0)

    # Don't use tight_layout() with cartopy - it causes issues with projections
    # We've already set the axes position explicitly above with ax.set_position()
    
    # Save with tight bbox to crop any remaining white space
    plt.savefig(out_png, dpi=300, facecolor="white", edgecolor="none", 
                bbox_inches="tight", pad_inches=0.02)
    print(f"[INFO] Saved map: {out_png}")
    plt.close(fig)


def plot_zoomed_tracking(
    tracking_npz,
    out_png,
    lat_range=(-78, -76),
    lon_range=(165, 190),
    marker_size=20,
    alpha=0.8,
    amsr_mat=None,
    target_date=None,
    polynya_threshold=0.7,
    polynya_dot_size=15,
    polynya_dot_stride=2,
    target_points=20,
    bedmachine_nc=None,
    show_ice_shelves=False,
    show_grounding_line=False,
    ice_motion_nc=None,
    ice_motion_subsample=2,
    ice_motion_arrow_gain=5.0,
    debug=False,
):
    """
    Create a zoomed-in figure showing backward tracking positions between specified latitude range.
    
    Parameters:
    -----------
    tracking_npz : str
        Path to NPZ file with tracking positions (from ice_tracking.py)
    out_png : str
        Output PNG file path
    lat_range : tuple
        Latitude range (min, max) in degrees (default: (-78, -76))
    lon_range : tuple
        Longitude range (min, max) in degrees (default: (165, 190))
    marker_size : float
        Size of tracking position markers (default: 20)
    alpha : float
        Alpha (transparency) for tracking positions (default: 0.8)
    debug : bool
        Print debug information
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not os.path.isfile(tracking_npz):
        raise FileNotFoundError(f"Tracking NPZ file not found: {tracking_npz}")
    
    # Load tracking positions
    if debug:
        print(f"[ZOOM] Loading tracking positions from: {tracking_npz}")
    
    tracking_data = load_tracking_positions(tracking_npz, debug=debug)
    
    lat_start = tracking_data['lat_start']
    lon_start = tracking_data['lon_start']
    lat_final = tracking_data['lat_final']
    lon_final = tracking_data['lon_final']
    valid = tracking_data['valid']
    
    # Convert to -180-180 longitude for plotting
    lon_start_plot = wrap_lon180(normalize_longitude_0_360(lon_start))
    lon_final_plot = wrap_lon180(normalize_longitude_0_360(lon_final))
    
    # Filter to specified latitude and longitude range
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    lon_start_0360 = normalize_longitude_0_360(lon_start)
    lon_final_0360 = normalize_longitude_0_360(lon_final)
    
    region_mask = (
        (lat_start >= lat_min) & (lat_start <= lat_max) &
        (lon_start_0360 >= lon_min) & (lon_start_0360 <= lon_max)
    ) & (
        (lat_final >= lat_min) & (lat_final <= lat_max) &
        (lon_final_0360 >= lon_min) & (lon_final_0360 <= lon_max)
    )
    
    valid = valid & region_mask
    
    # Filter to valid points
    valid_mask = valid & np.isfinite(lat_start) & np.isfinite(lon_start_plot) & \
                np.isfinite(lat_final) & np.isfinite(lon_final_plot)
    
    # Subsample tracking points to avoid too many dots
    n_valid = int(np.sum(valid_mask))
    if n_valid > target_points:
        # Calculate stride to get approximately target_points
        stride = max(1, n_valid // target_points)
        valid_indices = np.where(valid_mask)[0][::stride]
        valid_mask_sub = np.zeros_like(valid_mask, dtype=bool)
        valid_mask_sub[valid_indices] = True
        valid_mask = valid_mask_sub
        if debug:
            print(f"[ZOOM] Subsample: {n_valid} -> {int(np.sum(valid_mask))} points (stride={stride})")
    
    if debug:
        print(f"[ZOOM] Filtered to region: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
        print(f"[ZOOM] Valid points: {int(np.sum(valid_mask))}")
        if np.any(valid_mask):
            print(f"[ZOOM] Start positions: lon=[{np.nanmin(lon_start_plot[valid_mask]):.2f}, {np.nanmax(lon_start_plot[valid_mask]):.2f}], "
                  f"lat=[{np.nanmin(lat_start[valid_mask]):.2f}, {np.nanmax(lat_start[valid_mask]):.2f}]")
            print(f"[ZOOM] Final positions: lon=[{np.nanmin(lon_final_plot[valid_mask]):.2f}, {np.nanmax(lon_final_plot[valid_mask]):.2f}], "
                  f"lat=[{np.nanmin(lat_final[valid_mask]):.2f}, {np.nanmax(lat_final[valid_mask]):.2f}]")
    
    if not np.any(valid_mask):
        raise RuntimeError(f"No valid tracking positions in region: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
    
    # Create figure with South Polar Stereographic projection
    proj = ccrs.SouthPolarStereo(central_longitude=0.0)
    pc = ccrs.PlateCarree()
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=proj)
    
    # Calculate extent from data
    lon_all = np.concatenate([lon_start_plot[valid_mask], lon_final_plot[valid_mask]])
    lat_all = np.concatenate([lat_start[valid_mask], lat_final[valid_mask]])
    
    lon_center = np.mean([np.nanmin(lon_all), np.nanmax(lon_all)])
    lat_center = np.mean([np.nanmin(lat_all), np.nanmax(lat_all)])
    
    # Convert extent to projection coordinates
    lon_corners = np.array([np.nanmin(lon_all), np.nanmax(lon_all), 
                           np.nanmin(lon_all), np.nanmax(lon_all)])
    lat_corners = np.array([np.nanmin(lat_all), np.nanmin(lat_all),
                           np.nanmax(lat_all), np.nanmax(lat_all)])
    
    pts = proj.transform_points(pc, lon_corners, lat_corners)
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    
    # Add 10% buffer
    x_range = np.ptp(x_vals) * 1.1
    y_range = np.ptp(y_vals) * 1.1
    x_center = np.mean(x_vals)
    y_center = np.mean(y_vals)
    
    # Enforce reasonable aspect ratio
    aspect_ratio = y_range / x_range if x_range > 0 else 1.0
    max_aspect = 1.2
    if aspect_ratio > max_aspect:
        x_range = y_range / max_aspect
    
    extent_proj = (
        x_center - x_range / 2.0,
        x_center + x_range / 2.0,
        y_center - y_range / 2.0,
        y_center + y_range / 2.0
    )
    
    ax.set_xlim(extent_proj[0], extent_proj[1])
    ax.set_ylim(extent_proj[2], extent_proj[3])
    ax.set_aspect('equal', adjustable='box')
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="0.4", linewidth=0.3, zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3, zorder=0)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--', zorder=3)
    gl.top_labels = False
    gl.right_labels = False
    
    # --- Load and plot BedMachine data (ice shelves and grounding lines) ---
    # In zoom view, show BedMachine by default if file is provided
    if bedmachine_nc and os.path.isfile(bedmachine_nc):
        try:
            if debug:
                print(f"[ZOOM] Loading BedMachine data from: {bedmachine_nc}")
            
            # Get data bounds for BedMachine ROI
            lon_data = np.concatenate([lon_start_plot[valid_mask], lon_final_plot[valid_mask]])
            lat_data = np.concatenate([lat_start[valid_mask], lat_final[valid_mask]])
            
            bedmachine_data = load_bedmachine_mask(
                bedmachine_nc, lon_data=lon_data, lat_data=lat_data, 
                buffer_km=200, debug=debug
            )
            
            if bedmachine_data:
                bm_lon = bedmachine_data['lon']
                bm_lat = bedmachine_data['lat']
                bm_lon_plot = wrap_lon180(bm_lon)
                
                # Filter to zoom region
                bm_lon_0360 = normalize_longitude_0_360(bm_lon)
                bm_region_mask = (
                    (bm_lat >= lat_min) & (bm_lat <= lat_max) &
                    (bm_lon_0360 >= lon_min) & (bm_lon_0360 <= lon_max)
                )
                
                # Show ice shelves by default in zoom view (or if flag is set)
                if show_ice_shelves or True:  # Always show in zoom
                    ice_shelf_mask = bedmachine_data['ice_shelf_mask']
                    if np.any(ice_shelf_mask & bm_region_mask):
                        ice_shelf_ma = np.ma.masked_array(
                            ice_shelf_mask.astype(float),
                            mask=~(ice_shelf_mask & bm_region_mask)
                        )
                        ax.pcolormesh(
                            bm_lon_plot, bm_lat, ice_shelf_ma,
                            transform=pc, shading='auto',
                            cmap='gray', alpha=0.6,
                            vmin=0, vmax=3, zorder=10, edgecolors='none'
                        )
                        if debug:
                            print(f"[ZOOM] Plotted ice shelves: {np.sum(ice_shelf_mask & bm_region_mask)} pixels")
                    else:
                        if debug:
                            print(f"[ZOOM] No ice shelf pixels in zoom region")
                
                # # Show grounding line by default in zoom view (or if flag is set)
                # if show_grounding_line or True:  # Always show in zoom
                #     grounding_line_mask = bedmachine_data['grounding_line_mask']
                #     if np.any(grounding_line_mask & bm_region_mask):
                #         try:
                #             ax.contour(
                #                 bm_lon_plot, bm_lat, 
                #                 (grounding_line_mask & bm_region_mask).astype(float),
                #                 levels=[0.5], colors="darkred", linewidths=1.5,
                #                 transform=pc, zorder=5
                #             )
                #             if debug:
                #                 print(f"[ZOOM] Plotted grounding line: {np.sum(grounding_line_mask & bm_region_mask)} pixels")
                #         except Exception as e:
                #             if debug:
                #                 print(f"[ZOOM] Could not plot grounding line as contour: {e}")
                #             # Fallback to scatter
                #             gl_mask = grounding_line_mask & bm_region_mask
                #             gl_lon = bm_lon_plot[gl_mask]
                #             gl_lat = bm_lat[gl_mask]
                #             if len(gl_lon) > 0:
                #                 ax.scatter(gl_lon, gl_lat, s=1, c="darkred", marker=".", 
                #                          alpha=0.8, transform=pc, zorder=5)
                #                 if debug:
                #                     print(f"[ZOOM] Plotted grounding line (scatter): {len(gl_lon)} points")
                #     else:
                #         if debug:
                #             print(f"[ZOOM] No grounding line pixels in zoom region")
        except Exception as e:
            if debug:
                print(f"[ZOOM] Could not load BedMachine data: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"[ZOOM] Could not load BedMachine data: {e}")
    
    # --- Load and plot ice motion vectors ---
    if ice_motion_nc and os.path.isfile(ice_motion_nc):
        try:
            if debug:
                print(f"[ZOOM] Loading ice motion from: {ice_motion_nc}")
            
            lat_i, lon0360_i, u_e_ms, v_n_ms, valid_i = load_ice_motion(
                ice_motion_nc, target_date, subsample=ice_motion_subsample, debug=debug
            )
            
            # Filter to zoom region
            lon_i_0360 = normalize_longitude_0_360(lon0360_i)
            region_i = (
                (lat_i >= lat_min) & (lat_i <= lat_max) &
                (lon_i_0360 >= lon_min) & (lon_i_0360 <= lon_max)
            )
            valid_i = valid_i & region_i
            
            if debug:
                print(f"[ZOOM] Ice motion vectors in region: {int(np.sum(valid_i))}")
                if np.any(valid_i):
                    print(f"[ZOOM] Ice motion lat range: [{np.nanmin(lat_i[valid_i]):.2f}, {np.nanmax(lat_i[valid_i]):.2f}]")
                    print(f"[ZOOM] Ice motion lon range: [{np.nanmin(lon_i_0360[valid_i]):.2f}, {np.nanmax(lon_i_0360[valid_i]):.2f}]")
            
            if np.any(valid_i):
                # Extract valid points
                lon0 = wrap_lon180(lon0360_i[valid_i]).astype(float)
                lat0 = lat_i[valid_i].astype(float)
                u0 = u_e_ms[valid_i].astype(float)   # m/s
                v0 = v_n_ms[valid_i].astype(float)   # m/s
                
                # Safety: check for cm/s vs m/s
                speed0 = np.sqrt(u0**2 + v0**2)
                med_speed = float(np.nanmedian(speed0)) if speed0.size else np.nan
                if np.isfinite(med_speed) and med_speed > 2.0:
                    if debug:
                        print(f"[ZOOM][ICE_MOTION] Median speed={med_speed:.2f} m/s looks too large; converting cm/s -> m/s")
                    u0 = u0 / 100.0
                    v0 = v0 / 100.0
                    speed0 = np.sqrt(u0**2 + v0**2)
                
                # Use same approach as main function: calculate endpoint in geographic, then transform both
                # This properly accounts for projection distortion
                dt_seconds = 86400.0  # 24 hours (same as main function uses for display)
                lon1, lat1 = displace_lonlat(lon0, lat0, u0, v0, dt_seconds)
                
                # Project both start and end to map projection coords, then take dx/dy (meters)
                p0 = proj.transform_points(pc, lon0, lat0)
                p1 = proj.transform_points(pc, lon1, lat1)
                x0, y0 = p0[:, 0], p0[:, 1]
                dx, dy = (p1[:, 0] - p0[:, 0]), (p1[:, 1] - p0[:, 1])
                
                # Remove any bad transforms
                good = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(dx) & np.isfinite(dy)
                x0, y0, dx, dy, speed0, lon0, lat0 = x0[good], y0[good], dx[good], dy[good], speed0[good], lon0[good], lat0[good]
                
                if debug:
                    print(f"[ZOOM] After transform: {x0.size} valid vectors")
                    if x0.size > 0:
                        print(f"[ZOOM] Vector locations: lat=[{np.nanmin(lat0):.2f}, {np.nanmax(lat0):.2f}], "
                              f"lon=[{np.nanmin(lon0):.2f}, {np.nanmax(lon0):.2f}]")
                
                # Apply gain
                dx *= ice_motion_arrow_gain
                dy *= ice_motion_arrow_gain
                
                # Clip to current view extent (but set limits first if not already set)
                if ax.get_xlim()[0] == ax.get_xlim()[1] or ax.get_ylim()[0] == ax.get_ylim()[1]:
                    # Limits not set yet, use data bounds
                    xlim = (np.nanmin(x0) - 50000, np.nanmax(x0) + 50000)
                    ylim = (np.nanmin(y0) - 50000, np.nanmax(y0) + 50000)
                else:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                
                on = (x0 >= min(xlim)) & (x0 <= max(xlim)) & (y0 >= min(ylim)) & (y0 <= max(ylim))
                x0, y0, dx, dy = x0[on], y0[on], dx[on], dy[on]
                
                if debug:
                    print(f"[ZOOM] After clipping to view: {x0.size} vectors")
                    if x0.size == 0 and good.size > 0:
                        print(f"[ZOOM] WARNING: All vectors were clipped! View extent may be too small.")
                        print(f"[ZOOM] View extent: x=[{xlim[0]/1000:.1f}, {xlim[1]/1000:.1f}] km, "
                              f"y=[{ylim[0]/1000:.1f}, {ylim[1]/1000:.1f}] km")
                        print(f"[ZOOM] Vector range: x=[{np.nanmin(x0[~on])/1000:.1f}, {np.nanmax(x0[~on])/1000:.1f}] km, "
                              f"y=[{np.nanmin(y0[~on])/1000:.1f}, {np.nanmax(y0[~on])/1000:.1f}] km")
                
                if x0.size > 0:
                    if debug:
                        print(f"[ZOOM] Ice motion: {x0.size} vectors after clipping, median speed={np.nanmedian(speed0[on]):.3f} m/s")
                    q = ax.quiver(
                        x0, y0, dx, dy,
                        transform=proj,
                        angles="xy", scale_units="xy", scale=1.0,
                        width=0.0062, headwidth=5.0, headlength=5, headaxislength=5.5,
                        color="k", alpha=0.3, zorder=10
                    )
                    
                    # Add reference vector
                    ref_speed = float(np.nanmedian(speed0[on])) if np.any(on) and np.isfinite(np.nanmedian(speed0[on])) else 0.10
                    if ref_speed < 0.05:
                        ref_speed = 0.05
                    elif ref_speed < 0.10:
                        ref_speed = 0.10
                    elif ref_speed < 0.20:
                        ref_speed = 0.20
                    else:
                        ref_speed = round(ref_speed, 2)
                    
                    ref_disp = ref_speed * dt_seconds * ice_motion_arrow_gain
                    ax.quiverkey(
                        q, 0.07, 0.03, ref_disp,
                        f"{ref_speed:.2f} m/s",
                        coordinates='axes', labelpos='E', fontproperties={'size': 15}
                    )
                    
                    if debug:
                        print(f"[ZOOM] Plotted {x0.size} ice motion vectors")
        except Exception as e:
            if debug:
                print(f"[ZOOM] Could not load ice motion: {e}")
                import traceback
                traceback.print_exc()
    
    # --- Load and plot polynya areas if AMSR data is provided ---
    if amsr_mat and os.path.isfile(amsr_mat):
        try:
            if debug:
                print(f"[ZOOM] Loading polynya data from: {amsr_mat}")
            
            amsr = load_mat_any(amsr_mat)
            if "sic" not in amsr or "lat" not in amsr or "lon" not in amsr:
                if debug:
                    print("[ZOOM] AMSR MAT missing required fields (sic, lat, lon)")
            else:
                sic = np.array(amsr["sic"], dtype=float)
                lat_amsr = np.array(amsr["lat"], dtype=float)
                lon_amsr = np.array(amsr["lon"], dtype=float)
                
                # Ensure sic time dimension first
                if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
                    sic = np.moveaxis(sic, -1, 0)
                
                # Fix lat/lon orientation if needed
                if lat_amsr.shape != sic.shape[1:]:
                    if lat_amsr.T.shape == sic.shape[1:]:
                        lat_amsr = lat_amsr.T
                        lon_amsr = lon_amsr.T
                
                # If AMSR lat/lon are 1D vectors, expand to 2D to match SIC grid
                if lat_amsr.ndim == 1 and lon_amsr.ndim == 1 and sic.ndim == 2:
                    lon_amsr, lat_amsr = np.meshgrid(lon_amsr, lat_amsr)
                
                lon_amsr = normalize_longitude_0_360(lon_amsr)
                lon_amsr_plot = wrap_lon180(lon_amsr)
                
                # Get date for day index
                from datetime import datetime
                if target_date is None:
                    # Default to 2023-10-28 if not provided
                    target_date = datetime(2023, 10, 28)
                
                try:
                    doy = target_date.timetuple().tm_yday
                    doy = min(doy, sic.shape[0])
                    day_idx = doy - 1
                    sic_daily = sic[day_idx, :, :] if day_idx < sic.shape[0] else sic[-1, :, :]
                except:
                    sic_daily = sic[0, :, :] if sic.ndim == 3 else sic
                
                # Fraction (0..1)
                sic_frac = sic_daily / 100.0 if np.nanmax(sic_daily) > 1.0 else sic_daily.copy()
                
                # Polynya detection
                try:
                    open_polynya, coastal_polynya = polynya_output_amsr2(sic_frac, polynya_threshold)
                    if coastal_polynya is not None:
                        coastal_mask = ~np.isnan(coastal_polynya)
                    else:
                        coastal_mask = (sic_frac < polynya_threshold).astype(bool)
                    
                    # Filter to zoom region
                    region_polynya = (
                        (lat_amsr >= lat_min) & (lat_amsr <= lat_max) &
                        (lon_amsr >= lon_min) & (lon_amsr <= lon_max)
                    )
                    coastal_mask = coastal_mask & region_polynya
                    
                    if np.any(coastal_mask):
                        # Subsample polynya dots (same approach as main function)
                        dots_mask = coastal_mask.copy()
                        stride = int(max(1, polynya_dot_stride))
                        if stride > 1:
                            sampler = np.zeros_like(dots_mask, dtype=bool)
                            sampler[::stride, ::stride] = True
                            dots_mask = dots_mask & sampler
                        
                        yy, xx = np.where(dots_mask)
                        lon_d = lon_amsr_plot[yy, xx]
                        lat_d = lat_amsr[yy, xx]
                        
                        valid_coords = np.isfinite(lon_d) & np.isfinite(lat_d)
                        lon_d = lon_d[valid_coords]
                        lat_d = lat_d[valid_coords]
                        
                        if lon_d.size > 0:
                            ax.scatter(
                                lon_d, lat_d,
                                s=float(polynya_dot_size),
                                c="lightcoral", marker="o", linewidths=0,
                                alpha=0.55,
                                transform=pc, zorder=4,
                                label="Coastal polynya"
                            )
                            if debug:
                                print(f"[ZOOM] Plotted {lon_d.size} polynya dots (stride={stride})")
                except Exception as e:
                    if debug:
                        print(f"[ZOOM] Could not plot polynya: {e}")
        except Exception as e:
            if debug:
                print(f"[ZOOM] Could not load polynya data: {e}")
    
    # Rotate the entire view by 180 degrees (for visualization only)
    # This is equivalent to flipping both x and y axes.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[::-1])
    ax.set_ylim(ylim[::-1])

    # Plot start positions
    ax.scatter(
        lon_start_plot[valid_mask], lat_start[valid_mask],
        s=marker_size, c='lime', marker='s',
        alpha=alpha+0.2, edgecolors='yellowgreen', linewidths=0,
        transform=pc, zorder=9,
        label='Start positions'
    )
    
    # Plot end positions
    ax.scatter(
        lon_final_plot[valid_mask], lat_final[valid_mask],
        s=marker_size+15, c='orange', marker='h',
        alpha=alpha+0.2, edgecolors='orange', linewidths=0,
        transform=pc, zorder=9,
        label='End positions'
    )
    
    # Add legend (update to include polynya if plotted)
    ax.legend(loc='upper left', fontsize=15, framealpha=0.9)
    
    # # Title
    # ax.set_title(
    #     f"Backward Tracking: {lat_range[0]}° to {lat_range[1]}°S\n"
    #     f"Points: {int(np.sum(valid_mask))}",
    #     fontsize=14, fontweight="bold", pad=15
    # )
    
    # Save figure
    plt.savefig(out_png, dpi=300, facecolor="white", edgecolor="none", 
                bbox_inches="tight", pad_inches=0.02)
    print(f"[ZOOM] Saved zoomed tracking plot: {out_png}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot ESACCI SIT with AMSR coastal polynya outline and ice drift vectors."
    )

    ap.add_argument("--date", required=True, help="Date in YYYYMMDD (e.g., 20231028)")
    ap.add_argument("--esacci_dir", default="/Volumes/Yotta_1/SWOT/Antarctic_hi",
                    help="Directory containing ESACCI files")
    ap.add_argument("--amsr_mat", required=True, help="AMSR MAT file containing sic, lat, lon")

    ap.add_argument("--polynya_threshold", type=float, default=0.7,
                    help="SIC threshold for polynya (default 0.7)")
    ap.add_argument("--polynya_type", default="coastal", choices=["open", "coastal", "both"],
                    help="Which polynya to use for plotting/labeling")

    ap.add_argument("--out_png", default="sit_map.png", help="Output PNG")
    ap.add_argument("--vmin", type=float, default=0.0, help="SIT color min")
    ap.add_argument("--vmax", type=float, default=3.0, help="SIT color max")
    ap.add_argument("--central_lon", type=float, default=180.0, help="Polar stereo central longitude")
    ap.add_argument("--buffer_factor", type=float, default=1.2, help="Zoom buffer factor")
    ap.add_argument("--global_view", action="store_true", help="Show full Antarctica")

    ap.add_argument("--ice_motion_nc", default=None,
                    help="Ice motion netCDF (e.g., icemotion_daily_sh_25km_*.nc)")
    ap.add_argument("--ice_motion_subsample", type=int, default=5,
                    help="Subsample ice motion grid (default 5)")
    ap.add_argument("--ice_motion_dt_hours", type=float, default=24.0,
                    help="Displacement time window for arrows (hours, default 24)")
    ap.add_argument("--ice_motion_arrow_gain", type=float, default=1.0,
                    help="Multiply arrow displacement for visibility (default 1.0; try 2-5)")

    ap.add_argument("--constrain_ross_sea", action="store_true",
                    help="Constrain plot to Ross Sea (165-200°E, -78 to -74°S)")

    ap.add_argument("--bedmachine_nc", default=None,
                    help="BedMachine NetCDF file (e.g., BedMachineAntarctica_2020-07-15_v02.nc)")
    ap.add_argument("--show_ice_shelves", action="store_true",
                    help="Overlay ice shelves from BedMachine")
    ap.add_argument("--show_grounding_line", action="store_true",
                    help="Overlay grounding line from BedMachine")
    ap.add_argument("--bedmachine_subsample", type=int, default=5,
                    help="Subsample BedMachine grid (default 5)")

    ap.add_argument("--is2_mat", default=None,
                    help="IS2-SWOT collocation MATLAB file (e.g., 20231028_IS2_SWOT.mat) [DISABLED - use --tracking_npz instead]")
    ap.add_argument("--show_is2_tracks", action="store_true",
                    help="Overlay IS2 (ICESat-2) tracks [DISABLED - use --tracking_npz instead]")
    
    ap.add_argument("--tracking_npz", default=None,
                    help="NPZ file with tracking positions from ice_tracking.py (e.g., trajectories_backward.npz)")
    ap.add_argument("--show_tracking_start", action="store_true",
                    help="Show start positions from tracking (default: True if --tracking_npz provided)")
    ap.add_argument("--no_tracking_start", action="store_true",
                    help="Don't show start positions from tracking")
    ap.add_argument("--show_tracking_end", action="store_true",
                    help="Show end positions from tracking (default: True if --tracking_npz provided)")
    ap.add_argument("--no_tracking_end", action="store_true",
                    help="Don't show end positions from tracking")
    ap.add_argument("--tracking_start_color", default="lime",
                    help="Color for tracking start positions (default: lime)")
    ap.add_argument("--tracking_end_color", default="red",
                    help="Color for tracking end positions (default: red)")
    ap.add_argument("--tracking_marker_size", type=float, default=1,
                    help="Size of tracking position markers (default: 8)")
    ap.add_argument("--tracking_alpha", type=float, default=0.9,
                    help="Alpha (transparency) for tracking positions (default: 0.9)")
    ap.add_argument("--tracking_subsample", type=int, default=1,
                    help="Subsample tracking points (plot every Nth point, default: 1 = all points)")
    ap.add_argument("--zoom_tracking", action="store_true",
                    help="Create zoomed-in figure for tracking between -78 and -76 degrees latitude")
    ap.add_argument("--zoom_out_png", default=None,
                    help="Output PNG file for zoomed tracking plot (default: adds '_zoom' to --out_png)")
    ap.add_argument("--is2_track_color", default="orange",
                    help="Color for IS2 tracks (default: orange, used if not using colormap)")
    ap.add_argument("--is2_track_alpha", type=float, default=0.7,
                    help="Alpha (transparency) for IS2 tracks (default: 0.7)")
    ap.add_argument("--is2_track_linewidth", type=float, default=1.5,
                    help="Line width for IS2 tracks (default: 1.5, used if not using scatter)")
    ap.add_argument("--is2_use_scatter", action="store_true",
                    help="Use scatter plot (circles) instead of lines (default: True, enabled by default)")
    ap.add_argument("--is2_use_lines", action="store_true",
                    help="Use line plot instead of scatter (overrides --is2_use_scatter)")
    ap.add_argument("--is2_scatter_size", type=float, default=15,
                    help="Size of scatter circles (default: 15)")
    ap.add_argument("--is2_colormap", default="gray_r",
                    help="Colormap for scatter (light to dark: gray_r, viridis_r, etc., default: gray_r)")
    ap.add_argument("--is2_color_by", default="height", choices=["height", "index", "none"],
                    help="Color by: height, index (sequential), or none (single color, default: height)")
    ap.add_argument("--backtrack_is2", action="store_true",
                    help="Backtrack IS2 locations using ice drift to find where ice was during SWOT observations")
    ap.add_argument("--swot_start_time", default=None,
                    help="SWOT observation start time (format: YYYYMMDDTHHMMSS, e.g., 20231028T191642)")
    ap.add_argument("--swot_end_time", default=None,
                    help="SWOT observation end time (format: YYYYMMDDTHHMMSS, e.g., 20231028T215103)")
    ap.add_argument("--backtrack_dt_hours", type=float, default=1.0,
                    help="Time step for backtracking integration in hours (default: 1.0)")

    ap.add_argument("--no_quality_filter", action="store_true",
                    help="Disable ESACCI quality filtering (flag_miz)")
    ap.add_argument("--max_uncertainty", type=float, default=None,
                    help="Max allowed SIT uncertainty (m)")

    ap.add_argument("--debug", action="store_true", help="Verbose logging")

    args = ap.parse_args()

    if not os.path.isdir(args.esacci_dir):
        raise RuntimeError(f"ESACCI directory does not exist: {args.esacci_dir}")
    if not os.path.isfile(args.amsr_mat):
        raise RuntimeError(f"AMSR MAT does not exist: {args.amsr_mat}")
    if args.ice_motion_nc and (not os.path.isfile(args.ice_motion_nc)):
        raise RuntimeError(f"Ice motion file not found: {args.ice_motion_nc}")
    if args.bedmachine_nc and (not os.path.isfile(args.bedmachine_nc)):
        raise RuntimeError(f"BedMachine file not found: {args.bedmachine_nc}")
    if args.is2_mat and (not os.path.isfile(args.is2_mat)):
        raise RuntimeError(f"IS2 file not found: {args.is2_mat}")
    if args.tracking_npz and (not os.path.isfile(args.tracking_npz)):
        raise RuntimeError(f"Tracking NPZ file not found: {args.tracking_npz}")

    plot_sit_map(
        date_str=args.date,
        esacci_dir=args.esacci_dir,
        amsr_mat=args.amsr_mat,
        polynya_threshold=args.polynya_threshold,
        polynya_type=args.polynya_type,
        out_png=args.out_png,
        vmin=args.vmin,
        vmax=args.vmax,
        central_lon=args.central_lon,
        buffer_factor=args.buffer_factor,
        global_view=args.global_view,
        filter_quality=not args.no_quality_filter,
        max_uncertainty=args.max_uncertainty,
        ice_motion_nc=args.ice_motion_nc,
        ice_motion_subsample=args.ice_motion_subsample,
        ice_motion_dt_hours=args.ice_motion_dt_hours,
        ice_motion_arrow_gain=args.ice_motion_arrow_gain,
        constrain_ross_sea=args.constrain_ross_sea,
        bedmachine_nc=args.bedmachine_nc,
        show_ice_shelves=args.show_ice_shelves,
        show_grounding_line=args.show_grounding_line,
        bedmachine_subsample=args.bedmachine_subsample,
        is2_mat=args.is2_mat,
        show_is2_tracks=args.show_is2_tracks,
        is2_track_color=args.is2_track_color,
        is2_track_alpha=args.is2_track_alpha,
        is2_track_linewidth=args.is2_track_linewidth,
        is2_use_scatter=args.is2_use_scatter and not args.is2_use_lines,
        is2_scatter_size=args.is2_scatter_size,
        is2_colormap=args.is2_colormap,
        is2_color_by=args.is2_color_by if args.is2_color_by != "none" else None,
        backtrack_is2=args.backtrack_is2,
        swot_start_time_str=args.swot_start_time,
        swot_end_time_str=args.swot_end_time,
        backtrack_dt_hours=args.backtrack_dt_hours,
        tracking_npz=args.tracking_npz,
        show_tracking_start=(args.show_tracking_start or args.tracking_npz is not None) and not args.no_tracking_start,
        show_tracking_end=(args.show_tracking_end or args.tracking_npz is not None) and not args.no_tracking_end,
        tracking_start_color=args.tracking_start_color,
        tracking_end_color=args.tracking_end_color,
        tracking_marker_size=args.tracking_marker_size,
        tracking_alpha=args.tracking_alpha,
        tracking_subsample=args.tracking_subsample,
        debug=args.debug,
    )
    
    # Create zoomed tracking plot if requested
    if args.zoom_tracking and args.tracking_npz:
        zoom_out = args.zoom_out_png
        if zoom_out is None:
            # Add '_zoom' before the file extension
            base, ext = os.path.splitext(args.out_png)
            zoom_out = f"{base}_zoom{ext}"
        
        # Parse date for polynya data
        from datetime import datetime
        try:
            target_date = datetime.strptime(args.date, "%Y%m%d")
        except:
            target_date = datetime(2023, 10, 28)  # Default
        
        plot_zoomed_tracking(
            tracking_npz=args.tracking_npz,
            out_png=zoom_out,
            lat_range=(-78, -76),
            lon_range=(165, 190),
            marker_size=40,
            alpha=0.3,
            amsr_mat=args.amsr_mat if hasattr(args, 'amsr_mat') else None,
            target_date=target_date,
            polynya_threshold=args.polynya_threshold if hasattr(args, 'polynya_threshold') else 0.7,
            polynya_dot_size=80,
            polynya_dot_stride=1,
            target_points=1000,
            bedmachine_nc=args.bedmachine_nc if hasattr(args, 'bedmachine_nc') else None,
            show_ice_shelves=args.show_ice_shelves if hasattr(args, 'show_ice_shelves') else False,
            show_grounding_line=args.show_grounding_line if hasattr(args, 'show_grounding_line') else False,
            ice_motion_nc=args.ice_motion_nc if hasattr(args, 'ice_motion_nc') else None,
            ice_motion_subsample=args.ice_motion_subsample if hasattr(args, 'ice_motion_subsample') else 2,
            ice_motion_arrow_gain=args.ice_motion_arrow_gain if hasattr(args, 'ice_motion_arrow_gain') else 5.0,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
