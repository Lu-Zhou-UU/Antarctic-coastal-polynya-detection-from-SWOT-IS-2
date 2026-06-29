#!/usr/bin/env python3
"""
ice_tracking.py

Forward/backward tracking of ice parcels using ice drift data.

This script can track ice parcels forward or backward in time from initial positions
using daily ice motion data. Supports multiple input formats:
  - ICESat-2 tracks from HDF5 files (ATL07 format)
  - Manual coordinate lists
  - SWOT observation windows

The script can filter ICESat-2 data to the Ross Sea region and perform both
forward and backward tracking using EASE-Grid ice motion data.

Usage:
  # Backward tracking from ICESat-2 to SWOT window (with Ross Sea filtering)
  python ice_tracking.py \
      --mode backward \
      --start_time 20231028T191642 \
      --end_time 20231028T215103 \
      --is2_file /Volumes/Yotta_1/SWOT/ATL07-02_20231028223812_06102101_006_01.h5 \
      --ice_motion_nc /Volumes/Yotta_1/Antarctic/Ice_Drift/icemotion_daily_sh_25km_20230101_20231231_v4.1.nc \
      --filter_ross_sea \
      --out_npz trajectories_backward.npz \
      --out_png trajectories_backward.png

  # Forward tracking from ICESat-2 (with Ross Sea filtering)
  python ice_tracking.py \
      --mode forward \
      --start_time 20231028T000000 \
      --end_time 20231030T000000 \
      --is2_file /Volumes/Yotta_1/SWOT/ATL07-02_20231028223812_06102101_006_01.h5 \
      --ice_motion_nc /Volumes/Yotta_1/Antarctic/Ice_Drift/icemotion_daily_sh_25km_20230101_20231231_v4.1.nc \
      --filter_ross_sea \
      --out_npz trajectories_forward.npz \
      --out_png trajectories_forward.png

  # Forward tracking from manual coordinates
  python ice_tracking.py \
      --mode forward \
      --start_time 20231028T000000 \
      --end_time 20231030T000000 \
      --lat -75.0 -74.5 -74.0 \
      --lon 180.0 180.5 181.0 \
      --ice_motion_nc /Volumes/Yotta_1/Antarctic/Ice_Drift/icemotion_daily_sh_25km_20230101_20231231_v4.1.nc \
      --out_npz trajectories_forward.npz \
      --out_png trajectories_forward.png
"""

import argparse
import os
import re
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# Publication-quality settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
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

try:
    from scipy.interpolate import griddata
    HAS_SCIPY_INTERP = True
except Exception:
    HAS_SCIPY_INTERP = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_longitude_0_360(lon):
    """Normalize longitude to 0-360 range."""
    lon = np.asarray(lon, dtype=np.float64)
    return (lon + 360.0) % 360.0


def wrap_lon180(lon):
    """Normalize longitude to -180 to 180 range."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def ease_grid_south_lonlat_to_xy(lon, lat, grid_size_km=25.0, return_meters=True):
    """
    Convert geographic lat/lon to EASE-Grid South (25km) x,y coordinates.
    
    Based on IDL ssmi_convert function for Southern Hemisphere EASE-Grid.
    
    Parameters
    ----------
    lon : array
        Longitude in degrees (0-360 or -180-180)
    lat : array
        Latitude in degrees
    grid_size_km : float
        Grid cell size in km (default: 25.0 for 25km EASE-Grid)
    return_meters : bool
        If True, return coordinates in meters (default). If False, return grid indices.
    
    Returns
    -------
    x : array
        EASE-Grid x coordinates (meters if return_meters=True, else grid indices)
    y : array
        EASE-Grid y coordinates (meters if return_meters=True, else grid indices)
    """
    # EASE-Grid South parameters (25km resolution)
    RE_km = 6371.228  # Earth radius (authalic sphere)
    CELL_km = 25.067525  # Nominal cell size
    Rg = RE_km / CELL_km  # Grid radius
    map_scale_m = CELL_km * 1000.0  # Convert km to meters: 25067.525 m
    
    # Grid dimensions for 25km EASE-Grid South
    cols = 721
    rows = 721
    r0 = (cols - 1) / 2.0
    s0 = (rows - 1) / 2.0
    
    # Normalize longitude to 0-360
    lon = normalize_longitude_0_360(lon)
    
    # Convert to radians
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    
    # EASE-Grid South forward transformation
    # From IDL: rho = 2.0D * Rg * cos(PI/4 - phi/2)
    rho = 2.0 * Rg * np.cos(np.pi/4.0 - phi/2.0)
    
    # r = r0 + rho * sin(lam)
    # s = s0 - rho * cos(lam)  (negative for Southern Hemisphere)
    r = r0 + rho * np.sin(lam)
    s = s0 - rho * np.cos(lam)
    
    # Convert grid indices to meters if requested
    if return_meters:
        # From IDL wgs84_inverse: x = (r - r0) * map_scale_m, y = (s0 - s) * map_scale_m
        x = (r - r0) * map_scale_m
        y = (s0 - s) * map_scale_m
        return x, y
    else:
        return r, s


def ease_grid_south_xy_to_lonlat(x, y, grid_size_km=25.0, input_meters=True):
    """
    Convert EASE-Grid South (25km) x,y coordinates to geographic lat/lon.
    
    Based on IDL ssmi_inverse function for Southern Hemisphere EASE-Grid.
    
    Parameters
    ----------
    x : array
        EASE-Grid x coordinates (meters if input_meters=True, else grid indices)
    y : array
        EASE-Grid y coordinates (meters if input_meters=True, else grid indices)
    grid_size_km : float
        Grid cell size in km (default: 25.0 for 25km EASE-Grid)
    input_meters : bool
        If True, input coordinates are in meters (default). If False, input are grid indices.
    
    Returns
    -------
    lat : array
        Latitude in degrees
    lon : array
        Longitude in degrees (0-360)
    """
    # EASE-Grid South parameters (25km resolution)
    RE_km = 6371.228  # Earth radius (authalic sphere)
    CELL_km = 25.067525  # Nominal cell size
    Rg = RE_km / CELL_km  # Grid radius
    map_scale_m = CELL_km * 1000.0  # Convert km to meters: 25067.525 m
    
    # Grid dimensions for 25km EASE-Grid South
    cols = 721
    rows = 721
    r0 = (cols - 1) / 2.0
    s0 = (rows - 1) / 2.0
    
    # Convert from meters to grid indices if needed
    if input_meters:
        # From IDL wgs84_inverse: r = r0 + (x / map_scale_m), s = s0 - (y / map_scale_m)
        r = r0 + (np.asarray(x, dtype=float) / map_scale_m)
        s = s0 - (np.asarray(y, dtype=float) / map_scale_m)
    else:
        r = np.asarray(x, dtype=float)
        s = np.asarray(y, dtype=float)
    
    # Compute relative coordinates
    x_rel = r - r0
    y_rel = -(s - s0)  # Note: negative for Southern Hemisphere
    
    # Compute distance from pole
    rho = np.sqrt(x_rel**2 + y_rel**2)
    
    # Handle pole case
    lat = np.full_like(rho, np.nan)
    lon = np.full_like(rho, np.nan)
    
    pole_mask = rho == 0.0
    lat[pole_mask] = -90.0
    lon[pole_mask] = 0.0
    
    # Compute for non-pole points
    non_pole = ~pole_mask
    if np.any(non_pole):
        # Longitude
        y_rel_np = y_rel[non_pole]
        x_rel_np = x_rel[non_pole]
        r_np = r[non_pole]
        
        lon_np = np.full_like(y_rel_np, np.nan)
        zero_y = (y_rel_np == 0.0)
        lon_np[zero_y] = np.where(r_np[zero_y] <= r0, -90.0, 90.0)
        lon_np[~zero_y] = np.arctan2(x_rel_np[~zero_y], y_rel_np[~zero_y])
        
        # Latitude
        gamma = rho[non_pole] / (2.0 * Rg)
        valid_gamma = np.abs(gamma) <= 1.0
        if np.any(valid_gamma):
            c = 2.0 * np.arcsin(gamma[valid_gamma])
            sinphi1 = np.sin(-np.pi / 2.0)  # South pole
            cosphi1 = np.cos(-np.pi / 2.0)
            beta = np.cos(c) * sinphi1 + y_rel_np[valid_gamma] * np.sin(c) * (cosphi1 / rho[non_pole][valid_gamma])
            beta = np.clip(beta, -1.0, 1.0)
            phi = np.arcsin(beta)
            lat_np = np.full_like(gamma, np.nan)
            lat_np[valid_gamma] = np.rad2deg(phi)
            
            lat[non_pole] = lat_np
            lon[non_pole] = np.rad2deg(lon_np)
    
    # Normalize longitude to 0-360
    lon = normalize_longitude_0_360(lon)
    
    return lat, lon


def transform_ease_grid_to_geographic(u_grid, v_grid, lon_deg, lat_deg):
    """
    Transform EASE-Grid South velocity components to geographic (eastward, northward).
    
    For EASE-Grid South (Lambert Azimuthal Equal Area centered at South Pole):
    - The grid x-axis points eastward at longitude 0°
    - The grid y-axis points northward at longitude 0°
    - At other longitudes, the grid rotates relative to geographic coordinates
    
    This transformation is based on the EASE-Grid coordinate system where:
    - Grid coordinates (x, y) are in the EASE-Grid projection space
    - Geographic coordinates (lat, lon) are on the sphere
    - Velocity components need to be rotated by the longitude angle to convert
      from grid-aligned (along x, y axes) to geographic (eastward, northward)
    
    Parameters
    ----------
    u_grid : array
        Along-x component (EASE-Grid coordinates, cm/s or m/s)
    v_grid : array
        Along-y component (EASE-Grid coordinates, cm/s or m/s)
    lon_deg : array
        Longitude in degrees (for rotation angle)
    lat_deg : array
        Latitude in degrees (not directly used in velocity transformation,
        but kept for consistency with EASE-Grid coordinate system)
    
    Returns
    -------
    u_east : array
        Eastward velocity component (same units as input)
    v_north : array
        Northward velocity component (same units as input)
    
    Notes
    -----
    The transformation formula is:
        u_east = u_grid * sin(lon) + v_grid * cos(lon)
        v_north = u_grid * cos(lon) - v_grid * sin(lon)
    
    This accounts for the rotation of the EASE-Grid coordinate system relative
    to geographic coordinates as we move around the South Pole.
    """
    # Convert longitude to radians
    lon_rad = np.deg2rad(lon_deg)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    
    # For EASE-Grid South (centered at South Pole):
    # The transformation accounts for the rotation of the grid relative to
    # geographic coordinates as we move around the pole
    # 
    # Note: This transformation matches the working code that produces correct vectors
    # The formula rotates grid-aligned (x,y) to geographic (east, north)
    # IMPORTANT: Variable assignments match the confirmed working version
    v_north = u_grid * sin_lon + v_grid * cos_lon
    u_east = u_grid * cos_lon - v_grid * sin_lon
    
    return u_east, v_north


def transform_grid_to_geographic(u_grid, v_grid, lon_deg, lat_deg=None):
    """
    Transform grid-aligned velocities to geographic (eastward, northward).
    
    This is a wrapper that uses EASE-Grid specific transformation when appropriate.
    
    Parameters
    ----------
    u_grid : array
        Along-x component (grid coordinates)
    v_grid : array
        Along-y component (grid coordinates)
    lon_deg : array
        Longitude in degrees
    lat_deg : array, optional
        Latitude in degrees (for EASE-Grid transformation)
    
    Returns
    -------
    u_east : array
        Eastward velocity component (m/s)
    v_north : array
        Northward velocity component (m/s)
    """
    # Use EASE-Grid transformation (works for EASE-Grid South)
    if lat_deg is not None:
        return transform_ease_grid_to_geographic(u_grid, v_grid, lon_deg, lat_deg)
    else:
        # Fallback to simple rotation (less accurate but works if lat not available)
        # Note: This matches the confirmed working transformation formula
        lon_rad = np.deg2rad(lon_deg)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)
        
        # IMPORTANT: Variable assignments match the confirmed working version
        v_north = u_grid * sin_lon + v_grid * cos_lon
        u_east = u_grid * cos_lon - v_grid * sin_lon
        return u_east, v_north


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


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great circle distance between two points on Earth using Haversine formula.
    
    Parameters
    ----------
    lon1, lat1 : array
        Longitude and latitude of first point (degrees)
    lon2, lat2 : array
        Longitude and latitude of second point (degrees)
    
    Returns
    -------
    distance : array
        Distance in meters
    """
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    lon1_rad = np.deg2rad(lon1)
    lat1_rad = np.deg2rad(lat1)
    lon2_rad = np.deg2rad(lon2)
    lat2_rad = np.deg2rad(lat2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance = R * c
    return distance


def displace_lonlat(lon_deg, lat_deg, u_east_ms, v_north_ms, dt_seconds):
    """
    Move (lon,lat) by (u,v) over dt using small-angle spherical approximation.
    
    Parameters
    ----------
    lon_deg : array
        Longitude in degrees
    lat_deg : array
        Latitude in degrees
    u_east_ms : array
        Eastward velocity in m/s
    v_north_ms : array
        Northward velocity in m/s
    dt_seconds : float
        Time step in seconds
    
    Returns
    -------
    lon2 : array
        New longitude in degrees
    lat2 : array
        New latitude in degrees
    """
    R = 6371000.0  # Earth radius in meters
    lat_rad = np.deg2rad(lat_deg)
    
    dlat = (v_north_ms * dt_seconds) / R
    dlon = (u_east_ms * dt_seconds) / (R * np.cos(lat_rad))
    
    lat2 = lat_deg + np.rad2deg(dlat)
    lon2 = lon_deg + np.rad2deg(dlon)
    lon2 = wrap_lon180(lon2)
    return lon2, lat2


def _h5_read_dataset(ds):
    """Read HDF5 dataset and reverse axes to match MATLAB ordering."""
    arr = np.array(ds)
    if arr.shape == ():
        return arr
    if arr.ndim >= 2:
        arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
    return arr


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
    """Unwrap common MATLAB loadmat wrappers."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.shape == () and x.dtype != object:
            return x.item()
        if x.dtype == object and x.size == 1:
            return _unwrap_mat_scalar(x.flat[0])
        if x.dtype == object and x.shape == ():
            return _unwrap_mat_scalar(x.item())
    return x


def _mat_get_field(obj, field, default=None):
    """Robustly get a field from MATLAB struct."""
    obj = _unwrap_mat_scalar(obj)
    
    if obj is None:
        return default
    
    if isinstance(obj, dict):
        return obj.get(field, default)
    
    if hasattr(obj, "_fieldnames"):
        if field in obj._fieldnames:
            return getattr(obj, field)
        return default
    
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        if field in obj.dtype.names:
            return obj[field]
        return default
    
    return default


# ============================================================================
# Data Loading Functions
# ============================================================================

def filter_ross_sea(lat, lon, debug=False):
    """
    Filter coordinates to Ross Sea region.
    
    Parameters
    ----------
    lat : array
        Latitude in degrees
    lon : array
        Longitude in degrees (any range)
    debug : bool
        Print debug information
    
    Returns
    -------
    mask : array
        Boolean mask (True for points in Ross Sea)
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lon_0360 = normalize_longitude_0_360(lon)
    
    # Ross Sea bounds (approximate)
    # Longitude: 160°E to 220°E (or -140°W to -20°W)
    # Latitude: -78°S to -70°S
    ross_lon_min = 160.0
    ross_lon_max = 220.0
    ross_lat_min = -78.0
    ross_lat_max = -70.0
    
    mask = (
        (lat >= ross_lat_min) & (lat <= ross_lat_max) &
        (lon_0360 >= ross_lon_min) & (lon_0360 <= ross_lon_max)
    )
    
    if debug:
        n_total = lat.size
        n_ross = np.sum(mask)
        print(f"[ROSS_SEA] Filtered {n_ross}/{n_total} points within Ross Sea region")
        if n_ross > 0:
            print(f"[ROSS_SEA] Lat range: [{lat[mask].min():.2f}, {lat[mask].max():.2f}]")
            print(f"[ROSS_SEA] Lon range: [{lon_0360[mask].min():.2f}, {lon_0360[mask].max():.2f}]")
    
    return mask


def load_is2_tracks(is2_file, beams=None, target_date=None, filter_ross_sea_region=False, debug=False):
    """
    Load ICESat-2 tracks from HDF5 file (ATL07 format).
    
    Parameters
    ----------
    is2_file : str
        Path to ICESat-2 HDF5 file (e.g., ATL07-02_*.h5)
    beams : list, optional
        List of beam names to load (e.g., ['gt1l', 'gt1r', ...])
        If None, loads all available beams
    target_date : datetime, optional
        Target date for filtering
    filter_ross_sea_region : bool
        Filter to Ross Sea region only
    debug : bool
        Print debug information
    
    Returns
    -------
    dict:
        {
            'lat': array,      # Combined latitude array
            'lon': array,      # Combined longitude array
            'time': datetime,  # Observation time (or array if per-point)
            'height': array,   # Ice height (optional)
            'delta_time': array,  # Time offset from reference (seconds)
            'beams': dict      # Per-beam data
        }
    """
    if not os.path.exists(is2_file):
        raise FileNotFoundError(f"IS2 file not found: {is2_file}")
    
    if not HAS_H5PY:
        raise RuntimeError("h5py is required to read ICESat-2 HDF5 files")
    
    if debug:
        print(f"[IS2] Loading from HDF5: {is2_file}")
    
    # Default beam names for ICESat-2
    default_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    
    if beams is None:
        beams = default_beams
    
    all_lat = []
    all_lon = []
    all_height = []
    all_delta_time = []
    beam_data = {}
    
    with h5py.File(is2_file, 'r') as f:
        # Check available beams
        available_beams = []
        for beam in beams:
            beam_path = f"{beam}/sea_ice_segments"
            if beam_path in f:
                available_beams.append(beam)
        
        if len(available_beams) == 0:
            raise ValueError(f"No valid ICESat-2 beams found in {is2_file}")
        
        if debug:
            print(f"[IS2] Loading beams: {available_beams}")
        
        # Try to get reference time from metadata
        ref_time = None
        try:
            # Check for reference time in metadata
            if 'ancillary_data' in f:
                anc = f['ancillary_data']
                if 'atlas_sdp_gps_epoch' in anc:
                    ref_time_str = anc['atlas_sdp_gps_epoch'][()].decode('utf-8') if isinstance(anc['atlas_sdp_gps_epoch'][()], bytes) else str(anc['atlas_sdp_gps_epoch'][()])
                    # GPS epoch is typically 1980-01-06 00:00:00 UTC
                    # But we'll try to parse it
                    try:
                        ref_time = datetime.strptime(ref_time_str, "%Y-%m-%d %H:%M:%S")
                    except:
                        ref_time = datetime(1980, 1, 6, 0, 0, 0)  # Default GPS epoch
        except:
            pass
        
        if ref_time is None:
            # Try to extract from filename
            match = re.search(r'(\d{8})', os.path.basename(is2_file))
            if match:
                try:
                    ref_time = datetime.strptime(match.group(1), "%Y%m%d")
                except:
                    pass
        
        if ref_time is None:
            ref_time = datetime(1980, 1, 6, 0, 0, 0)  # Default GPS epoch
        
        for beam in available_beams:
            beam_path = f"{beam}/sea_ice_segments"
            
            try:
                # Read latitude
                lat_path = f"{beam_path}/latitude"
                if lat_path not in f:
                    if debug:
                        print(f"[IS2] Warning: {lat_path} not found for beam {beam}")
                    continue
                lat = np.array(f[lat_path][:], dtype=float)
                
                # Read longitude
                lon_path = f"{beam_path}/longitude"
                if lon_path not in f:
                    if debug:
                        print(f"[IS2] Warning: {lon_path} not found for beam {beam}")
                    continue
                lon = np.array(f[lon_path][:], dtype=float)
                
                # Read delta_time (time offset from reference, in seconds)
                delta_time = None
                time_path = f"{beam_path}/delta_time"
                if time_path in f:
                    delta_time = np.array(f[time_path][:], dtype=float)
                
                # Read height (if available)
                height = None
                height_paths = [
                    f"{beam_path}/height_segment_sea_ice",
                    f"{beam_path}/h_sea_ice",
                    f"{beam_path}/height_ice2",
                ]
                for hp in height_paths:
                    if hp in f:
                        height = np.array(f[hp][:], dtype=float)
                        break
                
                # Filter invalid values
                valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
                lat = lat[valid]
                lon = lon[valid]
                if delta_time is not None:
                    delta_time = delta_time[valid]
                if height is not None:
                    height = height[valid]
                
                if lat.size == 0:
                    if debug:
                        print(f"[IS2] Warning: No valid points for beam {beam}")
                    continue
                
                # Filter to Ross Sea if requested
                if filter_ross_sea_region:
                    ross_mask = filter_ross_sea(lat, lon, debug=False)
                    lat = lat[ross_mask]
                    lon = lon[ross_mask]
                    if delta_time is not None:
                        delta_time = delta_time[ross_mask]
                    if height is not None:
                        height = height[ross_mask]
                    
                    if lat.size == 0:
                        if debug:
                            print(f"[IS2] Warning: No points in Ross Sea for beam {beam}")
                        continue
                
                all_lat.append(lat)
                all_lon.append(lon)
                if height is not None:
                    all_height.append(height)
                if delta_time is not None:
                    all_delta_time.append(delta_time)
                
                beam_data[beam] = {
                    'lat': lat,
                    'lon': lon,
                    'height': height,
                    'delta_time': delta_time
                }
                
            except Exception as e:
                if debug:
                    print(f"[IS2] Warning: Error loading beam {beam}: {e}")
                continue
    
    if len(all_lat) == 0:
        raise ValueError("No valid IS2 data found")
    
    # Combine all beams
    lat_combined = np.concatenate(all_lat)
    lon_combined = np.concatenate(all_lon)
    
    # Compute observation time
    time_obs = None
    if len(all_delta_time) > 0:
        # Use delta_time to compute actual observation times
        delta_time_combined = np.concatenate(all_delta_time)
        # delta_time is typically in seconds since GPS epoch (1980-01-06)
        # But it might be in days since 2000-01-01 or other reference
        # For now, use the first delta_time as reference
        if target_date:
            time_obs = target_date
        else:
            # Estimate from delta_time (assuming seconds since GPS epoch)
            # GPS epoch: 1980-01-06 00:00:00 UTC
            gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
            # Use median delta_time to estimate observation time
            median_delta = np.nanmedian(delta_time_combined)
            time_obs = gps_epoch + timedelta(seconds=median_delta)
    else:
        # Fallback to filename or target_date
        if target_date:
            time_obs = target_date
        else:
            match = re.search(r'(\d{8})', os.path.basename(is2_file))
            if match:
                try:
                    time_obs = datetime.strptime(match.group(1), "%Y%m%d")
                except:
                    time_obs = datetime.now()
            else:
                time_obs = datetime.now()
    
    if debug:
        print(f"[IS2] Loaded {len(lat_combined)} points from {len(beam_data)} beams")
        print(f"[IS2] Lat range: [{lat_combined.min():.2f}, {lat_combined.max():.2f}]")
        print(f"[IS2] Lon range: [{lon_combined.min():.2f}, {lon_combined.max():.2f}]")
        print(f"[IS2] Observation time: {time_obs}")
        if filter_ross_sea_region:
            print(f"[IS2] Filtered to Ross Sea region")
    
    return {
        'lat': lat_combined,
        'lon': lon_combined,
        'time': time_obs,
        'height': np.concatenate(all_height) if all_height else None,
        'delta_time': np.concatenate(all_delta_time) if all_delta_time else None,
        'beams': beam_data
    }


# ============================================================================
# Tracking Functions
# ============================================================================

def track_ice_parcels(
    lat_init, lon_init, time_init,
    ice_motion_nc, time_start, time_end,
    direction='backward', dt_hours=1.0, debug=False
):
    """
    Track ice parcels forward or backward in time using ice drift data.
    
    Parameters
    ----------
    lat_init : array
        Initial latitudes (degrees)
    lon_init : array
        Initial longitudes (degrees, any range)
    time_init : datetime
        Initial observation time
    ice_motion_nc : str
        Path to ice motion NetCDF file
    time_start : datetime
        Start of tracking period
    time_end : datetime
        End of tracking period
    direction : str
        'forward' or 'backward' tracking
    dt_hours : float
        Time step for integration (hours, default: 1.0)
    debug : bool
        Print debug information
    
    Returns
    -------
    dict:
        {
            'trajectories': list,      # List of (time, lat, lon) arrays for each point
            'times': array,            # All time steps
            'valid': array,            # Boolean mask of valid tracked points
            'lat_final': array,        # Final positions
            'lon_final': array,
            'lat_start': array,        # Start positions (at time_start)
            'lon_start': array
        }
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is required for tracking")
    
    if not HAS_SCIPY_INTERP:
        raise RuntimeError("scipy.interpolate is required for tracking")
    
    # Normalize inputs
    lat_init = np.asarray(lat_init, dtype=float)
    lon_init = np.asarray(lon_init, dtype=float)
    lon_init_0360 = normalize_longitude_0_360(lon_init)
    
    n_points = lat_init.size
    if debug:
        print(f"[TRACK] Starting {direction} tracking for {n_points} points")
        print(f"[TRACK] Initial time: {time_init}")
        print(f"[TRACK] Tracking period: {time_start} to {time_end}")
    
    # Determine time range
    if direction == 'backward':
        # Track backwards from time_init to time_start
        t_current = time_init
        t_target = time_start
        time_steps = []
        t = t_current
        while t >= t_target:
            time_steps.append(t)
            if t <= t_target:
                break  # Reached target, don't go further
            t -= timedelta(hours=dt_hours)
        
        # Ensure we include the target time if not already included
        if len(time_steps) > 0 and time_steps[-1] > t_target:
            time_steps.append(t_target)
        
        time_steps.reverse()  # Now in forward chronological order
    else:  # forward
        # Track forward from time_init to time_end
        t_current = time_init
        t_target = time_end
        time_steps = []
        t = t_current
        while t <= t_target:
            time_steps.append(t)
            if t >= t_target:
                break  # Reached target, don't go further
            t += timedelta(hours=dt_hours)
        
        # Ensure we include the target time if not already included
        if len(time_steps) > 0 and time_steps[-1] < t_target:
            time_steps.append(t_target)
    
    if debug:
        print(f"[TRACK] Using {len(time_steps)} time steps (dt={dt_hours} hours)")
        if len(time_steps) > 0:
            print(f"[TRACK] Time step range: {time_steps[0]} to {time_steps[-1]}")
            if len(time_steps) > 1:
                actual_dt = (time_steps[-1] - time_steps[0]).total_seconds() / 3600.0
                print(f"[TRACK] Total tracking period: {actual_dt:.2f} hours")
    
    # Initialize positions
    lat_track = lat_init.copy()
    lon_track = lon_init_0360.copy()
    
    # Store trajectories: list of (time, lat, lon) tuples for each point
    trajectories = []
    for i in range(n_points):
        trajectories.append([(time_init, lat_init[i], lon_init_0360[i])])
    
    # Track valid points
    valid = np.ones(n_points, dtype=bool)
    
    # Open ice motion file
    nc = netCDF4.Dataset(ice_motion_nc, "r")
    
    # Get time variable info
    time_var = nc.variables["time"]
    time_units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    time_vals = np.array(time_var[:], dtype=float)
    
    # Handle time units properly
    # For this dataset: "days since 1970-01-01" with calendar="julian"
    if time_units is None:
        raise RuntimeError("Ice motion file must have time units specified")
    
    if debug:
        print(f"[TRACK] Time units: '{time_units}', calendar: '{calendar}'")
        print(f"[TRACK] Ice motion file contains {len(time_vals)} time steps")
    
    # Convert time values to datetime for display
    if len(time_vals) > 0:
        try:
            # Use netCDF4 date conversion with proper calendar
            time_dates = [netCDF4.num2date(tv, units=time_units, calendar=calendar) for tv in time_vals]
            time_start_file = time_dates[0]
            time_end_file = time_dates[-1]
            if debug:
                print(f"[TRACK] Available time range: {time_start_file} to {time_end_file}")
        except Exception as e:
            if debug:
                print(f"[TRACK] Warning: Could not convert time values to dates: {e}")
                print(f"[TRACK] Time values range: [{time_vals.min():.2f}, {time_vals.max():.2f}]")
    
    # Get EASE-Grid x, y coordinates (native grid coordinates)
    if "x" in nc.variables:
        x_grid = np.array(nc.variables["x"][:], dtype=float)
    else:
        nc.close()
        raise RuntimeError("Ice motion file must contain x coordinate (EASE-Grid)")
    
    if "y" in nc.variables:
        y_grid = np.array(nc.variables["y"][:], dtype=float)
    else:
        nc.close()
        raise RuntimeError("Ice motion file must contain y coordinate (EASE-Grid)")
    
    # Make 2D grid if needed
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        x_grid_2d, y_grid_2d = np.meshgrid(x_grid, y_grid)
    else:
        x_grid_2d = x_grid
        y_grid_2d = y_grid
    
    # Flatten EASE-Grid coordinates for interpolation
    x_grid_flat = x_grid_2d.ravel()
    y_grid_flat = y_grid_2d.ravel()
    
    # Also get lat/lon grid for transformation (needed for velocity transformation)
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
    
    # Integrate through time steps
    for step_idx, t_step in enumerate(time_steps):
        if step_idx == 0:
            continue  # Skip first step (initial time)
        
        # Find closest time in ice motion file
        target_num = netCDF4.date2num(t_step, units=time_units, calendar=calendar)
        time_idx = int(np.nanargmin(np.abs(time_vals - target_num)))
        actual_time_cf = netCDF4.num2date(time_vals[time_idx], units=time_units, calendar=calendar)
        
        # Calculate time difference using numeric values to avoid calendar mismatch
        # cftime.datetime objects can't be directly compared with datetime when calendars differ
        actual_time_num = time_vals[time_idx]
        time_diff_days = abs(actual_time_num - target_num)
        
        # Convert cftime.datetime to regular datetime for display/comparison
        try:
            # Try to convert cftime to regular datetime
            if hasattr(actual_time_cf, 'datetime'):
                actual_time = actual_time_cf.datetime
            elif hasattr(actual_time_cf, 'year'):
                # Manual conversion from cftime attributes
                actual_time = datetime(
                    actual_time_cf.year, actual_time_cf.month, actual_time_cf.day,
                    getattr(actual_time_cf, 'hour', 0),
                    getattr(actual_time_cf, 'minute', 0),
                    getattr(actual_time_cf, 'second', 0)
                )
            else:
                actual_time = actual_time_cf  # Keep original if conversion fails
        except Exception:
            # Fallback: use the cftime object as-is for display
            actual_time = actual_time_cf
        
        # Track which time indices we're using (for detecting daily data limitation)
        if not hasattr(track_ice_parcels, '_used_time_indices'):
            track_ice_parcels._used_time_indices = []
        track_ice_parcels._used_time_indices.append(time_idx)
        
        if debug:
            if step_idx <= 3 or step_idx % 10 == 0 or time_diff_days > 1.0:  # Print first few steps, every 10 steps, or if mismatch > 1 day
                print(f"[TRACK] Step {step_idx}/{len(time_steps)-1}: target={t_step}, "
                      f"using ice motion from day {time_idx+1}/{len(time_vals)}: {actual_time_cf} "
                      f"(diff: {time_diff_days:.2f} days)")
                # Check if we're using the same time index as previous step (would indicate daily data limitation)
                if step_idx > 1:
                    prev_time_idx = track_ice_parcels._used_time_indices[-2]
                    if prev_time_idx == time_idx:
                        print(f"[TRACK]   NOTE: Using same ice motion time index ({time_idx}) as previous step - "
                              f"this is expected for daily ice motion data")
            if time_diff_days > 1.0:
                print(f"[TRACK] Warning: Large time mismatch ({time_diff_days:.2f} days) - "
                      f"ice motion data may not match tracking time step")
        
        # Load u/v for this time
        u_var = nc.variables["u"]
        v_var = nc.variables["v"]
        u = np.array(u_var[time_idx, :, :], dtype=float)
        v = np.array(v_var[time_idx, :, :], dtype=float)
        
        # Get units and metadata
        u_units = getattr(u_var, "units", "").strip()
        v_units = getattr(v_var, "units", "").strip()
        u_fill = getattr(u_var, "_FillValue", None)
        v_fill = getattr(v_var, "_FillValue", None)
        u_valid_min = getattr(u_var, "valid_min", None)
        u_valid_max = getattr(u_var, "valid_max", None)
        v_valid_min = getattr(v_var, "valid_min", None)
        v_valid_max = getattr(v_var, "valid_max", None)
        
        if debug and step_idx == 1:
            print(f"[TRACK] Ice motion units: u='{u_units}', v='{v_units}'")
            print(f"[TRACK] Fill values: u={u_fill}, v={v_fill}")
            print(f"[TRACK] Valid ranges: u=[{u_valid_min}, {u_valid_max}], v=[{v_valid_min}, {v_valid_max}]")
        
        # Handle fill values (typically -9999.0 for this dataset)
        if u_fill is not None:
            u_fill_val = float(u_fill)
            u[np.abs(u - u_fill_val) < 1.0] = np.nan  # Use tolerance for float comparison
        if v_fill is not None:
            v_fill_val = float(v_fill)
            v[np.abs(v - v_fill_val) < 1.0] = np.nan
        
        # Apply valid range filter BEFORE unit conversion
        # Valid range is in original units (cm/s for this dataset: -200 to 200 cm/s)
        if u_valid_min is not None:
            u[u < float(u_valid_min)] = np.nan
        if u_valid_max is not None:
            u[u > float(u_valid_max)] = np.nan
        if v_valid_min is not None:
            v[v < float(v_valid_min)] = np.nan
        if v_valid_max is not None:
            v[v > float(v_valid_max)] = np.nan
        
        # Convert to m/s (from cm/s: multiply by 0.01)
        # Units are "cm / s" for this dataset
        # Keep velocities in grid-aligned coordinates for interpolation in EASE-Grid space
        fu = _units_to_mps(u_units)
        fv = _units_to_mps(v_units)
        u_grid_ms = u * fu  # Grid-aligned u (along-x) in m/s
        v_grid_ms = v * fv  # Grid-aligned v (along-y) in m/s
        
        if debug and step_idx == 1:
            n_valid_u = np.sum(np.isfinite(u_grid_ms))
            n_valid_v = np.sum(np.isfinite(v_grid_ms))
            n_total = u_grid_ms.size
            print(f"[TRACK] After filtering: {n_valid_u}/{n_total} valid u values, {n_valid_v}/{n_total} valid v values")
            if n_valid_u > 0:
                print(f"[TRACK] u range (m/s, grid-aligned): [{np.nanmin(u_grid_ms):.4f}, {np.nanmax(u_grid_ms):.4f}]")
            if n_valid_v > 0:
                print(f"[TRACK] v range (m/s, grid-aligned): [{np.nanmin(v_grid_ms):.4f}, {np.nanmax(v_grid_ms):.4f}]")
            # Check if this is EASE-Grid
            is_ease_grid = False
            if "crs" in nc.variables:
                crs_var = nc.variables["crs"]
                grid_mapping = getattr(crs_var, "grid_mapping_name", "").lower()
                if "lambert_azimuthal" in grid_mapping or "ease" in grid_mapping.lower():
                    is_ease_grid = True
                    print(f"[TRACK] Detected EASE-Grid projection: {grid_mapping}")
                    print(f"[TRACK] Interpolating in EASE-Grid (x, y) space, then transforming to geographic")
        
        # Flatten velocity fields (keep in grid-aligned coordinates)
        # We interpolate in EASE-Grid space, then transform to geographic.
        # User request: treat NaN ice motion as zero and use cubic interpolation.
        # Implement this by replacing NaNs in u/v with 0 BEFORE interpolation,
        # but still only using grid cells where x/y are finite.
        u_grid_ms_safe = np.where(np.isfinite(u_grid_ms), u_grid_ms, 0.0)
        v_grid_ms_safe = np.where(np.isfinite(v_grid_ms), v_grid_ms, 0.0)
        u_grid_flat = u_grid_ms_safe.ravel()  # Grid-aligned u (along-x) in m/s
        v_grid_flat = v_grid_ms_safe.ravel()  # Grid-aligned v (along-y) in m/s
        
        # Filter to only grid points with valid coordinates for interpolation (faster)
        valid_grid = np.isfinite(x_grid_flat) & np.isfinite(y_grid_flat)
        if np.sum(valid_grid) == 0:
            if debug:
                print(f"[TRACK] Warning: No valid grid points for interpolation at step {step_idx}")
            # Mark all points as invalid
            valid[:] = False
            continue
        
        # Convert tracked positions from lat-lon to EASE-Grid x, y coordinates
        # This is critical: interpolate in native EASE-Grid space
        x_track, y_track = ease_grid_south_lonlat_to_xy(lon_track, lat_track)
        
        # Interpolate velocities in EASE-Grid space (x, y coordinates)
        # Use only valid grid points for faster interpolation
        if debug and step_idx % 5 == 1:
            print(f"[TRACK] Interpolating in EASE-Grid (x, y) space for {np.sum(valid)} valid points (step {step_idx}/{len(time_steps)-1})...")
            # Debug: check EASE-Grid coordinate range
            valid_track_mask = valid & np.isfinite(x_track) & np.isfinite(y_track)
            if np.sum(valid_track_mask) > 0:
                print(f"[TRACK]   EASE-Grid x range: [{np.nanmin(x_track[valid_track_mask]):.1f}, {np.nanmax(x_track[valid_track_mask]):.1f}] m")
                print(f"[TRACK]   EASE-Grid y range: [{np.nanmin(y_track[valid_track_mask]):.1f}, {np.nanmax(y_track[valid_track_mask]):.1f}] m")
                print(f"[TRACK]   Grid x range: [{np.nanmin(x_grid_flat[valid_grid]):.1f}, {np.nanmax(x_grid_flat[valid_grid]):.1f}] m")
                print(f"[TRACK]   Grid y range: [{np.nanmin(y_grid_flat[valid_grid]):.1f}, {np.nanmax(y_grid_flat[valid_grid]):.1f}] m")
        
        # Interpolate grid-aligned velocities in EASE-Grid coordinate space
        # Check if tracked points are within grid bounds
        valid_track_mask = valid & np.isfinite(x_track) & np.isfinite(y_track)
        if np.sum(valid_track_mask) > 0:
            x_track_valid = x_track[valid_track_mask]
            y_track_valid = y_track[valid_track_mask]
            
            # Check if points are within grid bounds
            x_grid_min, x_grid_max = np.nanmin(x_grid_flat[valid_grid]), np.nanmax(x_grid_flat[valid_grid])
            y_grid_min, y_grid_max = np.nanmin(y_grid_flat[valid_grid]), np.nanmax(y_grid_flat[valid_grid])
            
            if debug and step_idx == 1:
                print(f"[TRACK]   Grid bounds: x=[{x_grid_min:.1f}, {x_grid_max:.1f}] m, y=[{y_grid_min:.1f}, {y_grid_max:.1f}] m")
                print(f"[TRACK]   Tracked points: x=[{np.nanmin(x_track_valid):.1f}, {np.nanmax(x_track_valid):.1f}] m, "
                      f"y=[{np.nanmin(y_track_valid):.1f}, {np.nanmax(y_track_valid):.1f}] m")
                n_outside = np.sum((x_track_valid < x_grid_min) | (x_track_valid > x_grid_max) |
                                  (y_track_valid < y_grid_min) | (y_track_valid > y_grid_max))
                if n_outside > 0:
                    print(f"[TRACK]   WARNING: {n_outside} tracked points are outside grid bounds!")
        
        # Interpolate grid-aligned velocities using cubic interpolation (fallback to linear if cubic fails)
        interp_points = (x_grid_flat[valid_grid], y_grid_flat[valid_grid])
        try:
            u_grid_interp = griddata(
                interp_points,
                u_grid_flat[valid_grid],
                (x_track, y_track),
                method='cubic',
                fill_value=0.0
            )
            v_grid_interp = griddata(
                interp_points,
                v_grid_flat[valid_grid],
                (x_track, y_track),
                method='cubic',
                fill_value=0.0
            )
            if debug and step_idx == 1:
                print("[TRACK] Using cubic interpolation for ice motion.")
        except Exception as e:
            if debug:
                print(f"[TRACK] Cubic interpolation failed ({e}), falling back to linear.")
            u_grid_interp = griddata(
                interp_points,
                u_grid_flat[valid_grid],
                (x_track, y_track),
                method='linear',
                fill_value=0.0
            )
            v_grid_interp = griddata(
                interp_points,
                v_grid_flat[valid_grid],
                (x_track, y_track),
                method='linear',
                fill_value=0.0
            )
        
        # Check interpolation success rate
        if debug and step_idx == 1:
            n_valid_interp = np.sum(np.isfinite(u_grid_interp) & np.isfinite(v_grid_interp))
            n_total_valid = np.sum(valid)
            print(f"[TRACK]   Interpolation success: {n_valid_interp}/{n_total_valid} points ({100*n_valid_interp/max(n_total_valid,1):.1f}%)")
            if n_valid_interp < n_total_valid * 0.5:
                print(f"[TRACK]   WARNING: Low interpolation success rate! Many points may be outside grid.")
        
        # Debug: check interpolated velocities
        if debug and step_idx == 1:
            valid_interp = np.isfinite(u_grid_interp) & np.isfinite(v_grid_interp)
            if np.sum(valid_interp) > 0:
                print(f"[TRACK]   Interpolated u_grid (m/s): min={np.nanmin(u_grid_interp[valid_interp]):.6f}, "
                      f"max={np.nanmax(u_grid_interp[valid_interp]):.6f}, "
                      f"mean={np.nanmean(u_grid_interp[valid_interp]):.6f}, "
                      f"std={np.nanstd(u_grid_interp[valid_interp]):.6f}")
                print(f"[TRACK]   Interpolated v_grid (m/s): min={np.nanmin(v_grid_interp[valid_interp]):.6f}, "
                      f"max={np.nanmax(v_grid_interp[valid_interp]):.6f}, "
                      f"mean={np.nanmean(v_grid_interp[valid_interp]):.6f}, "
                      f"std={np.nanstd(v_grid_interp[valid_interp]):.6f}")
                # Check if all values are the same (would indicate a problem)
                if np.nanstd(u_grid_interp[valid_interp]) < 1e-10:
                    print(f"[TRACK]   WARNING: All u_grid_interp values are identical!")
                if np.nanstd(v_grid_interp[valid_interp]) < 1e-10:
                    print(f"[TRACK]   WARNING: All v_grid_interp values are identical!")
        
        # Now transform interpolated grid-aligned velocities to geographic (eastward/northward)
        # This transformation accounts for the rotation of EASE-Grid relative to geographic coordinates
        u_interp, v_interp = transform_ease_grid_to_geographic(
            u_grid_interp, v_grid_interp, 
            lon_deg=lon_track, lat_deg=lat_track
        )
        
        # Debug: check transformed velocities
        if debug and step_idx == 1:
            valid_interp = np.isfinite(u_interp) & np.isfinite(v_interp)
            if np.sum(valid_interp) > 0:
                print(f"[TRACK]   Transformed u (east, m/s): min={np.nanmin(u_interp[valid_interp]):.6f}, "
                      f"max={np.nanmax(u_interp[valid_interp]):.6f}, "
                      f"mean={np.nanmean(u_interp[valid_interp]):.6f}, "
                      f"std={np.nanstd(u_interp[valid_interp]):.6f}")
                print(f"[TRACK]   Transformed v (north, m/s): min={np.nanmin(v_interp[valid_interp]):.6f}, "
                      f"max={np.nanmax(v_interp[valid_interp]):.6f}, "
                      f"mean={np.nanmean(v_interp[valid_interp]):.6f}, "
                      f"std={np.nanstd(v_interp[valid_interp]):.6f}")
        
        # Update valid mask
        valid = valid & np.isfinite(u_interp) & np.isfinite(v_interp) & np.isfinite(lat_track) & np.isfinite(lon_track)
        
        # Calculate time step (use actual time difference, not fixed dt_hours)
        if step_idx > 0:
            t_prev = time_steps[step_idx - 1]
            t_curr = time_steps[step_idx]
            if direction == 'backward':
                dt_sec = -(t_curr - t_prev).total_seconds()  # Negative for backwards
            else:
                dt_sec = (t_curr - t_prev).total_seconds()   # Positive for forwards
        else:
            # First step: use default dt
            if direction == 'backward':
                dt_sec = -dt_hours * 3600.0
            else:
                dt_sec = dt_hours * 3600.0
        
        # Update positions
        if debug and step_idx == 1:
            n_valid_before = np.sum(valid)
            if n_valid_before > 0:
                # Sample a few points to see actual displacement
                sample_idx = np.where(valid)[0][:3]
                lon_before = lon_track[sample_idx].copy()
                lat_before = lat_track[sample_idx].copy()
                u_sample = u_interp[sample_idx]
                v_sample = v_interp[sample_idx]
                print(f"[TRACK]   Sample displacements (step {step_idx}):")
                for idx, (lon_b, lat_b, u_s, v_s) in zip(sample_idx, zip(lon_before, lat_before, u_sample, v_sample)):
                    print(f"    Point {idx}: lon={lon_b:.4f}, lat={lat_b:.4f}, u={u_s*100:.4f} cm/s, v={v_s*100:.4f} cm/s")
        
        lon_track[valid], lat_track[valid] = displace_lonlat(
            lon_track[valid], lat_track[valid],
            u_interp[valid], v_interp[valid],
            dt_sec
        )
        
        if debug and step_idx == 1:
            n_valid_after = np.sum(valid)
            if n_valid_after > 0 and n_valid_before > 0:
                lon_after = lon_track[sample_idx]
                lat_after = lat_track[sample_idx]
                print(f"[TRACK]   After displacement:")
                for idx, (lon_a, lat_a) in zip(sample_idx, zip(lon_after, lat_after)):
                    dist = haversine_distance(lon_before[idx], lat_before[idx], lon_a, lat_a)
                    print(f"    Point {idx}: lon={lon_a:.4f}, lat={lat_a:.4f}, moved {dist:.2f} m")
        
        # Normalize longitude
        lon_track = normalize_longitude_0_360(lon_track)
        
        # Store trajectory
        for i in range(n_points):
            if valid[i]:
                trajectories[i].append((t_step, lat_track[i], lon_track[i]))
    
    nc.close()
    
    # Report on time indices used (for debugging daily data limitation)
    if debug and hasattr(track_ice_parcels, '_used_time_indices'):
        unique_indices = sorted(set(track_ice_parcels._used_time_indices))
        print(f"[TRACK] Used {len(unique_indices)} unique ice motion time indices: {unique_indices}")
        if len(unique_indices) == 1:
            print(f"[TRACK]   WARNING: All tracking steps used the same daily ice motion field!")
            print(f"[TRACK]   This is expected for daily data - velocities will be similar across all steps within the same day.")
        delattr(track_ice_parcels, '_used_time_indices')
    
    # Extract final and start positions
    lat_final = np.full(n_points, np.nan)
    lon_final = np.full(n_points, np.nan)
    lat_start = np.full(n_points, np.nan)
    lon_start = np.full(n_points, np.nan)
    
    for i in range(n_points):
        if valid[i] and len(trajectories[i]) > 0:
            # Final position: last in trajectory (earliest time for backward tracking)
            _, lat_final[i], lon_final[i] = trajectories[i][-1]
            # Start position: first in trajectory (latest time for backward tracking)
            _, lat_start[i], lon_start[i] = trajectories[i][0]
    
    if debug:
        n_valid = np.sum(valid)
        print(f"[TRACK] Completed: {n_valid}/{n_points} points successfully tracked")
    
    # Extract all times
    all_times = sorted(set(t for traj in trajectories for t, _, _ in traj))
    
    # Calculate distances traveled and mean ice motion
    distances = np.full(n_points, np.nan)
    mean_velocities = np.full(n_points, np.nan)
    total_times = np.full(n_points, np.nan)
    
    # Calculate total tracking time from time_start to time_end
    if isinstance(time_start, datetime) and isinstance(time_end, datetime):
        total_tracking_time_sec = abs((time_end - time_start).total_seconds())
    else:
        # Fallback: use time steps
        if len(time_steps) > 1:
            total_tracking_time_sec = abs((time_steps[-1] - time_steps[0]).total_seconds())
        else:
            total_tracking_time_sec = 0.0
    
    # Sample a few trajectories for debug output
    sample_indices = []
    if debug and n_points > 0:
        # Sample first, middle, and last valid trajectories
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            sample_indices = [
                valid_indices[0],
                valid_indices[len(valid_indices)//2],
                valid_indices[-1]
            ][:3]
    
    for i in range(n_points):
        if valid[i] and len(trajectories[i]) > 1:
            # Calculate total distance traveled
            traj = trajectories[i]
            
            # Calculate distance from start to end position
            t_start, lat_start_pt, lon_start_pt = traj[0]
            t_end, lat_end_pt, lon_end_pt = traj[-1]
            
            # Calculate cumulative distance along path (sum of all segments)
            cumulative_dist = 0.0
            for j in range(len(traj) - 1):
                _, lat1, lon1 = traj[j]
                _, lat2, lon2 = traj[j + 1]
                dist = haversine_distance(lon1, lat1, lon2, lat2)
                cumulative_dist += dist
            
            # Also calculate straight-line distance from start to end
            straight_dist = haversine_distance(lon_start_pt, lat_start_pt, lon_end_pt, lat_end_pt)
            
            # Use cumulative distance (more accurate for curved paths)
            distances[i] = cumulative_dist
            total_times[i] = total_tracking_time_sec
            
            # Mean velocity (m/s) = total distance / total time
            if total_tracking_time_sec > 0:
                mean_velocities[i] = cumulative_dist / total_tracking_time_sec
            else:
                mean_velocities[i] = np.nan
            
            # Debug output for sample trajectories
            if i in sample_indices and debug:
                print(f"[TRACK] Sample trajectory #{i}:")
                print(f"  Start: ({lon_start_pt:.4f}, {lat_start_pt:.4f}) at {t_start}")
                print(f"  End: ({lon_end_pt:.4f}, {lat_end_pt:.4f}) at {t_end}")
                print(f"  Cumulative distance: {cumulative_dist/1000:.4f} km")
                print(f"  Straight-line distance: {straight_dist/1000:.4f} km")
                print(f"  Mean velocity: {mean_velocities[i]*100:.2f} cm/s")
                print(f"  Trajectory length: {len(traj)} points")
                # Check if positions actually changed
                if cumulative_dist < 1.0:  # Less than 1 meter
                    print(f"  WARNING: Very small movement - positions may not have changed")
    
    # Filter to Ross Sea region for statistics
    ross_sea_mask = filter_ross_sea(lat_final, lon_final, debug=False)
    ross_sea_distances = distances[ross_sea_mask & valid & np.isfinite(distances)]
    ross_sea_velocities = mean_velocities[ross_sea_mask & valid & np.isfinite(mean_velocities)]
    
    if debug:
        n_ross_sea = np.sum(ross_sea_mask & valid)
        n_ross_sea_with_dist = len(ross_sea_distances)
        n_ross_sea_with_vel = len(ross_sea_velocities)
        print(f"[TRACK] Ross Sea statistics (from {n_ross_sea} points):")
        if n_ross_sea_with_dist > 0:
            dist_min = np.nanmin(ross_sea_distances)
            dist_max = np.nanmax(ross_sea_distances)
            dist_mean = np.nanmean(ross_sea_distances)
            dist_std = np.nanstd(ross_sea_distances)
            print(f"[TRACK]   Distance traveled: min={dist_min/1000:.4f} km, "
                  f"max={dist_max/1000:.4f} km, "
                  f"mean={dist_mean/1000:.4f} km, "
                  f"std={dist_std/1000:.4f} km")
            if dist_min == dist_max:
                print(f"[TRACK]   WARNING: All distances are identical! This suggests a calculation issue.")
        else:
            print(f"[TRACK]   Distance traveled: No valid distances in Ross Sea")
        
        if n_ross_sea_with_vel > 0:
            vel_min = np.nanmin(ross_sea_velocities)
            vel_max = np.nanmax(ross_sea_velocities)
            vel_mean = np.nanmean(ross_sea_velocities)
            vel_std = np.nanstd(ross_sea_velocities)
            print(f"[TRACK]   Mean ice motion: min={vel_min*100:.4f} cm/s, "
                  f"max={vel_max*100:.4f} cm/s, "
                  f"mean={vel_mean*100:.4f} cm/s, "
                  f"std={vel_std*100:.4f} cm/s")
            if vel_min == vel_max:
                print(f"[TRACK]   WARNING: All velocities are identical! This suggests a calculation issue.")
        else:
            print(f"[TRACK]   Mean ice motion: No valid velocities in Ross Sea")
            if debug:
                print(f"[TRACK]   Debug: total_tracking_time_sec={total_tracking_time_sec:.1f} s")
                print(f"[TRACK]   Debug: valid velocities count={np.sum(np.isfinite(mean_velocities))}")
    
    return {
        'trajectories': trajectories,
        'times': all_times,
        'valid': valid,
        'lat_final': lat_final,
        'lon_final': lon_final,
        'lat_start': lat_start,
        'lon_start': lon_start,
        'distances': distances,  # Total distance traveled (meters)
        'mean_velocities': mean_velocities,  # Mean velocity along track (m/s)
        'total_times': total_times,  # Total tracking time (seconds)
        'ross_sea_mask': ross_sea_mask  # Boolean mask for Ross Sea region
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_trajectories(
    tracking_result, out_png=None, title=None,
    show_start=True, show_end=True, show_path=True,
    constrain_ross_sea=False, plot_subsample=1, 
    plot_final_only=False, plot_one_example=False,
    simple_xy=False, debug=False
):
    """
    Plot tracked trajectories on a map.
    
    Parameters
    ----------
    tracking_result : dict
        Result from track_ice_parcels()
    out_png : str, optional
        Output PNG file path
    title : str, optional
        Plot title
    show_start : bool
        Show start positions
    show_end : bool
        Show end positions
    show_path : bool
        Show trajectory paths
    constrain_ross_sea : bool
        Constrain map to Ross Sea region
    debug : bool
        Print debug information
    """
    # Use simple X-Y plot if requested or if cartopy not available
    if simple_xy or not HAS_CARTOPY:
        if simple_xy and HAS_CARTOPY:
            print("[PLOT] Using simple X-Y (lat-lon) plot instead of map projection")
        
        if not HAS_CARTOPY:
            print("Warning: cartopy not available, using simple plot")
        
        # Simple Cartesian plot (lat vs lon)
        fig, ax = plt.subplots(figsize=(12, 10))
        trajectories = tracking_result['trajectories']
        valid = tracking_result['valid']
        
        # Collect valid trajectories
        valid_trajs = [(i, traj) for i, traj in enumerate(trajectories) if valid[i] and len(traj) > 1]
        
        if plot_final_only:
            # Plot final and start positions
            end_lons = []
            end_lats = []
            start_lons = []
            start_lats = []
            example_traj_idx = None
            
            for i, traj in valid_trajs:
                times, lats, lons = zip(*traj)
                lons = np.array(lons)
                lats = np.array(lats)
                
                # Store first valid trajectory for example
                if plot_one_example and example_traj_idx is None:
                    example_traj_idx = i
                    example_lons = lons
                    example_lats = lats
                
                # Collect start and end positions
                start_lons.append(lons[0])
                start_lats.append(lats[0])
                end_lons.append(lons[-1])
                end_lats.append(lats[-1])
            
            # Plot one example trajectory if requested
            if plot_one_example and example_traj_idx is not None:
                ax.plot(example_lons, example_lats, 'b-', alpha=0.7, linewidth=1.5, 
                       label='Example trajectory', zorder=3)
            
            # Plot start positions
            if len(start_lons) > 0:
                ax.scatter(start_lons, start_lats, c='green', s=15, marker='o',
                          label='Start positions', alpha=0.6, zorder=2)
            
            # Plot all final positions
            if len(end_lons) > 0:
                ax.scatter(end_lons, end_lats, c='red', s=15, marker='o',
                          label='Final positions', alpha=0.6, zorder=2)
        else:
            # Plot all trajectories (subsampled if requested)
            if plot_subsample > 1:
                valid_trajs = valid_trajs[::plot_subsample]
                if debug:
                    print(f"[PLOT] Subsampling trajectories: plotting every {plot_subsample}th trajectory")
            
            n_plotted = 0
            for i, traj in valid_trajs:
                times, lats, lons = zip(*traj)
                lons = np.array(lons)
                lats = np.array(lats)
                
                if show_path:
                    ax.plot(lons, lats, 'b-', alpha=0.3, linewidth=0.5, zorder=1)
                
                if show_start:
                    ax.plot(lons[0], lats[0], 'go', markersize=6, 
                           label='Start' if n_plotted == 0 else '', zorder=2)
                
                if show_end:
                    ax.plot(lons[-1], lats[-1], 'ro', markersize=6,
                           label='End' if n_plotted == 0 else '', zorder=2)
                
                n_plotted += 1
        
        ax.set_xlabel('Longitude (degrees)', fontsize=12)
        ax.set_ylabel('Latitude (degrees)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        else:
            n_total = len(valid_trajs) if not plot_final_only else len(end_lons)
            ax.set_title(f"Ice Parcel Trajectories ({n_total} points)", 
                        fontsize=14, fontweight='bold', pad=15)
        
        # Add legend
        if show_start or show_end or plot_final_only:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        if out_png:
            plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0.1)
            if debug:
                print(f"[PLOT] Saved: {out_png}")
        
        plt.show()
        return
    
    # Use cartopy for proper map projection
    fig = plt.figure(figsize=(14, 12))
    proj = ccrs.SouthPolarStereo()
    ax = plt.subplot(1, 1, 1, projection=proj)
    
    # Set extent
    # If plotting final only, default to Ross Sea zoom for better visualization
    if constrain_ross_sea or plot_final_only:
        # Smaller Ross Sea region - more zoomed in
        extent = [165, 175, -76.5, -74]  # Focused Ross Sea region
        if debug:
            print(f"[PLOT] Setting map extent to Ross Sea region: {extent}")
    else:
        extent = [-180, 180, -90, -60]  # Southern Ocean
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add map features for better visualization
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='black', zorder=10)
    ax.add_feature(cfeature.LAND, alpha=0.5, color='lightgray', zorder=5)
    ax.add_feature(cfeature.OCEAN, alpha=0.4, color='lightblue', zorder=1)
    
    # Add ice shelves if available (for Antarctic context)
    try:
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', 
                                                     '50m', edgecolor='cyan', 
                                                     facecolor='none', linewidth=0.8, zorder=9))
    except:
        pass  # Feature may not be available
    
    # Add gridlines with better formatting for Ross Sea
    gl = ax.gridlines(draw_labels=True, alpha=0.6, linestyle='--', linewidth=0.8, zorder=8, 
                      color='gray', xlocs=range(160, 180, 2), ylocs=range(-78, -72, 1))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Plot trajectories
    trajectories = tracking_result['trajectories']
    valid = tracking_result['valid']
    
    # Collect valid trajectories
    valid_trajs = [(i, traj) for i, traj in enumerate(trajectories) if valid[i] and len(traj) > 1]
    
    n_total = len(valid_trajs)
    if debug:
        print(f"[PLOT] Found {n_total} valid trajectories")
    
    # If plot_final_only: just plot final and start positions, no paths
    if plot_final_only:
        if debug:
            print(f"[PLOT] Plotting only final positions ({n_total} points)")
        
        # Collect final and start positions
        end_lons = []
        end_lats = []
        start_lons = []
        start_lats = []
        example_traj_idx = None
        
        for i, traj in valid_trajs:
            times, lats, lons = zip(*traj)
            lons = np.array(lons)
            lats = np.array(lats)
            lons_plot = wrap_lon180(lons)
            
            # Store first valid trajectory for example
            if plot_one_example and example_traj_idx is None:
                example_traj_idx = i
                example_lons = lons_plot
                example_lats = lats
            
            # Collect start and end positions
            start_lons.append(lons_plot[0])
            start_lats.append(lats[0])
            end_lons.append(lons_plot[-1])
            end_lats.append(lats[-1])
        
        # Plot one example trajectory if requested
        if plot_one_example and example_traj_idx is not None:
            if debug:
                print(f"[PLOT] Plotting example trajectory #{example_traj_idx}")
            ax.plot(example_lons, example_lats, 'b-', alpha=0.7, linewidth=1.5,
                   transform=ccrs.PlateCarree(), zorder=6, label='Example trajectory')
        
        # Plot start positions
        if len(start_lons) > 0:
            ax.scatter(start_lons, start_lats, c='green', s=15, marker='o',
                      transform=ccrs.PlateCarree(), zorder=7, label='Start positions', alpha=0.6)
        
        # Plot all final positions
        if len(end_lons) > 0:
            ax.scatter(end_lons, end_lats, c='red', s=15, marker='o',
                      transform=ccrs.PlateCarree(), zorder=7, label='Final positions', alpha=0.6)
        
        n_plotted = len(end_lons)
    
    else:
        # Original plotting logic (all trajectories)
        # Subsample trajectories for faster plotting
        if plot_subsample > 1:
            valid_trajs = valid_trajs[::plot_subsample]
            if debug:
                print(f"[PLOT] Subsampling trajectories: plotting every {plot_subsample}th trajectory")
        
        n_total = len(valid_trajs)
        if debug:
            print(f"[PLOT] Plotting {n_total} trajectories...")
        
        # Collect all points for vectorized plotting
        if show_path:
            all_path_lons = []
            all_path_lats = []
            
        if show_start:
            start_lons = []
            start_lats = []
        
        if show_end:
            end_lons = []
            end_lats = []
        
        n_plotted = 0
        for i, traj in valid_trajs:
            times, lats, lons = zip(*traj)
            lons = np.array(lons)
            lats = np.array(lats)
            
            # Convert to -180-180 for plotting
            lons_plot = wrap_lon180(lons)
            
            if show_path:
                all_path_lons.append(lons_plot)
                all_path_lats.append(lats)
            
            if show_start:
                start_lons.append(lons_plot[0])
                start_lats.append(lats[0])
            
            if show_end:
                end_lons.append(lons_plot[-1])
                end_lats.append(lats[-1])
            
            n_plotted += 1
        
        # Plot paths in batches (much faster than individual plots)
        if show_path and len(all_path_lons) > 0:
            for lons_plot, lats in zip(all_path_lons, all_path_lats):
                ax.plot(lons_plot, lats, 'b-', alpha=0.3, linewidth=0.5,
                       transform=ccrs.PlateCarree(), zorder=6)
        
        # Plot start/end markers in batches
        if show_start and len(start_lons) > 0:
            ax.scatter(start_lons, start_lats, c='green', s=10, marker='o',
                      transform=ccrs.PlateCarree(), zorder=7, label='Start', alpha=0.6)
        
        if show_end and len(end_lons) > 0:
            ax.scatter(end_lons, end_lats, c='red', s=10, marker='o',
                      transform=ccrs.PlateCarree(), zorder=7, label='End', alpha=0.6)
    
    if debug:
        print(f"[PLOT] Plotted {n_plotted} trajectories")
    
    # Add legend
    if show_start or show_end:
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f"Ice Parcel Trajectories ({n_plotted} tracks)", 
                    fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0.1)
        if debug:
            print(f"[PLOT] Saved: {out_png}")
    
    plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Forward/backward tracking of ice parcels using ice drift data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backward tracking from ICESat-2 to SWOT window (with Ross Sea filtering)
  python ice_tracking.py \\
      --mode backward \\
      --start_time 20231028T191642 \\
      --end_time 20231028T215103 \\
      --is2_file /path/to/ATL07-02_20231028223812_06102101_006_01.h5 \\
      --ice_motion_nc /path/to/icemotion.nc \\
      --filter_ross_sea \\
      --out_npz trajectories_backward.npz \\
      --out_png trajectories_backward.png

  # Forward tracking from ICESat-2
  python ice_tracking.py \\
      --mode forward \\
      --start_time 20231028T000000 \\
      --end_time 20231030T000000 \\
      --is2_file /path/to/ATL07-02_20231028223812_06102101_006_01.h5 \\
      --ice_motion_nc /path/to/icemotion.nc \\
      --filter_ross_sea \\
      --out_npz trajectories_forward.npz \\
      --out_png trajectories_forward.png

  # Forward tracking from manual coordinates
  python ice_tracking.py \\
      --mode forward \\
      --start_time 20231028T000000 \\
      --end_time 20231030T000000 \\
      --lat -75.0 -74.5 \\
      --lon 180.0 180.5 \\
      --ice_motion_nc /path/to/icemotion.nc \\
      --out_npz trajectories.npz
        """
    )
    
    # Mode and timing
    parser.add_argument("--mode", choices=['forward', 'backward'], default='backward',
                       help="Tracking direction (default: backward). "
                            "For backward: start_time < end_time (track from end_time back to start_time). "
                            "For forward: start_time < end_time (track from start_time forward to end_time).")
    parser.add_argument("--start_time", required=True,
                       help="Start time (format: YYYYMMDDTHHMMSS or YYYYMMDD). "
                            "For backward tracking: earliest time to track back to. "
                            "For forward tracking: initial time to start from.")
    parser.add_argument("--end_time", required=True,
                       help="End time (format: YYYYMMDDTHHMMSS or YYYYMMDD). "
                            "For backward tracking: latest time (where tracking starts from). "
                            "For forward tracking: final time to track forward to.")
    
    # Input data sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--is2_file", "--is2_h5", "--is2_mat",
                            dest="is2_file",
                            help="ICESat-2 HDF5 file path (ATL07 format, .h5)")
    input_group.add_argument("--lat", type=float, nargs='+',
                            help="Initial latitudes (degrees)")
    input_group.add_argument("--lon", type=float, nargs='+',
                            help="Initial longitudes (degrees)")
    
    # Ice motion
    parser.add_argument("--ice_motion_nc", required=True,
                       help="Ice motion NetCDF file path")
    
    # Tracking parameters
    parser.add_argument("--dt_hours", type=float, default=1.0,
                       help="Time step for integration in hours (default: 1.0)")
    parser.add_argument("--subsample_is2", type=int, default=1,
                       help="Subsample IS2 points by this factor (default: 1 = no subsampling). "
                            "Use 2 to use every 2nd point, 5 for every 5th point, etc. "
                            "Larger values = faster but less detailed tracking.")
    
    # Output
    parser.add_argument("--out_npz", default=None,
                       help="Output NPZ file to save trajectories")
    parser.add_argument("--out_png", default=None,
                       help="Output PNG file for trajectory plot")
    
    # Visualization options
    parser.add_argument("--no_plot", action="store_true",
                       help="Skip plotting")
    parser.add_argument("--constrain_ross_sea", action="store_true",
                       help="Constrain map to Ross Sea region")
    parser.add_argument("--filter_ross_sea", action="store_true",
                       help="Filter ICESat-2 data to Ross Sea region only")
    parser.add_argument("--no_start_marker", action="store_true",
                       help="Don't show start position markers")
    parser.add_argument("--no_end_marker", action="store_true",
                       help="Don't show end position markers")
    parser.add_argument("--no_path", action="store_true",
                       help="Don't show trajectory paths")
    parser.add_argument("--plot_subsample", type=int, default=1,
                       help="Subsample trajectories for plotting (default: 1 = plot all). "
                            "Use 10 to plot every 10th trajectory, 100 for every 100th, etc. "
                            "Larger values = faster plotting but less detail.")
    parser.add_argument("--plot_final_only", action="store_true",
                       help="Plot only final positions (much faster). Shows all end points as scatter.")
    parser.add_argument("--plot_one_example", action="store_true",
                       help="Plot one example trajectory along with final positions. "
                            "Useful with --plot_final_only to show trajectory pattern.")
    parser.add_argument("--simple_xy", action="store_true",
                       help="Plot in simple X-Y (lat-lon) Cartesian coordinates instead of map projection. "
                            "Better for seeing small differences in trajectories.")
    
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information")
    
    args = parser.parse_args()
    
    # Parse times
    def parse_time(s):
        try:
            if 'T' in s:
                return datetime.strptime(s, "%Y%m%dT%H%M%S")
            else:
                return datetime.strptime(s, "%Y%m%d")
        except ValueError as e:
            raise ValueError(f"Invalid time format: {s}. Use YYYYMMDDTHHMMSS or YYYYMMDD")
    
    time_start = parse_time(args.start_time)
    time_end = parse_time(args.end_time)
    
    # Validate time ordering
    if args.mode == 'backward':
        if time_start >= time_end:
            raise ValueError(
                f"For backward tracking, --start_time ({time_start}) must be EARLIER than "
                f"--end_time ({time_end}). We track backwards from end_time to start_time."
            )
    else:  # forward
        if time_start >= time_end:
            raise ValueError(
                f"For forward tracking, --start_time ({time_start}) must be EARLIER than "
                f"--end_time ({time_end}). We track forward from start_time to end_time."
            )
    
    # Load initial positions
    if args.is2_file:
        # For backward tracking: use end_time as the observation time (track back from there)
        # For forward tracking: use start_time as the observation time (track forward from there)
        is2_target_time = time_end if args.mode == 'backward' else time_start
        is2_data = load_is2_tracks(
            args.is2_file, 
            target_date=is2_target_time, 
            filter_ross_sea_region=args.filter_ross_sea,
            debug=args.debug
        )
        lat_init = is2_data['lat']
        lon_init = is2_data['lon']
        time_init = is2_data['time']
        
        # Subsample IS2 points if requested (for faster tracking)
        if args.subsample_is2 > 1:
            n_original = len(lat_init)
            lat_init = lat_init[::args.subsample_is2]
            lon_init = lon_init[::args.subsample_is2]
            if args.debug:
                print(f"[MAIN] Subsampled IS2 points: {n_original} -> {len(lat_init)} "
                      f"(every {args.subsample_is2}th point)")
    elif args.lat and args.lon:
        if len(args.lat) != len(args.lon):
            raise ValueError("--lat and --lon must have the same number of values")
        lat_init = np.array(args.lat)
        lon_init = np.array(args.lon)
        time_init = time_start if args.mode == 'forward' else time_end
    else:
        raise ValueError("Must provide either --is2_file or both --lat and --lon")
    
    if args.debug:
        print(f"[MAIN] Initial positions: {len(lat_init)} points")
        print(f"[MAIN] Initial time: {time_init}")
        if args.mode == 'backward':
            print(f"[MAIN] Tracking BACKWARD from {time_end} (IS2 time) to {time_start} (SWOT start)")
        else:
            print(f"[MAIN] Tracking FORWARD from {time_start} to {time_end}")
    
    # Perform tracking
    tracking_result = track_ice_parcels(
        lat_init, lon_init, time_init,
        args.ice_motion_nc, time_start, time_end,
        direction=args.mode, dt_hours=args.dt_hours, debug=args.debug
    )
    
    # Save results
    if args.out_npz:
        # Convert trajectories to saveable format
        # trajectories is a list of lists of (time, lat, lon) tuples
        # We'll save as separate arrays with trajectory indices
        trajectories = tracking_result['trajectories']
        valid = tracking_result['valid']
        
        # Flatten trajectories into arrays
        traj_times = []
        traj_lats = []
        traj_lons = []
        traj_indices = []  # Which trajectory each point belongs to
        
        for i, traj in enumerate(trajectories):
            if valid[i] and len(traj) > 0:
                for t, lat, lon in traj:
                    traj_times.append(t)
                    traj_lats.append(lat)
                    traj_lons.append(lon)
                    traj_indices.append(i)
        
        # Convert times to timestamps (seconds since epoch) for saving
        if len(traj_times) > 0:
            # Use first time as reference
            time_ref = traj_times[0]
            if isinstance(time_ref, datetime):
                traj_times_sec = np.array([(t - time_ref).total_seconds() for t in traj_times], dtype=float)
            else:
                # If times are already numeric, use as-is
                traj_times_sec = np.array(traj_times, dtype=float)
                time_ref = None
        else:
            traj_times_sec = np.array([], dtype=float)
            time_ref = time_init  # Fallback to initial time
        
        # Save trajectory lengths for reconstruction
        traj_lengths = np.array([len(traj) if valid[i] else 0 for i, traj in enumerate(trajectories)], dtype=int)
        
        # Prepare save dictionary
        save_dict = {
            # Trajectory data (flattened)
            'traj_times_sec': traj_times_sec,
            'traj_lats': np.array(traj_lats, dtype=float) if traj_lats else np.array([], dtype=float),
            'traj_lons': np.array(traj_lons, dtype=float) if traj_lons else np.array([], dtype=float),
            'traj_indices': np.array(traj_indices, dtype=int) if traj_indices else np.array([], dtype=int),
            'traj_lengths': traj_lengths,
            # Summary data
            'valid': tracking_result['valid'],
            'lat_final': tracking_result['lat_final'],
            'lon_final': tracking_result['lon_final'],
            'lat_start': tracking_result['lat_start'],
            'lon_start': tracking_result['lon_start'],
            'lat_init': lat_init,
            'lon_init': lon_init,
            'time_init': time_init,
            'time_start': time_start,
            'time_end': time_end,
            'mode': args.mode,
            'n_trajectories': len(trajectories),
            # Motion statistics
            'distances': tracking_result.get('distances', np.array([])),  # Total distance (meters)
            'mean_velocities': tracking_result.get('mean_velocities', np.array([])),  # Mean velocity (m/s)
            'total_times': tracking_result.get('total_times', np.array([])),  # Total time (seconds)
            'ross_sea_mask': tracking_result.get('ross_sea_mask', np.array([]))  # Ross Sea mask
        }
        
        # Add time reference if it's a datetime (save as string)
        if time_ref is not None and isinstance(time_ref, datetime):
            save_dict['time_ref_str'] = time_ref.isoformat()
        
        np.savez(
            args.out_npz,
            **save_dict
        )
        if args.debug:
            print(f"[MAIN] Saved trajectories to: {args.out_npz}")
            print(f"[MAIN] Saved {len(traj_times)} trajectory points from {np.sum(valid)} valid trajectories")
    
    # Plot results
    if not args.no_plot and args.out_png:
        plot_trajectories(
            tracking_result,
            out_png=args.out_png,
            title=f"Ice Parcel Tracking ({args.mode})",
            show_start=not args.no_start_marker,
            show_end=not args.no_end_marker,
            show_path=not args.no_path,
            constrain_ross_sea=args.constrain_ross_sea,
            plot_subsample=args.plot_subsample,
            plot_final_only=args.plot_final_only,
            plot_one_example=args.plot_one_example,
            simple_xy=args.simple_xy,
            debug=args.debug
        )
    elif not args.no_plot:
        # Plot without saving
        plot_trajectories(
            tracking_result,
            title=f"Ice Parcel Tracking ({args.mode})",
            show_start=not args.no_start_marker,
            show_end=not args.no_end_marker,
            show_path=not args.no_path,
            constrain_ross_sea=args.constrain_ross_sea,
            plot_subsample=args.plot_subsample,
            plot_final_only=args.plot_final_only,
            plot_one_example=args.plot_one_example,
            simple_xy=args.simple_xy,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
