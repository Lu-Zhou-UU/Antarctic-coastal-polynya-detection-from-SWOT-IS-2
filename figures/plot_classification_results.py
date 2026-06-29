#!/usr/bin/env python3
"""
plot_classification_results.py

Enhanced SWOT classification visualization with physics-based coloring.

Features:
- Robust polar projection with automatic extent computation
- Physics-based ice class colors (VIBRANT & DISTINCT palette)
  ✓ Vivid blues for water/thin ice
  ✓ Dark shades for thick ice
  ✓ Maximum visual contrast and appeal
- 0-360 longitude range to prevent dateline artifacts in Ross Sea
- BedMachine ice shelf and grounding line overlays
- Region constraining for focused plots
- High-quality publication-ready output

UPDATED COLOR PALETTE:
  - VIBRANT & DISTINCT: Bold, easily distinguishable colors
  - Alternative palettes available in enhanced_color_palettes.py
    (MODERN, HIGHCONTRAST, THERMAL, JEWELS, MINIMAL)
"""

import argparse
from datetime import datetime
from pathlib import Path
import warnings
import os
import sys

# Try to help pyproj find its database
def init_pyproj():
    """Initialize pyproj database path if possible."""
    if 'PROJ_LIB' not in os.environ:
        # Try to find PROJ data directory
        try:
            import pyproj
            proj_data = pyproj.datadir.get_data_dir()
            if proj_data and os.path.exists(proj_data):
                os.environ['PROJ_LIB'] = proj_data
                print(f"Set PROJ_LIB to: {proj_data}")
        except:
            pass

init_pyproj()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Layer z-order (shared by SSH and classification panels)
Z_DATA = 3
Z_PMW_OUTLINE = 4
Z_SHELF_MASK = 5

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 15,
    'axes.linewidth': 1.0,
})


def _as_str_list(x):
    if x is None:
        return []
    a = np.asarray(x)
    if a.dtype.kind in ("U", "S"):
        return [str(s) for s in a.tolist()]
    return [str(s) for s in a.tolist()]


def _robust_limits(a, p_lo=2.0, p_hi=98.0):
    a = np.asarray(a, dtype=np.float64)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return (0.0, 1.0)
    return (np.nanpercentile(v, p_lo), np.nanpercentile(v, p_hi))


def normalize_longitude_0_360(lon):
    """
    Normalize longitude to 0-360 range.
    This prevents dateline crossing artifacts in Ross Sea region.
    """
    lon = np.asarray(lon, dtype=np.float64)
    return (lon + 360.0) % 360.0


def constrain_to_region(lon, lat, data_dict, lon_range=(160, 220), lat_range=(-78, -70)):
    """
    Constrain data to a specific geographic region (e.g., Ross Sea).
    
    Parameters
    ----------
    lon, lat : arrays
        Coordinate arrays (lon should be in 0-360 range)
    data_dict : dict
        Dictionary of data arrays to constrain (same shape as lon/lat)
        Keys are names, values are arrays
    lon_range : tuple
        (lon_min, lon_max) in 0-360 range
    lat_range : tuple
        (lat_min, lat_max)
        
    Returns
    -------
    lon_crop, lat_crop, data_dict_crop
    """
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    
    # Create mask for region
    mask = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )
    
    # Find bounding box
    if not np.any(mask):
        print(f"Warning: No data in region lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")
        return lon, lat, data_dict
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    # Add small buffer (5 pixels)
    row_min = max(0, row_min - 5)
    row_max = min(mask.shape[0], row_max + 6)
    col_min = max(0, col_min - 5)
    col_max = min(mask.shape[1], col_max + 6)
    
    # Crop
    lon_crop = lon[row_min:row_max, col_min:col_max]
    lat_crop = lat[row_min:row_max, col_min:col_max]
    
    data_dict_crop = {}
    for key, arr in data_dict.items():
        if arr is not None:
            data_dict_crop[key] = arr[row_min:row_max, col_min:col_max]
        else:
            data_dict_crop[key] = None
    
    print(f"Constrained to region: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")
    print(f"  Original shape: {lon.shape}")
    print(f"  Cropped shape: {lon_crop.shape}")
    print(f"  Lon range in data: [{np.nanmin(lon_crop):.1f}, {np.nanmax(lon_crop):.1f}]")
    print(f"  Lat range in data: [{np.nanmin(lat_crop):.1f}, {np.nanmax(lat_crop):.1f}]")
    
    return lon_crop, lat_crop, data_dict_crop


def compute_projection_extent(lon, lat, proj, pc, buffer_factor=1.2):
    """
    Compute extent in projection coordinates for reliable zooming.
    
    Parameters
    ----------
    lon, lat : arrays
        Data coordinates (lon in 0-360 range)
    proj : cartopy projection
        Target projection
    pc : PlateCarree projection
        Geographic projection
    buffer_factor : float
        Multiplicative buffer (1.2 = 20% larger)
        
    Returns
    -------
    extent_proj : tuple
        (x_min, x_max, y_min, y_max) in projection coordinates
    """
    import cartopy.crs as ccrs
    
    # Remove invalid points
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        return None
    
    lon_v = lon[valid]
    lat_v = lat[valid]
    
    # Convert 0-360 to -180 to 180 for cartopy (it expects this range)
    lon_v_180 = np.where(lon_v > 180, lon_v - 360, lon_v)
    
    # Transform to projection coordinates
    x_proj, y_proj = proj.transform_points(pc, lon_v_180, lat_v)[:, 0:2].T
    
    # Remove any invalid transformed points
    valid_proj = np.isfinite(x_proj) & np.isfinite(y_proj)
    if not np.any(valid_proj):
        return None
    
    x_proj = x_proj[valid_proj]
    y_proj = y_proj[valid_proj]
    
    # Compute bounds with buffer
    x_mean = np.mean(x_proj)
    y_mean = np.mean(y_proj)
    x_range = np.ptp(x_proj)
    y_range = np.ptp(y_proj)
    
    x_min = x_mean - (x_range / 2) * buffer_factor
    x_max = x_mean + (x_range / 2) * buffer_factor
    y_min = y_mean - (y_range / 2) * buffer_factor
    y_max = y_mean + (y_range / 2) * buffer_factor
    
    return (x_min, x_max, y_min, y_max)


def load_bedmachine_mask(bedmachine_path, lon_data, lat_data, subsample=5,
                         lon_range=None, lat_range=None):
    """Load BedMachine mask for the data region."""
    try:
        import xarray as xr
        from pyproj import Transformer
        from scipy.ndimage import binary_dilation
    except ImportError as e:
        raise ImportError(
            "BedMachine support requires xarray, pyproj, and scipy"
        ) from e
    
    print(f"Loading BedMachine from: {bedmachine_path}")
    
    ds = xr.open_dataset(bedmachine_path)
    
    # Subsample
    if subsample > 1:
        ds = ds.isel(x=slice(None, None, subsample), 
                     y=slice(None, None, subsample))
    
    mask = ds['mask'].values
    x = ds['x'].values
    y = ds['y'].values
    
    # Transform coordinates - use proj strings to avoid EPSG database issues
    try:
        transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
        print("  Using EPSG codes")
    except Exception as e1:
        print(f"  EPSG codes failed: {e1}")
        try:
            # Fallback: proj string for Antarctic Polar Stereo, EPSG for WGS84
            proj_antarctic = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            transformer = Transformer.from_crs(proj_antarctic, "EPSG:4326", always_xy=True)
            print("  Using proj string for source, EPSG for target")
        except Exception as e2:
            print(f"  Hybrid approach failed: {e2}")
            # Last resort: proj strings for both
            proj_antarctic = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            proj_wgs84 = "+proj=longlat +datum=WGS84 +no_defs"
            transformer = Transformer.from_crs(proj_antarctic, proj_wgs84, always_xy=True)
            print("  Using proj strings for both source and target")
    
    X, Y = np.meshgrid(x, y)
    print("Transforming BedMachine coordinates...")
    lon, lat = transformer.transform(X, Y)
    
    # Normalize BedMachine longitude to 0-360
    lon = normalize_longitude_0_360(lon)
    
    # Subset to map region (prefer explicit bounds over SWOT swath footprint)
    if lon_range is not None and lat_range is not None:
        lon_data_min, lon_data_max = float(lon_range[0]), float(lon_range[1])
        lat_data_min, lat_data_max = float(lat_range[0]), float(lat_range[1])
        lon_buffer = max(0.5, (lon_data_max - lon_data_min) * 0.05)
        lat_buffer = max(0.5, (lat_data_max - lat_data_min) * 0.05)
    else:
        lon_data_min, lon_data_max = np.nanmin(lon_data), np.nanmax(lon_data)
        lat_data_min, lat_data_max = np.nanmin(lat_data), np.nanmax(lat_data)
        # Wider buffer so shelves/coast fill the zoomed map, not just the swath
        lon_buffer = max(2.0, (lon_data_max - lon_data_min) * 0.5)
        lat_buffer = max(2.0, (lat_data_max - lat_data_min) * 0.5)
    
    in_region = (
        (lat >= lat_data_min - lat_buffer) & 
        (lat <= lat_data_max + lat_buffer) &
        (lon >= lon_data_min - lon_buffer) &
        (lon <= lon_data_max + lon_buffer)
    )
    
    if not np.any(in_region):
        print("Warning: No BedMachine data in region")
        ds.close()
        return None
    
    # Find bounding box
    rows, cols = np.where(in_region)
    r_min, r_max = max(0, rows.min() - 10), min(mask.shape[0], rows.max() + 10)
    c_min, c_max = max(0, cols.min() - 10), min(mask.shape[1], cols.max() + 10)
    
    mask = mask[r_min:r_max, c_min:c_max]
    lon = lon[r_min:r_max, c_min:c_max]
    lat = lat[r_min:r_max, c_min:c_max]
    
    ice_shelf_mask = (mask == 3)
    grounded_mask = (mask == 2)
    # Rock + grounded ice + floating shelf (exclude open ocean and subglacial lakes)
    land_mask = (mask >= 1) & (mask <= 3)
    
    ice_shelf_expanded = binary_dilation(ice_shelf_mask)
    grounding_line_mask = ice_shelf_expanded & grounded_mask
    
    ds.close()
    
    print(f"BedMachine loaded: {mask.shape} pixels")
    print(f"  Floating ice shelf (mask=3): {ice_shelf_mask.sum()} pixels")
    print(f"  Land/ice fill (mask 1-3): {land_mask.sum()} pixels")
    print(f"  BedMachine lon range: [{np.nanmin(lon):.1f}, {np.nanmax(lon):.1f}]")
    
    return {
        'ice_shelf_mask': ice_shelf_mask,
        'land_mask': land_mask,
        'grounding_line_mask': grounding_line_mask,
        'lon': lon,
        'lat': lat,
    }


def get_physics_based_colors(class_names):
    """
    Get physics-based colors for sea ice classification.
    
    ✓ VIBRANT & DISTINCT palette with maximum visual appeal
    ✓ Bold colors with excellent contrast
    ✓ Water: vivid blues → Ice: darker shades
    ✓ Perfect for publications and presentations
    
    To change palette: See enhanced_color_palettes.py for MODERN, JEWELS, etc.
    """
    from matplotlib.colors import ListedColormap
    
    color_map = {
        'polynya': '#0D47A1',           # Sapphire
        'thin_ice': '#648b93',          # Turquoise
        'first_year_ice': '#d4dbd9',    # Light aquamarine
        'pack_ice': '#ecc49c',          # Slate
        'unsure': '#946847',            # Gray
        'excluded': '#F5F5F5',          # Off-white
    }
    
    # Build colormap from class names
    colors = []
    for name in class_names:
        name_lower = name.lower().replace(' ', '_').replace('-', '_')
        if name_lower in color_map:
            colors.append(color_map[name_lower])
        else:
            # Default: light gray for unknown classes
            colors.append('#D3D3D3')
    
    return ListedColormap(colors), colors


def _class_display_name(class_name):
    name_lower = class_name.lower().replace(" ", "_").replace("-", "_")
    if name_lower in ("polynya", "open_water"):
        return "Polynya/Lead"
    return class_name.replace("_", " ").capitalize()


def _class_percentages(labels_plot, n_classes, class_names):
    valid_mask = labels_plot >= 0
    total_valid = int(np.sum(valid_mask))
    pcts = {}
    if total_valid <= 0:
        return pcts, total_valid
    for i in range(min(n_classes, len(class_names))):
        class_count = int(np.sum(labels_plot == i))
        pcts[i] = 100.0 * class_count / total_valid
    return pcts, total_valid


def _crop_geo_grid(lat, lon360, *fields, lon_min, lon_max, lat_min, lat_max, buffer_deg=0.5):
    """Crop 2-D lat/lon grids (lon in 0-360) to a geographic box."""
    m = (
        (lat >= (lat_min - buffer_deg)) & (lat <= (lat_max + buffer_deg))
        & (lon360 >= (lon_min - buffer_deg)) & (lon360 <= (lon_max + buffer_deg))
    )
    if not np.any(m):
        raise ValueError(
            f"No AMSR grid points in lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]"
        )
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    out = [arr[r0 : r1 + 1, c0 : c1 + 1] for arr in (lat, lon360, *fields)]
    return tuple(out)


def _lon_plot_from_360(lon360):
    """Match SWOT panel convention: 0-360 -> [-180, 180] for east longitudes."""
    return np.where(lon360 > 180.0, lon360 - 360.0, lon360)


def _load_amsr_sic_crop(
    amsr_mat,
    scene_date,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    polynya_threshold=0.7,
    polynya_type="sic",
    buffer_deg=0.5,
):
    """
    Load AMSR2 SIC for one day, align lat/lon to sic grid, crop to map box.

    Returns lat, lon_plot, sic_frac on the same cropped grid.
    """
    from plot_esacci_sit_map import load_mat_any, polynya_output_amsr2

    amsr = load_mat_any(amsr_mat)
    sic = np.array(amsr["sic"], dtype=float)
    lat = np.array(amsr["lat"], dtype=float)
    lon = np.array(amsr["lon"], dtype=float)

    if sic.ndim == 3 and sic.shape[0] != 365 and sic.shape[-1] == 365:
        sic = np.moveaxis(sic, -1, 0)
    if lat.shape != sic.shape[1:] and lat.T.shape == sic.shape[1:]:
        lat, lon = lat.T, lon.T
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    lon360 = normalize_longitude_0_360(lon)
    dt = datetime.strptime(str(scene_date), "%Y%m%d")
    day_idx = min(int(dt.timetuple().tm_yday), 365) - 1
    sic_daily = sic[day_idx] if day_idx < sic.shape[0] else sic[-1]
    sic_frac = sic_daily / 100.0 if np.nanmax(sic_daily) > 1.0 else sic_daily

    if polynya_type == "sic":
        field = sic_frac
    else:
        open_pol, coastal_pol = polynya_output_amsr2(sic_frac, polynya_threshold)
        if polynya_type == "open":
            field = np.where(np.isfinite(open_pol), sic_frac, np.nan)
        elif polynya_type == "both":
            m = np.isfinite(open_pol) | np.isfinite(coastal_pol)
            field = np.where(m, sic_frac, np.nan)
        else:
            field = np.where(np.isfinite(coastal_pol), sic_frac, np.nan)

    lat_c, lon_c, field_c = _crop_geo_grid(
        lat, lon360, field,
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        buffer_deg=buffer_deg,
    )
    lon_p = _lon_plot_from_360(lon_c)
    return lat_c, lon_p, field_c


def _plot_pmw_outline(ax, lat, lon_plot, sic_frac, proj, pc, threshold, **line_kw):
    """
    Draw PMW SIC-threshold outline in projected coordinates.

    Contouring directly on lon/lat arrays that mix 170E with -170E creates
    spurious closed loops near the map cut; transform to projection xy first.
    """
    valid = np.isfinite(sic_frac) & np.isfinite(lat) & np.isfinite(lon_plot)
    sic_plot = np.array(sic_frac, dtype=float)
    sic_plot[~valid] = np.nan

    xy = proj.transform_points(pc, lon_plot, lat)
    x = xy[..., 0]
    y = xy[..., 1]
    bad = ~np.isfinite(x) | ~np.isfinite(y)
    sic_plot[bad] = np.nan

    ax.contour(
        x, y, sic_plot,
        levels=[float(threshold)],
        transform=proj,
        **line_kw,
    )


def _plot_bedmachine_ice_shelves(ax, bedmachine_data, proj, pc, args, zorder=Z_SHELF_MASK):
    """Draw BedMachine land/ice mask (same style on all panels)."""
    if not bedmachine_data or not args.show_ice_shelves:
        return

    bm_lon = np.where(bedmachine_data["lon"] > 180, bedmachine_data["lon"] - 360, bedmachine_data["lon"])
    bm_lat = bedmachine_data["lat"]
    fill_mask = bedmachine_data.get("land_mask", bedmachine_data["ice_shelf_mask"])
    shelf_mask = fill_mask.astype(float)

    ice_shelf_ma = np.ma.masked_where(~fill_mask, shelf_mask)
    shelf_cmap = ListedColormap([args.shelf_facecolor])
    ax.pcolormesh(
        bm_lon, bm_lat, ice_shelf_ma,
        transform=pc, shading="auto",
        cmap=shelf_cmap, alpha=args.shelf_alpha,
        vmin=0, vmax=1, zorder=zorder,
    )


def _plot_bedmachine_grounding_line(ax, bedmachine_data, proj, pc, args):
    """Optional grounding-line overlay (below shelf outline)."""
    if not bedmachine_data or not args.show_grounding_line:
        return

    bm_lon = np.where(bedmachine_data["lon"] > 180, bedmachine_data["lon"] - 360, bedmachine_data["lon"])
    bm_lat = bedmachine_data["lat"]
    gl = bedmachine_data["grounding_line_mask"].astype(float)

    xy = proj.transform_points(pc, bm_lon, bm_lat)
    x = xy[..., 0]
    y = xy[..., 1]
    gl_plot = np.array(gl, dtype=float)
    gl_plot[~np.isfinite(x) | ~np.isfinite(y)] = np.nan

    ax.contour(
        x, y, gl_plot,
        levels=[0.5],
        colors=args.gl_color,
        linestyles="-",
        linewidths=args.gl_linewidth,
        transform=proj,
        zorder=28,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--central_lon", type=float, default=180.0)
    ap.add_argument("--show_sigma", action="store_true")
    ap.add_argument("--show_land", action="store_true", default=True)
    ap.add_argument("--show_grid", action="store_true", default=False,
                    help="Show coordinate grid lines (default: False for cleaner plots)")
    ap.add_argument("--use_physics_colors", action="store_true", default=True,
                    help="Use physics-based ice classification colors (default: True)")
    
    # Region constraining
    ap.add_argument("--constrain_region", action="store_true",
                    help="Constrain plot to Ross Sea region (160-220°E, -78 to -70°S)")
    ap.add_argument("--lon_min", type=float, default=160.0,
                    help="Minimum longitude for region constraint (default: 160)")
    ap.add_argument("--lon_max", type=float, default=220.0,
                    help="Maximum longitude for region constraint (default: 220)")
    ap.add_argument("--lat_min", type=float, default=-78.0,
                    help="Minimum latitude for region constraint (default: -78)")
    ap.add_argument("--lat_max", type=float, default=-70.0,
                    help="Maximum latitude for region constraint (default: -70)")
    
    # BedMachine
    ap.add_argument("--bedmachine", type=str, default=None)
    ap.add_argument("--show_ice_shelves", action="store_true")
    ap.add_argument("--show_grounding_line", action="store_true")
    ap.add_argument("--bedmachine_subsample", type=int, default=2,
                    help="BedMachine grid subsample factor (default: 2; use 1 for full res)")
    ap.add_argument("--shelf_color", default="gray", 
                    help="Color for ice shelf outlines (default: gray)")
    ap.add_argument("--shelf_alpha", type=float, default=1.0,
                    help="Transparency for ice shelf fill (default: 1.0)")
    ap.add_argument("--shelf_facecolor", default="0.85",
                    help="Ice shelf fill color (default: 0.85, matches map land tone)")
    ap.add_argument("--shelf_linewidth", type=float, default=1.8,
                    help="Ice shelf outline linewidth (default: 1.8)")
    ap.add_argument("--shelf_linestyle", default="-",
                    help="Ice shelf outline linestyle (default: solid)")
    ap.add_argument("--gl_color", default="red")
    ap.add_argument("--gl_linewidth", type=float, default=1.5)

    # Passive-microwave polynya outline (classification panel only)
    ap.add_argument("--amsr_mat", type=str, default=None,
                    help="AMSR2 .mat file for PMW polynya outline on classification panel")
    ap.add_argument("--scene_date", type=str, default=None,
                    help="Scene date YYYYMMDD for AMSR2 mask (required with --amsr_mat)")
    ap.add_argument("--polynya_threshold", type=float, default=0.7,
                    help="SIC threshold for PMW polynya outline (default: 0.7)")
    ap.add_argument("--polynya_type", choices=["sic", "open", "coastal", "both"], default="sic",
                    help="PMW outline source: sic=SIC isoline (manuscript mask), "
                         "coastal/open/both=polynya_output_amsr2 components")
    ap.add_argument("--pmw_crop_buffer_deg", type=float, default=0.5,
                    help="Extra degrees around map box when cropping AMSR for outline")
    ap.add_argument("--pmw_outline_color", default="#d62728",
                    help="PMW polynya outline color (default: red)")
    ap.add_argument("--pmw_outline_style", default="--",
                    help="PMW polynya outline linestyle (default: solid)")
    ap.add_argument("--pmw_outline_lw", type=float, default=1.5,
                    help="PMW polynya outline linewidth (default: 1.0)")
    
    # Extent - now uses projection coordinates by default
    ap.add_argument("--buffer_factor", type=float, default=1.2,
                    help="Zoom buffer factor (1.2 = 20%% extra space)")
    ap.add_argument("--global_view", action="store_true",
                    help="Show entire Antarctica instead of zooming")
    
    # Style
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--sigma_cmap", default="YlGnBu",
                    help="Colormap for sigma0 (default: gray for grayscale shading)")
    ap.add_argument("--sigma_vmin", type=float, default=-5.0,
                    help="Minimum value for sigma0 colorbar (default: 0 dB)")
    ap.add_argument("--sigma_vmax", type=float, default=35.0,
                    help="Maximum value for sigma0 colorbar (default: 45 dB)")
    ap.add_argument("--class_cmap", default="tab20",
                    help="Fallback colormap if not using physics colors")
    ap.add_argument("--figsize", nargs=2, type=float, default=None)
    ap.add_argument("--overlay_points", type=str, default=None)
    
    args = ap.parse_args()

    # Load data
    p = Path(args.npz)
    z = np.load(p, allow_pickle=True)

    lat = z["lat"].astype(np.float64)
    lon = z["lon"].astype(np.float64)
    
    # CRITICAL: Normalize to 0-360 range (DO NOT use -180 to 180)
    lon = normalize_longitude_0_360(lon)
    print(f"Longitude range: [{np.nanmin(lon):.1f}, {np.nanmax(lon):.1f}]")

    class_labels = z["class_labels"].astype(np.int32)
    class_names = _as_str_list(z.get("class_names", None))
    sigma0 = z["sigma0"].astype(np.float32) if "sigma0" in z.files else None
    ssh = z["ssh"].astype(np.float32) if "ssh" in z.files else None
    exclude_mask = z["exclude_mask"].astype(bool) if "exclude_mask" in z.files and z["exclude_mask"] is not None else None

    labels_plot = np.array(class_labels, copy=True)
    labels_plot[labels_plot < 0] = -1
    if exclude_mask is not None:
        labels_plot[exclude_mask] = -1
        if sigma0 is not None:
            sigma0 = np.array(sigma0, copy=True)
            sigma0[exclude_mask] = np.nan
        if ssh is not None:
            ssh = np.array(ssh, copy=True)
            ssh[exclude_mask] = np.nan
    
    # Constrain to region if requested (e.g., Ross Sea: 160-220°E)
    # FIXED: Constrain all arrays together to ensure same shape
    if args.constrain_region:
        data_dict = {
            'sigma0': sigma0,
            'ssh': ssh,
            'labels_plot': labels_plot,
            'exclude_mask': exclude_mask
        }
        lon, lat, data_dict = constrain_to_region(
            lon, lat, data_dict,
            lon_range=(args.lon_min, args.lon_max),
            lat_range=(args.lat_min, args.lat_max)
        )
        sigma0 = data_dict['sigma0']
        ssh = data_dict['ssh']
        labels_plot = data_dict['labels_plot']
        exclude_mask = data_dict['exclude_mask']

    # Load BedMachine if requested
    bedmachine_data = None
    if (args.show_ice_shelves or args.show_grounding_line) and args.bedmachine:
        try:
            bm_lon_range = bm_lat_range = None
            if args.constrain_region:
                bm_lon_range = (args.lon_min, args.lon_max)
                bm_lat_range = (args.lat_min, args.lat_max)
            bedmachine_data = load_bedmachine_mask(
                args.bedmachine, lon, lat,
                subsample=args.bedmachine_subsample,
                lon_range=bm_lon_range,
                lat_range=bm_lat_range,
            )
        except Exception as e:
            warnings.warn(f"Failed to load BedMachine: {e}")

    # Load overlay points
    overlay_lon, overlay_lat = None, None
    if args.overlay_points:
        overlay_z = np.load(args.overlay_points, allow_pickle=True)
        if "lon" in overlay_z.files and "lat" in overlay_z.files:
            overlay_lon = normalize_longitude_0_360(overlay_z["lon"].astype(float))
            overlay_lat = overlay_z["lat"].astype(float)
            print(f"Loaded {len(overlay_lon)} overlay points")

    # Setup cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.SouthPolarStereo(central_longitude=float(args.central_lon))
    pc = ccrs.PlateCarree()

    # Compute extent in projection coordinates
    extent_proj = None
    if not args.global_view:
        # FIXED: Use labels_plot and exclude_mask that are already constrained
        if exclude_mask is not None:
            valid_mask = ~exclude_mask
        else:
            valid_mask = (labels_plot >= 0)
        
        lon_valid = lon[valid_mask]
        lat_valid = lat[valid_mask]
        
        if len(lon_valid) > 0:
            extent_proj = compute_projection_extent(
                lon_valid, lat_valid, proj, pc, 
                buffer_factor=args.buffer_factor
            )
            if extent_proj:
                print(f"Computed projection extent: x=[{extent_proj[0]:.0f}, {extent_proj[1]:.0f}], "
                      f"y=[{extent_proj[2]:.0f}, {extent_proj[3]:.0f}]")

    # Figure setup
    n_panels = 2 if args.show_sigma else 1
    figsize = tuple(args.figsize) if args.figsize else ((9, 8) if n_panels == 1 else (16, 7.5))
    
    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize,
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]

    # Styling function
    def prep_ax(ax, title):
        if extent_proj is not None:
            # Use projection coordinates directly for reliable zooming
            ax.set_xlim(extent_proj[0], extent_proj[1])
            ax.set_ylim(extent_proj[2], extent_proj[3])
            print(f"  Set limits: x=[{extent_proj[0]:.0f}, {extent_proj[1]:.0f}], "
                  f"y=[{extent_proj[2]:.0f}, {extent_proj[3]:.0f}]")
        else:
            ax.set_global()
            print("  Using global view")
        
        if args.show_grid:
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                            color='gray', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.bottom_labels = False  # Disable bottom labels to avoid messy lines
            gl.left_labels = True
            gl.right_labels = False
        
        # Use BedMachine shelf mask as the sole land fill when available
        use_cartopy_land = args.show_land and not (
            args.show_ice_shelves and bedmachine_data is not None
        )
        if use_cartopy_land:
            ax.add_feature(cfeature.LAND, facecolor="0.85", 
                          edgecolor="0.4", linewidth=0.3, zorder=1)
            ax.coastlines(linewidth=0.5, color="0.3", zorder=2)
        
        # ax.set_title(title, fontsize=13, pad=10)
        # Clean up axes appearance - no visible frame
        ax.spines['geo'].set_linewidth(0)  # Remove boundary line

    # Convert lon to -180 to 180 for pcolormesh (cartopy expects this)
    lon_plot = np.where(lon > 180, lon - 360, lon)
    
    # Panel 1: SSH (Unfiltered)
    if args.show_sigma:
        ax = axes[0]
        prep_ax(ax, "SSH Unfiltered (m)")
        
        if ssh is not None:
            # Use robust percentile range for SSH
            v0, v1 = _robust_limits(ssh, 1, 99)
            print(f"SSH range: [{v0:.2f}, {v1:.2f}] m")
            im = ax.pcolormesh(lon_plot, lat, ssh, transform=pc, 
                             shading="auto", vmin=v0, vmax=v1,
                             cmap='RdBu_r', rasterized=True, zorder=Z_DATA)
            
            if overlay_lon is not None:
                overlay_lon_plot = np.where(overlay_lon > 180, overlay_lon - 360, overlay_lon)
                ax.plot(overlay_lon_plot, overlay_lat, 'r*', markersize=8,
                       transform=pc, zorder=10)
            
            _plot_bedmachine_grounding_line(ax, bedmachine_data, proj, pc, args)
            _plot_bedmachine_ice_shelves(ax, bedmachine_data, proj, pc, args)
            
            cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
            cb.set_label("SSHA (m)", fontsize=15)

    # Panel 2: Classification
    ax = axes[-1]
    method_name = str(z.get("method", "Classification")).strip() if "method" in z.files else "Classification"
    prep_ax(ax, f"SWOT {method_name}")
    
    n_classes = (max(1, len(class_names)) if len(class_names) > 0 
                 else int(np.nanmax(labels_plot[labels_plot >= 0]) + 1) 
                 if np.any(labels_plot >= 0) else 1)
    
    # Use physics-based colors if class names are available
    if args.use_physics_colors and len(class_names) > 0:
        cmap, class_colors = get_physics_based_colors(class_names)
        print(f"\n✓ Using VIBRANT & DISTINCT physics-based colors:")
        for i, (name, color) in enumerate(zip(class_names, class_colors)):
            print(f"  {i}: {name:20s} → {color}")
    else:
        try:
            cmap = plt.colormaps[args.class_cmap].resampled(n_classes)
        except:
            cmap = plt.cm.get_cmap(args.class_cmap, n_classes)
        class_colors = [cmap(i) for i in range(n_classes)]
        print(f"Using standard colormap: {args.class_cmap}")
    
    # Calculate class percentages within SWOT swath (excluding masked/excluded areas)
    class_pcts, total_valid = _class_percentages(labels_plot, n_classes, class_names)
    if total_valid > 0:
        print(f"\n✓ Ice class distribution in SWOT swath ({total_valid:,} valid pixels):")
        for i in range(min(n_classes, len(class_names))):
            class_count = int(np.sum(labels_plot == i))
            class_pct = class_pcts.get(i, 0.0)
            display_name = _class_display_name(class_names[i])
            print(f"  {display_name:20s}: {class_pct:5.1f}% ({class_count:,} pixels)")
    else:
        print("\n⚠ No valid classification data in plotting window")

    # Draw order: SWOT data → PMW outline → land mask (same stack as SSH panel)
    lab_ma = np.ma.masked_where(labels_plot < 0, labels_plot)
    imc = ax.pcolormesh(lon_plot, lat, lab_ma, transform=pc, shading="auto",
                       cmap=cmap, vmin=-0.5, vmax=n_classes-0.5,
                       rasterized=True, zorder=Z_DATA)

    pmw_outline_plotted = False
    if args.amsr_mat:
        if not args.scene_date:
            warnings.warn("--amsr_mat given without --scene_date; skipping PMW outline")
        else:
            try:
                valid_swot = np.isfinite(lon) & np.isfinite(lat)
                if args.constrain_region:
                    pmw_lon_min = float(args.lon_min)
                    pmw_lon_max = float(args.lon_max)
                    pmw_lat_min = float(args.lat_min)
                    pmw_lat_max = float(args.lat_max)
                elif np.any(valid_swot):
                    lon360_swot = normalize_longitude_0_360(lon[valid_swot])
                    pmw_lon_min = float(np.nanmin(lon360_swot))
                    pmw_lon_max = float(np.nanmax(lon360_swot))
                    pmw_lat_min = float(np.nanmin(lat[valid_swot]))
                    pmw_lat_max = float(np.nanmax(lat[valid_swot]))
                else:
                    pmw_lon_min, pmw_lon_max = 160.0, 220.0
                    pmw_lat_min, pmw_lat_max = -78.0, -70.0

                lat_pmw, lon_pmw, sic_pmw = _load_amsr_sic_crop(
                    args.amsr_mat,
                    args.scene_date,
                    lon_min=pmw_lon_min,
                    lon_max=pmw_lon_max,
                    lat_min=pmw_lat_min,
                    lat_max=pmw_lat_max,
                    polynya_threshold=args.polynya_threshold,
                    polynya_type=args.polynya_type,
                    buffer_deg=float(args.pmw_crop_buffer_deg),
                )
                _plot_pmw_outline(
                    ax, lat_pmw, lon_pmw, sic_pmw, proj, pc,
                    threshold=args.polynya_threshold,
                    colors=args.pmw_outline_color,
                    linestyles=args.pmw_outline_style,
                    linewidths=args.pmw_outline_lw,
                    zorder=Z_PMW_OUTLINE,
                )
                pmw_outline_plotted = True
                print(
                    f"✓ PMW outline: SIC={args.polynya_threshold:.2f} isoline "
                    f"({args.polynya_type}) for {args.scene_date}; "
                    f"crop lon=[{pmw_lon_min:.1f}, {pmw_lon_max:.1f}], "
                    f"lat=[{pmw_lat_min:.1f}, {pmw_lat_max:.1f}]"
                )
            except Exception as e:
                warnings.warn(f"Failed to plot PMW polynya outline: {e}")

    _plot_bedmachine_grounding_line(ax, bedmachine_data, proj, pc, args)
    _plot_bedmachine_ice_shelves(ax, bedmachine_data, proj, pc, args)
    
    if overlay_lon is not None:
        overlay_lon_plot = np.where(overlay_lon > 180, overlay_lon - 360, overlay_lon)
        ax.plot(overlay_lon_plot, overlay_lat, 'r*', markersize=8,
               transform=pc, zorder=10)

    # Legend - reordered for importance (polynya first)
    if class_names:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Define priority order for legend (most important first)
        priority_order = [
            'polynya', 'open_water', 'lead', 'thin_ice', 
            'first_year_ice', 'young_ice', 'pack_ice', 
            'old_ice', 'multi_year_ice', 'unsure'
        ]
        
        # Create handles in priority order
        handles = []
        handled_indices = set()
        
        # First, add classes in priority order
        for priority_name in priority_order:
            for i in range(min(n_classes, len(class_names))):
                name_lower = class_names[i].lower().replace(' ', '_').replace('-', '_')
                if name_lower == priority_name and i not in handled_indices:
                    display_name = _class_display_name(class_names[i])
                    pct_suffix = f" ({class_pcts[i]:.1f}%)" if i in class_pcts else ""
                    handles.append(
                        Patch(facecolor=class_colors[i], edgecolor="0.3", linewidth=0.5,
                              label=f"{display_name}{pct_suffix}")
                    )
                    handled_indices.add(i)
                    break
        
        # Add any remaining classes not in priority list
        for i in range(min(n_classes, len(class_names))):
            if i not in handled_indices:
                display_name = _class_display_name(class_names[i])
                pct_suffix = f" ({class_pcts[i]:.1f}%)" if i in class_pcts else ""
                handles.append(
                    Patch(facecolor=class_colors[i], edgecolor="0.3", linewidth=0.5,
                          label=f"{display_name}{pct_suffix}")
                )
        if exclude_mask is not None and np.any(exclude_mask):
            handles.append(Patch(facecolor="white", edgecolor="0.5", 
                                alpha=0.5, label="Excluded"))
        if bedmachine_data:
            if args.show_ice_shelves:
                handles.append(
                    Patch(
                        facecolor=args.shelf_facecolor,
                        edgecolor="0.5",
                        alpha=1.0,
                        label="Ice shelves",
                    )
                )
            if args.show_grounding_line:
                handles.append(Line2D([0], [0], color=args.gl_color,
                                     linewidth=args.gl_linewidth,
                                     label="Grounding line"))
        if pmw_outline_plotted:
            handles.append(
                Line2D(
                    [0], [0],
                    color=args.pmw_outline_color,
                    linestyle=args.pmw_outline_style,
                    linewidth=args.pmw_outline_lw,
                    label=f"PMW SIC outline",
                )
            )
        
        ax.legend(handles=handles, loc="upper left", 
                 bbox_to_anchor=(1.005, 1.0), frameon=True,
                 framealpha=0.95, edgecolor="0.3")
    else:
        cb = fig.colorbar(imc, ax=ax, shrink=0.7, pad=0.03)
        cb.set_label("Class ID", fontsize=10)

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✓ Saved: {out}")


if __name__ == "__main__":
    main()