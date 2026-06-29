#!/usr/bin/env python
"""
align_mask_to_swot.py

Properly align BedMachine mask to SWOT lat/lon coordinates.
This is the CORRECT way to apply geographic masks to satellite swaths.

The key insight:
- BedMachine is a global Antarctic grid (x, y coordinates)
- SWOT is a swath with lat/lon coordinates
- We need to RESAMPLE BedMachine to SWOT's lat/lon grid

Usage:
  python align_mask_to_swot.py \
    --swot SWOT_file.nc \
    --bedmachine BedMachineAntarctica-v3.nc \
    --output swot_ice_mask.nc
"""

import argparse
import logging
import numpy as np
import xarray as xr
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def align_mask_to_swot(swot_file, bedmachine_file, output_file):
    """Align BedMachine mask to SWOT lat/lon coordinates."""
    logger.info("=" * 70)
    logger.info("ALIGNING BEDMACHINE MASK TO SWOT COORDINATES")
    logger.info("=" * 70)
    
    # Step 1: Load SWOT data
    logger.info("\n[1/4] Loading SWOT data...")
    with xr.open_dataset(swot_file) as swot_ds:
        # Get lat/lon
        if 'latitude' in swot_ds.variables:
            lat = swot_ds['latitude'].values
            lon = swot_ds['longitude'].values
        elif 'lat' in swot_ds.variables:
            lat = swot_ds['lat'].values
            lon = swot_ds['lon'].values
        else:
            logger.error("Could not find lat/lon in SWOT file!")
            return False
        
        logger.info(f"  SWOT shape: {lat.shape}")
        logger.info(f"  Lat range: {np.nanmin(lat):.2f}° to {np.nanmax(lat):.2f}°")
        logger.info(f"  Lon range: {np.nanmin(lon):.2f}° to {np.nanmax(lon):.2f}°")
    
    # Step 2: Load BedMachine
    logger.info("\n[2/4] Loading BedMachine...")
    with xr.open_dataset(bedmachine_file) as bm_ds:
        bm_mask = bm_ds['mask'].values
        bm_x = bm_ds['x'].values
        bm_y = bm_ds['y'].values
        
        logger.info(f"  BedMachine shape: {bm_mask.shape}")
        logger.info(f"  BedMachine x range: {bm_x.min():.0f} to {bm_x.max():.0f}")
        logger.info(f"  BedMachine y range: {bm_y.min():.0f} to {bm_y.max():.0f}")
    
    # Step 3: Convert SWOT lat/lon to BedMachine x/y (Polar Stereographic)
    logger.info("\n[3/4] Converting SWOT coordinates to BedMachine grid...")
    try:
        from pyproj import Proj
        
        # Antarctic Polar Stereographic projection (same as BedMachine)
        ps_proj = Proj(proj='stere', lat_0=-90, lon_0=0, k=1, x_0=0, y_0=0, 
                       ellps='WGS84', units='m')
        
        # Convert SWOT lat/lon to x/y
        swot_x, swot_y = ps_proj(lon, lat)
        
        logger.info(f"  ✓ Converted to Polar Stereographic coordinates")
        logger.info(f"  SWOT x range: {np.nanmin(swot_x)/1e6:.1f} to {np.nanmax(swot_x)/1e6:.1f} Mm")
        logger.info(f"  SWOT y range: {np.nanmin(swot_y)/1e6:.1f} to {np.nanmax(swot_y)/1e6:.1f} Mm")
        
    except ImportError:
        logger.error("  ❌ pyproj not installed!")
        logger.error("  Install with: pip install pyproj")
        return False
    
    # Step 4: Interpolate BedMachine mask to SWOT grid
    logger.info("\n[4/4] Interpolating BedMachine mask to SWOT grid...")
    try:
        from scipy.interpolate import griddata
        
        # BedMachine grid coordinates
        bm_xx, bm_yy = np.meshgrid(bm_x, bm_y)
        bm_points = np.column_stack([bm_xx.ravel(), bm_yy.ravel()])
        bm_values = bm_mask.ravel()
        
        # SWOT points
        swot_points = np.column_stack([swot_x.ravel(), swot_y.ravel()])
        
        # Interpolate (nearest neighbor for categorical mask)
        logger.info(f"  Interpolating {len(bm_points):,} BedMachine points to {len(swot_points):,} SWOT points...")
        swot_mask_interp = griddata(
            bm_points, 
            bm_values, 
            swot_points, 
            method='nearest'
        )
        
        # Reshape to SWOT grid
        swot_mask = swot_mask_interp.reshape(lat.shape).astype(np.int8)
        
        logger.info(f"  ✓ Interpolation complete")
        
        # Count masked pixels
        ice_or_land = np.isin(swot_mask, [1, 2, 3])
        logger.info(f"  Ice/land pixels: {np.sum(ice_or_land):,} ({100.0*np.sum(ice_or_land)/ice_or_land.size:.1f}%)")
        
    except ImportError:
        logger.error("  ❌ scipy not installed!")
        logger.error("  Install with: pip install scipy")
        return False
    except Exception as e:
        logger.error(f"  ❌ Interpolation failed: {e}")
        return False
    
    # Step 5: Save result
    logger.info(f"\n[5/5] Saving aligned mask...")
    try:
        output_ds = xr.Dataset(
            {
                'ice_mask': (['y', 'x'], ice_or_land.astype(np.int8)),
                'bedmachine_mask': (['y', 'x'], swot_mask.astype(np.int8)),
            },
            coords={
                'y': (['y'], np.arange(lat.shape[0])),
                'x': (['x'], np.arange(lat.shape[1])),
                'lat': (['y', 'x'], lat),
                'lon': (['y', 'x'], lon),
            },
            attrs={
                'source': 'BedMachineAntarctica-v3 aligned to SWOT',
                'description': 'Ice shelf/sheet mask for SWOT classification',
                'ice_mask_description': '1 = ice/land (mask out), 0 = water (keep)',
                'projection': 'Polar Stereographic (-90°S)',
            }
        )
        
        output_ds.to_netcdf(output_file)
        logger.info(f"  ✓ Saved to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to save: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Align BedMachine mask to SWOT swath coordinates"
    )
    parser.add_argument("--swot", required=True, type=str, help="SWOT file")
    parser.add_argument("--bedmachine", required=True, type=str, help="BedMachine file")
    parser.add_argument("--output", default="swot_ice_mask.nc", type=str, help="Output file")
    
    args = parser.parse_args()
    
    success = align_mask_to_swot(args.swot, args.bedmachine, args.output)
    
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✅ ALIGNMENT SUCCESSFUL!")
        logger.info(f"   Use this mask file: {args.output}")
        logger.info(f"   python example_classify_swot_with_mask.py \\")
        logger.info(f"     --file {args.swot} \\")
        logger.info(f"     --mask {args.output} \\")
        logger.info(f"     --region ross_sea")
    else:
        logger.error("❌ ALIGNMENT FAILED!")
    logger.info("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())