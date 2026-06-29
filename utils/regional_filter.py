"""
Regional Data Filtering Module

Extract specific geographic regions from SWOT and IS2 data.
Useful for focusing analysis on specific areas (e.g., Ross Sea).

Usage:
    filter = RegionalFilter()

    # Get Ross Sea region
    ross_sea_data = filter.extract_ross_sea(sigma0, ssh, lat, lon)

    # Or define custom region
    custom_data = filter.extract_region(sigma0, lat, lon,
                                       lat_range=(-78, -70),
                                       lon_range=(160, 230))
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RegionDefinition:
    """Definition of a geographic region"""
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    description: str = ""

    def __str__(self):
        return (f"{self.name}\n"
                f"  Latitude:  {self.lat_min}° to {self.lat_max}°\n"
                f"  Longitude: {self.lon_min}° to {self.lon_max}°")


# Pre-defined regions (using 0-360° longitude)
REGIONS = {
    'ross_sea': RegionDefinition(
        name='Ross Sea',
        lat_min=-78.0,
        lat_max=-70.0,
        lon_min=160.0,
        lon_max=230.0,  # 160°E to 230°E (crosses dateline, equivalent to 160°E to 130°W)
        description='Ross Sea ice shelf region, Antarctica'
    ),
    'weddell_sea': RegionDefinition(
        name='Weddell Sea',
        lat_min=-85.0,
        lat_max=-65.0,
        lon_min=300.0,   # 300°E (equivalent to 60°W)
        lon_max=60.0,    # to 60°E (crosses 0°/360° line)
        description='Weddell Sea ice shelf region, Antarctica'
    ),
    'amundsen_sea': RegionDefinition(
        name='Amundsen Sea',
        lat_min=-78.0,
        lat_max=-68.0,
        lon_min=240.0,   # 240°E (equivalent to 120°W)
        lon_max=280.0,   # to 280°E (equivalent to 80°W)
        description='Amundsen Sea ice shelf region, Antarctica'
    ),
    'east_antarctica': RegionDefinition(
        name='East Antarctica',
        lat_min=-78.0,
        lat_max=-65.0,
        lon_min=50.0,
        lon_max=160.0,
        description='East Antarctic ice shelf region'
    ),
}


class RegionalFilter:
    """Filter satellite data to specific geographic regions"""

    def __init__(self, verbose: bool = True):
        """
        Parameters
        ----------
        verbose : bool
            Enable logging
        """
        self.verbose = verbose

    @staticmethod
    def normalize_longitude_0_360(lon: np.ndarray) -> np.ndarray:
        """
        Normalize longitude to [0, 360) range
        
        This prevents dateline crossing artifacts, especially important
        for regions like Ross Sea that span the 180° meridian.

        Parameters
        ----------
        lon : ndarray
            Longitude values

        Returns
        -------
        ndarray
            Normalized longitude in [0, 360) range
        """
        lon_norm = np.asarray(lon, dtype=np.float64)
        return (lon_norm + 360.0) % 360.0

    @staticmethod
    def normalize_longitude(lon: np.ndarray) -> np.ndarray:
        """
        Normalize longitude to [0, 360) range
        
        Kept for backwards compatibility, but now uses 0-360 range.
        
        Parameters
        ----------
        lon : ndarray
            Longitude values

        Returns
        -------
        ndarray
            Normalized longitude in [0, 360) range
        """
        return RegionalFilter.normalize_longitude_0_360(lon)

    @staticmethod
    def is_in_region(lat: np.ndarray, lon: np.ndarray,
                     lat_range: Tuple[float, float],
                     lon_range: Tuple[float, float],
                     normalize_lon: bool = True) -> np.ndarray:
        """
        Create boolean mask for points in region
        
        Handles regions that cross 0°/360° line (e.g., Weddell Sea: 300° to 60°)
        and regions near 180° (e.g., Ross Sea: 160° to 230°)

        Parameters
        ----------
        lat : ndarray
            Latitude values
        lon : ndarray
            Longitude values
        lat_range : tuple
            (lat_min, lat_max)
        lon_range : tuple
            (lon_min, lon_max) in 0-360 range
        normalize_lon : bool
            Normalize longitude to [0, 360)

        Returns
        -------
        ndarray (bool)
            Mask where True = inside region
        """
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        # Normalize longitude to 0-360
        lon_work = lon.copy()
        if normalize_lon:
            lon_work = RegionalFilter.normalize_longitude_0_360(lon_work)

        # Create latitude mask
        lat_mask = (lat >= lat_min) & (lat <= lat_max)

        # Handle longitude
        # If lon_max < lon_min, region crosses 0°/360° line
        # Example: Weddell Sea (300° to 60°) means 300° to 360° OR 0° to 60°
        if lon_max < lon_min:
            # Region crosses 0°/360° line
            lon_mask = (lon_work >= lon_min) | (lon_work <= lon_max)
        else:
            # Normal case: region doesn't cross 0°/360°
            # This includes Ross Sea (160° to 230°) which doesn't cross 360°
            lon_mask = (lon_work >= lon_min) & (lon_work <= lon_max)

        return lat_mask & lon_mask

    @staticmethod
    def _ensure_2d_grid(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure lat/lon are 2D grids."""
        if lat.ndim == 1 and lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            return lat_grid, lon_grid
        return lat, lon

    @staticmethod
    def _ensure_2d_like(arr: np.ndarray, target_hw: Tuple[int, int], name: str) -> np.ndarray:
        """
        Ensure arr is either (H,W) or (W,) which will be broadcast to (H,W).
        """
        H, W = target_hw
        if arr.ndim == 2:
            if arr.shape != (H, W):
                raise ValueError(f"{name} shape {arr.shape} must match (H,W)=({H},{W})")
            return arr
        if arr.ndim == 1:
            if arr.shape[0] != W:
                raise ValueError(f"{name} is 1D but length {arr.shape[0]} != W={W}")
            return np.tile(arr[None, :], (H, 1))
        raise ValueError(f"{name} must be 1D (W,) or 2D (H,W), got ndim={arr.ndim}")

    def extract_region(self,
                       sigma0: np.ndarray,
                       lat: np.ndarray,
                       lon: np.ndarray,
                       ssh: Optional[np.ndarray] = None,
                       lat_range: Tuple[float, float] = None,
                       lon_range: Tuple[float, float] = None,
                       region_name: str = None,
                       incidence: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Extract data within a specific geographic region

        Parameters
        ----------
        sigma0 : ndarray
            Backscatter coefficient (2D image [H, W])
        lat : ndarray
            Latitude grid [H, W] or [H]
        lon : ndarray
            Longitude grid [H, W] or [W] (will be normalized to 0-360)
        ssh : ndarray, optional
            Sea surface height [H, W]
        lat_range : tuple, optional
            (lat_min, lat_max) for custom region
        lon_range : tuple, optional
            (lon_min, lon_max) for custom region in 0-360 range
        region_name : str, optional
            Pre-defined region name: 'ross_sea', 'weddell_sea', etc.
        incidence : ndarray, optional
            Incidence angle proxy. Can be:
              - 2D [H, W]
              - 1D [W] (per cross-track pixel), will be broadcast to [H, W]

        Returns
        -------
        dict or None
            Extracted region data:
            {
                'sigma0': [h, w] cropped image (masked outside region)
                'ssh': [h, w] cropped or None
                'incidence': [h, w] cropped or None
                'lat': [h, w] cropped
                'lon': [h, w] cropped (in 0-360 range)
                'mask': [H, W] boolean mask of full image
                'bounds': region bounds
                'stats': extraction statistics
            }
        """

        # Get region bounds
        if region_name:
            if region_name not in REGIONS:
                logger.error(f"Unknown region: {region_name}")
                logger.info(f"Available regions: {list(REGIONS.keys())}")
                return None
            region = REGIONS[region_name]
            lat_range = (region.lat_min, region.lat_max)
            lon_range = (region.lon_min, region.lon_max)
            region_str = region.name
        else:
            if lat_range is None or lon_range is None:
                logger.error("Must specify either region_name or lat_range + lon_range")
                return None
            region_str = f"Lat {lat_range[0]:.1f}-{lat_range[1]:.1f}, Lon {lon_range[0]:.1f}-{lon_range[1]:.1f}"

        logger.info(f"\nExtracting region: {region_str}")
        logger.info(f"  Latitude:  {lat_range[0]:.1f}° to {lat_range[1]:.1f}°")
        
        # Display longitude in user-friendly format
        lon_min, lon_max = lon_range
        if lon_max < lon_min:
            logger.info(f"  Longitude: {lon_min:.1f}° to 360° and 0° to {lon_max:.1f}° (crosses 0°/360°)")
        elif lon_max > 180:
            # Show both 0-360 and equivalent -180 to 180 for clarity
            lon_min_180 = lon_min if lon_min <= 180 else lon_min - 360
            lon_max_180 = lon_max if lon_max <= 180 else lon_max - 360
            logger.info(f"  Longitude: {lon_min:.1f}° to {lon_max:.1f}° (or {lon_min_180:.1f}° to {lon_max_180:.1f}°)")
        else:
            logger.info(f"  Longitude: {lon_min:.1f}° to {lon_max:.1f}°")

        # Handle 1D lat/lon arrays
        lat_grid, lon_grid = self._ensure_2d_grid(lat, lon)
        
        # Normalize longitude to 0-360 range
        lon_grid = self.normalize_longitude_0_360(lon_grid)
        logger.info(f"  ✓ Normalized longitude to 0-360° range")

        # Basic shape check
        if sigma0.ndim != 2:
            logger.error(f"sigma0 must be 2D [H,W], got shape {sigma0.shape}")
            return None
        H, W = sigma0.shape
        if lat_grid.shape != (H, W) or lon_grid.shape != (H, W):
            logger.error(f"lat/lon grids must match sigma0 shape {sigma0.shape}. "
                         f"Got lat {lat_grid.shape}, lon {lon_grid.shape}")
            return None

        # Ensure incidence is 2D (if provided)
        incidence_grid = None
        if incidence is not None:
            try:
                incidence_grid = self._ensure_2d_like(incidence, (H, W), name="incidence")
            except Exception as e:
                logger.error(f"Invalid incidence input: {e}")
                return None

        # Create mask (is_in_region expects lon in 0-360, which we now have)
        mask = self.is_in_region(lat_grid, lon_grid, lat_range, lon_range, normalize_lon=False)

        n_in_region = int(np.sum(mask))
        n_total = int(mask.size)
        coverage = 100.0 * n_in_region / n_total

        logger.info(f"  Pixels in region: {n_in_region:,} / {n_total:,} ({coverage:.1f}%)")

        if n_in_region == 0:
            logger.error("No pixels found in region!")
            return None

        # Find bounding box of region
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        sigma0_crop = sigma0[row_min:row_max+1, col_min:col_max+1]
        lat_crop = lat_grid[row_min:row_max+1, col_min:col_max+1]
        lon_crop = lon_grid[row_min:row_max+1, col_min:col_max+1]  # Already in 0-360
        mask_crop = mask[row_min:row_max+1, col_min:col_max+1]

        # Mask outside region to NaN
        sigma0_crop_masked = sigma0_crop.copy()
        sigma0_crop_masked[~mask_crop] = np.nan

        ssh_crop = None
        if ssh is not None:
            ssh_crop = ssh[row_min:row_max+1, col_min:col_max+1].copy()
            ssh_crop[~mask_crop] = np.nan

        # Crop/mask incidence
        incidence_crop = None
        if incidence_grid is not None:
            incidence_crop = incidence_grid[row_min:row_max+1, col_min:col_max+1].copy()
            incidence_crop[~mask_crop] = np.nan

        # Statistics
        sigma0_valid = sigma0_crop_masked[np.isfinite(sigma0_crop_masked)]

        stats = {
            'n_pixels_total': n_total,
            'n_pixels_in_region': n_in_region,
            'coverage_percent': coverage,
            'sigma0_mean': float(np.nanmean(sigma0_valid)),
            'sigma0_std': float(np.nanstd(sigma0_valid)),
            'sigma0_min': float(np.nanmin(sigma0_valid)),
            'sigma0_max': float(np.nanmax(sigma0_valid)),
        }

        # Incidence stats if provided
        if incidence_crop is not None:
            inc_valid = incidence_crop[np.isfinite(incidence_crop)]
            if inc_valid.size > 0:
                stats.update({
                    'incidence_mean': float(np.nanmean(inc_valid)),
                    'incidence_std': float(np.nanstd(inc_valid)),
                    'incidence_min': float(np.nanmin(inc_valid)),
                    'incidence_max': float(np.nanmax(inc_valid)),
                })

        result = {
            'sigma0': sigma0_crop_masked,
            'ssh': ssh_crop,
            'incidence': incidence_crop,
            'lat': lat_crop,
            'lon': lon_crop,  # Now in 0-360 range
            'mask': mask,                  # Full mask
            'mask_crop': mask_crop,        # Cropped mask
            'bounds': {
                'row_min': int(row_min),
                'row_max': int(row_max),
                'col_min': int(col_min),
                'col_max': int(col_max),
            },
            'stats': stats
        }

        logger.info(f"  Cropped to: {sigma0_crop_masked.shape}")
        logger.info(f"  Sigma0: {stats['sigma0_mean']:.2f} ± {stats['sigma0_std']:.2f} dB")
        logger.info(f"           ({stats['sigma0_min']:.2f} to {stats['sigma0_max']:.2f})")
        if incidence_crop is not None and 'incidence_mean' in stats:
            logger.info(f"  Incidence: {stats['incidence_mean']:.2f} ± {stats['incidence_std']:.2f}° (proxy)")
            logger.info(f"             ({stats['incidence_min']:.2f} to {stats['incidence_max']:.2f})")
        logger.info(f"  Longitude range in cropped region: {float(np.nanmin(lon_crop)):.1f}° to {float(np.nanmax(lon_crop)):.1f}°")

        return result

    def extract_ross_sea(self, sigma0: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                         ssh: Optional[np.ndarray] = None,
                         incidence: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Extract Ross Sea region from SWOT data"""
        return self.extract_region(sigma0, lat, lon, ssh, region_name='ross_sea', incidence=incidence)

    def extract_weddell_sea(self, sigma0: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                            ssh: Optional[np.ndarray] = None,
                            incidence: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Extract Weddell Sea region"""
        return self.extract_region(sigma0, lat, lon, ssh, region_name='weddell_sea', incidence=incidence)

    def extract_amundsen_sea(self, sigma0: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                             ssh: Optional[np.ndarray] = None,
                             incidence: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Extract Amundsen Sea region"""
        return self.extract_region(sigma0, lat, lon, ssh, region_name='amundsen_sea', incidence=incidence)

    @staticmethod
    def print_available_regions():
        """Print all available pre-defined regions"""
        print("\nAvailable Pre-Defined Regions:")
        print("="*70)
        for region_name, region in REGIONS.items():
            print(f"\n  {region_name.upper()}")
            
            # Show both 0-360 and -180 to 180 formats for clarity
            lon_min = region.lon_min
            lon_max = region.lon_max
            
            print(f"    {region.name}")
            print(f"      Latitude:  {region.lat_min}° to {region.lat_max}°")
            
            if lon_max < lon_min:
                # Crosses 0°/360° line
                print(f"      Longitude: {lon_min}° to 360° and 0° to {lon_max}° (0-360 range)")
                lon_min_180 = lon_min - 360
                print(f"                 {lon_min_180}° to {lon_max}° (-180 to 180 range)")
            elif lon_max > 180:
                # Show both ranges for regions beyond 180°
                lon_min_180 = lon_min if lon_min <= 180 else lon_min - 360
                lon_max_180 = lon_max - 360
                print(f"      Longitude: {lon_min}° to {lon_max}° (0-360 range)")
                print(f"                 {lon_min_180}° to {lon_max_180}° (-180 to 180 range)")
            else:
                print(f"      Longitude: {lon_min}° to {lon_max}°")
            
            if region.description:
                print(f"      {region.description}")


def example_filter_to_ross_sea():
    """Example: Filter SWOT data to Ross Sea"""

    logging.basicConfig(level=logging.INFO)

    from data_loader import SWOTISDataLoader

    # Load SWOT file
    swot_file = 'data/SWOT_L3_LR_SSH_Unsmoothed_*.nc'

    logger.info("Loading SWOT data...")
    loader = SWOTISDataLoader()
    swot = loader.load_swot_scene(swot_file)

    if swot is None:
        logger.error("Failed to load SWOT data")
        return

    # Filter to Ross Sea
    logger.info("\nFiltering to Ross Sea region...")
    filter = RegionalFilter()
    ross_sea = filter.extract_ross_sea(
        swot.sigma0, swot.lat, swot.lon, swot.ssh,
        incidence=getattr(swot, "incidence", None)
    )

    if ross_sea is None:
        logger.error("Failed to extract Ross Sea")
        return

    # Now classify just the Ross Sea region
    logger.info("\nClassifying Ross Sea region...")
    from swot_classifier import SWOTClassifier

    classifier = SWOTClassifier()
    result = classifier.classify_ensemble(
        ross_sea['sigma0'],
        ross_sea['ssh'],
        n_clusters=4
    )

    if result is not None:
        logger.info("\nRoss Sea Classification Results:")
        for class_name in result.class_names:
            stats = result.get_class_stats(class_name)
            logger.info(f"  {class_name}: {stats['fraction']*100:.1f}%")

    return ross_sea, result


if __name__ == '__main__':
    RegionalFilter.print_available_regions()

    # Uncomment to run example
    # example_filter_to_ross_sea()