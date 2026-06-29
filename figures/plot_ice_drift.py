import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta
import warnings

def transform_grid_to_geographic(u_grid, v_grid, lon, lat):
    """
    Transform grid-aligned velocities (along-x, along-y) to geographic velocities (eastward, northward).
    
    For EASE-Grid South (Lambert Azimuthal Equal Area centered at South Pole):
    - The grid x-axis points eastward at longitude 0°
    - The grid y-axis points northward at longitude 0°
    - At other longitudes, the grid rotates relative to geographic coordinates
    
    Parameters:
    -----------
    u_grid : array
        Along-x component of velocity (grid coordinates)
    v_grid : array  
        Along-y component of velocity (grid coordinates)
    lon : array
        Longitude array (degrees)
    lat : array
        Latitude array (degrees)
        
    Returns:
    --------
    u_east : array
        Eastward velocity component
    v_north : array
        Northward velocity component
    """
    # Convert longitude to radians
    lon_rad = np.deg2rad(lon)
    
    # For EASE-Grid South (Lambert Azimuthal Equal Area):
    # The transformation accounts for grid rotation relative to geographic coordinates
    # Standard transformation for polar azimuthal equal-area:
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    
    # Transform from grid coordinates to geographic coordinates
    # CORRECTED: u_east (eastward), v_north (northward)
    u_east = u_grid * sin_lon + v_grid * cos_lon
    v_north = u_grid * cos_lon - v_grid * sin_lon
    
    return u_east, v_north

def load_and_visualize_ice_motion(filename, time_index=0, subsample_step=10, 
                                vector_scale=5e2, extent=None, add_colorbar=True):
    """
    Load and visualize sea ice motion data from NetCDF file.
    
    Parameters:
    -----------
    filename : str
        Path to NetCDF file
    time_index : int
        Time index to plot (default: 0)
    subsample_step : int
        Step size for subsampling vectors (default: 10)
    vector_scale : float
        Scale factor for vector arrows (default: 5e2)
    extent : list or None
        Map extent [lon_min, lon_max, lat_min, lat_max]
    add_colorbar : bool
        Whether to add colorbar showing vector magnitude
    """
    
    try:
        # Load the NetCDF file
        print(f"Loading data from {filename}...")
        ds = xr.open_dataset(filename, decode_times=False)
        
        # Display basic info about the dataset
        print(f"Dataset dimensions: {dict(ds.dims)}")
        print(f"Available time steps: {ds.dims.get('time', 'N/A')}")
        
        # Check if requested time index exists
        if 'time' in ds.dims and time_index >= ds.dims['time']:
            print(f"Warning: Time index {time_index} exceeds available times. Using index 0.")
            time_index = 0
        
        # Extract variables (grid-aligned components)
        u_grid = ds['u'].isel(time=time_index) if 'time' in ds.dims else ds['u']  # Along-x component
        v_grid = ds['v'].isel(time=time_index) if 'time' in ds.dims else ds['v']  # Along-y component
        lat = ds['latitude']
        lon = ds['longitude']
        
        # Check units
        u_units = ds['u'].attrs.get('units', '').strip()
        v_units = ds['v'].attrs.get('units', '').strip()
        print(f"Original units: u='{u_units}', v='{v_units}'")
        
        # Convert to m/s if needed (data is typically in cm/s)
        if 'cm' in u_units.lower() or 'cm' in v_units.lower():
            print("Converting from cm/s to m/s...")
            u_grid = u_grid / 100.0  # cm/s to m/s
            v_grid = v_grid / 100.0  # cm/s to m/s
            units_str = "m/s"
        else:
            units_str = u_units
        
        print("Note: U and V are grid-aligned components (along-x, along-y), not geographic components")
        print("Transforming to geographic coordinates (eastward, northward)...")
        
        # Transform from grid coordinates to geographic coordinates
        u_east, v_north = transform_grid_to_geographic(u_grid, v_grid, lon, lat)
        
        # Handle potential missing values
        valid_mask = ~(np.isnan(u_east) | np.isnan(v_north))
        
        # Subsample for clarity (now using geographic components)
        u_sub = u_east[::subsample_step, ::subsample_step]
        v_sub = v_north[::subsample_step, ::subsample_step]
        lat_sub = lat[::subsample_step, ::subsample_step]
        lon_sub = lon[::subsample_step, ::subsample_step]
        valid_mask_sub = valid_mask[::subsample_step, ::subsample_step]
        
        # Calculate vector magnitude for coloring (in m/s)
        magnitude = np.sqrt(u_sub**2 + v_sub**2)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.SouthPolarStereo())
        
        # Set extent (default to Southern Ocean focus)
        if extent is None:
            extent = [-180, 180, -90, -50]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
        
        # Add gridlines with labels
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot vectors only where data is valid
        valid_points = valid_mask_sub.values
        if np.any(valid_points):
            if add_colorbar:
                # Color-coded quiver plot
                quiver = ax.quiver(
                    lon_sub.values[valid_points], 
                    lat_sub.values[valid_points], 
                    u_sub.values[valid_points], 
                    v_sub.values[valid_points],
                    magnitude.values[valid_points],
                    transform=ccrs.PlateCarree(), 
                    scale=vector_scale, 
                    width=0.003,
                    cmap='viridis',
                    alpha=0.8
                )
                
                # Add colorbar
                cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, pad=0.1)
                cbar.set_label(f'Ice Drift Speed ({units_str})', rotation=270, labelpad=20)
            else:
                # Simple quiver plot
                quiver = ax.quiver(
                    lon_sub.values[valid_points], 
                    lat_sub.values[valid_points], 
                    u_sub.values[valid_points], 
                    v_sub.values[valid_points],
                    transform=ccrs.PlateCarree(), 
                    scale=vector_scale, 
                    width=0.003,
                    color='red',
                    alpha=0.8
                )
            
            # Add reference vector (in m/s)
            ax.quiverkey(quiver, 0.9, 0.1, 0.5, f'0.5 {units_str}', 
                        labelpos='E', coordinates='figure')
            
        else:
            print("Warning: No valid data points found for plotting.")
        
        # Get time information if available
        time_str = ""
        if 'time' in ds.dims and 'time' in ds.variables:
            try:
                # Try to decode time
                time_val = ds['time'].values[time_index]
                if hasattr(time_val, 'item'):
                    time_val = time_val.item()
                time_str = f" (Time step {time_index})"
            except:
                time_str = f" (Time index {time_index})"
        
        plt.title(f"Sea Ice Drift Vectors (Geographic Components){time_str}", fontsize=14, pad=20)
        
        # Add stats (convert to cm/s for display to match typical ice drift speeds)
        if np.any(valid_points):
            mean_speed = np.nanmean(magnitude.values[valid_points])
            max_speed = np.nanmax(magnitude.values[valid_points])
            # Display in cm/s for readability (typical ice drift is 5-30 cm/s)
            plt.figtext(0.02, 0.02, 
                       f"Mean speed: {mean_speed*100:.3f} cm/s | Max speed: {max_speed*100:.3f} cm/s",
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Close dataset
        ds.close()
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    filename = "/Volumes/Yotta_1/Antarctic/Ice_Drift/icemotion_daily_sh_25km_20230101_20231231_v4.1.nc"
    
    # Basic visualization
    load_and_visualize_ice_motion(filename, time_index=300)
    
    # You can also try different parameters:
    # load_and_visualize_ice_motion(filename, time_index=5, subsample_step=15, 
    #                              vector_scale=3e2, add_colorbar=True)
    
    # Example: Compare grid vs geographic components for a small region
    # This shows the difference between raw U,V (grid) and transformed components
    def compare_grid_vs_geographic(filename, time_index=0):
        """Show side-by-side comparison of grid vs geographic components"""
        try:
            ds = xr.open_dataset(filename, decode_times=False)
            u_grid = ds['u'].isel(time=time_index) if 'time' in ds.dims else ds['u']
            v_grid = ds['v'].isel(time=time_index) if 'time' in ds.dims else ds['v']
            lat = ds['latitude']
            lon = ds['longitude']
            
            # Convert units
            u_units = ds['u'].attrs.get('units', '').strip()
            if 'cm' in u_units.lower():
                u_grid = u_grid / 100.0
                v_grid = v_grid / 100.0
            
            # Transform to geographic
            u_east, v_north = transform_grid_to_geographic(u_grid, v_grid, lon, lat)
            
            # Subsample
            step = 10
            u_grid_sub = u_grid[::step, ::step]
            v_grid_sub = v_grid[::step, ::step]
            u_east_sub = u_east[::step, ::step]
            v_north_sub = v_north[::step, ::step]
            lat_sub = lat[::step, ::step]
            lon_sub = lon[::step, ::step]
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                         subplot_kw={'projection': ccrs.SouthPolarStereo()})
            
            for ax in [ax1, ax2]:
                ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
                ax.coastlines()
                ax.gridlines(alpha=0.5)
            
            # Plot grid components
            ax1.quiver(lon_sub, lat_sub, u_grid_sub, v_grid_sub,
                      transform=ccrs.PlateCarree(), scale=7e2, width=0.003, color='red')
            ax1.set_title('Grid Components (Along-X, Along-Y)', fontsize=12)
            
            # Plot geographic components  
            ax2.quiver(lon_sub, lat_sub, u_east_sub, v_north_sub,
                      transform=ccrs.PlateCarree(), scale=5e2, width=0.003, color='blue')
            ax2.set_title('Geographic Components (Eastward, Northward)', fontsize=12)
            
            plt.tight_layout()
            plt.show()
            ds.close()
            
        except Exception as e:
            print(f"Error in comparison: {str(e)}")
            import traceback
            traceback.print_exc()
