"""
SWOT-ICeSat-2 Collocation from CSV Matches

Reads CSV file containing matched SWOT-ICeSat-2 pairs
For each row: loads SWOT and ICeSat-2 files, extracts collocation data, saves to MAT

CSV format (from overlap analysis):
Column 0: IS2 filename
Column 1: IS2 date/time
Column 2: IS2 obs count
Column 3: IS2 Ross Sea obs
Column 4: IS2 percentage
Column 5: SWOT filename
Column 6: SWOT date/time
Column 7+: Other metrics
"""

import numpy as np
import xarray as xr
import h5py
from scipy.io import savemat
import pandas as pd
from pathlib import Path
from glob import glob
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================

CSV_FILE = "ross_sea_matches_with_polynya_202310.csv"

BASE_PATH = "/Volumes/One Touch/SWOT/SWOT-Oceanography-85c046820f1b47545b27d2366f84e4a7d58cd866"
SWOT_FOLDER = f"{BASE_PATH}/swot_temp"
IS2_FOLDER = f"{BASE_PATH}/icesat2_temp"
ATL10_FOLDER = f"/Volumes/One Touch/SWOT/ATL10"

OUTPUT_FOLDER = f"{BASE_PATH}/SWOTIS2_mat"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Geographic region (Ross Sea)
MIN_LAT, MAX_LAT = -78, -70
MIN_LON, MAX_LON = 170, 210

# Distance threshold (meters)
DIST_THRESHOLD = 4000

BEAMS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

print("\n" + "="*80)
print("SWOT-ICeSat-2 COLLOCATION FROM CSV MATCHES")
print("="*80 + "\n")

print("Configuration:")
print(f"  CSV file: {CSV_FILE}")
print(f"  SWOT folder: {SWOT_FOLDER}")
print(f"  IS2 folder: {IS2_FOLDER}")
print(f"  Output folder: {OUTPUT_FOLDER}")
print(f"  Region: Lat [{MIN_LAT}, {MAX_LAT}], Lon [{MIN_LON}, {MAX_LON}]")
print(f"  Distance threshold: {DIST_THRESHOLD} m\n")

# =====================================================================
# UTILITIES
# =====================================================================

def wrap_lon_360(lon):
    """Convert longitude to 0-360 convention."""
    return np.mod(lon, 360.0)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in meters."""
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat_rad = np.radians(lat2_rad - lat1_rad)
    
    dlon = lon2 - lon1
    dlon[np.abs(dlon) > 180] = dlon[np.abs(dlon) > 180] - np.sign(dlon[np.abs(dlon) > 180]) * 360
    dlon_rad = np.radians(dlon)
    
    a = np.sin(dlat_rad/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def extract_date_from_filename(filename):
    """Extract YYYYMMDD from filename."""
    match = re.search(r'\d{8}', str(filename))
    if match:
        return match.group()
    return None

# =====================================================================
# READ CSV
# =====================================================================

print("Reading CSV file...")

if not Path(CSV_FILE).exists():
    print(f"ERROR: CSV file not found: {CSV_FILE}")
    exit(1)

# Read CSV without header
df = pd.read_csv(CSV_FILE, header=None)
n_matches = len(df)

print(f"✅ Found {n_matches} matched pairs\n")

# =====================================================================
# PROCESS EACH MATCHED PAIR
# =====================================================================

success_count = 0
fail_count = 0

for idx, row in df.iterrows():
    is2_filename = str(row[0]).strip()
    swot_filename = str(row[5]).strip()
    
    # Extract date
    date_str = extract_date_from_filename(swot_filename)
    if date_str is None:
        print(f"[{idx+1}/{n_matches}] ⊘ Cannot parse date from SWOT filename")
        fail_count += 1
        continue
    
    print(f"[{idx+1}/{n_matches}] Date: {date_str}")
    print(f"       IS2:  {is2_filename}")
    print(f"       SWOT: {swot_filename}")
    
    # =====================================================================
    # LOAD SWOT DATA
    # =====================================================================
    swot_filepath = Path(SWOT_FOLDER) / swot_filename
    
    if not swot_filepath.exists():
        print(f"  ✗ SWOT file not found")
        fail_count += 1
        continue
    
    try:
        ds = xr.open_dataset(swot_filepath)
        ssh = ds["ssha_filtered"].values
        mss = ds["mss"].values
        sigma0 = ds["sigma0"].values
        quality_flag = ds["quality_flag"].values
        lat = ds["latitude"].values
        lon = wrap_lon_360(ds["longitude"].values)
        ds.close()
        
        print(f"  ✅ SWOT data loaded: {sigma0.shape[0]}×{sigma0.shape[1]}")
    except Exception as e:
        print(f"  ✗ Error loading SWOT: {e}")
        fail_count += 1
        continue
    
    # =====================================================================
    # EXTRACT REGION FROM SWOT
    # =====================================================================
    try:
        # Region mask
        region_mask = (lat >= MIN_LAT) & (lat <= MAX_LAT) & \
                      (lon >= MIN_LON) & (lon <= MAX_LON)
        
        row_idx, col_idx = np.where(region_mask)
        
        if len(row_idx) == 0:
            print(f"  ✗ No SWOT data in region")
            fail_count += 1
            continue
        
        i_min, i_max = row_idx.min(), row_idx.max()
        j_min, j_max = col_idx.min(), col_idx.max()
        
        # Extract region
        lat_cut = lat[i_min:i_max+1, j_min:j_max+1]
        lon_cut = lon[i_min:i_max+1, j_min:j_max+1]
        ssh_cut = ssh[i_min:i_max+1, j_min:j_max+1].astype(float)
        mss_cut = mss[i_min:i_max+1, j_min:j_max+1].astype(float)
        sigma0_cut = sigma0[i_min:i_max+1, j_min:j_max+1].astype(float)
        quality_cut = quality_flag[i_min:i_max+1, j_min:j_max+1]
        
        # Quality control
        sigma0_cut[quality_cut >= 100] = np.nan
        ssh_cut[quality_cut >= 100] = np.nan
        mss_cut[quality_cut >= 100] = np.nan
        sigma0_cut[sigma0_cut <= 0] = np.nan
        
        # Convert to dB
        sigma0_db = 10 * np.log10(sigma0_cut)
        
        # Flatten and filter
        swot_lat_vec = lat_cut.flatten()
        swot_lon_vec = lon_cut.flatten()
        swot_sigma_vec = sigma0_db.flatten()
        ssh_vec = ssh_cut.flatten()
        mss_vec = mss_cut.flatten()
        quality_vec = quality_cut.flatten()
        
        valid = np.isfinite(swot_sigma_vec)
        swot_lat_vec = swot_lat_vec[valid]
        swot_lon_vec = swot_lon_vec[valid]
        swot_sigma_vec = swot_sigma_vec[valid]
        ssh_vec = ssh_vec[valid]
        mss_vec = mss_vec[valid]
        quality_vec = quality_vec[valid]
        
        print(f"  SWOT region: {lat_cut.shape[0]}×{lat_cut.shape[1]} | Valid σ₀: {len(swot_sigma_vec)}")
    except Exception as e:
        print(f"  ✗ Error extracting SWOT region: {e}")
        fail_count += 1
        continue
    
    # =====================================================================
    # FIND ICeSat-2 FILE
    # =====================================================================
    is2_filepath = Path(IS2_FOLDER) / is2_filename
    if not is2_filepath.exists():
        is2_filepath = Path(ATL10_FOLDER) / is2_filename
    
    if not is2_filepath.exists():
        print(f"  ✗ IS2 file not found")
        fail_count += 1
        continue
    
    # =====================================================================
    # PROCESS ICeSat-2 DATA
    # =====================================================================
    beam_data = {}
    n_beams = 0
    
    try:
        with h5py.File(is2_filepath, 'r') as f:
            for beam in BEAMS:
                try:
                    prefix = f"/{beam}/freeboard_segment"
                    
                    if prefix not in f:
                        continue
                    
                    # Read beam data
                    lat_ice = f[f"{prefix}/latitude"][()]
                    lon_ice = wrap_lon_360(f[f"{prefix}/longitude"][()])
                    fb_ice = f[f"{prefix}/beam_fb_height"][()]
                    
                    # Read height and MSS if available
                    try:
                        height_ice = f[f"{prefix}/heights/height_segment_height"][()]
                    except:
                        height_ice = np.full_like(fb_ice, np.nan)
                    
                    try:
                        mss_ice = f[f"{prefix}/geophysical/height_segment_mss"][()]
                    except:
                        mss_ice = np.full_like(fb_ice, np.nan)
                    
                    # Filter by region
                    idx_region = np.where((lat_ice >= MIN_LAT) & (lat_ice <= MAX_LAT) & 
                                        (lon_ice >= MIN_LON) & (lon_ice <= MAX_LON))[0]
                    
                    if len(idx_region) == 0:
                        continue
                    
                    lat_ice = lat_ice[idx_region]
                    lon_ice = lon_ice[idx_region]
                    fb_ice = fb_ice[idx_region]
                    height_ice = height_ice[idx_region]
                    mss_ice = mss_ice[idx_region]
                    
                    # Check if inside SWOT polygon
                    # Simple bounding box check
                    in_swot = (lat_ice >= lat_cut.min()) & (lat_ice <= lat_cut.max()) & \
                             (lon_ice >= lon_cut.min()) & (lon_ice <= lon_cut.max())
                    
                    if not np.any(in_swot):
                        continue
                    
                    lat_ice = lat_ice[in_swot]
                    lon_ice = lon_ice[in_swot]
                    fb_ice = fb_ice[in_swot]
                    height_ice = height_ice[in_swot]
                    mss_ice = mss_ice[in_swot]
                    
                    # Match with SWOT data
                    sigma_mean = []
                    ssh_mean = []
                    mss_swot_list = []
                    quality_all = []
                    dist_min = []
                    
                    for k in range(len(lat_ice)):
                        # Distances
                        lats = np.full_like(swot_lat_vec, lat_ice[k])
                        lons = np.full_like(swot_lon_vec, lon_ice[k])
                        dists = haversine(lats, lons, swot_lat_vec, swot_lon_vec)
                        
                        nearby = dists < DIST_THRESHOLD
                        
                        if np.any(nearby):
                            sigma_mean.append(swot_sigma_vec[nearby])
                            ssh_mean.append(ssh_vec[nearby])
                            mss_swot_list.append(mss_vec[nearby])
                            quality_all.append(quality_vec[nearby])
                            dist_min.append(dists[nearby])
                        else:
                            sigma_mean.append(np.array([]))
                            ssh_mean.append(np.array([]))
                            mss_swot_list.append(np.array([]))
                            quality_all.append(np.array([]))
                            dist_min.append(np.array([]))
                    
                    # Store beam data
                    beam_data[beam] = {
                        'lat': lat_ice,
                        'lon': lon_ice,
                        'fb': fb_ice,
                        'height_ice2': height_ice,
                        'mss_ice2': mss_ice,
                        'sigma_swot': sigma_mean,
                        'ssh_swot': ssh_mean,
                        'mss_swot': mss_swot_list,
                        'quality_all': quality_all,
                        'distance': dist_min
                    }
                    
                    n_beams += 1
                    n_matched = sum([len(x) for x in sigma_mean])
                    print(f"    ✅ Beam {beam}: {len(lat_ice)} points ({n_matched} matched)")
                    
                except Exception as e:
                    continue
    except Exception as e:
        print(f"  ✗ Error processing IS2: {e}")
        fail_count += 1
        continue
    
    if n_beams == 0:
        print(f"  ✗ No beams processed")
        fail_count += 1
        continue
    
    # =====================================================================
    # SAVE TO MAT FILE
    # =====================================================================
    try:
        output_mat = f"{OUTPUT_FOLDER}/{date_str}_IS2_SWOT.mat"
        
        # Convert to MATLAB-compatible format
        save_dict = {}
        for beam, data in beam_data.items():
            save_dict[f"{beam}_lat"] = data['lat']
            save_dict[f"{beam}_lon"] = data['lon']
            save_dict[f"{beam}_fb"] = data['fb']
            save_dict[f"{beam}_height_ice2"] = data['height_ice2']
            save_dict[f"{beam}_mss_ice2"] = data['mss_ice2']
            
            # For cells, compute means as representative values
            sigma_means = np.array([np.nanmean(x) if len(x) > 0 else np.nan for x in data['sigma_mean']])
            save_dict[f"{beam}_sigma_swot_mean"] = sigma_means
        
        savemat(output_mat, save_dict, appendmat=False)
        
        print(f"  ✅ Saved: {output_mat} ({n_beams} beams)")
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ Error saving MAT file: {e}")
        fail_count += 1
        continue

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total matches processed: {n_matches}")
print(f"Successful: {success_count}")
print(f"Failed: {fail_count}")
print(f"Output folder: {OUTPUT_FOLDER}")
print("✅ Complete!")