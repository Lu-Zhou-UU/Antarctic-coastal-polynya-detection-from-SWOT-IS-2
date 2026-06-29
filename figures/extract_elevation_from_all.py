import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.io import loadmat
from datetime import datetime
from collections import deque
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import h5py
import matplotlib as mpl

mpl.rcParams["path.simplify"] = False
mpl.rcParams["path.simplify_threshold"] = 0.0

def load_mat_file(path: str):
    """Load .mat file, with fallback for v7.3 HDF5 format"""
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return load_hdf5_mat_v73(path)
        
def load_hdf5_mat_v73(path: str):
    """Load MATLAB v7.3 .mat files (saved as HDF5)"""
    def _read(obj, f):
        if isinstance(obj, h5py.Dataset):
            # Handle reference types
            if obj.dtype == np.object_ or h5py.check_dtype(ref=obj.dtype) is not None:
                refs = obj[()]
                out = np.empty(refs.shape, dtype=object)
                it = np.nditer(refs, flags=["multi_index", "refs_ok"])
                for r in it:
                    out[it.multi_index] = _read(f[r.item()], f)
                return out
            val = obj[()]
            # Transpose arrays (MATLAB convention)
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                val = val.T
            # Handle string types
            if val.dtype == np.uint16:
                return "".join(chr(c) for c in val.flatten())
            return val
        if isinstance(obj, h5py.Group):
            return {k: _read(obj[k], f) for k in obj.keys()}
        return obj
    with h5py.File(path, "r") as f:
        return {k: _read(f[k], f) for k in f.keys()}

# ============================================================================
# LOAD AND INSPECT SWOT DATA
# ============================================================================

print("="*80)
print("LOADING SWOT MAT FILE")
print("="*80)

swot_file = "/Volumes/One Touch/SWOT/SWOT-Oceanography-85c046820f1b47545b27d2366f84e4a7d58cd866/20230921_temp.mat"

try:
    SWOT = load_mat_file(swot_file)
    print(f"\n✅ Successfully loaded: {swot_file}\n")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# ============================================================================
# DISPLAY TOP-LEVEL STRUCTURE
# ============================================================================

print("="*80)
print("TOP-LEVEL VARIABLES IN SWOT DICTIONARY")
print("="*80)

if isinstance(SWOT, dict):
    print(f"\nNumber of top-level variables: {len(SWOT)}\n")
    
    for i, key in enumerate(sorted(SWOT.keys()), 1):
        value = SWOT[key]
        
        # Get type and shape information
        if isinstance(value, np.ndarray):
            dtype = str(value.dtype)
            shape = value.shape
            size = value.size
            info = f"array{shape} | dtype={dtype} | size={size:,}"
        elif isinstance(value, dict):
            info = f"dict with {len(value)} keys"
        elif isinstance(value, list):
            info = f"list with {len(value)} elements"
        elif isinstance(value, str):
            info = f"string: '{value[:100]}...'" if len(value) > 100 else f"string: '{value}'"
        else:
            info = f"{type(value).__name__}"
        
        print(f"{i:2d}. {key:30s} : {info}")

# ============================================================================
# DETAILED INSPECTION OF EACH VARIABLE
# ============================================================================

print("\n" + "="*80)
print("DETAILED VARIABLE INSPECTION")
print("="*80)

def print_variable_details(key, value, max_rows=10):
    """Print detailed information about a variable"""
    print(f"\n{'─'*80}")
    print(f"Variable: {key}")
    print(f"{'─'*80}")
    
    if isinstance(value, np.ndarray):
        print(f"Type: numpy array")
        print(f"Shape: {value.shape}")
        print(f"Dtype: {value.dtype}")
        print(f"Size: {value.size:,} elements")
        print(f"Memory: {value.nbytes / 1e6:.2f} MB")
        print(f"Min/Max: {np.nanmin(value):.6e} / {np.nanmax(value):.6e}")
        
        # Show sample values
        if value.ndim == 1:
            if len(value) > 0:
                print(f"First 5 values: {value[:5]}")
                print(f"Last 5 values: {value[-5:]}")
        elif value.ndim == 2:
            print(f"Shape: {value.shape[0]} rows × {value.shape[1]} columns")
            if value.shape[0] > 0 and value.shape[1] > 0:
                print(f"First row: {value[0, :5]}")
        
    elif isinstance(value, dict):
        print(f"Type: dictionary")
        print(f"Keys: {sorted(value.keys())}")
        for k in sorted(value.keys())[:5]:  # Show first 5
            v = value[k]
            if isinstance(v, np.ndarray):
                print(f"  {k}: array{v.shape} dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__}")
    
    elif isinstance(value, (list, tuple)):
        print(f"Type: {type(value).__name__}")
        print(f"Length: {len(value)}")
        if len(value) > 0:
            print(f"First element type: {type(value[0]).__name__}")
    
    elif isinstance(value, str):
        print(f"Type: string")
        print(f"Length: {len(value)} characters")
        print(f"Value: {value}")
    
    else:
        print(f"Type: {type(value).__name__}")
        print(f"Value: {value}")

# Show details for each variable
for key in sorted(SWOT.keys()):
    print_variable_details(key, SWOT[key])

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nFile: {swot_file}")
print(f"Total variables: {len(SWOT)}")

# Count different types
array_vars = sum(1 for v in SWOT.values() if isinstance(v, np.ndarray))
dict_vars = sum(1 for v in SWOT.values() if isinstance(v, dict))
str_vars = sum(1 for v in SWOT.values() if isinstance(v, str))
other_vars = len(SWOT) - array_vars - dict_vars - str_vars

print(f"  - Arrays: {array_vars}")
print(f"  - Dicts: {dict_vars}")
print(f"  - Strings: {str_vars}")
print(f"  - Other: {other_vars}")

# Total data size
total_size = sum(
    v.nbytes / 1e6 if isinstance(v, np.ndarray) else 0 
    for v in SWOT.values()
)
print(f"Total data size: {total_size:.2f} MB")

# ============================================================================
# SPECIAL HANDLING FOR KNOWN STRUCTURES
# ============================================================================

print("\n" + "="*80)
print("STRUCTURE ANALYSIS")
print("="*80)

# Check for common SWOT/IS2 variables
expected_vars = {
    'latitude': 'Geographic latitude',
    'longitude': 'Geographic longitude',
    'lat': 'Geographic latitude (short)',
    'lon': 'Geographic longitude (short)',
    'ssh': 'Sea surface height',
    'sic': 'Sea ice concentration',
    'sigma0': 'Sigma-0 (backscatter)',
    'time': 'Time data',
    'gt1l': 'Beam gt1l',
    'gt1r': 'Beam gt1r',
    'gt2l': 'Beam gt2l',
    'gt2r': 'Beam gt2r',
    'gt3l': 'Beam gt3l',
    'gt3r': 'Beam gt3r',
}

print("\nExpected variables found:")
for var_name, description in expected_vars.items():
    if var_name in SWOT:
        value = SWOT[var_name]
        if isinstance(value, np.ndarray):
            print(f"  ✅ {var_name:15s} : {description:30s} {value.shape}")
        else:
            print(f"  ✅ {var_name:15s} : {description:30s} {type(value).__name__}")
    else:
        print(f"  ❌ {var_name:15s} : {description:30s} NOT FOUND")

# ============================================================================
# RECOMMENDATIONS FOR DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATIONS FOR DATA ACCESS")
print("="*80)

print("""
To access your data, use patterns like:

1. For arrays:
   >>> lat = SWOT['latitude']
   >>> lon = SWOT['longitude']
   >>> ssh = SWOT['ssh']

2. For nested structures (if present):
   >>> beam_data = SWOT['gt1l']
   >>> beam_lat = beam_data['latitude']

3. For lists/cells:
   >>> cells = SWOT['cell_var']
   >>> first_cell = cells[0]

4. Check data ranges:
   >>> print(f"Lat: {np.nanmin(SWOT['latitude']):.2f} to {np.nanmax(SWOT['latitude']):.2f}")
   >>> print(f"Lon: {np.nanmin(SWOT['longitude']):.2f} to {np.nanmax(SWOT['longitude']):.2f}")

5. Handle NaNs:
   >>> valid_mask = ~np.isnan(SWOT['ssh'])
   >>> valid_ssh = SWOT['ssh'][valid_mask]
""")

print("="*80)