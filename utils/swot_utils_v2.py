"""
SWOT Utility Functions - Consolidated

Shared utilities for:
- swot_classifier.py (main classifier with incidence correction)
- example_classify_swot_ross_sea.py (main pipeline)

Single source of truth for all helper functions.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import uniform_filter, binary_opening, binary_closing
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Incidence-Angle Utilities
# ============================================================================

def incidence_proxy_from_cross_track_distance(
    x_km: np.ndarray,
    R_earth: float = 6_371_000.0,
    h_orbit: float = 891_000.0
) -> np.ndarray:
    """
    Approximate incidence angle (deg) from cross-track distance (km).
    
    Uses spherical geometry:
      phi = |x| / R
      s   = sqrt((R+h)^2 + R^2 - 2R(R+h)cos(phi))
      inc = arccos(((R+h)cos(phi) - R) / s)
    
    Parameters
    ----------
    x_km : ndarray
        Cross-track distance in km (can be 1D or 2D)
    R_earth : float
        Earth radius in meters (default 6,371 km)
    h_orbit : float
        Orbital altitude in meters (default 891 km for SWOT)
        
    Returns
    -------
    ndarray
        Incidence angle in degrees
    """
    x_m = np.abs(x_km) * 1000.0
    phi = x_m / R_earth
    s = np.sqrt(
        (R_earth + h_orbit) ** 2
        + R_earth**2
        - 2.0 * R_earth * (R_earth + h_orbit) * np.cos(phi)
    )
    cos_inc = ((R_earth + h_orbit) * np.cos(phi) - R_earth) / s
    cos_inc = np.clip(cos_inc, -1.0, 1.0)
    inc_rad = np.arccos(cos_inc)
    return np.degrees(inc_rad)


def _broadcast_to_2d(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Broadcast array to 2D shape (H, W).
    
    Accepts: (H,W), (W,), (1,W), (H,1), (H, W) and broadcasts to (H,W).
    
    Parameters
    ----------
    arr : ndarray
        Input array
    shape_hw : tuple
        Target shape (H, W)
        
    Returns
    -------
    ndarray
        Broadcasted array of shape (H, W)
    """
    H, W = shape_hw
    a = np.asarray(arr)
    if a.shape == (H, W):
        return a
    if a.ndim == 1 and a.size == W:
        return np.broadcast_to(a[None, :], (H, W))
    if a.ndim == 2 and a.shape == (1, W):
        return np.broadcast_to(a, (H, W))
    if a.ndim == 1 and a.size == H:
        return np.broadcast_to(a[:, None], (H, W))
    if a.ndim == 2 and a.shape == (H, 1):
        return np.broadcast_to(a, (H, W))
    raise ValueError(f"Cannot broadcast array of shape {a.shape} to (H,W)={(H,W)}")


def _robust_linear_slope_from_binned_medians(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 18,
    clip_sigma: float = 4.0
) -> Tuple[float, float]:
    """
    Robustly estimate slope of y vs x using binned medians + polyfit.
    
    Key steps:
    1. Remove outliers (>4 sigma in y)
    2. Bin x, compute median of x and y in each bin
    3. Fit line through binned medians
    4. Return slope and x reference point
    
    Parameters
    ----------
    x : ndarray
        Independent variable (e.g., incidence angle)
    y : ndarray
        Dependent variable (e.g., sigma0)
    n_bins : int
        Number of bins for robustness (default 18)
    clip_sigma : float
        Outlier clipping threshold (default 4.0 sigma)
        
    Returns
    -------
    slope : float
        dσ0/dθ (change in sigma0 per degree incidence)
    x_ref : float
        Reference incidence angle (median x)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    
    if x.size < 50:
        return 0.0, float(np.nanmedian(x)) if x.size else 0.0

    # Outlier clipping in y
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) + 1e-12
    robust_std = 1.4826 * mad
    keep = np.abs(y - med) <= clip_sigma * robust_std
    x = x[keep]
    y = y[keep]
    
    if x.size < 50:
        return 0.0, float(np.nanmedian(x)) if x.size else 0.0

    # Bin in x
    x_min, x_max = np.nanpercentile(x, [2, 98])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return 0.0, float(np.nanmedian(x))

    edges = np.linspace(x_min, x_max, n_bins + 1)
    x_med = []
    y_med = []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if np.sum(m) >= 200:
            x_med.append(np.nanmedian(x[m]))
            y_med.append(np.nanmedian(y[m]))

    if len(x_med) < 3:
        # Fallback: direct fit
        p = np.polyfit(x, y, 1)
        slope = float(p[0])
        return slope, float(np.nanmedian(x))

    x_med = np.asarray(x_med)
    y_med = np.asarray(y_med)
    p = np.polyfit(x_med, y_med, 1)
    slope = float(p[0])
    return slope, float(np.nanmedian(x))


# ============================================================================
# Column Bias Removal
# ============================================================================

def remove_column_bias(
    field2d: np.ndarray,
    valid_mask2d: Optional[np.ndarray] = None,
    min_valid: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove additive cross-track bias by subtracting robust per-column median trend.
    
    This reduces:
    - Incidence angle artifacts (residual after incidence correction)
    - Cross-track striping from instrument
    - Rail artifacts near nadir gap edges
    
    Parameters
    ----------
    field2d : ndarray [H, W]
        2D field to detrend (e.g., sigma0, SSH)
    valid_mask2d : ndarray [H, W], optional
        Boolean mask where data are valid. If None, uses isfinite()
    min_valid : int
        Minimum valid pixels per column to compute median (default 200)
        
    Returns
    -------
    field_detrended : ndarray [H, W]
        Detrended field
    bias_per_column : ndarray [W]
        Per-column bias that was subtracted
    """
    f = field2d.astype(np.float64, copy=True)
    m = np.isfinite(f)
    if valid_mask2d is not None:
        m &= valid_mask2d.astype(bool)

    n_cross = f.shape[1]
    col_med = np.full(n_cross, np.nan, dtype=np.float64)

    # Compute median per column
    for j in range(n_cross):
        vals = f[m[:, j], j]
        if vals.size >= min_valid:
            col_med[j] = np.nanmedian(vals)

    # Fill gaps via interpolation
    jj = np.arange(n_cross)
    good = np.isfinite(col_med)
    if good.sum() < 2:
        return f, np.zeros(n_cross, dtype=np.float64)

    col_med_filled = np.interp(jj, jj[good], col_med[good])
    bias = col_med_filled - np.nanmedian(col_med_filled)

    f_detrended = f - bias[None, :]
    return f_detrended, bias


def make_inner_edge_mask(valid2d: np.ndarray, buffer_cols: int = 6) -> np.ndarray:
    """
    Identify columns adjacent to the nadir gap (inner-swath edge).
    
    These regions often have artifacts:
    - Incidence angle near 0° (near-nadir effects)
    - Misclassification (often as "open water" or "polynya")
    - Should be excluded from fits or flagged
    
    Parameters
    ----------
    valid2d : ndarray [H, W]
        Boolean mask where data are valid
    buffer_cols : int
        Number of columns on each side of gap edges to include (default 6)
        
    Returns
    -------
    mask2d : ndarray [H, W]
        Boolean mask True where pixels are in inner-edge buffer zone
    """
    valid2d = valid2d.astype(bool)
    col_valid_frac = np.mean(valid2d, axis=0)
    gap_cols = np.where(col_valid_frac < 0.05)[0]
    mask = np.zeros_like(valid2d, dtype=bool)

    if gap_cols.size == 0:
        return mask

    left_edge = gap_cols.min() - 1
    right_edge = gap_cols.max() + 1

    n_cross = valid2d.shape[1]
    left_start = max(0, left_edge - buffer_cols + 1)
    left_end = min(n_cross, left_edge + buffer_cols + 1)
    right_start = max(0, right_edge - buffer_cols)
    right_end = min(n_cross, right_edge + buffer_cols)

    if 0 <= left_edge < n_cross:
        mask[:, left_start:left_end] = True
    if 0 <= right_edge < n_cross:
        mask[:, right_start:right_end] = True

    return mask


# ============================================================================
# Texture & Smoothing
# ============================================================================

def local_std_nan(img: np.ndarray, size: int = 21) -> np.ndarray:
    """
    Compute local standard deviation with NaN-safe normalization.
    
    Uses uniform_filter to robustly handle NaN values.
    
    Parameters
    ----------
    img : ndarray
        Input image
    size : int
        Window size for local std computation (default 21)
        
    Returns
    -------
    output : ndarray
        Local standard deviation, same shape as input, NaN where input is NaN
    """
    if not HAS_SCIPY:
        logger.warning("SciPy not available; cannot compute local_std_nan()")
        return np.full_like(img, np.nan)
    
    img = img.astype(np.float32)
    m = np.isfinite(img)
    x = np.where(m, img, 0.0)

    # Count valid pixels in window
    w = uniform_filter(m.astype(np.float32), size=size)
    
    # Local mean
    mean = uniform_filter(x, size=size) / np.maximum(w, 1e-6)
    
    # Local mean of squares
    mean2 = uniform_filter(x * x, size=size) / np.maximum(w, 1e-6)

    # Variance (clipped to avoid numerical errors)
    var = np.maximum(mean2 - mean * mean, 0.0)
    out = np.sqrt(var)
    
    # Restore NaN
    out[~m] = np.nan
    return out


def majority_filter_3x3(labels2d: np.ndarray) -> np.ndarray:
    """
    Apply simple 3x3 majority filter (mode).
    
    Useful for smoothing classification labels and reducing salt-and-pepper noise.
    
    Parameters
    ----------
    labels2d : ndarray [H, W]
        Integer label array
        
    Returns
    -------
    output : ndarray [H, W]
        Smoothed labels via 3x3 majority voting
    """
    lab = labels2d.copy()
    pad = np.pad(lab, ((1, 1), (1, 1)), mode="edge")
    out = lab.copy()

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            win = pad[i:i+3, j:j+3].ravel()
            vals, counts = np.unique(win, return_counts=True)
            out[i, j] = vals[np.argmax(counts)]
    
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("SWOT utility functions loaded successfully")