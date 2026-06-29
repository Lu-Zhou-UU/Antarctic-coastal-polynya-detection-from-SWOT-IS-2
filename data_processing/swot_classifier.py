#!/usr/bin/env python3
"""
swot_classifier.py - PROPERLY FIXED TWO-STAGE VERSION

Key improvements over both previous versions:
1. Stage 1: KMeans on sigma0+texture ONLY (no SSH bias)
2. Stage 2: Use SSH to refine candidate classes (polynya vs leads)
3. Better edge artifact handling
4. Conservative incidence correction
5. SSH used ONLY where it helps (not everywhere)

This combines the working logic from the old algorithm with fixes for edge artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional deps
try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from scipy.ndimage import (
        uniform_filter,
        binary_dilation,
        binary_opening,
        binary_closing,
        label as scipy_label,
    )
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
@dataclass
class SWOTClassification:
    sigma0: np.ndarray
    ssh: Optional[np.ndarray]
    lat: Optional[np.ndarray]
    lon: Optional[np.ndarray]
    sigma0_corrected: Optional[np.ndarray]
    class_labels: np.ndarray
    class_names: List[str]
    class_map: Dict[str, int]
    confidence: np.ndarray
    method: str
    n_clusters: int
    meta: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)
    exclude_mask: Optional[np.ndarray] = None
    mask_vals: Optional[np.ndarray] = None

    def get_class_mask(self, class_name: str) -> np.ndarray:
        if class_name not in self.class_map:
            return np.zeros_like(self.class_labels, dtype=bool)
        return self.class_labels == int(self.class_map[class_name])

    def get_class_stats(self, class_name: str) -> Dict[str, float]:
        m = self.get_class_mask(class_name)
        n_total_valid = int(np.sum(self.class_labels >= 0))
        n = int(np.sum(m))
        frac = (n / n_total_valid) if n_total_valid > 0 else 0.0
        out = {
            "n_pixels": n,
            "fraction": float(frac),
            "sigma0_mean": float(np.nanmean(self.sigma0[m])) if n > 0 else np.nan,
            "sigma0_std": float(np.nanstd(self.sigma0[m])) if n > 0 else np.nan,
            "ssh_mean": np.nan,
            "ssh_std": np.nan,
        }
        if self.ssh is not None and n > 0:
            out["ssh_mean"] = float(np.nanmean(self.ssh[m]))
            out["ssh_std"] = float(np.nanstd(self.ssh[m]))
        return out


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _require_sklearn() -> None:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required: pip install scikit-learn")


def _require_scipy(feature: str = "this feature") -> None:
    if not HAS_SCIPY:
        raise ImportError(f"SciPy required for {feature}: pip install scipy")


def _combine_valid_and_exclude(valid2d: np.ndarray, exclude_mask: Optional[np.ndarray]) -> np.ndarray:
    if exclude_mask is None:
        return valid2d
    ex = np.asarray(exclude_mask, dtype=bool)
    if ex.shape != valid2d.shape:
        raise ValueError("exclude_mask shape mismatch")
    return valid2d & (~ex)


def _zscore_robust(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def make_inner_edge_mask(valid2d: np.ndarray, buffer_cols: int = 25) -> np.ndarray:
    """Mask columns adjacent to the nadir gap."""
    valid2d = np.asarray(valid2d, dtype=bool)
    col_valid_frac = np.mean(valid2d, axis=0)
    gap_cols = np.where(col_valid_frac < 0.05)[0]
    mask = np.zeros_like(valid2d, dtype=bool)

    if gap_cols.size == 0:
        return mask

    left_edge = int(gap_cols.min() - 1)
    right_edge = int(gap_cols.max() + 1)
    W = valid2d.shape[1]
    b = int(buffer_cols)

    left_start = max(0, left_edge - b + 1)
    left_end = min(W, left_edge + b + 1)
    right_start = max(0, right_edge - b)
    right_end = min(W, right_edge + b)

    if 0 <= left_edge < W:
        mask[:, left_start:left_end] = True
    if 0 <= right_edge < W:
        mask[:, right_start:right_end] = True

    return mask


def make_outer_edge_mask(valid2d: np.ndarray, buffer_cols: int = 15) -> np.ndarray:
    """
    NEW: Mask outer edges of swath (far range) where artifacts are common.
    """
    valid2d = np.asarray(valid2d, dtype=bool)
    H, W = valid2d.shape
    mask = np.zeros_like(valid2d, dtype=bool)
    
    b = int(buffer_cols)
    
    # Mask left and right edges
    mask[:, :b] = True  # Left edge
    mask[:, -b:] = True  # Right edge
    
    return mask


def detect_rail_cols(
    sigma0_corr: np.ndarray,
    valid2d: np.ndarray,
    *,
    z: float = 3.5,
    min_per_col: int = 50,
) -> np.ndarray:
    """Detect anomalously bright columns using MAD z-score."""
    s = np.asarray(sigma0_corr, dtype=np.float32)
    v = np.asarray(valid2d, dtype=bool)
    H, W = s.shape

    col_med = np.full(W, np.nan, dtype=np.float32)
    for j in range(W):
        vv = v[:, j] & np.isfinite(s[:, j])
        if int(np.sum(vv)) >= int(min_per_col):
            col_med[j] = np.nanmedian(s[vv, j])

    g = np.isfinite(col_med)
    if int(np.sum(g)) < 10:
        return np.zeros(W, dtype=bool)

    med = float(np.nanmedian(col_med[g]))
    mad = float(np.nanmedian(np.abs(col_med[g] - med))) + 1e-6
    zscore = (col_med - med) / (1.4826 * mad)
    return np.isfinite(zscore) & (zscore > float(z))


def expand_mask_with_cols(mask2d: np.ndarray, cols: np.ndarray, buffer: int) -> np.ndarray:
    """Expand mask to include specified columns with buffer."""
    mask = np.asarray(mask2d, dtype=bool).copy()
    cols = np.asarray(cols, dtype=bool)
    H, W = mask.shape
    b = int(buffer)

    jj = np.where(cols)[0]
    for j in jj:
        j0 = max(0, j - b)
        j1 = min(W, j + b + 1)
        mask[:, j0:j1] = True
    return mask


def local_std_nan(img: np.ndarray, size: int = 21) -> np.ndarray:
    """NaN-safe local standard deviation."""
    _require_scipy("texture")
    img = np.asarray(img, dtype=np.float32)
    m = np.isfinite(img)
    x = np.where(m, img, 0.0)

    w = uniform_filter(m.astype(np.float32), size=size)
    w_safe = np.maximum(w, 1e-9)

    mean = uniform_filter(x, size=size) / w_safe
    mean2 = uniform_filter(x * x, size=size) / w_safe

    var = np.maximum(mean2 - mean * mean, 0.0)
    out = np.sqrt(var)

    min_valid_frac = 0.1
    min_valid_count = max(1, int(size * size * min_valid_frac))
    insufficient = (w * size * size) < min_valid_count
    out[insufficient] = np.nan
    return out


def majority_filter_3x3(labels2d: np.ndarray) -> np.ndarray:
    """3x3 majority filter that ignores -1."""
    lab = labels2d.astype(np.int32, copy=True)
    pad = np.pad(lab, ((1, 1), (1, 1)), mode="edge")
    out = lab.copy()
    H, W = out.shape
    for i in range(H):
        for j in range(W):
            win = pad[i:i + 3, j:j + 3].ravel()
            win = win[win >= 0]
            if win.size == 0:
                out[i, j] = -1
            else:
                vals, counts = np.unique(win, return_counts=True)
                out[i, j] = int(vals[int(np.argmax(counts))])
    return out


def _rank_names_for_k(n: int) -> List[str]:
    """Generate class names from bright to dark."""
    # Standard names for different cluster counts
    if n == 3:
        return ["polynya", "thin_ice", "pack_ice"]
    elif n == 4:
        return ["polynya", "thin_ice", "first_year_ice", "pack_ice"]
    elif n == 5:
        return ["polynya", "thin_ice", "first_year_ice", "multi_year_ice", "pack_ice"]
    elif n == 6:
        return ["open_water", "thin_ice", "young_ice", "first_year_ice", "multi_year_ice", "pack_ice"]
    elif n == 7:
        return ["open_water", "thin_ice", "young_ice", "first_year_ice", "multi_year_ice", "old_ice", "pack_ice"]
    else:
        # Generic names for other counts
        base = ["open_water", "thin_ice", "young_ice", "first_year_ice", "multi_year_ice"]
        names = base[:max(0, n - 1)] + ["pack_ice"]
        while len(names) < n:
            names.insert(-1, f"ice_type_{len(names) + 1}")
        return names[:n]


def component_morphology_score(mask: np.ndarray, min_size: int = 300) -> float:
    """Higher score => more 'polynya-like' (large + compact)."""
    _require_scipy("component morphology")
    m = np.asarray(mask, dtype=bool)
    lab, n = scipy_label(m)
    if n == 0:
        return 0.0

    areas = []
    compacts = []
    for rid in range(1, n + 1):
        r = (lab == rid)
        a = int(np.sum(r))
        if a < int(min_size):
            continue
        perim = float(np.sum(binary_dilation(r) & (~r)))
        compact = (4.0 * np.pi * a) / (perim * perim + 1e-6)
        areas.append(float(a))
        compacts.append(float(compact))

    if len(areas) == 0:
        return 0.0

    return float(np.median(areas) * np.median(compacts))


# -----------------------------------------------------------------------------
# Incidence correction
# -----------------------------------------------------------------------------
def region_aware_incidence_correction(
    sigma0: np.ndarray,
    incidence: np.ndarray,
    fit_mask: np.ndarray,
    *,
    nadir_max: float = 1.8,  # Below this: apply column destriping (INCREASED to cover full striped region)
    outer_min: float = 3.2,  # Above this: bin-wise (avoid artifacts)
    n_bins_edge: int = 30,   # Number of bins for outer edge
    apply_column_destriping: bool = True,  # Destripe nadir/inner region
) -> Tuple[np.ndarray, Dict]:
    """
    Hybrid incidence correction:
    - Nadir (<nadir_max): COLUMN DESTRIPING (remove per-column offsets causing stripes)
    - Inner+Middle: EXPONENTIAL correction (physical model: σ0 decays exponentially)
    - Outer edge (>outer_min): BIN-WISE mean removal (robust to artifacts)
    
    Exponential correction: σ0_corr = σ0 - A * exp(B * (inc - inc_ref))
    This captures the physical backscatter behavior better than linear.
    """
    s = np.asarray(sigma0, dtype=np.float32)
    inc = np.asarray(incidence, dtype=np.float32)
    fm = np.asarray(fit_mask, dtype=bool) & np.isfinite(s) & np.isfinite(inc)
    
    # Clip outliers
    s = np.clip(s, -50, 50)
    
    if int(np.sum(fm)) < 1000:
        return s.copy(), {"used": False, "reason": "insufficient_data"}
    
    s_corr = s.copy()
    H, W = s.shape
    
    # =================================================================
    # REGION 1: Nadir (inc < nadir_max) - COLUMN DESTRIPING
    # =================================================================
    nadir_mask = np.isfinite(inc) & (inc < nadir_max)
    n_nadir = int(np.sum(nadir_mask))
    
    if apply_column_destriping and n_nadir > 0:
        # Compute global reference (median of all data or middle swath)
        ref_mask = fm & (inc >= nadir_max) & (inc <= outer_min)
        if int(np.sum(ref_mask)) > 0:
            global_ref = float(np.median(s[ref_mask]))
        else:
            global_ref = float(np.median(s[fm]))
        
        # Column-wise destriping: subtract median of each column
        n_destriped_cols = 0
        for col in range(W):
            col_nadir_mask = nadir_mask[:, col] & fm[:, col]
            n_col = int(np.sum(col_nadir_mask))
            if n_col >= 10:  # Need at least 10 pixels per column
                col_median = float(np.median(s[col_nadir_mask, col]))
                col_correction = col_median - global_ref
                s_corr[nadir_mask[:, col], col] = s[nadir_mask[:, col], col] - col_correction
                n_destriped_cols += 1
        
        logger.info(f"    → Destriped {n_destriped_cols}/{W} columns in nadir region")
    # If no destriping, nadir region stays unchanged
    
    # =================================================================
    # REGION 2: Inner + Middle (nadir_max <= inc <= outer_min) - EXPONENTIAL
    # =================================================================
    inner_middle_mask = (inc >= nadir_max) & (inc <= outer_min) & np.isfinite(inc)
    inner_middle_fit = fm & inner_middle_mask
    
    if int(np.sum(inner_middle_fit)) >= 500:
        inc_fit = inc[inner_middle_fit]
        s_fit = s[inner_middle_fit]
        
        # Exponential fit: sigma0 = A * exp(B * incidence) + C
        # Linearize: log(sigma0 - min) = log(A) + B * incidence
        # For backscatter, we expect exponential decay with incidence
        
        # Use scipy curve_fit for robust exponential fitting
        from scipy.optimize import curve_fit
        
        inc_ref = float(np.median(inc_fit))
        
        # Define exponential correction model: correction = A * exp(B * (inc - inc_ref))
        def exp_model(inc, A, B):
            return A * np.exp(B * (inc - inc_ref))
        
        # Fit the trend (robust to outliers)
        try:
            # Initial guess: A ~ range/2, B ~ -0.5 (typical backscatter decay)
            p0 = [5.0, -0.3]
            popt, _ = curve_fit(exp_model, inc_fit, s_fit - np.median(s_fit), p0=p0, maxfev=1000)
            A, B = popt
            
            # Apply exponential correction
            correction = exp_model(inc[inner_middle_mask], A, B)
            s_corr[inner_middle_mask] = s[inner_middle_mask] - correction
            
            n_inner_middle = int(np.sum(inner_middle_mask))
            exp_params = (float(A), float(B))
        except:
            # Fallback to linear if exponential fit fails
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(inc_fit, s_fit)
            s_corr[inner_middle_mask] = s[inner_middle_mask] - slope * (inc[inner_middle_mask] - inc_ref)
            n_inner_middle = int(np.sum(inner_middle_mask))
            exp_params = None
    else:
        inc_ref = 0.0
        n_inner_middle = 0
        exp_params = None
    
    # =================================================================
    # REGION 3: Outer edge (inc > outer_min) - BIN-WISE correction
    # =================================================================
    outer_mask = np.isfinite(inc) & (inc > outer_min)
    outer_fit = fm & outer_mask
    
    if int(np.sum(outer_fit)) >= 100:
        inc_outer = inc[outer_fit]
        
        # Compute global reference (median of inner+middle swath)
        ref_mask = fm & (inc >= nadir_max) & (inc <= outer_min)
        if int(np.sum(ref_mask)) > 0:
            global_ref = float(np.median(s[ref_mask]))
        else:
            global_ref = float(np.median(s[fm]))
        
        # Bin-wise correction for outer edge
        bins = np.linspace(inc_outer.min(), inc_outer.max(), n_bins_edge + 1)
        for i in range(len(bins) - 1):
            bin_mask = outer_mask & (inc >= bins[i]) & (inc < bins[i+1])
            bin_fit = fm & bin_mask
            if int(np.sum(bin_fit)) >= 10:
                bin_median = float(np.median(s[bin_fit]))
                correction = bin_median - global_ref
                s_corr[bin_mask] = s[bin_mask] - correction
        
        n_outer = int(np.sum(outer_mask))
    else:
        n_outer = 0
    
    # Clip after correction
    s_corr = np.clip(s_corr, -50, 50)
    
    meta = {
        "used": True,
        "method": "hybrid_exponential_binwise",
        "exp_params": exp_params,
        "inc_ref": float(inc_ref) if inc_ref != 0 else None,
        "nadir_max": float(nadir_max),
        "outer_min": float(outer_min),
        "n_nadir_preserved": n_nadir,
        "n_inner_middle_exponential": n_inner_middle,
        "n_outer_binwise": n_outer,
    }
    
    logger.info(f"✓ Hybrid incidence correction:")
    logger.info(f"    Nadir (<{nadir_max}°): {n_nadir:,} pixels - COLUMN DESTRIPING")
    if exp_params is not None:
        A, B = exp_params
        logger.info(f"    Inner+Middle ({nadir_max}-{outer_min}°): {n_inner_middle:,} pixels - EXPONENTIAL (A={A:.3f}, B={B:.3f})")
    else:
        logger.info(f"    Inner+Middle ({nadir_max}-{outer_min}°): {n_inner_middle:,} pixels - EXPONENTIAL (fallback to linear)")
    logger.info(f"    Outer (>{outer_min}°): {n_outer:,} pixels - BIN-WISE mean removal")
    
    return s_corr.astype(np.float32), meta


def simple_linear_incidence_correction(
    sigma0: np.ndarray,
    incidence: np.ndarray,
    fit_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Simple linear incidence correction - just fit slope and subtract trend.
    DEPRECATED: Use region_aware_incidence_correction instead.
    """
    s = np.asarray(sigma0, dtype=np.float32)
    inc = np.asarray(incidence, dtype=np.float32)
    fm = np.asarray(fit_mask, dtype=bool) & np.isfinite(s) & np.isfinite(inc)
    
    # CRITICAL: Clip extreme outliers before fitting (typical σ0 range: -15 to +25 dB)
    s = np.clip(s, -50, 50)
    
    if int(np.sum(fm)) < 1000:
        return s.copy(), {"used": False, "reason": "insufficient_data"}
    
    inc_fit = inc[fm]
    s_fit = s[fm]
    
    # Robust fit using quantile regression (50th percentile = median trend)
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(inc_fit, s_fit)
    
    # Reference angle: median incidence
    inc_ref = float(np.median(inc_fit))
    
    # Correct: remove trend but preserve reference level
    s_corr = s - slope * (inc - inc_ref)
    
    # Clip again after correction to ensure no outliers
    s_corr = np.clip(s_corr, -50, 50)
    
    meta = {
        "used": True,
        "method": "linear",
        "slope": float(slope),
        "inc_ref": float(inc_ref),
    }
    
    logger.info(f"✓ Linear incidence correction: slope={slope:.4f} dB/deg @ {inc_ref:.1f}°")
    return s_corr.astype(np.float32), meta


def bin_wise_incidence_normalization(
    sigma0: np.ndarray,
    incidence: np.ndarray,
    fit_mask: np.ndarray,
    *,
    n_bins: int = 50,
    quantile: float = 0.5,
    min_per_bin: int = 200,
) -> Tuple[np.ndarray, Dict]:
    """
    Bin-wise incidence normalization (NO linear fitting).
    
    Instead of fitting a slope and subtracting trend, we:
    1. Bin pixels by incidence angle
    2. Compute median/quantile per bin
    3. Subtract the bin's median from each pixel in that bin
    
    This is more robust because:
    - No assumption of linearity
    - Each bin independently normalized
    - No extrapolation errors
    - More conservative
    """
    s = np.asarray(sigma0, dtype=np.float32)
    inc = np.asarray(incidence, dtype=np.float32)
    fm = np.asarray(fit_mask, dtype=bool) & np.isfinite(s) & np.isfinite(inc)

    if int(np.sum(fm)) < 2000:
        logger.warning("Too few points for incidence normalization")
        return s.copy(), {"used": False, "reason": "insufficient_data"}

    inc_v = inc[fm]
    s_v = s[fm]

    # Find incidence range
    i0 = float(np.percentile(inc_v, 1))
    i1 = float(np.percentile(inc_v, 99))
    if (not np.isfinite(i0)) or (not np.isfinite(i1)) or (i1 <= i0):
        return s.copy(), {"used": False, "reason": "bad_incidence_range"}

    # Create bins
    edges = np.linspace(i0, i1, int(n_bins) + 1)
    
    # Compute median per bin (for pixels in fit_mask)
    bin_medians = {}
    bin_counts = {}
    for k in range(len(edges) - 1):
        sel = fm & (inc >= edges[k]) & (inc < edges[k + 1])
        n_bin = int(np.sum(sel))
        if n_bin >= int(min_per_bin):
            bin_medians[k] = float(np.quantile(s[sel], quantile))
            bin_counts[k] = n_bin

    if len(bin_medians) < 6:
        logger.warning(f"Too few bins ({len(bin_medians)}) for incidence normalization")
        return s.copy(), {"used": False, "reason": "insufficient_bins"}

    # CRITICAL FIX: Don't correct nadir region (0-1.5°)
    # Nadir has strong specular reflection - correction makes it worse!
    nadir_threshold = 1.5  # degrees
    
    # Compute global reference (median of all bin medians, excluding nadir)
    non_nadir_medians = {k: v for k, v in bin_medians.items() 
                         if edges[k] >= nadir_threshold}
    
    if len(non_nadir_medians) < 3:
        # Not enough bins, skip correction
        logger.warning(f"Too few non-nadir bins for correction")
        return s.copy(), {"used": False, "reason": "insufficient_non_nadir_bins"}
    
    global_ref = float(np.median(list(non_nadir_medians.values())))
    
    # Apply correction: remove per-bin deviation from global reference
    # SKIP nadir region to preserve bright polynya signal!
    s_corr = s.copy()
    n_corrected = 0
    
    for k, bin_median in bin_medians.items():
        # Skip nadir bins
        if edges[k] < nadir_threshold:
            continue
            
        # All pixels (not just fit_mask) in this incidence bin
        in_bin = (inc >= edges[k]) & (inc < edges[k + 1]) & np.isfinite(s)
        # Subtract (bin_median - global_ref) instead of bin_median
        # This removes incidence trend but preserves absolute brightness
        s_corr[in_bin] = s[in_bin] - (bin_median - global_ref)
        n_corrected += int(np.sum(in_bin))
    
    # For pixels outside corrected bins (including nadir), leave as-is
    corrected_mask = np.zeros_like(s, dtype=bool)
    for k in bin_medians.keys():
        if edges[k] >= nadir_threshold:
            corrected_mask |= (inc >= edges[k]) & (inc < edges[k + 1]) & np.isfinite(s)

    meta = {
        "used": True,
        "method": "bin_wise_normalization",
        "n_bins": len(bin_medians),
        "n_pixels_corrected": n_corrected,
        "inc_range": (float(i0), float(i1)),
        "quantile": float(quantile),
    }
    
    n_nadir = int(np.sum(np.isfinite(s) & (inc < nadir_threshold)))
    logger.info(f"✓ Bin-wise incidence normalization: {len(bin_medians)} bins, {n_corrected:,} pixels corrected, {n_nadir:,} nadir pixels preserved")
    return s_corr.astype(np.float32), meta


# -----------------------------------------------------------------------------
# Main classifier - PROPER TWO-STAGE
# -----------------------------------------------------------------------------
class SWOTClassifier:
    def __init__(self):
        pass

    def classify_kmeans_two_stage(
        self,
        sigma0_db: np.ndarray,
        ssh: Optional[np.ndarray] = None,
        *,
        incidence: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,

        # Stage 1
        n_clusters_stage1: int = 7,
        include_texture: bool = True,
        texture_win: int = 21,
        apply_incidence_correction: bool = False,

        # Candidate selection
        n_candidate_classes: int = 2,  # Select top 2 brightest for SSH refinement
        candidate_min_pixels: int = 500,  # Lower threshold to include smaller features
        polynya_sigma0_threshold: float = 12.0,  # If corrected σ0 > this, likely polynya (increased!)

        # Stage 2 (SSH refinement)
        do_stage2: bool = True,
        stage2_k: int = 2,
        ssh_localstd_win: int = 15,
        min_stage2_pixels: int = 1200,
        min_polynya_component: int = 600,
        min_polynya_score_ratio: float = 1.3,

        # Post rules
        enforce_polynya_requires_ssh: bool = True,
        apply_majority_filter: bool = True,
        random_state: int = 42,

        # Feature weights
        sigma0_weight: float = 3.0,
        texture_weight: float = 1.0,
        ssh_weight: float = 2.0,
        ssh_grad_weight: float = 1.5,

        # Artifact handling (IMPROVED)
        buffer_cols_inner: int = 1,  # Minimal inner masking
        buffer_cols_outer: int = 1,  # VERY aggressive outer masking - maximum edge removal
        rail_z: float = 3.5,
        exclude_inner_from_fit: bool = True,
        exclude_outer_from_fit: bool = True,  # NEW: exclude outer edges too
        force_artifacts_to_pack: bool = False,  # DISABLED - was overwriting good classification
        force_outer_to_pack: bool = False,  # DISABLED - use unsure instead
        mark_artifacts_as_unsure: bool = True,  # NEW: Mark artifacts as "unsure" class
    ) -> SWOTClassification:
        """
        PROPER TWO-STAGE CLASSIFICATION
        
        Stage 1: KMeans on sigma0+texture ONLY (no SSH)
                 - Fit excludes edges/artifacts
                 - Predict on all valid pixels
        
        Stage 2: Within candidates, use SSH to separate polynya from leads
                 - SSH used ONLY where it helps
                 - Not used as clustering feature for everyone
        """
        _require_sklearn()
        if include_texture:
            _require_scipy("texture")
        _require_scipy("stage2 morphology")

        sigma0 = np.asarray(sigma0_db, dtype=np.float32)
        H, W = sigma0.shape

        valid = np.isfinite(sigma0) & (sigma0 > -80.0)
        valid = _combine_valid_and_exclude(valid, exclude_mask)

        if int(np.sum(valid)) < 2000:
            raise RuntimeError("Too few valid pixels")

        logger.info(f"\n{'='*70}")
        logger.info("PROPER TWO-STAGE CLASSIFICATION")
        logger.info(f"{'='*70}")
        logger.info(f"Valid pixels: {int(np.sum(valid)):,} / {H*W:,}")

        # =================================================================
        # Build artifact mask (IMPROVED: inner + outer + rails)
        # =================================================================
        logger.info("\n[Step 1/5] Building artifact mask")
        
        inner_mask = make_inner_edge_mask(valid, buffer_cols=buffer_cols_inner)
        outer_mask = make_outer_edge_mask(valid, buffer_cols=buffer_cols_outer)
        rail_cols = detect_rail_cols(sigma0, valid, z=rail_z)
        
        artifact_mask = inner_mask | outer_mask
        if np.any(rail_cols):
            artifact_mask = expand_mask_with_cols(artifact_mask, rail_cols, buffer=buffer_cols_inner)
        
        n_artifacts = int(np.sum(artifact_mask))
        logger.info(f"  Inner edges: {int(np.sum(inner_mask)):,} pixels")
        logger.info(f"  Outer edges: {int(np.sum(outer_mask)):,} pixels")
        logger.info(f"  Total artifacts: {n_artifacts:,} pixels ({100*n_artifacts/valid.size:.1f}%)")

        # =================================================================
        # Incidence correction (conservative)
        # =================================================================
        logger.info("\n[Step 2/5] Incidence correction")
        sigma0_corr = sigma0.copy()
        inc_meta = {"used": False}
        
        # Log original sigma0 range
        valid_sigma0 = sigma0[valid & np.isfinite(sigma0)]
        if valid_sigma0.size > 0:
            logger.info(f"  Original σ0 range: [{np.min(valid_sigma0):.2f}, {np.max(valid_sigma0):.2f}] dB")

        if apply_incidence_correction and (incidence is not None):
            inc_arr = np.asarray(incidence, dtype=np.float32)
            if inc_arr.shape == (H, W):
                # Fit mask excludes artifacts
                fit_mask_inc = valid & (~artifact_mask) & np.isfinite(sigma0) & np.isfinite(inc_arr)
                n_fit = int(np.sum(fit_mask_inc))
                
                if n_fit >= 2000:
                    sigma0_corr, inc_meta = region_aware_incidence_correction(
                        sigma0, inc_arr, fit_mask_inc,
                        nadir_max=1.8, outer_min=3.2, n_bins_edge=30
                    )
                    logger.info(f"  Used {n_fit:,} pixels for hybrid correction")
                    # Log corrected range
                    valid_corr = sigma0_corr[valid & np.isfinite(sigma0_corr)]
                    if valid_corr.size > 0:
                        logger.info(f"  Corrected σ0 range: [{np.min(valid_corr):.2f}, {np.max(valid_corr):.2f}] dB")
                else:
                    logger.warning(f"  Too few pixels ({n_fit}) - skipped")

        # =================================================================
        # Texture
        # =================================================================
        logger.info("\n[Step 3/5] Computing texture")
        tex_f = None
        if include_texture:
            tex_f = local_std_nan(sigma0_corr, size=texture_win)
            tv = tex_f[valid & np.isfinite(tex_f)]
            tmed = float(np.median(tv)) if tv.size > 0 else 0.0
            tex_f = np.where(np.isfinite(tex_f), tex_f, tmed).astype(np.float32)
            logger.info(f"  ✓ Texture computed (window={texture_win})")

        # =================================================================
        # STAGE 1: KMeans on sigma0+texture ONLY
        # =================================================================
        logger.info(f"\n[Step 4/5] Stage 1: KMeans (sigma0+texture, K={n_clusters_stage1})")
        
        # Fit mask: exclude artifacts
        fit_mask1 = valid & np.isfinite(sigma0_corr)
        if exclude_inner_from_fit:
            fit_mask1 &= ~inner_mask
        if exclude_outer_from_fit:
            fit_mask1 &= ~outer_mask

        if int(np.sum(fit_mask1)) < 1000:
            raise RuntimeError("Too few pixels for Stage-1 fit")

        # Build features (NO SSH!)
        X_list = [sigma0_corr[fit_mask1].reshape(-1, 1)]
        w = [sigma0_weight]
        if tex_f is not None:
            X_list.append(tex_f[fit_mask1].reshape(-1, 1))
            w.append(texture_weight)

        X1 = np.column_stack(X_list)
        scaler1 = RobustScaler().fit(X1)
        X1w = scaler1.transform(X1) * np.array(w).reshape(1, -1)

        km1 = MiniBatchKMeans(n_clusters=n_clusters_stage1, n_init=10, max_iter=300, random_state=random_state, batch_size=10000)
        km1.fit(X1w)

        logger.info(f"  Fit on {X1.shape[0]:,} pixels (excluded {int(np.sum(artifact_mask)):,} artifacts)")

        # Predict on ALL valid pixels
        pred_mask1 = valid & np.isfinite(sigma0_corr)
        Xp_list = [sigma0_corr[pred_mask1].reshape(-1, 1)]
        if tex_f is not None:
            Xp_list.append(tex_f[pred_mask1].reshape(-1, 1))
        Xp = np.column_stack(Xp_list)
        lab1_pred = km1.predict(scaler1.transform(Xp) * np.array(w).reshape(1, -1))

        labels_stage1 = np.full((H, W), -1, dtype=np.int32)
        idx_all = np.where(pred_mask1.ravel())[0]
        labels_stage1.ravel()[idx_all] = lab1_pred

        logger.info(f"  Predicted on {int(np.sum(pred_mask1)):,} pixels")

        # =================================================================
        # Reorder clusters by brightness (bright→dark)
        # =================================================================
        core_for_stats = pred_mask1 & (~artifact_mask)
        K1 = n_clusters_stage1

        # Compute statistics for each cluster
        med_sig_raw = np.full(K1, np.nan)
        med_tex_raw = np.full(K1, np.nan)
        count_k_raw = np.zeros(K1, dtype=int)

        for k in range(K1):
            mk = (labels_stage1 == k) & core_for_stats
            n = int(np.sum(mk))
            count_k_raw[k] = n
            if n >= 200:
                med_sig_raw[k] = float(np.median(sigma0_corr[mk]))
                if tex_f is not None:
                    med_tex_raw[k] = float(np.median(tex_f[mk]))

        # DON'T reorder - instead, find brightest clusters directly
        med_sig = med_sig_raw
        med_tex = med_tex_raw
        count_k = count_k_raw
        
        # Log cluster statistics (no reordering)
        logger.info(f"  Cluster statistics (original KMeans labels):")
        total_valid = int(np.sum(pred_mask1))
        for k in range(K1):
            if np.isfinite(med_sig[k]) and count_k[k] > 0:
                pct = 100.0 * count_k[k] / total_valid if total_valid > 0 else 0
                tex_str = f", tex={med_tex[k]:.2f}" if np.isfinite(med_tex[k]) else ""
                logger.info(f"    Class {k}: σ0={med_sig[k]:.2f} dB, n={count_k[k]:,} ({pct:.1f}%){tex_str}")
        
        # Check if one class dominates
        max_class_pct = 100.0 * np.max(count_k) / total_valid if total_valid > 0 else 0
        if max_class_pct > 70:
            logger.warning(f"  ⚠ WARNING: One class has {max_class_pct:.1f}% of pixels! KMeans clustering may be poor.")
        
        # =================================================================
        # Candidate selection for Stage 2
        # =================================================================

        # Select candidates: pick brightest clusters (highest med_sig)
        # Score by brightness (higher = brighter = more likely polynya)
        valid_clusters = (np.isfinite(med_sig)) & (count_k >= candidate_min_pixels)
        
        if np.sum(valid_clusters) == 0:
            # Fallback: pick any cluster with pixels
            valid_clusters = (count_k > 0)
        
        # Sort by brightness (descending)
        brightness_order = np.argsort(-np.where(valid_clusters, med_sig, -999.0))
        
        # Pick top N brightest
        cand = []
        for k in brightness_order:
            if valid_clusters[k]:
                cand.append(int(k))
                if len(cand) >= n_candidate_classes:
                    break
        
        if len(cand) == 0:
            cand = [int(np.argmax(count_k))]

        candidate_mask = np.isin(labels_stage1, cand) & core_for_stats
        
        logger.info(f"  Selected {len(cand)} brightest clusters as candidates")

        logger.info(f"\n  Candidate classes for Stage 2: {cand}")
        for k in cand:
            if np.isfinite(med_sig[k]):
                tex_str = f", tex={med_tex[k]:.2f}" if np.isfinite(med_tex[k]) else ""
                logger.info(f"    Class {k}: {count_k[k]:,} pixels, σ0={med_sig[k]:.2f} dB{tex_str}")

        if int(np.sum(candidate_mask)) < candidate_min_pixels:
            logger.warning("Too few candidates - returning Stage 1 only")
            class_names = _rank_names_for_k(K1)
            class_map = {n: i for i, n in enumerate(class_names)}
            out = labels_stage1.copy()
            
            if apply_majority_filter:
                out = majority_filter_3x3(out)
            
            # Handle artifacts (same logic as main path)
            if mark_artifacts_as_unsure:
                unsure_id = len(class_names)
                class_names.append("unsure")
                class_map["unsure"] = unsure_id
                n_unsure = int(np.sum(artifact_mask & (out >= 0)))
                out[artifact_mask & (out >= 0)] = unsure_id
                logger.info(f"  ✓ Marked {n_unsure:,} artifact pixels as 'unsure'")
            elif force_artifacts_to_pack:
                pack_id = len(class_names) - 1
                out[artifact_mask & (out >= 0)] = pack_id
            
            return SWOTClassification(
                sigma0=sigma0,
                ssh=(np.asarray(ssh, dtype=np.float32) if ssh is not None else None),
                lat=None,
                lon=None,
                sigma0_corrected=sigma0_corr,
                class_labels=out,
                class_names=class_names,
                class_map=class_map,
                confidence=np.zeros((H, W, len(class_names)), dtype=np.float32),
                method="kmeans_two_stage",
                n_clusters=K1,
                meta={"incidence_correction": inc_meta, "stage2_used": False},
                exclude_mask=(np.asarray(exclude_mask, dtype=bool) if exclude_mask is not None else None),
            )

        # =================================================================
        # STAGE 2: SSH refinement within candidates
        # =================================================================
        logger.info(f"\n[Step 5/5] Stage 2: SSH refinement")
        
        stage2_used = False
        polynya_mask = np.zeros((H, W), dtype=bool)
        lead_mask = np.zeros((H, W), dtype=bool)

        if do_stage2 and (ssh is not None):
            ssh_arr = np.asarray(ssh, dtype=np.float32)
            if ssh_arr.shape == (H, W):
                # Stage2 region: candidates WITH valid SSH
                stage2_region = candidate_mask & np.isfinite(ssh_arr)
                n2 = int(np.sum(stage2_region))
                
                logger.info(f"  Stage 2 region: {n2:,} pixels (min={min_stage2_pixels})")
                
                if n2 >= min_stage2_pixels:
                    stage2_used = True

                    # Compute SSH local std
                    ssh_std = local_std_nan(ssh_arr, size=ssh_localstd_win)
                    sv = ssh_std[stage2_region & np.isfinite(ssh_std)]
                    smed = float(np.median(sv)) if sv.size > 0 else 0.0
                    ssh_std = np.where(np.isfinite(ssh_std), ssh_std, smed).astype(np.float32)

                    # Features: sigma0 + SSH + SSH_std
                    X2 = np.column_stack([
                        sigma0_corr[stage2_region],
                        ssh_arr[stage2_region],
                        ssh_std[stage2_region],
                    ])

                    scaler2 = RobustScaler().fit(X2)
                    w2 = np.array(
                        [float(sigma0_weight), float(ssh_weight), float(ssh_grad_weight)],
                        dtype=np.float32,
                    ).reshape(1, -1)
                    km2 = KMeans(n_clusters=stage2_k, n_init=20, max_iter=400, random_state=random_state)
                    lab2 = km2.fit_predict(scaler2.transform(X2) * w2)

                    # Assign to 2D
                    tmp0 = np.zeros((H, W), dtype=bool)
                    tmp1 = np.zeros((H, W), dtype=bool)
                    idx2 = np.where(stage2_region.ravel())[0]
                    tmp0.ravel()[idx2[lab2 == 0]] = True
                    tmp1.ravel()[idx2[lab2 == 1]] = True

                    # Identify which is polynya using SSH (simpler, more direct)
                    m0 = float(np.nanmean(ssh_arr[tmp0])) if np.any(tmp0) else np.nan
                    m1 = float(np.nanmean(ssh_arr[tmp1])) if np.any(tmp1) else np.nan
                    
                    sig0 = float(np.nanmean(sigma0_corr[tmp0])) if np.any(tmp0) else np.nan
                    sig1 = float(np.nanmean(sigma0_corr[tmp1])) if np.any(tmp1) else np.nan

                    logger.info(f"    Class 0: SSH={m0:.3f}m, σ0={sig0:.1f}dB, n={np.sum(tmp0):,}")
                    logger.info(f"    Class 1: SSH={m1:.1f}m, σ0={sig1:.1f}dB, n={np.sum(tmp1):,}")

                    # Polynya = higher SSH + brighter
                    if np.isfinite(m0) and np.isfinite(m1):
                        poly_is_0 = (m0 > m1)  # Higher SSH = polynya
                    elif np.isfinite(sig0) and np.isfinite(sig1):
                        poly_is_0 = (sig0 > sig1)  # Fallback: brighter = polynya
                    else:
                        poly_is_0 = (np.sum(tmp0) >= np.sum(tmp1))  # Fallback: larger = polynya

                    if poly_is_0 is not None:
                        polynya_mask = tmp0 if poly_is_0 else tmp1
                        lead_mask = tmp1 if poly_is_0 else tmp0
                        if enforce_polynya_requires_ssh:
                            polynya_mask &= np.isfinite(ssh_arr)
                        logger.info(f"  ✓ Polynya: {int(np.sum(polynya_mask)):,} pixels")
                        logger.info(f"  ✓ Lead: {int(np.sum(lead_mask)):,} pixels")
                    else:
                        logger.warning("  Ambiguous - no polynya assigned")
                        lead_mask = stage2_region.copy()
                else:
                    logger.info(f"  Not enough SSH data - skipped Stage 2")

        # =================================================================
        # Merge results - Assign names based on ACTUAL brightness
        # =================================================================
        labels_final = labels_stage1.copy()
        
        # Create class names array (will be updated)
        class_names = [f"class_{k}" for k in range(K1)]
        
        # Sort all clusters by brightness to assign proper names
        brightness_order = np.argsort(-np.where(np.isfinite(med_sig), med_sig, -999.0))
        std_names = _rank_names_for_k(K1)  # [open_water, thin_ice, ..., pack_ice]
        
        # Assign names: brightest cluster gets first name, etc.
        for rank, cluster_id in enumerate(brightness_order):
            if np.isfinite(med_sig[cluster_id]):
                class_names[cluster_id] = std_names[rank]
        
        class_map = {n: i for i, n in enumerate(class_names)}
        
        # Identify special classes
        poly_id = cand[0] if len(cand) > 0 else 0  # Brightest candidate
        thin_id = cand[1] if len(cand) > 1 else poly_id  # Second brightest
        
        # Pack ice = darkest cluster with enough pixels
        pack_id = brightness_order[-1] if len(brightness_order) > 0 else K1-1
        
        logger.info(f"\n  Class assignments (by brightness):")
        logger.info(f"    poly_id = {poly_id} (σ0={med_sig[poly_id]:.2f} dB, will be renamed to 'polynya')")
        if poly_id != thin_id:
            logger.info(f"    thin_id = {thin_id} (σ0={med_sig[thin_id]:.2f} dB, '{class_names[thin_id]}')")
        logger.info(f"    pack_id = {pack_id} (σ0={med_sig[pack_id]:.2f} dB, '{class_names[pack_id]}')")
        
        # Log initial distribution BEFORE merging
        logger.info(f"\n  Initial Stage 1 distribution (before merging):")
        for k in range(K1):
            n_k = int(np.sum(labels_final == k))
            if n_k > 0:
                logger.info(f"    Class {k} ({class_names[k]}): {n_k:,} pixels ({100*n_k/np.sum(labels_final>=0):.1f}%)")

        # Rename and merge candidate classes (CRITICAL FIX: merge ALL pixels, not just core)
        if stage2_used:
            # Assign Stage 2 results
            labels_final[polynya_mask] = poly_id
            labels_final[lead_mask] = thin_id
            class_names[poly_id] = "polynya"
            
            # DISABLED: Don't merge other bright classes - preserve first_year_ice, thin_ice, etc.
            # for cid in cand:
            #     if cid != poly_id and cid != thin_id:
            #         cand_pixels = (labels_final == cid)
            #         if np.any(cand_pixels):
            #             labels_final[cand_pixels] = poly_id
            #             logger.info(f"  ✓ Merged class {cid} ({class_names[cid]}, σ0={med_sig[cid]:.2f}) into polynya: {int(np.sum(cand_pixels)):,} pixels")
            logger.info(f"  ✓ Preserved all {len(cand)} candidate classes (no merging)")
        else:
            # Stage 2 not used - just rename brightest as polynya, keep others
            class_names[poly_id] = "polynya"
            logger.info(f"  ✓ Preserved all {len(cand)} candidate classes (no merging)")
            # DISABLED merging:
            # for cid in cand:
            #     if cid != poly_id:
            #         cand_pixels = (labels_final == cid)
            #         if np.any(cand_pixels):
            #             labels_final[cand_pixels] = poly_id
            #             logger.info(f"  ✓ Merged class {cid} ({class_names[cid]}, σ0={med_sig[cid]:.2f}) into polynya: {int(np.sum(cand_pixels)):,} pixels")
        
        # DISABLED: Don't merge young_ice or first_year_ice into polynya
        # User wants to keep these as separate classes
        # class_map_temp = {n: i for i, n in enumerate(class_names)}
        # if "young_ice" in class_map_temp:
        #     young_id = class_map_temp["young_ice"]
        #     if young_id != poly_id:
        #         young_pixels = (labels_final == young_id)
        #         if np.any(young_pixels):
        #             labels_final[young_pixels] = poly_id
        #             logger.info(f"  ✓ Merged young_ice into polynya: {int(np.sum(young_pixels)):,} pixels")
        
        # CRITICAL FIX: Remove duplicate class names and rebuild class list
        # After merging, some classes have 0 pixels and shouldn't appear in legend
        unique_labels = np.unique(labels_final[labels_final >= 0])
        new_class_names = []
        new_label_map = {}  # old_id -> new_id
        
        for new_id, old_id in enumerate(sorted(unique_labels)):
            old_id = int(old_id)
            new_class_names.append(class_names[old_id])
            new_label_map[old_id] = new_id
        
        # Remap labels to sequential 0, 1, 2, ...
        labels_remapped = np.full_like(labels_final, -1)
        for old_id, new_id in new_label_map.items():
            labels_remapped[labels_final == old_id] = new_id
        
        labels_final = labels_remapped
        class_names = new_class_names
        class_map = {n: i for i, n in enumerate(class_names)}
        
        # Update special IDs after remapping
        if poly_id in new_label_map:
            poly_id = new_label_map[poly_id]
        if pack_id in new_label_map:
            pack_id = new_label_map[pack_id]
        
        logger.info(f"\n  After merging: {len(class_names)} unique classes remain")

        # =================================================================
        # BRIGHTNESS-BASED POLYNYA OVERRIDE
        # Force very bright pixels to polynya (likely misclassified by KMeans)
        # =================================================================
        if "polynya" in class_map:
            poly_id = class_map["polynya"]
            
            # Find pixels that are VERY bright (corrected σ0 > threshold)
            very_bright_mask = (sigma0_corr > polynya_sigma0_threshold) & (labels_final >= 0) & (labels_final != poly_id)
            n_bright_override = int(np.sum(very_bright_mask))
            
            if n_bright_override > 0:
                labels_final[very_bright_mask] = poly_id
                logger.info(f"\n  ✓ Brightness override: {n_bright_override:,} very bright pixels (σ0>{polynya_sigma0_threshold} dB) → polynya")

        # Handle artifacts
        if mark_artifacts_as_unsure:
            # Add "unsure" as a new class
            unsure_id = len(class_names)
            class_names.append("unsure")
            class_map["unsure"] = unsure_id
            
            # Mark artifact regions as "unsure"
            n_unsure = int(np.sum(artifact_mask & (labels_final >= 0)))
            labels_final[artifact_mask & (labels_final >= 0)] = unsure_id
            logger.info(f"  ✓ Marked {n_unsure:,} artifact pixels as 'unsure'")
        elif force_artifacts_to_pack:
            n_artifact_forced = int(np.sum(artifact_mask & (labels_final >= 0)))
            labels_final[artifact_mask & (labels_final >= 0)] = pack_id
            logger.info(f"  ⚠ Forced {n_artifact_forced:,} artifact pixels to pack_ice")
        elif force_outer_to_pack:
            # Force ONLY outer edges to pack_ice (not inner edges or rails)
            n_outer_forced = int(np.sum(outer_mask & (labels_final >= 0)))
            labels_final[outer_mask & (labels_final >= 0)] = pack_id
            logger.info(f"  ⚠ Forced {n_outer_forced:,} outer edge pixels to pack_ice")
        else:
            logger.info(f"  ✓ Artifact handling disabled (preserving classification)")

        # Majority filter
        if apply_majority_filter:
            labels_final = majority_filter_3x3(labels_final)
            # Re-apply artifact handling after filtering
            if mark_artifacts_as_unsure:
                labels_final[artifact_mask & (labels_final >= 0)] = unsure_id
            elif force_artifacts_to_pack:
                labels_final[artifact_mask & (labels_final >= 0)] = pack_id
            elif force_outer_to_pack:
                labels_final[outer_mask & (labels_final >= 0)] = pack_id

        # =================================================================
        # Finalize
        # =================================================================
        logger.info("\n" + "="*70)
        logger.info("CLASSIFICATION COMPLETE")
        logger.info("="*70)
        
        for name in class_names:
            cid = class_map[name]
            mask = labels_final == cid
            n = int(np.sum(mask))
            frac = n / int(np.sum(labels_final >= 0)) if np.sum(labels_final >= 0) > 0 else 0
            mean_sig = np.nanmean(sigma0_corr[mask]) if n > 0 else np.nan
            mean_sig_raw = np.nanmean(sigma0[mask]) if n > 0 else np.nan
            logger.info(f"{name:>18s} (class {cid}): {n:>7,} ({100*frac:>5.1f}%)  σ0_corr={mean_sig:>6.2f} dB  σ0_raw={mean_sig_raw:>6.2f} dB")

        meta = {
            "incidence_correction": inc_meta,
            "stage2_used": stage2_used,
            "candidate_classes": cand,
            "n_candidate_pixels": int(np.sum(candidate_mask)),
            "n_artifacts_masked": n_artifacts,
        }

        return SWOTClassification(
            sigma0=sigma0,
            ssh=(np.asarray(ssh, dtype=np.float32) if ssh is not None else None),
            lat=None,
            lon=None,
            sigma0_corrected=sigma0_corr,
            class_labels=labels_final,
            class_names=class_names,
            class_map=class_map,
            confidence=np.zeros((H, W, len(class_names)), dtype=np.float32),
            method="kmeans_two_stage",
            n_clusters=len(class_names),
            meta=meta,
            exclude_mask=(np.asarray(exclude_mask, dtype=bool) if exclude_mask is not None else None),
        )

    def classify_kmeans_simple(
        self,
        sigma0_db: np.ndarray,
        ssh: Optional[np.ndarray] = None,
        *,
        incidence: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
        n_clusters: int = 5,
        include_texture: bool = True,
        texture_win: int = 21,
        apply_incidence_correction: bool = True,
        random_state: int = 42,
        buffer_cols_inner: int = 1,
        buffer_cols_outer: int = 1,
        sigma0_weight: float = 3.0,
        texture_weight: float = 1.0,
    ) -> SWOTClassification:
        """
        SIMPLE KMEANS - Just cluster on sigma0 + texture, no complicated merging.
        This is what's working in your reference plot!
        """
        _require_sklearn()
        
        sigma0 = np.asarray(sigma0_db, dtype=np.float32)
        H, W = sigma0.shape
        valid = np.isfinite(sigma0) & (sigma0 > -80.0)
        valid = _combine_valid_and_exclude(valid, exclude_mask)
        
        if int(np.sum(valid)) < 1000:
            raise RuntimeError("Too few valid pixels")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SIMPLE KMEANS CLASSIFICATION (n_clusters={n_clusters})")
        logger.info(f"{'='*70}")
        logger.info(f"Valid pixels: {int(np.sum(valid)):,}")
        
        # Minimal artifact masking
        inner_mask = make_inner_edge_mask(valid, buffer_cols=buffer_cols_inner)
        outer_mask = make_outer_edge_mask(valid, buffer_cols=buffer_cols_outer)
        artifact_mask = inner_mask | outer_mask
        
        # Log original sigma0 range
        valid_orig = sigma0[valid & np.isfinite(sigma0)]
        if valid_orig.size > 0:
            logger.info(f"  Original σ0: [{np.min(valid_orig):.2f}, {np.max(valid_orig):.2f}] dB, range={np.max(valid_orig)-np.min(valid_orig):.2f} dB")
        
        # Incidence correction (skip nadir)
        sigma0_corr = sigma0.copy()
        if apply_incidence_correction and (incidence is not None):
            inc_arr = np.asarray(incidence, dtype=np.float32)
            if inc_arr.shape == (H, W):
                fit_mask = valid & (~artifact_mask) & np.isfinite(sigma0) & np.isfinite(inc_arr)
                if int(np.sum(fit_mask)) >= 2000:
                    sigma0_corr, inc_meta = region_aware_incidence_correction(
                        sigma0, inc_arr, fit_mask,
                        nadir_max=1.8, outer_min=3.2, n_bins_edge=30
                    )
                    # Log corrected range
                    valid_corr = sigma0_corr[valid & np.isfinite(sigma0_corr)]
                    if valid_corr.size > 0:
                        corr_range = np.max(valid_corr) - np.min(valid_corr)
                        logger.info(f"  Corrected σ0: [{np.min(valid_corr):.2f}, {np.max(valid_corr):.2f}] dB, range={corr_range:.2f} dB")
                else:
                    logger.warning(f"  Skipped incidence correction (too few pixels)")
        else:
            logger.info(f"  Incidence correction: DISABLED")
        
        # Texture
        tex_f = None
        if include_texture:
            tex_f = local_std_nan(sigma0_corr, size=texture_win)
            tv = tex_f[valid & np.isfinite(tex_f)]
            tmed = float(np.median(tv)) if tv.size > 0 else 0.0
            tex_f = np.where(np.isfinite(tex_f), tex_f, tmed).astype(np.float32)
        
        # Build features
        fit_mask = valid & np.isfinite(sigma0_corr) & (~artifact_mask)
        X_list = [sigma0_corr[fit_mask].reshape(-1, 1)]
        if tex_f is not None:
            X_list.append(tex_f[fit_mask].reshape(-1, 1))
        
        X = np.column_stack(X_list)
        scaler = RobustScaler().fit(X)
        w = [float(sigma0_weight)]
        if tex_f is not None:
            w.append(float(texture_weight))
        w = np.asarray(w, dtype=np.float32).reshape(1, -1)
        X_scaled = scaler.transform(X) * w
        
        # MiniBatchKMeans - more memory efficient for large datasets
        km = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=random_state, batch_size=10000)
        km.fit(X_scaled)
        
        # Predict on all valid
        pred_mask = valid & np.isfinite(sigma0_corr)
        Xp_list = [sigma0_corr[pred_mask].reshape(-1, 1)]
        if tex_f is not None:
            Xp_list.append(tex_f[pred_mask].reshape(-1, 1))
        Xp = np.column_stack(Xp_list)
        labels_pred = km.predict(scaler.transform(Xp) * w)
        
        labels_final = np.full((H, W), -1, dtype=np.int32)
        labels_final[pred_mask] = labels_pred
        
        # Compute cluster stats and sort by brightness
        med_sig = np.full(n_clusters, np.nan)
        count_k = np.zeros(n_clusters, dtype=int)
        total_pixels = int(np.sum(valid))
        
        logger.info(f"\n  Initial KMeans clustering (BEFORE reordering):")
        for k in range(n_clusters):
            mk = (labels_final == k) & valid
            n = int(np.sum(mk))
            count_k[k] = n
            if n >= 100:
                med_sig[k] = float(np.median(sigma0_corr[mk]))
            
            pct = 100.0 * n / total_pixels if total_pixels > 0 else 0
            logger.info(f"    Cluster {k}: n={n:,} ({pct:.1f}%), σ0={med_sig[k]:.2f} dB")
        
        # Check if one cluster dominates
        max_pct = 100.0 * np.max(count_k) / total_pixels if total_pixels > 0 else 0
        if max_pct > 70:
            logger.warning(f"  ⚠ WARNING: One cluster has {max_pct:.1f}% of pixels!")
            logger.warning(f"  ⚠ KMeans failed to separate ice types properly!")
            logger.warning(f"  ⚠ Try: --apply_incidence_correction False or increase --clusters")
        
        # Sort by brightness and assign names
        brightness_order = np.argsort(-np.where(np.isfinite(med_sig), med_sig, -999.0))
        class_names_ordered = _rank_names_for_k(n_clusters)
        
        logger.info(f"\n  Brightness ordering (bright→dark):")
        for new_id, old_id in enumerate(brightness_order):
            logger.info(f"    {new_id}. {class_names_ordered[new_id]:>18s} ← Cluster {old_id} (σ0={med_sig[old_id]:.2f} dB)")
        
        # Remap to brightness order
        labels_remapped = np.full_like(labels_final, -1)
        class_names = [None] * n_clusters
        for new_id, old_id in enumerate(brightness_order):
            labels_remapped[labels_final == old_id] = new_id
            class_names[new_id] = class_names_ordered[new_id]
        
        labels_final = labels_remapped
        class_map = {n: i for i, n in enumerate(class_names)}
        
        # Log results
        logger.info(f"\n{'='*70}")
        logger.info("CLASSIFICATION COMPLETE")
        logger.info(f"{'='*70}")
        for i, name in enumerate(class_names):
            mask = labels_final == i
            n = int(np.sum(mask))
            frac = n / int(np.sum(labels_final >= 0)) if np.sum(labels_final >= 0) > 0 else 0
            mean_sig = np.nanmean(sigma0_corr[mask]) if n > 0 else np.nan
            logger.info(f"{name:>18s}: {n:>7,} ({100*frac:>5.1f}%)  σ0={mean_sig:>6.2f} dB")
        
        return SWOTClassification(
            sigma0=sigma0,
            ssh=(np.asarray(ssh, dtype=np.float32) if ssh is not None else None),
            lat=None,
            lon=None,
            sigma0_corrected=sigma0_corr,
            class_labels=labels_final,
            class_names=class_names,
            class_map=class_map,
            confidence=np.zeros((H, W, n_clusters), dtype=np.float32),
            method="kmeans_simple",
            n_clusters=n_clusters,
            meta={},
            exclude_mask=(np.asarray(exclude_mask, dtype=bool) if exclude_mask is not None else None),
        )
    
    # Backward compatibility
    def classify_enhanced(self, *args, **kwargs):
        return self.classify_kmeans_two_stage(*args, **kwargs)  # Use two-stage with linear correction!
    
    def classify_kmeans(self, *args, **kwargs):
        return self.classify_kmeans_two_stage(*args, **kwargs)
    
    def classify_kmeans_baseline(self, sigma0_db, exclude_mask=None, n_clusters=5, 
                                  include_texture=True, texture_win=21, random_state=42, **kwargs):
        """Baseline KMeans - no incidence correction, no SSH, just sigma0 + texture."""
        return self.classify_kmeans_simple(
            sigma0_db, 
            ssh=None,
            incidence=None,
            exclude_mask=exclude_mask,
            n_clusters=n_clusters,
            include_texture=include_texture,
            texture_win=texture_win,
            apply_incidence_correction=True,  # Force OFF
            buffer_cols_inner=5,  # Minimal
            buffer_cols_outer=5,  # Minimal
            random_state=random_state,
        )
