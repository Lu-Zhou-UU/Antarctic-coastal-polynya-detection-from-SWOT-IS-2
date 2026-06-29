#!/usr/bin/env python3
"""
IS2 thin-ice thickness in SWOT/AMSR polynya using JOINT (6-beam) sea-surface reference.

Key choices:
- Polynya along IS2 is determined by **AMSR SIC (<thresh)** and **SWOT class mask (KDTree)**.
- Decision rule along each IS2 beam:
    if (#SWOT polynya/open-water points) < (#AMSR SIC<thresh points)  -> use AMSR mask
    else                                                             -> use SWOT mask
  (If one source is unavailable, the other is used automatically.)
- "Lead-like" sea-surface candidates are detected per beam:
    - SWOT-feature mode (sigma0/ssh stats) when using SWOT polynya
    - IS2-only (height-only) mode when falling back (no polynya points) or you choose to disable SWOT features
- Sea-surface reference is computed JOINTLY across ALL beams, per LATITUDE BIN
  of size --segment_km (km) converted to degrees using 111.32 km/deg.
- Hybrid reference:
    if #candidates >= --min_cand_per_seg -> ref = quantile(candidates, --ref_quantile)
    else                                -> ref = quantile(all in-scope heights, --ref_quantile)

Outputs are named with a date tag extracted from input filenames (e.g. 20231015T232713).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date as _date, datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- optional deps ---
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    import mat73  # for v7.3 .mat
except Exception:
    mat73 = None

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from scipy.signal import find_peaks, peak_widths
    HAS_SCIPY_SIGNAL = True
except Exception:
    HAS_SCIPY_SIGNAL = False

try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = True
except Exception:
    HAS_KDTREE = False

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# =============================================================================
# Helpers: time tags, MATLAB-like AMSR SIC loading
# =============================================================================

def _matlab_day_index2(ymd: str) -> int:
    """
    MATLAB: day_index2 = days(datetime(YYYY,MM,DD) - datetime(2022,12,31))
    For 2023-01-01 -> 1, so Python 0-based index is (that - 1).
    """
    d = _dt.strptime(ymd, "%Y-%m-%d").date()
    base = _date(2022, 12, 31)
    return (d - base).days  # MATLAB 1-based day count for 2023


def _extract_yyyymmdd_from_tag(tag: str) -> Optional[str]:
    """
    tag could be '20231021T214726' or '20231021205450' etc.
    return '2023-10-21'
    """
    m = re.search(r"(?<!\d)(\d{8})(?:T?\d{6})", str(tag))
    if not m:
        return None
    ymd = m.group(1)
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"


def extract_swot_tag(path: str) -> str:
    """Extract SWOT tag as YYYYMMDDTHHMMSS (with 'T') from a path."""
    pat = re.compile(r"(\d{8}T\d{6})")
    m = pat.search(str(path))
    if m:
        return m.group(1)
    return Path(path).stem


def extract_is2_tag(path: str) -> str:
    """
    Extract IS2 tag as YYYYMMDDHHMMSS (14 digits) from a path.
    Prefers a 14-digit block if present (e.g. 20231015232845).
    Falls back to YYYYMMDDTHHMMSS with 'T' removed if needed.
    """
    s = str(path)

    m14 = re.findall(r"(?<!\d)(\d{14})(?!\d)", s)
    if m14:
        return m14[0]

    mT = re.search(r"(\d{8}T\d{6})", s)
    if mT:
        return mT.group(1).replace("T", "")

    return Path(path).stem


def make_output_tag(swot_path: str, is2_path: str) -> str:
    swot_tag = extract_swot_tag(swot_path)
    is2_tag = extract_is2_tag(is2_path)
    return f"{swot_tag}_SWOT_{is2_tag}_IS2"


# =============================================================================
# Geometry / projections
# =============================================================================

def get_transformer_4326_to_3031() -> "Transformer":
    """
    Robust transformer that does NOT require EPSG database.
    Falls back to explicit PROJ4 definitions if EPSG lookup fails.
    """
    if not HAS_PYPROJ:
        raise RuntimeError("Need pyproj. Install: pip install pyproj")

    from pyproj import CRS, Transformer

    try:
        return Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    except Exception:
        src = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
        dst = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 "
            "+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        return Transformer.from_crs(src, dst, always_xy=True)


def _wrap_lon180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized Haversine distance (km)."""
    R = 6371.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c


def moving_mean_nan(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware moving mean (1D)."""
    x = np.asarray(x, dtype=np.float64)
    if win <= 1:
        return x.copy()
    w = np.ones(win, dtype=np.float64)
    mask = np.isfinite(x).astype(np.float64)
    x0 = np.where(np.isfinite(x), x, 0.0)
    num = np.convolve(x0, w, mode="same")
    den = np.convolve(mask, w, mode="same")
    out = np.full_like(x, np.nan, dtype=np.float64)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def safe_nanquantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.nanquantile(x, q))


def _as_1d_float(a) -> np.ndarray:
    if a is None:
        return np.array([], dtype=np.float64)
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _cell_to_list(x) -> List:
    """Convert MATLAB cell-like content to Python list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    a = np.asarray(x)
    if a.dtype == object:
        return list(a.ravel())
    return [a]


# =============================================================================
# MAT loaders
# =============================================================================

def _load_mat_any(mat_path: str) -> Dict:
    """
    Load a .mat file as a python dict.
    - tries scipy.io.loadmat (v7)
    - falls back to mat73 (v7.3)
    """
    p = Path(mat_path)
    if not p.exists():
        raise FileNotFoundError(mat_path)

    if loadmat is not None:
        try:
            return loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception:
            pass

    if mat73 is not None:
        try:
            return mat73.loadmat(mat_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read MAT with mat73: {e}") from e

    raise RuntimeError(
        "Could not read .mat. If it is v7.3, install mat73 (pip install mat73). "
        "If it is v7, install scipy (pip install scipy)."
    )


# =============================================================================
# AMSR SIC index (KDTree) + MATLAB-style day selection
# =============================================================================

@dataclass
class AMSRSICIndex:
    tree: "cKDTree"
    sic: np.ndarray          # SIC value for each KDTree point
    max_nn_m: float
    transformer: "Transformer"

    def query_sic(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lonw = _wrap_lon180(lon)
        x, y = self.transformer.transform(lonw, lat)
        pts = np.column_stack([x, y])
        d, ii = self.tree.query(pts, k=1)
        out = np.full(len(lat), np.nan, dtype=np.float64)
        ok = (d <= self.max_nn_m) & np.isfinite(d) & np.isfinite(ii)
        out[ok] = self.sic[np.asarray(ii[ok], dtype=int)]
        return out

    def query_is_polynya(self, lat: np.ndarray, lon: np.ndarray, *, sic_thresh: float) -> np.ndarray:
        sicv = self.query_sic(lat, lon)
        return np.isfinite(sicv) & (sicv < float(sic_thresh))


def load_amsr2_sic_day_matlab_like(
    amsr_mat_path: str,
    *,
    ymd: str,
    out_shape: Tuple[int, int] = (1264, 1328),
    lat_name: str = "lat",
    lon_name: str = "lon",
    sic_name: str = "sic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce MATLAB logic:
      latt = double(lat');  lonn = double(lon');
      sic_day = reshape(sic(day_index2,:,:), 1264, 1328) ./ 100;
    """
    m = _load_mat_any(amsr_mat_path)

    if lat_name not in m or lon_name not in m or sic_name not in m:
        raise KeyError(f"AMSR mat missing one of: {lat_name}, {lon_name}, {sic_name}")

    lat = np.asarray(m[lat_name], dtype=np.float64)
    lon = np.asarray(m[lon_name], dtype=np.float64)
    sic = np.asarray(m[sic_name], dtype=np.float64)

    # MATLAB did lat', lon' (transpose)
    latt = lat.T if lat.ndim == 2 else lat
    lonn = lon.T if lon.ndim == 2 else lon

    # Pick day slice (MATLAB 1-based -> Python 0-based)
    di2 = _matlab_day_index2(ymd)
    idx0 = di2 - 1
    if idx0 < 0 or idx0 >= sic.shape[0]:
        raise IndexError(f"Requested date {ymd} -> idx {idx0} outside sic time dim {sic.shape[0]}")

    sl = np.asarray(sic[idx0, ...]).squeeze()

    # MATLAB reshape uses column-major
    n = int(np.prod(out_shape))
    if sl.size != n:
        raise ValueError(f"SIC day slice has {sl.size} elements; expected {n} for out_shape={out_shape}")

    sic_day = sl.reshape(out_shape, order="F") / 100.0

    # Ensure latt/lonn are 2D same shape
    if latt.ndim == 1 and lonn.ndim == 1:
        Lon, Lat = np.meshgrid(lonn, latt)
        lonn2, latt2 = Lon, Lat
    else:
        latt2, lonn2 = np.asarray(latt), np.asarray(lonn)

    if latt2.shape != sic_day.shape or lonn2.shape != sic_day.shape:
        if latt2.T.shape == sic_day.shape and lonn2.T.shape == sic_day.shape:
            latt2 = latt2.T
            lonn2 = lonn2.T
        else:
            raise ValueError(
                f"Shape mismatch: latt {latt2.shape}, lonn {lonn2.shape}, sic_day {sic_day.shape}. "
                "Check AMSR .mat variable shapes."
            )

    return latt2, lonn2, sic_day


def build_amsr2_day_index_from_matlab_like(
    amsr_mat_path: str,
    *,
    ymd: str,
    max_nn_km: float = 25.0,
) -> Tuple[AMSRSICIndex, Dict[str, float]]:
    if not (HAS_KDTREE and HAS_PYPROJ):
        raise RuntimeError("Need scipy.spatial.cKDTree and pyproj for AMSR index. Install: pip install scipy pyproj")

    latt, lonn, sic_day = load_amsr2_sic_day_matlab_like(amsr_mat_path, ymd=ymd)

    latv = latt.ravel()
    lonv = lonn.ravel()
    sicv = sic_day.ravel()

    ok = np.isfinite(latv) & np.isfinite(lonv) & np.isfinite(sicv)
    latv = latv[ok]
    lonv = _wrap_lon180(lonv[ok])
    sicv = sicv[ok]

    transformer = get_transformer_4326_to_3031()
    x, y = transformer.transform(lonv, latv)
    xy = np.column_stack([x, y])

    tree = cKDTree(xy)
    idx = AMSRSICIndex(tree=tree, sic=sicv, max_nn_m=float(max_nn_km) * 1000.0, transformer=transformer)

    meta = {"max_nn_km": float(max_nn_km)}
    return idx, meta


# =============================================================================
# SWOT polynya mask -> KDTree
# =============================================================================

@dataclass
class PolynyaMaskIndex:
    tree: "cKDTree"
    xy_polynya: np.ndarray  # [N,2] in EPSG:3031
    max_nn_m: float
    transformer: "Transformer"

    def query_in_polynya(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lonw = _wrap_lon180(lon)
        x, y = self.transformer.transform(lonw, lat)
        pts = np.column_stack([x, y])
        d, _ = self.tree.query(pts, k=1)
        return (d <= self.max_nn_m) & np.isfinite(d)


def _select_polynya_classes(
    class_names: np.ndarray,
    include: Optional[str],
    keywords: Tuple[str, ...] = ("polynya", "open", "water", "lead")
) -> List[int]:
    names = [str(n) for n in class_names.tolist()]
    lower = [n.lower() for n in names]
    if include:
        want = {s.strip().lower() for s in include.split(",") if s.strip()}
        return [i for i, n in enumerate(lower) if n in want]
    return [i for i, n in enumerate(lower) if any(k in n for k in keywords)]


def build_polynya_mask_index(
    swot_npz: str,
    *,
    class_include: Optional[str] = None,
    max_nn_km: float = 7.5,
) -> Tuple[PolynyaMaskIndex, Dict]:
    if not (HAS_KDTREE and HAS_PYPROJ):
        raise RuntimeError("Need scipy.spatial.cKDTree and pyproj. Install: pip install scipy pyproj")

    z = np.load(swot_npz, allow_pickle=True)
    lat = z["lat"]
    lon = z["lon"]
    labels = z["class_labels"]
    class_names = z["class_names"]

    pol_idx = _select_polynya_classes(class_names, class_include)
    if len(pol_idx) == 0:
        sigma0 = z["sigma0"] if "sigma0" in z.files else None
        ssh = z["ssh"] if "ssh" in z.files else None
        uniq = [u for u in np.unique(labels) if u >= 0]
        scores = []
        for u in uniq:
            m = labels == u
            sig_mean = float(np.nanmean(sigma0[m])) if (sigma0 is not None and np.isfinite(sigma0[m]).any()) else np.nan
            ssh_std  = float(np.nanstd(ssh[m]))    if (ssh is not None and np.isfinite(ssh[m]).any()) else np.nan
            scores.append((int(u), sig_mean, ssh_std))

        sig = np.array([s[1] for s in scores], float)
        std = np.array([s[2] for s in scores], float)

        def _norm(v):
            if not np.isfinite(v).any():
                return np.zeros_like(v)
            v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]))
            return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)

        S = _norm(sig) - _norm(std)
        order = np.argsort(S)[::-1]
        pol_idx = [scores[int(order[0])][0]]
        if len(order) > 1:
            pol_idx.append(scores[int(order[1])][0])
        print(f"[WARN] No class_names matched polynya keywords. Using heuristic classes: {pol_idx}")

    pol_mask = np.isin(labels, np.asarray(pol_idx, dtype=labels.dtype))
    ok = np.isfinite(lat) & np.isfinite(lon) & pol_mask

    lonw = _wrap_lon180(lon[ok])
    latv = lat[ok]

    transformer = get_transformer_4326_to_3031()
    x, y = transformer.transform(lonw, latv)
    xy = np.column_stack([x, y])

    tree = cKDTree(xy)
    idx = PolynyaMaskIndex(
        tree=tree,
        xy_polynya=xy,
        max_nn_m=float(max_nn_km) * 1000.0,
        transformer=transformer,
    )
    swot_dict = {k: z[k] for k in z.files}
    return idx, swot_dict


# =============================================================================
# Load MATLAB collocation MAT (expects: beam_data.(beam))
# =============================================================================

def load_is2_swot_mat(mat_path: str) -> Dict:
    return _load_mat_any(mat_path)


def _get_field(obj, name: str):
    if obj is None:
        raise KeyError(name)
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        raise KeyError(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    raise KeyError(name)


def iter_beams_from_mat(mat: Dict, beams: List[str]) -> Dict[str, object]:
    if "beam_data" not in mat:
        raise KeyError("Expected top-level variable 'beam_data' in MAT.")
    beam_data = mat["beam_data"]

    out = {}
    for b in beams:
        try:
            out[b] = _get_field(beam_data, b)
        except Exception:
            continue
    return out


def beam_to_arrays(D) -> Dict[str, object]:
    lat = _as_1d_float(_get_field(D, "lat"))
    lon = _as_1d_float(_get_field(D, "lon"))
    height = _as_1d_float(_get_field(D, "height_ice2"))

    N = lat.size

    # --- NEW: IS2 MSS (vector) ---
    try:
        mss_ice2 = _as_1d_float(_get_field(D, "mss_ice2"))
        if mss_ice2.size != N:
            # be strict but safe
            mss_ice2 = np.resize(mss_ice2, N).astype(np.float64)
    except Exception:
        mss_ice2 = np.full(N, np.nan, dtype=np.float64)

    sigma_swot = _cell_to_list(_get_field(D, "sigma_swot"))
    ssh_swot   = _cell_to_list(_get_field(D, "ssh_swot"))
    dist_min   = _cell_to_list(_get_field(D, "DIST_MIN"))

    # --- NEW: SWOT MSS (cell) ---
    try:
        mss_swot = _cell_to_list(_get_field(D, "mss_swot"))
    except Exception:
        mss_swot = [None] * N

    # truncate / align cell lists to N
    sigma_swot = list(np.asarray(sigma_swot, dtype=object).ravel())[:N]
    ssh_swot   = list(np.asarray(ssh_swot,   dtype=object).ravel())[:N]
    dist_min   = list(np.asarray(dist_min,   dtype=object).ravel())[:N]
    mss_swot   = list(np.asarray(mss_swot,   dtype=object).ravel())[:N]

    return dict(
        lat=lat, lon=lon, height=height, mss_ice2=mss_ice2,
        sigma_swot=sigma_swot, ssh_swot=ssh_swot, mss_swot=mss_swot, dist_min=dist_min
    )


# =============================================================================
# SWOT stats per IS2 point (within dist_max)
# =============================================================================

def compute_swot_stats_per_is2_point(
    sigma_cells: List,
    ssh_cells: List,
    dist_cells: List,
    *,
    mss_cells: Optional[List] = None,
    dist_max_m: float = 500.0,
) -> Dict[str, np.ndarray]:
    """
    Compute neighborhood stats (within dist_max_m) for:
      - sigma0 (mean/std)
      - ssh/ssha (mean/std)
      - mss (mean/std) if provided

    Each of sigma_cells/ssh_cells/dist_cells (and optionally mss_cells) is a per-IS2-point list/array.
    """
    N = len(dist_cells)

    sig_mean = np.full(N, np.nan)
    sig_std  = np.full(N, np.nan)
    ssh_mean = np.full(N, np.nan)
    ssh_std  = np.full(N, np.nan)

    mss_mean = np.full(N, np.nan)
    mss_std  = np.full(N, np.nan)

    has_mss = (mss_cells is not None) and (len(mss_cells) == N)

    for i in range(N):
        d = np.asarray(dist_cells[i], dtype=np.float64).reshape(-1) if dist_cells[i] is not None else np.array([])
        if d.size == 0 or not np.isfinite(d).any():
            continue

        ok = np.isfinite(d) & (d < float(dist_max_m))
        if not np.any(ok):
            continue

        sig = np.asarray(sigma_cells[i], dtype=np.float64).reshape(-1) if sigma_cells[i] is not None else np.array([])
        ssh = np.asarray(ssh_cells[i],   dtype=np.float64).reshape(-1) if ssh_cells[i]   is not None else np.array([])

        if sig.size == d.size and sig.size > 0:
            sigi = sig[ok]
            if np.isfinite(sigi).any():
                sig_mean[i] = np.nanmean(sigi)
                sig_std[i]  = np.nanstd(sigi)

        if ssh.size == d.size and ssh.size > 0:
            sshi = ssh[ok]
            if np.isfinite(sshi).any():
                ssh_mean[i] = np.nanmean(sshi)
                ssh_std[i]  = np.nanstd(sshi)

        if has_mss:
            mss = np.asarray(mss_cells[i], dtype=np.float64).reshape(-1) if mss_cells[i] is not None else np.array([])
            if mss.size == d.size and mss.size > 0:
                mssi = mss[ok]
                if np.isfinite(mssi).any():
                    mss_mean[i] = np.nanmean(mssi)
                    mss_std[i]  = np.nanstd(mssi)

    return dict(
        swot_sigma_mean=sig_mean,
        swot_sigma_std=sig_std,
        swot_ssh_mean=ssh_mean,
        swot_ssh_std=ssh_std,
        swot_mss_mean=mss_mean,
        swot_mss_std=mss_std,
    )


# =============================================================================
# Lead-like sea-surface candidate detection
# =============================================================================

def detect_sea_surface_candidates_matlab_style(
    X: np.ndarray,
    *,
    k: int = 3,
    rng_seed: int = 1,
    mahal_prct: float = 75.0,
    mahal_smooth_win: int = 10,
    peak_prom: float = 0.5,
    peak_dist: int = 10,
    sigma_smooth_win: int = 5,
    rolling_win: int = 10,
    sigma_prct: float = 75.0,
    sharpness_thresh: float = 0.03,
) -> Dict[str, np.ndarray]:
    if not (HAS_SKLEARN and HAS_SCIPY_SIGNAL):
        raise RuntimeError("Need scikit-learn and scipy.signal. Install: pip install scikit-learn scipy")

    N = X.shape[0]
    valid = np.all(np.isfinite(X), axis=1)

    out = {
        "valid": valid,
        "lead_flags_kmeans": np.zeros(N, dtype=bool),
        "peak_flags_mahal": np.zeros(N, dtype=bool),
        "is_lead_sigma": np.zeros(N, dtype=bool),
        "is_lead_sharpness": np.zeros(N, dtype=bool),
        "is_sea_surface_candidate": np.zeros(N, dtype=bool),
        "mahal": np.full(N, np.nan),
    }

    if np.sum(valid) < max(20, k * 5):
        return out

    Xv = X[valid].copy()

    scaler = StandardScaler().fit(Xv)
    Xn = scaler.transform(Xv)

    pca = PCA(n_components=min(4, Xn.shape[1]), random_state=rng_seed).fit(Xn)
    score = pca.transform(Xn)

    km = KMeans(n_clusters=k, n_init=80, random_state=rng_seed)
    cluster = km.fit_predict(score[:, :3])

    C_pc = km.cluster_centers_
    C_pad = np.zeros((k, score.shape[1]), dtype=float)
    C_pad[:, :3] = C_pc
    C_xn = pca.inverse_transform(C_pad)
    C_x = scaler.inverse_transform(C_xn)

    sig_mean = C_x[:, 0]
    sig_std  = C_x[:, 1]
    ssh_mean = C_x[:, 2]
    ssh_std  = C_x[:, 3]

    def _norm(v):
        v = np.asarray(v, float)
        return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)

    score_cluster = _norm(ssh_mean) + 0.5 * _norm(sig_std) + 0.5 * _norm(ssh_std)
    lead_cluster = int(np.argmin(score_cluster))
    lead_kmeans = (cluster == lead_cluster)

    sizes = np.bincount(cluster, minlength=k)
    others = [i for i in range(k) if i != lead_cluster]
    nonlead_cluster = int(others[int(np.argmax(sizes[others]))])

    Xn_non = Xn[cluster == nonlead_cluster]
    mu = np.mean(Xn_non, axis=0)
    Sigma = np.cov(Xn_non.T) + 1e-6 * np.eye(Xn_non.shape[1])

    try:
        L = np.linalg.cholesky(Sigma)
        Y = np.linalg.solve(L, (Xn - mu).T).T
        mahal = np.sqrt(np.sum(Y**2, axis=1))
    except np.linalg.LinAlgError:
        mahal = np.sqrt(np.sum((Xn - mu) ** 2, axis=1))

    mahal_full = np.full(N, np.nan)
    mahal_full[valid] = mahal
    out["mahal"] = mahal_full

    th = np.nanpercentile(mahal, mahal_prct)
    lead_kmeans = lead_kmeans & (mahal > th)

    mahal_s = moving_mean_nan(mahal_full, mahal_smooth_win)
    y = mahal_s[valid]
    peaks, _ = find_peaks(y, prominence=peak_prom, distance=peak_dist)
    peak_flags = np.zeros_like(y, dtype=bool)
    for p in peaks:
        a = max(0, p - 1); b = min(len(y), p + 2)
        peak_flags[a:b] = True

    sig_mean_full = np.full(N, np.nan)
    sig_mean_full[valid] = Xv[:, 0]
    sigma_smooth = moving_mean_nan(sig_mean_full, sigma_smooth_win)
    rolling = moving_mean_nan(sigma_smooth, rolling_win)
    sigma_thr = np.nanpercentile(sig_mean_full[valid], sigma_prct)
    is_lead_sigma = (sigma_smooth > rolling + 1.0) & (sigma_smooth > sigma_thr) & valid

    ysig = sigma_smooth[valid]
    peaks2, props = find_peaks(ysig, prominence=peak_prom, distance=peak_dist)
    sharp_flag = np.zeros_like(ysig, dtype=bool)
    if peaks2.size > 0:
        widths = peak_widths(ysig, peaks2, rel_height=0.5)[0]
        proms = props.get("prominences", np.ones_like(widths))
        sharpness = proms / (widths + 1e-12)
        keep = sharpness > sharpness_thresh
        for p in peaks2[keep]:
            a = max(0, int(p) - 1); b = min(len(ysig), int(p) + 2)
            sharp_flag[a:b] = True

    out["lead_flags_kmeans"][valid] = lead_kmeans
    out["peak_flags_mahal"][valid] = peak_flags
    out["is_lead_sigma"] = is_lead_sigma
    out["is_lead_sharpness"][valid] = sharp_flag

    out["is_sea_surface_candidate"] = (
        out["lead_flags_kmeans"] | out["peak_flags_mahal"] | out["is_lead_sigma"] | out["is_lead_sharpness"]
    )
    return out


def detect_is2_lead_candidates_height_only(
    height: np.ndarray,
    s_km: np.ndarray,
    *,
    win_km: float = 1.5,
    z_thresh: float = 3.0,
    local_q: float = 0.05,
    min_prom_m: float = 0.03,
    min_dist_km: float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    IS2-only lead / sea-surface candidate detection using height_ice2 only.

    - rolling median baseline
    - rolling MAD -> robust z-score
    - low outliers + local minima in detrended series
    """
    if not HAS_PANDAS:
        raise RuntimeError("IS2-only lead detection needs pandas. Install: pip install pandas")

    h = np.asarray(height, dtype=np.float64).reshape(-1)
    s = np.asarray(s_km, dtype=np.float64).reshape(-1)
    N = h.size

    out = {
        "valid": np.isfinite(h) & np.isfinite(s),
        "med": np.full(N, np.nan),
        "mad": np.full(N, np.nan),
        "detrended": np.full(N, np.nan),
        "z": np.full(N, np.nan),
        "cand_low": np.zeros(N, dtype=bool),
        "cand_minima": np.zeros(N, dtype=bool),
        "is_candidate": np.zeros(N, dtype=bool),
    }

    valid = out["valid"]
    if valid.sum() < 30:
        return out

    idx = np.where(valid)[0]
    order = np.argsort(s[idx])
    idxs = idx[order]
    hs = h[idxs]

    ds = np.diff(s[idxs])
    ds_med = float(np.nanmedian(ds[np.isfinite(ds) & (ds > 0)])) if np.any(np.isfinite(ds) & (ds > 0)) else np.nan
    if not np.isfinite(ds_med) or ds_med <= 0:
        ds_med = 0.02  # ~20 m fallback

    win = int(np.round(win_km / ds_med))
    win = max(21, win)
    if win % 2 == 0:
        win += 1

    ser = pd.Series(hs)
    med = ser.rolling(win, center=True, min_periods=max(5, win // 4)).median().to_numpy()
    detr = hs - med

    mad = pd.Series(np.abs(detr)).rolling(win, center=True, min_periods=max(5, win // 4)).median().to_numpy()
    mad = 1.4826 * mad
    z = detr / (mad + 1e-6)

    qloc = ser.rolling(win, center=True, min_periods=max(5, win // 4)).quantile(local_q).to_numpy()
    cand_low = (hs <= qloc) & (z < -float(z_thresh))

    cand_min = np.zeros_like(cand_low, dtype=bool)
    if HAS_SCIPY_SIGNAL:
        y = -detr
        min_dist = int(np.round(min_dist_km / ds_med))
        min_dist = max(5, min_dist)
        peaks, _ = find_peaks(y, prominence=float(min_prom_m), distance=min_dist)
        for p in peaks:
            a = max(0, int(p) - 1)
            b = min(len(y), int(p) + 2)
            cand_min[a:b] = True
        cand_min = cand_min & (hs <= qloc)

    out["med"][idxs] = med
    out["mad"][idxs] = mad
    out["detrended"][idxs] = detr
    out["z"][idxs] = z
    out["cand_low"][idxs] = cand_low
    out["cand_minima"][idxs] = cand_min
    out["is_candidate"] = out["cand_low"] | out["cand_minima"]
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Compute IS2 thin-ice SIT in polynya with joint 6-beam reference (SWOT + AMSR)."
    )

    ap.add_argument("--swot_class_npz", required=True)
    ap.add_argument("--is2_swot_mat", required=True)
    ap.add_argument("--outdir", default="results/is2_polynya_sit")

    # SWOT polynya
    ap.add_argument("--polynya_class_include", default=None)
    ap.add_argument("--polynya_nn_km", type=float, default=7.5)

    # AMSR polynya
    ap.add_argument("--amsr_sic_file", default=None, help="Path to AMSR2_2023.mat containing lat/lon/sic.")
    ap.add_argument("--amsr_sic_thresh", type=float, default=0.7, help="AMSR SIC threshold for polynya (e.g., 0.7).")
    ap.add_argument("--amsr_nn_km", type=float, default=25.0, help="Max NN distance for AMSR SIC query (km).")

    # SWOT stats radius
    ap.add_argument("--swot_dist_max_m", type=float, default=500.0)

    # Reference bins
    ap.add_argument(
        "--segment_km", type=float, default=2.0,
        help="Latitude bin size in km (converted using 111.32 km/deg)."
    )
    ap.add_argument("--ref_mode", choices=["lead", "low_quantile", "hybrid"], default="hybrid")
    ap.add_argument("--ref_quantile", type=float, default=0.05)
    ap.add_argument("--min_cand_per_seg", type=int, default=2)

    ap.add_argument("--disable_lead_detection", action="store_true")

    ap.add_argument("--rho_i", type=float, default=915.0)
    ap.add_argument("--rho_w", type=float, default=1024.0)

    ap.add_argument("--save_npz", action="store_true")

    args = ap.parse_args()

    if not HAS_PANDAS:
        raise RuntimeError("This script requires pandas. Install: pip install pandas")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = make_output_tag(args.swot_class_npz, args.is2_swot_mat)

    # --- AMSR daily SIC index (optional) ---
    amsr_idx = None
    amsr_ymd = None
    if args.amsr_sic_file:
        swot_tag = extract_swot_tag(args.swot_class_npz)
        amsr_ymd = _extract_yyyymmdd_from_tag(swot_tag)
        if amsr_ymd is None:
            raise RuntimeError("Cannot parse date from SWOT tag to select AMSR SIC day.")

        try:
            amsr_idx, _ = build_amsr2_day_index_from_matlab_like(
                args.amsr_sic_file,
                ymd=amsr_ymd,
                max_nn_km=float(args.amsr_nn_km),
            )
            print(f"[INFO] AMSR SIC day loaded for {amsr_ymd} from {args.amsr_sic_file}")
        except Exception as e:
            print(f"[WARN] AMSR SIC unavailable ({e}). AMSR-based polynya masking disabled.")
            amsr_idx = None

    # --- SWOT polynya mask index (optional) ---
    pol_idx = None
    try:
        pol_idx, _ = build_polynya_mask_index(
            args.swot_class_npz,
            class_include=args.polynya_class_include,
            max_nn_km=float(args.polynya_nn_km),
        )
    except Exception as e:
        print(f"[WARN] SWOT polynya mask unavailable ({e}). SWOT-based polynya masking disabled.")
        pol_idx = None

    # --- Load MATLAB collocation ---
    mat = load_is2_swot_mat(args.is2_swot_mat)
    beams = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
    beam_structs = iter_beams_from_mat(mat, beams)

    # --- Collect all points ---
    rows = []
    for beam in beams:
        if beam not in beam_structs:
            continue

        D = beam_structs[beam]
        B = beam_to_arrays(D)

        lat = B["lat"]; lon = B["lon"]; h = B["height"]

        # per-beam along-track distance (for ordering)
        if lat.size >= 2:
            dstep = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            s_km_beam = np.concatenate([[0.0], np.cumsum(dstep)])
        else:
            s_km_beam = np.zeros_like(lat, dtype=np.float64)

        stats = compute_swot_stats_per_is2_point(
            B["sigma_swot"], B["ssh_swot"], B["dist_min"],
            mss_cells=B.get("mss_swot", None),
            dist_max_m=float(args.swot_dist_max_m),
        )


        in_pol_swot = pol_idx.query_in_polynya(lat, lon) if pol_idx is not None else np.zeros_like(lat, dtype=bool)

        amsr_sic = amsr_idx.query_sic(lat, lon) if amsr_idx is not None else np.full_like(lat, np.nan, dtype=np.float64)
        in_pol_amsr = (
            np.isfinite(amsr_sic) & (amsr_sic < float(args.amsr_sic_thresh))
            if amsr_idx is not None else np.zeros_like(lat, dtype=bool)
        )

        for i in range(lat.size):
            rows.append({
                "beam": beam,
                "i": int(i),
                "lat": float(lat[i]) if np.isfinite(lat[i]) else np.nan,
                "lon": float(lon[i]) if np.isfinite(lon[i]) else np.nan,
                "height": float(h[i]) if np.isfinite(h[i]) else np.nan,
                "s_km_beam": float(s_km_beam[i]) if np.isfinite(s_km_beam[i]) else np.nan,

                # keep both sources
                "in_polynya_swot": bool(in_pol_swot[i]),
                "amsr_sic": float(amsr_sic[i]) if np.isfinite(amsr_sic[i]) else np.nan,
                "in_polynya_amsr": bool(in_pol_amsr[i]),

                # placeholder final (decided later)
                "in_polynya": False,
                "polynya_source": "",

                # SWOT stats (may be NaN if no collocated SWOT points)
                "swot_sigma_mean": float(stats["swot_sigma_mean"][i]),
                "swot_sigma_std":  float(stats["swot_sigma_std"][i]),
                "swot_ssh_mean":   float(stats["swot_ssh_mean"][i]),
                "swot_ssh_std":    float(stats["swot_ssh_std"][i]),
                "mss_ice2": float(B["mss_ice2"][i]) if np.isfinite(B["mss_ice2"][i]) else np.nan,

                "swot_mss_mean": float(stats["swot_mss_mean"][i]),
                "swot_mss_std":  float(stats["swot_mss_std"][i]),

            })

    df = pd.DataFrame(rows)
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df = df[np.isfinite(df["height"].values)].copy()

    # --- NEW: SWOT anomaly on IS2 MSS reference ---
    for c in ["mss_ice2", "swot_mss_mean", "swot_ssh_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # absolute SWOT SSH (ellipsoid-like) from collocation: SSHA + MSS
    df["swot_ssh_abs_mean"] = df["swot_ssh_mean"] + df["swot_mss_mean"]

    # requested aligned anomaly on IS2 MSS:
    # eta_swot = (ssha_swot + mss_swot) - mss_is2
    df["SWOT_SSHA_IS2MSS"] = df["swot_ssh_abs_mean"] - df["mss_ice2"]


    print(f"[INFO] Loaded points (finite height): {len(df)}")

    # --- Choose polynya mask per beam (AMSR-first rule) ---
    used_any_polynya = False
    for beam in beams:
        mb = (df["beam"].values == beam)
        if not np.any(mb):
            continue

        swot_count = int(np.sum(mb & df["in_polynya_swot"].values.astype(bool)))
        amsr_count = int(np.sum(mb & df["in_polynya_amsr"].values.astype(bool)))

        if amsr_idx is None and pol_idx is None:
            src = "NONE"
            mask_final = np.zeros(int(np.sum(mb)), dtype=bool)
        elif amsr_idx is None:
            src = "SWOT"
            mask_final = df.loc[mb, "in_polynya_swot"].values.astype(bool)
        elif pol_idx is None:
            src = "AMSR"
            mask_final = df.loc[mb, "in_polynya_amsr"].values.astype(bool)
        else:
            # requested rule
            if swot_count < amsr_count:
                src = "AMSR"
                mask_final = df.loc[mb, "in_polynya_amsr"].values.astype(bool)
            else:
                src = "SWOT"
                mask_final = df.loc[mb, "in_polynya_swot"].values.astype(bool)

        df.loc[mb, "polynya_source"] = src
        df.loc[mb, "in_polynya"] = mask_final
        used_any_polynya = used_any_polynya or bool(np.any(mask_final))

        print(f"[INFO] Beam {beam}: SWOT polynya={swot_count}, AMSR polynya={amsr_count} -> using {src}")

    n_pol = int(df["in_polynya"].sum()) if "in_polynya" in df.columns else 0
    print(f"[INFO] Points in polynya (final mask): {n_pol}")

    # Decide whether to use polynya-masked workflow or IS2-only fallback
    use_is2_only_fallback = (n_pol == 0)
    if use_is2_only_fallback:
        print("[WARN] Points in polynya (final mask) = 0 -> using IS2-only (height-only) lead detection + reference.")
        df["in_scope"] = np.isfinite(df["height"].values)
    else:
        df["in_scope"] = df["in_polynya"].values.astype(bool)

    # --- JOINT latitude segments (shared across all beams) ---
    seg_km = float(args.segment_km)
    dlat = seg_km / 111.32  # deg

    latv = df["lat"].to_numpy(dtype=float)
    lat0 = np.nanmin(latv)

    seg_lat = np.full_like(latv, np.nan, dtype=float)
    ok_lat = np.isfinite(latv)
    seg_lat[ok_lat] = np.floor((latv[ok_lat] - lat0) / dlat)

    df["t_km_joint"] = np.nan  # retained for compatibility
    df["seg_joint"] = pd.Series(seg_lat, index=df.index).astype("Int64")

    # --- Candidate detection ---
    df["is_sea_surface_candidate"] = False
    df["mahal"] = np.nan  # filled only in SWOT-feature mode

    if (not args.disable_lead_detection) and (args.ref_mode in ("lead", "hybrid")):
        if use_is2_only_fallback:
            # IS2-only lead detection per beam (height-only)
            for beam in beams:
                m = (
                    (df["beam"].values == beam)
                    & (df["in_scope"].values.astype(bool))
                    & np.isfinite(df["height"].values)
                    & np.isfinite(df["s_km_beam"].values)
                )
                if np.sum(m) < 30:
                    continue

                idx = np.where(m)[0]
                order = np.argsort(df.loc[idx, "s_km_beam"].values)
                idx_sorted = idx[order]

                det = detect_is2_lead_candidates_height_only(
                    df.loc[idx_sorted, "height"].values.astype(np.float64),
                    df.loc[idx_sorted, "s_km_beam"].values.astype(np.float64),
                    win_km=max(1.0, float(args.segment_km) * 0.75),
                    z_thresh=3.0,
                    local_q=max(0.01, float(args.ref_quantile)),
                    min_prom_m=0.03,
                    min_dist_km=0.2,
                )
                df.loc[idx_sorted, "is_sea_surface_candidate"] = det["is_candidate"]

        else:
            # SWOT-feature-based detection (within in_scope polynya points)
            if not (HAS_SKLEARN and HAS_SCIPY_SIGNAL):
                raise RuntimeError("Lead detection needs scikit-learn + scipy. Install: pip install scikit-learn scipy")

            feat_cols = ["swot_sigma_mean", "swot_sigma_std", "swot_ssh_mean", "swot_ssh_std"]

            for beam in beams:
                m = (
                    (df["beam"].values == beam)
                    & (df["in_scope"].values.astype(bool))
                    & np.all(np.isfinite(df[feat_cols].values), axis=1)
                    & np.isfinite(df["s_km_beam"].values)
                )
                if np.sum(m) < 30:
                    continue

                idx = np.where(m)[0]
                order = np.argsort(df.loc[idx, "s_km_beam"].values)
                idx_sorted = idx[order]

                X = df.loc[idx_sorted, feat_cols].values.astype(np.float64)
                det = detect_sea_surface_candidates_matlab_style(X)

                df.loc[idx_sorted, "is_sea_surface_candidate"] = det["is_sea_surface_candidate"]
                df.loc[idx_sorted, "mahal"] = det["mahal"]

    # --- Joint reference per latitude segment (pool ALL beams) ---
    df["ref_height"] = np.nan
    q = float(args.ref_quantile)
    min_cand = int(args.min_cand_per_seg)

    seg_ids = df.loc[df["in_scope"].values.astype(bool), "seg_joint"].dropna().unique()
    seg_ids = np.array(seg_ids, dtype=int)

    ref_records = []
    for seg in np.sort(seg_ids):
        idx_seg = (
            (df["seg_joint"].values.astype(float) == float(seg))
            & (df["in_scope"].values.astype(bool))
            & np.isfinite(df["height"].values)
        )
        h_seg = df.loc[idx_seg, "height"].values.astype(np.float64)

        ref = np.nan
        n_in_scope = int(np.sum(np.isfinite(h_seg)))

        if args.ref_mode == "low_quantile":
            ref = safe_nanquantile(h_seg, q)
            n_cand = 0
        else:
            idx_cand = idx_seg & (df["is_sea_surface_candidate"].values.astype(bool))
            h_cand = df.loc[idx_cand, "height"].values.astype(np.float64)
            n_cand = int(np.sum(np.isfinite(h_cand)))

            if n_cand >= min_cand:
                ref = safe_nanquantile(h_cand, q)
            else:
                ref = safe_nanquantile(h_seg, q)

        df.loc[idx_seg, "ref_height"] = ref
        ref_records.append({
            "seg_joint": int(seg),
            "ref_height": float(ref),
            "n_in_scope": n_in_scope,
            "n_cand": n_cand,
        })

    ref_df = pd.DataFrame.from_records(
        ref_records,
        columns=["seg_joint", "ref_height", "n_in_scope", "n_cand"],
    )
    if not ref_df.empty:
        ref_df = ref_df.sort_values("seg_joint")
    else:
        print("[WARN] No in-scope segments found. Reference table is empty.")

    # --- Freeboard + SIT ---
    rho_i = float(args.rho_i)
    rho_w = float(args.rho_w)

    df["freeboard"] = df["height"] - df["ref_height"]
    df.loc[~df["in_scope"].values.astype(bool), "freeboard"] = np.nan

    df["sit"] = np.nan
    ok_fb = (
        df["in_scope"].values.astype(bool)
        & np.isfinite(df["freeboard"].values)
        & (df["freeboard"].values > 0)
    )
    df.loc[ok_fb, "sit"] = df.loc[ok_fb, "freeboard"] * rho_w / (rho_w - rho_i)

    # --- Save outputs (tagged) ---
    points_csv = outdir / f"{tag}_is2_polynya_sit_points.csv"
    ref_csv    = outdir / f"{tag}_sea_surface_reference_by_latbin.csv"
    df.to_csv(points_csv, index=False)
    ref_df.to_csv(ref_csv, index=False)

    print(f"Saved points: {points_csv}")
    print(f"Saved refs  : {ref_csv}")

    if args.save_npz:
        npz_path = outdir / f"{tag}_is2_polynya_sit_outputs.npz"
        np.savez_compressed(
            npz_path,
            beam=df["beam"].values.astype(str),
            lat=df["lat"].values.astype(np.float64),
            lon=df["lon"].values.astype(np.float64),
            height=df["height"].values.astype(np.float64),
            s_km_beam=df["s_km_beam"].values.astype(np.float64),
            seg_joint=df["seg_joint"].fillna(-1).values.astype(np.int64),

            in_polynya=df["in_polynya"].values.astype(bool),
            in_polynya_swot=df["in_polynya_swot"].values.astype(bool),
            in_polynya_amsr=df["in_polynya_amsr"].values.astype(bool),
            amsr_sic=df["amsr_sic"].values.astype(np.float64),
            polynya_source=df["polynya_source"].values.astype("U"),

            in_scope=df["in_scope"].values.astype(bool),
            is_sea_surface_candidate=df["is_sea_surface_candidate"].values.astype(bool),
            ref_height=df["ref_height"].values.astype(np.float64),
            freeboard=df["freeboard"].values.astype(np.float64),
            sit=df["sit"].values.astype(np.float64),

            ref_seg_joint=ref_df["seg_joint"].values.astype(np.int64) if not ref_df.empty else np.array([], dtype=np.int64),
            ref_height_seg=ref_df["ref_height"].values.astype(np.float64) if not ref_df.empty else np.array([], dtype=np.float64),
            n_in_scope_seg=ref_df["n_in_scope"].values.astype(np.int64) if not ref_df.empty else np.array([], dtype=np.int64),
            n_cand_seg=ref_df["n_cand"].values.astype(np.int64) if not ref_df.empty else np.array([], dtype=np.int64),

            meta_tag=np.array([tag], dtype="U"),
            meta_segment_km=np.array([float(args.segment_km)], dtype=float),
            meta_ref_mode=np.array([str(args.ref_mode)], dtype="U"),
            meta_ref_quantile=np.array([float(args.ref_quantile)], dtype=float),
            meta_min_cand_per_seg=np.array([int(args.min_cand_per_seg)], dtype=int),
            meta_used_is2_only=np.array([bool(use_is2_only_fallback)], dtype=bool),

            meta_amsr_file=np.array([str(args.amsr_sic_file) if args.amsr_sic_file else ""], dtype="U"),
            meta_amsr_date=np.array([str(amsr_ymd) if amsr_ymd else ""], dtype="U"),
            meta_amsr_thresh=np.array([float(args.amsr_sic_thresh)], dtype=float),
            meta_amsr_nn_km=np.array([float(args.amsr_nn_km)], dtype=float),
            swot_sigma_mean=df["swot_sigma_mean"].values.astype(np.float64),
            swot_sigma_std =df["swot_sigma_std"].values.astype(np.float64),
            swot_ssh_mean  =df["swot_ssh_mean"].values.astype(np.float64),
            swot_ssh_std   =df["swot_ssh_std"].values.astype(np.float64),
            mahal          =df["mahal"].values.astype(np.float64),
            mss_ice2=df["mss_ice2"].values.astype(np.float64),
            swot_mss_mean=df["swot_mss_mean"].values.astype(np.float64),
            swot_mss_std=df["swot_mss_std"].values.astype(np.float64),

            swot_ssh_abs_mean=df["swot_ssh_abs_mean"].values.astype(np.float64),
            SWOT_SSHA_IS2MSS=df["SWOT_SSHA_IS2MSS"].values.astype(np.float64),

        )
        print(f"Saved NPZ  : {npz_path}")


if __name__ == "__main__":
    main()
