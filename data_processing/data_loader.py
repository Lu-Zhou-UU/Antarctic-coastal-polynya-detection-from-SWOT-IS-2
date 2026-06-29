#!/usr/bin/env python3
"""
load_data.py

Data Loading Module for SWOT and IS2 Satellite Data

Supports:
- SWOT NetCDF files (robust SSH/SSHA discovery + optional NetCDF group probing)
- IS2 HDF5 files (ATL10/ATL07-style beam groups)
- MATLAB .mat files (legacy)
- Creating collocation datasets (KDTree in meters, robust at high latitudes)

Key update (2026-01):
- Preference order updated to prioritize 'ssha_unfiltered' first (your NetCDF naming)
- Expose `ssh_var` override in load_swot_scene()
- Store chosen ssh variable name as SWOTSceneData.ssh_name
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY_SPATIAL = True
except ImportError:
    HAS_SCIPY_SPATIAL = False


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class BeamData:
    """Container for IS2 beam data."""
    lat: np.ndarray
    lon: np.ndarray
    height_ice2: np.ndarray
    ssh: Optional[np.ndarray] = None
    beam_name: str = "unknown"
    track_name: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "height_ice2": self.height_ice2,
            "ssh": self.ssh,
            "beam_name": self.beam_name,
            "track_name": self.track_name,
        }


@dataclass
class SWOTSceneData:
    """Container for SWOT 2D scene data."""
    sigma0: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    ssh: Optional[np.ndarray] = None
    ssh_name: Optional[str] = None
    time: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "sigma0": self.sigma0,
            "lat": self.lat,
            "lon": self.lon,
            "ssh": self.ssh,
            "ssh_name": self.ssh_name,
            "time": self.time,
        }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _wrap_lon_to_180(lon: np.ndarray) -> np.ndarray:
    """Wrap longitude to [-180, 180)."""
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _as_2d_latlon(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure lat/lon are 2D grids aligned with SWOT image arrays."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim == 2 and lon.ndim == 2:
        return lat, lon
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d, lon2d
    raise ValueError(f"Unexpected lat/lon dims: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")


def _robust_is_sigma0_db(values: np.ndarray, units: Optional[str]) -> bool:
    """
    Heuristic: determine if sigma0 is already in dB.
    - If units explicitly says dB/db -> True
    - Else check ranges
    """
    if units:
        u = str(units).strip().lower()
        if "db" in u:
            return True

    v = np.asarray(values, dtype=np.float64)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return False

    p1, p99 = np.nanpercentile(vv, [1, 99])
    if p99 <= 5.0 and p1 >= 0.0:
        return False  # linear-like
    if p1 >= -120.0 and p99 <= 120.0:
        return True
    return False


def _pick_ssh_var(
    ds: "xr.Dataset",
    target_shape: Optional[Tuple[int, int]] = None,
    prefer_unfiltered: bool = True,
) -> Optional[str]:
    """
    Pick an SSH/SSHA variable from a SWOT dataset.

    Strategy:
      1) Try preferred explicit names (order depends on prefer_unfiltered)
      2) Fall back to scanning any var containing 'ssh' or 'ssha' (excluding uncertainties/flags)
      3) Require 2D and (optionally) matching target_shape
    """
    # Updated preference order: SSHA_UNFILTERED first
    if prefer_unfiltered:
        preferred = [
            "ssha_unfiltered",
            "ssh_unfiltered",
            "ssha_filtered",
            "ssh_filtered",
            "ssha_unedited",
            "ssh_unedited",
            "ssha",
            "ssh",
            "sea_surface_height_anomaly_unfiltered",
            "sea_surface_height_anomaly_filtered",
            "sea_surface_height_anomaly",
            "sea_surface_height_unfiltered",
            "sea_surface_height_filtered",
            "sea_surface_height",
            "ssh_karin",
            "ssha_karin",
        ]
    else:
        preferred = [
            "ssha_filtered",
            "ssh_filtered",
            "ssha_unfiltered",
            "ssh_unfiltered",
            "ssha_unedited",
            "ssh_unedited",
            "ssha",
            "ssh",
            "sea_surface_height_anomaly_filtered",
            "sea_surface_height_anomaly_unfiltered",
            "sea_surface_height_anomaly",
            "sea_surface_height_filtered",
            "sea_surface_height_unfiltered",
            "sea_surface_height",
            "ssh_karin",
            "ssha_karin",
        ]

    def ok_name(v: str) -> bool:
        vl = v.lower()
        bad = ("uncert", "sigma", "std", "quality", "qual", "flag", "mask", "rms", "variance", "var_")
        return not any(b in vl for b in bad)

    def shape_ok(v: str) -> bool:
        a = ds[v]
        if a.ndim != 2:
            return False
        if target_shape is None:
            return True
        return tuple(a.shape) == tuple(target_shape)

    for v in preferred:
        if v in ds.data_vars and ok_name(v) and shape_ok(v):
            return v

    candidates: List[str] = []
    for v in ds.data_vars:
        vl = v.lower()
        if ("ssh" in vl or "ssha" in vl) and ok_name(v) and shape_ok(v):
            candidates.append(v)

    if not candidates:
        return None

    token_rank = ["unfiltered", "filtered", "unedited", "karin", "sea_surface_height", "ssh", "ssha"]

    def rank(v: str) -> int:
        vl = v.lower()
        hits = [token_rank.index(t) for t in token_rank if t in vl]
        return min(hits) if hits else 999

    candidates.sort(key=rank)
    return candidates[0]


def _try_open_swot_dataset(filepath: Path, groups_to_try: Optional[List[Optional[str]]] = None) -> "xr.Dataset":
    """Try opening a SWOT NetCDF with xarray, probing common groups."""
    if groups_to_try is None:
        groups_to_try = [None, "pixel_cloud", "karin", "swath", "ssh", "ssha", "data", "science"]

    last_err = None
    for g in groups_to_try:
        try:
            if g is None:
                ds = xr.open_dataset(filepath)
                ds.attrs["_opened_group"] = ""
            else:
                ds = xr.open_dataset(filepath, group=g)
                ds.attrs["_opened_group"] = g
            return ds
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not open {filepath.name} with any tried group. Last error: {last_err}")


def _latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Equirectangular local meters mapping for KDTree distances."""
    R = 6371000.0
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(float(lat0_deg))

    x = R * np.cos(lat0) * lon
    y = R * lat
    return x, y


# -----------------------------------------------------------------------------
# Main Loader
# -----------------------------------------------------------------------------
class SWOTISDataLoader:
    """Load and collate SWOT and IS2 data."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)

    # ------------------------ IS2 ------------------------
    def load_is2_track(
        self,
        filepath: str,
        beam: str = "gt1l",
        height_key: str = "h_li",
    ) -> Optional[BeamData]:
        """Load ICESat-2 ATL10/ATL07-like beam data from HDF5."""
        if not HAS_H5PY:
            logger.error("h5py not installed. Install with: pip install h5py")
            return None

        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        logger.info(f"Loading IS2 data from {filepath.name}")
        logger.info(f"  Beam: {beam}")

        try:
            with h5py.File(filepath, "r") as f:
                if beam not in f:
                    available = [k for k in f.keys() if k.startswith("gt")]
                    logger.error(f"Beam {beam} not found. Available: {available}")
                    return None

                g = f[beam]

                height_data = None
                for key in [height_key, "h_li", "h_te", "height"]:
                    if key in g:
                        height_data = g[key][:]
                        logger.info(f"  Loaded height data from '{key}'")
                        break
                if height_data is None:
                    logger.error(f"Height data not found in beam {beam}. Keys: {list(g.keys())[:50]}")
                    return None

                lat = g["latitude"][:] if "latitude" in g else None
                lon = g["longitude"][:] if "longitude" in g else None

                if lat is None:
                    if "lat" in g:
                        lat = g["lat"][:]
                    elif "y" in g:
                        lat = g["y"][:]
                if lon is None:
                    if "lon" in g:
                        lon = g["lon"][:]
                    elif "x" in g:
                        lon = g["x"][:]

                if lat is None or lon is None:
                    logger.error("Latitude or longitude not found in IS2 beam group.")
                    return None

                ssh = None
                for ssh_key in ["ssha_filtered", "ssha_unfiltered", "ssha_unedited", "ssh", "ssha"]:
                    if ssh_key in g:
                        ssh = g[ssh_key][:]
                        logger.info(f"  Loaded SSH-like field from '{ssh_key}'")
                        break

                lat = lat.astype(float)
                lon = _wrap_lon_to_180(lon.astype(float))
                height_data = height_data.astype(float)
                if ssh is not None:
                    ssh = np.asarray(ssh, dtype=float)

                logger.info(f"  Loaded {len(lat)} points")

                return BeamData(
                    lat=lat,
                    lon=lon,
                    height_ice2=height_data,
                    ssh=ssh,
                    beam_name=beam,
                    track_name=filepath.stem,
                )

        except Exception as e:
            logger.error(f"Error loading IS2 file: {e}")
            return None

    # ------------------------ SWOT ------------------------
    def load_swot_scene(
        self,
        filepath: str,
        *,
        prefer_unfiltered_ssh: bool = True,
        ssh_var: Optional[str] = None,
        sigma0_in_db: Optional[bool] = None,
        try_groups: bool = True,
    ) -> Optional[SWOTSceneData]:
        """
        Load SWOT scene data from NetCDF.

        Parameters
        ----------
        prefer_unfiltered_ssh : bool
            Auto selection preference if ssh_var is None.
        ssh_var : Optional[str]
            Force a specific SSH/SSHA variable name (e.g. "ssha_unfiltered").
            If provided but missing / wrong shape, we warn and fall back to auto.
        """
        if not HAS_XARRAY:
            logger.error("xarray not installed. Install with: pip install xarray")
            return None

        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        logger.info(f"Loading SWOT scene from {filepath.name}")

        try:
            ds = _try_open_swot_dataset(filepath) if try_groups else xr.open_dataset(filepath)
            opened_group = ds.attrs.get("_opened_group", "")
            if opened_group:
                logger.info(f"  Opened NetCDF group: '{opened_group}'")

            vars_ssh = [v for v in ds.data_vars if ("ssh" in v.lower() or "ssha" in v.lower())]
            logger.info(f"  Vars containing 'ssh'/'ssha' (first 50): {vars_ssh[:50]}")

            # Find sigma0
            sigma0_key = None
            for key in ["sigma0", "backscatter", "normalized_radar_cross_section"]:
                if key in ds.data_vars:
                    sigma0_key = key
                    break
            if sigma0_key is None:
                logger.error("Sigma0/backscatter data not found.")
                logger.error(f"Available variables (first 60): {list(ds.data_vars)[:60]}")
                ds.close()
                return None

            sigma0 = ds[sigma0_key].values
            if sigma0.ndim != 2:
                logger.error(f"sigma0 variable '{sigma0_key}' is not 2D (shape={sigma0.shape}).")
                ds.close()
                return None

            # dB conversion
            units = ds[sigma0_key].attrs.get("units", None)
            if sigma0_in_db is None:
                already_db = _robust_is_sigma0_db(sigma0, units)
            else:
                already_db = bool(sigma0_in_db)

            if not already_db:
                logger.info("  Converting sigma0 to dB (auto-detected linear units).")
                sigma0 = 10.0 * np.log10(np.maximum(sigma0, 1e-12))
            else:
                logger.info("  sigma0 appears to already be in dB (no conversion).")

            # Lat/Lon
            lat = ds["latitude"].values if "latitude" in ds.variables else None
            lon = ds["longitude"].values if "longitude" in ds.variables else None
            if lat is None and "lat" in ds.variables:
                lat = ds["lat"].values
            if lon is None and "lon" in ds.variables:
                lon = ds["lon"].values

            if lat is None or lon is None:
                logger.error("Latitude or longitude not found in SWOT dataset.")
                logger.error(f"Available variables (first 60): {list(ds.variables)[:60]}")
                ds.close()
                return None

            lat2d, lon2d = _as_2d_latlon(lat, lon)
            lon2d = _wrap_lon_to_180(lon2d)

            # SSH/SSHA
            ssh = None
            ssh_name = None

            # 1) forced
            if ssh_var is not None:
                if ssh_var in ds.data_vars and ds[ssh_var].ndim == 2 and tuple(ds[ssh_var].shape) == tuple(sigma0.shape):
                    ssh = ds[ssh_var].values
                    ssh_name = str(ssh_var)
                    logger.info(f"  Loaded SSH/SSHA from forced '{ssh_name}' (shape={ssh.shape})")
                else:
                    logger.warning(
                        f"  Forced ssh_var='{ssh_var}' missing or wrong shape; falling back to auto selection."
                    )

            # 2) auto
            if ssh is None:
                ssh_key = _pick_ssh_var(ds, target_shape=sigma0.shape, prefer_unfiltered=prefer_unfiltered_ssh)
                if ssh_key is not None:
                    ssh = ds[ssh_key].values
                    ssh_name = str(ssh_key)
                    logger.info(f"  Loaded SSH/SSHA from '{ssh_name}' (shape={ssh.shape})")
                else:
                    if vars_ssh:
                        logger.warning(
                            f"  SSH/SSHA-like vars exist but none matched shape={sigma0.shape} and 2D requirement."
                        )
                    else:
                        logger.warning("  No SSH/SSHA variables found in this opened dataset/group.")

            logger.info(f"  Loaded SWOT image sigma0 shape={sigma0.shape}")
            ds.close()

            return SWOTSceneData(
                sigma0=np.asarray(sigma0, dtype=float),
                lat=np.asarray(lat2d, dtype=float),
                lon=np.asarray(lon2d, dtype=float),
                ssh=np.asarray(ssh, dtype=float) if ssh is not None else None,
                ssh_name=ssh_name,
                time=str(filepath.stem),
            )

        except Exception as e:
            logger.error(f"Error loading SWOT file: {e}")
            return None

    # ------------------------ MATLAB legacy ------------------------
    def load_matlab_beam_day(self, filepath: str) -> Optional[Dict[str, BeamData]]:
        """Load legacy MATLAB Beam_day structure."""
        if not HAS_SCIPY:
            logger.error("scipy not installed. Install with: pip install scipy")
            return None

        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        logger.info(f"Loading MATLAB Beam_day from {filepath.name}")

        try:
            mat = loadmat(filepath)
            beam_day = mat.get("Beam_day", None)
            if beam_day is None:
                logger.error("Beam_day not found in MATLAB file")
                return None

            beams: Dict[str, BeamData] = {}
            beam_names = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

            for beam_name in beam_names:
                try:
                    beam_struct = beam_day[0, 0][beam_name][0, 0]
                    lat = beam_struct["lat"][:, 0].flatten().astype(float)
                    lon = _wrap_lon_to_180(beam_struct["lon"][:, 0].flatten().astype(float))
                    h = beam_struct["height_ice2"][:, 0].flatten().astype(float)

                    ssh = None
                    if "ssh" in beam_struct.dtype.names:
                        ssh = np.asarray(beam_struct["ssh"], dtype=float).reshape(-1)
                        if ssh.size == 0:
                            ssh = None

                    beams[beam_name] = BeamData(
                        lat=lat,
                        lon=lon,
                        height_ice2=h,
                        ssh=ssh,
                        beam_name=beam_name,
                        track_name=filepath.stem,
                    )
                    logger.info(f"  Loaded beam {beam_name}: {len(lat)} points")

                except Exception as e:
                    logger.debug(f"Could not load beam {beam_name}: {e}")

            return beams

        except Exception as e:
            logger.error(f"Error loading MATLAB file: {e}")
            return None

    # ------------------------ Collocation ------------------------
    def collate_is2_swot(
        self,
        beam_data: BeamData,
        swot_data: SWOTSceneData,
        dist_max: float = 3000.0,
        verbose: bool = True,
    ) -> Optional[Dict]:
        """Collate IS2 and SWOT by finding all SWOT pixels within dist_max (meters) of each IS2 point."""
        if not HAS_SCIPY_SPATIAL:
            logger.error("scipy.spatial not available. Install with: pip install scipy")
            return None

        logger.info(f"Collating IS2 ({len(beam_data.lat)} pts) and SWOT ({swot_data.sigma0.shape})")
        logger.info(f"  Max distance: {dist_max} m")

        lat2d, lon2d = _as_2d_latlon(swot_data.lat, swot_data.lon)
        lon2d = _wrap_lon_to_180(lon2d)

        sigma0 = np.asarray(swot_data.sigma0)
        ssh2d = np.asarray(swot_data.ssh) if swot_data.ssh is not None else None

        latf = lat2d.ravel()
        lonf = lon2d.ravel()
        sigf = sigma0.ravel()
        sshf = ssh2d.ravel() if ssh2d is not None else None

        ok = np.isfinite(latf) & np.isfinite(lonf) & np.isfinite(sigf)
        if ok.sum() == 0:
            logger.error("No valid SWOT pixels after filtering finite lat/lon/sigma0.")
            return None

        latf_ok = latf[ok]
        lonf_ok = lonf[ok]
        sigf_ok = sigf[ok]
        sshf_ok = sshf[ok] if sshf is not None else None

        lat0 = float(np.nanmean(latf_ok))
        xs, ys = _latlon_to_local_xy_m(latf_ok, lonf_ok, lat0_deg=lat0)
        swot_xy = np.column_stack([xs, ys])

        logger.info("  Building KD-tree of SWOT pixels in local meters...")
        tree = cKDTree(swot_xy)

        is2_lat = np.asarray(beam_data.lat, dtype=float)
        is2_lon = _wrap_lon_to_180(np.asarray(beam_data.lon, dtype=float))
        xi, yi = _latlon_to_local_xy_m(is2_lat, is2_lon, lat0_deg=lat0)
        is2_xy = np.column_stack([xi, yi])

        logger.info("  Querying collocations...")
        dist_min_list: List[np.ndarray] = []
        sigma_swot_list: List[np.ndarray] = []
        ssh_swot_list: List[np.ndarray] = []

        for i in range(is2_xy.shape[0]):
            idx = tree.query_ball_point(is2_xy[i], r=float(dist_max))

            if len(idx) == 0:
                dist_min_list.append(np.array([], dtype=float))
                sigma_swot_list.append(np.array([], dtype=float))
                ssh_swot_list.append(np.array([], dtype=float))
            else:
                pts = swot_xy[idx]
                d = np.linalg.norm(pts - is2_xy[i], axis=1)

                dist_min_list.append(d.astype(float))
                sigma_swot_list.append(sigf_ok[np.asarray(idx, dtype=int)].astype(float))

                if sshf_ok is not None:
                    ssh_swot_list.append(sshf_ok[np.asarray(idx, dtype=int)].astype(float))
                else:
                    ssh_swot_list.append(np.full(len(idx), np.nan, dtype=float))

            if verbose and (i + 1) % 50 == 0:
                logger.info(f"    Processed {i+1}/{is2_xy.shape[0]}")

        logger.info("  Collation complete")

        return {
            "lat": is2_lat,
            "lon": is2_lon,
            "height_ice2": np.asarray(beam_data.height_ice2, dtype=float),
            "DIST_MIN": [dist_min_list],
            "sigma_swot": [sigma_swot_list],
            "ssh_swot": [ssh_swot_list],
            "beam_name": beam_data.beam_name,
            "track_name": beam_data.track_name,
        }


def batch_load_is2_beams(filepath: str, beams: Optional[List[str]] = None) -> Dict[str, BeamData]:
    """Load multiple beams from one IS2 HDF5 file."""
    if beams is None:
        beams = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

    loader = SWOTISDataLoader()
    out: Dict[str, BeamData] = {}

    for b in beams:
        bd = loader.load_is2_track(filepath, beam=b)
        if bd is not None:
            out[b] = bd

    return out


def example_load_data():
    logging.basicConfig(level=logging.INFO)

    is2_file = "/path/to/ATL10_20230101_01234567_008_01.h5"
    swot_file = "/path/to/SWOT_L2_LR_SSH_20230101T000000_20230101T001000_V01.nc"

    loader = SWOTISDataLoader()

    print("\n" + "=" * 60)
    print("Loading IS2 Data")
    print("=" * 60)
    is2_beam = loader.load_is2_track(is2_file, beam="gt1l")
    if is2_beam is None:
        print("Failed to load IS2 data")
        return

    print(f"Loaded IS2 beam: {len(is2_beam.lat)} points")
    print(f"  Lat range: {np.nanmin(is2_beam.lat):.2f} to {np.nanmax(is2_beam.lat):.2f}")
    print(f"  Lon range: {np.nanmin(is2_beam.lon):.2f} to {np.nanmax(is2_beam.lon):.2f}")
    print(f"  Height range: {np.nanmin(is2_beam.height_ice2):.2f} to {np.nanmax(is2_beam.height_ice2):.2f} m")

    print("\n" + "=" * 60)
    print("Loading SWOT Data")
    print("=" * 60)
    swot_scene = loader.load_swot_scene(swot_file, prefer_unfiltered_ssh=True, try_groups=True)
    if swot_scene is None:
        print("Failed to load SWOT data")
        return

    print(f"Loaded SWOT scene: {swot_scene.sigma0.shape} pixels")
    print(f"  Lat range: {np.nanmin(swot_scene.lat):.2f} to {np.nanmax(swot_scene.lat):.2f}")
    print(f"  Lon range: {np.nanmin(swot_scene.lon):.2f} to {np.nanmax(swot_scene.lon):.2f}")
    print(f"  Sigma0 range: {np.nanmin(swot_scene.sigma0):.2f} to {np.nanmax(swot_scene.sigma0):.2f} dB")
    print(f"  SSH present? {'yes' if swot_scene.ssh is not None and np.isfinite(swot_scene.ssh).any() else 'no'}")
    print(f"  SSH var name: {swot_scene.ssh_name}")

    print("\n" + "=" * 60)
    print("Collating Data")
    print("=" * 60)
    beam_collated = loader.collate_is2_swot(is2_beam, swot_scene, dist_max=3000.0, verbose=True)

    if beam_collated is None:
        print("Collocation failed")
        return

    n_collocations = [len(d) for d in beam_collated["DIST_MIN"][0]]
    print("\nCollocation summary:")
    print(f"  Total IS2 points: {len(beam_collated['lat'])}")
    print(f"  Min collocations per point: {int(np.min(n_collocations))}")
    print(f"  Max collocations per point: {int(np.max(n_collocations))}")
    print(f"  Mean collocations per point: {float(np.mean(n_collocations)):.1f}")

    return beam_collated


if __name__ == "__main__":
    example_load_data()
