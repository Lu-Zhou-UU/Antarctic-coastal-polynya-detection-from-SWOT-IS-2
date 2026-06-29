#!/usr/bin/env python3
"""
Detect coastal polynyas from RACMO siconca using:
  1) Ross Sea regional mask (from lat/lon)
  2) Land-sea mask (LSM)
  3) Flooding (connected low-SIC regions touching land) within Ross Sea.

Inputs:
  - siconca_file: siconca.KNMI-2096.PXANT11.RACMO24p1v3_CESM_ssp370.DD.nc
       variable 'siconca(time, height, rlat, rlon)' in %, _FillValue=-9999
  - masks_file:   PXANT11_masks_NEW.nc
       variables 'LSM(rlat, rlon)', 'lat(rlat, rlon)', 'lon(rlat, rlon)'
"""

import argparse
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label
import matplotlib.pyplot as plt


def coastal_polynya_from_landmask(
    sic_pct_2d,
    lsm_2d,
    region_mask_2d=None,
    sic_threshold=0.7,
    sic_fill=-9999.0,
):
    """
    Compute coastal polynya mask from 2D SIC, LSM, and optional region mask.

    Parameters
    ----------
    sic_pct_2d : 2D array
        Sea-ice concentration in percent (%), with fill value for land.
    lsm_2d : 2D array
        Land-sea mask. Assumed >0 (or ~1) over land, 0 over ocean.
    region_mask_2d : 2D boolean or None
        True inside region of interest (e.g. Ross Sea). If None, whole domain.
    sic_threshold : float
        Threshold in fraction; SIC < threshold are candidate polynya cells.
    sic_fill : float
        Fill value for siconca.

    Returns
    -------
    mask_coastal : 2D boolean
        True where grid cell belongs to a coastal polynya.
    """
    sic = np.array(sic_pct_2d, dtype=float)
    lsm = np.array(lsm_2d, dtype=float)

    if region_mask_2d is None:
        region_mask = np.ones_like(sic, dtype=bool)
    else:
        region_mask = np.array(region_mask_2d, dtype=bool)

    # Land = where LSM is ~1 or nonzero; ocean = 0
    land = np.isfinite(lsm) & (lsm > 0.5)
    ocean = ~land

    # Restrict to region (Ross Sea)
    land = land & region_mask
    ocean = ocean & region_mask

    # Remove fill from SIC
    sic[sic == sic_fill] = np.nan

    # Convert % -> fraction if needed
    if np.nanmax(sic[ocean]) > 2.0:
        sic_frac = sic / 100.0
    else:
        sic_frac = sic

    # Candidate polynyas: low SIC over ocean, within region
    cand = ocean & np.isfinite(sic_frac) & (sic_frac < sic_threshold)
    if not np.any(cand):
        return np.zeros_like(cand, dtype=bool)

    # Label all low-SIC candidate regions (4-neighbour connectivity)
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)
    labeled, nlab = label(cand, structure=struct)
    if nlab == 0:
        return np.zeros_like(cand, dtype=bool)

    # Build a "land-neighbour" mask: ocean cells that touch land, all within region
    land_neigh = np.zeros_like(land, dtype=bool)
    land_neigh[1:, :] |= land[:-1, :]
    land_neigh[:-1, :] |= land[1:, :]
    land_neigh[:, 1:] |= land[:, :-1]
    land_neigh[:, :-1] |= land[:, 1:]

    # Intersection of candidate polynya cells that are adjacent to land
    seeds = cand & land_neigh
    seed_labels = np.unique(labeled[seeds])
    seed_labels = seed_labels[seed_labels > 0]  # ignore background (0)

    if seed_labels.size == 0:
        return np.zeros_like(cand, dtype=bool)

    mask_coastal = np.isin(labeled, seed_labels)
    return mask_coastal


def build_ross_sea_mask(lat, lon):
    """
    Simple Ross Sea mask from lat/lon.

    Assumes lon in degrees, may be -180..180 or 0..360.
    Uses a loose box: 160E–240E, 85S–60S (adjust as needed).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lon_mod = lon.copy()
    lon_mod[lon_mod < 0] += 360.0

    ross = (
        (lon_mod >= 160.0) & (lon_mod <= 240.0) &
        (lat <= -60.0) & (lat >= -85.0)
    )
    return ross


def main():
    ap = argparse.ArgumentParser(
        description="Detect Ross Sea coastal polynyas from siconca using LSM + flooding."
    )
    ap.add_argument("siconca_file", help="RACMO siconca NetCDF")
    ap.add_argument("masks_file", help="PXANT11_masks_NEW.nc (with LSM, lat, lon)")
    ap.add_argument("--time_index", type=int, default=208,
                    help="Time index (0-based, default 208)")
    ap.add_argument("--sic_threshold", type=float, default=0.7,
                    help="SIC threshold as fraction (default 0.7)")
    ap.add_argument("--out_png", default=None,
                    help="Optional PNG for visual check")
    args = ap.parse_args()

    # Read SIC
    with Dataset(args.siconca_file) as ds_sic:
        sic_var = ds_sic.variables["siconca"]  # (time, height, rlat, rlon)
        sic_t = sic_var[args.time_index, 0, :, :]
        sic_fill = getattr(sic_var, "_FillValue", -9999.0)

    # Read masks and lat/lon
    with Dataset(args.masks_file) as ds_mask:
        lsm = ds_mask.variables["LSM"][:]       # (rlat, rlon)
        lat = ds_mask.variables["lat"][:]       # (rlat, rlon)
        lon = ds_mask.variables["lon"][:]       # (rlat, rlon)

    # Build Ross Sea region mask
    ross_mask = build_ross_sea_mask(lat, lon)

    mask_coastal = coastal_polynya_from_landmask(
        sic_t,
        lsm,
        region_mask_2d=ross_mask,
        sic_threshold=args.sic_threshold,
        sic_fill=sic_fill,
    )

    print("siconca shape:", sic_t.shape)
    print("Number of coastal polynya cells (Ross Sea):", int(mask_coastal.sum()))

    sic_plot = np.array(sic_t, dtype=float)
    sic_plot[sic_plot == sic_fill] = np.nan

    if args.out_png:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(sic_plot / 100.0, origin="lower", vmin=0, vmax=1)
        plt.title("siconca fraction (t={})".format(args.time_index))
        plt.colorbar(im1, shrink=0.7)

        plt.subplot(1, 2, 2)
        im2 = plt.imshow(mask_coastal, origin="lower", cmap="Reds")
        plt.title("Ross Sea coastal polynya (SIC < {:.0f}%)".format(
            args.sic_threshold * 100))
        plt.colorbar(im2, shrink=0.7)

        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)
        print("Saved", args.out_png)
    else:
        plt.imshow(mask_coastal, origin="lower", cmap="Reds")
        plt.show()


if __name__ == "__main__":
    main()
