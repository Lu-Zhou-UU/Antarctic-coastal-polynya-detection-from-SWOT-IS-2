"""
plot_racmo_one_day_winds.py
─────────────────────────────────────────────────────────────────────────────
Plot RACMO2.4p1 near-surface wind speed (contourf) and wind vectors (quiver)
over Antarctica for a single calendar day on a South Polar Stereographic
projection.

Wind speed is shown on a regular lon/lat grid (contourf).  Wind vectors are
sampled on the **native RACMO curvilinear grid** (index stride) so arrow
density does not pile up at the pole.

Usage
-----
python plot_racmo_one_day_winds.py \
    --uas uas_RACMO2.4p1_ERA5_3h_ANT27_2010_2020.nc \
    --vas vas_RACMO2.4p1_ERA5_3h_ANT27_2010_2020.nc \
    --date 2015-10-15 \
    --output racmo_winds_20151015.png \
    [--cmap speed] [--arrow-color "#7EC8E3"] \
    [--quiver-scale 380] [--quiver-stride 16] [--lat-north -50]
"""

from __future__ import annotations

import argparse
import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset, num2date
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


# ─── colour-map helpers ──────────────────────────────────────────────────────

def _make_speed_cmap(name: str) -> mcolors.Colormap:
    """Return a sequential colormap for wind speed."""
    if name == "speed":
        try:
            import cmocean
            return cmocean.cm.speed
        except ImportError:
            pass
        return plt.cm.YlOrRd
    try:
        return plt.get_cmap(name)
    except ValueError:
        return plt.cm.YlOrRd


# ─── NetCDF helpers ──────────────────────────────────────────────────────────

def _day_indices(nc: Dataset, target_date: datetime.date) -> np.ndarray:
    """Return boolean mask of all time steps on *target_date*."""
    time_var = nc.variables["time"]
    dates = num2date(time_var[:], units=time_var.units,
                     calendar=getattr(time_var, "calendar", "standard"))
    mask = np.array(
        [d.year == target_date.year
         and d.month == target_date.month
         and d.day == target_date.day
         for d in dates],
        dtype=bool,
    )
    if not mask.any():
        raise ValueError(
            f"No time steps found for {target_date} in {nc.filepath()}.\n"
            f"File covers {dates[0].year}-{dates[0].month:02d}-{dates[0].day:02d} "
            f"to {dates[-1].year}-{dates[-1].month:02d}-{dates[-1].day:02d}."
        )
    return mask


def _squeeze_to_grid(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Return a 2-D field aligned with lon/lat (drops time/height singletons)."""
    out = np.squeeze(np.asarray(field, dtype=float))
    if out.ndim != 2:
        raise ValueError(
            f"Expected 2-D wind field matching lon/lat {lon.shape}, got shape {out.shape}"
        )
    if out.shape != lon.shape:
        raise ValueError(
            f"Wind field shape {out.shape} does not match lon/lat shape {lon.shape}"
        )
    return out


def load_day_wind(
    uas_path: str,
    vas_path: str,
    target_date: datetime.date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and daily-average RACMO near-surface winds for *target_date*.

    Returns
    -------
    lon, lat : 2-D arrays  (degrees)
    uas_mean, vas_mean : daily-mean eastward / northward wind (m s⁻¹)
    """
    with Dataset(uas_path) as nc_u, Dataset(vas_path) as nc_v:
        lon = nc_u.variables["lon"][:]
        lat = nc_u.variables["lat"][:]

        mask_u = _day_indices(nc_u, target_date)
        mask_v = _day_indices(nc_v, target_date)

        n_u = mask_u.sum()
        n_v = mask_v.sum()
        print(f"  Found {n_u} uas / {n_v} vas time steps on {target_date}")

        uas = _squeeze_to_grid(
            nc_u.variables["uas"][mask_u].mean(axis=0), lon, lat,
        )
        vas = _squeeze_to_grid(
            nc_v.variables["vas"][mask_v].mean(axis=0), lon, lat,
        )
        lon = np.array(lon)
        lat = np.array(lat)

    return lon, lat, uas, vas


# ─── regridding ──────────────────────────────────────────────────────────────

def _regular_lon_lat_mesh(
    lon: np.ndarray,
    lat: np.ndarray,
    nlon: int = 360,
    nlat: int = 200,
    lat_min: float = -90.0,
    lat_max: float = -50.0,
) -> tuple[np.ndarray, np.ndarray]:
    # Force full −180→180 coverage so no gap appears at the antimeridian
    reg_lon = np.linspace(-180.0, 180.0, nlon)
    reg_lat = np.linspace(lat_min, lat_max, nlat)
    return np.meshgrid(reg_lon, reg_lat)


def _native_lat_min(lat: np.ndarray, field: np.ndarray) -> float:
    valid = np.isfinite(field)
    if not valid.any():
        return -90.0
    return float(lat[valid].min())


def _pole_lat_cutoff(lat: np.ndarray, field: np.ndarray, buffer_deg: float) -> float:
    """Southernmost latitude for displaying regridded fields and vectors."""
    return _native_lat_min(lat, field) + buffer_deg


def _interp_curvilinear(
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    reg_lon2d: np.ndarray,
    reg_lat2d: np.ndarray,
    *,
    pole_buffer_deg: float = 2.0,
) -> np.ndarray:
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    val_flat = field.ravel()
    valid = np.isfinite(val_flat)

    # Duplicate source points shifted ±360° so the Delaunay triangulation
    # covers the periodic boundary and there is no gap at ±180°.
    lon_w = np.concatenate([lon_flat[valid] - 360,
                            lon_flat[valid],
                            lon_flat[valid] + 360])
    lat_w = np.concatenate([lat_flat[valid],
                            lat_flat[valid],
                            lat_flat[valid]])
    val_w = np.concatenate([val_flat[valid],
                            val_flat[valid],
                            val_flat[valid]])

    interp = LinearNDInterpolator(np.column_stack([lon_w, lat_w]), val_w)
    result = interp(reg_lon2d, reg_lat2d)

    # Do not plot where interpolation failed (outside native-grid coverage).
    result[~np.isfinite(result)] = np.nan

    # Mask the pole hole and a buffer where Delaunay fill is not trustworthy.
    pole_cutoff = _pole_lat_cutoff(lat, field, pole_buffer_deg)
    result[reg_lat2d < pole_cutoff] = np.nan

    return result


def _apply_native_coverage_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    reg_lon2d: np.ndarray,
    reg_lat2d: np.ndarray,
    values: np.ndarray,
    *,
    max_dist_deg: float,
) -> np.ndarray:
    """Blank values on the display grid that are far from any native RACMO node."""
    valid = np.isfinite(u)
    src_lon = lon[valid].ravel()
    src_lat = lat[valid].ravel()
    src_pts = np.column_stack([
        np.concatenate([src_lon, src_lon - 360, src_lon + 360]),
        np.concatenate([src_lat, src_lat, src_lat]),
    ])
    tree = cKDTree(src_pts)
    q_pts = np.column_stack([reg_lon2d.ravel(), reg_lat2d.ravel()])
    dist = tree.query(q_pts, k=1)[0].reshape(reg_lon2d.shape)
    out = values.copy()
    out[dist > max_dist_deg] = np.nan
    return out


def _prepare_speed_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    uas: np.ndarray,
    vas: np.ndarray,
    *,
    lat_north: float = -50.0,
    pole_buffer_deg: float = 2.0,
    coverage_max_deg: float = 0.9,
) -> dict:
    """Regrid wind speed to regular lon/lat for contourf."""
    reg_lon2d, reg_lat2d = _regular_lon_lat_mesh(lon, lat, lat_max=lat_north)
    uas_reg = _interp_curvilinear(
        lon, lat, uas, reg_lon2d, reg_lat2d, pole_buffer_deg=pole_buffer_deg,
    )
    vas_reg = _interp_curvilinear(
        lon, lat, vas, reg_lon2d, reg_lat2d, pole_buffer_deg=pole_buffer_deg,
    )
    speed_reg = np.sqrt(uas_reg**2 + vas_reg**2)
    speed_reg[~(np.isfinite(uas_reg) & np.isfinite(vas_reg))] = np.nan
    speed_reg = _apply_native_coverage_mask(
        lon, lat, uas, reg_lon2d, reg_lat2d, speed_reg,
        max_dist_deg=coverage_max_deg,
    )
    pole_cutoff = _pole_lat_cutoff(lat, uas, pole_buffer_deg)

    return dict(
        lon2d=reg_lon2d, lat2d=reg_lat2d,
        speed=speed_reg,
        pole_cutoff=pole_cutoff,
    )


def _native_quiver_points(
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    stride: int,
    lat_north: float,
    quiver_lat_min: float | None = None,
    pole_cutoff: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Subsample wind vectors on the native RACMO grid (i/j stride).

    Uses geographic lon/lat at native grid nodes — not a regular lon/lat mesh —
    so spacing stays roughly uniform in the model domain and avoids pole crowding.
    """
    stride = max(1, stride)
    off = stride // 2

    lon_q = lon[off::stride, off::stride]
    lat_q = lat[off::stride, off::stride]
    u_q = u[off::stride, off::stride]
    v_q = v[off::stride, off::stride]

    if quiver_lat_min is None:
        quiver_lat_min = (
            pole_cutoff if pole_cutoff is not None
            else _pole_lat_cutoff(lat, u, 2.0)
        )

    show = (
        np.isfinite(lon_q) & np.isfinite(lat_q)
        & np.isfinite(u_q) & np.isfinite(v_q)
        & (lat_q <= lat_north)
        & (lat_q >= quiver_lat_min)
    )

    return (
        lon_q[show], lat_q[show], u_q[show], v_q[show],
        float(quiver_lat_min),
    )


# ─── plotting ────────────────────────────────────────────────────────────────

def _plot_wind_vectors(
    ax: plt.Axes,
    grid: dict,
    arrow_color: str = "#7EC8E3",
    quiver_scale: float = 380,
) -> None:
    ax.quiver(
        grid["q_lon"], grid["q_lat"],
        grid["q_u"],   grid["q_v"],
        transform=ccrs.PlateCarree(),
        color=arrow_color,
        scale=quiver_scale,
        scale_units="width",
        width=0.0030,
        headwidth=3.2,
        headlength=3.8,
        headaxislength=3.2,
        alpha=0.92,
        zorder=4,
    )


def _plot_coast_and_shelves(ax: plt.Axes) -> None:
    """Draw prominent coast / ice-shelf outlines on top of the wind field."""
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=1.0, edgecolor="0.05", zorder=6,
    )
    try:
        ice = cfeature.NaturalEarthFeature(
            "physical", "antarctic_ice_shelves_polys", "10m",
            facecolor="none", edgecolor="0.05", linewidth=0.85,
        )
        ax.add_feature(ice, zorder=6)
    except Exception:
        pass


def plot_wind_one_day(
    uas_path: str,
    vas_path: str,
    target_date: datetime.date,
    output_path: str = "racmo_winds_oneday.png",
    cmap_name: str = "speed",
    arrow_color: str = "#7EC8E3",
    quiver_scale: float = 380,
    quiver_stride: int = 6,
    lat_north: float = -50.0,
    quiver_lat_min: float | None = None,
    pole_buffer_deg: float = 2.0,
    coverage_max_deg: float = 0.9,
    dpi: int = 200,
) -> None:
    """Plot daily-mean wind speed and vectors for *target_date*."""

    print(f"Loading RACMO winds for {target_date} …")
    lon, lat, uas, vas = load_day_wind(uas_path, vas_path, target_date)

    print("Regridding wind speed to regular lon/lat …")
    grid = _prepare_speed_grid(
        lon, lat, uas, vas,
        lat_north=lat_north,
        pole_buffer_deg=pole_buffer_deg,
        coverage_max_deg=coverage_max_deg,
    )

    q_lon, q_lat, q_u, q_v, q_lat_min = _native_quiver_points(
        lon, lat, uas, vas,
        stride=quiver_stride,
        lat_north=lat_north,
        quiver_lat_min=quiver_lat_min,
        pole_cutoff=grid["pole_cutoff"],
    )
    grid["q_lon"] = q_lon
    grid["q_lat"] = q_lat
    grid["q_u"] = q_u
    grid["q_v"] = q_v
    print(
        f"  Quiver points: {q_lon.size} on native RACMO grid "
        f"(stride={quiver_stride}, lat ≥ {q_lat_min:.1f}°; same pole cutoff as contourf)"
    )

    # ── figure / map ────────────────────────────────────────────────────────
    proj = ccrs.SouthPolarStereo(central_longitude=0)
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(projection=proj))
    fig.patch.set_facecolor("white")
    ax.set_extent([-180, 180, -90, lat_north], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.spines["geo"].set_visible(False)
    nodata_color = "#ffffff"

    # ── wind speed (filled contours + colorbar) ─────────────────────────────
    levels = np.linspace(0, 20, 21)
    cmap = _make_speed_cmap(cmap_name)
    cmap.set_bad(color=nodata_color)

    speed_plot = np.ma.masked_invalid(grid["speed"])
    cf = ax.contourf(
        grid["lon2d"], grid["lat2d"], speed_plot,
        levels=levels, cmap=cmap, extend="max",
        transform=ccrs.PlateCarree(), zorder=2,
        antialiased=True,
    )

    # ── wind vectors (under coastlines) ───────────────────────────────────────
    _plot_wind_vectors(ax, grid, arrow_color=arrow_color, quiver_scale=quiver_scale)

    # ── coast + ice shelves on top ────────────────────────────────────────────
    _plot_coast_and_shelves(ax)

    # ── gridlines ────────────────────────────────────────────────────────────
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=0.35, color="0.65", alpha=0.55, linestyle=":",
    )
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(range(-80, -49, 10))

    for lat_line in [-80, -70, -60]:
        ax.text(
            178, lat_line, f"{abs(lat_line)}°S",
            transform=ccrs.PlateCarree(),
            fontsize=7, color="0.45", ha="left", va="center",
        )

    # ── colourbar ────────────────────────────────────────────────────────────
    cb = fig.colorbar(
        cf, ax=ax, orientation="horizontal",
        pad=0.05, fraction=0.05, shrink=0.82, aspect=32,
        ticks=np.arange(0, 21, 2),
    )
    cb.set_label("10-m wind speed (m s$^{-1}$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"RACMO2.4p1 near-surface winds\n"
        f"{target_date.strftime('%d %B %Y')} (daily mean)",
        fontsize=11, pad=10,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot RACMO near-surface wind speed + vectors for one day."
    )
    p.add_argument("--uas", required=True,
                   help="Path to RACMO uas (eastward wind) NetCDF file.")
    p.add_argument("--vas", required=True,
                   help="Path to RACMO vas (northward wind) NetCDF file.")
    p.add_argument("--date", required=True,
                   help="Target date in YYYY-MM-DD format (e.g. 2015-10-15).")
    p.add_argument("--output", default=None,
                   help="Output image path. Defaults to racmo_winds_YYYYMMDD.png.")
    p.add_argument("--cmap", default="speed")
    p.add_argument("--arrow-color", default="#7EC8E3",
                   help="Quiver arrow colour (default: light blue).")
    p.add_argument("--quiver-scale", type=float, default=380)
    p.add_argument("--quiver-stride", type=int, default=18,
                   help="Native-grid quiver subsample stride in cells (default: 18).")
    p.add_argument("--lat-north", type=float, default=-50.0,
                   help="Northern map limit in degrees south (default: -50).")
    p.add_argument("--quiver-lat-min", type=float, default=None,
                   help="Do not draw arrows south of this latitude (default: pole cutoff + 2°).")
    p.add_argument("--pole-buffer-deg", type=float, default=2.0,
                   help="Pole buffer (°) applied to both contourf and vectors (default: 2).")
    p.add_argument("--coverage-max-deg", type=float, default=0.9,
                   help="Max distance (°) from a native RACMO node to draw contourf (default: 0.9).")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    target_date = datetime.date.fromisoformat(args.date)
    if target_date.month != 10:
        print(f"Warning: requested date {target_date} is not in October.")

    output = args.output or f"racmo_winds_{target_date.strftime('%Y%m%d')}.png"

    plot_wind_one_day(
        uas_path=args.uas,
        vas_path=args.vas,
        target_date=target_date,
        output_path=output,
        cmap_name=args.cmap,
        arrow_color=args.arrow_color,
        quiver_scale=args.quiver_scale,
        quiver_stride=args.quiver_stride,
        lat_north=args.lat_north,
        quiver_lat_min=args.quiver_lat_min,
        pole_buffer_deg=args.pole_buffer_deg,
        coverage_max_deg=args.coverage_max_deg,
        dpi=args.dpi,
    )
