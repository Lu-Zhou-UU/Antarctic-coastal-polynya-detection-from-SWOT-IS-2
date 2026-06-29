"""
plot_racmo_antarctic_winds.py
─────────────────────────────────────────────────────────────────────────────
Plot RACMO2.4p1 near-surface wind speed (contourf) and wind vectors (quiver)
over Antarctica on a South Polar Stereographic projection.

Usage
-----
python plot_racmo_antarctic_winds.py \
    --uas uas_RACMO2.4p1_ERA5_3h_ANT27_2010_2020.nc \
    --vas vas_RACMO2.4p1_ERA5_3h_ANT27_2010_2020.nc \
    --output racmo_winds_antarctica.png \
    [--year-start 2010] [--year-end 2020] \
    [--month-start 1] [--month-end 12] \
    [--cmap speed] [--arrow-color 0.15] \
    [--quiver-scale 380] [--quiver-stride 6]
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset, num2date
from scipy.interpolate import LinearNDInterpolator


# ─── colour-map helpers ──────────────────────────────────────────────────────

def _make_speed_cmap(name: str) -> mcolors.Colormap:
    """Return a sequential colormap for wind speed.

    Tries *cmocean* 'speed' first; falls back gracefully to the requested
    matplotlib name or to 'YlOrRd'.
    """
    if name == "speed":
        try:
            import cmocean
            return cmocean.cm.speed
        except ImportError:
            pass
        # Fallback if cmocean is absent
        return plt.cm.YlOrRd
    try:
        return plt.get_cmap(name)
    except ValueError:
        return plt.cm.YlOrRd


# ─── NetCDF helpers ──────────────────────────────────────────────────────────

def _time_indices(
    nc: Dataset,
    year_start: int,
    year_end: int,
    month_start: int,
    month_end: int,
) -> np.ndarray:
    """Return boolean mask of time steps inside the requested window."""
    time_var = nc.variables["time"]
    dates = num2date(time_var[:], units=time_var.units,
                     calendar=getattr(time_var, "calendar", "standard"))
    # Convert cftime → Python datetime for easy comparison
    mask = np.array([
        (year_start <= d.year <= year_end) and (month_start <= d.month <= month_end)
        for d in dates
    ], dtype=bool)
    if not mask.any():
        raise ValueError(
            f"No time steps found for {year_start}–{year_end}, "
            f"months {month_start}–{month_end}."
        )
    return mask


def load_mean_wind(
    uas_path: str,
    vas_path: str,
    year_start: int = 2010,
    year_end: int = 2020,
    month_start: int = 1,
    month_end: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and time-average RACMO near-surface winds.

    Returns
    -------
    lon, lat : 2-D arrays of grid-cell longitudes / latitudes  (degrees)
    uas_mean, vas_mean : time-mean eastward / northward wind (m s⁻¹)
    """
    with Dataset(uas_path) as nc_u, Dataset(vas_path) as nc_v:
        lon = nc_u.variables["lon"][:]
        lat = nc_u.variables["lat"][:]

        mask_u = _time_indices(nc_u, year_start, year_end, month_start, month_end)
        mask_v = _time_indices(nc_v, year_start, year_end, month_start, month_end)

        uas = nc_u.variables["uas"][mask_u].mean(axis=0)
        vas = nc_v.variables["vas"][mask_v].mean(axis=0)

    return lon, lat, uas, vas


# ─── regridding ──────────────────────────────────────────────────────────────

def _regular_lon_lat_mesh(
    lon: np.ndarray,
    lat: np.ndarray,
    nlon: int = 360,
    nlat: int = 180,
    lat_min: float = -90.0,
    lat_max: float = -55.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular lon/lat mesh covering the Southern Ocean."""
    lon_min, lon_max = float(lon.min()), float(lon.max())
    reg_lon = np.linspace(lon_min, lon_max, nlon)
    reg_lat = np.linspace(lat_min, lat_max, nlat)
    return np.meshgrid(reg_lon, reg_lat)


def _interp_curvilinear(
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    reg_lon2d: np.ndarray,
    reg_lat2d: np.ndarray,
) -> np.ndarray:
    """Interpolate a curvilinear field onto a regular lon/lat grid."""
    points = np.column_stack([lon.ravel(), lat.ravel()])
    values = field.ravel()
    valid = np.isfinite(values)
    interp = LinearNDInterpolator(points[valid], values[valid])
    return interp(reg_lon2d, reg_lat2d)


def _prepare_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    uas: np.ndarray,
    vas: np.ndarray,
    stride: int = 6,
) -> dict:
    """Regrid winds and subsample for quiver arrows."""
    reg_lon2d, reg_lat2d = _regular_lon_lat_mesh(lon, lat)
    uas_reg = _interp_curvilinear(lon, lat, uas, reg_lon2d, reg_lat2d)
    vas_reg = _interp_curvilinear(lon, lat, vas, reg_lon2d, reg_lat2d)
    speed_reg = np.sqrt(uas_reg**2 + vas_reg**2)

    # Subsample for quiver (avoid overcrowded arrows)
    ql = reg_lon2d[::stride, ::stride]
    qt = reg_lat2d[::stride, ::stride]
    qu = uas_reg[::stride, ::stride]
    qv = vas_reg[::stride, ::stride]

    return dict(
        lon2d=reg_lon2d,
        lat2d=reg_lat2d,
        speed=speed_reg,
        uas=uas_reg,
        vas=vas_reg,
        q_lon=ql,
        q_lat=qt,
        q_u=qu,
        q_v=qv,
    )


# ─── plotting ────────────────────────────────────────────────────────────────

def _plot_wind_vectors(
    ax: plt.Axes,
    grid: dict,
    arrow_color: str = "0.15",
    quiver_scale: float = 380,
    quiver_key_speed: float = 10.0,
) -> None:
    """Overlay quiver arrows and a reference key on *ax*."""
    qv = ax.quiver(
        grid["q_lon"],
        grid["q_lat"],
        grid["q_u"],
        grid["q_v"],
        transform=ccrs.PlateCarree(),
        color=arrow_color,
        scale=quiver_scale,
        scale_units="width",
        width=0.0022,
        headwidth=3.5,
        headlength=4,
        headaxislength=3.5,
        alpha=0.85,
        zorder=5,
    )
    ax.quiverkey(
        qv,
        X=0.88, Y=0.06,
        U=quiver_key_speed,
        label=f"{quiver_key_speed:.0f} m s$^{{-1}}$",
        labelpos="E",
        fontproperties=dict(size=8),
        color=arrow_color,
        zorder=6,
    )


def plot_wind_circulation(
    uas_path: str,
    vas_path: str,
    output_path: str = "racmo_winds_antarctica.png",
    year_start: int = 2010,
    year_end: int = 2020,
    month_start: int = 1,
    month_end: int = 12,
    cmap_name: str = "speed",
    arrow_color: str = "0.15",
    quiver_scale: float = 380,
    quiver_stride: int = 6,
    dpi: int = 200,
) -> None:
    """Main plotting routine."""

    # ── load & process ──────────────────────────────────────────────────────
    print("Loading RACMO winds …")
    lon, lat, uas, vas = load_mean_wind(
        uas_path, vas_path,
        year_start=year_start, year_end=year_end,
        month_start=month_start, month_end=month_end,
    )

    print("Regridding to regular lon/lat …")
    grid = _prepare_grid(lon, lat, uas, vas, stride=quiver_stride)

    # ── figure / map ────────────────────────────────────────────────────────
    proj = ccrs.SouthPolarStereo(central_longitude=0)
    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw=dict(projection=proj),
    )
    ax.set_extent([-180, 180, -90, -55], crs=ccrs.PlateCarree())

    # Ocean: light blue so low-speed areas are distinguishable from land
    ax.set_facecolor("#c8dff0")

    # ── wind speed contourf ──────────────────────────────────────────────────
    levels = np.linspace(0, 20, 41)          # 0–20 m s⁻¹, 0.5 m s⁻¹ steps
    cmap = _make_speed_cmap(cmap_name)

    cf = ax.contourf(
        grid["lon2d"],
        grid["lat2d"],
        grid["speed"],
        levels=levels,
        cmap=cmap,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # Thin contour lines every 5 m s⁻¹ for readability
    cs = ax.contour(
        grid["lon2d"],
        grid["lat2d"],
        grid["speed"],
        levels=np.arange(5, 21, 5),
        colors="0.30",
        linewidths=0.4,
        alpha=0.55,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.clabel(cs, fmt="%g", fontsize=6, inline=True, inline_spacing=2)

    # ── wind vectors ─────────────────────────────────────────────────────────
    _plot_wind_vectors(
        ax, grid,
        arrow_color=arrow_color,
        quiver_scale=quiver_scale,
    )

    # ── geographic features ──────────────────────────────────────────────────
    ax.add_feature(cfeature.LAND,
                   facecolor="#e8e0d0", edgecolor="0.35", linewidth=0.5,
                   zorder=4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.25", zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.50", zorder=4)
    # Ice shelves (Natural Earth "Antarctic ice shelves" if available)
    try:
        ice = cfeature.NaturalEarthFeature(
            "physical", "antarctic_ice_shelves_polys", "10m",
            facecolor="#ddeeff", edgecolor="0.50", linewidth=0.4,
        )
        ax.add_feature(ice, zorder=4)
    except Exception:
        pass

    # ── gridlines ────────────────────────────────────────────────────────────
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="gray",
        alpha=0.45,
        linestyle="--",
    )
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(range(-80, -54, 10))

    # Latitude labels manually (cartopy south-polar label support is limited)
    for lat_line in [-80, -70, -60]:
        ax.text(
            180, lat_line, f"{lat_line}°",
            transform=ccrs.PlateCarree(),
            fontsize=6, color="0.40", ha="left", va="center",
        )

    # ── colourbar ────────────────────────────────────────────────────────────
    cb = fig.colorbar(
        cf,
        ax=ax,
        orientation="horizontal",
        pad=0.04,
        fraction=0.046,
        shrink=0.75,
        aspect=30,
    )
    cb.set_label("10-m wind speed (m s$^{-1}$)", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks(np.arange(0, 21, 5))

    # ── title ─────────────────────────────────────────────────────────────────
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    period = (
        f"{month_names[month_start]}–{month_names[month_end]} "
        f"{year_start}–{year_end}"
        if not (month_start == 1 and month_end == 12)
        else f"{year_start}–{year_end} annual mean"
    )
    ax.set_title(
        f"RACMO2.4p1 near-surface winds\n{period}",
        fontsize=10,
        pad=8,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot RACMO near-surface wind speed + vectors over Antarctica."
    )
    p.add_argument("--uas", required=True,
                   help="Path to RACMO uas (eastward wind) NetCDF file.")
    p.add_argument("--vas", required=True,
                   help="Path to RACMO vas (northward wind) NetCDF file.")
    p.add_argument("--output", default="racmo_winds_antarctica.png",
                   help="Output image path.")
    p.add_argument("--year-start", type=int, default=2010)
    p.add_argument("--year-end",   type=int, default=2020)
    p.add_argument("--month-start", type=int, default=1,
                   help="First month of averaging window (1=Jan).")
    p.add_argument("--month-end",   type=int, default=12,
                   help="Last month of averaging window (12=Dec).")
    p.add_argument("--cmap", default="speed",
                   help="Colormap name for wind speed. 'speed' uses cmocean.speed "
                        "(falls back to YlOrRd). Any matplotlib name also works.")
    p.add_argument("--arrow-color", default="0.15",
                   help="Quiver arrow colour (matplotlib colour string).")
    p.add_argument("--quiver-scale", type=float, default=380,
                   help="Quiver scale parameter (larger = shorter arrows).")
    p.add_argument("--quiver-stride", type=int, default=6,
                   help="Subsample stride for quiver arrows.")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_wind_circulation(
        uas_path=args.uas,
        vas_path=args.vas,
        output_path=args.output,
        year_start=args.year_start,
        year_end=args.year_end,
        month_start=args.month_start,
        month_end=args.month_end,
        cmap_name=args.cmap,
        arrow_color=args.arrow_color,
        quiver_scale=args.quiver_scale,
        quiver_stride=args.quiver_stride,
        dpi=args.dpi,
    )
