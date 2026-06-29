#!/usr/bin/env python3
"""
Manuscript Figure 3: 2x2 SWOT–IS-2 reference + SIT for two representative dates.

Main figure (Andy revision):
  (a,b) 14 Oct: three latitude insets (fragmented polynya mask).
  (c,d) 28 Oct: full swath; grey beams, IS-2 reference, SWOT SSHA; SIT 0–2 m.

Supplementary:
  SY — six-beam coloured elevations (legacy style).
  SX — full SIT range including coastal outliers > 2 m.

Example:
  python plot_figure3_SWOT_IS2_manuscript.py \\
    --npz-oct14 results/is2_polynya_sit/20231014T205222_SWOT_20231014222007_IS2_is2_polynya_sit_outputs.npz \\
    --npz-oct28 results/is2_polynya_sit/20231028T215103_SWOT_20231028223812_IS2_is2_polynya_sit_outputs.npz \\
    --out-main SWOT_IS2_2013_oct_ice_thickness.png \\
    --out-supp-beams SuppFig_SY_SWOT_IS2_beams.png \\
    --out-supp-sit-full SuppFig_SX_SWOT_IS2_SIT_fullrange.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import plot_sit_from_npz as ps


BEAM_ORDER = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
BEAM_PALETTE = [
    "#852D86", "#B94B77", "#D37854", "#DEA961", "#D5D386", "#A4D9A6",
]
# Colourblind-friendly palette (Wong / ColorBrewer inspired)
COLOR_REF = "#222222"          # IS-2 sea-surface reference
COLOR_SSHA = "#0072B2"         # SWOT SSHA
COLOR_SIT = "#B2182B"          # SIT 2 km mean (deep red)
COLOR_SIT_PT = "#D6604D"       # SIT pointwise (salmon-red)
COLOR_BEAM_GREY = "#8A8A8A"    # IS-2 ATL07 six-beam cloud
COLOR_MASK = "#EBEBEB"


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def _lat_extent(scene, pad_deg=0.04):
    """Tight latitude limits from polynya-masked points."""
    m = scene.mask_final & np.isfinite(scene.lat)
    if m.sum() < 2:
        m = np.isfinite(scene.lat)
    if m.sum() < 2:
        return None
    lo, hi = float(np.nanmin(scene.lat[m])), float(np.nanmax(scene.lat[m]))
    pad = max(float(pad_deg), 0.01 * abs(hi - lo))
    return lo - pad, hi + pad


def _fragment_spans(scene, args):
    """Merged polynya latitude intervals (for fragmented scenes)."""
    spans = _spans(scene, args)
    if not spans:
        xlim = _lat_extent(scene, pad_deg=0.02)
        return [xlim] if xlim else []
    return sorted(spans, key=lambda s: s[0])


def _span_xlim(lo, hi, pad_deg=0.012):
    width = abs(hi - lo)
    pad = max(float(pad_deg), 0.12 * width)
    return lo - pad, hi + pad


def _fragment_width_ratios(fragments, pad_deg=0.012):
    return [max(_span_xlim(lo, hi, pad_deg)[1] - _span_xlim(lo, hi, pad_deg)[0], 0.02)
            for lo, hi in fragments]


def _draw_axis_breaks(fig, axes, *, size=0.007):
    """Diagonal break marks between disjoint latitude segments."""
    for ax_l, ax_r in zip(axes[:-1], axes[1:]):
        bb_l, bb_r = ax_l.get_position(), ax_r.get_position()
        x = 0.5 * (bb_l.x1 + bb_r.x0)
        for y in (bb_l.y0 + 0.012, bb_l.y1 - 0.012):
            for dx, dy in ((-size, size), (size, size)):
                fig.add_artist(Line2D(
                    [x + dx, x], [y - dy, y],
                    transform=fig.transFigure, color="#333333", lw=1.0, clip_on=False,
                ))


def _scene_in_lat_window(scene, lat_lo, lat_hi, pad_deg=0.012):
    """Subset scene to one latitude fragment (for inset panels)."""
    m = (
        np.isfinite(scene.lat)
        & (scene.lat >= lat_lo - pad_deg)
        & (scene.lat <= lat_hi + pad_deg)
    )
    return SimpleNamespace(
        beam=scene.beam[m],
        lat=scene.lat[m],
        height=scene.height[m],
        ref_height=scene.ref_height[m],
        sit=scene.sit[m],
        s_km=scene.s_km[m],
        mask_final=scene.mask_final[m],
        ssha=scene.ssha[m] if scene.ssha is not None else None,
        seg_joint=scene.seg_joint[m] if scene.seg_joint is not None else None,
    )


def _panel_tag(ax, letter):
    ax.text(
        0.03, 0.97, letter,
        transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="top", ha="left", zorder=30, color="#222222",
    )


def _clean_spines(ax, *, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, linestyle=":", linewidth=0.5, color="#d8d8d8", zorder=0)
        ax.set_axisbelow(True)


def _load_scene(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    beam = ps._as_str_array(z["beam"])
    lat = z["lat"].astype(float)
    height = z["height"].astype(float)
    ref_height = z["ref_height"].astype(float)
    sit = z["sit"].astype(float)
    if "s_km_beam" in z.files:
        s_km = z["s_km_beam"].astype(float)
    elif "t_km_joint" in z.files:
        s_km = z["t_km_joint"].astype(float)
    else:
        s_km = np.full_like(lat, np.nan)
    seg_joint = z["seg_joint"].astype(int) if "seg_joint" in z.files else None
    masks = ps._get_masks_from_npz(z, n=len(lat))
    mask_final = masks.get("final", masks[list(masks.keys())[0]])
    ssha_key = ps._detect_ssha_key(z.files)
    ssha = np.asarray(z[ssha_key], float).reshape(-1) if ssha_key else None
    return SimpleNamespace(
        z=z, beam=beam, lat=lat, height=height, ref_height=ref_height, sit=sit,
        s_km=s_km, seg_joint=seg_joint, mask_final=mask_final, ssha_key=ssha_key, ssha=ssha,
    )


def _spans(scene, args):
    order_key = scene.s_km if np.isfinite(scene.s_km).sum() >= 10 else None
    return ps._polynya_lat_spans_per_beam(
        beam=scene.beam, lat=scene.lat, in_pol=scene.mask_final,
        order_key=order_key, gap_lat_deg=args.gap_lat_deg,
        pad_merge=args.shade_merge_pad_deg,
    )


def _plot_ref_line(ax, scene, args, *, mask_only=False):
    m = np.isfinite(scene.lat) & np.isfinite(scene.ref_height) & np.isfinite(scene.s_km)
    if mask_only:
        m &= scene.mask_final
    if m.sum() < args.min_per_bin:
        return
    latm, refm, _, idxm = ps.running_mean_std_by_distance(
        scene.lat[m], scene.ref_height[m], scene.s_km[m],
        window_km=args.run_window_km, step_km=args.run_step_km,
        min_per_win=max(3, args.min_per_bin),
    )
    if latm.size < 2:
        return
    x, y = ps.break_line_on_gaps(latm, refm, idx=idxm, gap_lat_deg=args.gap_lat_deg)
    ps._plot_with_white_edge(
        ax, x, y, color=COLOR_REF, lw=2.2, ls="-", alpha=1.0, zorder=8, edge_lw=2.6,
    )


def _plot_ssha_combined(ax, scene, args, *, mask_only=False, sparse=False):
    """Dashed SSHA: 2 km running mean on left beams (gt1l/gt2l/gt3l); sparse mode for short fragments."""
    if scene.ssha is None or scene.ssha.size != scene.lat.size:
        return False
    y = scene.ssha.astype(float)
    left_beams = ["gt1l", "gt2l", "gt3l"]
    m = np.isfinite(scene.lat) & np.isfinite(y) & np.isfinite(scene.s_km)
    m &= np.isin(scene.beam, left_beams)
    if mask_only:
        m &= scene.mask_final
    min_pts = 3 if sparse else args.min_per_bin
    if m.sum() < min_pts:
        return False

    if sparse:
        ax.scatter(
            scene.lat[m], y[m], s=12, alpha=0.5, color=COLOR_SSHA,
            edgecolors="none", zorder=8, rasterized=True,
        )

    win_km = 0.5 if sparse else args.run_window_km
    step_km = 0.2 if sparse else args.run_step_km
    latm, ym, _, idxm = ps.running_mean_std_by_distance(
        scene.lat[m], y[m], scene.s_km[m],
        window_km=win_km, step_km=step_km,
        min_per_win=min_pts,
    )
    ok = np.isfinite(latm) & np.isfinite(ym)
    if ok.sum() >= 2:
        latm, ym, idxm = latm[ok], ym[ok], idxm[ok]
        gap = max(0.01, args.gap_lat_deg * (0.5 if sparse else 1.0))
        x, yline = ps.break_line_on_gaps(latm, ym, idx=idxm, gap_lat_deg=gap)
        ax.plot(
            x, yline, color=COLOR_SSHA, lw=2.2, ls=(0, (6, 4)),
            solid_capstyle="round", zorder=9,
        )
        return True
    return sparse and m.sum() > 0


def _sit_scatter_style(n_pts):
    n = max(int(n_pts), 1)
    if n > 800:
        return 6.0, 0.22
    if n > 200:
        return 10.0, 0.38
    return 16.0, 0.62


def _beam_scatter_style(n_pts, *, colored=False, base_alpha=0.28, base_s=14):
    """Adaptive marker size/alpha so dense swaths do not turn into grey mud."""
    if colored:
        return base_alpha, base_s
    n = max(int(n_pts), 1)
    alpha = float(np.clip(1200.0 / n, 0.06, base_alpha))
    size = float(np.clip(9000.0 / n, 4.0, base_s))
    return alpha, size


def _plot_beam_scatter(ax, scene, *, colored=False, mask_only=True, alpha=0.28, s=14):
    m_base = np.isfinite(scene.lat) & np.isfinite(scene.height)
    if mask_only:
        m_base &= scene.mask_final
    n_pts = int(m_base.sum())
    alpha, s = _beam_scatter_style(n_pts, colored=colored, base_alpha=alpha, base_s=s)
    beams = [b for b in BEAM_ORDER if np.any(scene.beam == b)]
    beams += [b for b in np.unique(scene.beam) if b not in beams]
    for i, b in enumerate(beams):
        m = m_base & (scene.beam == b)
        if m.sum() == 0:
            continue
        c = BEAM_PALETTE[i % len(BEAM_PALETTE)] if colored else COLOR_BEAM_GREY
        ax.scatter(
            scene.lat[m], scene.height[m], s=s, alpha=alpha, color=c,
            edgecolors="none", zorder=2, rasterized=True,
        )


def _plot_sit_panel(ax, scene, args, *, ylim02=True, colored_beams=False, show_scatter=True):
    m = np.isfinite(scene.lat) & np.isfinite(scene.sit) & np.isfinite(scene.s_km) & scene.mask_final
    if colored_beams:
        beams = [b for b in BEAM_ORDER if np.any(scene.beam == b)]
        for i, b in enumerate(beams):
            mb = m & (scene.beam == b)
            if mb.sum() == 0:
                continue
            c = BEAM_PALETTE[i % len(BEAM_PALETTE)]
            sa, ss = _sit_scatter_style(mb.sum())
            ax.scatter(scene.lat[mb], scene.sit[mb], s=max(ss, 8), alpha=min(sa + 0.15, 0.6),
                         color=c, edgecolors="none", zorder=3, rasterized=True)
            latm, sitm, _, idxm = ps.running_mean_std_by_distance(
                scene.lat[mb], scene.sit[mb], scene.s_km[mb],
                window_km=args.run_window_km, step_km=args.run_step_km,
                min_per_win=max(3, args.min_per_bin),
            )
            if latm.size >= 2:
                x, y = ps.break_line_on_gaps(latm, sitm, idx=idxm, gap_lat_deg=args.gap_lat_deg)
                ps._plot_with_white_edge(ax, x, y, color=c, lw=2.2, ls="-", alpha=0.75, zorder=5, edge_lw=2.0)
    else:
        if show_scatter and m.sum() > 0:
            s, alpha = _sit_scatter_style(m.sum())
            ax.scatter(
                scene.lat[m], scene.sit[m], s=s, alpha=alpha,
                color=COLOR_SIT_PT, edgecolors="none", zorder=3, rasterized=True,
            )
        latm, sitm, _, idxm = ps.running_mean_std_by_distance(
            scene.lat[m], scene.sit[m], scene.s_km[m],
            window_km=args.run_window_km, step_km=args.run_step_km,
            min_per_win=max(3, args.min_per_bin),
        )
        if latm.size >= 2:
            x, y = ps.break_line_on_gaps(latm, sitm, idx=idxm, gap_lat_deg=args.gap_lat_deg)
            ps._plot_with_white_edge(ax, x, y, color=COLOR_SIT, lw=2.4, ls="-", alpha=1.0, zorder=6, edge_lw=2.2)

    if ylim02:
        ax.set_ylim(0, 2)
    else:
        ymax = float(np.nanpercentile(scene.sit[m], 99)) if m.sum() else 3.0
        ax.set_ylim(-0.05, max(2.5, ymax * 1.05))


def _style_elev_axis(ax):
    ax.set_ylabel("Elevation (m)", labelpad=6)
    ax.set_ylim(-0.4, 0.1)
    _clean_spines(ax, grid=True)


def _style_ssha_axis(ax_ssha):
    """Hide twin-axis ticks/spines; SSHA scale is drawn in a dedicated column."""
    if ax_ssha is None or not ax_ssha.get_visible():
        return
    ax_ssha.tick_params(axis="y", right=False, labelright=False)
    for sp in ("top", "bottom", "left", "right"):
        ax_ssha.spines[sp].set_visible(False)


def _split_elev_cell(gs_cell):
    """Elevation cell = [plot area | SSHA scale strip]."""
    return gs_cell.subgridspec(1, 2, width_ratios=[1.0, 0.075], wspace=0.06)


def _add_ssha_scale_axis(fig, gs_scale, ssha_ylim):
    """Dedicated right-hand SSHA axis (shared scale, no overlap with SIT column)."""
    ax = fig.add_subplot(gs_scale)
    if ssha_ylim is not None:
        ax.set_ylim(ssha_ylim)
    ax.set_ylabel("SWOT SSHA (m)", color=COLOR_SSHA, fontsize=10, labelpad=2)
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", colors=COLOR_SSHA, labelcolor=COLOR_SSHA, labelsize=8.5, length=3)
    ax.set_xticks([])
    ax.patch.set_visible(False)
    for sp in ("top", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(COLOR_SSHA)
    ax.spines["right"].set_linewidth(0.9)
    return ax


def _style_sit_axis(ax, *, ylim02=True):
    ax.set_ylabel("SIT (m)", labelpad=8)
    if ylim02:
        ax.set_ylim(0, 2)
    _clean_spines(ax, grid=True)


def _add_spans(ax, spans):
    ps._add_span_layer(ax, spans, face=COLOR_MASK, edge="none", alpha=0.9, lw=0, zorder=0)


def _ssha_limits(scenes):
    """Shared SSHA y-limits for both dates (left beams, polynya mask)."""
    left = ["gt1l", "gt2l", "gt3l"]
    vals = []
    for scene in scenes:
        m = scene.mask_final & np.isfinite(scene.ssha) & np.isin(scene.beam, left)
        if m.sum() >= 5:
            vals.append(scene.ssha[m])
    if not vals:
        return None
    y = np.concatenate(vals)
    lo, hi = np.nanpercentile(y, [1, 99])
    pad = max(0.015, 0.12 * (hi - lo))
    return lo - pad, hi + pad


def _plot_elev_on_ax(
    ax, scene, args, *, ax_ssha=None, ssha_ylim=None,
    show_ylabel=True, show_elev_ticks=True, sparse_ssha=False,
):
    ax.set_facecolor(COLOR_MASK)
    _plot_beam_scatter(ax, scene, colored=False, mask_only=True)
    _plot_ref_line(ax, scene, args, mask_only=True)
    has_ssha = False
    if ax_ssha is not None:
        has_ssha = _plot_ssha_combined(
            ax_ssha, scene, args, mask_only=True, sparse=sparse_ssha,
        )
        if has_ssha and ssha_ylim is not None:
            ax_ssha.set_ylim(ssha_ylim)
        elif not has_ssha:
            ax_ssha.set_visible(False)
    _style_elev_axis(ax)
    if has_ssha:
        _style_ssha_axis(ax_ssha)
    if not show_ylabel:
        ax.set_ylabel("")
    if not show_elev_ticks:
        ax.tick_params(labelleft=False)
    return has_ssha


def _plot_sit_on_ax(ax, scene, args, *, show_ylabel=True, show_scatter=True, show_yticks=True):
    ax.set_facecolor(COLOR_MASK)
    _plot_sit_panel(ax, scene, args, ylim02=True, colored_beams=False, show_scatter=show_scatter)
    _style_sit_axis(ax, ylim02=True)
    if not show_ylabel:
        ax.set_ylabel("")
    if not show_yticks:
        ax.tick_params(labelleft=False)


def _style_fragment_xaxis(ax, xlim, *, show_labels=True):
    ax.set_xlim(xlim)
    width = xlim[1] - xlim[0]
    nbins = 2 if width < 0.05 else 3
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(
        axis="x", labelbottom=show_labels, labelsize=8,
        pad=2, length=3,
    )


def _plot_fragment_row(
    fig, gs_e_plot, gs_s, scene, args, fragments, panel_letters, ssha_ylim, date_lbl,
):
    """Top row: broken latitude axis (panel width ∝ fragment length)."""
    n = len(fragments)
    ratios = _fragment_width_ratios(fragments)
    sub_e = gs_e_plot.subgridspec(1, n, width_ratios=ratios, wspace=0.05)
    sub_s = gs_s.subgridspec(1, n, width_ratios=ratios, wspace=0.05)
    elev_axes, sit_axes, ssha_axes = [], [], []

    for i, (lo, hi) in enumerate(fragments):
        frag = _scene_in_lat_window(scene, lo, hi)
        xlim = _span_xlim(lo, hi)

        ax_e = fig.add_subplot(sub_e[0, i])
        ax_s = fig.add_subplot(sub_s[0, i])
        ax_ssha = ax_e.twinx()

        has_ssha = _plot_elev_on_ax(
            ax_e, frag, args, ax_ssha=ax_ssha, ssha_ylim=ssha_ylim,
            show_ylabel=(i == 0), show_elev_ticks=(i == 0), sparse_ssha=True,
        )
        _style_fragment_xaxis(ax_e, xlim, show_labels=True)
        ax_e.set_ylim(-0.4, 0.1)
        elev_axes.append(ax_e)
        if has_ssha:
            ssha_axes.append(ax_ssha)

        _plot_sit_on_ax(
            ax_s, frag, args, show_ylabel=(i == 0), show_scatter=True,
            show_yticks=(i == 0),
        )
        _style_fragment_xaxis(ax_s, xlim, show_labels=True)
        ax_s.set_ylim(0, 2)
        sit_axes.append(ax_s)

    _draw_axis_breaks(fig, elev_axes)
    _draw_axis_breaks(fig, sit_axes)

    bbox_e = elev_axes[0].get_position()
    bbox_s = sit_axes[-1].get_position()
    cx = 0.5 * (bbox_e.x0 + bbox_s.x1)
    fig.text(cx, bbox_e.y1 + 0.018, date_lbl, ha="center", va="bottom", fontsize=11)
    # fig.text(cx, bbox_e.y0 - 0.045, "Latitude (disjoint polynya segments)",
            #  ha="center", va="top", fontsize=9.5, style="italic", color="#444444")
    return elev_axes, sit_axes, ssha_axes


def _main_legend(fig, ref_ax):
    """Legend tucked just below the bottom row of panels."""
    handles = [
        Patch(facecolor=COLOR_MASK, edgecolor="none", alpha=0.9, label="Polynya mask"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_BEAM_GREY,
                markersize=6, linestyle="None", label="IS-2 ATL07"),
        Line2D([0], [0], color=COLOR_REF, linewidth=2.2, label="IS-2 $h_{\\mathrm{ref}}$"),
        Line2D([0], [0], color=COLOR_SSHA, linewidth=2.0, linestyle=(0, (5, 3)), label="SWOT SSHA"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SIT_PT,
                markersize=5, alpha=0.85, linestyle="None", label="SIT (pointwise)"),
        Line2D([0], [0], color=COLOR_SIT, linewidth=2.4, label="SIT (2 km mean)"),
    ]
    bb = ref_ax.get_position()
    fig.legend(
        handles=handles, loc="upper center", ncol=6, frameon=False,
        columnspacing=1.3, handletextpad=0.45,
        bbox_to_anchor=(0.5, bb.y0 - 0.07),
    )


def _plot_supp_broken_elev(fig, gs_cell, scene, args, ssha_ylim, title):
    """Coloured-beam elevation for a fragmented scene (supplement SY, top row)."""
    fragments = _fragment_spans(scene, args)
    if len(fragments) < 2:
        ax = fig.add_subplot(gs_cell)
        _add_spans(ax, _spans(scene, args))
        _plot_beam_scatter(ax, scene, colored=True, mask_only=True, alpha=0.4, s=18)
        _plot_ref_line(ax, scene, args, mask_only=True)
        ax_ssha = ax.twinx()
        _plot_ssha_combined(ax_ssha, scene, args, mask_only=True, sparse=True)
        if ssha_ylim is not None:
            ax_ssha.set_ylim(ssha_ylim)
        _style_elev_axis(ax)
        _style_ssha_axis(ax_ssha)
        xlim = _lat_extent(scene, pad_deg=0.03)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_title(title, fontsize=11, pad=6)
        return [ax]

    ratios = _fragment_width_ratios(fragments)
    sub = gs_cell.subgridspec(1, len(fragments), width_ratios=ratios, wspace=0.05)
    axes = []
    ssha_ok = []
    for lo, hi in fragments:
        frag = _scene_in_lat_window(scene, lo, hi)
        left = ["gt1l", "gt2l", "gt3l"]
        m = (
            frag.mask_final & np.isfinite(frag.ssha) & np.isfinite(frag.s_km)
            & np.isin(frag.beam, left)
        )
        ssha_ok.append(m.sum() >= 3)
    last_ssha_i = max((i for i, ok in enumerate(ssha_ok) if ok), default=len(fragments) - 1)

    for i, (lo, hi) in enumerate(fragments):
        frag = _scene_in_lat_window(scene, lo, hi)
        ax = fig.add_subplot(sub[0, i])
        _plot_beam_scatter(ax, frag, colored=True, mask_only=True, alpha=0.5, s=22)
        _plot_ref_line(ax, frag, args, mask_only=True)
        ax_ssha = ax.twinx()
        has_ssha = _plot_ssha_combined(ax_ssha, frag, args, mask_only=True, sparse=True)
        if has_ssha and ssha_ylim is not None:
            ax_ssha.set_ylim(ssha_ylim)
        _style_elev_axis(ax)
        if has_ssha:
            _style_ssha_axis(ax_ssha)
        _style_fragment_xaxis(ax, _span_xlim(lo, hi))
        ax.set_ylim(-0.4, 0.1)
        if i == 0:
            ax.set_ylabel("Elevation (m)")
        axes.append(ax)
    _draw_axis_breaks(fig, axes)
    bb = axes[0].get_position()
    fig.text(0.5 * (bb.x0 + axes[-1].get_position().x1), bb.y1 + 0.02,
             title, ha="center", va="bottom", fontsize=11)
    # fig.text(0.5 * (bb.x0 + axes[-1].get_position().x1), bb.y0 - 0.05,
    #          "Latitude (disjoint segments)", ha="center", va="top", fontsize=9, style="italic", color="#444")
    return axes


def _plot_supp_broken_sit(fig, gs_cell, scene, args, title, *, ylim02=False):
    """Per-beam SIT for a fragmented scene (supplement SX, top row)."""
    fragments = _fragment_spans(scene, args)
    if len(fragments) < 2:
        ax = fig.add_subplot(gs_cell)
        _add_spans(ax, _spans(scene, args))
        _plot_sit_panel(ax, scene, args, ylim02=ylim02, colored_beams=True, show_scatter=True)
        ax.set_ylabel("Sea ice thickness (m)")
        xlim = _lat_extent(scene, pad_deg=0.03)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_title(title, fontsize=11, pad=6)
        _clean_spines(ax, grid=True)
        return [ax]

    ratios = _fragment_width_ratios(fragments)
    sub = gs_cell.subgridspec(1, len(fragments), width_ratios=ratios, wspace=0.05)
    axes = []
    for i, (lo, hi) in enumerate(fragments):
        frag = _scene_in_lat_window(scene, lo, hi)
        ax = fig.add_subplot(sub[0, i])
        _plot_sit_panel(ax, frag, args, ylim02=ylim02, colored_beams=True, show_scatter=True)
        _style_fragment_xaxis(ax, _span_xlim(lo, hi))
        if i == 0:
            ax.set_ylabel("Sea ice thickness (m)")
        axes.append(ax)
    _draw_axis_breaks(fig, axes)
    bb = axes[0].get_position()
    fig.text(0.5 * (bb.x0 + axes[-1].get_position().x1), bb.y1 + 0.02,
             title, ha="center", va="bottom", fontsize=11)
    # fig.text(0.5 * (bb.x0 + axes[-1].get_position().x1), bb.y0 - 0.05,
    #          "Latitude (disjoint segments)", ha="center", va="top", fontsize=9, style="italic", color="#444")
    return axes


def plot_main_two_scenes(scenes, labels, args):
    """
    Layout: (a) elev Oct14  (b) SIT Oct14
            (c) elev Oct28  (d) SIT Oct28
    """
    _setup_style()
    fig = plt.figure(figsize=(12.4, 7.6), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, height_ratios=[0.88, 1.0], width_ratios=[1.08, 0.92],
        hspace=0.32, wspace=0.34, left=0.11, right=0.92, top=0.91, bottom=0.17,
    )
    panel_letters = [["a", "b"], ["c", "d"]]
    ssha_ylim = _ssha_limits(scenes)
    ssha_axes = []

    # Row 0 (14 Oct): three latitude insets per column
    fragments = _fragment_spans(scenes[0], args)
    if len(fragments) >= 2:
        gs_e_split = _split_elev_cell(gs[0, 0])
        elev_frag, _, frag_ssha = _plot_fragment_row(
            fig, gs_e_split[0, 0], gs[0, 1], scenes[0], args, fragments,
            panel_letters[0], ssha_ylim, labels[0],
        )
        ssha_axes.extend(frag_ssha)
        if ssha_ylim is not None:
            _add_ssha_scale_axis(fig, gs_e_split[0, 1], ssha_ylim)
    else:
        # fallback: single panel if not fragmented
        gs_e_split = _split_elev_cell(gs[0, 0])
        ax_e = fig.add_subplot(gs_e_split[0, 0])
        ax_ssha = ax_e.twinx()
        has_ssha = _plot_elev_on_ax(ax_e, scenes[0], args, ax_ssha=ax_ssha, ssha_ylim=ssha_ylim)
        if has_ssha:
            ssha_axes.append(ax_ssha)
        if ssha_ylim is not None:
            _add_ssha_scale_axis(fig, gs_e_split[0, 1], ssha_ylim)
        ax_s = fig.add_subplot(gs[0, 1])
        _plot_sit_on_ax(ax_s, scenes[0], args, show_scatter=True)
        xlim = _lat_extent(scenes[0], pad_deg=0.025)
        if xlim:
            ax_e.set_xlim(xlim)
            ax_s.set_xlim(xlim)
        ax_e.tick_params(labelbottom=False)
        ax_s.tick_params(labelbottom=False)
        bbox_e = ax_e.get_position()
        bbox_s = ax_s.get_position()
        fig.text(0.5 * (bbox_e.x0 + bbox_s.x1), bbox_e.y1 + 0.02, labels[0],
                 ha="center", va="bottom", fontsize=11)

    # Row 1 (28 Oct): full continuous swath
    scene = scenes[1]
    date_lbl = labels[1]
    xlim = _lat_extent(scene, pad_deg=0.04)

    gs_e_split = _split_elev_cell(gs[1, 0])
    ax_e = fig.add_subplot(gs_e_split[0, 0])
    ax_ssha = ax_e.twinx()
    ax_e.set_facecolor("white")
    _add_spans(ax_e, _spans(scene, args))
    has_ssha = _plot_elev_on_ax(
        ax_e, scene, args, ax_ssha=ax_ssha, ssha_ylim=ssha_ylim,
    )
    if has_ssha:
        ssha_axes.append(ax_ssha)
    if ssha_ylim is not None:
        _add_ssha_scale_axis(fig, gs_e_split[0, 1], ssha_ylim)
    ax_s = fig.add_subplot(gs[1, 1], sharex=ax_e)
    ax_s.set_facecolor("white")
    _add_spans(ax_s, _spans(scene, args))
    _plot_sit_on_ax(ax_s, scene, args, show_scatter=True)

    if xlim is not None:
        ax_e.set_xlim(xlim)
        ax_s.set_xlim(xlim)
    ax_e.set_xlabel("Latitude", labelpad=4)
    ax_s.set_xlabel("Latitude", labelpad=4)
    bbox_e = ax_e.get_position()
    bbox_s = ax_s.get_position()
    fig.text(0.5 * (bbox_e.x0 + bbox_s.x1), bbox_e.y1 + 0.018, date_lbl,
             ha="center", va="bottom", fontsize=11)

    _main_legend(fig, ax_s)
    return fig


def _beam_legend(fig, ref_ax):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BEAM_PALETTE[i],
               markersize=7, linestyle="None", label=b)
        for i, b in enumerate(BEAM_ORDER)
    ]
    handles += [
        Line2D([0], [0], color=COLOR_REF, linewidth=2.0, label="IS-2 $h_{\\mathrm{ref}}$"),
        Line2D([0], [0], color=COLOR_SSHA, linewidth=1.8, linestyle=(0, (5, 3)), label="SWOT SSHA"),
    ]
    bb = ref_ax.get_position()
    fig.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
        columnspacing=1.2, handletextpad=0.4, bbox_to_anchor=(0.5, bb.y0 - 0.06),
    )


def plot_supp_beams(scenes, labels, args):
    """SY: six-beam elevations; 14 Oct uses broken-axis fragments."""
    _setup_style()
    fig = plt.figure(figsize=(12.2, 6.8), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.40, 0.60], hspace=0.30,
        left=0.10, right=0.92, top=0.96, bottom=0.16,
    )
    ssha_ylim = _ssha_limits(scenes)
    _plot_supp_broken_elev(fig, gs[0], scenes[0], args, ssha_ylim, labels[0])

    ax = fig.add_subplot(gs[1])
    scene = scenes[1]
    _add_spans(ax, _spans(scene, args))
    _plot_beam_scatter(ax, scene, colored=True, mask_only=True, alpha=0.35, s=14)
    _plot_ref_line(ax, scene, args, mask_only=True)
    ax_ssha = ax.twinx()
    _plot_ssha_combined(ax_ssha, scene, args, mask_only=True)
    if ssha_ylim is not None:
        ax_ssha.set_ylim(ssha_ylim)
    _style_elev_axis(ax)
    _style_ssha_axis(ax_ssha)
    xlim = _lat_extent(scene, pad_deg=0.04)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Latitude", labelpad=4)
    ax.set_title(labels[1], fontsize=11, pad=6)
    _beam_legend(fig, ax)
    return fig


def plot_supp_sit_full(scenes, labels, args):
    """SX: per-beam SIT full range; 14 Oct broken-axis; point + 2 km mean."""
    _setup_style()
    fig = plt.figure(figsize=(12.2, 6.8), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.40, 0.60], hspace=0.30,
        left=0.10, right=0.92, top=0.96, bottom=0.16,
    )
    _plot_supp_broken_sit(fig, gs[0], scenes[0], args, labels[0], ylim02=False)

    ax = fig.add_subplot(gs[1])
    scene = scenes[1]
    _add_spans(ax, _spans(scene, args))
    _plot_sit_panel(ax, scene, args, ylim02=False, colored_beams=True, show_scatter=True)
    ax.set_ylabel("Sea ice thickness (m)")
    xlim = _lat_extent(scene, pad_deg=0.04)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Latitude", labelpad=4)
    ax.set_title(labels[1], fontsize=11, pad=6)
    _clean_spines(ax, grid=True)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BEAM_PALETTE[i],
               markersize=6, linestyle="None", label=b)
        for i, b in enumerate(BEAM_ORDER)
    ]
    handles.append(Line2D([0], [0], color="#333333", linewidth=2.0, label="2 km beam mean"))
    bb = ax.get_position()
    fig.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
        columnspacing=1.2, bbox_to_anchor=(0.5, bb.y0 - 0.06),
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description="Manuscript Fig. 3: SWOT–IS-2 reference + SIT (Andy layout).")
    ap.add_argument("--npz-oct14", required=True)
    ap.add_argument("--npz-oct28", required=True)
    ap.add_argument("--out-main", default="SWOT_IS2_2013_oct_ice_thickness.png")
    ap.add_argument("--out-supp-beams", default="SuppFig_SY_SWOT_IS2_beams.png")
    ap.add_argument("--out-supp-sit-full", default="SuppFig_SX_SWOT_IS2_SIT_fullrange.png")
    ap.add_argument("--run_window_km", type=float, default=2.0)
    ap.add_argument("--run_step_km", type=float, default=0.5)
    ap.add_argument("--min_per_bin", type=int, default=5)
    ap.add_argument("--gap_lat_deg", type=float, default=0.03)
    ap.add_argument("--shade_merge_pad_deg", type=float, default=0.002)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    _setup_style()
    scenes = [_load_scene(args.npz_oct14), _load_scene(args.npz_oct28)]
    labels = ["14 October 2023", "28 October 2023"]

    fig = plot_main_two_scenes(scenes, labels, args)
    fig.savefig(args.out_main, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"[INFO] Wrote main figure: {args.out_main}")
    plt.close(fig)

    fig_b = plot_supp_beams(scenes, labels, args)
    fig_b.savefig(args.out_supp_beams, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"[INFO] Wrote supplementary beams: {args.out_supp_beams}")
    plt.close(fig_b)

    fig_s = plot_supp_sit_full(scenes, labels, args)
    fig_s.savefig(args.out_supp_sit_full, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"[INFO] Wrote supplementary SIT full range: {args.out_supp_sit_full}")
    plt.close(fig_s)


if __name__ == "__main__":
    main()
