#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe


def _as_str_array(a):
    a = np.asarray(a)
    if a.dtype.kind in ("U", "S"):
        return a.astype(str)
    return np.array([str(x) for x in a], dtype=str)


def _plot_with_white_edge(
    ax, x, y, *,
    color="k", lw=2.5, ls="-", alpha=1.0, zorder=5,
    edge_lw=2.5, edge_alpha=0.95
):
    """
    Plot a line with a white outline (halo) so it stands out over dense scatter.
    edge_lw is the EXTRA width added on top of lw.
    """
    (ln,) = ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    ln.set_path_effects([
        pe.Stroke(linewidth=lw + edge_lw, foreground="white", alpha=edge_alpha),
        pe.Normal(),
    ])
    return ln


def running_mean_std_by_distance(lat, y, s_km, window_km=2.0, step_km=0.5, min_per_win=5):
    """
    Running mean (and std) of y in a sliding distance window along s_km.
    Returns lat_c, y_c, std_c, idx (consecutive index for break_line_on_gaps)
    """
    lat = np.asarray(lat, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    s = np.asarray(s_km, float).reshape(-1)

    ok = np.isfinite(lat) & np.isfinite(y) & np.isfinite(s)
    if ok.sum() < int(min_per_win):
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=int)

    lat = lat[ok]
    y = y[ok]
    s = s[ok]

    ii = np.argsort(s)
    lat, y, s = lat[ii], y[ii], s[ii]

    w = float(window_km)
    step = float(step_km)
    if w <= 0 or step <= 0:
        raise ValueError("window_km and step_km must be positive.")

    half = 0.5 * w
    smin, smax = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=int)

    centers = np.arange(smin + half, smax - half + 1e-9, step)
    if centers.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=int)

    csum_y = np.concatenate(([0.0], np.cumsum(y)))
    csum_y2 = np.concatenate(([0.0], np.cumsum(y * y)))
    csum_lat = np.concatenate(([0.0], np.cumsum(lat)))

    lat_c = np.full(centers.shape, np.nan, dtype=float)
    y_c = np.full(centers.shape, np.nan, dtype=float)
    std_c = np.full(centers.shape, np.nan, dtype=float)
    idx = np.arange(centers.size, dtype=int)

    for k, c in enumerate(centers):
        left = np.searchsorted(s, c - half, side="left")
        right = np.searchsorted(s, c + half, side="right")
        n = right - left
        if n < int(min_per_win):
            continue

        sum_y = csum_y[right] - csum_y[left]
        sum_y2 = csum_y2[right] - csum_y2[left]
        sum_lat = csum_lat[right] - csum_lat[left]

        mean_y = sum_y / n
        var_y = max(0.0, (sum_y2 / n) - mean_y * mean_y)

        lat_c[k] = sum_lat / n
        y_c[k] = mean_y
        std_c[k] = np.sqrt(var_y)

    return lat_c, y_c, std_c, idx


def bin_means_by_distance(lat, y, s_km, bin_km=2.0, min_per_bin=5):
    lat = np.asarray(lat, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    s_km = np.asarray(s_km, dtype=float).reshape(-1)

    ok = np.isfinite(lat) & np.isfinite(y) & np.isfinite(s_km)
    if ok.sum() == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int)

    lat = lat[ok]
    y = y[ok]
    s_km = s_km[ok]

    bin_km = float(bin_km)
    if not np.isfinite(bin_km) or bin_km <= 0:
        raise ValueError(f"bin_km must be positive; got {bin_km}")

    b = np.floor(s_km / bin_km).astype(int)
    bins = np.unique(b)

    lat_bin, y_bin, b_out = [], [], []
    for bi in np.sort(bins):
        m = (b == bi)
        if m.sum() < int(min_per_bin):
            continue
        lat_bin.append(np.nanmean(lat[m]))
        y_bin.append(np.nanmean(y[m]))
        b_out.append(int(bi))

    return (
        np.asarray(lat_bin, dtype=float),
        np.asarray(y_bin, dtype=float),
        np.asarray(b_out, dtype=int),
    )


def segment_means(lat, y, seg_id, min_per_seg=1):
    lat = np.asarray(lat, float)
    y = np.asarray(y, float)
    seg_id = np.asarray(seg_id)

    ok = np.isfinite(lat) & np.isfinite(y) & np.isfinite(seg_id)
    if ok.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    lat = lat[ok]
    y = y[ok]
    seg = seg_id[ok].astype(int)

    segs = np.unique(seg)
    xs, ys, ss = [], [], []
    for s in np.sort(segs):
        m = seg == s
        if m.sum() < int(min_per_seg):
            continue
        xs.append(np.nanmean(lat[m]))
        ys.append(np.nanmedian(y[m]))
        ss.append(s)

    return np.asarray(xs), np.asarray(ys), np.asarray(ss, dtype=int)


def break_line_on_gaps(x, y, idx=None, gap_lat_deg=0.03):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if idx is not None:
        idx = np.asarray(idx, int)

    xb, yb = [], []
    for i in range(len(x)):
        if i > 0:
            bad = False

            if not (np.isfinite(x[i]) and np.isfinite(x[i - 1]) and np.isfinite(y[i]) and np.isfinite(y[i - 1])):
                bad = True

            if np.isfinite(x[i]) and np.isfinite(x[i - 1]) and abs(x[i] - x[i - 1]) > float(gap_lat_deg):
                bad = True

            if idx is not None and (idx[i] - idx[i - 1]) != 1:
                bad = True

            if bad:
                xb.append(np.nan)
                yb.append(np.nan)

        xb.append(x[i])
        yb.append(y[i])

    return np.asarray(xb), np.asarray(yb)


def _merge_spans(spans, pad=0.0):
    spans = [(float(a), float(b)) for a, b in spans if np.isfinite(a) and np.isfinite(b)]
    if not spans:
        return []
    spans = [(min(a, b), max(a, b)) for a, b in spans]
    spans.sort(key=lambda t: t[0])

    merged = []
    cur_a, cur_b = spans[0]
    for a, b in spans[1:]:
        if a <= cur_b + float(pad):
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return merged


def _polynya_lat_spans_per_beam(beam, lat, in_pol, order_key=None, gap_lat_deg=0.03, pad_merge=0.0):
    beam = _as_str_array(beam)
    lat = np.asarray(lat, float)
    in_pol = np.asarray(in_pol, bool)

    spans = []
    for b in np.unique(beam):
        m = (beam == b) & np.isfinite(lat)
        if m.sum() < 2:
            continue

        x = lat[m]
        flag = in_pol[m]

        if order_key is not None:
            ok_order = np.isfinite(order_key[m])
            if ok_order.sum() >= 2:
                o = order_key[m].astype(float)
                ii = np.argsort(o)
            else:
                ii = np.argsort(x)
        else:
            ii = np.argsort(x)

        x = x[ii]
        flag = flag[ii]

        start = None
        for i in range(len(x)):
            if not flag[i]:
                if start is not None and i - start >= 1:
                    spans.append((np.nanmin(x[start:i]), np.nanmax(x[start:i])))
                start = None
                continue

            if start is None:
                start = i
                continue

            if np.isfinite(x[i]) and np.isfinite(x[i - 1]) and abs(x[i] - x[i - 1]) > float(gap_lat_deg):
                if i - start >= 1:
                    spans.append((np.nanmin(x[start:i]), np.nanmax(x[start:i])))
                start = i

        if start is not None and len(x) - start >= 1:
            spans.append((np.nanmin(x[start:]), np.nanmax(x[start:])))

    return _merge_spans(spans, pad=float(pad_merge))


def _add_span_layer(ax, spans, *, face="none", edge="0.4", alpha=0.18, lw=1.2, ls="-", zorder=0):
    if not spans:
        return
    for x0, x1 in spans:
        ax.axvspan(
            x0, x1,
            facecolor=face,
            edgecolor=edge,
            alpha=float(alpha),
            linewidth=float(lw),
            linestyle=str(ls),
            zorder=zorder,
        )


def _get_masks_from_npz(z, n):
    swot = np.asarray(z["in_polynya_swot"], bool) if "in_polynya_swot" in z.files else None
    amsr = np.asarray(z["in_polynya_amsr"], bool) if "in_polynya_amsr" in z.files else None
    stored_final = np.asarray(z["in_polynya"], bool) if "in_polynya" in z.files else None

    def _fix(m, name):
        if m is None:
            return None
        m = np.asarray(m, bool).reshape(-1)
        if m.size != n:
            raise RuntimeError(f"Mask size mismatch for {name}: expected {n}, got {m.size}")
        return m

    swot = _fix(swot, "in_polynya_swot")
    amsr = _fix(amsr, "in_polynya_amsr")
    stored_final = _fix(stored_final, "in_polynya")

    both = None
    union = None
    if swot is not None and amsr is not None:
        both = swot & amsr
        union = swot | amsr
    elif swot is not None:
        union = swot.copy()
    elif amsr is not None:
        union = amsr.copy()

    if union is not None:
        final = union.copy()
    elif stored_final is not None:
        final = stored_final.copy()
    else:
        final = np.ones(n, dtype=bool)

    return {"swot": swot, "amsr": amsr, "both": both, "union": union, "final": final, "stored_final": stored_final}


def _detect_ssha_key(npz_files):
    candidates = [
        "swot_ssh_mean", "swot_ssha_mean",
        "swot_ssh", "swot_ssha",
        "ssha", "SSHA",
        "swot_ssha", "SWOT_SSHA",
        "ssha_swot", "SSHA_SWOT",
        "karin_ssha", "KaRIn_SSHA",
        "ssh_anom", "ssh_anomaly",
        "ssha_joint", "ssha_neigh",
        "ssh", "SSH",
    ]
    for k in candidates:
        if k in npz_files:
            return k
    return None


def _guess_ssha_std_key(z_files, mean_key):
    if mean_key is None:
        return None
    if mean_key.endswith("_mean"):
        k = mean_key.replace("_mean", "_std")
        if k in z_files:
            return k
    candidates = [
        "swot_ssh_std", "swot_ssha_std",
        "ssha_std", "ssh_std", "SWOT_SSHA_STD",
        "ssha_sigma", "ssh_sigma",
    ]
    for k in candidates:
        if k in z_files:
            return k
    return None


def _print_manuscript_statistics(
    lat, height, ref_height, sit, s_km, beam, mask_polynya, mask_sit,
    ssha_key=None, z=None, bin_km=2.0, min_per_bin=5
):
    """
    Calculate and print statistics needed for manuscript text.
    """
    print("\n" + "=" * 80)
    print("MANUSCRIPT STATISTICS")
    print("=" * 80)
    
    # 1. ATL07 elevation statistics within polynya mask
    m_polynya = mask_polynya & np.isfinite(lat) & np.isfinite(height)
    if m_polynya.sum() > 0:
        h_polynya = height[m_polynya]
        h_mean = np.nanmean(h_polynya)
        h_median = np.nanmedian(h_polynya)
        h_q25 = np.nanpercentile(h_polynya, 25)
        h_q75 = np.nanpercentile(h_polynya, 75)
        print(f"\n1. ATL07 elevations within polynya mask ({m_polynya.sum():,} points):")
        print(f"   Mean:     {h_mean:.3f} m")
        print(f"   Median:   {h_median:.3f} m")
        print(f"   IQR:      {h_q25:.3f} -- {h_q75:.3f} m")
        
        # Beam-to-beam spread within 2 km bins
        lat_pol = lat[m_polynya]
        h_pol = height[m_polynya]
        s_pol = s_km[m_polynya]
        beam_pol = beam[m_polynya]
        
        # Bin by distance
        if np.isfinite(s_pol).sum() >= min_per_bin:
            s_min, s_max = np.nanmin(s_pol), np.nanmax(s_pol)
            bins = np.arange(s_min, s_max + bin_km, bin_km)
            bin_spreads = []
            for i in range(len(bins) - 1):
                in_bin = (s_pol >= bins[i]) & (s_pol < bins[i + 1])
                if in_bin.sum() >= min_per_bin:
                    h_bin = h_pol[in_bin]
                    if len(np.unique(beam_pol[in_bin])) >= 2:  # Need at least 2 beams
                        bin_spread = np.nanmax(h_bin) - np.nanmin(h_bin)
                        bin_spreads.append(bin_spread)
            if len(bin_spreads) > 0:
                spread_median = np.median(bin_spreads)
                print(f"   Beam-to-beam spread (within {bin_km} km bins): {spread_median:.3f} m (median)")
    else:
        print("\n1. ATL07 elevations: No data within polynya mask")
    
    # 2. Reference mismatch with SWOT SSHA
    if ssha_key and z is not None and ssha_key in z.files:
        ssha = np.asarray(z[ssha_key], float).reshape(-1)
        if ssha.size == lat.size:
            m_ref_ssha = np.isfinite(lat) & np.isfinite(ref_height) & np.isfinite(ssha) & mask_polynya
            if m_ref_ssha.sum() > 0:
                delta_h = ref_height[m_ref_ssha] - ssha[m_ref_ssha]
                delta_mean = np.nanmean(delta_h)
                delta_median = np.nanmedian(delta_h)
                print(f"\n2. Reference mismatch (IS-2 ref - SWOT SSHA) ({m_ref_ssha.sum():,} points):")
                print(f"   Mean:   {delta_mean:.3f} m")
                print(f"   Median: {delta_median:.3f} m")
            else:
                print("\n2. Reference mismatch: No valid data")
        else:
            print(f"\n2. Reference mismatch: SSHA size mismatch ({ssha.size} vs {lat.size})")
    else:
        print("\n2. Reference mismatch: SSHA not available")
    
    # 3. Fraction of 2 km bins with valid reference
    if np.isfinite(s_km).sum() >= min_per_bin:
        s_min, s_max = np.nanmin(s_km), np.nanmax(s_km)
        bins = np.arange(s_min, s_max + bin_km, bin_km)
        n_bins_total = len(bins) - 1
        n_bins_with_ref = 0
        for i in range(len(bins) - 1):
            in_bin = (s_km >= bins[i]) & (s_km < bins[i + 1])
            if in_bin.sum() >= min_per_bin:
                has_ref = np.any(np.isfinite(ref_height[in_bin]) & mask_polynya[in_bin])
                if has_ref:
                    n_bins_with_ref += 1
        if n_bins_total > 0:
            frac_ref = 100.0 * n_bins_with_ref / n_bins_total
            print(f"\n3. Fraction of {bin_km} km bins with valid reference: {frac_ref:.1f}% ({n_bins_with_ref}/{n_bins_total})")
    else:
        print("\n3. Fraction of bins with reference: Insufficient distance data")
    
    # 4. SIT statistics within polynya mask
    m_sit = mask_sit & np.isfinite(lat) & np.isfinite(sit) & (sit > 0)
    if m_sit.sum() > 0:
        sit_polynya = sit[m_sit]
        sit_median = np.nanmedian(sit_polynya)
        sit_q25 = np.nanpercentile(sit_polynya, 25)
        sit_q75 = np.nanpercentile(sit_polynya, 75)
        sit_q90 = np.nanpercentile(sit_polynya, 90)
        sit_peak = np.nanmax(sit_polynya)
        print(f"\n4. SIT within polynya mask ({m_sit.sum():,} points):")
        print(f"   Median:        {sit_median:.2f} m")
        print(f"   IQR:           {sit_q25:.2f} -- {sit_q75:.2f} m")
        print(f"   90th percentile: {sit_q90:.2f} m")
        print(f"   Peak:          {sit_peak:.2f} m")
        
        # 5. Across-beam standard deviation of SIT within 2 km bins
        lat_sit = lat[m_sit]
        sit_vals = sit[m_sit]
        s_sit = s_km[m_sit]
        beam_sit = beam[m_sit]
        
        if np.isfinite(s_sit).sum() >= min_per_bin:
            s_min, s_max = np.nanmin(s_sit), np.nanmax(s_sit)
            bins = np.arange(s_min, s_max + bin_km, bin_km)
            bin_stds = []
            for i in range(len(bins) - 1):
                in_bin = (s_sit >= bins[i]) & (s_sit < bins[i + 1])
                if in_bin.sum() >= min_per_bin:
                    sit_bin = sit_vals[in_bin]
                    beam_bin = beam_sit[in_bin]
                    # Calculate std across beams (need at least 2 beams)
                    if len(np.unique(beam_bin)) >= 2:
                        # Get mean SIT per beam in this bin
                        beam_means = []
                        for b in np.unique(beam_bin):
                            beam_mask = (beam_bin == b)
                            if beam_mask.sum() > 0:
                                beam_means.append(np.nanmean(sit_bin[beam_mask]))
                        if len(beam_means) >= 2:
                            bin_std = np.std(beam_means)
                            bin_stds.append(bin_std)
            if len(bin_stds) > 0:
                std_median = np.median(bin_stds)
                std_q25 = np.nanpercentile(bin_stds, 25)
                std_q75 = np.nanpercentile(bin_stds, 75)
                print(f"\n5. Across-beam std of SIT (within {bin_km} km bins):")
                print(f"   Median: {std_median:.3f} m")
                print(f"   IQR:    {std_q25:.3f} -- {std_q75:.3f} m")
    else:
        print("\n4. SIT statistics: No valid SIT data within polynya mask")
    
    print("\n" + "=" * 80 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="2-panel plot from *_is2_polynya_sit_outputs.npz with polynya detection from SWOT + AMSR."
    )
    ap.add_argument("--npz", required=True, help="Path to *_is2_polynya_sit_outputs.npz")
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument("--outdir", default=None, help="Alias for --out (backward compatibility)")
    ap.add_argument("--bin_km", type=float, default=2.0, help="(Unused for mean lines now) kept for compatibility.")
    ap.add_argument("--min_per_bin", type=int, default=5, help="Min points per window/bin (used for running mean).")
    ap.add_argument("--gap_lat_deg", type=float, default=0.03, help="Break lines if |Δlat| exceeds this (deg).")
    ap.add_argument("--alpha_scatter", type=float, default=0.25, help="Base scatter alpha.")

    ap.add_argument("--run_window_km", type=float, default=2.0, help="Running-mean window (km). Default=2.")
    ap.add_argument("--run_step_km", type=float, default=0.5, help="Running-mean step (km). Default=0.5.")

    ap.add_argument("--mask_for_highlight", choices=["final", "swot", "amsr", "union", "both"], default="final")
    ap.add_argument("--mask_for_sit", choices=["final", "swot", "amsr", "union", "both"], default="final")

    ap.add_argument("--polynya_only", action="store_true",
                    help="Top panel: plot only points inside --mask_for_highlight.")
    ap.add_argument("--sit_all_points", action="store_true",
                    help="Bottom panel: plot SIT for all points (ignores --mask_for_sit).")

    ap.add_argument("--shade_merge_pad_deg", type=float, default=0.002, help="Merge/bridge close spans (deg).")

    ap.add_argument("--show_swot", action="store_true")
    ap.add_argument("--show_amsr", action="store_true")

    ap.add_argument("--show_ssha", action="store_true",
                    help="Upper panel: overlay SWOT SSHA/SSH on a right y-axis by default.")
    ap.add_argument("--ssha_key", default=None,
                    help="NPZ key for SSHA/SSH (if not provided, auto-detect; prefers swot_ssh_mean).")
    ap.add_argument("--ssha_left_axis", action="store_true",
                    help="Plot SSHA/SSH on the left axis (only if it shares the same vertical datum as IS2 heights).")
    ap.add_argument("--ssha_scale", type=float, default=1.0)
    ap.add_argument("--ssha_offset", type=float, default=0.0)
    ap.add_argument("--ssha_plot_std", action="store_true",
                    help="If a matching std key exists, shade ±1 std around SSHA running-mean lines.")
    ap.add_argument("--ssha_std_key", default=None,
                    help="Explicit NPZ key for SSHA/SSH std (optional).")

    args = ap.parse_args()

    if not (args.show_swot or args.show_amsr):
        args.show_swot = True
        args.show_amsr = True

    out = args.out
    if out is None and args.outdir is not None:
        out = args.outdir
    if out is None:
        out = "elev_ref_and_sit_vs_lat_all6.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)

    beam = _as_str_array(z["beam"])
    lat = z["lat"].astype(float)

    if "height" not in z.files or "ref_height" not in z.files:
        raise RuntimeError("NPZ must contain 'height' and 'ref_height' for the upper panel.")
    height = z["height"].astype(float)
    ref_height = z["ref_height"].astype(float)

    if "sit" not in z.files:
        raise RuntimeError("NPZ must contain 'sit' for the lower panel.")
    sit = z["sit"].astype(float)

    if "s_km_beam" in z.files:
        s_km = z["s_km_beam"].astype(float)
    elif "t_km_joint" in z.files:
        s_km = z["t_km_joint"].astype(float)
    else:
        s_km = np.full_like(lat, np.nan)

    seg_joint = z["seg_joint"].astype(int) if "seg_joint" in z.files else None
    if seg_joint is not None:
        seg_joint_valid = seg_joint.astype(float)
        seg_joint_valid[seg_joint < 0] = np.nan
    else:
        seg_joint_valid = None

    masks = _get_masks_from_npz(z, n=len(lat))

    def _resolve_mask(name: str):
        m = masks.get(name, None)
        return masks["final"] if m is None else m

    mask_highlight = _resolve_mask(args.mask_for_highlight)
    mask_sit = _resolve_mask(args.mask_for_sit)

    beams_order = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
    unique = [b for b in beams_order if np.any(beam == b)]
    unique += [b for b in np.unique(beam) if b not in unique]

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]},
        facecolor="white"
    )

    # (kept your palette as-is; change here if you want)
    palette6 = [
        "#852D86",  # Purple
        "#B94B77",  # Burgundy-pink
        "#D37854",  # Orange
        "#DEA961",  # Golden
        "#D5D386",  # Olive-yellow
        "#A4D9A6",  # Light green
        "#78D9C7",  # Teal
        "#6CC7DE",  # Light blue
    ]
    beam_color = {b: palette6[i % len(palette6)] for i, b in enumerate(unique)}

    # ---- build spans for SWOT ∪ AMSR backgrounds
    order_key = None
    if np.isfinite(s_km).sum() >= 10:
        order_key = s_km
    elif seg_joint_valid is not None and np.isfinite(seg_joint_valid).sum() >= 10:
        order_key = seg_joint_valid

    union_mask = masks["final"]
    spans_union = _polynya_lat_spans_per_beam(
        beam=beam,
        lat=lat,
        in_pol=union_mask,
        order_key=order_key,
        gap_lat_deg=float(args.gap_lat_deg),
        pad_merge=float(args.shade_merge_pad_deg),
    )

    _add_span_layer(ax0, spans_union, face="0.95", edge="0.40", alpha=0.45, lw=1.2, ls="--", zorder=0)
    _add_span_layer(ax1, spans_union, face="0.95", edge="0.40", alpha=0.45, lw=1.2, ls="--", zorder=0)

    # -------------------------
    # TOP: IS2 elevation (scatter)
    # -------------------------
    for b in unique:
        c = beam_color[b]

        m_all = (beam == b) & np.isfinite(lat) & np.isfinite(height)
        if m_all.sum() == 0:
            continue

        if not args.polynya_only:
            m_out = m_all & (~mask_highlight)
            if m_out.sum() > 0:
                ax0.scatter(
                    lat[m_out], height[m_out],
                    s=40,
                    alpha=max(0.15, float(args.alpha_scatter) * 0.65),
                    color=c,
                    edgecolors="none",
                    linewidths=0.0,
                    label="_nolegend_",
                    zorder=3,
                )

        m_in = m_all & mask_highlight
        if m_in.sum() > 0:
            ax0.scatter(
                lat[m_in], height[m_in],
                s=40,
                alpha=min(0.85, float(args.alpha_scatter) * 1.0),
                color=c,
                edgecolors="none",
                linewidths=0.0,
                label="_nolegend_",
                zorder=4,
            )

    # -------------------------
    # TOP: IS2 Reference elevation (ONE combined line) — 2 km running mean
    # -------------------------
    ref_line_color = "k"
    ref_line_lw = 2.6
    ref_line_alpha = 0.85

    m_ref = np.isfinite(lat) & np.isfinite(ref_height) & np.isfinite(s_km)
    if args.polynya_only:
        m_ref &= mask_highlight

    if m_ref.sum() >= max(10, int(args.min_per_bin)):
        latm, refm, _, idxm = running_mean_std_by_distance(
            lat[m_ref], ref_height[m_ref], s_km[m_ref],
            window_km=float(args.run_window_km),
            step_km=float(args.run_step_km),
            min_per_win=max(3, int(args.min_per_bin)),
        )
        if latm.size >= 2:
            xline, yline = break_line_on_gaps(latm, refm, idx=idxm, gap_lat_deg=float(args.gap_lat_deg))
            _plot_with_white_edge(
                ax0, xline, yline,
                color=ref_line_color, lw=ref_line_lw, ls="-",
                alpha=ref_line_alpha, zorder=6,
                edge_lw=2.8, edge_alpha=0.95
            )
    else:
        if seg_joint_valid is not None:
            m_ref2 = np.isfinite(lat) & np.isfinite(ref_height) & np.isfinite(seg_joint_valid)
            if args.polynya_only:
                m_ref2 &= mask_highlight
            if m_ref2.sum() >= 2:
                lat_ref, ref_ref, segs = segment_means(
                    lat[m_ref2], ref_height[m_ref2], seg_joint_valid[m_ref2], min_per_seg=1
                )
                if lat_ref.size >= 2:
                    order = np.argsort(segs)
                    xline, yline = break_line_on_gaps(
                        lat_ref[order], ref_ref[order], idx=segs[order], gap_lat_deg=float(args.gap_lat_deg)
                    )
                    _plot_with_white_edge(
                        ax0, xline, yline,
                        color=ref_line_color, lw=ref_line_lw, ls="-",
                        alpha=ref_line_alpha, zorder=6,
                        edge_lw=2.8, edge_alpha=0.95
                    )

    # -------------------------
    # TOP: SWOT SSHA/SSH overlay — only gt1l, gt2l, gt3l lines, 2 km running mean
    # -------------------------
    ax0_ssha = None
    ssha_plotted = False

    ssha_left_beams = ["gt1l", "gt2l", "gt3l"]
    ssha_mean_ls = "--"
    ssha_mean_lw = 2.2
    ssha_mean_alpha = 0.95

    if args.show_ssha:
        key = args.ssha_key or _detect_ssha_key(z.files)
        if key is None:
            print("[WARN] --show_ssha requested but no SSHA/SSH-like key found in NPZ.")
        else:
            ssha = np.asarray(z[key], float).reshape(-1)
            if ssha.size != lat.size:
                print(f"[WARN] SSHA/SSH key '{key}' has size {ssha.size}, expected {lat.size}. Skipping overlay.")
            else:
                y_ssha = args.ssha_scale * ssha + args.ssha_offset

                plot_on_right = not args.ssha_left_axis
                if plot_on_right:
                    ax0_ssha = ax0.twinx()
                    ax_plot = ax0_ssha
                    ax0_ssha.spines["top"].set_visible(False)
                    ax0_ssha.tick_params(labelsize=11)
                    ax0_ssha.set_ylabel("SWOT SSHA (m)", fontsize=15)
                    ax0_ssha.set_ylim(-0.6, -0.3)
                else:
                    ax_plot = ax0

                m_ssha_all = np.isfinite(lat) & np.isfinite(y_ssha) & np.isfinite(s_km)
                if args.polynya_only:
                    m_ssha_all &= mask_highlight

                # optional light SSHA points (very subtle)
                if m_ssha_all.sum() >= 2:
                    ax_plot.scatter(
                        lat[m_ssha_all], y_ssha[m_ssha_all],
                        s=15, alpha=0.06, color="0.35",
                        edgecolors="none", linewidths=0.0,
                        zorder=2, label="_nolegend_",
                    )

                std_key = None
                ssha_std_all = None
                if args.ssha_plot_std:
                    std_key = args.ssha_std_key or _guess_ssha_std_key(z.files, key)
                    if std_key is not None:
                        ssha_std_all = np.asarray(z[std_key], float).reshape(-1)
                        if ssha_std_all.size != lat.size:
                            print(f"[WARN] std key '{std_key}' size mismatch; skipping std shading.")
                            std_key = None
                            ssha_std_all = None
                    else:
                        print("[WARN] --ssha_plot_std set but no std key found; skipping std shading.")

                for b in ssha_left_beams:
                    if not np.any(beam == b):
                        continue
                    mb = (beam == b) & m_ssha_all
                    if mb.sum() < max(10, int(args.min_per_bin)):
                        continue

                    latm, ym, _, idxm = running_mean_std_by_distance(
                        lat[mb], y_ssha[mb], s_km[mb],
                        window_km=float(args.run_window_km),
                        step_km=float(args.run_step_km),
                        min_per_win=max(3, int(args.min_per_bin)),
                    )
                    if latm.size < 2:
                        continue

                    xline, yline = break_line_on_gaps(latm, ym, idx=idxm, gap_lat_deg=float(args.gap_lat_deg))

                    _plot_with_white_edge(
                        ax_plot, xline, yline,
                        color=beam_color.get(b, "0.25"),
                        lw=ssha_mean_lw, ls=ssha_mean_ls,
                        alpha=ssha_mean_alpha, zorder=7,
                        edge_lw=2.0, edge_alpha=0.90
                    )

                    if std_key is not None and ssha_std_all is not None:
                        y_std = np.abs(args.ssha_scale) * ssha_std_all
                        latm2, stdm2, _, idxm2 = running_mean_std_by_distance(
                            lat[mb], y_std[mb], s_km[mb],
                            window_km=float(args.run_window_km),
                            step_km=float(args.run_step_km),
                            min_per_win=max(3, int(args.min_per_bin)),
                        )
                        if latm2.size >= 2:
                            x2, m2 = break_line_on_gaps(latm2, stdm2, idx=idxm2, gap_lat_deg=float(args.gap_lat_deg))
                            ok = np.isfinite(xline) & np.isfinite(yline) & np.isfinite(m2)
                            ax_plot.fill_between(
                                xline, yline - m2, yline + m2,
                                where=ok,
                                alpha=0.10,
                                color=beam_color.get(b, "0.25"),
                                linewidth=0.0,
                                zorder=1,
                            )

                    ssha_plotted = True

                print(f"[INFO] Plotted SSHA/SSH running-mean lines (2 km) for gt1l/gt2l/gt3l from key '{key}'.")

    ax0.set_ylabel("Elevation (m)", fontsize=15)
    ax0.grid(True, alpha=0.2, linestyle="--", linewidth=0.7)
    ax0.set_ylim(-0.4, 0.1)
    ax0.minorticks_on()
    ax0.tick_params(labelsize=11)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # -------------------------
    # BOTTOM: SIT scatter + 2 km running-mean lines
    # -------------------------
    for b in unique:
        c = beam_color[b]

        m_all = (beam == b) & np.isfinite(lat) & np.isfinite(sit) & np.isfinite(s_km)
        if m_all.sum() == 0:
            continue

        if not args.sit_all_points:
            m_use = m_all & mask_sit
            if m_use.sum() == 0:
                continue
            lat_use, sit_use, s_use = lat[m_use], sit[m_use], s_km[m_use]
            ax1.scatter(
                lat_use, sit_use,
                s=40, alpha=min(0.85, float(args.alpha_scatter) * 1.0),
                color=c, edgecolors="none", linewidths=0.0,
                label="_nolegend_", zorder=4
            )
        else:
            m_out = m_all & (~mask_sit)
            if m_out.sum() > 0:
                ax1.scatter(
                    lat[m_out], sit[m_out],
                    s=40, alpha=max(0.15, float(args.alpha_scatter) * 0.65),
                    color=c, edgecolors="none", linewidths=0.0,
                    label="_nolegend_", zorder=3
                )
            m_in = m_all & mask_sit
            if m_in.sum() == 0:
                continue
            lat_use, sit_use, s_use = lat[m_in], sit[m_in], s_km[m_in]
            ax1.scatter(
                lat_use, sit_use,
                s=40, alpha=min(0.85, float(args.alpha_scatter) * 1.0),
                color=c, edgecolors="none", linewidths=0.0,
                label="_nolegend_", zorder=4
            )

        # 2 km running-mean SIT line (WITH white edge)
        if lat_use.size >= max(10, int(args.min_per_bin)):
            latm, sitm, _, idxm = running_mean_std_by_distance(
                lat_use, sit_use, s_use,
                window_km=float(args.run_window_km),
                step_km=float(args.run_step_km),
                min_per_win=max(3, int(args.min_per_bin)),
            )
            if latm.size >= 2:
                xline, yline = break_line_on_gaps(latm, sitm, idx=idxm, gap_lat_deg=float(args.gap_lat_deg))
                _plot_with_white_edge(
                    ax1, xline, yline,
                    color=c, lw=2.5, ls="-",
                    alpha=0.75, zorder=5,
                    edge_lw=2.2, edge_alpha=0.90
                )

    ax1.set_xlabel("Latitude", fontsize=15)
    ax1.set_ylabel("Sea Ice Thickness (m)", fontsize=15)
    ax1.set_ylim(-0.05, 5)
    ax1.grid(True, alpha=0.2, linestyle="--", linewidth=0.7)
    ax1.minorticks_on()
    ax1.tick_params(labelsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # -------------------------
    # Legend (bottom panel)
    # -------------------------
    handles = [
        Patch(facecolor="0.95", edgecolor="0.40", alpha=0.45, linestyle="--", linewidth=1.2,
              label="SWOT ∪ AMSR polynya"),
        Line2D([0], [0], color="k", linewidth=ref_line_lw, alpha=ref_line_alpha,
               label="IS-2 reference elevation"),
    ]

    if ssha_plotted:
        handles.append(
            Line2D([0], [0], color="0.35", linewidth=ssha_mean_lw, linestyle=ssha_mean_ls, alpha=ssha_mean_alpha,
                   label="SWOT SSHA")
        )

    for b in unique:
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=beam_color[b],
                   markersize=10, linestyle="None", label=b)
        )

    ax1.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.4,
        labelspacing=0.8,
        fontsize=15,
        edgecolor="0.3",
    )

    # -------------------------
    # Calculate and print statistics for manuscript
    # -------------------------
    ssha_key_for_stats = None
    if args.show_ssha:
        ssha_key_for_stats = args.ssha_key or _detect_ssha_key(z.files)
    
    _print_manuscript_statistics(
        lat=lat, height=height, ref_height=ref_height, sit=sit, s_km=s_km,
        beam=beam, mask_polynya=mask_highlight, mask_sit=mask_sit,
        ssha_key=ssha_key_for_stats,
        z=z, bin_km=float(args.bin_km), min_per_bin=int(args.min_per_bin)
    )

    plt.tight_layout(pad=1.3)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
