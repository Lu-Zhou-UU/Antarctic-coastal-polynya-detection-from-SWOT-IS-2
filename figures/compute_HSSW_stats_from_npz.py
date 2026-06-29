#!/usr/bin/env python3
"""
Compute HSSW statistics from the NPZ saved by plot_HSSW_from_SIT_RACMO_Sohey.py
with --save_npz_out. Prints mean, s.d., min, max, and pairwise correlations
for pasting into the manuscript (manuscript_HSSW_subsection_enriched.tex).

Usage:
  python compute_HSSW_stats_from_npz.py hssw_series.npz

Example (after running the plot script with --save_npz_out):
  python plot_HSSW_from_SIT_RACMO_Sohey.py ... --save_npz_out hssw_series.npz
  python ls hssw_series.npz
"""

import argparse
import numpy as np


def _stats(name, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"  {name}: no valid values")
        return None
    n = x.size
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x, ddof=1)) if n > 1 else 0.0
    min_, max_ = float(np.nanmin(x)), float(np.nanmax(x))
    print(f"  {name}: mean = {mean:.4f} Sv, s.d. = {std:.4f} Sv, n = {n}")
    print(f"         min = {min_:.4f}, max = {max_:.4f} Sv")
    return {"mean": mean, "std": std, "n": n, "min": min_, "max": max_}


def _align_by_date(times_a, vals_a, times_b, vals_b):
    """Align two series by date (string match on date part). Returns (a_aligned, b_aligned)."""
    times_a = np.asarray(times_a).ravel()
    times_b = np.asarray(times_b).ravel()
    vals_a = np.asarray(vals_a, dtype=float).ravel()
    vals_b = np.asarray(vals_b, dtype=float).ravel()
    # Normalize to date string YYYY-MM-DD for matching
    def to_date_str(t):
        s = str(t).split()[0].split("T")[0]
        return s[:10] if len(s) >= 10 else s
    dates_a = [to_date_str(t) for t in times_a]
    dates_b = [to_date_str(t) for t in times_b]
    set_b = set(dates_b)
    a_alg, b_alg = [], []
    for i, da in enumerate(dates_a):
        if da in set_b:
            j = dates_b.index(da)
            a_alg.append(vals_a[i])
            b_alg.append(vals_b[j])
    return np.array(a_alg), np.array(b_alg)


def _corr_and_ci(x, y, n_bootstrap=2000):
    """Pearson r and 95% CI via bootstrap. x, y must be same length."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size != y.size:
        return np.nan, (np.nan, np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return np.nan, (np.nan, np.nan)
    r = np.corrcoef(x, y)[0, 1]
    r_bs = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        idx = rng.integers(0, x.size, size=x.size)
        r_bs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    r_bs = np.array(r_bs)
    ci = (float(np.percentile(r_bs, 2.5)), float(np.percentile(r_bs, 97.5)))
    return r, ci


def main():
    ap = argparse.ArgumentParser(description="Compute HSSW stats from plot script NPZ output")
    ap.add_argument("npz", help="Path to NPZ from --save_npz_out")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for correlation CI")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)

    print("=" * 60)
    print("HSSW statistics (Sv) for manuscript placeholders")
    print("=" * 60)

    stats_racmo = None
    if "hssw_racmo" in d:
        stats_racmo = _stats("RACMO (single region)", d["hssw_racmo"])
    else:
        for suf in ("RSP", "TNBP", "MSP"):
            key = f"hssw_racmo_{suf}"
            if key in d:
                _stats(f"RACMO {suf}", d[key])
        if "hssw_racmo_RSP" in d and "hssw_racmo_TNBP" in d and "hssw_racmo_MSP" in d:
            rsp = np.nan_to_num(d["hssw_racmo_RSP"], nan=0.0)
            tnb = np.nan_to_num(d["hssw_racmo_TNBP"], nan=0.0)
            msp = np.nan_to_num(d["hssw_racmo_MSP"], nan=0.0)
            combined = rsp + tnb + msp
            stats_racmo = _stats("RACMO (RSP+TNBP+MSP combined)", combined)

    if "hssw_sohey" in d:
        _stats("Sohey PMW", d["hssw_sohey"])
    if "hssw_sit" in d:
        _stats("SWOT--IS2", d["hssw_sit"])

    print()
    print("Pairwise correlations (r, 95% CI), aligned by date where lengths differ:")
    racmo_series = None
    racmo_times = d.get("racmo_times_str")
    sohey_times = d.get("sohey_times_str")
    sit_times = d.get("sit_times_str")

    if "hssw_racmo" in d:
        racmo_series = d["hssw_racmo"]
    elif "hssw_racmo_RSP" in d and "hssw_racmo_TNBP" in d and "hssw_racmo_MSP" in d:
        racmo_series = np.nan_to_num(d["hssw_racmo_RSP"], nan=0) + np.nan_to_num(d["hssw_racmo_TNBP"], nan=0) + np.nan_to_num(d["hssw_racmo_MSP"], nan=0)

    if racmo_series is not None and "hssw_sohey" in d:
        if sohey_times is not None and racmo_times is not None:
            ra, so = _align_by_date(racmo_times, racmo_series, sohey_times, d["hssw_sohey"])
            r, ci = _corr_and_ci(ra, so, args.bootstrap)
            print(f"  RACMO vs Sohey: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})  (n = {len(ra)} aligned dates)")
        else:
            r, ci = _corr_and_ci(racmo_series, d["hssw_sohey"], args.bootstrap)
            print(f"  RACMO vs Sohey: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})")
    if "hssw_racmo_RSP" in d and "hssw_sohey" in d and "hssw_racmo" not in d and racmo_times is not None and sohey_times is not None:
        ra, so = _align_by_date(racmo_times, d["hssw_racmo_RSP"], sohey_times, d["hssw_sohey"])
        r, ci = _corr_and_ci(ra, so, args.bootstrap)
        print(f"  RACMO RSP vs Sohey: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})  (n = {len(ra)} aligned dates)")
    if "hssw_sit" in d and "hssw_sohey" in d:
        if sit_times is not None and sohey_times is not None:
            si, so = _align_by_date(sit_times, d["hssw_sit"], sohey_times, d["hssw_sohey"])
            r, ci = _corr_and_ci(si, so, args.bootstrap)
            print(f"  SWOT--IS2 vs Sohey: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})  (n = {len(si)} aligned dates)")
        else:
            r, ci = _corr_and_ci(d["hssw_sit"], d["hssw_sohey"], args.bootstrap)
            print(f"  SWOT--IS2 vs Sohey: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})")
    if "hssw_sit" in d and racmo_series is not None:
        if sit_times is not None and racmo_times is not None:
            si, ra = _align_by_date(sit_times, d["hssw_sit"], racmo_times, racmo_series)
            r, ci = _corr_and_ci(si, ra, args.bootstrap)
            print(f"  SWOT--IS2 vs RACMO: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})  (n = {len(si)} aligned dates)")
        else:
            r, ci = _corr_and_ci(d["hssw_sit"], racmo_series, args.bootstrap)
            print(f"  SWOT--IS2 vs RACMO: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})")

    print()
    print("Manuscript placeholders (copy into .tex):")
    print("  RACMO mean ± s.d.:  [fill from RACMO line above]")
    print("  Sohey mean ± s.d.:  [fill from Sohey line above]")
    print("  SWOT--IS2 mean ± s.d.: [fill if present]")
    print("  Correlation RACMO–Sohey: r = [X.XX] ([X.XX]--[X.XX] 95% CI)")
    print("  TNB benchmark: 0.34--0.55 Sv; RISp: 0.1--0.4 Sv (already in text)")


if __name__ == "__main__":
    main()
