#!/usr/bin/env python3
"""
Compute SIT/RACMO/Sohey/ESACCI statistics from the NPZ saved by plot_SWOT_PM_RACMO_SIT.py
with --save_npz_out. All statistics and correlations are restricted to dates where
SWOT-IS-2 has a valid SIT value (fair comparison). Prints mean, s.d., n, min, max and
pairwise correlations for use in manuscripts. Optionally exports monthly means to CSV.

Usage:
  python compute_SIT_stats_from_npz.py merged_plot_elements.npz
  python compute_SIT_stats_from_npz.py merged_plot_elements.npz --out_monthly_csv monthly_means.csv

The NPZ should contain at least sit_times_str, sit_mean, sit_std; optionally
racmo_times_str, racmo_values; sohey_times_str, sohey_values; esacci_times_str,
esacci_values, esacci_std.
"""

import argparse
import csv
import re
import numpy as np
from datetime import datetime
from collections import defaultdict


def _parse_dt(s):
    s = str(s).strip().replace("T", " ")
    if "." in s:
        base, frac = s.split(".", 1)
        m = re.match(r"(\d+)", frac)
        frac_digits = (m.group(1) if m else "")[:6]
        s = f"{base}.{frac_digits}"
    return datetime.fromisoformat(s)


def _to_date_str(t):
    s = str(t).split()[0].split("T")[0]
    return s[:10] if len(s) >= 10 else s


def _build_date_to_value_map(times, values):
    """Build dict date_str -> value. times/values can be arrays or lists."""
    if times is None or values is None:
        return {}
    times = np.asarray(times).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if len(times) != len(values):
        return {}
    out = {}
    for i in range(len(times)):
        d = _to_date_str(times[i])
        v = values[i]
        if np.isfinite(v):
            out[d] = float(v)
    return out


def _corr_and_ci(x, y, n_bootstrap=2000):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size != y.size:
        return np.nan, (np.nan, np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return np.nan, (np.nan, np.nan)
    r = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng()
    r_bs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, x.size, size=x.size)
        r_bs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    r_bs = np.array(r_bs)
    ci = (float(np.percentile(r_bs, 2.5)), float(np.percentile(r_bs, 97.5)))
    return r, ci


def _print_series_stats(name, vals, unit="m", n_dates=None):
    v = np.asarray(vals, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"  {name}: no valid values" + (f" (on {n_dates} SWOT-IS-2 dates)" if n_dates is not None else ""))
        return
    n = v.size
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    mn, mx = float(np.min(v)), float(np.max(v))
    suffix = f"  [on {n_dates} SWOT-IS-2 dates]" if n_dates is not None else ""
    print(f"  {name}: mean = {mean:.4f} {unit}, s.d. = {std:.4f}, n = {n}{suffix}")
    print(f"         min = {mn:.4f}, max = {mx:.4f} {unit}")


def main():
    ap = argparse.ArgumentParser(
        description="Compute SIT/RACMO/Sohey/ESACCI stats from plot script NPZ (only when SWOT-IS-2 SIT is valid)"
    )
    ap.add_argument("npz", help="Path to NPZ from --save_npz_out of plot_SWOT_PM_RACMO_SIT.py")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out_monthly_csv", default=None,
                    help="Export monthly means to CSV. Each product's mean is computed from its "
                         "own original time series (all valid days in that month). Rows are all "
                         "months that appear in any product.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)

    sit_times = d.get("sit_times_str")
    sit_mean = d.get("sit_mean")
    racmo_times = d.get("racmo_times_str")
    racmo_vals = d.get("racmo_values")
    sohey_times = d.get("sohey_times_str")
    sohey_vals = d.get("sohey_values")
    esacci_times = d.get("esacci_times_str")
    esacci_vals = d.get("esacci_values")

    if sit_times is None or sit_mean is None:
        raise SystemExit("NPZ must contain sit_times_str and sit_mean.")

    sit_times = np.asarray(sit_times).ravel()
    sit_mean = np.asarray(sit_mean, dtype=float).ravel()
    if len(sit_times) != len(sit_mean):
        raise SystemExit("sit_times_str and sit_mean length mismatch.")

    # Only dates where SWOT-IS-2 has valid SIT
    valid_mask = np.isfinite(sit_mean)
    if np.sum(valid_mask) == 0:
        raise SystemExit("No valid SWOT-IS-2 SIT values in NPZ.")

    sit_dates_valid = [_to_date_str(sit_times[i]) for i in range(len(sit_times)) if valid_mask[i]]
    sit_vals_valid = sit_mean[valid_mask]
    n_swot_dates = len(sit_dates_valid)

    # Build date -> value for other products
    racmo_map = _build_date_to_value_map(racmo_times, racmo_vals)
    sohey_map = _build_date_to_value_map(sohey_times, sohey_vals)
    esacci_map = _build_date_to_value_map(esacci_times, esacci_vals)

    # Align all to SWOT-IS-2 valid dates (same order)
    sit_aligned = np.array(sit_vals_valid, dtype=float)
    racmo_aligned = np.array([racmo_map.get(d, np.nan) for d in sit_dates_valid], dtype=float)
    sohey_aligned = np.array([sohey_map.get(d, np.nan) for d in sit_dates_valid], dtype=float)
    esacci_aligned = np.array([esacci_map.get(d, np.nan) for d in sit_dates_valid], dtype=float)

    print("=" * 60)
    print("SIT statistics (all restricted to dates where SWOT-IS-2 SIT is valid)")
    print("=" * 60)
    print(f"  Number of SWOT-IS-2 dates with valid SIT: {n_swot_dates}")
    print()

    _print_series_stats("SWOT-IS-2 (SIT)", sit_aligned, "m", n_dates=n_swot_dates)
    _print_series_stats("RACMO (ice production)", racmo_aligned, "m/month", n_dates=n_swot_dates)
    _print_series_stats("Sohey PMW (HI)", sohey_aligned, "m", n_dates=n_swot_dates)
    _print_series_stats("ESACCI", esacci_aligned, "m", n_dates=n_swot_dates)

    print()
    print("Pairwise correlations (r, 95% CI) on SWOT-IS-2 valid dates (n where both finite):")

    series = [
        ("SWOT-IS-2", sit_aligned),
        ("RACMO", racmo_aligned),
        ("Sohey", sohey_aligned),
        ("ESACCI", esacci_aligned),
    ]

    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            name_a, vals_a = series[i]
            name_b, vals_b = series[j]
            r, ci = _corr_and_ci(vals_a, vals_b, args.bootstrap)
            n_both = int(np.sum(np.isfinite(vals_a) & np.isfinite(vals_b)))
            print(f"  {name_a} vs {name_b}: r = {r:.3f} ({ci[0]:.3f}--{ci[1]:.3f})  (n = {n_both})")
    print()

    # ---- Monthly means from each product's own original data ----
    if args.out_monthly_csv:
        def monthly_means_from_series(times, values):
            """Group by (year, month), return dict (year, month) -> (mean, n) from that product's own data."""
            if times is None or values is None:
                return {}
            times = np.asarray(times).ravel()
            values = np.asarray(values, dtype=float).ravel()
            if len(times) != len(values):
                return {}
            out = defaultdict(list)
            for i in range(len(times)):
                t = times[i]
                v = values[i]
                if not np.isfinite(v):
                    continue
                try:
                    s = str(t).split()[0].split("T")[0][:10]
                    dt = datetime.strptime(s, "%Y-%m-%d")
                    out[(dt.year, dt.month)].append(float(v))
                except (ValueError, TypeError):
                    continue
            result = {}
            for key, vals in out.items():
                arr = np.array(vals, dtype=float)
                result[key] = (float(np.mean(arr)), len(arr))
            return result

        sit_monthly = monthly_means_from_series(sit_times, sit_mean)
        racmo_monthly = monthly_means_from_series(racmo_times, racmo_vals)
        sohey_monthly = monthly_means_from_series(sohey_times, sohey_vals)
        esacci_monthly = monthly_means_from_series(esacci_times, esacci_vals)

        all_months = set()
        for d in (sit_monthly, racmo_monthly, sohey_monthly, esacci_monthly):
            all_months.update(d.keys())
        month_keys = sorted(all_months)

        fieldnames = ["year", "month",
                      "sit_swot_is2_mean", "n_swot_is2",
                      "sit_racmo_mean", "n_racmo",
                      "sit_sohey_mean", "n_sohey",
                      "sit_esacci_mean", "n_esacci"]
        with open(args.out_monthly_csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            for (year, month) in month_keys:
                row = {"year": year, "month": month,
                       "sit_swot_is2_mean": "", "n_swot_is2": "",
                       "sit_racmo_mean": "", "n_racmo": "",
                       "sit_sohey_mean": "", "n_sohey": "",
                       "sit_esacci_mean": "", "n_esacci": ""}
                for name, monthly_dict in [
                    ("sit_swot_is2", sit_monthly),
                    ("sit_racmo", racmo_monthly),
                    ("sit_sohey", sohey_monthly),
                    ("sit_esacci", esacci_monthly),
                ]:
                    if (year, month) in monthly_dict:
                        mean_val, n_val = monthly_dict[(year, month)]
                        row[f"{name}_mean"] = round(mean_val, 4)
                        row[f"n_{'swot_is2' if name == 'sit_swot_is2' else name.split('_')[1]}"] = n_val
                w.writerow(row)
        print(f"[INFO] Exported monthly means (each product's own data): {args.out_monthly_csv} ({len(month_keys)} months)")


if __name__ == "__main__":
    main()
