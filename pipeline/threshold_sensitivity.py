#!/usr/bin/env python3
"""
threshold_sensitivity.py - CAR-cutoff sensitivity (Table 4).

For each cutoff, re-select the sample from the full-universe screen (all common
stocks with an FF3 CAR over the Jan 11 - Feb 3 2021 window), re-match controls
(Lyon, Barber and Tsai 1999), and recompute BHAR at 36 and 48 months.
Sub-100% stocks' returns come from the control-returns file; 100%+ stocks from
the meme file.

Output: meme_stock_analysis/v13_outputs/threshold_sensitivity.csv
  threshold_pct, n_sample, n_matched, bhar36, t36, bhar48, t48, pct_neg48

Run: python3 threshold_sensitivity.py --allow-run
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _base as B  # noqa: E402
from _guard import check_run_guard  # noqa: E402

THRESHOLDS = [50, 75, 100, 125, 150, 200, 300]


def main():
    check_run_guard("threshold_sensitivity.py")
    B.ensure_out()
    fu = B.screen_full_universe()
    ret_permnos = set(B._returns_by_permno().keys())

    rows = []
    for thr in THRESHOLDS:
        sample = fu[fu["car_pct"] > thr]
        # keep only stocks for which we actually have a return series
        perms = [int(p) for p in sample["permno"] if int(p) in ret_permnos]
        matches = B.build_matches(perms)
        panel = B.bhar_panel(matches, horizons=[36, 48])
        b36 = panel[panel.horizon_mo == 36].dropna(subset=["bhar"])["bhar"].values
        b48 = panel[panel.horizon_mo == 48].dropna(subset=["bhar"])["bhar"].values
        m36, t36, _ = B.tstat(b36)
        m48, t48, n48 = B.tstat(b48)
        pct_neg = (sum(1 for v in b48 if v < 0) / len(b48) * 100
                   if len(b48) else float("nan"))
        rows.append([thr, len(perms), len(matches),
                     round(m36 * 100, 1), round(t36, 2),
                     round(m48 * 100, 1), round(t48, 2), round(pct_neg, 1)])
        print(f"  CAR>{thr:>3}%  n={len(perms):>3} matched={len(matches):>3}  "
              f"48mo BHAR={m48 * 100:7.1f}% (t={t48:6.2f})")
    out = pd.DataFrame(rows, columns=["threshold_pct", "n_sample", "n_matched",
                                      "bhar36", "t36", "bhar48", "t48",
                                      "pct_neg48"])
    out.to_csv(f"{B.OUT}/threshold_sensitivity.csv", index=False)
    print("\nWrote threshold_sensitivity.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
