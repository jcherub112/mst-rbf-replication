#!/usr/bin/env python3
"""
bhar_alt_specs.py - Alternative inference for long-run BHAR (Table C1, Appendix C).

Long-run BHARs are skewed, so the conventional t-test is misspecified. Three
alternatives:

  1. Skewness-adjusted t (Johnson 1978; Lyon, Barber and Tsai 1999).
  2. Bootstrap p-value (10,000 resamples) for the mean BHAR.
  3. Monthly cross-sectional mean: time-series average of the monthly
     equal-weighted excess return, with a time-series t-statistic.

Output: meme_stock_analysis/v13_outputs/bhar_alt_specs.csv
  horizon_months, mean_bhar, conventional_t, skew, skew_adj_t,
  bootstrap_p, fm_monthly_abn_ret, fm_t

Run: python3 bhar_alt_specs.py --allow-run
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _base as B  # noqa: E402
from _guard import check_run_guard  # noqa: E402

RNG = np.random.default_rng(20260603)
N_BOOT = 10000


def skew_adjusted_t(x):
    """Johnson (1978) skewness-adjusted t, as used in Lyon-Barber-Tsai (1999).
    t_sa = sqrt(n) * (S + (1/3) g S^2 + (1/(6n)) g),  S = mean/std, g = skew."""
    x = np.asarray(x, float)
    n = len(x)
    s = x.std(ddof=1)
    S = x.mean() / s
    g = np.sum((x - x.mean()) ** 3) / (n * s ** 3)  # sample skewness coeff
    return np.sqrt(n) * (S + (g * S ** 2) / 3 + g / (6 * n)), g


def bootstrap_p(x):
    """Two-sided bootstrap p-value for H0: mean BHAR = 0.
    Center resampled means on 0 and measure tail probability of observed mean."""
    x = np.asarray(x, float)
    n = len(x)
    obs = x.mean()
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot_means = x[idx].mean(axis=1) - obs  # center under H0
    p = np.mean(np.abs(boot_means) >= abs(obs))
    return p


def fama_macbeth_monthly():
    """Monthly cross-sectional mean abnormal return (meme excess minus matched
    control excess proxied by the EW calendar-time abnormal series), averaged
    Fama-MacBeth style with a time-series t-stat."""
    cal = pd.read_csv(f"{B.DATA}/analysis_calendar_time.csv")
    # monthly abnormal return = portfolio return minus risk-free (excess);
    # FM estimate = time-series mean, t = mean / (std / sqrt(T))
    r = cal["excess_ret"].values
    T = len(r)
    mean = r.mean()
    t = mean / (r.std(ddof=1) / np.sqrt(T))
    return mean * 100, t


def main():
    check_run_guard("bhar_alt_specs.py")
    B.ensure_out()
    matches = B.build_matches(B.screen().permno.tolist())
    panel = B.bhar_panel(matches)

    fm_ret, fm_t = fama_macbeth_monthly()
    rows = []
    for h in [36, 48]:
        x = panel[panel.horizon_mo == h].dropna(subset=["bhar"])["bhar"].values
        mean = x.mean() * 100
        conv_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        sat, g = skew_adjusted_t(x)
        bp = bootstrap_p(x)
        rows.append([h, round(mean, 1), round(conv_t, 2), round(g, 2),
                     round(sat, 2), round(bp, 4), round(fm_ret, 2),
                     round(fm_t, 2)])
        print(f"  {h}mo: mean={mean:6.1f}%  conv_t={conv_t:5.2f}  skew={g:5.2f}"
              f"  skew_adj_t={sat:5.2f}  boot_p={bp:.4f}")
    out = pd.DataFrame(rows, columns=[
        "horizon_months", "mean_bhar", "conventional_t", "skew",
        "skew_adj_t", "bootstrap_p", "fm_monthly_abn_ret", "fm_t"])
    out.to_csv(f"{B.OUT}/bhar_alt_specs.csv", index=False)
    print(f"\nFama-MacBeth monthly abnormal return: {fm_ret:.2f}%/mo "
          f"(t={fm_t:.2f})")
    print("Wrote bhar_alt_specs.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
