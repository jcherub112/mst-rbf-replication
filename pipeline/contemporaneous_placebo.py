#!/usr/bin/env python3
"""
contemporaneous_placebo.py - Same-window benchmark and long-short (Table 2).

Compares the meme cohort (event-window CAR above 100%) with every other stock in
the same January 2021 screen universe whose CAR is between 50% and 100%. Holding
the window fixed holds the market regime fixed. Both groups are matched with the
same rules (_base.build_matches), and a meme-minus-benchmark calendar-time
long-short is estimated on FF5+MOM with Newey-West standard errors.

Run:  python3 contemporaneous_placebo.py
Out:  meme_stock_analysis/v13_outputs/contemp_placebo_*.csv  + printed summary
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _base as B  # noqa: E402

try:
    from scipy import stats
except Exception:
    stats = None


def estimands(x):
    x = np.asarray([v for v in x if pd.notna(v)], float)
    out = {"n": len(x), "raw": x.mean() * 100, "median": np.median(x) * 100,
           "pct_under": (x < 0).mean() * 100}
    if stats is not None:
        out["winsor"] = float(np.mean(stats.mstats.winsorize(x, limits=[0.05, 0.05]))) * 100
    else:
        lo, hi = np.percentile(x, [5, 95]); out["winsor"] = np.clip(x, lo, hi).mean() * 100
    out["floor"] = np.clip(x, -1.0, None).mean() * 100
    return out


def monthly_port(permnos):
    """EW monthly portfolio return (event firms), Feb 2021 - Dec 2024."""
    recs = []
    for p in permnos:
        r = B.returns_for(int(p))
        if r is None or len(r) == 0:
            continue
        r = r[(r["date"] >= B.CALTIME_START) & (r["date"] <= B.SAMPLE_END)].copy()
        if r.empty:
            continue
        r["ym"] = r["date"].dt.to_period("M")
        m = r.groupby("ym")["ret"].apply(lambda s: np.prod(1 + s.values) - 1)
        recs.append(m.rename(int(p)))
    if not recs:
        return pd.DataFrame()
    wide = pd.concat(recs, axis=1)
    return wide.mean(axis=1).rename("port").to_frame()  # EW across available firms


def alpha(port, cols, nlags):
    cal = port.merge(B.factors_monthly(), left_on=port.index.astype(str),
                     right_on=B.factors_monthly()["ym"].astype(str)) \
        if "ym" not in port else port
    return None  # placeholder; real impl below


def main():
    B.ensure_out()
    fu = B.screen_full_universe()
    meme = set(B.screen().permno.astype(int))
    # contemporaneous placebo: full-universe Jan-2021 >100% CAR, excluding the 44 meme names
    col_car = "car_pct" if "car_pct" in fu.columns else [c for c in fu.columns if "car" in c.lower()][0]
    n_over = int(((fu[col_car] > 100) & (~fu.permno.astype(int).isin(meme))).sum())
    print(f"KEY FACT: non-meme stocks with >100% CAR in the Jan-2021 window: {n_over} "
          f"(the 44 >100% names ARE the meme set; no contemporaneous extreme-winner placebo exists, "
          f"which is why the cross-decade placebo is needed).")
    # Same-window near-miss winners: non-meme, CAR in [LO,100). A contemporaneous,
    # same-regime benchmark of strong-but-not-extreme 2021 winners.
    LO = 50.0
    q = fu[(fu[col_car] >= LO) & (fu[col_car] <= 100) & (~fu.permno.astype(int).isin(meme))].permno.astype(int).tolist()
    q = sorted(set(q))
    print(f"Contemporaneous same-window benchmark: non-meme, CAR in [{LO:.0f}%,100%]: {len(q)} stocks")

    # match both samples identically (size + 1-digit SIC + B/M), exclude meme from controls
    m_match = B.build_matches(sorted(meme))
    p_match = B.build_matches(q, exclude=meme)
    m_panel = B.bhar_panel(m_match)
    p_panel = B.bhar_panel(p_match)
    meme48 = m_panel[m_panel.horizon_mo == 48].dropna(subset=["bhar"])
    plac48 = p_panel[p_panel.horizon_mo == 48].dropna(subset=["bhar"])
    print(f"matched+returns: meme {meme48.permno.nunique()}, contemp-placebo {plac48.permno.nunique()}")

    me, pl = estimands(meme48.bhar.values), estimands(plac48.bhar.values)
    print(f"\n{'estimand':12}{'MEME':>9}{'C-PLAC':>9}{'M-P':>8}")
    for k in ["raw", "winsor", "floor", "median", "pct_under"]:
        print(f"{k:12}{me[k]:>9.1f}{pl[k]:>9.1f}{me[k]-pl[k]:>8.1f}")
    if stats is not None:
        t, pt = stats.ttest_ind(meme48.bhar, plac48.bhar, equal_var=False)
        u, pu = stats.mannwhitneyu(meme48.bhar, plac48.bhar, alternative="two-sided")
        print(f"  Welch t={t:.2f} p={pt:.3f} | Mann-Whitney p={pu:.3f} "
              f"| n_meme={len(meme48)} n_plac={len(plac48)}")

    # ---- meme-minus-placebo calendar-time long-short (both Feb2021-Dec2024) ----
    mport = monthly_port(meme48.permno.tolist())
    pport = monthly_port(plac48.permno.tolist())
    fac = B.factors_monthly().copy()
    fac["ymk"] = fac["ym"].astype(str)
    j = mport.join(pport, lsuffix="_m", rsuffix="_p", how="inner")
    j.index = j.index.astype(str)
    j = j.join(fac.set_index("ymk"))
    j = j.dropna(subset=["port_m", "port_p", "mktrf"])
    ls = (j["port_m"] - j["port_p"]).values  # long meme, short contemp-placebo
    print(f"\nLong-short (meme - contemp-placebo) calendar-time, {len(ls)} months:")
    print(f"  mean monthly LS return = {ls.mean()*100:.2f}%")
    cols = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    have = [c for c in cols if c in j.columns]
    X = np.column_stack([np.ones(len(j))] + [j[c].values for c in have])
    y = ls - j["rf"].values if "rf" in j.columns else ls
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    for L in (3, 12):
        se = B.newey_west_se(X, resid, L)
        print(f"  FF5+MOM alpha = {beta[0]*100:.2f}%/mo  (NW {L} lags t={beta[0]/se[0]:.2f})")

    # save
    out = B.OUT
    pd.DataFrame([{"group": "meme", **me}, {"group": "contemp_placebo", **pl}]).to_csv(
        f"{out}/contemp_placebo_estimands.csv", index=False)
    plac48.to_csv(f"{out}/contemp_placebo_bhar48.csv", index=False)
    print(f"\nwrote {out}/contemp_placebo_estimands.csv + contemp_placebo_bhar48.csv")


if __name__ == "__main__":
    main()
