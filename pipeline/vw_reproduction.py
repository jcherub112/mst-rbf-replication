#!/usr/bin/env python3
"""
vw_reproduction.py - Equal- vs value-weighted BHAR and calendar-time alphas
(Tables 5 and 6) and the equity-issuance split (Section 3.7).

VW BHAR weights each stock's BHAR by its 2021-02-01 market cap. VW calendar-time
weights each month's cross-section by beginning-of-month market cap. Calendar-time
standard errors are Newey-West with 12 lags.

Outputs (meme_stock_analysis/v13_outputs/):
  table1_bhar_vw.csv      horizon_months, ew_bhar, ew_t, vw_bhar, vw_t, n
  table2_caltime_vw.csv   model, ew_alpha, ew_t, vw_alpha, vw_t
  table4_issuance_vw.csv  group, n, ew_bhar_48, vw_bhar_48

Run: python3 vw_reproduction.py --allow-run
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _base as B  # noqa: E402
from _guard import check_run_guard  # noqa: E402


def weighted_mean_t(vals, weights):
    """Weighted mean and an approximate t-stat (weighted mean / weighted SE)."""
    v = np.asarray(vals, float)
    w = np.asarray(weights, float)
    keep = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    v, w = v[keep], w[keep]
    if len(v) < 2:
        return np.nan, np.nan, len(v)
    w = w / w.sum()
    wm = np.sum(w * v)
    # variance of a weighted mean with reliability weights
    var = np.sum(w * (v - wm) ** 2) * (len(v) / (len(v) - 1))
    se = np.sqrt(var * np.sum(w ** 2))
    return wm, (wm / se if se > 0 else np.nan), len(v)


def vw_calendar_time(weight_by_cap):
    """Monthly meme portfolio returns; EW or VW by beginning-of-month cap."""
    rm = B.returns_meme().copy()
    rm = rm[(rm["date"] >= B.CALTIME_START) & (rm["date"] <= B.SAMPLE_END)]
    rm["ym"] = rm["date"].dt.to_period("M")
    rm["cap"] = rm["prc"].abs() * rm["shrout"]
    recs = []
    for (p, ym), d in rm.groupby(["permno", "ym"]):
        d = d.sort_values("date")
        mret = np.prod(1.0 + d["ret"].dropna().values) - 1.0
        recs.append((p, ym, mret, d["cap"].iloc[0]))
    mo = pd.DataFrame(recs, columns=["permno", "ym", "mret", "cap"])
    rows = []
    for ym, g in mo.groupby("ym"):
        if weight_by_cap and g["cap"].fillna(0).sum() > 0:
            w = g["cap"].fillna(0).values
            w = w / w.sum()
        else:
            w = np.ones(len(g)) / len(g)
        rows.append((ym, float(np.sum(w * g["mret"].values))))
    port = pd.DataFrame(rows, columns=["ym", "port_ret"])
    cal = port.merge(B.factors_monthly(), on="ym")
    cal["excess_ret"] = cal["port_ret"] - cal["rf"]
    return cal


def alpha_t(cal, cols):
    y = cal["excess_ret"].values
    X = np.column_stack([np.ones(len(y))] + [cal[c].values for c in cols])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = B.newey_west_se(X, resid, 12)
    return beta[0] * 100, beta[0] / se[0]


def main():
    check_run_guard("vw_reproduction.py")
    B.ensure_out()
    matches = B.build_matches(B.screen().permno.tolist())
    panel = B.bhar_panel(matches)
    caps = B.meme_market_caps()

    # ---- Table 1 VW ----
    rows = []
    for h in B.HORIZONS:
        sub = panel[panel.horizon_mo == h].dropna(subset=["bhar"])
        ew_m, ew_t, n = B.tstat(sub["bhar"].values)
        w = [caps.get(int(p), np.nan) for p in sub["permno"]]
        vw_m, vw_t, _ = weighted_mean_t(sub["bhar"].values, w)
        rows.append([h, round(ew_m * 100, 1), round(ew_t, 2),
                     round(vw_m * 100, 1), round(vw_t, 2), n])
    t1 = pd.DataFrame(rows, columns=["horizon_months", "ew_bhar", "ew_t",
                                     "vw_bhar", "vw_t", "n"])
    t1.to_csv(f"{B.OUT}/table1_bhar_vw.csv", index=False)
    print("Table 1 (EW vs VW BHAR):")
    print(t1.to_string(index=False))

    # ---- Table 2 VW ----
    cal_ew, cal_vw = vw_calendar_time(False), vw_calendar_time(True)
    specs = {"CAPM": ["mktrf"], "FF3": ["mktrf", "smb", "hml"],
             "FF5+MOM": ["mktrf", "smb", "hml", "rmw", "cma", "umd"]}
    rows = []
    for name, cols in specs.items():
        ea, et = alpha_t(cal_ew, cols)
        va, vt = alpha_t(cal_vw, cols)
        rows.append([name, round(ea, 2), round(et, 2),
                     round(va, 2), round(vt, 2)])
    t2 = pd.DataFrame(rows, columns=["model", "ew_alpha", "ew_t",
                                     "vw_alpha", "vw_t"])
    t2.to_csv(f"{B.OUT}/table2_caltime_vw.csv", index=False)
    print("\nTable 2 (EW vs VW calendar-time alpha %/mo):")
    print(t2.to_string(index=False))

    # ---- Table 4 VW ----
    dil = pd.read_csv(f"{B.DATA}/analysis_dilution.csv")
    b48 = panel[panel.horizon_mo == 48].dropna(subset=["bhar"]) \
        .set_index("permno")["bhar"]
    rows = []
    for label, flag in [("Issuers", 1), ("Non-issuers", 0)]:
        perms = [int(p) for p in dil[dil.is_issuer == flag].permno
                 if int(p) in b48.index]
        vals = [b48[p] for p in perms]
        w = [caps.get(p, np.nan) for p in perms]
        ew_m, _, n = B.tstat(vals)
        vw_m, _, _ = weighted_mean_t(vals, w)
        rows.append([label, n, round(ew_m * 100, 1), round(vw_m * 100, 1)])
    t4 = pd.DataFrame(rows, columns=["group", "n", "ew_bhar_48", "vw_bhar_48"])
    t4.to_csv(f"{B.OUT}/table4_issuance_vw.csv", index=False)
    print("\nTable 4 (issuance EW vs VW, 48mo BHAR):")
    print(t4.to_string(index=False))


if __name__ == "__main__":
    main()
