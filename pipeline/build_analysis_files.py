#!/usr/bin/env python3
"""
build_analysis_files.py - Build the analysis files from the WRDS extracts.

Methodology:
  Matching: Lyon, Barber and Tsai (1999). Same 1-digit SIC, control mktcap in
            [0.3x, 2.0x] of the sample firm's mktcap, nearest book-to-market,
            without replacement.
  BHAR:     Start 2021-02-04. Horizons 6/12/24/36/48/58 months. Buy-and-hold
            return of the sample firm minus its matched control. Shumway (1997)
            -30% for performance-related delistings (codes 400-599) with missing
            dlret.
  Cal-time: Start 2021-02-01. Compound each stock's daily returns to monthly,
            then equal-weight the cross-section; regress excess return on factors.
  Dilution: first vs last shares-outstanding observation; >20% growth = issuer.

Outputs (written to wrds_data/):
  analysis_bhar_matched.csv     permno, ticker, horizon_mo, bhar, meme_bhr, ctrl_bhr
  analysis_calendar_time.csv    ym, port_ret, rf, excess_ret, mktrf, smb, hml, rmw, cma, umd
  analysis_dilution.csv         permno, ticker, shrout_first, shrout_last, pct_change, is_issuer
  analysis_matched_pairs.csv    permno, ticker, ctrl_permno, ctrl_sic1, size_ratio, bm_diff

Run: python3 build_analysis_files.py
"""
import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "meme_stock_analysis", "wrds_data")

BHAR_START = pd.Timestamp("2021-02-04")
CALTIME_START = pd.Timestamp("2021-02-01")
HORIZONS = [6, 12, 24, 36, 48, 58]
SAMPLE_END = pd.Timestamp("2025-12-31")


def load():
    screen = pd.read_csv(f"{DATA}/wrds_01_screen.csv")
    meme_ccm = pd.read_csv(f"{DATA}/wrds_02_ccm_meme.csv")
    full_ccm = pd.read_csv(f"{DATA}/wrds_02_ccm_full.csv")
    rm = pd.read_csv(f"{DATA}/wrds_03_returns_meme.csv", parse_dates=["date"])
    de = pd.read_csv(f"{DATA}/wrds_03_delisting.csv")
    sh = pd.read_csv(f"{DATA}/wrds_04_shares_outstanding.csv", parse_dates=["date"])
    return screen, meme_ccm, full_ccm, rm, de, sh


def delist_map(de):
    """permno -> (delist_date, delist_ret) applying Shumway (1997).
    Performance-related codes 400-599: use dlret if present else -0.30.
    Other codes (mergers 200s, exchange 300s): use dlret if present else 0.
    Code 100 = still listed, skip."""
    out = {}
    for _, r in de.iterrows():
        code = r.get("dlstcd")
        if pd.isna(code) or int(code) == 100:
            continue
        code = int(code)
        dlret = r.get("dlret_adj")
        if pd.isna(dlret):
            dlret = r.get("dlret")
        if pd.isna(dlret):
            dlret = -0.30 if 400 <= code <= 599 else 0.0
        out[int(r["permno"])] = (pd.Timestamp(r["dlstdt"]), float(dlret))
    return out


def buy_hold(rets, start, end, dmap, permno):
    """Compound (1+ret) over [start, end]. If the firm delists at/<=end and
    after start, append the delisting return as the final factor."""
    s = rets[(rets["date"] >= start) & (rets["date"] <= end)]
    s = s[s["ret"].notna()]  # CRSP missing returns: skip day (don't poison prod)
    if s.empty:
        return None
    bhr = float(np.prod(1.0 + s["ret"].values)) - 1.0
    if permno in dmap:
        ddate, dret = dmap[permno]
        last = s["date"].max()
        # delisting falls within window and after the last observed return
        if start <= ddate <= end and ddate > last:
            bhr = (1.0 + bhr) * (1.0 + dret) - 1.0
    return bhr


def build_matches(screen, meme_ccm, full_ccm, control_permnos):
    """LBT matching. Returns dict meme_permno -> ctrl_permno and a pairs DF.

    Requires same 1-digit SIC + control mktcap in [0.3x, 2.0x]. Among
    candidates: rank by nearest book-to-market when the meme stock HAS a B/M
    (22 of 44 do); fall back to nearest size ratio when B/M is missing
    (negative/missing book equity firms). Without replacement."""
    meme = meme_ccm.set_index("permno")
    # control pool: non-meme firms with size+SIC AND return data. B/M optional
    # (only needed when the meme stock being matched has a B/M to compare to).
    pool = full_ccm[(full_ccm["is_meme"] == 0)].copy()
    pool = pool[pool["permno"].isin(control_permnos)]
    pool = pool.dropna(subset=["mktcap_m", "sic"])
    pool["sic1"] = pool["sic"].astype(str).str[0]
    used = set()
    matches, rows = {}, []
    # process meme stocks largest-CAR first (deterministic order)
    order = screen.sort_values("car_pct", ascending=False)["permno"].tolist()
    for mp in order:
        if mp not in meme.index:
            continue
        m = meme.loc[mp]
        if pd.isna(m["mktcap_m"]) or pd.isna(m["sic"]):
            continue
        sic1 = str(m["sic"])[0]
        lo, hi = 0.3 * m["mktcap_m"], 2.0 * m["mktcap_m"]
        cand = pool[(pool["sic1"] == sic1)
                    & (pool["mktcap_m"] >= lo)
                    & (pool["mktcap_m"] <= hi)
                    & (~pool["permno"].isin(used))].copy()
        if cand.empty:
            continue
        cand["size_ratio"] = cand["mktcap_m"] / m["mktcap_m"]
        if pd.notna(m["bm_ratio"]):
            # nearest B/M among controls that have a B/M; else nearest size
            with_bm = cand.dropna(subset=["bm_ratio"])
            if not with_bm.empty:
                with_bm = with_bm.assign(
                    bm_diff=(with_bm["bm_ratio"] - m["bm_ratio"]).abs())
                best = with_bm.sort_values("bm_diff").iloc[0]
                bm_diff = float(best["bm_diff"])
            else:
                best = cand.assign(sr=(cand["size_ratio"] - 1).abs()) \
                           .sort_values("sr").iloc[0]
                bm_diff = np.nan
        else:
            best = cand.assign(sr=(cand["size_ratio"] - 1).abs()) \
                       .sort_values("sr").iloc[0]
            bm_diff = np.nan
        cp = int(best["permno"])
        used.add(cp)
        matches[mp] = cp
        rows.append({"permno": mp, "ticker": m.get("ticker", ""),
                     "ctrl_permno": cp, "ctrl_sic1": sic1,
                     "size_ratio": best["size_ratio"], "bm_diff": bm_diff})
    return matches, pd.DataFrame(rows)


def main():
    screen, meme_ccm, full_ccm, rm, de, sh = load()
    dmap = delist_map(de)
    tick = dict(zip(screen["permno"], screen["ticker"]))

    # control returns: stream by chunks to keep memory sane
    print("Loading control returns (107MB)...")
    cr = pd.read_csv(f"{DATA}/wrds_03_returns_controls.csv", parse_dates=["date"])
    control_permnos = set(cr["permno"].unique())
    print(f"  {len(control_permnos)} control permnos with return data")

    # ---- Matching ----
    matches, pairs = build_matches(screen, meme_ccm, full_ccm, control_permnos)
    print(f"Matched {len(matches)} / {len(screen)} meme stocks")
    pairs.to_csv(f"{DATA}/analysis_matched_pairs.csv", index=False)

    # index returns by permno for speed
    rm_g = {p: d.sort_values("date") for p, d in rm.groupby("permno")}
    cr_g = {p: d.sort_values("date") for p, d in cr.groupby("permno")}

    # ---- Table 1: BHAR matched ----
    rows = []
    for mp, cp in matches.items():
        meme_r = rm_g.get(mp)
        ctrl_r = cr_g.get(cp)
        if meme_r is None or ctrl_r is None:
            continue
        for h in HORIZONS:
            end = BHAR_START + relativedelta(months=h)
            mbhr = buy_hold(meme_r, BHAR_START, end, dmap, mp)
            cbhr = buy_hold(ctrl_r, BHAR_START, end, dmap, cp)
            if mbhr is None or cbhr is None:
                continue
            rows.append({"permno": mp, "ticker": tick.get(mp, ""),
                         "horizon_mo": h, "bhar": mbhr - cbhr,
                         "meme_bhr": mbhr, "ctrl_bhr": cbhr})
    bhar = pd.DataFrame(rows)
    bhar.to_csv(f"{DATA}/analysis_bhar_matched.csv", index=False)
    print(f"BHAR rows: {len(bhar)}; n@48mo = {len(bhar[bhar.horizon_mo==48])}")

    # ---- Table 2: calendar-time portfolio (EW) ----
    # compound each meme stock's daily returns to monthly, then EW cross-section
    rm2 = rm[(rm["date"] >= CALTIME_START) & (rm["date"] <= SAMPLE_END)].copy()
    rm2["ym"] = rm2["date"].dt.to_period("M")
    # monthly compound per permno
    monthly = (rm2.groupby(["permno", "ym"])["ret"]
               .apply(lambda x: np.prod(1.0 + x.values) - 1.0)
               .reset_index(name="mret"))
    port = monthly.groupby("ym")["mret"].mean().reset_index(name="port_ret")
    # factors monthly (compound daily FF5+MOM)
    ff = pd.read_csv(f"{DATA}/wrds_03_ff5mom.csv", parse_dates=["date"])
    ff = ff[(ff["date"] >= CALTIME_START) & (ff["date"] <= SAMPLE_END)].copy()
    ff["ym"] = ff["date"].dt.to_period("M")
    fac_cols = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    ffm = (ff.groupby("ym")[fac_cols]
           .apply(lambda d: pd.Series({c: np.prod(1.0 + d[c].values) - 1.0
                                       for c in fac_cols})).reset_index())
    rfm = ff.groupby("ym")["rf"].apply(lambda x: np.prod(1.0 + x.values) - 1.0).reset_index(name="rf")
    cal = port.merge(ffm, on="ym").merge(rfm, on="ym")
    cal["excess_ret"] = cal["port_ret"] - cal["rf"]
    cal.to_csv(f"{DATA}/analysis_calendar_time.csv", index=False)
    print(f"Calendar-time months: {len(cal)}")

    # ---- Table 4: dilution ----
    rows = []
    for p, d in sh.groupby("permno"):
        d = d.sort_values("date")
        first, last = d["shrout"].iloc[0], d["shrout"].iloc[-1]
        if first and first > 0:
            pct = (last - first) / first
            rows.append({"permno": p, "ticker": tick.get(p, ""),
                         "shrout_first": first, "shrout_last": last,
                         "pct_change": pct, "is_issuer": int(pct > 0.20)})
    dil = pd.DataFrame(rows)
    dil.to_csv(f"{DATA}/analysis_dilution.csv", index=False)
    print(f"Dilution rows: {len(dil)}; issuers: {dil.is_issuer.sum()}")
    print("DONE. Wrote analysis_*.csv to", DATA)


if __name__ == "__main__":
    main()
