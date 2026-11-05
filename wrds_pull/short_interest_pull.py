"""
short_interest_pull.py - January 2021 short interest for the 118 same-window stocks.
================================================================================
Tests whether a measured squeeze characteristic (short interest) predicts
long-run reversal.

Universe: the 118 stocks clearing a 50% Jan-2021 event-window CAR screen
(44 with CAR>100% + 74 with CAR 50-100%).

Short interest is in `comp.sec_shortint` (gvkey-keyed). gvkey is linked to
permno by CUSIP, so the CRSP-Compustat link schema (crsp_a_ccm) is not required:
  crsp.dsenames (permno -> ncusip8)  x  comp.security (cusip -> gvkey).
Shares outstanding for the short-interest ratio come from crsp.dsf.

RUN (in the folder holding wrds_01_screen.csv):
    python3 short_interest_pull.py --test
    python3 short_interest_pull.py
OUTPUT: ./wrds_05_short_interest_118.csv  + a peak-SI summary by group.
"""
import os
import sys
import numpy as np
import pandas as pd
import wrds

CAND_PATHS = [
    "wrds_01_screen_full_universe.csv",
    os.path.expanduser("~/Documents/meme_stock_analysis/wrds_data/wrds_01_screen_full_universe.csv"),
    "../meme_stock_analysis/wrds_data/wrds_01_screen_full_universe.csv",
]
MEME_FILE = "wrds_01_screen.csv"
START, END = "2020-11-01", "2021-06-30"


def load_universe():
    for p in CAND_PATHS:
        if os.path.exists(p):
            fu = pd.read_csv(p)
            col = "car_pct" if "car_pct" in fu.columns else [c for c in fu.columns if "car" in c.lower()][0]
            sub = fu.loc[fu[col] >= 50, ["permno", "ticker", col]].copy()
            sub.columns = ["permno", "ticker", "car_pct"]
            meme = set(pd.read_csv(MEME_FILE).permno.astype(int)) if os.path.exists(MEME_FILE) else set()
            sub["meme_flag"] = sub.permno.astype(int).isin(meme).astype(int)
            print(f"Universe from {p}: {len(sub)} stocks (meme {int(sub.meme_flag.sum())}, "
                  f"near-miss {int((sub.meme_flag == 0).sum())})")
            return sub
    sys.exit("Could not find wrds_01_screen_full_universe.csv.")


def smoke_test():
    db = wrds.Connection()
    for t in ["comp.sec_shortint", "comp.security", "crsp.dsenames", "crsp.dsf"]:
        lib, tbl = t.split(".")
        try:
            n = db.raw_sql(f"SELECT * FROM {lib}.{tbl} LIMIT 1")
            print(f"  [OK ] {t} ({len(n.columns)} cols): {list(n.columns)[:8]}")
        except Exception as e:
            print(f"  [FAIL] {t} -> {str(e)[:80]}")
    db.close()


def main():
    if "--test" in sys.argv:
        smoke_test(); return
    uni = load_universe()
    permnos = ",".join(str(int(p)) for p in uni.permno)
    db = wrds.Connection()

    # permno -> ncusip8 (valid around the event)
    names = db.raw_sql(f"""SELECT permno, ncusip, namedt, nameendt FROM crsp.dsenames
        WHERE permno IN ({permnos}) AND ncusip IS NOT NULL AND ncusip <> ''""", date_cols=["namedt", "nameendt"])
    names = names[(names.namedt <= "2021-01-29") & (names.nameendt >= "2021-01-29")]
    names["cusip8"] = names.ncusip.str[:8]
    names = names[["permno", "cusip8"]].drop_duplicates("permno")
    cus = [c for c in names.cusip8.dropna().unique() if c]
    cl = ",".join("'" + c + "'" for c in cus)

    # cusip8 -> gvkey
    sec = db.raw_sql(f"""SELECT gvkey, cusip FROM comp.security
        WHERE substr(cusip,1,8) IN ({cl})""")
    sec["cusip8"] = sec.cusip.str[:8]
    g2c = sec[["gvkey", "cusip8"]].drop_duplicates()
    link = names.merge(g2c, on="cusip8", how="inner")   # permno, cusip8, gvkey
    print(f"Linked {link.permno.nunique()} of {len(uni)} permnos to a gvkey.")
    gvs = ",".join("'" + str(g) + "'" for g in link.gvkey.dropna().unique())

    # short interest (shares) by gvkey
    si = db.raw_sql(f"""SELECT gvkey, datadate AS date, shortint, shortintadj
        FROM comp.sec_shortint
        WHERE gvkey IN ({gvs}) AND datadate BETWEEN '{START}' AND '{END}'
        ORDER BY gvkey, datadate""", date_cols=["date"])
    if si.empty:
        print("No comp.sec_shortint rows returned; send me the --test output."); db.close(); return
    si = si.merge(link[["permno", "gvkey"]].drop_duplicates(), on="gvkey", how="left")

    # shares outstanding from crsp.dsf (merge_asof on date, per permno)
    dsf = db.raw_sql(f"""SELECT permno, date, shrout FROM crsp.dsf
        WHERE permno IN ({permnos}) AND date BETWEEN '{START}' AND '{END}'
        AND shrout IS NOT NULL""", date_cols=["date"])
    si = si.sort_values("date"); dsf = dsf.sort_values("date")
    si = pd.merge_asof(si, dsf, on="date", by="permno", direction="nearest")
    si["si_pct"] = si["shortint"] / (si["shrout"] * 1000.0) * 100.0   # shrout in thousands
    out = si.merge(uni, on="permno", how="left")[
        ["permno", "ticker", "meme_flag", "car_pct", "date", "shortint", "shrout", "si_pct"]]
    out.to_csv("wrds_05_short_interest_118.csv", index=False)
    print(f"\nWrote wrds_05_short_interest_118.csv ({len(out)} rows, {out.permno.nunique()} stocks)")

    peak = out.dropna(subset=["si_pct"]).groupby(["permno", "meme_flag"])["si_pct"].max().reset_index()
    for f, lbl in [(1, "meme (CAR>100%)"), (0, "near-miss (50-100%)")]:
        s = peak[peak.meme_flag == f].si_pct
        print(f"  {lbl:22} peak SI%%: median {s.median():.1f}  mean {s.mean():.1f}  n={len(s)}")
    db.close()


if __name__ == "__main__":
    main()
