#!/usr/bin/env python3
"""
sector_split.py - Sector composition and biotech-excluded robustness (Table 7, Appendix A).

Outputs (meme_stock_analysis/v13_outputs/):
  sample_list.csv          permno, ticker, comnam, siccd, sector, car_pct,
                           mktcap_m, bhar48, is_biotech
  sector_composition.csv   sector, n, pct
  biotech_excluded.csv     group, n, bhar48, t48   (full vs ex-biotech)

Run: python3 sector_split.py --allow-run
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _base as B  # noqa: E402
from _guard import check_run_guard  # noqa: E402

# pharma/biotech + commercial biological research
BIOTECH_SIC = {2833, 2834, 2835, 2836, 8731}


def sic_division(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if 100 <= s <= 999:
        return "Agriculture/Forestry/Fishing"
    if 1000 <= s <= 1499:
        return "Mining"
    if 1500 <= s <= 1799:
        return "Construction"
    if 2000 <= s <= 3999:
        return "Manufacturing"
    if 4000 <= s <= 4999:
        return "Transportation/Communications/Utilities"
    if 5000 <= s <= 5199:
        return "Wholesale Trade"
    if 5200 <= s <= 5999:
        return "Retail Trade"
    if 6000 <= s <= 6799:
        return "Finance/Insurance/Real Estate"
    if 7000 <= s <= 8999:
        return "Services"
    return "Public Admin/Other"


def best_sic(row):
    """Prefer Compustat SIC (CCM); fall back to CRSP siccd. CRSP codes 9999
    ('non-classifiable') are common here, but Compustat populates the real
    industry for most (e.g. TLRY cannabis, VIR/ACRS biotech)."""
    cs = row.get("sic_ccm")
    if pd.notna(cs) and int(cs) != 9999:
        return int(cs)
    return int(row["siccd"]) if pd.notna(row["siccd"]) else 9999


def main():
    check_run_guard("sector_split.py")
    B.ensure_out()
    scr = B.screen().copy()
    ccm = B.meme_ccm()[["permno", "sic"]].rename(columns={"sic": "sic_ccm"})
    scr = scr.merge(ccm, on="permno", how="left")
    scr["sic_best"] = scr.apply(best_sic, axis=1)
    matches = B.build_matches(scr.permno.tolist())
    panel = B.bhar_panel(matches)
    b48 = panel[panel.horizon_mo == 48].dropna(subset=["bhar"]) \
        .set_index("permno")["bhar"]

    scr["sector"] = scr["sic_best"].apply(
        lambda s: "Unclassified (SIC 9999)" if s == 9999 else sic_division(s))
    scr["is_biotech"] = scr["sic_best"].apply(lambda s: int(s in BIOTECH_SIC))
    scr["bhar48"] = scr["permno"].map(lambda p: b48.get(int(p)))

    sample = scr[["permno", "ticker", "comnam", "sic_best", "sector",
                  "car_pct", "mktcap_m", "bhar48", "is_biotech"]] \
        .rename(columns={"sic_best": "sic"}) \
        .sort_values("car_pct", ascending=False)
    sample.to_csv(f"{B.OUT}/sample_list.csv", index=False)

    comp = (scr.groupby("sector").size().reset_index(name="n")
            .sort_values("n", ascending=False))
    comp["pct"] = (comp["n"] / comp["n"].sum() * 100).round(1)
    comp.to_csv(f"{B.OUT}/sector_composition.csv", index=False)
    print("Sector composition:")
    print(comp.to_string(index=False))

    rows = []
    full = b48.dropna().values
    fm, ft, fn = B.tstat(full)
    rows.append(["Full sample", fn, round(fm * 100, 1), round(ft, 2)])
    ex_perms = scr[scr.is_biotech == 0]["permno"].astype(int)
    ex = [b48[p] for p in ex_perms if p in b48.index and pd.notna(b48[p])]
    em, et, en = B.tstat(ex)
    rows.append(["Ex-biotech", en, round(em * 100, 1), round(et, 2)])
    bio = scr[scr.is_biotech == 1]
    out = pd.DataFrame(rows, columns=["group", "n", "bhar48", "t48"])
    out.to_csv(f"{B.OUT}/biotech_excluded.csv", index=False)
    print(f"\nBiotech/pharma stocks in sample: {bio.is_biotech.sum()} "
          f"({', '.join(bio.ticker.tolist())})")
    print("Biotech-excluded robustness (48mo BHAR):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
