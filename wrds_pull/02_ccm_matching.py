"""
Query 2: CCM Matching Characteristics
=======================================
For each meme stock identified in Query 1, pull Dec 2020 fundamentals
from CRSP/Compustat Merged for matched-control selection.

Also pulls the FULL universe of potential controls (same characteristics)
so we can do Lyon-Barber-Tsai (1999) matching locally.

Run on WRDS. Requires wrds_01_screen.csv from Query 1.
Output: wrds_02_ccm.csv, wrds_02_control_universe.csv
"""

import wrds
import pandas as pd
import numpy as np

# ── Load meme stock sample from Query 1 ──
meme = pd.read_csv('wrds_01_screen.csv')
meme_permnos = meme['permno'].tolist()
print(f"Meme stock sample: {len(meme_permnos)} PERMNOs")

# ── Connect to WRDS ──
db = wrds.Connection()

# ── Step 1: Pull Compustat annual fundamentals (fiscal year ending in 2020) ──
# We want: total assets (at), book equity (ceq + txdb - pstk), SIC code
# Linked to CRSP via CCM link table

print("Pulling Compustat fundamentals for fiscal year 2020...")

comp = db.raw_sql("""
    SELECT a.gvkey, a.datadate, a.fyear,
           a.at,           /* total assets */
           a.ceq,          /* common equity */
           a.txdb,         /* deferred taxes */
           a.pstk,         /* preferred stock */
           a.csho,         /* shares outstanding */
           a.prcc_f,       /* price at fiscal year end */
           a.sich,         /* historical SIC code */
           a.revt,         /* revenue */
           a.ni,           /* net income */
           b.lpermno AS permno,
           b.linkprim, b.linktype
    FROM comp.funda AS a
    INNER JOIN crsp.ccmxpf_lnkhist AS b
        ON a.gvkey = b.gvkey
        AND b.linktype IN ('LU', 'LC')
        AND b.linkprim IN ('P', 'C')
        AND a.datadate >= b.linkdt
        AND (a.datadate <= b.linkenddt OR b.linkenddt IS NULL)
    WHERE a.datafmt = 'STD'
      AND a.consol = 'C'
      AND a.indfmt = 'INDL'
      AND a.popsrc = 'D'
      AND a.fyear = 2020
""", date_cols=['datadate'])

print(f"  Compustat rows (FY2020): {len(comp):,}")
print(f"  Unique PERMNOs with Compustat data: {comp['permno'].nunique():,}")

# ── Step 2: Pull CRSP market cap as of Dec 31, 2020 ──
print("Pulling CRSP market cap at Dec 31, 2020...")

mktcap = db.raw_sql("""
    SELECT permno, date, abs(prc) * shrout / 1000 AS mktcap_m,
           abs(prc) AS price, shrout
    FROM crsp.dsf
    WHERE date BETWEEN '2020-12-28' AND '2020-12-31'
      AND shrcd IN (10, 11)
      AND exchcd IN (1, 2, 3)
    ORDER BY permno, date DESC
""", date_cols=['date'])

# Keep last available trading day per permno
mktcap = mktcap.sort_values(['permno', 'date']).groupby('permno').last().reset_index()
print(f"  Market caps: {len(mktcap):,}")

# ── Step 3: Compute book-to-market ──
# Book equity = CEQ + TXDB - PSTK (Fama-French definition)
comp['be'] = comp['ceq'].fillna(0) + comp['txdb'].fillna(0) - comp['pstk'].fillna(0)
comp.loc[comp['be'] <= 0, 'be'] = np.nan  # drop negative book equity

# Merge with market cap
chars = comp.merge(mktcap[['permno', 'mktcap_m', 'price']], on='permno', how='inner')
chars['bm'] = chars['be'] / (chars['mktcap_m'] * 1000 / chars['csho'].where(chars['csho'] > 0))
# Simpler BM: book equity / market equity
chars['bm2'] = (chars['be'] * 1e6) / (chars['mktcap_m'] * 1e6)  # both in $M

# Use SIC from Compustat (sich) or fall back to CRSP
chars['sic'] = chars['sich']

# ── Step 4: Flag meme stocks vs control universe ──
chars['is_meme'] = chars['permno'].isin(meme_permnos).astype(int)

# Keep relevant columns
output_cols = ['permno', 'gvkey', 'datadate', 'sic', 'mktcap_m', 'price',
               'at', 'be', 'bm2', 'revt', 'ni', 'csho', 'is_meme']
chars_out = chars[output_cols].copy()
chars_out.columns = ['permno', 'gvkey', 'datadate', 'sic', 'mktcap_m', 'price',
                     'total_assets', 'book_equity', 'bm_ratio', 'revenue',
                     'net_income', 'shares_out', 'is_meme']

# ── Step 5: Save ──
# Meme stocks only
meme_chars = chars_out[chars_out['is_meme'] == 1]
print(f"\nMeme stocks with Compustat match: {len(meme_chars)}")
print(meme_chars[['permno', 'sic', 'mktcap_m', 'bm_ratio', 'total_assets']].to_string())

# Full control universe
control = chars_out[chars_out['is_meme'] == 0]
print(f"Control universe size: {len(control):,}")

meme_chars.to_csv('wrds_02_ccm_meme.csv', index=False)
chars_out.to_csv('wrds_02_ccm_full.csv', index=False)
print(f"\nSaved wrds_02_ccm_meme.csv ({len(meme_chars)} rows)")
print(f"Saved wrds_02_ccm_full.csv ({len(chars_out)} rows)")

# ── Step 6: Also pull CRSP exchange/SIC for any meme stocks missing from Compustat ──
missing = set(meme_permnos) - set(meme_chars['permno'].tolist())
if missing:
    print(f"\nWARNING: {len(missing)} meme stocks not in Compustat. PERMNOs: {missing}")
    print("These may be micro-caps without Compustat coverage. Will use CRSP-only data for matching.")

    crsp_only = db.raw_sql(f"""
        SELECT DISTINCT a.permno, b.siccd, b.ticker, b.comnam,
               abs(a.prc) * a.shrout / 1000 AS mktcap_m
        FROM crsp.dsf AS a
        LEFT JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
        WHERE a.permno IN ({','.join(str(p) for p in missing)})
          AND a.date BETWEEN '2020-12-28' AND '2020-12-31'
    """, date_cols=[])
    crsp_only.to_csv('wrds_02_ccm_crsp_only.csv', index=False)
    print(f"Saved wrds_02_ccm_crsp_only.csv ({len(crsp_only)} rows)")

db.close()
