"""
Query 3b: 2025 CRSP Daily Returns (CIZ / v2 schema) - horizon extension
========================================================================
The SIZ pull (crsp.dsf) ends 2024-12-31. crsp.dsf_v2 (CIZ) reaches 2025-12-31.
This script pulls only the 2025 portion for the meme cohort and the control
universe, to be appended onto the Feb-2021..Dec-2024 series (no schema splice for
delistings: every meme delisting occurred before 2025, so 2025 rows are surviving
common stocks with plain total returns; CIZ dlyret already includes any rare 2025
delisting return in-series).

Run on WRDS. Requires wrds_01_screen.csv.
Outputs (this directory):
    wrds_03_returns_meme_2025.csv      permno,date,ret,prc,shrout,vol,ticker
    wrds_03_returns_controls_2025.csv  permno,date,ret
    wrds_03_delflag_2025.csv           QC: any meme permno flagged delisted in 2025
"""
import wrds
import pandas as pd

meme = pd.read_csv('wrds_01_screen.csv')
permnos = meme['permno'].tolist()
plist = ','.join(str(int(p)) for p in permnos)
print(f"Meme cohort: {len(permnos)} permnos")

db = wrds.Connection()

# ── Meme cohort 2025 daily returns (permno-only filter; keep all cohort names) ──
print("Pulling 2025 meme returns (CIZ dsf_v2)...")
meme25 = db.raw_sql(f"""
    SELECT permno, dlycaldt AS date, dlyret AS ret, dlyprc AS prc,
           shrout, dlyvol AS vol, ticker, dlydelflg
    FROM crsp.dsf_v2
    WHERE permno IN ({plist})
      AND dlycaldt BETWEEN '2025-01-01' AND '2025-12-31'
    ORDER BY permno, dlycaldt
""", date_cols=['date'])
print(f"  meme 2025 rows: {len(meme25):,}; permnos with 2025 data: {meme25['permno'].nunique()}")

# QC: any meme name that delisted during 2025 (dlyret already includes the delist return)
delflag = meme25[meme25['dlydelflg'] == 'Y'][['permno', 'date', 'ret', 'ticker', 'dlydelflg']]
if len(delflag):
    print("  meme names with a 2025 delisting flag (return is in-series):")
    print(delflag.to_string())
else:
    print("  no meme names delisted in 2025 (all survivors).")
delflag.to_csv('wrds_03_delflag_2025.csv', index=False)
meme25.drop(columns=['dlydelflg']).to_csv('wrds_03_returns_meme_2025.csv', index=False)

# ── Control universe 2025 daily returns (common stock, major exchanges) ──
print("Pulling 2025 control-universe returns (this is the slow one)...")
ctrl25 = db.raw_sql("""
    SELECT permno, dlycaldt AS date, dlyret AS ret
    FROM crsp.dsf_v2
    WHERE dlycaldt BETWEEN '2025-01-01' AND '2025-12-31'
      AND sharetype = 'NS'
      AND securitytype = 'EQTY'
      AND securitysubtype = 'COM'
      AND primaryexch IN ('N', 'A', 'Q')
      AND dlyret IS NOT NULL
    ORDER BY permno, dlycaldt
""", date_cols=['date'])
print(f"  control 2025 rows: {len(ctrl25):,}; permnos: {ctrl25['permno'].nunique():,}")
ctrl25.to_csv('wrds_03_returns_controls_2025.csv', index=False)

print("\nDONE. Three files written:")
for f in ['wrds_03_returns_meme_2025.csv', 'wrds_03_returns_controls_2025.csv', 'wrds_03_delflag_2025.csv']:
    print("  ", f)
db.close()
