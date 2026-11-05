"""
Query 3: CRSP Daily Returns + Delisting Returns for Meme Stock Sample
=======================================================================
Pulls daily returns from Feb 2021 through Mar 2026 for the meme stock sample,
INCLUDING delisting returns (critical for BBBY, EXPR, NAKD, etc.).

Also pulls returns for the full control universe (needed for matched-firm BHAR).

Run on WRDS. Requires wrds_01_screen.csv from Query 1.
Output: wrds_03_returns_meme.csv, wrds_03_returns_controls.csv, wrds_03_delisting.csv
"""

import wrds
import pandas as pd
import numpy as np

# ── Load meme stock sample ──
meme = pd.read_csv('wrds_01_screen.csv')
meme_permnos = meme['permno'].tolist()
print(f"Meme stock sample: {len(meme_permnos)} PERMNOs")

# ── Connect ──
db = wrds.Connection()

# ── Step 1: Daily returns for meme stocks, Feb 2021 – Mar 2026 ──
print("Pulling daily returns for meme stock sample...")

permno_list = ','.join(str(p) for p in meme_permnos)

meme_ret = db.raw_sql(f"""
    SELECT a.permno, a.date, a.ret, a.prc, a.shrout, a.vol,
           b.ticker, b.comnam, b.shrcd, b.exchcd
    FROM crsp.dsf AS a
    LEFT JOIN crsp.dsenames AS b
        ON a.permno = b.permno
        AND b.namedt <= a.date
        AND a.date <= b.nameendt
    WHERE a.permno IN ({permno_list})
      AND a.date BETWEEN '2021-02-01' AND '2026-03-31'
      AND a.shrcd IN (10, 11)
""", date_cols=['date'])

print(f"  Meme stock daily returns: {len(meme_ret):,} rows")
print(f"  PERMNOs with data: {meme_ret['permno'].nunique()}")

# Check last available date per stock (identifies early delisting)
last_dates = meme_ret.groupby(['permno', 'ticker'])['date'].max().reset_index()
last_dates.columns = ['permno', 'ticker', 'last_date']
print("\nLast trading date per meme stock:")
print(last_dates.sort_values('last_date').to_string())

# ── Step 2: Delisting returns ──
# CRITICAL: CRSP delisting returns capture the final return when a stock is
# removed from trading (bankruptcy, merger, going private, etc.)
print("\nPulling delisting returns...")

delist = db.raw_sql(f"""
    SELECT permno, dlstdt, dlstcd, dlret, dlprc, dlpdt
    FROM crsp.dsedelist
    WHERE permno IN ({permno_list})
      AND dlstdt >= '2021-01-01'
""", date_cols=['dlstdt', 'dlpdt'])

if len(delist) > 0:
    print(f"  Delisting events found: {len(delist)}")
    print(delist.to_string())

    # Delisting code interpretation:
    # 200-299: mergers (usually not a loss)
    # 300-399: exchange-related (transfer, etc.)
    # 400-499: liquidation (bad - often large negative return)
    # 500-599: dropped by exchange
    delist['delist_type'] = 'other'
    delist.loc[delist['dlstcd'].between(200, 299), 'delist_type'] = 'merger'
    delist.loc[delist['dlstcd'].between(400, 499), 'delist_type'] = 'liquidation'
    delist.loc[delist['dlstcd'].between(500, 599), 'delist_type'] = 'dropped'

    # For missing delisting returns on liquidations, use Shumway (1997) convention: -30%
    # This is standard in the literature
    delist['dlret_adj'] = delist['dlret']
    mask = delist['dlret'].isna() & delist['dlstcd'].between(400, 599)
    delist.loc[mask, 'dlret_adj'] = -0.30
    print("\nAdjusted delisting returns (Shumway 1997 for missing):")
    print(delist[['permno', 'dlstdt', 'dlstcd', 'delist_type', 'dlret', 'dlret_adj']].to_string())
else:
    print("  No delisting events found for sample stocks in this period.")

# ── Step 3: Fama-French daily factors for the full post-event period ──
print("\nPulling Fama-French factors (Feb 2021 – Mar 2026)...")

ff = db.raw_sql("""
    SELECT date, mktrf, smb, hml, rf
    FROM ff.factors_daily
    WHERE date BETWEEN '2021-02-01' AND '2026-03-31'
""", date_cols=['date'])

# Also pull FF5 + momentum if available
try:
    ff5 = db.raw_sql("""
        SELECT a.date, a.mktrf, a.smb, a.hml, a.rmw, a.cma, a.rf,
               b.umd
        FROM ff.fivefactors_daily AS a
        LEFT JOIN ff.factors_daily AS b ON a.date = b.date
        WHERE a.date BETWEEN '2021-02-01' AND '2026-03-31'
    """, date_cols=['date'])
    ff5.to_csv('wrds_03_ff5mom.csv', index=False)
    print(f"  FF5+MOM: {len(ff5)} days saved to wrds_03_ff5mom.csv")
except Exception as e:
    print(f"  FF5 query failed ({e}), using FF3 only")
    # Alternative: pull momentum separately
    try:
        mom = db.raw_sql("""
            SELECT date, umd
            FROM ff.factors_daily
            WHERE date BETWEEN '2021-02-01' AND '2026-03-31'
        """, date_cols=['date'])
        ff = ff.merge(mom, on='date', how='left')
    except:
        pass

# Convert percentages to decimals if needed
for col in ['mktrf', 'smb', 'hml', 'rf']:
    if col in ff.columns and ff[col].abs().max() > 1:
        ff[col] = ff[col] / 100.0

# ── Step 4: Pull control universe returns ──
# We need returns for ALL stocks in the CCM universe to do matched-firm BHAR
# This is a large pull - restrict to stocks that could plausibly be controls
# (same exchanges, common stock)
print("\nPulling control universe returns (this may take a few minutes)...")

control_ret = db.raw_sql("""
    SELECT a.permno, a.date, a.ret
    FROM crsp.dsf AS a
    WHERE a.date BETWEEN '2021-02-01' AND '2026-03-31'
      AND a.shrcd IN (10, 11)
      AND a.exchcd IN (1, 2, 3)
      AND a.ret IS NOT NULL
""", date_cols=['date'])

print(f"  Control universe: {len(control_ret):,} rows, {control_ret['permno'].nunique():,} stocks")

# ── Save everything ──
meme_ret.to_csv('wrds_03_returns_meme.csv', index=False)
if len(delist) > 0:
    delist.to_csv('wrds_03_delisting.csv', index=False)
ff.to_csv('wrds_03_ff_daily.csv', index=False)
control_ret.to_csv('wrds_03_returns_controls.csv', index=False)

print(f"\nSaved wrds_03_returns_meme.csv ({len(meme_ret):,} rows)")
print(f"Saved wrds_03_delisting.csv ({len(delist) if len(delist) > 0 else 0} rows)")
print(f"Saved wrds_03_ff_daily.csv ({len(ff)} rows)")
print(f"Saved wrds_03_returns_controls.csv ({len(control_ret):,} rows)")
print("\nNOTE: wrds_03_returns_controls.csv will be large (50-100MB). This is expected.")

db.close()
