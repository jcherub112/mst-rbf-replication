"""
Query 4: Thomson S34 / 13F Institutional Holdings
====================================================
Quarterly institutional ownership for meme stock sample,
Q4 2020 through Q4 2025 (pre-squeeze baseline through present).

This lets us track the institutional ownership trajectory -
did institutions flee meme stocks? When? How fast?

Run on WRDS. Requires wrds_01_screen.csv from Query 1.
Output: wrds_04_holdings.csv
"""

import wrds
import pandas as pd

# ── Load meme stock sample ──
meme = pd.read_csv('wrds_01_screen.csv')
meme_permnos = meme['permno'].tolist()
print(f"Meme stock sample: {len(meme_permnos)} PERMNOs")

# ── Connect ──
db = wrds.Connection()

# ── Step 1: Map PERMNOs to CUSIPs (Thomson uses 8-digit CUSIP) ──
print("Mapping PERMNOs to CUSIPs...")

permno_list = ','.join(str(p) for p in meme_permnos)

cusip_map = db.raw_sql(f"""
    SELECT DISTINCT permno, ncusip
    FROM crsp.dsenames
    WHERE permno IN ({permno_list})
      AND ncusip IS NOT NULL
    ORDER BY permno
""")

# Thomson S34 uses 8-digit CUSIP (first 8 chars)
cusip_map['cusip8'] = cusip_map['ncusip'].str[:8]
print(f"  Mapped {len(cusip_map)} PERMNO-CUSIP pairs")
print(cusip_map.to_string())

cusip_list = "'" + "','".join(cusip_map['cusip8'].unique()) + "'"

# ── Step 2: Pull Thomson S34 holdings ──
# s34type1 = quarterly 13F filings
print("\nPulling Thomson S34 quarterly holdings (Q4 2020 – Q4 2025)...")

holdings = db.raw_sql(f"""
    SELECT a.cusip, a.rdate, a.fdate,
           a.mgrno, a.mgrname,
           a.shares, a.shrout2,
           a.prc
    FROM tfn.s34 AS a
    WHERE a.cusip IN ({cusip_list})
      AND a.rdate BETWEEN '2020-10-01' AND '2025-12-31'
      AND a.fdate BETWEEN '2020-10-01' AND '2026-03-31'
    ORDER BY a.cusip, a.rdate, a.mgrno
""", date_cols=['rdate', 'fdate'])

print(f"  Holdings rows: {len(holdings):,}")

if len(holdings) == 0:
    # Try alternative table name (WRDS sometimes uses different schema)
    print("  Trying alternative table: tfn.s34type1...")
    holdings = db.raw_sql(f"""
        SELECT a.cusip, a.rdate, a.fdate,
               a.mgrno, a.mgrname,
               a.shares, a.shrout2,
               a.prc
        FROM tfn.s34type1 AS a
        WHERE a.cusip IN ({cusip_list})
          AND a.rdate BETWEEN '2020-10-01' AND '2025-12-31'
        ORDER BY a.cusip, a.rdate, a.mgrno
    """, date_cols=['rdate', 'fdate'])
    print(f"  Holdings rows (alt table): {len(holdings):,}")

if len(holdings) == 0:
    # Final fallback: try the wrdsapps 13F summary table
    print("  Trying WRDS 13F summary: wrdsapps.holdings_summary...")
    try:
        holdings = db.raw_sql(f"""
            SELECT cusip, rdate, num_holders, total_shares, total_value
            FROM wrdsapps.holdings_summary
            WHERE cusip IN ({cusip_list})
              AND rdate BETWEEN '2020-10-01' AND '2025-12-31'
            ORDER BY cusip, rdate
        """, date_cols=['rdate'])
        print(f"  Summary rows: {len(holdings):,}")
    except Exception as e:
        print(f"  Summary table also failed: {e}")

# ── Step 3: Aggregate to quarterly institutional ownership ──
if len(holdings) > 0 and 'shares' in holdings.columns:
    # Merge CUSIP back to PERMNO
    holdings = holdings.merge(cusip_map[['cusip8', 'permno']],
                               left_on='cusip', right_on='cusip8', how='left')

    # Aggregate: total institutional shares per stock per quarter
    quarterly = (holdings.groupby(['permno', 'cusip', 'rdate'])
                 .agg(n_institutions=('mgrno', 'nunique'),
                      total_inst_shares=('shares', 'sum'))
                 .reset_index())

    # We need shares outstanding to compute ownership fraction
    # Pull from CRSP quarterly
    print("\nPulling shares outstanding for ownership computation...")
    shrout = db.raw_sql(f"""
        SELECT permno, date, shrout
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '2020-10-01' AND '2025-12-31'
    """, date_cols=['date'])

    # Get last shrout per quarter
    shrout['qtr'] = shrout['date'].dt.to_period('Q')
    shrout_q = (shrout.sort_values('date')
                .groupby(['permno', 'qtr'])['shrout']
                .last()
                .reset_index())
    shrout_q['rdate_approx'] = shrout_q['qtr'].dt.to_timestamp('Q')

    # Merge - approximate date matching
    quarterly['qtr'] = quarterly['rdate'].dt.to_period('Q')
    quarterly = quarterly.merge(shrout_q[['permno', 'qtr', 'shrout']],
                                 on=['permno', 'qtr'], how='left')

    # Institutional ownership fraction
    # Thomson shares are in actual shares; CRSP shrout is in thousands
    quarterly['inst_own_pct'] = (quarterly['total_inst_shares'] /
                                  (quarterly['shrout'] * 1000)) * 100

    print("\nQuarterly institutional ownership summary:")
    for permno in quarterly['permno'].unique():
        sub = quarterly[quarterly['permno'] == permno].sort_values('rdate')
        ticker = meme.loc[meme['permno'] == permno, 'ticker'].values
        tk = ticker[0] if len(ticker) > 0 else str(permno)
        print(f"\n  {tk} (PERMNO {permno}):")
        for _, row in sub.iterrows():
            print(f"    {row['rdate'].strftime('%Y-Q%q') if hasattr(row['rdate'], 'strftime') else row['rdate']}: "
                  f"{row['n_institutions']} institutions, {row['inst_own_pct']:.1f}% ownership")

    quarterly.to_csv('wrds_04_holdings.csv', index=False)
    print(f"\nSaved wrds_04_holdings.csv ({len(quarterly)} rows)")
else:
    print("\nNo holdings data retrieved. Check WRDS subscription for Thomson Reuters.")

# ── Step 4: Also grab CRSP shares outstanding history for the paper ──
# (useful for tracking dilution from equity issuances)
print("\nPulling monthly shares outstanding for dilution analysis...")

shrout_monthly = db.raw_sql(f"""
    SELECT permno, date, shrout, cfacshr
    FROM crsp.msf
    WHERE permno IN ({permno_list})
      AND date BETWEEN '2020-12-01' AND '2026-03-31'
    ORDER BY permno, date
""", date_cols=['date'])

shrout_monthly.to_csv('wrds_04_shares_outstanding.csv', index=False)
print(f"Saved wrds_04_shares_outstanding.csv ({len(shrout_monthly)} rows)")

db.close()
