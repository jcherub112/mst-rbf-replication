"""
Query 5: Short Interest Data (Optional)
=========================================
Bi-monthly short interest for the meme stock sample, from pre-squeeze through
the end of the sample.

Run on WRDS. Requires wrds_01_screen.csv from Query 1.
Output: wrds_05_short_interest.csv
"""

import wrds
import pandas as pd

# ── Load meme stock sample ──
meme = pd.read_csv('wrds_01_screen.csv')
meme_permnos = meme['permno'].tolist()
permno_list = ','.join(str(p) for p in meme_permnos)

# ── Connect ──
db = wrds.Connection()

# ── Try WRDS short interest (Compustat supplemental) ──
print("Pulling short interest data...")

# Method 1: Compustat short interest
try:
    si = db.raw_sql(f"""
        SELECT a.permno, a.shortintdate, a.shortint, a.shrout,
               a.shortint * 1.0 / NULLIF(a.shrout, 0) AS si_ratio
        FROM comp_seg_shortint.shortint AS a
        WHERE a.permno IN ({permno_list})
          AND a.shortintdate BETWEEN '2020-06-01' AND '2026-03-31'
        ORDER BY a.permno, a.shortintdate
    """, date_cols=['shortintdate'])
    print(f"  Compustat short interest: {len(si)} rows")
except Exception as e:
    print(f"  Compustat short interest failed: {e}")
    si = pd.DataFrame()

# Method 2: FINRA/NYSE short interest via WRDS
if len(si) == 0:
    try:
        si = db.raw_sql(f"""
            SELECT permno, settledate, shortinterest, shrout
            FROM wrdsapps.short_interest
            WHERE permno IN ({permno_list})
              AND settledate BETWEEN '2020-06-01' AND '2026-03-31'
            ORDER BY permno, settledate
        """, date_cols=['settledate'])
        si.rename(columns={'settledate': 'shortintdate',
                           'shortinterest': 'shortint'}, inplace=True)
        si['si_ratio'] = si['shortint'] / (si['shrout'] * 1000)
        print(f"  WRDS short interest app: {len(si)} rows")
    except Exception as e:
        print(f"  WRDS short interest app also failed: {e}")

# Method 3: Direct FINRA table
if len(si) == 0:
    try:
        si = db.raw_sql(f"""
            SELECT *
            FROM finra.shortinterest
            WHERE permno IN ({permno_list})
              AND settlementdate BETWEEN '2020-06-01' AND '2026-03-31'
            ORDER BY permno, settlementdate
        """, date_cols=['settlementdate'])
        print(f"  FINRA table: {len(si)} rows")
    except Exception as e:
        print(f"  FINRA table failed: {e}")
        print("\n  Short interest data not available in your WRDS subscription.")
        print("  This query is optional; the rest of the pipeline does not need it.")

if len(si) > 0:
    si.to_csv('wrds_05_short_interest.csv', index=False)
    print(f"\nSaved wrds_05_short_interest.csv ({len(si)} rows)")

    # Quick summary
    for permno in si['permno'].unique():
        sub = si[si['permno'] == permno].sort_values(si.columns[1])
        ticker = meme.loc[meme['permno'] == permno, 'ticker'].values
        tk = ticker[0] if len(ticker) > 0 else str(permno)
        print(f"\n  {tk}: {len(sub)} observations")
        if 'si_ratio' in sub.columns:
            print(f"    SI ratio range: {sub['si_ratio'].min():.2%} – {sub['si_ratio'].max():.2%}")

db.close()
