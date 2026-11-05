"""
Query 1: CRSP Full-Universe Meme Stock Screen
===============================================
Identifies all common stocks with CAR > 100% during the Jan 11 – Feb 3, 2021
meme stock squeeze window, using FF3 expected returns from a 2016-2020 estimation window.

Run on WRDS (Python wrds library) or adapt for SAS Studio.
Output: wrds_01_screen.csv
"""

import wrds
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ── Connect to WRDS ──
db = wrds.Connection()  # will prompt for credentials first time

# ── Step 1: Pull CRSP daily returns for ALL common stocks ──
# shrcd in (10, 11) = common stocks only
# exchcd in (1, 2, 3) = NYSE, AMEX, NASDAQ
# We need estimation window (2016-2020) + event window (Jan 2021)

print("Pulling CRSP daily stock data (2016-01-01 to 2021-02-05)...")

crsp = db.raw_sql("""
    SELECT a.permno, a.date, a.ret, a.shrcd, a.exchcd,
           b.ticker, b.comnam, b.siccd,
           a.prc, a.shrout
    FROM crsp.dsf AS a
    LEFT JOIN crsp.dsenames AS b
        ON a.permno = b.permno
        AND b.namedt <= a.date
        AND a.date <= b.nameendt
    WHERE a.date BETWEEN '2016-01-01' AND '2021-02-05'
      AND a.shrcd IN (10, 11)
      AND a.exchcd IN (1, 2, 3)
      AND a.ret IS NOT NULL
""", date_cols=['date'])

print(f"  Rows: {len(crsp):,}")
print(f"  Unique PERMNOs: {crsp['permno'].nunique():,}")

# ── Step 2: Pull Fama-French 3-factor daily data from WRDS ──
print("Pulling Fama-French 3-factor daily data...")

ff = db.raw_sql("""
    SELECT date, mktrf, smb, hml, rf
    FROM ff.factors_daily
    WHERE date BETWEEN '2016-01-01' AND '2021-02-05'
""", date_cols=['date'])

# Convert from percentage to decimal if needed
# (WRDS ff.factors_daily stores as percentages)
for col in ['mktrf', 'smb', 'hml', 'rf']:
    if ff[col].abs().max() > 1:
        ff[col] = ff[col] / 100.0

print(f"  FF rows: {len(ff)}")

# ── Step 3: Merge and compute CAR for each PERMNO ──
print("Computing CARs for event window Jan 11 – Feb 3, 2021...")

EVENT_START = pd.Timestamp('2021-01-11')
EVENT_END = pd.Timestamp('2021-02-03')
EST_START = pd.Timestamp('2016-01-01')
EST_END = pd.Timestamp('2020-12-31')

# Merge CRSP with FF
merged = crsp.merge(ff, on='date', how='inner')

results = []
permnos = merged['permno'].unique()
total = len(permnos)

for i, permno in enumerate(permnos):
    if (i + 1) % 500 == 0:
        print(f"  Processing {i+1}/{total}...")

    sub = merged[merged['permno'] == permno].sort_values('date')

    # Need sufficient estimation window
    est = sub[(sub['date'] >= EST_START) & (sub['date'] <= EST_END)].dropna(subset=['ret'])
    if len(est) < 250:
        continue

    # Need event window data
    ev = sub[(sub['date'] >= EVENT_START) & (sub['date'] <= EVENT_END)].dropna(subset=['ret'])
    if len(ev) < 10:  # need most of the 17 trading days
        continue

    # FF3 regression on estimation window
    try:
        y = est['ret'] - est['rf']
        X = sm.add_constant(est[['mktrf', 'smb', 'hml']])
        model = sm.OLS(y.astype(float), X.astype(float)).fit()
        alpha, b_mkt, b_smb, b_hml = model.params
    except Exception:
        continue

    # Compute abnormal returns in event window
    ev = ev.copy()
    ev['expected'] = ev['rf'] + alpha + b_mkt * ev['mktrf'] + b_smb * ev['smb'] + b_hml * ev['hml']
    ev['ar'] = ev['ret'] - ev['expected']

    # CAR (compounded)
    car = ((1 + ev['ar']).prod() - 1) * 100

    # Raw return
    raw_ret = ((1 + ev['ret']).prod() - 1) * 100

    # Grab identifying info from first event-window row
    info = ev.iloc[0]

    results.append({
        'permno': int(permno),
        'ticker': info.get('ticker', ''),
        'comnam': info.get('comnam', ''),
        'siccd': info.get('siccd', ''),
        'exchcd': info.get('exchcd', ''),
        'car_pct': car,
        'raw_ret_pct': raw_ret,
        'n_est_days': len(est),
        'n_event_days': len(ev),
        'mktcap_m': abs(info['prc']) * info['shrout'] / 1000 if pd.notna(info['prc']) else None
    })

df = pd.DataFrame(results)
print(f"\nTotal stocks screened: {len(df):,}")

# ── Step 4: Filter for CAR > 100% ──
qualified = df[df['car_pct'] > 100].sort_values('car_pct', ascending=False)
print(f"Stocks with CAR > 100%: {len(qualified)}")
print(f"Stocks with CAR > 300%: {len(df[df['car_pct'] > 300])}")

# Also show top 50 for inspection
print("\nTop 50 by CAR:")
print(qualified.head(50)[['permno', 'ticker', 'comnam', 'car_pct', 'raw_ret_pct', 'mktcap_m']].to_string())

# ── Save ──
qualified.to_csv('wrds_01_screen.csv', index=False)
df.to_csv('wrds_01_screen_full_universe.csv', index=False)  # keep full results for robustness checks
print(f"\nSaved wrds_01_screen.csv ({len(qualified)} rows)")
print(f"Saved wrds_01_screen_full_universe.csv ({len(df)} rows)")

db.close()
