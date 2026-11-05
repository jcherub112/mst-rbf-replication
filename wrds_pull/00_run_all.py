"""
Runs WRDS queries 01 to 05 in sequence and saves CSVs to this directory.
Requires WRDS Python access (pip install wrds). Each query can also be run alone.

SETUP:
    pip install wrds pandas numpy statsmodels
    # First run prompts for your WRDS credentials

EXPECTED RUNTIME: ~15-30 minutes (the control-universe returns in Query 3 dominate)

OUTPUT FILES:
    wrds_01_screen.csv               - Meme stock sample (CAR > 100%)
    wrds_01_screen_full_universe.csv - Full screen results (all stocks)
    wrds_02_ccm_meme.csv             - Compustat characteristics for meme stocks
    wrds_02_ccm_full.csv             - Full CCM universe (for control matching)
    wrds_02_ccm_crsp_only.csv        - CRSP-only data for micro-caps missing Compustat
    wrds_03_returns_meme.csv         - Daily returns for meme stocks
    wrds_03_returns_controls.csv     - Daily returns for control universe (large, ~100MB)
    wrds_03_delisting.csv            - Delisting returns for sample stocks
    wrds_03_ff_daily.csv             - Fama-French daily factors
    wrds_03_ff5mom.csv               - FF5 + momentum daily factors
    wrds_04_holdings.csv             - Quarterly institutional holdings
    wrds_04_shares_outstanding.csv   - Monthly shares outstanding
    wrds_05_short_interest.csv       - Bi-monthly short interest (if available)

TOTAL DISK: ~200-300MB (mostly the control-universe returns)

Then move the CSVs to meme_stock_analysis/wrds_data/ and run pipeline/ (see README).
"""

import subprocess
import sys
import os

scripts = [
    '01_crsp_screen.py',
    '02_ccm_matching.py',
    '03_crsp_returns.py',
    '04_institutional_holdings.py',
    '05_short_interest.py',
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"RUNNING: {script}")
    print(f"{'='*60}\n")

    result = subprocess.run([sys.executable, script],
                           capture_output=False,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\nWARNING: {script} exited with code {result.returncode}")
        cont = input("Continue to next query? (y/n): ").strip().lower()
        if cont != 'y':
            print("Stopping.")
            sys.exit(1)

print(f"\n{'='*60}")
print("ALL QUERIES COMPLETE")
print(f"{'='*60}")
print("\nOutput files in current directory:")
for f in sorted(os.listdir('.')):
    if f.startswith('wrds_') and f.endswith('.csv'):
        size = os.path.getsize(f) / (1024*1024)
        print(f"  {f:45s} {size:7.1f} MB")
