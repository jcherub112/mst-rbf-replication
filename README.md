# Extreme winners, not special memes: replication code

Code for:

> Cherubini, J. (2026), "Extreme winners, not special memes? No robust evidence of a meme-specific long-run reversal after the 2021 retail squeeze", *Review of Behavioral Finance*, Vol. ahead-of-print No. ahead-of-print. https://doi.org/10.1108/RBF-06-2026-0335

Accepted 7 October 2026; published online ahead of print (EarlyCite), November 2026. The repository holds the sample-construction and analysis code only. No CRSP, Compustat or other licensed data are included (see "Data access" below).

## What is here

| Folder | Contents |
|---|---|
| `wrds_pull/` | Scripts that query WRDS: the January 2021 CRSP screen and event-window CARs (`01`), Compustat characteristics for matching (`02`), daily returns, delistings and Fama-French factors (`03`, with the 2025 extension in `03b`), institutional holdings and shares outstanding (`04`), short interest (`05`, `short_interest_pull.py`), and the 2010 to 2017 cross-decade placebo (`placebo_pull_v2.py`). |
| `pipeline/` | Analysis run on the extracted CSVs: matching, matched-firm BHAR, calendar-time portfolios, and the robustness modules. `_base.py` holds the shared matching, buy-and-hold and Newey-West logic. |
| `appendix_d/` | The scripts that built Appendix D (sample identifiers and matched-control assignments). |

## Data access

All inputs come from WRDS and require your own subscription:

- CRSP daily stock file and names (`crsp.dsf`, `crsp.dsenames`; `crsp.dsf_v2` for the 2025 extension), delisting returns, monthly shares outstanding
- Compustat annual fundamentals via the CRSP/Compustat link (size, SIC, book equity)
- Fama-French daily factors and momentum as distributed on WRDS
- Thomson Reuters 13F holdings and Compustat short interest (used only for the attention and squeeze proxies)

The WRDS scripts open a connection with `wrds.Connection()`, which prompts for your own credentials. No credentials are stored in this repository. The pull scripts write CSVs named `wrds_*.csv` to the working directory.

## Folder layout and run order

The pipeline scripts locate data relative to their own folder. Place the extracted CSVs so that this layout holds:

```
<parent>/
  mst-rbf-replication/      (this repository)
    pipeline/
  meme_stock_analysis/
    wrds_data/              (wrds_*.csv from wrds_pull/)
    v13_outputs/            (created by the pipeline)
```

1. On a WRDS-connected machine, run the scripts in `wrds_pull/` (`00_run_all.py` runs `01` to `05`; run `03b_crsp_returns_2025_v2.py`, `placebo_pull_v2.py` and `short_interest_pull.py` separately). Move the CSVs into `meme_stock_analysis/wrds_data/`.
2. From `pipeline/`:
   ```
   python3 build_analysis_files.py
   python3 bhar_alt_specs.py --allow-run
   python3 contemporaneous_placebo.py
   python3 threshold_sensitivity.py --allow-run
   python3 vw_reproduction.py --allow-run
   python3 sector_split.py --allow-run
   python3 social_media_intensity.py --allow-run --no-trends
   ```
   Most modules exit without doing anything unless `--allow-run` is passed (a guard left in from development).
3. Appendix D: `appendix_d/` contains the scripts as they were run. They expect the pipeline at `<base>/replication_v32/pipeline` and the data at `<base>/wrds_data`, so either create that layout (a symlink is enough) or edit the path lines at the top of each script. Run `check_meme.py <base> <out>`, then `regen_bench.py <base> <out>`, then `build_supp.py <base> <out>`. `check_meme.py` also compares against a saved copy of the matched-control assignments and can be skipped.

Requirements are in `requirements.txt`.

## Which results each script produces

See `VERIFICATION.md` for the check of these scripts against the published numbers, including the results that are not produced by any script in this repository.

| Paper result | Script |
|---|---|
| Table 1 (matched-firm BHAR, 6 to 58 months) | `build_analysis_files.py`, `vw_reproduction.py` |
| Table 2 (same-window benchmark, long-short alpha) | `contemporaneous_placebo.py` |
| Table 3 inputs (placebo firms, matches, BHAR) | `wrds_pull/placebo_pull_v2.py` |
| Table 4 (threshold sensitivity) | `threshold_sensitivity.py` |
| Tables 5 and 6 (equal- vs value-weighted) | `vw_reproduction.py` |
| Table 7 and Appendix A (sector, biotech exclusion) | `sector_split.py` |
| Section 3.7 (equity issuance split) | `vw_reproduction.py` |
| Table C1 (skewness-adjusted t, bootstrap) | `bhar_alt_specs.py` |
| Appendix D, Tables D1 and D2 | `appendix_d/` |

## Licence

Code is released under the MIT Licence (`LICENSE`). The licence covers the code only, not any data you extract with it.

## Citation

See `CITATION.cff`. Please cite the article above.
