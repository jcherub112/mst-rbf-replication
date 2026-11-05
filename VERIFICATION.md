# Verification against the published article

On 4 November 2026 the `pipeline/` and `appendix_d/` scripts in this repository were re-run on the author's WRDS extract (the `wrds_*.csv` files described in the README). Outputs were compared with the numbers in the article proof. Python 3.13, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, python-dateutil 2.9.0. The WRDS pull scripts were not re-run. Edits made to the scripts for this repository are limited to comments, docstrings and printed messages (notes about the development history removed, long dashes replaced by hyphens, and one outdated figure removed from a status message in `wrds_pull/placebo_pull_v2.py`). A comparison of syntax trees confirms no computational code changed, and the pipeline outputs were byte-identical before and after these edits.

## Reproduced

| Result | Published | Re-run |
|---|---|---|
| Matching | 44 of 44 matched; 22 by book-to-market, 22 by size | same |
| Table 1, mean BHAR (t), 6/12/24/36/48/58 months | -30.5 (-3.41), -34.3 (-4.78), -45.4 (-4.18), -68.9 (-3.62), -77.5 (-3.79), -98.7 (-3.54) | same |
| Table 1, median and % negative, all horizons | e.g. 48 months: -27.4%, 81.8% | same, computed from `analysis_bhar_matched.csv` |
| Table 2, raw mean meme / benchmark | -77.5% / -34.0% (n = 44 / 74) | same |
| Table 2, median and % underperforming | -27.4% / -8.0%; 81.8% / 66.2% | same |
| Table 2, Welch p and Mann-Whitney p | 0.08, 0.07 | 0.082, 0.065 |
| Table 2, long-short FF5+MOM alpha | -0.31% per month, t = -0.37 (NW, 3 lags) | same |
| Table 4, all seven cutoffs (N, 36- and 48-month BHAR and t, % negative) | e.g. >200%: N = 10, -110.3% (-2.23) | same |
| Table 5, value-weighted BHAR 12 to 58 months | e.g. 48 months: -125.4% (-2.75) | same |
| Table 6, EW and VW alphas, CAPM / FF3 / FF5+MOM | EW -3.39 (-3.74), -2.02 (-2.00), -1.75 (-1.84); VW -4.24 (-4.24), -2.56 (-1.72), -2.52 (-1.69) | same |
| Table 7, full sample and excluding biotech/pharma | -77.5% (-3.79), N = 44; -84.6% (-3.42), N = 36 | same; same eight biotech tickers |
| Section 3.7, issuers vs non-issuers | 27 vs 17; -91.8% vs -54.8% | same |
| Table C1 | skew-adj. t -5.21 / -5.38; bootstrap p 0.0010 / 0.0004 | same |
| Appendix C, monthly cross-sectional mean | -2.04% per month, t = -1.35 | same |
| Appendix D, Tables D1 and D2 | control assignments and BHARs | identical to the published spreadsheet (all 44 and 74 rows) |
| Appendix D, screen counts | 3,394 screened; 44 above 100%; 74 in (50%, 100%] | same |

## Differences

- **Table 2, winsorized mean row.** Published: -71.0% (meme) and -30.1% (benchmark), with values beyond each group's 5th and 95th percentiles set to those percentiles. `contemporaneous_placebo.py` uses `scipy.stats.mstats.winsorize`, which prints -71.4% and -29.0%. Clipping at `numpy.percentile(x, [5, 95])` reproduces the published -71.0% and -30.1%; the script's own fallback branch (used when scipy is absent) does this.
- **Table 2 note, long-short t at 12 lags.** The note reports t = -0.37 "at 3, 6 and 12 lags". The script prints -0.37 at 3 lags and -0.39 at 12 lags.

## Not produced by scripts in this repository

These results in the article were computed outside the scripts above, and the code for them is not included here:

- Table 3 comparison of meme and placebo estimands (the placebo BHARs themselves come from `wrds_pull/placebo_pull_v2.py`, which was not re-run)
- The within-2021 pooled regression of BHAR on CAR and a meme indicator (-51 points, t = -1.76)
- The short-interest and turnover regressions in Sections 3.2 and 4
- Table C2 (winsorized, trimmed and tail-excluded means) and the mean floored at -100%
- Table 7, biotech/pharma-only row
- The minimum detectable differences in Section 3.2
- Appendix B (May 2024 event)
