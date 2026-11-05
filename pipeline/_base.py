#!/usr/bin/env python3
"""
_base.py - Shared base for the analysis modules.

Matching, buy-and-hold, delisting, calendar-time factor and Newey-West logic
used by every module in pipeline/. Methodology: Lyon, Barber and Tsai (1999)
matched-firm BHAR; Shumway (1997) delisting adjustment.
"""
import os
import functools
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "meme_stock_analysis", "wrds_data")
OUT = os.path.join(ROOT, "meme_stock_analysis", "v13_outputs")

BHAR_START = pd.Timestamp("2021-02-04")
CALTIME_START = pd.Timestamp("2021-02-01")
SAMPLE_END = pd.Timestamp("2025-12-31")
HORIZONS = [6, 12, 24, 36, 48, 58]


# ----------------------------- loaders -----------------------------
@functools.lru_cache(maxsize=1)
def screen():
    return pd.read_csv(f"{DATA}/wrds_01_screen.csv")


@functools.lru_cache(maxsize=1)
def screen_full_universe():
    return pd.read_csv(f"{DATA}/wrds_01_screen_full_universe.csv")


@functools.lru_cache(maxsize=1)
def meme_ccm():
    return pd.read_csv(f"{DATA}/wrds_02_ccm_meme.csv")


@functools.lru_cache(maxsize=1)
def full_ccm():
    return pd.read_csv(f"{DATA}/wrds_02_ccm_full.csv")


@functools.lru_cache(maxsize=1)
def delisting():
    return pd.read_csv(f"{DATA}/wrds_03_delisting.csv")


@functools.lru_cache(maxsize=1)
def returns_meme():
    return pd.read_csv(f"{DATA}/wrds_03_returns_meme.csv", parse_dates=["date"])


@functools.lru_cache(maxsize=1)
def returns_controls():
    return pd.read_csv(f"{DATA}/wrds_03_returns_controls.csv",
                       parse_dates=["date"])


@functools.lru_cache(maxsize=1)
def shares():
    return pd.read_csv(f"{DATA}/wrds_04_shares_outstanding.csv",
                       parse_dates=["date"])


@functools.lru_cache(maxsize=1)
def factors_monthly():
    """FF5+MOM compounded daily->monthly over sample period."""
    ff = pd.read_csv(f"{DATA}/wrds_03_ff5mom.csv", parse_dates=["date"])
    ff = ff[(ff["date"] >= CALTIME_START) & (ff["date"] <= SAMPLE_END)].copy()
    ff["ym"] = ff["date"].dt.to_period("M")
    cols = ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
    out = (ff.groupby("ym")[cols]
           .apply(lambda d: pd.Series({c: np.prod(1.0 + d[c].values) - 1.0
                                       for c in cols})).reset_index())
    return out


@functools.lru_cache(maxsize=1)
def delist_map():
    """permno -> (delist_date, delist_ret) with Shumway (1997) adjustment.
    Performance codes 400-599 with missing dlret -> -0.30; others -> 0."""
    out = {}
    for _, r in delisting().iterrows():
        code = r.get("dlstcd")
        if pd.isna(code) or int(code) == 100:
            continue
        code = int(code)
        dlret = r.get("dlret_adj")
        if pd.isna(dlret):
            dlret = r.get("dlret")
        if pd.isna(dlret):
            dlret = -0.30 if 400 <= code <= 599 else 0.0
        out[int(r["permno"])] = (pd.Timestamp(r["dlstdt"]), float(dlret))
    return out


@functools.lru_cache(maxsize=1)
def _returns_by_permno():
    """Combined daily-return lookup: permno -> sorted DataFrame[date, ret].
    Pulls from both the meme file and the control file (the threshold sweep
    needs returns for sub-100%-CAR stocks that live only in the control file)."""
    g = {}
    rm = returns_meme()[["permno", "date", "ret"]]
    for p, d in rm.groupby("permno"):
        g[int(p)] = d.sort_values("date")
    cr = returns_controls()[["permno", "date", "ret"]]
    for p, d in cr.groupby("permno"):
        g.setdefault(int(p), d.sort_values("date"))
    return g


def returns_for(permno):
    return _returns_by_permno().get(int(permno))


# ----------------------------- core compute -----------------------------
def buy_hold(rets, start, end, permno, dmap=None):
    """Compound (1+ret) over [start,end], skipping CRSP-missing returns,
    appending the delisting return if the firm delists within the window."""
    if rets is None:
        return None
    if dmap is None:
        dmap = delist_map()
    s = rets[(rets["date"] >= start) & (rets["date"] <= end)]
    s = s[s["ret"].notna()]
    if s.empty:
        return None
    bhr = float(np.prod(1.0 + s["ret"].values)) - 1.0
    if int(permno) in dmap:
        ddate, dret = dmap[int(permno)]
        if start <= ddate <= end and ddate > s["date"].max():
            bhr = (1.0 + bhr) * (1.0 + dret) - 1.0
    return bhr


def build_matches(sample_permnos, exclude=None):
    """LBT(1999) match each sample permno to a non-sample control.
    Same 1-digit SIC, control mktcap in [0.3x,2.0x], nearest B/M (fallback
    nearest size when sample B/M missing), without replacement.
    Returns dict sample_permno -> ctrl_permno.

    Characteristics for sample stocks are looked up in full_ccm (covers the
    whole screened universe), so this works for arbitrary CAR-threshold samples.
    """
    fc = full_ccm()
    chars = fc.set_index("permno")
    ret_permnos = set(_returns_by_permno().keys())
    sample_set = set(int(p) for p in sample_permnos)
    excl = set(int(p) for p in (exclude or [])) | sample_set

    pool = fc[~fc["permno"].isin(excl)].dropna(subset=["mktcap_m", "sic"]).copy()
    pool = pool[pool["permno"].isin(ret_permnos)]
    pool["sic1"] = pool["sic"].astype(str).str[0]

    # deterministic order: largest mktcap first
    order = sorted(sample_set,
                   key=lambda p: -(chars.loc[p, "mktcap_m"]
                                   if p in chars.index
                                   and pd.notna(chars.loc[p, "mktcap_m"])
                                   else 0))
    used, matches = set(), {}
    for sp in order:
        if sp not in chars.index:
            continue
        m = chars.loc[sp]
        if pd.isna(m["mktcap_m"]) or pd.isna(m["sic"]):
            continue
        sic1 = str(m["sic"])[0]
        lo, hi = 0.3 * m["mktcap_m"], 2.0 * m["mktcap_m"]
        cand = pool[(pool["sic1"] == sic1)
                    & (pool["mktcap_m"] >= lo)
                    & (pool["mktcap_m"] <= hi)
                    & (~pool["permno"].isin(used))].copy()
        if cand.empty:
            continue
        cand["size_ratio"] = cand["mktcap_m"] / m["mktcap_m"]
        if pd.notna(m["bm_ratio"]):
            wb = cand.dropna(subset=["bm_ratio"])
            if not wb.empty:
                wb = wb.assign(bm_diff=(wb["bm_ratio"] - m["bm_ratio"]).abs())
                best = wb.sort_values("bm_diff").iloc[0]
            else:
                best = cand.assign(sr=(cand["size_ratio"] - 1).abs()) \
                           .sort_values("sr").iloc[0]
        else:
            best = cand.assign(sr=(cand["size_ratio"] - 1).abs()) \
                       .sort_values("sr").iloc[0]
        cp = int(best["permno"])
        used.add(cp)
        matches[sp] = cp
    return matches


def bhar_panel(matches, horizons=HORIZONS):
    """Return DataFrame[permno, horizon_mo, bhar, meme_bhr, ctrl_bhr]."""
    dmap = delist_map()
    rows = []
    for sp, cp in matches.items():
        sr, cr = returns_for(sp), returns_for(cp)
        for h in horizons:
            end = BHAR_START + relativedelta(months=h)
            mb = buy_hold(sr, BHAR_START, end, sp, dmap)
            cb = buy_hold(cr, BHAR_START, end, cp, dmap)
            if mb is None or cb is None:
                continue
            rows.append({"permno": sp, "horizon_mo": h, "bhar": mb - cb,
                         "meme_bhr": mb, "ctrl_bhr": cb})
    return pd.DataFrame(rows)


def meme_market_caps():
    """permno -> market cap ($M) at 2021-02-01 (BHAR/VW weighting base).
    From returns_meme: |prc| * shrout (shrout in thousands -> $M)."""
    rm = returns_meme()
    first = (rm.sort_values("date").groupby("permno").first().reset_index())
    caps = {}
    for _, r in first.iterrows():
        prc, sh = r.get("prc"), r.get("shrout")
        if pd.notna(prc) and pd.notna(sh):
            caps[int(r["permno"])] = abs(prc) * sh / 1000.0  # $M
    return caps


def tstat(x):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)])
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() / se, len(x)


def newey_west_se(X, resid, nlags=12):
    n, k = X.shape
    S0 = np.zeros((k, k))
    for t in range(n):
        S0 += resid[t] ** 2 * np.outer(X[t], X[t])
    S0 /= n
    for L in range(1, nlags + 1):
        w = 1 - L / (nlags + 1)
        G = np.zeros((k, k))
        for t in range(L, n):
            G += resid[t] * resid[t - L] * np.outer(X[t], X[t - L])
        G /= n
        S0 += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X / n)
    V = XtXi @ S0 @ XtXi / n
    return np.sqrt(np.diag(V))


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
