"""
Cross-decade placebo: screen, match and BHAR for non-squeeze windows, 2010-2017.
================================================================================
Applies the same FF3 CAR screen (>100%) and Shumway (1997) buy-and-hold logic to
earlier non-squeeze windows, and matches each qualifier on size + 1-digit SIC +
book-to-market (with a size + SIC alternative). Also saves monthly event-time
returns for a calendar-time long-short, event-date observables for covariate
balance, and a NONOVERLAP option for window-clustered inference.

NEEDS: crsp.dsf, crsp.dsenames, crsp.dsedelist, ff.factors_daily, comp.funda.
  Book equity is linked to permno by CUSIP (comp.funda cusip[1:8] ==
  crsp.dsenames ncusip), so the CCM link schema crsp_a_ccm is not required.
  If comp.funda is not accessible in --test, the B/M stage disables itself and
  the rest still runs.

RUN:
    pip install wrds pandas numpy statsmodels
    cd <folder with this file and wrds_01_screen.csv>
    python3 placebo_pull_v2.py --test     # access check (6 tables)
    python3 placebo_pull_v2.py            # full run, ~45-90 min

OUTPUT (./placebo_data_v2/):
    placebo_screen_all.csv        every >100%-CAR qualifier + window + observables
    placebo_bm.csv                book equity + B/M per qualifier and control
    placebo_bhar.csv              per-stock matched BHAR (36 & 48mo), size+SIC+B/M
    placebo_bhar_sizeSIC.csv      per-stock matched BHAR, size+SIC only
    placebo_monthly_returns.csv   monthly ret per qualifier+control, event_month 1..48
    placebo_observables.csv       turnover, idio_vol, amihud, price, regime per name
    placebo_summary.csv           pooled + per-window mean/median/t, both matchings
    placebo_meta.txt              run config + counts + table-access status
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import wrds

# ----------------------------- CONFIG -----------------------------
# Denser, less-overlapping window set (annual mid-Jan anchors 2010-2019). Each has a
# full 48-month post-window ending before the Jan-2021 squeeze. Set NONOVERLAP=True
# to keep only every-other year (clean non-overlap for window-clustered inference).
WINDOW_STARTS = [
    "2010-01-12", "2011-01-12", "2012-01-12", "2013-01-12", "2014-01-12",
    "2015-01-12", "2016-01-12", "2016-07-12", "2017-01-12",
]
NONOVERLAP = False          # True -> drop windows whose 48mo horizons overlap
EVENT_TDAYS = 17
CAR_THRESHOLD = 100.0
EST_YEARS = 3
MIN_EST_DAYS = 250
HORIZONS = [36, 48]
MONTHLY_MAX_MONTHS = 48     # event-time months to save for the long-short
OUT = "placebo_data_v2"
MEME_PERMNOS_FILE = "wrds_01_screen.csv"


def trading_window(all_dates, start, n):
    d = all_dates[all_dates >= pd.Timestamp(start)]
    return d[:n]


def shumway_delist(de):
    out = {}
    for _, r in de.iterrows():
        code = r.dlstcd
        if pd.isna(code) or int(code) == 100:
            continue
        code = int(code)
        dl = r.dlret
        if pd.isna(dl):
            dl = -0.30 if 400 <= code <= 599 else 0.0
        out[int(r.permno)] = (pd.Timestamp(r.dlstdt), float(dl))
    return out


def buy_hold(ret_df, start, end, permno, dmap):
    s = ret_df[(ret_df.date >= start) & (ret_df.date <= end)].dropna(subset=["ret"])
    if s.empty:
        return None
    bhr = float(np.prod(1 + s.ret.values)) - 1
    if permno in dmap:
        dd, dr = dmap[permno]
        if start <= dd <= end and dd > s.date.max():
            bhr = (1 + bhr) * (1 + dr) - 1
    return bhr


def monthly_from_daily(ret_df, start, end, permno, dmap, max_months):
    """Compound daily -> calendar-monthly ret over the event window, event_month 1..N."""
    s = ret_df[(ret_df.date >= start) & (ret_df.date <= end)].dropna(subset=["ret"]).copy()
    if s.empty:
        return []
    s["ym"] = s.date.dt.to_period("M")
    rows = []
    for i, (ym, g) in enumerate(s.groupby("ym"), start=1):
        if i > max_months:
            break
        mret = float(np.prod(1 + g.ret.values)) - 1
        rows.append({"permno": permno, "event_month": i,
                     "ym": str(ym), "ret": mret})
    # splice delisting return into the month it occurs
    if permno in dmap:
        dd, dr = dmap[permno]
        if start <= dd <= end and rows:
            dperiod = str(dd.to_period("M"))
            for r in rows:
                if r["ym"] == dperiod:
                    r["ret"] = (1 + r["ret"]) * (1 + dr) - 1
    return rows


def book_equity_table(db, permnos, year_lo, year_hi):
    """Time-series of Compustat book equity per permno, linked by CUSIP (comp.funda
    cusip 1-8 == crsp.dsenames ncusip), which avoids the crsp_a_ccm (CCM link) schema
    that this WRDS seat lacks. Book equity = ceq + txdb - pstk (Fama-French), drop <=0.
    Returns DataFrame[permno, datadate, be]; empty df if nothing resolves."""
    empty = pd.DataFrame(columns=["permno", "datadate", "be"])
    if not permnos:
        return empty
    pl = ",".join(map(str, permnos))
    # permno <-> 8-char cusip from dsenames (accessible)
    names = db.raw_sql(f"""SELECT DISTINCT permno, ncusip FROM crsp.dsenames
        WHERE permno IN ({pl}) AND ncusip IS NOT NULL AND ncusip <> ''""")
    if names.empty:
        return empty
    names["cusip8"] = names.ncusip.str[:8]
    cus = [c for c in names.cusip8.dropna().unique() if c]
    if not cus:
        return empty
    cl = ",".join("'" + c + "'" for c in cus)
    try:
        f = db.raw_sql(f"""
            SELECT cusip, datadate, ceq, txdb, pstk
            FROM comp.funda
            WHERE datafmt='STD' AND consol='C' AND indfmt='INDL' AND popsrc='D'
              AND fyear BETWEEN {year_lo} AND {year_hi}
              AND substr(cusip,1,8) IN ({cl})
        """, date_cols=["datadate"])
    except Exception as e:
        print(f"   (comp.funda book-equity pull failed: {str(e)[:70]} -> B/M off)")
        return empty
    if f.empty:
        return empty
    f["cusip8"] = f.cusip.str[:8]
    f["be"] = f.ceq.fillna(0) + f.txdb.fillna(0) - f.pstk.fillna(0)
    f.loc[f.be <= 0, "be"] = np.nan
    f = f.dropna(subset=["be"])
    m = f.merge(names[["permno", "cusip8"]], on="cusip8", how="inner")
    return m[["permno", "datadate", "be"]].copy()


def smoke_test():
    db = wrds.Connection()
    # B/M now links by CUSIP via comp.funda + crsp.dsenames (no crsp_a_ccm needed).
    need = ["crsp.dsf", "crsp.dsenames", "crsp.dsedelist", "ff.factors_daily",
            "comp.funda"]
    print("\nChecking required tables:")
    ok = True
    for t in need:
        lib, tbl = t.split(".")
        try:
            n = db.raw_sql(f"SELECT * FROM {lib}.{tbl} LIMIT 1")
            print(f"  [OK ] {t}  ({len(n.columns)} cols)")
        except Exception as e:
            ok = ok and (t == "comp.funda")  # comp.funda optional (B/M off if missing)
            tag = "WARN" if t == "comp.funda" else "FAIL"
            print(f"  [{tag}] {t}  -> {str(e)[:80]}")
    db.close()
    print("\nAll tables OK (B/M match via CUSIP link will run)." if ok else
          "\nA CORE table is missing; send me the FAIL line.")


def main():
    if "--test" in sys.argv:
        smoke_test()
        return
    windows = WINDOW_STARTS
    if NONOVERLAP:
        keep, last_end = [], pd.Timestamp("1900-01-01")
        for ws in sorted(windows):
            if pd.Timestamp(ws) >= last_end:
                keep.append(ws)
                last_end = pd.Timestamp(ws) + pd.DateOffset(months=max(HORIZONS) + 1)
        windows = keep
    os.makedirs(OUT, exist_ok=True)
    db = wrds.Connection()
    meme_permnos = set()
    if os.path.exists(MEME_PERMNOS_FILE):
        meme_permnos = set(pd.read_csv(MEME_PERMNOS_FILE).permno.astype(int))
        print(f"Excluding {len(meme_permnos)} meme permnos from placebo.")

    span_lo = (pd.Timestamp(min(windows)) - pd.DateOffset(years=EST_YEARS)).strftime("%Y-%m-%d")
    span_hi = (pd.Timestamp(max(windows)) + pd.DateOffset(months=max(HORIZONS) + 2)).strftime("%Y-%m-%d")
    print(f"FF factors {span_lo}..{span_hi}")
    ff = db.raw_sql(f"""SELECT date, mktrf, smb, hml, rf FROM ff.factors_daily
                        WHERE date BETWEEN '{span_lo}' AND '{span_hi}'""", date_cols=["date"])
    for c in ["mktrf", "smb", "hml", "rf"]:
        if ff[c].abs().max() > 1:
            ff[c] = ff[c] / 100.0
    # market-state per day for regime tagging
    ff["mkt_ann_252"] = (1 + ff["mktrf"] + ff["rf"]).rolling(252).apply(np.prod, raw=True) - 1
    ff["mkt_vol_63"] = (ff["mktrf"]).rolling(63).std() * np.sqrt(252)

    screen_rows, obs_rows = [], []
    univ_chars = {}
    for ws in windows:
        est_start = pd.Timestamp(ws) - pd.DateOffset(years=EST_YEARS)
        ev_hi = (pd.Timestamp(ws) + pd.DateOffset(days=40)).strftime("%Y-%m-%d")
        print(f"\n[{ws}] pulling universe {est_start.strftime('%Y-%m-%d')}..{ev_hi}")
        crsp = db.raw_sql(f"""
            SELECT a.permno,a.date,a.ret,a.prc,a.shrout,a.vol,b.ticker,b.comnam,b.siccd
            FROM crsp.dsf a
            LEFT JOIN crsp.dsenames b ON a.permno=b.permno
                AND b.namedt<=a.date AND a.date<=b.nameendt
            WHERE a.date BETWEEN '{est_start.strftime('%Y-%m-%d')}' AND '{ev_hi}'
              AND b.shrcd IN (10,11) AND b.exchcd IN (1,2,3) AND a.ret IS NOT NULL
        """, date_cols=["date"])
        crsp = crsp.merge(ff, on="date", how="inner")
        all_dates = pd.Index(sorted(crsp.date.unique()))
        ev_dates = trading_window(all_dates, ws, EVENT_TDAYS)
        ev_end = ev_dates.max()
        est_end = ev_dates.min() - pd.Timedelta(days=1)
        regime = ff[ff.date == ev_end][["mkt_ann_252", "mkt_vol_63"]]
        regime = regime.iloc[0].to_dict() if len(regime) else {"mkt_ann_252": np.nan, "mkt_vol_63": np.nan}
        chars = []
        for permno, sub in crsp.groupby("permno"):
            if int(permno) in meme_permnos:
                continue
            sub = sub.sort_values("date")
            ev = sub[sub.date.isin(ev_dates)].dropna(subset=["ret"])
            if len(ev) < 10:
                continue
            info = ev.iloc[0]
            if pd.isna(info.prc) or pd.isna(info.shrout) or pd.isna(info.siccd):
                continue
            mktcap = abs(info.prc) * info.shrout / 1000.0
            try:
                sic1 = str(int(info.siccd))[0]
            except Exception:
                continue
            chars.append({"permno": int(permno), "sic1": sic1, "mktcap_m": mktcap})
            est = sub[(sub.date >= est_start) & (sub.date <= est_end)].dropna(subset=["ret"])
            if len(est) < MIN_EST_DAYS:
                continue
            try:
                y = (est.ret - est.rf).astype(float)
                X = sm.add_constant(est[["mktrf", "smb", "hml"]].astype(float))
                fit = sm.OLS(y, X).fit()
                p = fit.params
                idio_vol = float(fit.resid.std() * np.sqrt(252))
            except Exception:
                continue
            exp = ev.rf + p["const"] + p["mktrf"] * ev.mktrf + p["smb"] * ev.smb + p["hml"] * ev.hml
            car = ((1 + (ev.ret - exp)).prod() - 1) * 100
            if car > CAR_THRESHOLD:
                screen_rows.append({"permno": int(permno), "window": ws,
                                    "ev_end": ev_end.strftime("%Y-%m-%d"),
                                    "ticker": info.ticker, "comnam": info.comnam,
                                    "siccd": int(info.siccd), "sic1": sic1,
                                    "car_pct": car, "mktcap_m": mktcap})
                # observables at event (Barber-Odean turnover, Amihud illiq, idio vol)
                evd = ev.copy()
                turn = (evd.vol / (info.shrout * 1000.0)).mean() if "vol" in evd else np.nan
                amihud = (evd.ret.abs() / (evd.vol * abs(evd.prc)).replace(0, np.nan)).mean() \
                    if "vol" in evd else np.nan
                obs_rows.append({"permno": int(permno), "window": ws,
                                 "price": abs(float(info.prc)), "turnover": float(turn),
                                 "idio_vol": idio_vol, "amihud": float(amihud) if pd.notna(amihud) else np.nan,
                                 "mkt_ann_252": regime["mkt_ann_252"], "mkt_vol_63": regime["mkt_vol_63"]})
        univ_chars[ws] = pd.DataFrame(chars)
        print(f"   stocks with chars: {len(chars)} | cumulative qualifiers: {len(screen_rows)}")

    screen = pd.DataFrame(screen_rows)
    screen.to_csv(f"{OUT}/placebo_screen_all.csv", index=False)
    pd.DataFrame(obs_rows).to_csv(f"{OUT}/placebo_observables.csv", index=False)
    print(f"\nTotal placebo qualifiers across windows: {len(screen)}")
    if screen.empty:
        print("No qualifiers; widen WINDOW_STARTS or lower CAR_THRESHOLD.")
        db.close(); return
    all_qualifiers = set(screen.permno.astype(int))

    # ---- book equity / B/M (CUSIP link) for qualifiers AND their candidate pools ----
    pool_permnos = set(screen.permno.astype(int))
    for ws in windows:
        pool_permnos |= set(univ_chars[ws].permno.astype(int))
    ylo = pd.Timestamp(min(windows)).year - 3
    yhi = pd.Timestamp(max(windows)).year
    print(f"\nPulling Compustat book equity (CUSIP link) for {len(pool_permnos)} permnos, FY {ylo}-{yhi}...")
    betab = book_equity_table(db, sorted(pool_permnos), ylo, yhi)
    bm_available = not betab.empty
    print(f"   book-equity rows: {len(betab)}; B/M match {'ON' if bm_available else 'OFF'}")
    be_by_permno = {p: g.sort_values("datadate") for p, g in betab.groupby("permno")} if bm_available else {}

    def be_asof(permno, asof_ts):
        g = be_by_permno.get(int(permno))
        if g is None:
            return None
        gg = g[g.datadate <= asof_ts]
        return float(gg.be.iloc[-1]) if len(gg) else None

    bm_rows = []
    for ws in windows:
        wts = pd.Timestamp(ws)
        for _, r in univ_chars[ws].iterrows():
            b = be_asof(r.permno, wts)              # most recent FY before the event window
            bm = (b / r.mktcap_m) if (b is not None and r.mktcap_m) else np.nan
            bm_rows.append({"permno": int(r.permno), "window": ws,
                            "mktcap_m": r.mktcap_m, "sic1": r.sic1,
                            "book_equity_m": b if b is not None else np.nan, "bm_ratio": bm})
    bm_df = pd.DataFrame(bm_rows)
    bm_df.to_csv(f"{OUT}/placebo_bm.csv", index=False)
    bm_map = {(int(x.permno), x.window): x.bm_ratio for x in bm_df.itertuples() if pd.notna(x.bm_ratio)}

    # ---- Stage 2: match (BOTH size+SIC and size+SIC+B/M), BHAR + monthly returns ----
    def match_pool(grp, pool, use_bm):
        used, matches = set(), {}
        for _, q in grp.sort_values("mktcap_m", ascending=False).iterrows():
            qp = int(q.permno)
            lo, hi = 0.3 * q.mktcap_m, 2.0 * q.mktcap_m
            cand = pool[(pool.sic1 == q.sic1) & (pool.mktcap_m.between(lo, hi))
                        & (~pool.permno.isin(used))].copy()
            if cand.empty:
                continue
            qbm = bm_map.get((qp, q.window))
            if use_bm and qbm is not None:
                cand["bm"] = cand.permno.map(lambda p: bm_map.get((int(p), q.window), np.nan))
                cb = cand.dropna(subset=["bm"])
                if not cb.empty:
                    cb["d"] = (cb.bm - qbm).abs()
                    cp = int(cb.sort_values("d").iloc[0].permno)
                    used.add(cp); matches[qp] = cp
                    continue
            cand["d"] = (cand.mktcap_m / q.mktcap_m - 1).abs()  # size fallback
            cp = int(cand.sort_values("d").iloc[0].permno)
            used.add(cp); matches[qp] = cp
        return matches

    bhar_bm, bhar_ss, monthly_rows = [], [], []
    for ws, grp in screen.groupby("window"):
        ev_end = pd.Timestamp(grp.ev_end.iloc[0])
        bhar_start = ev_end + pd.Timedelta(days=1)
        post_hi = (ev_end + pd.DateOffset(months=max(HORIZONS) + 1)).strftime("%Y-%m-%d")
        pool = univ_chars[ws]
        pool = pool[~pool.permno.isin(all_qualifiers)]
        m_bm = match_pool(grp, pool, use_bm=bm_available)
        m_ss = match_pool(grp, pool, use_bm=False)
        need = list(set(m_bm) | set(m_bm.values()) | set(m_ss) | set(m_ss.values()))
        if not need:
            continue
        print(f"[{ws}] matched bm={len(m_bm)} sizeSIC={len(m_ss)}; pulling {len(need)} return series")
        rets = db.raw_sql(f"""SELECT permno,date,ret FROM crsp.dsf
            WHERE permno IN ({','.join(map(str,need))})
              AND date BETWEEN '{bhar_start.strftime('%Y-%m-%d')}' AND '{post_hi}'
              AND ret IS NOT NULL""", date_cols=["date"])
        de = db.raw_sql(f"""SELECT permno,dlstdt,dlstcd,dlret FROM crsp.dsedelist
            WHERE permno IN ({','.join(map(str,need))}) AND dlstcd IS NOT NULL""",
            date_cols=["dlstdt"])
        dmap = shumway_delist(de) if len(de) else {}
        rg = {int(p): d.sort_values("date") for p, d in rets.groupby("permno")}
        cmap = dict(zip(grp.permno.astype(int), grp.car_pct))
        smap = dict(zip(grp.permno.astype(int), grp.siccd))
        mmap = dict(zip(grp.permno.astype(int), grp.mktcap_m))

        def emit(matches, sink):
            for qp, cp in matches.items():
                if qp not in rg or cp not in rg:
                    continue
                row = {"permno": qp, "window": ws, "ctrl_permno": cp,
                       "car_pct": cmap[qp], "siccd": smap[qp], "mktcap_m": mmap[qp]}
                for h in HORIZONS:
                    end = bhar_start + pd.DateOffset(months=h)
                    mb = buy_hold(rg[qp], bhar_start, end, qp, dmap)
                    cb = buy_hold(rg[cp], bhar_start, end, cp, dmap)
                    row[f"bhar_{h}"] = (mb - cb) if (mb is not None and cb is not None) else np.nan
                sink.append(row)

        emit(m_bm, bhar_bm)
        emit(m_ss, bhar_ss)
        # monthly series (B/M matched pairs) for the calendar-time long-short
        mend = bhar_start + pd.DateOffset(months=MONTHLY_MAX_MONTHS)
        for qp, cp in m_bm.items():
            if qp in rg:
                for r in monthly_from_daily(rg[qp], bhar_start, mend, qp, dmap, MONTHLY_MAX_MONTHS):
                    r.update({"window": ws, "role": "event", "pair_permno": qp}); monthly_rows.append(r)
            if cp in rg:
                for r in monthly_from_daily(rg[cp], bhar_start, mend, cp, dmap, MONTHLY_MAX_MONTHS):
                    r.update({"window": ws, "role": "control", "pair_permno": qp}); monthly_rows.append(r)

    pd.DataFrame(bhar_bm).to_csv(f"{OUT}/placebo_bhar.csv", index=False)
    pd.DataFrame(bhar_ss).to_csv(f"{OUT}/placebo_bhar_sizeSIC.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(f"{OUT}/placebo_monthly_returns.csv", index=False)

    def stats(v):
        v = np.asarray([x for x in v if pd.notna(x)])
        if len(v) < 2:
            return (np.nan, np.nan, np.nan, len(v))
        return (v.mean() * 100, np.median(v) * 100,
                v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), len(v))
    srows = []
    for label, bh in [("bm", pd.DataFrame(bhar_bm)), ("sizeSIC", pd.DataFrame(bhar_ss))]:
        if bh.empty:
            continue
        for h in HORIZONS:
            m, md, t, n = stats(bh[f"bhar_{h}"])
            srows.append({"matching": label, "window": "POOLED", "horizon": h,
                          "n": n, "mean_bhar": m, "median_bhar": md, "t": t})
        for ws, g in bh.groupby("window"):
            for h in HORIZONS:
                m, md, t, n = stats(g[f"bhar_{h}"])
                srows.append({"matching": label, "window": ws, "horizon": h,
                              "n": n, "mean_bhar": m, "median_bhar": md, "t": t})
    pd.DataFrame(srows).to_csv(f"{OUT}/placebo_summary.csv", index=False)
    with open(f"{OUT}/placebo_meta.txt", "w") as f:
        f.write(f"windows={windows}\nnonoverlap={NONOVERLAP}\nevent_tdays={EVENT_TDAYS}\n"
                f"car_threshold={CAR_THRESHOLD}\nest_years={EST_YEARS}\n"
                f"matching=size+1digitSIC AND size+1digitSIC+B/M\n"
                f"bm_available={bm_available}\nn_qualifiers={len(screen)}\n"
                f"n_bhar_bm={len(bhar_bm)}\nn_bhar_sizeSIC={len(bhar_ss)}\n")
    print("\n==== POOLED PLACEBO (both matchings) ====")
    print(pd.DataFrame(srows)[lambda d: d.window == "POOLED"].to_string(index=False))
    print(f"\nWrote {OUT}/")
    db.close()


if __name__ == "__main__":
    main()
