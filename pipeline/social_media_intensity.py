#!/usr/bin/env python3
"""
social_media_intensity.py - Attention-intensity split of the 44 meme stocks.

Builds a composite attention score, flags a "pure-meme" subsample, and compares
BHAR for that subsample with the rest. Signals:

  1. TURNOVER (local, reproducible): mean daily share turnover over
     Feb-Mar 2021 from the CRSP volume data, a standard retail-attention proxy
     (Barber and Odean).

  2. GOOGLE TRENDS / GSVI (optional, needs network access): search interest,
     queried in batches of 4 against a common anchor (GME) so tickers share one
     scale (Da, Engelberg and Gao 2011). Skip with --no-trends.

Optional: --manual-wsb CSV (cols ticker,wsb_count) adds hand-collected WSB
mention counts as a third signal.

Composite = rank-average across available signals. pure_meme_flag =
composite >= --threshold (default 0.65).

Outputs (meme_stock_analysis/v13_outputs/):
  social_intensity_scores.csv    permno,ticker,name,turnover,gtrends,wsb,
                                  composite,pure_meme_flag,signals_used
  pure_meme_subsample.csv
  identification_bhar_split.csv   group(pure/rest),n,bhar36,t36,bhar48,t48

Install: pip install pandas requests pytrends
Run:     python3 social_media_intensity.py --allow-run --no-trends
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _guard import check_run_guard  # noqa: E402
import _base as B  # noqa: E402

ANCHOR = "GME"                       # common Trends anchor across batches
TURN_START = pd.Timestamp("2021-02-01")
TURN_END = pd.Timestamp("2021-03-31")
TRENDS_WINDOW = "2020-12-01 2021-03-31"


def turnover_signal():
    """Mean daily turnover (vol / shares outstanding) per meme stock over the
    post-squeeze window. Fully local + reproducible. Returns permno->float."""
    rm = B.returns_meme()
    w = rm[(rm["date"] >= TURN_START) & (rm["date"] <= TURN_END)].copy()
    # vol in shares; shrout in thousands -> shares outstanding = shrout*1000
    w["turn"] = w["vol"] / (w["shrout"] * 1000.0)
    w = w.replace([np.inf, -np.inf], np.nan)
    return w.groupby("permno")["turn"].mean().to_dict()


def gtrends_batched(tickers, sleep=20.0, retries=3):
    """Peak in-window search interest per ticker, normalized to the anchor
    (GME) so all batches share one scale. Returns ticker->float (NaN on fail)."""
    try:
        from pytrends.request import TrendReq
    except Exception:
        print("  (pytrends not installed; skipping Trends)")
        return {}
    others = [t for t in tickers if t != ANCHOR]
    batches = [others[i:i + 4] for i in range(0, len(others), 4)]
    out = {ANCHOR: 1.0}
    for bi, batch in enumerate(batches):
        terms = [ANCHOR] + batch
        peaks = None
        for attempt in range(retries):
            try:
                py = TrendReq(hl="en-US", tz=300, retries=2, backoff_factor=0.5)
                py.build_payload(terms, timeframe=TRENDS_WINDOW, geo="US")
                df = py.interest_over_time()
                if df is not None and not df.empty:
                    peaks = {t: float(df[t].max()) for t in terms if t in df}
                break
            except Exception:
                time.sleep(sleep * (attempt + 1))
        if peaks and peaks.get(ANCHOR, 0) > 0:
            ref = peaks[ANCHOR]
            for t in batch:
                out[t] = peaks.get(t, np.nan) / ref   # relative to GME
        else:
            for t in batch:
                out[t] = np.nan
        print(f"  Trends batch {bi+1}/{len(batches)}: "
              f"{ {t: round(out.get(t), 2) if not np.isnan(out.get(t, np.nan)) else None for t in batch} }")
        time.sleep(sleep)
    return out


def pct_rank(series):
    return pd.Series(series, dtype="float64").rank(pct=True)


def main():
    check_run_guard("social_media_intensity.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--no-trends", action="store_true",
                    help="skip Google Trends; use turnover (+manual) only")
    ap.add_argument("--sleep", type=float, default=20.0,
                    help="seconds between Trends batches (raise if rate-limited)")
    ap.add_argument("--manual-wsb", default=None)
    args = ap.parse_args()
    B.ensure_out()

    scr = B.screen()[["permno", "ticker", "comnam"]].rename(
        columns={"comnam": "name"})
    df = scr.copy()

    # signal 1: turnover (always)
    turn = turnover_signal()
    df["turnover"] = df["permno"].map(turn)

    # signal 2: trends (optional)
    if args.no_trends:
        df["gtrends"] = np.nan
    else:
        gt = gtrends_batched(df["ticker"].tolist(), sleep=args.sleep)
        df["gtrends"] = df["ticker"].map(gt)

    # signal 3: manual WSB (optional)
    if args.manual_wsb and os.path.exists(args.manual_wsb):
        man = pd.read_csv(args.manual_wsb).set_index("ticker")["wsb_count"]
        df["wsb"] = df["ticker"].map(man)
        print(f"  using manual WSB counts from {args.manual_wsb}")
    else:
        df["wsb"] = np.nan

    parts, used = [], []
    for col in ("turnover", "gtrends", "wsb"):
        if df[col].notna().sum() >= 2:
            parts.append(pct_rank(df[col]))
            used.append(col)
    df["composite"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    df["pure_meme_flag"] = (df["composite"] >= args.threshold).astype(int)
    df["signals_used"] = ",".join(used)
    df = df.sort_values("composite", ascending=False)
    df.to_csv(f"{B.OUT}/social_intensity_scores.csv", index=False)
    pure = df[df.pure_meme_flag == 1]
    pure.to_csv(f"{B.OUT}/pure_meme_subsample.csv", index=False)
    print(f"\nSignals used: {used}")
    print(f"Pure-meme subsample (composite >= {args.threshold}): n={len(pure)}")
    print("  " + ", ".join(pure.ticker))

    # identification test
    matches = B.build_matches(scr.permno.tolist())
    panel = B.bhar_panel(matches)
    pure_set = set(pure.permno)
    rows = []
    for label, perms in [("pure_meme", pure_set),
                         ("rest", set(df.permno) - pure_set)]:
        sub = panel[panel.permno.isin(perms)]
        out = {"group": label}
        for h in (36, 48):
            x = sub[sub.horizon_mo == h].dropna(subset=["bhar"])["bhar"].values
            m, tt, n = B.tstat(x)
            out["n"] = n
            out[f"bhar{h}"] = round(m * 100, 1) if n else np.nan
            out[f"t{h}"] = round(tt, 2) if n else np.nan
        rows.append(out)
    idsplit = pd.DataFrame(rows)
    idsplit.to_csv(f"{B.OUT}/identification_bhar_split.csv", index=False)
    print("\nIdentification test (BHAR: pure-meme vs rest):")
    print(idsplit.to_string(index=False))


if __name__ == "__main__":
    main()
