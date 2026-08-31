"""Standalone panic-index block for the daily briefing.

Self-contained on purpose: the daily task runs in a sandbox that cannot see this
project, so this file imports nothing local. It downloads its own data and
prints an HTML fragment on stdout.

    pip install yfinance pandas numpy requests
    python briefing_panic.py            # HTML fragment to stdout
    python briefing_panic.py --json     # machine-readable instead
    python briefing_panic.py --out x.html

Weights and formulas are copied verbatim from the validated research; see
README.md for what these numbers can and cannot tell you.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings

warnings.filterwarnings("ignore")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import numpy as np
import pandas as pd

PCT_WIN, PCT_MIN = 1260, 252
START = "2000-01-01"          # long enough that the "historical percentile" means something

# ticker -> (显示名, 所属市场). Market picks the relative-strength benchmark:
# an SGX name measured against the S&P would mostly read the time-zone gap.
WATCHLIST = {
    "AAPL":    ("苹果", "us"),
    "NVDA":    ("英伟达", "us"),
    "META":    ("Meta", "us"),
    "TSLA":    ("特斯拉", "us"),
    "TLT":     ("20年期美债ETF", "us"),
    "INTC":    ("英特尔", "us"),
    "XLU":     ("公用事业ETF", "us"),
    "AGQ":     ("2倍做多白银ETF", "us"),
    "C38U.SI": ("CapitaLand综合商业信托", "sg"),
    "AJBU.SI": ("吉宝数据中心REIT", "sg"),
    "S59.SI":  ("新翔集团", "sg"),
    "Z74.SI":  ("新加坡电信", "sg"),
}
LEVERAGED = {"AGQ"}      # daily-compounding product: decays in chop

US_STOCKS = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","JNJ",
             "V","PG","UNH","HD","MA","XOM","CVX","BAC","ABBV","PFE","KO","PEP",
             "MRK","COST","WMT","CSCO","ADBE","CRM","ACN","MCD","DIS","NFLX",
             "INTC","AMD","QCOM","TXN","IBM","GE","CAT","BA","MMM","HON","UPS",
             "LMT","GS","MS","C","WFC","AXP","BLK","T","VZ","CMCSA","NKE","SBUX",
             "LOW","TGT","CVS","AMGN","GILD","LLY","TMO","DHR","ABT","BMY","SO",
             "DUK","NEE","D","EXC"]

SG_STOCKS = ["D05.SI","O39.SI","U11.SI","C38U.SI","A17U.SI","M44U.SI","ME8U.SI",
             "N2IU.SI","AJBU.SI","J69U.SI","BUOU.SI","K71U.SI","T82U.SI","P40U.SI",
             "HMN.SI","CY6U.SI","TS0U.SI","Q5T.SI","CRPU.SI","UD1U.SI","A7RU.SI",
             "Z74.SI","9CI.SI","C07.SI","BN4.SI","S68.SI","S63.SI","U96.SI",
             "F34.SI","Y92.SI","BS6.SI","G13.SI","C09.SI","U14.SI","V03.SI",
             "5E2.SI","H78.SI","D01.SI","C6L.SI","J36.SI","S58.SI","G07.SI",
             "CC3.SI","OV8.SI","BSL.SI","S59.SI","E5H.SI","H02.SI","NS8U.SI",
             "AWX.SI","S07.SI"]
SG_REITS = ["C38U.SI","A17U.SI","M44U.SI","ME8U.SI","N2IU.SI","AJBU.SI","J69U.SI",
            "BUOU.SI","K71U.SI","T82U.SI","P40U.SI","HMN.SI","CY6U.SI","TS0U.SI",
            "Q5T.SI","CRPU.SI","UD1U.SI","A7RU.SI"]
SG_BANKS = ["D05.SI","O39.SI","U11.SI"]

INDICES = ["^GSPC","^VIX","^VIX3M","SPY","TLT","^STI","SGD=X"]
# watchlist names that are not already in the breadth universes
EXTRA = ["INTC","XLU","AGQ","C38U.SI","AJBU.SI","S59.SI","Z74.SI"]

US_WEIGHTS = {"spx_dist_200dma":.130,"us_breadth_200dma":.129,"vix_level":.127,
              "stlfsi":.118,"vix_term_structure":.113,"us_avg_corr":.107,
              "credit_baa_chg60":.097,"us_volume_surge":.090,"equity_vs_bond":.089}
SG_WEIGHTS = {"sg_net_high_low":.170,"sg_reit_drawdown":.158,"sti_rvol":.135,
              "sti_dist_200dma":.128,"sg_volume_surge":.121,"sgd_strength_60d":.115,
              "sg_avg_corr":.093,"sg_bank_vs_sti":.079}
STOCK_WEIGHTS = {"drawdown_252":.18,"dist_200dma":.16,"pos_in_52w":.14,
                 "rvol_20":.13,"rel_vs_spx_60":.13,"mom_120":.10,
                 "volume_surge":.09,"down_day_share_60":.07}

BANDS = [(0,20,"极度恐慌"),(20,40,"恐慌"),(40,60,"中性"),(60,80,"乐观"),(80,101,"亢奋")]
# measured at historical crash troughs -- see README
TROUGH_REF = {"us":"13次崩盘底部均值 12.4，最高 19.5","sg":"10次崩盘底部均值 21.3，最高 51.1"}


def band(v):
    for lo, hi, nm in BANDS:
        if lo <= v < hi:
            return nm
    return "n/a"


def pit(s, invert=False):
    r = s.rolling(PCT_WIN, min_periods=PCT_MIN).rank(pct=True) * 100
    return (100 - r) if invert else r


def rvol(px, w=20):
    return px.pct_change().rolling(w, min_periods=w//2).std() * np.sqrt(252) * 100


def vol_surge(panel, min_names=8):
    v = panel.replace(0, np.nan)
    ratio = v.rolling(5, min_periods=3).mean() / v.rolling(60, min_periods=30).mean()
    return ratio.median(axis=1).where(ratio.notna().sum(axis=1) >= min_names)


def avg_corr(panel, w=60, min_names=8):
    ret = panel.pct_change()
    out = (ret.mean(axis=1).rolling(w, min_periods=w//2).var()
           / ret.rolling(w, min_periods=w//2).var().mean(axis=1).replace(0, np.nan))
    return out.where(ret.notna().sum(axis=1) >= min_names)


def net_high_low(panel, w=252, band_=.05, min_names=8):
    hi, lo = panel.rolling(w, min_periods=w//2).max(), panel.rolling(w, min_periods=w//2).min()
    ok = hi.notna() & lo.notna() & panel.notna()
    n = ok.sum(axis=1)
    net = (((panel >= hi*(1-band_)) & ok).sum(axis=1)
           - ((panel <= lo*(1+band_)) & ok).sum(axis=1))
    return (net / n.replace(0, np.nan) * 100).where(n >= min_names)


def basket(panel, cols, min_names=3):
    cols = [c for c in cols if c in panel.columns]
    ret = panel[cols].pct_change()
    eq = ret.mean(axis=1).where(ret.notna().sum(axis=1) >= min_names)
    first = eq.first_valid_index()
    if first is None:
        return pd.Series(np.nan, index=panel.index)
    return (1 + eq.loc[first:].fillna(0)).cumprod().reindex(panel.index)


def fred(series_id):
    import requests
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                     timeout=45)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[series_id].apply(pd.to_numeric, errors="coerce")


def download():
    import yfinance as yf
    tickers = sorted(set(INDICES + US_STOCKS + SG_STOCKS + EXTRA + list(WATCHLIST)))
    raw = yf.download(tickers, start=START, auto_adjust=True, progress=False,
                      threads=True, group_by="column")
    close = raw["Close"].dropna(axis=1, how="all")
    vol = raw["Volume"].dropna(axis=1, how="all")
    for d in (close, vol):
        d.index = pd.to_datetime(d.index).tz_localize(None)
    return close.sort_index(), vol.sort_index()


def composite(score, weights, min_present=.5):
    cols = [c for c in weights if c in score.columns]
    w = pd.Series({c: weights[c] for c in cols})
    sub = score[cols]
    present = sub.notna().mul(w, axis=1).sum(axis=1)
    return (sub.mul(w, axis=1).sum(axis=1, min_count=1)
            / present.replace(0, np.nan)).where(present >= min_present)


def us_index(close, vol):
    spx = close["^GSPC"].dropna()
    stk = close[[c for c in US_STOCKS if c in close.columns]]
    sv = vol[[c for c in US_STOCKS if c in vol.columns]]
    r = pd.DataFrame(index=spx.index)
    r["spx_dist_200dma"] = spx / spx.rolling(200, min_periods=100).mean() - 1
    ma = stk.rolling(200, min_periods=100).mean()
    ok = ma.notna() & stk.notna()
    r["us_breadth_200dma"] = ((stk > ma) & ok).sum(axis=1) / ok.sum(axis=1).replace(0, np.nan) * 100
    r["vix_level"] = close["^VIX"].reindex(spx.index)
    r["vix_term_structure"] = close["^VIX"] / close["^VIX3M"]
    r["us_avg_corr"] = avg_corr(stk, min_names=20).reindex(spx.index)
    r["us_volume_surge"] = vol_surge(sv, min_names=20).reindex(spx.index)
    if {"SPY", "TLT"} <= set(close.columns):
        r["equity_vs_bond"] = ((close.SPY/close.SPY.shift(60))
                               - (close.TLT/close.TLT.shift(60))).reindex(spx.index)
    try:
        r["stlfsi"] = fred("STLFSI4").reindex(spx.index).ffill(limit=10)
        baa = fred("BAA10Y").reindex(spx.index).ffill(limit=5)
        r["credit_baa_chg60"] = baa - baa.shift(60)
    except Exception as e:                                     # keep going without FRED
        print(f"<!-- FRED unavailable: {e} -->", file=sys.stderr)
    fear = {"spx_dist_200dma":0,"us_breadth_200dma":0,"vix_level":1,"stlfsi":1,
            "vix_term_structure":1,"us_avg_corr":1,"credit_baa_chg60":1,
            "us_volume_surge":1,"equity_vs_bond":0}
    sc = pd.DataFrame({c: pit(r[c], bool(fear[c])) for c in fear if c in r},
                      index=r.index)
    return composite(sc, US_WEIGHTS), sc


def sg_index(close, vol):
    if "^STI" not in close.columns:
        return pd.Series(dtype=float), pd.DataFrame()
    sti = close["^STI"].dropna()
    stk = close[[c for c in SG_STOCKS if c in close.columns]]
    sv = vol[[c for c in SG_STOCKS if c in vol.columns]]
    r = pd.DataFrame(index=sti.index)
    r["sti_dist_200dma"] = sti / sti.rolling(200, min_periods=100).mean() - 1
    r["sti_rvol"] = rvol(sti)
    r["sg_net_high_low"] = net_high_low(stk).reindex(sti.index)
    reit = basket(stk, SG_REITS)
    r["sg_reit_drawdown"] = (reit / reit.rolling(252, min_periods=60).max() - 1).reindex(sti.index)
    bank = basket(stk, SG_BANKS)
    r["sg_bank_vs_sti"] = ((bank/bank.shift(60))
                           - (sti.reindex(bank.index).ffill()/sti.reindex(bank.index).ffill().shift(60))
                           ).reindex(sti.index)
    r["sg_avg_corr"] = avg_corr(stk).reindex(sti.index)
    r["sg_volume_surge"] = vol_surge(sv).reindex(sti.index)
    if "SGD=X" in close.columns:
        sgd = close["SGD=X"].dropna()
        r["sgd_strength_60d"] = -(sgd/sgd.shift(60) - 1).reindex(sti.index)
    fear = {"sg_net_high_low":0,"sg_reit_drawdown":0,"sti_rvol":1,"sti_dist_200dma":0,
            "sg_volume_surge":1,"sgd_strength_60d":0,"sg_avg_corr":1,"sg_bank_vs_sti":0}
    sc = pd.DataFrame({c: pit(r[c], bool(fear[c])) for c in fear if c in r},
                      index=r.index)
    return composite(sc, SG_WEIGHTS), sc


def stock_index(close, vol, tk, mkt="us"):
    px = close[tk].dropna()
    bench_tk = "^STI" if mkt == "sg" else "^GSPC"
    spx = close[bench_tk].reindex(px.index).ffill()
    r = pd.DataFrame(index=px.index)
    r["drawdown_252"] = px / px.rolling(252, min_periods=60).max() - 1
    r["dist_200dma"] = px / px.rolling(200, min_periods=100).mean() - 1
    r["mom_120"] = px / px.shift(120) - 1
    r["rvol_20"] = rvol(px)
    r["rel_vs_spx_60"] = (px/px.shift(60)) - (spx/spx.shift(60))
    lo = px.rolling(252, min_periods=60).min()
    hi = px.rolling(252, min_periods=60).max()
    r["pos_in_52w"] = (px - lo) / (hi - lo)
    r["down_day_share_60"] = (px.pct_change() < 0).rolling(60, min_periods=30).mean() * 100
    if tk in vol.columns:
        v = vol[tk].replace(0, np.nan)
        r["volume_surge"] = v.rolling(5, min_periods=3).mean() / v.rolling(60, min_periods=30).mean()
    fear = {"drawdown_252":0,"dist_200dma":0,"mom_120":0,"rvol_20":1,
            "rel_vs_spx_60":0,"pos_in_52w":0,"down_day_share_60":1,"volume_surge":1}
    sc = pd.DataFrame({c: pit(r[c], bool(fear[c])) for c in fear if c in r},
                      index=r.index)
    return composite(sc, STOCK_WEIGHTS)


def summarise(s: pd.Series) -> dict | None:
    s = s.dropna()
    if s.empty:
        return None
    v = float(s.iloc[-1])
    return {"score": round(v, 1), "band": band(v),
            "pct": int(round(float((s < v).mean() * 100))),
            "date": str(s.index[-1].date()),
            "d5": round(v - float(s.iloc[-6]), 1) if len(s) > 6 else None,
            "d20": round(v - float(s.iloc[-21]), 1) if len(s) > 21 else None}


def cls(v):
    return "down" if v < 40 else ("up" if v >= 60 else "flat")


def arrow(d):
    return "▲" if (d or 0) > 0 else ("▼" if (d or 0) < 0 else "—")


def render(mkt: dict, stocks: dict) -> str:
    p = []
    stamp = (mkt.get("us") or mkt.get("sg") or {}).get("date", "unknown")
    p.append(f"<!-- DATA_DATE={stamp} -->")
    p.append('  <div class="ticker-section">')
    p.append('    <div class="ticker-label">恐慌指数 · 0=极度恐慌 100=亢奋 · 5年滚动分位</div>')
    p.append('    <div class="ticker-grid">')
    for key, label in (("us", "美股大盘 S&amp;P 500"), ("sg", "新加坡 STI")):
        d = mkt.get(key)
        if not d:
            continue
        p.append(f'      <div class="tick"><span class="name">{label}</span>'
                 f'<span class="val">{d["score"]}</span>'
                 f'<span class="chg {cls(d["score"])}">{d["band"]} '
                 f'{arrow(d["d20"])}</span>'
                 f'<span class="ctx">历史分位 {d["pct"]}% · 20日 {d["d20"]:+.1f} · '
                 f'{TROUGH_REF[key]}</span></div>')
    p.append('    </div>')
    p.append('  </div>')

    for mkt, label in (("us", "美股自选 · 个股恐慌指数"), ("sg", "新交所自选 · 个股恐慌指数")):
        rows = [(tk, d) for tk, d in stocks.items()
                if d and WATCHLIST[tk][1] == mkt]
        if not rows:
            continue
        p.append('  <div class="ticker-section">')
        p.append(f'    <div class="ticker-label">{label}</div>')
        p.append('    <div class="ticker-grid watch-grid">')
        for tk, d in sorted(rows, key=lambda x: x[1]["score"]):
            lev = ' · <strong>2倍杠杆</strong>' if tk in LEVERAGED else ''
            p.append(f'      <div class="tick">'
                     f'<span class="name">{tk} · {WATCHLIST[tk][0]}</span>'
                     f'<span class="val">{d["score"]}</span>'
                     f'<span class="chg {cls(d["score"])}">{d["band"]} {arrow(d["d20"])}</span>'
                     f'<span class="ctx">历史分位 {d["pct"]}% · 5日 {d["d5"]:+.1f} · '
                     f'20日 {d["d20"]:+.1f}{lev}</span></div>')
        p.append('    </div>')
        p.append('  </div>')

    p.append('  <div class="note" style="margin-top:10px;">'
             '<strong>怎么读：</strong>这是状态温度计，不是买卖信号。回测 138 次个股极度恐慌'
             '（读数&lt;20，12只自选股 1988-2026）：12个月后为正 84/134（63%），中位数 +11.6%，'
             '<strong>但信号后中位数还要再跌 12.3%，最差一次 −80.9%</strong>。'
             '个股赔率明显差于大盘，且单只股票可以归零、杠杆ETF会衰减。</div>')
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    close, vol = download()
    us, _ = us_index(close, vol)
    sg, _ = sg_index(close, vol)
    mkt = {"us": summarise(us), "sg": summarise(sg)}
    stocks = {tk: summarise(stock_index(close, vol, tk, mkt))
              for tk, (_, mkt) in WATCHLIST.items() if tk in close.columns}

    out = (json.dumps({"market": mkt, "stocks": stocks}, ensure_ascii=False, indent=2)
           if a.json else render(mkt, stocks))
    if a.out:
        open(a.out, "w", encoding="utf-8").write(out)
        print(f"written to {a.out}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
