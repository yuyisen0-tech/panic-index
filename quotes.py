"""Closing prices for the watchlist, as a fragment the briefing can embed.

Lives alongside the panic index because the briefing's cloud sandbox cannot
reach any market-data host — it can only read raw.githubusercontent.com, so
whatever it needs has to be computed here and committed.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import yfinance as yf

from briefing_panic import LEVERAGED, WATCHLIST

CCY = {"C38U.SI": "S$", "AJBU.SI": "S$", "S59.SI": "S$", "Z74.SI": "S$"}


def main() -> None:
    tks = list(WATCHLIST)
    df = yf.download(tks, period="10d", auto_adjust=False, progress=False,
                     threads=True, group_by="column")
    rows, asof = [], None
    for tk in tks:
        try:
            c = df["Close"][tk].dropna()
            hi = df["High"][tk].dropna()
            lo = df["Low"][tk].dropna()
        except (KeyError, TypeError):
            continue
        if len(c) < 2:
            continue
        last, prev = float(c.iloc[-1]), float(c.iloc[-2])
        chg = (last / prev - 1) * 100
        d = c.index[-1].date()
        asof = max(asof, d) if asof else d
        cur = CCY.get(tk, "$")
        cls_ = "up" if chg > 0 else ("down" if chg < 0 else "flat")
        arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "—")
        lev = " · 2倍杠杆" if tk in LEVERAGED else ""
        rng = (f"{cur}{float(lo.iloc[-1]):,.2f}–{cur}{float(hi.iloc[-1]):,.2f}"
               if len(hi) and len(lo) else "n/a")
        rows.append(
            f'      <div class="tick">\n'
            f'        <span class="name">{tk} · {WATCHLIST[tk][0]}</span>\n'
            f'        <span class="val">{cur}{last:,.2f}</span>\n'
            f'        <span class="chg {cls_}">{chg:+.2f}% {arrow}</span>\n'
            f'        <span class="ctx">{d} 收盘 · 日内 {rng}{lev}</span>\n'
            f'      </div>')

    print(f"<!-- DATA_DATE={asof} -->")
    print('    <div class="ticker-label">你的自选股 · 收盘价</div>')
    print('    <div class="ticker-grid watch-grid">')
    print("\n".join(rows))
    print("    </div>")


if __name__ == "__main__":
    main()
