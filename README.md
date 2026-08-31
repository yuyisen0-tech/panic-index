# panic-index

Daily market-stress readings for the Straits Times Index, the S&P 500 and a
12-name watchlist. Rebuilt every weekday by GitHub Actions so the numbers stay
current whether or not any particular machine is switched on.

`data/panic-block.html` and `data/quotes-block.html` are fragments meant to be
pasted into a briefing page; `data/panic.json` is the same readings as data.

Scores run 0 (maximum stress) to 100 (maximum calm), each indicator ranked
against its own trailing five years. **This measures the present, it does not
forecast** — see the research write-up for what that distinction cost to
establish.

Only public market data lives here. No account, position or credential data.
