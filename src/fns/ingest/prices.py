"""
Stage 1a -- daily OHLCV ingestion.

Source: Yahoo Finance via `yfinance`. Free, no API key, survivorship-bias caveats
noted in the README.

Two non-obvious decisions, both of which affect correctness downstream:

1. `auto_adjust=False`. We keep BOTH `close` (what the tape printed) and
   `adj_close` (back-adjusted for splits and dividends). Returns MUST be computed
   from adj_close: on a 4-for-1 split day, raw close drops ~75% and a raw-close
   return would hand the model a fake -75% "crash" as a training label. Raw close
   is retained only for display.

2. A warm-up buffer before `start_date` and a tail buffer after `end_date`.
   Rolling features (SMA-30, 20-day volume z-score) need ~30 prior sessions or
   they emit NaN for the first month of the sample; the forward return needs one
   session PAST the end. Fetching a buffer and trimming later is cheaper and less
   error-prone than special-casing the edges.
"""
from __future__ import annotations

import pandas as pd

from ..config import SETTINGS
from ..db import prices, upsert

WARMUP_DAYS = 90   # calendar days of history before start_date (~60 sessions)
TAIL_DAYS = 10     # calendar days after end_date, so the last label exists


def fetch_prices(
    tickers: tuple[str, ...] | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily bars and return a tidy long DataFrame.

    Returns columns: ticker, date, open, high, low, close, adj_close, volume.
    """
    import yfinance as yf

    tickers = list(tickers or SETTINGS.tickers)
    start_ts = pd.Timestamp(start or SETTINGS.start_date) - pd.Timedelta(days=WARMUP_DAYS)
    end_ts = pd.Timestamp(end or SETTINGS.end_date) + pd.Timedelta(days=TAIL_DAYS)

    # One batched HTTP request for all tickers instead of N sequential ones.
    try:
        raw = yf.download(
            tickers,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            auto_adjust=False,     # see docstring note 1
            progress=False,
            threads=True,
            group_by="column",     # -> MultiIndex columns (field, ticker)
        )
    except Exception as exc:                                   # pragma: no cover
        print(f"[prices] yfinance raised {exc!r}; falling back to Stooq")
        raw = pd.DataFrame()

    # Yahoo is an unofficial, frequently-changing endpoint. Rather than let the
    # whole pipeline die when they rotate their auth scheme, fall back to Stooq,
    # which serves plain CSV over HTTPS with no key. Two independent vendors is
    # cheap insurance for a data dependency you do not control.
    if raw.empty:
        print("[prices] yfinance returned nothing; using Stooq fallback")
        return _fetch_stooq(tickers, start_ts, end_ts)

    # yfinance returns flat columns for a single ticker and a MultiIndex for many.
    # Normalise both into the same long/tidy shape so callers never branch.
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level=1, future_stack=True).rename_axis(["Date", "Ticker"]).reset_index()
    else:
        df = raw.reset_index()
        df["Ticker"] = tickers[0]

    df = df.rename(columns={
        "Date": "date", "Ticker": "ticker", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    keep = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[keep]

    # A row with no close is a non-trading day artefact of the union calendar
    # (e.g. a holiday one exchange observes and another doesn't). Drop it: an
    # imputed price would create a fake 0% return day.
    df = df.dropna(subset=["adj_close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _fetch_stooq(tickers: list[str], start_ts, end_ts) -> pd.DataFrame:
    """Backup vendor. Stooq exposes daily bars at a stable CSV endpoint.

    Caveat carried into the README: Stooq's `close` is already split-adjusted but
    NOT dividend-adjusted, so we copy it into `adj_close`. For 1-day directional
    prediction the dividend drop-through (~0.5%/yr on these names) is immaterial,
    but it is a real approximation and should be stated, not hidden.
    """
    import io
    import urllib.request

    frames = []
    for t in tickers:
        url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            csv = urllib.request.urlopen(req, timeout=30).read().decode()
            d = pd.read_csv(io.StringIO(csv))
        except Exception as exc:
            print(f"[prices]   stooq {t} failed: {exc}")
            continue
        if d.empty or "Close" not in d.columns:
            continue
        d.columns = [c.lower() for c in d.columns]
        d["ticker"] = t
        d["adj_close"] = d["close"]
        frames.append(d)

    if not frames:
        raise RuntimeError("both yfinance and Stooq failed -- check connectivity")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    return df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]


def ingest_prices(**kwargs) -> int:
    """Fetch + persist. Returns row count written."""
    df = fetch_prices(**kwargs)
    n = upsert(prices, df)
    print(f"[prices] {n:,} rows for {df.ticker.nunique()} tickers "
          f"({df.date.min()} -> {df.date.max()})")
    return n


if __name__ == "__main__":
    from ..db import init_db
    init_db()
    ingest_prices()
