"""
Stage 1b -- financial headline ingestion + trading-session attribution.

Source: FNSPID (Financial News and Stock Price Integration Dataset), the
NASDAQ-100 / post-2019 slice hosted on the HuggingFace Hub. ~62k REAL Nasdaq.com
articles with UTC publication timestamps, ticker tags and publishers.

======================================================================
THE MOST IMPORTANT FUNCTION IN THIS PROJECT: assign_trade_date()
======================================================================
Every naive "news sentiment predicts stock moves" project dies on lookahead bias.
The failure mode: you take all news dated 2023-06-15 and regress it against
2023-06-15's return. But an earnings beat published at 16:30 ET is AFTER the
close -- the return it "predicts" already happened, and worse, the market's
reaction is what generated the headline's tone. Accuracy looks like 70%. It is
pure leakage and it evaporates the moment you trade it.

The fix is a point-in-time cutoff:
  * convert the UTC publish time into US/Eastern (handles EST/EDT automatically),
  * news published BEFORE 16:00 ET can act on TODAY's session,
  * news published AT/AFTER 16:00 ET -- or on a weekend/holiday -- rolls forward
    to the NEXT session the exchange was actually open.

We derive the trading calendar from the distinct dates already in the `prices`
table, so it is the real observed calendar (holidays, half-days, the 2020 shutdowns)
rather than an approximation, and it needs no extra dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RAW_DIR, SETTINGS
from ..db import get_engine, headlines, make_uid, read_sql, upsert

# The dataset ships the article body and four different auto-summaries; each is
# hundreds of KB per 1k rows. We only need the headline. Reading a column subset
# out of Parquet is a columnar projection -- it never touches those bytes on disk,
# which turns a 204 MB read into roughly 8 MB.
NEEDED_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Publisher", "Url"]


def download_corpus() -> list[str]:
    """Fetch (and locally cache) the FNSPID parquet shards. Returns file paths.

    hf_hub_download is content-addressed: the second call is a cache hit and
    costs nothing, so this is safe to call on every pipeline run.
    """
    from huggingface_hub import hf_hub_download

    return [
        hf_hub_download(
            repo_id=SETTINGS.hf_dataset_repo,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(RAW_DIR),   # keep the shards in the project, not ~/.cache
        )
        for fname in SETTINGS.hf_dataset_files
    ]


def load_raw(paths: list[str], tickers: list[str]) -> pd.DataFrame:
    """Read the parquet shards with a column + row projection to our universe.

    We filter each shard down to `tickers` BEFORE concatenating. Reading all
    shards into one frame first would materialise ~350k rows of article text in
    RAM for no reason; this keeps peak memory to a few hundred MB.
    """
    import pyarrow.parquet as pq

    frames = []
    for path in paths:
        df = pq.read_table(path, columns=NEEDED_COLUMNS).to_pandas()
        df = df.rename(columns={
            "Date": "published_at", "Article_title": "title",
            "Stock_symbol": "ticker", "Publisher": "publisher", "Url": "url",
        })
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        frames.append(df[df["ticker"].isin(tickers)].copy())
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Text + timestamp hygiene. Every drop here is a deliberate choice."""
    n0 = len(df)

    # Timestamps arrive as "2023-12-16 22:00:00 UTC". format="mixed" tolerates the
    # handful of rows with a different layout; errors="coerce" turns the
    # unparseable ones into NaT so one bad row can't kill the run.
    df["published_at"] = pd.to_datetime(
        df["published_at"], utc=True, format="mixed", errors="coerce"
    )
    df = df.dropna(subset=["published_at"])

    df["title"] = df["title"].astype(str).str.strip()
    # Collapse runs of whitespace: the corpus has embedded newlines/tabs that
    # would otherwise become distinct tokens and distinct hashes for one article.
    df["title"] = df["title"].str.replace(r"\s+", " ", regex=True)

    # Sub-15-character "headlines" are scraping artefacts ("Read more", tickers
    # alone). They carry no sentiment and would dilute the daily average.
    df = df[df["title"].str.len() >= 15]

    # Exact-duplicate articles syndicated across the wire on the same day.
    df = df.drop_duplicates(subset=["ticker", "published_at", "title"])

    df["publisher"] = df["publisher"].fillna("").astype(str).str.slice(0, 128)
    df["url"] = df["url"].fillna("").astype(str)
    print(f"[headlines] cleaned {n0:,} -> {len(df):,} rows")
    return df


def trading_sessions() -> np.ndarray:
    """The real trading calendar, read back out of the prices table."""
    s = read_sql("SELECT DISTINCT date FROM prices ORDER BY date")
    if s.empty:
        raise RuntimeError("prices table is empty -- run price ingestion first")
    return pd.to_datetime(s["date"]).values.astype("datetime64[D]")


def assign_trade_date(published_at_utc: pd.Series, sessions: np.ndarray,
                      extra_lag: int | None = None) -> pd.Series:
    """Map each UTC publish timestamp to the session it may first act upon.

    Vectorised (no Python loop over 60k rows) via searchsorted.
    """
    # 1) UTC -> exchange local time. pandas applies the correct EST/EDT offset
    #    per-timestamp, so a July article and a January article are both right.
    local = published_at_utc.dt.tz_convert(SETTINGS.market_tz)

    # 2) Split the cutoff ("16:00") into hour/minute and compare against the
    #    wall clock. `after_close` is True for anything at or past the bell.
    cut_h, cut_m = (int(x) for x in SETTINGS.session_cutoff_et.split(":"))
    minutes = local.dt.hour * 60 + local.dt.minute
    after_close = minutes >= (cut_h * 60 + cut_m)

    # 3) Candidate calendar day: same day if it beat the bell, else tomorrow.
    #    tz_localize(None) drops the tz so we can do plain date arithmetic.
    cand = local.dt.tz_localize(None).dt.normalize()
    cand = cand + pd.to_timedelta(after_close.astype(int), unit="D")

    # 4) Snap forward to the next REAL session. searchsorted with side="left"
    #    returns the index of the first session >= cand, which is exactly
    #    "today if the market is open, otherwise the next open day" -- this
    #    absorbs weekends and every market holiday in one step.
    idx = np.searchsorted(sessions, cand.values.astype("datetime64[D]"), side="left")

    # 5) Conservative extra lag. See Settings.extra_lag_sessions: this corpus
    #    only records a publication DATE, and the lead-lag study proves day-D
    #    headlines are contemporaneous with day-D returns. Pushing them one more
    #    session guarantees they were public before the close we trade against.
    lag = SETTINGS.extra_lag_sessions if extra_lag is None else extra_lag
    idx = idx + lag

    # Articles published after the last session we have prices for cannot be
    # evaluated; mark them NaT and drop upstream.
    valid = idx < len(sessions)
    out = np.full(len(cand), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    out[valid] = sessions[idx[valid]]
    return pd.Series(pd.to_datetime(out), index=published_at_utc.index)


def ingest_headlines(tickers: tuple[str, ...] | list[str] | None = None,
                     extra_lag: int | None = None) -> int:
    tickers = list(tickers or SETTINGS.tickers)
    paths = download_corpus()
    df = load_raw(paths, tickers)
    df = clean(df)

    sessions = trading_sessions()
    df["trade_date"] = assign_trade_date(df["published_at"], sessions, extra_lag)
    df = df.dropna(subset=["trade_date"])

    # Clamp to the configured study window.
    lo, hi = pd.Timestamp(SETTINGS.start_date), pd.Timestamp(SETTINGS.end_date)
    df = df[(df["trade_date"] >= lo) & (df["trade_date"] <= hi)]

    # Store timestamps as naive UTC -- SQLite has no tz-aware type, and keeping
    # one canonical zone in the DB avoids "which zone is this?" bugs later.
    df["published_at"] = df["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["trade_date"] = df["trade_date"].dt.date

    df["headline_uid"] = [
        make_uid(t, p, x) for t, p, x in zip(df["ticker"], df["published_at"], df["title"])
    ]
    df = df.drop_duplicates(subset=["headline_uid"])

    # Re-ingesting with different settings (a new lag, a shorter window) produces
    # a DIFFERENT set of uids. upsert() only inserts and updates -- it never
    # deletes -- so rows from a previous run whose uid is no longer produced
    # would linger with a stale trade_date and silently pollute the feature
    # build. Clearing this universe's rows first makes ingestion a true
    # REPLACE for the tickers in scope, so the table always reflects exactly
    # one consistent configuration. The sentiment cache is keyed independently
    # by uid, so nothing expensive is lost.
    with get_engine().begin() as conn:
        deleted = conn.execute(
            headlines.delete().where(headlines.c.ticker.in_(tickers))
        ).rowcount
    if deleted:
        print(f"[headlines] cleared {deleted:,} existing rows for this universe")

    n = upsert(headlines, df)
    print(f"[headlines] {n:,} rows | {df.ticker.nunique()} tickers | "
          f"{df.trade_date.min()} -> {df.trade_date.max()}")
    return n


if __name__ == "__main__":
    ingest_headlines()
