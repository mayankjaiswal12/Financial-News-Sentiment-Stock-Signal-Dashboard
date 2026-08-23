"""
SQL schema + tiny persistence helpers.

WHY SQLAlchemy Core (Table/MetaData) and not the ORM:
this is an analytics pipeline, not a CRUD app. We move DataFrames in and out in
bulk. The ORM's unit-of-work/identity-map buys us nothing here and costs a lot of
speed. Core gives us portable DDL (same code runs on SQLite and Postgres),
real indexes, and typed columns -- without object hydration overhead.

WHY a database at all instead of just parquet files:
1. Idempotency. Re-running ingest must not duplicate rows. A UNIQUE key + upsert
   gives us that for free; with files you end up hand-rolling dedupe.
2. The expensive step (FinBERT over ~15k headlines) is cached in `sentiment`,
   keyed by a content hash. Re-running the pipeline costs ~0 GPU seconds.
3. The API and the dashboard query it directly -- no pickle passing.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd
from sqlalchemy import (
    Column, Date, DateTime, Float, Integer, MetaData, String, Table,
    create_engine, Index, text,
)
from sqlalchemy.engine import Engine

from .config import DB_URL

metadata = MetaData()

# ---------------------------------------------------------------------------
# headlines: one row per (ticker, article). Raw text, never mutated after insert.
# ---------------------------------------------------------------------------
headlines = Table(
    "headlines", metadata,
    Column("headline_uid", String(40), primary_key=True),  # sha1 -> stable, idempotent
    Column("ticker", String(12), nullable=False),
    Column("published_at", DateTime, nullable=False),      # stored as naive UTC
    Column("trade_date", Date, nullable=False),            # ET session it may act on
    Column("title", String, nullable=False),
    Column("publisher", String(128)),
    Column("url", String),
    Index("ix_headlines_ticker_date", "ticker", "trade_date"),
)

# ---------------------------------------------------------------------------
# sentiment: kept SEPARATE from headlines, keyed by (uid, model).
# WHY: lets us score the same corpus with FinBERT today and another model
# tomorrow, compare them, and never re-run inference we already paid for.
# ---------------------------------------------------------------------------
sentiment = Table(
    "sentiment", metadata,
    # Composite PRIMARY KEY, not merely a UNIQUE constraint: `upsert()` builds its
    # ON CONFLICT target from the primary key, so a table with no PK would emit
    # `ON CONFLICT ()` -- a syntax error. The PK is what makes re-scoring idempotent.
    Column("headline_uid", String(40), primary_key=True),
    Column("model", String(64), primary_key=True),
    Column("p_neg", Float, nullable=False),
    Column("p_neu", Float, nullable=False),
    Column("p_pos", Float, nullable=False),
    Column("score", Float, nullable=False),   # p_pos - p_neg, in [-1, 1]
    Column("label", String(8), nullable=False),
)

# ---------------------------------------------------------------------------
# prices: daily OHLCV. PK(ticker, date) makes re-download a no-op upsert.
# ---------------------------------------------------------------------------
prices = Table(
    "prices", metadata,
    Column("ticker", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float),
    Column("close", Float), Column("adj_close", Float), Column("volume", Float),
)

# ---------------------------------------------------------------------------
# features: the modelling table -- one row per (ticker, trading day).
# This is the join of sentiment aggregates and price/technical controls, plus
# the forward-looking label. It is the ONLY table train.py reads.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: tuple[str, ...] = (
    # --- sentiment block (all computed from news strictly BEFORE the session) ---
    "n_headlines",       # news volume; attention proxy
    "sent_mean",         # average FinBERT score of the day's headlines
    "sent_std",          # disagreement across headlines
    "sent_pos_frac",     # share classified positive
    "sent_neg_frac",     # share classified negative
    "sent_net",          # pos_frac - neg_frac; robust to score miscalibration
    "sent_ewma3",        # 3-day exponentially weighted sentiment (momentum)
    "sent_surprise",     # today's sentiment vs its own 10-day mean (novelty)
    "news_vol_z",        # headline count vs its 20-day norm (attention spike)
    # --- price/technical controls -------------------------------------------
    "ret_1d", "ret_5d",
    "vol_5d",            # realised volatility, 5d
    "ma_gap",            # close / SMA10 - 1 : short-term stretch
    "ma_cross",          # SMA10 - SMA30, normalised: the trend baseline's signal
    "volume_z",
)

features = Table(
    "features", metadata,
    Column("ticker", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    *[Column(c, Float) for c in FEATURE_COLUMNS],
    Column("fwd_ret_1d", Float),      # next session's close-to-close return
    Column("target_next_up", Integer),  # 1 if fwd_ret_1d > 0 else 0  <- the label
)

# ---------------------------------------------------------------------------
# predictions: what the model said, and (filled in later) what actually happened.
# The monitoring layer joins these two halves.
# ---------------------------------------------------------------------------
predictions = Table(
    "predictions", metadata,
    Column("model", String(64), primary_key=True),
    Column("ticker", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("prob_up", Float, nullable=False),
    Column("pred_up", Integer, nullable=False),
    Column("realized_ret", Float),      # NULL until the outcome is known
    Column("realized_up", Integer),
    Column("created_at", DateTime),
    Index("ix_pred_model_date", "model", "date"),
)

# ---------------------------------------------------------------------------
# monitor_log: append-only audit trail of rolling accuracy + drift checks.
# ---------------------------------------------------------------------------
monitor_log = Table(
    "monitor_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model", String(64), nullable=False),
    Column("asof_date", Date, nullable=False),
    Column("window", Integer, nullable=False),
    Column("n_obs", Integer, nullable=False),
    Column("rolling_accuracy", Float),
    Column("baseline_accuracy", Float),   # what "always predict up" would score
    Column("psi_sent_mean", Float),       # feature-drift score vs training window
    Column("status", String(16)),         # OK | WARN | DRIFT
    Column("note", String),
    Column("created_at", DateTime),
    Index("ix_monitor_model_date", "model", "asof_date"),
)


# ---------------------------------------------------------------------------
# Engine / helpers
# ---------------------------------------------------------------------------
_engine: Engine | None = None


def get_engine() -> Engine:
    """Process-wide singleton engine.

    `future=True` opts into SQLAlchemy 2.0 semantics. For SQLite we also switch
    the journal to WAL so the FastAPI app can read while a pipeline job writes.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, future=True)
        if _engine.dialect.name == "sqlite":
            with _engine.begin() as c:
                c.execute(text("PRAGMA journal_mode=WAL"))
                c.execute(text("PRAGMA synchronous=NORMAL"))
    return _engine


def init_db(drop: bool = False) -> None:
    """Create every table. `checkfirst` makes this safe to call repeatedly."""
    eng = get_engine()
    if drop:
        metadata.drop_all(eng)
    metadata.create_all(eng, checkfirst=True)


def make_uid(ticker: str, published_at, title: str) -> str:
    """Content-addressed id for a headline.

    WHY hash instead of an autoincrement int: the same article is often
    re-published verbatim. Hashing (ticker, timestamp, title) means a re-ingest
    of overlapping data collapses onto the same row instead of duplicating it,
    and the sentiment cache stays valid across runs. sha1 truncated to 40 hex
    chars is plenty -- collision risk at 1e5 rows is negligible.
    """
    raw = f"{ticker}|{pd.Timestamp(published_at).isoformat()}|{title.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# SQLite compiles a multi-row INSERT into one statement with one host parameter
# per cell, and refuses more than SQLITE_MAX_VARIABLE_NUMBER (32,766) of them.
# So the safe batch size depends on the table's WIDTH, not on a fixed row count:
# 5,000 rows x 18 feature columns = 90,000 params and a hard failure. We derive
# the row count from the column count instead of hardcoding it.
MAX_SQL_PARAMS = 30_000


def upsert(table: Table, df: pd.DataFrame, chunk: int | None = None) -> int:
    """Insert-or-replace a DataFrame into `table`.

    WHY not df.to_sql(if_exists='append'): that raises on duplicate primary keys,
    so a partially-failed run could never be resumed. Here we build a dialect-
    native UPSERT so every stage of the pipeline is safely re-runnable.
    """
    if df.empty:
        return 0
    cols = [c.name for c in table.columns if c.name in df.columns]
    df = df[cols].where(pd.notnull(df[cols]), None)
    chunk = chunk or max(1, MAX_SQL_PARAMS // max(len(cols), 1))
    eng = get_engine()
    dialect = eng.dialect.name

    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _insert
    elif dialect.startswith("postgres"):
        from sqlalchemy.dialects.postgresql import insert as _insert
    else:                                    # pragma: no cover
        raise RuntimeError(f"upsert not implemented for dialect {dialect!r}")

    pk = [c.name for c in table.primary_key.columns]
    total = 0
    with eng.begin() as conn:
        for i in range(0, len(df), chunk):
            rows = df.iloc[i:i + chunk].to_dict("records")
            stmt = _insert(table).values(rows)
            update_cols = {c: stmt.excluded[c] for c in cols if c not in pk}
            if update_cols:
                stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update_cols)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=pk)
            conn.execute(stmt)
            total += len(rows)
    return total


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """Thin pandas wrapper so callers never touch connection lifecycles."""
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})
