"""
Stage 3 -- build the modelling table.

Joins daily sentiment aggregates onto the price panel, derives price/technical
controls, and attaches the forward-looking label.

THE ONE RULE THIS FILE ENFORCES
-------------------------------
Every feature on row (ticker, t) must be computable from information available
at the CLOSE of session t. Only `fwd_ret_1d` and `target_next_up` may look ahead.

pandas makes this easy to get right and easy to get wrong:
  * `.rolling(w)` and `.ewm()` are BACKWARD-looking -- window ends at the current
    row. Safe. Including the current row is fine: close(t) is known at close(t).
  * `.shift(-1)` looks FORWARD. It appears exactly once in this file, to build
    the label.
  * `.pct_change()` is backward. Safe.
If you ever add a `.shift(-n)` or a `center=True` rolling window to a feature
column, you have built a time machine and your backtest is fiction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SETTINGS
from .db import FEATURE_COLUMNS, features as features_tbl, read_sql, upsert


def load_daily_sentiment() -> pd.DataFrame:
    """Collapse headline-level scores into one row per (ticker, trade_date).

    The aggregation happens in SQL, not pandas: the database is already indexed
    on (ticker, trade_date), so it groups 80k rows into ~15k without ever
    materialising the headline text in Python memory.

    Why these five statistics:
      * n_headlines  -- news volume is an attention/uncertainty proxy in its own
                        right, independent of tone.
      * sent_mean    -- the central signal, confidence-weighted (see finbert.py).
      * sent_std     -- disagreement. Ten headlines split 5 bullish / 5 bearish
                        mean something very different from ten neutral ones, yet
                        both average to ~0. Only the dispersion separates them.
      * pos/neg_frac -- a rank-based view of the same day that is robust to a few
                        extreme scores dragging the mean around.
    """
    return read_sql(
        """
        SELECT h.ticker,
               h.trade_date                                  AS date,
               COUNT(*)                                      AS n_headlines,
               AVG(s.score)                                  AS sent_mean,
               AVG(s.score * s.score)                        AS sent_sq_mean,
               AVG(CASE WHEN s.label = 'pos' THEN 1.0 ELSE 0.0 END) AS sent_pos_frac,
               AVG(CASE WHEN s.label = 'neg' THEN 1.0 ELSE 0.0 END) AS sent_neg_frac
        FROM headlines h
        JOIN sentiment s
          ON s.headline_uid = h.headline_uid AND s.model = :model
        GROUP BY h.ticker, h.trade_date
        """,
        {"model": SETTINGS.hf_model},
    )


def load_prices() -> pd.DataFrame:
    return read_sql("SELECT ticker, date, adj_close, volume FROM prices ORDER BY ticker, date")


def build_features() -> pd.DataFrame:
    px = load_prices()
    sen = load_daily_sentiment()
    px["date"] = pd.to_datetime(px["date"])
    sen["date"] = pd.to_datetime(sen["date"])

    # SQL has no population-variance-from-moments helper we can trust across
    # dialects, so derive the standard deviation from E[x^2] - E[x]^2 here.
    # clip(lower=0) guards the tiny negative values floating-point error can
    # produce when every score in a group is identical.
    sen["sent_std"] = np.sqrt((sen["sent_sq_mean"] - sen["sent_mean"] ** 2).clip(lower=0))
    sen = sen.drop(columns=["sent_sq_mean"])
    sen["sent_net"] = sen["sent_pos_frac"] - sen["sent_neg_frac"]

    # LEFT join on the price panel: the panel defines the row universe, so every
    # trading session exists for every ticker even when no news was published.
    # An inner join here would silently delete quiet days and break the rolling
    # windows below, which assume evenly spaced sessions.
    df = px.merge(sen, on=["ticker", "date"], how="left").sort_values(["ticker", "date"])

    # A session with no news genuinely has zero headlines...
    df["n_headlines"] = df["n_headlines"].fillna(0.0)
    # ...but it does NOT have "neutral sentiment". Filling sent_mean with 0 would
    # tell the model a quiet day is a confidently-neutral day. We leave it NaN
    # and drop those rows at the end (see min_headlines_per_day).

    g = df.groupby("ticker", observed=True, group_keys=False)

    # ---------------- sentiment dynamics ---------------------------------
    # A single day's tone is noisy. These two features ask the more useful
    # questions: which way has tone been trending, and is today unusual?
    #
    # ignore_na=True makes the EWMA skip quiet days rather than treating them as
    # gaps, so the average reflects the last 3 days that actually had news.
    df["sent_ewma3"] = g["sent_mean"].transform(
        lambda s: s.ewm(span=3, ignore_na=True, min_periods=1).mean()
    )
    # "Surprise" = today's tone minus its own recent norm. A stock that is always
    # written about positively carries a permanent positive mean; only the
    # DEVIATION from that baseline is news. This de-means each name adaptively.
    df["sent_surprise"] = df["sent_mean"] - g["sent_mean"].transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    # Attention spike: today's headline count vs its own 20-day norm. +1e-9
    # avoids a divide-by-zero for names with a dead-quiet stretch.
    df["news_vol_z"] = g["n_headlines"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean())
        / (s.rolling(20, min_periods=5).std() + 1e-9)
    )

    # ---------------- price / technical controls --------------------------
    # These matter for an honest evaluation. Without them we could not tell
    # whether "sentiment predicts returns" or whether sentiment is just a proxy
    # for momentum. Including them forces the sentiment block to earn its keep.
    df["ret_1d"] = g["adj_close"].transform(lambda s: s.pct_change())
    df["ret_5d"] = g["adj_close"].transform(lambda s: s.pct_change(5))
    df["vol_5d"] = g["ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).std())
    sma10 = g["adj_close"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    sma30 = g["adj_close"].transform(lambda s: s.rolling(30, min_periods=10).mean())
    df["ma_gap"] = df["adj_close"] / sma10 - 1.0
    # This is the moving-average crossover baseline expressed as a feature:
    # positive => fast above slow => classic trend-following "long" signal.
    df["ma_cross"] = (sma10 - sma30) / sma30
    df["volume_z"] = g["volume"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean())
        / (s.rolling(20, min_periods=5).std() + 1e-9)
    )

    # ---------------- label (the ONLY forward-looking line) ---------------
    # shift(-1) pulls TOMORROW's close onto today's row: the return we are
    # trying to predict, from close(t) to close(t+1).
    df["fwd_ret_1d"] = g["adj_close"].transform(lambda s: s.shift(-1) / s - 1.0)
    df["target_next_up"] = (df["fwd_ret_1d"] > 0).astype("float")
    df.loc[df["fwd_ret_1d"].isna(), "target_next_up"] = np.nan

    # ---------------- trim ------------------------------------------------
    # Drop the warm-up buffer we deliberately downloaded for the rolling windows,
    # and the final session (no label exists for it yet).
    df = df[(df["date"] >= SETTINGS.start_date) & (df["date"] <= SETTINGS.end_date)]
    df = df[df["n_headlines"] >= SETTINGS.min_headlines_per_day]
    df = df.dropna(subset=list(FEATURE_COLUMNS) + ["target_next_up"])

    df["target_next_up"] = df["target_next_up"].astype(int)
    df["date"] = df["date"].dt.date
    return df[["ticker", "date", *FEATURE_COLUMNS, "fwd_ret_1d", "target_next_up"]]


def ingest_features() -> pd.DataFrame:
    df = build_features()
    # Rebuild from scratch each time: features are a pure function of the raw
    # tables, so stale rows from an earlier feature definition are never wanted.
    from .db import get_engine
    with get_engine().begin() as conn:
        conn.execute(features_tbl.delete())
    upsert(features_tbl, df)
    print(f"[features] {len(df):,} ticker-days | {df.ticker.nunique()} tickers | "
          f"{df.date.min()} -> {df.date.max()} | up-rate={df.target_next_up.mean():.3f}")
    return df


if __name__ == "__main__":
    ingest_features()
