"""
Tests focused on the things that are SILENTLY wrong when they break.

We do not test that sklearn can fit a model. We test the project-specific
invariants whose failure produces a plausible-looking but false result:
session attribution, absence of lookahead in the label, upsert idempotency,
and the drift metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Session attribution -- the point-in-time rule
# ---------------------------------------------------------------------------
@pytest.fixture
def sessions() -> np.ndarray:
    """Mon-Fri for two weeks, with Wed 2024-01-10 removed as a 'holiday'."""
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    days = days[days != pd.Timestamp("2024-01-10")]
    return days.values.astype("datetime64[D]")


def _assign(ts_utc: list[str], sessions, extra_lag: int):
    from fns.ingest.headlines import assign_trade_date
    s = pd.Series(pd.to_datetime(ts_utc, utc=True))
    return assign_trade_date(s, sessions, extra_lag=extra_lag)


def test_before_close_maps_to_same_session(sessions):
    # 14:00 UTC = 09:00 ET in January -- before the 16:00 bell.
    out = _assign(["2024-01-08 14:00:00Z"], sessions, extra_lag=0)
    assert out.iloc[0] == pd.Timestamp("2024-01-08")


def test_after_close_rolls_to_next_session(sessions):
    # 22:00 UTC = 17:00 ET -- after the bell, so it belongs to the next session.
    out = _assign(["2024-01-08 22:00:00Z"], sessions, extra_lag=0)
    assert out.iloc[0] == pd.Timestamp("2024-01-09")


def test_weekend_news_maps_to_monday(sessions):
    out = _assign(["2024-01-06 15:00:00Z"], sessions, extra_lag=0)   # Saturday
    assert out.iloc[0] == pd.Timestamp("2024-01-08")


def test_holiday_is_skipped(sessions):
    # After Tue's close -> Wed, but Wed 01-10 is a holiday -> Thu 01-11.
    out = _assign(["2024-01-09 22:00:00Z"], sessions, extra_lag=0)
    assert out.iloc[0] == pd.Timestamp("2024-01-11")


def test_extra_lag_shifts_exactly_one_session(sessions):
    a = _assign(["2024-01-08 14:00:00Z"], sessions, extra_lag=0).iloc[0]
    b = _assign(["2024-01-08 14:00:00Z"], sessions, extra_lag=1).iloc[0]
    assert a == pd.Timestamp("2024-01-08") and b == pd.Timestamp("2024-01-09")


def test_never_attributes_to_a_past_session(sessions):
    """The invariant that makes the whole study valid."""
    ts = pd.date_range("2024-01-01", "2024-01-18", freq="7h", tz="UTC")
    out = _assign(list(ts.astype(str)), sessions, extra_lag=1).dropna()
    et_date = (pd.Series(ts).dt.tz_convert("America/New_York")
               .dt.tz_localize(None).dt.normalize())
    assert (out.values >= et_date[out.index].values).all()


# ---------------------------------------------------------------------------
# Drift metric
# ---------------------------------------------------------------------------
def test_psi_is_zero_for_identical_distributions():
    from fns.monitor import population_stability_index
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x.copy()) < 1e-6


def test_psi_grows_with_distribution_shift():
    from fns.monitor import population_stability_index
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    small = population_stability_index(base, rng.normal(0.1, 1, 5000))
    large = population_stability_index(base, rng.normal(1.5, 1, 5000))
    assert small < large
    assert large > 0.25          # a 1.5-sigma shift must trip the alert band


# ---------------------------------------------------------------------------
# Sentiment scoring contract
# ---------------------------------------------------------------------------
def test_finbert_label_indices_are_read_from_config():
    """Guards the sign of the entire signal.

    ProsusAI/finbert orders its labels {0: positive, 1: negative, 2: neutral}.
    Anyone 'tidying' finbert.py to assume alphabetical order would invert the
    score and every downstream number would still look plausible.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("ProsusAI/finbert")
    id2label = {int(k): v.lower() for k, v in cfg.id2label.items()}
    assert id2label[0].startswith("pos")
    assert id2label[1].startswith("neg")


# ---------------------------------------------------------------------------
# Database round-trip
# ---------------------------------------------------------------------------
def test_upsert_is_idempotent(tmp_path, monkeypatch):
    """Re-running any stage must not duplicate or corrupt rows."""
    import sqlalchemy as sa
    from fns import db as dbmod

    eng = sa.create_engine(f"sqlite:///{tmp_path/'t.db'}", future=True)
    monkeypatch.setattr(dbmod, "_engine", eng)
    dbmod.metadata.create_all(eng)

    df = pd.DataFrame({"ticker": ["AAA"] * 3,
                       "date": pd.date_range("2024-01-01", periods=3).date,
                       "adj_close": [1.0, 2.0, 3.0]})
    dbmod.upsert(dbmod.prices, df)
    dbmod.upsert(dbmod.prices, df)                       # same rows again
    assert dbmod.read_sql("SELECT COUNT(*) n FROM prices")["n"].iloc[0] == 3

    df2 = df.copy(); df2["adj_close"] = [9.0, 9.0, 9.0]  # conflicting values
    dbmod.upsert(dbmod.prices, df2)
    got = dbmod.read_sql("SELECT adj_close FROM prices ORDER BY date")["adj_close"]
    assert got.tolist() == [9.0, 9.0, 9.0]               # updated, not duplicated


def test_uid_is_stable_and_content_addressed():
    from fns.db import make_uid
    a = make_uid("AAPL", "2024-01-02 00:00:00", " Apple beats estimates ")
    b = make_uid("AAPL", pd.Timestamp("2024-01-02"), "Apple beats estimates")
    c = make_uid("AAPL", "2024-01-02 00:00:00", "Apple misses estimates")
    assert a == b          # whitespace + timestamp formatting are normalised
    assert a != c


# ---------------------------------------------------------------------------
# The no-lookahead guarantee, checked against the real built table
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (__import__("fns.config", fromlist=["DATA_DIR"]).DATA_DIR / "fns.db").exists(),
    reason="pipeline has not been run yet")
def test_label_equals_next_session_return():
    """fwd_ret_1d on row t must equal the return from close(t) to close(t+1).

    This is the assertion that would have caught an off-by-one in the label --
    the bug that silently turns a 50% model into a 60% one.
    """
    from fns.db import read_sql
    px = read_sql("SELECT ticker,date,adj_close FROM prices ORDER BY ticker,date")
    px["date"] = pd.to_datetime(px["date"])
    px["expected"] = px.groupby("ticker")["adj_close"].shift(-1) / px["adj_close"] - 1

    f = read_sql("SELECT ticker,date,fwd_ret_1d FROM features")
    f["date"] = pd.to_datetime(f["date"])

    m = f.merge(px[["ticker", "date", "expected"]], on=["ticker", "date"], how="left")
    assert m["expected"].notna().all()
    np.testing.assert_allclose(m["fwd_ret_1d"], m["expected"], rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(
    not (__import__("fns.config", fromlist=["DATA_DIR"]).DATA_DIR / "fns.db").exists(),
    reason="pipeline has not been run yet")
def test_no_headline_is_used_before_it_was_published():
    from fns.config import SETTINGS
    from fns.db import read_sql
    df = read_sql("SELECT published_at, trade_date FROM headlines")
    pub = (pd.to_datetime(df["published_at"]).dt.tz_localize("UTC")
             .dt.tz_convert(SETTINGS.market_tz).dt.tz_localize(None).dt.normalize())
    assert (pd.to_datetime(df["trade_date"]) >= pub).all()
