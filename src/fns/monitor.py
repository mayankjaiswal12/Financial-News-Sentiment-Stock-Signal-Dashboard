"""
Stage 6 -- production monitoring.

A model that was 52% accurate in backtest is not a model that IS 52% accurate.
Markets change; the news mix changes; the vendor silently alters a feed. This
layer answers two different questions every day, because they fail differently:

  1. PERFORMANCE DECAY -- "are we still right as often as we were?"
     Measured by rolling directional accuracy against realised outcomes.
     This is the ground truth, but it is SLOW: with a ~50% base rate you need
     hundreds of observations before a real drop is statistically distinguishable
     from luck.

  2. FEATURE DRIFT -- "does the input still look like what we trained on?"
     Measured by Population Stability Index on the sentiment distribution.
     This is a leading indicator: it fires the day the input shifts, long before
     enough outcomes have accumulated to prove accuracy fell. It is also the
     only one of the two that works when labels are delayed or missing.

Watching only accuracy means finding out too late. Watching only drift means
crying wolf over harmless shifts. Production systems need both.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import SETTINGS
from .db import get_engine, monitor_log, read_sql, upsert


# ---------------------------------------------------------------------------
def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between a reference ('expected') and a recent ('actual') sample.

        PSI = sum_over_bins( (a_i - e_i) * ln(a_i / e_i) )

    Rule of thumb used across credit risk and ML monitoring:
        < 0.10  no meaningful shift
        0.10-0.25  moderate shift, investigate
        > 0.25  major shift, the model is being asked about a different world

    Implementation notes that matter:
      * Bin edges come from the EXPECTED (training) sample via quantiles, then
        are frozen and applied to the actual sample. Re-binning each sample
        separately would compare two different rulers and always return ~0.
      * `np.unique` on the edges guards against a degenerate feature where many
        quantiles collapse to the same value.
      * Empty bins are floored at a small epsilon: a zero in the denominator
        would send PSI to infinity on a single missing bin.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) < bins:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf     # catch out-of-range values

    e = np.histogram(expected, bins=edges)[0].astype(float)
    a = np.histogram(actual, bins=edges)[0].astype(float)
    e, a = e / e.sum(), a / a.sum()
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------------------
def backfill_outcomes() -> int:
    """Attach realised outcomes to stored predictions.

    In a live system predictions are written BEFORE the outcome exists, so this
    runs on a lag and fills them in. Here it reconciles `predictions` against
    the `features` table, which is also a genuine integrity check: if a
    prediction has no matching realised return, something upstream is broken.
    """
    sql = """
        UPDATE predictions
           SET realized_ret = (SELECT f.fwd_ret_1d FROM features f
                                WHERE f.ticker = predictions.ticker
                                  AND f.date   = predictions.date),
               realized_up  = (SELECT f.target_next_up FROM features f
                                WHERE f.ticker = predictions.ticker
                                  AND f.date   = predictions.date)
         WHERE realized_up IS NULL
    """
    from sqlalchemy import text
    with get_engine().begin() as conn:
        return conn.execute(text(sql)).rowcount


def rolling_report(model: str | None = None,
                   window: int | None = None) -> pd.DataFrame:
    """Daily portfolio-level accuracy plus its rolling average.

    We aggregate to one row per DATE first (mean hit rate across the 12 tickers),
    then roll over dates. Rolling over raw ticker-day rows would make the window
    length depend on how many names had news that day -- a "60-row" window could
    span three days or fifteen. Rolling over dates keeps the window a fixed
    amount of TIME, which is what "60-day accuracy" is supposed to mean.
    """
    window = window or SETTINGS.monitor_window
    q = """SELECT model, ticker, date, prob_up, pred_up, realized_up, realized_ret
           FROM predictions WHERE realized_up IS NOT NULL"""
    params: dict = {}
    if model:
        q += " AND model = :m"
        params["m"] = model
    df = read_sql(q, params)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["correct"] = (df["pred_up"] == df["realized_up"]).astype(float)
    # `always up` is the reference every accuracy number must be read against.
    df["baseline_correct"] = df["realized_up"].astype(float)

    daily = (df.groupby(["model", "date"], observed=True)
               .agg(n=("correct", "size"),
                    accuracy=("correct", "mean"),
                    baseline=("baseline_correct", "mean"),
                    mean_prob=("prob_up", "mean"))
               .reset_index()
               .sort_values(["model", "date"]))

    g = daily.groupby("model", observed=True, group_keys=False)
    daily["rolling_accuracy"] = g["accuracy"].transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean())
    daily["rolling_baseline"] = g["baseline"].transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean())
    daily["edge"] = daily["rolling_accuracy"] - daily["rolling_baseline"]
    return daily


def check_drift(model: str, feature: str = "sent_mean",
                window: int | None = None, persist: bool = True) -> dict:
    """Evaluate the newest window and raise OK / WARN / DRIFT.

    Two independent triggers, deliberately not combined into one score:
      * rolling accuracy below `accuracy_floor` -> the model has stopped working.
      * PSI above `psi_threshold`               -> the inputs have moved.
    Either one alone justifies a human look, and the note says which fired.
    """
    window = window or SETTINGS.monitor_window
    daily = rolling_report(model=model, window=window)
    if daily.empty:
        return {"status": "NO_DATA", "note": "no realised predictions yet"}

    latest = daily.iloc[-1]

    # PSI reference = the TRAINING period; actual = the most recent `window` days
    # of live scoring. That is the comparison that answers "is production input
    # still the input this model was fitted on?"
    feats = read_sql(f"SELECT date, {feature} FROM features")
    feats["date"] = pd.to_datetime(feats["date"])
    ref = feats[feats["date"] < SETTINGS.test_start][feature].to_numpy()
    recent_dates = daily["date"].tail(window)
    cur = feats[feats["date"].isin(recent_dates)][feature].to_numpy()
    psi = population_stability_index(ref, cur)

    acc = float(latest["rolling_accuracy"]) if pd.notna(latest["rolling_accuracy"]) else float("nan")
    base = float(latest["rolling_baseline"]) if pd.notna(latest["rolling_baseline"]) else float("nan")

    notes, status = [], "OK"
    if np.isfinite(acc) and acc < SETTINGS.accuracy_floor:
        status = "DRIFT"
        notes.append(f"rolling accuracy {acc:.3f} < floor {SETTINGS.accuracy_floor:.2f}")
    if np.isfinite(acc) and np.isfinite(base) and acc < base:
        # Beating a coin flip is not the bar; beating "always up" is.
        if status == "OK":
            status = "WARN"
        notes.append(f"underperforming always-up baseline ({acc:.3f} vs {base:.3f})")
    if np.isfinite(psi) and psi > SETTINGS.psi_threshold:
        status = "DRIFT"
        notes.append(f"PSI {psi:.3f} > {SETTINGS.psi_threshold:.2f} on {feature}")

    rec = {
        "model": model,
        "asof_date": latest["date"].date(),
        "window": int(window),
        "n_obs": int(daily["n"].tail(window).sum()),
        "rolling_accuracy": acc,
        "baseline_accuracy": base,
        "psi_sent_mean": psi,
        "status": status,
        "note": "; ".join(notes) or "within tolerance",
        "created_at": datetime.now(timezone.utc),
    }
    if persist:
        upsert(monitor_log, pd.DataFrame([rec]))
    return rec


def run(model: str | None = None) -> dict:
    n = backfill_outcomes()
    if n:
        print(f"[monitor] backfilled {n:,} realised outcomes")
    if model is None:
        models = read_sql("SELECT DISTINCT model FROM predictions")["model"].tolist()
    else:
        models = [model]
    out = {}
    for m in models:
        rec = check_drift(m)
        out[m] = rec
        print(f"[monitor] {m:28s} {rec['status']:5s} "
              f"acc={rec.get('rolling_accuracy', float('nan')):.4f} "
              f"base={rec.get('baseline_accuracy', float('nan')):.4f} "
              f"psi={rec.get('psi_sent_mean', float('nan')):.4f} | {rec['note']}")
    return out


if __name__ == "__main__":
    run()
