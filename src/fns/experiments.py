"""
The leakage experiment -- the centrepiece result of this project.

Everything else in the repo measures a signal. This measures the MEASUREMENT:
it runs the identical pipeline twice, changing exactly one thing -- whether
day-D headlines are allowed to inform a position taken at close(D) -- and shows
how much apparent skill that one decision manufactures.

  extra_lag = 0 : the setup most tutorials publish. Headlines dated D are
                  treated as known at the close of session D. But the lead-lag
                  study shows those headlines correlate +0.23 with session D's
                  OWN return: many were written to REPORT that day's move. The
                  model gets a partial peek at the answer.

  extra_lag = 1 : headlines are withheld until the following session, by which
                  point every day-D article is unambiguously public.

The gap between the two accuracy numbers is not alpha. It is the price of a
one-line indexing mistake, and it is why the honest result in this project is
"no edge" rather than a headline number.
"""
from __future__ import annotations

import pandas as pd

from .features import ingest_features
from .ingest.headlines import ingest_headlines
from .sentiment.finbert import score_headlines
from .train import run as train_run


def leakage_comparison(lags: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    frames = []
    for lag in lags:
        print("\n" + "=" * 78)
        print(f"  REBUILDING PIPELINE WITH extra_lag_sessions = {lag}"
              f"{'   <- LEAKY (news dated D used at close(D))' if lag == 0 else '   <- leak-free'}")
        print("=" * 78)
        # Re-ingest re-attributes every headline to a different session. The
        # FinBERT cache is keyed by content hash, NOT by trade_date, so this
        # costs zero GPU time -- only the 23-second parquet read.
        ingest_headlines(extra_lag=lag)
        score_headlines()          # no-op: cache is warm
        ingest_features()
        res = train_run(persist=False)
        res.insert(0, "extra_lag", lag)
        frames.append(res)

    out = pd.concat(frames, ignore_index=True)

    print("\n" + "=" * 78)
    print("  LEAKY vs LEAK-FREE -- same code, same data, one indexing decision")
    print("=" * 78)
    piv = out.pivot_table(index="model", columns="extra_lag",
                          values=["accuracy", "roc_auc"])
    piv.columns = [f"{m}_lag{l}" for m, l in piv.columns]
    if {"accuracy_lag0", "accuracy_lag1"} <= set(piv.columns):
        piv["accuracy_INFLATION"] = piv["accuracy_lag0"] - piv["accuracy_lag1"]
        piv["auc_INFLATION"] = piv["roc_auc_lag0"] - piv["roc_auc_lag1"]
        piv = piv.sort_values("accuracy_INFLATION", ascending=False)
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))
    return out


def same_day_leak_demo() -> pd.DataFrame:
    """The catastrophic version of the mistake, quantified.

    `leakage_comparison()` above measures a SUBTLE leak: news dated D used at
    close(D), still predicting D+1. The bias there is real but small (~1pp),
    because the label still lies in the future.

    This function demonstrates the version that actually ruins projects:
    predicting session D's OWN return from news dated D. No future label is
    involved, so nothing looks wrong -- train/test are still split
    chronologically, no row is duplicated, sklearn raises no warning. It simply
    is not a prediction. The lead-lag study already told us the answer
    (rho = +0.23 at k=0); this converts that correlation into the accuracy
    figure such a project would proudly report.

    Only SENTIMENT features are used, so the result cannot be attributed to a
    price column trivially encoding the target.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    from .db import read_sql
    from .train import SENTIMENT_FEATURES

    # Rebuild attribution with lag 0 so a headline dated D sits on session D.
    ingest_headlines(extra_lag=0)
    ingest_features()

    df = read_sql("SELECT * FROM features")
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for label, target in [
        ("HONEST: predict NEXT session (t+1)", df["target_next_up"]),
        ("LEAKY: predict THIS session (t)", (df["ret_1d"] > 0).astype(int)),
    ]:
        d = df.assign(y=target)
        tr, te = d[d.date < "2023-01-01"], d[d.date >= "2023-01-01"]
        m = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.03, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=42)
        m.fit(tr[SENTIMENT_FEATURES], tr["y"])
        prob = m.predict_proba(te[SENTIMENT_FEATURES])[:, 1]
        rows.append({
            "setup": label,
            "accuracy": (prob >= 0.5).astype(int).__eq__(te["y"].values).mean(),
            "balanced_accuracy": balanced_accuracy_score(te["y"], (prob >= 0.5).astype(int)),
            "roc_auc": roc_auc_score(te["y"], prob),
            "n": len(te),
        })

    out = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  SAME-DAY LEAK DEMO -- sentiment features only, identical model")
    print("=" * 78)
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\n  The second row is what news-sentiment projects usually report.")
    print("  It is not forecasting -- it is reading the newspaper about today.")

    # Leave the database in the project's default (leak-free) state.
    ingest_headlines()
    ingest_features()
    return out


if __name__ == "__main__":
    leakage_comparison()
    same_day_leak_demo()
