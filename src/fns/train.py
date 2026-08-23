"""
Stage 5 -- train, benchmark, and persist the classifier.

Design decisions that matter more than the model choice:

1. CHRONOLOGICAL SPLIT. Train on 2019-2022, test on 2023. A random split would
   let the model learn from 2023 to predict 2021. On a panel it is even worse:
   the same day appears for 12 tickers, so a random split puts NVDA's Tuesday in
   train and AAPL's Tuesday in test -- and market-wide moves make those two
   labels highly correlated. That leaks the market factor and can add 5-10
   accuracy points of pure fantasy.

2. TimeSeriesSplit FOR TUNING. Inside the training period we still need to pick
   a regularisation strength. Ordinary k-fold would reintroduce the same
   lookahead; TimeSeriesSplit only ever validates on data later than it trains on.

3. ABLATION BY FEATURE BLOCK. We fit the same model on sentiment-only,
   price-only, and combined feature sets. This is the only way to answer the
   question the project actually poses: does news sentiment add anything BEYOND
   what the price series already tells you? A single combined accuracy number
   cannot distinguish "sentiment works" from "momentum works and sentiment is
   along for the ride".

4. WE REPORT ECONOMICS, NOT JUST ACCURACY. Directional accuracy is a weak proxy:
   being right on twenty +0.1% days and wrong on one -5% day is a losing
   strategy with 95% accuracy. So we also compute the mean daily return and
   Sharpe ratio of acting on the signal.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import ALL_BASELINES
from .config import MODEL_DIR, SETTINGS
from .db import FEATURE_COLUMNS, predictions as pred_tbl, read_sql, upsert

# Feature blocks for the ablation. Splitting on the source of the information,
# not on statistical convenience.
SENTIMENT_FEATURES = [c for c in FEATURE_COLUMNS
                      if c.startswith(("sent_", "n_headlines", "news_vol"))]
PRICE_FEATURES = [c for c in FEATURE_COLUMNS if c not in SENTIMENT_FEATURES]
FEATURE_BLOCKS: dict[str, list[str]] = {
    "sentiment_only": SENTIMENT_FEATURES,
    "price_only": PRICE_FEATURES,
    "combined": list(FEATURE_COLUMNS),
}


# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    df = read_sql("SELECT * FROM features ORDER BY date, ticker")
    df["date"] = pd.to_datetime(df["date"])
    return df


def chronological_split(df: pd.DataFrame, test_start: str | None = None):
    """Split strictly in time. Nothing from the test period informs training."""
    cut = pd.Timestamp(test_start or SETTINGS.test_start)
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    if train.empty or test.empty:
        raise ValueError(f"bad split at {cut.date()}: train={len(train)} test={len(test)}")
    return train, test


# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_prob, fwd_ret) -> dict:
    """Statistical accuracy AND trading economics.

    strategy_ret: take a long position when we predict up and a short when we
    predict down, i.e. position = 2*pred - 1, and earn position * fwd_ret. This
    is gross of costs -- stated plainly rather than buried, because at daily
    turnover realistic costs (~1-2 bps a side) would consume most of the edges
    reported here.

    sharpe: annualised with sqrt(252), the standard scaling for daily data.
    """
    pos = 2 * np.asarray(y_pred) - 1
    r = pos * np.asarray(fwd_ret)
    sharpe = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # balanced accuracy averages the per-class recall, so a model that
        # simply always predicts the majority class scores 0.5 here no matter
        # how imbalanced the data is. It is the honest headline number.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "f1_up": float(f1_score(y_true, y_pred, zero_division=0)),
        "pred_up_rate": float(np.mean(y_pred)),
        "mean_ret_bps": float(1e4 * r.mean()),
        "sharpe": sharpe,
        "n": int(len(y_true)),
    }


def make_models(random_state: int) -> dict:
    """Two deliberately different learners.

    LogisticRegression: linear, heavily regularised, wrapped in a StandardScaler.
      Scaling is REQUIRED here -- L2 penalises large coefficients, and raw
      features range from ~1e-3 (returns) to ~1e2 (headline counts), so without
      scaling the penalty would fall almost entirely on the small-scale columns.

    HistGradientBoosting: non-linear, captures interactions (e.g. "negative
      sentiment matters only when volume is elevated"). It needs no scaling
      because trees split on order, not distance. Kept shallow with a strong
      learning-rate/depth cap: with a near-zero signal-to-noise ratio, a deep
      forest will memorise noise and test worse than the linear model.
    """
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, max_iter=2000, class_weight="balanced",
                random_state=random_state)),
        ]),
        "hgb": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.03,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=random_state),
    }


def tune_logreg_C(X: pd.DataFrame, y: np.ndarray, random_state: int) -> float:
    """Pick L2 strength with forward-chaining CV (never validate on the past)."""
    best_c, best_score = 1.0, -np.inf
    tscv = TimeSeriesSplit(n_splits=4)
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        scores = []
        for tr, va in tscv.split(X):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000,
                                           class_weight="balanced",
                                           random_state=random_state)),
            ]).fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            # AUC, not accuracy: it grades the RANKING and is insensitive to the
            # 0.5 threshold, which is what we want from a model selection metric.
            scores.append(roc_auc_score(y[va], p) if len(set(y[va])) > 1 else 0.5)
        m = float(np.mean(scores))
        if m > best_score:
            best_c, best_score = C, m
    return best_c


# ---------------------------------------------------------------------------
def run(test_start: str | None = None, persist: bool = True) -> pd.DataFrame:
    df = load_dataset()
    train, test = chronological_split(df, test_start)
    y_tr = train["target_next_up"].to_numpy()
    y_te = test["target_next_up"].to_numpy()
    fwd_te = test["fwd_ret_1d"].to_numpy()

    print(f"[train] train={len(train):,} ({train.date.min().date()} -> {train.date.max().date()})  "
          f"test={len(test):,} ({test.date.min().date()} -> {test.date.max().date()})")
    print(f"[train] up-rate train={y_tr.mean():.4f} test={y_te.mean():.4f}  "
          f"<- 'always up' scores exactly the test up-rate\n")

    rows, fitted = [], {}

    # ---- baselines ------------------------------------------------------
    for B in ALL_BASELINES:
        b = B().fit(train[list(FEATURE_COLUMNS)], y_tr)
        prob = b.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]
        pred = b.predict(test[list(FEATURE_COLUMNS)])
        rows.append({"model": b.name, "block": "-", **evaluate(y_te, pred, prob, fwd_te)})
        fitted[b.name] = (b, list(FEATURE_COLUMNS))

    # ---- learned models, one fit per feature block ----------------------
    for block, cols in FEATURE_BLOCKS.items():
        Xtr, Xte = train[cols], test[cols]
        best_c = tune_logreg_C(Xtr, y_tr, SETTINGS.random_state)
        models = make_models(SETTINGS.random_state)
        models["logreg"].set_params(clf__C=best_c)
        for mname, model in models.items():
            model.fit(Xtr, y_tr)
            prob = model.predict_proba(Xte)[:, 1]
            pred = (prob >= 0.5).astype(int)
            name = f"{mname}__{block}"
            rows.append({"model": name, "block": block,
                         **evaluate(y_te, pred, prob, fwd_te),
                         **({"C": best_c} if mname == "logreg" else {})})
            fitted[name] = (model, cols)

    res = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)

    # ---- persist --------------------------------------------------------
    if persist:
        import joblib
        champion = res.iloc[0]["model"]
        model, cols = fitted[champion]
        joblib.dump({"model": model, "features": cols, "name": champion},
                    MODEL_DIR / "champion.joblib")
        # Also keep the best COMBINED model for the dashboard, so the served
        # signal always uses the news features even if a price-only variant
        # happened to edge it out on this particular test window.
        comb = res[res.block == "combined"].iloc[0]["model"]
        cmodel, ccols = fitted[comb]
        joblib.dump({"model": cmodel, "features": ccols, "name": comb},
                    MODEL_DIR / "combined.joblib")
        (MODEL_DIR / "metrics.json").write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "test_start": str(test_start or SETTINGS.test_start),
            "extra_lag_sessions": SETTINGS.extra_lag_sessions,
            "champion": champion, "combined": comb,
            "results": res.to_dict("records"),
        }, indent=2, default=float))

        # Write test-period predictions for the monitoring layer to consume.
        for name in (champion, comb):
            m, c = fitted[name]
            prob = m.predict_proba(test[c])[:, 1]
            out = pd.DataFrame({
                "model": name, "ticker": test["ticker"].values,
                "date": test["date"].dt.date.values,
                "prob_up": prob, "pred_up": (prob >= 0.5).astype(int),
                "realized_ret": fwd_te, "realized_up": y_te,
                "created_at": datetime.now(timezone.utc),
            })
            upsert(pred_tbl, out)
        print(f"[train] champion='{champion}'  saved -> {MODEL_DIR}\n")

    return res


def print_results(res: pd.DataFrame) -> None:
    cols = ["model", "accuracy", "balanced_accuracy", "roc_auc",
            "pred_up_rate", "mean_ret_bps", "sharpe", "n"]
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    print_results(run())
