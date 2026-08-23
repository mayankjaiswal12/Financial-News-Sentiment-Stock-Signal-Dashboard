"""
Stage 9 -- cross-sectional ranking.

WHY THIS IS A DIFFERENT (AND BETTER-POSED) QUESTION
---------------------------------------------------
`train.py` asks "will AAPL go up tomorrow?". That question is dominated by the
market: on a day when the Nasdaq rallies 2%, essentially every name is up, so a
model can score 54.8% by learning "stocks go up" and nothing else. The signal we
care about is buried under a factor we are not trying to predict.

This module asks instead: "of these 12 names, which will BEAT THE OTHERS
tomorrow?" That reframing does three things:

1. It removes the market factor by construction. Every name on a given day
   experiences the same market return, so ranking within the day cancels it.
2. The base rate becomes 50% exactly -- half the names are above the median by
   definition. The "always predict up" trap disappears, so accuracy becomes an
   honest metric again.
3. It matches how the signal would actually be traded: a dollar-neutral
   long-short book, long the top names and short the bottom, rebalanced daily.

There is direct evidence this is the right frame. Cross-sectionally demeaned
features carry the only statistically significant signal found anywhere in this
project:

    volume_z    (demeaned)  IC = -0.0326  p = 0.001
    news_vol_z  (demeaned)  IC = -0.0232  p = 0.022
    sent_mean   (raw)       IC = -0.0013  p = 0.895   <- nothing

EVALUATION IS DIFFERENT TOO
---------------------------
Accuracy is a poor way to grade a ranking signal. The quant-standard metrics are
used instead:

  * Daily rank IC  -- Spearman correlation between predicted score and realised
                      return, computed WITHIN each day, then averaged over days.
  * IC t-statistic -- is the mean IC distinguishable from zero?
  * Information Ratio -- annualised return/risk of the long-short book.
  * Turnover and net-of-cost returns -- a daily-rebalanced signal that needs
                                        150% turnover a day is not tradeable
                                        however good its IC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SETTINGS
from .db import FEATURE_COLUMNS, read_sql

# Costs are charged per side, in basis points. 2 bps is a realistic all-in
# estimate (spread + impact) for mega-cap US equities at modest size.
COST_BPS_PER_SIDE = 2.0

# --------------------------------------------------------------------------
# WHY A COMPOSITE INSTEAD OF SIX SENTIMENT COLUMNS
# --------------------------------------------------------------------------
# sent_mean, sent_net, sent_surprise and sent_ewma3 correlate at 0.75-0.91 --
# they are four measurements of one underlying quantity. Handing all of them to
# a fitted model does not add information; it adds ways to split one weight four
# arbitrary ways. Measured consequence: the fitted logistic put -0.034 on
# sent_pos_frac and +0.037 on sent_neg_frac -- two coefficients that are both
# inverted AND cancel each other out.
#
# Averaging them into a single z-scored composite is justified a priori by that
# correlation matrix, NOT by any hold-out score, so it is not a form of
# test-set peeking.
SENT_COMPOSITE_PARTS = ("sent_mean", "sent_net", "sent_surprise", "sent_ewma3")

# --------------------------------------------------------------------------
# ECONOMIC SIGN PRIORS
# --------------------------------------------------------------------------
# The direction each feature *should* act in, stated before looking at results:
#   +1 positive tone should predict outperformance
#   -1 short-term reversal: yesterday's winners underperform today
#   -1 attention/volume spikes precede mean reversion
# The magnitudes are deliberately all 1: with an IC of ~0.015 the data cannot
# distinguish 0.8 from 1.2, so pretending to estimate a magnitude is false
# precision. This vector IS the prior that the shrinkage below pulls toward.
SIGN_PRIOR: dict[str, float] = {
    "sent_composite": +1.0,
    "news_vol_z": -1.0,
    "ret_1d": -1.0,
    "volume_z": -1.0,
}
MIN_NAMES_PER_DAY = 15     # below this a cross-section is too thin to rank.
                           # Scaled with the 43-name universe: a day where only a
                           # handful of names traded produces unstable z-scores.


# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    df = read_sql("SELECT * FROM features ORDER BY date, ticker")
    df["date"] = pd.to_datetime(df["date"])
    # Drop days with too few names: a "rank" among 2 stocks is noise, and the
    # z-scores below would be wildly unstable.
    counts = df.groupby("date")["ticker"].transform("size")
    return df[counts >= MIN_NAMES_PER_DAY].copy()


def cross_sectional_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace each feature by its z-score WITHIN its own date.

        x_demeaned(i, t) = (x(i, t) - mean_j x(j, t)) / std_j x(j, t)

    Two reasons this is the right transform, not merely a normalisation:

    * It strips the market factor. On a day when everything gaps up, every
      name's raw ret_1d is positive; after demeaning only the RELATIVE move
      survives, which is what a ranking model should see.
    * It removes slow drifts in the feature's own level. Headline volume grows
      ~6x from 2019 to 2023 in this corpus, so raw `n_headlines` means something
      different in 2019 than in 2023 -- a classic source of drift. A daily
      z-score is self-normalising and immune to that.

    No lookahead: the mean and std come from the SAME day's cross-section, which
    is fully known at that day's close.
    """
    out = df.copy()
    g = out.groupby("date", observed=True)
    for c in cols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std")
        # A day where every name shares a value gives sd = 0; the z-score is
        # then genuinely 0 (no cross-sectional information), not infinity.
        out[c] = ((out[c] - mu) / sd.replace(0, np.nan)).fillna(0.0)
    return out


def add_sentiment_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the correlated sentiment columns into one z-scored factor.

    Must run AFTER cross_sectional_zscore so the parts are already on a common
    scale -- averaging a raw score in [-1,1] with a count would be meaningless.
    """
    out = df.copy()
    parts = [c for c in SENT_COMPOSITE_PARTS if c in out.columns]
    out["sent_composite"] = out[parts].mean(axis=1)
    # Re-z-score within date so the composite has the same unit variance as the
    # other features and the prior weights below are comparable.
    g = out.groupby("date", observed=True)["sent_composite"]
    out["sent_composite"] = ((out["sent_composite"] - g.transform("mean"))
                             / g.transform("std").replace(0, np.nan)).fillna(0.0)
    return out


def prior_score(df: pd.DataFrame) -> np.ndarray:
    """Zero-parameter signal: the sign-weighted sum of the prior features.

    Nothing here is estimated, so there is nothing to overfit. This is the
    benchmark every fitted model must beat, and in practice it is hard to beat
    precisely because a fitted sign is unreliable when |IC| ~ 0.015.
    """
    return sum(w * df[c].to_numpy() for c, w in SIGN_PRIOR.items() if c in df.columns)


def add_relative_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label = did this name beat the cross-sectional median that day?

    Using the MEDIAN rather than the mean makes the label robust: one name
    printing +18% on earnings would drag a mean-based threshold up and mislabel
    several otherwise-normal names as underperformers.
    """
    out = df.copy()
    med = out.groupby("date", observed=True)["fwd_ret_1d"].transform("median")
    out["fwd_ret_rel"] = out["fwd_ret_1d"] - med          # continuous, for IC

    # Label = top half of the day by return. Getting the base rate to exactly
    # 0.50 -- the whole point of this framing -- needs care on ODD-sized days:
    # with 9 names there is a genuine middle observation that is neither in the
    # top nor the bottom half, and assigning it either way tilts the base rate
    # (a plain `> median` gave 0.463, a plain rank-split gave 0.537). We drop
    # that single middle name on odd days instead, which leaves an exact 50/50
    # split and costs ~1 row per odd day.
    g = out.groupby("date", observed=True)["fwd_ret_1d"]
    rank = g.rank(method="first")                 # 1..n, ties broken by order
    n = g.transform("size")
    is_middle = (n % 2 == 1) & (rank == (n + 1) / 2)
    out = out[~is_middle].copy()

    g = out.groupby("date", observed=True)["fwd_ret_1d"]
    out["target_outperform"] = (g.rank(method="first") > g.transform("size") / 2).astype(int)
    return out


# ---------------------------------------------------------------------------
def daily_rank_ic(df: pd.DataFrame, score_col: str,
                  ret_col: str = "fwd_ret_1d") -> pd.DataFrame:
    """Spearman IC computed WITHIN each day, then stacked into a time series.

    Pooling all ticker-days into one correlation (what analysis.py does) mixes
    cross-sectional signal with time-series drift. Computing it per day and
    averaging isolates the question this module asks: on a given day, does the
    score rank that day's names correctly?
    """
    rows = []
    for d, g in df.groupby("date", observed=True):
        if len(g) < MIN_NAMES_PER_DAY or g[ret_col].nunique() < 2:
            continue
        rho, _ = spearmanr(g[score_col], g[ret_col])
        if np.isfinite(rho):
            rows.append({"date": d, "ic": rho, "n": len(g)})
    return pd.DataFrame(rows)


def summarise_ic(ic: pd.DataFrame) -> dict:
    """Mean IC plus the t-stat that says whether it is real.

        t = mean(IC) / (std(IC) / sqrt(n_days))

    This is a one-sample t-test against zero. Rules of thumb: |t| > 2 is the
    usual bar for "not luck"; a mean IC of 0.02 with t = 0.4 is nothing.
    """
    if ic.empty:
        return {"mean_ic": np.nan, "ic_std": np.nan, "t_stat": np.nan,
                "hit_rate": np.nan, "n_days": 0}
    m, s, n = ic["ic"].mean(), ic["ic"].std(ddof=1), len(ic)
    return {
        "mean_ic": float(m),
        "ic_std": float(s),
        "t_stat": float(m / (s / np.sqrt(n))) if s > 0 else np.nan,
        # Share of days the IC was positive -- consistency, not magnitude.
        "hit_rate": float((ic["ic"] > 0).mean()),
        "n_days": int(n),
    }


# ---------------------------------------------------------------------------
def long_short_backtest(df: pd.DataFrame, score_col: str,
                        n_side: int | None = None,
                        cost_bps: float = COST_BPS_PER_SIDE) -> tuple[pd.DataFrame, dict]:
    """Dollar-neutral book: long the top-ranked names, short the bottom.

    Weights are equal within each leg and sum to +1 long / -1 short, so the book
    is market-neutral and its return is purely the spread between the legs.

    Turnover is |w_t - w_{t-1}| summed over names and halved (one "trade" moves
    weight out of one name and into another, and halving avoids double-counting).
    Cost is charged on that turnover at `cost_bps` per side. Reporting the net
    line matters: a daily-rebalanced 12-name book turns over a lot, and gross
    Sharpe figures that ignore this are meaningless.
    """
    d = df.sort_values(["date", score_col]).copy()
    rows, prev_w = [], {}

    for date, g in d.groupby("date", observed=True):
        k = n_side or max(2, len(g) // 4)          # 3 names a side for a 12-name panel
        if len(g) < 2 * k:
            continue
        g = g.sort_values(score_col)
        shorts, longs = g.head(k), g.tail(k)

        w = {t: -1.0 / k for t in shorts["ticker"]}
        w.update({t: 1.0 / k for t in longs["ticker"]})

        gross = (longs["fwd_ret_1d"].mean() - shorts["fwd_ret_1d"].mean())
        names = set(w) | set(prev_w)
        turnover = sum(abs(w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in names) / 2
        cost = turnover * (cost_bps / 1e4) * 2      # entry + exit
        rows.append({"date": date, "gross_ret": gross, "turnover": turnover,
                     "cost": cost, "net_ret": gross - cost})
        prev_w = w

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {}

    def _sharpe(x):
        return float(np.sqrt(252) * x.mean() / x.std()) if x.std() > 0 else 0.0

    stats = {
        "n_days": int(len(bt)),
        "gross_ret_bps_day": float(1e4 * bt["gross_ret"].mean()),
        "net_ret_bps_day": float(1e4 * bt["net_ret"].mean()),
        "gross_sharpe": _sharpe(bt["gross_ret"]),
        "net_sharpe": _sharpe(bt["net_ret"]),
        "avg_turnover": float(bt["turnover"].mean()),
        "cost_bps_day": float(1e4 * bt["cost"].mean()),
        "win_rate": float((bt["net_ret"] > 0).mean()),
    }
    return bt, stats


# ---------------------------------------------------------------------------
class ShrunkToPriorRanker:
    """Logistic weights shrunk toward the economic sign prior.

        w_final = lam * w_fitted_normalised + (1 - lam) * w_prior_normalised

    WHY THIS IS THE RIGHT SHAPE OF FIX
    ----------------------------------
    The diagnosis was not "the model is too weak" but "the model estimates a
    sign the data cannot determine". Measured: the fitted coefficient on
    sent_mean was -0.032, -0.027, +0.006, -0.009 in successive years -- it is
    a coin flip. Anything that keeps estimating that sign freely inherits the
    coin flip; heavier L2 only shrinks toward ZERO, not toward *correct*.

    Shrinking toward a prior is different from shrinking toward zero: at
    lam = 0 this is exactly `prior_score`, at lam = 1 it is the free fit, and
    in between the data may tilt the weights but cannot flip a sign unless it
    is strongly evidenced. `lam` is chosen on TRAINING data only.

    Both weight vectors are L2-normalised before blending so that `lam` means
    the same thing regardless of how large the raw logistic coefficients are.
    """

    def __init__(self, lam: float = 0.25, C: float = 0.1, random_state: int = 42):
        self.lam, self.C, self.random_state = lam, C, random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.cols_ = list(X.columns)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=self.C, max_iter=2000,
                                                    random_state=self.random_state))])
        pipe.fit(X, y)
        w_fit = pipe.named_steps["clf"].coef_[0]
        w_pri = np.array([SIGN_PRIOR.get(c, 0.0) for c in self.cols_])

        def _unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        self.w_ = self.lam * _unit(w_fit) + (1 - self.lam) * _unit(w_pri)
        self.scaler_ = pipe.named_steps["sc"]
        return self

    def score_rows(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(X[self.cols_]) @ self.w_


def select_lambda_walkforward(panel: pd.DataFrame, cols: list[str],
                              train_end: str, lams=(0.0, 0.25, 0.5, 0.75, 1.0)) -> float:
    """Choose the shrinkage weight using ONLY pre-test data.

    This is the part that stops variant selection from becoming the overfit.
    Eight candidate fixes were tried against the 2023 hold-out earlier and the
    apparent winner (a 3-feature model, IC +0.009 on 2023) reversed to -0.021
    under walk-forward. The lesson is that any choice made by looking at the
    test window is itself a fitted parameter. So `lam` is selected by expanding-
    window walk-forward INSIDE the training period, and the hold-out is touched
    exactly once, at the end.
    """
    tr = panel[panel["date"] < train_end]
    years = sorted(tr["date"].dt.year.unique())
    best, best_ic = lams[0], -np.inf
    for lam in lams:
        ics = []
        for yr in years[1:]:                       # first year is history only
            hist = tr[tr["date"].dt.year < yr]
            fold = tr[tr["date"].dt.year == yr]
            if len(hist) < 500 or fold.empty:
                continue
            m = ShrunkToPriorRanker(lam=lam).fit(hist[cols], hist["target_outperform"])
            f = fold.copy()
            f["score"] = m.score_rows(fold[cols])
            ics.append(summarise_ic(daily_rank_ic(f, "score"))["mean_ic"])
        mean_ic = float(np.nanmean(ics)) if ics else -np.inf
        if mean_ic > best_ic:
            best, best_ic = lam, mean_ic
    print(f"[xs] selected lambda={best} by walk-forward on TRAIN only "
          f"(mean in-train IC {best_ic:+.4f})")
    return best


def run(test_start: str | None = None, verbose: bool = True) -> dict:
    cols = list(FEATURE_COLUMNS)
    panel = add_relative_target(load_panel())
    panel = cross_sectional_zscore(panel, cols)
    panel = add_sentiment_composite(panel)      # collapses the collinear block

    # Feature set for the shrunk model: the composite plus the three
    # non-sentiment features that carry an economic prior. Deliberately small --
    # every column here has a sign we are willing to state in advance.
    prior_cols = [c for c in SIGN_PRIOR if c in panel.columns]

    cut = pd.Timestamp(test_start or SETTINGS.test_start)
    train, test = panel[panel["date"] < cut], panel[panel["date"] >= cut]

    y_tr = train["target_outperform"].to_numpy()
    y_te = test["target_outperform"].to_numpy()

    models = {
        "xs_logreg": Pipeline([("sc", StandardScaler()),
                               ("clf", LogisticRegression(C=0.1, max_iter=2000,
                                                          random_state=SETTINGS.random_state))]),
        "xs_hgb": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.03, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=SETTINGS.random_state),
    }

    results, out = [], {}

    def _record(name, test_df, score, full=False):
        t = test_df.copy(); t["score"] = score
        ic = daily_rank_ic(t, "score"); st = summarise_ic(ic)
        bt, bstats = long_short_backtest(t, "score")
        acc = float(((t["score"] >= np.median(t["score"])).astype(int)
                     == t["target_outperform"]).mean())
        results.append({"model": name, "accuracy": acc, **st, **bstats})
        out[name] = {"ic": ic, "backtest": bt}

    for name, model in models.items():
        model.fit(train[cols], y_tr)
        te = test.copy()
        te["score"] = model.predict_proba(test[cols])[:, 1]

        ic = daily_rank_ic(te, "score")
        s = summarise_ic(ic)
        bt, bstats = long_short_backtest(te, "score")
        acc = float(((te["score"] >= 0.5).astype(int) == y_te).mean())
        results.append({"model": name, "accuracy": acc, **s, **bstats})
        out[name] = {"ic": ic, "backtest": bt}

    # ---- the two fixes ---------------------------------------------------
    # 1. Zero-parameter sign-prior composite. Nothing estimated, nothing to
    #    overfit; this is the benchmark.
    _record("prior_composite", test, prior_score(test))

    # 2. Shrunk-to-prior, with lambda chosen on TRAIN ONLY.
    lam = select_lambda_walkforward(panel, prior_cols, cut)
    shrunk = ShrunkToPriorRanker(lam=lam).fit(train[prior_cols],
                                              train["target_outperform"].to_numpy())
    _record(f"shrunk_to_prior(lam={lam})", test, shrunk.score_rows(test[prior_cols]))

    # Both evaluated on the FULL sample too: the prior model fits nothing, so it
    # needs no hold-out, and 5x the days makes the t-statistic meaningful.
    _record("prior_composite_FULLSAMPLE", panel, prior_score(panel))

    # Reference: rank by raw demeaned sentiment alone -- no learning at all.
    te = test.copy()
    te["score"] = te["sent_mean"]
    ic = daily_rank_ic(te, "score")
    bt, bstats = long_short_backtest(te, "score")
    results.append({"model": "rank_by_sentiment", "accuracy": float(
        ((te["sent_mean"] >= 0).astype(int) == y_te).mean()), **summarise_ic(ic), **bstats})
    out["rank_by_sentiment"] = {"ic": ic, "backtest": bt}

    # This rule FITS NOTHING, so it cannot overfit and needs no hold-out. That
    # means it can legitimately be evaluated on the whole 2019-2023 sample --
    # ~5x the days, and the t-statistic scales with sqrt(n_days), so this is a
    # far more powerful test of whether the effect is real than 243 days can be.
    full = panel.copy()
    full["score"] = full["sent_mean"]
    ic_full = daily_rank_ic(full, "score")
    bt_full, bstats_full = long_short_backtest(full, "score")
    results.append({"model": "rank_by_sentiment_FULLSAMPLE",
                    "accuracy": float(((full["sent_mean"] >= 0).astype(int)
                                       == full["target_outperform"]).mean()),
                    **summarise_ic(ic_full), **bstats_full})
    out["rank_by_sentiment_full"] = {"ic": ic_full, "backtest": bt_full}

    res = pd.DataFrame(results)
    out["results"] = res

    if verbose:
        print(f"\n[xs] panel={len(panel):,} ticker-days  train={len(train):,} test={len(test):,}")
        print(f"[xs] base rate (outperform) = {y_te.mean():.4f}  "
              f"<- ~0.50 by construction, so accuracy is honest here\n")
        print("=== CROSS-SECTIONAL RANKING -- hold-out ===")
        print(res[["model", "accuracy", "mean_ic", "t_stat", "hit_rate", "n_days"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        print("\n=== LONG-SHORT BOOK (top-3 vs bottom-3, daily rebalance) ===")
        print(res[["model", "gross_ret_bps_day", "net_ret_bps_day", "gross_sharpe",
                   "net_sharpe", "avg_turnover", "cost_bps_day", "win_rate"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        print("\n  |t| > 2 on mean IC is the usual bar for 'not luck'.")
    return out


if __name__ == "__main__":
    run()
