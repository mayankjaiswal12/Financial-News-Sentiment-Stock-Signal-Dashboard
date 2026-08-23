"""
Stage 4 -- does sentiment actually predict anything?

This module exists to answer that question BEFORE any model is trained. If the
raw signal has no edge, no amount of gradient boosting will manufacture one, and
knowing that early is worth more than a high-accuracy number you cannot trust.

Three diagnostics:
  1. lead_lag_study()      -- WHEN does the market react? Detects lookahead bias.
  2. information_coefficient() -- HOW strong is the edge, in units quants use.
  3. sentiment_buckets()   -- is the relationship MONOTONE, or driven by tails?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import SETTINGS
from .db import read_sql


# ---------------------------------------------------------------------------
# 1. Lead-lag: the lookahead-bias detector
# ---------------------------------------------------------------------------
def lead_lag_study(min_headlines: int = 3, max_lag: int = 3) -> pd.DataFrame:
    """Correlate daily sentiment dated D against session returns at D+k.

    WHY THIS IS THE MOST IMPORTANT DIAGNOSTIC IN THE PROJECT:
    a genuinely predictive signal peaks at k >= 1 (the future). A signal that
    peaks at k = 0 is not predicting the move -- it is DESCRIBING one that has
    already happened ("Nvidia soars 12% on blowout guidance" is written after the
    12%). Regressing that against the same day's return yields a beautiful,
    entirely fake result. Running this before modelling is how you catch it.

    We use Spearman (rank) rather than Pearson correlation because daily equity
    returns are fat-tailed: a handful of +15% earnings days would otherwise
    dominate a Pearson estimate and make the correlation a statement about four
    outliers rather than about the other 8,000 observations.
    """
    px = read_sql("SELECT ticker, date, adj_close FROM prices ORDER BY ticker, date")
    px["date"] = pd.to_datetime(px["date"])
    px["ret"] = px.groupby("ticker", observed=True)["adj_close"].pct_change()

    sen = read_sql(
        """SELECT h.ticker, h.published_at, s.score
           FROM headlines h JOIN sentiment s
             ON s.headline_uid = h.headline_uid AND s.model = :m""",
        {"m": SETTINGS.hf_model},
    )
    # Deliberately use the RAW publication date here, not the lagged trade_date.
    # The whole point is to measure the alignment that the lag is derived from;
    # using trade_date would bake in the answer we are trying to test.
    sen["news_date"] = pd.to_datetime(sen["published_at"]).dt.normalize()

    daily = (sen.groupby(["ticker", "news_date"], observed=True)["score"]
                .agg(["mean", "count"]).reset_index())
    daily.columns = ["ticker", "news_date", "sent", "n"]
    daily = daily[daily["n"] >= min_headlines]

    sessions = np.sort(px["date"].unique())
    rows = []
    for k in range(-max_lag, max_lag + 1):
        parts = []
        for tkr, g in daily.groupby("ticker", observed=True):
            ret = px[px["ticker"] == tkr].set_index("date")["ret"]
            # s0 = index of the first session on/after the news date; +k steps
            # along the trading calendar (so we skip weekends correctly).
            s0 = np.searchsorted(sessions, g["news_date"].values, side="left") + k
            ok = (s0 >= 0) & (s0 < len(sessions))
            parts.append(pd.DataFrame({
                "sent": g["sent"].values[ok],
                "ret": pd.Series(sessions[s0[ok]]).map(ret).values,
            }))
        d = pd.concat(parts).dropna()
        rho, pval = spearmanr(d["sent"], d["ret"])
        rows.append({"lag_k": k, "spearman": rho, "p_value": pval, "n": len(d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Information Coefficient
# ---------------------------------------------------------------------------
def information_coefficient(feature: str = "sent_mean") -> pd.DataFrame:
    """Rank-IC of a feature against the NEXT-day return, overall and per year.

    IC is the standard quant measure of raw signal strength. Calibration:
      |IC| < 0.01  noise
      0.01 - 0.03  weak but potentially tradeable at scale / low cost
      0.03 - 0.05  good for a daily equity signal
      > 0.10       be suspicious: usually lookahead bias or a data error

    The per-year breakdown matters more than the headline number. A signal that
    is +0.06 in 2020 and -0.01 everywhere else is not a signal; it is the
    pandemic. Stability across regimes is what makes an edge real.
    """
    df = read_sql(f"SELECT ticker, date, {feature}, fwd_ret_1d FROM features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[feature, "fwd_ret_1d"])

    out = []
    rho, p = spearmanr(df[feature], df["fwd_ret_1d"])
    out.append({"period": "ALL", "ic": rho, "p_value": p, "n": len(df)})
    for yr, g in df.groupby(df["date"].dt.year):
        rho, p = spearmanr(g[feature], g["fwd_ret_1d"])
        out.append({"period": str(yr), "ic": rho, "p_value": p, "n": len(g)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 3. Bucket / monotonicity check
# ---------------------------------------------------------------------------
def sentiment_buckets(feature: str = "sent_mean", q: int = 5) -> pd.DataFrame:
    """Split days into sentiment quantiles and report the forward outcome.

    A usable signal is MONOTONE: the most-negative bucket should have the lowest
    next-day return and hit rate, rising steadily to the most-positive bucket.
    A single strong bucket at one extreme with noise elsewhere usually means the
    edge lives entirely in rare events and will not survive transaction costs.

    `qcut` splits by rank (equal-population buckets) rather than by value, so
    each bucket carries the same statistical weight.
    """
    df = read_sql(f"SELECT {feature}, fwd_ret_1d, target_next_up FROM features")
    df = df.dropna(subset=[feature, "fwd_ret_1d"])
    df["bucket"] = pd.qcut(df[feature].rank(method="first"), q,
                           labels=[f"Q{i+1}" for i in range(q)])
    out = df.groupby("bucket", observed=True).agg(
        n=("fwd_ret_1d", "size"),
        mean_fwd_ret_bps=("fwd_ret_1d", lambda s: 1e4 * s.mean()),
        hit_rate=("target_next_up", "mean"),
        mean_sentiment=(feature, "mean"),
    ).reset_index()
    return out


def run_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    res = {
        "lead_lag": lead_lag_study(),
        "ic": information_coefficient(),
        "buckets": sentiment_buckets(),
    }
    if verbose:
        print("\n=== 1. LEAD-LAG (sentiment dated D vs return at session D+k) ===")
        print(res["lead_lag"].to_string(index=False,
              formatters={"spearman": "{:+.4f}".format, "p_value": "{:.2e}".format}))
        peak = int(res["lead_lag"].loc[res["lead_lag"].spearman.idxmax(), "lag_k"])
        print(f"  -> peak correlation at k={peak:+d}. "
              f"{'CONTEMPORANEOUS: news reacts to the move; a same-day label would leak.' if peak == 0 else 'predictive.'}")

        print("\n=== 2. INFORMATION COEFFICIENT (sent_mean vs NEXT-day return) ===")
        print(res["ic"].to_string(index=False,
              formatters={"ic": "{:+.4f}".format, "p_value": "{:.3f}".format}))

        print("\n=== 3. SENTIMENT QUANTILES vs forward return ===")
        print(res["buckets"].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    return res


if __name__ == "__main__":
    run_all()
