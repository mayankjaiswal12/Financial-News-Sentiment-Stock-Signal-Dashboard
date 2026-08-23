"""
Stage 7 -- figures for the dashboard and the write-up.

Charting rules applied throughout (each one is a real readability decision, not
decoration):

* NO DUAL-AXIS CHARTS. The obvious way to draw "sentiment vs price" is one panel
  with price on the left axis and sentiment on the right. Don't: the crossing
  point of the two lines is then an artefact of two arbitrary scales, and you can
  make the series appear to lead or lag each other just by rescaling. We use two
  stacked panels sharing one x-axis, so comparisons are made date-by-date and no
  false correlation is implied.
* Price is INDEXED to 100 at the sample start, so a $600 name and a $40 name are
  visually comparable on one axis.
* Colours come from a colourblind-validated categorical set (blue/orange checked
  for deuteranopia/tritanopia separation), not matplotlib's default cycle.
* Grid and spines are recessive; the data is the darkest thing on the page.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # headless backend: no display needed on a server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import REPORT_DIR, SETTINGS
from .db import read_sql

# Validated categorical slots (light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL, GOOD = "#e34948", "#008300"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd6"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "figure.dpi": 110,
    })


def _save(fig, name: str) -> str:
    path = REPORT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path}")
    return str(path)


# ---------------------------------------------------------------------------
def plot_lead_lag(df: pd.DataFrame | None = None) -> str:
    """The leakage diagnostic, drawn so the k=0 spike is unmissable."""
    from .analysis import lead_lag_study
    df = lead_lag_study() if df is None else df
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    # Colour by MEANING, not by rank: the contemporaneous bar is the failure
    # state we are warning about, so it takes the reserved 'critical' colour.
    colors = [CRITICAL if k == 0 else BLUE for k in df["lag_k"]]
    ax.bar(df["lag_k"], df["spearman"], color=colors, width=0.62, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    for _, r in df.iterrows():
        ax.annotate(f"{r.spearman:+.3f}", (r.lag_k, r.spearman),
                    ha="center", va="bottom" if r.spearman >= 0 else "top",
                    fontsize=9, color=INK,
                    xytext=(0, 3 if r.spearman >= 0 else -12),
                    textcoords="offset points")
    ax.set_xlabel("sessions between news date and the return being measured  (k)")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("News sentiment reacts to the move — it does not predict it")
    ax.annotate("k=0: same-session return.\nUsing this as a label is lookahead bias.",
                xy=(0, df.loc[df.lag_k == 0, "spearman"].iloc[0]), xytext=(0.9, 0.19),
                fontsize=9, color=CRITICAL,
                arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.2))
    ax.set_xticks(df["lag_k"])
    # Headroom so the value labels on the negative bars clear the tick labels.
    lo, hi = df["spearman"].min(), df["spearman"].max()
    ax.set_ylim(min(lo - 0.045, -0.03), hi + 0.045)
    return _save(fig, "fig_lead_lag.png")


def plot_sentiment_vs_price(ticker: str = "NVDA", smooth: int = 10) -> str:
    """Two stacked panels, one shared x-axis. Never a second y-axis.

    Price is read from `prices` (every session) while sentiment comes from
    `features` (only sessions that had news), then REINDEXED onto the price
    calendar. That reindex matters: the FNSPID corpus has real per-ticker
    coverage gaps -- NVDA has no headlines at all between 2020-07 and 2021-07.
    Plotting only the rows that exist would draw a straight line across those
    thirteen months and imply data we do not have. Reindexing leaves NaN, and
    matplotlib breaks the line there, so the gap is visible instead of invented.
    """
    px = read_sql("SELECT date, adj_close FROM prices WHERE ticker = :t ORDER BY date",
                  {"t": ticker})
    fe = read_sql("SELECT date, sent_mean FROM features WHERE ticker = :t ORDER BY date",
                  {"t": ticker})
    if px.empty:
        raise ValueError(f"no prices for {ticker}")
    px["date"] = pd.to_datetime(px["date"])
    px = px[(px["date"] >= SETTINGS.start_date) & (px["date"] <= SETTINGS.end_date)]
    fe["date"] = pd.to_datetime(fe["date"])

    # Index to 100 so the panel is about SHAPE, not the dollar price.
    px["price_idx"] = 100 * px["adj_close"] / px["adj_close"].iloc[0]
    s = (fe.set_index("date")["sent_mean"]
           .reindex(px["date"])                     # <- gaps become NaN
           .rolling(smooth, min_periods=3).mean())

    _style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    ax1.plot(px["date"], px["price_idx"], color=BLUE, lw=1.6)
    ax1.set_ylabel("price (indexed, start = 100)")
    ax1.set_title(f"{ticker} - price and news sentiment on a shared timeline")

    ax2.axhline(0, color=MUTED, lw=1)
    ax2.plot(px["date"].values, s.values, color=ORANGE, lw=1.6)
    ax2.fill_between(px["date"].values, 0, s.values,
                     where=s.values >= 0, color=ORANGE, alpha=0.18)
    ax2.fill_between(px["date"].values, 0, s.values,
                     where=s.values < 0, color=ORANGE, alpha=0.08)
    ax2.set_ylabel(f"FinBERT score\n({smooth}-day mean)")

    # Shade the no-coverage stretches so the break reads as "no data",
    # not as "sentiment was zero".
    missing = s.isna().values
    if missing.any():
        d = px["date"].values
        edges = np.flatnonzero(np.diff(missing.astype(int)) != 0) + 1
        for a, b in zip(np.r_[0, edges], np.r_[edges, len(missing)]):
            if missing[a] and (b - a) > 10:
                for ax in (ax1, ax2):
                    ax.axvspan(d[a], d[b - 1], color=MUTED, alpha=0.09, lw=0)
        ax2.annotate("no headline coverage", xy=(0.5, 0.06), xycoords="axes fraction",
                     ha="center", fontsize=9, color=MUTED)
    return _save(fig, f"fig_sentiment_vs_price_{ticker}.png")


def plot_coverage() -> str:
    """Headlines per ticker-month -- makes the corpus gaps explicit.

    A sequential single-hue ramp (light = few, dark = many) because the value is
    a magnitude, not a category. Zero-coverage cells are left white so they read
    as absent rather than as a low count.
    """
    h = read_sql("SELECT ticker, trade_date FROM headlines")
    h["trade_date"] = pd.to_datetime(h["trade_date"])
    h["month"] = h["trade_date"].dt.to_period("M").dt.to_timestamp()
    piv = (h.pivot_table(index="ticker", columns="month",
                         values="trade_date", aggfunc="count")
             .fillna(0).astype(int).sort_index())

    _style()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    sns.heatmap(piv.replace(0, np.nan), cmap="Blues", ax=ax,
                cbar_kws={"label": "headlines / month"},
                linewidths=0.4, linecolor="white")
    ax.set_xticks(np.arange(0, piv.shape[1], 6) + 0.5)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in piv.columns[::6]], rotation=0)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title("Headline coverage is uneven - white cells are months with NO news")
    return _save(fig, "fig_coverage.png")


def plot_model_comparison() -> str:
    """Every model against the only bar that matters: 'always up'."""
    import json
    from .config import MODEL_DIR
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    # Sort by the metric actually drawn, so the bar order reads monotonically.
    res = pd.DataFrame(metrics["results"]).sort_values("accuracy")

    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    base = res.loc[res.model == "baseline_always_up", "accuracy"].iloc[0]
    colors = [ORANGE if m.startswith("baseline") else BLUE for m in res["model"]]
    ax.barh(res["model"], res["accuracy"], color=colors, height=0.62, zorder=3)
    ax.axvline(base, color=CRITICAL, lw=1.6, ls="--", zorder=4,
               label=f"always-up baseline = {base:.3f}")
    for y, (_, r) in enumerate(res.iterrows()):
        ax.annotate(f"{r.accuracy:.3f}", (r.accuracy, y), va="center",
                    xytext=(4, 0), textcoords="offset points", fontsize=9)
    ax.set_xlim(0.40, max(0.60, res["accuracy"].max() + 0.03))
    ax.set_xlabel("directional accuracy on the 2023 hold-out")
    ax.set_title("No model beats 'always predict up' — the honest result")
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, "fig_model_comparison.png")


def plot_monitoring(model: str | None = None) -> str:
    """Rolling accuracy vs its baseline, with the drift floor drawn in."""
    from .monitor import rolling_report
    daily = rolling_report(model=model)
    if daily.empty:
        raise ValueError("no realised predictions -- run training + monitor first")
    model = model or daily["model"].iloc[0]
    d = daily[daily["model"] == model].dropna(subset=["rolling_accuracy"])

    _style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(d["date"], d["rolling_accuracy"], color=BLUE, lw=1.8, label="model")
    ax.plot(d["date"], d["rolling_baseline"], color=ORANGE, lw=1.8,
            ls="-", label="always-up baseline")
    ax.axhline(SETTINGS.accuracy_floor, color=CRITICAL, lw=1.2, ls="--",
               label=f"drift floor = {SETTINGS.accuracy_floor:.2f}")
    # Shade only where the model is actually losing to the baseline -- the
    # condition the monitor alerts on, made visible.
    ax.fill_between(d["date"], d["rolling_accuracy"], d["rolling_baseline"],
                    where=d["rolling_accuracy"] < d["rolling_baseline"],
                    color=CRITICAL, alpha=0.10, interpolate=True)
    ax.set_ylabel(f"{SETTINGS.monitor_window}-day rolling accuracy")
    ax.set_title(f"Monitoring — {model}")
    ax.legend(loc="best", frameon=False, ncols=3)
    return _save(fig, "fig_monitoring.png")


def plot_sentiment_buckets() -> str:
    from .analysis import sentiment_buckets
    b = sentiment_buckets()
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(b["bucket"].astype(str), b["mean_fwd_ret_bps"], color=BLUE,
           width=0.62, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    for i, r in b.iterrows():
        ax.annotate(f"{r.mean_fwd_ret_bps:.1f}", (i, r.mean_fwd_ret_bps),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xlabel("sentiment quantile  (Q1 = most negative, Q5 = most positive)")
    ax.set_ylabel("mean next-day return (bps)")
    ax.set_title("Not monotone — the relationship is noise, not signal")
    return _save(fig, "fig_sentiment_buckets.png")


def generate_all(ticker: str = "NVDA") -> list[str]:
    out = [
        plot_lead_lag(),
        plot_coverage(),
        plot_sentiment_vs_price(ticker),
        plot_model_comparison(),
        plot_sentiment_buckets(),
    ]
    try:
        out.append(plot_monitoring())
    except Exception as exc:                                  # pragma: no cover
        print(f"[plots] skipped monitoring figure: {exc}")
    return out


if __name__ == "__main__":
    generate_all()
