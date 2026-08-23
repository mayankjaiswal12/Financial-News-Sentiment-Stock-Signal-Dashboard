# Financial News Sentiment & Stock Signal Dashboard

A leakage-controlled pipeline that scores real financial headlines with a
Hugging Face transformer (FinBERT), joins them to daily prices in SQL, trains a
scikit-learn classifier for next-day direction, benchmarks it against honest
baselines, serves it from FastAPI, and monitors it for drift.

**Python · Transformers · scikit-learn · FastAPI · SQL · Pandas/NumPy · Matplotlib/Seaborn**

---

## The headline result

> **Sentiment dated day *D* correlates +0.234 with day *D*'s own return, and
> +0.018 with day *D+1*'s.**
>
> Financial headlines mostly **report** moves rather than predict them. Once
> that is corrected for, **no model beats "always predict up."**

That is the finding, and the project is built to prove it rather than to hide it.
An earlier, sloppier version of this same code reports **58.9% accuracy** —
entirely from letting a day's headlines describe that day's return.

| Setup (identical model & features) | Accuracy | ROC AUC |
|---|---|---|
| Predict **this** session from today's news *(the common mistake)* | **0.5892** | **0.6063** |
| Predict **next** session from today's news *(honest)* | 0.5286 | 0.5022 |

![lead-lag](reports/fig_lead_lag.png)

---

## Hold-out results (2023, 2,599 ticker-days)

The bar every model must clear is **0.5483** — the rate at which these names
simply went up in 2023.

| Model | Accuracy | Balanced acc. | ROC AUC | Sharpe |
|---|---|---|---|---|
| `baseline_always_up` | **0.5483** | 0.5000 | 0.5000 | 1.84 |
| `hgb__sentiment_only` | 0.5452 | 0.5039 | 0.4890 | 1.66 |
| `hgb__price_only` | 0.5271 | 0.5010 | 0.5000 | 0.71 |
| `hgb__combined` | 0.5267 | 0.5024 | 0.4989 | 0.69 |
| `baseline_sentiment_sign` | 0.5087 | 0.4918 | 0.4852 | 0.76 |
| `baseline_ma_cross` | 0.4783 | 0.4622 | 0.4513 | −0.27 |

Every balanced accuracy sits within noise of 0.50 and every AUC within noise of
0.50. **Daily FinBERT headline sentiment carries no exploitable next-day
directional edge on mega-cap US tech.** The `always_up` Sharpe of 1.84 is not
skill — it is 2023's tech rally.

Where signal *does* faintly appear (rank IC vs next-day return):

| Feature | IC | p |
|---|---|---|
| `ret_1d` | −0.0380 | 0.000 |
| `volume_z` (cross-sectionally demeaned) | −0.0326 | 0.001 |
| `news_vol_z` (cross-sectionally demeaned) | −0.0232 | 0.022 |
| `sent_mean` | −0.0013 | 0.895 |

i.e. short-term **reversal** and **attention spikes**, not tone.

---

## Data

| | |
|---|---|
| **Headlines** | 80,343 real Nasdaq.com articles, 2019‑01‑02 → 2023‑12‑29, from [FNSPID](https://huggingface.co/datasets/benstaf/FNSPID-filtered-nasdaq-100) |
| **Prices** | 15,900 daily bars from Yahoo Finance (Stooq fallback), split/dividend adjusted |
| **Universe** | AAPL MSFT AMZN GOOG NVDA TSLA AMD INTC MU QCOM NFLX COST |
| **Panel** | 9,781 ticker-days after feature warm-up and filtering |
| **Sentiment** | `ProsusAI/finbert`, 757 headlines/sec on Apple MPS (80k in 106 s) |

The universe was chosen by **measured headline coverage**, not brand
recognition: META has zero rows in this corpus and GOOGL only ~1.7k (coverage
sits on the GOOG line), so including them would have injected thousands of empty
ticker-days.

---

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python -m fns.cli all      # ingest → score → features → analyse → train → monitor → plots
python -m fns.cli serve    # dashboard at http://127.0.0.1:8000
```

First run downloads ~1 GB (headline corpus + FinBERT weights) and takes roughly
10 minutes; everything is cached, so later runs take seconds. No API keys.

Individual stages (each idempotent and separately re-runnable):

```bash
python -m fns.cli ingest      # prices + headlines  (prices FIRST: they define the trading calendar)
python -m fns.cli score       # FinBERT; skips anything already scored
python -m fns.cli features    # build the modelling table
python -m fns.cli analyse     # lead-lag, information coefficient, quantile buckets
python -m fns.cli train       # baselines + logreg/HGB, ablated by feature block
python -m fns.cli monitor     # rolling accuracy + PSI drift
python -m fns.cli experiment  # the leaky-vs-honest comparison
pytest -q                     # 13 tests
```

---

## How correctness is enforced

**1. Point-in-time attribution.** `ingest/headlines.py:assign_trade_date()`
converts each UTC timestamp to US/Eastern, rolls anything at/after the 16:00
bell to the next session, snaps to the **real** trading calendar (read back out
of the `prices` table, so holidays are exact), then applies one extra session of
lag. That last step is not paranoia — it is what the lead-lag study demands.

**2. Chronological splits everywhere.** Train 2019–2022, test 2023. Tuning uses
`TimeSeriesSplit`. A random split on a 12-name panel would put NVDA's Tuesday in
train and AAPL's Tuesday in test, leaking the market factor.

**3. Baselines reported beside every number.** `always_up`, MA-crossover, and
raw sentiment sign.

**4. Ablation by feature block.** Sentiment-only vs price-only vs combined — the
only way to tell "sentiment works" from "momentum works and sentiment came along".

**5. Economics, not just accuracy.** Mean return in bps and Sharpe, gross of
costs (stated, not hidden — daily turnover at ~1–2 bps/side would consume these).

**6. Tests on the invariants that fail silently.** Weekend/holiday rollover,
`fwd_ret_1d` equalling the true next-session return, upsert idempotency, PSI
behaviour, and a guard on FinBERT's non-alphabetical label order.

---

## Architecture

```
FNSPID parquet ─┐
                ├─► headlines ──► FinBERT ──► sentiment ─┐
Yahoo/Stooq ────┴─► prices ──────────────────────────────┼─► features ─► train ─► predictions
                                                          │                          │
                                                          └──────────► monitor ◄──────┘
                                                                          │
                                                              FastAPI + dashboard
```

```
src/fns/
├── config.py            one source of truth for universe, dates, cutoff, lag
├── db.py                SQLAlchemy Core schema + dialect-native UPSERT
├── ingest/
│   ├── prices.py        yfinance + Stooq fallback, adjusted closes
│   └── headlines.py     corpus load, cleaning, SESSION ATTRIBUTION
├── sentiment/finbert.py batched GPU inference, SQL-cached by content hash
├── features.py          sentiment aggregates + price controls + label
├── analysis.py          lead-lag, information coefficient, quantile buckets
├── baselines.py         always-up, MA crossover, sentiment sign
├── train.py             chronological split, ablation, benchmarking
├── experiments.py       the leaky-vs-honest comparison
├── monitor.py           rolling accuracy + PSI drift alerting
├── plots.py             Matplotlib/Seaborn figures
├── api/main.py          FastAPI + Jinja dashboard
└── cli.py               `python -m fns.cli <stage>`
```

Six SQL tables: `headlines`, `sentiment`, `prices`, `features`, `predictions`,
`monitor_log`. SQLite by default; set `FNS_DB_URL` to point the same code at
Postgres.

---

## Monitoring

Two signals, because they fail differently:

* **Rolling directional accuracy** — ground truth, but slow to become
  significant at a ~50% base rate.
* **Population Stability Index** on the sentiment distribution — a leading
  indicator that fires the day the *input* shifts, before enough outcomes exist
  to prove accuracy fell.

Current state on 2023 data: **DRIFT** — PSI 0.325 (the 2023 news distribution
genuinely differs from 2019–2022) and rolling accuracy 0.540 below the
always-up baseline of 0.587. The monitor is correctly refusing to certify a
model that does not work.

![monitoring](reports/fig_monitoring.png)

---

## Known limitations

* **One corpus, date-only timestamps.** FNSPID stamps every article at 00:00
  UTC, so intraday precision is impossible; the extra session of lag is the
  conservative response. Real intraday timestamps would let the cutoff do the
  work and would likely recover some decayed signal.
* **12 mega-caps.** The most efficiently-priced, most-covered names in the
  market — the hardest place to find news alpha. Small caps would be a fairer
  test of the hypothesis.
* **Survivorship.** The universe is chosen with hindsight from today's index.
* **Gross of costs.** No spread, slippage or borrow.
* **FinBERT scores tone, not surprise.** "Beats by $0.02" and "beats by $2.00"
  read alike; the market only cares about the gap to expectations.

## Next steps

Cross-sectional ranking instead of absolute direction; intraday timestamps and
minute-bar reaction windows; event-type classification (earnings/guidance/legal)
so tone is conditioned on what kind of news it is; analyst-estimate data to turn
tone into surprise.

## Licence & attribution

Code MIT. Headlines: FNSPID (CC-BY-4.0). Prices: Yahoo Finance / Stooq, personal
research use. Model: `ProsusAI/finbert`.
