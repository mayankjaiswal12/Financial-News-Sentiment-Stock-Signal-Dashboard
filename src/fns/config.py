"""
Central configuration.

WHY a config module instead of scattering constants:
every stage of the pipeline (ingest -> score -> features -> train -> serve) has to
agree on the SAME universe, date range and session cutoff. If the feature builder
thought the cutoff was 16:00 ET and the API thought it was 09:30 ET, the model
would be served features it was never trained on. One import, one source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Everything is derived from the repo root so the code works no matter
# what directory you launch it from (uvicorn, pytest and __main__ all differ).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]      # .../FNS
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                       # cached downloads (parquet, csv)
REPORT_DIR = ROOT / "reports"                    # generated PNG figures
MODEL_DIR = ROOT / "models"                      # pickled sklearn pipelines

for _d in (DATA_DIR, RAW_DIR, REPORT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# SQLite by default (zero-setup, single file, perfectly adequate for ~1e5 rows).
# Override with FNS_DB_URL=postgresql+psycopg://... to run the exact same code
# against Postgres -- SQLAlchemy Core makes that a config change, not a rewrite.
DB_URL = os.getenv("FNS_DB_URL", f"sqlite:///{DATA_DIR / 'fns.db'}")


@dataclass(frozen=True)
class Settings:
    # ---- universe -------------------------------------------------------
    # Universe selected by MEASURED corpus coverage, not brand recognition.
    # Criteria (see analysis in docs/CONCEPTS.md): a ticker is included if, over
    # the 2019-2023 study window, it has headlines in >=80% of months and >=1,200
    # headlines total. That yields 41 usable names and ~120k headlines.
    # (EA and WBA meet the coverage bar but were taken private in 2025, so no
    #  price vendor serves them any more -- a reverse-survivorship limitation
    #  worth naming: we cannot study the names that left the index.)
    #
    # Two things this fixes at once:
    #  * BREADTH. The Fundamental Law of Active Management says information ratio
    #    scales as IC * sqrt(N). Going from 12 to 43 names is a ~1.9x multiplier
    #    on the same per-name signal -- and breadth is what makes a weak
    #    cross-sectional effect estimable at all.
    #  * COVERAGE GAPS. The previous 12-name list was chosen for liquidity and
    #    contained large holes (NVDA had 13 consecutive zero-headline months;
    #    AAPL/MSFT/AMZN had almost no pre-2022 coverage). Only 6 of those 12
    #    clear the coverage bar. Dropping famous tickers that the corpus barely
    #    covers is a data-quality decision, not a liquidity one.
    tickers: tuple[str, ...] = (
        "ADBE", "ADI", "ADSK", "ALGN", "AMAT", "AMD", "AMGN", "ASML", "BIIB",
        "BKNG", "BKR", "CDNS", "CHTR", "CMCSA", "COST", "CRWD", "CSX", "DLTR",
        "DXCM", "EBAY", "ENPH", "FANG", "FTNT", "GILD", "GOOG", "INTC",
        "KHC", "MRVL", "MU", "PANW", "PDD", "PEP", "PYPL", "QCOM", "SBUX",
        "TMUS", "TXN", "VRTX", "WDAY", "ZM", "ZS",
    )
    start_date: str = "2019-01-01"
    end_date: str = "2023-12-31"     # FNSPID headline coverage ends here

    # ---- point-in-time discipline --------------------------------------
    # A headline published at 21:00 UTC on Monday lands AFTER the 16:00 ET
    # (=21:00 UTC in winter) close, so it cannot inform Monday's return -- it is
    # attributed to Tuesday's session. `session_cutoff_et` is the wall-clock
    # time in US/Eastern that separates "today's tradeable news" from
    # "tomorrow's". Everything after it rolls to the next trading day.
    session_cutoff_et: str = "16:00"
    market_tz: str = "America/New_York"

    # Extra whole-session lag applied AFTER the cutoff rule. Default 1. Why:
    # the FNSPID corpus stamps every article at 00:00 UTC, i.e. it records a
    # publication DATE, not a time. So we cannot tell whether an article dated D
    # appeared at 09:00 ET (tradeable into D's close) or 18:00 ET (not).
    # `analysis.lead_lag_study()` settles it empirically: sentiment dated D
    # correlates +0.23 with session D's OWN return and only +0.02 with D+1's.
    # Day-D headlines are therefore mostly REACTIONS to day D's move. Treating
    # them as known at close(D) would leak the answer into the features.
    # With extra_lag_sessions=1 the news is only used from the following session
    # onward, by which point every day-D article is certainly public.
    # Set to 0 to reproduce the (leaky) setup most tutorials publish -- the CLI
    # exposes it precisely so the two can be compared side by side.
    extra_lag_sessions: int = 1

    # ---- model ----------------------------------------------------------
    hf_model: str = "ProsusAI/finbert"   # BERT fine-tuned on financial phrasebank
    max_seq_len: int = 64                # headlines are short; 64 tokens covers >99%
    batch_size: int = 64

    # ---- training --------------------------------------------------------
    # Chronological split. NEVER random-split a time series: a random split lets
    # the model see 2023 while predicting 2021, which inflates accuracy and is
    # the single most common bug in retail "stock prediction" projects.
    test_start: str = "2023-01-01"
    min_headlines_per_day: int = 1       # drop ticker-days with no news at all

    # ---- monitoring -------------------------------------------------------
    # Rolling window used by the drift monitor, and the accuracy floor below
    # which we raise a DRIFT flag.
    monitor_window: int = 60             # trading days
    accuracy_floor: float = 0.50         # coin-flip; alert if we fall under it
    psi_threshold: float = 0.20          # population-stability index alert level

    random_state: int = 42

    # ---- headline corpus ---------------------------------------------------
    hf_dataset_repo: str = "benstaf/FNSPID-filtered-nasdaq-100"
    hf_dataset_files: tuple[str, ...] = (
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
    )


SETTINGS = Settings()
