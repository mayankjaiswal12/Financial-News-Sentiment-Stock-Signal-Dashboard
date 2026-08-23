"""
Stage 8 -- FastAPI service + dashboard.

Serves three things:
  * JSON endpoints for the signal, the evaluation table and the monitor
  * an on-demand /predict that runs the saved sklearn pipeline
  * a server-rendered HTML dashboard at /

DESIGN DECISIONS

1. The model is loaded ONCE at startup via the lifespan hook, not per request.
   Unpickling a sklearn pipeline takes tens of milliseconds -- irrelevant once,
   fatal at 100 req/s. The lifespan context is the modern replacement for the
   deprecated @app.on_event("startup").

2. Pydantic response models on every endpoint. They are not ceremony: FastAPI
   uses them to generate the OpenAPI schema at /docs, and they fail loudly if the
   pipeline ever starts returning a NaN where a float is promised.

3. Endpoints read from SQL rather than recomputing. The dashboard must stay
   responsive while a training job is running; WAL mode (see db.py) lets readers
   proceed during writes.

4. /predict accepts RAW FEATURES rather than a ticker+date. That keeps the
   service honest: it cannot accidentally serve a prediction built from data
   that would not have existed at decision time. Point-in-time correctness is
   the caller's contract, and the schema makes it explicit.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import MODEL_DIR, REPORT_DIR, SETTINGS
from ..db import read_sql

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model + metrics once, at process start."""
    import joblib
    bundle_path = MODEL_DIR / "combined.joblib"
    if bundle_path.exists():
        STATE["bundle"] = joblib.load(bundle_path)
        print(f"[api] loaded model '{STATE['bundle']['name']}' "
              f"({len(STATE['bundle']['features'])} features)")
    else:
        STATE["bundle"] = None
        print("[api] WARNING no trained model found -- run `fns train` first")

    mpath = MODEL_DIR / "metrics.json"
    STATE["metrics"] = json.loads(mpath.read_text()) if mpath.exists() else {}
    yield
    STATE.clear()


app = FastAPI(
    title="Financial News Sentiment & Stock Signal API",
    version="0.1.0",
    description="FinBERT headline sentiment -> next-day direction signal, "
                "with leakage-controlled evaluation and drift monitoring.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Health(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    tickers: list[str]
    n_headlines: int
    n_features: int


class PredictRequest(BaseModel):
    """Feature values for one ticker-day. Any omitted feature defaults to 0.0
    (the post-standardisation mean), which is the neutral choice."""
    features: dict[str, float] = Field(
        ..., description="Feature name -> value; see /features/schema",
        json_schema_extra={"example": {"sent_mean": 0.21, "n_headlines": 8,
                                       "ret_1d": -0.012, "ma_cross": 0.03}},
    )


class PredictResponse(BaseModel):
    model_name: str
    prob_up: float
    pred_up: int
    used_features: list[str]
    missing_filled_with_zero: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=Health, tags=["meta"])
def health() -> Health:
    counts = read_sql(
        "SELECT (SELECT COUNT(*) FROM headlines) h, (SELECT COUNT(*) FROM features) f")
    b = STATE.get("bundle")
    return Health(
        status="ok", model_loaded=b is not None,
        model_name=b["name"] if b else None,
        tickers=list(SETTINGS.tickers),
        n_headlines=int(counts["h"].iloc[0]), n_features=int(counts["f"].iloc[0]),
    )


@app.get("/features/schema", tags=["meta"])
def feature_schema() -> dict:
    b = STATE.get("bundle")
    if not b:
        raise HTTPException(503, "no model loaded")
    return {"model": b["name"], "features": b["features"]}


@app.get("/signal", tags=["signal"])
def signal(ticker: str | None = None, limit: int = Query(200, le=5000)) -> list[dict]:
    """Most recent sentiment + price + realised outcome rows."""
    q = """SELECT f.ticker, f.date, f.n_headlines, f.sent_mean, f.sent_net,
                  f.ret_1d, f.fwd_ret_1d, f.target_next_up, p.adj_close
           FROM features f JOIN prices p
             ON p.ticker=f.ticker AND p.date=f.date"""
    params: dict = {}
    if ticker:
        q += " WHERE f.ticker = :t"
        params["t"] = ticker.upper()
    q += f" ORDER BY f.date DESC LIMIT {int(limit)}"
    df = read_sql(q, params)
    if df.empty:
        raise HTTPException(404, f"no rows for ticker={ticker}")
    return df.to_dict("records")


@app.get("/predictions", tags=["signal"])
def predictions(model: str | None = None, limit: int = Query(200, le=5000)) -> list[dict]:
    q = "SELECT * FROM predictions"
    params: dict = {}
    if model:
        q += " WHERE model = :m"
        params["m"] = model
    q += f" ORDER BY date DESC LIMIT {int(limit)}"
    return read_sql(q, params).to_dict("records")


@app.get("/metrics", tags=["eval"])
def metrics() -> dict:
    if not STATE.get("metrics"):
        raise HTTPException(503, "no metrics.json -- run training first")
    return STATE["metrics"]


@app.get("/monitor", tags=["monitor"])
def monitor(model: str | None = None) -> list[dict]:
    """Latest drift check per model, recomputed live."""
    from ..monitor import run as monitor_run
    return list(monitor_run(model).values())


@app.get("/monitor/history", tags=["monitor"])
def monitor_history(model: str | None = None, window: int | None = None) -> list[dict]:
    from ..monitor import rolling_report
    d = rolling_report(model=model, window=window)
    if d.empty:
        return []
    d = d.copy()
    d["date"] = d["date"].astype(str)
    # JSON has no NaN literal; None serialises correctly as null.
    return d.replace({np.nan: None}).to_dict("records")


@app.post("/predict", response_model=PredictResponse, tags=["signal"])
def predict(req: PredictRequest) -> PredictResponse:
    b = STATE.get("bundle")
    if not b:
        raise HTTPException(503, "no model loaded -- run `fns train`")
    cols = b["features"]
    missing = [c for c in cols if c not in req.features]
    # Build the row in the model's EXACT column order. sklearn pipelines are
    # positional under the hood, so a dict that happens to iterate in a
    # different order would silently feed volume_z into the ret_1d slot.
    row = pd.DataFrame([[float(req.features.get(c, 0.0)) for c in cols]], columns=cols)
    prob = float(b["model"].predict_proba(row)[0, 1])
    return PredictResponse(
        model_name=b["name"], prob_up=prob, pred_up=int(prob >= 0.5),
        used_features=cols, missing_filled_with_zero=missing,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.get("/figures/{name}", tags=["dashboard"])
def figure(name: str):
    """Serve a generated PNG.

    `Path(name).name` strips any directory component, so a request for
    `../../../etc/passwd` collapses to `passwd` and cannot escape REPORT_DIR.
    Never interpolate a user-supplied string into a filesystem path unguarded.
    """
    from fastapi.responses import FileResponse
    safe = Path(name).name
    path = REPORT_DIR / safe
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(404, f"no figure {safe!r}")
    return FileResponse(path, media_type="image/png")


@app.get("/", response_class=HTMLResponse, tags=["dashboard"])
def dashboard() -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html"]),   # escape by default: XSS guard
    )
    m = STATE.get("metrics") or {}
    results = pd.DataFrame(m.get("results", []))

    counts = read_sql("""SELECT (SELECT COUNT(*) FROM headlines) h,
                                (SELECT COUNT(*) FROM features)  f,
                                (SELECT COUNT(*) FROM sentiment) s""")
    try:
        from ..monitor import run as monitor_run
        mon = list(monitor_run().values())
    except Exception:
        mon = []

    figs = sorted(p.name for p in REPORT_DIR.glob("*.png"))
    return env.get_template("dashboard.html").render(
        results=results.to_dict("records") if not results.empty else [],
        champion=m.get("champion"), test_start=m.get("test_start"),
        extra_lag=m.get("extra_lag_sessions"),
        n_headlines=int(counts["h"].iloc[0]), n_features=int(counts["f"].iloc[0]),
        n_scored=int(counts["s"].iloc[0]), tickers=list(SETTINGS.tickers),
        monitor=mon, figures=figs,
    )
