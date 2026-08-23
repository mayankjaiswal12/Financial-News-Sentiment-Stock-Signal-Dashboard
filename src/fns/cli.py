"""
Single entry point for every stage.

    python -m fns.cli all          # full pipeline, end to end
    python -m fns.cli serve        # dashboard on :8000

Why argparse subcommands rather than a pile of scripts: each stage is separately
re-runnable and each is idempotent, so you can re-run one without redoing the
expensive ones. `all` just calls them in dependency order.
"""
from __future__ import annotations

import argparse
import sys


def cmd_initdb(a):
    from .db import init_db
    init_db(drop=a.drop)
    print("[db] schema ready" + (" (recreated)" if a.drop else ""))


def cmd_ingest(a):
    from .db import init_db
    from .ingest.headlines import ingest_headlines
    from .ingest.prices import ingest_prices
    init_db()
    # Prices FIRST: headline session-attribution reads the trading calendar
    # back out of the prices table, so this order is a hard dependency.
    ingest_prices()
    ingest_headlines(extra_lag=a.extra_lag)


def cmd_score(a):
    from .sentiment.finbert import score_headlines
    score_headlines(limit=a.limit, batch_size=a.batch_size)


def cmd_features(a):
    from .features import ingest_features
    ingest_features()


def cmd_analyse(a):
    from .analysis import run_all
    run_all()


def cmd_xs(a):
    from .cross_sectional import run
    run(test_start=a.test_start)


def cmd_train(a):
    from .train import print_results, run
    print_results(run(test_start=a.test_start))


def cmd_monitor(a):
    from .monitor import run
    run(a.model)


def cmd_plots(a):
    from .plots import generate_all
    generate_all(a.ticker)


def cmd_experiment(a):
    from .experiments import leakage_comparison, same_day_leak_demo
    leakage_comparison()
    same_day_leak_demo()


def cmd_all(a):
    cmd_ingest(a); cmd_score(a); cmd_features(a)
    cmd_analyse(a); cmd_train(a); cmd_xs(a); cmd_monitor(a); cmd_plots(a)
    print("\nDone. Launch the dashboard with:  python -m fns.cli serve")


def cmd_serve(a):
    import uvicorn
    uvicorn.run("fns.api.main:app", host=a.host, port=a.port, reload=a.reload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fns", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("initdb", help="create tables");  s.add_argument("--drop", action="store_true"); s.set_defaults(func=cmd_initdb)
    s = sub.add_parser("ingest", help="download prices + headlines")
    s.add_argument("--extra-lag", type=int, default=None, dest="extra_lag",
                   help="sessions to withhold news (default from config; 0 = leaky)")
    s.set_defaults(func=cmd_ingest)
    s = sub.add_parser("score", help="run FinBERT over un-scored headlines")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    s.set_defaults(func=cmd_score)
    s = sub.add_parser("features", help="build the modelling table"); s.set_defaults(func=cmd_features)
    s = sub.add_parser("analyse", help="lead-lag, IC, bucket diagnostics"); s.set_defaults(func=cmd_analyse)
    s = sub.add_parser("train", help="train + benchmark")
    s.add_argument("--test-start", default=None, dest="test_start"); s.set_defaults(func=cmd_train)
    s = sub.add_parser("xs", help="cross-sectional ranking + long-short backtest")
    s.add_argument("--test-start", default=None, dest="test_start"); s.set_defaults(func=cmd_xs)
    s = sub.add_parser("monitor", help="drift + rolling accuracy")
    s.add_argument("--model", default=None); s.set_defaults(func=cmd_monitor)
    s = sub.add_parser("plots", help="render figures")
    s.add_argument("--ticker", default="NVDA"); s.set_defaults(func=cmd_plots)
    s = sub.add_parser("experiment", help="leaky vs leak-free comparison"); s.set_defaults(func=cmd_experiment)

    s = sub.add_parser("all", help="run the whole pipeline")
    s.add_argument("--extra-lag", type=int, default=None, dest="extra_lag")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    s.add_argument("--test-start", default=None, dest="test_start")
    s.add_argument("--model", default=None)
    s.add_argument("--ticker", default="NVDA")
    s.set_defaults(func=cmd_all)

    s = sub.add_parser("serve", help="run the FastAPI dashboard")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true"); s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
