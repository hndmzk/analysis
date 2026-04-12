from __future__ import annotations

import argparse
import json

from market_prediction_agent.audits.monitor_audit import build_monitor_audit, build_monitor_audit_note, persist_monitor_audit
from market_prediction_agent.audits.public_snapshot_seed import seed_public_snapshots
from market_prediction_agent.config import load_settings, resolve_storage_path, update_settings
from market_prediction_agent.pipeline import MarketPredictionPipeline
from market_prediction_agent.schemas.validator import validate_payload
from market_prediction_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic or public real-data OOS monitor audits.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument(
        "--dataset-type",
        choices=["synthetic", "public_real_market", "both"],
        default="both",
        help="Audit dataset type.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=["null_random_walk", "predictable_momentum"],
        default="predictable_momentum",
        help="Dummy mode used when dataset-type includes synthetic.",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated ticker list for public real-data audit. Defaults to config data.default_tickers.",
    )
    parser.add_argument("--history-days", type=int, default=None, help="Optional lookback override.")
    parser.add_argument("--as-of-date", default=None, help="Optional as-of date override in YYYY-MM-DD format.")
    parser.add_argument("--dummy-ticker-count", type=int, default=None, help="Optional synthetic ticker count override.")
    parser.add_argument("--cpcv-max-splits", type=int, default=None, help="Optional CPCV split cap override.")
    parser.add_argument(
        "--seed-public-snapshots",
        action="store_true",
        help="Seed public-data cache/snapshots before running the public real-data audit.",
    )
    return parser.parse_args()


def _apply_runtime_overrides(settings, args: argparse.Namespace):
    data_updates: dict[str, object] = {}
    model_updates: dict[str, object] = {}
    if args.history_days is not None:
        data_updates["dummy_days"] = args.history_days
    if args.dummy_ticker_count is not None:
        data_updates["dummy_ticker_count"] = args.dummy_ticker_count
    if args.cpcv_max_splits is not None:
        model_updates["cpcv"] = {"max_splits": args.cpcv_max_splits}
    updates: dict[str, object] = {}
    if data_updates:
        updates["data"] = data_updates
    if model_updates:
        updates["model_settings"] = model_updates
    if not updates:
        return settings
    return update_settings(settings, **updates)


def _run_synthetic_audit(args: argparse.Namespace) -> dict[str, object]:
    settings = load_settings(args.config)
    settings = _apply_runtime_overrides(settings, args)
    settings = update_settings(
        settings,
        data={
            "source_mode": "dummy",
            "dummy_mode": args.synthetic_mode,
        },
    )
    configure_logging(settings.app.log_level)
    tickers = [f"TICK{i:03d}" for i in range(settings.data.dummy_ticker_count)]
    result = MarketPredictionPipeline(settings).run(tickers=tickers, as_of_time=args.as_of_date)
    payload = build_monitor_audit(
        settings=settings,
        result=result,
        dataset_type="synthetic",
        tickers=tickers,
        macro_source="dummy",
        notes=[
            build_monitor_audit_note("synthetic"),
            f"dummy_mode={args.synthetic_mode}",
        ],
    )
    validate_payload("monitor_audit", payload)
    persist_monitor_audit(settings, payload)
    return payload


def _run_public_real_audit(args: argparse.Namespace) -> dict[str, object]:
    settings = load_settings(args.config)
    settings = _apply_runtime_overrides(settings, args)
    settings = update_settings(
        settings,
        data={
            "source_mode": "live",
            "primary_source": "yahoo_chart",
            "fallback_source": "fred_market_proxy",
            "macro_source": "fred_csv",
        },
    )
    tickers = settings.data.default_tickers
    if args.tickers:
        tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required for public real-data audit.")
    configure_logging(settings.app.log_level)
    if args.seed_public_snapshots:
        seed_public_snapshots(settings, tickers=tickers, history_days=args.history_days)
    try:
        result = MarketPredictionPipeline(settings).run(tickers=tickers, as_of_time=args.as_of_date)
    except RuntimeError as exc:
        if "no cache or snapshot fallback" in str(exc):
            raise RuntimeError(
                "Public real-data audit failed before a local snapshot was available. "
                "Run `uv run python scripts/seed_public_snapshots.py --tickers SPY,QQQ,DIA,GLD` first, "
                "or rerun this command with `--seed-public-snapshots` while network access is available."
            ) from exc
        raise
    payload = build_monitor_audit(
        settings=settings,
        result=result,
        dataset_type="public_real_market",
        tickers=tickers,
        macro_source=settings.data.macro_source,
        notes=[
            build_monitor_audit_note("public_real_market"),
            "This audit is research-only and must not be used as an execution signal.",
        ],
    )
    validate_payload("monitor_audit", payload)
    persist_monitor_audit(settings, payload)
    return payload


def main() -> None:
    args = parse_args()
    payloads: list[dict[str, object]] = []
    if args.dataset_type in {"synthetic", "both"}:
        payloads.append(_run_synthetic_audit(args))
    if args.dataset_type in {"public_real_market", "both"}:
        payloads.append(_run_public_real_audit(args))
    print(
        json.dumps(
            {
                "storage_path": str(resolve_storage_path(load_settings(args.config))),
                "audits": payloads,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
