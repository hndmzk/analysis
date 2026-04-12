from __future__ import annotations

import argparse
import json

from market_prediction_agent.config import load_settings, resolve_storage_path, update_settings
from market_prediction_agent.pipeline import MarketPredictionPipeline
from market_prediction_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the market prediction backtest pipeline.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument(
        "--dummy-mode",
        choices=["null_random_walk", "predictable_momentum"],
        default=None,
        help="Override dummy generation mode.",
    )
    parser.add_argument(
        "--comparison-models",
        default=None,
        help="Optional comma-separated comparison model list. Use 'none' to disable comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.dummy_mode:
        settings = update_settings(settings, data={"dummy_mode": args.dummy_mode})
    if args.comparison_models is not None:
        if args.comparison_models.strip().lower() == "none":
            comparison_models: list[str] = []
        else:
            comparison_models = [
                item.strip() for item in args.comparison_models.split(",") if item.strip()
            ]
        settings = update_settings(
            settings, model_settings={"comparison_models": comparison_models}
        )
    configure_logging(settings.app.log_level)
    pipeline = MarketPredictionPipeline(settings)
    result = pipeline.run()
    summary = {
        "storage_path": str(resolve_storage_path(settings)),
        "dummy_mode": settings.data.dummy_mode,
        "backtest_id": result.backtest_result["backtest_id"],
        "forecast_id": result.forecast_output["forecast_id"],
        "report_id": result.report_payload["report_id"],
        "hit_rate_mean": result.backtest_result["aggregate_metrics"]["hit_rate_mean"],
        "ece_mean": result.backtest_result["aggregate_metrics"].get("ece_mean"),
        "information_ratio": result.backtest_result["cost_adjusted_metrics"]["information_ratio"],
        "max_psi": result.backtest_result.get("drift_monitor", {}).get("max_psi"),
        "pbo": result.backtest_result.get("pbo"),
        "pbo_label": result.backtest_result.get("cpcv", {}).get("pbo_summary", {}).get("label"),
        "regime": result.backtest_result.get("regime_monitor", {}).get("current_regime"),
        "should_retrain": result.backtest_result.get("retraining_monitor", {}).get("should_retrain"),
        "approval": result.risk_review["approval"],
        "model_comparison": [
            {
                "model_name": item.get("model_name"),
                "status": item.get("status"),
                "hit_rate_mean": item.get("aggregate_metrics", {}).get("hit_rate_mean"),
                "ece_mean": item.get("aggregate_metrics", {}).get("ece_mean"),
                "information_ratio": item.get("cost_adjusted_metrics", {}).get("information_ratio"),
            }
            for item in result.backtest_result.get("model_comparison", [])
        ],
        "paper_trading_batch_id": result.paper_trading_batch["batch_id"],
        "weekly_review_id": result.weekly_review["review_id"],
        "retraining_event_id": result.retraining_event["event_id"] if result.retraining_event else None,
        "used_source": result.evidence_bundle["data_snapshot"]["source_metadata"]["used_source"],
        "fallback_used": result.evidence_bundle["data_snapshot"]["source_metadata"]["fallback_used"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
