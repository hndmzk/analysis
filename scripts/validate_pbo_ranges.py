from __future__ import annotations

import argparse
import json
from typing import Any, cast

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.pipeline import MarketPredictionPipeline


VALID_PBO_LABELS = {
    "low_overfit_risk",
    "moderate_overfit_risk",
    "high_overfit_risk",
    "severe_overfit_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CPCV PBO outputs against dummy and real-like synthetic data. "
            "Reports both candidate-level and cluster-adjusted PBO."
        )
    )
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--dummy-ticker-count", type=int, default=20, help="Synthetic ticker count.")
    parser.add_argument("--dummy-days", type=int, default=900, help="Synthetic business-day history.")
    parser.add_argument("--initial-train-days", type=int, default=500, help="Walk-forward initial training window.")
    parser.add_argument("--eval-days", type=int, default=60, help="Walk-forward evaluation window.")
    parser.add_argument("--step-days", type=int, default=60, help="Walk-forward step size.")
    parser.add_argument("--cpcv-group-count", type=int, default=4, help="CPCV group count.")
    parser.add_argument("--cpcv-test-groups", type=int, default=1, help="CPCV test-group count.")
    parser.add_argument("--cpcv-max-splits", type=int, default=2, help="CPCV split cap.")
    return parser.parse_args()


def _require_pbo(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise AssertionError(f"{field_name} must be numeric, got {value!r}")
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise AssertionError(f"{field_name} must be in [0.0, 1.0], got {resolved:.3f}")
    return resolved


def _require_label(value: object, *, field_name: str) -> str:
    label = str(value)
    if label not in VALID_PBO_LABELS:
        raise AssertionError(f"{field_name} has unexpected label {label!r}")
    return label


def _run_mode(mode: str, args: argparse.Namespace) -> dict[str, object]:
    settings = load_settings(args.config)
    settings = update_settings(
        settings,
        data={
            "dummy_mode": mode,
            "storage_path": f"./storage/pbo_validation/{mode}",
            "dummy_ticker_count": args.dummy_ticker_count,
            "dummy_days": args.dummy_days,
        },
        model_settings={
            "walk_forward": {
                "initial_train_days": args.initial_train_days,
                "eval_days": args.eval_days,
                "step_days": args.step_days,
            },
            "cpcv": {
                "group_count": args.cpcv_group_count,
                "test_groups": args.cpcv_test_groups,
                "max_splits": args.cpcv_max_splits,
            },
        },
    )
    result = MarketPredictionPipeline(settings).run()
    cpcv_result = cast(dict[str, Any], result.backtest_result["cpcv"])
    cost_adjusted_metrics = cast(dict[str, Any], result.backtest_result["cost_adjusted_metrics"])
    aggregate_metrics = cast(dict[str, Any], result.backtest_result["aggregate_metrics"])
    candidate_pbo = _require_pbo(result.backtest_result["pbo"], field_name=f"{mode}.pbo")
    candidate_pbo_label = _require_label(
        cast(dict[str, Any], cpcv_result["pbo_summary"])["label"],
        field_name=f"{mode}.pbo_label",
    )
    cluster_adjusted_pbo = _require_pbo(
        cpcv_result["cluster_adjusted_pbo"],
        field_name=f"{mode}.cluster_adjusted_pbo",
    )
    cluster_adjusted_pbo_label = _require_label(
        cast(dict[str, Any], cpcv_result["cluster_adjusted_pbo_summary"])["label"],
        field_name=f"{mode}.cluster_adjusted_pbo_label",
    )
    return {
        "mode": mode,
        "pbo": candidate_pbo,
        "pbo_label": candidate_pbo_label,
        "candidate_level_pbo": candidate_pbo,
        "candidate_level_pbo_label": candidate_pbo_label,
        "cluster_adjusted_pbo": cluster_adjusted_pbo,
        "cluster_adjusted_pbo_label": cluster_adjusted_pbo_label,
        "information_ratio": cost_adjusted_metrics["information_ratio"],
        "hit_rate_mean": aggregate_metrics["hit_rate_mean"],
    }


def main() -> None:
    args = parse_args()
    dummy_result = _run_mode("null_random_walk", args)
    real_like_result = _run_mode("predictable_momentum", args)

    payload = {
        "definition": (
            "candidate_level_pbo is the fraction of CPCV splits where the best in-sample portfolio-rule candidate "
            "lands below the median out-of-sample rank. cluster_adjusted_pbo repeats the same audit after regrouping "
            "nearby parameter candidates into configured clusters."
        ),
        "thresholds": {
            "low_overfit_risk": "<0.20",
            "moderate_overfit_risk": "0.20-0.49",
            "high_overfit_risk": "0.50-0.79",
            "severe_overfit_risk": ">=0.80",
        },
        "validation": {
            "checks": [
                "candidate-level PBO is present and in [0, 1]",
                "cluster-adjusted PBO is present and in [0, 1]",
                "both PBO labels use the fixed interpretation thresholds",
            ],
            "note": (
                "predictable_momentum can still show severe candidate-level PBO when close portfolio-rule candidates "
                "swap ranks; use cluster_adjusted_pbo for the parameter-cluster view."
            ),
        },
        "results": [dummy_result, real_like_result],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
