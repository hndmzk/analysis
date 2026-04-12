from __future__ import annotations

from datetime import timezone
import hashlib
from typing import Any, Mapping, cast
from uuid import uuid4

import pandas as pd


def detect_regime(macro: pd.DataFrame, vix_stress_threshold: float) -> str:
    vix_rows = macro.loc[macro["series_id"] == "VIXCLS"]
    if vix_rows.empty:
        return "unknown"
    latest_vix = float(vix_rows.sort_values("available_at")["value"].iloc[-1])
    return "high_vol" if latest_vix > vix_stress_threshold else "low_vol"


def build_forecast_output(
    predictions: pd.DataFrame,
    model_version: str,
    horizon: str,
    regime: str,
) -> dict[str, object]:
    forecast_id = str(uuid4())
    generated_at = pd.Timestamp.now(tz=timezone.utc).isoformat()
    payload_predictions = []
    for _, row in predictions.iterrows():
        payload_predictions.append(
            {
                "ticker": row["ticker"],
                "direction": row["direction"],
                "probabilities": {
                    "UP": float(row["prob_up"]),
                    "DOWN": float(row["prob_down"]),
                    "FLAT": float(row["prob_flat"]),
                },
                "expected_return": float(row["expected_return"]) if pd.notna(row["expected_return"]) else None,
                "predicted_volatility": float(row["predicted_volatility"]) if pd.notna(row["predicted_volatility"]) else None,
                "confidence": row["confidence"],
                "top_features": row["top_features"],
                "stale_data_flag": bool(row["stale_data_flag"]),
            }
        )
    return {
        "forecast_id": forecast_id,
        "generated_at": generated_at,
        "model_version": model_version,
        "horizon": horizon,
        "regime": regime,
        "predictions": payload_predictions,
    }


def build_evidence_bundle(
    forecast_id: str,
    features: pd.DataFrame,
    macro: pd.DataFrame,
    model_name: str,
    model_version: str,
    trained_at: str,
    training_samples: int,
    stale_tickers: list[str],
    source_metadata: dict[str, object],
    hyperparameters: Mapping[str, object] | None = None,
    feature_importance: list[dict[str, object]] | None = None,
    feature_catalog: list[dict[str, object]] | None = None,
    feature_family_importance: list[dict[str, object]] | None = None,
    calibration_summary: Mapping[str, object] | None = None,
    drift_summary: Mapping[str, object] | None = None,
    regime_summary: Mapping[str, object] | None = None,
    retraining_monitor: Mapping[str, object] | None = None,
    news_summary: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    features_hash = hashlib.sha256(features.to_csv(index=False).encode("utf-8")).hexdigest()
    stale_ratio = float(len(stale_tickers) / max(len(features), 1))
    return {
        "bundle_id": str(uuid4()),
        "forecast_id": forecast_id,
        "created_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "data_snapshot": {
            "ohlcv_as_of": pd.to_datetime(features["date"].max(), utc=True).isoformat(),
            "macro_as_of": pd.to_datetime(macro["available_at"].max(), utc=True).isoformat(),
            "features_hash": features_hash,
            "stale_tickers": stale_tickers,
            "ohlcv_stale_ratio": stale_ratio,
            "source_metadata": source_metadata,
            "feature_catalog": feature_catalog or [],
            "drift_monitor": dict(drift_summary or {}),
            "regime_monitor": dict(regime_summary or {}),
        },
        "model_info": {
            "model_name": model_name,
            "model_version": model_version,
            "trained_at": trained_at,
            "training_samples": int(training_samples),
            "hyperparameters": dict(hyperparameters or {}),
            "feature_importance_top": feature_importance or [],
            "feature_family_importance": feature_family_importance or [],
            "calibration_summary": dict(calibration_summary or {}),
            "retraining_monitor": dict(retraining_monitor or {}),
        },
        "news_summary": news_summary or [],
    }


def build_risk_review(
    forecast_id: str,
    forecast_output: dict[str, object],
    backtest_result: dict[str, object],
    max_drawdown_limit: float,
) -> dict[str, object]:
    predictions = cast(list[dict[str, Any]], forecast_output["predictions"])
    stale_ratio = sum(1 for item in predictions if item["stale_data_flag"]) / max(len(predictions), 1)
    concentrated = 0.0
    if predictions:
        concentrated = max(max(item["probabilities"].values()) for item in predictions)
    backtest_config = cast(dict[str, object], backtest_result["config"])
    dummy_mode = cast(str | None, backtest_config.get("dummy_mode"))
    cost_adjusted_metrics = cast(dict[str, float], backtest_result["cost_adjusted_metrics"])
    drift_monitor = cast(dict[str, Any], backtest_result.get("drift_monitor", {}))
    regime_monitor = cast(dict[str, Any], backtest_result.get("regime_monitor", {}))
    retraining_monitor = cast(dict[str, Any], backtest_result.get("retraining_monitor", {}))
    supplementary_drift = cast(dict[str, Any], drift_monitor.get("supplementary_analysis", {}))
    max_psi = float(drift_monitor.get("max_psi", 0.0) or 0.0)
    max_raw_psi = float(drift_monitor.get("max_raw_psi", max_psi) or max_psi)
    psi_warning_threshold = float(drift_monitor.get("warning_threshold", 0.2) or 0.2)
    psi_critical_threshold = float(drift_monitor.get("critical_threshold", 0.25) or 0.25)
    critical_features = cast(list[str], drift_monitor.get("critical_features", []))
    warning_features = cast(list[str], drift_monitor.get("warning_features", []))
    current_regime = cast(str, regime_monitor.get("current_regime", forecast_output.get("regime", "unknown")))
    regime_shift_flag = bool(regime_monitor.get("regime_shift_flag", False))
    state_probability = float(regime_monitor.get("state_probability", 1.0) or 1.0)
    retrain_triggers = cast(list[dict[str, Any]], retraining_monitor.get("triggers", []))
    retrain_required = bool(retraining_monitor.get("should_retrain", False))
    drift_status = "PASS"
    if critical_features:
        drift_status = "FAIL"
    elif warning_features:
        drift_status = "WARNING"
    if dummy_mode is not None and drift_status == "FAIL":
        drift_status = "WARNING"
    max_drawdown = cost_adjusted_metrics["max_drawdown"]
    drift_detail = (
        f"Adjusted PSI max={max_psi:.3f} (raw={max_raw_psi:.3f}); "
        f"warning_features={len(warning_features)}, critical_features={len(critical_features)}, "
        f"cause={cast(str, supplementary_drift.get('primary_cause', 'mixed_or_unknown'))}."
    )
    if dummy_mode is not None and critical_features:
        drift_detail += " Synthetic dummy runs downgrade drift FAIL to WARNING."
    checks: list[dict[str, Any]] = [
        {
            "check_name": "data_freshness",
            "status": "WARNING" if stale_ratio > 0.1 else "PASS",
            "detail": f"Stale ratio={stale_ratio:.2%}",
            "metric_value": stale_ratio,
            "threshold": 0.1,
        },
        {
            "check_name": "feature_drift",
            "status": drift_status,
            "detail": drift_detail,
            "metric_value": max_psi,
            "threshold": psi_critical_threshold if drift_status == "FAIL" else psi_warning_threshold,
        },
        {
            "check_name": "prediction_concentration",
            "status": "WARNING" if concentrated > 0.75 else "PASS",
            "detail": f"Max class probability={concentrated:.2f}",
            "metric_value": concentrated,
            "threshold": 0.75,
        },
        {
            "check_name": "model_staleness",
            "status": "WARNING" if retrain_required else "PASS",
            "detail": (
                "Retraining trigger active: "
                + "; ".join(cast(str, item["detail"]) for item in retrain_triggers)
                if retrain_triggers
                else "No retraining trigger fired in the current run."
            ),
            "metric_value": float(retraining_monitor.get("trigger_count", 0) or 0.0),
            "threshold": 1.0,
        },
        {
            "check_name": "liquidity_check",
            "status": "PASS",
            "detail": "Synthetic large-cap universe in MVP.",
            "metric_value": None,
            "threshold": None,
        },
        {
            "check_name": "regime_consistency",
            "status": "WARNING" if regime_shift_flag or current_regime == "transition" else "PASS",
            "detail": (
                f"Detected regime={current_regime}, "
                f"regime_shift={regime_shift_flag}, "
                f"state_probability={state_probability:.2f}."
            ),
            "metric_value": state_probability,
            "threshold": 0.6,
        },
        {
            "check_name": "hallucination_guard",
            "status": "PASS",
            "detail": "No live trading or free-form LLM output in MVP report.",
            "metric_value": None,
            "threshold": None,
        },
        {
            "check_name": "source_integrity",
            "status": "PASS",
            "detail": "Data sources are adapter controlled and schema validated.",
            "metric_value": None,
            "threshold": None,
        },
    ]
    statuses = [cast(str, item["status"]) for item in checks]
    approval_notes: list[str] = []
    if "FAIL" in statuses:
        risk_level = "CRITICAL"
        approval = "BLOCKED"
        approval_notes.append("At least one risk check failed; blocked pending human review.")
    elif max_drawdown < -max_drawdown_limit or statuses.count("WARNING") >= 2:
        risk_level = "HIGH"
        approval = "MANUAL_REVIEW_REQUIRED"
        approval_notes.append("High-risk profile detected from drawdown or multiple warnings.")
    elif "WARNING" in statuses:
        risk_level = "MEDIUM"
        approval = "MANUAL_REVIEW_REQUIRED"
        approval_notes.append("At least one warning requires human review.")
    else:
        risk_level = "LOW"
        approval = "AUTO_APPROVED"
        approval_notes.append("No blocking checks or warnings were triggered.")
    if dummy_mode == "null_random_walk":
        if approval == "AUTO_APPROVED":
            approval = "MANUAL_REVIEW_REQUIRED"
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
        approval_notes.append("Sanity dummy runs are never auto-approved.")
    if dummy_mode == "predictable_momentum":
        approval_notes.append("Predictable dummy mode is for development/demo only, not investment use.")
    return {
        "review_id": str(uuid4()),
        "forecast_id": forecast_id,
        "reviewed_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "checks": checks,
        "overall_risk_level": risk_level,
        "approval": approval,
        "approval_notes": approval_notes,
        "recommendations": [
            "Live trading remains disabled.",
            "Treat forecast output as research support only.",
            *(
                ["Retrain the model before the next production-style run."]
                if retrain_required
                else ["No immediate retraining is required by drift/regime/calibration checks."]
            ),
        ],
        "retraining_monitor": retraining_monitor,
    }


def build_report_payload(
    forecast_output: dict[str, object],
    backtest_result: dict[str, object],
    risk_review: dict[str, object],
) -> dict[str, object]:
    predictions = cast(list[dict[str, Any]], forecast_output["predictions"])
    aggregate_metrics = cast(dict[str, float], backtest_result["aggregate_metrics"])
    cost_adjusted_metrics = cast(dict[str, float], backtest_result["cost_adjusted_metrics"])
    drift_monitor = cast(dict[str, Any], backtest_result.get("drift_monitor", {}))
    regime_monitor = cast(dict[str, Any], backtest_result.get("regime_monitor", {}))
    retraining_monitor = cast(dict[str, Any], risk_review.get("retraining_monitor", {}))
    pbo_summary = cast(dict[str, Any], cast(dict[str, object], backtest_result.get("cpcv", {})).get("pbo_summary", {}))
    risk_level = cast(str, risk_review["overall_risk_level"])
    approval = cast(str, risk_review["approval"])
    top_prediction = max(
        predictions,
        key=lambda item: max(item["probabilities"].values()),
        default=None,
    )
    summary = "No predictions were generated."
    if top_prediction is not None:
        summary = (
            f"{len(predictions)} tickers forecasted for {cast(str, forecast_output['horizon'])}; "
            f"highest-confidence call is {top_prediction['ticker']} {top_prediction['direction']}."
        )
    return {
        "report_id": str(uuid4()),
        "report_type": "backtest_report",
        "generated_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "summary": summary[:500],
        "sections": [
            {
                "title": "Forecast Snapshot",
                "content": summary,
            },
            {
                "title": "Backtest Metrics",
                "content": (
                    f"Hit rate mean={aggregate_metrics['hit_rate_mean']:.3f}, "
                    f"log loss mean={aggregate_metrics['log_loss_mean']:.3f}, "
                    f"ECE mean={aggregate_metrics.get('ece_mean', 0.0):.3f}, "
                    f"PBO={backtest_result.get('pbo') if backtest_result.get('pbo') is not None else 'n/a'} "
                    f"({cast(str, pbo_summary.get('label', 'not_available'))}), "
                    f"IR={cost_adjusted_metrics['information_ratio']:.3f}."
                ),
            },
            {
                "title": "Risk Review",
                "content": (
                    f"Risk level={risk_level}, "
                    f"approval={approval}, "
                    f"max PSI={float(drift_monitor.get('max_psi', 0.0) or 0.0):.3f}, "
                    f"regime={cast(str, regime_monitor.get('current_regime', 'unknown'))}, "
                    f"retrain={bool(retraining_monitor.get('should_retrain', False))}."
                ),
            },
        ],
        "forecast_ids": [cast(str, forecast_output["forecast_id"])],
        "risk_review_id": cast(str, risk_review["review_id"]),
        "disclaimers": [
            "This system is for market analysis only and does not place live trades.",
            "Outputs are synthetic/offline-safe by default and are not investment advice.",
        ],
    }
