from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import cast
from uuid import uuid4

import numpy as np
import pandas as pd

from market_prediction_agent.backtest.cpcv import run_cpcv_backtest
from market_prediction_agent.config import Settings
from market_prediction_agent.data.universe import resolve_active_constituents
from market_prediction_agent.evaluation.metrics import (
    RETURN_REGRESSION_METRIC_KEYS,
    VOLATILITY_REGRESSION_METRIC_KEYS,
    compute_cost_adjusted_metrics,
    compute_fold_metrics,
    diebold_mariano,
    return_regression_metrics,
    volatility_regression_metrics,
)
from market_prediction_agent.models.factory import build_model


@dataclass(slots=True)
class WalkForwardWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


def _selected_portfolio_rule(settings: Settings) -> dict[str, float | int | str]:
    rule = settings.model_settings.portfolio_rule
    return {
        "strategy_name": rule.strategy_name,
        "probability_threshold": float(rule.probability_threshold),
        "top_bucket_fraction": float(rule.top_bucket_fraction),
        "bottom_bucket_fraction": float(rule.bottom_bucket_fraction),
        "holding_days": int(rule.holding_days),
        "min_edge": float(rule.min_edge),
        "bucket_hysteresis": float(rule.bucket_hysteresis),
        "hysteresis_edge_buffer": float(rule.hysteresis_edge_buffer),
        "reentry_cooldown_days": int(rule.reentry_cooldown_days),
        "max_turnover_per_day": float(rule.max_turnover_per_day),
        "participation_volume_floor": float(rule.participation_volume_floor),
        "participation_volume_ceiling": float(rule.participation_volume_ceiling),
    }


def _legacy_two_sided_rule() -> dict[str, float | int | str]:
    return {
        "strategy_name": "classified_two_sided",
        "probability_threshold": 0.40,
        "top_bucket_fraction": 0.20,
        "bottom_bucket_fraction": 0.20,
        "holding_days": 1,
        "min_edge": 0.0,
        "bucket_hysteresis": 0.0,
        "hysteresis_edge_buffer": 0.0,
        "reentry_cooldown_days": 0,
        "max_turnover_per_day": 0.0,
        "participation_volume_floor": 1.0,
        "participation_volume_ceiling": 1.0,
    }


def _portfolio_rule_analysis(
    aggregate_metrics: dict[str, float],
    selected_rule: dict[str, float | int | str],
    selected_metrics: dict[str, float],
    uncontrolled_metrics: dict[str, float],
    legacy_metrics: dict[str, float],
    cpcv_result: dict[str, object],
) -> dict[str, object]:
    reasons: list[str] = []
    if legacy_metrics["active_days_ratio"] <= 0.05 and selected_metrics["active_days_ratio"] > legacy_metrics["active_days_ratio"]:
        reasons.append(
            "Legacy two-sided classification rule left most days flat, so classification hit rate could not monetize."
        )
    if selected_metrics["gross_annual_return"] <= 0:
        reasons.append(
            "Gross cross-sectional return remained non-positive, so directional accuracy did not convert into usable PnL."
        )
    if selected_metrics["cost_drag_annual_return"] > max(abs(selected_metrics["gross_annual_return"]) * 0.25, 0.01):
        reasons.append("Turnover and trading costs materially eroded gross return.")
    if selected_metrics["selection_stability"] < 0.25:
        reasons.append("Selected names were unstable across rebalances, which weakens realized monetization.")
    if selected_metrics["one_sided_signal_days_ratio"] > selected_metrics["two_sided_signal_days_ratio"]:
        reasons.append("One-sided signals dominated the sample, so strict market-neutral rules under-deployed the model.")
    cpcv_candidates = cast(list[dict[str, object]], cpcv_result.get("candidate_strategies", []))
    if cpcv_candidates:
        best_candidate = cpcv_candidates[0]
    else:
        best_candidate = {}
    if cast(float | None, cpcv_result.get("pbo")) is not None and cast(float, cpcv_result["pbo"]) >= 0.8:
        reasons.append("Candidate family remained unstable across CPCV, so severe PBO still warns against overfitting.")
    control_effect = {
        "information_ratio_delta": float(selected_metrics["information_ratio"] - uncontrolled_metrics["information_ratio"]),
        "avg_daily_turnover_delta": float(selected_metrics["avg_daily_turnover"] - uncontrolled_metrics["avg_daily_turnover"]),
        "cost_drag_annual_return_delta": float(selected_metrics["cost_drag_annual_return"] - uncontrolled_metrics["cost_drag_annual_return"]),
        "selection_stability_delta": float(selected_metrics["selection_stability"] - uncontrolled_metrics["selection_stability"]),
        "rebalance_thinned_days_ratio": float(selected_metrics.get("rebalance_thinned_days_ratio", 0.0)),
    }
    if control_effect["information_ratio_delta"] > 0.0:
        reasons.append("Turnover controls improved realized IR without adopting the CPCV best candidate.")
    if control_effect["cost_drag_annual_return_delta"] < 0.0:
        reasons.append("Turnover controls reduced cost drag versus the uncontrolled selected rule.")
    if control_effect["selection_stability_delta"] > 0.0:
        reasons.append("Selection stability improved after hysteresis, cooldown, and rebalance throttling.")
    if not reasons:
        reasons.append("Selected portfolio rule monetized the classification signal without a dominant failure mode.")
    return {
        "classification_hit_rate_mean": aggregate_metrics["hit_rate_mean"],
        "selected_rule": {
            **selected_rule,
            "metrics": selected_metrics,
        },
        "uncontrolled_selected_rule": {
            **selected_rule,
            "min_edge": 0.0,
            "bucket_hysteresis": 0.0,
            "hysteresis_edge_buffer": 0.0,
            "reentry_cooldown_days": 0,
            "max_turnover_per_day": 0.0,
            "participation_volume_floor": 1.0,
            "participation_volume_ceiling": 1.0,
            "metrics": uncontrolled_metrics,
        },
        "control_effect": control_effect,
        "legacy_two_sided_rule": {
            **_legacy_two_sided_rule(),
            "metrics": legacy_metrics,
        },
        "best_cpcv_candidate": best_candidate,
        "candidate_family_summary": cpcv_result.get("portfolio_rule_summary", {}),
        "primary_reasons": reasons,
    }


def _feature_catalog_lookup(feature_catalog: list[dict[str, object]] | None) -> dict[str, dict[str, object]]:
    return {
        str(item.get("feature")): dict(item)
        for item in (feature_catalog or [])
        if item.get("feature") is not None
    }


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _annotate_feature_importance(
    feature_importance: list[dict[str, object]],
    feature_catalog_lookup: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for item in feature_importance:
        feature = str(item.get("feature", "unknown"))
        catalog_entry = feature_catalog_lookup.get(feature, {})
        annotated.append(
            {
                **item,
                "feature_family": catalog_entry.get("feature_family"),
                "data_source": catalog_entry.get("data_source"),
                "missing_rate": catalog_entry.get("missing_rate"),
                "stale_rate": catalog_entry.get("stale_rate"),
            }
        )
    return annotated


def _feature_family_importance_summary(
    feature_importance: list[dict[str, object]],
    feature_catalog_lookup: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for item in feature_importance:
        feature = str(item.get("feature", "unknown"))
        catalog_entry = feature_catalog_lookup.get(feature, {})
        family = str(catalog_entry.get("feature_family", "unknown"))
        entry = grouped.setdefault(
            family,
            {
                "feature_family": family,
                "feature_count": 0,
                "mean_abs_shap": [],
                "missing_rate": [],
                "stale_rate": [],
                "data_sources": set(),
            },
        )
        feature_count = cast(int, entry["feature_count"])
        entry["feature_count"] = feature_count + 1
        cast(list[float], entry["mean_abs_shap"]).append(_as_float(item.get("mean_abs_shap", 0.0)))
        cast(list[float], entry["missing_rate"]).append(_as_float(catalog_entry.get("missing_rate", 0.0)))
        cast(list[float], entry["stale_rate"]).append(_as_float(catalog_entry.get("stale_rate", 0.0)))
        cast(set[str], entry["data_sources"]).add(str(catalog_entry.get("data_source", "unknown")))
    summary: list[dict[str, object]] = []
    for family, item in grouped.items():
        summary.append(
            {
                "feature_family": family,
                "feature_count": cast(int, item["feature_count"]),
                "mean_abs_shap": float(np.mean(cast(list[float], item["mean_abs_shap"]))),
                "mean_missing_rate": float(np.mean(cast(list[float], item["missing_rate"]))),
                "mean_stale_rate": float(np.mean(cast(list[float], item["stale_rate"]))),
                "data_sources": sorted(cast(set[str], item["data_sources"])),
            }
        )
    summary.sort(key=lambda entry: cast(float, entry["mean_abs_shap"]), reverse=True)
    return summary


def build_walk_forward_windows(
    dates: list[pd.Timestamp],
    initial_train_days: int,
    eval_days: int,
    step_days: int,
    embargo_days: int,
) -> list[WalkForwardWindow]:
    windows: list[WalkForwardWindow] = []
    fold_id = 1
    train_end_index = initial_train_days - 1
    while True:
        eval_start_index = train_end_index + embargo_days + 1
        eval_end_index = eval_start_index + eval_days - 1
        if eval_end_index >= len(dates):
            break
        windows.append(
            WalkForwardWindow(
                fold_id=fold_id,
                train_start=dates[0],
                train_end=dates[train_end_index],
                eval_start=dates[eval_start_index],
                eval_end=dates[eval_end_index],
            )
        )
        train_end_index += step_days
        fold_id += 1
    return windows


def run_walk_forward_backtest(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    settings: Settings,
    *,
    model_name: str | None = None,
    include_feature_importance: bool = True,
    include_cpcv: bool = True,
    feature_catalog: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    walk = settings.model_settings.walk_forward
    effective_model_name = model_name or settings.model_settings.primary
    unique_dates = sorted(training_frame["date"].drop_duplicates().tolist())
    windows = build_walk_forward_windows(
        unique_dates,
        walk.initial_train_days,
        walk.eval_days,
        walk.step_days,
        walk.embargo_days,
    )
    if not windows:
        raise ValueError("Not enough history to build walk-forward windows.")
    fold_entries: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    loss_differentials: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []
    importance_by_feature: dict[str, list[float]] = {feature: [] for feature in feature_columns}
    feature_catalog_lookup = _feature_catalog_lookup(feature_catalog)
    for window in windows:
        train_frame = training_frame.loc[
            (training_frame["date"] >= window.train_start) & (training_frame["date"] <= window.train_end)
        ].copy()
        eval_frame = training_frame.loc[
            (training_frame["date"] >= window.eval_start) & (training_frame["date"] <= window.eval_end)
        ].copy()
        active_constituents = resolve_active_constituents(settings, as_of_date=window.eval_end)
        if active_constituents is not None:
            train_frame = train_frame.loc[train_frame["ticker"].isin(active_constituents)].copy()
            eval_frame = eval_frame.loc[eval_frame["ticker"].isin(active_constituents)].copy()
        if train_frame.empty or eval_frame.empty:
            continue
        model = build_model(settings=settings, model_name=effective_model_name, version=settings.model_settings.version)
        model.fit(train_frame, feature_columns)
        predictions = model.predict(eval_frame, include_explanations=False)
        fold_feature_importance = model.feature_importance_top(eval_frame, limit=10) if include_feature_importance else []
        fold_feature_importance = _annotate_feature_importance(fold_feature_importance, feature_catalog_lookup)
        class_prior = train_frame["direction_label"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0)
        predictions["baseline_prob_down"] = class_prior.loc[0]
        predictions["baseline_prob_flat"] = class_prior.loc[1]
        predictions["baseline_prob_up"] = class_prior.loc[2]
        predictions["direction_label"] = eval_frame["direction_label"].to_numpy(dtype=int)
        predictions["future_simple_return"] = eval_frame["future_simple_return"].to_numpy(dtype=float)
        predictions["target_return"] = eval_frame["target_return"].to_numpy(dtype=float)
        predictions["future_volatility_20d"] = eval_frame["future_volatility_20d"].to_numpy(dtype=float)
        predictions["volume_ratio_20d"] = eval_frame["volume_ratio_20d"].to_numpy(dtype=float)
        predictions["fold_id"] = window.fold_id
        metrics, differential = compute_fold_metrics(predictions)
        metrics.update(return_regression_metrics(predictions))
        metrics.update(volatility_regression_metrics(predictions))
        fold_metrics.append(metrics)
        loss_differentials.append(differential)
        prediction_frames.append(predictions)
        fold_entries.append(
            {
                "fold_id": window.fold_id,
                "train_start": window.train_start.date().isoformat(),
                "train_end": window.train_end.date().isoformat(),
                "eval_start": window.eval_start.date().isoformat(),
                "eval_end": window.eval_end.date().isoformat(),
                "n_train": int(len(train_frame)),
                "n_eval": int(len(eval_frame)),
                "metrics": metrics,
                "calibration": model.calibration_summary,
                "feature_importance": fold_feature_importance,
            }
        )
        if include_feature_importance:
            for item in fold_feature_importance:
                importance_by_feature[cast(str, item["feature"])].append(cast(float, item["mean_abs_shap"]))
    if not prediction_frames:
        raise ValueError("No non-empty walk-forward folds remained after universe filtering.")
    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    aggregate = {
        "hit_rate_mean": float(np.mean([metrics["hit_rate"] for metrics in fold_metrics])),
        "hit_rate_std": float(np.std([metrics["hit_rate"] for metrics in fold_metrics], ddof=0)),
        "log_loss_mean": float(np.mean([metrics["log_loss"] for metrics in fold_metrics])),
        "brier_score_mean": float(np.mean([metrics["brier_score"] for metrics in fold_metrics])),
        "ece_mean": float(np.mean([metrics["ece"] for metrics in fold_metrics])),
        "calibration_gap_mean": float(np.mean([metrics["calibration_gap"] for metrics in fold_metrics])),
        "ic_mean": float(np.mean([metrics["ic"] for metrics in fold_metrics])),
        "rank_ic_mean": float(np.mean([metrics["rank_ic"] for metrics in fold_metrics])),
        "ece_warning_breach_count": int(
            sum(metrics["ece"] >= settings.model_settings.calibration.ece_warning for metrics in fold_metrics)
        ),
        "ece_warning_breach_ratio": float(
            np.mean([metrics["ece"] >= settings.model_settings.calibration.ece_warning for metrics in fold_metrics])
        ),
        "calibration_gap_warning_breach_count": int(
            sum(
                metrics["calibration_gap"] >= settings.model_settings.calibration.calibration_gap_warning
                for metrics in fold_metrics
            )
        ),
        "calibration_gap_warning_breach_ratio": float(
            np.mean(
                [
                    metrics["calibration_gap"] >= settings.model_settings.calibration.calibration_gap_warning
                    for metrics in fold_metrics
                ]
            )
        ),
        **{
            f"{metric_name}_mean": float(np.mean([metrics[metric_name] for metrics in fold_metrics]))
            for metric_name in (*RETURN_REGRESSION_METRIC_KEYS, *VOLATILITY_REGRESSION_METRIC_KEYS)
        },
    }
    aggregate_feature_importance: list[dict[str, object]] = []
    if include_feature_importance:
        aggregate_feature_importance = [
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(values)),
            }
            for feature, values in importance_by_feature.items()
            if values
        ]
        aggregate_feature_importance.sort(key=lambda item: cast(float, item["mean_abs_shap"]), reverse=True)
        aggregate_feature_importance = _annotate_feature_importance(aggregate_feature_importance, feature_catalog_lookup)
    one_way_cost_bps = settings.trading.cost_bps.equity_oneway + settings.trading.slippage_bps.equity_oneway
    selected_rule = _selected_portfolio_rule(settings)
    cost_metrics = compute_cost_adjusted_metrics(
        combined_predictions,
        one_way_cost_bps=one_way_cost_bps,
        probability_threshold=cast(float, selected_rule["probability_threshold"]),
        top_bucket_fraction=cast(float, selected_rule["top_bucket_fraction"]),
        bottom_bucket_fraction=cast(float, selected_rule["bottom_bucket_fraction"]),
        strategy_name=cast(str, selected_rule["strategy_name"]),
        holding_days=cast(int, selected_rule["holding_days"]),
        min_edge=cast(float, selected_rule["min_edge"]),
        bucket_hysteresis=cast(float, selected_rule["bucket_hysteresis"]),
        hysteresis_edge_buffer=cast(float, selected_rule["hysteresis_edge_buffer"]),
        reentry_cooldown_days=cast(int, selected_rule["reentry_cooldown_days"]),
        max_turnover_per_day=cast(float, selected_rule["max_turnover_per_day"]),
        participation_volume_floor=cast(float, selected_rule["participation_volume_floor"]),
        participation_volume_ceiling=cast(float, selected_rule["participation_volume_ceiling"]),
    )
    uncontrolled_selected_metrics = compute_cost_adjusted_metrics(
        combined_predictions,
        one_way_cost_bps=one_way_cost_bps,
        probability_threshold=cast(float, selected_rule["probability_threshold"]),
        top_bucket_fraction=cast(float, selected_rule["top_bucket_fraction"]),
        bottom_bucket_fraction=cast(float, selected_rule["bottom_bucket_fraction"]),
        strategy_name=cast(str, selected_rule["strategy_name"]),
        holding_days=cast(int, selected_rule["holding_days"]),
    )
    legacy_rule = _legacy_two_sided_rule()
    legacy_cost_metrics = compute_cost_adjusted_metrics(
        combined_predictions,
        one_way_cost_bps=one_way_cost_bps,
        probability_threshold=cast(float, legacy_rule["probability_threshold"]),
        top_bucket_fraction=cast(float, legacy_rule["top_bucket_fraction"]),
        bottom_bucket_fraction=cast(float, legacy_rule["bottom_bucket_fraction"]),
        strategy_name=cast(str, legacy_rule["strategy_name"]),
        holding_days=cast(int, legacy_rule["holding_days"]),
    )
    dm = diebold_mariano(np.concatenate(loss_differentials))
    cpcv_result: dict[str, object] = {}
    pbo: float | None = None
    if include_cpcv:
        cpcv_result, pbo = run_cpcv_backtest(training_frame, feature_columns, settings)
        cpcv_result["pbo"] = pbo
    portfolio_rule_analysis = _portfolio_rule_analysis(
        aggregate_metrics=aggregate,
        selected_rule=selected_rule,
        selected_metrics=cost_metrics,
        uncontrolled_metrics=uncontrolled_selected_metrics,
        legacy_metrics=legacy_cost_metrics,
        cpcv_result=cpcv_result,
    )
    result = {
        "backtest_id": str(uuid4()),
        "config": {
            "model_name": effective_model_name,
            "model_version": settings.model_settings.version,
            "start_date": unique_dates[0].date().isoformat(),
            "end_date": unique_dates[-1].date().isoformat(),
            "horizon": settings.data.forecast_horizon,
            "initial_train_days": walk.initial_train_days,
            "eval_days": walk.eval_days,
            "embargo_days": walk.embargo_days,
            "feature_version": settings.model_settings.feature_version,
            "universe": settings.data.universe,
            "dummy_mode": settings.data.dummy_mode if settings.data.source_mode == "dummy" else None,
            "portfolio_rule": selected_rule,
        },
        "completed_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "folds": fold_entries,
        "aggregate_metrics": aggregate,
        "cost_adjusted_metrics": cost_metrics,
        "portfolio_rule_analysis": portfolio_rule_analysis,
        "feature_importance_summary": aggregate_feature_importance[:10],
        "feature_catalog": feature_catalog or [],
        "feature_family_importance_summary": _feature_family_importance_summary(
            aggregate_feature_importance,
            feature_catalog_lookup,
        ),
        "cpcv": cpcv_result,
        "pbo": pbo,
        "diebold_mariano": dm,
    }
    return result, combined_predictions


def run_model_comparisons(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    settings: Settings,
    primary_backtest_result: dict[str, object],
    *,
    feature_catalog: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    primary_config = cast(dict[str, object], primary_backtest_result["config"])
    primary_model_name = cast(str, primary_config["model_name"])
    primary_aggregate = cast(dict[str, float], primary_backtest_result["aggregate_metrics"])
    primary_cost = cast(dict[str, float], primary_backtest_result["cost_adjusted_metrics"])
    comparison_results: list[dict[str, object]] = []
    for comparison_model_name in settings.model_settings.comparison_models:
        if comparison_model_name == primary_model_name:
            continue
        try:
            comparison_backtest, _ = run_walk_forward_backtest(
                training_frame,
                feature_columns,
                settings,
                model_name=comparison_model_name,
                include_feature_importance=False,
                include_cpcv=False,
                feature_catalog=feature_catalog,
            )
            comparison_aggregate = cast(dict[str, float], comparison_backtest["aggregate_metrics"])
            comparison_cost = cast(dict[str, float], comparison_backtest["cost_adjusted_metrics"])
            comparison_results.append(
                {
                    "model_name": comparison_model_name,
                    "model_version": settings.model_settings.version,
                    "status": "completed",
                    "aggregate_metrics": comparison_aggregate,
                    "cost_adjusted_metrics": comparison_cost,
                    "diebold_mariano": comparison_backtest["diebold_mariano"],
                    "comparison_to_primary": {
                        "hit_rate_mean_delta": float(comparison_aggregate["hit_rate_mean"] - primary_aggregate["hit_rate_mean"]),
                        "log_loss_mean_delta": float(comparison_aggregate["log_loss_mean"] - primary_aggregate["log_loss_mean"]),
                        "ece_mean_delta": float(comparison_aggregate["ece_mean"] - primary_aggregate["ece_mean"]),
                        "information_ratio_delta": float(
                            comparison_cost["information_ratio"] - primary_cost["information_ratio"]
                        ),
                    },
                }
            )
        except Exception as exc:
            comparison_results.append(
                {
                    "model_name": comparison_model_name,
                    "model_version": settings.model_settings.version,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return comparison_results

