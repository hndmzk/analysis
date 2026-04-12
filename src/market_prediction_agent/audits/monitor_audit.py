from __future__ import annotations

from pathlib import Path
import json
from typing import Any, cast
from uuid import uuid4

from market_prediction_agent.config import Settings, resolve_storage_path
from market_prediction_agent.pipeline import PipelineRunResult


def build_monitor_audit(
    *,
    settings: Settings,
    result: PipelineRunResult,
    dataset_type: str,
    tickers: list[str],
    macro_source: str,
    notes: list[str] | None = None,
) -> dict[str, object]:
    backtest_result = cast(dict[str, object], result.backtest_result)
    report_payload = cast(dict[str, object], result.report_payload)
    paper_trading_batch = cast(dict[str, object], result.paper_trading_batch)
    paper_trading_metrics = cast(dict[str, object], paper_trading_batch.get("metrics", {}))
    paper_trading_execution = cast(dict[str, object], paper_trading_batch.get("execution_diagnostics", {}))
    weekly_review = cast(dict[str, object], result.weekly_review)
    weekly_execution = cast(dict[str, object], weekly_review.get("execution_diagnostics", {}))
    evidence_bundle = cast(dict[str, object], result.evidence_bundle)
    data_snapshot = cast(dict[str, object], evidence_bundle.get("data_snapshot", {}))
    source_metadata = cast(dict[str, object], data_snapshot.get("source_metadata", {}))
    feature_sources = cast(dict[str, object], source_metadata.get("feature_sources", {}))
    ohlcv_transport = cast(dict[str, object], source_metadata.get("public_data_transport", {}))
    macro_transport = cast(dict[str, object], source_metadata.get("macro_public_data_transport", {}))
    drift_monitor = cast(dict[str, object], backtest_result.get("drift_monitor", {}))
    regime_monitor = cast(dict[str, object], backtest_result.get("regime_monitor", {}))
    retraining_monitor = cast(dict[str, object], backtest_result.get("retraining_monitor", {}))
    cpcv_result = cast(dict[str, object], backtest_result.get("cpcv", {}))
    pbo_summary = cast(dict[str, object], cpcv_result.get("pbo_summary", {}))
    pbo_diagnostics = cast(dict[str, object], cpcv_result.get("pbo_diagnostics", {}))
    cluster_adjusted_pbo_summary = cast(dict[str, object], cpcv_result.get("cluster_adjusted_pbo_summary", {}))
    cluster_adjusted_pbo_diagnostics = cast(dict[str, object], cpcv_result.get("cluster_adjusted_pbo_diagnostics", {}))
    aggregate_metrics = cast(dict[str, object], backtest_result.get("aggregate_metrics", {}))
    cost_adjusted_metrics = cast(dict[str, object], backtest_result.get("cost_adjusted_metrics", {}))
    portfolio_rule_analysis = cast(dict[str, object], backtest_result.get("portfolio_rule_analysis", {}))
    return {
        "audit_id": str(uuid4()),
        "dataset_type": dataset_type,
        "generated_at": cast(str, report_payload["generated_at"]),
        "window": {
            "start_date": cast(str, cast(dict[str, object], backtest_result["config"])["start_date"]),
            "end_date": cast(str, cast(dict[str, object], backtest_result["config"])["end_date"]),
            "horizon": cast(str, cast(dict[str, object], backtest_result["config"])["horizon"]),
        },
        "data_sources": {
            "source_mode": settings.data.source_mode,
            "ohlcv_source": source_metadata.get("used_source"),
            "proxy_ohlcv_used": source_metadata.get("used_source") == "fred_market_proxy",
            "macro_source": macro_source,
            "feature_sources": feature_sources,
            "fallback_used": source_metadata.get("fallback_used"),
            "fallback_reason": source_metadata.get("fallback_reason"),
            "dummy_mode": source_metadata.get("dummy_mode"),
            "ohlcv_transport": ohlcv_transport,
            "macro_transport": macro_transport,
        },
        "universe": tickers,
        "backtest": {
            "backtest_id": cast(str, backtest_result["backtest_id"]),
            "hit_rate_mean": aggregate_metrics.get("hit_rate_mean"),
            "ece_mean": aggregate_metrics.get("ece_mean"),
            "information_ratio": cost_adjusted_metrics.get("information_ratio"),
            "pbo": backtest_result.get("pbo"),
            "pbo_summary": pbo_summary,
            "cluster_adjusted_pbo": cpcv_result.get("cluster_adjusted_pbo"),
            "cluster_adjusted_pbo_summary": cluster_adjusted_pbo_summary,
            "portfolio_rule_analysis": portfolio_rule_analysis,
            "feature_importance_summary": backtest_result.get("feature_importance_summary", []),
            "feature_family_importance_summary": backtest_result.get("feature_family_importance_summary", []),
        },
        "feature_lineage": {
            "feature_catalog": backtest_result.get("feature_catalog", []),
            "feature_importance_summary": backtest_result.get("feature_importance_summary", []),
            "feature_family_importance_summary": backtest_result.get("feature_family_importance_summary", []),
        },
        "drift_monitor": drift_monitor,
        "regime_monitor": regime_monitor,
        "retraining_monitor": retraining_monitor,
        "paper_trading_summary": {
            "batch_id": cast(str, paper_trading_batch["batch_id"]),
            "approval": paper_trading_batch.get("approval"),
            "avg_round_trip_cost_bps": paper_trading_metrics.get("avg_round_trip_cost_bps"),
            "avg_participation_rate": paper_trading_metrics.get("avg_participation_rate"),
            "liquidity_capped_trades_this_run": paper_trading_metrics.get("liquidity_capped_trades_this_run"),
            "liquidity_blocked_trades_this_run": paper_trading_metrics.get("liquidity_blocked_trades_this_run"),
            "execution_diagnostics": paper_trading_execution,
            "weekly_execution_diagnostics": weekly_execution,
        },
        "notes": _build_notes(
            dataset_type=dataset_type,
            ohlcv_source=cast(str | None, source_metadata.get("used_source")),
            ohlcv_transport=ohlcv_transport,
            macro_transport=macro_transport,
            drift_monitor=drift_monitor,
            portfolio_rule_analysis=portfolio_rule_analysis,
            pbo_diagnostics=pbo_diagnostics,
            cluster_adjusted_pbo_diagnostics=cluster_adjusted_pbo_diagnostics,
            retraining_monitor=retraining_monitor,
            paper_trading_execution=paper_trading_execution,
            weekly_execution=weekly_execution,
            feature_sources=feature_sources,
            feature_family_importance=cast(
                list[dict[str, object]],
                backtest_result.get("feature_family_importance_summary", []),
            ),
            notes=notes or [],
        ),
    }


def persist_monitor_audit(settings: Settings, payload: dict[str, Any]) -> Path:
    storage_root = resolve_storage_path(settings)
    generated_at = payload.get("generated_at")
    audit_date = "unknown"
    if isinstance(generated_at, str):
        audit_date = generated_at[:10]
    target = storage_root / "outputs" / "monitor_audits" / str(payload["dataset_type"]) / audit_date / f"{payload['audit_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def build_monitor_audit_note(dataset_type: str) -> str:
    if dataset_type == "public_real_market":
        return "Public real-data audit uses live public OHLCV and macro series, so results vary as upstream data updates."
    return "Synthetic audit remains deterministic for regression/sanity checks and must not be treated as market evidence."


def _build_notes(
    *,
    dataset_type: str,
    ohlcv_source: str | None,
    ohlcv_transport: dict[str, object],
    macro_transport: dict[str, object],
    drift_monitor: dict[str, object],
    portfolio_rule_analysis: dict[str, object],
    pbo_diagnostics: dict[str, object],
    cluster_adjusted_pbo_diagnostics: dict[str, object],
    retraining_monitor: dict[str, object],
    paper_trading_execution: dict[str, object],
    weekly_execution: dict[str, object],
    feature_sources: dict[str, object],
    feature_family_importance: list[dict[str, object]],
    notes: list[str],
) -> list[str]:
    built = list(notes)
    if dataset_type == "public_real_market" and ohlcv_source == "fred_market_proxy":
        built.append("fred_market_proxy converts public close series into OHLCV and volume proxy fields; it is not tape-native OHLCV.")
    if ohlcv_transport:
        built.append(
            "OHLCV transport used retry/cache/snapshot safeguards: "
            f"origins={ohlcv_transport.get('origins')}, cache_used={ohlcv_transport.get('cache_used')}, "
            f"snapshot_used={ohlcv_transport.get('snapshot_used')}."
        )
    if macro_transport:
        built.append(
            "Macro transport used retry/cache/snapshot safeguards: "
            f"origins={macro_transport.get('origins')}, cache_used={macro_transport.get('cache_used')}, "
            f"snapshot_used={macro_transport.get('snapshot_used')}."
        )
    supplementary = cast(dict[str, object], drift_monitor.get("supplementary_analysis", {}))
    if portfolio_rule_analysis:
        selected_rule = cast(dict[str, object], portfolio_rule_analysis.get("selected_rule", {}))
        selected_metrics = cast(dict[str, object], selected_rule.get("metrics", {}))
        uncontrolled_rule = cast(dict[str, object], portfolio_rule_analysis.get("uncontrolled_selected_rule", {}))
        uncontrolled_metrics = cast(dict[str, object], uncontrolled_rule.get("metrics", {}))
        legacy_rule = cast(dict[str, object], portfolio_rule_analysis.get("legacy_two_sided_rule", {}))
        legacy_metrics = cast(dict[str, object], legacy_rule.get("metrics", {}))
        control_effect = cast(dict[str, object], portfolio_rule_analysis.get("control_effect", {}))
        built.append(
            "Portfolio rule analysis: "
            f"selected={selected_rule.get('strategy_name')} top={selected_rule.get('top_bucket_fraction')} "
            f"bottom={selected_rule.get('bottom_bucket_fraction')} hold={selected_rule.get('holding_days')}, "
            f"min_edge={selected_rule.get('min_edge')}, hysteresis={selected_rule.get('bucket_hysteresis')}, "
            f"edge_buffer={selected_rule.get('hysteresis_edge_buffer')}, cooldown={selected_rule.get('reentry_cooldown_days')}, "
            f"max_turnover_per_day={selected_rule.get('max_turnover_per_day')}, "
            f"IR={selected_metrics.get('information_ratio')}, active_days={selected_metrics.get('active_days_ratio')}, "
            f"selection_stability={selected_metrics.get('selection_stability')}."
        )
        built.append(
            "Legacy monetization check: "
            f"legacy_ir={legacy_metrics.get('information_ratio')}, legacy_active_days={legacy_metrics.get('active_days_ratio')}, "
            f"cost_drag_annual_return={selected_metrics.get('cost_drag_annual_return')}, total_cost_bps={selected_metrics.get('total_cost_bps')}."
        )
        built.append(
            "Turnover-control effect: "
            f"uncontrolled_ir={uncontrolled_metrics.get('information_ratio')}, "
            f"ir_delta={control_effect.get('information_ratio_delta')}, "
            f"turnover_delta={control_effect.get('avg_daily_turnover_delta')}, "
            f"cost_drag_delta={control_effect.get('cost_drag_annual_return_delta')}, "
            f"selection_stability_delta={control_effect.get('selection_stability_delta')}, "
            f"rebalance_thinned_days_ratio={control_effect.get('rebalance_thinned_days_ratio')}."
        )
        built.append("Best CPCV candidate remains a parallel comparison only; the default rule is not auto-promoted from CPCV.")
        reasons = cast(list[str], portfolio_rule_analysis.get("primary_reasons", []))
        if reasons:
            built.append("Monetization diagnosis: " + " ".join(reasons))
    if supplementary:
        feature_diagnostics = cast(list[dict[str, object]], supplementary.get("feature_diagnostics", []))
        trigger_count = sum(str(item.get("retrain_action", "ignore")) == "trigger" for item in feature_diagnostics)
        watch_count = sum(str(item.get("retrain_action", "ignore")) == "watch" for item in feature_diagnostics)
        built.append(
            "Drift supplementary analysis: "
            f"primary_cause={supplementary.get('primary_cause')}, "
            f"proxy_sensitive_flagged={len(cast(list[object], supplementary.get('proxy_sensitive_flagged_features', [])))}, "
            f"non_proxy_flagged={len(cast(list[object], supplementary.get('non_proxy_flagged_features', [])))}, "
            f"trigger_features={trigger_count}, watch_features={watch_count}."
        )
    if feature_sources:
        built.append(
            "Feature lineage sources: "
            + ", ".join(
                f"{domain}={cast(dict[str, object], details).get('used_source')}"
                for domain, details in feature_sources.items()
            )
            + "."
        )
        news_source = cast(dict[str, object], feature_sources.get("news", {}))
        utility_comparison = cast(dict[str, object], news_source.get("utility_comparison", {}))
        if utility_comparison:
            session_buckets = cast(list[dict[str, object]], utility_comparison.get("session_buckets", []))
            source_mix_buckets = cast(list[dict[str, object]], utility_comparison.get("source_mix_buckets", []))
            source_advantage = cast(dict[str, object], utility_comparison.get("source_advantage_analysis", {}))
            mixed_session = cast(dict[str, object], utility_comparison.get("mixed_session_conditions", {}))
            weighting_modes = {
                str(item.get("mode", "")): cast(dict[str, object], item)
                for item in cast(list[dict[str, object]], utility_comparison.get("weighting_mode_comparison", []))
                if str(item.get("mode", ""))
            }
            learned_summary = cast(dict[str, object], utility_comparison.get("learned_weighting", {}))
            multi_source_weighting = cast(
                dict[str, object],
                utility_comparison.get("multi_source_weighting_improvement", {}),
            )
            learned_source_weights = cast(list[dict[str, object]], learned_summary.get("source_weights", []))
            learned_session_weights = cast(list[dict[str, object]], learned_summary.get("session_weights", []))
            best_session = max(
                session_buckets,
                key=lambda item: (
                    float(cast(float, item.get("abs_ic", 0.0) or 0.0)),
                    float(cast(float, item.get("coverage", 0.0) or 0.0)),
                ),
                default={},
            )
            best_source_mix = max(
                source_mix_buckets,
                key=lambda item: (
                    float(cast(float, item.get("abs_ic", 0.0) or 0.0)),
                    float(cast(float, item.get("coverage", 0.0) or 0.0)),
                ),
                default={},
            )
            built.append(
                "News utility comparison: "
                f"baseline_coverage={cast(dict[str, object], utility_comparison.get('baseline_1d', {})).get('coverage')}, "
                f"baseline_abs_ic={cast(dict[str, object], utility_comparison.get('baseline_1d', {})).get('abs_ic')}, "
                f"baseline_overlap_rate={cast(dict[str, object], utility_comparison.get('baseline_1d', {})).get('overlap_rate')}, "
                f"best_window={utility_comparison.get('best_aggregation_window')}, "
                f"best_window_coverage={utility_comparison.get('best_aggregation_coverage')}, "
                f"best_window_abs_ic={utility_comparison.get('best_aggregation_abs_ic')}, "
                f"best_window_overlap_rate={utility_comparison.get('best_aggregation_overlap_rate')}, "
                f"best_decay={utility_comparison.get('best_decay_halflife')}, "
                f"best_decay_coverage={utility_comparison.get('best_decay_coverage')}, "
                f"best_decay_abs_ic={utility_comparison.get('best_decay_abs_ic')}, "
                f"best_variant={utility_comparison.get('best_feature_variant')}, "
                f"best_weighted_variant={utility_comparison.get('best_weighted_variant')}, "
                f"best_session={best_session.get('session_bucket')} coverage={best_session.get('coverage')} abs_ic={best_session.get('abs_ic')}, "
                f"best_source_mix={best_source_mix.get('source_mix')} coverage={best_source_mix.get('coverage')} abs_ic={best_source_mix.get('abs_ic')}, "
                f"source_advantage_reason={source_advantage.get('dominant_reason')}, "
                f"mixed_session_is_best={mixed_session.get('mixed_session_is_best')}."
            )
            built.append(
                "News weighting modes: "
                f"selected={utility_comparison.get('selected_weighting_mode')}, "
                f"unweighted_abs_ic={weighting_modes.get('unweighted', {}).get('abs_ic')}, "
                f"fixed_abs_ic={weighting_modes.get('fixed', {}).get('abs_ic')}, "
                f"learned_abs_ic={weighting_modes.get('learned', {}).get('abs_ic')}, "
                f"learned_overlap_rate={weighting_modes.get('learned', {}).get('overlap_rate')}, "
                f"learned_fallback_rate={learned_summary.get('fallback_rate')}, "
                f"multi_source_abs_ic_delta_vs_fixed={multi_source_weighting.get('learned_abs_ic_delta_vs_fixed')}, "
                f"multi_source_overlap_delta_vs_fixed={multi_source_weighting.get('learned_overlap_rate_delta_vs_fixed')}."
            )
            if learned_source_weights or learned_session_weights:
                top_source = max(
                    learned_source_weights,
                    key=lambda item: float(cast(float, item.get("mean_weight", 0.0) or 0.0)),
                    default={},
                )
                top_session = max(
                    learned_session_weights,
                    key=lambda item: float(cast(float, item.get("mean_weight", 0.0) or 0.0)),
                    default={},
                )
                built.append(
                    "Learned weighting summary: "
                    f"target={learned_summary.get('target')}, "
                    f"fit_days={learned_summary.get('fit_day_count')}, "
                    f"fallback_days={learned_summary.get('fallback_day_count')}, "
                    f"top_source={top_source.get('source_name')} weight={top_source.get('mean_weight')}, "
                    f"top_session={top_session.get('session_bucket')} weight={top_session.get('mean_weight')}."
                )
    if feature_family_importance:
        top_families = feature_family_importance[:3]
        built.append(
            "Feature family contribution: "
            + ", ".join(
                (
                    f"{item.get('feature_family')} shap={item.get('mean_abs_shap')} "
                    f"missing={item.get('mean_missing_rate')} stale={item.get('mean_stale_rate')}"
                )
                for item in top_families
            )
            + "."
        )
    if pbo_diagnostics:
        family_contribution = cast(list[dict[str, object]], pbo_diagnostics.get("family_contribution", []))
        threshold_contribution = cast(list[dict[str, object]], pbo_diagnostics.get("threshold_contribution", []))
        bucket_contribution = cast(list[dict[str, object]], pbo_diagnostics.get("bucket_contribution", []))
        holding_contribution = cast(list[dict[str, object]], pbo_diagnostics.get("holding_days_contribution", []))
        competition = cast(dict[str, object], pbo_diagnostics.get("near_candidate_competition", {}))
        if family_contribution or threshold_contribution or bucket_contribution or holding_contribution or competition:
            top_family = family_contribution[0] if family_contribution else {}
            top_threshold = threshold_contribution[0] if threshold_contribution else {}
            top_bucket = bucket_contribution[0] if bucket_contribution else {}
            top_holding = holding_contribution[0] if holding_contribution else {}
            built.append(
                "PBO diagnostics: "
                f"family={top_family.get('strategy_name')} share={top_family.get('overfit_share')}, "
                f"threshold={top_threshold.get('probability_threshold')} share={top_threshold.get('overfit_share')}, "
                f"bucket={top_bucket.get('bucket_pair')} share={top_bucket.get('overfit_share')}, "
                f"holding_days={top_holding.get('holding_days')} share={top_holding.get('overfit_share')}, "
                f"close_competition_ratio={competition.get('close_split_ratio')}, "
                f"competition_dominated={competition.get('competition_dominated')}."
            )
            competition_reason = str(competition.get("competition_reason", "")).strip()
            if competition_reason:
                built.append("PBO competition note: " + competition_reason)
    if cluster_adjusted_pbo_diagnostics:
        cluster_competition = cast(
            dict[str, object],
            cluster_adjusted_pbo_diagnostics.get("near_candidate_competition", {}),
        )
        built.append(
            "Cluster-adjusted PBO diagnostics: "
            f"close_competition_ratio={cluster_competition.get('close_split_ratio')}, "
            f"competition_dominated={cluster_competition.get('competition_dominated')}, "
            f"dominant_axis={cluster_competition.get('dominant_axis')}, "
            f"dominant_value={cluster_competition.get('dominant_value')}."
        )
    if retraining_monitor:
        drift_signal = cast(dict[str, object], retraining_monitor.get("drift_signal", {}))
        calibration_signal = cast(dict[str, object], retraining_monitor.get("calibration_signal", {}))
        pbo_signal = cast(dict[str, object], retraining_monitor.get("pbo_signal", {}))
        if drift_signal:
            built.append(
                "Retraining drift gate: "
                f"weighted_score={drift_signal.get('weighted_score')}, "
                f"threshold={drift_signal.get('weighted_threshold')}, "
                f"families={drift_signal.get('trigger_family_count')}, "
                f"features={drift_signal.get('trigger_feature_count')}, "
                f"persistent={drift_signal.get('persistent')}."
            )
        if calibration_signal:
            built.append(
                "Retraining calibration gate: "
                f"ece_breach_ratio={calibration_signal.get('ece_breach_ratio')}, "
                f"gap_breach_ratio={calibration_signal.get('calibration_gap_breach_ratio')}, "
                f"min_ratio={calibration_signal.get('minimum_breach_ratio')}, "
                f"min_folds={calibration_signal.get('minimum_persistent_folds')}, "
                f"min_runs={calibration_signal.get('minimum_persistent_runs')}, "
                f"min_span_days={calibration_signal.get('minimum_persistent_span_business_days')}, "
                f"ece_observations={calibration_signal.get('ece_observation_count')}, "
                f"ece_span_days={calibration_signal.get('ece_span_business_days')}."
            )
        if pbo_signal:
            built.append(
                "Retraining PBO gate: "
                f"value={pbo_signal.get('value')}, "
                f"candidate_value={retraining_monitor.get('candidate_level_pbo')}, "
                f"history_matches={pbo_signal.get('history_matches')}, "
                f"observation_count={pbo_signal.get('observation_count')}, "
                f"min_observations={pbo_signal.get('minimum_persistent_observations')}, "
                f"span_business_days={pbo_signal.get('span_business_days')}, "
                f"min_span_days={pbo_signal.get('minimum_persistent_span_business_days')}, "
                f"competition_dominated={pbo_signal.get('competition_dominated')}, "
                f"persistent={pbo_signal.get('persistent')}."
            )
        built.append(
            "Retraining policy: "
            f"base={retraining_monitor.get('base_should_retrain')}, "
            f"effective={retraining_monitor.get('should_retrain')}, "
            f"decision={retraining_monitor.get('policy_decision')}, "
            f"cooloff_active={retraining_monitor.get('cooloff_active')}, "
            f"remaining_business_days={retraining_monitor.get('cooloff_remaining_business_days')}, "
            f"history_lookback_count={retraining_monitor.get('history_lookback_count')}."
        )
        policy_notes = cast(list[str], retraining_monitor.get("policy_notes", []))
        if policy_notes:
            built.append("Retraining policy notes: " + " ".join(policy_notes))
    if paper_trading_execution or weekly_execution:
        built.append(
            "Execution diagnostics: "
            f"fill_rate={paper_trading_execution.get('fill_rate')}, "
            f"partial_fill_rate={paper_trading_execution.get('partial_fill_rate')}, "
            f"missed_trade_rate={paper_trading_execution.get('missed_trade_rate')}, "
            f"realized_vs_intended_exposure={weekly_execution.get('realized_vs_intended_exposure')}, "
            f"execution_cost_drag={weekly_execution.get('execution_cost_drag')}, "
            f"gap_slippage_bps={weekly_execution.get('gap_slippage_bps')}."
        )
    return built
