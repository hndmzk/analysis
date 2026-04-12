from __future__ import annotations

import argparse
import gc
import json
from typing import cast

from market_prediction_agent.audits.monitor_audit import build_monitor_audit, build_monitor_audit_note, persist_monitor_audit
from market_prediction_agent.audits.public_audit_suite import (
    build_news_feature_coverage_analysis,
    build_public_audit_suite,
    persist_public_audit_suite,
    resolve_public_audit_profile,
)
from market_prediction_agent.audits.public_snapshot_seed import seed_public_snapshots
from market_prediction_agent.config import load_settings, resolve_storage_path, update_settings
from market_prediction_agent.evaluation.retraining import build_retraining_history_entry
from market_prediction_agent.pipeline import MarketPredictionPipeline
from market_prediction_agent.schemas.validator import validate_payload
from market_prediction_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated public real-data OOS audits and summarize distributions.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument(
        "--profile",
        choices=["fast", "standard", "full_light", "full"],
        default="full_light",
        help="Audit profile preset. full_light is the default routine monitor; full is reserved for deep-dive/replay style audits.",
    )
    parser.add_argument("--anchor-date", default=None, help="Optional anchor date used when a profile generates as-of dates.")
    parser.add_argument(
        "--as-of-dates",
        default=None,
        help="Optional comma-separated as-of dates in YYYY-MM-DD format. Overrides profile-generated dates.",
    )
    parser.add_argument(
        "--ticker-sets",
        default=None,
        help="Ticker-set groups separated by '|', each group comma-separated. Overrides the profile defaults.",
    )
    parser.add_argument("--history-days", type=int, default=None, help="Optional lookback override.")
    parser.add_argument("--cpcv-max-splits", type=int, default=None, help="Optional CPCV split cap override.")
    parser.add_argument(
        "--source-mode",
        choices=["dummy", "live"],
        default=None,
        help="Optional source-mode override. Defaults to the configured value.",
    )
    parser.add_argument(
        "--seed-public-snapshots",
        action="store_true",
        help="Seed public-data cache/snapshots for the union of all ticker sets before running the suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    source_mode = args.source_mode or settings.data.source_mode
    data_updates: dict[str, object] = {
        "source_mode": source_mode,
    }
    if source_mode == "live":
        data_updates.update(
            primary_source="yahoo_chart",
            fallback_source="fred_market_proxy",
            macro_source="fred_csv",
        )
    profile = resolve_public_audit_profile(
        profile_name=args.profile,
        default_tickers=settings.data.default_tickers,
        anchor_date=args.anchor_date,
        as_of_dates_override=args.as_of_dates,
        ticker_sets_override=args.ticker_sets,
        history_days_override=args.history_days,
        cpcv_max_splits_override=args.cpcv_max_splits,
    )
    data_updates["dummy_days"] = profile.history_days
    settings = update_settings(
        settings,
        data=data_updates,
        model_settings={"cpcv": {"max_splits": profile.cpcv_max_splits}},
    )
    configure_logging(settings.app.log_level)
    if args.seed_public_snapshots and source_mode != "live":
        raise ValueError("--seed-public-snapshots requires --source-mode live.")
    if args.seed_public_snapshots:
        union_tickers = sorted({ticker for ticker_set in profile.ticker_sets for ticker in ticker_set})
        seed_public_snapshots(settings, tickers=union_tickers, history_days=profile.history_days)

    def _transport_origin_label(transport: dict[str, object]) -> str:
        origins = cast(list[object], transport.get("origins", []))
        normalized = [str(item) for item in origins if str(item)]
        return "|".join(normalized) if normalized else ""

    run_rows: list[dict[str, object]] = []
    history_by_ticker_set: dict[str, list[dict[str, object]]] = {
        ",".join(ticker_set): [] for ticker_set in profile.ticker_sets
    }
    for as_of_date in profile.as_of_dates:
        for ticker_set in profile.ticker_sets:
            ticker_label = ",".join(ticker_set)
            pipeline = MarketPredictionPipeline(settings)
            result = pipeline.run(
                tickers=ticker_set,
                as_of_time=as_of_date,
                retraining_policy_history=list(history_by_ticker_set[ticker_label]),
            )
            payload = build_monitor_audit(
                settings=settings,
                result=result,
                dataset_type="public_real_market",
                tickers=ticker_set,
                macro_source=settings.data.macro_source,
                notes=[
                    build_monitor_audit_note("public_real_market"),
                    "This audit is research-only and must not be used as an execution signal.",
                    f"suite_profile={profile.name}",
                    f"suite_profile_role={profile.role}",
                    f"suite_as_of_date={as_of_date}",
                ],
            )
            validate_payload("monitor_audit", payload)
            persist_monitor_audit(settings, payload)
            backtest = result.backtest_result
            cost_metrics = cast(dict[str, object], backtest["cost_adjusted_metrics"])
            aggregate_metrics = cast(dict[str, object], backtest["aggregate_metrics"])
            cpcv_result = cast(dict[str, object], backtest.get("cpcv", {}))
            pbo_summary = cast(dict[str, object], cpcv_result.get("pbo_summary", {}))
            pbo_diagnostics = cast(dict[str, object], cpcv_result.get("pbo_diagnostics", {}))
            cluster_adjusted_pbo_summary = cast(dict[str, object], cpcv_result.get("cluster_adjusted_pbo_summary", {}))
            cluster_adjusted_pbo_diagnostics = cast(
                dict[str, object],
                cpcv_result.get("cluster_adjusted_pbo_diagnostics", {}),
            )
            regime_monitor = cast(dict[str, object], backtest.get("regime_monitor", {}))
            candidate_pbo_competition = cast(dict[str, object], pbo_diagnostics.get("near_candidate_competition", {}))
            cluster_pbo_competition = cast(
                dict[str, object],
                cluster_adjusted_pbo_diagnostics.get("near_candidate_competition", {}),
            )
            retraining_monitor = cast(dict[str, object], backtest.get("retraining_monitor", {}))
            drift_signal = cast(dict[str, object], retraining_monitor.get("drift_signal", {}))
            calibration_signal = cast(dict[str, object], retraining_monitor.get("calibration_signal", {}))
            regime_signal = cast(dict[str, object], retraining_monitor.get("regime_signal", {}))
            drift_monitor = cast(dict[str, object], backtest.get("drift_monitor", {}))
            drift_supplementary_analysis = cast(dict[str, object], drift_monitor.get("supplementary_analysis", {}))
            data_sources = cast(dict[str, object], payload.get("data_sources", {}))
            ohlcv_transport = cast(dict[str, object], data_sources.get("ohlcv_transport", {}))
            macro_transport = cast(dict[str, object], data_sources.get("macro_transport", {}))
            feature_sources = cast(dict[str, object], data_sources.get("feature_sources", {}))
            news_source = cast(dict[str, object], feature_sources.get("news", {}))
            feature_catalog = cast(list[dict[str, object]], backtest.get("feature_catalog", []))
            news_catalog = [
                item for item in feature_catalog if str(item.get("feature_family", "")) == "news"
            ]
            processed_news = pipeline.store.read_frame("processed/news/news.parquet")
            processed_ohlcv = pipeline.store.read_frame("processed/ohlcv/ohlcv.parquet")
            news_coverage_analysis = build_news_feature_coverage_analysis(
                ohlcv=processed_ohlcv,
                news=processed_news,
                weighting_mode=settings.data.news_weighting_mode,
                learned_weighting=settings.data.learned_weighting,
            )
            lookback_entries = {
                int(cast(int, item.get("window_days", 0))): cast(dict[str, object], item)
                for item in cast(list[dict[str, object]], news_coverage_analysis.get("lookback_windows", []))
            }
            decay_entries = {
                int(cast(int, item.get("halflife_days", 0))): cast(dict[str, object], item)
                for item in cast(list[dict[str, object]], news_coverage_analysis.get("decay_halflives", []))
            }
            news_missing_rate = (
                float(
                    sum(float(cast(float, item.get("missing_rate", 0.0))) for item in news_catalog)
                    / max(len(news_catalog), 1)
                )
                if news_catalog
                else 1.0
            )
            news_stale_rate = (
                float(
                    sum(float(cast(float, item.get("stale_rate", 0.0))) for item in news_catalog)
                    / max(len(news_catalog), 1)
                )
                if news_catalog
                else float(cast(float, news_source.get("stale_rate", 0.0) or 0.0))
            )
            news_coverage = max(0.0, 1.0 - news_missing_rate)
            news_transport = cast(dict[str, object], news_source.get("transport", {}))
            run_rows.append(
                {
                    "audit_id": payload["audit_id"],
                    "backtest_id": backtest["backtest_id"],
                    "as_of_date": as_of_date,
                    "ticker_set": ticker_label,
                    "information_ratio": float(cast(float, cost_metrics["information_ratio"])),
                    "pbo": float(cast(float, backtest.get("pbo") or 0.0)),
                    "pbo_label": pbo_summary.get("label"),
                    "cluster_adjusted_pbo": float(
                        cast(float, cpcv_result.get("cluster_adjusted_pbo", backtest.get("pbo") or 0.0))
                    ),
                    "cluster_adjusted_pbo_label": cluster_adjusted_pbo_summary.get("label"),
                    "candidate_pbo_competition_dominated": bool(
                        candidate_pbo_competition.get("competition_dominated", False)
                    ),
                    "pbo_competition_dominated": bool(cluster_pbo_competition.get("competition_dominated", False)),
                    "candidate_pbo_dominant_axis": str(candidate_pbo_competition.get("dominant_axis", "")),
                    "pbo_dominant_axis": str(cluster_pbo_competition.get("dominant_axis", "")),
                    "candidate_pbo_dominant_value": str(candidate_pbo_competition.get("dominant_value", "")),
                    "pbo_dominant_value": str(cluster_pbo_competition.get("dominant_value", "")),
                    "candidate_pbo_close_competition_ratio": float(
                        cast(float, candidate_pbo_competition.get("close_split_ratio", 0.0))
                    ),
                    "pbo_close_competition_ratio": float(cast(float, cluster_pbo_competition.get("close_split_ratio", 0.0))),
                    "selection_stability": float(cast(float, cost_metrics["selection_stability"])),
                    "hit_rate_mean": float(cast(float, aggregate_metrics["hit_rate_mean"])),
                    "current_regime": str(regime_monitor.get("current_regime", "unknown")),
                    "dominant_recent_regime": str(regime_monitor.get("dominant_recent_regime", "unknown")),
                    "regime_shift_flag": bool(regime_monitor.get("regime_shift_flag", False)),
                    "state_probability": float(cast(float, regime_monitor.get("state_probability", 1.0))),
                    "transition_rate": float(cast(float, regime_monitor.get("transition_rate", 0.0))),
                    "ohlcv_source": str(data_sources.get("ohlcv_source", "")),
                    "macro_source": str(data_sources.get("macro_source", "")),
                    "ohlcv_transport_origin": _transport_origin_label(ohlcv_transport),
                    "macro_transport_origin": _transport_origin_label(macro_transport),
                    "news_used_source": str(news_source.get("used_source", "")),
                    "news_requested_source": str(news_source.get("requested_source", "")),
                    "news_fallback_used": bool(news_source.get("fallback_used", False)),
                    "news_transport_origin": _transport_origin_label(news_transport),
                    "news_feature_missing_rate": news_missing_rate,
                    "news_feature_coverage": news_coverage,
                    "news_feature_stale_rate": news_stale_rate,
                    "news_coverage_analysis": news_coverage_analysis,
                    "news_best_aggregation_window": news_coverage_analysis.get("best_aggregation_window"),
                    "news_best_aggregation_coverage": news_coverage_analysis.get("best_aggregation_coverage", 0.0),
                    "news_best_aggregation_abs_ic": news_coverage_analysis.get("best_aggregation_abs_ic", 0.0),
                    "news_best_decay_halflife": news_coverage_analysis.get("best_decay_halflife"),
                    "news_best_decay_coverage": news_coverage_analysis.get("best_decay_coverage", 0.0),
                    "news_best_decay_abs_ic": news_coverage_analysis.get("best_decay_abs_ic", 0.0),
                    "news_aggregation_improves_coverage": news_coverage_analysis.get("aggregation_improves_coverage", False),
                    "news_aggregation_improves_utility": news_coverage_analysis.get("aggregation_improves_utility", False),
                    "news_decay_improves_coverage": news_coverage_analysis.get("decay_improves_coverage", False),
                    "news_decay_improves_utility": news_coverage_analysis.get("decay_improves_utility", False),
                    "news_coverage_lookback_1d": float(cast(float, lookback_entries.get(1, {}).get("coverage", 0.0))),
                    "news_coverage_lookback_3d": float(cast(float, lookback_entries.get(3, {}).get("coverage", 0.0))),
                    "news_coverage_lookback_5d": float(cast(float, lookback_entries.get(5, {}).get("coverage", 0.0))),
                    "news_coverage_lookback_10d": float(cast(float, lookback_entries.get(10, {}).get("coverage", 0.0))),
                    "news_abs_ic_lookback_1d": float(cast(float, lookback_entries.get(1, {}).get("abs_ic", 0.0))),
                    "news_abs_ic_lookback_3d": float(cast(float, lookback_entries.get(3, {}).get("abs_ic", 0.0))),
                    "news_abs_ic_lookback_5d": float(cast(float, lookback_entries.get(5, {}).get("abs_ic", 0.0))),
                    "news_abs_ic_lookback_10d": float(cast(float, lookback_entries.get(10, {}).get("abs_ic", 0.0))),
                    "news_decay_coverage_halflife_3d": float(cast(float, decay_entries.get(3, {}).get("coverage", 0.0))),
                    "news_decay_coverage_halflife_5d": float(cast(float, decay_entries.get(5, {}).get("coverage", 0.0))),
                    "news_decay_abs_ic_halflife_3d": float(cast(float, decay_entries.get(3, {}).get("abs_ic", 0.0))),
                    "news_decay_abs_ic_halflife_5d": float(cast(float, decay_entries.get(5, {}).get("abs_ic", 0.0))),
                    "transition_profile": str(regime_signal.get("transition_profile", "not_transition")),
                    "transition_history_matches": int(cast(int, regime_signal.get("transition_history_matches", 0))),
                    "transition_observation_count": int(
                        cast(int, regime_signal.get("transition_observation_count", 0))
                    ),
                    "transition_span_business_days": int(
                        cast(int, regime_signal.get("transition_span_business_days", 0))
                    ),
                    "transition_persistent": bool(regime_signal.get("transition_persistent", False)),
                    "stable_transition": bool(regime_signal.get("stable_transition", False)),
                    "unstable_transition": bool(regime_signal.get("unstable_transition", False)),
                    "immediate_transition": bool(regime_signal.get("immediate_transition", False)),
                    "state_probability_bucket": str(regime_signal.get("state_probability_bucket", "stable")),
                    "transition_rate_bucket": str(regime_signal.get("transition_rate_bucket", "zero")),
                    "drift_weighted_score": float(cast(float, drift_signal.get("weighted_score", 0.0))),
                    "drift_pre_suppression_weighted_score": float(
                        cast(float, drift_signal.get("pre_suppression_weighted_score", 0.0))
                    ),
                    "drift_weighted_threshold": float(cast(float, drift_signal.get("weighted_threshold", 0.0))),
                    "drift_low_vol_threshold": float(cast(float, drift_signal.get("low_vol_threshold", 0.0))),
                    "drift_immediate_threshold": float(cast(float, drift_signal.get("immediate_threshold", 0.0))),
                    "drift_trigger_feature_count": int(cast(int, drift_signal.get("trigger_feature_count", 0))),
                    "drift_trigger_family_count": int(cast(int, drift_signal.get("trigger_family_count", 0))),
                    "drift_pre_suppression_trigger_feature_count": int(
                        cast(int, drift_signal.get("pre_suppression_trigger_feature_count", 0))
                    ),
                    "drift_pre_suppression_trigger_family_count": int(
                        cast(int, drift_signal.get("pre_suppression_trigger_family_count", 0))
                    ),
                    "drift_trigger_families": [
                        str(item) for item in cast(list[object], drift_signal.get("trigger_families", []))
                    ],
                    "drift_pre_suppression_trigger_families": [
                        str(item)
                        for item in cast(list[object], drift_signal.get("pre_suppression_trigger_families", []))
                    ],
                    "drift_proxy_sensitive_profile": str(drift_signal.get("proxy_sensitive_profile", "none")),
                    "drift_proxy_sensitive_trigger_feature_count": int(
                        cast(int, drift_signal.get("proxy_sensitive_trigger_feature_count", 0))
                    ),
                    "drift_non_proxy_trigger_feature_count": int(
                        cast(int, drift_signal.get("non_proxy_trigger_feature_count", 0))
                    ),
                    "drift_pre_suppression_proxy_sensitive_trigger_feature_count": int(
                        cast(int, drift_signal.get("pre_suppression_proxy_sensitive_trigger_feature_count", 0))
                    ),
                    "drift_pre_suppression_non_proxy_trigger_feature_count": int(
                        cast(int, drift_signal.get("pre_suppression_non_proxy_trigger_feature_count", 0))
                    ),
                    "drift_history_matches": int(cast(int, drift_signal.get("history_matches", 0))),
                    "drift_span_business_days": int(cast(int, drift_signal.get("span_business_days", 0))),
                    "drift_suppressed_families": [
                        str(item) for item in cast(list[object], drift_signal.get("suppressed_families", []))
                    ],
                    "drift_stable_transition_suppressed_families": [
                        str(item)
                        for item in cast(list[object], drift_signal.get("stable_transition_suppressed_families", []))
                    ],
                    "drift_primary_cause": str(drift_supplementary_analysis.get("primary_cause", "")),
                    "drift_family_persistence_would_suppress": bool(
                        drift_signal.get("family_persistence_would_suppress", False)
                    ),
                    "drift_family_persistence_counterfactual_score": float(
                        cast(float, drift_signal.get("family_persistence_counterfactual_score", 0.0))
                    ),
                    "drift_threshold_delta_to_suppress": float(
                        cast(float, drift_signal.get("threshold_delta_to_suppress", 0.0))
                    ),
                    "drift_low_vol_threshold_would_suppress": bool(
                        drift_signal.get("low_vol_threshold_would_suppress", False)
                    ),
                    "drift_stable_transition_suppression_would_suppress": bool(
                        drift_signal.get("stable_transition_suppression_would_suppress", False)
                    ),
                    "drift_stable_transition_counterfactual_score": float(
                        cast(float, drift_signal.get("stable_transition_counterfactual_score", 0.0))
                    ),
                    "ece_breach_count": int(cast(int, calibration_signal.get("ece_breach_count", 0))),
                    "ece_breach_ratio": float(cast(float, calibration_signal.get("ece_breach_ratio", 0.0))),
                    "calibration_gap_breach_count": int(
                        cast(int, calibration_signal.get("calibration_gap_breach_count", 0))
                    ),
                    "calibration_gap_breach_ratio": float(
                        cast(float, calibration_signal.get("calibration_gap_breach_ratio", 0.0))
                    ),
                    "ece_fold_persistent": bool(calibration_signal.get("ece_fold_persistent", False)),
                    "calibration_gap_fold_persistent": bool(
                        calibration_signal.get("calibration_gap_fold_persistent", False)
                    ),
                    "ece_run_persistent": bool(calibration_signal.get("ece_run_persistent", False)),
                    "calibration_gap_run_persistent": bool(
                        calibration_signal.get("calibration_gap_run_persistent", False)
                    ),
                    "ece_history_matches": int(cast(int, calibration_signal.get("ece_history_matches", 0))),
                    "calibration_gap_history_matches": int(
                        cast(int, calibration_signal.get("calibration_gap_history_matches", 0))
                    ),
                    "ece_observation_count": int(cast(int, calibration_signal.get("ece_observation_count", 0))),
                    "calibration_gap_observation_count": int(
                        cast(int, calibration_signal.get("calibration_gap_observation_count", 0))
                    ),
                    "ece_span_business_days": int(cast(int, calibration_signal.get("ece_span_business_days", 0))),
                    "calibration_gap_span_business_days": int(
                        cast(int, calibration_signal.get("calibration_gap_span_business_days", 0))
                    ),
                    "base_should_retrain": bool(retraining_monitor.get("base_should_retrain", False)),
                    "should_retrain": bool(retraining_monitor.get("should_retrain", False)),
                    "policy_decision": str(retraining_monitor.get("policy_decision", "watch_only")),
                    "trigger_names": [
                        str(name) for name in cast(list[object], retraining_monitor.get("effective_trigger_names", []))
                    ],
                    "base_trigger_names": [
                        str(name) for name in cast(list[object], retraining_monitor.get("base_trigger_names", []))
                    ],
                    "suppressed_trigger_names": [
                        str(name) for name in cast(list[object], retraining_monitor.get("suppressed_trigger_names", []))
                    ],
                    "policy_notes": [str(note) for note in cast(list[object], retraining_monitor.get("policy_notes", []))],
                }
            )
            history_by_ticker_set[ticker_label].append(
                build_retraining_history_entry(
                    as_of_date=as_of_date,
                    retraining_monitor=retraining_monitor,
                    regime_summary=cast(dict[str, object], backtest.get("regime_monitor", {})),
                    tickers=ticker_set,
                    source_mode=settings.data.source_mode,
                    pbo=cast(float | None, cpcv_result.get("cluster_adjusted_pbo", backtest.get("pbo"))),
                    pbo_summary=cluster_adjusted_pbo_summary or pbo_summary,
                )
            )
            del (
                pipeline,
                result,
                payload,
                backtest,
                cost_metrics,
                aggregate_metrics,
                cpcv_result,
                pbo_summary,
                pbo_diagnostics,
                cluster_adjusted_pbo_summary,
                cluster_adjusted_pbo_diagnostics,
                regime_monitor,
                candidate_pbo_competition,
                cluster_pbo_competition,
                retraining_monitor,
                drift_signal,
                calibration_signal,
                regime_signal,
                drift_monitor,
                drift_supplementary_analysis,
                data_sources,
                ohlcv_transport,
                macro_transport,
                feature_sources,
                news_source,
                feature_catalog,
                news_catalog,
                processed_news,
                processed_ohlcv,
                news_coverage_analysis,
                lookback_entries,
                decay_entries,
                news_transport,
            )
            gc.collect()
    suite_payload = build_public_audit_suite(
        runs=run_rows,
        as_of_dates=profile.as_of_dates,
        ticker_sets=profile.ticker_sets,
        cpcv_max_splits=settings.model_settings.cpcv.max_splits,
        profile_name=profile.name,
        profile_role=profile.role,
        analysis_mode="live_suite",
    )
    validate_payload("monitor_audit_suite", suite_payload)
    suite_path = persist_public_audit_suite(settings, suite_payload)
    print(
        json.dumps(
            {
                "storage_path": str(resolve_storage_path(settings)),
                "profile": {
                    "name": profile.name,
                    "role": profile.role,
                    "as_of_dates": profile.as_of_dates,
                    "ticker_sets": profile.ticker_sets,
                    "history_days": profile.history_days,
                    "cpcv_max_splits": profile.cpcv_max_splits,
                },
                "suite_path": str(suite_path),
                "distribution_summary": suite_payload["distribution_summary"],
                "run_count": suite_payload["run_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
