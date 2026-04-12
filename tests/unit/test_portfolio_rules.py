from __future__ import annotations

import pandas as pd

from market_prediction_agent.audits.monitor_audit import build_monitor_audit
from market_prediction_agent.backtest.cpcv import _portfolio_candidates
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.evaluation.metrics import compute_cost_adjusted_metrics
from market_prediction_agent.pipeline import PipelineRunResult


def _prediction_frame() -> pd.DataFrame:
    rows = []
    for day, aaa_return, bbb_return in [
        ("2026-01-02", 0.02, -0.01),
        ("2026-01-05", 0.015, -0.005),
        ("2026-01-06", 0.01, -0.004),
    ]:
        rows.extend(
            [
                {
                    "date": pd.Timestamp(day, tz="UTC"),
                    "ticker": "AAA",
                    "direction": "UP",
                    "prob_up": 0.65,
                    "prob_down": 0.15,
                    "prob_flat": 0.20,
                    "signal": 0.8,
                    "volume_ratio_20d": 1.0,
                    "future_simple_return": aaa_return,
                },
                {
                    "date": pd.Timestamp(day, tz="UTC"),
                    "ticker": "BBB",
                    "direction": "FLAT",
                    "prob_up": 0.2,
                    "prob_down": 0.2,
                    "prob_flat": 0.6,
                    "signal": -0.2,
                    "volume_ratio_20d": 1.0,
                    "future_simple_return": bbb_return,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_classified_directional_uses_one_sided_days_where_two_sided_stays_flat() -> None:
    predictions = _prediction_frame()
    two_sided = compute_cost_adjusted_metrics(
        predictions,
        one_way_cost_bps=0.0,
        probability_threshold=0.35,
        strategy_name="classified_two_sided",
        top_bucket_fraction=0.5,
        bottom_bucket_fraction=0.5,
        holding_days=1,
    )
    directional = compute_cost_adjusted_metrics(
        predictions,
        one_way_cost_bps=0.0,
        probability_threshold=0.35,
        strategy_name="classified_directional",
        top_bucket_fraction=0.5,
        bottom_bucket_fraction=0.5,
        holding_days=1,
    )
    assert two_sided["active_days_ratio"] == 0.0
    assert directional["active_days_ratio"] > 0.0
    assert directional["annual_return"] > two_sided["annual_return"]
    assert directional["one_sided_signal_days_ratio"] > 0.0


def test_portfolio_metrics_expose_selection_stability_and_cost_drag() -> None:
    predictions = _prediction_frame()
    metrics = compute_cost_adjusted_metrics(
        predictions,
        one_way_cost_bps=10.0,
        strategy_name="rank_long_only",
        top_bucket_fraction=0.5,
        bottom_bucket_fraction=0.0,
        holding_days=3,
    )
    assert metrics["selection_stability"] == 1.0
    assert metrics["cost_drag_annual_return"] >= 0.0
    assert metrics["avg_gross_exposure"] > 0.0


def test_turnover_controls_reduce_cost_drag_and_expose_thinning() -> None:
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02", tz="UTC"),
                "ticker": "AAA",
                "direction": "UP",
                "prob_up": 0.62,
                "prob_down": 0.18,
                "prob_flat": 0.20,
                "signal": 0.70,
                "volume_ratio_20d": 0.8,
                "future_simple_return": 0.010,
            },
            {
                "date": pd.Timestamp("2026-01-02", tz="UTC"),
                "ticker": "BBB",
                "direction": "UP",
                "prob_up": 0.58,
                "prob_down": 0.20,
                "prob_flat": 0.22,
                "signal": 0.55,
                "volume_ratio_20d": 0.8,
                "future_simple_return": 0.008,
            },
            {
                "date": pd.Timestamp("2026-01-05", tz="UTC"),
                "ticker": "AAA",
                "direction": "UP",
                "prob_up": 0.54,
                "prob_down": 0.24,
                "prob_flat": 0.22,
                "signal": 0.40,
                "volume_ratio_20d": 0.7,
                "future_simple_return": 0.006,
            },
            {
                "date": pd.Timestamp("2026-01-05", tz="UTC"),
                "ticker": "BBB",
                "direction": "UP",
                "prob_up": 0.64,
                "prob_down": 0.16,
                "prob_flat": 0.20,
                "signal": 0.72,
                "volume_ratio_20d": 0.7,
                "future_simple_return": 0.011,
            },
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "ticker": "AAA",
                "direction": "UP",
                "prob_up": 0.63,
                "prob_down": 0.17,
                "prob_flat": 0.20,
                "signal": 0.71,
                "volume_ratio_20d": 0.8,
                "future_simple_return": 0.010,
            },
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "ticker": "BBB",
                "direction": "UP",
                "prob_up": 0.53,
                "prob_down": 0.25,
                "prob_flat": 0.22,
                "signal": 0.38,
                "volume_ratio_20d": 0.8,
                "future_simple_return": 0.006,
            },
        ]
    )
    uncontrolled = compute_cost_adjusted_metrics(
        predictions,
        one_way_cost_bps=10.0,
        probability_threshold=0.35,
        strategy_name="classified_directional",
        top_bucket_fraction=0.5,
        bottom_bucket_fraction=0.0,
        holding_days=1,
    )
    controlled = compute_cost_adjusted_metrics(
        predictions,
        one_way_cost_bps=10.0,
        probability_threshold=0.35,
        strategy_name="classified_directional",
        top_bucket_fraction=0.5,
        bottom_bucket_fraction=0.0,
        holding_days=1,
        min_edge=0.02,
        bucket_hysteresis=1.0,
        hysteresis_edge_buffer=0.02,
        reentry_cooldown_days=2,
        max_turnover_per_day=0.25,
        participation_volume_floor=0.75,
        participation_volume_ceiling=1.25,
    )
    assert controlled["avg_daily_turnover"] < uncontrolled["avg_daily_turnover"]
    assert controlled["cost_drag_annual_return"] <= uncontrolled["cost_drag_annual_return"]
    assert controlled["rebalance_thinned_days_ratio"] > 0.0


def test_cpcv_candidates_include_strategy_bucket_and_holding_days() -> None:
    settings = update_settings(
        load_settings("config/default.yaml"),
        model_settings={
            "cpcv": {
                "strategy_names": ["classified_directional", "rank_long_short"],
                "portfolio_thresholds": [0.35],
                "top_bucket_fractions": [0.25],
                "bottom_bucket_fractions": [0.25],
                "holding_days": [3, 5],
            }
        },
    )
    candidates = _portfolio_candidates(settings)
    names = {str(candidate["name"]) for candidate in candidates}
    assert "classified_directional_thr_0.35_top_0.25_bottom_0.25_hold_3" in names
    assert "rank_long_short_top_0.25_bottom_0.25_hold_5" in names


def test_monitor_audit_notes_include_portfolio_rule_diagnosis() -> None:
    settings = load_settings("config/default.yaml")
    result = PipelineRunResult(
        backtest_result={
            "backtest_id": "11111111-1111-4111-8111-111111111111",
            "config": {"start_date": "2024-01-01", "end_date": "2026-04-03", "horizon": "1d"},
            "aggregate_metrics": {"hit_rate_mean": 0.44, "ece_mean": 0.03},
            "cost_adjusted_metrics": {"information_ratio": 0.2},
            "portfolio_rule_analysis": {
                "selected_rule": {
                    "strategy_name": "classified_directional",
                    "top_bucket_fraction": 0.25,
                    "bottom_bucket_fraction": 0.25,
                    "holding_days": 5,
                    "min_edge": 0.02,
                    "reentry_cooldown_days": 3,
                    "metrics": {
                        "information_ratio": 0.2,
                        "active_days_ratio": 0.35,
                        "selection_stability": 0.4,
                        "cost_drag_annual_return": 0.03,
                        "total_cost_bps": 18.0,
                    },
                },
                "uncontrolled_selected_rule": {
                    "metrics": {"information_ratio": 0.15},
                },
                "control_effect": {
                    "information_ratio_delta": 0.05,
                    "avg_daily_turnover_delta": -0.03,
                    "cost_drag_annual_return_delta": -0.01,
                    "selection_stability_delta": 0.08,
                    "rebalance_thinned_days_ratio": 0.2,
                },
                "legacy_two_sided_rule": {
                    "metrics": {"information_ratio": 0.0, "active_days_ratio": 0.0},
                },
                "primary_reasons": ["Legacy two-sided classification rule left most days flat, so classification hit rate could not monetize."],
            },
            "pbo": 0.35,
            "cpcv": {"pbo_summary": {"label": "moderate_overfit_risk"}},
            "drift_monitor": {"max_psi": 0.1},
            "regime_monitor": {"current_regime": "low_vol"},
            "retraining_monitor": {"should_retrain": False, "trigger_count": 0, "triggers": []},
        },
        forecast_output={
            "forecast_id": "22222222-2222-4222-8222-222222222222",
            "generated_at": pd.Timestamp("2026-04-03", tz="UTC").isoformat(),
            "predictions": [],
        },
        evidence_bundle={
            "data_snapshot": {
                "source_metadata": {
                    "used_source": "yahoo_chart",
                    "fallback_used": False,
                    "fallback_reason": None,
                    "dummy_mode": None,
                    "public_data_transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
                    "macro_public_data_transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
                }
            }
        },
        risk_review={},
        report_payload={"generated_at": pd.Timestamp("2026-04-03", tz="UTC").isoformat()},
        paper_trading_batch={
            "batch_id": "33333333-3333-4333-8333-333333333333",
            "approval": "MANUAL_REVIEW_REQUIRED",
            "metrics": {
                "avg_round_trip_cost_bps": 20.0,
                "avg_participation_rate": 0.01,
                "liquidity_capped_trades_this_run": 1,
                "liquidity_blocked_trades_this_run": 0,
            },
        },
        weekly_review={},
        retraining_event=None,
    )
    payload = build_monitor_audit(
        settings=settings,
        result=result,
        dataset_type="public_real_market",
        tickers=["SPY", "QQQ"],
        macro_source="fred_csv",
        notes=["x"],
    )
    joined_notes = " ".join(str(note) for note in payload["notes"])
    assert "Portfolio rule analysis" in joined_notes
    assert "Turnover-control effect" in joined_notes
    assert "Monetization diagnosis" in joined_notes

