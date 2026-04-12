from __future__ import annotations

import pytest

from market_prediction_agent.config import ExecutionConfig
from market_prediction_agent.execution.market_impact import MarketImpactModel
from market_prediction_agent.execution.order_simulator import MarketState, Order, OrderSimulator


def test_market_impact_model_matches_square_root_formula() -> None:
    model = MarketImpactModel()

    impact = model.estimate_impact(
        order_notional=1_000_000.0,
        adv_dollar_volume=100_000_000.0,
        volatility=0.02,
        participation_rate=0.01,
    )

    assert impact["temporary_impact_bps"] == pytest.approx(28.4)
    assert impact["permanent_impact_bps"] == pytest.approx(0.628)
    assert impact["total_impact_bps"] == pytest.approx(29.028)


def test_market_order_simulate_execution_returns_full_fill() -> None:
    simulator = OrderSimulator(ExecutionConfig(enabled=True), commission_bps=5.0)

    result = simulator.simulate_execution(
        Order(
            ticker="AAA",
            side="buy",
            notional=100_000.0,
            order_type="market",
            urgency=1.0,
        ),
        MarketState(
            reference_price=100.0,
            adv_dollar_volume=100_000_000.0,
            volatility=0.02,
            bid_ask_spread_bps=10.0,
        ),
    )

    assert result.filled_notional == pytest.approx(100_000.0)
    assert result.fill_rate == pytest.approx(1.0)
    assert result.filled_quantity > 0.0
    assert result.average_fill_price > 100.0


def test_twap_order_reduces_impact_vs_market_order() -> None:
    simulator = OrderSimulator(
        ExecutionConfig(
            enabled=True,
            twap_slices=10,
        ),
        commission_bps=5.0,
    )
    market_state = MarketState(
        reference_price=100.0,
        adv_dollar_volume=50_000_000.0,
        volatility=0.03,
        bid_ask_spread_bps=10.0,
    )
    market_order = Order(ticker="AAA", side="buy", notional=1_000_000.0, order_type="market", urgency=1.0)
    twap_order = Order(ticker="AAA", side="buy", notional=1_000_000.0, order_type="twap", urgency=1.0)

    market_result = simulator.simulate_execution(market_order, market_state)
    twap_result = simulator.simulate_execution(twap_order, market_state)

    assert twap_result.fill_rate == pytest.approx(1.0)
    assert twap_result.market_impact_bps < market_result.market_impact_bps
    assert twap_result.total_cost_bps < market_result.total_cost_bps


def test_limit_order_can_return_partial_fill() -> None:
    simulator = OrderSimulator(ExecutionConfig(enabled=True), commission_bps=5.0)

    result = simulator.simulate_execution(
        Order(
            ticker="AAA",
            side="buy",
            notional=1_000_000.0,
            order_type="limit",
            urgency=0.2,
        ),
        MarketState(
            reference_price=100.0,
            adv_dollar_volume=50_000_000.0,
            volatility=0.03,
            bid_ask_spread_bps=10.0,
        ),
    )

    assert 0.0 < result.fill_rate < 1.0
    assert 0.0 < result.filled_notional < 1_000_000.0


def test_execution_quality_score_is_bounded() -> None:
    simulator = OrderSimulator(ExecutionConfig(enabled=True), commission_bps=5.0)

    result = simulator.simulate_execution(
        Order(
            ticker="AAA",
            side="sell",
            notional=750_000.0,
            order_type="limit",
            urgency=0.15,
        ),
        MarketState(
            reference_price=50.0,
            adv_dollar_volume=25_000_000.0,
            volatility=0.04,
            bid_ask_spread_bps=14.0,
        ),
    )

    assert 0.0 <= result.execution_quality_score <= 1.0
