from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from market_prediction_agent.config import ExecutionConfig

from .market_impact import MarketImpactModel


OrderType = Literal["market", "limit", "twap"]
OrderSide = Literal["buy", "sell"]


@dataclass(slots=True, frozen=True)
class Order:
    ticker: str
    side: OrderSide
    notional: float
    order_type: OrderType
    urgency: float


@dataclass(slots=True, frozen=True)
class MarketState:
    reference_price: float
    adv_dollar_volume: float
    volatility: float
    bid_ask_spread_bps: float


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    filled_notional: float
    filled_quantity: float
    fill_rate: float
    average_fill_price: float
    market_impact_bps: float
    total_cost_bps: float
    slippage_bps: float
    execution_quality_score: float


class OrderSimulator:
    def __init__(self, config: ExecutionConfig, *, commission_bps: float = 0.0) -> None:
        self.config = config
        self.commission_bps = float(commission_bps)
        self.market_impact_model = MarketImpactModel(config)

    def simulate_execution(self, order: Order, market_state: MarketState) -> ExecutionResult:
        if order.order_type not in self.config.order_types:
            raise ValueError(f"Unsupported order type '{order.order_type}' for current execution config.")
        if order.order_type == "market":
            return self._simulate_market(order, market_state)
        if order.order_type == "limit":
            return self._simulate_limit(order, market_state)
        return self._simulate_twap(order, market_state)

    def _simulate_market(self, order: Order, market_state: MarketState) -> ExecutionResult:
        return self._build_result(
            order=order,
            market_state=market_state,
            filled_notional=max(float(order.notional), 0.0),
            realized_spread_bps=self._half_spread_bps(market_state),
            impact_bps=self._impact_bps(float(order.notional), market_state),
            baseline_total_cost_bps=self._baseline_total_cost_bps(order, market_state),
        )

    def _simulate_limit(self, order: Order, market_state: MarketState) -> ExecutionResult:
        urgency = min(max(float(order.urgency), 0.0), 1.0)
        half_spread_bps = self._half_spread_bps(market_state)
        baseline_total_cost_bps = self._baseline_total_cost_bps(order, market_state)
        required_cross_bps = max(baseline_total_cost_bps - self.commission_bps, 1e-9)
        threshold_bps = max(half_spread_bps * max(urgency, 0.1), 0.01)
        fill_rate = min(threshold_bps / required_cross_bps, 1.0)
        filled_notional = max(float(order.notional) * fill_rate, 0.0)
        realized_spread_bps = min(threshold_bps, half_spread_bps)
        return self._build_result(
            order=order,
            market_state=market_state,
            filled_notional=filled_notional,
            realized_spread_bps=realized_spread_bps,
            impact_bps=self._impact_bps(filled_notional, market_state),
            baseline_total_cost_bps=baseline_total_cost_bps,
        )

    def _simulate_twap(self, order: Order, market_state: MarketState) -> ExecutionResult:
        slices = max(int(self.config.twap_slices), 1)
        slice_notional = max(float(order.notional), 0.0) / slices
        slice_impact_bps = self._impact_bps(slice_notional, market_state)
        baseline_total_cost_bps = self._baseline_total_cost_bps(order, market_state)
        return self._build_result(
            order=order,
            market_state=market_state,
            filled_notional=max(float(order.notional), 0.0),
            realized_spread_bps=self._half_spread_bps(market_state),
            impact_bps=slice_impact_bps,
            baseline_total_cost_bps=baseline_total_cost_bps,
        )

    def _build_result(
        self,
        *,
        order: Order,
        market_state: MarketState,
        filled_notional: float,
        realized_spread_bps: float,
        impact_bps: float,
        baseline_total_cost_bps: float,
    ) -> ExecutionResult:
        total_cost_bps = impact_bps + realized_spread_bps + self.commission_bps if filled_notional > 0 else 0.0
        average_fill_price = (
            self._apply_bps(
                float(market_state.reference_price),
                impact_bps + realized_spread_bps,
                order.side,
            )
            if filled_notional > 0
            else 0.0
        )
        filled_quantity = (filled_notional / average_fill_price) if average_fill_price > 0 else 0.0
        fill_rate = (filled_notional / float(order.notional)) if order.notional > 0 else 0.0
        slippage_bps = abs(total_cost_bps - baseline_total_cost_bps) if filled_notional > 0 else baseline_total_cost_bps
        execution_quality_score = self._quality_score(fill_rate, total_cost_bps, slippage_bps)
        return ExecutionResult(
            filled_notional=filled_notional,
            filled_quantity=filled_quantity,
            fill_rate=fill_rate,
            average_fill_price=average_fill_price,
            market_impact_bps=impact_bps,
            total_cost_bps=total_cost_bps,
            slippage_bps=slippage_bps,
            execution_quality_score=execution_quality_score,
        )

    def _baseline_total_cost_bps(self, order: Order, market_state: MarketState) -> float:
        impact_bps = self._impact_bps(float(order.notional), market_state)
        return impact_bps + self._half_spread_bps(market_state) + self.commission_bps

    def _impact_bps(self, order_notional: float, market_state: MarketState) -> float:
        impact = self.market_impact_model.estimate_impact(
            order_notional=order_notional,
            adv_dollar_volume=float(market_state.adv_dollar_volume),
            volatility=float(market_state.volatility),
            participation_rate=float(self.config.market_impact.max_participation_rate),
        )
        return float(impact["total_impact_bps"])

    @staticmethod
    def _half_spread_bps(market_state: MarketState) -> float:
        return max(float(market_state.bid_ask_spread_bps), 0.0) / 2.0

    @staticmethod
    def _apply_bps(reference_price: float, bps: float, side: OrderSide) -> float:
        if reference_price <= 0:
            return 0.0
        direction = 1.0 if side == "buy" else -1.0
        return reference_price * (1.0 + direction * (bps / 10_000.0))

    @staticmethod
    def _quality_score(fill_rate: float, total_cost_bps: float, slippage_bps: float) -> float:
        normalized_cost = min(max(total_cost_bps, 0.0) / 100.0, 1.0)
        normalized_slippage = min(max(slippage_bps, 0.0) / 50.0, 1.0)
        penalty = ((1.0 - fill_rate) * 0.55) + (normalized_cost * 0.30) + (normalized_slippage * 0.15)
        return max(0.0, min(1.0, 1.0 - penalty))
