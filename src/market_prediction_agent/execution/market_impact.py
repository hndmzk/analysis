from __future__ import annotations

from math import sqrt

from market_prediction_agent.config import ExecutionConfig, MarketImpactConfig


class MarketImpactModel:
    def __init__(self, config: MarketImpactConfig | ExecutionConfig | None = None) -> None:
        if isinstance(config, ExecutionConfig):
            self.config = config.market_impact
        else:
            self.config = config or MarketImpactConfig()

    def estimate_impact(
        self,
        order_notional: float,
        adv_dollar_volume: float,
        volatility: float,
        participation_rate: float,
    ) -> dict[str, float]:
        if order_notional <= 0 or adv_dollar_volume <= 0 or volatility <= 0:
            return {
                "temporary_impact_bps": 0.0,
                "permanent_impact_bps": 0.0,
                "total_impact_bps": 0.0,
            }

        participation = max(order_notional / adv_dollar_volume, 0.0)
        max_participation_rate = max(
            float(participation_rate),
            float(self.config.max_participation_rate),
            1e-9,
        )
        temporary_impact_bps = (
            float(self.config.eta)
            * float(volatility)
            * sqrt(participation / max_participation_rate)
            * 10_000.0
        )
        permanent_impact_bps = (
            float(self.config.gamma)
            * float(volatility)
            * participation
            * 10_000.0
        )
        total_impact_bps = temporary_impact_bps + permanent_impact_bps
        return {
            "temporary_impact_bps": temporary_impact_bps,
            "permanent_impact_bps": permanent_impact_bps,
            "total_impact_bps": total_impact_bps,
        }
