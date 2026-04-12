from __future__ import annotations

from typing import cast

from market_prediction_agent.config import Settings
from market_prediction_agent.reporting.builders import build_risk_review


class RiskAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def review(self, forecast_output: dict[str, object], backtest_result: dict[str, object]) -> dict[str, object]:
        return build_risk_review(
            forecast_id=cast(str, forecast_output["forecast_id"]),
            forecast_output=forecast_output,
            backtest_result=backtest_result,
            max_drawdown_limit=self.settings.risk.max_drawdown_limit,
        )
