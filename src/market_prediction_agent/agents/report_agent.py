from __future__ import annotations

from market_prediction_agent.reporting.builders import build_report_payload


class ReportAgent:
    def generate(
        self,
        forecast_output: dict[str, object],
        backtest_result: dict[str, object],
        risk_review: dict[str, object],
    ) -> dict[str, object]:
        return build_report_payload(
            forecast_output=forecast_output,
            backtest_result=backtest_result,
            risk_review=risk_review,
        )

