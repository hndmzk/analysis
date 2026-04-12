from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pandas as pd

from market_prediction_agent.config import Settings
from market_prediction_agent.execution.order_simulator import MarketState, Order, OrderSimulator
from market_prediction_agent.evaluation.retraining import build_retraining_history_entry
from market_prediction_agent.retraining.ledger_service import RetrainingEventLedgerService
from market_prediction_agent.storage.parquet_store import ParquetStore


logger = logging.getLogger(__name__)


LEDGER_PATH = Path("outputs") / "paper_trading" / "trade_ledger.parquet"
LEDGER_COLUMNS = [
    "trade_id",
    "forecast_id",
    "forecast_date",
    "signal_time",
    "execution_time",
    "execution_delay_business_days",
    "settlement_date",
    "week_id",
    "ticker",
    "direction",
    "confidence",
    "prob_up",
    "prob_down",
    "prob_flat",
    "expected_return",
    "expected_trade_return",
    "expected_net_return",
    "predicted_volatility",
    "stale_data_flag",
    "regime",
    "approval",
    "dummy_mode",
    "used_source",
    "should_retrain",
    "signal_reference_price",
    "reference_close",
    "reference_volume",
    "estimated_dollar_volume",
    "adv_dollar_volume",
    "fillable_notional_cap",
    "target_notional",
    "intended_order_quantity",
    "executed_notional",
    "filled_quantity",
    "unfilled_quantity",
    "unfilled_notional",
    "execution_price",
    "participation_rate",
    "fill_rate",
    "partial_fill",
    "missed_trade",
    "liquidity_capped",
    "liquidity_blocked",
    "gap_slippage_bps",
    "round_trip_fee_bps",
    "round_trip_slippage_bps",
    "round_trip_cost_bps",
    "execution_cost_drag",
    "execution_cost_drag_bps",
    "execution_reason",
    "status",
    "asset_return",
    "gross_trade_return",
    "net_trade_return",
    "realized_return",
    "realized_direction",
    "hit",
    "settled_at",
    "created_at",
]


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _week_id(timestamp: pd.Timestamp) -> str:
    iso = timestamp.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _settlement_date(forecast_date: pd.Timestamp, horizon_days: int) -> pd.Timestamp:
    return forecast_date + pd.tseries.offsets.BDay(horizon_days)


def _execution_date(forecast_date: pd.Timestamp, delay_business_days: int) -> pd.Timestamp:
    return forecast_date + pd.tseries.offsets.BDay(delay_business_days)


def _execution_timestamp(execution_date: pd.Timestamp, execution_time_utc: str) -> pd.Timestamp:
    hours, minutes, seconds = (int(part) for part in execution_time_utc.split(":"))
    return execution_date + pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _realized_direction(realized_return: float, threshold: float) -> str:
    if realized_return <= -threshold:
        return "DOWN"
    if realized_return >= threshold:
        return "UP"
    return "FLAT"


def _directional_return(direction: str, asset_return: float) -> float:
    if direction == "UP":
        return asset_return
    if direction == "DOWN":
        return -asset_return
    return 0.0


def _adverse_gap_slippage_bps(direction: str, signal_price: float, execution_price: float) -> float:
    if signal_price <= 0 or execution_price <= 0:
        return 0.0
    if direction == "UP":
        return max((execution_price / signal_price) - 1.0, 0.0) * 10_000.0
    if direction == "DOWN":
        return max(1.0 - (execution_price / signal_price), 0.0) * 10_000.0
    return 0.0


def _empty_trade_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _empty_execution_diagnostics() -> dict[str, object]:
    return {
        "intended_trades": 0,
        "attempted_trades": 0,
        "filled_trades": 0,
        "partial_fills": 0,
        "missed_trades": 0,
        "fill_rate": 0.0,
        "partial_fill_rate": 0.0,
        "missed_trade_rate": 0.0,
        "realized_vs_intended_exposure": 0.0,
        "execution_cost_drag": 0.0,
        "execution_cost_drag_bps": 0.0,
        "gap_slippage_bps": 0.0,
    }


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(cast(float | int | str, value))


def _optional_timestamp(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _to_utc_timestamp(value).isoformat()


@dataclass(slots=True)
class PaperTradingArtifacts:
    batch_log: dict[str, object]
    weekly_review: dict[str, object]
    retraining_event: dict[str, object] | None


class PaperTradingService:
    def __init__(self, settings: Settings, store: ParquetStore) -> None:
        self.settings = settings
        self.store = store
        self.retraining_ledger_service = RetrainingEventLedgerService(settings, store)
        self.order_simulator = (
            OrderSimulator(
                settings.execution,
                commission_bps=float(settings.trading.cost_bps.equity_oneway),
            )
            if settings.execution.enabled
            else None
        )

    def update(
        self,
        forecast_output: dict[str, object],
        backtest_result: dict[str, object],
        risk_review: dict[str, object],
        source_metadata: dict[str, object],
        feature_frame: pd.DataFrame,
    ) -> PaperTradingArtifacts:
        generated_at = _to_utc_timestamp(forecast_output["generated_at"])
        generated_at_for_ledger = generated_at.floor("s")
        forecast_date = generated_at.normalize()
        ledger = self._load_trade_ledger()
        market_context = self._prepare_market_context(feature_frame)
        settled_count = self._settle_pending_trades(ledger, market_context, forecast_date, generated_at_for_ledger)
        executed_rows = self._execute_queued_trades(ledger, market_context, forecast_date, generated_at_for_ledger)
        market_snapshot = self._build_market_snapshot(market_context, forecast_date)
        new_batch_rows = self._build_new_trade_rows(
            forecast_output=forecast_output,
            risk_review=risk_review,
            source_metadata=source_metadata,
            market_snapshot=market_snapshot,
            forecast_date=forecast_date,
            generated_at=generated_at_for_ledger,
        )
        if new_batch_rows:
            ledger = pd.concat([ledger, pd.DataFrame(new_batch_rows)], ignore_index=True)
        ledger = self._normalize_ledger(ledger)
        self.store.write_frame(LEDGER_PATH, ledger)
        batch_log = self._build_batch_log(
            forecast_output=forecast_output,
            risk_review=risk_review,
            generated_at=generated_at,
            batch_rows=new_batch_rows,
            executed_rows=executed_rows,
            settled_count=settled_count,
            ledger=ledger,
        )
        weekly_review = self._build_weekly_review(ledger, forecast_date)
        self._append_retraining_policy_history(
            forecast_output=forecast_output,
            backtest_result=backtest_result,
            risk_review=risk_review,
            source_metadata=source_metadata,
            generated_at=generated_at,
        )
        retraining_event = self._build_retraining_event(
            forecast_output=forecast_output,
            backtest_result=backtest_result,
            risk_review=risk_review,
            generated_at=generated_at,
        )
        return PaperTradingArtifacts(
            batch_log=batch_log,
            weekly_review=weekly_review,
            retraining_event=retraining_event,
        )

    def _normalize_ledger(self, ledger: pd.DataFrame) -> pd.DataFrame:
        if ledger.empty:
            return _empty_trade_ledger()
        normalized = ledger.copy()
        round_trip_fee_bps = 2 * float(self.settings.trading.cost_bps.equity_oneway)
        round_trip_slippage_bps = 2 * float(self.settings.trading.slippage_bps.equity_oneway)
        defaults: dict[str, object] = {
            "signal_time": None,
            "execution_time": None,
            "execution_delay_business_days": int(self.settings.trading.execution_delay_business_days),
            "expected_trade_return": None,
            "expected_net_return": None,
            "signal_reference_price": None,
            "reference_close": None,
            "reference_volume": None,
            "estimated_dollar_volume": None,
            "adv_dollar_volume": None,
            "fillable_notional_cap": None,
            "target_notional": None,
            "intended_order_quantity": 0.0,
            "executed_notional": 0.0,
            "filled_quantity": 0.0,
            "unfilled_quantity": 0.0,
            "unfilled_notional": 0.0,
            "execution_price": None,
            "participation_rate": 0.0,
            "fill_rate": 0.0,
            "partial_fill": False,
            "missed_trade": False,
            "liquidity_capped": False,
            "liquidity_blocked": False,
            "gap_slippage_bps": 0.0,
            "round_trip_fee_bps": round_trip_fee_bps,
            "round_trip_slippage_bps": round_trip_slippage_bps,
            "round_trip_cost_bps": round_trip_fee_bps + round_trip_slippage_bps,
            "execution_cost_drag": 0.0,
            "execution_cost_drag_bps": 0.0,
            "execution_reason": "legacy_row",
            "asset_return": None,
            "gross_trade_return": None,
            "net_trade_return": None,
            "realized_return": None,
            "realized_direction": None,
            "hit": None,
            "settled_at": None,
        }
        for column in LEDGER_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = defaults.get(column)
        normalized["forecast_date"] = pd.to_datetime(normalized["forecast_date"], utc=True)
        normalized["signal_time"] = pd.to_datetime(normalized["signal_time"], utc=True, errors="coerce")
        normalized["execution_time"] = pd.to_datetime(normalized["execution_time"], utc=True, errors="coerce")
        normalized["settlement_date"] = pd.to_datetime(normalized["settlement_date"], utc=True)
        normalized["settled_at"] = pd.to_datetime(normalized["settled_at"], utc=True, errors="coerce")
        normalized["created_at"] = pd.to_datetime(normalized["created_at"], utc=True, errors="coerce")
        normalized["stale_data_flag"] = normalized["stale_data_flag"].fillna(False).astype(bool)
        normalized["should_retrain"] = normalized["should_retrain"].fillna(False).astype(bool)
        normalized["partial_fill"] = normalized["partial_fill"].fillna(False).astype(bool)
        normalized["missed_trade"] = normalized["missed_trade"].fillna(False).astype(bool)
        normalized["liquidity_capped"] = normalized["liquidity_capped"].fillna(False).astype(bool)
        normalized["liquidity_blocked"] = normalized["liquidity_blocked"].fillna(False).astype(bool)
        normalized["execution_delay_business_days"] = normalized["execution_delay_business_days"].fillna(
            self.settings.trading.execution_delay_business_days
        ).astype(int)
        return normalized.loc[:, LEDGER_COLUMNS].copy()

    def _load_trade_ledger(self) -> pd.DataFrame:
        path = self.store.resolve(LEDGER_PATH)
        if not path.exists():
            return _empty_trade_ledger()
        try:
            ledger = self.store.read_frame(LEDGER_PATH)
        except (OSError, ValueError) as exc:
            logger.warning("Paper trading ledger at '%s' could not be read and will be reinitialized: %s", path, exc)
            return _empty_trade_ledger()
        if ledger.empty:
            return _empty_trade_ledger()
        return self._normalize_ledger(ledger)

    def _append_retraining_policy_history(
        self,
        *,
        forecast_output: dict[str, object],
        backtest_result: dict[str, object],
        risk_review: dict[str, object],
        source_metadata: dict[str, object],
        generated_at: pd.Timestamp,
    ) -> None:
        retraining_monitor = cast(dict[str, object], risk_review.get("retraining_monitor", {}))
        regime_summary = cast(dict[str, object], backtest_result.get("regime_monitor", {}))
        pbo_summary = cast(
            dict[str, object],
            cast(dict[str, object], backtest_result.get("cpcv", {})).get("pbo_summary", {}),
        )
        predictions = cast(list[dict[str, object]], forecast_output.get("predictions", []))
        tickers = [str(item.get("ticker", "")) for item in predictions if item.get("ticker")]
        entry = build_retraining_history_entry(
            as_of_date=generated_at.date().isoformat(),
            retraining_monitor=retraining_monitor,
            regime_summary=regime_summary,
            tickers=tickers,
            source_mode=self.settings.data.source_mode,
            dummy_mode=cast(str | None, source_metadata.get("dummy_mode")),
            pbo=cast(float | None, backtest_result.get("pbo")),
            pbo_summary=pbo_summary,
            created_at=generated_at.isoformat(),
        )
        self.retraining_ledger_service.append_entry(entry)

    def _prepare_market_context(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "ticker",
            "date",
            "open",
            "close",
            "volume",
            "dollar_volume",
            "adv_dollar_volume",
            "volatility",
            "bid_ask_spread_bps",
        ]
        if feature_frame.empty or "ticker" not in feature_frame.columns or "date" not in feature_frame.columns:
            return pd.DataFrame(columns=columns)
        market = feature_frame.copy()
        market["date"] = pd.to_datetime(market["date"], utc=True).dt.normalize()
        if "close" not in market.columns:
            market["close"] = 0.0
        market["close"] = pd.to_numeric(market["close"], errors="coerce").fillna(0.0)
        if "open" not in market.columns:
            market["open"] = market["close"]
        else:
            market["open"] = pd.to_numeric(market["open"], errors="coerce").fillna(market["close"])
        if "volume" not in market.columns:
            market["volume"] = 0.0
        market["volume"] = pd.to_numeric(market["volume"], errors="coerce").fillna(0.0)
        market = market.loc[:, ["ticker", "date", "open", "close", "volume"]]
        market = market.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
        market["dollar_volume"] = market["close"] * market["volume"]
        lookback = max(int(self.settings.trading.adv_lookback_days), 1)
        market["adv_dollar_volume"] = market.groupby("ticker")["dollar_volume"].transform(
            lambda series: series.rolling(window=lookback, min_periods=1).mean()
        )
        market["daily_return"] = market.groupby("ticker")["close"].pct_change().fillna(0.0)
        market["volatility"] = market.groupby("ticker")["daily_return"].transform(
            lambda series: series.rolling(window=lookback, min_periods=2).std()
        ).fillna(0.0)
        market["bid_ask_spread_bps"] = 2.0 * float(self.settings.trading.slippage_bps.equity_oneway)
        return market.loc[:, columns].copy()

    def _build_market_snapshot(self, market_context: pd.DataFrame, forecast_date: pd.Timestamp) -> dict[str, dict[str, float]]:
        if market_context.empty:
            return {}
        eligible = market_context.loc[market_context["date"] <= forecast_date].copy()
        if eligible.empty:
            return {}
        latest_rows = eligible.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        snapshot: dict[str, dict[str, float]] = {}
        for _, row in latest_rows.iterrows():
            ticker = cast(str, row["ticker"])
            snapshot[ticker] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "dollar_volume": float(row["dollar_volume"]),
                "adv_dollar_volume": float(row["adv_dollar_volume"]),
                "volatility": float(row["volatility"]),
                "bid_ask_spread_bps": float(row["bid_ask_spread_bps"]),
            }
        return snapshot

    def _build_market_lookup(self, market_context: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
        if market_context.empty:
            return {}
        lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
        for _, row in market_context.iterrows():
            lookup[(cast(str, row["ticker"]), _to_utc_timestamp(row["date"]).normalize())] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "dollar_volume": float(row["dollar_volume"]),
                "adv_dollar_volume": float(row["adv_dollar_volume"]),
                "volatility": float(row["volatility"]),
                "bid_ask_spread_bps": float(row["bid_ask_spread_bps"]),
            }
        return lookup

    def _settle_pending_trades(
        self,
        ledger: pd.DataFrame,
        market_context: pd.DataFrame,
        current_forecast_date: pd.Timestamp,
        generated_at: pd.Timestamp,
    ) -> int:
        if ledger.empty:
            return 0
        market_lookup = self._build_market_lookup(market_context)
        settled = 0
        for index, row in ledger.loc[ledger["status"] == "PENDING"].iterrows():
            settlement_date = _to_utc_timestamp(row["settlement_date"]).normalize()
            if settlement_date > current_forecast_date:
                continue
            market_row = market_lookup.get((cast(str, row["ticker"]), settlement_date))
            if market_row is None:
                continue
            execution_price = _optional_float(row["execution_price"]) or 0.0
            settlement_close = float(market_row["close"])
            if execution_price <= 0 or settlement_close <= 0:
                continue
            asset_return = (settlement_close / execution_price) - 1.0
            trade_direction = cast(str, row["direction"])
            gross_trade_return = _directional_return(trade_direction, asset_return)
            round_trip_cost_rate = float(cast(float, row["round_trip_cost_bps"]) or 0.0) / 10_000.0
            net_trade_return = gross_trade_return - round_trip_cost_rate
            realized_direction = _realized_direction(asset_return, self.settings.model_settings.direction_threshold)
            ledger.at[index, "asset_return"] = asset_return
            ledger.at[index, "gross_trade_return"] = gross_trade_return
            ledger.at[index, "net_trade_return"] = net_trade_return
            ledger.at[index, "realized_return"] = net_trade_return
            ledger.at[index, "realized_direction"] = realized_direction
            ledger.at[index, "hit"] = bool(trade_direction == realized_direction)
            ledger.at[index, "status"] = "SETTLED"
            ledger.at[index, "settled_at"] = generated_at
            settled += 1
        return settled

    def _simulate_fill(
        self,
        *,
        row: pd.Series,
        market_row: dict[str, float] | None,
        signal_reference_price: float,
        intended_quantity: float,
        target_notional: float,
        min_trade_notional: float,
        min_daily_dollar_volume: float,
        max_participation_rate: float,
    ) -> dict[str, object]:
        if not self.settings.execution.enabled or self.order_simulator is None:
            return self._simulate_fill_legacy(
                row=row,
                market_row=market_row,
                signal_reference_price=signal_reference_price,
                intended_quantity=intended_quantity,
                target_notional=target_notional,
                min_trade_notional=min_trade_notional,
                min_daily_dollar_volume=min_daily_dollar_volume,
                max_participation_rate=max_participation_rate,
            )
        return self._simulate_fill_detailed(
            row=row,
            market_row=market_row,
            signal_reference_price=signal_reference_price,
            intended_quantity=intended_quantity,
            target_notional=target_notional,
            min_trade_notional=min_trade_notional,
            min_daily_dollar_volume=min_daily_dollar_volume,
            max_participation_rate=max_participation_rate,
        )

    def _simulate_fill_legacy(
        self,
        *,
        row: pd.Series,
        market_row: dict[str, float] | None,
        signal_reference_price: float,
        intended_quantity: float,
        target_notional: float,
        min_trade_notional: float,
        min_daily_dollar_volume: float,
        max_participation_rate: float,
    ) -> dict[str, object]:
        if market_row is None:
            return {
                "status": "SKIPPED",
                "missed_trade": intended_quantity > 0,
                "unfilled_quantity": intended_quantity,
                "unfilled_notional": target_notional,
                "liquidity_blocked": True,
                "execution_reason": "missing_execution_market_data",
            }

        execution_price = float(market_row["open"])
        adv_dollar_volume = float(market_row["adv_dollar_volume"])
        fillable_notional_cap = adv_dollar_volume * max_participation_rate if adv_dollar_volume > 0 else 0.0
        filled_quantity_cap = fillable_notional_cap / execution_price if execution_price > 0 else 0.0

        blocked = adv_dollar_volume < min_daily_dollar_volume or fillable_notional_cap < min_trade_notional
        filled_quantity = min(intended_quantity, filled_quantity_cap) if not blocked else 0.0
        filled_notional = filled_quantity * execution_price if execution_price > 0 else 0.0

        if blocked or execution_price <= 0 or intended_quantity <= 0 or filled_notional < min_trade_notional:
            return {
                "status": "SKIPPED",
                "execution_price": execution_price if execution_price > 0 else None,
                "adv_dollar_volume": adv_dollar_volume,
                "fillable_notional_cap": fillable_notional_cap,
                "filled_quantity": 0.0,
                "executed_notional": 0.0,
                "unfilled_quantity": intended_quantity,
                "unfilled_notional": target_notional,
                "fill_rate": 0.0,
                "missed_trade": intended_quantity > 0,
                "partial_fill": False,
                "participation_rate": 0.0,
                "liquidity_blocked": True,
                "execution_reason": "insufficient_liquidity",
            }

        unfilled_quantity = max(intended_quantity - filled_quantity, 0.0)
        unfilled_notional = max(target_notional - filled_notional, 0.0)
        fill_rate = filled_quantity / intended_quantity if intended_quantity > 0 else 0.0
        participation_rate = filled_notional / adv_dollar_volume if adv_dollar_volume > 0 else 0.0
        partial_fill = filled_quantity + 1e-9 < intended_quantity
        gap_slippage_bps = _adverse_gap_slippage_bps(cast(str, row["direction"]), signal_reference_price, execution_price)
        round_trip_cost_bps = float(cast(float, row["round_trip_cost_bps"]) or 0.0)
        execution_cost_drag_bps = round_trip_cost_bps + gap_slippage_bps
        return {
            "status": "PENDING",
            "execution_price": execution_price,
            "adv_dollar_volume": adv_dollar_volume,
            "fillable_notional_cap": fillable_notional_cap,
            "filled_quantity": filled_quantity,
            "executed_notional": filled_notional,
            "unfilled_quantity": unfilled_quantity,
            "unfilled_notional": unfilled_notional,
            "fill_rate": fill_rate,
            "partial_fill": partial_fill,
            "missed_trade": False,
            "participation_rate": participation_rate,
            "liquidity_capped": partial_fill,
            "liquidity_blocked": False,
            "gap_slippage_bps": gap_slippage_bps,
            "execution_cost_drag_bps": execution_cost_drag_bps,
            "execution_cost_drag": execution_cost_drag_bps / 10_000.0,
            "execution_reason": "partial_fill_participation_cap" if partial_fill else "filled_next_open",
        }

    def _simulate_fill_detailed(
        self,
        *,
        row: pd.Series,
        market_row: dict[str, float] | None,
        signal_reference_price: float,
        intended_quantity: float,
        target_notional: float,
        min_trade_notional: float,
        min_daily_dollar_volume: float,
        max_participation_rate: float,
    ) -> dict[str, object]:
        if market_row is None:
            return {
                "status": "SKIPPED",
                "missed_trade": intended_quantity > 0,
                "unfilled_quantity": intended_quantity,
                "unfilled_notional": target_notional,
                "liquidity_blocked": True,
                "execution_reason": "missing_execution_market_data",
            }

        reference_price = float(market_row["open"])
        adv_dollar_volume = float(market_row["adv_dollar_volume"])
        fillable_notional_cap = adv_dollar_volume * max_participation_rate if adv_dollar_volume > 0 else 0.0
        blocked = adv_dollar_volume < min_daily_dollar_volume or fillable_notional_cap < min_trade_notional
        simulated_notional = min(target_notional, fillable_notional_cap)

        if blocked or reference_price <= 0 or intended_quantity <= 0 or simulated_notional < min_trade_notional:
            return {
                "status": "SKIPPED",
                "execution_price": reference_price if reference_price > 0 else None,
                "adv_dollar_volume": adv_dollar_volume,
                "fillable_notional_cap": fillable_notional_cap,
                "filled_quantity": 0.0,
                "executed_notional": 0.0,
                "unfilled_quantity": intended_quantity,
                "unfilled_notional": target_notional,
                "fill_rate": 0.0,
                "missed_trade": intended_quantity > 0,
                "partial_fill": False,
                "participation_rate": 0.0,
                "liquidity_blocked": True,
                "execution_reason": "insufficient_liquidity",
            }

        side: Literal["buy", "sell"] = "buy" if cast(str, row["direction"]) == "UP" else "sell"
        assert self.order_simulator is not None
        result = self.order_simulator.simulate_execution(
            Order(
                ticker=cast(str, row["ticker"]),
                side=side,
                notional=simulated_notional,
                order_type="market",
                urgency=1.0,
            ),
            MarketState(
                reference_price=reference_price,
                adv_dollar_volume=adv_dollar_volume,
                volatility=float(market_row.get("volatility", 0.0)),
                bid_ask_spread_bps=float(market_row.get("bid_ask_spread_bps", 0.0)),
            ),
        )
        filled_notional = float(result.filled_notional)
        filled_quantity = float(result.filled_quantity)
        execution_price = float(result.average_fill_price) if result.average_fill_price > 0 else reference_price
        unfilled_quantity = max(intended_quantity - filled_quantity, 0.0)
        unfilled_notional = max(target_notional - filled_notional, 0.0)
        fill_rate = filled_quantity / intended_quantity if intended_quantity > 0 else 0.0
        partial_fill = filled_quantity + 1e-9 < intended_quantity
        participation_rate = filled_notional / adv_dollar_volume if adv_dollar_volume > 0 else 0.0
        gap_slippage_bps = _adverse_gap_slippage_bps(cast(str, row["direction"]), signal_reference_price, execution_price)
        one_way_fee_bps = float(self.settings.trading.cost_bps.equity_oneway)
        round_trip_fee_bps = 2.0 * one_way_fee_bps
        round_trip_slippage_bps = 2.0 * max(float(result.total_cost_bps) - one_way_fee_bps, 0.0)
        round_trip_cost_bps = round_trip_fee_bps + round_trip_slippage_bps
        execution_cost_drag_bps = round_trip_cost_bps + gap_slippage_bps
        return {
            "status": "PENDING",
            "execution_price": execution_price,
            "adv_dollar_volume": adv_dollar_volume,
            "fillable_notional_cap": fillable_notional_cap,
            "filled_quantity": filled_quantity,
            "executed_notional": filled_notional,
            "unfilled_quantity": unfilled_quantity,
            "unfilled_notional": unfilled_notional,
            "fill_rate": fill_rate,
            "partial_fill": partial_fill,
            "missed_trade": False,
            "participation_rate": participation_rate,
            "liquidity_capped": partial_fill,
            "liquidity_blocked": False,
            "gap_slippage_bps": gap_slippage_bps,
            "round_trip_fee_bps": round_trip_fee_bps,
            "round_trip_slippage_bps": round_trip_slippage_bps,
            "round_trip_cost_bps": round_trip_cost_bps,
            "execution_cost_drag_bps": execution_cost_drag_bps,
            "execution_cost_drag": execution_cost_drag_bps / 10_000.0,
            "execution_reason": "execution_model_partial_fill" if partial_fill else "execution_model_market_fill",
        }

    def _execute_queued_trades(
        self,
        ledger: pd.DataFrame,
        market_context: pd.DataFrame,
        current_forecast_date: pd.Timestamp,
        generated_at: pd.Timestamp,
    ) -> pd.DataFrame:
        del generated_at
        if ledger.empty:
            return _empty_trade_ledger()
        market_lookup = self._build_market_lookup(market_context)
        executed_rows: list[dict[str, object]] = []
        min_trade_notional = float(self.settings.trading.min_trade_notional)
        min_daily_dollar_volume = float(self.settings.trading.min_daily_dollar_volume)
        max_participation_rate = float(self.settings.trading.max_participation_rate)

        for index, row in ledger.loc[ledger["status"] == "QUEUED"].iterrows():
            execution_time = row["execution_time"]
            execution_date = (
                _to_utc_timestamp(execution_time).normalize()
                if execution_time is not None and not pd.isna(execution_time)
                else _execution_date(
                    _to_utc_timestamp(row["forecast_date"]).normalize(),
                    self.settings.trading.execution_delay_business_days,
                )
            )
            if execution_date > current_forecast_date:
                continue

            ticker = cast(str, row["ticker"])
            market_row = market_lookup.get((ticker, execution_date))
            signal_reference_price = _optional_float(row["signal_reference_price"]) or 0.0
            intended_quantity = float(cast(float, row["intended_order_quantity"]) or 0.0)
            target_notional = float(cast(float, row["target_notional"]) or 0.0)
            fill_result = self._simulate_fill(
                row=row,
                market_row=market_row,
                signal_reference_price=signal_reference_price,
                intended_quantity=intended_quantity,
                target_notional=target_notional,
                min_trade_notional=min_trade_notional,
                min_daily_dollar_volume=min_daily_dollar_volume,
                max_participation_rate=max_participation_rate,
            )
            for column, value in fill_result.items():
                ledger.at[index, column] = value
            executed_rows.append(ledger.loc[index].to_dict())

        if not executed_rows:
            return _empty_trade_ledger()
        return self._normalize_ledger(pd.DataFrame(executed_rows))

    def _build_new_trade_rows(
        self,
        forecast_output: dict[str, object],
        risk_review: dict[str, object],
        source_metadata: dict[str, object],
        market_snapshot: dict[str, dict[str, float]],
        forecast_date: pd.Timestamp,
        generated_at: pd.Timestamp,
    ) -> list[dict[str, object]]:
        predictions = cast(list[dict[str, Any]], forecast_output["predictions"])
        trades: list[dict[str, object]] = []
        one_way_fee_bps = float(self.settings.trading.cost_bps.equity_oneway)
        one_way_slippage_bps = float(self.settings.trading.slippage_bps.equity_oneway)
        round_trip_fee_bps = 2 * one_way_fee_bps
        round_trip_slippage_bps = 2 * one_way_slippage_bps
        round_trip_cost_bps = round_trip_fee_bps + round_trip_slippage_bps
        base_target_notional = float(self.settings.trading.paper_capital * self.settings.trading.max_position_pct)
        execution_date = _execution_date(forecast_date, self.settings.trading.execution_delay_business_days)
        execution_time = _execution_timestamp(execution_date, self.settings.trading.execution_time_utc)

        for prediction in predictions:
            ticker = cast(str, prediction["ticker"])
            direction = cast(str, prediction["direction"])
            market_row = market_snapshot.get(
                ticker,
                {"close": 0.0, "open": 0.0, "volume": 0.0, "dollar_volume": 0.0, "adv_dollar_volume": 0.0},
            )
            signal_reference_price = float(market_row["close"])
            reference_volume = float(market_row["volume"])
            estimated_dollar_volume = float(market_row["dollar_volume"])
            adv_dollar_volume = float(market_row["adv_dollar_volume"])
            fillable_notional_cap = adv_dollar_volume * float(self.settings.trading.max_participation_rate)
            target_notional = base_target_notional if direction != "FLAT" else 0.0
            intended_order_quantity = (target_notional / signal_reference_price) if signal_reference_price > 0 else 0.0
            expected_return = prediction.get("expected_return")
            expected_trade_return = (
                _directional_return(direction, float(expected_return))
                if expected_return is not None and pd.notna(expected_return)
                else None
            )
            expected_net_return = (
                float(expected_trade_return) - round_trip_cost_bps / 10_000.0
                if expected_trade_return is not None
                else None
            )
            settlement_date = max(
                _settlement_date(forecast_date, self.settings.data.horizon_days),
                execution_date,
            )
            status = "QUEUED"
            execution_reason = "queued_for_next_open"
            missed_trade = False
            liquidity_blocked = False
            if direction == "FLAT":
                status = "SKIPPED"
                execution_reason = "flat_signal"
            elif signal_reference_price <= 0:
                status = "SKIPPED"
                execution_reason = "missing_signal_market_data"
                missed_trade = True
                liquidity_blocked = True
            trades.append(
                {
                    "trade_id": str(uuid4()),
                    "forecast_id": cast(str, forecast_output["forecast_id"]),
                    "forecast_date": forecast_date,
                    "signal_time": generated_at,
                    "execution_time": execution_time,
                    "execution_delay_business_days": self.settings.trading.execution_delay_business_days,
                    "settlement_date": settlement_date,
                    "week_id": _week_id(forecast_date),
                    "ticker": ticker,
                    "direction": direction,
                    "confidence": prediction["confidence"],
                    "prob_up": float(prediction["probabilities"]["UP"]),
                    "prob_down": float(prediction["probabilities"]["DOWN"]),
                    "prob_flat": float(prediction["probabilities"]["FLAT"]),
                    "expected_return": expected_return,
                    "expected_trade_return": expected_trade_return,
                    "expected_net_return": expected_net_return,
                    "predicted_volatility": prediction.get("predicted_volatility"),
                    "stale_data_flag": bool(prediction["stale_data_flag"]),
                    "regime": cast(str, forecast_output["regime"]),
                    "approval": cast(str, risk_review["approval"]),
                    "dummy_mode": source_metadata.get("dummy_mode"),
                    "used_source": cast(str, source_metadata.get("used_source", "unknown")),
                    "should_retrain": bool(
                        cast(dict[str, object], risk_review.get("retraining_monitor", {})).get("should_retrain", False)
                    ),
                    "signal_reference_price": signal_reference_price if signal_reference_price > 0 else None,
                    "reference_close": signal_reference_price if signal_reference_price > 0 else None,
                    "reference_volume": reference_volume,
                    "estimated_dollar_volume": estimated_dollar_volume,
                    "adv_dollar_volume": adv_dollar_volume,
                    "fillable_notional_cap": fillable_notional_cap,
                    "target_notional": target_notional,
                    "intended_order_quantity": intended_order_quantity,
                    "executed_notional": 0.0,
                    "filled_quantity": 0.0,
                    "unfilled_quantity": intended_order_quantity,
                    "unfilled_notional": target_notional,
                    "execution_price": None,
                    "participation_rate": 0.0,
                    "fill_rate": 0.0,
                    "partial_fill": False,
                    "missed_trade": missed_trade,
                    "liquidity_capped": False,
                    "liquidity_blocked": liquidity_blocked,
                    "gap_slippage_bps": 0.0,
                    "round_trip_fee_bps": round_trip_fee_bps if direction != "FLAT" else 0.0,
                    "round_trip_slippage_bps": round_trip_slippage_bps if direction != "FLAT" else 0.0,
                    "round_trip_cost_bps": round_trip_cost_bps if direction != "FLAT" else 0.0,
                    "execution_cost_drag": 0.0,
                    "execution_cost_drag_bps": 0.0,
                    "execution_reason": execution_reason,
                    "status": status,
                    "asset_return": None,
                    "gross_trade_return": None,
                    "net_trade_return": None,
                    "realized_return": None,
                    "realized_direction": None,
                    "hit": None,
                    "settled_at": None,
                    "created_at": generated_at,
                }
            )
        return trades

    def _build_execution_diagnostics(self, frame: pd.DataFrame) -> dict[str, object]:
        if frame.empty:
            return _empty_execution_diagnostics()
        diagnostics_frame = frame.copy()
        if "status" in diagnostics_frame.columns:
            diagnostics_frame = diagnostics_frame.loc[diagnostics_frame["status"] != "QUEUED"].copy()
        if diagnostics_frame.empty:
            return _empty_execution_diagnostics()
        intended_frame = diagnostics_frame.loc[diagnostics_frame["intended_order_quantity"].astype(float) > 0].copy()
        if intended_frame.empty:
            return _empty_execution_diagnostics()
        attempted_trades = int(len(intended_frame))
        filled_frame = intended_frame.loc[intended_frame["filled_quantity"].astype(float) > 0].copy()
        intended_quantity_total = float(intended_frame["intended_order_quantity"].astype(float).sum())
        filled_quantity_total = float(intended_frame["filled_quantity"].astype(float).sum())
        intended_notional_total = float(intended_frame["target_notional"].astype(float).sum())
        executed_notional_total = float(intended_frame["executed_notional"].astype(float).sum())
        weighted_notional = filled_frame["executed_notional"].astype(float) if not filled_frame.empty else pd.Series(dtype=float)
        execution_cost_drag_bps = (
            float((filled_frame["execution_cost_drag_bps"].astype(float) * weighted_notional).sum() / weighted_notional.sum())
            if not filled_frame.empty and float(weighted_notional.sum()) > 0
            else 0.0
        )
        gap_slippage_bps = (
            float((filled_frame["gap_slippage_bps"].astype(float) * weighted_notional).sum() / weighted_notional.sum())
            if not filled_frame.empty and float(weighted_notional.sum()) > 0
            else 0.0
        )
        partial_fills = int(intended_frame["partial_fill"].sum())
        missed_trades = int(intended_frame["missed_trade"].sum())
        return {
            "intended_trades": int(len(intended_frame)),
            "attempted_trades": attempted_trades,
            "filled_trades": int(len(filled_frame)),
            "partial_fills": partial_fills,
            "missed_trades": missed_trades,
            "fill_rate": (filled_quantity_total / intended_quantity_total) if intended_quantity_total > 0 else 0.0,
            "partial_fill_rate": (partial_fills / attempted_trades) if attempted_trades > 0 else 0.0,
            "missed_trade_rate": (missed_trades / attempted_trades) if attempted_trades > 0 else 0.0,
            "realized_vs_intended_exposure": (
                executed_notional_total / intended_notional_total if intended_notional_total > 0 else 0.0
            ),
            "execution_cost_drag": execution_cost_drag_bps / 10_000.0,
            "execution_cost_drag_bps": execution_cost_drag_bps,
            "gap_slippage_bps": gap_slippage_bps,
        }

    def _serialize_trade_for_batch(self, row: dict[str, object]) -> dict[str, object]:
        return {
            "trade_id": str(row["trade_id"]),
            "ticker": cast(str, row["ticker"]),
            "direction": cast(str, row["direction"]),
            "confidence": cast(str, row["confidence"]),
            "prob_up": float(cast(float, row["prob_up"])),
            "prob_down": float(cast(float, row["prob_down"])),
            "prob_flat": float(cast(float, row["prob_flat"])),
            "expected_return": _optional_float(row.get("expected_return")),
            "expected_trade_return": _optional_float(row.get("expected_trade_return")),
            "expected_net_return": _optional_float(row.get("expected_net_return")),
            "predicted_volatility": _optional_float(row.get("predicted_volatility")),
            "stale_data_flag": bool(row["stale_data_flag"]),
            "signal_time": _optional_timestamp(row.get("signal_time")),
            "execution_time": _optional_timestamp(row.get("execution_time")),
            "signal_reference_price": _optional_float(row.get("signal_reference_price")),
            "execution_price": _optional_float(row.get("execution_price")),
            "status": cast(str, row["status"]),
            "target_notional": _optional_float(row.get("target_notional")),
            "executed_notional": _optional_float(row.get("executed_notional")),
            "intended_order_quantity": _optional_float(row.get("intended_order_quantity")),
            "filled_quantity": _optional_float(row.get("filled_quantity")),
            "unfilled_quantity": _optional_float(row.get("unfilled_quantity")),
            "unfilled_notional": _optional_float(row.get("unfilled_notional")),
            "fillable_notional_cap": _optional_float(row.get("fillable_notional_cap")),
            "adv_dollar_volume": _optional_float(row.get("adv_dollar_volume")),
            "participation_rate": float(cast(float, row["participation_rate"]) or 0.0),
            "fill_rate": float(cast(float, row["fill_rate"]) or 0.0),
            "partial_fill": bool(row["partial_fill"]),
            "missed_trade": bool(row["missed_trade"]),
            "liquidity_capped": bool(row["liquidity_capped"]),
            "liquidity_blocked": bool(row["liquidity_blocked"]),
            "round_trip_cost_bps": float(cast(float, row["round_trip_cost_bps"]) or 0.0),
            "gap_slippage_bps": float(cast(float, row["gap_slippage_bps"]) or 0.0),
            "execution_cost_drag": float(cast(float, row["execution_cost_drag"]) or 0.0),
            "execution_reason": cast(str, row["execution_reason"]),
        }

    def _build_batch_log(
        self,
        forecast_output: dict[str, object],
        risk_review: dict[str, object],
        generated_at: pd.Timestamp,
        batch_rows: list[dict[str, object]],
        executed_rows: pd.DataFrame,
        settled_count: int,
        ledger: pd.DataFrame,
    ) -> dict[str, object]:
        batch_id = str(uuid4())
        week_id = _week_id(generated_at)
        batch_frame = self._normalize_ledger(pd.DataFrame(batch_rows)) if batch_rows else _empty_trade_ledger()
        execution_diagnostics = self._build_execution_diagnostics(executed_rows)
        executed_fills = executed_rows.loc[executed_rows["filled_quantity"].astype(float) > 0].copy() if not executed_rows.empty else executed_rows
        return {
            "batch_id": batch_id,
            "forecast_id": cast(str, forecast_output["forecast_id"]),
            "created_at": generated_at.isoformat(),
            "forecast_date": generated_at.date().isoformat(),
            "week_id": week_id,
            "ledger_path": str(LEDGER_PATH.as_posix()),
            "approval": cast(str, risk_review["approval"]),
            "should_retrain": bool(cast(dict[str, object], risk_review.get("retraining_monitor", {})).get("should_retrain", False)),
            "metrics": {
                "new_trades": len(batch_rows),
                "queued_trades_this_run": int((batch_frame["status"] == "QUEUED").sum()) if not batch_frame.empty else 0,
                "executed_trades_this_run": int(cast(int, execution_diagnostics["filled_trades"])),
                "settled_trades_this_run": settled_count,
                "pending_trades_total": int((ledger["status"] == "PENDING").sum()),
                "queued_trades_total": int((ledger["status"] == "QUEUED").sum()),
                "settled_trades_total": int((ledger["status"] == "SETTLED").sum()),
                "skipped_trades_total": int((ledger["status"] == "SKIPPED").sum()),
                "liquidity_capped_trades_this_run": int(executed_rows["liquidity_capped"].sum()) if not executed_rows.empty else 0,
                "liquidity_blocked_trades_this_run": int(executed_rows["liquidity_blocked"].sum()) if not executed_rows.empty else 0,
                "avg_participation_rate": (
                    float(executed_fills["participation_rate"].astype(float).mean()) if not executed_fills.empty else 0.0
                ),
                "avg_round_trip_cost_bps": (
                    float(executed_fills["round_trip_cost_bps"].astype(float).mean()) if not executed_fills.empty else 0.0
                ),
            },
            "execution_diagnostics": execution_diagnostics,
            "trades": [self._serialize_trade_for_batch(row) for row in batch_rows],
        }

    def _build_weekly_review(self, ledger: pd.DataFrame, forecast_date: pd.Timestamp) -> dict[str, object]:
        week_id = _week_id(forecast_date)
        current_week = ledger.loc[ledger["week_id"] == week_id].copy() if not ledger.empty else _empty_trade_ledger()
        settled = current_week.loc[current_week["status"] == "SETTLED"].copy() if not current_week.empty else current_week
        approval_breakdown = current_week["approval"].value_counts().to_dict() if not current_week.empty else {}
        retraining_batches = (
            int(current_week.loc[current_week["should_retrain"].astype(bool), "forecast_id"].nunique()) if not current_week.empty else 0
        )
        execution_diagnostics = self._build_execution_diagnostics(current_week)
        week_start = (forecast_date - pd.Timedelta(days=forecast_date.dayofweek)).date().isoformat()
        week_end = (forecast_date - pd.Timedelta(days=forecast_date.dayofweek) + pd.Timedelta(days=4)).date().isoformat()
        return {
            "review_id": str(uuid4()),
            "week_id": week_id,
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "window_start": week_start,
            "window_end": week_end,
            "summary": (
                f"Weekly paper trading review for {week_id}: "
                f"{int(len(current_week))} trades, {int(len(settled))} settled, "
                f"{int((current_week['status'] == 'PENDING').sum()) if not current_week.empty else 0} pending, "
                f"{int((current_week['status'] == 'QUEUED').sum()) if not current_week.empty else 0} queued, "
                f"{int((current_week['status'] == 'SKIPPED').sum()) if not current_week.empty else 0} skipped, "
                f"fill_rate={float(cast(float, execution_diagnostics['fill_rate'])):.2f}, "
                f"partial_fill_rate={float(cast(float, execution_diagnostics['partial_fill_rate'])):.2f}, "
                f"missed_trade_rate={float(cast(float, execution_diagnostics['missed_trade_rate'])):.2f}."
            ),
            "metrics": {
                "total_trades": int(len(current_week)),
                "settled_trades": int(len(settled)),
                "pending_trades": int((current_week["status"] == "PENDING").sum()) if not current_week.empty else 0,
                "queued_trades": int((current_week["status"] == "QUEUED").sum()) if not current_week.empty else 0,
                "skipped_trades": int((current_week["status"] == "SKIPPED").sum()) if not current_week.empty else 0,
                "hit_rate": float(settled["hit"].astype(float).mean()) if not settled.empty else None,
                "avg_realized_return": float(settled["realized_return"].astype(float).mean()) if not settled.empty else None,
                "avg_gross_trade_return": (
                    float(settled["gross_trade_return"].astype(float).mean()) if not settled.empty else None
                ),
                "avg_net_trade_return": (
                    float(settled["net_trade_return"].astype(float).mean()) if not settled.empty else None
                ),
                "avg_expected_return": (
                    float(current_week["expected_return"].dropna().astype(float).mean()) if not current_week.empty else None
                ),
                "avg_expected_net_return": (
                    float(current_week["expected_net_return"].dropna().astype(float).mean()) if not current_week.empty else None
                ),
                "avg_round_trip_cost_bps": (
                    float(current_week["round_trip_cost_bps"].astype(float).mean()) if not current_week.empty else 0.0
                ),
                "avg_participation_rate": (
                    float(current_week["participation_rate"].astype(float).mean()) if not current_week.empty else 0.0
                ),
                "liquidity_capped_trades": int(current_week["liquidity_capped"].sum()) if not current_week.empty else 0,
                "liquidity_blocked_trades": int(current_week["liquidity_blocked"].sum()) if not current_week.empty else 0,
                "retraining_batches": retraining_batches,
            },
            "execution_diagnostics": execution_diagnostics,
            "approval_breakdown": approval_breakdown,
        }

    def _build_retraining_event(
        self,
        forecast_output: dict[str, object],
        backtest_result: dict[str, object],
        risk_review: dict[str, object],
        generated_at: pd.Timestamp,
    ) -> dict[str, object] | None:
        retraining_monitor = cast(dict[str, object], risk_review.get("retraining_monitor", {}))
        if not bool(retraining_monitor.get("should_retrain", False)):
            return None
        regime_monitor = cast(dict[str, object], backtest_result.get("regime_monitor", {}))
        drift_monitor = cast(dict[str, object], backtest_result.get("drift_monitor", {}))
        pbo_summary = cast(
            dict[str, object],
            cast(dict[str, object], backtest_result.get("cpcv", {})).get("pbo_summary", {}),
        )
        aggregate_metrics = cast(dict[str, float], backtest_result.get("aggregate_metrics", {}))
        return {
            "event_id": str(uuid4()),
            "forecast_id": cast(str, forecast_output["forecast_id"]),
            "created_at": generated_at.isoformat(),
            "week_id": _week_id(generated_at),
            "approval": cast(str, risk_review["approval"]),
            "regime": cast(str, regime_monitor.get("current_regime", forecast_output.get("regime", "unknown"))),
            "max_psi": cast(float, drift_monitor.get("max_psi", 0.0) or 0.0),
            "ece_mean": float(aggregate_metrics.get("ece_mean", 0.0) or 0.0),
            "pbo": backtest_result.get("pbo"),
            "pbo_label": pbo_summary.get("label"),
            "trigger_count": cast(int, retraining_monitor.get("trigger_count", 0) or 0),
            "triggers": cast(list[dict[str, object]], retraining_monitor.get("triggers", [])),
            "action": "retrain_before_next_production_style_run",
        }

