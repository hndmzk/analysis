from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field
import yaml

from market_prediction_agent.utils.paths import resolve_repo_path


ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

ModelT = TypeVar("ModelT", bound=BaseModel)


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class AppConfig(FrozenConfigModel):
    name: str
    version: str
    seed: int
    log_level: str


class DataThresholdConfig(FrozenConfigModel):
    ohlcv: int
    macro: int
    news: int
    fundamentals: int
    sector: int


class PublicDataConfig(FrozenConfigModel):
    cache_path: str
    snapshot_path: str
    cache_ttl_hours: int
    retry_count: int
    retry_backoff_seconds: float


class JpEquityConfig(FrozenConfigModel):
    enabled: bool
    universe: str
    source: str
    tickers_file: str


class LearnedWeightingConfig(FrozenConfigModel):
    regularization_lambda: float
    min_samples: int
    lookback_days: int
    target: str
    min_weight: float
    fallback_mode: str


class DataConfig(FrozenConfigModel):
    source_mode: str
    dummy_mode: str
    primary_source: str
    fallback_source: str
    crypto_source: str
    crypto_tickers: list[str]
    crypto_enabled: bool
    macro_source: str
    news_source: str
    news_secondary_sources: list[str]
    news_fallback_source: str
    news_source_weights: dict[str, float]
    news_session_weights: dict[str, float]
    news_weighting_mode: str
    learned_weighting: LearnedWeightingConfig
    jp_equity: JpEquityConfig
    fundamentals_source: str
    fundamentals_fallback_source: str
    sector_source: str
    universe: str
    default_tickers: list[str]
    storage_path: str
    ohlcv_granularity: str
    forecast_horizon: str
    horizon_days: int
    dummy_ticker_count: int
    dummy_days: int
    stale_threshold_hours: DataThresholdConfig
    public_data: PublicDataConfig
    macro_series: list[str]


class WalkForwardConfig(FrozenConfigModel):
    initial_train_days: int
    eval_days: int
    step_days: int
    embargo_days: int


class CalibrationConfig(FrozenConfigModel):
    method: str
    fraction: float
    min_days: int
    ece_warning: float
    calibration_gap_warning: float


class LightGBMConfig(FrozenConfigModel):
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    reg_lambda: float
    max_shap_samples: int


class XGBoostConfig(FrozenConfigModel):
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    max_shap_samples: int


class LSTMConfig(FrozenConfigModel):
    hidden_size: int
    num_layers: int
    dropout: float
    sequence_length: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    patience: int


class TransformerConfig(FrozenConfigModel):
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    patch_length: int
    sequence_length: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    patience: int


class CPCVConfig(FrozenConfigModel):
    group_count: int
    test_groups: int
    max_splits: int
    strategy_names: list[str]
    portfolio_thresholds: list[float]
    top_bucket_fractions: list[float]
    bottom_bucket_fractions: list[float]
    holding_days: list[int]
    threshold_cluster_tolerance: float
    bucket_cluster_tolerance: float
    holding_days_cluster_tolerance: int


class PortfolioRuleConfig(FrozenConfigModel):
    strategy_name: str
    probability_threshold: float
    top_bucket_fraction: float
    bottom_bucket_fraction: float
    holding_days: int
    min_edge: float
    bucket_hysteresis: float
    hysteresis_edge_buffer: float
    reentry_cooldown_days: int
    max_turnover_per_day: float
    participation_volume_floor: float
    participation_volume_ceiling: float


class HMMConfig(FrozenConfigModel):
    n_states: int
    min_history_days: int
    regime_shift_lookback_days: int


class RetrainingConfig(FrozenConfigModel):
    regime_shift_requires_retrain: bool
    transition_regime_requires_retrain: bool
    high_vol_requires_retrain: bool
    regime_shift_requires_confirmation: bool
    regime_shift_standalone_state_probability: float
    regime_shift_standalone_transition_rate: float
    transition_regime_state_probability_threshold: float
    transition_regime_transition_rate_threshold: float
    transition_regime_unstable_state_probability_threshold: float
    transition_regime_unstable_transition_rate_threshold: float
    transition_regime_immediate_state_probability_threshold: float
    transition_regime_immediate_transition_rate_threshold: float
    transition_regime_min_persistent_runs: int
    transition_regime_min_persistent_span_business_days: int
    transition_regime_persistence_business_days: int
    regime_shift_requires_drift_confirmation: bool
    pbo_warning: float
    pbo_min_persistent_observations: int
    pbo_min_persistent_span_business_days: int
    pbo_persistence_business_days: int
    pbo_immediate_threshold: float
    pbo_close_competition_margin_threshold: float
    pbo_close_competition_ratio_threshold: float
    pbo_competition_dominance_threshold: float
    drift_min_persistent_features: int
    drift_min_persistent_families: int
    drift_family_weight_threshold: float
    drift_immediate_weight_threshold: float
    calibration_min_persistent_folds: int
    calibration_min_fold_breach_ratio: float
    calibration_min_persistent_runs: int
    calibration_min_persistent_span_business_days: int
    calibration_persistence_business_days: int
    calibration_immediate_multiplier: float
    cooloff_business_days: int
    cause_cooloff_business_days: int
    cooloff_same_regime_only: bool
    cooloff_override_triggers: list[str]
    drift_threshold_multipliers: dict[str, float]
    drift_immediate_threshold_multipliers: dict[str, float]
    calibration_breach_ratio_multipliers: dict[str, float]
    family_retrain_suppression: dict[str, list[str]]
    stable_transition_family_retrain_suppression: list[str]


class ModelConfig(FrozenConfigModel):
    primary: str
    comparison_models: list[str]
    version: str
    feature_version: str
    direction_threshold: float
    ridge_alpha: float
    portfolio_rule: PortfolioRuleConfig
    walk_forward: WalkForwardConfig
    retrain_interval_days: int
    calibration: CalibrationConfig
    lightgbm: LightGBMConfig
    xgboost: XGBoostConfig
    lstm: LSTMConfig
    transformer: TransformerConfig
    cpcv: CPCVConfig
    hmm: HMMConfig
    retraining: RetrainingConfig


class RiskConfig(FrozenConfigModel):
    psi_warning: float
    psi_critical: float
    max_drawdown_limit: float
    vix_stress_threshold: float
    vix_extreme_threshold: float
    drift: DriftConfig


class DriftFamilyThresholdConfig(FrozenConfigModel):
    warning: float
    critical: float


class DriftFamilyWeightsConfig(FrozenConfigModel):
    price_momentum: float
    volatility: float
    volume: float
    macro: float
    calendar: float


class DriftConfig(FrozenConfigModel):
    bucket_count: int
    price_momentum: DriftFamilyThresholdConfig
    volatility: DriftFamilyThresholdConfig
    volume: DriftFamilyThresholdConfig
    macro: DriftFamilyThresholdConfig
    calendar: DriftFamilyThresholdConfig
    proxy_sensitive_features: list[str]
    family_trigger_weights: DriftFamilyWeightsConfig


class TradingBpsConfig(FrozenConfigModel):
    equity_oneway: float
    crypto_oneway: float


class TradingConfig(FrozenConfigModel):
    enable_live_trading: bool
    max_position_pct: float
    max_daily_orders: int
    paper_capital: float
    execution_delay_business_days: int
    execution_time_utc: str
    adv_lookback_days: int
    min_trade_notional: float
    min_daily_dollar_volume: float
    max_participation_rate: float
    cost_bps: TradingBpsConfig
    slippage_bps: TradingBpsConfig


class MarketImpactConfig(FrozenConfigModel):
    eta: float = 0.142
    gamma: float = 0.314
    max_participation_rate: float = 0.01


class ExecutionConfig(FrozenConfigModel):
    enabled: bool = False
    market_impact: MarketImpactConfig = Field(default_factory=MarketImpactConfig)
    order_types: list[str] = Field(default_factory=lambda: ["market", "limit", "twap"])
    twap_slices: int = 10
    twap_interval_minutes: int = 5


class ApiKeysConfig(FrozenConfigModel):
    polygon: str
    alphavantage: str
    fred: str
    finnhub: str
    coingecko: str
    binance_key: str
    binance_secret: str
    anthropic: str


class Settings(FrozenConfigModel):
    app: AppConfig
    data: DataConfig
    model_settings: ModelConfig = Field(alias="model")
    risk: RiskConfig
    trading: TradingConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    api_keys: ApiKeysConfig


RiskConfig.model_rebuild()
Settings.model_rebuild()


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            return os.getenv(match.group(1), "")

        return ENV_PATTERN.sub(replace, value)
    return value


def _merge_model_data(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _merge_model_data(current, value)
        else:
            merged[key] = value
    return merged


def update_model(model: ModelT, /, **updates: Any) -> ModelT:
    payload = model.model_dump(mode="python", by_alias=False)
    return type(model).model_validate(_merge_model_data(payload, updates))


def update_settings(settings: Settings, /, **updates: Any) -> Settings:
    return update_model(settings, **updates)


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = resolve_repo_path(os.getenv("CONFIG_PATH", str(path or "config/default.yaml")))
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return Settings.model_validate(_expand_env(raw))


def resolve_storage_path(settings: Settings) -> Path:
    return resolve_repo_path(settings.data.storage_path)
