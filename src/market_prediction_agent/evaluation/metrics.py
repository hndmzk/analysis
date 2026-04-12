from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


RETURN_REGRESSION_METRIC_KEYS = (
    "return_ic",
    "return_rank_ic",
    "return_mae",
    "return_rmse",
)
VOLATILITY_REGRESSION_METRIC_KEYS = (
    "vol_mae",
    "vol_mape",
    "vol_rmse",
)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def multiclass_log_loss(y_true: np.ndarray, probability_matrix: np.ndarray) -> float:
    clipped = np.clip(probability_matrix, 1e-9, 1 - 1e-9)
    chosen = clipped[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(chosen)))


def multiclass_brier_score(y_true: np.ndarray, probability_matrix: np.ndarray) -> float:
    target = np.zeros_like(probability_matrix)
    target[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probability_matrix - target) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probability_matrix: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    confidence = probability_matrix.max(axis=1)
    predicted = probability_matrix.argmax(axis=1)
    correctness = (predicted == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        bucket_accuracy = float(correctness[mask].mean())
        bucket_confidence = float(confidence[mask].mean())
        ece += abs(bucket_accuracy - bucket_confidence) * (float(mask.sum()) / len(y_true))
    return float(ece)


def calibration_gap(y_true: np.ndarray, probability_matrix: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    confidence = probability_matrix.max(axis=1)
    predicted = probability_matrix.argmax(axis=1)
    accuracy = float(np.mean(predicted == y_true))
    return float(abs(float(confidence.mean()) - accuracy))


def correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) < 2 or np.std(values_a) == 0 or np.std(values_b) == 0:
        return 0.0
    return float(np.corrcoef(values_a, values_b)[0, 1])


def rank_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    rank_a = pd.Series(values_a).rank(method="average").to_numpy()
    rank_b = pd.Series(values_b).rank(method="average").to_numpy()
    return correlation(rank_a, rank_b)


def _pairwise_numeric_frame(predictions: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:
    frame = predictions[[column_a, column_b]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame


def _mae(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) == 0:
        return 0.0
    return float(np.mean(np.abs(values_a - values_b)))


def _rmse(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) == 0:
        return 0.0
    return float(np.sqrt(np.mean((values_a - values_b) ** 2)))


def return_regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    ic_frame = _pairwise_numeric_frame(predictions, "expected_return", "future_simple_return")
    error_frame = _pairwise_numeric_frame(predictions, "expected_return", "target_return")
    expected_for_ic = ic_frame["expected_return"].to_numpy(dtype=float)
    realized_return = ic_frame["future_simple_return"].to_numpy(dtype=float)
    expected_for_error = error_frame["expected_return"].to_numpy(dtype=float)
    target_return = error_frame["target_return"].to_numpy(dtype=float)
    return {
        "return_ic": correlation(expected_for_ic, realized_return),
        "return_rank_ic": rank_correlation(expected_for_ic, realized_return),
        "return_mae": _mae(expected_for_error, target_return),
        "return_rmse": _rmse(expected_for_error, target_return),
    }


def volatility_regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    frame = _pairwise_numeric_frame(predictions, "predicted_volatility", "future_volatility_20d")
    predicted_volatility = frame["predicted_volatility"].to_numpy(dtype=float)
    realized_volatility = frame["future_volatility_20d"].to_numpy(dtype=float)
    non_zero_mask = realized_volatility != 0.0
    if non_zero_mask.any():
        vol_mape = float(
            np.mean(np.abs(predicted_volatility[non_zero_mask] - realized_volatility[non_zero_mask]) / np.abs(realized_volatility[non_zero_mask]))
        )
    else:
        vol_mape = 0.0
    return {
        "vol_mae": _mae(predicted_volatility, realized_volatility),
        "vol_mape": vol_mape,
        "vol_rmse": _rmse(predicted_volatility, realized_volatility),
    }


def compute_fold_metrics(predictions: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    y_true = predictions["direction_label"].to_numpy(dtype=int)
    probability_matrix = predictions[["prob_down", "prob_flat", "prob_up"]].to_numpy(dtype=float)
    predicted_class = probability_matrix.argmax(axis=1)
    metrics = {
        "hit_rate": hit_rate(y_true, predicted_class),
        "log_loss": multiclass_log_loss(y_true, probability_matrix),
        "brier_score": multiclass_brier_score(y_true, probability_matrix),
        "ece": expected_calibration_error(y_true, probability_matrix),
        "calibration_gap": calibration_gap(y_true, probability_matrix),
        "ic": correlation(predictions["signal"].to_numpy(dtype=float), predictions["future_simple_return"].to_numpy(dtype=float)),
        "rank_ic": rank_correlation(predictions["signal"].to_numpy(dtype=float), predictions["future_simple_return"].to_numpy(dtype=float)),
    }
    baseline_probability = predictions[["baseline_prob_down", "baseline_prob_flat", "baseline_prob_up"]].to_numpy(dtype=float)
    model_loss = -np.log(np.clip(probability_matrix[np.arange(len(y_true)), y_true], 1e-9, 1.0))
    baseline_loss = -np.log(np.clip(baseline_probability[np.arange(len(y_true)), y_true], 1e-9, 1.0))
    loss_diff = baseline_loss - model_loss
    return metrics, loss_diff


def diebold_mariano(loss_differentials: np.ndarray) -> dict[str, float | str]:
    if len(loss_differentials) < 2:
        return {"vs_baseline": "class_prior", "dm_statistic": 0.0, "p_value": 1.0}
    mean = float(np.mean(loss_differentials))
    std = float(np.std(loss_differentials, ddof=1))
    if std == 0:
        return {"vs_baseline": "class_prior", "dm_statistic": 0.0, "p_value": 1.0}
    statistic = mean / (std / math.sqrt(len(loss_differentials)))
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(statistic) / math.sqrt(2))))
    return {"vs_baseline": "class_prior", "dm_statistic": float(statistic), "p_value": float(p_value)}


@dataclass(slots=True)
class DailySelection:
    date: pd.Timestamp
    target_weights: dict[str, float]
    executed_weights: dict[str, float]
    signed_selection: set[str]
    long_count: int
    short_count: int
    proposed_turnover: float
    target_turnover: float
    rebalance_thinned: bool
    liquidity_multiplier: float


def _bucket_size(universe_size: int, fraction: float) -> int:
    if universe_size <= 0 or fraction <= 0:
        return 0
    return max(1, int(round(universe_size * fraction)))


def _build_weights(longs: list[str], shorts: list[str]) -> tuple[dict[str, float], set[str]]:
    weights: dict[str, float] = {}
    signed_selection: set[str] = set()
    if longs:
        weight = 1.0 / len(longs)
        for ticker_name in longs:
            weights[ticker_name] = weights.get(ticker_name, 0.0) + weight
            signed_selection.add(f"L:{ticker_name}")
    if shorts:
        weight = -1.0 / len(shorts)
        for ticker_name in shorts:
            weights[ticker_name] = weights.get(ticker_name, 0.0) + weight
            signed_selection.add(f"S:{ticker_name}")
    return weights, signed_selection


def _candidate_pool(
    group: pd.DataFrame,
    *,
    side: str,
    strategy_name: str,
    probability_threshold: float,
    minimum_edge: float,
) -> pd.DataFrame:
    pool = group.copy()
    if strategy_name in {"classified_two_sided", "classified_directional"}:
        if side == "long":
            pool = pool.loc[(pool["direction"] == "UP") & (pool["prob_up"] >= probability_threshold)].copy()
            pool["edge"] = pool["prob_up"] - pool[["prob_down", "prob_flat"]].max(axis=1)
            return pool.sort_values(["edge", "signal"], ascending=[False, False]).reset_index(drop=True)
        pool = pool.loc[(pool["direction"] == "DOWN") & (pool["prob_down"] >= probability_threshold)].copy()
        pool["edge"] = pool["prob_down"] - pool[["prob_up", "prob_flat"]].max(axis=1)
        return pool.sort_values(["edge", "signal"], ascending=[False, True]).reset_index(drop=True)
    if strategy_name == "rank_long_short":
        if side == "long":
            pool["edge"] = pool.get("signal", 0.0)
            pool = pool.loc[pool["edge"] >= minimum_edge].copy()
            return pool.sort_values(["edge", "signal"], ascending=[False, False]).reset_index(drop=True)
        pool["edge"] = np.maximum(pool.get("prob_down", 0.0) - pool.get("prob_up", 0.0), -pool["signal"])
        pool = pool.loc[pool["edge"] >= minimum_edge].copy()
        return pool.sort_values(["edge", "signal"], ascending=[False, True]).reset_index(drop=True)
    if strategy_name == "rank_long_only":
        if side == "short":
            return group.iloc[0:0].copy()
        pool["edge"] = pool.get("signal", 0.0)
        pool = pool.loc[pool["edge"] >= minimum_edge].copy()
        return pool.sort_values(["edge", "signal"], ascending=[False, False]).reset_index(drop=True)
    raise ValueError(f"Unsupported portfolio strategy: {strategy_name}")


def _select_side_candidates(
    pool: pd.DataFrame,
    *,
    bucket_size: int,
    side_prefix: str,
    previous_selected: set[str],
    cooldown_until: dict[str, int],
    day_index: int,
    minimum_edge: float,
    bucket_hysteresis: float,
    hysteresis_edge_buffer: float,
) -> list[str]:
    if bucket_size <= 0 or pool.empty:
        return []
    selected: list[str] = []
    retention_limit = bucket_size + int(math.ceil(bucket_size * max(bucket_hysteresis, 0.0)))
    rank_map = {str(row["ticker"]): index for index, (_, row) in enumerate(pool.iterrows())}
    edge_map = {str(row["ticker"]): float(row["edge"]) for _, row in pool.iterrows()}
    for ticker in previous_selected:
        if ticker not in rank_map:
            continue
        if len(selected) >= bucket_size:
            break
        rank = rank_map[ticker]
        edge_value = edge_map[ticker]
        retain_on_edge = hysteresis_edge_buffer > 0 and edge_value >= max(0.0, minimum_edge - hysteresis_edge_buffer)
        if rank < retention_limit or retain_on_edge:
            selected.append(ticker)
    for _, row in pool.iterrows():
        ticker = str(row["ticker"])
        token = f"{side_prefix}:{ticker}"
        if len(selected) >= bucket_size:
            break
        if ticker in selected or float(row["edge"]) < minimum_edge:
            continue
        if cooldown_until.get(token, -1) >= day_index:
            continue
        selected.append(ticker)
    return selected


def _turnover(weights_a: dict[str, float], weights_b: dict[str, float]) -> float:
    symbols = set(weights_a) | set(weights_b)
    return 0.5 * sum(abs(weights_a.get(symbol, 0.0) - weights_b.get(symbol, 0.0)) for symbol in symbols)


def _selection_from_weights(weights: dict[str, float], tolerance: float = 1e-6) -> set[str]:
    selected: set[str] = set()
    for ticker, weight in weights.items():
        if weight > tolerance:
            selected.add(f"L:{ticker}")
        elif weight < -tolerance:
            selected.add(f"S:{ticker}")
    return selected


def _extract_previous_tickers(selection: set[str], prefix: str) -> set[str]:
    return {item.split(":", maxsplit=1)[1] for item in selection if item.startswith(f"{prefix}:")}


def _liquidity_multiplier(
    group: pd.DataFrame,
    selected_tickers: list[str],
    *,
    participation_volume_floor: float,
    participation_volume_ceiling: float,
) -> float:
    if "volume_ratio_20d" not in group.columns or not selected_tickers:
        return 1.0
    values = (
        group.loc[group["ticker"].astype(str).isin(selected_tickers), "volume_ratio_20d"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    if values.empty:
        return 1.0
    return float(np.clip(values.median(), participation_volume_floor, participation_volume_ceiling))


def _apply_turnover_budget(
    desired_weights: dict[str, float],
    previous_weights: dict[str, float],
    *,
    max_turnover_per_day: float,
    liquidity_multiplier: float,
) -> tuple[dict[str, float], float, float, bool]:
    proposed_turnover = _turnover(desired_weights, previous_weights)
    effective_cap = max_turnover_per_day * liquidity_multiplier if max_turnover_per_day > 0 else 0.0
    if proposed_turnover == 0.0 or effective_cap <= 0.0 or proposed_turnover <= effective_cap:
        return desired_weights, proposed_turnover, proposed_turnover, False
    scale = effective_cap / proposed_turnover
    throttled = {
        symbol: previous_weights.get(symbol, 0.0) + scale * (desired_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
        for symbol in set(previous_weights) | set(desired_weights)
    }
    throttled = {symbol: weight for symbol, weight in throttled.items() if abs(weight) > 1e-4}
    return throttled, proposed_turnover, _turnover(throttled, previous_weights), True


def _annualized_return(series: np.ndarray) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean() * 252)


def _annualized_volatility(series: np.ndarray) -> float:
    if len(series) < 2:
        return 0.0
    return float(series.std(ddof=1) * np.sqrt(252))


def _information_ratio(series: np.ndarray) -> float:
    annual_return = _annualized_return(series)
    annual_volatility = _annualized_volatility(series)
    if annual_volatility == 0.0:
        return 0.0
    return float(annual_return / annual_volatility)


def _selection_stability(selections: list[DailySelection]) -> float:
    scores: list[float] = []
    previous: set[str] | None = None
    for selection in selections:
        current = selection.signed_selection
        if not current:
            continue
        if previous is not None:
            union = previous | current
            if union:
                scores.append(len(previous & current) / len(union))
        previous = current
    return float(np.mean(scores)) if scores else 0.0


def compute_cost_adjusted_metrics(
    predictions: pd.DataFrame,
    one_way_cost_bps: float,
    bucket_fraction: float = 0.2,
    probability_threshold: float = 0.4,
    *,
    top_bucket_fraction: float | None = None,
    bottom_bucket_fraction: float | None = None,
    strategy_name: str = "classified_two_sided",
    holding_days: int = 1,
    min_edge: float = 0.0,
    bucket_hysteresis: float = 0.0,
    hysteresis_edge_buffer: float = 0.0,
    reentry_cooldown_days: int = 0,
    max_turnover_per_day: float = 0.0,
    participation_volume_floor: float = 1.0,
    participation_volume_ceiling: float = 1.0,
) -> dict[str, float]:
    if predictions.empty:
        return {
            "information_ratio": 0.0,
            "gross_information_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "annual_return": 0.0,
            "gross_annual_return": 0.0,
            "annual_volatility": 0.0,
            "avg_daily_turnover": 0.0,
            "total_cost_bps": 0.0,
            "active_days_ratio": 0.0,
            "signal_days_ratio": 0.0,
            "two_sided_signal_days_ratio": 0.0,
            "one_sided_signal_days_ratio": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_selected_names": 0.0,
            "selection_stability": 0.0,
            "cost_drag_annual_return": 0.0,
            "gross_positive_day_ratio": 0.0,
            "net_positive_day_ratio": 0.0,
            "rebalance_thinned_days_ratio": 0.0,
            "avg_liquidity_multiplier": 1.0,
            "avg_target_turnover": 0.0,
        }

    effective_top_fraction = float(top_bucket_fraction if top_bucket_fraction is not None else bucket_fraction)
    effective_bottom_fraction = float(bottom_bucket_fraction if bottom_bucket_fraction is not None else bucket_fraction)
    effective_holding_days = max(1, int(holding_days))
    ordered_predictions = predictions.sort_values(["date", "ticker"]).copy()
    selections: list[DailySelection] = []
    previous_target_weights: dict[str, float] = {}
    previous_target_selection: set[str] = set()
    cooldown_until: dict[str, int] = {}
    grouped_frames = {
        pd.Timestamp(date): group.set_index("ticker")
        for date, group in ordered_predictions.groupby("date", sort=True)
    }
    for day_index, (date, group) in enumerate(ordered_predictions.groupby("date", sort=True)):
        top_bucket = _bucket_size(len(group), effective_top_fraction)
        bottom_bucket = _bucket_size(len(group), effective_bottom_fraction)
        previous_longs = _extract_previous_tickers(previous_target_selection, "L")
        previous_shorts = _extract_previous_tickers(previous_target_selection, "S")
        long_pool = _candidate_pool(
            group,
            side="long",
            strategy_name=strategy_name,
            probability_threshold=probability_threshold,
            minimum_edge=min_edge,
        )
        short_pool = _candidate_pool(
            group,
            side="short",
            strategy_name=strategy_name,
            probability_threshold=probability_threshold,
            minimum_edge=min_edge,
        )
        selected_longs = _select_side_candidates(
            long_pool,
            bucket_size=top_bucket,
            side_prefix="L",
            previous_selected=previous_longs,
            cooldown_until=cooldown_until,
            day_index=day_index,
            minimum_edge=min_edge,
            bucket_hysteresis=bucket_hysteresis,
            hysteresis_edge_buffer=hysteresis_edge_buffer,
        )
        selected_shorts = _select_side_candidates(
            short_pool,
            bucket_size=bottom_bucket,
            side_prefix="S",
            previous_selected=previous_shorts,
            cooldown_until=cooldown_until,
            day_index=day_index,
            minimum_edge=min_edge,
            bucket_hysteresis=bucket_hysteresis,
            hysteresis_edge_buffer=hysteresis_edge_buffer,
        )
        if strategy_name == "classified_two_sided" and (not selected_longs or not selected_shorts):
            selected_longs = []
            selected_shorts = []
        desired_weights, _ = _build_weights(selected_longs, selected_shorts)
        liquidity_multiplier = _liquidity_multiplier(
            group,
            selected_longs + selected_shorts,
            participation_volume_floor=participation_volume_floor,
            participation_volume_ceiling=participation_volume_ceiling,
        )
        target_weights, proposed_turnover, target_turnover, rebalance_thinned = _apply_turnover_budget(
            desired_weights,
            previous_target_weights,
            max_turnover_per_day=max_turnover_per_day,
            liquidity_multiplier=liquidity_multiplier,
        )
        target_selection = _selection_from_weights(target_weights)
        for token in previous_target_selection - target_selection:
            cooldown_until[token] = day_index + max(0, reentry_cooldown_days)
        previous_target_weights = target_weights
        previous_target_selection = target_selection
        selections.append(
            DailySelection(
                date=pd.Timestamp(date),
                target_weights=target_weights,
                executed_weights={},
                signed_selection=target_selection,
                long_count=len(selected_longs),
                short_count=len(selected_shorts),
                proposed_turnover=proposed_turnover,
                target_turnover=target_turnover,
                rebalance_thinned=rebalance_thinned,
                liquidity_multiplier=liquidity_multiplier,
            )
        )

    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnover_values: list[float] = []
    target_turnover_values: list[float] = []
    gross_exposures: list[float] = []
    selected_names: list[int] = []
    liquidity_multipliers: list[float] = []
    previous_weights: dict[str, float] = {}
    active_days = 0
    signal_days = 0
    two_sided_signal_days = 0
    one_sided_signal_days = 0
    rebalance_thinned_days = 0

    for index, selection in enumerate(selections):
        if selection.target_weights:
            signal_days += 1
            selected_names.append(selection.long_count + selection.short_count)
        if selection.long_count > 0 and selection.short_count > 0:
            two_sided_signal_days += 1
        elif selection.long_count > 0 or selection.short_count > 0:
            one_sided_signal_days += 1
        aggregated_weights: dict[str, float] = {}
        for prior_selection in selections[max(0, index - effective_holding_days + 1) : index + 1]:
            if not prior_selection.target_weights:
                continue
            for ticker, weight in prior_selection.target_weights.items():
                aggregated_weights[ticker] = aggregated_weights.get(ticker, 0.0) + (weight / effective_holding_days)
        selection.executed_weights = aggregated_weights
        selection.signed_selection = _selection_from_weights(aggregated_weights)
        gross_exposure = float(sum(abs(weight) for weight in aggregated_weights.values()))
        gross_exposures.append(gross_exposure)
        if aggregated_weights:
            active_days += 1
        frame = grouped_frames[selection.date]
        gross_return = 0.0
        for ticker, weight in aggregated_weights.items():
            if ticker not in frame.index:
                continue
            gross_return += weight * float(frame.loc[ticker, "future_simple_return"])
        turnover = _turnover(aggregated_weights, previous_weights)
        cost = turnover * (one_way_cost_bps / 10_000.0)
        gross_returns.append(gross_return)
        net_returns.append(gross_return - cost)
        turnover_values.append(turnover)
        target_turnover_values.append(selection.target_turnover)
        liquidity_multipliers.append(selection.liquidity_multiplier)
        rebalance_thinned_days += int(selection.rebalance_thinned)
        previous_weights = aggregated_weights

    gross_series = np.array(gross_returns, dtype=float)
    net_series = np.array(net_returns, dtype=float)
    if len(net_series) == 0:
        return {
            "information_ratio": 0.0,
            "gross_information_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "annual_return": 0.0,
            "gross_annual_return": 0.0,
            "annual_volatility": 0.0,
            "avg_daily_turnover": 0.0,
            "total_cost_bps": 0.0,
            "active_days_ratio": 0.0,
            "signal_days_ratio": 0.0,
            "two_sided_signal_days_ratio": 0.0,
            "one_sided_signal_days_ratio": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_selected_names": 0.0,
            "selection_stability": 0.0,
            "cost_drag_annual_return": 0.0,
            "gross_positive_day_ratio": 0.0,
            "net_positive_day_ratio": 0.0,
            "rebalance_thinned_days_ratio": 0.0,
            "avg_liquidity_multiplier": 1.0,
            "avg_target_turnover": 0.0,
        }

    annual_return = _annualized_return(net_series)
    gross_annual_return = _annualized_return(gross_series)
    annual_volatility = _annualized_volatility(net_series)
    downside = np.where(net_series < 0, net_series, 0.0)
    downside_std = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(net_series) > 1 else 0.0
    cumulative = np.cumprod(1 + net_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1.0
    max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0
    sharpe = _information_ratio(net_series)
    sortino = annual_return / downside_std if downside_std else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown else 0.0
    total_cost_bps = float(np.sum(np.array(turnover_values, dtype=float) * (one_way_cost_bps)))
    return {
        "information_ratio": sharpe,
        "gross_information_ratio": _information_ratio(gross_series),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "annual_return": annual_return,
        "gross_annual_return": gross_annual_return,
        "annual_volatility": annual_volatility,
        "avg_daily_turnover": float(np.mean(turnover_values)) if turnover_values else 0.0,
        "total_cost_bps": total_cost_bps,
        "active_days_ratio": float(active_days / len(selections)) if selections else 0.0,
        "signal_days_ratio": float(signal_days / len(selections)) if selections else 0.0,
        "two_sided_signal_days_ratio": float(two_sided_signal_days / len(selections)) if selections else 0.0,
        "one_sided_signal_days_ratio": float(one_sided_signal_days / len(selections)) if selections else 0.0,
        "avg_gross_exposure": float(np.mean(gross_exposures)) if gross_exposures else 0.0,
        "avg_selected_names": float(np.mean(selected_names)) if selected_names else 0.0,
        "selection_stability": _selection_stability(selections),
        "cost_drag_annual_return": float(gross_annual_return - annual_return),
        "gross_positive_day_ratio": float(np.mean(gross_series > 0)) if len(gross_series) else 0.0,
        "net_positive_day_ratio": float(np.mean(net_series > 0)) if len(net_series) else 0.0,
        "rebalance_thinned_days_ratio": float(rebalance_thinned_days / len(selections)) if selections else 0.0,
        "avg_liquidity_multiplier": float(np.mean(liquidity_multipliers)) if liquidity_multipliers else 1.0,
        "avg_target_turnover": float(np.mean(target_turnover_values)) if target_turnover_values else 0.0,
    }
