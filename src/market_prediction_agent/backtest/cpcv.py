from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, cast

import numpy as np
import pandas as pd

from market_prediction_agent.config import Settings
from market_prediction_agent.evaluation.metrics import compute_cost_adjusted_metrics, compute_fold_metrics
from market_prediction_agent.evaluation.pbo import interpret_pbo
from market_prediction_agent.models.lightgbm_calibrated import LightGBMCalibratedModel


@dataclass(slots=True)
class CPCVSplit:
    split_id: int
    train_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]


def build_cpcv_splits(
    dates: list[pd.Timestamp],
    group_count: int,
    test_groups: int,
    embargo_days: int,
    max_splits: int,
) -> list[CPCVSplit]:
    date_count = len(dates)
    if date_count < group_count:
        return []
    grouped_indices = [list(chunk) for chunk in np.array_split(np.arange(date_count), group_count) if len(chunk) > 0]
    group_combinations = list(combinations(range(len(grouped_indices)), test_groups))
    if not group_combinations:
        return []
    if len(group_combinations) > max_splits:
        selector = np.linspace(0, len(group_combinations) - 1, max_splits, dtype=int)
        group_combinations = [group_combinations[index] for index in selector.tolist()]
    splits: list[CPCVSplit] = []
    for split_id, test_group_ids in enumerate(group_combinations, start=1):
        test_indices = sorted(index for group_id in test_group_ids for index in grouped_indices[group_id])
        purged_indices = set(test_indices)
        for index in test_indices:
            lower = max(0, index - embargo_days)
            upper = min(date_count - 1, index + embargo_days)
            purged_indices.update(range(lower, upper + 1))
        train_indices = [index for index in range(date_count) if index not in purged_indices]
        if not train_indices or not test_indices:
            continue
        splits.append(
            CPCVSplit(
                split_id=split_id,
                train_dates=[dates[index] for index in train_indices],
                test_dates=[dates[index] for index in test_indices],
            )
        )
    return splits


def _portfolio_candidates(settings: Settings) -> list[dict[str, float | int | str | None]]:
    candidates: list[dict[str, float | int | str | None]] = []
    strategy_names = settings.model_settings.cpcv.strategy_names or [settings.model_settings.portfolio_rule.strategy_name]
    threshold_strategies = {"classified_two_sided", "classified_directional"}
    if len(settings.model_settings.cpcv.top_bucket_fractions) == len(settings.model_settings.cpcv.bottom_bucket_fractions):
        bucket_pairs = list(zip(settings.model_settings.cpcv.top_bucket_fractions, settings.model_settings.cpcv.bottom_bucket_fractions, strict=True))
    else:
        bucket_pairs = list(product(settings.model_settings.cpcv.top_bucket_fractions, settings.model_settings.cpcv.bottom_bucket_fractions))
    for strategy_name in strategy_names:
        threshold_values = settings.model_settings.cpcv.portfolio_thresholds if strategy_name in threshold_strategies else [None]
        for threshold, (top_bucket_fraction, bottom_bucket_fraction), holding_days in product(
            threshold_values,
            bucket_pairs,
            settings.model_settings.cpcv.holding_days,
        ):
            name_parts = [
                strategy_name,
                f"top_{float(top_bucket_fraction):.2f}",
                f"bottom_{float(bottom_bucket_fraction):.2f}",
                f"hold_{int(holding_days)}",
            ]
            if threshold is not None:
                name_parts.insert(1, f"thr_{float(threshold):.2f}")
            candidates.append(
                {
                    "name": "_".join(name_parts),
                    "strategy_name": strategy_name,
                    "probability_threshold": float(threshold) if threshold is not None else None,
                    "top_bucket_fraction": float(top_bucket_fraction),
                    "bottom_bucket_fraction": float(bottom_bucket_fraction),
                    "holding_days": int(holding_days),
                }
            )
    return candidates


def _prepare_predictions(
    model: LightGBMCalibratedModel,
    frame: pd.DataFrame,
    baseline_prior: pd.Series,
) -> pd.DataFrame:
    predictions = model.predict(frame, include_explanations=False)
    predictions["baseline_prob_down"] = baseline_prior.loc[0]
    predictions["baseline_prob_flat"] = baseline_prior.loc[1]
    predictions["baseline_prob_up"] = baseline_prior.loc[2]
    predictions["direction_label"] = frame["direction_label"].to_numpy(dtype=int)
    predictions["future_simple_return"] = frame["future_simple_return"].to_numpy(dtype=float)
    predictions["volume_ratio_20d"] = frame["volume_ratio_20d"].to_numpy(dtype=float)
    return predictions


def _portfolio_controls(settings: Settings) -> dict[str, float | int]:
    rule = settings.model_settings.portfolio_rule
    return {
        "min_edge": float(rule.min_edge),
        "bucket_hysteresis": float(rule.bucket_hysteresis),
        "hysteresis_edge_buffer": float(rule.hysteresis_edge_buffer),
        "reentry_cooldown_days": int(rule.reentry_cooldown_days),
        "max_turnover_per_day": float(rule.max_turnover_per_day),
        "participation_volume_floor": float(rule.participation_volume_floor),
        "participation_volume_ceiling": float(rule.participation_volume_ceiling),
    }


def _rank_percentile(values: list[float], index: int) -> float:
    if not values:
        return 1.0
    order = np.argsort(values)
    rank = int(np.where(order == index)[0][0]) + 1
    return float(rank / len(values))


def _mean_metric(entries: list[dict[str, float]], key: str) -> float:
    values = [float(entry.get(key, 0.0) or 0.0) for entry in entries]
    return float(np.mean(values)) if values else 0.0


def _threshold_key(value: object) -> str:
    if value is None:
        return "none"
    return f"{cast(float, value):.2f}"


def _bucket_key(top_bucket_fraction: float, bottom_bucket_fraction: float) -> str:
    return f"top={float(top_bucket_fraction):.2f}|bottom={float(bottom_bucket_fraction):.2f}"


def _cluster_numeric_values(values: list[float], tolerance: float, *, integer: bool = False) -> dict[float, str]:
    unique_values = sorted({float(value) for value in values})
    if not unique_values:
        return {}
    if tolerance <= 0:
        return {
            value: (str(int(round(value))) if integer else f"{value:.2f}")
            for value in unique_values
        }
    grouped_labels: dict[float, str] = {}
    cluster_members: list[float] = [unique_values[0]]
    for value in unique_values[1:]:
        if value - cluster_members[-1] <= tolerance:
            cluster_members.append(value)
            continue
        label = (
            str(int(round(cluster_members[0])))
            if integer and cluster_members[0] == cluster_members[-1]
            else f"{int(round(cluster_members[0]))}-{int(round(cluster_members[-1]))}"
            if integer
            else f"{cluster_members[0]:.2f}"
            if cluster_members[0] == cluster_members[-1]
            else f"{cluster_members[0]:.2f}-{cluster_members[-1]:.2f}"
        )
        for member in cluster_members:
            grouped_labels[member] = label
        cluster_members = [value]
    label = (
        str(int(round(cluster_members[0])))
        if integer and cluster_members[0] == cluster_members[-1]
        else f"{int(round(cluster_members[0]))}-{int(round(cluster_members[-1]))}"
        if integer
        else f"{cluster_members[0]:.2f}"
        if cluster_members[0] == cluster_members[-1]
        else f"{cluster_members[0]:.2f}-{cluster_members[-1]:.2f}"
    )
    for member in cluster_members:
        grouped_labels[member] = label
    return grouped_labels


def _cluster_bucket_pairs(bucket_pairs: list[tuple[float, float]], tolerance: float) -> dict[tuple[float, float], str]:
    unique_pairs = sorted({(float(top), float(bottom)) for top, bottom in bucket_pairs})
    if not unique_pairs:
        return {}
    if tolerance <= 0:
        return {pair: _bucket_key(pair[0], pair[1]) for pair in unique_pairs}
    labels: dict[tuple[float, float], str] = {}
    cluster_members: list[tuple[float, float]] = [unique_pairs[0]]
    for pair in unique_pairs[1:]:
        previous = cluster_members[-1]
        if max(abs(pair[0] - previous[0]), abs(pair[1] - previous[1])) <= tolerance:
            cluster_members.append(pair)
            continue
        start = cluster_members[0]
        end = cluster_members[-1]
        label = (
            _bucket_key(start[0], start[1])
            if start == end
            else f"top={start[0]:.2f}-{end[0]:.2f}|bottom={start[1]:.2f}-{end[1]:.2f}"
        )
        for member in cluster_members:
            labels[member] = label
        cluster_members = [pair]
    start = cluster_members[0]
    end = cluster_members[-1]
    label = (
        _bucket_key(start[0], start[1])
        if start == end
        else f"top={start[0]:.2f}-{end[0]:.2f}|bottom={start[1]:.2f}-{end[1]:.2f}"
    )
    for member in cluster_members:
        labels[member] = label
    return labels


def _apply_candidate_clusters(
    candidates: list[dict[str, float | int | str | None]],
    settings: Settings,
) -> list[dict[str, float | int | str | None]]:
    threshold_map = _cluster_numeric_values(
        [cast(float, candidate["probability_threshold"]) for candidate in candidates if candidate["probability_threshold"] is not None],
        settings.model_settings.cpcv.threshold_cluster_tolerance,
    )
    bucket_map = _cluster_bucket_pairs(
        [
            (cast(float, candidate["top_bucket_fraction"]), cast(float, candidate["bottom_bucket_fraction"]))
            for candidate in candidates
        ],
        settings.model_settings.cpcv.bucket_cluster_tolerance,
    )
    holding_map = _cluster_numeric_values(
        [float(cast(int, candidate["holding_days"])) for candidate in candidates],
        float(settings.model_settings.cpcv.holding_days_cluster_tolerance),
        integer=True,
    )
    clustered: list[dict[str, float | int | str | None]] = []
    for candidate in candidates:
        threshold_value = candidate["probability_threshold"]
        threshold_cluster_key = "none" if threshold_value is None else threshold_map[float(cast(float, threshold_value))]
        bucket_pair = (
            cast(float, candidate["top_bucket_fraction"]),
            cast(float, candidate["bottom_bucket_fraction"]),
        )
        bucket_cluster_key = bucket_map[bucket_pair]
        holding_days_value = int(cast(int, candidate["holding_days"]))
        holding_days_cluster_key = holding_map[float(holding_days_value)]
        cluster_name = (
            f"{candidate['strategy_name']}_thr_cluster_{threshold_cluster_key}_"
            f"bucket_cluster_{bucket_cluster_key}_hold_cluster_{holding_days_cluster_key}"
        )
        clustered.append(
            {
                **candidate,
                "threshold_cluster_key": threshold_cluster_key,
                "bucket_cluster_key": bucket_cluster_key,
                "holding_days_cluster_key": holding_days_cluster_key,
                "cluster_name": cluster_name,
            }
        )
    return clustered


def _cluster_score_vectors(
    candidates: list[dict[str, float | int | str | None]],
    in_sample_scores: list[float],
    out_sample_scores: list[float],
) -> tuple[list[str], list[float], list[float], dict[str, dict[str, object]]]:
    cluster_meta: dict[str, dict[str, object]] = {}
    for candidate, in_sample_score, out_sample_score in zip(candidates, in_sample_scores, out_sample_scores, strict=True):
        cluster_name = str(candidate["cluster_name"])
        meta = cluster_meta.setdefault(
            cluster_name,
            {
                "cluster_name": cluster_name,
                "strategy_name": str(candidate["strategy_name"]),
                "threshold_cluster_key": str(candidate["threshold_cluster_key"]),
                "bucket_cluster_key": str(candidate["bucket_cluster_key"]),
                "holding_days_cluster_key": str(candidate["holding_days_cluster_key"]),
                "member_names": [],
                "in_sample_scores": [],
                "out_sample_scores": [],
            },
        )
        cast(list[str], meta["member_names"]).append(str(candidate["name"]))
        cast(list[float], meta["in_sample_scores"]).append(float(in_sample_score))
        cast(list[float], meta["out_sample_scores"]).append(float(out_sample_score))
    cluster_names = sorted(cluster_meta)
    cluster_in_sample_scores = [
        float(np.mean(cast(list[float], cluster_meta[name]["in_sample_scores"]))) for name in cluster_names
    ]
    cluster_out_sample_scores = [
        float(np.mean(cast(list[float], cluster_meta[name]["out_sample_scores"]))) for name in cluster_names
    ]
    for name in cluster_names:
        cluster_meta[name]["member_count"] = len(cast(list[str], cluster_meta[name]["member_names"]))
    return cluster_names, cluster_in_sample_scores, cluster_out_sample_scores, cluster_meta


def _group_pbo_contribution(
    split_entries: list[dict[str, object]],
    *,
    entry_key: str,
    label_name: str,
) -> list[dict[str, object]]:
    total_splits = max(len(split_entries), 1)
    total_overfit_splits = max(sum(bool(entry.get("overfit_flag", False)) for entry in split_entries), 1)
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in split_entries:
        label = str(entry.get(entry_key, "unknown"))
        grouped.setdefault(label, []).append(entry)
    summary: list[dict[str, object]] = []
    for label, items in grouped.items():
        overfit_items = [item for item in items if bool(item.get("overfit_flag", False))]
        summary.append(
            {
                label_name: label,
                "selected_split_count": len(items),
                "selected_split_ratio": float(len(items) / total_splits),
                "overfit_split_count": len(overfit_items),
                "overfit_share": float(len(overfit_items) / total_overfit_splits),
                "overfit_rate_when_selected": float(len(overfit_items) / max(len(items), 1)),
                "mean_in_sample_margin": float(
                    np.mean([cast(float, item.get("best_vs_runner_up_in_sample_margin", 0.0)) for item in items])
                ),
                "mean_oos_margin_vs_runner_up": float(
                    np.mean([cast(float, item.get("best_vs_runner_up_oos_margin", 0.0)) for item in items])
                ),
                "mean_best_candidate_test_information_ratio": float(
                    np.mean([cast(float, item.get("best_candidate_test_information_ratio", 0.0)) for item in items])
                ),
            }
        )
    summary.sort(
        key=lambda item: (
            cast(float, item["overfit_share"]),
            cast(float, item["overfit_rate_when_selected"]),
            cast(float, item["mean_best_candidate_test_information_ratio"]),
        ),
        reverse=True,
    )
    return summary


def _aggregate_pbo_diagnostics(split_entries: list[dict[str, object]], settings: Settings) -> dict[str, object]:
    close_margin_threshold = float(settings.model_settings.retraining.pbo_close_competition_margin_threshold)
    close_entries = [
        entry
        for entry in split_entries
        if float(cast(float, entry.get("best_vs_runner_up_in_sample_margin", 0.0))) <= close_margin_threshold
    ]
    family_contribution = _group_pbo_contribution(
        split_entries,
        entry_key="best_strategy_name",
        label_name="strategy_name",
    )
    threshold_contribution = _group_pbo_contribution(
        split_entries,
        entry_key="best_threshold_key",
        label_name="probability_threshold",
    )
    bucket_contribution = _group_pbo_contribution(
        split_entries,
        entry_key="best_bucket_key",
        label_name="bucket_pair",
    )
    holding_days_contribution = _group_pbo_contribution(
        split_entries,
        entry_key="best_holding_days_key",
        label_name="holding_days",
    )
    dominant_entries = [
        ("threshold", threshold_contribution[0]) if threshold_contribution else None,
        ("bucket", bucket_contribution[0]) if bucket_contribution else None,
        ("holding_days", holding_days_contribution[0]) if holding_days_contribution else None,
    ]
    dominant_entries = [entry for entry in dominant_entries if entry is not None]
    dominant_axis = "none"
    dominant_value = "none"
    dominant_overfit_share = 0.0
    if dominant_entries:
        dominant_axis, dominant_item = max(
            cast(list[tuple[str, dict[str, object]]], dominant_entries),
            key=lambda item: cast(float, item[1]["overfit_share"]),
        )
        dominant_overfit_share = float(cast(float, dominant_item["overfit_share"]))
        if dominant_axis == "threshold":
            dominant_value = str(dominant_item["probability_threshold"])
        elif dominant_axis == "bucket":
            dominant_value = str(dominant_item["bucket_pair"])
        else:
            dominant_value = str(dominant_item["holding_days"])
    close_split_ratio = float(len(close_entries) / max(len(split_entries), 1))
    near_candidate_competition = {
        "close_margin_threshold": close_margin_threshold,
        "close_split_count": len(close_entries),
        "close_split_ratio": close_split_ratio,
        "mean_in_sample_margin": float(
            np.mean([cast(float, entry.get("best_vs_runner_up_in_sample_margin", 0.0)) for entry in split_entries])
        ),
        "median_in_sample_margin": float(
            np.median([cast(float, entry.get("best_vs_runner_up_in_sample_margin", 0.0)) for entry in split_entries])
        ),
        "mean_oos_margin_vs_runner_up": float(
            np.mean([cast(float, entry.get("best_vs_runner_up_oos_margin", 0.0)) for entry in split_entries])
        ),
        "median_oos_margin_vs_runner_up": float(
            np.median([cast(float, entry.get("best_vs_runner_up_oos_margin", 0.0)) for entry in split_entries])
        ),
        "same_family_runner_up_ratio": float(
            np.mean([bool(entry.get("same_family_runner_up", False)) for entry in close_entries]) if close_entries else 0.0
        ),
        "threshold_flip_ratio": float(
            np.mean([not bool(entry.get("same_threshold_runner_up", False)) for entry in close_entries])
            if close_entries
            else 0.0
        ),
        "bucket_flip_ratio": float(
            np.mean([not bool(entry.get("same_bucket_runner_up", False)) for entry in close_entries])
            if close_entries
            else 0.0
        ),
        "holding_days_flip_ratio": float(
            np.mean([not bool(entry.get("same_holding_days_runner_up", False)) for entry in close_entries])
            if close_entries
            else 0.0
        ),
        "dominant_axis": dominant_axis,
        "dominant_value": dominant_value,
        "dominant_overfit_share": dominant_overfit_share,
    }
    competition_dominated = bool(
        close_split_ratio >= settings.model_settings.retraining.pbo_close_competition_ratio_threshold
        and dominant_overfit_share >= settings.model_settings.retraining.pbo_competition_dominance_threshold
    )
    near_candidate_competition["competition_dominated"] = competition_dominated
    if competition_dominated:
        competition_reason = (
            f"Close runner-up competition dominated CPCV winners; axis={dominant_axis}, value={dominant_value}, "
            f"overfit_share={dominant_overfit_share:.2f}, close_split_ratio={close_split_ratio:.2f}."
        )
    else:
        competition_reason = (
            f"PBO was not dominated by close candidate competition; axis={dominant_axis}, value={dominant_value}, "
            f"overfit_share={dominant_overfit_share:.2f}, close_split_ratio={close_split_ratio:.2f}."
        )
    near_candidate_competition["competition_reason"] = competition_reason
    return {
        "family_contribution": family_contribution,
        "threshold_contribution": threshold_contribution,
        "bucket_contribution": bucket_contribution,
        "holding_days_contribution": holding_days_contribution,
        "near_candidate_competition": near_candidate_competition,
    }


def _aggregate_candidate_family(candidate_entries: list[dict[str, object]]) -> dict[str, object]:
    by_strategy: dict[str, list[dict[str, object]]] = {}
    by_threshold: dict[str, list[dict[str, object]]] = {}
    by_holding_days: dict[int, list[dict[str, object]]] = {}
    by_bucket_pair: dict[str, list[dict[str, object]]] = {}
    for entry in candidate_entries:
        strategy_name = cast(str, entry["strategy_name"])
        by_strategy.setdefault(strategy_name, []).append(entry)
        threshold = entry.get("probability_threshold")
        threshold_key = "none" if threshold is None else f"{cast(float, threshold):.2f}"
        by_threshold.setdefault(threshold_key, []).append(entry)
        holding_days = cast(int, entry["holding_days"])
        by_holding_days.setdefault(holding_days, []).append(entry)
        bucket_key = f"top={cast(float, entry['top_bucket_fraction']):.2f}|bottom={cast(float, entry['bottom_bucket_fraction']):.2f}"
        by_bucket_pair.setdefault(bucket_key, []).append(entry)

    def summarize(grouped_entries: dict[Any, list[dict[str, object]]], label_name: str) -> list[dict[str, object]]:
        summary: list[dict[str, object]] = []
        for label, items in grouped_entries.items():
            summary.append(
                {
                    label_name: label,
                    "candidate_count": len(items),
                    "mean_oos_information_ratio": float(
                        np.mean([cast(float, item["mean_oos_information_ratio"]) for item in items])
                    ),
                    "mean_oos_gross_information_ratio": float(
                        np.mean([cast(float, item["mean_oos_gross_information_ratio"]) for item in items])
                    ),
                    "mean_oos_annual_return": float(np.mean([cast(float, item["mean_oos_annual_return"]) for item in items])),
                    "mean_oos_cost_drag_annual_return": float(
                        np.mean([cast(float, item["mean_oos_cost_drag_annual_return"]) for item in items])
                    ),
                    "mean_oos_active_days_ratio": float(
                        np.mean([cast(float, item["mean_oos_active_days_ratio"]) for item in items])
                    ),
                    "mean_oos_selection_stability": float(
                        np.mean([cast(float, item["mean_oos_selection_stability"]) for item in items])
                    ),
                    "mean_oos_total_cost_bps": float(
                        np.mean([cast(float, item["mean_oos_total_cost_bps"]) for item in items])
                    ),
                    "mean_oos_rebalance_thinned_days_ratio": float(
                        np.mean([cast(float, item.get("mean_oos_rebalance_thinned_days_ratio", 0.0)) for item in items])
                    ),
                    "mean_oos_avg_target_turnover": float(
                        np.mean([cast(float, item.get("mean_oos_avg_target_turnover", 0.0)) for item in items])
                    ),
                }
            )
        summary.sort(key=lambda item: cast(float, item["mean_oos_information_ratio"]), reverse=True)
        return summary

    return {
        "strategy_analysis": summarize(by_strategy, "strategy_name"),
        "threshold_analysis": summarize(by_threshold, "probability_threshold"),
        "holding_rule_analysis": summarize(by_holding_days, "holding_days"),
        "bucket_analysis": summarize(by_bucket_pair, "bucket_pair"),
    }


def _aggregate_cluster_entries(candidate_entries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_cluster: dict[str, list[dict[str, object]]] = {}
    for entry in candidate_entries:
        by_cluster.setdefault(str(entry["cluster_name"]), []).append(entry)
    summary: list[dict[str, object]] = []
    for cluster_name, items in by_cluster.items():
        summary.append(
            {
                "cluster_name": cluster_name,
                "strategy_name": str(items[0]["strategy_name"]),
                "threshold_cluster_key": str(items[0]["threshold_cluster_key"]),
                "bucket_cluster_key": str(items[0]["bucket_cluster_key"]),
                "holding_days_cluster_key": str(items[0]["holding_days_cluster_key"]),
                "member_count": len(items),
                "member_names": [str(item["name"]) for item in items],
                "mean_oos_information_ratio": float(
                    np.mean([cast(float, item["mean_oos_information_ratio"]) for item in items])
                ),
                "mean_oos_gross_information_ratio": float(
                    np.mean([cast(float, item["mean_oos_gross_information_ratio"]) for item in items])
                ),
                "mean_oos_annual_return": float(np.mean([cast(float, item["mean_oos_annual_return"]) for item in items])),
                "mean_oos_cost_drag_annual_return": float(
                    np.mean([cast(float, item["mean_oos_cost_drag_annual_return"]) for item in items])
                ),
                "mean_oos_active_days_ratio": float(
                    np.mean([cast(float, item["mean_oos_active_days_ratio"]) for item in items])
                ),
                "mean_oos_selection_stability": float(
                    np.mean([cast(float, item["mean_oos_selection_stability"]) for item in items])
                ),
            }
        )
    summary.sort(key=lambda item: cast(float, item["mean_oos_information_ratio"]), reverse=True)
    return summary


def run_cpcv_backtest(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    settings: Settings,
) -> tuple[dict[str, object], float | None]:
    walk = settings.model_settings.walk_forward
    unique_dates = sorted(training_frame["date"].drop_duplicates().tolist())
    splits = build_cpcv_splits(
        dates=unique_dates,
        group_count=settings.model_settings.cpcv.group_count,
        test_groups=settings.model_settings.cpcv.test_groups,
        embargo_days=walk.embargo_days,
        max_splits=settings.model_settings.cpcv.max_splits,
    )
    if not splits:
        return {"splits": [], "aggregate_metrics": {}, "candidate_strategies": [], "pbo_summary": interpret_pbo(None)}, None

    one_way_cost_bps = settings.trading.cost_bps.equity_oneway + settings.trading.slippage_bps.equity_oneway
    controls = _portfolio_controls(settings)
    candidates = _apply_candidate_clusters(_portfolio_candidates(settings), settings)
    split_entries: list[dict[str, object]] = []
    cluster_split_entries: list[dict[str, object]] = []
    split_metrics: list[dict[str, float]] = []
    overfit_flags: list[float] = []
    cluster_overfit_flags: list[float] = []
    candidate_oos_metrics: dict[str, list[dict[str, float]]] = {cast(str, candidate["name"]): [] for candidate in candidates}

    for split in splits:
        train_frame = training_frame.loc[training_frame["date"].isin(split.train_dates)].copy()
        test_frame = training_frame.loc[training_frame["date"].isin(split.test_dates)].copy()
        if train_frame.empty or test_frame.empty:
            continue
        model = LightGBMCalibratedModel(settings=settings, version=settings.model_settings.version)
        model.fit(train_frame, feature_columns)
        class_prior = train_frame["direction_label"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0)
        train_predictions = _prepare_predictions(model, train_frame, class_prior)
        test_predictions = _prepare_predictions(model, test_frame, class_prior)
        metrics, _ = compute_fold_metrics(test_predictions)
        split_metrics.append(metrics)

        in_sample_scores: list[float] = []
        out_sample_scores: list[float] = []
        out_sample_metric_snapshots: list[dict[str, float]] = []
        for candidate in candidates:
            probability_threshold = float(candidate["probability_threshold"] or 0.0)
            train_cost_metrics = compute_cost_adjusted_metrics(
                train_predictions,
                one_way_cost_bps=one_way_cost_bps,
                probability_threshold=probability_threshold,
                top_bucket_fraction=cast(float, candidate["top_bucket_fraction"]),
                bottom_bucket_fraction=cast(float, candidate["bottom_bucket_fraction"]),
                strategy_name=cast(str, candidate["strategy_name"]),
                holding_days=cast(int, candidate["holding_days"]),
                min_edge=cast(float, controls["min_edge"]),
                bucket_hysteresis=cast(float, controls["bucket_hysteresis"]),
                hysteresis_edge_buffer=cast(float, controls["hysteresis_edge_buffer"]),
                reentry_cooldown_days=cast(int, controls["reentry_cooldown_days"]),
                max_turnover_per_day=cast(float, controls["max_turnover_per_day"]),
                participation_volume_floor=cast(float, controls["participation_volume_floor"]),
                participation_volume_ceiling=cast(float, controls["participation_volume_ceiling"]),
            )
            test_cost_metrics = compute_cost_adjusted_metrics(
                test_predictions,
                one_way_cost_bps=one_way_cost_bps,
                probability_threshold=probability_threshold,
                top_bucket_fraction=cast(float, candidate["top_bucket_fraction"]),
                bottom_bucket_fraction=cast(float, candidate["bottom_bucket_fraction"]),
                strategy_name=cast(str, candidate["strategy_name"]),
                holding_days=cast(int, candidate["holding_days"]),
                min_edge=cast(float, controls["min_edge"]),
                bucket_hysteresis=cast(float, controls["bucket_hysteresis"]),
                hysteresis_edge_buffer=cast(float, controls["hysteresis_edge_buffer"]),
                reentry_cooldown_days=cast(int, controls["reentry_cooldown_days"]),
                max_turnover_per_day=cast(float, controls["max_turnover_per_day"]),
                participation_volume_floor=cast(float, controls["participation_volume_floor"]),
                participation_volume_ceiling=cast(float, controls["participation_volume_ceiling"]),
            )
            in_sample_scores.append(train_cost_metrics["information_ratio"])
            out_sample_scores.append(test_cost_metrics["information_ratio"])
            out_sample_metric_snapshots.append(test_cost_metrics)
            candidate_oos_metrics[cast(str, candidate["name"])].append(test_cost_metrics)

        cluster_names, cluster_in_sample_scores, cluster_out_sample_scores, cluster_meta = _cluster_score_vectors(
            candidates,
            in_sample_scores,
            out_sample_scores,
        )
        ranking = np.argsort(in_sample_scores)[::-1]
        best_index = int(ranking[0])
        runner_up_index = int(ranking[1]) if len(ranking) > 1 else best_index
        best_candidate = candidates[best_index]
        runner_up_candidate = candidates[runner_up_index]
        best_in_sample_score = float(in_sample_scores[best_index])
        runner_up_in_sample_score = float(in_sample_scores[runner_up_index])
        best_out_sample_score = float(out_sample_scores[best_index])
        runner_up_out_sample_score = float(out_sample_scores[runner_up_index])
        oos_rank_percentile = _rank_percentile(out_sample_scores, best_index)
        omega = float(np.clip(oos_rank_percentile, 1e-6, 1 - 1e-6))
        logit_omega = float(np.log(omega / (1 - omega)))
        overfit_flag = 1.0 if logit_omega < 0 else 0.0
        overfit_flags.append(overfit_flag)

        cluster_ranking = np.argsort(cluster_in_sample_scores)[::-1]
        cluster_best_index = int(cluster_ranking[0])
        cluster_runner_up_index = int(cluster_ranking[1]) if len(cluster_ranking) > 1 else cluster_best_index
        cluster_best_name = cluster_names[cluster_best_index]
        cluster_runner_up_name = cluster_names[cluster_runner_up_index]
        cluster_best_meta = cluster_meta[cluster_best_name]
        cluster_runner_up_meta = cluster_meta[cluster_runner_up_name]
        cluster_oos_rank_percentile = _rank_percentile(cluster_out_sample_scores, cluster_best_index)
        cluster_omega = float(np.clip(cluster_oos_rank_percentile, 1e-6, 1 - 1e-6))
        cluster_logit_omega = float(np.log(cluster_omega / (1 - cluster_omega)))
        cluster_overfit_flag = 1.0 if cluster_logit_omega < 0 else 0.0
        cluster_overfit_flags.append(cluster_overfit_flag)

        split_entries.append(
            {
                "split_id": split.split_id,
                "train_start": min(split.train_dates).date().isoformat(),
                "train_end": max(split.train_dates).date().isoformat(),
                "test_start": min(split.test_dates).date().isoformat(),
                "test_end": max(split.test_dates).date().isoformat(),
                "n_train": int(len(train_frame)),
                "n_test": int(len(test_frame)),
                "metrics": metrics,
                "best_candidate": best_candidate,
                "runner_up_candidate": runner_up_candidate,
                "best_candidate_in_sample_information_ratio": best_in_sample_score,
                "runner_up_candidate_in_sample_information_ratio": runner_up_in_sample_score,
                "best_candidate_test_information_ratio": best_out_sample_score,
                "runner_up_candidate_test_information_ratio": runner_up_out_sample_score,
                "best_candidate_test_active_days_ratio": out_sample_metric_snapshots[best_index]["active_days_ratio"],
                "best_candidate_test_selection_stability": out_sample_metric_snapshots[best_index]["selection_stability"],
                "best_vs_runner_up_in_sample_margin": float(best_in_sample_score - runner_up_in_sample_score),
                "best_vs_runner_up_oos_margin": float(best_out_sample_score - runner_up_out_sample_score),
                "best_candidate_oos_rank_percentile": oos_rank_percentile,
                "best_candidate_logit_omega": logit_omega,
                "overfit_flag": bool(overfit_flag),
                "best_strategy_name": str(best_candidate["strategy_name"]),
                "best_threshold_key": _threshold_key(best_candidate.get("probability_threshold")),
                "best_bucket_key": _bucket_key(
                    cast(float, best_candidate["top_bucket_fraction"]),
                    cast(float, best_candidate["bottom_bucket_fraction"]),
                ),
                "best_holding_days_key": str(cast(int, best_candidate["holding_days"])),
                "same_family_runner_up": str(best_candidate["strategy_name"]) == str(runner_up_candidate["strategy_name"]),
                "same_threshold_runner_up": _threshold_key(best_candidate.get("probability_threshold"))
                == _threshold_key(runner_up_candidate.get("probability_threshold")),
                "same_bucket_runner_up": _bucket_key(
                    cast(float, best_candidate["top_bucket_fraction"]),
                    cast(float, best_candidate["bottom_bucket_fraction"]),
                )
                == _bucket_key(
                    cast(float, runner_up_candidate["top_bucket_fraction"]),
                    cast(float, runner_up_candidate["bottom_bucket_fraction"]),
                ),
                "same_holding_days_runner_up": int(cast(int, best_candidate["holding_days"]))
                == int(cast(int, runner_up_candidate["holding_days"])),
            }
        )
        cluster_split_entries.append(
            {
                "split_id": split.split_id,
                "train_start": min(split.train_dates).date().isoformat(),
                "train_end": max(split.train_dates).date().isoformat(),
                "test_start": min(split.test_dates).date().isoformat(),
                "test_end": max(split.test_dates).date().isoformat(),
                "n_train": int(len(train_frame)),
                "n_test": int(len(test_frame)),
                "best_cluster": {
                    "cluster_name": cluster_best_name,
                    "strategy_name": cluster_best_meta["strategy_name"],
                    "threshold_cluster_key": cluster_best_meta["threshold_cluster_key"],
                    "bucket_cluster_key": cluster_best_meta["bucket_cluster_key"],
                    "holding_days_cluster_key": cluster_best_meta["holding_days_cluster_key"],
                    "member_count": cluster_best_meta["member_count"],
                    "member_names": cluster_best_meta["member_names"],
                },
                "runner_up_cluster": {
                    "cluster_name": cluster_runner_up_name,
                    "strategy_name": cluster_runner_up_meta["strategy_name"],
                    "threshold_cluster_key": cluster_runner_up_meta["threshold_cluster_key"],
                    "bucket_cluster_key": cluster_runner_up_meta["bucket_cluster_key"],
                    "holding_days_cluster_key": cluster_runner_up_meta["holding_days_cluster_key"],
                    "member_count": cluster_runner_up_meta["member_count"],
                    "member_names": cluster_runner_up_meta["member_names"],
                },
                "best_candidate_test_information_ratio": float(cluster_out_sample_scores[cluster_best_index]),
                "best_vs_runner_up_in_sample_margin": float(
                    cluster_in_sample_scores[cluster_best_index] - cluster_in_sample_scores[cluster_runner_up_index]
                ),
                "best_vs_runner_up_oos_margin": float(
                    cluster_out_sample_scores[cluster_best_index] - cluster_out_sample_scores[cluster_runner_up_index]
                ),
                "best_candidate_oos_rank_percentile": cluster_oos_rank_percentile,
                "best_candidate_logit_omega": cluster_logit_omega,
                "overfit_flag": bool(cluster_overfit_flag),
                "best_strategy_name": str(cluster_best_meta["strategy_name"]),
                "best_threshold_key": str(cluster_best_meta["threshold_cluster_key"]),
                "best_bucket_key": str(cluster_best_meta["bucket_cluster_key"]),
                "best_holding_days_key": str(cluster_best_meta["holding_days_cluster_key"]),
                "same_family_runner_up": str(cluster_best_meta["strategy_name"])
                == str(cluster_runner_up_meta["strategy_name"]),
                "same_threshold_runner_up": str(cluster_best_meta["threshold_cluster_key"])
                == str(cluster_runner_up_meta["threshold_cluster_key"]),
                "same_bucket_runner_up": str(cluster_best_meta["bucket_cluster_key"])
                == str(cluster_runner_up_meta["bucket_cluster_key"]),
                "same_holding_days_runner_up": str(cluster_best_meta["holding_days_cluster_key"])
                == str(cluster_runner_up_meta["holding_days_cluster_key"]),
            }
        )

    if not split_entries:
        return {"splits": [], "aggregate_metrics": {}, "candidate_strategies": [], "pbo_summary": interpret_pbo(None)}, None

    aggregate_metrics = {
        "hit_rate_mean": float(np.mean([item["hit_rate"] for item in split_metrics])),
        "log_loss_mean": float(np.mean([item["log_loss"] for item in split_metrics])),
        "ece_mean": float(np.mean([item["ece"] for item in split_metrics])),
        "information_ratio_mean": float(
            np.mean([split_entry["best_candidate_test_information_ratio"] for split_entry in split_entries])
        ),
    }
    candidate_entries: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_name = cast(str, candidate["name"])
        metrics_list = candidate_oos_metrics[candidate_name]
        candidate_entries.append(
            {
                **candidate,
                "mean_oos_information_ratio": _mean_metric(metrics_list, "information_ratio"),
                "mean_oos_gross_information_ratio": _mean_metric(metrics_list, "gross_information_ratio"),
                "mean_oos_annual_return": _mean_metric(metrics_list, "annual_return"),
                "mean_oos_gross_annual_return": _mean_metric(metrics_list, "gross_annual_return"),
                "mean_oos_cost_drag_annual_return": _mean_metric(metrics_list, "cost_drag_annual_return"),
                "mean_oos_active_days_ratio": _mean_metric(metrics_list, "active_days_ratio"),
                "mean_oos_signal_days_ratio": _mean_metric(metrics_list, "signal_days_ratio"),
                "mean_oos_avg_daily_turnover": _mean_metric(metrics_list, "avg_daily_turnover"),
                "mean_oos_total_cost_bps": _mean_metric(metrics_list, "total_cost_bps"),
                "mean_oos_selection_stability": _mean_metric(metrics_list, "selection_stability"),
                "mean_oos_two_sided_signal_days_ratio": _mean_metric(metrics_list, "two_sided_signal_days_ratio"),
                "mean_oos_one_sided_signal_days_ratio": _mean_metric(metrics_list, "one_sided_signal_days_ratio"),
                "mean_oos_rebalance_thinned_days_ratio": _mean_metric(metrics_list, "rebalance_thinned_days_ratio"),
                "mean_oos_avg_target_turnover": _mean_metric(metrics_list, "avg_target_turnover"),
            }
        )
    candidate_entries.sort(key=lambda item: cast(float, item["mean_oos_information_ratio"]), reverse=True)
    cluster_entries = _aggregate_cluster_entries(candidate_entries)
    pbo = float(np.mean(overfit_flags)) if overfit_flags else None
    cluster_adjusted_pbo = float(np.mean(cluster_overfit_flags)) if cluster_overfit_flags else None
    pbo_diagnostics = _aggregate_pbo_diagnostics(split_entries, settings)
    cluster_adjusted_pbo_diagnostics = _aggregate_pbo_diagnostics(cluster_split_entries, settings)
    return {
        "splits": split_entries,
        "cluster_adjusted_splits": cluster_split_entries,
        "aggregate_metrics": aggregate_metrics,
        "candidate_strategies": candidate_entries,
        "cluster_candidate_strategies": cluster_entries,
        "portfolio_rule_summary": _aggregate_candidate_family(candidate_entries),
        "cluster_adjusted_pbo": cluster_adjusted_pbo,
        "cluster_adjusted_pbo_summary": interpret_pbo(cluster_adjusted_pbo),
        "cluster_adjusted_pbo_diagnostics": cluster_adjusted_pbo_diagnostics,
        "pbo_diagnostics": pbo_diagnostics,
        "pbo_summary": interpret_pbo(pbo),
    }, pbo

