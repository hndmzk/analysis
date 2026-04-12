from __future__ import annotations

import pandas as pd

from market_prediction_agent.data.normalizer import apply_stale_flag


def test_apply_stale_flag_marks_stale_and_non_stale_rows() -> None:
    as_of_time = pd.Timestamp("2026-01-02T12:00:00Z")
    frame = pd.DataFrame(
        [
            {"ticker": "AAA", "fetched_at": "2026-01-02T11:30:00Z"},
            {"ticker": "BBB", "fetched_at": "2026-01-02T08:00:00Z"},
        ]
    )
    updated = apply_stale_flag(frame, as_of_time=as_of_time, threshold_hours=2)
    assert bool(updated.loc[0, "stale_data_flag"]) is False
    assert bool(updated.loc[1, "stale_data_flag"]) is True
    assert updated.loc[1, "data_age_hours"] > updated.loc[0, "data_age_hours"]
