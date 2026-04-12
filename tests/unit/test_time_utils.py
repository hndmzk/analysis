from __future__ import annotations

import pandas as pd

from market_prediction_agent.utils.time_utils import is_business_day, point_in_time_filter


def test_point_in_time_filter_respects_available_at() -> None:
    frame = pd.DataFrame(
        {
            "available_at": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
            "value": [1, 2],
        }
    )
    filtered = point_in_time_filter(frame, "2026-01-02T00:00:00Z")
    assert filtered["value"].tolist() == [1]


def test_is_business_day() -> None:
    assert is_business_day(pd.Timestamp("2026-01-02"))
    assert not is_business_day(pd.Timestamp("2026-01-03"))

