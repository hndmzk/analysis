from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from market_prediction_agent.config import Settings, load_settings


def test_settings_frozen() -> None:
    settings = load_settings("config/default.yaml")

    with pytest.raises((TypeError, ValidationError)):
        settings.data.storage_path = str(Path(".test-artifacts") / "mutated")


def test_settings_validation_error() -> None:
    payload = load_settings("config/default.yaml").model_dump(mode="python", by_alias=True)
    payload["trading"]["max_daily_orders"] = {"bad": "type"}

    with pytest.raises(ValidationError):
        Settings.model_validate(payload)
