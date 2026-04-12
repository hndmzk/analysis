from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from market_prediction_agent.utils.paths import resolve_repo_path


@lru_cache(maxsize=None)
def load_schema(schema_name: str) -> dict[str, Any]:
    path = resolve_repo_path("config") / "schemas" / f"{schema_name}.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_payload(schema_name: str, payload: dict[str, Any]) -> None:
    validator = Draft202012Validator(load_schema(schema_name), format_checker=FormatChecker())
    validator.validate(payload)
