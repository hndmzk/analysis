from __future__ import annotations

from pathlib import Path

from market_prediction_agent.config import load_settings, resolve_storage_path
from market_prediction_agent.utils.paths import get_repo_root


def test_load_settings_and_storage_path_are_repo_relative(monkeypatch) -> None:
    monkeypatch.setenv("CONFIG_PATH", "config/default.yaml")
    monkeypatch.chdir(Path(".claude"))
    settings = load_settings()
    assert resolve_storage_path(settings) == get_repo_root() / "storage"

