from __future__ import annotations

from pathlib import Path
import os


def get_repo_root() -> Path:
    configured_root = os.getenv("MARKET_PREDICTION_AGENT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (get_repo_root() / candidate).resolve()
