from __future__ import annotations

from pathlib import Path
import os


def pytest_configure() -> None:
    os.environ.setdefault("CONFIG_PATH", str(Path("config") / "default.yaml"))

