from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def write_frame(self, relative_path: str | Path, frame: pd.DataFrame) -> Path:
        target = self.base_path / Path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        return target

    def read_frame(self, relative_path: str | Path) -> pd.DataFrame:
        target = self.base_path / Path(relative_path)
        return pd.read_parquet(target)

    def resolve(self, relative_path: str | Path) -> Path:
        return self.base_path / Path(relative_path)

