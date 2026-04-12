from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from market_prediction_agent.utils.paths import get_repo_root


EXCLUDE_PATTERNS = [
    ".claude",
    ".git",
    ".venv",
    "__pycache__",
    "*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_tmp",
    ".pytest_tmp*",
    ".uv-cache",
    ".test-artifacts",
    "pytest-cache-files-*",
    "build",
    "dist",
    "htmlcov",
]
ROOT_ONLY_EXCLUDES = {"storage"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clean distributable copy of the repository.")
    parser.add_argument(
        "--output",
        default="dist/package_clean",
        help="Output directory for the cleaned package copy.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a zip archive next to the cleaned copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    output_path = (repo_root / args.output).resolve()
    if output_path.exists():
        shutil.rmtree(output_path)

    def ignore(directory: str, entries: list[str]) -> set[str]:
        ignored = set(shutil.ignore_patterns(*EXCLUDE_PATTERNS)(directory, entries))
        current = Path(directory).resolve()
        if current == repo_root.resolve():
            ignored.update(name for name in entries if name in ROOT_ONLY_EXCLUDES)
        return ignored

    shutil.copytree(repo_root, output_path, ignore=ignore)
    print(f"Clean package created at {output_path}")
    if args.zip:
        archive_base = output_path.parent / output_path.name
        zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_path.parent, base_dir=output_path.name))
        print(f"Zip archive created at {zip_path}")


if __name__ == "__main__":
    main()
