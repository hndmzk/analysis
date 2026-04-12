from __future__ import annotations

import argparse
import json

from market_prediction_agent.audits.public_snapshot_seed import seed_public_snapshots
from market_prediction_agent.config import load_settings
from market_prediction_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed public-data cache and snapshots for public real-data audits.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--tickers", default=None, help="Optional comma-separated ticker list.")
    parser.add_argument("--history-days", type=int, default=None, help="Optional lookback override for snapshot seeding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    tickers = None
    if args.tickers:
        tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    try:
        result = seed_public_snapshots(settings, tickers=tickers, history_days=args.history_days)
    except RuntimeError as exc:
        if "no cache or snapshot fallback" in str(exc):
            raise RuntimeError(
                "Snapshot seed failed because the public upstream did not return data and no local snapshot exists yet. "
                "Retry when network access to the public source is available, or seed snapshots on another machine and copy "
                "`storage/public_data/snapshots/` into this repository before running the public real-data audit."
            ) from exc
        raise
    print(json.dumps(result.payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
