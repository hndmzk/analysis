from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_prediction_agent.audits.public_audit_suite import (
    load_public_audit_suite,
    persist_public_audit_suite,
    replay_public_audit_suite,
)
from market_prediction_agent.config import load_settings, resolve_storage_path
from market_prediction_agent.schemas.validator import validate_payload
from market_prediction_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a stored public audit suite with the current retraining policy.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--suite-path", required=True, help="Path to an existing monitor audit suite JSON artifact.")
    parser.add_argument(
        "--profile",
        choices=["fast", "standard", "full_light", "full"],
        default="full",
        help="Profile label attached to the replay artifact metadata. full is the default deep-dive/replay role.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    suite_path = Path(args.suite_path)
    source_payload = load_public_audit_suite(suite_path)
    replay_payload = replay_public_audit_suite(
        settings=settings,
        source_payload=source_payload,
        source_suite_path=str(suite_path),
        profile_name=args.profile,
    )
    validate_payload("monitor_audit_suite", replay_payload)
    replay_path = persist_public_audit_suite(settings, replay_payload)
    print(
        json.dumps(
            {
                "storage_path": str(resolve_storage_path(settings)),
                "source_suite_path": str(suite_path),
                "replay_suite_path": str(replay_path),
                "comparison_to_source": replay_payload.get("comparison_to_source", {}),
                "distribution_summary": replay_payload["distribution_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
