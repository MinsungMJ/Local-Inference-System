from __future__ import annotations

import json
from pathlib import Path

from lis_verify.product_contract import StageState


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "tools" / "test_fixtures" / "lis_verify_contract" / "report_examples"


def load_example(name: str) -> dict:
    return json.loads((EXAMPLE_ROOT / f"{name}.json").read_text())


def drive_pre_cleanup_stages(context, raw: dict) -> dict:
    for stage in raw["stages"][:-1]:
        context.start_stage(stage["name"])
        context.finish_stage(
            StageState(stage["state"]),
            result_ref=stage["result_ref"],
            evidence_tier=stage["evidence_tier"],
            failure_class=stage["failure_class"],
            reason=stage["reason"],
            blocker=stage["blocker"],
        )
    return raw
