#!/usr/bin/env python3
"""Freeze one capability_need query per positive runtime Stage transition.

This is an experiment-preparation step. It invokes the same capability_need extractor
used by retrieval evaluation exactly once for each post-S1 stage with new_required,
prints the resulting query for human audit, and relies on the completion adapter's
--audit-dir to persist exact prompt/completion replay records.

After this command succeeds, all retrieval experiments should use the same directory
through capability_need_codex.py --replay-dir so no model calls or query drift occur
inside the experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from skill_control_plane.evals import load_stage_transition_gold
from skill_control_plane.runtime.capability_need import (
    LLMCapabilityNeedExtractor,
    command_completer,
    extract_query,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument(
        "--completion-command",
        required=True,
        help=(
            "Capability-need completion command. For freezing with Codex, pass "
            "'python scripts/capability_need_codex.py --audit-dir <dir>'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL summary of frozen queries for human inspection.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional case filter; may be repeated.",
    )
    parser.add_argument(
        "--stage-id",
        action="append",
        default=[],
        help="Optional stage filter; may be repeated.",
    )
    args = parser.parse_args()

    cases = load_stage_transition_gold(args.gold)
    extractor = LLMCapabilityNeedExtractor(
        command_completer(args.completion_command)
    )

    rows: list[dict[str, object]] = []
    for case in cases:
        if args.case_id and case.case_id not in set(args.case_id):
            continue
        for stage in case.stages[1:]:
            if args.stage_id and stage.stage_id not in set(args.stage_id):
                continue
            if not stage.new_required:
                continue
            result = extract_query(
                extractor,
                case.initial_task,
                stage.runtime_evidence,
            )
            row = {
                "case_id": case.case_id,
                "stage_id": stage.stage_id,
                "new_required": list(stage.new_required),
                "query": result.query,
                "status": result.status,
                "confidence": (
                    result.need.confidence if result.need is not None else None
                ),
                "evidence_basis": (
                    result.need.evidence_basis if result.need is not None else ""
                ),
            }
            rows.append(row)

    failed = [row for row in rows if row["status"] != "ok"]
    for row in rows:
        print(
            f"{row['case_id']}/{row['stage_id']} -> "
            f"{','.join(row['new_required'])}\n"
            f"  query: {row['query']}\n"
            f"  status={row['status']} confidence={row['confidence']}\n"
        )

    print(f"target_transitions: {len(rows)}")
    print(f"ok: {len(rows) - len(failed)}")
    print(f"failed: {len(failed)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        print(f"summary: {args.output}")

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
