"""Explainability helpers for RetrievalCard field ablations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from skill_control_plane.evals.control_plane import StageTransitionGoldCase
from skill_control_plane.retrieval.cards import RETRIEVAL_CARD_FIELDS

RETRIEVAL_CARD_FIELD_ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("full", RETRIEVAL_CARD_FIELDS),
    *tuple(
        (
            f"minus-{field.replace('_', '-')}",
            tuple(candidate for candidate in RETRIEVAL_CARD_FIELDS if candidate != field),
        )
        for field in RETRIEVAL_CARD_FIELDS
    ),
)


def target_transition_count(cases: Sequence[StageTransitionGoldCase]) -> int:
    """Count runtime transitions that introduce at least one newly required Skill."""
    return sum(
        bool(stage.new_required)
        for case in cases
        for stage in case.stages[1:]
    )


def require_min_target_transitions(
    cases: Sequence[StageTransitionGoldCase],
    *,
    minimum: int = 20,
    allow_small_sample: bool = False,
) -> int:
    """Protect explanatory experiments from accidental tiny-sample conclusions."""
    if minimum < 1:
        raise ValueError("minimum target transitions must be >= 1")
    count = target_transition_count(cases)
    if count < minimum and not allow_small_sample:
        raise ValueError(
            "explanatory retrieval evaluation requires at least "
            f"{minimum} target Stage transitions; found {count}. "
            "Expand the Stage-transition Gold set, or pass --allow-small-sample "
            "for smoke testing only."
        )
    return count


def summarize_field_ablation_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize leave-one-field-out drops relative to the full RetrievalCard."""
    if "metadata-v0.1" not in reports:
        raise ValueError("metadata-v0.1 report is required")
    if "full" not in reports:
        raise ValueError("full RetrievalCard report is required")

    full = reports["full"]
    cutoffs = tuple(int(value) for value in full["cutoffs"])

    def ranked_metrics(report: Mapping[str, Any], arm: str) -> dict[str, float]:
        data = report[arm]
        return {
            f"recall_at_{cutoff}": float(data[f"recall_at_{cutoff}"])
            for cutoff in cutoffs
        }

    def compact(report: Mapping[str, Any]) -> dict[str, Any]:
        union = report["union"]
        return {
            "dense": ranked_metrics(report, "dense"),
            "bm25": ranked_metrics(report, "bm25"),
            "rrf": ranked_metrics(report, "rrf"),
            "union": {
                "candidate_recall": float(union["candidate_recall"]),
                "full_transition_coverage": float(union["full_transition_coverage"]),
                "average_candidate_set_size": float(union["average_candidate_set_size"]),
            },
        }

    full_compact = compact(full)
    variants: dict[str, Any] = {
        "metadata-v0.1": compact(reports["metadata-v0.1"]),
        "full": full_compact,
    }
    field_importance: dict[str, Any] = {}

    for label, _fields in RETRIEVAL_CARD_FIELD_ABLATIONS:
        if label == "full":
            continue
        if label not in reports:
            raise ValueError(f"missing ablation report: {label}")
        current = compact(reports[label])
        variants[label] = current
        field_name = label.removeprefix("minus-")
        field_importance[field_name] = {
            arm: {
                key: full_compact[arm][key] - current[arm][key]
                for key in full_compact[arm]
            }
            for arm in ("dense", "bm25", "rrf")
        }
        field_importance[field_name]["union"] = {
            key: full_compact["union"][key] - current["union"][key]
            for key in ("candidate_recall", "full_transition_coverage")
        }

    return {
        "experiment": "retrieval-card leave-one-field-out ablation",
        "interpretation": (
            "positive drop_from_full means removing that field reduced retrieval quality"
        ),
        "target_transition_count": int(full["target_transition_count"]),
        "new_required_skill_occurrences": int(full["new_required_skill_occurrences"]),
        "cutoffs": list(cutoffs),
        "variants": variants,
        "drop_from_full": field_importance,
    }


def print_field_ablation_summary(report: Mapping[str, Any]) -> None:
    print("=== RETRIEVAL CARD FIELD ABLATION ===")
    print(f"target_transitions: {report['target_transition_count']}")
    print(
        "new_required_skill_occurrences: "
        f"{report['new_required_skill_occurrences']}"
    )
    print()
    print("Positive drop means the removed field was helping retrieval.")
    for field, drops in report["drop_from_full"].items():
        print()
        print(f"[remove {field}]")
        for arm in ("dense", "bm25", "rrf"):
            values = drops[arm]
            formatted = " ".join(
                f"{key}={float(value):+.4f}"
                for key, value in values.items()
            )
            print(f"  {arm}: {formatted}")
        union = drops["union"]
        print(
            "  union: "
            f"candidate_recall={float(union['candidate_recall']):+.4f} "
            f"full_transition_coverage="
            f"{float(union['full_transition_coverage']):+.4f}"
        )
