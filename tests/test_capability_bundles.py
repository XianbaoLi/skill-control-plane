from __future__ import annotations

from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.runtime.bundles import (
    build_capability_shelf,
    integrate_retrieval_delta,
)


def _record(skill_id: str, category: str, description: str = "") -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=skill_id,
        description=description,
        body="",
        source_path=f"/skills/{category}/{skill_id}/SKILL.md",
        category=category,
    )


def _candidate(skill_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(skill_id=skill_id, score=1.0 / rank, rank=rank)


def test_initial_retrieval_builds_active_bundle_and_shelf() -> None:
    records = {
        "pytest": _record("pytest", "testing", "Diagnose Python test failures."),
        "github-actions": _record("github-actions", "ci", "Inspect CI workflows."),
        "dependency": _record(
            "dependency", "runtime", "Diagnose runtime dependencies."
        ),
        "flaky": _record("flaky", "testing", "Diagnose nondeterministic tests."),
    }
    candidates = [
        _candidate("pytest", 1),
        _candidate("github-actions", 2),
        _candidate("dependency", 3),
        _candidate("flaky", 4),
    ]

    shelf = build_capability_shelf(candidates, records)

    assert shelf.active_bundle_ids == ("testing",)
    assert shelf.get("testing").skill_ids == ("pytest", "flaky")
    assert shelf.get("ci").skill_ids == ("github-actions",)
    assert shelf.get("runtime").skill_ids == ("dependency",)
    assert shelf.registered_skill_ids == (
        "pytest",
        "flaky",
        "github-actions",
        "dependency",
    )


def test_later_retrieval_expands_existing_shelf_bundle_and_activates_it() -> None:
    records = {
        "pytest": _record("pytest", "testing"),
        "dependency": _record("dependency", "runtime"),
        "cuda": _record("cuda", "runtime"),
        "shared-lib": _record("shared-lib", "runtime"),
    }
    initial = build_capability_shelf(
        [_candidate("pytest", 1), _candidate("dependency", 2)],
        records,
    )

    updated = integrate_retrieval_delta(
        initial,
        [_candidate("cuda", 1), _candidate("shared-lib", 2)],
        records,
    )

    assert updated.get("runtime").skill_ids == (
        "dependency",
        "cuda",
        "shared-lib",
    )
    assert updated.active_bundle_ids == ("testing", "runtime")
    assert set(initial.registered_skill_ids).issubset(updated.registered_skill_ids)


def test_later_retrieval_creates_new_bundle_without_dropping_old_ones() -> None:
    records = {
        "pytest": _record("pytest", "testing"),
        "oauth": _record("oauth", "api-access"),
    }
    initial = build_capability_shelf([_candidate("pytest", 1)], records)

    updated = integrate_retrieval_delta(
        initial,
        [_candidate("oauth", 1)],
        records,
    )

    assert updated.get("testing") is not None
    assert updated.get("api-access").skill_ids == ("oauth",)
    assert updated.active_bundle_ids == ("testing", "api-access")
    assert updated.registered_skill_ids == ("pytest", "oauth")
