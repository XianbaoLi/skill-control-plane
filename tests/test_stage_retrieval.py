from __future__ import annotations

from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.evals.legacy.evidence import EvidencePool
from skill_control_plane.evals.legacy.stage_retrieval import (
    RetrievalPhase,
    StageRetrievalContext,
    retrieve_for_stage,
)


class FakeRetriever:
    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        self.queries.append(query)
        return [
            RetrievalCandidate(skill_id=skill_id, score=1.0 / rank, rank=rank)
            for rank, skill_id in enumerate(self.results.get(query, [])[:k], start=1)
        ]


class FakeEnhancer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def build_context(self, evidence: EvidencePool) -> StageRetrievalContext:
        self.calls += 1
        return StageRetrievalContext(text=self.text, source="fake")


def top_is(skill_id: str):
    return lambda candidates: bool(candidates) and candidates[0].skill_id == skill_id


def test_raw_retrieval_stops_without_semantic_escalation_when_sufficient() -> None:
    evidence = EvidencePool(raw_runtime=("docker: command not found",))
    retriever = FakeRetriever({"docker: command not found": ["docker", "shell"]})
    enhancer = FakeEnhancer("container runtime diagnostics")

    result = retrieve_for_stage(
        evidence,
        retriever,
        sufficiency=top_is("docker"),
        enhancer=enhancer,
    )

    assert result.phase is RetrievalPhase.RAW
    assert enhancer.calls == 0
    assert retriever.queries == ["docker: command not found"]


def test_agent_interpretation_is_reused_before_extra_enhancer() -> None:
    evidence = EvidencePool(
        raw_runtime=("CI fails intermittently",),
        agent_interpretation=("likely cache contamination after restore",),
    )
    retriever = FakeRetriever(
        {
            "CI fails intermittently": ["generic-ci"],
            "CI fails intermittently\nlikely cache contamination after restore": [
                "cache-debug",
                "generic-ci",
            ],
        }
    )
    enhancer = FakeEnhancer("CI cache contamination diagnosis")

    result = retrieve_for_stage(
        evidence,
        retriever,
        sufficiency=top_is("cache-debug"),
        enhancer=enhancer,
    )

    assert result.phase is RetrievalPhase.AGENT_AUGMENTED
    assert enhancer.calls == 0
    assert len(result.traces) == 2


def test_src_is_used_only_after_raw_and_agent_paths_are_insufficient() -> None:
    evidence = EvidencePool(
        raw_runtime=("tests fail only in CI",),
        agent_interpretation=("environment may differ",),
    )
    src_text = "CI environment isolation and dependency mismatch diagnosis"
    retriever = FakeRetriever(
        {
            "tests fail only in CI": ["pytest"],
            "tests fail only in CI\nenvironment may differ": ["generic-ci"],
            src_text: ["env-isolation", "dependency-debug"],
        }
    )
    enhancer = FakeEnhancer(src_text)

    result = retrieve_for_stage(
        evidence,
        retriever,
        sufficiency=top_is("env-isolation"),
        enhancer=enhancer,
    )

    assert result.phase is RetrievalPhase.SRC_ENHANCED
    assert enhancer.calls == 1
    assert result.src is not None
    assert result.src.text == src_text
    assert "env-isolation" in {item.skill_id for item in result.candidates}
    assert [trace.phase for trace in result.traces] == [
        RetrievalPhase.RAW,
        RetrievalPhase.AGENT_AUGMENTED,
        RetrievalPhase.SRC_ENHANCED,
    ]


def test_no_enhancer_means_no_extra_semantic_call() -> None:
    evidence = EvidencePool(raw_runtime=("unknown runtime failure",))
    retriever = FakeRetriever({"unknown runtime failure": ["generic-debug"]})

    result = retrieve_for_stage(
        evidence,
        retriever,
        sufficiency=top_is("target"),
    )

    assert result.phase is RetrievalPhase.RAW
    assert result.candidates[0].skill_id == "generic-debug"
