from skill_control_plane.evals.legacy.evidence import EvidencePool


def test_evidence_pool_reuses_agent_interpretation_in_direct_retrieval_text() -> None:
    evidence = EvidencePool(
        raw_runtime=("CI fails after cache restore",),
        structured_signals=("exit_code=1",),
        agent_interpretation=("possible cache contamination or nondeterminism",),
    )

    text = evidence.direct_retrieval_text()

    assert "CI fails after cache restore" in text
    assert "exit_code=1" in text
    assert "cache contamination" in text
