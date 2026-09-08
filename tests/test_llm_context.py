from skill_control_plane.runtime.evidence import EvidencePool
from skill_control_plane.runtime.llm_context import (
    LLMStageContextEnhancer,
    build_src_prompt,
)


def test_src_prompt_is_narrow_and_contains_runtime_evidence() -> None:
    evidence = EvidencePool(
        raw_runtime=("tests fail after cache restore",),
        agent_interpretation=("possible nondeterminism",),
    )

    prompt = build_src_prompt(evidence)

    assert "tests fail after cache restore" in prompt
    assert "possible nondeterminism" in prompt
    assert "Do not solve the underlying task" in prompt


def test_llm_enhancer_returns_provider_neutral_stage_context() -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return " CI cache contamination and flaky-test diagnosis "

    enhancer = LLMStageContextEnhancer(complete=complete)
    context = enhancer.build_context(
        EvidencePool(raw_runtime=("CI cache restore causes intermittent failures",))
    )

    assert context.text == "CI cache contamination and flaky-test diagnosis"
    assert context.source == "llm"
    assert len(prompts) == 1
