import json

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.retrieval.cards import (
    LLMRetrievalCardExtractor,
    RETRIEVAL_CARD_VERSION,
    apply_retrieval_cards,
    build_retrieval_card_cache,
    build_retrieval_card_prompt,
    load_retrieval_cards,
    validate_retrieval_cards,
)


def _skill(content_hash="hash-1"):
    return SkillRecord(
        skill_id="python-debugpy",
        name="python-debugpy",
        description="Debug Python with pdb and debugpy.",
        body=(
            "Use when a traceback does not reveal why a value is wrong. "
            "Step through a function and watch a collection mutate."
        ),
        source_path="/skills/python-debugpy/SKILL.md",
        tags=("debugging", "python", "breakpoints"),
        content_hash=content_hash,
        category="development",
    )


def _completion(_prompt):
    return json.dumps(
        {
            "purpose": "Interactively debug Python execution.",
            "use_when": [
                "A traceback does not reveal why a value is wrong.",
                "Step through a function and watch a collection mutate.",
            ],
            "capabilities": [
                "Set breakpoints and step through Python code.",
                "Inspect changing variables and collections.",
            ],
            "lexical_cues": [
                "step through",
                "watch variable",
                "breakpoint",
                "pdb",
                "debugpy",
            ],
        }
    )


def test_prompt_is_single_skill_only_and_card_contains_trigger_semantics():
    skill = _skill()
    prompt = build_retrieval_card_prompt(skill)
    assert "Gold" in prompt  # explicit prohibition, not benchmark content
    assert "python-debugpy" in prompt
    assert "Step through a function" in prompt

    card = LLMRetrievalCardExtractor(_completion).extract(skill)
    text = card.search_text(skill)
    assert card.version == RETRIEVAL_CARD_VERSION
    assert card.source_content_hash == "hash-1"
    assert "traceback does not reveal" in text
    assert "step through" in text
    assert "breakpoint" in text


def test_cache_reuses_matching_content_hash_and_projects_to_metadata(tmp_path):
    skill = _skill()
    output = tmp_path / "cards.jsonl"

    first = build_retrieval_card_cache(
        [skill],
        extractor=LLMRetrievalCardExtractor(_completion),
        output=output,
    )
    assert first == {"skill_count": 1, "extracted": 1, "reused": 0}

    def should_not_run(_prompt):
        raise AssertionError("matching card should be reused")

    second = build_retrieval_card_cache(
        [skill],
        extractor=LLMRetrievalCardExtractor(should_not_run),
        output=output,
    )
    assert second == {"skill_count": 1, "extracted": 0, "reused": 1}

    cards = load_retrieval_cards(output)
    indexed = apply_retrieval_cards([skill], cards)
    assert indexed[0].body == ""
    assert indexed[0].tags == skill.tags
    assert "use when:" in indexed[0].description
    assert "watch a collection mutate" in indexed[0].description


def test_stale_card_is_rejected(tmp_path):
    skill = _skill()
    output = tmp_path / "cards.jsonl"
    build_retrieval_card_cache(
        [skill],
        extractor=LLMRetrievalCardExtractor(_completion),
        output=output,
    )
    cards = load_retrieval_cards(output)

    with pytest.raises(ValueError, match="stale"):
        validate_retrieval_cards([_skill("hash-2")], cards)
