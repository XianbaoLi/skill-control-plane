import json

import pytest

from skill_control_plane.evals.control_plane import StageGold, StageTransitionGoldCase
from skill_control_plane.evals.query_robustness import (
    LLMQueryParaphraser,
    QueryVariantSet,
    evaluate_query_variant_retrieval,
)
from skill_control_plane.evals.representation_ablation import (
    require_min_target_transitions,
)
from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.retrieval.cards import RetrievalCard, apply_retrieval_cards


class MappingSearch:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, query, k=5):
        return [
            RetrievalCandidate(skill_id, 1.0 / rank, rank)
            for rank, skill_id in enumerate(self.mapping[query][:k], start=1)
        ]


def _case():
    return StageTransitionGoldCase(
        "CASE",
        "background",
        (
            StageGold("S1", (), ("old",), ("old",), (), ()),
            StageGold(
                "S2",
                ("runtime fact",),
                ("target",),
                ("target",),
                (),
                (),
            ),
        ),
        "rationale",
        "snapshot",
    )


def test_retrieval_card_leave_one_field_out_keeps_metadata_fixed():
    skill = SkillRecord(
        skill_id="mail",
        name="Mail",
        description="legacy description",
        body="full body must not enter retrieval index",
        source_path="mail/SKILL.md",
        tags=("email",),
        content_hash="abc",
    )
    card = RetrievalCard(
        skill_id="mail",
        source_content_hash="abc",
        purpose="Handle mailbox work.",
        use_when=("Need to find a message.", "Need to answer a sender."),
        capabilities=("Search mail.", "Send replies."),
        lexical_cues=("inbox", "email reply", "mail search"),
    )

    full = apply_retrieval_cards([skill], {"mail": card})[0]
    minus_capabilities = apply_retrieval_cards(
        [skill],
        {"mail": card},
        include_fields=("purpose", "use_when", "lexical_cues"),
    )[0]

    assert "legacy description" in full.description
    assert "purpose:" in full.description
    assert "use when:" in full.description
    assert "capabilities:" in full.description
    assert "lexical cues:" in full.description
    assert "capabilities:" not in minus_capabilities.description
    assert "purpose:" in minus_capabilities.description
    assert "use when:" in minus_capabilities.description
    assert "lexical cues:" in minus_capabilities.description
    assert full.body == ""
    assert minus_capabilities.body == ""


def test_explainability_guard_rejects_tiny_target_set_by_default():
    cases = [_case()]
    with pytest.raises(ValueError, match="at least 20 target Stage transitions"):
        require_min_target_transitions(cases)
    assert require_min_target_transitions(
        cases,
        allow_small_sample=True,
    ) == 1


def test_query_paraphraser_requires_unique_non_original_variants():
    paraphraser = LLMQueryParaphraser(
        lambda _prompt: json.dumps(
            {"paraphrases": ["Locate the mail and reply.", "Find the message, then answer."]}
        )
    )
    result = paraphraser.paraphrase("Find the email and reply.", 2)
    assert result == (
        "Locate the mail and reply.",
        "Find the message, then answer.",
    )


def test_query_robustness_reports_recall_and_all_variant_stability():
    queries = ("original", "paraphrase one", "paraphrase two")
    dense = MappingSearch(
        {
            "original": ("target", "d2", "d3", "d4", "d5", "d6", "d7"),
            "paraphrase one": ("d1", "d2", "d3", "d4", "d5", "target", "d7"),
            "paraphrase two": ("d1", "d2", "target", "d4", "d5", "d6", "d7"),
        }
    )
    bm25 = MappingSearch(
        {
            query: ("b1", "target", "b3", "b4", "b5", "b6", "b7")
            for query in queries
        }
    )
    query_sets = {
        ("CASE", "S2"): QueryVariantSet(
            "CASE",
            "S2",
            "original",
            queries,
        )
    }

    report = evaluate_query_variant_retrieval(
        [_case()],
        query_sets=query_sets,
        bm25=bm25,
        dense=dense,
        per_retriever_k=10,
        rrf_k=60,
        cutoffs=(5, 10),
        skill_representation="test",
    )

    assert report["dense"]["recall_at_5"] == pytest.approx(2 / 3)
    assert report["dense"]["original_recall_at_5"] == 1
    assert report["dense"]["paraphrase_recall_at_5"] == pytest.approx(1 / 2)
    assert report["dense"]["all_variants_hit_at_5"] == 0
    assert report["dense"]["recall_at_10"] == 1
    assert report["dense"]["all_variants_hit_at_10"] == 1
    assert report["query_variant_count"] == 3
