import json

from skill_control_plane.cli import _build_parser
from skill_control_plane.evals.capability_facets import (
    evaluate_capability_facet_retrieval,
)
from skill_control_plane.evals.control_plane import StageGold, StageTransitionGoldCase
from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.runtime.capability_facets import (
    CapabilityFacets,
    LLMCapabilityFacetExtractor,
    build_capability_facets_prompt,
    extract_facet_queries,
)
from skill_control_plane.runtime.capability_need import CapabilityNeed


class MappingSearch:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, query, k=5):
        ids = self.mapping.get(query, ())
        return [
            RetrievalCandidate(skill_id, 1.0 / rank, rank)
            for rank, skill_id in enumerate(ids[:k], start=1)
        ]


class OldExtractor:
    def extract(self, initial_task, runtime_evidence):
        assert initial_task == "background"
        assert tuple(runtime_evidence) == ("runtime fact",)
        return CapabilityNeed("old rewrite", 0.9, "runtime fact")


class FacetExtractor:
    def extract(self, initial_task, runtime_evidence):
        assert initial_task == "background"
        assert tuple(runtime_evidence) == ("runtime fact",)
        return CapabilityFacets(
            ("root cause analysis", "runtime inspection", "failure reproduction"),
            0.9,
            "runtime fact",
        )


def case():
    return StageTransitionGoldCase(
        "CASE",
        "background",
        (
            StageGold("S1", (), ("old-skill",), ("old-skill",), (), ()),
            StageGold(
                "S2",
                ("runtime fact",),
                ("target-skill",),
                ("target-skill",),
                (),
                (),
            ),
        ),
        "gold rationale must never reach extraction",
        "snapshot",
    )


def test_facets_multi_query_can_recover_target_missed_by_old_rewrite():
    mapping = {
        "old rewrite": ("d1", "d2", "d3", "d4", "d5", "target-skill"),
        "root cause analysis": ("target-skill", "d1"),
        "runtime inspection": ("d2", "target-skill"),
        "failure reproduction": ("target-skill", "d3"),
    }
    search = MappingSearch(mapping)
    report = evaluate_capability_facet_retrieval(
        [case()],
        bm25=search,
        dense=search,
        old_extractor=OldExtractor(),
        facet_extractor=FacetExtractor(),
        per_query_k=5,
        rrf_k=60,
    )

    assert report["old"]["recall_at_5"] == 0
    assert report["facets"]["recall_at_5"] == 1
    stage = report["stages"][0]
    assert stage["old"]["new_required_ranks"]["target-skill"] is None
    assert stage["facets"]["new_required_ranks"]["target-skill"] == 1
    assert stage["facets"]["queries"] == [
        "root cause analysis",
        "runtime inspection",
        "failure reproduction",
    ]


def test_facets_prompt_contains_only_runtime_inputs():
    prompt = build_capability_facets_prompt("background", ["runtime fact"])
    payload = json.loads(prompt.split("\nInput:\n")[1])
    assert payload == {
        "initial_task": "background",
        "runtime_evidence": ["runtime fact"],
    }
    assert "Skill ID" in prompt
    assert "Gold" not in prompt
    assert "gold rationale" not in prompt


def test_invalid_or_low_confidence_facets_fall_back_to_raw_evidence():
    malformed = LLMCapabilityFacetExtractor(
        lambda _: json.dumps({"capabilities": ["only one"], "confidence": 0.9})
    )
    result = extract_facet_queries(malformed, "task", ["raw", "fact"])
    assert result.queries == ("raw\nfact",)
    assert result.status == "error:ValueError"

    low = LLMCapabilityFacetExtractor(
        lambda _: json.dumps(
            {
                "capabilities": ["root cause analysis", "runtime inspection", "failure reproduction"],
                "confidence": 0.2,
            }
        )
    )
    result = extract_facet_queries(low, "task", ["raw"])
    assert result.queries == ("raw",)
    assert result.status == "low-confidence"


def test_capability_facets_cli_has_frozen_v04_defaults():
    args = _build_parser().parse_args(
        [
            "eval",
            "capability-facets",
            "/skills",
            "--gold",
            "gold.jsonl",
            "--manifest",
            "manifest.json",
            "--old-rewrite-command",
            "old-provider",
            "--capability-facets-command",
            "hermes-provider",
        ]
    )
    assert args.per_query_k == 10
    assert args.rrf_k == 60
    assert args.dense_backend == "sentence-transformers"
    assert args.bigmodel_embedding_model == "embedding-3"
    assert args.bigmodel_embedding_dimensions == 2048
    assert args.old_rewrite_command == "old-provider"
    assert args.capability_facets_command == "hermes-provider"
