from skill_control_plane.cli import _build_parser
from skill_control_plane.evals.control_plane import StageGold, StageTransitionGoldCase
from skill_control_plane.evals.retrieval_ablation import (
    evaluate_frozen_query_retrieval_ablation,
)
from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.evals.legacy.capability_need import CapabilityNeed


class MappingSearch:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, query, k=5):
        ids = self.mapping[query]
        return [
            RetrievalCandidate(skill_id, 1.0 / rank, rank)
            for rank, skill_id in enumerate(ids[:k], start=1)
        ]


class FrozenOldExtractor:
    def extract(self, initial_task, runtime_evidence):
        assert initial_task == "background"
        assert tuple(runtime_evidence) == ("runtime fact",)
        return CapabilityNeed("frozen old query", 0.9, "runtime fact")


def _case():
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
        "gold rationale",
        "snapshot",
    )


def test_frozen_query_ablation_separates_union_and_ranked_budget():
    dense = MappingSearch(
        {
            "frozen old query": (
                "d1", "d2", "d3", "d4", "d5", "target-skill", "d7"
            )
        }
    )
    bm25 = MappingSearch(
        {
            "frozen old query": (
                "b1", "b2", "b3", "b4", "b5", "target-skill", "b7"
            )
        }
    )

    report = evaluate_frozen_query_retrieval_ablation(
        [_case()],
        bm25=bm25,
        dense=dense,
        old_extractor=FrozenOldExtractor(),
        per_retriever_k=10,
        rrf_k=60,
        cutoffs=(5, 10),
    )

    assert report["dense"]["recall_at_5"] == 0
    assert report["bm25"]["recall_at_5"] == 0
    assert report["union"]["candidate_recall"] == 1
    assert report["rrf"]["recall_at_5"] == 1
    assert report["stages"][0]["query"] == "frozen old query"
    assert report["stages"][0]["dense"]["new_required_ranks"]["target-skill"] == 6
    assert report["stages"][0]["bm25"]["new_required_ranks"]["target-skill"] == 6
    assert report["stages"][0]["rrf"]["new_required_ranks"]["target-skill"] == 1
    assert report["skill_representation"] == "metadata-v0.1"
    assert report["target_transition_count"] == 1


def test_retrieval_ablation_cli_defaults_match_v04_experiment():
    args = _build_parser().parse_args(
        [
            "eval",
            "retrieval-ablation",
            "/skills",
            "--gold",
            "gold.jsonl",
            "--manifest",
            "manifest.json",
            "--old-rewrite-command",
            "replay-old",
        ]
    )
    assert args.per_retriever_k == 10
    assert args.rrf_k == 60
    assert args.dense_backend == "sentence-transformers"
    assert args.bigmodel_embedding_model == "embedding-3"
    assert args.bigmodel_embedding_dimensions == 2048
    assert args.old_rewrite_command == "replay-old"
    assert args.skill_representation == "metadata"
    assert args.retrieval_cards is None
