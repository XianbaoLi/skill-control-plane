from __future__ import annotations

import json
from pathlib import Path

from skill_control_plane.evals import evaluate_union_retrieval, load_runtime_retrieval_gold
from skill_control_plane.models import RetrievalCandidate


class FakeRetriever:
    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(skill_id=skill_id, score=1.0 / rank, rank=rank)
            for rank, skill_id in enumerate(self.results[query][:k], start=1)
        ]


def _write_gold(path: Path) -> None:
    rows = [
        {
            "case_id": "A",
            "query": "query-a",
            "required": ["target-a"],
            "useful": [],
            "hard_negative": [],
            "rationale": "",
            "snapshot_id": "snapshot-1",
        },
        {
            "case_id": "B",
            "query": "query-b",
            "required": ["target-b", "target-c"],
            "useful": [],
            "hard_negative": [],
            "rationale": "",
            "snapshot_id": "snapshot-1",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_union_retrieval_reports_recall_and_candidate_size(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_gold(gold_path)
    cases = load_runtime_retrieval_gold(gold_path)

    bm25 = FakeRetriever(
        {
            "query-a": ["target-a", "noise-1"],
            "query-b": ["noise-2", "target-b"],
        }
    )
    dense = FakeRetriever(
        {
            "query-a": ["noise-1", "noise-3"],
            "query-b": ["target-c", "noise-2"],
        }
    )

    report = evaluate_union_retrieval(cases, bm25=bm25, dense=dense, k=2)

    assert report["candidate_recall"] == 1.0
    assert report["full_case_coverage"] == 1.0
    assert report["average_candidate_set_size"] == 3.0
    assert report["cases"][1]["union_required_hits"] == ["target-b", "target-c"]
