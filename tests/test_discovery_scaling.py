import json

import pytest

from skill_control_plane.evals.discovery_scaling import (
    DiscoveryCase, EmbeddingCacheMiss, HashedEmbeddingCache,
    audit_distractors, build_frozen_distractors, run_arm, subset_ids, summarize,
)
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery


def call(name, arguments):
    return {"role": "assistant", "tool_calls": [{"id": name + str(arguments), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class ScriptedClient:
    model = "fake-glm"
    def __init__(self, messages): self.messages = iter(messages); self.requests = []
    def complete_messages(self, messages, *, tools):
        self.requests.append((messages, tools))
        return next(self.messages), {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


def records():
    return {record.skill_id: record for record in SkillRegistry([
        SkillRecord("a", "Alpha", "alpha capability", "", ""),
        SkillRecord("b", "Beta", "beta capability", "", ""),
        SkillRecord("c", "Gamma", "gamma capability", "", ""),
        *[SkillRecord(f"d{i}", f"D{i}", "distractor", "", "") for i in range(20)],
    ])}


def test_subset_keeps_required_and_is_stable():
    case = DiscoveryCase("case", "need", ("a", "b"), "test")
    all_ids = tuple(records())
    first = subset_ids(case, 20, all_ids)
    assert first == subset_ids(case, 20, all_ids)
    assert {"a", "b"} <= set(first)
    assert len(first) == 20


def test_full_catalog_has_only_catalog_and_equal_select_protocol():
    available = records(); case = DiscoveryCase("case", "need", ("a",), "test")
    client = ScriptedClient([call("select_skills", {"skill_ids": ["a"]})])
    row = run_arm(arm="full_catalog", case=case, ids=tuple(available), records=available,
                  client=client, discovery=None)
    assert row["full_required_set_coverage"] == 1.0
    assert row["model_call_count"] == 1 and row["search_count"] == 0
    system = client.requests[0][0][0]["content"]
    assert '"skill_id":"a"' in system and "body" not in system and "retrieval" not in system.lower()
    assert [tool["function"]["name"] for tool in client.requests[0][1]] == ["select_skills"]


def test_retrieval_pool_merges_two_searches_then_selects_both():
    available = records(); case = DiscoveryCase("case", "need", ("a", "b"), "test",
                                                search_phrases=("alpha", "beta"))
    client = ScriptedClient([call("load_capability", {"need": "alpha"}), call("load_capability", {"need": "beta"}),
                             call("select_skills", {"skill_ids": ["a", "b"]})])
    row = run_arm(arm="retrieval_first", case=case, ids=tuple(available), records=available, client=client,
                  discovery=SkillDiscovery(SkillRegistry(available.values())), max_steps=3)
    assert row["error"] is None
    assert row["search_count"] == 2
    assert row["target_entered_candidate_pool"]
    assert row["full_required_set_coverage"] == 1.0
    assert set(row["searches"][1]["pool_after"]) >= {"a", "b"}
    search = row["searches"][0]
    assert set(search) == {"need", "model_visible_payload",
                           "internal_retrieval_record", "pool_after"}
    assert search["internal_retrieval_record"]["representations"]
    assert set(search["model_visible_payload"]) == {
        "query", "candidates", "search_control"}


def test_retrieval_failure_categories_distinguish_pool_from_selection():
    available = records(); discovery = SkillDiscovery(SkillRegistry(available.values()))
    case = DiscoveryCase("case", "need", ("a",), "test",
                         search_phrases=("beta", "alpha beta"))
    no_pool = run_arm(arm="retrieval_first", case=case, ids=tuple(available), records=available,
                      client=ScriptedClient([call("load_capability", {"need": "beta"}), call("select_skills", {"skill_ids": ["b"]})]),
                      discovery=discovery, max_steps=2)
    assert no_pool["failure_type"] == "retrieval_miss"
    assert no_pool["target_not_in_candidate_pool"]
    non_required_candidate = next(candidate.skill_id for candidate in discovery.discover_skills("alpha beta", k=10).candidates
                                  if candidate.skill_id != "a")
    in_pool = run_arm(arm="retrieval_first", case=case, ids=tuple(available), records=available,
                      client=ScriptedClient([call("load_capability", {"need": "alpha beta"}), call("select_skills", {"skill_ids": [non_required_candidate]})]),
                      discovery=discovery, max_steps=2)
    assert in_pool["failure_type"] == "llm_selection_miss"
    assert in_pool["target_in_pool_llm_not_selected"]


def test_rejected_selection_is_not_counted_as_committed_recall():
    available = records(); discovery = SkillDiscovery(SkillRegistry(available.values()))
    case = DiscoveryCase("case", "alpha", ("a",), "test", search_phrases=("alpha",))
    row = run_arm(arm="retrieval_first", case=case, ids=tuple(available), records=available,
                  client=ScriptedClient([call("load_capability", {"need": "alpha"}),
                                         call("select_skills", {"skill_ids": ["invented"]})]),
                  discovery=discovery, max_steps=2)
    assert row["error"] == "ValueError"
    assert row["selected_skill_ids"] == [] and row["required_skill_recall"] == 0.0
    assert row["failure_type"] == "llm_selection_miss"
    assert row["target_in_pool_llm_not_selected"]


def test_summary_groups_costs_and_retrieval_failures():
    base = {"case_id": "x", "source": "test", "query": "q", "required_skill_ids": ["a"],
            "corpus_skill_ids": ["a"], "selected_skill_ids": ["a"], "searches": [], "model_calls": [],
            "error": None, "required_skill_recall": 1.0, "full_required_set_coverage": 1.0, "precision": 1.0,
            "extra_selected_skills": [], "extra_selected_skill_count": 0, "search_count": 1, "model_call_count": 2, "input_tokens": 5,
            "output_tokens": 3, "total_tokens": 8, "wall_time_seconds": 2.0, "latency_seconds_per_call": 1.0}
    full = {**base, "arm": "full_catalog", "corpus_size": 20}
    retrieval = {**base, "arm": "retrieval_first", "corpus_size": 20,
                 "retriever_candidate_recall": 1.0, "first_search_resolution_rate": 1.0,
                 "failure_type": "llm_selection_miss",
                 "target_in_pool_llm_not_selected": True}
    result = summarize([full, retrieval])
    row = next(row for row in result if row["arm"] == "retrieval_first" and row["corpus_size"] == 20)
    assert row["avg_total_tokens"] == 8 and row["target_in_pool_llm_not_selected"] == 1


def test_embedding_cache_is_snapshot_keyed_and_evaluation_is_read_only(tmp_path):
    path = tmp_path / "cache.json"
    calls = []
    def provider(texts):
        calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]
    prepared = HashedEmbeddingCache(path, model="m", dimensions=2,
                                    corpus_identity="snapshot-a", readonly=False,
                                    provider=provider)
    assert prepared(["alpha", "beta"])[0] == [5.0, 1.0]
    assert len(calls) == 1
    evaluated = HashedEmbeddingCache(path, model="m", dimensions=2,
                                     corpus_identity="snapshot-a", readonly=True)
    assert evaluated(["alpha"])[0] == [5.0, 1.0]
    with pytest.raises(EmbeddingCacheMiss):
        evaluated(["uncached query"])


def test_hard_distractors_are_reproducible_provenance_audited():
    available = records()
    cases = [DiscoveryCase("x", "alpha", ("a",), "test")]
    first = build_frozen_distractors(available, cases, count=10)
    assert first == build_frozen_distractors(available, cases, count=10)
    audit = audit_distractors(first, {"a"}, set(available) - {"a"})
    assert audit["record_count"] == audit["unique_description_count"] == 10
    assert all(row["audit"]["required_capability_satisfied"] is False for row in first)
