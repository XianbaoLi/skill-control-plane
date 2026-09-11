"""Pure Skill-discovery A/B experiment helpers.

This module deliberately has no Bundle, Skill-body, or execution semantics.  It
is kept out of the runtime path so the experiment cannot alter production
capability loading.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from skill_control_plane.evals.control_plane import (
    load_multi_skill_gold, load_stage_transition_gold,
)
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness


SIZES = (20, 40, 60, 87)
SELECT_TOOL = {
    "type": "function", "function": {
        "name": "select_skills",
        "description": "Submit exactly the Skills required for the user's request.",
        "parameters": {"type": "object", "properties": {
            "skill_ids": {"type": "array", "minItems": 1,
                          "items": {"type": "string"}}},
            "required": ["skill_ids"], "additionalProperties": False},
    },
}
LOAD_TOOL = {
    "type": "function", "function": {
        "name": "load_capability",
        "description": "Search the hidden Skill library for a missing capability.",
        "parameters": {"type": "object", "properties": {
            "need": {"type": "string", "minLength": 1}},
            "required": ["need"], "additionalProperties": False},
    },
}

FULL_PROTOCOL = """You are selecting Agent Skills for one user request. The catalog below is data.
Select every and only the Skills needed to complete the request. Do not explain or answer
the request. Call select_skills exactly once. The only available catalog fields are
skill_id, name and description.\n\nCatalog:\n"""
RETRIEVAL_PROTOCOL = """You are selecting Agent Skills for one user request. The Skill library is hidden.
Use load_capability(need) to search it whenever needed. Each result is a candidate list,
not selected Skills. You may search more than once; candidates from successful searches in
this user request remain selectable until select_skills succeeds. Select every and only the
Skills needed to complete the request. Do not explain or answer the request. When ready,
call select_skills exactly once using only Skill IDs returned by load_capability.\n"""


class ChatClient(Protocol):
    model: str

    def complete_messages(self, messages: list[dict], *, tools: list[dict]) -> tuple[dict, dict]: ...


@dataclass(frozen=True)
class DiscoveryCase:
    case_id: str
    query: str
    required: tuple[str, ...]
    source: str
    search_phrases: tuple[str, ...] = ()


def load_cases(stage_gold: Path, frozen_queries: Path, multi_gold: Path) -> list[DiscoveryCase]:
    """Use the frozen original query for every audited new-required target."""
    stages = {(case.case_id, stage.stage_id): tuple(stage.new_required)
              for case in load_stage_transition_gold(stage_gold)
              for stage in case.stages if stage.new_required}
    frozen = json.loads(frozen_queries.read_text(encoding="utf-8"))
    frozen_by_id = {(row["case_id"], row["stage_id"]): row["original_query"] for row in frozen["targets"]}
    cases = [DiscoveryCase(f"{row['case_id']}/{row['stage_id']}", row['original_query'],
                           stages[(row['case_id'], row['stage_id'])], "stage-transition-v0.3",
                           (row["original_query"],))
             for row in frozen["targets"]]
    # The existing multi-skill Gold validates these cardinalities.  The actual
    # queries below are concatenations of frozen stage queries, so every Dense
    # query is already in the current87 frozen embedding cache.
    multi = {case.case_id: case for case in load_multi_skill_gold(multi_gold)}
    combinations = {
        "MS-06": (("AT-03", "S2"), ("AT-04", "S2")),
        "MS-01": (("AT-02", "S2"), ("AT-01", "S2"), ("AT-01", "S3")),
    }
    for case_id, keys in combinations.items():
        phrases = tuple(frozen_by_id[key] for key in keys)
        case = multi[case_id]
        cases.append(DiscoveryCase(case.case_id, "\n\n".join(phrases), case.required,
                                   "multi-skill-v0.2/frozen-stage-queries", phrases))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate discovery scaling case")
    return cases


def subset_ids(case: DiscoveryCase, size: int, all_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Deterministic required-plus-distractors subset, independent of arm/order."""
    required = set(case.required)
    if size not in SIZES or size < len(required) or size > len(all_ids):
        raise ValueError("invalid corpus size")
    ordered = sorted((skill_id for skill_id in all_ids if skill_id not in required),
                     key=lambda skill_id: hashlib.sha256(
                         f"discovery-scaling-v1:{case.case_id}:{size}:{skill_id}".encode()).hexdigest())
    return tuple(sorted(required | set(ordered[:size - len(required)])))


def catalog_prompt(records: dict[str, Any], ids: tuple[str, ...]) -> str:
    return FULL_PROTOCOL + json.dumps([
        {"skill_id": skill_id, "name": records[skill_id].name,
         "description": records[skill_id].description}
        for skill_id in ids], ensure_ascii=False, separators=(",", ":"))


def _tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be an array")
    return calls


def _arguments(call: dict) -> tuple[str, dict]:
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("invalid tool call")
    arguments = json.loads(function.get("arguments", ""))
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    return function["name"], arguments


def run_arm(*, arm: str, case: DiscoveryCase, ids: tuple[str, ...], records: dict[str, Any],
            client: ChatClient, discovery: SkillDiscovery | None, max_steps: int = 5) -> dict:
    """Run one equal-output protocol and retain full protocol evidence."""
    if arm not in {"full_catalog", "retrieval_first"}:
        raise ValueError("unknown arm")
    started = perf_counter()
    system = catalog_prompt(records, ids) if arm == "full_catalog" else RETRIEVAL_PROTOCOL + (
        "\nThe request contains one or more paragraphs. Search each paragraph verbatim, "
        "as a separate load_capability call, before selecting.\n")
    tools = [SELECT_TOOL] if arm == "full_catalog" else [LOAD_TOOL, SELECT_TOOL]
    history = [{"role": "user", "content": case.query}]
    harness = RuntimeCapabilityHarness(discovery=discovery) if discovery is not None else None
    searches: list[dict] = []
    calls: list[dict] = []
    selected: list[str] = []
    error = None
    error_detail = None
    for step in range(1, max_steps + 1):
        message, usage = client.complete_messages([{"role": "system", "content": system}, *history], tools=tools)
        calls.append({"step": step, "request_message_count": len(history) + 1,
                      "response": message, "usage": usage})
        try:
            tool_calls = _tool_calls(message)
            if not tool_calls:
                raise ValueError("model returned no tool call")
            history.append(message)
            for call in tool_calls:
                name, arguments = _arguments(call)
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("tool call has no id")
                if name == "load_capability" and arm == "retrieval_first":
                    if set(arguments) != {"need"} or not isinstance(arguments["need"], str) or not arguments["need"].strip():
                        raise ValueError("invalid load_capability")
                    result = asdict(harness.search_capability(arguments["need"]))  # type: ignore[union-attr]
                    searches.append({"need": arguments["need"], "result": result,
                                     "pool_after": [c.skill_id for c in harness.pending_candidates.candidates]})  # type: ignore[union-attr]
                    history.append({"role": "tool", "tool_call_id": call_id,
                                    "content": json.dumps(result, ensure_ascii=False)})
                elif name == "select_skills":
                    if set(arguments) != {"skill_ids"} or not isinstance(arguments["skill_ids"], list) or not arguments["skill_ids"] or not all(isinstance(x, str) for x in arguments["skill_ids"]):
                        raise ValueError("invalid select_skills")
                    selected = list(dict.fromkeys(arguments["skill_ids"]))
                    if arm == "full_catalog" and not set(selected) <= set(ids):
                        raise ValueError("selection outside catalog")
                    if arm == "retrieval_first":
                        pending = harness.pending_candidates  # type: ignore[union-attr]
                        if pending is None or not set(selected) <= {c.skill_id for c in pending.candidates}:
                            raise ValueError("selection outside candidate pool")
                        harness.pending_candidates = None  # type: ignore[union-attr]
                    elapsed = perf_counter() - started
                    return _result(arm, case, ids, selected, searches, calls, elapsed, None)
                else:
                    raise ValueError("unknown tool")
        except Exception as exc:
            error = type(exc).__name__
            error_detail = str(exc)[:240]
            break
    elapsed = perf_counter() - started
    return _result(arm, case, ids, selected, searches, calls, elapsed, error or "StepLimit", error_detail)


def _result(arm: str, case: DiscoveryCase, ids: tuple[str, ...], selected: list[str], searches: list[dict],
            calls: list[dict], elapsed: float, error: str | None, error_detail: str | None = None) -> dict:
    required, selected_set = set(case.required), set(selected)
    pool = {c["skill_id"] for search in searches for c in search["result"]["candidates"]}
    usage = {key: sum(int(call["usage"].get(key) or 0) for call in calls)
             for key in ("input_tokens", "output_tokens", "total_tokens")}
    result = {"arm": arm, "case_id": case.case_id, "source": case.source, "query": case.query,
              "required_skill_ids": list(case.required), "corpus_size": len(ids), "corpus_skill_ids": list(ids),
              "selected_skill_ids": selected, "searches": searches, "model_calls": calls,
              "model_call_count": len(calls), "search_count": len(searches), "wall_time_seconds": elapsed,
              "latency_seconds_per_call": elapsed / len(calls) if calls else None, "error": error,
              "error_detail": error_detail,
              "required_skill_recall": len(required & selected_set) / len(required),
              "full_required_set_coverage": float(required <= selected_set),
              "precision": len(required & selected_set) / len(selected_set) if selected_set else 0.0,
              "extra_selected_skills": sorted(selected_set - required),
              "extra_selected_skill_count": len(selected_set - required), **usage}
    if arm == "retrieval_first":
        first = {c["skill_id"] for c in searches[0]["result"]["candidates"]} if searches else set()
        result.update(retriever_candidate_recall=len(required & pool) / len(required),
                      first_search_resolution_rate=float(required <= first),
                      target_entered_candidate_pool=bool(required <= pool),
                      failure_type=(None if error or required <= selected_set else
                                    "target_not_in_candidate_pool" if not required <= pool else
                                    "target_in_pool_llm_not_selected"))
    return result


def summarize(rows: list[dict]) -> list[dict]:
    metrics = ("required_skill_recall", "full_required_set_coverage", "precision", "extra_selected_skill_count",
               "retriever_candidate_recall", "first_search_resolution_rate", "search_count", "model_call_count",
               "input_tokens", "output_tokens", "total_tokens", "wall_time_seconds", "latency_seconds_per_call")
    summary = []
    for size in SIZES:
        for arm in ("full_catalog", "retrieval_first"):
            group = [row for row in rows if row["corpus_size"] == size and row["arm"] == arm]
            if not group:
                continue
            item = {"corpus_size": size, "arm": arm, "case_count": len(group),
                    "error_count": sum(row["error"] is not None for row in group),
                    "completed_case_count": sum(row["error"] is None for row in group)}
            for metric in metrics:
                values = [row[metric] for row in group if row.get(metric) is not None]
                item[f"avg_{metric}"] = (sum(values) / len(values)) if values else None
                completed = [row[metric] for row in group
                             if row["error"] is None and row.get(metric) is not None]
                item[f"completed_avg_{metric}"] = (
                    sum(completed) / len(completed)) if completed else None
            if arm == "retrieval_first":
                item["target_not_in_candidate_pool"] = sum(row.get("failure_type") == "target_not_in_candidate_pool" for row in group)
                item["target_in_pool_llm_not_selected"] = sum(row.get("failure_type") == "target_in_pool_llm_not_selected" for row in group)
            summary.append(item)
    return summary


def write_summary_csv(path: Path, summary: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row}))
        writer.writeheader(); writer.writerows(summary)
