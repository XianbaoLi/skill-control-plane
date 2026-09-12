"""Pure Skill-discovery A/B experiment helpers.

This module deliberately has no Bundle, Skill-body, or execution semantics.  It
is kept out of the runtime path so the experiment cannot alter production
capability loading.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from skill_control_plane.evals.control_plane import (
    load_multi_skill_gold, load_stage_transition_gold,
)
from skill_control_plane.models import SkillRecord
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness


SIZES = (20, 40, 60, 87, 150, 300, 500, 1000)
DEFAULT_SIZES = SIZES[:-1]
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
        "description": "Search the hidden Skill library. The need value must copy one complete user-request paragraph verbatim; do not shorten or paraphrase it.",
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
call select_skills exactly once using only Skill IDs returned by load_capability.
For every load_capability call, copy one complete user-request paragraph verbatim into
need. Never shorten, extract, rewrite or paraphrase the paragraph.\n"""


class ChatClient(Protocol):
    model: str

    def complete_messages(self, messages: list[dict], *, tools: list[dict]) -> tuple[dict, dict]: ...


@dataclass(frozen=True)
class DiscoveryCase:
    case_id: str
    query: str
    required: tuple[str, ...]
    source: str
    case_type: str = "single_skill_hard_paraphrase"
    search_phrases: tuple[str, ...] = ()


class InfrastructureError(RuntimeError):
    """Experiment setup/provider failures that must not become semantic misses."""


class EmbeddingCacheMiss(InfrastructureError):
    pass


class HashedEmbeddingCache:
    """Snapshot-bound persistent embedding cache with explicit read-only evaluation."""

    VERSION = "discovery-scaling-embedding-cache-v0.2"

    def __init__(self, path: Path, *, model: str, dimensions: int,
                 corpus_identity: str, readonly: bool,
                 provider: Callable[[Sequence[str]], list[list[float]]] | None = None):
        self.path = path
        self.model = model
        self.dimensions = dimensions
        self.corpus_identity = corpus_identity
        self.readonly = readonly
        self.provider = provider
        self.hits = 0
        self.misses = 0
        self.provider_calls = 0
        self.provider_seconds = 0.0
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            expected = {"version": self.VERSION, "model": model,
                        "dimensions": dimensions, "corpus_identity": corpus_identity}
            if any(self.data.get(key) != value for key, value in expected.items()):
                raise InfrastructureError("embedding cache identity/configuration mismatch")
        elif readonly:
            raise EmbeddingCacheMiss(f"embedding cache does not exist: {path}")
        else:
            self.data = {"version": self.VERSION, "model": model,
                         "dimensions": dimensions, "corpus_identity": corpus_identity,
                         "entries": {}, "preparation": {}}

    def _key(self, text: str) -> str:
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        material = json.dumps({"model": self.model, "dimensions": self.dimensions,
                               "text_sha256": text_sha,
                               "corpus_identity": self.corpus_identity},
                              sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        unique_missing = list(dict.fromkeys(
            text for text in texts if self._key(text) not in self.data["entries"]))
        self.hits += len(texts) - len(unique_missing)
        self.misses += len(unique_missing)
        if unique_missing:
            if self.readonly:
                hashes = [hashlib.sha256(text.encode()).hexdigest() for text in unique_missing]
                raise EmbeddingCacheMiss(f"read-only embedding cache missing text SHA256: {hashes}")
            if self.provider is None:
                raise InfrastructureError("embedding preparation requires a provider")
            started = perf_counter()
            try:
                vectors = self.provider(unique_missing)
            except Exception as exc:
                raise InfrastructureError(
                    f"embedding provider failed during preparation: {type(exc).__name__}") from exc
            self.provider_seconds += perf_counter() - started
            self.provider_calls += 1
            if len(vectors) != len(unique_missing):
                raise InfrastructureError("embedding provider returned wrong vector count")
            for text, vector in zip(unique_missing, vectors, strict=True):
                if len(vector) != self.dimensions:
                    raise InfrastructureError("embedding dimensions mismatch")
                text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self.data["entries"][self._key(text)] = {
                    "model": self.model, "dimensions": self.dimensions,
                    "text_sha256": text_sha, "corpus_identity": self.corpus_identity,
                    "vector": vector,
                }
            self.flush()
        return [self.data["entries"][self._key(text)]["vector"] for text in texts]

    def flush(self) -> None:
        if self.readonly:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["preparation"] = {
            "provider_calls": self.provider_calls,
            "provider_seconds": self.provider_seconds,
            "entry_count": len(self.data["entries"]),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def verify_complete(self, texts: Sequence[str]) -> None:
        missing = [hashlib.sha256(text.encode()).hexdigest() for text in dict.fromkeys(texts)
                   if self._key(text) not in self.data["entries"]]
        if missing:
            raise EmbeddingCacheMiss(f"embedding cache incomplete: {missing}")


def load_cases(stage_gold: Path, frozen_queries: Path, multi_gold: Path) -> list[DiscoveryCase]:
    """Load three frozen variants per stage target plus existing multi-skill Gold."""
    stages = {(case.case_id, stage.stage_id): tuple(stage.new_required)
              for case in load_stage_transition_gold(stage_gold)
              for stage in case.stages if stage.new_required}
    frozen = json.loads(frozen_queries.read_text(encoding="utf-8"))
    frozen_by_id = {(row["case_id"], row["stage_id"]): row["original_query"] for row in frozen["targets"]}
    cases = []
    case_types = ("stage_transition_wording", "single_skill_hard_paraphrase",
                  "ambiguous_capability_wording")
    for row in frozen["targets"]:
        for variant_index, case_type in enumerate(case_types):
            query = row["variants"][variant_index]
            cases.append(DiscoveryCase(
                f"{row['case_id']}/{row['stage_id']}/V{variant_index}", query,
                stages[(row['case_id'], row['stage_id'])],
                "robustness-current87-13target-65query", case_type, (query,)))
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
                                   "multi-skill-v0.2", "multi_skill", phrases))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate discovery scaling case")
    return cases


def subset_ids(case: DiscoveryCase, size: int, all_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Deterministic required-plus-distractors subset, independent of arm/order."""
    required = set(case.required)
    if size not in SIZES or size < len(required) or size > len(all_ids):
        raise ValueError("invalid corpus size")
    real = tuple(skill_id for skill_id in all_ids if not skill_id.startswith("synthetic-near-miss-"))
    synthetic = tuple(skill_id for skill_id in all_ids if skill_id.startswith("synthetic-near-miss-"))
    pool = real if size <= len(real) else real + synthetic
    ordered = sorted((skill_id for skill_id in pool if skill_id not in required),
                     key=lambda skill_id: hashlib.sha256(
                         f"discovery-scaling-v1:{case.case_id}:{size}:{skill_id}".encode()).hexdigest())
    if size > len(real):
        ordered = [skill_id for skill_id in real if skill_id not in required] + list(synthetic)
    return tuple(sorted(required | set(ordered[:size - len(required)])))


def build_frozen_distractors(records: dict[str, SkillRecord], cases: Sequence[DiscoveryCase],
                             *, count: int = 913,
                             seed: str = "discovery-scaling-v0.2") -> list[dict]:
    """Create deterministic, explicitly capability-negated near-miss records."""
    gold_ids = sorted({skill_id for case in cases for skill_id in case.required})
    source_ids = sorted(set(records) - set(gold_ids))
    if not source_ids or not gold_ids:
        raise ValueError("distractor generation requires gold and non-gold real Skills")
    rows = []
    for index in range(count):
        target_id = gold_ids[index % len(gold_ids)]
        offset = int(hashlib.sha256(f"{seed}:{index}".encode()).hexdigest(), 16)
        source_id = source_ids[offset % len(source_ids)]
        target, source = records[target_id], records[source_id]
        qualifier = ("terminology reference", "planning checklist", "read-only overview")[index % 3]
        description = (
            f"{qualifier.title()} {index + 1:04d} near {target.name}: discusses {target.description.rstrip('.')} "
            f"but cannot perform, execute, or guide that capability; adjacent material: "
            f"{source.description.rstrip('.')}.")
        rows.append({
            "skill_id": f"synthetic-near-miss-{index + 1:04d}",
            "name": f"{target.name} {qualifier} {index + 1:04d}",
            "description": description,
            "retrieval_representation": "\n".join((
                target.name, description,
                f"near-miss lexical context only: {target.retrieval_representation}",
                "capability exclusion: reference-only; cannot perform, execute, or guide the target capability.",
            )),
            "provenance": {
                "kind": "frozen_semantic_near_miss",
                "seed": seed,
                "target_skill_id": target_id,
                "source_non_gold_skill_id": source_id,
                "generation_rule": "target lexical context + non-gold adjacent source + explicit capability negation",
            },
            "audit": {
                "status": "excluded_by_construction",
                "required_capability_satisfied": False,
                "reason": "The record is reference-only and explicitly denies performance or guidance of the target capability.",
            },
        })
    audit_distractors(rows, set(gold_ids), set(source_ids))
    return rows


def audit_distractors(rows: Sequence[dict], gold_ids: set[str],
                      non_gold_ids: set[str]) -> dict:
    ids, descriptions = set(), set()
    by_target: dict[str, int] = {}
    for row in rows:
        skill_id, description = row.get("skill_id"), row.get("description")
        provenance, audit = row.get("provenance", {}), row.get("audit", {})
        target, source = provenance.get("target_skill_id"), provenance.get("source_non_gold_skill_id")
        if (not isinstance(skill_id, str) or not skill_id.startswith("synthetic-near-miss-")
                or skill_id in ids or not isinstance(description, str) or description in descriptions):
            raise ValueError("invalid, duplicate, or non-frozen synthetic distractor")
        if target not in gold_ids or source not in non_gold_ids or source in gold_ids:
            raise ValueError("distractor provenance is not gold/non-gold separated")
        if "cannot perform, execute, or guide" not in description:
            raise ValueError("synthetic distractor lacks capability exclusion")
        if audit.get("required_capability_satisfied") is not False:
            raise ValueError("synthetic distractor audit did not exclude target capability")
        ids.add(skill_id); descriptions.add(description)
        by_target[target] = by_target.get(target, 0) + 1
    return {"record_count": len(rows), "unique_description_count": len(descriptions),
            "target_distribution": dict(sorted(by_target.items())),
            "source_policy": "real non-gold Skill only",
            "capability_audit": "all records explicitly exclude the target capability"}


def load_frozen_distractors(path: Path, records: dict[str, SkillRecord],
                            cases: Sequence[DiscoveryCase]) -> tuple[dict[str, SkillRecord], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    gold_ids = {skill_id for case in cases for skill_id in case.required}
    audit = audit_distractors(rows, gold_ids, set(records) - gold_ids)
    loaded = dict(records)
    for row in rows:
        loaded[row["skill_id"]] = SkillRecord(
            row["skill_id"], row["name"], row["description"], "",
            str(path), retrieval_representation=row["retrieval_representation"])
    return loaded, audit


def corpus_identity(records: dict[str, SkillRecord]) -> str:
    material = [{"skill_id": skill_id,
                 "retrieval_sha256": hashlib.sha256(
                     record.retrieval_representation.encode()).hexdigest()}
                for skill_id, record in sorted(records.items())]
    return hashlib.sha256(json.dumps(material, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


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
    failure_type = None
    retrieval_seconds = 0.0
    for step in range(1, max_steps + 1):
        try:
            message, usage = client.complete_messages(
                [{"role": "system", "content": system}, *history], tools=tools)
            calls.append({"step": step, "request_message_count": len(history) + 1,
                          "response": message, "usage": usage})
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
                    allowed_queries = case.search_phrases or (case.query,)
                    if arguments["need"] not in allowed_queries:
                        raise ValueError("load_capability need must be one frozen user paragraph verbatim")
                    retrieval_started = perf_counter()
                    internal = harness.search_capability(arguments["need"])  # type: ignore[union-attr]
                    retrieval_seconds += perf_counter() - retrieval_started
                    result = harness.model_visible_candidates(internal)  # type: ignore[union-attr]
                    searches.append({"need": arguments["need"],
                                     "model_visible_payload": result,
                                     "internal_retrieval_record": asdict(internal),
                                     "pool_after": [c.skill_id for c in harness.pending_candidates.candidates]})  # type: ignore[union-attr]
                    history.append({"role": "tool", "tool_call_id": call_id,
                                    "content": json.dumps(result, ensure_ascii=False)})
                elif name == "select_skills":
                    if set(arguments) != {"skill_ids"} or not isinstance(arguments["skill_ids"], list) or not arguments["skill_ids"] or not all(isinstance(x, str) for x in arguments["skill_ids"]):
                        raise ValueError("invalid select_skills")
                    proposed = list(dict.fromkeys(arguments["skill_ids"]))
                    if arm == "full_catalog" and not set(proposed) <= set(ids):
                        raise ValueError("selection outside catalog")
                    if arm == "retrieval_first":
                        pending = harness.pending_candidates  # type: ignore[union-attr]
                        if pending is None or not set(proposed) <= {c.skill_id for c in pending.candidates}:
                            raise ValueError("selection outside candidate pool")
                        harness.pending_candidates = None  # type: ignore[union-attr]
                    selected = proposed
                    elapsed = perf_counter() - started
                    return _result(arm, case, ids, selected, searches, calls, elapsed,
                                   retrieval_seconds, None, None, None)
                else:
                    raise ValueError("unknown tool")
        except InfrastructureError as exc:
            error = type(exc).__name__
            error_detail = str(exc)[:500]
            failure_type = "infra_error"
            break
        except Exception as exc:
            error = type(exc).__name__
            error_detail = str(exc)[:240]
            failure_type = "protocol_error"
            break
    elapsed = perf_counter() - started
    if error is None:
        error, failure_type = "StepLimit", "llm_selection_miss"
    return _result(arm, case, ids, selected, searches, calls, elapsed,
                   retrieval_seconds, error, error_detail, failure_type)


def _result(arm: str, case: DiscoveryCase, ids: tuple[str, ...], selected: list[str], searches: list[dict],
            calls: list[dict], elapsed: float, retrieval_seconds: float,
            error: str | None, error_detail: str | None,
            failure_type: str | None) -> dict:
    required, selected_set = set(case.required), set(selected)
    pool = {c["skill_id"] for search in searches
            for c in search["model_visible_payload"]["candidates"]}
    usage = {key: sum(int(call["usage"].get(key) or 0) for call in calls)
             for key in ("input_tokens", "output_tokens", "total_tokens")}
    infra_error = failure_type == "infra_error"
    result = {"arm": arm, "case_id": case.case_id, "case_type": case.case_type,
              "source": case.source, "query": case.query,
              "required_skill_ids": list(case.required), "corpus_size": len(ids), "corpus_skill_ids": list(ids),
              "selected_skill_ids": selected, "searches": searches, "model_calls": calls,
              "model_call_count": len(calls), "search_count": len(searches), "wall_time_seconds": elapsed,
              "latency_seconds_per_call": elapsed / len(calls) if calls else None, "error": error,
              "error_detail": error_detail, "infra_error": infra_error,
              "retrieval_latency_seconds": retrieval_seconds,
              "model_latency_seconds": sum(float(call["usage"].get("provider_latency_seconds") or 0)
                                           for call in calls),
              "required_skill_recall": None if infra_error else len(required & selected_set) / len(required),
              "full_required_set_coverage": None if infra_error else float(required <= selected_set),
              "precision": None if infra_error else (
                  len(required & selected_set) / len(selected_set) if selected_set else 0.0),
              "extra_selected_skills": sorted(selected_set - required),
              "extra_selected_skill_count": len(selected_set - required), **usage}
    if arm == "retrieval_first":
        first = {c["skill_id"] for c in searches[0]["model_visible_payload"]["candidates"]} if searches else set()
        semantic_failure = (None if infra_error or required <= selected_set else
                            "retrieval_miss" if not required <= pool else
                            "llm_selection_miss")
        result.update(retriever_candidate_recall=None if infra_error else len(required & pool) / len(required),
                      first_search_resolution_rate=None if infra_error else float(required <= first),
                      target_entered_candidate_pool=None if infra_error else bool(required <= pool),
                      failure_type=(failure_type if infra_error else
                                    semantic_failure or failure_type),
                      target_not_in_candidate_pool=semantic_failure == "retrieval_miss",
                      target_in_pool_llm_not_selected=semantic_failure == "llm_selection_miss")
    else:
        result["failure_type"] = failure_type or (
            None if required <= selected_set else "llm_selection_miss")
    return result


def summarize(rows: list[dict]) -> list[dict]:
    metrics = ("required_skill_recall", "full_required_set_coverage", "precision", "extra_selected_skill_count",
               "retriever_candidate_recall", "first_search_resolution_rate", "search_count", "model_call_count",
               "input_tokens", "output_tokens", "total_tokens", "wall_time_seconds", "latency_seconds_per_call",
               "retrieval_latency_seconds", "model_latency_seconds")
    summary = []
    for size in sorted({row["corpus_size"] for row in rows}):
        for arm in ("full_catalog", "retrieval_first"):
            group = [row for row in rows if row["corpus_size"] == size and row["arm"] == arm]
            if not group:
                continue
            evaluated = [row for row in group if not row.get("infra_error")]
            item = {"corpus_size": size, "arm": arm, "case_count": len(group),
                    "evaluated_case_count": len(evaluated),
                    "infra_error_count": sum(bool(row.get("infra_error")) for row in group),
                    "error_count": sum(row["error"] is not None for row in group),
                    "completed_case_count": sum(row["error"] is None for row in group)}
            for metric in metrics:
                values = [row[metric] for row in evaluated if row.get(metric) is not None]
                item[f"avg_{metric}"] = (sum(values) / len(values)) if values else None
                completed = [row[metric] for row in group
                             if row["error"] is None and row.get(metric) is not None]
                item[f"completed_avg_{metric}"] = (
                    sum(completed) / len(completed)) if completed else None
            if arm == "retrieval_first":
                item["target_not_in_candidate_pool"] = sum(bool(row.get("target_not_in_candidate_pool")) for row in evaluated)
                item["target_in_pool_llm_not_selected"] = sum(bool(row.get("target_in_pool_llm_not_selected")) for row in evaluated)
            summary.append(item)
    return summary


def write_summary_csv(path: Path, summary: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row}))
        writer.writeheader(); writer.writerows(summary)
