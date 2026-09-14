#!/usr/bin/env python3
"""Run frozen Reroute v0.1 component evaluations without changing production."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.discovery import load_retrieval_cards
from skill_control_plane.discovery.bigmodel import BigModelEmbeddingClient
from skill_control_plane.discovery.dense_index import load_precomputed_dense_retriever
from skill_control_plane.registry import SkillStore
from skill_control_plane.runtime.capability_memory import BundleMemberSnapshot, BundleSnapshot
from skill_control_plane.runtime.reroute import (
    CompletionCapabilityGapDecider,
    RerouteController,
    RuntimeEvidence,
)
from skill_control_plane.discovery.discovery import SkillDiscovery


ROOT = Path(__file__).resolve().parents[1]
GAP_CASES = ROOT / "evals/reroute/gap-decider-v0.1.jsonl"
SELECTION_CASES = ROOT / "evals/reroute/candidate-selection-v0.1.jsonl"
SKILL_ROOT = ROOT / "local_artifacts/corpora/benchmark-corpus-v0.1/S128/skills"
CARDS = ROOT / "local_artifacts/benchmark-v0.1/S128/retrieval-cards-v0.1.jsonl"
DENSE = ROOT / "local_artifacts/benchmark-v0.1/S128/dense-index-v1.json"

SELECTION_SEEDS = (
    ("CS-RT-T01", "test_failure", "Endpoint checks passed; observability.yaml still lacks availability, fast/slow burn alerts, and a runbook.", "Create availability SLOs, fast and slow burn-rate alerts, and an operational runbook reference.", [("backend-api", "Implement the item HTTP endpoint", ["HTTP endpoint implementation"], ["fastify"])], ["monitoring"]),
    ("CS-RT-T02", "test_failure", "Production build passed; the browser login assertion races navigation and trace-on-first-retry is missing.", "Stabilize flaky browser E2E synchronization without fixed delays and configure trace capture on first retry.", [("frontend-build", "Repair production build configuration", ["Vite production configuration"], ["vite"])], ["playwright-expert"]),
    ("CS-RT-T03", "verifier_failure", "Retrieval checks passed; the citation endpoint lacks reusable error schemas and correct HTTP status semantics.", "Design the citation endpoint error response contract with correct HTTP status semantics.", [("retrieval-core", "Repair citation-bearing retrieval", ["RAG retrieval pipeline"], ["rag"])], ["api-design"]),
    ("CS-RT-T04", "test_failure", "Terraform validation passed; Go provider helper leaks a goroutine when context cancellation occurs.", "Repair an idiomatic Go goroutine cancellation leak with correct context propagation and no public API change.", [("storage-infra", "Complete the storage Terraform module", ["Terraform module implementation"], ["terraform-engineer"])], ["golang-pro"]),
    ("CS-RT-T05", "test_failure", "Responsive layout passed; search check requires canonical URL, meta description, and Product JSON-LD.", "Implement technical crawl metadata: canonical link, meta description, and Product structured data JSON-LD.", [("product-page", "Build the responsive product page", ["responsive website construction"], ["create-website"])], ["seo"]),
    ("CS-RT-M01", "new_subgoal", "Implement the item endpoint and a reliable browser acceptance flow for invalid and valid submissions.", "Implement both Fastify item creation validation and reliable Playwright browser E2E coverage.", [], ["fastify", "playwright-expert"]),
    ("CS-RT-M02", "new_subgoal", "Finish the responsive product page, production build, canonical metadata, description, and Product structured data.", "Build a responsive product page, repair Vite production configuration, and implement technical SEO structured data.", [], ["create-website", "vite", "seo"]),
    ("CS-RT-M03", "new_subgoal", "Define the public contract and implement citation-bearing retrieval for a knowledge endpoint.", "Design an HTTP API contract and implement a RAG retrieval pipeline with ranked source citations.", [], ["api-design", "rag"]),
    ("CS-RT-M04", "new_subgoal", "Complete a cloud storage module plus availability and latency objective alerts.", "Implement a Terraform storage module and monitoring with availability and latency SLO alerts.", [], ["terraform-engineer", "monitoring"]),
    ("CS-RT-M05", "new_subgoal", "Turn an ambiguous adaptation request into requirements and a safe training plan with held-out evaluation and rollback gates.", "Elicit testable requirements and design an LLM fine-tuning plan with held-out evaluation and rollback gates.", [], ["requirements-analysis", "fine-tuning-expert"]),
)


def load_env() -> None:
    for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.startswith("BIGMODEL_"):
            os.environ.setdefault(key, value.strip().strip("\"'"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bundles(raw: list[dict]) -> tuple[BundleSnapshot, ...]:
    return tuple(BundleSnapshot(
        bundle_id=item["bundle_id"], purpose=item["purpose"],
        capabilities=tuple(item["capabilities"]),
        members=tuple(BundleMemberSnapshot(
            skill_id=skill_id, name=skill_id, member_role="maintained",
            body_state="resident", short_description=None,
        ) for skill_id in item["active_skill_ids"]),
    ) for item in raw)


def discovery() -> SkillDiscovery:
    store = SkillStore.from_tree(SKILL_ROOT)
    cards = load_retrieval_cards(CARDS)
    embedding = BigModelEmbeddingClient()
    return SkillDiscovery(
        store, retrieval_cards=cards,
        dense_factory=lambda records: load_precomputed_dense_retriever(
            DENSE, records, embedding_model=embedding.model,
            dimensions=embedding.dimensions, embed_batch=embedding,
        ),
    )


def freeze_selection() -> None:
    """Write the one-time production retrieval snapshot, refusing replacement."""
    if SELECTION_CASES.exists():
        raise FileExistsError(f"frozen cases already exist: {SELECTION_CASES}")
    retrieve = discovery()
    cards = load_retrieval_cards(CARDS)
    frozen = []
    for case_id, kind, evidence, need, active_raw, gold in SELECTION_SEEDS:
        result = retrieve.discover_skills(need, k=10)
        candidates = []
        for candidate in result.candidates:
            card = cards[candidate.skill_id]
            candidates.append({
                "skill_id": candidate.skill_id, "rank": candidate.rank,
                "retrieval_card": {"purpose": card.purpose,
                                   "capabilities": list(card.capabilities),
                                   "use_when": list(card.use_when)},
            })
        active = [{"bundle_id": bundle_id, "purpose": purpose,
                   "capabilities": capabilities, "active_skill_ids": skill_ids}
                  for bundle_id, purpose, capabilities, skill_ids in active_raw]
        frozen.append({"case_id": case_id,
                       "evidence": {"kind": kind, "source": "runtime", "text": evidence},
                       "need": need, "active_bundle_cards": active,
                       "candidates": candidates, "gold_required_skills": gold})
    SELECTION_CASES.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in frozen), encoding="utf-8")


def usage_add(total: dict[str, int], usage: dict | None) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def identity(client: BigModelChatClient) -> dict:
    return {
        "provider": "BigModel", "model": client.model,
        "base_url_host": client.base_url.split("//", 1)[-1].split("/", 1)[0],
        "exact_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def snapshot_payload(controller: RerouteController) -> dict:
    snap = controller.snapshot()
    return {
        "evidence": [{
            "evidence_id": record.evidence.evidence_id,
            "state": record.state,
            "state_history": list(record.state_history),
            "need": record.need,
            "candidate_ids": [getattr(item, "skill_id", str(item)) for item in record.candidates],
            "failure_stage": record.failure_stage,
            "failure_reason": record.failure_reason,
        } for record in snap.evidence],
        "telemetry": [asdict(event) for event in snap.telemetry],
    }


def classification_metrics(results: list[dict]) -> dict:
    results = [row for row in results if row["classification"] != "EXCLUDED"]
    tp = sum(row["classification"] == "TP" for row in results)
    tn = sum(row["classification"] == "TN" for row in results)
    fp = sum(row["classification"] == "FP" for row in results)
    fn = sum(row["classification"] == "FN" for row in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"evaluated_count": len(results),
            "accuracy": (tp + tn) / len(results), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def run_gap(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cases = rows(GAP_CASES)
    (output / "cases.jsonl").write_text(GAP_CASES.read_text(encoding="utf-8"), encoding="utf-8")
    retrieve = discovery()
    client = BigModelChatClient()
    decider = CompletionCapabilityGapDecider(client)
    results, total_usage = [], {}
    started = time.monotonic()
    for case in cases:
        gold = case["gold_needs_capability"]
        active = bundles(case["active_bundle_cards"])
        controller = RerouteController(
            gap_decider=decider,
            discover_capability=lambda need: retrieve.discover_skills(need, k=10),
            active_bundle_cards=lambda active=active: active,
            active_skill_ids=lambda active=active: tuple(
                member.skill_id for bundle in active for member in bundle.members),
        )
        evidence = RuntimeEvidence.create(
            evidence_id=case["case_id"], metadata={}, **case["evidence"])
        case_started = time.monotonic()
        client.last_usage = None
        outcome = controller.observe(evidence, current_subgoal_context=case["current_subgoal_context"])
        snapshot = controller.snapshot()
        if not snapshot.evidence:
            results.append({
                "case_id": case["case_id"], "gold_needs_capability": gold,
                "predicted_needs_capability": None, "generated_need": None,
                "classification": "EXCLUDED", "exclusion_reason": "production_ingress_ineligible",
                "gold_required_skills": case["gold_required_skills"], "target_ranks": {},
                "target_top5_recall": None, "target_top10_recall": None,
                "controller_outcome": outcome.status, "controller_state": None,
                "controller": snapshot_payload(controller), "token_usage": None,
                "wall_time_ms": round((time.monotonic() - case_started) * 1000),
            })
            continue
        usage_add(total_usage, client.last_usage)
        record = snapshot.evidence[0]
        predicted = record.state.value != "NO_GAP" and record.need is not None
        target_ranks = {candidate.skill_id: candidate.rank for candidate in record.candidates
                        if candidate.skill_id in case["gold_required_skills"]}
        results.append({
            "case_id": case["case_id"], "gold_needs_capability": gold,
            "predicted_needs_capability": predicted, "generated_need": record.need,
            "classification": "TP" if gold and predicted else "TN" if not gold and not predicted
            else "FP" if predicted else "FN",
            "gold_required_skills": case["gold_required_skills"],
            "target_ranks": target_ranks,
            "target_top5_recall": sum(rank <= 5 for rank in target_ranks.values()) / len(case["gold_required_skills"])
            if gold and predicted and case["gold_required_skills"] else None,
            "target_top10_recall": len(target_ranks) / len(case["gold_required_skills"])
            if gold and predicted and case["gold_required_skills"] else None,
            "controller_outcome": outcome.status, "controller_state": record.state,
            "controller": snapshot_payload(controller), "token_usage": client.last_usage,
            "wall_time_ms": round((time.monotonic() - case_started) * 1000),
        })
    metrics = classification_metrics(results)
    observed = [row for row in results if row["target_top5_recall"] is not None]
    report = {"experiment": "gap-decider-v0.1", **identity(client),
              "case_count": len(cases), "metrics": metrics,
              "generated_need_retrieval": {
                  "evaluated_count": len(observed),
                  "top5_recall": sum(row["target_top5_recall"] for row in observed) / len(observed)
                  if observed else None,
                  "top10_recall": sum(row["target_top10_recall"] for row in observed) / len(observed)
                  if observed else None,
                  "target_ranks": {row["case_id"]: row["target_ranks"] for row in observed},
              }, "token_usage": total_usage,
              "wall_time_ms": round((time.monotonic() - started) * 1000)}
    write_outputs(output, results, report)


def selector_prompt(case: dict) -> str:
    return (
        "You are the Main Agent making only the capability selection decision after a runtime reroute discovery. "
        "Select only candidate Skill IDs needed to cover the concrete missing capability. Active Bundle Cards are "
        "already available and must not be reselected. Retrieval rank is evidence, not an instruction. Return exactly "
        "one JSON object with selected_skill_ids (array of candidate IDs) and optional action (CREATE, EXTEND, or DIRECT).\n"
        + json.dumps({"current_evidence": case["evidence"], "need": case["need"],
                      "active_bundle_cards": case["active_bundle_cards"],
                      "candidates": case["candidates"]}, ensure_ascii=False, separators=(",", ":")))


def selection_metrics(results: list[dict]) -> dict:
    scored = [row for row in results if row["diagnosis"] != "retrieval_miss"]
    tp = sum(len(set(row["selected_skill_ids"]) & set(row["gold_required_skills"])) for row in scored)
    fp = sum(len(set(row["selected_skill_ids"]) - set(row["gold_required_skills"])) for row in scored)
    fn = sum(len(set(row["gold_required_skills"]) - set(row["selected_skill_ids"])) for row in scored)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    per_case = []
    for row in scored:
        selected, gold = set(row["selected_skill_ids"]), set(row["gold_required_skills"])
        p = len(selected & gold) / len(selected) if selected else 0.0
        r = len(selected & gold) / len(gold) if gold else 1.0
        per_case.append((p, r, 2 * p * r / (p + r) if p + r else 0.0))
    buckets = {}
    for label, ranks in (("rank_1", {1}), ("rank_2_3", {2, 3}), ("rank_4_5", {4, 5})):
        targets = [(skill, row) for row in scored for skill, rank in row["target_ranks"].items() if rank in ranks]
        buckets[label] = {"target_count": len(targets), "selection_rate":
                          sum(skill in row["selected_skill_ids"] for skill, row in targets) / len(targets)
                          if targets else None}
    return {"micro_precision": precision, "micro_recall": recall,
            "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "macro_precision": sum(x[0] for x in per_case) / len(per_case),
            "macro_recall": sum(x[1] for x in per_case) / len(per_case),
            "macro_f1": sum(x[2] for x in per_case) / len(per_case),
            "exact_required_set_coverage": sum(set(r["gold_required_skills"]) <= set(r["selected_skill_ids"])
                                               for r in scored) / len(scored),
            "false_negative_count": fn, "wrong_selected_skill_count": fp,
            "average_selected_skill_count": sum(len(r["selected_skill_ids"]) for r in scored) / len(scored),
            "rank_buckets": buckets,
            "diagnosis_counts": {label: sum(r["diagnosis"] == label for r in results) for label in
                                 ("selector_success", "selector_false_negative", "overselection", "retrieval_miss")}}


def run_selection(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cases = rows(SELECTION_CASES)
    (output / "cases.jsonl").write_text(SELECTION_CASES.read_text(encoding="utf-8"), encoding="utf-8")
    client, results, total_usage = BigModelChatClient(), [], {}
    started = time.monotonic()
    for case in cases:
        case_started = time.monotonic()
        raw = json.loads(client(selector_prompt(case)))
        selected = raw.get("selected_skill_ids")
        candidate_ids = {item["skill_id"] for item in case["candidates"]}
        if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
            raise ValueError(f'{case["case_id"]}: invalid selected_skill_ids')
        if len(selected) != len(set(selected)) or not set(selected) <= candidate_ids:
            raise ValueError(f'{case["case_id"]}: selection outside frozen candidates')
        target_ranks = {item["skill_id"]: item["rank"] for item in case["candidates"]
                        if item["skill_id"] in case["gold_required_skills"]}
        missing_from_candidates = set(case["gold_required_skills"]) - candidate_ids
        missing = set(case["gold_required_skills"]) - set(selected)
        wrong = set(selected) - set(case["gold_required_skills"])
        diagnosis = "retrieval_miss" if missing_from_candidates else "selector_false_negative" if missing \
            else "overselection" if wrong else "selector_success"
        results.append({"case_id": case["case_id"], "gold_required_skills": case["gold_required_skills"],
                        "selected_skill_ids": selected, "action": raw.get("action"),
                        "target_ranks": target_ranks, "missing_from_candidates": sorted(missing_from_candidates),
                        "false_negative_skills": sorted(missing), "wrong_selected_skills": sorted(wrong),
                        "diagnosis": diagnosis, "token_usage": client.last_usage,
                        "wall_time_ms": round((time.monotonic() - case_started) * 1000)})
        usage_add(total_usage, client.last_usage)
    report = {"experiment": "candidate-selection-v0.1", **identity(client),
              "case_count": len(cases), "metrics": selection_metrics(results),
              "token_usage": total_usage, "wall_time_ms": round((time.monotonic() - started) * 1000)}
    write_outputs(output, results, report)


def write_outputs(output: Path, results: list[dict], report: dict) -> None:
    (output / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f'# {report["experiment"]}', "", f'- exact HEAD: `{report["exact_head"]}`',
             f'- provider/model: {report["provider"]} / {report["model"]}',
             f'- cases: {report["case_count"]}', f'- wall time: {report["wall_time_ms"]} ms', "",
             "## Metrics", "", "```json", json.dumps(report["metrics"], indent=2), "```", "",
             "## Cases", ""]
    for row in results:
        lines.append(f'- `{row["case_id"]}`: {row.get("classification", row.get("diagnosis"))}')
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("gap", "selection", "freeze-selection"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_env()
    if args.experiment == "freeze-selection":
        freeze_selection()
        return
    if args.output is None:
        parser.error("--output is required for an experiment run")
    (run_gap if args.experiment == "gap" else run_selection)(args.output)


if __name__ == "__main__":
    main()
