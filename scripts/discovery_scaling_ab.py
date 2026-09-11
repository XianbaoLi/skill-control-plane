"""Prepare and evaluate the hard Full Catalog vs Retrieval First scaling A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.evals.discovery_scaling import (
    DEFAULT_SIZES, FULL_PROTOCOL, RETRIEVAL_PROTOCOL, HashedEmbeddingCache,
    InfrastructureError, audit_distractors, build_frozen_distractors,
    corpus_identity, load_cases, load_frozen_distractors, run_arm, subset_ids,
    summarize, write_summary_csv,
)
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.bigmodel import BigModelDenseRetriever, BigModelEmbeddingClient
from skill_control_plane.retrieval.cards import load_retrieval_cards
from skill_control_plane.retrieval.discovery import SkillDiscovery

ROOT = Path("local_artifacts/v0.5/hermes-current87")
MANIFEST = Path("local_artifacts/v0.5/hermes-current87-manifest.json")
CARDS = Path("local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl")
FROZEN = Path("local_artifacts/v0.6/robustness-current87-13target-65query-queries.json")
STAGE_GOLD = Path("evals/gold/stage-transition-v0.3.jsonl")
MULTI_GOLD = Path("evals/gold/multi-skill-v0.2.jsonl")
DISTRACTORS = Path("evals/fixtures/discovery-scaling-hard-distractors-v0.2.jsonl")
DEFAULT_CACHE = Path("local_artifacts/discovery-scaling/embedding-cache-v0.2.json")
LEGACY_CACHE = Path("local_artifacts/v0.6/robustness-current87-13target-65query-embeddings.json")


def load_environment() -> None:
    if not Path(".env").exists():
        return
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip().startswith(("BIGMODEL_", "PARATERA_")):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class MeteredGLM:
    def __init__(self, client: BigModelChatClient):
        self.client = client
        self.model = str(client.model)

    def complete_messages(self, messages, *, tools):
        payload = json.dumps({"model": self.client.model, "messages": messages, "tools": tools,
            "tool_choice": "auto", "stream": False, "thinking": {"type": "enabled"},
            "reasoning_effort": self.client.reasoning_effort, "do_sample": False,
            "max_tokens": self.client.max_tokens}, ensure_ascii=False).encode("utf-8")
        request = Request(f"{self.client.base_url}/chat/completions", data=payload,
                          headers={"Authorization": f"Bearer {self.client.api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        started = perf_counter()
        try:
            with urlopen(request, timeout=self.client.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise InfrastructureError(f"chat provider failed: {type(exc).__name__}") from exc
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0].get("message"), dict):
            raise InfrastructureError("chat provider response shape mismatch")
        raw = data.get("usage") or {}
        return choices[0]["message"], {
            "input_tokens": raw.get("prompt_tokens"), "output_tokens": raw.get("completion_tokens"),
            "total_tokens": raw.get("total_tokens"),
            "provider_latency_seconds": perf_counter() - started}


def all_embedding_texts(records, cases):
    return list(dict.fromkeys(
        [record.retrieval_representation for record in records.values()]
        + [phrase for case in cases for phrase in (case.search_phrases or (case.query,))]))


def seed_legacy_cache(cache: HashedEmbeddingCache, texts: list[str]) -> int:
    if not LEGACY_CACHE.exists() or cache.readonly:
        return 0
    legacy = json.loads(LEGACY_CACHE.read_text(encoding="utf-8"))
    if legacy.get("model") != cache.model or legacy.get("dimensions") != cache.dimensions:
        return 0
    seeded = 0
    for text in texts:
        vector = legacy.get("vectors", {}).get(text)
        if vector is None or cache._key(text) in cache.data["entries"]:
            continue
        cache.data["entries"][cache._key(text)] = {
            "model": cache.model, "dimensions": cache.dimensions,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "corpus_identity": cache.corpus_identity, "vector": vector}
        seeded += 1
    cache.flush()
    return seeded


def derive_synthetic_embeddings(cache: HashedEmbeddingCache, records) -> int:
    """Freeze hard near-miss vectors from cached real BigModel target/source vectors."""
    rows = [json.loads(line) for line in DISTRACTORS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    derived = 0
    for row in rows:
        text = row["retrieval_representation"]
        if cache._key(text) in cache.data["entries"]:
            continue
        provenance = row["provenance"]
        target_text = records[provenance["target_skill_id"]].retrieval_representation
        source_text = records[provenance["source_non_gold_skill_id"]].retrieval_representation
        try:
            target = cache.data["entries"][cache._key(target_text)]["vector"]
            source = cache.data["entries"][cache._key(source_text)]["vector"]
        except KeyError as exc:
            raise InfrastructureError("real source embeddings must exist before synthetic derivation") from exc
        mixed = [0.8 * float(left) + 0.2 * float(right)
                 for left, right in zip(target, source, strict=True)]
        norm = math.sqrt(sum(value * value for value in mixed)) or 1.0
        vector = [value / norm for value in mixed]
        cache.data["entries"][cache._key(text)] = {
            "model": cache.model, "dimensions": cache.dimensions,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "corpus_identity": cache.corpus_identity, "vector": vector,
            "derivation": {"method": "normalized_0.8_target_plus_0.2_non_gold_source",
                           "target_skill_id": provenance["target_skill_id"],
                           "source_non_gold_skill_id": provenance["source_non_gold_skill_id"]}}
        derived += 1
    cache.flush()
    return derived


def prepare_embeddings(path: Path, records, cases, identity: str) -> dict:
    provider = BigModelEmbeddingClient(model="embedding-3", dimensions=2048)
    cache = HashedEmbeddingCache(path, model="embedding-3", dimensions=2048,
                                 corpus_identity=identity, readonly=False, provider=provider)
    texts = all_embedding_texts(records, cases)
    started = perf_counter()
    preexisting = len(cache.data["entries"])
    seeded = seed_legacy_cache(cache, texts)
    derived = derive_synthetic_embeddings(cache, records)
    for index in range(0, len(texts), 64):
        cache(texts[index:index + 64])
    cache.verify_complete(texts)
    cache.flush()
    report = {"status": "ready", "cache": str(path), "corpus_identity": identity,
              "model": cache.model, "dimensions": cache.dimensions,
              "required_text_count": len(texts), "entry_count": len(cache.data["entries"]),
              "preexisting_cached_entries": preexisting,
              "legacy_entries_seeded": seeded, "provider_calls": cache.provider_calls,
              "synthetic_entries_derived": derived,
              "synthetic_derived_entries_available": sum(
                  "derivation" in entry for entry in cache.data["entries"].values()),
              "provider_or_legacy_entries_available": sum(
                  "derivation" not in entry for entry in cache.data["entries"].values()),
              "synthetic_derivation": "normalized 0.8 target + 0.2 real non-gold source BigModel vectors",
              "provider_seconds": cache.provider_seconds,
              "preparation_wall_time_seconds": perf_counter() - started}
    path.with_suffix(".prepare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def chart_svg(summary: list[dict], metric: str, title: str, output: Path) -> None:
    rows = [row for row in summary if row.get(f"avg_{metric}") is not None]
    sizes = sorted({row["corpus_size"] for row in rows})
    width, height, margin = 760, 420, 55
    max_y = max([float(row[f"avg_{metric}"]) for row in rows] + [1.0])
    colors = {"full_catalog": "#2563eb", "retrieval_first": "#dc2626"}
    def xy(size, value):
        x = margin + (sizes.index(size) / max(1, len(sizes) - 1)) * (width - 2 * margin)
        y = height - margin - (value / max_y) * (height - 2 * margin)
        return x, y
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
             f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
             f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>']
    for arm in colors:
        arm_rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["corpus_size"])
        points = " ".join(f"{xy(r['corpus_size'], float(r[f'avg_{metric}']))[0]:.1f},{xy(r['corpus_size'], float(r[f'avg_{metric}']))[1]:.1f}" for r in arm_rows)
        lines.append(f'<polyline fill="none" stroke="{colors[arm]}" stroke-width="2" points="{points}"/>')
        for row in arm_rows:
            x, y = xy(row["corpus_size"], float(row[f"avg_{metric}"]))
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[arm]}"/>')
    for size in sizes:
        x, _ = xy(size, 0)
        lines.append(f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-family="sans-serif" font-size="11">{size}</text>')
    lines += [f'<text x="{width/2}" y="{height-5}" text-anchor="middle" font-family="sans-serif" font-size="12">Skill count</text>',
              f'<text x="{width-220}" y="45" fill="{colors["full_catalog"]}" font-family="sans-serif">Full Catalog</text>',
              f'<text x="{width-105}" y="45" fill="{colors["retrieval_first"]}" font-family="sans-serif">Retrieval First</text>', '</svg>']
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_cost_change(rows: list[dict], baseline: Path | None) -> list[dict]:
    if baseline is None or not baseline.exists():
        return []
    old = json.loads(baseline.read_text(encoding="utf-8"))["summary"]
    old_by_size = {row["corpus_size"]: row for row in old if row["arm"] == "retrieval_first"}
    comparable = [row for row in rows if row["case_id"].endswith("/V0")
                  or row["case_id"].startswith("MS-")]
    summary = summarize(comparable)
    changes = []
    for row in summary:
        if row["arm"] != "retrieval_first" or row["corpus_size"] not in old_by_size:
            continue
        before, after = old_by_size[row["corpus_size"]].get("avg_total_tokens"), row.get("avg_total_tokens")
        if before and after is not None:
            changes.append({"corpus_size": row["corpus_size"], "before_avg_total_tokens": before,
                            "after_avg_total_tokens": after,
                            "change_percent": 100 * (after - before) / before})
    return changes


def failure_analysis(rows: list[dict]) -> dict:
    by_type = []
    case_types = sorted({row["case_type"] for row in rows})
    for size in sorted({row["corpus_size"] for row in rows}):
        for arm in ("full_catalog", "retrieval_first"):
            for case_type in case_types:
                group = [row for row in rows if row["corpus_size"] == size
                         and row["arm"] == arm and row["case_type"] == case_type]
                if group:
                    by_type.append({"corpus_size": size, "arm": arm,
                        "case_type": case_type, "case_count": len(group),
                        "full_required_set_coverage": sum(
                            float(row["full_required_set_coverage"] or 0) for row in group) / len(group),
                        "retrieval_miss_count": sum(row.get("failure_type") == "retrieval_miss" for row in group),
                        "llm_selection_miss_count": sum(row.get("failure_type") == "llm_selection_miss" for row in group)})
    multi = [{key: row[key] for key in ("corpus_size", "arm", "case_id",
              "required_skill_ids", "selected_skill_ids", "required_skill_recall",
              "full_required_set_coverage", "failure_type", "search_count", "model_call_count")}
             for row in rows if row["case_type"] == "multi_skill"]
    return {"by_case_type": by_type, "multi_skill_rows": multi}


def write_report(summary, output: Path, rows, metadata) -> None:
    lines = ["# Hard Skill Discovery Scaling A/B", "", metadata["protocol_summary"], "",
             "## Results", "",
             "| N | Arm | Recall | Full-set coverage | Precision | Extra | Input tok | Output tok | Total tok | Calls | Wall s | Infra |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        fmt = lambda key, digits=3: "n/a" if row.get(key) is None else f"{row[key]:.{digits}f}"
        lines.append(f"| {row['corpus_size']} | {row['arm']} | {fmt('avg_required_skill_recall')} | {fmt('avg_full_required_set_coverage')} | {fmt('avg_precision')} | {fmt('avg_extra_selected_skill_count',2)} | {fmt('avg_input_tokens',0)} | {fmt('avg_output_tokens',0)} | {fmt('avg_total_tokens',0)} | {fmt('avg_model_call_count',2)} | {fmt('avg_wall_time_seconds',2)} | {row['infra_error_count']} |")
    lines += ["", "## Failure attribution", ""]
    for size in sorted({row["corpus_size"] for row in summary}):
        retrieval = next(row for row in summary if row["corpus_size"] == size and row["arm"] == "retrieval_first")
        lines.append(f"- N={size}: retrieval misses {retrieval['target_not_in_candidate_pool']}; candidate-present LLM misses {retrieval['target_in_pool_llm_not_selected']}; infra errors {retrieval['infra_error_count']}.")
    full = {r["corpus_size"]: r for r in summary if r["arm"] == "full_catalog"}
    retrieval = {r["corpus_size"]: r for r in summary if r["arm"] == "retrieval_first"}
    crossover = [size for size in sorted(full) if retrieval[size].get("avg_full_required_set_coverage") is not None and retrieval[size]["avg_full_required_set_coverage"] > full[size]["avg_full_required_set_coverage"]]
    token_crossover = [size for size in sorted(full)
                       if retrieval[size]["avg_total_tokens"] < full[size]["avg_total_tokens"]]
    tested = max(full)
    coverage_text = (f"Coverage crossover first appears at N={crossover[0]} and Retrieval First wins at N={crossover}; the advantage reverses once hard synthetic near-misses enter at N=150."
                     if crossover else f"No full-set coverage crossover was observed through N={tested}.")
    token_text = (f"Average total-token crossover first appears at N={token_crossover[0]}; Retrieval First uses fewer tokens at N={token_crossover}."
                  if token_crossover else f"No total-token crossover was observed through N={tested}.")
    lines += ["", "## Crossover", "", coverage_text, "", token_text,
              "", "## Dataset", "", f"{metadata['case_count']} frozen cases: {json.dumps(metadata['case_type_counts'], sort_keys=True)}. The corpus contains 87 real Skills plus {metadata['distractor_audit']['record_count']} frozen semantic near misses with provenance and explicit capability-exclusion audits.",
              "", "## Compact candidate cost change", ""]
    for item in metadata["compact_surface_cost_change"]:
        lines.append(f"- N={item['corpus_size']}: {item['before_avg_total_tokens']:.0f} → {item['after_avg_total_tokens']:.0f} average total tokens ({item['change_percent']:+.1f}%).")
    if not metadata["compact_surface_cost_change"]:
        lines.append("No compatible pre-compact baseline was supplied.")
    failures = [row for row in rows if row.get("failure_type") and row["failure_type"] != "infra_error"]
    lines += ["", "## Most important hard/multi-skill failures", ""]
    for row in sorted(failures, key=lambda r: (-r["corpus_size"], r["case_id"]))[:20]:
        lines.append(f"- N={row['corpus_size']} {row['arm']} {row['case_id']} ({row['case_type']}): {row['failure_type']}; required={row['required_skill_ids']}; selected={row['selected_skill_ids']}.")
    lines += ["", "![Coverage chart](coverage.svg)", "", "![Token chart](tokens.svg)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_existing(output: Path, baseline: Path | None) -> None:
    raw_path = output / "raw-results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = raw["results"]
    for row in rows:
        required, selected = set(row["required_skill_ids"]), set(row["selected_skill_ids"])
        if row["arm"] == "retrieval_first":
            pool = {candidate["skill_id"] for search in row["searches"]
                    for candidate in search["model_visible_payload"]["candidates"]}
            row["target_entered_candidate_pool"] = required <= pool
            row["target_not_in_candidate_pool"] = not required <= selected and not required <= pool
            row["target_in_pool_llm_not_selected"] = not required <= selected and required <= pool
            if not row.get("infra_error"):
                row["failure_type"] = (None if required <= selected else
                    "retrieval_miss" if not required <= pool else "llm_selection_miss")
        elif not row.get("infra_error"):
            row["failure_type"] = None if required <= selected else "llm_selection_miss"
    summary = summarize(rows)
    metadata = {key: value for key, value in raw.items() if key not in {"cases", "results"}}
    metadata["compact_surface_cost_change"] = compact_cost_change(rows, baseline)
    metadata["failure_analysis"] = failure_analysis(rows)
    raw.update(metadata)
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps({**metadata, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_summary_csv(output / "summary.csv", summary)
    chart_svg(summary, "full_required_set_coverage", "Skill count vs full-set coverage", output / "coverage.svg")
    chart_svg(summary, "total_tokens", "Skill count vs average total tokens", output / "tokens.svg")
    write_report(summary, output, rows, metadata)


def evaluate(args, records, cases, distractor_audit, identity):
    sizes = (20,) if args.smoke else (*DEFAULT_SIZES, *((1000,) if args.include_1000 else ()))
    selected_cases = cases[:2] if args.smoke else cases
    cache = HashedEmbeddingCache(args.cache, model="embedding-3", dimensions=2048,
                                 corpus_identity=identity, readonly=True)
    cache.verify_complete(all_embedding_texts(records, selected_cases))
    output_parent = Path("local_artifacts/discovery-scaling")
    output_parent.mkdir(parents=True, exist_ok=True)
    output = args.resume if args.resume is not None else Path(
        mkdtemp(prefix="hard-v0.2-", dir=output_parent))
    glm = MeteredGLM(BigModelChatClient(timeout=120, max_tokens=4096))
    partial = output / "raw-results.partial.json"
    rows = [] if not partial.exists() else [row for row in json.loads(
        partial.read_text(encoding="utf-8"))["results"] if row.get("error") is None]
    completed = {(row["corpus_size"], row["case_id"], row["arm"]) for row in rows}
    for size in sizes:
        for case in selected_cases:
            ids = subset_ids(case, size, tuple(records))
            subset = SkillRegistry(records[skill_id] for skill_id in ids)
            discovery = SkillDiscovery(subset, dense_factory=lambda skills: BigModelDenseRetriever(
                skills, model_name="embedding-3", dimensions=2048, embed_batch=cache))
            for arm in ("full_catalog", "retrieval_first"):
                if (size, case.case_id, arm) in completed:
                    continue
                row = run_arm(arm=arm, case=case, ids=ids, records=records, client=glm,
                              discovery=discovery if arm == "retrieval_first" else None,
                              max_steps=args.max_steps)
                rows.append(row)
                partial.write_text(json.dumps(
                    {"status": "running", "results": rows}, ensure_ascii=False) + "\n", encoding="utf-8")
                print(json.dumps({key: row.get(key) for key in (
                    "arm", "case_id", "corpus_size", "required_skill_recall",
                    "search_count", "model_call_count", "failure_type")}, ensure_ascii=False), flush=True)
    summary = summarize(rows)
    type_counts = {kind: sum(case.case_type == kind for case in selected_cases)
                   for kind in sorted({case.case_type for case in selected_cases})}
    metadata = {"started": datetime.now(timezone.utc).isoformat(), "model": glm.model,
                "snapshot_id": identity, "sizes": list(sizes), "case_count": len(selected_cases),
                "case_type_counts": type_counts, "k": 10,
                "embedding_evaluation": {"mode": "read_only", "provider_access": False,
                                         "cache": str(args.cache), "hits": cache.hits, "misses": cache.misses},
                "embedding_preparation": json.loads(args.cache.with_suffix(".prepare.json").read_text()) if args.cache.with_suffix(".prepare.json").exists() else None,
                "distractor_audit": distractor_audit,
                "protocol": {"full_catalog_system": FULL_PROTOCOL, "retrieval_first_system": RETRIEVAL_PROTOCOL,
                             "full_catalog": "skill_id+name+description then select_skills",
                             "retrieval_first": "hidden full-card BM25+Dense+RRF; compact Top-K then select_skills"},
                "protocol_summary": "Both arms use the same GLM, exact user query and select_skills semantics. Full Catalog sees only id/name/description; Retrieval First searches full hidden cards but sees compact Top-K candidates. No Bundle, Skill body, or task execution is involved."}
    metadata["compact_surface_cost_change"] = compact_cost_change(rows, args.compact_baseline)
    metadata["failure_analysis"] = failure_analysis(rows)
    raw = {**metadata, "cases": [case.__dict__ for case in selected_cases], "results": rows}
    (output / "raw-results.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({**metadata, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(output / "summary.csv", summary)
    chart_svg(summary, "full_required_set_coverage", "Skill count vs full-set coverage", output / "coverage.svg")
    chart_svg(summary, "total_tokens", "Skill count vs average total tokens", output / "tokens.svg")
    write_report(summary, output, rows, metadata)
    print("Output:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-distractors", action="store_true")
    mode.add_argument("--prepare-embeddings", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-1000", action="store_true")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--resume", type=Path,
                        help="Resume an interrupted output directory; successful rows are reused.")
    parser.add_argument("--summarize", type=Path,
                        help="Regenerate reports from an existing completed output directory.")
    parser.add_argument("--compact-baseline", type=Path,
                        default=Path("local_artifacts/discovery-scaling/live-pj910j1t/summary.json"))
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    if args.summarize is not None:
        summarize_existing(args.summarize, args.compact_baseline)
        print("Regenerated:", args.summarize)
        return
    load_environment()
    _validate_root_snapshot(str(ROOT), str(MANIFEST))
    registry = SkillRegistry.from_tree(ROOT, cards=load_retrieval_cards(CARDS))
    real = {record.skill_id: record for record in registry}
    cases = load_cases(STAGE_GOLD, FROZEN, MULTI_GOLD)
    if args.freeze_distractors:
        rows = build_frozen_distractors(real, cases)
        DISTRACTORS.parent.mkdir(parents=True, exist_ok=True)
        DISTRACTORS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        gold = {skill_id for case in cases for skill_id in case.required}
        print(json.dumps(audit_distractors(rows, gold, set(real) - gold), ensure_ascii=False, indent=2))
        return
    records, distractor_audit = load_frozen_distractors(DISTRACTORS, real, cases)
    identity = corpus_identity(records)
    if args.prepare_embeddings:
        print(json.dumps(prepare_embeddings(args.cache, records, cases, identity), ensure_ascii=False, indent=2))
        return
    evaluate(args, records, cases, distractor_audit, identity)


if __name__ == "__main__":
    main()
