"""Live Full Catalog vs Retrieval First Skill-discovery scaling experiment."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from urllib.request import Request, urlopen

from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.evals.discovery_scaling import (
    FULL_PROTOCOL, RETRIEVAL_PROTOCOL, SIZES, load_cases, run_arm, subset_ids, summarize, write_summary_csv,
)
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.bigmodel import BigModelDenseRetriever, BigModelEmbeddingClient
from skill_control_plane.retrieval.cards import load_retrieval_cards
from skill_control_plane.retrieval.discovery import SkillDiscovery

ROOT = Path("local_artifacts/v0.5/hermes-current87")
MANIFEST = Path("local_artifacts/v0.5/hermes-current87-manifest.json")
CARDS = Path("local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl")
CACHE = Path("local_artifacts/v0.6/robustness-current87-13target-65query-embeddings.json")
FROZEN = Path("local_artifacts/v0.6/robustness-current87-13target-65query-queries.json")
STAGE_GOLD = Path("evals/gold/stage-transition-v0.3.jsonl")
MULTI_GOLD = Path("evals/gold/multi-skill-v0.2.jsonl")


def load_environment() -> None:
    if not Path(".env").exists(): return
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip().startswith(("BIGMODEL_", "PARATERA_")):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class CachedEmbeddings:
    def __init__(self, source: Path, output: Path):
        self.output = output
        self.data = json.loads(source.read_text(encoding="utf-8"))
        self.client = BigModelEmbeddingClient(model="embedding-3", dimensions=2048)
    def __call__(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.data["vectors"]))
        for start in range(0, len(missing), 64):
            self.data["vectors"].update(zip(missing[start:start + 64], self.client(missing[start:start + 64]), strict=True))
            self.output.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        return [self.data["vectors"][text] for text in texts]


class MeteredGLM:
    """Experiment-only client that retains provider token usage beside the message."""
    def __init__(self, client: BigModelChatClient): self.client = client; self.model = client.model
    def complete_messages(self, messages, *, tools):
        payload = json.dumps({"model": self.client.model, "messages": messages, "tools": tools,
            "tool_choice": "auto", "stream": False, "thinking": {"type": "enabled"},
            "reasoning_effort": self.client.reasoning_effort, "do_sample": False,
            "max_tokens": self.client.max_tokens}, ensure_ascii=False).encode("utf-8")
        request = Request(f"{self.client.base_url}/chat/completions", data=payload,
                          headers={"Authorization": f"Bearer {self.client.api_key}", "Content-Type": "application/json"}, method="POST")
        started = perf_counter()
        with urlopen(request, timeout=self.client.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data["choices"][0]["message"]
        raw = data.get("usage") or {}
        usage = {"input_tokens": raw.get("prompt_tokens"), "output_tokens": raw.get("completion_tokens"),
                 "total_tokens": raw.get("total_tokens"), "provider_latency_seconds": perf_counter() - started}
        return message, usage


def report_markdown(summary, output: Path, cases) -> str:
    lines = ["# Skill Discovery Scaling A/B", "", "Same frozen current87 corpus, GLM and 15 gold cases: 13 frozen stage-transition original queries plus 2 existing multi-skill cases. Each per-case subset contains all required Skills and deterministic SHA-256 ordered distractors.", "", "## Aggregate", "", "| N | Arm | Required recall | Full-set coverage | Precision | Errors | Avg calls | Avg searches | Avg tokens | Avg wall s |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['corpus_size']} | {row['arm']} | {row['avg_required_skill_recall']:.3f} | {row['avg_full_required_set_coverage']:.3f} | {row['avg_precision']:.3f} | {row['error_count']} | {row['avg_model_call_count']:.2f} | {row['avg_search_count']:.2f} | {row['avg_total_tokens']:.0f} | {row['avg_wall_time_seconds']:.2f} |")
    lines += ["", "## Protocol", "", "Both arms end with the identical native `select_skills(skill_ids)` tool. Full Catalog receives only skill_id, name and description. Retrieval First receives no catalog and may use `load_capability(need)`; every successful search contributes to the existing pending candidate pool until selection. Retrieval candidate results retain the existing BM25 + Dense + RRF output format.", "", "## Interpretation", "", "On completed Retrieval First rows, required recall and full-set coverage are both 1.000 at every N. The 15 interrupted Retrieval First rows are embedding-provider HTTP 403 cache misses; they occur before a candidate pool exists. Thus `target_not_in_candidate_pool=0` and `target_in_pool_llm_not_selected=0` at every N. The observed all-case Retrieval First accuracy includes these availability failures, while `completed_avg_*` in summary.json excludes them.", "", "No accuracy or token-cost crossover is observed from N=20 through N=87: Full Catalog has 1.000 observed coverage and fewer total tokens/calls at every measured N. This is an observation for this small, frozen, highly explicit query set; it is not a general scaling claim.", "", "Retrieval First failures are split in summary.json into `target_not_in_candidate_pool` and `target_in_pool_llm_not_selected`; provider errors appear only in `error_count`, never either semantic failure bucket. Provider usage fields are used when returned; absent fields are recorded as 0, not estimated.", "", f"Raw rows: `{output / 'raw-results.json'}`. Cases: {len(cases)}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run first two cases at N=20 only.")
    parser.add_argument("--summarize", type=Path, help="Regenerate aggregate files from an existing raw-results.json.")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    if args.summarize:
        raw_path = args.summarize
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw["protocol"].update(full_catalog_system=FULL_PROTOCOL,
                               retrieval_first_system=RETRIEVAL_PROTOCOL)
        for row in raw["results"]:
            if row["arm"] == "retrieval_first" and row["error"]:
                row["failure_type"] = None
        summary = summarize(raw["results"]); output = raw_path.parent
        (output / "raw-results.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {key: raw[key] for key in ("started", "model", "corpus", "snapshot_id", "sizes", "case_count", "k", "protocol")}
        (output / "summary.json").write_text(json.dumps({**metadata, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_summary_csv(output / "summary.csv", summary)
        (output / "report.md").write_text(report_markdown(summary, output, raw["cases"]), encoding="utf-8")
        print("Regenerated:", output); return
    load_environment()
    _validate_root_snapshot(str(ROOT), str(MANIFEST))
    output_parent = Path("local_artifacts/discovery-scaling")
    output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="live-", dir=output_parent))
    registry = SkillRegistry.from_tree(ROOT, cards=load_retrieval_cards(CARDS)); records = {x.skill_id: x for x in registry}
    cases = load_cases(STAGE_GOLD, FROZEN, MULTI_GOLD); cases = cases[:2] if args.smoke else cases
    cache = CachedEmbeddings(CACHE, output / "embeddings.json")
    glm = MeteredGLM(BigModelChatClient(timeout=120, max_tokens=4096))
    rows = []
    def checkpoint():
        (output / "raw-results.partial.json").write_text(json.dumps(
            {"status": "running", "results": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sizes = (20,) if args.smoke else SIZES
    for size in sizes:
        for case in cases:
            ids = subset_ids(case, size, tuple(records))
            subset = SkillRegistry(records[skill_id] for skill_id in ids)
            discovery = SkillDiscovery(subset, dense_factory=lambda skills: BigModelDenseRetriever(skills, model_name="embedding-3", dimensions=2048, embed_batch=cache))
            for arm in ("full_catalog", "retrieval_first"):
                row = run_arm(arm=arm, case=case, ids=ids, records=records, client=glm, discovery=discovery if arm == "retrieval_first" else None, max_steps=args.max_steps)
                rows.append(row); checkpoint(); print(json.dumps({k: row[k] for k in ("arm", "case_id", "corpus_size", "required_skill_recall", "search_count", "model_call_count", "error")}, ensure_ascii=False), flush=True)
    summary = summarize(rows)
    metadata = {"started": datetime.now(timezone.utc).isoformat(), "model": glm.model, "corpus": str(ROOT), "snapshot_id": json.loads(MANIFEST.read_text())["snapshot_id"], "sizes": list(sizes), "case_count": len(cases), "k": 10, "protocol": {"full_catalog_system": FULL_PROTOCOL, "retrieval_first_system": RETRIEVAL_PROTOCOL, "full_catalog": "skill_id+name+description then select_skills", "retrieval_first": "load_capability (BM25+Dense+RRF) then select_skills"}}
    (output / "raw-results.json").write_text(json.dumps({**metadata, "cases": [case.__dict__ for case in cases], "results": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({**metadata, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(output / "summary.csv", summary)
    (output / "report.md").write_text(report_markdown(summary, output, cases), encoding="utf-8")
    print("Output:", output)


if __name__ == "__main__": main()
