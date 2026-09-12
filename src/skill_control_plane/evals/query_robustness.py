"""Query-paraphrase robustness evaluation for Stage reroute retrieval."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from skill_control_plane.evals.control_plane import StageTransitionGoldCase
from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.discovery import candidate_union, reciprocal_rank_fusion
from skill_control_plane.discovery.base import Retriever
from skill_control_plane.runtime.capability_need import (
    CapabilityNeedExtractor,
    extract_query,
)

QUERY_VARIANT_VERSION = "stage-query-paraphrase-v0.1"
TextCompleter = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class QueryVariantSet:
    case_id: str
    stage_id: str
    original_query: str
    variants: tuple[str, ...]
    version: str = QUERY_VARIANT_VERSION


def build_query_paraphrase_prompt(query: str, paraphrase_count: int = 4) -> str:
    """Build a leakage-resistant prompt from the frozen retrieval query only."""
    if paraphrase_count < 1:
        raise ValueError("paraphrase_count must be >= 1")
    return f"""Generate {paraphrase_count} natural-language paraphrases of ONE Skill-retrieval query.

Goal:
Test retrieval robustness to wording changes while keeping the capability need exactly
the same.

Rules:
- Use ONLY the original query below.
- Preserve every requested capability and constraint.
- Do not add capabilities, tools, products, Skill names, implementation details, or
  assumptions that are absent from the original.
- Do not remove any capability or constraint.
- Vary wording and syntax meaningfully; do not merely swap one adjective.
- Each paraphrase must be independently understandable.
- Return ONLY one JSON object with exactly one key:
  {{"paraphrases": ["...", "..."]}}
- The paraphrases array must contain exactly {paraphrase_count} unique strings.

Original query:
{query}
"""


@dataclass(slots=True)
class LLMQueryParaphraser:
    complete: TextCompleter

    def paraphrase(self, query: str, paraphrase_count: int = 4) -> tuple[str, ...]:
        raw = self.complete(build_query_paraphrase_prompt(query, paraphrase_count))
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {"paraphrases"}:
            raise ValueError("paraphrase completion must contain only 'paraphrases'")
        values = data["paraphrases"]
        if not isinstance(values, list):
            raise ValueError("paraphrases must be a JSON array")

        cleaned: list[str] = []
        original_norm = " ".join(query.split()).strip()
        for value in values:
            if not isinstance(value, str):
                raise ValueError("paraphrases must contain only strings")
            text = " ".join(value.split()).strip()
            if not text or text == original_norm or text in cleaned:
                continue
            cleaned.append(text)

        if len(cleaned) != paraphrase_count:
            raise ValueError(
                "paraphrase completion must yield exactly "
                f"{paraphrase_count} unique non-original strings; got {len(cleaned)}"
            )
        return tuple(cleaned)


def load_query_variant_sets(path: str | Path) -> dict[tuple[str, str], QueryVariantSet]:
    sets: dict[tuple[str, str], QueryVariantSet] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"query variant line {line_number} must be an object")
        version = str(data.get("version", ""))
        if version != QUERY_VARIANT_VERSION:
            raise ValueError(
                f"query variant line {line_number} has version={version!r}; "
                f"expected {QUERY_VARIANT_VERSION!r}"
            )
        variants = data.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            raise ValueError(
                f"query variant line {line_number} must contain original + paraphrases"
            )
        cleaned = tuple(" ".join(str(value).split()).strip() for value in variants)
        original = " ".join(str(data["original_query"]).split()).strip()
        if cleaned[0] != original:
            raise ValueError(
                f"query variant line {line_number}: variants[0] must equal original_query"
            )
        if any(not value for value in cleaned) or len(set(cleaned)) != len(cleaned):
            raise ValueError(
                f"query variant line {line_number} contains empty or duplicate variants"
            )
        item = QueryVariantSet(
            case_id=str(data["case_id"]),
            stage_id=str(data["stage_id"]),
            original_query=original,
            variants=cleaned,
            version=version,
        )
        key = (item.case_id, item.stage_id)
        if key in sets:
            raise ValueError(f"duplicate query variants for {key}")
        sets[key] = item
    if not sets:
        raise ValueError("query variant set is empty")
    return sets


def _write_query_variant_sets(
    path: Path,
    sets: Mapping[tuple[str, str], QueryVariantSet],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(
        json.dumps(asdict(sets[key]), ensure_ascii=False, sort_keys=True) + "\n"
        for key in sorted(sets)
    )
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def build_query_variant_cache(
    cases: Sequence[StageTransitionGoldCase],
    *,
    old_extractor: CapabilityNeedExtractor,
    paraphraser: LLMQueryParaphraser,
    output: str | Path,
    paraphrase_count: int = 4,
    force: bool = False,
) -> dict[str, int]:
    """Freeze one original query plus paraphrases for every target transition."""
    if paraphrase_count < 1:
        raise ValueError("paraphrase_count must be >= 1")

    output_path = Path(output)
    existing = (
        load_query_variant_sets(output_path)
        if output_path.exists() and not force
        else {}
    )
    current: dict[tuple[str, str], QueryVariantSet] = {}
    extracted = 0
    reused = 0

    for case in cases:
        for stage in case.stages[1:]:
            if not stage.new_required:
                continue
            frozen = extract_query(
                old_extractor,
                case.initial_task,
                stage.runtime_evidence,
            )
            if frozen.status != "ok":
                raise ValueError(
                    f"frozen capability_need extraction failed for "
                    f"{case.case_id}/{stage.stage_id}: {frozen.status}"
                )
            original = " ".join(frozen.query.split()).strip()
            key = (case.case_id, stage.stage_id)
            cached = existing.get(key)
            if (
                cached is not None
                and cached.original_query == original
                and len(cached.variants) == paraphrase_count + 1
            ):
                current[key] = cached
                reused += 1
                continue

            paraphrases = paraphraser.paraphrase(original, paraphrase_count)
            current[key] = QueryVariantSet(
                case_id=case.case_id,
                stage_id=stage.stage_id,
                original_query=original,
                variants=(original, *paraphrases),
            )
            extracted += 1
            _write_query_variant_sets(output_path, current)

    _write_query_variant_sets(output_path, current)
    return {
        "target_transition_count": len(current),
        "query_variant_set_count": len(current),
        "paraphrases_per_transition": paraphrase_count,
        "variants_per_transition": paraphrase_count + 1,
        "extracted": extracted,
        "reused": reused,
    }


def _required_ranks(
    required: Sequence[str],
    ranking: Sequence[RetrievalCandidate],
) -> dict[str, int | None]:
    positions = {candidate.skill_id: candidate.rank for candidate in ranking}
    return {skill_id: positions.get(skill_id) for skill_id in required}


def _rank_hits(ranks: Mapping[str, int | None], cutoff: int) -> int:
    return sum(rank is not None and rank <= cutoff for rank in ranks.values())


def _rr_sum(ranks: Mapping[str, int | None], cutoff: int) -> float:
    return sum(
        1.0 / rank
        for rank in ranks.values()
        if rank is not None and rank <= cutoff
    )


def evaluate_query_variant_retrieval(
    cases: Sequence[StageTransitionGoldCase],
    *,
    query_sets: Mapping[tuple[str, str], QueryVariantSet],
    bm25: Retriever,
    dense: Retriever,
    per_retriever_k: int = 10,
    rrf_k: int = 60,
    cutoffs: tuple[int, ...] = (5, 10),
    skill_representation: str,
) -> dict[str, Any]:
    """Evaluate the same frozen query variants against one Skill representation."""
    if per_retriever_k < 1:
        raise ValueError("per_retriever_k must be >= 1")
    if max(cutoffs) > per_retriever_k:
        raise ValueError("cutoffs cannot exceed per_retriever_k")
    cutoffs = tuple(sorted(set(cutoffs)))

    expected_keys = {
        (case.case_id, stage.stage_id)
        for case in cases
        for stage in case.stages[1:]
        if stage.new_required
    }
    if set(query_sets) != expected_keys:
        missing = sorted(expected_keys - set(query_sets))
        extra = sorted(set(query_sets) - expected_keys)
        raise ValueError(
            f"query variant/Gold mismatch: missing={missing[:8]} extra={extra[:8]}"
        )

    arm_names = ("dense", "bm25", "rrf")
    total_hits = {arm: {k: 0 for k in cutoffs} for arm in arm_names}
    original_hits = {arm: {k: 0 for k in cutoffs} for arm in arm_names}
    paraphrase_hits = {arm: {k: 0 for k in cutoffs} for arm in arm_names}
    rr_sums = {arm: {k: 0.0 for k in cutoffs} for arm in arm_names}
    all_variant_hits = {arm: {k: 0 for k in cutoffs} for arm in arm_names}

    union_hits = 0
    union_original_hits = 0
    union_paraphrase_hits = 0
    union_all_variant_hits = 0
    union_pool_sizes = 0

    required_occurrences = 0
    variant_required_occurrences = 0
    paraphrase_required_occurrences = 0
    variant_count_total = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        for stage in case.stages[1:]:
            required = tuple(stage.new_required)
            if not required:
                continue
            query_set = query_sets[(case.case_id, stage.stage_id)]
            required_occurrences += len(required)
            variant_required_occurrences += len(required) * len(query_set.variants)
            paraphrase_required_occurrences += len(required) * (len(query_set.variants) - 1)
            variant_count_total += len(query_set.variants)

            ranks_by_arm: dict[str, list[dict[str, int | None]]] = {
                arm: [] for arm in arm_names
            }
            union_hits_by_variant: list[set[str]] = []
            variant_rows: list[dict[str, Any]] = []

            for variant_index, query in enumerate(query_set.variants):
                dense_ranking = dense.search(query, k=per_retriever_k)
                bm25_ranking = bm25.search(query, k=per_retriever_k)
                sources = {"bm25": bm25_ranking, "dense": dense_ranking}
                rrf_ranking = reciprocal_rank_fusion(sources, k=rrf_k)
                union_ranking = candidate_union(sources)

                rankings = {
                    "dense": dense_ranking,
                    "bm25": bm25_ranking,
                    "rrf": rrf_ranking,
                }
                variant_arm_rows: dict[str, Any] = {}
                for arm, ranking in rankings.items():
                    ranks = _required_ranks(required, ranking)
                    ranks_by_arm[arm].append(ranks)
                    variant_arm_rows[arm] = {"new_required_ranks": ranks}
                    for cutoff in cutoffs:
                        hits = _rank_hits(ranks, cutoff)
                        total_hits[arm][cutoff] += hits
                        rr_sums[arm][cutoff] += _rr_sum(ranks, cutoff)
                        if variant_index == 0:
                            original_hits[arm][cutoff] += hits
                        else:
                            paraphrase_hits[arm][cutoff] += hits

                union_ids = {candidate.skill_id for candidate in union_ranking}
                current_union_hits = set(required) & union_ids
                union_hits_by_variant.append(current_union_hits)
                union_hits += len(current_union_hits)
                union_pool_sizes += len(union_ranking)
                if variant_index == 0:
                    union_original_hits += len(current_union_hits)
                else:
                    union_paraphrase_hits += len(current_union_hits)

                variant_rows.append(
                    {
                        "variant_index": variant_index,
                        "query": query,
                        "is_original": variant_index == 0,
                        **variant_arm_rows,
                        "union": {
                            "new_required_hits": sorted(current_union_hits),
                            "candidate_set_size": len(union_ranking),
                        },
                    }
                )

            for arm in arm_names:
                for cutoff in cutoffs:
                    for skill_id in required:
                        if all(
                            ranks[skill_id] is not None and ranks[skill_id] <= cutoff
                            for ranks in ranks_by_arm[arm]
                        ):
                            all_variant_hits[arm][cutoff] += 1

            for skill_id in required:
                if all(skill_id in hits for hits in union_hits_by_variant):
                    union_all_variant_hits += 1

            rows.append(
                {
                    "case_id": case.case_id,
                    "stage_id": stage.stage_id,
                    "new_required": list(required),
                    "variants": variant_rows,
                }
            )

    required_den = required_occurrences or 1
    variant_den = variant_required_occurrences or 1
    paraphrase_den = paraphrase_required_occurrences or 1

    arms: dict[str, Any] = {}
    for arm in arm_names:
        metrics: dict[str, float] = {}
        for cutoff in cutoffs:
            metrics[f"recall_at_{cutoff}"] = total_hits[arm][cutoff] / variant_den
            metrics[f"original_recall_at_{cutoff}"] = (
                original_hits[arm][cutoff] / required_den
            )
            metrics[f"paraphrase_recall_at_{cutoff}"] = (
                paraphrase_hits[arm][cutoff] / paraphrase_den
            )
            metrics[f"mrr_at_{cutoff}"] = rr_sums[arm][cutoff] / variant_den
            metrics[f"all_variants_hit_at_{cutoff}"] = (
                all_variant_hits[arm][cutoff] / required_den
            )
        arms[arm] = metrics

    return {
        "experiment": "query paraphrase robustness",
        "skill_representation": skill_representation,
        "target_transition_count": len(rows),
        "new_required_skill_occurrences": required_occurrences,
        "query_variant_count": variant_count_total,
        "variants_per_transition": (
            variant_count_total / len(rows) if rows else 0.0
        ),
        "per_retriever_k": per_retriever_k,
        "rrf_k": rrf_k,
        "cutoffs": list(cutoffs),
        **arms,
        "union": {
            "candidate_recall": union_hits / variant_den,
            "original_candidate_recall": union_original_hits / required_den,
            "paraphrase_candidate_recall": union_paraphrase_hits / paraphrase_den,
            "all_variants_covered": union_all_variant_hits / required_den,
            "average_candidate_set_size": (
                union_pool_sizes / variant_count_total if variant_count_total else 0.0
            ),
        },
        "stages": rows,
    }


def compare_query_robustness_reports(
    metadata: Mapping[str, Any],
    retrieval_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare robustness under identical frozen query variants."""
    for key in (
        "target_transition_count",
        "new_required_skill_occurrences",
        "query_variant_count",
        "cutoffs",
    ):
        if metadata[key] != retrieval_card[key]:
            raise ValueError(f"robustness reports disagree on {key}")

    arm_names = ("dense", "bm25", "rrf")
    delta: dict[str, Any] = {}
    for arm in arm_names:
        delta[arm] = {
            key: float(retrieval_card[arm][key]) - float(metadata[arm][key])
            for key in metadata[arm]
        }
    delta["union"] = {
        key: float(retrieval_card["union"][key]) - float(metadata["union"][key])
        for key in (
            "candidate_recall",
            "original_candidate_recall",
            "paraphrase_candidate_recall",
            "all_variants_covered",
        )
    }

    return {
        "experiment": "metadata vs RetrievalCard query-paraphrase robustness",
        "interpretation": (
            "positive delta means RetrievalCard is more robust under the same query variants"
        ),
        "target_transition_count": metadata["target_transition_count"],
        "new_required_skill_occurrences": metadata["new_required_skill_occurrences"],
        "query_variant_count": metadata["query_variant_count"],
        "variants_per_transition": metadata["variants_per_transition"],
        "cutoffs": metadata["cutoffs"],
        "metadata-v0.1": {
            arm: metadata[arm] for arm in (*arm_names, "union")
        },
        "retrieval-card-v0.1": {
            arm: retrieval_card[arm] for arm in (*arm_names, "union")
        },
        "delta_retrieval_card_minus_metadata": delta,
    }


def print_query_robustness_comparison(report: Mapping[str, Any]) -> None:
    print("=== QUERY PARAPHRASE ROBUSTNESS ===")
    print(f"target_transitions: {report['target_transition_count']}")
    print(f"query_variants: {report['query_variant_count']}")
    print(
        "variants_per_transition: "
        f"{float(report['variants_per_transition']):.2f}"
    )
    metadata = report["metadata-v0.1"]
    card = report["retrieval-card-v0.1"]
    delta = report["delta_retrieval_card_minus_metadata"]
    for arm in ("dense", "bm25", "rrf"):
        print()
        print(f"[{arm}]")
        for cutoff in report["cutoffs"]:
            key = f"recall_at_{cutoff}"
            stable_key = f"all_variants_hit_at_{cutoff}"
            print(
                f"Recall@{cutoff}: metadata={float(metadata[arm][key]):.4f} "
                f"card={float(card[arm][key]):.4f} "
                f"delta={float(delta[arm][key]):+.4f}"
            )
            print(
                f"AllVariantsHit@{cutoff}: "
                f"metadata={float(metadata[arm][stable_key]):.4f} "
                f"card={float(card[arm][stable_key]):.4f} "
                f"delta={float(delta[arm][stable_key]):+.4f}"
            )
