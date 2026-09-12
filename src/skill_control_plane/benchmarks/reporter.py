"""Raw metric aggregation and paired Control Plane minus Native deltas."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

PAIRED_METRICS = ("task_success", "required_skill_recall", "wrong_skill_count",
                  "llm_input_tokens", "llm_output_tokens", "llm_total_tokens",
                  "cached_tokens", "wall_time_ms", "total_tool_calls")
RUNTIME_ONLY_PAIRED_METRICS = ("reroute_success", "new_required_skill_recall",
                               "premature_activation_count")


def _avg(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return mean(values) if values else None


def paired_report(rows: list[dict[str, Any]], suite: str) -> dict[str, Any]:
    metrics = PAIRED_METRICS if suite == "scaling" else PAIRED_METRICS + RUNTIME_ONLY_PAIRED_METRICS
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["suite"] == suite:
            grouped[(row["corpus"], row["arm"])].append(row)
    summary = []
    for (corpus, arm), group in sorted(grouped.items()):
        summary.append({"corpus": corpus, "arm": arm, "run_count": len(group),
                        **{metric: _avg(group, metric) for metric in metrics}})
    paired = []
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["suite"] == suite:
            by_pair[(row["task_id"], row["corpus"])][row["arm"]] = row
    for (task_id, corpus), arms in sorted(by_pair.items()):
        if set(arms) != {"native", "control-plane"}:
            continue
        paired.append({"task_id": task_id, "corpus": corpus,
                       **{metric: (None if arms["native"].get(metric) is None or
                           arms["control-plane"].get(metric) is None else
                           float(arms["control-plane"][metric]) - float(arms["native"][metric]))
                          for metric in metrics}})
    return {"suite": suite, "summary": summary, "paired_delta_control_plane_minus_native": paired}
