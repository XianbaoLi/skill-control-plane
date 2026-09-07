from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from skill_control_plane.registry import load_skill_tree


def _snapshot_id(entries: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        f"{entry['relative_path']}:{entry['content_hash']}"
        for entry in sorted(entries, key=lambda item: item["relative_path"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_corpus_manifest(
    root: str | Path,
    *,
    source: str,
    harness_commit: str | None = None,
) -> dict[str, Any]:
    """Build a content-free manifest for a local Skill tree."""

    root_path = Path(root).resolve()
    skills = load_skill_tree(root_path)

    entries: list[dict[str, Any]] = []
    for skill in skills:
        skill_path = Path(skill.source_path).resolve()
        relative_path = skill_path.relative_to(root_path).as_posix()
        parts = Path(relative_path).parts
        entries.append(
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "description": skill.description,
                "tags": list(skill.tags),
                "category": parts[0] if len(parts) > 1 else "",
                "relative_path": relative_path,
                "content_hash": skill.content_hash,
                "body_chars": len(skill.body),
            }
        )

    id_counts = Counter(entry["skill_id"] for entry in entries)
    duplicate_skill_ids = sorted(
        skill_id for skill_id, count in id_counts.items() if count > 1
    )

    return {
        "schema_version": 1,
        "source": source,
        "harness_commit": harness_commit,
        "skill_count": len(entries),
        "duplicate_skill_ids": duplicate_skill_ids,
        "snapshot_id": _snapshot_id(entries),
        "skills": sorted(entries, key=lambda item: item["relative_path"]),
    }


def write_corpus_manifest(
    root: str | Path,
    output: str | Path,
    *,
    source: str,
    harness_commit: str | None = None,
) -> dict[str, Any]:
    manifest = build_corpus_manifest(
        root,
        source=source,
        harness_commit=harness_commit,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
