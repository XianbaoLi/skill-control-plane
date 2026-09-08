from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from skill_control_plane.models import SkillRecord


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(
            i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration:
        return {}, text
    parsed = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, "\n".join(lines[end + 1 :])


def _extract_tags(frontmatter: dict[str, Any]) -> tuple[str, ...]:
    tags: list[str] = []
    direct = frontmatter.get("tags")
    if isinstance(direct, list):
        tags.extend(str(item) for item in direct)
    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        hermes = metadata.get("hermes")
        if isinstance(hermes, dict) and isinstance(hermes.get("tags"), list):
            tags.extend(str(item) for item in hermes["tags"])
    return tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))


def load_skill_tree(root: str | Path) -> list[SkillRecord]:
    """Load Agent-Skills-style directories containing SKILL.md files."""

    root_path = Path(root).resolve()
    skills: list[SkillRecord] = []
    for skill_md in sorted(root_path.rglob("SKILL.md")):
        raw = skill_md.read_text(encoding="utf-8-sig", errors="replace")
        frontmatter, body = _split_frontmatter(raw)
        skill_id = str(frontmatter.get("name") or skill_md.parent.name).strip()
        relative_path = skill_md.resolve().relative_to(root_path)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        skills.append(
            SkillRecord(
                skill_id=skill_id,
                name=str(frontmatter.get("name") or skill_md.parent.name).strip(),
                description=str(frontmatter.get("description") or "").strip(),
                body=body.strip(),
                source_path=str(skill_md),
                tags=_extract_tags(frontmatter),
                content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                category=category,
            )
        )
    return skills
