from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import yaml

from skill_control_plane.models import SkillRecord


def decode_skill_bytes(data: bytes) -> str:
    """Match read_text(encoding='utf-8-sig', errors='replace', newline=None).

    Manifest hashes identify decoded text, not raw bytes: strip a UTF-8 BOM,
    replace invalid UTF-8, and translate CRLF/bare CR to LF. Do not strip text.
    """
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", errors="replace") as stream:
        return stream.read()


def skill_content_hash(data: bytes) -> str:
    """The registry/manifest identity hash, also usable for historical Git blobs."""
    return hashlib.sha256(decode_skill_bytes(data).encode("utf-8")).hexdigest()


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
        data = skill_md.read_bytes()
        raw = decode_skill_bytes(data)
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
                content_hash=skill_content_hash(data),
                category=category,
            )
        )
    return skills
