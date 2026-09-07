from pathlib import Path

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry, load_skill_tree


def test_registry_rejects_duplicate_ids() -> None:
    skill = SkillRecord("python-env", "Python Env", "desc", "body", "x")
    registry = SkillRegistry([skill])
    with pytest.raises(ValueError, match="duplicate skill_id"):
        registry.add(skill)


def test_loader_reads_agent_skill_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "python-env"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: python-env
description: Diagnose Python environment failures.
metadata:
  hermes:
    tags: [python, uv]
---

# Workflow
Inspect the environment before changing dependencies.
""",
        encoding="utf-8",
    )

    skills = load_skill_tree(tmp_path)

    assert len(skills) == 1
    assert skills[0].skill_id == "python-env"
    assert skills[0].tags == ("python", "uv")
    assert "Inspect the environment" in skills[0].body
    assert skills[0].content_hash
