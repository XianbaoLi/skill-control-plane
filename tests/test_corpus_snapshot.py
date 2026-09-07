from pathlib import Path

from skill_control_plane.corpus import build_corpus_manifest, write_corpus_manifest


def _write_skill(root: Path, category: str, name: str, body: str) -> None:
    skill_dir = root / category / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Skill for {name}.
tags: [{category}, test]
---

# Workflow
{body}
""",
        encoding="utf-8",
    )


def test_manifest_is_content_free_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "devops", "docker-debugging", "secret body text")
    _write_skill(root, "github", "pr-review", "another private body")

    first = build_corpus_manifest(
        root,
        source="test-local",
        harness_commit="abc123",
    )
    second = build_corpus_manifest(
        root,
        source="test-local",
        harness_commit="abc123",
    )

    assert first["skill_count"] == 2
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["harness_commit"] == "abc123"
    assert first["skills"][0]["relative_path"]
    assert "body" not in first["skills"][0]
    assert "secret body text" not in str(first)
    assert str(tmp_path) not in str(first)


def test_snapshot_id_changes_when_skill_content_changes(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "devops", "docker-debugging", "version one")
    first = build_corpus_manifest(root, source="test-local")

    skill_path = root / "devops" / "docker-debugging" / "SKILL.md"
    skill_path.write_text(
        skill_path.read_text(encoding="utf-8").replace("version one", "version two"),
        encoding="utf-8",
    )
    second = build_corpus_manifest(root, source="test-local")

    assert first["snapshot_id"] != second["snapshot_id"]


def test_writer_creates_parent_directory(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "research", "paper-reading", "Read evidence first")
    output = tmp_path / "artifacts" / "manifest.json"

    manifest = write_corpus_manifest(root, output, source="test-local")

    assert output.exists()
    assert manifest["skill_count"] == 1
