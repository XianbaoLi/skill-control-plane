from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from skill_control_plane.corpus.freezing import (
    FreezeError,
    SourcePackage,
    _audit_source_packages,
    _assert_selected_roots_do_not_overlap,
    _ensure_disjoint,
    _hash_package,
    _select_subsets,
    freeze_benchmark_corpus,
)


SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_package(
    root: Path,
    category: str,
    name: str,
    *,
    body: str = "Use this skill for deterministic tests.",
    with_files: bool = False,
) -> None:
    package = root / "skills" / category / name
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\n\n# {name}\n{body}\n",
        encoding="utf-8",
    )
    if with_files:
        (package / "references").mkdir()
        (package / "references" / "guide.md").write_text("guide", encoding="utf-8")
        (package / "scripts").mkdir()
        (package / "scripts" / "helper.py").write_text("print('not executed')\n", encoding="utf-8")
        (package / "assets").mkdir()
        (package / "assets" / "table.csv").write_text("id\n1\n", encoding="utf-8")


def _make_source(
    root: Path,
    *,
    category_counts: dict[str, int] | None = None,
    with_nested: bool = False,
) -> Path:
    category_counts = category_counts or {"alpha": 64, "beta": 32, "gamma": 32}
    source_root = root / "source"
    for category, count in category_counts.items():
        for index in range(count):
            _write_package(
                source_root,
                category,
                f"{category}-{index:03d}",
                with_files=index % 3 == 0,
            )
    if with_nested:
        _write_package(source_root, "alpha", "nested-router")
        nested_root = source_root / "skills" / "alpha" / "nested-router"
        (nested_root / "inner").mkdir()
        (nested_root / "inner" / "SKILL.md").write_text(
            "---\nname: nested-inner\ndescription: Nested.\n---\n\nbody\n",
            encoding="utf-8",
        )
    _init_git_repo(source_root)
    return source_root


def _init_git_repo(source_root: Path) -> None:
    def git(*arguments: str) -> None:
        environment = {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "PATH": "/usr/bin:/bin",
        }
        subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "source")


def _source_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _freeze(source_root: Path, tmp_path: Path, suffix: str = ""):
    manifest_path = tmp_path / f"manifest{suffix}.json"
    materialized_root = tmp_path / f"corpus{suffix}"
    manifest = freeze_benchmark_corpus(
        source_root,
        source_commit=_source_commit(source_root),
        source_repository="https://github.com/example/source",
        manifest_path=manifest_path,
        materialized_root=materialized_root,
    )
    return manifest, manifest_path, materialized_root


def _package_hash(root: Path) -> str:
    files = tuple(
        path
        for path in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
        if path.is_file()
    )
    return _hash_package(root, files)[0]


def test_selection_is_deterministic_and_nested(tmp_path: Path) -> None:
    source_root = _make_source(tmp_path)
    audited, _ = _audit_source_packages(source_root)
    eligible = [package for package in audited if package.exclusion_reason is None]

    first = _select_subsets(SOURCE_COMMIT, eligible)
    second = _select_subsets(SOURCE_COMMIT, eligible)
    assert first == second
    assert set(first) == {"S32", "S64", "S128"}
    ids32 = {package.skill_id for package in first["S32"]}
    ids64 = {package.skill_id for package in first["S64"]}
    ids128 = {package.skill_id for package in first["S128"]}
    assert ids32 < ids64 < ids128
    assert [
        len({package.skill_id for package in first[name]})
        for name in ("S32", "S64", "S128")
    ] == [32, 64, 128]


def test_freeze_selects_exact_128_and_uses_stratification(tmp_path: Path) -> None:
    source_root = _make_source(tmp_path)
    manifest, _, _ = _freeze(source_root, tmp_path)

    assert manifest["selected_count"] == 128
    assert manifest["package_hash_algorithm"]["name"] == "package-hash-v1"
    assert manifest["eligible_skill_count"] == 128
    assert [len(manifest["subsets"][name]) for name in ("S32", "S64", "S128")] == [32, 64, 128]
    assert set(manifest["subsets"]["S32"]) < set(manifest["subsets"]["S64"])
    assert set(manifest["subsets"]["S64"]) < set(manifest["subsets"]["S128"])
    assert manifest["category_counts"]["subsets"]["S32"] == {
        "alpha": 16,
        "beta": 8,
        "gamma": 8,
    }
    assert manifest["category_counts"]["subsets"]["S64"] == {
        "alpha": 32,
        "beta": 16,
        "gamma": 16,
    }
    assert manifest["category_counts"]["subsets"]["S128"] == {
        "alpha": 64,
        "beta": 32,
        "gamma": 32,
    }
    assert manifest["package_report"]["package_file_count"] > 128
    assert manifest["package_report"]["with_references"] > 0
    assert manifest["package_report"]["with_scripts"] > 0
    assert manifest["package_report"]["with_assets"] > 0


def test_same_source_commit_rerun_produces_same_manifest(tmp_path: Path) -> None:
    source_root = _make_source(tmp_path)
    first, first_path, first_root = _freeze(source_root, tmp_path, suffix="-first")
    second, second_path, second_root = _freeze(source_root, tmp_path, suffix="-second")

    assert first == second
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
    for subset_name in ("S32", "S64", "S128"):
        assert _package_hash(first_root / subset_name) == _package_hash(
            second_root / subset_name
        )


def test_duplicate_and_malformed_skills_are_excluded(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_package(source_root, "alpha", "valid")
    _write_package(source_root, "beta", "beta-valid")
    _write_package(source_root, "alpha", "broken", body="")
    (source_root / "skills" / "alpha" / "broken" / "SKILL.md").write_text(
        "---\nname: broken\n---\n", encoding="utf-8"
    )

    audited, _ = _audit_source_packages(source_root)
    by_path = {package.relative_path: package for package in audited}
    assert by_path["skills/beta/beta-valid"].exclusion_reason is None
    assert by_path["skills/alpha/broken"].exclusion_reason == "body is empty"

    duplicate_source = tmp_path / "duplicate-source"
    _write_package(duplicate_source, "alpha", "same-id")
    _write_package(duplicate_source, "beta", "other")
    (duplicate_source / "skills" / "beta" / "same-id").mkdir()
    (duplicate_source / "skills" / "beta" / "same-id" / "SKILL.md").write_text(
        "---\nname: same-id\ndescription: Duplicate.\n---\n\nbody\n",
        encoding="utf-8",
    )
    audited, _ = _audit_source_packages(duplicate_source)
    reasons = {
        package.relative_path: package.exclusion_reason
        for package in audited
        if package.exclusion_reason is not None
    }
    assert len(reasons) == 1
    assert next(iter(reasons.values())) == "duplicate skill_id"


def test_package_hash_is_deterministic_and_changes_with_package_files(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("skill\n", encoding="utf-8")
    original = _package_hash(root)
    assert original == _package_hash(root)

    (root / "references").mkdir()
    (root / "references" / "guide.md").write_text("first\n", encoding="utf-8")
    with_reference = _package_hash(root)
    assert with_reference != original

    (root / "scripts").mkdir()
    (root / "scripts" / "helper.py").write_text("print('no execution')\n", encoding="utf-8")
    with_script = _package_hash(root)
    assert with_script != with_reference

    (root / "assets").mkdir()
    (root / "assets" / "asset.txt").write_text("asset\n", encoding="utf-8")
    with_asset = _package_hash(root)
    assert with_asset != with_script


def test_package_hash_v1_changes_when_relative_path_changes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root, filename in ((first, "guide.md"), (second, "renamed.md")):
        root.mkdir()
        (root / "SKILL.md").write_text("skill\n", encoding="utf-8")
        (root / filename).write_text("reference\n", encoding="utf-8")

    assert _package_hash(first) == _package_hash(first)
    assert _package_hash(first) != _package_hash(second)


def test_package_hash_v1_frames_ambiguous_concatenations(tmp_path: Path) -> None:
    two_files = tmp_path / "two-files"
    one_file = tmp_path / "one-file"
    two_files.mkdir()
    one_file.mkdir()
    (two_files / "SKILL.md").write_text("skill\n", encoding="utf-8")
    (two_files / "a").write_bytes(b"b\0c")
    (two_files / "c").write_bytes(b"")
    (one_file / "SKILL.md").write_text("skill\n", encoding="utf-8")
    (one_file / "a").write_bytes(b"b\0cc\0")

    def unframed_bytes(root: Path) -> bytes:
        payload = b""
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            if path.is_file():
                relative = path.relative_to(root).as_posix().encode("utf-8")
                payload += relative + b"\0" + path.read_bytes()
        return payload

    assert hashlib.sha256(unframed_bytes(two_files)).digest() == hashlib.sha256(
        unframed_bytes(one_file)
    ).digest()
    assert _package_hash(two_files) != _package_hash(one_file)


def test_selected_package_roots_are_pairwise_non_overlapping() -> None:
    def package(relative_path: str, skill_id: str) -> SourcePackage:
        return SourcePackage(
            relative_path=relative_path,
            category="alpha",
            skill_id=skill_id,
            content_hash="0" * 64,
            package_hash="0" * 64,
            package_file_count=1,
            package_byte_size=1,
        )

    parent = package("skills/alpha/parent", "parent")
    child = package("skills/alpha/parent/child", "child")
    with pytest.raises(FreezeError, match="selected package roots overlap"):
        _assert_selected_roots_do_not_overlap([parent, child])
    with pytest.raises(FreezeError, match="selected package roots overlap"):
        _assert_selected_roots_do_not_overlap([child, parent])


def test_nested_source_packages_are_not_selected(tmp_path: Path) -> None:
    source_root = _make_source(tmp_path, category_counts={"alpha": 128}, with_nested=True)
    audited, _ = _audit_source_packages(source_root)
    by_path = {package.relative_path: package for package in audited}
    assert by_path["skills/alpha/nested-router"].exclusion_reason == (
        "package contains nested SKILL.md files"
    )
    assert "skills/alpha/nested-router/inner" not in by_path

    manifest, _, _ = _freeze(source_root, tmp_path)
    selected_paths = {row["source_relative_path"] for row in manifest["skills"]}
    assert manifest["selected_count"] == 128
    assert "skills/alpha/nested-router" not in selected_paths
    assert "skills/alpha/nested-router/inner" not in selected_paths


def test_symlink_is_excluded_and_disjoint_roots_are_required(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_package(source_root, "alpha", "safe")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    unsafe_package = source_root / "skills" / "alpha" / "unsafe"
    unsafe_package.mkdir()
    (unsafe_package / "SKILL.md").write_text(
        "---\nname: unsafe\ndescription: Unsafe.\n---\n\nbody\n",
        encoding="utf-8",
    )
    (unsafe_package / "linked.md").symlink_to(outside)

    audited, _ = _audit_source_packages(source_root)
    by_path = {package.relative_path: package for package in audited}
    assert by_path["skills/alpha/safe"].exclusion_reason is None
    assert by_path["skills/alpha/unsafe"].exclusion_reason == "symlink: linked.md"
    assert not (tmp_path / "linked.md").exists()

    with pytest.raises(FreezeError, match="disjoint"):
        _ensure_disjoint(source_root, source_root / "materialized")


def test_source_commit_mismatch_fails_closed(tmp_path: Path) -> None:
    source_root = _make_source(tmp_path, category_counts={"alpha": 128})
    with pytest.raises(FreezeError, match="source commit is missing|source commit mismatch"):
        freeze_benchmark_corpus(
            source_root,
            source_commit="f" * 40,
            source_repository="https://github.com/example/source",
            manifest_path=tmp_path / "manifest.json",
            materialized_root=tmp_path / "corpus",
        )


def test_materialized_subsets_load_and_manifest_has_no_machine_path(tmp_path: Path) -> None:
    source_root = _make_source(tmp_path)
    manifest, manifest_path, materialized_root = _freeze(source_root, tmp_path)

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert json.loads(manifest_text) == manifest
    for row in manifest["skills"]:
        copied_root = (
            materialized_root
            / "S128"
            / "skills"
            / row["category"]
            / row["skill_id"]
        )
        assert _package_hash(copied_root) == row["package_hash"]
    for subset_name in ("S32", "S64", "S128"):
        subset_manifest = materialized_root / "manifests" / f"{subset_name}.json"
        assert str(tmp_path) not in subset_manifest.read_text(encoding="utf-8")
        subset = json.loads(subset_manifest.read_text(encoding="utf-8"))
        assert subset["selected_count"] == int(subset_name[1:])
    assert manifest["validation"] == {
        "S32": {"skill_count": 32, "unique_skill_count": 32, "valid": True},
        "S64": {"skill_count": 64, "unique_skill_count": 64, "valid": True},
        "S128": {"skill_count": 128, "unique_skill_count": 128, "valid": True},
    }
