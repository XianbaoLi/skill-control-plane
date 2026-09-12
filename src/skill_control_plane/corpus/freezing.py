from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from skill_control_plane.registry import SkillStore
from skill_control_plane.registry.loader import decode_skill_bytes, skill_content_hash


CORPUS_VERSION = "benchmark-corpus-v0.1"
SELECTION_ALGORITHM_NAME = "deterministic-stratified-largest-remainder"
SELECTION_ALGORITHM_VERSION = "v1"
SUBSET_SIZES = (32, 64, 128)
SKILL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class SourcePackage:
    relative_path: str
    category: str
    skill_id: str
    content_hash: str
    package_hash: str
    package_file_count: int
    package_byte_size: int
    exclusion_reason: str | None = None


class FreezeError(RuntimeError):
    """Raised when a source tree cannot be frozen safely or deterministically."""


def _sorted_entries(path: Path) -> list[Path]:
    return sorted(path.iterdir(), key=lambda entry: entry.name)


def _package_files(package_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []

    def walk(directory: Path) -> None:
        for entry in _sorted_entries(directory):
            if entry.is_symlink():
                relative = entry.relative_to(package_root).as_posix()
                raise ValueError(f"symlink: {relative}")
            if entry.is_dir():
                walk(entry)
            elif entry.is_file():
                files.append(entry)
            else:
                relative = entry.relative_to(package_root).as_posix()
                raise ValueError(f"unsupported file type: {relative}")

    walk(package_root)
    return tuple(sorted(files, key=lambda path: path.relative_to(package_root).as_posix()))


def _hash_package(package_root: Path, files: tuple[Path, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_size = 0
    for file_path in files:
        relative_path = file_path.relative_to(package_root).as_posix()
        content = file_path.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        total_size += len(content)
    return digest.hexdigest(), total_size


def _parse_skill_md(data: bytes) -> dict[str, Any]:
    text = decode_skill_bytes(data)
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("frontmatter delimiters missing")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as error:
        raise ValueError("frontmatter is unclosed") from error
    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as error:
        raise ValueError("frontmatter is unparseable") from error
    if not isinstance(frontmatter, dict):
        raise ValueError("frontmatter is not a mapping")
    if not "\n".join(lines[end + 1 :]).strip():
        raise ValueError("body is empty")
    return frontmatter


def _has_descendant_skill_md(package_root: Path) -> bool:
    skill_md = package_root / "SKILL.md"
    return any(
        path != skill_md
        for path in package_root.rglob("SKILL.md")
        if not any(parent.is_symlink() for parent in path.parents)
    )


def _count_skill_md_files(skills_root: Path) -> int:
    count = 0
    for directory, dirnames, filenames in os.walk(skills_root, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not (Path(directory) / name).is_symlink()
        )
        count += sum(filename == "SKILL.md" for filename in filenames)
    return count


def _verify_source_commit(source_root: Path, expected_source_commit: str) -> None:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FreezeError(f"source git command failed: {result.stderr.strip()}")
        return result.stdout.strip()

    try:
        git("cat-file", "-e", f"{expected_source_commit}^{{commit}}")
    except FreezeError as error:
        raise FreezeError(f"source commit is missing: {expected_source_commit}") from error
    actual_source_commit = git("rev-parse", "HEAD")
    if actual_source_commit != expected_source_commit:
        raise FreezeError(
            "source commit mismatch: "
            f"expected={expected_source_commit} actual={actual_source_commit}"
        )
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise FreezeError("source worktree is not clean")


def _audit_source_packages(source_root: Path) -> tuple[list[SourcePackage], int]:
    skills_root = source_root / "skills"
    if not skills_root.is_dir():
        raise FreezeError("source does not contain a skills directory")

    packages: list[SourcePackage] = []
    for category_dir in _sorted_entries(skills_root):
        if category_dir.is_symlink() or not category_dir.is_dir():
            continue
        for package_dir in _sorted_entries(category_dir):
            if package_dir.is_symlink() or not package_dir.is_dir():
                continue
            relative_path = package_dir.relative_to(source_root).as_posix()
            category = category_dir.name
            skill_id = ""
            content_hash = ""

            try:
                files = _package_files(package_dir)
            except ValueError as error:
                packages.append(
                    SourcePackage(relative_path, category, "", "", "", 0, 0, str(error))
                )
                continue

            skill_md = package_dir / "SKILL.md"
            if not skill_md.is_file():
                packages.append(
                    SourcePackage(
                        relative_path,
                        category,
                        "",
                        "",
                        "",
                        len(files),
                        sum(path.stat().st_size for path in files),
                        "SKILL.md missing",
                    )
                )
                continue

            content_hash = skill_content_hash(skill_md.read_bytes())
            package_hash, package_size = _hash_package(package_dir, files)
            try:
                frontmatter = _parse_skill_md(skill_md.read_bytes())
            except ValueError as error:
                packages.append(
                    SourcePackage(
                        relative_path,
                        category,
                        "",
                        content_hash,
                        package_hash,
                        len(files),
                        package_size,
                        str(error),
                    )
                )
                continue

            skill_id = str(frontmatter.get("name") or package_dir.name).strip()
            exclusion_reason: str | None = None
            if not SKILL_ID_PATTERN.fullmatch(skill_id):
                exclusion_reason = "frontmatter name is illegal"
            elif _has_descendant_skill_md(package_root=package_dir):
                exclusion_reason = "package contains nested SKILL.md files"

            packages.append(
                SourcePackage(
                    relative_path=relative_path,
                    category=category,
                    skill_id=skill_id,
                    content_hash=content_hash,
                    package_hash=package_hash,
                    package_file_count=len(files),
                    package_byte_size=package_size,
                    exclusion_reason=exclusion_reason,
                )
            )

    packages.sort(key=lambda package: package.relative_path)
    id_counts = Counter(
        package.skill_id
        for package in packages
        if package.exclusion_reason is None and package.skill_id
    )
    marked: list[SourcePackage] = []
    seen_duplicate_ids: set[str] = set()
    for package in packages:
        if package.exclusion_reason is None and id_counts[package.skill_id] > 1:
            if package.skill_id in seen_duplicate_ids:
                package = SourcePackage(
                    relative_path=package.relative_path,
                    category=package.category,
                    skill_id=package.skill_id,
                    content_hash=package.content_hash,
                    package_hash=package.package_hash,
                    package_file_count=package.package_file_count,
                    package_byte_size=package.package_byte_size,
                    exclusion_reason="duplicate skill_id",
                )
            else:
                seen_duplicate_ids.add(package.skill_id)
        marked.append(package)
    return marked, _count_skill_md_files(skills_root)


def _selection_order(source_commit: str, packages: list[SourcePackage]) -> list[SourcePackage]:
    def rank(package: SourcePackage) -> tuple[str, str]:
        value = hashlib.sha256(
            f"{source_commit}\0{package.relative_path}".encode("utf-8")
        ).hexdigest()
        return value, package.relative_path

    return sorted(packages, key=rank)


def _category_quotas(counts: dict[str, int], selected_count: int) -> dict[str, int]:
    total = sum(counts.values())
    if not counts or selected_count > total:
        raise FreezeError("selected count exceeds eligible Skill count")

    ideal = {
        category: selected_count * count / total
        for category, count in counts.items()
    }
    quotas = {category: int(value) for category, value in ideal.items()}
    remaining = selected_count - sum(quotas.values())
    order = sorted(
        ideal,
        key=lambda category: (-(ideal[category] - quotas[category]), category),
    )
    for category in order[:remaining]:
        quotas[category] += 1
    if any(quota > counts[category] for category, quota in quotas.items()):
        raise FreezeError("category allocation exceeds eligible source skills")
    return dict(sorted(quotas.items()))


def _select_subsets(
    source_commit: str, eligible: list[SourcePackage]
) -> dict[str, list[SourcePackage]]:
    counts = dict(sorted(Counter(package.category for package in eligible).items()))
    by_category = {
        category: _selection_order(
            source_commit,
            [package for package in eligible if package.category == category],
        )
        for category in counts
    }
    quotas = {
        f"S{size}": _category_quotas(counts, size) for size in SUBSET_SIZES
    }
    for smaller, larger in zip(SUBSET_SIZES, SUBSET_SIZES[1:]):
        small_key = f"S{smaller}"
        large_key = f"S{larger}"
        if any(
            quotas[small_key][category] > quotas[large_key][category]
            for category in counts
        ):
            raise FreezeError("subset category quotas are not nested")

    subsets: dict[str, list[SourcePackage]] = {}
    for subset_name, subset_quotas in quotas.items():
        selected: list[SourcePackage] = []
        for category in counts:
            selected.extend(by_category[category][: subset_quotas[category]])
        if len(selected) != int(subset_name[1:]):
            raise FreezeError(f"selected count mismatch for {subset_name}")
        if len({package.skill_id for package in selected}) != len(selected):
            raise FreezeError(f"duplicate selected skill_id for {subset_name}")
        subsets[subset_name] = sorted(selected, key=lambda package: package.relative_path)
    return dict(sorted(subsets.items()))


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _ensure_disjoint(source_root: Path, materialized_root: Path) -> None:
    if _is_within(source_root, materialized_root) or _is_within(materialized_root, source_root):
        raise FreezeError("source and materialized roots must be disjoint")


def _copy_package(
    package: SourcePackage,
    source_root: Path,
    destination_root: Path,
) -> None:
    source_package_root = source_root / package.relative_path
    destination_package_root = destination_root / "skills" / package.category / package.skill_id
    destination_package_root.mkdir(parents=True, exist_ok=False)

    for source_file in _package_files(source_package_root):
        relative = source_file.relative_to(source_package_root)
        destination_file = destination_package_root / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination_file)

    copied_files = _package_files(destination_package_root)
    copied_hash, copied_size = _hash_package(destination_package_root, copied_files)
    if copied_hash != package.package_hash or copied_size != package.package_byte_size:
        raise FreezeError(f"copy verification failed for {package.skill_id}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _subset_manifest(
    main_manifest: dict[str, Any], subset_name: str
) -> dict[str, Any]:
    skill_ids = set(main_manifest["subsets"][subset_name])
    records = [row for row in main_manifest["skills"] if row["skill_id"] in skill_ids]
    category_counts = Counter(record["category"] for record in records)
    return {
        "corpus_version": main_manifest["corpus_version"],
        "source_repository": main_manifest["source_repository"],
        "source_commit": main_manifest["source_commit"],
        "subset": subset_name,
        "selected_count": len(records),
        "category_counts": dict(sorted(category_counts.items())),
        "skills": sorted(records, key=lambda row: row["source_relative_path"]),
    }


def _validate_materialized(
    materialized_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    manifest_by_id = {row["skill_id"]: row for row in manifest["skills"]}
    for subset_name, expected_ids in manifest["subsets"].items():
        subset_root = materialized_root / subset_name
        skill_tree_root = subset_root / "skills"
        store = SkillStore.from_tree(skill_tree_root)
        ids = {record.skill_id for record in store}
        if len(store) != len(ids) or ids != set(expected_ids):
            raise FreezeError(f"SkillStore validation failed for {subset_name}")
        for record in store:
            row = manifest_by_id[record.skill_id]
            skill_md = Path(record.source_path).resolve()
            subset_root_resolved = skill_tree_root.resolve()
            if not _is_within(skill_md, subset_root_resolved):
                raise FreezeError(f"materialized source_path outside subset for {record.skill_id}")
            if not record.body.strip():
                raise FreezeError(f"empty Skill body for {record.skill_id}")
            if record.category != row["category"]:
                raise FreezeError(f"category mismatch for {record.skill_id}")
        validation[subset_name] = {
            "skill_count": len(store),
            "unique_skill_count": len(ids),
            "valid": True,
        }
    return validation


def _package_report(manifest: dict[str, Any]) -> dict[str, Any]:
    skills = manifest["skills"]
    return {
        "package_file_count": sum(row["package_file_count"] for row in skills),
        "package_byte_size": sum(row["package_byte_size"] for row in skills),
        "with_references": sum(
            row["contains_references"] for row in skills
        ),
        "with_scripts": sum(row["contains_scripts"] for row in skills),
        "with_assets": sum(row["contains_assets"] for row in skills),
    }


def freeze_benchmark_corpus(
    source_root: str | Path,
    *,
    source_commit: str,
    source_repository: str,
    manifest_path: str | Path,
    materialized_root: str | Path,
    clean_output: bool = False,
) -> dict[str, Any]:
    """Audit, select, materialize, and validate a frozen benchmark corpus."""

    source_path = Path(source_root).resolve()
    materialized_path = Path(materialized_root).resolve()
    output_manifest_path = Path(manifest_path).resolve()
    _ensure_disjoint(source_path, materialized_path)
    _verify_source_commit(source_path, source_commit)

    audited, source_skill_md_count = _audit_source_packages(source_path)
    eligible = [package for package in audited if package.exclusion_reason is None]
    if len(eligible) < SUBSET_SIZES[-1]:
        raise FreezeError("not enough eligible Skills")
    subsets = _select_subsets(source_commit, eligible)
    selected = subsets["S128"]

    if clean_output and materialized_path.exists():
        shutil.rmtree(materialized_path)
    if materialized_path.exists():
        raise FreezeError(f"materialized root already exists: {materialized_path}")

    for subset_name, subset in subsets.items():
        subset_root = materialized_path / subset_name
        for package in subset:
            _copy_package(package, source_path, subset_root)

    category_counts = Counter(package.category for package in eligible)
    exclusions = [
        {"source_relative_path": package.relative_path, "reason": package.exclusion_reason}
        for package in audited
        if package.exclusion_reason is not None
    ]
    skills: list[dict[str, Any]] = []
    for package in selected:
        package_root = source_path / package.relative_path
        files = _package_files(package_root)
        subset_names = sorted(
            name for name, members in subsets.items() if package in members
        )
        relative_names = {path.relative_to(package_root).as_posix() for path in files}
        def contains_dir(name: str) -> bool:
            return any(
                relative == name or relative.startswith(f"{name}/")
                for relative in relative_names
            )

        skills.append(
            {
                "skill_id": package.skill_id,
                "category": package.category,
                "source_relative_path": package.relative_path,
                "skill_md_content_hash": package.content_hash,
                "package_hash": package.package_hash,
                "package_file_count": package.package_file_count,
                "package_byte_size": package.package_byte_size,
                "contains_references": contains_dir("references"),
                "contains_scripts": contains_dir("scripts"),
                "contains_assets": contains_dir("assets"),
                "subsets": subset_names,
            }
        )

    selected_counts = Counter(package.category for package in selected)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "corpus_version": CORPUS_VERSION,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "selection_algorithm": {
            "name": SELECTION_ALGORITHM_NAME,
            "version": SELECTION_ALGORITHM_VERSION,
            "order_hash": "sha256(source_commit + NUL + source_relative_path)",
        },
        "source_skill_md_count": source_skill_md_count,
        "source_skill_count": len(audited),
        "eligible_skill_count": len(eligible),
        "excluded_skill_count": len(exclusions),
        "selected_count": len(selected),
        "category_counts": {
            "eligible": dict(sorted(category_counts.items())),
            "selected_s128": dict(sorted(selected_counts.items())),
            "subsets": {
                name: dict(sorted(Counter(row["category"] for row in skills if name in row["subsets"]).items()))
                for name in ("S32", "S64", "S128")
            },
        },
        "exclusions": exclusions,
        "subsets": {
            name: [row["skill_id"] for row in skills if name in row["subsets"]]
            for name in ("S32", "S64", "S128")
        },
        "skills": sorted(skills, key=lambda row: row["source_relative_path"]),
        "source_license": {
            "spdx_identifier": "MIT",
            "source_path": "LICENSE",
            "attribution": source_repository,
        },
    }
    manifest["package_report"] = _package_report(manifest)

    if any(Path(part).is_absolute() or ".." in Path(part).parts for row in skills for part in (row["source_relative_path"],)):
        raise FreezeError("manifest contains an unsafe source path")
    validation = _validate_materialized(materialized_path, manifest)
    manifest["validation"] = validation

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for subset_name in ("S32", "S64", "S128"):
        _write_json(
            materialized_path / "manifests" / f"{subset_name}.json",
            _subset_manifest(manifest, subset_name),
        )
    _write_json(
        materialized_path / "source-metadata.json",
        {
            "corpus_version": CORPUS_VERSION,
            "source_repository": source_repository,
            "source_commit": source_commit,
            "source_skill_md_count": source_skill_md_count,
            "source_skill_count": len(audited),
            "eligible_skill_count": len(eligible),
            "excluded_skill_count": len(exclusions),
            "source_license": manifest["source_license"],
        },
    )
    return manifest
