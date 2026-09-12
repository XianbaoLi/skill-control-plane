import ast
import json
from pathlib import Path

from skill_control_plane.discovery import SkillDiscovery
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillStore
from skill_control_plane.runtime import CapabilityMemory, DiscoverySession


PACKAGE = Path(__file__).parents[1] / "src" / "skill_control_plane"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_v1_production_dependency_boundaries():
    forbidden = {
        "registry": ("skill_control_plane.discovery", "skill_control_plane.runtime",
                     "skill_control_plane.integrations", "skill_control_plane.evals"),
        "discovery": ("skill_control_plane.runtime", "skill_control_plane.integrations",
                      "skill_control_plane.evals"),
        "runtime": ("skill_control_plane.integrations", "skill_control_plane.evals"),
        "integrations": ("skill_control_plane.evals",),
    }
    violations = []
    for layer, prefixes in forbidden.items():
        for path in (PACKAGE / layer).glob("*.py"):
            for imported in _imports(path):
                if imported.startswith(prefixes):
                    violations.append((str(path.relative_to(PACKAGE)), imported))
    assert violations == []


def test_discovery_session_owns_closure_and_sufficiency_without_bundle_commit():
    store = SkillStore([
        SkillRecord("ocr", "OCR", "extract scanned text", "OCR BODY", "/ocr"),
    ])
    discovery = SkillDiscovery(store)
    memory = CapabilityMemory(store)
    session = DiscoverySession(discovery, memory)

    session.search("extract scanned text")
    assert session.audit()["sufficiency_transitions"] == ["SEARCH_MORE"]
    result = session.apply(json.dumps({
        "action": "DIRECT",
        "skill_ids": ["ocr"],
        "reason": "one-turn operation",
        "coverage": [{"need": "scan", "covered_by": "skill:ocr"}],
        "remaining_gaps": [],
    }))

    assert result["skill_bodies"] == [{"skill_id": "ocr", "body": "OCR BODY"}]
    assert session.audit()["capability_sufficiency_outcome"] == "COVERED"
    assert json.loads(memory.render_bundle_card())["maintained_bundles"] == []


def test_capability_memory_owns_bundle_body_residency_and_exact_reload():
    store = SkillStore([
        SkillRecord("slides", "Slides", "create slides", "SLIDE BODY", "/slides"),
    ])
    memory = CapabilityMemory(store)
    bundle_id, _ = memory.commit(
        action="CREATE", skill_ids=("slides",), purpose="Presentation work")
    assert bundle_id
    assert memory.state.skill_body_states == {"slides": "resident"}

    memory.mark_all_skill_bodies_evicted()
    assert memory.state.skill_body_states == {"slides": "evicted"}
    assert memory.load_skill_body("slides")["body"] == "SLIDE BODY"
    assert memory.state.skill_body_states == {"slides": "resident"}
