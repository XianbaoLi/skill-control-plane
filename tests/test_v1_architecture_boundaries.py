import ast
import inspect
import json
from pathlib import Path

import pytest

from skill_control_plane import (
    CapabilityDecision,
    CoverageClaim,
    SkillControlPlane,
)
from skill_control_plane.discovery import SkillDiscovery
from skill_control_plane.integrations.reference_agent import (
    CAPABILITY_TOOLS,
    ReferenceSkillAgent,
    parse_tool_arguments,
)
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillStore
from skill_control_plane.runtime import CapabilityMemory, DiscoverySession
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness


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


def _store() -> SkillStore:
    return SkillStore([
        SkillRecord("ocr", "OCR", "extract scanned text", "OCR BODY", "/ocr"),
        SkillRecord("slides", "Slides", "create slides", "SLIDE BODY", "/slides"),
    ])


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


def test_reference_agent_depends_only_on_control_plane_public_methods():
    source = inspect.getsource(ReferenceSkillAgent)
    assert "RuntimeCapabilityHarness" not in source
    assert ".state" not in source
    assert ".pending_candidates" not in source
    assert ".discovery_session" not in source
    assert "._store" not in source
    assert "._discovery" not in source
    assert "._memory" not in source
    assert "._session" not in source
    assert "_strict_object" not in source
    assert "self.control_plane" in source


def test_core_has_no_provider_prompt_tool_schema_or_json_serialization():
    core_files = [
        PACKAGE / "runtime" / "control_plane.py",
        PACKAGE / "runtime" / "discovery_session.py",
        PACKAGE / "runtime" / "capability_memory.py",
    ]
    for path in core_files:
        source = path.read_text(encoding="utf-8")
        assert "CAPABILITY_TOOLS" not in source
        assert "AGENT_INSTRUCTIONS" not in source
        assert "object_pairs_hook" not in source
        assert "import json" not in source
        assert "render_system_context" not in source


def test_reference_agent_registers_exactly_three_capability_tools():
    assert [tool["function"]["name"] for tool in CAPABILITY_TOOLS] == [
        "load_capability", "apply_capability", "load_skill_body"]
    assert len(CAPABILITY_TOOLS) == 3


def test_discovery_session_owns_closure_and_sufficiency_without_bundle_commit():
    store = _store()
    discovery = SkillDiscovery(store)
    memory = CapabilityMemory(store)
    session = DiscoverySession(discovery, memory)

    session.search("extract scanned text")
    assert session.audit().sufficiency_transitions == ("SEARCH_MORE",)
    result = session.apply(CapabilityDecision(
        action="DIRECT",
        skill_ids=("ocr",),
        reason="one-turn operation",
        coverage=(CoverageClaim("scan", "skill:ocr"),),
        remaining_gaps=(),
    ))

    assert [(body.skill_id, body.body) for body in result.skill_bodies] == [
        ("ocr", "OCR BODY")]
    assert session.audit().capability_sufficiency_outcome == "COVERED"
    assert memory.snapshot().maintained_bundles == ()


def test_capability_memory_owns_bundle_body_residency_and_exact_reload():
    store = _store()
    memory = CapabilityMemory(store)
    bundle_id, _ = memory.commit(
        action="CREATE", skill_ids=("slides",), purpose="Presentation work")
    assert bundle_id
    assert memory.state.skill_body_states == {"slides": "resident"}

    memory.mark_all_skill_bodies_evicted()
    assert memory.state.skill_body_states == {"slides": "evicted"}
    assert memory.load_skill_body("slides").body == "SLIDE BODY"
    assert memory.state.skill_body_states == {"slides": "resident"}


def test_public_control_plane_full_structured_lifecycle():
    store = _store()
    plane = SkillControlPlane(store, discovery=SkillDiscovery(store))
    plane.begin_turn()
    search = plane.search_capability("create slides")
    assert search.candidates[0].skill_id == "slides"
    assert search.retrieval_trace.backend == "bm25"

    applied = plane.apply_capability(CapabilityDecision(
        action="CREATE",
        skill_ids=("slides",),
        reason="reusable presentation work",
        purpose="Presentation work",
        coverage=(CoverageClaim("slides", "skill:slides"),),
    ))
    assert applied.skill_bodies[0].body == "SLIDE BODY"
    snapshot = plane.context_snapshot()
    assert snapshot.maintained_bundles[0].members[0].body_state == "resident"

    plane.mark_skill_body_evicted("slides")
    assert plane.context_snapshot().skill_body_states == {"slides": "evicted"}
    reloaded = plane.load_skill_body("slides")
    assert reloaded.status == "loaded" and reloaded.body == "SLIDE BODY"
    plane.mark_all_skill_bodies_evicted()
    assert plane.context_snapshot().skill_body_states == {"slides": "evicted"}
    assert plane.turn_audit().body_load_count == 1


def test_strict_tool_argument_parsing_is_public_integration_behavior():
    parsed = parse_tool_arguments("apply_capability", json.dumps({
        "action": "DIRECT",
        "skill_ids": ["ocr"],
        "reason": "one turn",
        "coverage": [{"need": "scan", "covered_by": "skill:ocr"}],
        "remaining_gaps": [],
    }))
    assert isinstance(parsed, CapabilityDecision)
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_tool_arguments("load_capability", '{"need":"a","need":"b"}')


def test_historical_harness_is_deprecated_compatible_and_not_formal_api():
    store = _store()
    with pytest.warns(DeprecationWarning, match="deprecated"):
        harness = RuntimeCapabilityHarness(discovery=SkillDiscovery(store))
    harness.begin_turn()
    result = harness.search_capability("extract scanned text")
    assert result.candidates[0].skill_id == "ocr"
    applied = harness.apply_capability({
        "action": "DIRECT",
        "skill_ids": ["ocr"],
        "reason": "historical experiment",
        "coverage": [{"need": "scan", "covered_by": "skill:ocr"}],
        "remaining_gaps": [],
    })
    assert applied["skill_bodies"] == [{"skill_id": "ocr", "body": "OCR BODY"}]
    assert "RuntimeCapabilityHarness" not in __import__(
        "skill_control_plane").__all__
    assert not hasattr(SkillControlPlane, "load_capability")
