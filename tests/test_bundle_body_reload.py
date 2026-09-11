import json
from copy import deepcopy
from uuid import uuid4

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.experimental_agent import CAPABILITY_TOOLS, ExperimentalSkillAgent


def tool(name, arguments):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'call-' + uuid4().hex, 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)},
    }]}


class ScriptedClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def complete_messages(self, messages, *, tools):
        assert tools == CAPABILITY_TOOLS
        self.calls.append(deepcopy(messages))
        action = next(self.actions)
        return action(messages) if callable(action) else action


@pytest.fixture
def harness():
    registry = SkillRegistry([
        SkillRecord('powerpoint', 'PowerPoint', 'create and edit presentation slides',
                    'FULL_POWERPOINT_SKILL_BODY', '/skills/powerpoint/SKILL.md'),
        SkillRecord('pdf', 'PDF', 'read PDF documents',
                    'FULL_PDF_SKILL_BODY', '/skills/pdf/SKILL.md'),
    ])
    return RuntimeCapabilityHarness(discovery=SkillDiscovery(registry))


def create_powerpoint(harness):
    harness.search_capability('create and edit presentation slides')
    return harness.apply_capability(json.dumps({
        'action': 'CREATE', 'skill_ids': ['powerpoint'],
        'reason': 'maintain the presentation workflow',
        'purpose': 'Create and revise the document presentation',
    }))


def test_apply_create_injects_once_and_marks_bundle_body_resident(harness):
    result = create_powerpoint(harness)
    assert result['skill_bodies'] == [
        {'skill_id': 'powerpoint', 'body': 'FULL_POWERPOINT_SKILL_BODY'}]
    assert harness.state.active_bundles[0].skill_ids == ('powerpoint',)
    assert harness.state.skill_body_states == {'powerpoint': 'resident'}


def test_evicted_exact_reload_uses_store_without_retrieval(harness, monkeypatch):
    create_powerpoint(harness)
    harness.mark_skill_body_evicted('powerpoint')
    calls = {'retrieval': 0, 'body': 0}

    def no_retrieval(*args, **kwargs):
        calls['retrieval'] += 1
        pytest.fail('exact body reload must not run retrieval')

    exact_load = harness.discovery.registry.load_skill_body

    def count_body(skill_id):
        calls['body'] += 1
        return exact_load(skill_id)

    monkeypatch.setattr(harness.discovery, 'discover_skills', no_retrieval)
    monkeypatch.setattr(harness.discovery.registry, 'load_skill_body', count_body)
    result = harness.load_skill_body('powerpoint')
    assert result == {
        'status': 'loaded', 'skill_id': 'powerpoint',
        'bundle_ids': [harness.state.active_bundles[0].bundle_id],
        'body': 'FULL_POWERPOINT_SKILL_BODY',
    }
    assert calls == {'retrieval': 0, 'body': 1}
    assert harness.state.skill_body_states['powerpoint'] == 'resident'


def test_resident_reload_is_idempotent_and_does_not_read_store(harness, monkeypatch):
    create_powerpoint(harness)
    monkeypatch.setattr(harness.discovery.registry, 'load_skill_body',
                        lambda skill_id: pytest.fail('must not inject again'))
    result = harness.load_skill_body('powerpoint')
    assert result['status'] == 'already_resident'
    assert 'body' not in result
    assert harness.state.skill_body_states == {'powerpoint': 'resident'}


@pytest.mark.parametrize('skill_id', ['pdf', 'unknown'])
def test_non_bundle_reload_rejected_without_retrieval_or_state_change(
        harness, monkeypatch, skill_id):
    create_powerpoint(harness)
    before = deepcopy(harness.state)
    monkeypatch.setattr(harness.discovery, 'discover_skills',
                        lambda *args, **kwargs: pytest.fail('must not retrieve'))
    monkeypatch.setattr(harness.discovery.registry, 'load_skill_body',
                        lambda exact_id: pytest.fail('must not read store'))
    with pytest.raises(ValueError, match='Bundle member'):
        harness.load_skill_body(skill_id)
    assert harness.state == before


def test_eviction_and_reload_change_only_residency(harness):
    create_powerpoint(harness)
    bundles = deepcopy(harness.state.active_bundles)
    direct = set(harness.state.direct_skills)
    harness.mark_all_skill_bodies_evicted()
    assert harness.state.active_bundles == bundles
    assert harness.state.direct_skills == direct
    assert harness.state.skill_body_states == {'powerpoint': 'evicted'}
    harness.load_skill_body('powerpoint')
    assert harness.state.active_bundles == bundles
    assert harness.state.direct_skills == direct
    assert len(harness.state.active_bundles) == 1
    assert harness.state.active_bundles[0].skill_ids == ('powerpoint',)


def test_bundle_surface_is_compact_and_reports_body_state(harness):
    create_powerpoint(harness)
    resident = harness.render_bundle_context()
    assert 'FULL_POWERPOINT_SKILL_BODY' not in resident
    member = json.loads(resident)['maintained_bundles'][0]['members'][0]
    assert member['skill_id'] == 'powerpoint'
    assert member['body_state'] == 'resident'
    harness.mark_all_skill_bodies_evicted()
    member = json.loads(harness.render_bundle_context())['maintained_bundles'][0]['members'][0]
    assert member['body_state'] == 'evicted'


def test_same_agent_second_turn_exact_reload_has_zero_retrieval(harness, monkeypatch):
    def apply_from_candidates(messages):
        assert json.loads(messages[-1]['content'])['candidates'][0]['skill_id'] == 'powerpoint'
        return tool('apply_capability', {
            'action': 'CREATE', 'skill_ids': ['powerpoint'],
            'reason': 'reusable presentation work',
            'purpose': 'Create and revise the document presentation',
        })

    def reload_from_evicted_bundle(messages):
        assert ('Authoritative evicted Bundle Skill IDs (reload before use): '
                '["powerpoint"]') in messages[0]['content']
        surface = json.loads(messages[0]['content'].split(
            'Runtime Bundles (metadata only)\n', 1)[1])
        member = surface['maintained_bundles'][0]['members'][0]
        assert member['skill_id'] == 'powerpoint'
        assert member['body_state'] == 'evicted'
        return tool('load_skill_body', {'skill_id': member['skill_id']})

    client = ScriptedClient([
        tool('load_capability', {'need': 'read a document and create presentation slides'}),
        apply_from_candidates, {'role': 'assistant', 'content': 'PPT created.'},
        reload_from_evicted_bundle,
        {'role': 'assistant', 'content': 'Third slide revised.'},
    ])
    retrieval_count = body_count = 0
    discover = harness.discovery.discover_skills
    exact_load = harness.discovery.registry.load_skill_body

    def count_retrieval(*args, **kwargs):
        nonlocal retrieval_count
        retrieval_count += 1
        return discover(*args, **kwargs)

    def count_body(skill_id):
        nonlocal body_count
        body_count += 1
        return exact_load(skill_id)

    monkeypatch.setattr(harness.discovery, 'discover_skills', count_retrieval)
    monkeypatch.setattr(harness.discovery.registry, 'load_skill_body', count_body)
    agent = ExperimentalSkillAgent(harness, client)
    assert agent.run('读取这个文档并做成 PPT') == 'PPT created.'
    bundle_before = deepcopy(harness.state.active_bundles)
    assert harness.state.skill_body_states == {'powerpoint': 'resident'}
    harness.mark_all_skill_bodies_evicted()
    assert harness.state.skill_body_states == {'powerpoint': 'evicted'}
    assert agent.run('继续修改刚才 PPT 的第三页') == 'Third slide revised.'

    assert retrieval_count == 1  # Turn 1 only; Turn 2 is zero retrieval.
    assert body_count == 1
    assert harness.state.skill_body_states == {'powerpoint': 'resident'}
    assert harness.state.active_bundles == bundle_before
    assert [event['tool'] for row in agent.trace
            for event in row['tool_executions']] == ['load_skill_body']
    reload_result = json.loads(agent.history[-2]['content'])
    assert reload_result['body'] == 'FULL_POWERPOINT_SKILL_BODY'
    assert reload_result['status'] == 'loaded'


def test_native_exact_reload_schema():
    schema = CAPABILITY_TOOLS[2]['function']
    assert schema['name'] == 'load_skill_body'
    assert schema['parameters']['required'] == ['skill_id']
    assert schema['parameters']['additionalProperties'] is False
