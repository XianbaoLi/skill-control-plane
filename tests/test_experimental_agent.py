import json
from copy import deepcopy

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.capability_loading import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.runtime.experimental_agent import ExperimentalSkillAgent, AgentStepLimitError


@pytest.fixture
def harness():
    discovery = SkillDiscovery(SkillRegistry([
        SkillRecord('pdf', 'PDF', 'extract PDF text', 'PDF_BODY_ONLY_SELECTED', ''),
        SkillRecord('slides', 'Slides', 'presentation slides', 'SLIDES_BODY_ONLY_SELECTED', ''),
        SkillRecord('mail', 'Mail', 'email inbox', 'MAIL_BODY_ONLY_SELECTED', ''),
        SkillRecord('hidden', 'Astronomy', 'unrelated astronomy', 'HIDDEN_BODY_NEVER_VISIBLE', ''),
    ]))
    return RuntimeCapabilityHarness(discovery=discovery)


class ScriptedClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def complete_messages(self, messages):
        self.calls.append(deepcopy(messages))
        action = next(self.actions)
        if callable(action):
            action = action(messages)
        return action if isinstance(action, str) else json.dumps(action)


def load(need='extract PDF text and presentation slides'):
    return {'type': 'load_capability', 'need': need}


def apply(action='DIRECT', **kwargs):
    return {'type': 'apply_capability', 'action': action, 'skill_ids': ['pdf'], 'reason': 'task need', **kwargs}


FINAL = {'type': 'final', 'content': 'Task guidance complete.'}


@pytest.mark.parametrize('action', ['DIRECT', 'CREATE', 'EXTEND'])
def test_one_conversation_search_apply_inject_final(harness, action):
    harness.state = RuntimeCapabilityState([ActiveBundle('mail-work', 'Mail work', ('mail',))])
    initial = deepcopy(harness.state)
    def check_search(messages):
        assert harness.state == initial
        assert harness.pending_candidates is not None
        tool = json.loads(messages[-1]['content'])
        assert tool['type'] == 'capability_tool_result'
        assert tool['tool'] == 'load_capability'
        assert {c['skill_id'] for c in tool['result']['candidates']} == {'pdf', 'slides'}
        assert tool['result']['representations']
        assert all('evidence' in c for c in tool['result']['candidates'])
        kwargs = {'purpose': 'Document workflow'} if action == 'CREATE' else {}
        if action == 'EXTEND':
            kwargs['target_bundle_id'] = 'mail-work'
        return apply(action, **kwargs)
    client = ScriptedClient([load(), check_search, FINAL])
    agent = ExperimentalSkillAgent(harness, client)
    assert agent.run('Help process the document') == FINAL['content']
    assert len(client.calls) == 3
    for index, call in enumerate(client.calls):
        assert call[0]['role'] == 'system'
        assert call[1] == {'role': 'user', 'content': 'Help process the document'}
        assert len(call) == 2 + index * 2
        if index:
            assert call[1:len(client.calls[index-1])] == client.calls[index-1][1:]
        assert 'BODY_' not in call[0]['content']
        assert 'MAIL_BODY_ONLY_SELECTED' not in json.dumps(call)
        assert ('PDF_BODY_ONLY_SELECTED' in json.dumps(call[1:])) == (index == 2)
        assert 'SLIDES_BODY_ONLY_SELECTED' not in json.dumps(call)
        assert 'HIDDEN_BODY_NEVER_VISIBLE' not in json.dumps(call)
    metadata = json.loads(client.calls[-1][0]['content'].split('Runtime Bundles (metadata only)\n')[1])
    body_result = json.loads(client.calls[-1][-1]['content'])
    assert body_result['tool'] == 'apply_capability'
    assert body_result['result']['skill_bodies'] == [{'skill_id': 'pdf', 'body': 'PDF_BODY_ONLY_SELECTED'}]
    assert 'direct_skills' not in metadata
    if action == 'DIRECT':
        assert harness.state.direct_skills == {'pdf'}
        assert len(metadata['maintained_bundles']) == 1
    elif action == 'EXTEND':
        assert [m['skill_id'] for m in metadata['maintained_bundles'][0]['members']] == ['mail', 'pdf']
    else:
        assert len(metadata['maintained_bundles']) == 2
        assert any(b['purpose'] == 'Document workflow' and [m['skill_id'] for m in b['members']] == ['pdf'] for b in metadata['maintained_bundles'])
    assert agent.trace[0]['state_before'] == agent.trace[0]['state_after']
    assert agent.trace[1]['selected_skill_ids'] == ['pdf']
    assert agent.trace[1]['action'] == action
    assert agent.trace[2]['history_skill_body_ids'] == ['pdf']
    assert agent.trace[1]['appended_skill_body_ids'] == ['pdf']
    assert all(row['system_skill_body_ids'] == [] for row in agent.trace)
    assert harness.pending_candidates is None


@pytest.mark.parametrize('decision', [
    apply(skill_ids=['invented']), apply(skill_ids=['hidden']),
    apply('EXTEND', target_bundle_id='invented'),
    apply('CREATE', purpose=''), apply(extra='invalid'),
])
def test_invalid_application_preserves_state(harness, decision):
    before = deepcopy(harness.state)
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load(), decision]))
    with pytest.raises(ValueError):
        agent.run('Document task')
    assert harness.state == before
    assert agent.trace[-1]['state_before'] == agent.trace[-1]['state_after']
    assert not harness.loaded_skill_ids


def test_only_latest_candidates_and_single_use(harness):
    harness.search_capability('PDF')
    harness.search_capability('email inbox')
    with pytest.raises(ValueError, match='outside supplied'):
        harness.apply_capability(json.dumps({'action': 'DIRECT', 'skill_ids': ['pdf'], 'reason': 'x'}))
    harness.apply_capability(json.dumps({'action': 'DIRECT', 'skill_ids': ['mail'], 'reason': 'x'}))
    with pytest.raises(ValueError, match='fresh'):
        harness.apply_capability(json.dumps({'action': 'DIRECT', 'skill_ids': ['mail'], 'reason': 'x'}))
    assert harness.loaded_skill_ids == ('mail',)


def test_multiple_loads_in_same_history_and_multiple_direct(harness):
    client = ScriptedClient([load('email inbox'), load(), apply(skill_ids=['pdf', 'slides']), FINAL, FINAL])
    agent = ExperimentalSkillAgent(harness, client)
    agent.run('first task')
    assert harness.loaded_skill_ids == ('pdf', 'slides')
    previous_history = deepcopy(agent.history)
    agent.run('follow-up task')
    assert client.calls[-1][1:-1] == previous_history
    assert client.calls[-1][-1]['content'] == 'follow-up task'
    assert 'BODY_' not in client.calls[-1][0]['content']
    assert 'PDF_BODY_ONLY_SELECTED' in json.dumps(client.calls[-1][1:])
    assert 'SLIDES_BODY_ONLY_SELECTED' in json.dumps(client.calls[-1][1:])
    assert len(agent.trace) == 1


def test_empty_initial_context_and_no_unnecessary_load(harness):
    client = ScriptedClient([FINAL])
    agent = ExperimentalSkillAgent(harness, client)
    agent.run('Say hello')
    assert not harness.loaded_skill_ids
    assert 'BODY_' not in client.calls[0][0]['content']
    assert agent.trace[0]['retrieved_skill_ids'] == []


@pytest.mark.parametrize('action', [
    apply(), {'type': 'unknown'}, {'type': 'final', 'content': ''},
    {'type': 'load_capability', 'need': ''},
    '{"type":"final","type":"final","content":"x"}',
    {'type': 'load_capability', 'need': 3},
])
def test_protocol_errors_are_traced_and_leave_state_unchanged(harness, action):
    agent = ExperimentalSkillAgent(harness, ScriptedClient([action]))
    with pytest.raises(ValueError):
        agent.run('task')
    assert not harness.loaded_skill_ids
    assert agent.trace[0]['error']['type'] == 'ValueError'


def test_step_limit_and_provider_failure(harness):
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load()]), max_steps=1)
    with pytest.raises(AgentStepLimitError):
        agent.run('task')
    assert agent.trace[-1]['error']['type'] == 'AgentStepLimitError'
    def fail(messages):
        raise RuntimeError('provider unavailable')
    agent = ExperimentalSkillAgent(harness, ScriptedClient([fail]))
    with pytest.raises(RuntimeError):
        agent.run('task')
    assert agent.trace[0]['state_before'] == agent.trace[0]['state_after']
    assert agent.trace[0]['error']['type'] == 'RuntimeError'


def test_extend_requires_new_member_and_changes_only_target(harness):
    harness.state = RuntimeCapabilityState([
        ActiveBundle('a', 'Docs', ('pdf',)), ActiveBundle('b', 'Mail', ('mail',))])
    harness.search_capability('PDF')
    with pytest.raises(ValueError, match='new skill'):
        harness.apply_capability(json.dumps({'action': 'EXTEND', 'target_bundle_id': 'a', 'skill_ids': ['pdf'], 'reason': 'x'}))
    agent = ExperimentalSkillAgent(harness, ScriptedClient([
        load(), apply('EXTEND', target_bundle_id='a', skill_ids=['slides']), FINAL]))
    agent.run('Make slides')
    assert harness.state.active_bundles[0].skill_ids == ('pdf', 'slides')
    assert harness.state.active_bundles[1] == ActiveBundle('b', 'Mail', ('mail',))


def test_agent_never_constructs_or_calls_independent_resolver(harness, monkeypatch):
    from skill_control_plane.runtime import capability_loading
    def forbidden(*args, **kwargs):
        pytest.fail('independent resolver/combined loading must not run')
    monkeypatch.setattr(capability_loading.LLMCapabilityResolver, '__init__', forbidden)
    monkeypatch.setattr(capability_loading.RuntimeCapabilityLoader, 'load_capability', forbidden)
    monkeypatch.setattr(harness, 'load_capability', forbidden)
    client = ScriptedClient([load(), apply(), FINAL])
    ExperimentalSkillAgent(harness, client).run('Document task')
    assert len(client.calls) == 3


def test_failed_new_search_invalidates_previous_candidates(harness, monkeypatch):
    harness.search_capability('PDF')
    def fail(*args, **kwargs):
        raise RuntimeError('dense provider failure')
    monkeypatch.setattr(harness.discovery, 'discover_skills', fail)
    with pytest.raises(RuntimeError):
        harness.search_capability('email')
    assert harness.pending_candidates is None
    assert not harness.loaded_skill_ids


def test_empty_search_result_does_not_change_state(harness):
    result = harness.search_capability('zzzznonexistent')
    assert not result.candidates
    with pytest.raises(ValueError, match='outside supplied'):
        harness.apply_capability(json.dumps({'action': 'DIRECT', 'skill_ids': ['pdf'], 'reason': 'x'}))
    assert not harness.loaded_skill_ids


def test_bundle_surface_uses_only_member_metadata(harness):
    from dataclasses import replace
    record = harness.discovery.records['pdf']
    harness.discovery.records['pdf'] = replace(
        record, description='  extract\n PDF text  ' + 'detail ' * 100,
        retrieval_representation='SECRET_CARD_USE_WHEN_CAPABILITIES_LEXICAL_CUES')
    harness.state = RuntimeCapabilityState([
        ActiveBundle('z', 'Documents', ('slides', 'pdf')),
        ActiveBundle('a', 'Mail', ('mail',)),
    ], {'hidden'})
    surface = harness.render_bundle_context()
    data = json.loads(surface)
    assert list(data) == ['maintained_bundles']
    assert [b['bundle_id'] for b in data['maintained_bundles']] == ['a', 'z']
    members = data['maintained_bundles'][1]['members']
    assert [m['skill_id'] for m in members] == ['pdf', 'slides']
    assert members[0]['name'] == 'PDF'
    assert members[0]['short_description'].startswith('extract PDF text detail')
    assert len(members[0]['short_description']) == 240
    assert 'SECRET_CARD' not in surface
    assert 'BODY_' not in surface
    assert 'hidden' not in surface
    agent = ExperimentalSkillAgent(harness, ScriptedClient([FINAL]))
    agent.run('Describe available workflows')
    assert 'BODY_' not in json.dumps(agent.history)
    assert 'BODY_' not in agent.render_system_context()


def test_repeated_selection_appends_each_body_once_across_runs(harness):
    client = ScriptedClient([
        load(), apply(skill_ids=['pdf', 'pdf']), FINAL,
        load(), apply('CREATE', purpose='Docs', skill_ids=['pdf', 'slides']), FINAL,
        FINAL,
    ])
    agent = ExperimentalSkillAgent(harness, client)
    agent.run('One-off PDF')
    agent.run('Maintain document work')
    agent.run('Use the earlier instructions again')
    results = [json.loads(m['content'])['result'] for m in agent.history
               if m['role'] == 'user' and m['content'].startswith('{')
               and json.loads(m['content']).get('tool') == 'apply_capability']
    assert results[1]['selected_skill_ids'] == ['pdf', 'slides']
    assert results[1]['skill_bodies'] == [{'skill_id': 'slides', 'body': 'SLIDES_BODY_ONLY_SELECTED'}]
    for marker in ('PDF_BODY_ONLY_SELECTED', 'SLIDES_BODY_ONLY_SELECTED'):
        assert json.dumps(agent.history).count(marker) == 1
        assert json.dumps(client.calls[-1][1:]).count(marker) == 1
    assert all('BODY_' not in call[0]['content'] for call in client.calls)
    assert 'HIDDEN_BODY_NEVER_VISIBLE' not in json.dumps(client.calls)
    assert agent.trace[0]['history_skill_body_ids'] == ['pdf', 'slides']


def test_harness_apply_returns_selected_bodies_only(harness):
    harness.search_capability('PDF and presentation slides')
    result = harness.apply_capability(json.dumps({
        'action': 'DIRECT', 'skill_ids': ['pdf'], 'reason': 'one-off'}))
    assert result == {'action': 'DIRECT', 'affected_bundle_id': None,
                      'selected_skill_ids': ['pdf'],
                      'skill_bodies': [{'skill_id': 'pdf', 'body': 'PDF_BODY_ONLY_SELECTED'}]}
