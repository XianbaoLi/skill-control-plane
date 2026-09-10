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
        assert 'MAIL_BODY_ONLY_SELECTED' in call[0]['content']
        assert ('PDF_BODY_ONLY_SELECTED' in call[0]['content']) == (index == 2)
        assert 'SLIDES_BODY_ONLY_SELECTED' not in json.dumps(call)
        assert 'HIDDEN_BODY_NEVER_VISIBLE' not in json.dumps(call)
    metadata = json.loads(client.calls[-1][0]['content'].split('Runtime capability metadata\n')[1].split('\n')[1])
    if action == 'DIRECT':
        assert metadata['direct_skills'] == ['pdf']
    elif action == 'EXTEND':
        assert metadata['maintained_bundles'][0]['skill_ids'] == ['mail', 'pdf']
    else:
        assert len(metadata['maintained_bundles']) == 2
        assert any(b['purpose'] == 'Document workflow' and b['skill_ids'] == ['pdf'] for b in metadata['maintained_bundles'])
    assert agent.trace[0]['state_before'] == agent.trace[0]['state_after']
    assert agent.trace[1]['selected_skill_ids'] == ['pdf']
    assert agent.trace[1]['action'] == action
    assert agent.trace[2]['injected_skill_ids'] == ['mail', 'pdf']
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
    assert 'PDF_BODY_ONLY_SELECTED' in client.calls[-1][0]['content']
    assert 'SLIDES_BODY_ONLY_SELECTED' in client.calls[-1][0]['content']
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
