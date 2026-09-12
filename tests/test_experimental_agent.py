import json
from copy import deepcopy
from uuid import uuid4

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.capability_loading import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.runtime.experimental_agent import (
    CAPABILITY_TOOLS, ExperimentalSkillAgent, AgentStepLimitError,
)


@pytest.fixture
def harness():
    return RuntimeCapabilityHarness(discovery=SkillDiscovery(SkillRegistry([
        SkillRecord('pdf', 'PDF', 'extract PDF text', 'PDF_BODY_ONLY_SELECTED', ''),
        SkillRecord('slides', 'Slides', 'presentation slides', 'SLIDES_BODY_ONLY_SELECTED', ''),
        SkillRecord('mail', 'Mail', 'email inbox', 'MAIL_BODY_ONLY_SELECTED', ''),
        SkillRecord('hidden', 'Astronomy', 'unrelated astronomy', 'HIDDEN_BODY_NEVER_VISIBLE', ''),
    ])))


def tool(name, args):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call-' + uuid4().hex, 'type': 'function',
         'function': {'name': name, 'arguments': json.dumps(args)}}]}


def load(need='extract PDF text and presentation slides'):
    return tool('load_capability', {'need': need})


def apply(action='DIRECT', **kwargs):
    skill_ids = kwargs.get('skill_ids', ['pdf'])
    audited = {
        'coverage': [
            {'need': f'use {skill_id}', 'covered_by': f'skill:{skill_id}'}
            for skill_id in dict.fromkeys(skill_ids)
        ],
        'remaining_gaps': [],
    }
    return tool('apply_capability', {
        'action': action, 'skill_ids': ['pdf'], 'reason': 'task need',
        **audited, **kwargs})


FINAL = {'role': 'assistant', 'content': 'Task guidance complete. Quotes "and"\nnewlines are ordinary text.'}


class ScriptedClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def complete_messages(self, messages, *, tools):
        assert tools == CAPABILITY_TOOLS
        self.calls.append(deepcopy(messages))
        action = next(self.actions)
        return action(messages) if callable(action) else action


@pytest.mark.parametrize('action', ['DIRECT', 'CREATE', 'EXTEND'])
def test_native_search_apply_bodies_history_and_bundle_surface(harness, action):
    harness.state = RuntimeCapabilityState([ActiveBundle('mail-work', 'Mail work', ('mail',))])
    initial = deepcopy(harness.state)
    def check_search(messages):
        assert harness.state == initial
        assert messages[-1]['role'] == 'tool'
        assert messages[-1]['tool_call_id'] == messages[-2]['tool_calls'][0]['id']
        result = json.loads(messages[-1]['content'])
        assert {c['skill_id'] for c in result['candidates']} == {'pdf', 'slides'}
        assert set(result) == {'query', 'candidates', 'search_control'}
        assert all(set(c) == {'skill_id', 'name', 'description', 'rank',
                              'minimal_evidence'} for c in result['candidates'])
        kwargs = {'purpose': 'Documents'} if action == 'CREATE' else {}
        if action == 'EXTEND':
            kwargs['target_bundle_id'] = 'mail-work'
        return apply(action, **kwargs)
    client = ScriptedClient([load(), check_search, FINAL])
    agent = ExperimentalSkillAgent(harness, client)
    assert agent.run('Document task') == FINAL['content']
    assert len(client.calls) == 3
    for index, call in enumerate(client.calls):
        assert len(call) == 2 + 2 * index
        if index:
            assert call[1:len(client.calls[index-1])] == client.calls[index-1][1:]
        assert 'BODY_' not in call[0]['content']
        assert ('PDF_BODY_ONLY_SELECTED' in json.dumps(call[1:])) == (index == 2)
        assert all(marker not in json.dumps(call) for marker in (
            'MAIL_BODY_ONLY_SELECTED', 'SLIDES_BODY_ONLY_SELECTED', 'HIDDEN_BODY_NEVER_VISIBLE'))
    last = client.calls[-1]
    assert last[-1]['tool_call_id'] == last[-2]['tool_calls'][0]['id']
    result = json.loads(last[-1]['content'])
    assert result['skill_bodies'] == [{'skill_id': 'pdf', 'body': 'PDF_BODY_ONLY_SELECTED'}]
    assert result['state']['direct_skills'] == (['pdf'] if action == 'DIRECT' else [])
    surface = json.loads(last[0]['content'].split('Runtime Bundles (metadata only)\n')[1])
    if action == 'DIRECT':
        assert len(surface['maintained_bundles']) == 1
    elif action == 'EXTEND':
        assert [m['skill_id'] for m in surface['maintained_bundles'][0]['members']] == ['mail', 'pdf']
    else:
        assert len(surface['maintained_bundles']) == 2
    assert agent.trace[0]['state_before'] == agent.trace[0]['state_after']
    retrieval_event = agent.trace[0]['tool_executions'][0]
    assert set(retrieval_event) >= {'model_visible_payload', 'internal_retrieval_record'}
    assert retrieval_event['internal_retrieval_record']['representations']
    assert agent.trace[2]['history_skill_body_ids'] == ['pdf']
    assert harness.pending_candidates is None


@pytest.mark.parametrize('decision', [
    apply(skill_ids=['invented']), apply(skill_ids=['hidden']),
    apply('EXTEND', target_bundle_id='missing'), apply('CREATE', purpose=''), apply(extra='invalid'),
])
def test_invalid_native_application_preserves_state_and_pairs_error_result(harness, decision):
    before = deepcopy(harness.state)
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load(), decision, FINAL]))
    assert agent.run('Document task') == FINAL['content']
    assert harness.state == before
    assert agent.history[-2]['tool_call_id'] == decision['tool_calls'][0]['id']
    assert 'error' in json.loads(agent.history[-2]['content'])
    assert 'BODY_' not in json.dumps(agent.history)


def test_merged_candidates_and_single_use(harness):
    first = harness.search_capability('PDF')
    second = harness.search_capability('email inbox')
    assert [c.skill_id for c in first.candidates] == ['pdf']
    assert [c.skill_id for c in second.candidates] == ['mail']
    assert {c.skill_id for c in harness.pending_candidates.candidates} == {'pdf', 'mail'}
    decision = {'action': 'DIRECT', 'skill_ids': ['pdf', 'mail'], 'reason': 'x',
                'coverage': [
                    {'need': 'PDF', 'covered_by': 'skill:pdf'},
                    {'need': 'mail', 'covered_by': 'skill:mail'}],
                'remaining_gaps': []}
    harness.apply_capability(json.dumps(decision))
    assert harness.state.direct_skills == {'pdf', 'mail'}
    assert harness.pending_candidates is None
    with pytest.raises(ValueError, match='fresh'):
        harness.apply_capability(json.dumps(decision))
    harness.search_capability('slides')
    with pytest.raises(ValueError, match='outside supplied'):
        harness.apply_capability(json.dumps(decision))


def test_multiple_loads_multi_direct_and_repeated_selection_across_runs(harness):
    client = ScriptedClient([load('email'), load(), apply(skill_ids=['pdf', 'pdf']), FINAL,
                            load(), apply('CREATE', purpose='Docs', skill_ids=['pdf', 'slides']), FINAL, FINAL])
    agent = ExperimentalSkillAgent(harness, client)
    agent.run('first task')
    previous = deepcopy(agent.history)
    agent.run('Maintain documents')
    assert client.calls[4][1:-1] == previous
    agent.run('Use earlier instructions')
    assert harness.state.direct_skills == {'pdf'}
    assert harness.loaded_skill_ids == ('pdf', 'slides')
    for marker in ('PDF_BODY_ONLY_SELECTED', 'SLIDES_BODY_ONLY_SELECTED'):
        assert json.dumps(agent.history).count(marker) == 1
        assert marker in json.dumps(client.calls[-1][1:])
    results = [json.loads(m['content']) for m in agent.history if m['role'] == 'tool']
    assert results[-1]['selected_skill_ids'] == ['pdf', 'slides']
    assert results[-1]['skill_bodies'] == [{'skill_id': 'slides', 'body': 'SLIDES_BODY_ONLY_SELECTED'}]
    assert all('BODY_' not in c[0]['content'] for c in client.calls)
    assert len(agent.trace) == 1


@pytest.mark.parametrize('content', ['hello', '{not JSON}', 'text with "quotes"\nand newlines'])
def test_plain_assistant_content_is_final(harness, content):
    client = ScriptedClient([{'role': 'assistant', 'content': content}])
    agent = ExperimentalSkillAgent(harness, client)
    assert agent.run('say hello') == content
    assert agent.history[-1]['content'] == content
    assert not harness.loaded_skill_ids
    assert 'BODY_' not in client.calls[0][0]['content']


@pytest.mark.parametrize('message', [
    apply(), tool('unknown', {}), load(''), load(3),
    {'role': 'assistant', 'content': None},
    {'role': 'assistant', 'tool_calls': [{'id': 'x', 'type': 'function', 'function': {}}]},
])
def test_protocol_errors_traced_without_mutating_state(harness, message):
    agent = ExperimentalSkillAgent(harness, ScriptedClient([message, FINAL]))
    if not message.get('tool_calls'):
        with pytest.raises(ValueError):
            agent.run('task')
        return
    if not isinstance(message['tool_calls'][0].get('function', {}).get('arguments'), str):
        with pytest.raises(ValueError):
            agent.run('task')
        return
    assert agent.run('task') == FINAL['content']
    assert not harness.loaded_skill_ids
    assert agent.trace[0]['tool_error']['type'] == 'ValueError'


def test_duplicate_argument_keys_rejected(harness):
    message = load()
    message['tool_calls'][0]['function']['arguments'] = '{"need":"PDF","need":"slides"}'
    agent = ExperimentalSkillAgent(harness, ScriptedClient([message, FINAL]))
    assert agent.run('task') == FINAL['content']
    assert not harness.loaded_skill_ids
    assert agent.history[-2]['tool_call_id'] == message['tool_calls'][0]['id']


def test_step_limit_and_provider_failure(harness):
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load()]), max_steps=1)
    with pytest.raises(AgentStepLimitError):
        agent.run('task')
    assert agent.history[-1]['role'] == 'tool'
    def fail(messages):
        raise RuntimeError('provider unavailable')
    agent = ExperimentalSkillAgent(harness, ScriptedClient([fail]))
    with pytest.raises(RuntimeError):
        agent.run('task')
    assert agent.trace[0]['state_before'] == agent.trace[0]['state_after']


def test_extend_requires_new_member_and_changes_only_target(harness):
    harness.state = RuntimeCapabilityState([
        ActiveBundle('a', 'Docs', ('pdf',)), ActiveBundle('b', 'Mail', ('mail',))])
    agent = ExperimentalSkillAgent(harness, ScriptedClient([
        load(), apply('EXTEND', target_bundle_id='a'), FINAL]))
    assert agent.run('task') == FINAL['content']
    assert agent.trace[1]['tool_error']['type'] == 'ValueError'
    agent = ExperimentalSkillAgent(harness, ScriptedClient([
        load(), apply('EXTEND', target_bundle_id='a', skill_ids=['slides']), FINAL]))
    agent.run('Make slides')
    assert harness.state.active_bundles[0].skill_ids == ('pdf', 'slides')
    assert harness.state.active_bundles[1] == ActiveBundle('b', 'Mail', ('mail',))


def test_no_independent_resolver_or_combined_loader(harness, monkeypatch):
    from skill_control_plane.runtime import capability_loading
    def forbidden(*args, **kwargs):
        pytest.fail('independent resolver/combined loader must not run')
    monkeypatch.setattr(capability_loading.LLMCapabilityResolver, '__init__', forbidden)
    monkeypatch.setattr(capability_loading.RuntimeCapabilityLoader, 'load_capability', forbidden)
    monkeypatch.setattr(harness, 'load_capability', forbidden)
    ExperimentalSkillAgent(harness, ScriptedClient([load(), apply(), FINAL])).run('task')


def test_failed_new_search_preserves_previous_candidates(harness, monkeypatch):
    previous = harness.search_capability('PDF')
    def fail(*args, **kwargs):
        raise RuntimeError('dense failure')
    monkeypatch.setattr(harness.discovery, 'discover_skills', fail)
    with pytest.raises(RuntimeError):
        harness.search_capability('email')
    assert harness.pending_candidates is previous
    harness.apply_capability(json.dumps({
        'action': 'DIRECT', 'skill_ids': ['pdf'], 'reason': 'x',
        'coverage': [{'need': 'PDF', 'covered_by': 'skill:pdf'}],
        'remaining_gaps': []}))


def test_empty_search_cannot_apply(harness):
    assert not harness.search_capability('zzzznonexistent').candidates
    with pytest.raises(ValueError, match='outside supplied'):
        harness.apply_capability(json.dumps({
            'action': 'DIRECT', 'skill_ids': ['pdf'], 'reason': 'x',
            'coverage': [{'need': 'PDF', 'covered_by': 'skill:pdf'}],
            'remaining_gaps': []}))


def test_bundle_surface_uses_member_metadata_only(harness):
    from dataclasses import replace
    harness.discovery.records['pdf'] = replace(harness.discovery.records['pdf'],
        description='  extract\n PDF text  ' + 'detail ' * 100,
        retrieval_representation='SECRET_CARD')
    harness.state = RuntimeCapabilityState([
        ActiveBundle('z', 'Documents', ('slides', 'pdf')), ActiveBundle('a', 'Mail', ('mail',))], {'hidden'})
    surface = harness.render_bundle_context()
    data = json.loads(surface)
    assert [b['bundle_id'] for b in data['maintained_bundles']] == ['a', 'z']
    member = data['maintained_bundles'][1]['members'][0]
    assert member['name'] == 'PDF'
    assert set(member) == {'skill_id', 'name', 'body_state'}
    assert 'short_description' not in surface
    assert all(s not in surface for s in ['SECRET_CARD', 'BODY_', 'hidden'])


def test_load_capability_model_surface_hides_full_retrieval_record(harness):
    from dataclasses import replace
    harness.discovery.records['pdf'] = replace(
        harness.discovery.records['pdf'], retrieval_representation='SECRET_FULL_CARD')
    harness.discovery.texts['pdf'] = 'SECRET_FULL_CARD'
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load('PDF'), FINAL]))
    agent.run('task')
    tool_payload = json.loads(agent.history[-2]['content'])
    rendered = json.dumps(tool_payload)
    assert 'SECRET_FULL_CARD' not in rendered
    assert all(set(candidate) == {'skill_id', 'name', 'description', 'rank',
                                  'minimal_evidence'}
               for candidate in tool_payload['candidates'])
    event = agent.trace[0]['tool_executions'][0]
    assert event['model_visible_payload'] == tool_payload
    assert 'SECRET_FULL_CARD' in json.dumps(event['internal_retrieval_record'])


def test_full_assistant_preserved_multiple_calls_and_paired_batch_failure(harness):
    message = load('PDF')
    message.update(content='Let me look.', reasoning_content='Preserve provider reasoning')
    message['tool_calls'].append(load('slides')['tool_calls'][0])
    client = ScriptedClient([message, FINAL])
    agent = ExperimentalSkillAgent(harness, client)
    agent.run('task')
    assert client.calls[1][2] == message
    assert [m['tool_call_id'] for m in client.calls[1][3:]] == [c['id'] for c in message['tool_calls']]
    bad = tool('unknown', {})
    bad['tool_calls'].append(load()['tool_calls'][0])
    agent = ExperimentalSkillAgent(harness, ScriptedClient([bad, FINAL, FINAL]))
    assert agent.run('task') == FINAL['content']
    tool_results = [m for m in agent.history if m['role'] == 'tool']
    assert [m['tool_call_id'] for m in tool_results[-2:]] == [c['id'] for c in bad['tool_calls']]
    assert json.loads(tool_results[-1]['content'])['error']['type'] == 'NotExecuted'
    assert agent.run('next task') == FINAL['content']


def test_duplicate_call_ids_rejected_before_history_append(harness):
    message = load()
    message['tool_calls'] *= 2
    agent = ExperimentalSkillAgent(harness, ScriptedClient([message]))
    with pytest.raises(ValueError, match='duplicate'):
        agent.run('task')
    assert len(agent.history) == 1


def test_native_schema_required_fields_and_action_enum():
    schema = CAPABILITY_TOOLS[1]['function']['parameters']
    assert set(schema['required']) == {
        'action', 'skill_ids', 'reason', 'coverage', 'remaining_gaps'}
    assert schema['properties']['action']['enum'] == ['DIRECT', 'EXTEND', 'CREATE']
    assert 'sufficiency' not in schema['properties']
    assert {'target_bundle_id', 'purpose'} <= schema['properties'].keys()


@pytest.mark.parametrize('bad', [apply(skill_ids=['invented']), apply('CREATE', purpose=''),
                               apply('EXTEND', target_bundle_id='missing')])
def test_multi_search_retry_and_search_local_history(harness, bad):
    before = deepcopy(harness.state)
    def retry(messages):
        assert harness.state == before
        assert {c.skill_id for c in harness.pending_candidates.candidates} == {'pdf', 'mail'}
        assert 'error' in json.loads(messages[-1]['content'])
        searches = [json.loads(m['content']) for m in messages if m['role'] == 'tool'][:2]
        assert [[c['skill_id'] for c in r['candidates']] for r in searches] == [['pdf'], ['mail']]
        return apply(skill_ids=['pdf', 'mail'])
    agent = ExperimentalSkillAgent(harness, ScriptedClient([
        load('PDF'), load('email inbox'), bad, retry, FINAL]))
    agent.run('two capabilities')
    assert harness.state.direct_skills == {'pdf', 'mail'}
    assert agent.trace[2]['pending_pool_before'] == agent.trace[2]['pending_pool_after'] == ['pdf', 'mail']
    assert agent.trace[3]['pending_pool_after'] == []


def test_final_discards_uncommitted_pool(harness):
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load('PDF'), FINAL, apply(), FINAL]))
    agent.run('search only')
    assert agent.trace[-1]['pending_pool_before'] == ['pdf']
    assert agent.trace[-1]['pending_pool_after'] == []
    assert harness.pending_candidates is None
    agent.run('try stale candidate')
    assert agent.trace[0]['tool_error']['type'] == 'ValueError'
    assert not harness.loaded_skill_ids


def test_new_turn_discards_pool_left_by_interrupted_turn(harness):
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load('PDF'), apply()]), max_steps=1)
    with pytest.raises(AgentStepLimitError):
        agent.run('interrupted')
    assert harness.pending_candidates is not None
    with pytest.raises(AgentStepLimitError):
        agent.run('new turn')
    assert agent.trace[0]['pending_pool_before'] == []
    assert agent.trace[0]['tool_error']['type'] == 'ValueError'
    assert not harness.loaded_skill_ids


def test_pool_merge_deduplicates_and_preserves_representations(harness):
    harness.search_capability('PDF')
    harness.search_capability('PDF presentation slides')
    pending = harness.pending_candidates
    assert [c.skill_id for c in pending.candidates].count('pdf') == 1
    assert set(dict(pending.representations)) == {c.skill_id for c in pending.candidates}


def test_second_search_provider_failure_can_recover_in_same_agent(harness, monkeypatch):
    discover = harness.discovery.discover_skills
    def search(need, **kwargs):
        if need == 'email':
            raise RuntimeError('dense failure')
        return discover(need, **kwargs)
    monkeypatch.setattr(harness.discovery, 'discover_skills', search)
    agent = ExperimentalSkillAgent(harness, ScriptedClient([
        load('PDF'), load('email'), apply(), FINAL]))
    agent.run('task')
    assert agent.trace[1]['pending_pool_before'] == agent.trace[1]['pending_pool_after'] == ['pdf']
    assert agent.trace[1]['state_before'] == agent.trace[1]['state_after']
    assert agent.trace[1]['tool_error']['type'] == 'RuntimeError'
    assert harness.state.direct_skills == {'pdf'}


def test_missing_reason_preserves_pool_for_retry(harness):
    missing = tool('apply_capability', {
        'action': 'DIRECT', 'skill_ids': ['pdf'],
        'coverage': [{'need': 'PDF', 'covered_by': 'skill:pdf'}],
        'remaining_gaps': []})
    agent = ExperimentalSkillAgent(harness, ScriptedClient([load('PDF'), missing, apply(), FINAL]))
    agent.run('task')
    assert agent.trace[1]['tool_error']['type'] == 'ValueError'
    assert agent.trace[1]['state_before'] == agent.trace[1]['state_after']
    assert agent.trace[1]['pending_pool_before'] == agent.trace[1]['pending_pool_after'] == ['pdf']
    assert harness.state.direct_skills == {'pdf'}


@pytest.mark.parametrize('queries,ids,expected', [
    (['PDF', 'email'], ['pdf', 'mail'], True),
    (['PDF', 'PDF'], ['pdf'], False),
    (['PDF', 'email'], ['mail'], False),
])
def test_live_multi_search_audit_requires_joint_exclusive_selection(harness, queries, ids, expected):
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    spec = spec_from_file_location('live_agent_e2e', Path(__file__).parents[1] /
                                   'scripts/experimental_skill_agent_e2e.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = ExperimentalSkillAgent(harness, ScriptedClient([
        *[load(q) for q in queries], apply(skill_ids=ids), FINAL]))
    agent.run('task')
    assert module.audit_multi_search(agent)['passed'] is expected
