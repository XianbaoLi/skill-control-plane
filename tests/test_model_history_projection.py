import json
from copy import deepcopy

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.capability_loading import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.runtime.experimental_agent import (
    EVICTED_BODY_TOMBSTONE, SUPERSEDED_BODY_TOMBSTONE,
    ExperimentalSkillAgent,
)


POWERPOINT_BODY = 'FULL POWERPOINT SKILL BODY'
PDF_BODY = 'FULL PDF SKILL BODY'


class NoCallClient:
    def complete_messages(self, messages, *, tools):
        return {'role': 'assistant', 'content': 'done'}


def make_agent(*, powerpoint='resident', pdf='resident'):
    registry = SkillRegistry([
        SkillRecord('powerpoint', 'PowerPoint', 'slides', POWERPOINT_BODY, '/ppt/SKILL.md'),
        SkillRecord('pdf', 'PDF', 'documents', PDF_BODY, '/pdf/SKILL.md'),
    ])
    state = RuntimeCapabilityState([
        ActiveBundle('documents', 'Document work', ('powerpoint', 'pdf')),
    ], skill_body_states={'powerpoint': powerpoint, 'pdf': pdf})
    return ExperimentalSkillAgent(
        RuntimeCapabilityHarness(discovery=SkillDiscovery(registry), state=state),
        NoCallClient(),
    )


def paired_apply_history():
    assistant = {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'apply-1', 'type': 'function', 'function': {
            'name': 'apply_capability', 'arguments': '{}'},
    }]}
    result = {'action': 'CREATE', 'selected_skill_ids': ['powerpoint', 'pdf'],
              'skill_bodies': [
                  {'skill_id': 'powerpoint', 'body': POWERPOINT_BODY},
                  {'skill_id': 'pdf', 'body': PDF_BODY},
              ]}
    tool = {'role': 'tool', 'tool_call_id': 'apply-1',
            'content': json.dumps(result)}
    return [assistant, tool]


def paired_reload_history(call_id='reload-1'):
    return [
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': call_id, 'type': 'function', 'function': {
                'name': 'load_skill_body',
                'arguments': '{"skill_id":"powerpoint"}'},
        }]},
        {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps({
            'status': 'loaded', 'skill_id': 'powerpoint',
            'bundle_ids': ['documents'], 'body': POWERPOINT_BODY,
        })},
    ]


def tool_payload(history, call_id):
    return json.loads(next(message['content'] for message in history
                           if message.get('tool_call_id') == call_id))


def assert_pairing_unchanged(canonical, projected):
    canonical_assistants = [message for message in canonical if message.get('tool_calls')]
    projected_assistants = [message for message in projected if message.get('tool_calls')]
    assert projected_assistants == canonical_assistants
    expected_ids = [call['id'] for message in projected_assistants
                    for call in message['tool_calls']]
    actual_ids = [message['tool_call_id'] for message in projected
                  if message['role'] == 'tool']
    assert actual_ids == expected_ids


def test_resident_body_remains_visible_without_mutating_canonical_history():
    agent = make_agent()
    agent.history = paired_apply_history()
    canonical = deepcopy(agent.history)
    projected = agent.build_model_history()
    assert POWERPOINT_BODY in json.dumps(projected)
    assert PDF_BODY in json.dumps(projected)
    assert agent.history == canonical
    assert agent._last_history_projection == {
        'canonical_body_ids': ['pdf', 'powerpoint'],
        'model_visible_body_ids': ['pdf', 'powerpoint'],
        'redacted_body_ids': [],
    }


def test_one_evicted_body_is_redacted_without_erasing_apply_siblings():
    agent = make_agent(powerpoint='evicted')
    agent.history = paired_apply_history()
    canonical = deepcopy(agent.history)
    projected = agent.build_model_history()
    payload = tool_payload(projected, 'apply-1')
    powerpoint, pdf = payload['skill_bodies']
    assert powerpoint == {
        'skill_id': 'powerpoint', 'body_state': 'evicted',
        'body': EVICTED_BODY_TOMBSTONE,
    }
    assert pdf == {'skill_id': 'pdf', 'body': PDF_BODY}
    assert POWERPOINT_BODY not in json.dumps(projected)
    assert POWERPOINT_BODY in json.dumps(agent.history)
    assert agent.history == canonical
    assert agent._last_history_projection == {
        'canonical_body_ids': ['pdf', 'powerpoint'],
        'model_visible_body_ids': ['pdf'],
        'redacted_body_ids': ['powerpoint'],
    }


def test_exact_reload_becomes_visible_after_residency_transition():
    agent = make_agent(powerpoint='evicted')
    agent.history = paired_apply_history()
    assert POWERPOINT_BODY not in json.dumps(agent.build_model_history())
    result = agent.harness.load_skill_body('powerpoint')
    agent.history.extend(paired_reload_history())
    projected = agent.build_model_history()
    assert result['body'] == POWERPOINT_BODY
    assert tool_payload(projected, 'apply-1')['skill_bodies'][0] == {
        'skill_id': 'powerpoint', 'body_state': 'superseded',
        'body': SUPERSEDED_BODY_TOMBSTONE,
    }
    assert tool_payload(projected, 'reload-1')['body'] == POWERPOINT_BODY
    assert agent.harness.state.skill_body_states['powerpoint'] == 'resident'
    assert agent._last_history_projection['model_visible_body_ids'] == [
        'pdf', 'powerpoint']
    assert agent._last_history_projection['redacted_body_ids'] == ['powerpoint']


def test_three_occurrences_leave_only_latest_reload_visible_and_keep_pairing():
    agent = make_agent()
    agent.history = paired_apply_history()
    agent.harness.mark_skill_body_evicted('powerpoint')
    first = agent.harness.load_skill_body('powerpoint')
    assert first['status'] == 'loaded'
    agent.history.extend(paired_reload_history('reload-1'))
    agent.harness.mark_skill_body_evicted('powerpoint')
    second = agent.harness.load_skill_body('powerpoint')
    assert second['status'] == 'loaded'
    agent.history.extend(paired_reload_history('reload-2'))

    canonical = deepcopy(agent.history)
    projected = agent.build_model_history()
    assert json.dumps(agent.history).count(POWERPOINT_BODY) == 3
    assert json.dumps(projected).count(POWERPOINT_BODY) == 1
    assert json.dumps(projected).count(SUPERSEDED_BODY_TOMBSTONE) == 2
    assert tool_payload(projected, 'apply-1')['skill_bodies'][0]['body'] \
        == SUPERSEDED_BODY_TOMBSTONE
    assert tool_payload(projected, 'reload-1')['body'] == SUPERSEDED_BODY_TOMBSTONE
    assert tool_payload(projected, 'reload-2')['body'] == POWERPOINT_BODY
    assert agent.history == canonical
    assert_pairing_unchanged(canonical, projected)


def test_multi_skill_reloads_keep_each_skills_latest_body_visible_once():
    agent = make_agent()
    agent.history = paired_apply_history()
    for call_id in ('reload-1', 'reload-2'):
        agent.harness.mark_skill_body_evicted('powerpoint')
        assert agent.harness.load_skill_body('powerpoint')['status'] == 'loaded'
        agent.history.extend(paired_reload_history(call_id))

    canonical = deepcopy(agent.history)
    projected = agent.build_model_history()
    rendered = json.dumps(projected)
    assert rendered.count(POWERPOINT_BODY) == 1
    assert rendered.count(PDF_BODY) == 1
    apply_bodies = tool_payload(projected, 'apply-1')['skill_bodies']
    assert apply_bodies[0]['body'] == SUPERSEDED_BODY_TOMBSTONE
    assert apply_bodies[1]['body'] == PDF_BODY
    assert tool_payload(projected, 'reload-1')['body'] == SUPERSEDED_BODY_TOMBSTONE
    assert tool_payload(projected, 'reload-2')['body'] == POWERPOINT_BODY
    assert json.dumps(agent.history).count(POWERPOINT_BODY) == 3
    assert json.dumps(agent.history).count(PDF_BODY) == 1
    assert agent.history == canonical
    assert_pairing_unchanged(canonical, projected)


def test_repeated_eviction_redacts_every_historical_body_version():
    agent = make_agent()
    agent.history = [*paired_apply_history(), *paired_reload_history()]
    agent.harness.mark_skill_body_evicted('powerpoint')
    canonical = deepcopy(agent.history)
    projected = agent.build_model_history()
    assert POWERPOINT_BODY not in json.dumps(projected)
    assert json.dumps(projected).count(EVICTED_BODY_TOMBSTONE) == 2
    assert tool_payload(projected, 'apply-1')['skill_bodies'][0]['body_state'] == 'evicted'
    assert tool_payload(projected, 'reload-1')['body_state'] == 'evicted'
    assert json.dumps(agent.history).count(POWERPOINT_BODY) == 2
    assert agent.history == canonical


def test_projection_preserves_native_tool_pairing_and_non_body_fields():
    agent = make_agent(powerpoint='evicted')
    agent.history = [
        {'role': 'user', 'content': 'task'},
        *paired_apply_history(),
        *paired_reload_history(),
    ]
    projected = agent.build_model_history()
    assert_pairing_unchanged(agent.history, projected)
    payload = tool_payload(projected, 'apply-1')
    assert payload['action'] == 'CREATE'
    assert payload['selected_skill_ids'] == ['powerpoint', 'pdf']


def test_projection_fails_clearly_on_malformed_canonical_tool_json():
    agent = make_agent()
    agent.history = [{'role': 'tool', 'tool_call_id': 'bad', 'content': '{bad'}]
    canonical = deepcopy(agent.history)
    with pytest.raises(ValueError, match='canonical tool result content is invalid JSON'):
        agent.build_model_history()
    assert agent.history == canonical


def test_eviction_projection_does_not_change_runtime_or_pending_pool():
    agent = make_agent()
    pending = agent.harness.search_capability('spreadsheet')
    retrieval_count = agent.harness.retrieval_call_count
    bundles = deepcopy(agent.harness.state.active_bundles)
    agent.history = paired_apply_history()
    agent.harness.mark_skill_body_evicted('powerpoint')
    agent.build_model_history()
    assert agent.harness.state.active_bundles == bundles
    assert agent.harness.pending_candidates is pending
    assert agent.harness.retrieval_call_count == retrieval_count
    assert agent.harness.body_load_count == 0
    assert agent.harness.discovery.registry.load_skill_body('powerpoint') == POWERPOINT_BODY
