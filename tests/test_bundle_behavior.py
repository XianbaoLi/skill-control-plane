"""Scripted decisions test experiment plumbing, not live model judgment."""
import json
from copy import deepcopy
from uuid import uuid4

import pytest

from skill_control_plane.evals.bundle_behavior import TASKS, evaluate_turns, run_experiment
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery
from tests.support import full_discovery


@pytest.fixture
def discovery():
    store = SkillRegistry([
        SkillRecord(s, s, desc, s.upper() + '_PRIVATE_BODY', '') for s, desc in [
            ('pdf', 'scanned PDF OCR document production'),
            ('ocr-and-documents', 'scanned PDF OCR document production'),
            ('powerpoint', 'presentation editable slides document production'),
            ('xlsx', 'spreadsheet Excel workbook formula'),
            ('systematic-debugging', 'GitHub CI failure debugging reproduction'),
            ('hidden', 'unrelated astronomy'),
        ]
    ])
    return full_discovery(store)


def native(name, arguments):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': uuid4().hex, 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments)}}]}


class ScriptedConversation:
    def __init__(self, unrelated='DIRECT', redundant_load=False, related='EXTEND'):
        self.calls = []
        self.searches = 0
        self.unrelated = unrelated
        self.related = related
        self.redundant_load = redundant_load
        self.did_redundant_load = False

    def complete_messages(self, messages, *, tools):
        self.calls.append(deepcopy(messages))
        last = messages[-1]['content']
        surface = json.loads(messages[0]['content'].split('Runtime Bundles (metadata only)\n')[1])
        doc = next(iter(surface['maintained_bundles']), None)
        if last in TASKS:
            index = TASKS.index(last)
            if index == 1 and not self.redundant_load:
                assert any(m['skill_id'] == 'powerpoint' for m in doc['members'])
                assert 'POWERPOINT_PRIVATE_BODY' in json.dumps(messages[1:])
                return {'role': 'assistant', 'content': 'Use the existing presentation instructions.'}
            query = ['scanned PDF OCR document production presentation', 'presentation',
                     'spreadsheet Excel workbook formula', 'GitHub CI failure debugging'][index]
            self.searches += 1
            return native('load_capability', {'need': query})
        result = json.loads(last)
        if 'selected_skill_ids' in result:
            return {'role': 'assistant', 'content': 'Workflow ready.'}
        if self.redundant_load and result['query'] == 'presentation':
            return {'role': 'assistant', 'content': 'Redundant search completed.'}
        if doc is None:
            action = {'action': 'CREATE', 'purpose': 'Document production',
                      'skill_ids': ['pdf', 'ocr-and-documents', 'powerpoint']}
        elif result['query'].startswith('spreadsheet'):
            action = {'action': self.related, 'skill_ids': ['xlsx']}
            action.update({'target_bundle_id': doc['bundle_id']} if self.related == 'EXTEND'
                          else {'purpose': 'Separate spreadsheet workflow'})
        else:
            action = {'action': self.unrelated, 'skill_ids': ['systematic-debugging']}
            if self.unrelated == 'CREATE':
                action['purpose'] = 'Software diagnosis'
            elif self.unrelated == 'EXTEND':
                action['target_bundle_id'] = doc['bundle_id']
        coverage = [
            {'need': f'use {skill_id}', 'covered_by': f'skill:{skill_id}'}
            for skill_id in action['skill_ids']
        ]
        return native('apply_capability', {
            'reason': 'Scripted test decision', 'coverage': coverage,
            'remaining_gaps': [], **action})


def test_four_turns_preserve_history_extend_related_and_create_unrelated(discovery):
    client = ScriptedConversation('CREATE')
    checkpoints = []
    report = run_experiment(discovery, client, checkpoint=lambda r: checkpoints.append(deepcopy(r)))
    assert report['status'] == 'completed'
    assert report['matches_all_expectations']
    assert len(client.calls) == 10
    assert client.searches == 3
    assert [t['load_triggered'] for t in report['turns']] == [True, False, True, True]
    assert [len(t['model_actions']) for t in report['turns']] == [2, 0, 2, 2]
    assert report['turns'][1]['state_before'] == report['turns'][1]['state_after']
    initial_document = report['turns'][0]['state_after']['active_bundles'][0]
    extended_document = report['turns'][2]['state_after']['active_bundles'][0]
    assert extended_document['bundle_id'] == initial_document['bundle_id']
    assert set(extended_document['skill_ids']) == set(initial_document['skill_ids']) | {'xlsx'}
    final_document = next(bundle for bundle in
        report['turns'][3]['state_after']['active_bundles']
        if bundle['bundle_id'] == extended_document['bundle_id'])
    assert final_document == extended_document
    for previous, current in zip(report['turns'], report['turns'][1:]):
        assert current['state_before'] == previous['state_after']
        assert current['history_start'] == previous['history_end']
    for previous, current in zip(client.calls, client.calls[1:]):
        assert current[1:len(previous)] == previous[1:]
    assert [c['turn'] for c in report['model_calls']] == [1, 1, 1, 2, 3, 3, 3, 4, 4, 4]
    for task in TASKS:
        assert any(m['content'] == task for m in client.calls[-1])
    assert all('PRIVATE_BODY' not in call[0]['content'] for call in client.calls)
    assert json.dumps(report['history']).count('POWERPOINT_PRIVATE_BODY') == 1
    assert 'HIDDEN_PRIVATE_BODY' not in json.dumps(report)
    turn_three_surface = report['turns'][2]['system_bundle_metadata'][-1]
    assert 'xlsx' in {m['skill_id'] for m in turn_three_surface[0]['members']}
    payload = report['turns'][2]['retrieval_results'][0]
    assert set(payload) == {'query', 'candidates', 'search_control'}
    assert all(set(c) == {'skill_id', 'name', 'description', 'rank', 'minimal_evidence'}
               for c in payload['candidates'])
    assert checkpoints[-1]['status'] == 'completed'
    assert checkpoints[-1]['turns'][0]['state_after'] == report['turns'][0]['state_after']


def test_behavior_evaluator_detects_unrelated_extend_without_changing_runtime_policy(discovery):
    report = run_experiment(discovery, ScriptedConversation(unrelated='EXTEND'))
    assert report['status'] == 'completed'  # Schema-valid but semantically inappropriate.
    assert not report['matches_all_expectations']
    checks = report['evaluations'][3]['checks']
    assert not checks['no_document_extend']
    assert not checks['document_bundle_unchanged']
    assert 'systematic-debugging' in report['turns'][3]['state_after']['active_bundles'][0]['skill_ids']


def test_behavior_evaluator_detects_redundant_search(discovery):
    report = run_experiment(discovery, ScriptedConversation(redundant_load=True))
    assert not report['evaluations'][1]['checks']['no_retrieval']
    assert not report['matches_all_expectations']


def test_behavior_evaluator_detects_related_create_instead_of_extend(discovery):
    report = run_experiment(discovery, ScriptedConversation(related='CREATE'))
    checks = report['evaluations'][2]['checks']
    assert not checks['xlsx_extended_into_document']
    assert not checks['no_replacement_bundle']


def test_failure_keeps_partial_trace_and_does_not_restart_agent(discovery):
    class Failure:
        def complete_messages(self, messages, *, tools):
            raise RuntimeError('provider failure')
    report = run_experiment(discovery, Failure())
    assert report['status'] == 'error'
    assert len(report['turns']) == 1
    assert report['turns'][0]['state_before'] == report['turns'][0]['state_after']
    assert report['turns'][0]['trace'][0]['error']['type'] == 'RuntimeError'
    assert not report['matches_all_expectations']


def test_protocol_error_keeps_same_conversation_for_following_tasks(discovery):
    class InvalidFirstFinal(ScriptedConversation):
        def complete_messages(self, messages, *, tools):
            if len(self.calls) == 3:
                return {'role': 'assistant', 'content': 'Continue after the rejected tool call.'}
            response = super().complete_messages(messages, tools=tools)
            if len(self.calls) == 3:
                bad = native('load_capability', {})
                bad['tool_calls'][0]['function']['arguments'] = '{bad JSON}'
                return bad
            return response
    client = InvalidFirstFinal()
    report = run_experiment(discovery, client)
    assert report['status'] == 'completed'
    assert len(report['turns']) == 4
    assert report['turns'][0]['trace'][2]['tool_error']['type'] == 'ValueError'
    assert report['evaluations'][0]['checks']['turn_completed']
    assert report['turns'][1]['status'] == 'completed'
    assert report['turns'][1]['state_before'] == report['turns'][0]['state_after']
    assert any(message['role'] == 'tool' and json.loads(message['content']).get('error', {}).get('type') == 'ValueError'
               for message in report['history'])
