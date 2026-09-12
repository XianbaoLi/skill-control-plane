import json
from copy import deepcopy
from uuid import uuid4

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime import SkillControlPlane
from skill_control_plane.runtime.capability_memory import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.integrations.reference_agent import (
    ExperimentalSkillAgent, dto_payload, parse_tool_arguments,
)
from tests.support import full_discovery


def harness(state=None):
    records = [
        SkillRecord('pdf', 'PDF', 'read PDF documents', 'PDF BODY', ''),
        SkillRecord('ocr', 'OCR', 'extract text from scanned documents', 'OCR BODY', ''),
        SkillRecord('powerpoint', 'PowerPoint', 'create presentation slides',
                    'PPT BODY', ''),
    ]
    store = SkillRegistry(records)
    return SkillControlPlane(store, discovery=full_discovery(store), state=state)


def native(name, arguments):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': uuid4().hex, 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)},
    }]}


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete_messages(self, messages, *, tools):
        return next(self.responses)


def decision(skill_ids, *, remaining_gaps=(), bundle_coverage=()):
    return {
        'action': 'DIRECT', 'skill_ids': list(skill_ids), 'reason': 'credible match',
        'coverage': [
            *({'need': need, 'covered_by': f'bundle:{bundle_id}'}
              for need, bundle_id in bundle_coverage),
            *({'need': f'use {skill_id}', 'covered_by': f'skill:{skill_id}'}
              for skill_id in skill_ids),
        ],
        'remaining_gaps': list(remaining_gaps),
    }


def parsed(data):
    return parse_tool_arguments('apply_capability', json.dumps(data))


def test_coverage_references_are_authorized_and_selected():
    runtime = harness(RuntimeCapabilityState([
        ActiveBundle('documents', 'Read documents', ('pdf',))]))
    runtime.search_capability('OCR')
    accepted = decision(
        ('ocr',), bundle_coverage=(('read PDF', 'documents'),))
    result = runtime.apply_capability(parsed(accepted))
    assert dto_payload(result)['coverage'] == accepted['coverage']
    for covered_by in ('bundle:missing', 'skill:invented'):
        runtime.begin_turn()
        runtime.search_capability('OCR')
        invalid = decision(('ocr',))
        invalid['coverage'][0]['covered_by'] = covered_by
        with pytest.raises(ValueError, match='coverage references'):
            runtime.apply_capability(parsed(invalid))
    runtime.begin_turn()
    runtime.search_capability('OCR presentation')
    unselected = decision(('ocr',))
    unselected['coverage'].append({
        'need': 'presentation', 'covered_by': 'skill:powerpoint'})
    with pytest.raises(ValueError, match='must be selected'):
        runtime.apply_capability(parsed(unselected))


def test_partial_commit_preserves_gap_without_model_supplied_outcome():
    runtime = harness()
    runtime.search_capability('OCR')
    partial = decision(('ocr',), remaining_gaps=('control a quantum teleporter',))
    result = runtime.apply_capability(parsed(partial))
    assert set(runtime.context_snapshot().direct_skill_ids) == {'ocr'}
    payload = dto_payload(result)
    assert 'sufficiency' not in payload
    assert payload['remaining_gaps'] == ['control a quantum teleporter']
    assert payload['selected_skill_ids'] == ['ocr']


def test_apply_rejects_model_supplied_sufficiency_as_an_extra_field():
    runtime = harness()
    runtime.search_capability('OCR')
    invalid = decision(('ocr',))
    invalid['sufficiency'] = 'COVERED'
    with pytest.raises(ValueError, match='invalid action or action-specific fields'):
        runtime.apply_capability(parsed(invalid))


def test_search_budget_caps_retrieval_and_preserves_pending_pool():
    runtime = harness()
    for query in ('OCR', 'presentation', 'documents'):
        runtime.search_capability(query)
    before_pool = deepcopy(runtime.context_snapshot().pending_candidate_skill_ids)
    before_retrieval = runtime.turn_audit().retrieval_call_count
    with pytest.raises(ValueError, match='budget exhausted'):
        runtime.search_capability('another gap')
    audit = runtime.turn_audit()
    assert audit.session.search_count == 3
    assert audit.session.search_attempt_count == 4
    assert audit.session.search_budget_hits == 1
    assert audit.retrieval_call_count == before_retrieval
    assert runtime.context_snapshot().pending_candidate_skill_ids == before_pool


def test_no_progress_result_tells_model_to_stop_repeating():
    runtime = harness()
    runtime.search_capability('OCR presentation documents')
    second = runtime.search_capability('OCR presentation documents')
    control = second.search_control
    assert not control.meaningful_expansion
    assert control.repeated_query
    assert control.new_candidate_skill_ids == ()
    assert control.message.startswith('No meaningful new candidates')
    assert runtime.turn_audit().session.repeated_search_count == 1


def test_second_empty_search_counts_as_repeated_no_progress():
    runtime = harness()
    runtime.search_capability('cryogenic qubit calibration')
    runtime.search_capability('superconducting microwave pulse tuning')
    audit = runtime.turn_audit().session
    assert audit.no_progress_search_count == 2
    assert audit.repeated_search_count == 1


def test_agent_turn_audit_reports_partial_and_resets_budget_next_turn():
    runtime = harness()
    agent = ExperimentalSkillAgent(runtime, Client([
        native('load_capability', {'need': 'OCR'}),
        native('apply_capability', decision(
            ('ocr',), remaining_gaps=('quantum teleporter',))),
        {'role': 'assistant', 'content': 'OCR covered; teleporter unresolved.'},
        {'role': 'assistant', 'content': 'No new capability gap.'},
    ]))
    agent.run('OCR plus quantum teleporter')
    assert agent.turn_audit['search_count'] == 1
    assert agent.turn_audit['capability_sufficiency_outcome'] == 'UNSATISFIED'
    assert agent.turn_audit['sufficiency_transitions'] == [
        'SEARCH_MORE', 'UNSATISFIED']
    assert agent.turn_audit['unresolved_gaps'] == ['quantum teleporter']
    assert agent.turn_audit['apply_count'] == 1
    agent.run('continue with current capability')
    assert agent.turn_audit['search_count'] == 0
    assert agent.turn_audit['search_attempt_count'] == 0
    assert agent.turn_audit['capability_sufficiency_outcome'] == 'COVERED'


def test_missing_skill_stops_without_apply_and_is_inferred_unsatisfied():
    runtime = harness()
    agent = ExperimentalSkillAgent(runtime, Client([
        native('load_capability', {'need': 'quantum teleporter control'}),
        {'role': 'assistant', 'content': 'No credible Skill; gap unresolved.'},
    ]))
    agent.run('control a quantum teleporter')
    assert agent.turn_audit['search_count'] == 1
    assert agent.turn_audit['apply_count'] == 0
    assert agent.turn_audit['capability_sufficiency_outcome'] == 'UNSATISFIED'
    assert agent.turn_audit['sufficiency_transitions'] == ['SEARCH_MORE']
    assert agent.turn_audit['unresolved_gaps'] == ['quantum teleporter control']


def test_multi_gap_search_more_is_inferred_before_first_apply():
    runtime = harness()
    agent = ExperimentalSkillAgent(runtime, Client([
        native('load_capability', {'need': 'OCR'}),
        native('load_capability', {'need': 'presentation slides'}),
        native('apply_capability', decision(('ocr', 'powerpoint'))),
        {'role': 'assistant', 'content': 'Both capabilities committed.'},
    ]))
    agent.run('OCR then make slides')
    assert agent.turn_audit['apply_count'] == 1
    assert agent.turn_audit['sufficiency_transitions'] == [
        'SEARCH_MORE', 'SEARCH_MORE', 'COVERED']
    assert agent.turn_audit['capability_sufficiency_outcome'] == 'COVERED'


def test_final_outcome_uses_last_successful_discovery_behavior():
    runtime = harness()
    agent = ExperimentalSkillAgent(runtime, Client([
        native('load_capability', {'need': 'OCR'}),
        native('apply_capability', decision(('ocr',))),
        native('load_capability', {'need': 'quantum teleporter control'}),
        {'role': 'assistant', 'content': 'The new residual gap is unresolved.'},
    ]))
    agent.run('OCR, then control a quantum teleporter')
    assert agent.turn_audit['apply_count'] == 1
    assert agent.turn_audit['sufficiency_transitions'] == [
        'SEARCH_MORE', 'COVERED', 'SEARCH_MORE']
    assert agent.turn_audit['capability_sufficiency_outcome'] == 'UNSATISFIED'
    assert agent.turn_audit['unresolved_gaps'] == ['quantum teleporter control']
