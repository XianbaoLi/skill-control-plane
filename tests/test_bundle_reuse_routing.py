import json
from copy import deepcopy

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.cards import RetrievalCard
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.capability_loading import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.runtime.experimental_agent import AGENT_INSTRUCTIONS, ExperimentalSkillAgent


def card(skill_id, capabilities, use_when=()):
    return RetrievalCard(skill_id, 'hash', 'purpose', tuple(use_when),
                         tuple(capabilities), ('cue',))


def harness(body_state='resident'):
    records = [
        SkillRecord('powerpoint', 'PowerPoint', 'presentation metadata',
                    'PRIVATE_POWERPOINT_BODY' + ' body detail' * 500,
                    '/powerpoint/SKILL.md'),
        SkillRecord('xlsx', 'Excel', 'spreadsheet metadata',
                    'PRIVATE_XLSX_BODY', '/xlsx/SKILL.md'),
    ]
    cards = {'powerpoint': card('powerpoint', (
        'create presentation slides',
        'edit existing presentations',
        'EDIT EXISTING PRESENTATIONS',
        'x' * 180,
        'revise slide content and layout',
        'add and reorder slides',
        'update slide charts',
        'render presentations',
        'export presentations',
        'ninth bounded item',
    ), ('continue presentation work',))}
    registry = SkillRegistry(records, retrieval_cards=cards)
    state = RuntimeCapabilityState(
        [ActiveBundle('presentation-work', 'Create and revise presentations',
                      ('powerpoint',))],
        skill_body_states={'powerpoint': body_state},
    )
    return RuntimeCapabilityHarness(discovery=SkillDiscovery(registry), state=state)


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete_messages(self, messages, *, tools):
        return next(self.responses)


def native(name, arguments):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'call-1', 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)},
    }]}


def test_bundle_capabilities_are_structured_deduplicated_and_bounded():
    runtime = harness()
    surface = runtime.render_bundle_context()
    bundle = json.loads(surface)['maintained_bundles'][0]
    assert len(bundle['capabilities']) == 8
    assert bundle['capabilities'][:3] == [
        'create presentation slides',
        'edit existing presentations',
        'x' * 119 + '…',
    ]
    assert all(len(phrase) <= 120 for phrase in bundle['capabilities'])
    assert 'PRIVATE_POWERPOINT_BODY' not in surface
    assert 'cue' not in surface
    assert len(surface) < len(runtime.discovery.records['powerpoint'].body)


def test_registry_from_tree_retains_structured_card_accessor(tmp_path):
    skill_dir = tmp_path / 'powerpoint'
    skill_dir.mkdir()
    (skill_dir / 'SKILL.md').write_text(
        '---\nname: powerpoint\ndescription: slides\n---\nBODY')
    from skill_control_plane.registry.loader import load_skill_tree
    record = load_skill_tree(tmp_path)[0]
    cards = {'powerpoint': RetrievalCard(
        'powerpoint', record.content_hash, 'presentations',
        ('when revising slides',), ('edit existing presentations',), ('slides',))}
    registry = SkillRegistry.from_tree(tmp_path, cards=cards)
    assert registry.capability_phrases('powerpoint') == (
        'edit existing presentations', 'when revising slides')
    assert registry.load_skill_body('powerpoint') == 'BODY'


def test_bundle_first_policy_is_explicit_without_new_decision_protocol():
    assert 'Before calling load_capability' in AGENT_INSTRUCTIONS
    assert "Bundle's purpose" in AGENT_INSTRUCTIONS
    assert 'capabilities' in AGENT_INSTRUCTIONS
    assert 'residual capability gap' in AGENT_INSTRUCTIONS
    assert 'New user wording does not imply' in AGENT_INSTRUCTIONS
    assert 'ONLY the uncovered capability gap' in AGENT_INSTRUCTIONS
    assert 'Return JSON' not in AGENT_INSTRUCTIONS


def test_resident_reuse_turn_audit_has_zero_tool_and_retrieval_calls():
    runtime = harness()
    agent = ExperimentalSkillAgent(
        runtime, Client([{'role': 'assistant', 'content': 'Reused.'}]))
    before = runtime.retrieval_call_count
    assert agent.run('继续修改这个 presentation') == 'Reused.'
    audit = agent.turn_audit
    assert audit['bundle_state_before'][0]['capabilities']
    assert audit['bundle_member_body_state_before'] == {'powerpoint': 'resident'}
    assert audit['tool_calls'] == []
    assert audit['retrieval_call_count_before'] == before
    assert audit['retrieval_call_count_after'] == before
    assert audit['body_load_count_before'] == audit['body_load_count_after'] == 0
    assert not audit['load_capability_called']
    assert not audit['load_skill_body_called']
    assert audit['bundle_reuse_outcome'] == 'resident_reuse'
    assert agent.trace[-1]['turn_audit'] == audit


def test_evicted_reload_turn_audit_has_exact_load_only():
    runtime = harness('evicted')
    agent = ExperimentalSkillAgent(runtime, Client([
        native('load_skill_body', {'skill_id': 'powerpoint'}),
        {'role': 'assistant', 'content': 'Reloaded.'},
    ]))
    assert agent.run('继续改刚才 PPT 的第三页') == 'Reloaded.'
    audit = agent.turn_audit
    assert not audit['load_capability_called']
    assert audit['load_skill_body_called']
    assert audit['retrieval_call_count_before'] == audit['retrieval_call_count_after'] == 0
    assert audit['body_load_count_before'] == 0
    assert audit['body_load_count_after'] == 1
    assert audit['bundle_reuse_outcome'] == 'evicted_reload'
    assert audit['bundle_member_body_state_after'] == {'powerpoint': 'resident'}


def test_evicted_metadata_reuse_is_not_mislabeled_resident_reuse():
    runtime = harness('evicted')
    agent = ExperimentalSkillAgent(
        runtime, Client([{'role': 'assistant', 'content': 'Skipped.'}]))
    agent.run('继续改刚才 PPT 的第三页')
    assert agent.turn_audit['bundle_reuse_outcome'] == 'bundle_metadata_reuse'


def test_gap_trace_records_discovery_without_host_side_shortcut():
    runtime = harness()
    state_before = deepcopy(runtime.state)
    agent = ExperimentalSkillAgent(runtime, Client([
        native('load_capability', {'need': 'export spreadsheet data to Excel'}),
        {'role': 'assistant', 'content': 'Candidates inspected.'},
    ]))
    assert agent.run('继续改 PPT，并把数据另存 Excel') == 'Candidates inspected.'
    audit = agent.turn_audit
    assert audit['load_capability_called']
    assert not audit['load_skill_body_called']
    assert audit['retrieval_call_count_after'] == audit['retrieval_call_count_before'] + 1
    assert audit['bundle_reuse_outcome'] == 'discovery'
    assert audit['tool_calls'][0]['arguments']['need'] == 'export spreadsheet data to Excel'
    assert runtime.state == state_before  # Search/trace add no routing state.
