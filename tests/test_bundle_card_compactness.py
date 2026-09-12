import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.cards import RetrievalCard
from skill_control_plane.discovery.discovery import SkillDiscovery


class Client:
    def complete_messages(self, messages, *, tools):
        return {'role': 'assistant', 'content': 'done'}


def module():
    path = Path(__file__).parents[1] / 'scripts/bundle_card_compactness_ab_e2e.py'
    spec = spec_from_file_location('bundle_card_compactness_ab', path)
    loaded = module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def discovery():
    records = [
        SkillRecord('powerpoint', 'PowerPoint', 'presentation description',
                    'POWERPOINT BODY', ''),
        SkillRecord('xlsx', 'Excel', 'spreadsheet description', 'XLSX BODY', ''),
    ]
    cards = {'powerpoint': RetrievalCard(
        'powerpoint', 'hash', 'presentations', ('when editing slides',),
        ('create slides', 'edit layouts', 'render presentations',
         'export presentations', 'validate slides', 'extra capability'), ('ppt',))}
    return SkillDiscovery(SkillRegistry(records), retrieval_cards=cards)


def test_ab_agents_differ_only_in_bundle_surface():
    experiment = module()
    old = experiment.make_agent(
        discovery(), Client(), category='evicted_simple', surface='old')
    compact = experiment.make_agent(
        discovery(), Client(), category='evicted_simple', surface='compact')
    old_system = old.render_system_context()
    compact_system = compact.render_system_context()
    old_prefix, old_card = old_system.split(experiment.SURFACE_MARKER, 1)
    compact_prefix, compact_card = compact_system.split(experiment.SURFACE_MARKER, 1)
    assert old_prefix == compact_prefix
    assert old.history == compact.history
    assert old.control_plane.context_snapshot() == compact.control_plane.context_snapshot()
    assert old_card != compact_card
    assert 'short_description' in old_card
    assert 'short_description' not in compact_card
    assert len(json.loads(compact_card)['maintained_bundles'][0]['capabilities']) == 5


def test_behavior_cases_cover_required_routing_categories():
    experiment = module()
    categories = [category for category, _task in experiment.CASES]
    assert len(experiment.CASES) == 12
    assert categories.count('resident_reuse') == 4
    assert categories.count('evicted_simple') == 2
    assert categories.count('evicted_detailed') == 2
    assert categories.count('capability_gap') == 2
    assert categories.count('mixed_reuse_gap') == 1
    assert categories.count('unrelated_gap') == 1


def test_classification_covers_reuse_reload_and_true_gaps():
    experiment = module()

    def audit(*tools):
        return {'tool_calls': [
            {'tool': tool, 'arguments': {'skill_id': 'powerpoint'}
             if tool == 'load_skill_body' else {'need': 'spreadsheet'}}
            for tool in tools]}

    assert experiment.classify(
        'resident_reuse', audit(), [], 'done')['behavior_success']
    assert experiment.classify(
        'evicted_simple', audit(), [], 'done')['behavior_success']
    assert experiment.classify(
        'evicted_detailed', audit('load_skill_body'), [],
        'done')['behavior_success']
    assert experiment.classify(
        'capability_gap', audit('load_capability'), [], 'done')['behavior_success']
    assert not experiment.classify(
        'unrelated_gap', audit(), [], 'done')['behavior_success']
    repaired = experiment.classify(
        'capability_gap', audit('load_capability'), [{'tool_error': {
            'type': 'ValueError'}}], 'repaired and done')
    assert repaired['behavior_success']
    assert repaired['recoverable_tool_error_count'] == 1


def test_scaling_uses_realistic_fixed_bundles_and_compact_is_smaller():
    experiment = module()
    assert len(experiment.SCALING_BUNDLES) == 10
    records = []
    cards = {}
    for bundle in experiment.SCALING_BUNDLES:
        skill_id = bundle.skill_ids[0]
        records.append(SkillRecord(
            skill_id, skill_id, f'{skill_id} long member description', '', ''))
        cards[skill_id] = RetrievalCard(
            skill_id, 'hash', bundle.purpose, (f'use {skill_id}',),
            tuple(f'{skill_id} capability number {index}' for index in range(6)),
            (skill_id,))
    rows = experiment.scaling_report(SkillDiscovery(
        SkillRegistry(records), retrieval_cards=cards))
    assert [row['bundle_count'] for row in rows] == [1, 3, 5, 10]
    assert all(row['compact']['chars'] < row['old']['chars'] for row in rows)
    assert all(row['compact']['estimated_tokens_per_bundle']
               < row['old']['estimated_tokens_per_bundle'] for row in rows)
    assert experiment.estimate_tokens('x' * 9) == 3


def test_regression_count_only_includes_compact_loss_after_old_success():
    experiment = module()
    common = {
        'case_id': 1, 'category': 'capability_gap', 'task': 'spreadsheet',
        'bundle_card': {'old': 'surface'},
    }
    old = [{**common, 'behavior_success': True,
            'behavior_outcome': 'gap_discovered'}]
    compact = [{**common, 'behavior_success': False,
                'behavior_outcome': 'wrong_suppression'}]
    regressions = experiment.regression_analysis(old, compact)
    assert len(regressions) == 1
    assert regressions[0]['likely_cause'] == \
        'purpose_too_broad_or_model_stochastic_behavior'
