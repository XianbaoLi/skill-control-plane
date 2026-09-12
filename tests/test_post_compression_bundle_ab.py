from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery
from tests.support import full_discovery


def experiment_module():
    path = Path(__file__).parents[1] / 'scripts/post_compression_bundle_ab_e2e.py'
    spec = spec_from_file_location('post_compression_bundle_ab', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Client:
    def complete_messages(self, messages, *, tools):
        return {'role': 'assistant', 'content': 'done'}


def discovery():
    store = SkillRegistry([
        SkillRecord('powerpoint', 'PowerPoint', 'presentation slides',
                    'FULL PRIVATE BODY', '/powerpoint/SKILL.md'),
    ])
    return full_discovery(store)


def test_frozen_snapshot_and_compressed_history_are_identical_across_arms():
    module = experiment_module()
    visible = module.make_agent(discovery(), Client(), visible=True)
    hidden = module.make_agent(discovery(), Client(), visible=False)
    assert visible.control_plane.context_snapshot() == hidden.control_plane.context_snapshot()
    assert visible.history == hidden.history == list(module.COMPRESSED_HISTORY)
    snapshot = visible.control_plane.context_snapshot()
    assert snapshot.skill_body_states == {'powerpoint': 'evicted'}
    assert snapshot.maintained_bundles[0].bundle_id == 'presentation-work'
    assert tuple(member.skill_id for member in snapshot.maintained_bundles[0].members) \
        == ('powerpoint',)
    assert json.loads(json.dumps(module.runtime_snapshot()))['direct_skills'] == []


def test_hidden_first_input_has_no_capability_state_leakage():
    module = experiment_module()
    hidden = module.make_agent(discovery(), Client(), visible=False)
    messages = module.first_model_input(hidden, module.PARAPHRASES[0])
    audit = module.audit_hidden_input(messages, full_body='FULL PRIVATE BODY')
    assert not audit['bundle_hidden_leakage']
    assert all(audit['checks'].values())
    assert messages[-1]['content'] == module.PARAPHRASES[0]


def test_leakage_audit_rejects_old_tool_state_and_body():
    module = experiment_module()
    hidden = module.make_agent(discovery(), Client(), visible=False)
    messages = module.first_model_input(hidden, module.PARAPHRASES[0])
    leaked = deepcopy(messages)
    leaked.insert(-1, {'role': 'tool', 'tool_call_id': 'old',
                       'content': '{"selected_skill_ids":["powerpoint"],'
                                  '"body":"FULL PRIVATE BODY"}'})
    audit = module.audit_hidden_input(leaked, full_body='FULL PRIVATE BODY')
    assert audit['bundle_hidden_leakage']
    assert not audit['checks']['no_skill_id_in_compressed_history']
    assert not audit['checks']['no_selected_skill_ids']
    assert not audit['checks']['no_full_body']
    assert not audit['checks']['no_capability_tool_messages']


def test_system_ablation_changes_only_dynamic_bundle_visibility():
    module = experiment_module()
    visible = module.make_agent(discovery(), Client(), visible=True)
    hidden = module.make_agent(discovery(), Client(), visible=False)
    visible_system = visible.render_system_context()
    hidden_system = hidden.render_system_context()
    prefix, _ = visible_system.split('Runtime Bundles (metadata only)\n', 1)
    import re
    prefix = re.sub(
        r'Authoritative evicted Bundle Skill IDs: \[[^\n]*\]',
        'Authoritative evicted Bundle Skill IDs: []', prefix)
    assert (prefix + 'Runtime Bundles (metadata only)\n'
            '{"maintained_bundles":[]}') == hidden_system
    assert 'presentation-work' in visible_system
    assert 'powerpoint' in visible_system
    assert 'presentation-work' not in hidden_system
    assert 'powerpoint' not in hidden_system.casefold()


def test_experiment_has_ten_paired_natural_paraphrases():
    module = experiment_module()
    assert len(module.PARAPHRASES) >= 10
    assert len(set(module.PARAPHRASES)) == len(module.PARAPHRASES)
    assert sum('powerpoint' in query.casefold() for query in module.PARAPHRASES) == 0


def test_summary_excludes_leaked_cases_and_computes_rates():
    module = experiment_module()
    included = {
        'excluded': False, 'load_capability_called': False,
        'load_skill_body_called': True, 'exact_reload': True,
        'rediscovery': False, 'reuse_success': True,
        'retrieval_calls': 0, 'body_loads': 1, 'model_calls': 2,
        'input_tokens': 100, 'total_tokens': 130, 'wall_time_seconds': 1.5,
    }
    excluded = {'excluded': True}
    summary = module.summarize([included, excluded], visible=True)
    assert summary['included_cases'] == 1
    assert summary['load_capability_rate'] == 0
    assert summary['load_skill_body_rate'] == 1
    assert summary['exact_reload_rate'] == 1
    assert summary['reuse_success_rate'] == 1
    assert summary['body_load_count'] == 1
    assert summary['input_tokens'] == 100
    assert summary['total_tokens'] == 130
