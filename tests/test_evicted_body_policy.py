from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.capability_loading import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.runtime.experimental_agent import ExperimentalSkillAgent


class Client:
    def complete_messages(self, messages, *, tools):
        return {'role': 'assistant', 'content': 'done'}


def harness():
    registry = SkillRegistry([SkillRecord(
        'powerpoint', 'PowerPoint', 'presentation slides',
        'PRIVATE BODY', '/powerpoint/SKILL.md')])
    return RuntimeCapabilityHarness(
        discovery=SkillDiscovery(registry),
        state=RuntimeCapabilityState(
            [ActiveBundle('presentation-work', 'Presentations', ('powerpoint',))],
            skill_body_states={'powerpoint': 'evicted'}),
    )


def experiment_module():
    path = Path(__file__).parents[1] / 'scripts/evicted_body_policy_ab_e2e.py'
    spec = spec_from_file_location('evicted_body_policy_ab', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_policy_allows_metadata_reuse_without_relying_on_old_body():
    system = ExperimentalSkillAgent(harness(), Client()).render_system_context()
    normalized = ' '.join(system.split())
    assert 'An evicted body is unavailable' in normalized
    assert 'never rely on it' in normalized
    assert 'whether the current step can be handled without detailed' in normalized
    assert 'If that metadata is sufficient, continue without reloading' in normalized
    assert 'changing presentation content, titles, page order, or layout' in normalized
    assert 'The absence of an execution tool is not a reason to reload' in normalized
    assert 'exact Skill-specific procedures' in normalized
    assert 'you MUST call load_skill_body' not in normalized


def test_old_baseline_changes_only_body_policy_instruction():
    new = ExperimentalSkillAgent(harness(), Client())
    old = ExperimentalSkillAgent(
        harness(), Client(), require_evicted_body_reload=True)
    new_system = new.render_system_context()
    old_system = old.render_system_context()
    assert 'mandatory-reload baseline' in old_system
    assert 'you MUST call load_skill_body' in old_system
    assert 'metadata is sufficient, continue without reloading' not in old_system
    new_tail = new_system.split('Authoritative evicted Bundle Skill IDs:', 1)[1]
    old_tail = old_system.split('Authoritative evicted Bundle Skill IDs:', 1)[1]
    assert new_tail == old_tail
    assert new.harness.state == old.harness.state


def test_evicted_no_tool_is_metadata_reuse_not_missing_reload():
    agent = ExperimentalSkillAgent(harness(), Client())
    assert agent.run('adjust the title') == 'done'
    assert agent.turn_audit['bundle_reuse_outcome'] == 'bundle_metadata_reuse'
    assert agent.turn_audit['bundle_member_body_state_after'] == {
        'powerpoint': 'evicted'}
    assert agent.trace[0]['model_visible_body_ids'] == []


def test_experiment_has_balanced_simple_and_detailed_cases():
    module = experiment_module()
    simple = [case for case in module.CASES if case[0] == 'simple']
    detailed = [case for case in module.CASES if case[0] == 'detailed']
    assert len(module.CASES) == 12
    assert len(simple) == len(detailed) == 6
    old = module.make_agent(discovery=harness().discovery, client=Client(),
                            policy='old_mandatory')
    new = module.make_agent(discovery=harness().discovery, client=Client(),
                            policy='new_metadata_first')
    assert old.harness.state == new.harness.state == module.frozen_state()
    assert old.history == new.history == list(module.COMPRESSED_HISTORY)
    assert old.require_evicted_body_reload
    assert not new.require_evicted_body_reload


def test_behavior_classification_distinguishes_cost_and_accuracy_outcomes():
    module = experiment_module()

    def audit(tool=None):
        return {'tool_calls': [] if tool is None else [{
            'tool': tool, 'arguments': {'skill_id': 'powerpoint'}}]}

    assert module.classify_case(
        category='simple', audit=audit(), trace=[], answer='done')['outcome'] \
        == 'reused_bundle_only'
    assert module.classify_case(
        category='simple', audit=audit('load_skill_body'), trace=[],
        answer='done')['outcome'] == 'unnecessary_reload'
    assert module.classify_case(
        category='detailed', audit=audit(), trace=[], answer='done')['outcome'] \
        == 'missing_reload_failure'
    assert module.classify_case(
        category='detailed', audit=audit('load_skill_body'), trace=[],
        answer='done')['outcome'] == 'exact_reloaded'
    discovery_audit = {'tool_calls': [{
        'tool': 'load_capability', 'arguments': {'need': 'presentation'}}]}
    assert module.classify_case(
        category='simple', audit=discovery_audit, trace=[],
        answer='done')['outcome'] == 'rediscovered'


def test_summary_metrics_include_reload_avoidance_and_failure_rates():
    module = experiment_module()
    base = {
        'rediscovered': False, 'retrieval_calls': 0,
        'load_skill_body_called': False, 'body_loads': 0,
        'model_calls': 1, 'input_tokens': 10, 'output_tokens': 2,
        'total_tokens': 12, 'wall_time_seconds': 0.5,
    }
    cases = [
        {**base, 'category': 'simple', 'behavior_success': True,
         'outcome': 'reused_bundle_only'},
        {**base, 'category': 'simple', 'behavior_success': True,
         'load_skill_body_called': True, 'body_loads': 1,
         'outcome': 'unnecessary_reload'},
        {**base, 'category': 'detailed', 'behavior_success': False,
         'outcome': 'missing_reload_failure'},
        {**base, 'category': 'detailed', 'behavior_success': True,
         'load_skill_body_called': True, 'body_loads': 1,
         'outcome': 'exact_reloaded'},
    ]
    summary = module.summarize(cases)
    assert summary['full_body_reload_avoided_rate'] == 0.25
    assert summary['unnecessary_reload_rate'] == 0.5
    assert summary['missing_reload_failure_rate'] == 0.5
    assert summary['behavior_success_rate'] == 0.75
    assert summary['simple']['load_skill_body_rate'] == 0.5
    assert summary['detailed']['behavior_success_rate'] == 0.5
