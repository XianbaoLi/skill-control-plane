from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / 'scripts/capability_sufficiency_e2e.py'
    spec = spec_from_file_location('capability_sufficiency_e2e', path)
    loaded = module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_live_cases_cover_all_required_scenarios():
    experiment = module()
    assert [case['case_id'] for case in experiment.CASES] == [
        'no_gap', 'single_gap', 'multi_gap', 'bundle_pending_coverage',
        'missing_skill', 'partial_success', 'repeated_no_progress']
    assert experiment.CASES[0]['bundles']
    assert not experiment.CASES[2]['bundles']
    assert experiment.CASES[2]['candidate_window'] == 2
    assert experiment.CASES[3]['candidate_window'] == 1
    assert experiment.CASES[3]['bundles'][0][2] == ('nano-pdf',)
    assert experiment.CASES[4]['corpus'] == 'limited'


def test_evaluator_rejects_forced_selection_and_budget_excess():
    experiment = module()

    class Agent:
        trace = [{'selected_skill_ids': ['pdf']}]
        turn_audit = {
            'search_count': 4, 'search_needs': ['missing'] * 4,
            'apply_count': 0,
            'capability_sufficiency_outcome': 'UNSATISFIED',
            'unresolved_gaps': ['missing'], 'no_progress_search_count': 1,
        }

    result = experiment.evaluate(experiment.CASES[4], Agent(), 'done')
    assert not result['passed']
    assert result['forced_skill_ids'] == ['pdf']
    assert result['selected_skill_precision'] == 0.0


def test_trace_outcomes_are_derived_from_executed_tool_behavior():
    experiment = module()
    assert experiment.derived_tool_outcome({
        'tool': 'load_capability', 'model_visible_payload': {}}) == 'SEARCH_MORE'
    assert experiment.derived_tool_outcome({'tool': 'load_capability'}) is None
    assert experiment.derived_tool_outcome({
        'tool': 'apply_capability', 'remaining_gaps': []}) == 'COVERED'
    assert experiment.derived_tool_outcome({
        'tool': 'apply_capability', 'remaining_gaps': ['missing']}) == 'UNSATISFIED'


def test_summary_records_search_distribution_and_precision():
    experiment = module()
    base = {
        'passed': True, 'selected_skill_precision': 1.0,
        'forced_skill_selection_count': 0, 'model_calls': 2,
        'input_tokens': 10, 'total_tokens': 12, 'wall_time_seconds': 0.5,
    }
    rows = [
        {**base, 'case_id': 'a', 'audit': {
            'search_count': 0, 'apply_count': 0,
            'capability_sufficiency_outcome': 'COVERED',
            'sufficiency_transitions': [],
            'unresolved_gaps': [], 'repeated_search_count': 0,
            'no_progress_search_count': 0, 'search_budget_hits': 0}},
        {**base, 'case_id': 'b', 'audit': {
            'search_count': 2, 'apply_count': 1,
            'capability_sufficiency_outcome': 'UNSATISFIED',
            'sufficiency_transitions': ['SEARCH_MORE', 'UNSATISFIED'],
            'unresolved_gaps': ['missing'], 'repeated_search_count': 1,
            'no_progress_search_count': 1, 'search_budget_hits': 0}},
    ]
    summary = experiment.summarize(rows)
    assert summary['search_count_distribution'] == {'0': 1, '2': 1}
    assert summary['outcomes'] == {
        'COVERED': 1, 'SEARCH_MORE': 0, 'UNSATISFIED': 1}
    assert summary['forced_skill_selection_count'] == 0
