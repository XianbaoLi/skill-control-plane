"""Audit experiment aggregation independently of online embedding calls."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('robustness_current87', Path(__file__).parents[1] / 'scripts/robustness_current87.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def stage(ranks):
    return {'case_id': 'C', 'stage_id': 'S2', 'new_required': ['target'],
            'variants': [{**{arm: {'new_required_ranks': {'target': rank}} for arm in runner.ARMS},
                          'union': {'new_required_hits': ['target'] if rank else []}} for rank in ranks]}


def test_censored_worst_rank_and_cutoff_mrr():
    metrics = runner.target_metrics(stage([1, 5, 6, 10, None]), 'dense')
    assert metrics['worst_rank'] is None
    assert metrics['missing_from_retrieved_candidates'] == 1
    assert metrics['recall_at_5'] == 2 / 5
    assert metrics['recall_at_10'] == 4 / 5
    assert metrics['mrr_at_5'] == pytest.approx(1.2 / 5)
    assert metrics['mrr_at_10'] == pytest.approx((1 + .2 + 1/6 + .1) / 5)
    assert metrics['all_variants_hit_at_10'] == 0
    assert runner.target_metrics(stage([1, 2, 3, 4, 5]), 'dense')['worst_rank'] == 5


def report(ranks):
    row = stage(ranks)
    return {'target_transition_count': 1, 'new_required_skill_occurrences': 1,
            'query_variant_count': 5, 'variants_per_transition': 5, 'cutoffs': [5, 10],
            **{arm: {'recall_at_5': runner.target_metrics(row, arm)['recall_at_5']} for arm in runner.ARMS},
            'union': dict(candidate_recall=1, original_candidate_recall=1,
                          paraphrase_candidate_recall=1, all_variants_covered=1),
            'stages': [row]}


def test_pairwise_direction_ties_and_recall_vs_rank():
    pairs = runner.pairwise({'left': report([5, 5, 5, 5, 5]), 'right': report([1, 1, 1, 1, 1])})
    assert len(pairs) == 1
    outcomes = pairs[0]['per_target'][0]['arms']['dense']
    assert outcomes['recall_at_5']['outcome'] == 'tie'
    assert outcomes['mrr_at_5']['outcome'] == 'win'
    assert outcomes['mrr_at_5']['delta'] == pytest.approx(.8)
    assert pairs[0]['win_tie_loss_counts']['dense']['mrr_at_5'] == dict(win=1, tie=0, loss=0)
    reverse = runner.pairwise({'right': report([1]*5), 'left': report([5]*5)})
    assert reverse[0]['per_target'][0]['arms']['dense']['mrr_at_5']['outcome'] == 'loss'


def local_inputs():
    if not runner.SOURCE.exists() or not runner.QUERIES.exists():
        pytest.skip('local experiment inputs not distributed with repository')
    return (json.loads(runner.QUERIES.read_text()), json.loads(runner.SOURCE.read_text()),
            runner.load_stage_transition_gold(runner.GOLD))


def test_frozen_queries_match_originals_and_reject_drift():
    frozen, source, cases = local_inputs()
    assert len(runner.validate_queries(frozen, source, cases)) == 13
    changed = copy.deepcopy(frozen)
    changed['targets'][0]['variants'][0] += ' '
    with pytest.raises(AssertionError):
        runner.validate_queries(changed, source, cases)
    changed = copy.deepcopy(frozen)
    changed['targets'][0]['variants'][4] = changed['targets'][0]['variants'][3]
    with pytest.raises(AssertionError):
        runner.validate_queries(changed, source, cases)


def test_artifact_metrics_recomputed_from_all_65_ranks():
    path = runner.BASE / f'{runner.PREFIX}.json'
    if not path.exists():
        pytest.skip('local completed experiment not distributed with repository')
    data = json.loads(path.read_text())
    assert len(data['reports']) == 6 and len(data['pairwise']) == 15
    for path, expected in data['input_sha256'].items():
        assert runner.sha(path) == expected
    for report in data['reports'].values():
        assert len(report['stages']) == 13
        for arm in runner.ARMS:
            for cutoff in (5, 10):
                ranks = [v[arm]['new_required_ranks'][s['new_required'][0]] for s in report['stages'] for v in s['variants']]
                assert len(ranks) == 65
                assert report[arm][f'recall_at_{cutoff}'] == sum(r is not None and r <= cutoff for r in ranks) / 65
                assert report[arm][f'mrr_at_{cutoff}'] == pytest.approx(sum(1/r for r in ranks if r is not None and r <= cutoff) / 65)
                all_hits = sum(all(v[arm]['new_required_ranks'][s['new_required'][0]] is not None and v[arm]['new_required_ranks'][s['new_required'][0]] <= cutoff for v in s['variants']) for s in report['stages'])
                assert report[arm][f'all_variants_hit_at_{cutoff}'] == all_hits / 13
