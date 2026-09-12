from dataclasses import replace
import inspect

import pytest

from skill_control_plane.evals.legacy import soft_bundle
from skill_control_plane.evals.legacy.soft_bundle import SoftBundleRetriever, SoftBudget, explicit_priors
from skill_control_plane.evals.legacy.capability_need import CapabilityNeed
from skill_control_plane.evals.soft_bundle import evaluate_soft_bundle_cases
from skill_control_plane.evals.hierarchical import evaluate_hierarchical_cases
from skill_control_plane.evals.control_plane import StageGold, StageTransitionGoldCase
from skill_control_plane.models import SkillRecord, RetrievalCandidate
from skill_control_plane.cli import _build_parser, main


class Search:
    def __init__(self, ids, score=.8):
        self.ids, self.score, self.calls = ids, score, []
    def search(self, query, k=5):
        self.calls.append((query, k))
        return [RetrievalCandidate(s, self.score, i+1) for i, s in enumerate(self.ids[:k])]


class Extractor:
    def __init__(self, error=None):
        self.calls, self.error = [], error
    def extract(self, task, evidence):
        self.calls.append((task, evidence))
        if self.error:
            raise self.error
        return CapabilityNeed('repair query', .9)


def setup(score=.8):
    records = {s: SkillRecord(s, s, s, '', '', category=g) for s,g in
               [('local', 'github'), ('debug', 'software-development'), ('outside', 'email'),
                ('second', 'email'), ('repair', 'email')]}
    bm, dn = Search(['outside', 'second']), Search(['outside', 'second'], score)
    factory = lambda subset: Search([s.skill_id for s in subset])
    return records, dict(bm25=bm, dense=dn, dense_factory=factory)


def test_rules_multiple_no_match_and_no_labels():
    assert explicit_priors('PR #8 Python traceback', {'github','software-development'}) == ['github','software-development']
    assert explicit_priors('CI failed', {'github'}) == []
    assert explicit_priors('email calendar invitation', {'email','productivity'}) == ['email','productivity']
    assert explicit_priors('PR #2', {'email'}) == []
    assert 'gold' not in inspect.getsource(soft_bundle).lower()
    assert list(inspect.signature(explicit_priors).parameters) == ['query','available','limit']


@pytest.mark.parametrize('query', ['PR #7', 'CI failed', 'PR #7 traceback'])
def test_always_global_coexisting_local_and_dedup(query):
    records, args = setup()
    router = SoftBundleRetriever(records, **args)
    pool, trace = router.retrieve('task', [query])
    ids = [c.skill_id for c in pool]
    assert 'outside' in ids
    assert trace['global_lane'] and args['bm25'].calls and args['dense'].calls
    assert len(ids) == len(set(ids)) <= 10
    assert trace['rewrite_calls'] == 0
    if 'PR' in query:
        assert 'local' in ids
    if 'traceback' in query:
        assert 'debug' in ids


def test_budget_and_shared_local_cap():
    records = {str(i): SkillRecord(str(i), str(i), '', '', '', category='github' if i<8 else 'software-development') for i in range(16)}
    bm, dn = Search([str(i) for i in range(16)]), Search([str(i) for i in range(15,-1,-1)])
    router = SoftBundleRetriever(records, bm25=bm, dense=dn,
        dense_factory=lambda subset: Search([s.skill_id for s in subset]), budget=SoftBudget(local_limit=3))
    pool, trace = router.search('PR #7 traceback')
    assert trace['local_candidate_count'] <= 3
    assert trace['global_candidate_count'] == 6
    assert len(pool) <= 9
    assert bm.calls == [('PR #7 traceback', 3)]
    with pytest.raises(ValueError):
        SoftBudget(global_k=0)


def test_successful_first_pass_no_rewrite():
    records, args = setup()
    ex = Extractor()
    pool, trace = SoftBundleRetriever(records, **args).retrieve('task', ['PR #8'], extractor=ex)
    assert not ex.calls and not trace['repair_triggered']
    assert trace['retrieval_passes'] == 1


@pytest.mark.parametrize('error', [TimeoutError(), RuntimeError(), ValueError()])
def test_provider_failure_falls_back(error):
    records, args = setup(.1)
    router, ex = SoftBundleRetriever(records, **args), Extractor(error)
    first, _ = router.search('raw')
    pool, trace = router.retrieve('task', ['raw'], extractor=ex)
    assert pool == first and len(ex.calls) == 1
    assert trace['repair_status'].startswith('error:')
    assert trace['retrieval_passes'] == 1


def test_repair_union_and_retrieval_failure():
    records, args = setup(.1)
    class QuerySearch(Search):
        def search(self, query, k=5):
            return Search(['repair'] if query == 'repair query' else ['outside'], .1).search(query,k)
    args['bm25'] = args['dense'] = QuerySearch([])
    router = SoftBundleRetriever(records, **args)
    pool, trace = router.retrieve('task', ['raw'], extractor=Extractor())
    assert [c.skill_id for c in pool] == ['outside', 'repair']
    assert trace['repair_added_ids'] == ['repair'] and trace['retrieval_passes'] == 2
    original = router.search
    def failing(query):
        if query == 'repair query':
            raise RuntimeError()
        return original(query)
    router.search = failing
    pool, trace = router.retrieve('task', ['raw'], extractor=Extractor())
    assert [c.skill_id for c in pool] == ['outside']
    assert trace['repair_status'] == 'retrieval-error:RuntimeError'


def case():
    return StageTransitionGoldCase(case_id='case', initial_task='initial', snapshot_id='snapshot', rationale='', stages=(
        StageGold('S1', (), ('outside',), ('outside',), (), ()),
        StageGold('S2', ('PR #8 traceback',), ('debug',), ('debug',), (), ()),
        StageGold('S3', ('email',), ('second',), ('second',), (), ())))


def test_exact_A_frozen_independence_and_label_perturbation():
    records,args = setup(.1)
    old = evaluate_hierarchical_cases([case()], records, **args)['hierarchical']
    ex = Extractor()
    report = evaluate_soft_bundle_cases([case()], records, **args, extractor=ex)
    assert report['baseline_A'] == old
    for arm in ('A','B','C'):
        for a,b in zip(report['sequential']['A']['stages'], report['frozen'][arm]['stages']):
            assert a['pre_shelf'] == b['pre_shelf']
    again = evaluate_soft_bundle_cases([case()], records, **args, extractor=Extractor())
    assert again == report
    other = replace(case(), case_id='another', stages=tuple(replace(s, required_now=('repair',), new_required=('repair',)) for s in case().stages))
    ex2 = Extractor()
    changed = evaluate_soft_bundle_cases([other], records, **args, extractor=ex2)
    assert ex.calls == ex2.calls
    for mode in ('sequential','frozen'):
        for arm in ('A','B','C'):
            assert [r['candidate_ids'] for r in changed[mode][arm]['stages']] == [r['candidate_ids'] for r in report[mode][arm]['stages']]
    for arm in ('B','C'):
        assert report['sequential'][arm]['metrics']['global_lane_rate'] == 1


def test_cli_opt_in_and_exclusion():
    args = ['eval','stage-bundle','root','--gold','gold','--manifest','manifest']
    parser = _build_parser()
    parsed = parser.parse_args(args)
    assert not parsed.soft_bundle_ab and not parsed.soft_bundle_repair_ab
    for flag in ('--hierarchical-ab','--capability-need-ab','--soft-bundle-ab'):
        with pytest.raises(SystemExit):
            parser.parse_args(args+[flag,'--soft-bundle-repair-ab'])
    with pytest.raises(ValueError, match='requires'):
        main(args+['--soft-bundle-repair-ab'])
    with pytest.raises(ValueError, match='requires'):
        main(args+['--soft-bundle-ab','--capability-need-command','unused'])


def test_recovery_metric_and_trigger_denominator():
    from skill_control_plane.evals.soft_bundle import summarize
    metrics = summarize([
        dict(case_id='x', stage_id='S2', repair_triggered=True, rewrite_calls=1,
             retrieval_passes=2, required_skill_recovered_by_repair=['recovered'],
             candidate_pool_size=4, required_skill_candidate_recall=1),
        dict(case_id='x', stage_id='S3', repair_triggered=True, rewrite_calls=1,
             retrieval_passes=1, required_skill_recovered_by_repair=[],
             candidate_pool_size=2, required_skill_candidate_recall=0),
    ])
    assert metrics['repair_success_rate'] == .5
    assert metrics['mean_retrieval_passes'] == 1.5
    assert metrics['candidate_recall_per_candidate'] == pytest.approx(1/6)


def test_global_overlap_signal_independently_triggers():
    records, args = setup()
    args['dense'] = Search(['repair'], .9)
    ex = Extractor()
    _, trace = SoftBundleRetriever(records, **args).retrieve('background', ['raw'], extractor=ex)
    assert trace['uncertainty_reasons'] == ['global-bm25-dense-no-overlap']
    assert ex.calls == [('background', ['raw'])]
