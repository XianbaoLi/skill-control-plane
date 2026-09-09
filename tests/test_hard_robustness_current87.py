"""Independent checks for frozen hard-input integrity and censored aggregation."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('hard_robustness_current87',SCRIPTS/'hard_robustness_current87.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_metrics_and_censored_comparison():
    m=runner.metrics([1,6,None])
    assert m['recall_at_5']==1/3
    assert m['recall_at_10']==2/3
    assert m['mrr_at_10']==pytest.approx((1+1/6)/3)
    d=runner.rank_comparison(5,None,'dense')
    assert d['rank_delta_ablation_minus_full'] is None
    assert d['capped_rank_delta']==6 and d['outcome']=='win'
    assert d['top5_crossing']=='full_only'
    assert runner.rank_comparison(6,5,'rrf')['top5_crossing']=='ablation_only'
    assert runner.rank_comparison(None,None,'dense')['outcome']=='tie'
    assert runner.rank_comparison(4,2,'dense')['rank_delta_ablation_minus_full']==-2


def test_frozen_validation_rejects_gold_kind_and_source_drift():
    if not runner.QUERIES.exists():
        pytest.skip('local frozen queries unavailable')
    frozen=json.loads(runner.QUERIES.read_text())
    previous=json.loads(runner.PREVIOUS.read_text())
    cases=runner.prior.load_stage_transition_gold(runner.prior.GOLD)
    runner.validate(frozen,previous,cases)
    for field,value in [('target','wrong'),('case_id','wrong')]:
        bad=copy.deepcopy(frozen); bad['targets'][0][field]=value
        with pytest.raises((AssertionError,KeyError)):
            runner.validate(bad,previous,cases)
    bad=copy.deepcopy(frozen);bad['targets'][0]['variants'][0]['kind']='V0'
    with pytest.raises(AssertionError):runner.validate(bad,previous,cases)
    bad=copy.deepcopy(frozen);bad['source_sha256']='wrong'
    with pytest.raises(AssertionError):runner.validate(bad,previous,cases)


def test_completed_artifact_recomputed_independently():
    path=runner.BASE/f'{runner.PREFIX}.json'
    if not path.exists():pytest.skip('local completed experiment unavailable')
    data=json.loads(path.read_text()); previous=json.loads(runner.PREVIOUS.read_text())
    assert data['evaluation_count']==234 and len(data['reports'])==6
    assert data['queries']==json.loads(runner.QUERIES.read_text())
    for p,h in data['input_sha256'].items():assert runner.prior.sha(p)==h
    for label,r in data['reports'].items():
        assert len(r['stages'])==13
        variants=[v for s in r['stages'] for v in s['variants']]
        assert len(variants)==39
        for arm in runner.ARMS:
            for s in r['stages']:
                for v in s['variants']:
                    positions={c['skill_id']:c['rank'] for c in v[arm]['ranking']}
                    assert v[arm]['rank']==positions.get(s['target'])
            for k in (5,10):
                ranks=[v[arm]['rank'] for v in variants]
                recall=sum(x is not None and x<=k for x in ranks)/39
                mrr=sum(1/x for x in ranks if x is not None and x<=k)/39
                assert r[arm][f'recall_at_{k}']==recall
                assert r[arm][f'mrr_at_{k}']==pytest.approx(mrr)
                assert r[arm][f'all_variants_hit_at_{k}']==sum(all(v[arm]['rank'] is not None and v[arm]['rank']<=k for v in s['variants']) for s in r['stages'])/13
                old=[s['variants'][0][arm]['new_required_ranks'][s['new_required'][0]] for s in previous['reports'][label]['stages']]
                for metric,ov,hv in [('recall',sum(x is not None and x<=k for x in old)/13,recall),('mrr',sum(1/x for x in old if x is not None and x<=k)/13,mrr)]:
                    assert data['hard_degradation'][label][arm]['degradation'][f'{metric}_at_{k}']==pytest.approx(ov-hv)
            for kind in runner.KINDS:
                ranks=[v[arm]['rank'] for v in variants if v['kind']==kind]
                assert len(ranks)==13
                assert r['by_kind'][kind][arm]==runner.metrics(ranks)
        for s in r['stages']:
            for v in s['variants']:
                ids={c['skill_id'] for arm in ('dense','bm25') for c in v[arm]['ranking']}
                assert set(v['union']['skill_ids'])==ids
                assert v['union']['candidate_set_size']==len(ids)
                assert v['union']['hit']==(s['target'] in ids)
        assert r['union']['candidate_recall']==sum(v['union']['hit'] for v in variants)/39
        assert r['union']['average_candidate_pool_size']==sum(v['union']['candidate_set_size'] for v in variants)/39
    for label,arms in data['field_attribution'].items():
        for arm,a in arms.items():
            assert sum(a['win_tie_loss'].values())==39
            assert a['top5_boundary_crossings']==[d for d in a['per_query'] if d['top5_crossing']]
            for d in a['per_query']:
                for key,val in runner.rank_comparison(d['full_rank'],d['ablation_rank'],arm).items():assert d[key]==val
