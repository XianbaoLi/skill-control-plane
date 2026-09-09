import pytest

from skill_control_plane.evals.control_plane import StageGold, StageTransitionGoldCase
from skill_control_plane.evals.hierarchical import evaluate_hierarchical_cases
from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.runtime.bundles import build_capability_shelf, integrate_retrieval_delta
from skill_control_plane.runtime.hierarchical import ShelfAwareRetriever


def record(skill_id, category):
    return SkillRecord(skill_id, skill_id, skill_id, '', '', category=category)


def candidate(skill_id, score=1.0, rank=1):
    return RetrievalCandidate(skill_id, score, rank)


class StaticRetriever:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    def search(self, query, k=5):
        self.calls.append((query, k))
        return self.candidates[:k]


def test_match_searches_unregistered_members_and_activates_monotonically():
    records = {r.skill_id: r for r in [record('known', 'code'), record('debug', 'code'), record('mail', 'email')]}
    shelf = build_capability_shelf([candidate('known')], records)
    global_search = StaticRetriever([candidate('mail')])
    subsets = []

    def factory(skills):
        subsets.append({s.skill_id for s in skills})
        if skills[0].skill_id == 'code':
            return StaticRetriever([candidate('code', .8)])
        return StaticRetriever([candidate('debug')])

    router = ShelfAwareRetriever(records, bm25=global_search, dense=global_search, dense_factory=factory)
    route = router.search('debug', shelf)
    assert route.route == 'bundle-local'
    assert route.matched_bundle_id == 'code'
    assert route.corpus_size == 2
    assert subsets == [{'code'}, {'known', 'debug'}]
    assert not global_search.calls
    expanded = integrate_retrieval_delta(shelf, route.candidates, records)
    assert set(expanded.registered_skill_ids) == {'known', 'debug'}
    assert expanded.active_bundle_ids == ('code',)
    assert set(shelf.registered_skill_ids) <= set(expanded.registered_skill_ids)
    again = integrate_retrieval_delta(expanded, route.candidates, records)
    assert again == expanded


@pytest.mark.parametrize('scores', [[], [.2], [.8, .79]])
def test_no_confident_match_falls_back_and_creates_new_bundle(scores):
    records = {r.skill_id: r for r in [record('known', 'code'), record('memo', 'notes'), record('mail', 'email')]}
    shelf = build_capability_shelf([candidate('known'), candidate('memo', rank=2)], records)
    global_search = StaticRetriever([candidate('mail')])
    matcher = StaticRetriever([candidate(s, score, i+1) for i, (s, score) in enumerate(zip(['code', 'notes'], scores))])
    router = ShelfAwareRetriever(records, bm25=global_search, dense=global_search, dense_factory=lambda _: matcher)
    route = router.search('email', shelf)
    assert route.route == 'global'
    assert route.corpus_size == 3
    assert route.matched_bundle_id is None
    updated = integrate_retrieval_delta(shelf, route.candidates, records)
    assert updated.get('email').skill_ids == ('mail',)
    assert set(updated.registered_skill_ids) == {'known', 'memo', 'mail'}
    assert set(shelf.active_bundle_ids) <= set(updated.active_bundle_ids)
    assert 'email' in updated.active_bundle_ids


def test_global_policy_skips_matcher_and_empty_evidence_falls_back():
    records = {'known': record('known', 'code')}
    shelf = build_capability_shelf([candidate('known')], records)
    search = StaticRetriever([candidate('known')])
    def forbidden(_):
        raise AssertionError('matcher must not run')
    router = ShelfAwareRetriever(records, bm25=search, dense=search, dense_factory=forbidden)
    assert router.search('query', shelf, hierarchical=False).route == 'global'
    assert router.search('', shelf).route == 'global'
    with pytest.raises(ValueError):
        router.search('query', shelf, k=0)


def test_ab_metric_denominators_and_wrong_match_for_new_capability():
    records = {r.skill_id: r for r in [record('known', 'code'), record('mail', 'email')]}
    search = StaticRetriever([candidate('known')])
    factory = lambda skills: StaticRetriever([candidate(skills[0].skill_id, .9)])
    case = StageTransitionGoldCase(
        case_id='test', initial_task='start', snapshot_id='test', rationale='',
        stages=(StageGold(stage_id='S1', runtime_evidence=(), required_now=('known',), new_required=('known',), useful=(), hard_negative=()),
                StageGold(stage_id='S2', runtime_evidence=('email',), required_now=('known', 'mail'), new_required=('mail',), useful=(), hard_negative=())),
    )
    r = evaluate_hierarchical_cases([case], records, bm25=search, dense=search, dense_factory=factory)
    a, b = r['global-only'], r['hierarchical']
    assert a['global_retrieval_rate'] == 1
    assert b['bundle_local_retrieval_rate'] == 1
    assert b['global_retrieval_rate'] == 0
    assert b['wrong_bundle_match_rate'] == 1
    assert b['mean_search_corpus_size'] == 1
    assert b['mean_matcher_corpus_size'] == 1
    assert b['mean_registered_surface_growth'] == 0
    assert b['mean_active_surface_growth'] == 0
    assert b['mean_shelf_required_recall'] == .75
    assert a['required_skill_recall'] == b['required_skill_recall'] == .75


def test_ab_global_arm_preserves_existing_evaluator():
    from skill_control_plane.evals.bundles import evaluate_stage_bundle_cases
    records = {r.skill_id: r for r in [record('known', 'code'), record('mail', 'email')]}
    search = StaticRetriever([candidate('known'), candidate('mail', rank=2)])
    case = StageTransitionGoldCase(
        case_id='test', initial_task='start', snapshot_id='test', rationale='',
        stages=(StageGold('S1', (), ('known',), ('known',), (), ()),
                StageGold('S2', ('mail',), ('mail',), ('mail',), (), ())),
    )
    old = evaluate_stage_bundle_cases([case], records, bm25=search, dense=search)
    ab = evaluate_hierarchical_cases(
        [case], records, bm25=search, dense=search,
        dense_factory=lambda _: StaticRetriever([]),
    )['global-only']
    for key in ('mean_active_required_recall', 'mean_shelf_required_recall'):
        assert ab[key] == old[key]
    for before, after in zip(old['cases'][0]['stages'], ab['cases'][0]['stages']):
        assert all(after[key] == value for key, value in before.items())


def test_dense_matcher_uses_only_registered_metadata_and_reuses_encoder():
    from skill_control_plane.retrieval.dense import DenseRetriever
    from skill_control_plane.runtime.hierarchical import dense_factory_for
    records = {r.skill_id: r for r in [record('known', 'code'), record('debug', 'code'), record('mail', 'email')]}
    class Encoder:
        def __init__(self):
            self.texts = []
        def encode(self, texts, normalize_embeddings=True):
            self.texts.extend(texts)
            return [[1.0, 0.0] if 'mail' not in text else [0.0, 1.0] for text in texts]
    encoder = Encoder()
    dense = DenseRetriever(list(records.values()), model=encoder)
    factory = dense_factory_for(dense)
    assert factory([]).model is encoder
    encoder.texts.clear()
    shelf = build_capability_shelf([candidate('known')], records)
    router = ShelfAwareRetriever(records, bm25=StaticRetriever([]), dense=dense, dense_factory=factory)
    route = router.search('debug', shelf)
    descriptor = encoder.texts[0]
    assert 'Code' in descriptor and 'known' in descriptor
    assert 'debug' not in descriptor and 'mail' not in descriptor
    assert route.route == 'bundle-local'
    assert {c.skill_id for c in route.candidates} == {'known', 'debug'}
