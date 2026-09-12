import json

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.evals.legacy.resolver_capability_loading import (
    ActiveBundle, CapabilityDecision, LLMCapabilityResolver, ResolverError,
    ResolverResult, RuntimeCapabilityLoader, RuntimeCapabilityState,
    build_resolver_prompt, validate_decision,
)


@pytest.fixture
def discovery():
    return SkillDiscovery(SkillRegistry([
        SkillRecord(s, s, text, '', '') for s, text in [
            ('github', 'GitHub pull request review'), ('git-read', 'GitHub diff read'),
            ('github-actions', 'GitHub Actions CI check failure'),
            ('pdf', 'extract PDF text'), ('slides', 'presentation slides'),
            ('email', 'email inbox replies'), ('hidden', 'unrelated astronomy'),
        ]
    ]))


class StubResolver:
    def __init__(self, decision):
        self.decision = decision

    def resolve_capability(self, *args):
        return ResolverResult(self.decision, ())


def run(discovery, state, decision, need='GitHub Actions CI failure'):
    return RuntimeCapabilityLoader(discovery, StubResolver(decision)).load_capability(need, state)


def test_multiple_direct_without_bundle(discovery):
    state = RuntimeCapabilityState()
    result = run(discovery, state, CapabilityDecision('DIRECT', ('pdf', 'slides'), 'one-off'),
                 'extract PDF text and presentation slides')
    assert result.resulting_state.direct_skills == {'pdf', 'slides'}
    assert result.resulting_state.active_bundles == []
    assert state == RuntimeCapabilityState()


def test_create_preserves_existing_surface(discovery):
    state = RuntimeCapabilityState(direct_skills={'pdf'})
    result = run(discovery, state, CapabilityDecision('CREATE', ('email', 'email'), 'ongoing', purpose='Manage email'), 'email replies')
    assert len(result.resulting_state.active_bundles) == 1
    bundle = result.resulting_state.active_bundles[0]
    assert bundle.purpose == 'Manage email' and bundle.skill_ids == ('email',)
    assert bundle.bundle_id == result.affected_bundle_id
    assert result.resulting_state.direct_skills == {'pdf'}
    assert not state.active_bundles


@pytest.mark.parametrize('multiple', [False, True])
def test_extend_only_target(discovery, multiple):
    github = ActiveBundle('github-review', 'Review GitHub pull requests', ('github', 'git-read'))
    other = ActiveBundle('email-work', 'Manage email', ('email',))
    state = RuntimeCapabilityState([github, other] if multiple else [github])
    result = run(discovery, state, CapabilityDecision('EXTEND', ('github-actions', 'github-actions'), 'CI continuation', target_bundle_id='github-review'))
    assert len(result.resulting_state.active_bundles) == len(state.active_bundles)
    assert result.resulting_state.active_bundles[0].skill_ids == ('github', 'git-read', 'github-actions')
    if multiple:
        assert result.resulting_state.active_bundles[1] is other
    assert state.active_bundles[0] == github


@pytest.mark.parametrize('decision', [
    CapabilityDecision('DIRECT', ('invented',), 'bad'),
    CapabilityDecision('DIRECT', ('hidden',), 'known but not candidate'),
    CapabilityDecision('EXTEND', ('github-actions',), 'bad', target_bundle_id='missing'),
    CapabilityDecision('REUSE', ('github-actions',), 'forbidden'),
])
def test_invalid_stub_cannot_mutate_state(discovery, decision):
    state = RuntimeCapabilityState()
    with pytest.raises(ValueError):
        run(discovery, state, decision)
    assert state == RuntimeCapabilityState()


def test_repair_once_and_fail_explicitly(discovery):
    responses = iter(['not json', '{"action":"DIRECT","skill_ids":["pdf"],"reason":"one task"}'])
    calls = []
    def complete(prompt):
        calls.append(prompt)
        return next(responses)
    loader = RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(complete))
    result = loader.load_capability('extract PDF text', RuntimeCapabilityState())
    assert len(result.resolver_result.attempts) == len(calls) == 2
    assert 'Repair your JSON once' in calls[1]
    with pytest.raises(ResolverError) as failure:
        RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(lambda _: '{}')).load_capability('PDF', RuntimeCapabilityState())
    assert len(failure.value.attempts) == 2


def test_provider_failures_are_not_schema_retries(discovery):
    calls = []
    def complete(prompt):
        calls.append(prompt)
        raise RuntimeError('provider unavailable')
    with pytest.raises(RuntimeError, match='provider unavailable'):
        RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(complete)).load_capability('PDF', RuntimeCapabilityState())
    assert len(calls) == 1


def test_client_side_json_rejection_can_repair(discovery):
    count = 0
    def complete(prompt):
        nonlocal count
        count += 1
        if count == 1:
            raise ValueError('client rejected malformed JSON')
        return '{"action":"DIRECT","skill_ids":["pdf"],"reason":"one task"}'
    result = RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(complete)).load_capability('PDF', RuntimeCapabilityState())
    assert len(result.resolver_result.attempts) == 2


def test_prompt_only_contains_narrowed_context(discovery):
    state = RuntimeCapabilityState([
        ActiveBundle('review', 'GitHub review', ('github',)),
        ActiveBundle('private-mail', 'mail', ('email',)),
    ])
    skills = discovery.discover_skills('GitHub Actions', k=2)
    prompt = build_resolver_prompt('GitHub Actions', skills, state)
    assert 'private-mail' in prompt and 'review' in prompt
    assert 'unrelated astronomy' not in prompt
    assert 'one OR MORE' in prompt
    assert 'fixed coverage threshold' in prompt


@pytest.mark.parametrize('raw', [
    '{"action":"DIRECT","action":"DIRECT","skill_ids":["pdf"],"reason":"x"}',
    '{"action":"DIRECT","skill_ids":["pdf"],"reason":"x","purpose":"extra"}',
    '{"action":"DIRECT","skill_ids":[],"reason":"x"}',
    '{"action":"CREATE","skill_ids":["pdf"],"reason":"x","purpose":""}',
    '{"action":"DIRECT","skill_ids":[1],"reason":"x"}',
])
def test_strict_schema(discovery, raw):
    with pytest.raises(ValueError):
        validate_decision(raw, discovery.discover_skills('PDF'), RuntimeCapabilityState())


def test_existing_bundle_without_lexical_match_can_be_extended(discovery):
    state = RuntimeCapabilityState([ActiveBundle('email', 'mail', ('email',))])
    raw = '{"action":"EXTEND","target_bundle_id":"email","skill_ids":["pdf"],"reason":"x"}'
    decision = validate_decision(raw, discovery.discover_skills('PDF'), state)
    result = run(discovery, state, decision, 'PDF')
    assert result.resulting_state.active_bundles[0].skill_ids == ('email', 'pdf')


def test_extend_requires_new_skill(discovery):
    state = RuntimeCapabilityState([ActiveBundle('review', 'GitHub review', ('github-actions',))])
    with pytest.raises(ValueError, match='new skill'):
        validate_decision(json.dumps({'action':'EXTEND','target_bundle_id':'review',
            'skill_ids':['github-actions'],'reason':'x'}), discovery.discover_skills('GitHub Actions'), state)


def test_default_resolver_reuses_existing_client(monkeypatch):
    import skill_control_plane.evals.legacy.resolver_capability_loading as runtime
    client = lambda prompt: '{}'
    monkeypatch.setattr(runtime, 'BigModelChatClient', lambda: client)
    assert runtime.LLMCapabilityResolver().complete is client


def test_create_can_coexist_with_existing_bundle(discovery):
    github = ActiveBundle('review', 'GitHub review', ('github',))
    state = RuntimeCapabilityState([github], {'pdf'})
    result = run(discovery, state, CapabilityDecision('CREATE', ('email',), 'new ongoing context', purpose='Handle email'), 'email replies')
    assert len(result.resulting_state.active_bundles) == 2
    assert result.resulting_state.active_bundles[0] is github
    assert result.resulting_state.direct_skills == {'pdf'}
    assert state.active_bundles == [github]


def context_data(harness):
    return json.loads(harness.render_context())


def test_harness_stable_compact_context_without_library(discovery, monkeypatch):
    from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
    bundles = [ActiveBundle('z', 'Mail operations', ('email', 'pdf')),
               ActiveBundle('a', 'Code review', ('github',))]
    loader = RuntimeCapabilityLoader(discovery, StubResolver(None))
    def unexpected_search(*args, **kwargs):
        pytest.fail('rendering must not perform retrieval')
    monkeypatch.setattr(discovery, 'discover_skills', unexpected_search)
    harness = RuntimeCapabilityHarness(loader, RuntimeCapabilityState(bundles, {'pdf', 'slides'}))
    reordered = RuntimeCapabilityHarness(loader, RuntimeCapabilityState([
        bundles[1], ActiveBundle('z', 'Mail operations', ('pdf', 'email')),
    ], {'slides', 'pdf'}))
    assert harness.render_context() == reordered.render_context()
    data = context_data(harness)
    assert data['direct_skills'] == ['pdf', 'slides']
    assert data['maintained_bundles'] == [
        {'bundle_id': 'a', 'purpose': 'Code review', 'skill_ids': ['github'],
         'members': [{'skill_id': 'github', 'body_state': 'evicted'}]},
        {'bundle_id': 'z', 'purpose': 'Mail operations', 'skill_ids': ['email', 'pdf'],
         'members': [{'skill_id': 'email', 'body_state': 'evicted'},
                     {'skill_id': 'pdf', 'body_state': 'evicted'}]},
    ]
    assert 'tools' not in data
    assert 'load_capability(need)' not in harness.render_context()
    assert 'hidden' not in harness.render_context()
    assert 'unrelated astronomy' not in harness.render_context()
    assert 'GitHub pull request review' not in harness.render_context()


def test_harness_turn_to_turn_direct_create_extend(discovery):
    from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
    prompts = []
    def complete(prompt):
        prompts.append(json.loads(prompt.split('Input:\n', 1)[1]))
        if len(prompts) == 1:
            return json.dumps({'action': 'DIRECT', 'skill_ids': ['pdf', 'slides'], 'reason': 'one-off'})
        if len(prompts) == 2:
            return json.dumps({'action': 'CREATE', 'skill_ids': ['github'], 'reason': 'ongoing', 'purpose': 'Review code'})
        return json.dumps({'action': 'EXTEND', 'skill_ids': ['github-actions'],
                           'reason': 'check CI', 'target_bundle_id': prompts[-1]['maintained_bundles'][0]['bundle_id']})
    harness = RuntimeCapabilityHarness(RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(complete)))
    assert context_data(harness)['direct_skills'] == []
    assert context_data(harness)['maintained_bundles'] == []
    harness.load_capability('extract PDF text and presentation slides')
    assert context_data(harness)['direct_skills'] == ['pdf', 'slides']
    assert context_data(harness)['maintained_bundles'] == []
    created = harness.load_capability('GitHub pull request review')
    bundle_id = created.affected_bundle_id
    assert context_data(harness)['maintained_bundles'] == [
        {'bundle_id': bundle_id, 'purpose': 'Review code', 'skill_ids': ['github'],
         'members': [{'skill_id': 'github', 'body_state': 'evicted'}]}]
    harness.load_capability('GitHub Actions CI failure')
    assert context_data(harness)['maintained_bundles'][0]['skill_ids'] == ['github', 'github-actions']
    assert context_data(harness)['direct_skills'] == ['pdf', 'slides']
    assert created.resulting_state.active_bundles[0].skill_ids == ('github',)
    assert prompts[-1]['maintained_bundles'][0]['bundle_id'] == bundle_id


def test_resolver_receives_all_maintained_bundles_and_skill_evidence(discovery, monkeypatch):
    state = RuntimeCapabilityState([
        ActiveBundle(f'mail-{i}', f'Mail workflow {i}', ('email',)) for i in range(6)
    ])
    calls = []
    original = discovery.discover_skills
    def search(need, k):
        calls.append((need, k))
        return original(need, k)
    monkeypatch.setattr(discovery, 'discover_skills', search)
    def complete(prompt):
        data = json.loads(prompt.split('Input:\n', 1)[1])
        assert data['need'] == 'extract PDF text'
        assert len(data['maintained_bundles']) == 6
        assert data['maintained_bundles'][-1]['bundle_id'] == 'mail-5'
        assert data['representations']['pdf'] == discovery.texts['pdf']
        assert all('evidence' in c for c in data['skill_candidates'])
        return json.dumps({'action': 'EXTEND', 'target_bundle_id': 'mail-5',
                           'skill_ids': ['pdf'], 'reason': 'Process attachments'})
    result = RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(complete)).load_capability('extract PDF text', state)
    assert calls == [('extract PDF text', 10)]
    assert result.maintained_bundles == tuple(state.active_bundles)
    assert result.resulting_state.active_bundles[-1].skill_ids == ('email', 'pdf')


@pytest.mark.parametrize('failure', ['validation', 'provider', 'empty', 'blank'])
def test_harness_failure_preserves_state_and_context(discovery, failure):
    from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
    def complete(prompt):
        if failure == 'provider':
            raise RuntimeError('provider unavailable')
        return '{}'
    initial = RuntimeCapabilityState([ActiveBundle('mail', 'Manage mail', ('email',))], {'slides'})
    harness = RuntimeCapabilityHarness(RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(complete)), initial)
    before = harness.render_context()
    need = {'empty': 'zzzznonexistent', 'blank': ' '}.get(failure, 'PDF')
    with pytest.raises((ValueError, RuntimeError)):
        harness.load_capability(need)
    assert harness.state is initial
    assert harness.render_context() == before
    assert initial == RuntimeCapabilityState([ActiveBundle('mail', 'Manage mail', ('email',))], {'slides'})
