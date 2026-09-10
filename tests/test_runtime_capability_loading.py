import json

import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_loading import (
    ActiveBundle, CapabilityDecision, LLMCapabilityResolver, ResolverError,
    ResolverResult, RuntimeCapabilityLoader, RuntimeCapabilityState,
    build_resolver_prompt, discover_active_bundles, validate_decision,
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
    bundles = discover_active_bundles('GitHub Actions', state, discovery, n=1)
    assert bundles[0].bundle_id == 'review'
    prompt = build_resolver_prompt('GitHub Actions', skills, bundles)
    assert 'private-mail' not in prompt and 'unrelated astronomy' not in prompt
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
        validate_decision(raw, discovery.discover_skills('PDF'), (), RuntimeCapabilityState())


def test_existing_but_non_candidate_bundle_rejected(discovery):
    state = RuntimeCapabilityState([ActiveBundle('email', 'mail', ('email',))])
    raw = '{"action":"EXTEND","target_bundle_id":"email","skill_ids":["pdf"],"reason":"x"}'
    with pytest.raises(ValueError, match='outside supplied'):
        validate_decision(raw, discovery.discover_skills('PDF'), (), state)


def test_extend_requires_new_skill(discovery):
    state = RuntimeCapabilityState([ActiveBundle('review', 'GitHub review', ('github-actions',))])
    bundles = discover_active_bundles('GitHub Actions', state, discovery)
    with pytest.raises(ValueError, match='new skill'):
        validate_decision(json.dumps({'action':'EXTEND','target_bundle_id':'review',
            'skill_ids':['github-actions'],'reason':'x'}), discovery.discover_skills('GitHub Actions'), bundles, state)


def test_default_resolver_reuses_existing_client(monkeypatch):
    import skill_control_plane.runtime.capability_loading as runtime
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
