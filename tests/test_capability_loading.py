import json

import pytest

from skill_control_plane.capability_loading import (
    CapabilityLoader, Decision, load_capability, resolve_bundle,
)
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.registry.bundles import Bundle, BundleRegistry
from skill_control_plane.retrieval.cards import RetrievalCard, apply_retrieval_cards
from skill_control_plane.retrieval.dense import DenseRetriever, metadata_text
from skill_control_plane.retrieval.discovery import SkillDiscovery, discover_skills


def registry():
    return SkillRegistry([
        SkillRecord('a', 'a', 'alpha delta', 'original a', '/a'),
        SkillRecord('b', 'b', 'beta delta', 'original b', '/b'),
        SkillRecord('c', 'c', 'gamma', 'original c', '/c'),
        SkillRecord('d', 'd', 'omega', 'original d', '/d'),
    ])


@pytest.mark.parametrize('need,bundle,decision,add', [
    ('alpha', None, Decision.DIRECT, ()),
    ('delta', Bundle('ab', 'delta', ('a', 'b'), validated=True), Decision.REUSE, ()),
    ('delta gamma', Bundle('ab', 'delta gamma', ('a', 'b'), validated=True), Decision.EXTEND, ('c',)),
    ('gamma omega', Bundle('ab', 'delta', ('a', 'b'), validated=True), Decision.CREATE, ()),
])
def test_acceptance(need, bundle, decision, add):
    skills = registry()
    bundles = BundleRegistry(skills, [bundle] if bundle else [])
    before = bundles.values()
    loader = CapabilityLoader(SkillDiscovery(skills), bundles)
    result = load_capability(need, loader=loader)
    assert result.decision == decision
    assert result.resolution.add_skills == add
    assert result.resolution.existing_bundle == (bundle if decision in {Decision.REUSE, Decision.EXTEND} else None)
    assert {s.skill_id for s in result.skills} == set(result.resolution.skill_ids)
    assert all(s.body.startswith('original') for s in result.skills)
    assert bundles.values() == before
    print(json.dumps(dict(need=need, decision=result.decision,
                          bundle_id=result.resolution.existing_bundle.bundle_id if result.resolution.existing_bundle else None,
                          skill_ids=result.resolution.skill_ids, add_skills=add)))


def test_registration_cards_and_hash_validation(tmp_path):
    path = tmp_path / 'a'
    path.mkdir()
    (path / 'SKILL.md').write_text('---\nname: a\ndescription: alpha\n---\nOriginal body')
    raw = SkillRegistry.from_tree(tmp_path)
    skill = raw.get('a')
    card = RetrievalCard('a', skill.content_hash, 'convert quasar', ('quasar input',), ('quasar',), ('quasar',))
    cards = {'a': card}
    registered = SkillRegistry.from_tree(tmp_path, cards=cards)
    assert registered.get('a').body == 'Original body'
    assert registered.get('a').retrieval_representation == metadata_text(apply_retrieval_cards(raw, cards)[0])
    discovery = SkillDiscovery(registered)
    (path / 'SKILL.md').write_text('changed')
    for _ in range(2):
        result = discover_skills('quasar', discovery=discovery)
        assert result.candidates[0].skill_id == 'a'
        assert 'quasar' in result.representations[0][1]
        assert result.candidates[0].evidence
    with pytest.raises(ValueError, match='corpus mismatch'):
        SkillRegistry.from_tree(tmp_path, cards=cards)


def test_shared_membership_and_ambiguity():
    skills = registry()
    bundles = BundleRegistry(skills, [Bundle(bid, 'delta', ('a', 'b'), validated=True) for bid in ('x', 'y')])
    assert bundles.skill_to_bundles['a'] == {'x', 'y'}
    with pytest.raises(TypeError):
        bundles.skill_to_bundles['a'] = frozenset()
    result = resolve_bundle('delta', ('a', 'b'), bundles)
    assert result.decision == Decision.CREATE
    assert result.ambiguity == ('x', 'y')
    assert len(bundles.values()) == 2


@pytest.mark.parametrize('purpose,status,validated', [
    ('different', 'active', True), ('delta', 'inactive', True), ('delta', 'active', False),
])
def test_ineligible_bundle(purpose, status, validated):
    bundles = BundleRegistry(registry(), [Bundle('ab', purpose, ('a', 'b'), status, validated)])
    result = resolve_bundle('delta', ('a', 'b'), bundles)
    assert result.decision == Decision.CREATE
    assert result.matches


def test_coverage_superset_and_extension_boundary():
    skills = registry()
    bundles = BundleRegistry(skills, [Bundle('abc', 'DELTA', ('a', 'b', 'c'), validated=True)])
    result = resolve_bundle(' delta ', ('b', 'a', 'a'), bundles)
    assert result.decision == Decision.REUSE
    assert result.skill_ids == ('a', 'b', 'c')
    bundles = BundleRegistry(skills, [Bundle('a', 'delta', ('a',), validated=True)])
    assert resolve_bundle('delta', ('a', 'b'), bundles).decision == Decision.CREATE
    assert resolve_bundle('delta', ('a', 'b', 'c'), bundles).decision == Decision.CREATE


def test_invalid_and_empty_inputs():
    skills = registry()
    bundles = BundleRegistry(skills)
    discovery = SkillDiscovery(skills)
    loader = CapabilityLoader(discovery, bundles)
    for query, k in [('', 5), ('alpha', 0)]:
        with pytest.raises(ValueError):
            loader.load_capability(query, k=k)
    with pytest.raises(ValueError, match='no skills'):
        loader.load_capability('nonexistent')
    with pytest.raises(ValueError, match='unknown'):
        bundles.add(Bundle('bad', 'bad', ('unknown',)))
    with pytest.raises(ValueError, match='unknown'):
        resolve_bundle('bad', ('unknown',), bundles)
    bundles.add(Bundle('a', 'alpha', ('a',)))
    with pytest.raises(ValueError, match='duplicate'):
        bundles.add(Bundle('a', 'alpha', ('a',)))
    with pytest.raises(ValueError, match='share a skill corpus'):
        CapabilityLoader(discovery, BundleRegistry(SkillRegistry()))
    result = loader.load_capability('delta', k=1)
    assert result.discovery.truncated
    assert any('truncated' in w for w in result.warnings)


def test_dense_rrf_uses_same_prepared_text_once():
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings=True):
            self.calls.append(texts)
            return [[1.0, 0.0] if 'alpha' in t else [0.0, 1.0] for t in texts]

    encoder = Encoder()
    discovery = SkillDiscovery(registry(), dense_factory=lambda records: DenseRetriever(records, model=encoder))
    result = discovery.discover_skills('alpha')
    assert result.backend == 'rrf'
    assert result.candidates[0].skill_id == 'a'
    assert set(result.candidates[0].source_scores) == {'bm25', 'dense'}
    discovery.discover_skills('alpha')
    assert len(encoder.calls) == 3  # one corpus encoding, two query encodings
    assert encoder.calls[0] == list(discovery.texts.values())
