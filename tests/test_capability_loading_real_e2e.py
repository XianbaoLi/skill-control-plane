"""Local real-corpus integration checks, not a claim of semantic acceptance.

No network and no mocked rankings: Dense reuses recorded provider vectors.
Missing untracked experiment assets skip these tests in portable CI.
"""
import json
from pathlib import Path

import pytest

from skill_control_plane import Bundle, BundleRegistry, CapabilityLoader, SkillDiscovery
from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.bigmodel import BigModelDenseRetriever
from skill_control_plane.discovery.cards import load_retrieval_cards

FIXTURE = Path(__file__).parents[1] / 'fixtures/capability_loading/hermes87-real-e2e.json'


@pytest.fixture
def real_inputs():
    data = json.loads(FIXTURE.read_text())
    if not all(Path(data[k]).exists() for k in ('corpus', 'manifest', 'cards', 'embedding_cache')):
        pytest.skip('Frozen Hermes experiment assets are local and not distributed')
    _validate_root_snapshot(data['corpus'], data['manifest'])
    assert json.loads(Path(data['manifest']).read_text())['snapshot_id'] == 'f670e9d5ecdbb299a5c5ac4fcc48ece35e0adadcfb7e24b96850a4ec5379e509'
    skills = SkillRegistry.from_tree(data['corpus'])
    cards = load_retrieval_cards(data['cards'])
    assert len(skills) == 87
    bundles = BundleRegistry(skills, [Bundle(**{**b, 'skill_ids': tuple(b['skill_ids'])}) for b in data['bundles']])
    return data, skills, cards, bundles


@pytest.mark.parametrize('index,top', [(0, 'arxiv'), (1, 'github-code-review'),
                                       (2, 'systematic-debugging'), (3, 'ocr-and-documents')])
def test_real_discovery_reaches_resolver_without_candidate_injection(real_inputs, index, top):
    data, skills, cards, bundles = real_inputs
    loader = CapabilityLoader(
        SkillDiscovery(skills, retrieval_cards=cards), bundles)
    before = bundles.values()
    result = loader.load_capability(data['cases'][index]['need'])
    assert result.discovery.candidates[0].skill_id == top
    assert result.decision == 'CREATE'  # Observed limitation, NOT the desired semantics.
    assert len(result.discovery.candidates) == 5
    assert set(result.resolution.skill_ids) == {c.skill_id for c in result.discovery.candidates}
    assert all(loader.discovery.texts[c.skill_id] for c in result.discovery.candidates)
    assert bundles.values() == before


def test_real_top1_direct_does_not_establish_sufficiency(real_inputs):
    data, skills, cards, bundles = real_inputs
    loader = CapabilityLoader(
        SkillDiscovery(skills, retrieval_cards=cards), bundles)
    # A multi-capability request also becomes DIRECT solely by setting k=1.
    need = data['cases'][3]['need']
    top1 = loader.load_capability(need, k=1)
    assert top1.decision == 'DIRECT'
    assert top1.discovery.truncated
    assert top1.skills[0].skill_id == 'ocr-and-documents'
    assert loader.load_capability(need, k=5).decision == 'CREATE'


def test_real_cached_dense_rrf_preserves_historical_target_ranks(real_inputs):
    data, skills, cards, bundles = real_inputs
    report_path = Path('local_artifacts/v0.6/robustness-current87-13target-65query-full.json')
    if not report_path.exists():
        pytest.skip('Historical ranking report unavailable')
    cache = json.loads(Path(data['embedding_cache']).read_text())
    assert (cache['model'], cache['dimensions']) == ('embedding-3', 2048)

    def recorded_vectors(texts):
        # Exact text lookup fails on a missing input; never returns synthetic vectors.
        return [cache['vectors'][text] for text in texts]

    discovery = SkillDiscovery(
        skills, dense_factory=lambda records: BigModelDenseRetriever(
            records, embed_batch=recorded_vectors), source_k=10,
        retrieval_cards=cards)
    loader = CapabilityLoader(discovery, bundles)
    selected = {('ST-01', 'S3'), ('ST-01', 'S4'), ('AT-07', 'S2'), ('AT-02', 'S2')}
    checked = 0
    for stage in json.loads(report_path.read_text())['stages']:
        if (stage['case_id'], stage['stage_id']) not in selected:
            continue
        variant = stage['variants'][0]
        result = loader.load_capability(variant['query'])
        assert result.discovery.backend == 'rrf'
        assert any(set(c.source_scores) == {'bm25', 'dense'} for c in result.discovery.candidates)
        actual = {c.skill_id: c.rank for c in result.discovery.candidates}
        for target, rank in variant['rrf']['new_required_ranks'].items():
            assert actual.get(target) == (rank if rank is not None and rank <= 5 else None)
        checked += 1
    assert checked == 4
