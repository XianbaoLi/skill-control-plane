"""Real local acceptance; never inject candidates or regenerate cards.

Run: PYTHONPATH=src .venv/bin/python scripts/capability_loading_real_e2e.py
Uses existing BigModel and embedding cache; outputs local full evidence JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from robustness_current87 import CachedEmbeddings
from skill_control_plane import Bundle, BundleRegistry, CapabilityLoader, SkillDiscovery
from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.bigmodel import BigModelDenseRetriever
from skill_control_plane.discovery.cards import load_retrieval_cards

FIXTURE = Path('fixtures/capability_loading/hermes87-real-e2e.json')
OUTPUT = Path('local_artifacts/v0.7/real-e2e')


def load_environment():
    # Read configuration as data, without shell evaluation or printing secrets.
    if Path('.env').exists():
        for line in Path('.env').read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                name, value = line.split('=', 1)
                if name.strip().startswith('BIGMODEL_'):
                    os.environ.setdefault(name.strip(), value.strip().strip('\"\''))


def trace(loader, case, k):
    result = loader.load_capability(case['need'], k=k)
    ids = tuple(c.skill_id for c in result.discovery.candidates)
    comparisons = []
    for bundle in loader.bundles.values():
        comparisons.append({
            'bundle_id': bundle.bundle_id,
            'purpose': bundle.purpose,
            'covered_skills': [s for s in ids if s in bundle.skill_ids],
            'missing_skills': [s for s in ids if s not in bundle.skill_ids],
            'purpose_compatible': ' '.join(case['need'].casefold().split()) == ' '.join(bundle.purpose.casefold().split()),
        })
    reasons = {
        'DIRECT': 'One returned candidate; resolver does not check semantic sufficiency.',
        'REUSE': 'Unique active validated purpose-compatible bundle covers every returned candidate.',
        'EXTEND': 'Unique active validated purpose-compatible bundle covers >=2/3 and lacks <=1 candidate.',
        'CREATE': 'No unique qualifying REUSE/EXTEND; see coverage, missing skills and purpose gates.',
    }
    return {'case_id': case['case_id'], 'expected': case['expected'], 'k': k,
            'need': case['need'], 'actual': result.decision,
            'expectation_met': result.decision == case['expected'],
            'decision_reason': reasons[result.decision], 'bundle_comparisons': comparisons,
            'result': asdict(result)}


def main():
    load_environment()
    fixture = json.loads(FIXTURE.read_text())
    _validate_root_snapshot(fixture['corpus'], fixture['manifest'])
    skills = SkillRegistry.from_tree(fixture['corpus'])
    cards = load_retrieval_cards(fixture['cards'])
    assert len(skills) == 87
    bundles = BundleRegistry(skills, [Bundle(**{**b, 'skill_ids': tuple(b['skill_ids'])}) for b in fixture['bundles']])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT / 'embeddings.json'
    if not cache_path.exists():
        cache_path.write_bytes(Path(fixture['embedding_cache']).read_bytes())
    embed = CachedEmbeddings(cache_path)
    discovery = SkillDiscovery(
        skills, dense_factory=lambda records: BigModelDenseRetriever(
            records, model_name='embedding-3', dimensions=2048, embed_batch=embed),
        source_k=10, retrieval_cards=cards)
    loader = CapabilityLoader(discovery, bundles)
    limitation = None
    try:
        rows = [trace(loader, case, fixture['k']) for case in fixture['cases']]
    except RuntimeError as exc:
        # Preserve provider error type/status only, never response bodies or credentials.
        cause = exc.__cause__
        limitation = f"Live Dense unavailable: {type(cause).__name__}, status={getattr(cause, 'code', None)}"
        loader = CapabilityLoader(
            SkillDiscovery(skills, retrieval_cards=cards), bundles)
        rows = [trace(loader, case, fixture['k']) for case in fixture['cases']]
    frozen = json.loads(Path('local_artifacts/v0.6/robustness-current87-13target-65query-queries.json').read_text())
    selected = {('ST-01', 'S3'), ('ST-01', 'S4'), ('AT-07', 'S2'), ('AT-02', 'S2')}
    cached_rows = []
    cached_loader = CapabilityLoader(discovery, bundles)
    for row in frozen['targets']:
        if (row['case_id'], row['stage_id']) in selected:
            query = row['original_query']
            assert query in embed.data['vectors']
            cached_rows.append(trace(cached_loader, {'case_id': row['case_id'] + '/' + row['stage_id'],
                                                     'expected': 'diagnostic', 'need': query}, fixture['k']))
    diagnostic = trace(loader, fixture['cases'][0], 1)
    compound_diagnostic = trace(loader, fixture['cases'][3], 1)
    report = {'corpus': fixture['corpus'], 'snapshot_id': json.loads(Path(fixture['manifest']).read_text())['snapshot_id'],
              'skill_count': len(skills), 'backend': rows[0]['result']['discovery']['backend'], 'live_dense_limitation': limitation,
              'cached_rrf_backend': 'BM25 + cached BigModel embedding-3/2048 + RRF(60)',
              'source_k': 10, 'input_hashes': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in [FIXTURE, fixture['manifest'], fixture['cards'], fixture['embedding_cache']]},
              'cases': rows, 'cached_rrf_diagnostics': cached_rows, 'top1_diagnostic_not_acceptance': diagnostic, 'compound_top1_diagnostic': compound_diagnostic}
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for row in [*rows, *cached_rows, diagnostic, compound_diagnostic]:
        print(json.dumps({k: v for k, v in row.items() if k != 'result'}, ensure_ascii=False))
        for c in row['result']['discovery']['candidates']:
            print(json.dumps(c, ensure_ascii=False))
    print('Full report:', OUTPUT / 'report.json')


if __name__ == '__main__':
    main()
