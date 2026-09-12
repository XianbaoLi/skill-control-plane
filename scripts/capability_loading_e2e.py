"""Live resolver acceptance. Source .env before running; no replayed decisions."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.bigmodel import (
    BigModelDenseRetriever,
    BigModelEmbeddingClient,
)
from skill_control_plane.discovery.cards import load_retrieval_cards
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.evals.legacy.resolver_capability_loading import (
    ActiveBundle, LLMCapabilityResolver, ResolverError, RuntimeCapabilityLoader,
    RuntimeCapabilityState,
)

ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
CASES = (
    ('DIRECT', 'Read this PDF, extract the useful text, and turn it into a presentation for this one task.'),
    ('EXTEND', 'Inspect why the GitHub Actions checks for this pull request are failing.'),
    ('CREATE', 'Create and maintain an ongoing email operations capability that repeatedly searches my inbox, triages relevant messages, and prepares replies.'),
)


def safe_error(exc):
    chain, current = [], exc
    while current is not None:
        chain.append({'type': type(current).__name__, 'http_status': getattr(current, 'code', None)})
        current = current.__cause__
    return chain


def write(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                               default=lambda value: sorted(value) if isinstance(value, set) else str(value)) + '\n')


def main():
    _validate_root_snapshot(str(ROOT), str(MANIFEST))
    skills = SkillRegistry.from_tree(ROOT)
    cards = load_retrieval_cards(CARDS)
    assert len(skills) == 87
    base = Path('local_artifacts/runtime-capability-loading')
    base.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=base))
    report = {'started': datetime.now(timezone.utc).isoformat(), 'live_api': True,
              'snapshot_id': json.loads(MANIFEST.read_text())['snapshot_id'],
              'skill_count': len(skills), 'cases': []}
    client = BigModelChatClient(timeout=45, max_tokens=4096)
    report['model'] = client.model
    embedding_client = BigModelEmbeddingClient()
    discovery = SkillDiscovery(
        skills,
        dense_factory=lambda records: BigModelDenseRetriever(
            records,
            model_name=embedding_client.model,
            dimensions=embedding_client.dimensions,
            embed_batch=embedding_client,
        ),
        retrieval_cards=cards,
    )
    # Preflight all query vectors; provider failure aborts instead of falling back.
    for _, need in CASES:
        discovery.discover_skills(need, k=10)
    report['retrieval'] = 'BM25 + Paratera GLM-Embedding-3 Dense + RRF'
    report['dense_model'] = embedding_client.model
    report['dense_dimensions'] = embedding_client.dimensions
    report['dense_fallback'] = False
    for intended, need in CASES:
        state = RuntimeCapabilityState([] if intended == 'DIRECT' else [
            ActiveBundle('github-review', 'Review and inspect GitHub pull requests',
                         ('github-code-review', 'github-auth'))])
        preview = discovery.discover_skills(need, k=10)
        row = {'intended': intended, 'need': need, 'initial_state': asdict(state),
               'skill_candidates': asdict(preview), 'maintained_bundles': [asdict(b) for b in state.active_bundles],
               'api_calls': []}
        def complete(prompt):
            call = {'prompt': prompt}
            row['api_calls'].append(call)
            try:
                response = client(prompt)
                call['adapter_response'] = response  # Existing adapter canonicalizes provider JSON.
                return response
            except Exception as exc:
                call['error'] = safe_error(exc)
                raise
        try:
            result = RuntimeCapabilityLoader(discovery, LLMCapabilityResolver(complete)).load_capability(need, state)
            row['result'] = asdict(result)
            actual = result.resolver_result.decision.action
            row['actual'] = actual
            row['status'] = 'success' if actual == intended else 'decision_mismatch'
        except Exception as exc:
            row['status'] = 'resolver_error' if isinstance(exc, ResolverError) else 'provider_error'
            row['error'] = safe_error(exc)
            if isinstance(exc, ResolverError):
                row['attempts'] = [asdict(a) for a in exc.attempts]
            row['resulting_state'] = asdict(state)
        report['cases'].append(row)
        write(output / 'report.json', report)
        print(json.dumps({'case': intended, 'status': row['status'],
                          'decision': row.get('result', {}).get('resolver_result'),
                          'error': row.get('error')}, ensure_ascii=False), flush=True)
    print('Report:', output / 'report.json', flush=True)
    if any(row['status'] != 'success' for row in report['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
