"""Live four-turn Bundle behavior observation; source .env before running."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

from experimental_skill_agent_e2e import ROOT, MANIFEST, CARDS, audit_context_flow
from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.evals.bundle_behavior import run_experiment
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.bigmodel import BigModelDenseRetriever, BigModelEmbeddingClient
from skill_control_plane.discovery.cards import load_retrieval_cards
from skill_control_plane.discovery.discovery import SkillDiscovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-steps', type=int, default=8)
    args = parser.parse_args()
    base = Path('local_artifacts/bundle-behavior')
    base.mkdir(parents=True, exist_ok=True)
    path = Path(mkdtemp(prefix='live-', dir=base)) / 'trace.json'
    metadata = {'version': 'v0.3-native-tools', 'started': datetime.now(timezone.utc).isoformat(),
                'live_api': True, 'retrieval': 'BM25 + Dense + RRF', 'dense_fallback': False}
    report = {'status': 'initializing'}
    logged_turns = set()

    def save(current):
        nonlocal report
        report = current
        path.write_text(json.dumps({**metadata, **current}, ensure_ascii=False, indent=2) + '\n')
        for turn in current.get('turns', []):
            if turn['status'] != 'running' and turn['turn'] not in logged_turns:
                logged_turns.add(turn['turn'])
                print(json.dumps({'turn': turn['turn'], 'status': turn['status'],
                                  'actions': turn['model_actions'],
                                  'state_after': turn['state_after']}, ensure_ascii=False), flush=True)

    try:
        _validate_root_snapshot(str(ROOT), str(MANIFEST))
        cards = load_retrieval_cards(CARDS)
        registry = SkillRegistry.from_tree(ROOT)
        chat = BigModelChatClient(timeout=60, max_tokens=8192)
        embedding = BigModelEmbeddingClient(timeout=45)
        metadata.update(model=chat.model, dense_model=embedding.model,
                        dense_dimensions=embedding.dimensions, skill_count=len(registry),
                        snapshot_id=json.loads(MANIFEST.read_text())['snapshot_id'])
        save(report)
        print('Building real Dense index...', flush=True)
        discovery = SkillDiscovery(
            registry, dense_factory=lambda records: BigModelDenseRetriever(
                records, model_name=embedding.model, dimensions=embedding.dimensions,
                embed_batch=embedding), retrieval_cards=cards)
        report = run_experiment(discovery, chat, max_steps=args.max_steps, checkpoint=save)
        report['context_audit'] = audit_context_flow(report['model_calls'], discovery.records)
        report['context_audit_status'] = 'passed'
    except Exception as exc:
        report.update(status='error', error_type=type(exc).__name__)
    finally:
        save(report)
        print('Trace:', path, flush=True)
        print('Status:', report['status'], 'Expectations:', report.get('matches_all_expectations'), flush=True)
    # Behavioral mismatches are preserved observations; transport/validation failure is a run error.
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
