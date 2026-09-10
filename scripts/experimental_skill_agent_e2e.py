"""Opt-in live single-agent dynamic Skill loading. Source .env before running."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.bigmodel import BigModelDenseRetriever, BigModelEmbeddingClient
from skill_control_plane.retrieval.cards import load_retrieval_cards
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.experimental_agent import ExperimentalSkillAgent

ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
DEFAULT_TASK = (
    '请按本环境已有的文档处理操作规范，为团队建立持续复用的文档处理工作流：'
    '从扫描版 PDF 提取文字和表格，再制作一份可编辑的演示文稿。'
    '需要说明所用工具、必要依赖、关键命令和验证步骤；请依据实际可用的操作说明，'
    '不要凭空猜命令。当前没有输入文件，不需要执行命令或生成文件。'
)


def audit_context_flow(calls, records):
    """Check actual wire messages, independently of the agent's trace counters."""
    audits = []
    previous_history = []
    for number, call in enumerate(calls, 1):
        messages = call['messages']
        assert messages[0]['role'] == 'system'
        assert all(m['role'] != 'system' for m in messages[1:])
        system = messages[0]['content']
        surface = json.loads(system.split('Runtime Bundles (metadata only)\n', 1)[1])
        assert set(surface) == {'maintained_bundles'}
        for bundle in surface['maintained_bundles']:
            assert set(bundle) == {'bundle_id', 'purpose', 'members'}
            for member in bundle['members']:
                assert set(member) == {'skill_id', 'name', 'short_description'}
                record = records[member['skill_id']]
                assert member['name'] == record.name
                assert member['short_description'] == ' '.join(record.description.split())[:240]
        for record in records.values():
            if record.body:
                assert record.body not in system
                assert json.dumps(record.body, ensure_ascii=False)[1:-1] not in system
        history = messages[1:]
        assert history[:len(previous_history)] == previous_history
        if number > 1:
            assert history[len(previous_history)] == {
                'role': 'assistant', 'content': calls[number - 2]['response']}
        previous_history = history
        bodies, results = [], []
        for message in history:
            if message['role'] != 'user':
                continue
            try:
                envelope = json.loads(message['content'])
            except ValueError:
                continue
            if not isinstance(envelope, dict) or envelope.get('type') != 'capability_tool_result':
                continue
            result = envelope['result']
            results.append(envelope['tool'])
            if envelope['tool'] == 'apply_capability':
                for body in result['skill_bodies']:
                    skill_id = body['skill_id']
                    assert skill_id in result['selected_skill_ids']
                    assert skill_id not in bodies
                    assert body['body'] == records[skill_id].body
                    bodies.append(skill_id)
            else:
                assert envelope['tool'] == 'load_capability'
                assert result['backend'] == 'rrf'
                assert 'skill_bodies' not in result
        for record in records.values():
            if record.body and record.skill_id not in bodies:
                assert record.body not in '\n'.join(m['content'] for m in history)
                assert json.dumps(record.body, ensure_ascii=False)[1:-1] not in '\n'.join(m['content'] for m in history)
        audits.append({'call': number, 'message_count': len(messages),
                       'system_bundle_ids': [b['bundle_id'] for b in surface['maintained_bundles']],
                       'system_skill_body_ids': [], 'history_skill_body_ids': bodies,
                       'history_tool_results': results})
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=DEFAULT_TASK)
    parser.add_argument('--max-steps', type=int, default=8)
    args = parser.parse_args()
    base = Path('local_artifacts/experimental-skill-agent')
    base.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=base)) / 'trace.json'
    report = {'started': datetime.now(timezone.utc).isoformat(), 'task': args.task,
              'version': 'v0.2', 'live_api': True, 'retrieval': 'BM25 + Dense + RRF',
              'dense_fallback': False, 'model_calls': [], 'trace': []}
    agent = None

    def save():
        if agent is not None:
            report['trace'] = agent.trace
            report['history'] = agent.history
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    class RecordingClient:
        def __init__(self, client):
            self.client = client

        def complete_messages(self, messages):
            call = {'messages': deepcopy(messages)}
            report['model_calls'].append(call)
            save()
            response = self.client.complete_messages(messages)
            call['response'] = response
            save()
            print(json.dumps({'step': len(report['model_calls']),
                              'response': response}, ensure_ascii=False), flush=True)
            return response

    try:
        _validate_root_snapshot(str(ROOT), str(MANIFEST))
        registry = SkillRegistry.from_tree(ROOT, cards=load_retrieval_cards(CARDS))
        report['skill_count'] = len(registry)
        report['snapshot_id'] = json.loads(MANIFEST.read_text())['snapshot_id']
        chat = BigModelChatClient(timeout=60, max_tokens=8192)
        embedding = BigModelEmbeddingClient(timeout=45)
        report.update(model=chat.model, dense_model=embedding.model,
                      dense_dimensions=embedding.dimensions)
        save()
        print('Building real Dense index...', flush=True)
        discovery = SkillDiscovery(registry, dense_factory=lambda records: BigModelDenseRetriever(
            records, model_name=embedding.model, dimensions=embedding.dimensions,
            embed_batch=embedding))
        harness = RuntimeCapabilityHarness(discovery=discovery)
        agent = ExperimentalSkillAgent(harness, RecordingClient(chat), max_steps=args.max_steps)
        report['final'] = agent.run(args.task)
        # Acceptance measures observed actions; it does not prescribe model decisions.
        search_steps = [r for r in agent.trace if r['model_action']['type'] == 'load_capability']
        applications = [r for r in agent.trace if r['action'] in {'DIRECT', 'EXTEND', 'CREATE'}]
        report['context_audit'] = audit_context_flow(report['model_calls'], discovery.records)
        injected_later = any(
            set(applied['selected_skill_ids']) <= set(later['history_skill_body_ids'])
            for applied in applications for later in report['context_audit'] if later['call'] > applied['step'])
        report['status'] = 'success' if search_steps and applications and injected_later else 'incomplete_capability_cycle'
    except Exception as exc:
        chain, current = [], exc
        while current is not None:
            chain.append({'type': type(current).__name__, 'http_status': getattr(current, 'code', None)})
            current = current.__cause__
        report.update(status='error', error=chain)
    finally:
        save()
        print('Trace:', output, flush=True)
        print('Status:', report.get('status'), flush=True)
    if report.get('status') != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
