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
    '请按本环境已有的文档处理操作规范，为这次一次性需求制定具体操作方案：'
    '从扫描版 PDF 提取文字和表格，再制作一份可编辑的演示文稿。'
    '需要说明所用工具、必要依赖、关键命令和验证步骤；请依据实际可用的操作说明，'
    '不要凭空猜命令。当前没有输入文件，不需要执行命令或生成文件。'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=DEFAULT_TASK)
    parser.add_argument('--max-steps', type=int, default=8)
    args = parser.parse_args()
    base = Path('local_artifacts/experimental-skill-agent')
    base.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=base)) / 'trace.json'
    report = {'started': datetime.now(timezone.utc).isoformat(), 'task': args.task,
              'live_api': True, 'retrieval': 'BM25 + Dense + RRF',
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
        injected_later = any(
            set(applied['selected_skill_ids']) <= set(later['injected_skill_ids'])
            for applied in applications for later in agent.trace if later['step'] > applied['step'])
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
