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
from skill_control_plane.discovery.bigmodel import BigModelDenseRetriever, BigModelEmbeddingClient
from skill_control_plane.discovery.cards import load_retrieval_cards
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.integrations.reference_agent import ExperimentalSkillAgent

ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
DEFAULT_TASK = (
    '请按本环境已有的文档处理操作规范，为团队建立持续复用的文档处理工作流：'
    '从扫描版 PDF 提取文字和表格，再制作一份可编辑的演示文稿。'
    '需要说明所用工具、必要依赖、关键命令和验证步骤；请依据实际可用的操作说明，'
    '不要凭空猜命令。当前没有输入文件，不需要执行命令或生成文件。'
)


MULTI_SEARCH_TASK = (
    '请为一次性项目交接准备两部分操作方案：一是从扫描 PDF 中提取文字和表格并验证识别质量；'
    '二是排查 GitHub Python 项目中仅在 CI 出现的间歇性测试失败，涵盖读取 PR 差异与日志、'
    '复现、根因定位和修复验证。两部分都必须依据本环境实际可用的操作说明，不猜命令。'
    '本次请分开检索这两个明显不同的能力缺口：先调用一次 load_capability 搜索文档识别能力，'
    '读完结果后，再调用一次 load_capability 搜索 CI 调试能力；读完两次结果后，再用一次 '
    'apply_capability 联合选择支持两部分工作的 Skill。查询和候选选择由你决定。'
    '当前没有文件或外部访问权限，只给方案，不执行；回答控制在 800 字内。'
)


def audit_multi_search(agent):
    """Require a real sequential search/search/apply and exclusive selections.

    Pure observation after execution: no forced calls, IDs or runtime policy.
    """
    events = []
    for row in agent.trace:
        transitions = {t['tool_call_id']: t for t in row.get('pending_pool_transitions', [])}
        for event in row['tool_executions']:
            result = next(json.loads(m['content']) for m in agent.history
                          if m['role'] == 'tool' and m['tool_call_id'] == event['tool_call_id'])
            events.append({**event, 'step': row['step'], 'result': result,
                           'pool': transitions[event['tool_call_id']]})
    searches = [e for e in events if e['tool'] == 'load_capability' and 'candidates' in e['result']]
    applications = [e for e in events if e['tool'] == 'apply_capability' and 'error' not in e['result']]
    checks = {'two_successful_searches': len(searches) == 2,
              'one_successful_apply': len(applications) == 1,
              'no_protocol_or_tool_errors': not any(r.get('error') or r.get('tool_error') for r in agent.trace),
              'pool_empty_at_end': agent.harness.pending_candidates is None}
    if len(searches) == 2 and len(applications) == 1:
        first, second = [{c['skill_id'] for c in e['result']['candidates']} for e in searches]
        application = applications[0]
        selected = set(application['result']['selected_skill_ids'])
        checks.update(
            sequential_search_search_apply=searches[0]['step'] < searches[1]['step'] < application['step'],
            selected_first_exclusive=bool(selected & (first - second)),
            selected_second_exclusive=bool(selected & (second - first)),
            selection_within_union=selected <= first | second,
            second_search_preserves_first=set(searches[1]['pool']['after']) == first | second,
            apply_pool_before=set(application['pool']['before']) == first | second,
            apply_clears_pool=application['pool']['after'] == [])
    return {'checks': checks, 'passed': all(checks.values()), 'events': events}


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
            assert set(bundle) == {'bundle_id', 'purpose', 'capabilities', 'members'}
            assert len(bundle['capabilities']) <= 5
            assert all(isinstance(phrase, str) and len(phrase) <= 96
                       for phrase in bundle['capabilities'])
            for member in bundle['members']:
                assert set(member) == {'skill_id', 'name', 'body_state'}
                assert member['body_state'] in {'resident', 'evicted'}
                record = records[member['skill_id']]
                assert member['name'] == record.name
        for record in records.values():
            if record.body:
                assert record.body not in system
                assert json.dumps(record.body, ensure_ascii=False)[1:-1] not in system
        history = messages[1:]
        assert history[:len(previous_history)] == previous_history
        if number > 1:
            assert history[len(previous_history)] == calls[number - 2]['response']
        previous_history = history
        bodies, results = [], []
        pending = {}
        for message in history:
            if message['role'] == 'assistant':
                assert not pending
                for tool_call in message.get('tool_calls') or []:
                    assert tool_call['id'] not in pending
                    pending[tool_call['id']] = tool_call['function']['name']
                continue
            if message['role'] != 'tool':
                assert not pending
                continue
            name = pending.pop(message['tool_call_id'])
            result = json.loads(message['content'])
            results.append(name)
            if 'error' in result:
                continue
            if name == 'apply_capability':
                for body in result['skill_bodies']:
                    skill_id = body['skill_id']
                    assert skill_id in result['selected_skill_ids']
                    assert skill_id not in bodies
                    assert body['body'] == records[skill_id].body
                    bodies.append(skill_id)
            elif name == 'load_skill_body':
                if result['status'] == 'loaded':
                    assert result['body'] == records[result['skill_id']].body
                    bodies.append(result['skill_id'])
            else:
                assert name == 'load_capability'
                assert result['backend'] == 'rrf'
                assert 'skill_bodies' not in result
        assert not pending
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
    parser.add_argument('--multi-search', action='store_true', help='Run and audit the two-capability live case')
    args = parser.parse_args()
    if args.multi_search:
        args.task = MULTI_SEARCH_TASK
    base = Path('local_artifacts/experimental-skill-agent')
    base.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=base)) / 'trace.json'
    report = {'started': datetime.now(timezone.utc).isoformat(), 'task': args.task,
              'version': 'native-tools', 'live_api': True, 'retrieval': 'BM25 + Dense + RRF',
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

        def complete_messages(self, messages, *, tools):
            call = {'messages': deepcopy(messages), 'tools': deepcopy(tools)}
            report['model_calls'].append(call)
            save()
            response = self.client.complete_messages(messages, tools=tools)
            call['response'] = response
            save()
            print(json.dumps({'step': len(report['model_calls']),
                              'response': response}, ensure_ascii=False), flush=True)
            return response

    try:
        _validate_root_snapshot(str(ROOT), str(MANIFEST))
        cards = load_retrieval_cards(CARDS)
        registry = SkillRegistry.from_tree(ROOT)
        report['skill_count'] = len(registry)
        report['snapshot_id'] = json.loads(MANIFEST.read_text())['snapshot_id']
        chat = BigModelChatClient(timeout=60, max_tokens=8192)
        embedding = BigModelEmbeddingClient(timeout=45)
        report.update(model=chat.model, dense_model=embedding.model,
                      dense_dimensions=embedding.dimensions)
        save()
        print('Building real Dense index...', flush=True)
        discovery = SkillDiscovery(
            registry, dense_factory=lambda records: BigModelDenseRetriever(
                records, model_name=embedding.model, dimensions=embedding.dimensions,
                embed_batch=embedding), retrieval_cards=cards)
        harness = RuntimeCapabilityHarness(discovery=discovery)
        agent = ExperimentalSkillAgent(harness, RecordingClient(chat), max_steps=args.max_steps)
        report['final'] = agent.run(args.task)
        # Acceptance measures observed actions; it does not prescribe model decisions.
        search_steps = [r for r in agent.trace if any(
            e['tool'] == 'load_capability' for e in r['tool_executions'])]
        applications = [r for r in agent.trace if r['action'] in {'DIRECT', 'EXTEND', 'CREATE'}]
        report['context_audit'] = audit_context_flow(report['model_calls'], discovery.records)
        injected_later = any(
            set(applied['selected_skill_ids']) <= set(later['history_skill_body_ids'])
            for applied in applications for later in report['context_audit'] if later['call'] > applied['step'])
        report['status'] = 'success' if search_steps and applications and injected_later else 'incomplete_capability_cycle'
        if args.multi_search:
            report['multi_search_audit'] = audit_multi_search(agent)
            if not report['multi_search_audit']['passed']:
                report['status'] = 'multi_search_expectation_mismatch'
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
