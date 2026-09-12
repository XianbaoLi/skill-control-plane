"""Live GLM cost/behavior A/B for mandatory vs metadata-first body reload."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.cards import load_retrieval_cards
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.capability_memory import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.integrations.reference_agent import ExperimentalSkillAgent


ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')

CASES = (
    ('simple', '继续调整现有演示稿的标题'),
    ('simple', '优化已有 slides 的版面布局'),
    ('simple', '给之前的 presentation 再补一页'),
    ('simple', '精简现有 deck 每页的文字'),
    ('simple', '调整演示文稿里的图表布局'),
    ('simple', '把结论页移动到最后并重新排版'),
    ('detailed', '严格按照该 Skill 规定的渲染与视觉校验流程检查现有演示稿'),
    ('detailed', '使用该 Skill 的标准 JSON spec 工作流生成新页，并遵守所有字段约束'),
    ('detailed', '按该 Skill 对模板占位符和 layout 的具体约束填充公司演示模板'),
    ('detailed', '按照该 Skill 的专用流程 patch 图表 categories 和 series 数据'),
    ('detailed', '执行该 Skill 规定的 LibreOffice 与 poppler 验证步骤并解释失败处理'),
    ('detailed', '按该 Skill 的完整导出 PDF 和逐页视觉 QA 流程完成验收'),
)

COMPRESSED_HISTORY = (
    {'role': 'user', 'content': '用户正在继续之前的演示文稿编辑工作。'},
    {'role': 'assistant', 'content': '已保留普通任务语义，可以继续处理下一项修改。'},
)


def frozen_state():
    return RuntimeCapabilityState(
        active_bundles=[ActiveBundle(
            'presentation-work', 'Create and revise presentation slides',
            ('powerpoint',))],
        skill_body_states={'powerpoint': 'evicted'},
    )


def snapshot():
    data = asdict(frozen_state())
    data['direct_skills'] = sorted(data['direct_skills'])
    return data


class RecordingClient:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def complete_messages(self, messages, *, tools):
        started = monotonic()
        response = self.client.complete_messages(messages, tools=tools)
        self.calls.append({
            'usage': deepcopy(self.client.last_usage),
            'wall_time_seconds': monotonic() - started,
            'response': deepcopy(response),
        })
        return response


def make_agent(discovery, client, *, policy):
    harness = RuntimeCapabilityHarness(
        discovery=discovery, state=deepcopy(frozen_state()))
    agent = ExperimentalSkillAgent(
        harness.control_plane,
        client,
        max_steps=8,
        require_evicted_body_reload=(policy == 'old_mandatory'),
    )
    agent.history = deepcopy(list(COMPRESSED_HISTORY))
    return agent


def classify_case(*, category, audit, trace, answer):
    tools = audit['tool_calls']
    rediscovered = any(row['tool'] == 'load_capability' for row in tools)
    exact_reloaded = any(
        row['tool'] == 'load_skill_body'
        and row['arguments'].get('skill_id') == 'powerpoint'
        for row in tools)
    tool_failure = any(row.get('tool_error') or row.get('error') for row in trace)
    completed = isinstance(answer, str) and bool(answer.strip()) and not tool_failure
    if rediscovered:
        outcome = 'rediscovered'
        success = False
    elif category == 'detailed' and not exact_reloaded:
        outcome = 'missing_reload_failure'
        success = False
    elif category == 'simple' and exact_reloaded:
        outcome = 'unnecessary_reload'
        success = completed
    elif exact_reloaded:
        outcome = 'exact_reloaded'
        success = completed
    else:
        outcome = 'reused_bundle_only'
        success = completed
    return {
        'outcome': outcome,
        'behavior_success': success,
        'rediscovered': rediscovered,
        'exact_reloaded': exact_reloaded,
        'load_skill_body_called': any(
            row['tool'] == 'load_skill_body' for row in tools),
        'completed': completed,
    }


def run_case(discovery, glm, *, policy, category, task, case_id):
    recorder = RecordingClient(glm)
    agent = make_agent(discovery, recorder, policy=policy)
    started = monotonic()
    answer = agent.run(task)
    wall_time = monotonic() - started
    audit = deepcopy(agent.turn_audit)
    classification = classify_case(
        category=category, audit=audit, trace=agent.trace, answer=answer)
    usage = [call.get('usage') or {} for call in recorder.calls]
    return {
        'case_id': case_id,
        'category': category,
        'task': task,
        'policy': policy,
        'bundle_surface': audit['bundle_state_before'],
        'body_state': audit['bundle_member_body_state_before'],
        'model_visible_body_ids_before':
            agent.trace[0].get('model_visible_body_ids', []),
        'tool_call_trace': [
            {'tool': row['tool'], 'arguments': row['arguments']}
            for row in audit['tool_calls']],
        'retrieval_calls': audit['retrieval_call_count_after']
            - audit['retrieval_call_count_before'],
        'body_loads': audit['body_load_count_after']
            - audit['body_load_count_before'],
        'model_calls': len(recorder.calls),
        'input_tokens': sum(int(row.get('prompt_tokens', 0)) for row in usage),
        'output_tokens': sum(int(row.get('completion_tokens', 0)) for row in usage),
        'total_tokens': sum(int(row.get('total_tokens', 0)) for row in usage),
        'wall_time_seconds': wall_time,
        'answer': answer,
        'trace': deepcopy(agent.trace),
        **classification,
    }


def summarize(cases):
    count = len(cases)
    total = lambda key: sum(case[key] for case in cases)
    simple = [case for case in cases if case['category'] == 'simple']
    detailed = [case for case in cases if case['category'] == 'detailed']
    return {
        'cases': count,
        'load_capability_rate': total('rediscovered') / count,
        'retrieval_calls': total('retrieval_calls'),
        'load_skill_body_rate': total('load_skill_body_called') / count,
        'body_load_count': total('body_loads'),
        'model_call_count': total('model_calls'),
        'input_tokens': total('input_tokens'),
        'output_tokens': total('output_tokens'),
        'total_tokens': total('total_tokens'),
        'wall_time_seconds': round(total('wall_time_seconds'), 3),
        'behavior_success_rate': total('behavior_success') / count,
        'full_body_reload_avoided_rate': sum(
            not case['load_skill_body_called'] and case['behavior_success']
            for case in cases) / count,
        'unnecessary_reload_rate': sum(
            case['outcome'] == 'unnecessary_reload' for case in simple) / len(simple),
        'missing_reload_failure_rate': sum(
            case['outcome'] == 'missing_reload_failure' for case in detailed)
            / len(detailed),
        'simple': {
            'cases': len(simple),
            'load_skill_body_rate': sum(
                case['load_skill_body_called'] for case in simple) / len(simple),
            'behavior_success_rate': sum(
                case['behavior_success'] for case in simple) / len(simple),
        },
        'detailed': {
            'cases': len(detailed),
            'load_skill_body_rate': sum(
                case['load_skill_body_called'] for case in detailed) / len(detailed),
            'behavior_success_rate': sum(
                case['behavior_success'] for case in detailed) / len(detailed),
        },
    }


def main():
    output_dir = Path('local_artifacts/evicted-body-policy-ab')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=output_dir)) / 'trace.json'
    report = {
        'started': datetime.now(timezone.utc).isoformat(),
        'live_api': True,
        'runtime_snapshot': snapshot(),
        'compressed_history': deepcopy(list(COMPRESSED_HISTORY)),
        'cases_definition': [
            {'case_id': index, 'category': category, 'task': task}
            for index, (category, task) in enumerate(CASES, 1)],
        'cases': [],
        'failures': [],
    }

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    try:
        _validate_root_snapshot(str(ROOT), str(MANIFEST))
        cards = load_retrieval_cards(CARDS)
        registry = SkillRegistry.from_tree(ROOT)
        discovery = SkillDiscovery(registry, retrieval_cards=cards)
        glm = BigModelChatClient(timeout=60, max_tokens=4096)
        report['model'] = glm.model
        for case_id, (category, task) in enumerate(CASES, 1):
            for policy in ('old_mandatory', 'new_metadata_first'):
                try:
                    case = run_case(
                        discovery, glm, policy=policy, category=category,
                        task=task, case_id=case_id)
                except Exception as exc:
                    case = {
                        'case_id': case_id, 'category': category,
                        'task': task, 'policy': policy,
                        'error': {'type': type(exc).__name__, 'message': str(exc)},
                    }
                    report['failures'].append(case)
                report['cases'].append(case)
                save()
        complete = [case for case in report['cases'] if 'error' not in case]
        old = [case for case in complete if case['policy'] == 'old_mandatory']
        new = [case for case in complete if case['policy'] == 'new_metadata_first']
        report['behavior_failures'] = [
            {
                'case_id': case['case_id'],
                'category': case['category'],
                'policy': case['policy'],
                'outcome': case['outcome'],
            }
            for case in complete if not case['behavior_success']
        ]
        report['reload_inefficiencies'] = [
            {
                'case_id': case['case_id'],
                'category': case['category'],
                'policy': case['policy'],
                'outcome': case['outcome'],
            }
            for case in complete if case['outcome'] == 'unnecessary_reload'
        ]
        report['old_mandatory'] = summarize(old)
        report['new_metadata_first'] = summarize(new)
        report['delta_new_minus_old'] = {
            key: report['new_metadata_first'][key] - report['old_mandatory'][key]
            for key in ('load_capability_rate', 'retrieval_calls',
                        'load_skill_body_rate', 'body_load_count',
                        'model_call_count', 'input_tokens', 'output_tokens',
                        'total_tokens', 'wall_time_seconds',
                        'behavior_success_rate', 'full_body_reload_avoided_rate',
                        'unnecessary_reload_rate', 'missing_reload_failure_rate')
        }
        report['status'] = 'success' if not report['failures'] else 'completed_with_failures'
    except Exception as exc:
        report.update(status='error', error={'type': type(exc).__name__,
                                             'message': str(exc)})
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({
            'trace': str(output), 'status': report['status'],
            'old': report.get('old_mandatory'),
            'new': report.get('new_metadata_first'),
            'delta': report.get('delta_new_minus_old'),
            'failure_count': len(report['failures']),
        }, ensure_ascii=False), flush=True)
    if report['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
