"""Clean live post-compression Bundle-visible/hidden causal A/B experiment."""
from __future__ import annotations

import json
import re
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
from skill_control_plane.runtime import SkillControlPlane
from skill_control_plane.runtime.capability_memory import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.integrations.reference_agent import ExperimentalSkillAgent


ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')

PARAPHRASES = (
    '继续修改之前的演示文稿第三页',
    '把现有 slides 的标题调整一下',
    '继续优化刚才那套幻灯片的排版',
    '给现有演示稿再补一页',
    '把之前 presentation 的第三页重做一下',
    '继续完善刚才的汇报材料',
    '把已有 deck 的结论页重新排版',
    '继续调整演示文稿里的图表',
    '把刚才的幻灯片内容再精简一些',
    '继续编辑之前那套展示材料',
)

COMPRESSED_HISTORY = (
    {'role': 'user', 'content': '用户正在继续之前的演示文稿编辑工作。'},
    {'role': 'assistant', 'content': '已保留普通任务语义，可以继续处理下一项修改。'},
)


def frozen_runtime_state():
    return RuntimeCapabilityState(
        active_bundles=[ActiveBundle(
            'presentation-work',
            'Create and revise presentation slides',
            ('powerpoint',),
        )],
        skill_body_states={'powerpoint': 'evicted'},
    )


def runtime_snapshot():
    data = asdict(frozen_runtime_state())
    data['direct_skills'] = sorted(data['direct_skills'])
    return data


class HiddenBundleAgent(ExperimentalSkillAgent):
    """Experiment-only rendering ablation; runtime state remains untouched."""

    def render_system_context(self) -> str:
        visible = super().render_system_context()
        prefix, _surface = visible.split('Runtime Bundles (metadata only)\n', 1)
        prefix = re.sub(
            r'Authoritative evicted Bundle Skill IDs: \[[^\n]*\]',
            'Authoritative evicted Bundle Skill IDs: []',
            prefix,
        )
        return (prefix + 'Runtime Bundles (metadata only)\n'
                '{"maintained_bundles":[]}')


def make_agent(discovery, client, *, visible):
    control_plane = SkillControlPlane(
        discovery.registry, discovery=discovery,
        state=deepcopy(frozen_runtime_state()),
    )
    agent_type = ExperimentalSkillAgent if visible else HiddenBundleAgent
    agent = agent_type(control_plane, client, max_steps=8)
    agent.history = deepcopy(list(COMPRESSED_HISTORY))
    return agent


def first_model_input(agent, task):
    return [
        {'role': 'system', 'content': agent.render_system_context()},
        *agent.build_model_history(),
        {'role': 'user', 'content': task},
    ]


def audit_hidden_input(messages, *, full_body):
    """Audit state leakage while excluding the allowed natural-language query."""

    system = messages[0]['content']
    compressed = messages[1:-1]
    compressed_text = json.dumps(compressed, ensure_ascii=False)
    checks = {
        'bundle_surface_empty': system.endswith('{"maintained_bundles":[]}'),
        'no_skill_id_in_system': 'powerpoint' not in system.casefold(),
        'no_bundle_id_in_system': 'presentation-work' not in system,
        'no_skill_id_in_compressed_history': 'powerpoint' not in compressed_text.casefold(),
        'no_bundle_id_in_compressed_history': 'presentation-work' not in compressed_text,
        'no_selected_skill_ids': 'selected_skill_ids' not in compressed_text,
        'no_full_body': full_body not in compressed_text,
        'no_capability_tool_messages': not any(
            message.get('role') == 'tool' or message.get('tool_calls')
            for message in compressed),
        'no_prior_bundle_state': 'skill_body_states' not in compressed_text
            and 'maintained_bundles' not in compressed_text,
    }
    return {'checks': checks, 'bundle_hidden_leakage': not all(checks.values())}


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


def run_case(discovery, glm, *, visible, task, case_id, full_body):
    recorder = RecordingClient(glm)
    agent = make_agent(discovery, recorder, visible=visible)
    input_audit = None
    if not visible:
        input_audit = audit_hidden_input(
            first_model_input(agent, task), full_body=full_body)
        if input_audit['bundle_hidden_leakage']:
            return {'case_id': case_id, 'arm': 'hidden', 'task': task,
                    'excluded': True, 'input_audit': input_audit}

    started = monotonic()
    answer = agent.run(task)
    wall_time = monotonic() - started
    audit = deepcopy(agent.turn_audit)
    tools = audit['tool_calls']
    load_capability = [row for row in tools if row['tool'] == 'load_capability']
    exact_loads = [row for row in tools if row['tool'] == 'load_skill_body'
                   and row['arguments'].get('skill_id') == 'powerpoint']
    usage = [call.get('usage') or {} for call in recorder.calls]
    return {
        'case_id': case_id,
        'arm': 'visible' if visible else 'hidden',
        'task': task,
        'excluded': False,
        'input_audit': input_audit,
        'answer': answer,
        'turn_audit': audit,
        'trace': deepcopy(agent.trace),
        'tool_call_trace': [
            {'tool': row['tool'], 'arguments': row['arguments']} for row in tools],
        'load_capability_called': bool(load_capability),
        'load_skill_body_called': any(row['tool'] == 'load_skill_body' for row in tools),
        'exact_reload': bool(exact_loads) and not load_capability,
        'rediscovery': bool(load_capability),
        'reuse_success': bool(exact_loads) and not load_capability,
        'retrieval_calls': audit['retrieval_call_count_after']
            - audit['retrieval_call_count_before'],
        'body_loads': audit['body_load_count_after']
            - audit['body_load_count_before'],
        'model_calls': len(recorder.calls),
        'input_tokens': sum(int(row.get('prompt_tokens', 0)) for row in usage),
        'total_tokens': sum(int(row.get('total_tokens', 0)) for row in usage),
        'wall_time_seconds': wall_time,
    }


def summarize(cases, *, visible):
    included = [case for case in cases if not case['excluded']]
    count = len(included)
    if not count:
        return {'included_cases': 0}
    total = lambda key: sum(case[key] for case in included)
    return {
        'included_cases': count,
        'load_capability_rate': total('load_capability_called') / count,
        'load_skill_body_rate': total('load_skill_body_called') / count,
        'exact_reload_rate': total('exact_reload') / count,
        'rediscovery_rate': total('rediscovery') / count,
        'reuse_success_rate': total('reuse_success') / count if visible else None,
        'retrieval_call_count': total('retrieval_calls'),
        'body_load_count': total('body_loads'),
        'model_call_count': total('model_calls'),
        'input_tokens': total('input_tokens'),
        'total_tokens': total('total_tokens'),
        'wall_time_seconds': round(total('wall_time_seconds'), 3),
    }


def main():
    output_dir = Path('local_artifacts/post-compression-bundle-ab')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=output_dir)) / 'trace.json'
    report = {
        'started': datetime.now(timezone.utc).isoformat(),
        'live_api': True,
        'paraphrases': list(PARAPHRASES),
        'runtime_snapshot': runtime_snapshot(),
        'compressed_model_visible_history': deepcopy(list(COMPRESSED_HISTORY)),
        'canonical_audit_history': {
            'retained_outside_active_context': True,
            'body_ids': ['powerpoint'],
            'contains_capability_management_history': True,
        },
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
        full_body = registry.load_skill_body('powerpoint')
        glm = BigModelChatClient(timeout=60, max_tokens=4096)
        report['model'] = glm.model
        for case_id, task in enumerate(PARAPHRASES, 1):
            for visible in (True, False):
                try:
                    case = run_case(
                        discovery, glm, visible=visible, task=task,
                        case_id=case_id, full_body=full_body)
                except Exception as exc:
                    case = {
                        'case_id': case_id,
                        'arm': 'visible' if visible else 'hidden',
                        'task': task,
                        'excluded': True,
                        'error': {'type': type(exc).__name__, 'message': str(exc)},
                    }
                report['cases'].append(case)
                save()

        visible_cases = [case for case in report['cases'] if case['arm'] == 'visible']
        hidden_cases = [case for case in report['cases'] if case['arm'] == 'hidden']
        report['visible'] = summarize(visible_cases, visible=True)
        report['hidden'] = summarize(hidden_cases, visible=False)
        for case in report['cases']:
            if case['excluded']:
                report['failures'].append({
                    'case_id': case['case_id'], 'arm': case['arm'],
                    'reason': 'excluded leakage/error',
                    'detail': case.get('input_audit') or case.get('error'),
                })
            elif case['arm'] == 'visible' and not case['reuse_success']:
                report['failures'].append({
                    'case_id': case['case_id'], 'arm': 'visible',
                    'reason': 'visible arm did not exact-reload without discovery',
                    'tool_call_trace': case['tool_call_trace'],
                })
        report['causal_effect'] = {
            'rediscovery_rate_difference_hidden_minus_visible':
                report['hidden'].get('rediscovery_rate', 0)
                - report['visible'].get('rediscovery_rate', 0),
            'retrieval_call_difference_hidden_minus_visible':
                report['hidden'].get('retrieval_call_count', 0)
                - report['visible'].get('retrieval_call_count', 0),
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
            'visible': report.get('visible'), 'hidden': report.get('hidden'),
            'causal_effect': report.get('causal_effect'),
            'failure_count': len(report['failures']),
        }, ensure_ascii=False), flush=True)
    if report['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
