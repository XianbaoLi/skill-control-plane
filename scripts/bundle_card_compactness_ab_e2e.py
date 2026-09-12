"""Live GLM behavior/cost A/B for old versus compact Bundle cards."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.cards import load_retrieval_cards
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.capability_loading import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.runtime.experimental_agent import ExperimentalSkillAgent


ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
SURFACE_MARKER = 'Runtime Bundles (metadata only)\n'

CASES = (
    ('resident_reuse', '修改现有演示稿的标题'),
    ('resident_reuse', '调整已有 slides 的排版'),
    ('resident_reuse', '给现有 presentation 增加一页'),
    ('resident_reuse', '修改演示文稿里的图表'),
    ('evicted_simple', '继续精简已有 deck 每页的文字'),
    ('evicted_simple', '把结论页移动到最后并重新排版'),
    ('evicted_detailed', '严格按照该 Skill 的专用渲染和视觉校验流程检查演示稿'),
    ('evicted_detailed', '使用该 Skill 的标准 JSON spec 工作流并遵守全部字段约束'),
    ('capability_gap', '读取并修改 Excel 工作簿中的公式和表格'),
    ('capability_gap', '把 CSV 数据整理成带图表的 spreadsheet'),
    ('mixed_reuse_gap', '继续修改 PPT，同时把数据导出成 Excel 工作簿'),
    ('unrelated_gap', '调试 GitHub 项目中只在 CI 出现的测试失败'),
)

SCALING_BUNDLES = (
    ActiveBundle('presentations', 'Create and revise presentation slides', ('powerpoint',)),
    ActiveBundle('spreadsheets', 'Create and revise spreadsheet workbooks', ('xlsx',)),
    ActiveBundle('pdf-documents', 'Read, edit, and validate PDF documents', ('pdf',)),
    ActiveBundle('github-review', 'Review and maintain GitHub projects', ('github-code-review',)),
    ActiveBundle('email-work', 'Triage and respond to email', ('email-inbox-triage',)),
    ActiveBundle('research', 'Research and write grounded technical material', ('grounded-citations',)),
    ActiveBundle('visual-design', 'Create diagrams and visual explanations', ('architecture-diagram',)),
    ActiveBundle('media', 'Work with audio and video content', ('youtube-content',)),
    ActiveBundle('notes', 'Manage reusable notes and knowledge', ('obsidian',)),
    ActiveBundle('debugging', 'Inspect and debug software failures', ('systematic-debugging',)),
)


def behavior_state(category):
    body_state = 'evicted' if category.startswith('evicted_') else 'resident'
    return RuntimeCapabilityState(
        active_bundles=[ActiveBundle(
            'presentation-work', 'Create and revise presentation slides',
            ('powerpoint',))],
        skill_body_states={'powerpoint': body_state},
    )


def canonical_history(powerpoint_body):
    return [
        {'role': 'user', 'content': '创建并维护一份演示文稿。'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'seed-apply', 'type': 'function',
            'function': {'name': 'apply_capability', 'arguments': '{}'},
        }]},
        {'role': 'tool', 'tool_call_id': 'seed-apply', 'content': json.dumps({
            'action': 'CREATE', 'affected_bundle_id': 'presentation-work',
            'selected_skill_ids': ['powerpoint'],
            'skill_bodies': [{'skill_id': 'powerpoint', 'body': powerpoint_body}],
        }, ensure_ascii=False)},
        {'role': 'assistant', 'content': '演示文稿工作流已经建立。'},
    ]


class OldBundleCardAgent(ExperimentalSkillAgent):
    """A/B-only surface override; policy, history, tools and state are unchanged."""

    def render_system_context(self):
        compact = super().render_system_context()
        prefix, _surface = compact.split(SURFACE_MARKER, 1)
        return prefix + SURFACE_MARKER + self.harness.render_bundle_context(compact=False)


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


def make_agent(discovery, client, *, category, surface):
    harness = RuntimeCapabilityHarness(
        discovery=discovery, state=deepcopy(behavior_state(category)))
    agent_type = OldBundleCardAgent if surface == 'old' else ExperimentalSkillAgent
    agent = agent_type(harness, client, max_steps=8)
    agent.history = canonical_history(discovery.records['powerpoint'].body)
    return agent


def estimate_tokens(text):
    """Stable tokenizer-free estimate for relative Bundle surface scaling."""
    return ceil(len(text) / 4)


def surface_metrics(surface):
    return {'chars': len(surface), 'estimated_tokens': estimate_tokens(surface)}


def classify(category, audit, trace, answer):
    calls = audit['tool_calls']
    discovery_calls = [row for row in calls if row['tool'] == 'load_capability']
    body_calls = [row for row in calls if row['tool'] == 'load_skill_body']
    exact_reload = any(
        row['arguments'].get('skill_id') == 'powerpoint' for row in body_calls)
    recoverable_tool_errors = sum(bool(row.get('tool_error')) for row in trace)
    completed = isinstance(answer, str) and bool(answer.strip())
    if category == 'resident_reuse':
        success = completed and not discovery_calls and not body_calls
        outcome = 'resident_reuse' if success else 'resident_rediscovery'
    elif category == 'evicted_simple':
        success = completed and not discovery_calls
        outcome = ('metadata_reuse' if not body_calls else 'exact_reloaded') \
            if success else 'evicted_rediscovery'
    elif category == 'evicted_detailed':
        success = completed and exact_reload and not discovery_calls
        outcome = 'exact_reloaded' if success else 'missing_reload'
    else:
        success = completed and bool(discovery_calls)
        outcome = 'gap_discovered' if success else 'wrong_suppression'
    return {
        'behavior_success': success,
        'behavior_outcome': outcome,
        'load_capability_called': bool(discovery_calls),
        'load_skill_body_called': bool(body_calls),
        'exact_reload': exact_reload,
        'completed': completed,
        'recoverable_tool_error_count': recoverable_tool_errors,
    }


def run_case(discovery, glm, *, case_id, category, task, surface):
    recorder = RecordingClient(glm)
    agent = make_agent(discovery, recorder, category=category, surface=surface)
    card = agent.harness.render_bundle_context(compact=(surface == 'compact'))
    model_history_before = agent.build_model_history()
    projection_before = deepcopy(agent._last_history_projection)
    started = monotonic()
    answer = agent.run(task)
    elapsed = monotonic() - started
    audit = deepcopy(agent.turn_audit)
    usage = [call.get('usage') or {} for call in recorder.calls]
    result = {
        'case_id': case_id, 'category': category, 'task': task,
        'surface': surface, 'bundle_card': json.loads(card),
        'bundle_card_metrics': surface_metrics(card),
        'body_state': audit['bundle_member_body_state_before'],
        'model_visible_body_ids_before': projection_before[
            'model_visible_body_ids'],
        'projected_history_before': model_history_before,
        'tool_calls': deepcopy(audit['tool_calls']),
        'retrieval_calls': audit['retrieval_call_count_after']
            - audit['retrieval_call_count_before'],
        'body_loads': audit['body_load_count_after']
            - audit['body_load_count_before'],
        'model_calls': len(recorder.calls),
        'input_tokens': sum(int(row.get('prompt_tokens', 0)) for row in usage),
        'output_tokens': sum(int(row.get('completion_tokens', 0)) for row in usage),
        'total_tokens': sum(int(row.get('total_tokens', 0)) for row in usage),
        'wall_time_seconds': elapsed,
        'answer': answer,
        'trace': deepcopy(agent.trace),
    }
    result.update(classify(category, audit, agent.trace, answer))
    return result


def summarize(cases):
    count = len(cases)
    return {
        'cases': count,
        'bundle_surface_tokens': sum(
            row['bundle_card_metrics']['estimated_tokens'] for row in cases),
        'input_tokens': sum(row['input_tokens'] for row in cases),
        'output_tokens': sum(row['output_tokens'] for row in cases),
        'total_tokens': sum(row['total_tokens'] for row in cases),
        'model_calls': sum(row['model_calls'] for row in cases),
        'recoverable_tool_error_count': sum(
            row['recoverable_tool_error_count'] for row in cases),
        'load_capability_rate': sum(row['load_capability_called'] for row in cases) / count,
        'retrieval_calls': sum(row['retrieval_calls'] for row in cases),
        'load_skill_body_rate': sum(row['load_skill_body_called'] for row in cases) / count,
        'exact_reload_rate': sum(row['exact_reload'] for row in cases) / count,
        'behavior_success_rate': sum(row['behavior_success'] for row in cases) / count,
        'wall_time_seconds': round(sum(row['wall_time_seconds'] for row in cases), 3),
        'by_category': {
            category: {
                'cases': len(group),
                'success_rate': sum(row['behavior_success'] for row in group) / len(group),
                'load_capability_rate': sum(
                    row['load_capability_called'] for row in group) / len(group),
                'load_skill_body_rate': sum(
                    row['load_skill_body_called'] for row in group) / len(group),
            }
            for category in sorted({row['category'] for row in cases})
            for group in [[row for row in cases if row['category'] == category]]
        },
    }


def scaling_report(discovery):
    rows = []
    for count in (1, 3, 5, 10):
        bundles = list(SCALING_BUNDLES[:count])
        states = {skill_id: 'evicted' for bundle in bundles
                  for skill_id in bundle.skill_ids}
        harness = RuntimeCapabilityHarness(
            discovery=discovery,
            state=RuntimeCapabilityState(bundles, skill_body_states=states))
        old = harness.render_bundle_context(compact=False)
        compact = harness.render_bundle_context()
        old_metrics = surface_metrics(old)
        compact_metrics = surface_metrics(compact)
        rows.append({
            'bundle_count': count,
            'old': {**old_metrics,
                    'estimated_tokens_per_bundle': old_metrics['estimated_tokens'] / count},
            'compact': {**compact_metrics,
                        'estimated_tokens_per_bundle': compact_metrics['estimated_tokens'] / count},
            'token_reduction': old_metrics['estimated_tokens']
                - compact_metrics['estimated_tokens'],
            'token_reduction_rate': 1 - (
                compact_metrics['estimated_tokens'] / old_metrics['estimated_tokens']),
            'old_bundle_card': json.loads(old),
            'compact_bundle_card': json.loads(compact),
        })
    return rows


def regression_analysis(old_cases, compact_cases):
    regressions = []
    old_by_id = {row['case_id']: row for row in old_cases}
    compact_by_id = {row['case_id']: row for row in compact_cases}
    for case_id in sorted(old_by_id.keys() & compact_by_id.keys()):
        old = old_by_id[case_id]
        compact = compact_by_id[case_id]
        if old['behavior_success'] and not compact['behavior_success']:
            old_text = json.dumps(old['bundle_card'], ensure_ascii=False).casefold()
            compact_text = json.dumps(compact['bundle_card'], ensure_ascii=False).casefold()
            removed = sorted(set(old_text.split()) - set(compact_text.split()))
            if compact['behavior_outcome'] == 'wrong_suppression':
                cause = 'purpose_too_broad_or_model_stochastic_behavior'
            elif removed:
                cause = 'capability_phrase_or_member_description_removed'
            else:
                cause = 'model_stochastic_behavior_or_other'
            regressions.append({
                'case_id': compact['case_id'], 'category': compact['category'],
                'task': compact['task'], 'old_outcome': old['behavior_outcome'],
                'compact_outcome': compact['behavior_outcome'],
                'likely_cause': cause, 'removed_surface_terms_sample': removed[:12],
            })
    return regressions


def main():
    output_dir = Path('local_artifacts/bundle-card-compactness-ab')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=output_dir)) / 'trace.json'
    report = {
        'started': datetime.now(timezone.utc).isoformat(), 'live_api': True,
        'cases_definition': [
            {'case_id': index, 'category': category, 'task': task}
            for index, (category, task) in enumerate(CASES, 1)],
        'cases': [], 'execution_failures': [],
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
        report['scaling'] = scaling_report(discovery)
        for case_id, (category, task) in enumerate(CASES, 1):
            for surface in ('old', 'compact'):
                try:
                    case = run_case(
                        discovery, glm, case_id=case_id, category=category,
                        task=task, surface=surface)
                except Exception as exc:
                    case = {
                        'case_id': case_id, 'category': category, 'task': task,
                        'surface': surface,
                        'error': {'type': type(exc).__name__, 'message': str(exc)},
                    }
                    report['execution_failures'].append(case)
                report['cases'].append(case)
                save()
        complete = [row for row in report['cases'] if 'error' not in row]
        old = [row for row in complete if row['surface'] == 'old']
        compact = [row for row in complete if row['surface'] == 'compact']
        report['old'] = summarize(old)
        report['compact'] = summarize(compact)
        report['bundle_surface_token_reduction'] = 1 - (
            report['compact']['bundle_surface_tokens']
            / report['old']['bundle_surface_tokens'])
        report['total_token_reduction'] = 1 - (
            report['compact']['total_tokens'] / report['old']['total_tokens'])
        report['behavior_regressions'] = regression_analysis(old, compact)
        report['behavior_regression_count'] = len(report['behavior_regressions'])
        report['status'] = ('success' if not report['execution_failures']
                            else 'completed_with_failures')
    except Exception as exc:
        report.update(status='error', error={
            'type': type(exc).__name__, 'message': str(exc)})
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({
            'trace': str(output), 'status': report['status'],
            'old': report.get('old'), 'compact': report.get('compact'),
            'bundle_surface_token_reduction': report.get(
                'bundle_surface_token_reduction'),
            'total_token_reduction': report.get('total_token_reduction'),
            'behavior_regression_count': report.get('behavior_regression_count'),
            'execution_failure_count': len(report['execution_failures']),
        }, ensure_ascii=False), flush=True)
    if report['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
