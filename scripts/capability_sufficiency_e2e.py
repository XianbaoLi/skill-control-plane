"""Live GLM trigger and capability-sufficiency behavior experiment."""
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
from skill_control_plane.discovery.bigmodel import (
    BigModelDenseRetriever,
    BigModelEmbeddingClient,
)
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime import SkillControlPlane
from skill_control_plane.runtime.capability_memory import ActiveBundle, RuntimeCapabilityState
from skill_control_plane.integrations.reference_agent import ExperimentalSkillAgent


ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
LIMITED_IDS = ('ocr-and-documents', 'powerpoint', 'pdf', 'xlsx')

CASES = (
    {
        'case_id': 'no_gap', 'corpus': 'full',
        'bundles': (('presentation-work', 'Create and revise presentation slides',
                     ('powerpoint',)),),
        'task': '继续修改现有 PPT 的标题和第三页排版。',
        'allowed_selected': (),
    },
    {
        'case_id': 'single_gap', 'corpus': 'full',
        'bundles': (('presentation-work', 'Create and revise presentation slides',
                     ('powerpoint',)),),
        'task': '新的子目标：把数据整理成包含公式和图表的 Excel 工作簿。',
        'allowed_selected': ('xlsx',),
    },
    {
        'case_id': 'multi_gap', 'corpus': 'limited', 'bundles': (),
        'candidate_window': 2,
        'task': ('处理扫描 PDF：先提取文字和表格，然后把结果制作成可编辑演示文稿。'
                 '第一次评估只处理当前 OCR concrete step，search need 只描述 image-based '
                 'text recognition，不混入下游产物；读完候选后，再把 presentation 作为 '
                 'residual gap 评估并继续 discovery。'),
        'allowed_selected': ('ocr-and-documents', 'pdf', 'powerpoint'),
    },
    {
        'case_id': 'bundle_pending_coverage', 'corpus': 'full',
        'candidate_window': 1,
        'bundles': (('pdf-work', 'Edit text in existing PDF documents', ('nano-pdf',)),),
        'task': ('现有 PDF Bundle 已覆盖编辑 PDF 文字，但不覆盖扫描 OCR。新的步骤先对扫描页 '
                 'OCR，再把结果制作成 PPT。'
                 '分别评估 OCR 与 presentation；不要重新搜索 PDF，也不要重搜 pending '
                 '已经可信覆盖的能力。'),
        'allowed_selected': ('ocr-and-documents', 'powerpoint'),
    },
    {
        'case_id': 'missing_skill', 'corpus': 'limited', 'bundles': (),
        'task': ('校准超导量子处理器的低温微波脉冲硬件。当前 Skill 库若没有可信能力，'
                 '请明确 UNSATISFIED；不要用文档、PPT、PDF 或 spreadsheet Skill 填补。'),
        'allowed_selected': (),
    },
    {
        'case_id': 'partial_success', 'corpus': 'limited', 'bundles': (),
        'task': ('从扫描 PDF 做 OCR，同时校准超导量子处理器的低温控制脉冲。可以提交可信的 '
                 'OCR Skill，但若量子硬件能力不存在，必须保留为 unresolved，不选择弱相关 Skill。'),
        'allowed_selected': ('ocr-and-documents', 'pdf'),
    },
    {
        'case_id': 'repeated_no_progress', 'corpus': 'limited', 'bundles': (),
        'task': ('先寻找扫描文档 OCR 能力，再针对低温量子脉冲校准这个 residual gap 搜索一次。'
                 '如果第二次没有产生新 candidate，遵守 no-progress 提示立即停止，以 '
                 'UNSATISFIED 保留该 gap，不重复相同搜索。'),
        'allowed_selected': ('ocr-and-documents', 'pdf'),
    },
)


def state_for(case):
    bundles = [ActiveBundle(*bundle) for bundle in case['bundles']]
    body_states = {skill_id: 'resident' for bundle in bundles
                   for skill_id in bundle.skill_ids}
    return RuntimeCapabilityState(bundles, skill_body_states=body_states)


def seeded_history(case, discovery):
    members = [skill_id for bundle in case['bundles'] for skill_id in bundle[2]]
    if not members:
        return [
            {'role': 'user', 'content': '开始一个新的 capability planning task。'},
            {'role': 'assistant', 'content': '当前没有已维护的 Bundle。'},
        ]
    return [
        {'role': 'user', 'content': '建立已有工作流。'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'seed-apply', 'type': 'function',
            'function': {'name': 'apply_capability', 'arguments': '{}'},
        }]},
        {'role': 'tool', 'tool_call_id': 'seed-apply', 'content': json.dumps({
            'action': 'CREATE', 'selected_skill_ids': members,
            'skill_bodies': [
                {'skill_id': skill_id, 'body': discovery.records[skill_id].body}
                for skill_id in members],
        }, ensure_ascii=False)},
        {'role': 'assistant', 'content': '已有工作流保持可用。'},
    ]


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


class CaseControlPlane(SkillControlPlane):
    """Experiment-only candidate window; retrieval order remains unchanged."""

    def __init__(self, *args, candidate_window=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_window = candidate_window

    def search_capability(self, need, *, k=10):
        return super().search_capability(need, k=self.candidate_window)


def selected_ids(agent):
    return sorted({skill_id for row in agent.trace
                   for skill_id in row['selected_skill_ids']})


def derived_tool_outcome(event):
    """Label trace events from executed behavior, never model arguments."""
    if event['tool'] == 'load_capability' and 'model_visible_payload' in event:
        return 'SEARCH_MORE'
    if event['tool'] == 'apply_capability' and 'remaining_gaps' in event:
        return 'UNSATISFIED' if event['remaining_gaps'] else 'COVERED'
    return None


def evaluate(case, agent, answer):
    audit = agent.turn_audit
    selected = set(selected_ids(agent))
    allowed = set(case['allowed_selected'])
    forced = sorted(selected - allowed)
    case_id = case['case_id']
    common = bool(answer.strip()) and audit['search_count'] <= 3 and not forced
    if case_id == 'no_gap':
        passed = common and audit['search_count'] == 0 and not selected
    elif case_id == 'single_gap':
        passed = common and audit['search_count'] == 1 and selected == {'xlsx'}
    elif case_id == 'multi_gap':
        passed = common and 2 <= audit['search_count'] <= 3 \
            and {'ocr-and-documents', 'powerpoint'} <= selected \
            and audit['sufficiency_transitions'][:2] == [
                'SEARCH_MORE', 'SEARCH_MORE'] \
            and audit['apply_count'] == 1
    elif case_id == 'bundle_pending_coverage':
        bundle_members = {skill_id for bundle in case['bundles'] for skill_id in bundle[2]}
        passed = common and 2 <= audit['search_count'] <= 3 \
            and {'ocr-and-documents', 'powerpoint'} <= selected \
            and not selected & bundle_members
    elif case_id == 'missing_skill':
        passed = common and 1 <= audit['search_count'] <= 3 and not selected \
            and audit['apply_count'] == 0 \
            and audit['capability_sufficiency_outcome'] == 'UNSATISFIED'
    elif case_id == 'partial_success':
        passed = common and 'ocr-and-documents' in selected and selected <= allowed \
            and audit['apply_count'] == 1 \
            and audit['capability_sufficiency_outcome'] == 'UNSATISFIED' \
            and bool(audit['unresolved_gaps'])
    else:
        passed = common and audit['no_progress_search_count'] >= 1 \
            and audit['search_count'] <= 3 \
            and audit['capability_sufficiency_outcome'] == 'UNSATISFIED'
    precision = 1.0 if not selected else len(selected & allowed) / len(selected)
    return {
        'passed': passed, 'selected_skill_ids': sorted(selected),
        'selected_skill_precision': precision,
        'forced_skill_ids': forced,
        'forced_skill_selection_count': len(forced),
    }


def run_case(case, discovery, glm):
    recorder = RecordingClient(glm)
    control_plane = CaseControlPlane(
        discovery.registry, discovery=discovery, state=deepcopy(state_for(case)),
        candidate_window=case.get('candidate_window', 10))
    agent = ExperimentalSkillAgent(control_plane, recorder, max_steps=10)
    agent.history = seeded_history(case, discovery)
    started = monotonic()
    answer = agent.run(case['task'])
    elapsed = monotonic() - started
    usage = [call.get('usage') or {} for call in recorder.calls]
    result = {
        'case_id': case['case_id'], 'task': case['task'],
        'corpus': case['corpus'], 'corpus_size': len(discovery.records),
        'candidate_window': control_plane.candidate_window,
        'bundle_before': agent.turn_audit['bundle_state_before'],
        'tool_call_trace': [
            {'tool': event['tool'], 'arguments': event['arguments'],
             'derived_outcome': derived_tool_outcome(event),
             'coverage': event.get('coverage'),
             'remaining_gaps': event.get('remaining_gaps')}
            for row in agent.trace for event in row['tool_executions']],
        'search_controls': [
            event['model_visible_payload']['search_control']
            for row in agent.trace for event in row['tool_executions']
            if event['tool'] == 'load_capability'
            and 'model_visible_payload' in event],
        'audit': deepcopy(agent.turn_audit),
        'model_calls': len(recorder.calls),
        'input_tokens': sum(int(row.get('prompt_tokens', 0)) for row in usage),
        'output_tokens': sum(int(row.get('completion_tokens', 0)) for row in usage),
        'total_tokens': sum(int(row.get('total_tokens', 0)) for row in usage),
        'wall_time_seconds': elapsed,
        'answer': answer,
        'trace': deepcopy(agent.trace),
    }
    result.update(evaluate(case, agent, answer))
    return result


def summarize(cases):
    search_distribution = {}
    for row in cases:
        count = str(row['audit']['search_count'])
        search_distribution[count] = search_distribution.get(count, 0) + 1
    return {
        'cases': len(cases),
        'passed': sum(row['passed'] for row in cases),
        'search_count_distribution': search_distribution,
        'load_capability_calls': sum(row['audit']['search_count'] for row in cases),
        'apply_count': sum(row['audit']['apply_count'] for row in cases),
        'average_selected_skill_precision': sum(
            row['selected_skill_precision'] for row in cases) / len(cases),
        'forced_skill_selection_count': sum(
            row['forced_skill_selection_count'] for row in cases),
        'outcomes': {
            state: sum(row['audit']['capability_sufficiency_outcome'] == state
                       for row in cases)
            for state in ('COVERED', 'SEARCH_MORE', 'UNSATISFIED')
        },
        'sufficiency_transition_counts': {
            state: sum(row['audit']['sufficiency_transitions'].count(state)
                       for row in cases)
            for state in ('COVERED', 'SEARCH_MORE', 'UNSATISFIED')
        },
        'unresolved_gaps': [
            {'case_id': row['case_id'], 'gaps': row['audit']['unresolved_gaps']}
            for row in cases if row['audit']['unresolved_gaps']],
        'repeated_search_count': sum(
            row['audit']['repeated_search_count'] for row in cases),
        'no_progress_search_count': sum(
            row['audit']['no_progress_search_count'] for row in cases),
        'max_budget_hits': sum(row['audit']['search_budget_hits'] for row in cases),
        'model_calls': sum(row['model_calls'] for row in cases),
        'input_tokens': sum(row['input_tokens'] for row in cases),
        'total_tokens': sum(row['total_tokens'] for row in cases),
        'wall_time_seconds': round(sum(row['wall_time_seconds'] for row in cases), 3),
    }


def main():
    output_dir = Path('local_artifacts/capability-sufficiency-v0.1')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=output_dir)) / 'trace.json'
    report = {
        'started': datetime.now(timezone.utc).isoformat(), 'live_api': True,
        'cases_definition': deepcopy(CASES), 'cases': [], 'failures': [],
    }

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    try:
        _validate_root_snapshot(str(ROOT), str(MANIFEST))
        cards = load_retrieval_cards(CARDS)
        full_registry = SkillRegistry.from_tree(ROOT)
        embedding = BigModelEmbeddingClient(timeout=45)

        def dense_factory(records):
            return BigModelDenseRetriever(
                records,
                model_name=embedding.model,
                dimensions=embedding.dimensions,
                embed_batch=embedding,
            )

        discoveries = {
            'full': SkillDiscovery(
                full_registry,
                retrieval_cards=cards,
                dense_factory=dense_factory,
            ),
            'limited': SkillDiscovery(
                SkillRegistry([full_registry.get(skill_id) for skill_id in LIMITED_IDS]),
                retrieval_cards={skill_id: cards[skill_id] for skill_id in LIMITED_IDS},
                dense_factory=dense_factory,
            ),
        }
        glm = BigModelChatClient(timeout=60, max_tokens=4096)
        report['model'] = glm.model
        report['corpora'] = {name: sorted(discovery.records)
                             for name, discovery in discoveries.items()}
        for case in CASES:
            try:
                result = run_case(case, discoveries[case['corpus']], glm)
            except Exception as exc:
                result = {
                    'case_id': case['case_id'],
                    'error': {'type': type(exc).__name__, 'message': str(exc)}}
                report['failures'].append(result)
            report['cases'].append(result)
            save()
        complete = [row for row in report['cases'] if 'error' not in row]
        report['summary'] = summarize(complete)
        report['behavior_failures'] = [
            {'case_id': row['case_id'],
             'selected_skill_ids': row['selected_skill_ids'],
             'forced_skill_ids': row['forced_skill_ids'],
             'audit': row['audit']}
            for row in complete if not row['passed']]
        report['status'] = ('success' if not report['failures']
                            else 'completed_with_failures')
    except Exception as exc:
        report.update(status='error', error={
            'type': type(exc).__name__, 'message': str(exc)})
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({
            'trace': str(output), 'status': report['status'],
            'summary': report.get('summary'),
            'behavior_failures': report.get('behavior_failures'),
            'execution_failures': report['failures'],
        }, ensure_ascii=False), flush=True)
    if report['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
