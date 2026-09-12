"""Observe a fixed four-turn conversation without changing runtime policy."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from collections.abc import Callable

from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.integrations.reference_agent import ConversationClient, ExperimentalSkillAgent

TASKS = (
    '请按本环境已有的文档处理操作规范，为团队建立持续复用的文档制作工作流：'
    '从扫描版 PDF 提取文字和表格，再制作可编辑的演示文稿。请依据实际可用的操作说明，'
    '简要列出工具、依赖、关键命令和验证步骤，不要猜命令。当前没有输入文件，不执行或生成文件。',
    '继续刚才这份演示文稿：把标题改成“季度业务回顾”，调整幻灯片顺序，并更新演讲备注。'
    '请依据已有操作说明给出具体修改和验证方案；仍无实际文件，不执行。回答控制在 600 字内。',
    '继续同一套文档制作工作流：把从 PDF 提取的表格进一步导出为可编辑的 Excel 工作簿，'
    '包含多个工作表、汇总公式和数字格式，并检查公式与数据是否正确。'
    '请依据本环境的操作说明给出具体方案，仍不执行。回答控制在 600 字内。',
    '现在换一个与文档制作无关的软件仓库问题：GitHub 上一个 Python 项目的 pull request '
    '在 CI 中间歇性测试失败，本地却通过。请依据本环境已有操作规范，给出读取 PR 差异及失败日志、'
    '复现、定位根因和验证修复的排查流程。当前无仓库文件或访问权限，只给方案，不执行。'
    '回答控制在 600 字内。',
)
DOCUMENT_SKILLS = {'pdf', 'ocr-and-documents', 'powerpoint'}


def state_snapshot(harness: RuntimeCapabilityHarness) -> dict:
    state = asdict(harness.state)
    state['direct_skills'] = sorted(state['direct_skills'])
    return state


def evaluate_turns(turns: list[dict]) -> list[dict]:
    """Post-hoc checks only; never supplied to the model or runtime validator.

    These check observed choices, not semantic correctness or causality.
    Missing prerequisites fail explicitly rather than silently passing.
    """
    document_id = None
    evaluations = []
    for turn in turns:
        number = turn['turn']
        before = {b['bundle_id']: b for b in turn['state_before']['active_bundles']}
        after = {b['bundle_id']: b for b in turn['state_after']['active_bundles']}
        organizations = [a for a in turn['model_actions'] if a.get('tool') == 'apply_capability']
        selected = {s for a in organizations for s in a.get('skill_ids', [])}
        retrieved = {c['skill_id'] for result in turn['retrieval_results'] for c in result['candidates']}
        checks = {'turn_completed': turn['status'] == 'completed'}
        if number == 1:
            created = [b for bid, b in after.items() if bid not in before
                       and DOCUMENT_SKILLS <= set(b['skill_ids'])]
            document_id = created[0]['bundle_id'] if created else None
            checks.update(load_triggered=turn['load_triggered'],
                          document_skills_selected=DOCUMENT_SKILLS <= selected,
                          document_bundle_created=document_id is not None
                          and any(a['action'] == 'CREATE' for a in organizations))
        elif number == 2:
            checks.update(document_bundle_available=document_id in before,
                          no_retrieval=not turn['load_triggered'],
                          state_unchanged=turn['state_before'] == turn['state_after'])
        elif number == 3:
            checks.update(document_bundle_available=document_id in before,
                          load_triggered=turn['load_triggered'], xlsx_retrieved='xlsx' in retrieved,
                          xlsx_extended_into_document=any(
                              a['action'] == 'EXTEND' and a.get('target_bundle_id') == document_id
                              and 'xlsx' in a['skill_ids'] for a in organizations)
                          and document_id in after and 'xlsx' in after[document_id]['skill_ids'],
                          no_replacement_bundle=set(before) == set(after))
        elif number == 4:
            checks.update(document_bundle_available=document_id in before,
                          load_triggered=turn['load_triggered'],
                          no_document_extend=not any(
                              a['action'] == 'EXTEND' and a.get('target_bundle_id') == document_id
                              for a in organizations),
                          document_bundle_unchanged=document_id in before
                          and after.get(document_id) == before[document_id],
                          separate_organization=any(a['action'] in {'DIRECT', 'CREATE'} for a in organizations))
        evaluations.append({'turn': number, 'checks': checks,
                            'matches_expectation': all(checks.values()),
                            'document_bundle_id': document_id})
    return evaluations


def run_experiment(discovery: SkillDiscovery, client: ConversationClient, *,
                   max_steps: int = 8, checkpoint: Callable[[dict], None] | None = None) -> dict:
    """One Agent/client/history across all tasks; preserve deviations and failures."""
    report = {'tasks': list(TASKS), 'turns': [], 'model_calls': [], 'history': [],
              'status': 'running'}
    active_turn = None
    agent = None

    def save():
        if agent is not None:
            report['history'] = deepcopy(agent.history)
            if active_turn is not None:
                active_turn['trace'] = deepcopy(agent.trace)
        if checkpoint is not None:
            checkpoint(report)

    class RecordingClient:
        def complete_messages(self, messages, *, tools):
            surface = json.loads(messages[0]['content'].split('Runtime Bundles (metadata only)\n', 1)[1])
            call = {'turn': active_turn['turn'], 'messages': deepcopy(messages),
                    'system_bundle_metadata': surface['maintained_bundles'], 'tools': deepcopy(tools)}
            report['model_calls'].append(call)
            save()
            response = client.complete_messages(messages, tools=tools)
            call['response'] = response
            save()
            return response

    harness = RuntimeCapabilityHarness(discovery=discovery)
    agent = ExperimentalSkillAgent(
        harness.control_plane, RecordingClient(), max_steps=max_steps)
    for number, task in enumerate(TASKS, 1):
        history_start = len(agent.history)
        call_start = len(report['model_calls'])
        active_turn = {'turn': number, 'user_task': task, 'state_before': state_snapshot(harness),
                       'history_start': history_start, 'status': 'running'}
        report['turns'].append(active_turn)
        try:
            active_turn['final'] = agent.run(task)
            active_turn['status'] = 'completed'
        except ValueError as exc:
            # Observe the next task with the exact surviving state/history. No repair.
            active_turn.update(status='error', error_type=type(exc).__name__,
                               continue_next_turn=True)
        except Exception as exc:
            active_turn.update(status='error', error_type=type(exc).__name__)
            report['status'] = 'error'
        finally:
            active_turn['state_after'] = state_snapshot(harness)
            active_turn['history_end'] = len(agent.history)
            active_turn['model_actions'] = [
                {'tool': event['tool'], **deepcopy(event['arguments'])}
                for row in agent.trace for event in row['tool_executions']]
            active_turn['load_triggered'] = any(a.get('tool') == 'load_capability'
                                               for a in active_turn['model_actions'])
            names = {call['id']: call['function']['name']
                     for message in agent.history[history_start:] if message['role'] == 'assistant'
                     for call in (message.get('tool_calls') or [])}
            active_turn['retrieval_results'] = []
            for message in agent.history[history_start:]:
                if message['role'] == 'tool' and names.get(message['tool_call_id']) == 'load_capability':
                    result = json.loads(message['content'])
                    if 'candidates' in result:
                        active_turn['retrieval_results'].append(result)
            active_turn['system_bundle_metadata'] = [deepcopy(c['system_bundle_metadata'])
                                                     for c in report['model_calls'][call_start:]]
            save()
        if report['status'] == 'error':
            break
    if report['status'] != 'error':
        report['status'] = ('completed_with_turn_errors' if any(
            t['status'] == 'error' for t in report['turns']) else 'completed')
    report['evaluations'] = evaluate_turns(report['turns'])
    report['matches_all_expectations'] = (len(report['turns']) == 4 and all(
        e['matches_expectation'] for e in report['evaluations']))
    save()
    return report
