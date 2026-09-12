"""Live GLM Bundle-first reuse, capability-gap, and visibility A/B experiment."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

from skill_control_plane.cli import _validate_root_snapshot
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.discovery.cards import load_retrieval_cards
from skill_control_plane.discovery.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.integrations.reference_agent import (
    AGENT_INSTRUCTIONS, ExperimentalSkillAgent,
)


ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')

BOOTSTRAP = (
    '读取这个文档并做成 PPT。当前没有实际文件，只给出依据现有 Skill 的方案。'
    '这是持续演示文稿工作：先发现所需能力，再用 CREATE 建立只包含 powerpoint Skill 的 Bundle。'
    '调用 apply_capability 时遵守现有 schema，包含非空 reason。'
)
REPEATED = (
    '继续改刚才 PPT 的第三页',
    '把演示稿标题再调整一下',
    '把刚才那套 slides 的排版优化一下',
    '继续修改这个 presentation',
    '在现有幻灯片后面再补一页',
)
EVICTED = '继续修改刚才 PPT 的第三页'
GAP = '把表格另外导出成 Excel 文件'
MIXED = '继续改 PPT，并把里面的数据另外导出 Excel'
UNRELATED = '排查 GitHub Python 项目中只在 CI 出现的间歇性测试失败'


class RecordingClient:
    def __init__(self, client):
        self.client = client
        self.label = 'unlabeled'
        self.calls = []

    def complete_messages(self, messages, *, tools):
        call = {
            'label': self.label,
            'input_chars': len(json.dumps(messages, ensure_ascii=False)),
            'message_count': len(messages),
        }
        self.calls.append(call)
        response = self.client.complete_messages(messages, tools=tools)
        call['usage'] = deepcopy(self.client.last_usage)
        call['response'] = deepcopy(response)
        return response


class HiddenBundleAgent(ExperimentalSkillAgent):
    """Experiment-only treatment: same state/history, no Bundle system card."""

    def render_system_context(self) -> str:
        return ('RUNTIME BODY GATE — no Bundle metadata is visible in this '
                'experiment arm.\n' + AGENT_INSTRUCTIONS
                + '\nRuntime Bundles (metadata only)\n'
                + '{"maintained_bundles":[]}')


def clone_agent(source, client, *, hidden=False):
    harness = RuntimeCapabilityHarness(
        discovery=source.harness.discovery,
        state=deepcopy(source.harness.state),
    )
    agent_type = HiddenBundleAgent if hidden else ExperimentalSkillAgent
    agent = agent_type(harness, client, max_steps=8)
    agent.history = deepcopy(source.history)
    agent._history_skill_ids = set(source._history_skill_ids)
    return agent


def run_turn(agent, client, label, task):
    client.label = label
    call_start = len(client.calls)
    answer = agent.run(task)
    calls = deepcopy(client.calls[call_start:])
    return {
        'label': label,
        'task': task,
        'answer': answer,
        'turn_audit': deepcopy(agent.turn_audit),
        'trace': deepcopy(agent.trace),
        'model_calls': calls,
        'model_call_count': len(calls),
        'input_tokens': sum(
            int(call.get('usage', {}).get('prompt_tokens', 0)) for call in calls),
        'input_chars': sum(call['input_chars'] for call in calls),
    }


def capability_needs(turn):
    return [call['arguments']['need'] for call in turn['turn_audit']['tool_calls']
            if call['tool'] == 'load_capability']


def no_repeated_presentation_capability(needs):
    """Audit requested actions, while allowing slides to name the data source."""
    repeated_actions = (
        'create powerpoint', 'edit powerpoint', 'modify powerpoint',
        'create presentation', 'edit presentation', 'modify presentation',
        'create slides', 'edit slides', 'modify slides',
        '制作 ppt', '制作ppt', '创建 ppt', '创建ppt', '修改 ppt', '修改ppt',
        '编辑 ppt', '编辑ppt', '制作演示', '创建演示', '修改演示', '编辑演示',
        '制作幻灯', '创建幻灯', '修改幻灯', '编辑幻灯',
    )
    return all(not any(action in need.casefold() for action in repeated_actions)
               for need in needs)


def summarize_group(turns):
    return {
        'turns': len(turns),
        'load_capability_turns': sum(
            turn['turn_audit']['load_capability_called'] for turn in turns),
        'load_skill_body_turns': sum(
            turn['turn_audit']['load_skill_body_called'] for turn in turns),
        'retrieval_calls': sum(
            turn['turn_audit']['retrieval_call_count_after']
            - turn['turn_audit']['retrieval_call_count_before'] for turn in turns),
        'model_calls': sum(turn['model_call_count'] for turn in turns),
        'input_tokens': sum(turn['input_tokens'] for turn in turns),
        'input_chars': sum(turn['input_chars'] for turn in turns),
    }


def main():
    output_dir = Path('local_artifacts/bundle-reuse-routing')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=output_dir)) / 'trace.json'
    report = {'started': datetime.now(timezone.utc).isoformat(),
              'live_api': True, 'failures': []}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    try:
        _validate_root_snapshot(str(ROOT), str(MANIFEST))
        cards = load_retrieval_cards(CARDS)
        registry = SkillRegistry.from_tree(ROOT)
        discovery = SkillDiscovery(registry, retrieval_cards=cards)
        glm = BigModelChatClient(timeout=60, max_tokens=4096)
        client = RecordingClient(glm)
        bootstrap = ExperimentalSkillAgent(
            RuntimeCapabilityHarness(discovery=discovery), client, max_steps=8)
        report['model'] = glm.model
        report['bootstrap'] = run_turn(
            bootstrap, client, 'bootstrap', BOOTSTRAP)
        powerpoint_bundles = [
            bundle for bundle in bootstrap.harness.state.active_bundles
            if bundle.skill_ids == ('powerpoint',)
        ]
        if not powerpoint_bundles:
            raise RuntimeError('bootstrap did not CREATE a powerpoint-only Bundle')
        report['bootstrap_bundle_id'] = powerpoint_bundles[0].bundle_id

        visible = clone_agent(bootstrap, client)
        hidden = clone_agent(bootstrap, client, hidden=True)
        report['resident_visible'] = [
            run_turn(visible, client, f'resident-visible-{index}', task)
            for index, task in enumerate(REPEATED, 1)
        ]
        report['resident_hidden'] = [
            run_turn(hidden, client, f'resident-hidden-{index}', task)
            for index, task in enumerate(REPEATED, 1)
        ]

        evicted = clone_agent(bootstrap, client)
        evicted.harness.mark_skill_body_evicted('powerpoint')
        report['evicted'] = run_turn(evicted, client, 'evicted', EVICTED)

        gap = clone_agent(bootstrap, client)
        report['gap'] = run_turn(gap, client, 'gap', GAP)
        mixed = clone_agent(bootstrap, client)
        report['mixed'] = run_turn(mixed, client, 'mixed', MIXED)
        unrelated = clone_agent(bootstrap, client)
        report['unrelated'] = run_turn(unrelated, client, 'unrelated', UNRELATED)

        visible_summary = summarize_group(report['resident_visible'])
        hidden_summary = summarize_group(report['resident_hidden'])
        report['visibility_ab'] = {
            'visible': visible_summary,
            'hidden': hidden_summary,
            'load_capability_rate_visible':
                visible_summary['load_capability_turns'] / visible_summary['turns'],
            'load_capability_rate_hidden':
                hidden_summary['load_capability_turns'] / hidden_summary['turns'],
            'retrieval_calls_saved_visible_vs_hidden':
                hidden_summary['retrieval_calls'] - visible_summary['retrieval_calls'],
            'input_tokens_saved_visible_vs_hidden':
                hidden_summary['input_tokens'] - visible_summary['input_tokens'],
            'history_confound': (
                'Both arms retain the identical bootstrap history, including the '
                'earlier powerpoint body. A zero-search hidden arm would show that '
                'history alone can support reuse and prevent a causal Bundle claim.'),
        }

        visible_failures = [turn['label'] for turn in report['resident_visible']
                            if turn['turn_audit']['load_capability_called']
                            or turn['turn_audit']['load_skill_body_called']]
        if visible_failures:
            report['failures'].append({
                'case': 'resident_visible',
                'unexpected_tool_turns': visible_failures,
            })
        evicted_audit = report['evicted']['turn_audit']
        if (evicted_audit['load_capability_called']
                or not evicted_audit['load_skill_body_called']
                or evicted_audit['body_load_count_after']
                != evicted_audit['body_load_count_before'] + 1):
            report['failures'].append({'case': 'evicted', 'audit': evicted_audit})
        for name in ('gap', 'mixed'):
            turn = report[name]
            needs = capability_needs(turn)
            if not needs or not no_repeated_presentation_capability(needs):
                report['failures'].append({
                    'case': name, 'load_capability_needs': needs,
                    'reason': 'missing discovery or presentation capability repeated',
                })
        if not report['unrelated']['turn_audit']['load_capability_called']:
            report['failures'].append({
                'case': 'unrelated', 'reason': 'new debugging gap was suppressed'})
        report['status'] = 'success' if not report['failures'] else 'behavior_mismatch'
    except Exception as exc:
        report.update(status='error', error={'type': type(exc).__name__,
                                             'message': str(exc)})
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({
            'trace': str(output), 'status': report['status'],
            'visibility_ab': report.get('visibility_ab'),
            'failures': report.get('failures'),
        }, ensure_ascii=False), flush=True)
    if report['status'] != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
