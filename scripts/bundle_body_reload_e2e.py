"""Opt-in live two-turn Bundle-aware Skill body reload experiment."""
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
from skill_control_plane.integrations.reference_agent import ExperimentalSkillAgent


ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
TURN_1 = (
    '读取这个文档并做成 PPT。当前没有实际输入文件，所以先依据本环境已有 Skill 给出执行方案。'
    '这是会持续修改的演示文稿工作，请 discovery 后把 powerpoint Skill 放入一个 CREATE '
    'document Bundle；不要使用 DIRECT。调用 apply_capability 时遵守现有 schema，包含非空 reason。'
)
TURN_2 = '继续修改刚才 PPT 的第三页；沿用已有 Bundle，按当前 body_state 选择正确的 native tool。'


def snapshot(harness):
    return {
        'bundles': [
            {'bundle_id': bundle.bundle_id, 'purpose': bundle.purpose,
             'skill_ids': list(bundle.skill_ids)}
            for bundle in harness.state.active_bundles
        ],
        'body_residency': dict(harness.state.skill_body_states),
        'retrieval_call_count': harness.retrieval_call_count,
        'body_load_count': harness.body_load_count,
    }


def main():
    output_dir = Path('local_artifacts/bundle-body-reload')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='live-', dir=output_dir)) / 'trace.json'
    report = {
        'started': datetime.now(timezone.utc).isoformat(),
        'live_api': True,
        'turns': [],
    }

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    try:
        _validate_root_snapshot(str(ROOT), str(MANIFEST))
        cards = load_retrieval_cards(CARDS)
        registry = SkillRegistry.from_tree(ROOT)
        harness = RuntimeCapabilityHarness(
            discovery=SkillDiscovery(registry, retrieval_cards=cards))
        chat = BigModelChatClient(timeout=60, max_tokens=4096)
        powerpoint_body = registry.load_skill_body('powerpoint')
        wire_audit = []
        phase = 'turn_1'

        def visible_body_ids(messages):
            ids = set()
            for message in messages:
                if message.get('role') != 'tool':
                    continue
                result = json.loads(message['content'])
                for body in result.get('skill_bodies', []):
                    if (body.get('skill_id') == 'powerpoint'
                            and body.get('body') == powerpoint_body):
                        ids.add('powerpoint')
                if (result.get('skill_id') == 'powerpoint'
                        and result.get('body') == powerpoint_body):
                    ids.add('powerpoint')
            return sorted(ids)

        class WireAuditClient:
            def complete_messages(self, messages, *, tools):
                rendered = json.dumps(messages, ensure_ascii=False)
                wire_audit.append({
                    'phase': phase,
                    'visible_body_ids': visible_body_ids(messages),
                    'eviction_tombstone_visible':
                        '[evicted from active context]' in rendered,
                })
                return chat.complete_messages(messages, tools=tools)

        client = WireAuditClient()
        agent = ExperimentalSkillAgent(harness, client, max_steps=8)

        before = snapshot(harness)
        answer = agent.run(TURN_1)
        after = snapshot(harness)
        report['turns'].append({
            'task': TURN_1, 'answer': answer, 'before': before, 'after': after,
            'tool_trace': deepcopy(agent.trace),
        })
        powerpoint_bundles = [
            bundle for bundle in harness.state.active_bundles
            if 'powerpoint' in bundle.skill_ids
        ]
        if not powerpoint_bundles:
            raise RuntimeError('Turn 1 did not CREATE a Bundle containing powerpoint')
        if harness.state.skill_body_states['powerpoint'] != 'resident':
            raise RuntimeError('powerpoint body was not resident after apply')

        bundle_before = deepcopy(harness.state.active_bundles)
        harness.mark_all_skill_bodies_evicted()
        canonical_body_present_before = 'powerpoint' in visible_body_ids(agent.history)
        evicted = snapshot(harness)
        retrieval_before = harness.retrieval_call_count
        body_loads_before = harness.body_load_count
        phase = 'turn_2'
        answer = agent.run(TURN_2)
        after = snapshot(harness)
        second_tools = [event['tool'] for row in agent.trace
                        for event in row['tool_executions']]
        report['turns'].append({
            'task': TURN_2, 'answer': answer, 'before': evicted, 'after': after,
            'tool_trace': deepcopy(agent.trace),
        })
        report['wire_audit'] = wire_audit
        turn_two_wire = [row for row in wire_audit if row['phase'] == 'turn_2']
        checks = {
            'canonical_body_retained_after_eviction': canonical_body_present_before,
            'first_turn_2_call_redacts_old_body': bool(turn_two_wire)
                and 'powerpoint' not in turn_two_wire[0]['visible_body_ids']
                and turn_two_wire[0]['eviction_tombstone_visible'],
            'post_reload_call_sees_body': len(turn_two_wire) >= 2
                and 'powerpoint' in turn_two_wire[-1]['visible_body_ids'],
            'turn_2_zero_retrieval': harness.retrieval_call_count == retrieval_before,
            'turn_2_one_exact_body_load': harness.body_load_count == body_loads_before + 1,
            'turn_2_only_load_skill_body': second_tools == ['load_skill_body'],
            'powerpoint_resident_after':
                harness.state.skill_body_states['powerpoint'] == 'resident',
            'bundle_unchanged': harness.state.active_bundles == bundle_before,
        }
        report['checks'] = checks
        report['counterfactual_baseline'] = {
            'kind': 'derived, not executed',
            'without_exact_reload_minimum_retrieval_calls': 1,
            'without_exact_reload_minimum_candidate_payloads': 1,
            'without_exact_reload_minimum_additional_model_calls': 3,
            'exact_reload_retrieval_calls': 0,
            'exact_reload_candidate_payloads': 0,
            'exact_reload_additional_model_calls': len(agent.trace),
        }
        report['status'] = 'success' if all(checks.values()) else 'expectation_mismatch'
    except Exception as exc:
        report.update(status='error', error={'type': type(exc).__name__,
                                             'message': str(exc)})
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({'trace': str(output), 'status': report['status'],
                          'checks': report.get('checks')}, ensure_ascii=False))
    if report['status'] != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
