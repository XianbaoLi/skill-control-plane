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
from skill_control_plane.retrieval.cards import load_retrieval_cards
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.experimental_agent import ExperimentalSkillAgent


ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
TURN_1 = (
    '读取这个文档并做成 PPT。当前没有实际输入文件，所以先依据本环境已有 Skill 给出执行方案。'
    '这是会持续修改的演示文稿工作，请 discovery 后把 powerpoint Skill 放入一个 CREATE '
    'document Bundle；不要使用 DIRECT。'
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
        registry = SkillRegistry.from_tree(ROOT, cards=load_retrieval_cards(CARDS))
        harness = RuntimeCapabilityHarness(discovery=SkillDiscovery(registry))
        client = BigModelChatClient(timeout=60, max_tokens=4096)
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
        evicted = snapshot(harness)
        retrieval_before = harness.retrieval_call_count
        body_loads_before = harness.body_load_count
        answer = agent.run(TURN_2)
        after = snapshot(harness)
        second_tools = [event['tool'] for row in agent.trace
                        for event in row['tool_executions']]
        report['turns'].append({
            'task': TURN_2, 'answer': answer, 'before': evicted, 'after': after,
            'tool_trace': deepcopy(agent.trace),
        })
        checks = {
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
