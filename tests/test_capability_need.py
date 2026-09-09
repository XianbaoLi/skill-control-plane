import json
import subprocess
from dataclasses import replace

import pytest

from skill_control_plane.cli import _build_parser, main
from skill_control_plane.evals.control_plane import StageGold, StageTransitionGoldCase
from skill_control_plane.evals.hierarchical import (
    evaluate_capability_need_cases, evaluate_hierarchical_cases,
    print_capability_need_report,
)
from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.runtime.capability_need import (
    CapabilityNeed, LLMCapabilityNeedExtractor, build_capability_need_prompt,
    command_completer, extract_query,
)
from skill_control_plane.runtime.hierarchical import ShelfAwareRetriever


class Search:
    def __init__(self, ids):
        self.ids = ids
    def search(self, query, k=5):
        return [RetrievalCandidate(s, .9, i+1) for i, s in enumerate(self.ids[:k])]


class Extractor:
    def __init__(self, result=CapabilityNeed('new semantic action', .9)):
        self.result = result
        self.calls = []
    def extract(self, initial_task, runtime_evidence):
        self.calls.append((initial_task, tuple(runtime_evidence)))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def inputs():
    records = {s: SkillRecord(s, s, s, '', '', category=g)
               for s, g in [('old', 'ongoing'), ('new', 'different')]}
    case = StageTransitionGoldCase(
        'SECRET_CASE_ID', 'initial background',
        (StageGold('S1', (), ('old',), ('old',), (), ()),
         StageGold('S2', ('raw fact',), ('new',), ('new',), ('SECRET_USEFUL',), ('SECRET_NEGATIVE',))),
        'SECRET_RATIONALE', 'SECRET_SNAPSHOT',
    )
    search = Search(['old'])
    kwargs = dict(bm25=search, dense=search,
                  dense_factory=lambda rs: Search([rs[0].skill_id]))
    return records, case, kwargs


def test_only_runtime_inputs_reach_extractor_and_need_reaches_matcher(monkeypatch):
    records, case, kwargs = inputs()
    calls = []
    original = ShelfAwareRetriever.search
    def spy(self, query, shelf, **options):
        calls.append((query, options, self.min_score, self.min_margin))
        return original(self, query, shelf, **options)
    monkeypatch.setattr(ShelfAwareRetriever, 'search', spy)
    extractor = Extractor()
    report = evaluate_capability_need_cases([case], records, extractor=extractor, **kwargs)
    assert extractor.calls == [('initial background', ('raw fact',))]
    assert [c[0] for c in calls] == ['initial background', 'raw fact', 'initial background', 'new semantic action']
    assert all(c[2:] == (.35, .05) for c in calls)
    assert calls[-1][1]['hierarchical'] is True
    r1 = report['R1']['cases'][0]['stages']
    assert 'capability_need' not in r1[0]
    assert r1[1]['raw_evidence'] == ['raw fact']
    assert r1[1]['capability_need'] == 'new semantic action'


def test_prompt_has_only_background_and_evidence_not_gold():
    records, case, kwargs = inputs()
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        return json.dumps({'capability_need': 'inspect a fault', 'confidence': None})
    extractor = LLMCapabilityNeedExtractor(complete)
    evaluate_capability_need_cases([case], records, extractor=extractor, **kwargs)
    changed = replace(case, case_id='OTHER', rationale='OTHER', snapshot_id='OTHER',
                      stages=(case.stages[0], replace(case.stages[1], required_now=('old',), new_required=('old',))))
    evaluate_capability_need_cases([changed], records, extractor=extractor, **kwargs)
    assert prompts[0] == prompts[1]
    payload = json.loads(prompts[0].split('\nInput:\n')[1])
    assert payload == {'initial_task': 'initial background', 'runtime_evidence': ['raw fact']}
    assert 'SECRET_' not in prompts[0]
    assert 'semantic background only' in prompts[0]
    assert "outside the ongoing task's domain" in prompts[0]
    assert 'not the original task' in prompts[0]


@pytest.mark.parametrize('result,status', [
    (CapabilityNeed('', .9), 'empty'),
    (CapabilityNeed('  ', .9), 'empty'),
    (CapabilityNeed('uncertain', .2), 'low-confidence'),
    (RuntimeError('SECRET_API_KEY'), 'error:RuntimeError'),
    (None, 'error:ValueError'),
    (CapabilityNeed('bad', float('nan')), 'error:ValueError'),
    (CapabilityNeed('bad', 2), 'error:ValueError'),
])
def test_unusable_extraction_forces_raw_global_fallback(result, status, monkeypatch):
    records, case, kwargs = inputs()
    seen = []
    original = ShelfAwareRetriever.search
    def spy(self, query, shelf, **options):
        seen.append((query, options['hierarchical']))
        return original(self, query, shelf, **options)
    monkeypatch.setattr(ShelfAwareRetriever, 'search', spy)
    report = evaluate_capability_need_cases([case], records, extractor=Extractor(result), **kwargs)
    stage = report['R1']['cases'][0]['stages'][1]
    assert seen[-1] == ('raw fact', False)
    assert stage['route'] == 'global'
    assert stage['matcher_corpus_size'] == 0
    assert stage['extraction_status'] == status
    assert 'SECRET_API_KEY' not in json.dumps(report)


@pytest.mark.parametrize('response', ['not json', '{}', '[]', '{"capability_need":1}',
                                       '{"capability_need":"need","confidence":true}'])
def test_malformed_model_output_falls_back(response):
    result = extract_query(LLMCapabilityNeedExtractor(lambda _: response), 'background', ['evidence'])
    assert result.query == 'evidence'
    assert not result.allow_bundle_match
    assert result.status.startswith('error:')


def test_unknown_confidence_is_audited_and_does_not_block_valid_need():
    result = extract_query(Extractor(CapabilityNeed('inspect the failure', None)), 'task', ['fact'])
    assert result.allow_bundle_match
    assert result.need.confidence is None


def test_r0_reproduces_prior_hierarchical_and_runtime_denominators(capsys):
    records, case, kwargs = inputs()
    old = evaluate_hierarchical_cases([case], records, **kwargs)['hierarchical']
    report = evaluate_capability_need_cases([case], records, extractor=Extractor(), **kwargs)
    r0 = report['R0']
    for before, after in zip(old['cases'][0]['stages'], r0['cases'][0]['stages']):
        assert all(after[k] == v for k, v in before.items())
    # S1 recall is 1, transition recall is 0: new experiment excludes S1.
    assert old['mean_shelf_required_recall'] == .5
    assert r0['mean_shelf_required_recall'] == 0
    assert r0['required_skill_recall'] == 0
    assert r0['bundle_local_retrieval_rate'] == 1
    assert r0['global_retrieval_rate'] == 0
    assert r0['wrong_bundle_match_rate'] == 1
    assert r0['mean_registered_surface_growth'] == 0
    assert r0['mean_final_registered_skill_count'] == 1
    assert r0['mean_search_corpus_size'] == 1
    assert r0['mean_matcher_corpus_size'] == 1
    print_capability_need_report(report)
    output = capsys.readouterr().out
    for field in ['raw_evidence', 'capability_need', 'extractor_confidence', 'match_margin', 'missing_shelf_ids']:
        assert field in output


def test_initial_only_trajectory_does_not_call_extractor():
    records, case, kwargs = inputs()
    extractor = Extractor(RuntimeError('must not call'))
    report = evaluate_capability_need_cases([replace(case, stages=case.stages[:1])], records,
                                            extractor=extractor, **kwargs)
    assert extractor.calls == []
    assert report['R1']['transition_count'] == 0
    assert report['R1']['global_retrieval_rate'] == 0
    assert report['R1']['mean_shelf_required_recall'] == 0


def test_cli_experiment_flags_mutually_exclusive_and_default_unchanged():
    args = ['eval', 'stage-bundle', '/skills', '--gold', 'gold', '--manifest', 'manifest']
    default = _build_parser().parse_args(args)
    assert not default.capability_need_ab and not default.hierarchical_ab
    assert default.capability_need_command is None
    with pytest.raises(SystemExit):
        _build_parser().parse_args(args + ['--hierarchical-ab', '--capability-need-ab'])
    with pytest.raises(ValueError, match='requires --capability-need-command'):
        main(args + ['--capability-need-ab'])


def test_command_completer_no_shell_and_prompt_only_on_stdin(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, '{"capability_need":"inspect","confidence":0.8}')
    monkeypatch.setattr(subprocess, 'run', run)
    complete = command_completer('provider --config "some path"')
    assert json.loads(complete('raw $(do-not-execute)'))['capability_need'] == 'inspect'
    assert calls[0][0] == ['provider', '--config', 'some path']
    assert calls[0][1]['input'] == 'raw $(do-not-execute)'
    assert calls[0][1].get('shell', False) is False
    assert calls[0][1]['timeout'] == 120


def test_provider_timeout_is_safe_global_fallback():
    def complete(_):
        raise subprocess.TimeoutExpired('provider', 120)
    result = extract_query(LLMCapabilityNeedExtractor(complete), 'task', ['raw'])
    assert result.status == 'error:TimeoutExpired'
    assert result.query == 'raw' and not result.allow_bundle_match


@pytest.mark.parametrize('tool_use', [False, True])
def test_codex_adapter_audits_text_and_rejects_tool_use(monkeypatch, tmp_path, capsys, tool_use):
    import importlib.util
    import io
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('need_adapter',
        Path(__file__).parents[1] / 'scripts' / 'capability_need_codex.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def run(argv, **kwargs):
        assert kwargs['input'] == 'isolated prompt'
        assert 'features.shell_tool=false' in argv
        assert 'features.skip_host_skill_discovery=true' in argv
        assert 'project_doc_max_bytes=0' in argv
        assert '--ignore-user-config' in argv and '--ephemeral' in argv
        output = Path(argv[argv.index('-o') + 1])
        output.write_text('{"capability_need":"inspect"}')
        event = {'type': 'item.completed', 'item': {'type': 'command_execution' if tool_use else 'agent_message'}}
        return subprocess.CompletedProcess(argv, 0, json.dumps(event) + '\n', '')
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['adapter', '--audit-dir', str(tmp_path)])
    monkeypatch.setattr(sys, 'stdin', io.StringIO('isolated prompt'))
    if tool_use:
        with pytest.raises(RuntimeError, match='used a tool'):
            module.main()
    else:
        module.main()
        assert json.loads(capsys.readouterr().out)['capability_need'] == 'inspect'
    audit = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert audit['prompt'] == 'isolated prompt'


def test_exact_prompt_replay_never_calls_provider(monkeypatch, tmp_path, capsys):
    import hashlib
    import importlib.util
    import io
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('need_replay',
        Path(__file__).parents[1] / 'scripts' / 'capability_need_codex.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt = 'unchanged input'
    key = hashlib.sha256(prompt.encode()).hexdigest()
    artifact = tmp_path / f'{key}.json'
    artifact.write_text(json.dumps({'prompt': prompt, 'returncode': 0,
        'events': [{'item': {'type': 'agent_message'}}],
        'completion': '{"capability_need":"inspect"}'}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Replay must not call provider')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    monkeypatch.setattr(sys, 'argv', ['adapter', '--replay-dir', str(tmp_path)])
    monkeypatch.setattr(sys, 'stdin', io.StringIO(prompt))
    module.main()
    assert json.loads(capsys.readouterr().out)['capability_need'] == 'inspect'
    data = json.loads(artifact.read_text())
    data['prompt'] = 'different input'
    artifact.write_text(json.dumps(data))
    monkeypatch.setattr(sys, 'stdin', io.StringIO(prompt))
    with pytest.raises(ValueError, match='exact successful prompt'):
        module.main()
