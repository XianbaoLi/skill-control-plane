"""Frozen 13-target x 5-query experiment; reuses production retrieval evaluators."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from skill_control_plane.cli import _validate_root_snapshot, _validate_snapshot_id
from skill_control_plane.evals.control_plane import load_stage_transition_gold
from skill_control_plane.evals.query_robustness import (
    QueryVariantSet, compare_query_robustness_reports, evaluate_query_variant_retrieval,
)
from skill_control_plane.evals.representation_ablation import RETRIEVAL_CARD_FIELD_ABLATIONS
from skill_control_plane.registry import load_skill_tree
from skill_control_plane.discovery import BM25Retriever, BigModelDenseRetriever
from skill_control_plane.discovery.bigmodel import BigModelEmbeddingClient
from skill_control_plane.discovery.cards import apply_retrieval_cards, load_retrieval_cards
from skill_control_plane.discovery.dense import metadata_text

BASE = Path('local_artifacts/v0.6')
PREFIX = 'robustness-current87-13target-65query'
SOURCE = BASE / 'field-ablation-current87-13target.json'
QUERIES = BASE / f'{PREFIX}-queries.json'
GOLD = Path('evals/gold/stage-transition-v0.3.jsonl')
ROOT = Path('local_artifacts/v0.5/hermes-current87')
MANIFEST = Path('local_artifacts/v0.5/hermes-current87-manifest.json')
CARDS = Path('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl')
ARMS = ('dense', 'bm25', 'rrf')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def validate_queries(frozen, source, cases):
    assert frozen['source_sha256'] == sha(SOURCE)
    originals = {(s['case_id'], s['stage_id']): s for s in source['reports']['metadata-v0.1']['stages'] if s['new_required']}
    gold = {(c.case_id, s.stage_id): s for c in cases for s in c.stages[1:] if s.new_required}
    query_sets = {}
    for row in frozen['targets']:
        key = (row['case_id'], row['stage_id'])
        assert key not in query_sets
        assert row['original_query'] == row['variants'][0] == originals[key]['query']
        assert len(row['variants']) == len(set(row['variants'])) == 5
        assert all(isinstance(q, str) and q.strip() for q in row['variants'])
        assert list(gold[key].new_required) == originals[key]['new_required']
        assert len(gold[key].new_required) == 1
        query_sets[key] = QueryVariantSet(**{**row, 'variants': tuple(row['variants'])})
    assert set(query_sets) == set(originals) == set(gold)
    assert len(query_sets) == 13 and frozen['query_count'] == 65
    for report in source['reports'].values():
        assert (report['per_retriever_k'], report['rrf_k'], report['cutoffs']) == (10, 60, [5, 10])
        for row in report['stages']:
            if row['new_required']:
                original = originals[(row['case_id'], row['stage_id'])]
                assert row['query'] == original['query']
                assert row['new_required'] == original['new_required']
    return query_sets


class CachedEmbeddings:
    """Persist identical API inputs/outputs; no changes to similarity or ranking."""
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {
            'model': 'embedding-3', 'dimensions': 2048, 'vectors': {}}
        assert self.data['model'] == 'embedding-3' and self.data['dimensions'] == 2048
        self.client = BigModelEmbeddingClient(model='embedding-3', dimensions=2048)

    def __call__(self, texts):
        vectors = self.data['vectors']
        missing = list(dict.fromkeys(t for t in texts if t not in vectors))
        for start in range(0, len(missing), 64):
            batch = missing[start:start + 64]
            result = self.client(batch)
            assert all(len(v) == 2048 for v in result)
            vectors.update(zip(batch, result, strict=True))
            write_json(self.path, self.data)
        return [vectors[t] for t in texts]


class RecordedRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
        self.rankings = {}

    def search(self, query, k=10):
        ranking = self.retriever.search(query, k=k)
        self.rankings[query] = [asdict(candidate) for candidate in ranking]
        return ranking


def target_metrics(stage, arm):
    target, = stage['new_required']
    ranks = [v[arm]['new_required_ranks'][target] for v in stage['variants']]
    metrics = {}
    for k in (5, 10):
        metrics[f'recall_at_{k}'] = sum(r is not None and r <= k for r in ranks) / 5
        metrics[f'mrr_at_{k}'] = sum(1/r for r in ranks if r is not None and r <= k) / 5
        metrics[f'all_variants_hit_at_{k}'] = float(all(r is not None and r <= k for r in ranks))
    return {
        'ranks': ranks, 'worst_rank': None if None in ranks else max(ranks),
        'missing_from_retrieved_candidates': sum(r is None for r in ranks),
        **metrics,
    }


def pairwise(reports):
    pairs = []
    for left, right in itertools.combinations(reports, 2):
        delta = compare_query_robustness_reports(reports[left], reports[right])['delta_retrieval_card_minus_metadata']
        rows = []
        counts = {}
        for a, b in zip(reports[left]['stages'], reports[right]['stages'], strict=True):
            assert (a['case_id'], a['stage_id'], a['new_required']) == (b['case_id'], b['stage_id'], b['new_required'])
            row = {'case_id': a['case_id'], 'stage_id': a['stage_id'], 'target': a['new_required'][0], 'arms': {}}
            for arm in ARMS:
                am, bm = target_metrics(a, arm), target_metrics(b, arm)
                outcomes = {}
                for metric in ('recall_at_5', 'recall_at_10', 'mrr_at_5', 'mrr_at_10', 'all_variants_hit_at_5', 'all_variants_hit_at_10'):
                    diff = bm[metric] - am[metric]
                    outcome = 'win' if diff > 1e-12 else 'loss' if diff < -1e-12 else 'tie'
                    outcomes[metric] = {'delta': diff, 'outcome': outcome}
                    counts.setdefault(arm, {}).setdefault(metric, dict(win=0, tie=0, loss=0))[outcome] += 1
                row['arms'][arm] = outcomes
            target = row['target']
            union_outcomes = {}
            av = [target in v['union']['new_required_hits'] for v in a['variants']]
            bv = [target in v['union']['new_required_hits'] for v in b['variants']]
            for metric, va, vb in (
                ('candidate_recall', sum(av) / 5, sum(bv) / 5),
                ('all_variants_covered', float(all(av)), float(all(bv))),
            ):
                diff = vb - va
                outcome = 'win' if diff > 1e-12 else 'loss' if diff < -1e-12 else 'tie'
                union_outcomes[metric] = {'delta': diff, 'outcome': outcome}
                counts.setdefault('union', {}).setdefault(metric, dict(win=0, tie=0, loss=0))[outcome] += 1
            row['arms']['union'] = union_outcomes
            rows.append(row)
        pairs.append({'left': left, 'right': right, 'direction': 'right minus left; win means right is better', 'metric_deltas': delta, 'win_tie_loss_counts': counts, 'per_target': rows})
    return pairs


def summary_table(reports):
    lines = ['| Representation | Arm | Recall@5/10 | MRR@5/10 | Original R@5/10 | Paraphrase R@5/10 | AllVariantsHit@5/10 |', '|---|---|---|---|---|---|---|']
    for label, report in reports.items():
        for arm in ARMS:
            m = report[arm]
            values = [' / '.join(f'{m[f"{metric}_{k}"]:.4f}' for k in (5, 10)) for metric in ('recall_at', 'mrr_at', 'original_recall_at', 'paraphrase_recall_at', 'all_variants_hit_at')]
            lines.append('| ' + ' | '.join([label, arm, *values]) + ' |')
    lines += ['', 'Union is an unordered candidate pool (Dense Top-10 + BM25 Top-10); no artificial Union rank or MRR is assigned.', '', '| Representation | Union recall | Original | Paraphrase | All variants covered | Mean pool size |', '|---|---|---|---|---|---|']
    for label, report in reports.items():
        lines.append('| ' + ' | '.join([label, *(f'{report["union"][key]:.4f}' for key in ('candidate_recall', 'original_candidate_recall', 'paraphrase_candidate_recall', 'all_variants_covered', 'average_candidate_set_size'))]) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    source = json.loads(SOURCE.read_text())
    frozen = json.loads(QUERIES.read_text())
    cases = load_stage_transition_gold(GOLD)
    _validate_root_snapshot(str(ROOT), str(MANIFEST))
    for case in cases:
        _validate_snapshot_id(case.snapshot_id, str(MANIFEST))
    query_sets = validate_queries(frozen, source, cases)
    inputs = {str(p): sha(p) for p in (SOURCE, QUERIES, GOLD, MANIFEST, CARDS)}
    skills = load_skill_tree(ROOT)
    assert len(skills) == 87
    cards = load_retrieval_cards(CARDS)
    representations = {'metadata-v0.1': [replace(s, body='') for s in skills]}
    representations.update({label: apply_retrieval_cards(skills, cards, include_fields=fields) for label, fields in RETRIEVAL_CARD_FIELD_ABLATIONS})
    assert set(representations) == set(source['reports'])
    # Save every exact indexed representation for independent audit, before retrieval.
    indexed = {label: [{'skill_id': s.skill_id, 'dense_text': metadata_text(s), 'bm25_text': s.search_text} for s in rows] for label, rows in representations.items()}
    indexed_path = BASE / f'{PREFIX}-indexed-texts.json'
    write_json(indexed_path, indexed)
    inputs[str(indexed_path)] = sha(indexed_path)
    cache_path = BASE / f'{PREFIX}-embeddings.json'
    embed = CachedEmbeddings(cache_path)
    embed([q for qs in query_sets.values() for q in qs.variants])
    reports, original_checks = {}, []
    for label, rows in representations.items():
        print(f'Evaluating {label}', flush=True)
        bm25 = RecordedRetriever(BM25Retriever(rows))
        dense = RecordedRetriever(BigModelDenseRetriever(rows, model_name='embedding-3', dimensions=2048, embed_batch=embed))
        report = evaluate_query_variant_retrieval(cases, query_sets=query_sets, bm25=bm25, dense=dense, per_retriever_k=10, rrf_k=60, cutoffs=(5, 10), skill_representation=source['reports'][label]['skill_representation'])
        originals = {(s['case_id'], s['stage_id']): s for s in source['reports'][label]['stages'] if s['new_required']}
        for stage in report['stages']:
            old = originals[(stage['case_id'], stage['stage_id'])]
            stage['target_metrics'] = {arm: target_metrics(stage, arm) for arm in ARMS}
            for variant in stage['variants']:
                for arm, retriever in (('dense', dense), ('bm25', bm25)):
                    variant[arm]['ranking'] = retriever.rankings[variant['query']]
            for arm in ARMS:
                actual = stage['variants'][0][arm]['new_required_ranks']
                expected = old[arm]['new_required_ranks']
                check = {'representation': label, 'case_id': stage['case_id'], 'stage_id': stage['stage_id'], 'arm': arm, 'matches': actual == expected, 'expected': expected, 'actual': actual}
                if arm in ('dense', 'bm25'):
                    check['expected_top10'] = old[arm]['top10']
                    check['actual_top10'] = [c['skill_id'] for c in stage['variants'][0][arm]['ranking']]
                    check['top10_matches'] = check['expected_top10'] == check['actual_top10']
                original_checks.append(check)
        reports[label] = report
        write_json(BASE / f'{PREFIX}-{label}.json', report)
    assert all(sha(p) == digest for p, digest in inputs.items())
    _validate_root_snapshot(str(ROOT), str(MANIFEST))
    result = {
        'experiment': PREFIX, 'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'retrieval only; no runtime rerouting or activation; no tuning',
        'configuration': {'corpus': 'current87', 'skill_count': 87, 'snapshot_id': cases[0].snapshot_id, 'model': 'embedding-3', 'dimensions': 2048, 'per_retriever_k': 10, 'rrf_k': 60, 'cutoffs': [5, 10], 'bm25_k1': 1.5, 'bm25_b': 0.75},
        'input_sha256': inputs, 'embedding_cache_sha256': sha(cache_path),
        'queries': frozen, 'reports': reports, 'pairwise': pairwise(reports),
        'original_rank_reproduction': {'all_match': all(r['matches'] for r in original_checks), 'checks': original_checks},
        'rank_semantics': 'Dense/BM25 ranks censored at 10; RRF ranks over union of two Top-10 lists (up to 20). null worst_rank means at least one query absent from retrieved candidates, not an exact rank. Union is unordered.',
        'limitations': '13 targets with 5 correlated wordings each, not 65 independent targets. Intermediate robustness experiment; no statistical significance claim. Historical field-ablation has no embedded indexed-text hashes; current manifest/cards validated and V0 ranks compared, but historical byte identity cannot be independently established from that report alone.',
    }
    write_json(BASE / f'{PREFIX}.json', result)
    table = summary_table(reports)
    (BASE / f'{PREFIX}-summary.md').write_text(table)
    print(table)
    print('Original rank reproduction:', result['original_rank_reproduction']['all_match'])


if __name__ == '__main__':
    main()
