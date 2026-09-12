"""Frozen hard evaluation; no query generation, retriever tuning or activation."""
from __future__ import annotations
import json
from dataclasses import asdict, replace
from pathlib import Path
import robustness_current87 as prior
from skill_control_plane.discovery import candidate_union, reciprocal_rank_fusion

BASE = prior.BASE
PREFIX = 'hard-robustness-current87-13target-39query'
QUERIES = BASE / f'{PREFIX}-queries.json'
PREVIOUS = BASE / f'{prior.PREFIX}.json'
KINDS = ('H1', 'H2', 'H3')
ARMS = prior.ARMS
COMMAND = 'set -a; source .env; set +a; PYTHONPATH=src .venv/bin/python scripts/hard_robustness_current87.py'


def metrics(ranks):
    return {f'{metric}_at_{k}': value for k in (5, 10) for metric, value in (
        ('recall', sum(r is not None and r <= k for r in ranks)/len(ranks)),
        ('mrr', sum(1/r for r in ranks if r is not None and r <= k)/len(ranks)))}


def validate(frozen, previous, cases):
    assert frozen['source_sha256'] == prior.sha(prior.QUERIES)
    assert frozen['query_count'] == 39 and len(frozen['targets']) == 13
    gold = {(c.case_id, s.stage_id): list(s.new_required) for c in cases for s in c.stages[1:] if s.new_required}
    seen = set()
    for row in frozen['targets']:
        key = row['case_id'], row['stage_id']
        assert key not in seen
        seen.add(key)
        assert gold[key] == [row['target']]
        assert [v['kind'] for v in row['variants']] == list(KINDS)
        qs = [v['query'] for v in row['variants']]
        assert len(set(qs)) == 3 and all(isinstance(q, str) and q.strip() for q in qs)
    assert seen == set(gold)
    for p, digest in previous['input_sha256'].items():
        assert prior.sha(p) == digest, f'Frozen input drift: {p}'
    expected = dict(corpus='current87', skill_count=87, model='embedding-3', dimensions=2048,
                    per_retriever_k=10, rrf_k=60, cutoffs=[5,10], bm25_k1=1.5, bm25_b=.75)
    assert all(previous['configuration'][k] == v for k,v in expected.items())


def rank_comparison(full, ablated, arm):
    # Exact deltas only when both ranks are observed. Capped deltas are explicit.
    cap = 21 if arm == 'rrf' else 11
    delta = None if full is None or ablated is None else ablated-full
    capped = (cap if ablated is None else ablated)-(cap if full is None else full)
    return dict(rank_delta_ablation_minus_full=delta, capped_rank_delta=capped,
                outcome='win' if capped>0 else 'loss' if capped<0 else 'tie',
                top5_crossing='full_only' if full is not None and full<=5 and (ablated is None or ablated>5)
                else 'ablation_only' if ablated is not None and ablated<=5 and (full is None or full>5) else None)


def aggregate(stages):
    report = {}
    for arm in ARMS:
        groups = [[v[arm]['rank'] for v in s['variants']] for s in stages]
        report[arm] = {**metrics([r for group in groups for r in group]),
            **{f'all_variants_hit_at_{k}': sum(all(r is not None and r<=k for r in g) for g in groups)/len(groups) for k in (5,10)}}
    variants = [v for s in stages for v in s['variants']]
    report['union'] = dict(candidate_recall=sum(v['union']['hit'] for v in variants)/len(variants),
        average_candidate_pool_size=sum(v['union']['candidate_set_size'] for v in variants)/len(variants))
    report['by_kind'] = {kind:{arm:metrics([v[arm]['rank'] for v in variants if v['kind']==kind]) for arm in ARMS} for kind in KINDS}
    return report


def analyze(reports, previous):
    degradation, attribution = {}, {}
    for label, report in reports.items():
        old = previous['reports'][label]
        degradation[label] = {}
        for arm in ARMS:
            original = metrics([s['variants'][0][arm]['new_required_ranks'][s['new_required'][0]] for s in old['stages']])
            degradation[label][arm] = {'original':original, 'hard':{k:report[arm][k] for k in original},
                'degradation':{k:val-report[arm][k] for k,val in original.items()}}
    full = reports['full']
    for label, report in reports.items():
        if not label.startswith('minus-'):
            continue
        attribution[label] = {}
        for arm in ARMS:
            details = []
            for a,b in zip(full['stages'], report['stages'], strict=True):
                assert a['target'] == b['target']
                for av,bv in zip(a['variants'],b['variants'],strict=True):
                    details.append(dict(target=a['target'],kind=av['kind'], full_rank=av[arm]['rank'], ablation_rank=bv[arm]['rank'],
                                        **rank_comparison(av[arm]['rank'],bv[arm]['rank'],arm)))
            observed = [d['rank_delta_ablation_minus_full'] for d in details if d['rank_delta_ablation_minus_full'] is not None]
            attribution[label][arm] = dict(win_tie_loss={o:sum(d['outcome']==o for d in details) for o in ('win','tie','loss')},
                mean_rank_delta=sum(observed)/len(observed) if observed else None, comparable_count=len(observed),
                mean_capped_rank_delta=sum(d['capped_rank_delta'] for d in details)/len(details),
                top5_boundary_crossings=[d for d in details if d['top5_crossing']],
                driving_targets=sorted({d['target'] for d in details if d['outcome']!='tie'}), per_query=details)
    return degradation, attribution


def markdown(result):
    lines = ['# Frozen 39-query hard robustness evaluation', '',
        '13 targets × H1/H2/H3 × 6 representations = 234 query/representation evaluations. No input or retrieval tuning.', '',
        'H1: lexical abstraction; H2: natural operational wording; H3: terminology shift / distractor resistance.', '',
        '## Overall', '', '| Representation | Retriever | Recall@5/10 | MRR@5/10 | AllVariantsHit@5/10 |', '|---|---|---|---|---|']
    for label,r in result['reports'].items():
        for arm in ARMS:
            cells = [' / '.join(f'{r[arm][f"{m}_at_{k}"]:.4f}' for k in (5,10)) for m in ('recall','mrr','all_variants_hit')]
            lines.append('| '+' | '.join([label,arm,*cells])+' |')
    lines += ['', '| Representation | Union candidate recall | Average pool size |', '|---|---|---|']
    for label,r in result['reports'].items():
        lines.append(f'| {label} | {r["union"]["candidate_recall"]:.4f} | {r["union"]["average_candidate_pool_size"]:.4f} |')
    lines += ['', '## Hard degradation (saved V0 minus hard)', '', '| Representation | Dense R@5 | Dense MRR@5 | RRF R@5 | RRF MRR@5 |','|---|---|---|---|---|']
    for label,d in result['hard_degradation'].items():
        lines.append('| '+' | '.join([label,*[f'{d[a]["degradation"][m]:+.4f}' for a in ('dense','rrf') for m in ('recall_at_5','mrr_at_5')]])+' |')
    lines += ['', '## H1/H2/H3 metrics', '', '| Representation | Kind | Retriever | Recall@5/10 | MRR@5/10 |','|---|---|---|---|---|']
    for label,r in result['reports'].items():
        for kind,arms in r['by_kind'].items():
            for arm,m in arms.items():
                lines.append('| '+' | '.join([label,kind,arm,*[' / '.join(f'{m[f"{metric}_at_{k}"]:.4f}' for k in (5,10)) for metric in ('recall','mrr')]])+' |')
    lines += ['', '## Field attribution', '', result['rank_semantics'], '',
        'Win means full has a better rank; positive delta = ablation rank − full rank. Mean exact delta uses only jointly observed ranks. Capped mean assigns MISS 11 for Dense/BM25 and 21 for RRF; this is a reporting convention, not an inferred rank. Two MISS values count as a capped tie.', '',
        '| Ablation | Retriever | Full win/tie/loss | Mean exact delta (n) | Capped mean | Top-5 full-only / ablation-only | Driving targets |', '|---|---|---|---|---|---|---|']
    for label,arms in result['field_attribution'].items():
        for arm,d in arms.items():
            w='/'.join(str(d['win_tie_loss'][o]) for o in ('win','tie','loss'))
            crossings='/'.join(str(sum(x['top5_crossing']==v for x in d['top5_boundary_crossings'])) for v in ('full_only','ablation_only'))
            lines.append(f'| {label} | {arm} | {w} | {d["mean_rank_delta"]:.4f} ({d["comparable_count"]}) | {d["mean_capped_rank_delta"]:.4f} | {crossings} | {", ".join(d["driving_targets"])} |')
    for target in [s['target'] for s in result['reports']['full']['stages']]:
        lines += ['', f'## Target: {target}', '', '| Representation | Retriever | H1 | H2 | H3 | Worst | MISS count | AllHardHit@5/10 | Delta vs full H1/H2/H3 |','|---|---|---|---|---|---|---|---|---|']
        full = next(s for s in result['reports']['full']['stages'] if s['target']==target)
        # Queries precede the rank table in the qualitative section below for special cases.
        for label,r in result['reports'].items():
            stage = next(s for s in r['stages'] if s['target']==target)
            for arm in ARMS:
                ranks=[v[arm]['rank'] for v in stage['variants']]
                delta=[rank_comparison(f[arm]['rank'],v[arm]['rank'],arm)['rank_delta_ablation_minus_full'] for f,v in zip(full['variants'],stage['variants'],strict=True)]
                show=lambda x: 'MISS' if x is None else str(x)
                allhit='/'.join(str(int(all(x is not None and x<=k for x in ranks))) for k in (5,10))
                lines.append('| '+' | '.join([label,arm,*map(show,ranks),show(None if None in ranks else max(ranks)),str(ranks.count(None)),allhit,' / '.join('censored' if d is None else f'{d:+d}' for d in delta)])+' |')
        if target in ('python-debugpy','google-workspace'):
            lines += ['', '### Qualitative hard case (false positives relative to frozen single-target Gold)', '']
            for v in full['variants']:
                lines += [f'**{v["kind"]}**: {v["query"]}', '']
                for arm in ARMS:
                    fp=[f'{c["skill_id"]} (#{c["rank"]})' for c in v[arm]['ranking'][:5] if c['skill_id']!=target]
                    lines.append(f'- full {arm}: '+', '.join(fp))
                lines.append('')
    lines += ['', '## Interpretation', '', *result.get('findings', []), '', '## Reproducibility', '', '```bash',COMMAND,'```','',
        'Frozen input hashes, saved V0 provenance, exact ranks and candidates, crossings and deltas are in the JSON companion.', '',
        result['limitations'], '', 'Tests are recorded in the companion tests log; all failures are retained without query or representation edits.']
    return '\n'.join(lines)+'\n'


def main():
    frozen=json.loads(QUERIES.read_text()); previous=json.loads(PREVIOUS.read_text())
    cases=prior.load_stage_transition_gold(prior.GOLD)
    validate(frozen,previous,cases)
    prior._validate_root_snapshot(str(prior.ROOT),str(prior.MANIFEST))
    skills=prior.load_skill_tree(prior.ROOT); assert len(skills)==87
    cards=prior.load_retrieval_cards(prior.CARDS)
    representations={'metadata-v0.1':[replace(s,body='') for s in skills]}
    representations.update({label:prior.apply_retrieval_cards(skills,cards,include_fields=fields) for label,fields in prior.RETRIEVAL_CARD_FIELD_ABLATIONS})
    indexed={label:[dict(skill_id=s.skill_id,dense_text=prior.metadata_text(s),bm25_text=s.search_text) for s in rows] for label,rows in representations.items()}
    assert indexed==json.loads((BASE/f'{prior.PREFIX}-indexed-texts.json').read_text())
    old_cache=BASE/f'{prior.PREFIX}-embeddings.json'
    assert prior.sha(old_cache)==previous['embedding_cache_sha256']
    inputs={**previous['input_sha256'],str(PREVIOUS):prior.sha(PREVIOUS),str(QUERIES):prior.sha(QUERIES),str(old_cache):prior.sha(old_cache)}
    cache=BASE/f'{PREFIX}-embeddings.json'
    if not cache.exists():
        cache.write_bytes(old_cache.read_bytes())
    embed=prior.CachedEmbeddings(cache)
    old_vectors=json.loads(old_cache.read_text())['vectors']
    assert all(embed.data['vectors'].get(t)==v for t,v in old_vectors.items())
    assert all(s['dense_text'] in old_vectors for rows in indexed.values() for s in rows)
    embed([v['query'] for row in frozen['targets'] for v in row['variants']])
    reports={}
    for label,rows in representations.items():
        print(f'Evaluating {label}',flush=True)
        dense=prior.BigModelDenseRetriever(rows,model_name='embedding-3',dimensions=2048,embed_batch=embed)
        bm25=prior.BM25Retriever(rows,k1=1.5,b=.75)
        stages=[]
        for row in frozen['targets']:
            variants=[]
            for v in row['variants']:
                dr=dense.search(v['query'],k=10); br=bm25.search(v['query'],k=10)
                sources={'bm25':br,'dense':dr}; rr=reciprocal_rank_fusion(sources,k=60); union=candidate_union(sources)
                variants.append({**v,**{arm:dict(rank=next((c.rank for c in ranking if c.skill_id==row['target']),None),ranking=[asdict(c) for c in ranking]) for arm,ranking in (('dense',dr),('bm25',br),('rrf',rr))},
                    'union':dict(hit=any(c.skill_id==row['target'] for c in union),candidate_set_size=len(union),skill_ids=[c.skill_id for c in union])})
            stages.append({**row,'variants':variants,'target_metrics':{arm:{**metrics([v[arm]['rank'] for v in variants]),'worst_rank':None if any(v[arm]['rank'] is None for v in variants) else max(v[arm]['rank'] for v in variants),'miss_count':sum(v[arm]['rank'] is None for v in variants),**{f'all_hard_variants_hit_at_{k}':all(v[arm]['rank'] is not None and v[arm]['rank']<=k for v in variants) for k in (5,10)}} for arm in ARMS}})
        reports[label]={**aggregate(stages),'stages':stages}
    degradation,attribution=analyze(reports,previous)
    assert all(prior.sha(p)==h for p,h in inputs.items())
    prior._validate_root_snapshot(str(prior.ROOT),str(prior.MANIFEST))
    result=dict(experiment=PREFIX,configuration=previous['configuration'],query_count=39,evaluation_count=234,input_sha256=inputs,
        embedding_cache_sha256=prior.sha(cache),queries=frozen,reports=reports,hard_degradation=degradation,field_attribution=attribution,
        exact_command=COMMAND,original_baseline=str(PREVIOUS)+' reports[*].stages[*].variants[0]',
        rank_semantics='Dense/BM25 ranks are censored at 10; RRF ranks cover the union of Top-10 lists (up to 20). MISS/null is absence from that pool. Union is unordered; no Union MRR or arbitrary Top-k rank.',
        limitations='13 targets with correlated hard variants, not 39 independent targets. Hard labels are author-defined; no significance claim. Saved V0 from the 65-query run is used directly. That run recorded one historical RRF target-rank mismatch (minus-capabilities google-workspace 3→2); no baseline is re-embedded or overwritten. Indexed strings, cards, Gold, corpus and cached document vectors match the preceding frozen run.')
    prior.write_json(BASE/f'{PREFIX}.json',result)
    (BASE/f'{PREFIX}-summary.md').write_text(markdown(result))
    print('Completed 234 evaluations.',flush=True)

if __name__=='__main__':
    main()
