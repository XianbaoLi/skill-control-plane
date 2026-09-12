"""Opt-in bounded soft priors. Runtime inputs never include evaluation labels."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from itertools import zip_longest

from skill_control_plane.discovery import BM25Retriever, candidate_union
from skill_control_plane.runtime.bundles import default_group_key
from skill_control_plane.runtime.capability_need import extract_query

# Ordered, deliberately narrow indicators; ambiguous CI failure emits no prior.
PRIOR_RULES = (
    ('github', r'\bpull request\b|\bPR\s*#?\d+\b|\breview (?:the )?diff\b|\bgithub\b'),
    ('software-development', r'\bpytest\b|\btraceback\b|\bbreakpoint\b|\bdebugger\b|\bstack trace\b'),
    ('email', r'\bemail\b|\binbox\b|\bsender\b'),
    ('productivity', r'\bcalendar\b|\bmeeting invite\b|\binvitation\b'),
)


@dataclass(frozen=True)
class SoftBudget:
    global_k: int = 3
    local_k: int = 2
    prior_limit: int = 2
    local_limit: int = 4

    def __post_init__(self):
        if min(self.global_k, self.local_k, self.prior_limit, self.local_limit) < 1:
            raise ValueError('Soft budgets must be positive')


def explicit_priors(query, available, limit=2):
    return [group for group, pattern in PRIOR_RULES
            if group in available and re.search(pattern, query, re.I)][:limit]


class SoftBundleRetriever:
    def __init__(self, records, *, bm25, dense, dense_factory, budget=SoftBudget()):
        self.records, self.bm25, self.dense = records, bm25, dense
        self.factory, self.budget = dense_factory, budget
        self.groups = {default_group_key(r) for r in records.values()}
        self.local = {}

    def search(self, query):
        b = self.budget
        priors = explicit_priors(query, self.groups, b.prior_limit)
        gb = self.bm25.search(query, k=b.global_k)
        gd = self.dense.search(query, k=b.global_k)
        global_pool = candidate_union({'bm25': gb, 'dense': gd})
        local_pools, local_sizes = [], []
        for group in priors:
            if group not in self.local:
                subset = [r for r in self.records.values() if default_group_key(r) == group]
                self.local[group] = (BM25Retriever([replace(r, body='') for r in subset]),
                                     self.factory(subset), len(subset))
            bm, dn, size = self.local[group]
            local_sizes.append(size)
            local_pools.append(candidate_union({'bm25': bm.search(query, k=b.local_k),
                                                'dense': dn.search(query, k=b.local_k)}))
        # Round-robin priors before applying the shared local budget.
        interleaved = [c for row in zip_longest(*local_pools) for c in row if c is not None]
        local_pool = candidate_union({'local': interleaved})[:b.local_limit]
        pool = candidate_union({'local': local_pool, 'global': global_pool})
        overlap = set(c.skill_id for c in gb) & set(c.skill_id for c in gd)
        top = max((c.score for c in gd), default=0.0)
        reasons = []
        if top < .35:
            reasons.append('global-dense-top-below-0.35')
        if not overlap:
            reasons.append('global-bm25-dense-no-overlap')
        if len(pool) < 2:
            reasons.append('pool-below-2')
        return pool, {
            'query': query, 'priors': priors, 'global_lane': True,
            'local_retrieval': bool(priors), 'local_candidate_count': len(local_pool),
            'global_candidate_count': len(global_pool),
            'local_candidate_ids': [c.skill_id for c in local_pool],
            'global_candidate_ids': [c.skill_id for c in global_pool],
            'local_unbudgeted_count': len(interleaved),
            'deduped_candidate_count': len(pool), 'candidate_pool_size': len(pool),
            'search_corpus_size': len(self.records),
            'search_record_visits': len(self.records) + sum(local_sizes),
            'global_dense_top_score': top, 'global_retriever_overlap': len(overlap),
            'uncertainty_reasons': reasons,
        }

    def retrieve(self, initial_task, runtime_evidence, *, extractor=None):
        first, trace = self.search('\n'.join(runtime_evidence))
        trace.update(first_candidate_ids=[c.skill_id for c in first], repair_triggered=False,
                     rewrite_calls=0, retrieval_passes=1, repair_status='not-triggered',
                     repair_candidate_ids=[], repair_added_ids=[], passes=[])
        trace['passes'] = [{k: v for k, v in trace.items() if k != 'passes'}]
        if extractor is None or not trace['uncertainty_reasons']:
            return first, trace
        trace.update(repair_triggered=True, rewrite_calls=1)
        extraction = extract_query(extractor, initial_task, runtime_evidence)
        trace.update(repair_status=extraction.status, capability_need=(
            extraction.need.capability_need if extraction.need else None))
        if extraction.status != 'ok':
            return first, trace
        try:
            second, second_trace = self.search(extraction.query)
        except Exception as exc:
            trace.update(repair_status=f'retrieval-error:{type(exc).__name__}', retrieval_passes=2)
            return first, trace
        merged = candidate_union({'first': first, 'repair': second})
        first_ids = {c.skill_id for c in first}
        trace.update(retrieval_passes=2, repair_candidate_ids=[c.skill_id for c in second],
                     repair_added_ids=[c.skill_id for c in second if c.skill_id not in first_ids],
                     candidate_pool_size=len(merged), deduped_candidate_count=len(merged),
                     search_record_visits=trace['search_record_visits'] + second_trace['search_record_visits'])
        trace['passes'].append(second_trace)
        return merged, trace
