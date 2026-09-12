"""ABC sequential and A-pre-state frozen ablations, scoring after retrieval."""
from dataclasses import asdict

from skill_control_plane.evals.hierarchical import evaluate_hierarchical_cases
from skill_control_plane.evals.legacy.stage_bundles import (
    CapabilityShelf, build_capability_shelf, integrate_retrieval_delta, default_group_key,
)
from skill_control_plane.evals.legacy.hierarchical import ShelfAwareRetriever
from skill_control_plane.evals.legacy.soft_bundle import SoftBundleRetriever, SoftBudget


def active(shelf):
    return {s for b in shelf.bundles if b.bundle_id in shelf.active_bundle_ids for s in b.skill_ids}


def score(stage, candidates, before, after, records):
    ids = {c.skill_id for c in candidates}
    required = set(stage.required_now)
    registered, activated = set(after.registered_skill_ids), active(after)
    prior_groups = {b.bundle_id for b in before.bundles}
    failures = {}
    for sid in sorted(required - activated):
        if sid in registered:
            kind = 'Failure-D'
        elif sid in ids:
            kind = 'Failure-B'
        elif default_group_key(records[sid]) in prior_groups:
            kind = 'Failure-C'
        else:
            kind = 'Failure-A'
        failures[sid] = kind
    recall = lambda found: len(required & found) / len(required) if required else 1.0
    return dict(stage_id=stage.stage_id, candidate_ids=[c.skill_id for c in candidates],
                candidate_pool_size=len(ids), required_skill_candidate_recall=recall(ids),
                required_skill_shelf_recall=recall(registered), required_skill_active_recall=recall(activated),
                missing_candidate_ids=sorted(required-ids), missing_shelf_ids=sorted(required-registered),
                missing_active_ids=sorted(required-activated), failures=failures,
                pre_shelf=asdict(before), post_shelf=asdict(after),
                registered_surface_growth=len(registered)-len(before.registered_skill_ids),
                active_surface_growth=len(activated)-len(active(before)),
                registered_skill_count=len(registered), active_skill_count=len(activated))


def summarize(rows, finals=()):
    mean = lambda key: sum(r.get(key, 0) for r in rows)/len(rows) if rows else 0.0
    result = {'transition_count': len(rows)}
    for key in ('required_skill_candidate_recall', 'required_skill_shelf_recall',
                'required_skill_active_recall', 'local_candidate_count', 'global_candidate_count'):
        result[key] = mean(key)
    for key in ('candidate_pool_size', 'search_corpus_size', 'search_record_visits',
                'registered_surface_growth', 'active_surface_growth', 'retrieval_passes'):
        result['mean_'+key] = mean(key)
    result.update(prior_rule_hit_rate=mean('prior_hit'), mean_prior_bundle_count=mean('prior_count'),
                  local_retrieval_rate=mean('local_retrieval'), global_lane_rate=mean('global_lane'),
                  repair_trigger_rate=mean('repair_triggered'), rewrite_rate=mean('rewrite_calls'),
                  extra_llm_calls_per_transition=mean('rewrite_calls'))
    triggers = sum(r.get('repair_triggered', False) for r in rows)
    result['repair_success_rate'] = sum(bool(r.get('required_skill_recovered_by_repair')) for r in rows)/triggers if triggers else 0.0
    result['required_skill_recovered_by_repair'] = [
        {'case_id': r['case_id'], 'stage_id': r['stage_id'], 'skill_ids': r['required_skill_recovered_by_repair']}
        for r in rows if r.get('required_skill_recovered_by_repair')]
    result['candidate_recall_per_candidate'] = result['required_skill_candidate_recall']/result['mean_candidate_pool_size'] if result['mean_candidate_pool_size'] else 0.0
    for key in ('registered_skill_count', 'active_skill_count'):
        result['mean_final_'+key] = sum(r[key] for r in finals)/len(finals) if finals else None
    return result


def evaluate_soft_bundle_cases(cases, records, *, bm25, dense, dense_factory, extractor=None,
                               k=5, max_bundles=4, max_skills_per_bundle=4):
    args = dict(bm25=bm25, dense=dense, dense_factory=dense_factory)
    # The old evaluator, including every historical stage field, is untouched.
    baseline = evaluate_hierarchical_cases(cases, records, **args, k=k,
        max_bundles=max_bundles, max_skills_per_bundle=max_skills_per_bundle)['hierarchical']
    hard = ShelfAwareRetriever(records, **args)
    soft = SoftBundleRetriever(records, **args)
    arms = ('A', 'B', 'C') if extractor is not None else ('A', 'B')
    result = dict(budget=asdict(SoftBudget()), k_per_retriever=k, max_bundles=max_bundles,
                  max_skills_per_bundle=max_skills_per_bundle, baseline_A=baseline,
                  rate_denominator='runtime transitions only; macro average over required_now per stage',
                  search_cost_definition='unique records per transition; visits sum global+local per pass, excludes duplicate BM25/Dense visits',
                  sequential={}, frozen={})
    frozen_states = {}
    for mode in ('sequential', 'frozen'):
        for arm in arms:
            rows, finals = [], []
            for case in cases:
                shelf = CapabilityShelf()
                for index, stage in enumerate(case.stages):
                    if mode == 'frozen' and index:
                        shelf = frozen_states[(case.case_id, stage.stage_id)]
                    before = shelf
                    if mode == 'sequential' and arm == 'A' and index:
                        frozen_states[(case.case_id, stage.stage_id)] = before
                    if not index or arm == 'A':
                        query = case.initial_task if not index else '\n'.join(stage.runtime_evidence)
                        route = hard.search(query, shelf, k=k)
                        candidates = route.candidates
                        local = route.route == 'bundle-local'
                        trace = dict(query=query, route=route.route, search_corpus_size=route.corpus_size,
                            search_record_visits=route.corpus_size, global_lane=not local,
                            local_retrieval=local, priors=[], local_candidate_count=len(candidates) if local else 0,
                            global_candidate_count=0 if local else len(candidates), retrieval_passes=1,
                            repair_triggered=False, rewrite_calls=0, repair_added_ids=[])
                    else:
                        candidates, trace = soft.retrieve(case.initial_task, stage.runtime_evidence,
                                                          extractor=extractor if arm == 'C' else None)
                    if not index:
                        shelf = build_capability_shelf(candidates, records, max_bundles=max_bundles,
                                                       max_skills_per_bundle=max_skills_per_bundle)
                    else:
                        shelf = integrate_retrieval_delta(before, candidates, records, max_new_bundles=max_bundles,
                                                         max_skills_per_bundle=max_skills_per_bundle)
                    row = {**trace, **score(stage, candidates, before, shelf, records), 'case_id': case.case_id}
                    row.update(prior_hit=bool(trace['priors']), prior_count=len(trace['priors']),
                               required_skill_recovered_by_repair=sorted(set(stage.required_now) & set(trace['repair_added_ids'])))
                    if index:
                        rows.append(row)
                finals.append(row)
            result[mode][arm] = {'metrics': summarize(rows, finals if mode == 'sequential' else ()), 'stages': rows}
    return result
