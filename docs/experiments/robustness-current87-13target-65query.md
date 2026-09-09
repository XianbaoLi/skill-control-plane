# current87: 13 targets × 5 frozen queries

Completed retrieval-only robustness evaluation over all six current field-ablation
representations. No runtime rerouting, activation, Gold edits, representation edits,
or tuning. Each target contributes V0 verbatim and four wording variants, frozen
before retrieval. The author had prior field-ablation context; this is not claimed
as a blinded paraphrase experiment. No wording was optimized from the new results.

The existing `evaluate_query_variant_retrieval` evaluator, BM25, Dense, card
projection, and RRF implementations are reused without modification. Embeddings
are BigModel embedding-3, dimensions=2048; exact-input API vectors are cached and
shared across representations. BM25 uses k1=1.5, b=0.75; source depth is 10,
RRF k=60, and cutoffs are 5 and 10. Current87 manifest snapshot:
`f670e9d5ecdbb299a5c5ac4fcc48ece35e0adadcfb7e24b96850a4ec5379e509`.

The runner validates 87 corpus records, manifest/Gold snapshots, card content
hashes, original-query equality across all six field-ablation reports, 13 identical
Gold targets, and five distinct strings per target. The outputs freeze indexed
Dense/BM25 texts, source file hashes, embedding vectors, and ranked candidates.

## Exact commands

Run from the repository root:

```bash
set -a
source .env
set +a
PYTHONPATH=src .venv/bin/python scripts/robustness_current87.py > local_artifacts/v0.6/robustness-current87-13target-65query-run.log 2>&1
```

The query artifact must already exist; the runner never generates or tunes queries.
The final run uses saved embeddings. The local virtual environment needed pytest
and PyYAML installed before running tests.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_robustness_current87.py tests/test_retrieval_explainability.py tests/test_retrieval_ablation.py tests/test_retrieval_cards.py tests/test_retrieval_eval.py
```

Result: **17 passed**. Tests include independent recomputation of all 65-query
Recall/MRR/all-hit metrics, rejecting original-query drift and duplicate wording,
and checking censored worst ranks and pairwise win/loss direction.

## Artifacts

All files are in `local_artifacts/v0.6/`, with prefix
`robustness-current87-13target-65query`:

- `.json`: complete report, 65 queries, six reports, 15 pairwise comparisons,
  per-target win/tie/loss for each metric, worst ranks, reproduction checks.
- `-queries.json`: independently frozen original queries and paraphrases.
- `-comparison.md`: compact six-representation comparison table.
- `-summary.md`: all requested aggregate metrics for Dense/BM25/RRF, plus Union.
- `-indexed-texts.json`: exact strings indexed for all six representations.
- `-embeddings.json`: raw embedding API vectors for reproducibility.
- `-tests.log` and `-run.log`: verification and evaluation output.

Win/tie/loss is defined separately for each metric, per target, as **right minus
left**. It is not an unspecified overall winner. A null worst rank means at least
one query missed the retrieved pool: Dense/BM25 are censored at 10; RRF is ranked
within the union of source Top-10 lists. Union remains unordered and therefore
has candidate recall/coverage rather than artificial MRR or Top-K metrics.

## Results and limitations

Full-card BM25 and RRF retrieve all 65 targets within Top-5. Full-card Dense misses
Top-5 only for ST-01/S3 V3 (python-debugpy, rank 6), giving Recall@5=64/65 and
all-variants-hit@5=12/13. Its Recall@10 is 65/65.

Removing use_when lowers Dense Recall@5 from 64/65 to 57/65, Recall@10 to 64/65,
and RRF all-variants-hit@5 from 13/13 to 12/13. Removing purpose does not reduce
these aggregate metrics in this sample. These observations do not trigger changes
to representations.

Full vs metadata per-target Recall@5 win/tie/loss: Dense 2/11/0, BM25 4/9/0,
RRF 1/12/0. For MRR@5: Dense 3/9/1, BM25 9/3/1, RRF 6/6/1.

233 of 234 original target ranks reproduce. The sole difference is
minus-capabilities ST-03/S2 RRF: rank 3 becomes 2. Dense non-target candidates
himalaya and meeting-action-items exchange ranks 3/4; the target remains rank 5.
All original BM25 Top-10 lists match. Four original Dense Top-10 lists differ
from historical results; historical embedding vectors are unavailable, so their
cause cannot be established conclusively. All original Recall@5/@10 metrics
match the field-ablation report. This discrepancy is preserved in the artifact.

The historical report does not embed indexed-text hashes, so historical byte
identity cannot be independently proven from that report alone. The current
corpus manifest and cards are validated, their exact indexed strings are frozen,
and no input was changed. There are 13 targets with correlated paraphrases, not
65 independent targets; this is an intermediate sample, not a significance claim.
