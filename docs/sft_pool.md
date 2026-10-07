# Clean math SFT problem pool (Stage 1)

Open `notebooks/build_clean_sft_pool.ipynb` in Google Colab, enable a write-capable
`HF_TOKEN` secret, and run all cells. The notebook is self-contained: it bundles the
project modules, tests, and reference configurations rather than cloning a branch.
It builds the pool, saves the result to Google Drive, and uploads a verified commit
to `Seungjun/clean-math-sft-pool`. The destination is private by default, configurable
in the notebook. No model inference is performed.

Use a CPU runtime with several GB of free disk and allow hours. The full 904,715-row
source run has not been executed here; no full-corpus runtime estimate is available.
SQLite stores deduplication postings and problem payloads locally. Interrupted runs
restart from the beginning, reusing any surviving Hugging Face download cache.
Completed exports live in `MyDrive/lg-korean-aime/data/sft_pool/<content-id>`.

## Inputs and provenance

- MATH: `EleutherAI/hendrycks_math`, all seven train configurations, 7,500 rows.
- Numina: `AI-MO/NuminaMath-1.5`, default/train, 896,215 rows.
- s1: `simplescaling/s1K`, original default/train, 1,000 rows. This intentionally
  uses the original set requested in the specification, rather than s1K-1.1.
- Evaluation: exact repository commits, IDs, and original-content hashes from
  `configs/english_eval_suite.json` for all five benchmarks.
- RL: exact content and ID digests from `configs/dapo.yaml`; resolve the configured
  revision to a commit, verify the export provenance, and record that commit.

Training revisions were verified on the Hub and are pinned in `configs/sft_pool.yaml`.
Every loaded dataset revision, dependency versions, configuration, repository base
commit, and bundled source hash are recorded. The code manifest explicitly records
that the new implementation is not yet committed, so the base Git SHA alone is not
misrepresented as the complete implementation. The bundle hashes identify the code.

## Filtering and matching

Conservative rule-based filtering excludes proofs, multiple choice, multiple required
answers, unsupported gold answers, invalid Numina rows, and Numina's `amc_aime`
sub-source by default. All reasons are logged; one primary reason owns each removal
for additive counts. Unknown s1 domains are excluded. In particular, a `math` cot_type
is insufficient for physics sources or mixed-domain TheoremQA without math evidence.
Gold extraction uses the shared brace-aware boxed-answer utility and Math-Verify.
Only question, gold answer, and allowed metadata survive; generated traces are omitted.

Deduplication uses a disk-backed **exact Jaccard prefix join** over distinct word
5-grams, configurable independently of reference overlap grams. A consistent total
ordering and length bounds generate candidates without probabilistic misses, then
full shingle sets verify the threshold. Connected components form clusters. Source
priority selects representatives. Every member's answer must be Math-Verify-equivalent
to the representative; conflicting clusters are quarantined in full, including exact
text conflicts. This deliberately conservative policy is recorded in provenance.

Reference matching uses inverted indexes over distinct 8-grams. Scores are computed
against each individual reference; they are never added across references. Short
references use normalized substring matching. Eval and RL reports are separate.
Exact numeric-template matches additionally flag number-swapped copies below the
ordinary near-miss lower bound, preserving the actual numeric coverage. This does
not promise detection of every paraphrase or semantic variant.

Lexical normalization lowercases, removes formatting, canonicalizes fraction command
variants, and tokenizes words/numbers while discarding punctuation. It is not symbolic
algebra and can conflate operators; answer conflicts and audit logs mitigate this.
The original problem text is never modified. Nested source metadata is serialized
losslessly as JSON text in the final pool to keep a stable Hugging Face column schema.

## Outputs and publishing

`pool.jsonl`, `summary.json`, `removed.jsonl`, `merged_clusters.jsonl`,
`conflicts.jsonl`, `eval_near_misses.jsonl`, `rl_near_misses.jsonl`, and
`provenance.json` are published with a dataset card and completion checksums.
Near-miss entries are reference-level (multiple entries per retained problem are
possible). Eval near misses are recorded before RL exclusion, so some may later be
removed by the RL stage. Manual review candidates are retained as the specification
requests; they are clearly flagged in the final pool.

The upload cell requires a successful completion marker, verifies artifacts, makes
one Hub commit guarded by the previous revision, and downloads every published file
at the new revision to verify checksums. It never writes to source or reference repos.
Colab secrets are only accessible when running inside Colab; no token is bundled.

## Validation

Run `python -m pytest tests/test_sft_pool.py -q` and rebuild with
`python scripts/build_sft_pool_notebook.py` after editing any bundled file.
Tests cover normalization, distinct/per-reference coverage, verbatim/embedded/short
references, numeric variants, source priority, gold equivalence/conflicts, prefix join
versus brute force, reproducible bytes, count reconciliation, notebook payload integrity,
and rejection of changed artifacts before upload. An optional cached MATH-500 fixture
is also tested; the notebook checks a real pinned MATH-500 reference during its run.

Read-only real-source filter smoke checks retained 1,597 of 1,744 MATH algebra rows
and 502 of 1,000 original s1K rows before deduplication/reference exclusion. These are
filter smoke results, not final pool counts. The full Numina row count was verified
through the public Hub size endpoint; the full corpus has not been processed locally.
