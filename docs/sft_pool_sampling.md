# Eight base-model responses per eligible SFT pool problem (v4)

Open `notebooks/sample_clean_sft_pool_qwen3b_8.ipynb` in GPU Colab and enable
`HF_TOKEN` in Colab Secrets. The notebook embeds all required project files.
Use the new default `qwen25-3b-base-eight-v4` run name; older journals must not mix
with the revised grading and quality policy. Full generation and upload default off.

## What changed

The scorer still uses Math-Verify 0.9.0 (`parse` and `verify`, no string fallback).
Its extraction policy is now `explicit-final-v2`:

- Preserve complete equations instead of splitting at `=`. Bounded rational
  univariate polynomial equations are compared up to a nonzero constant. Variable
  renaming is permitted only for questions asking to construct/find an equation.
  Variable factors are not canceled, since that can add or remove roots.
- Require complete polynomial factors over the rationals on factorization tasks.
  Returning the original reducible polynomial, multiplying it by 1, or leaving a
  reducible factor receives no credit. Equivalence to the reference is also required.
  The supported domain is univariate rational polynomials of degree at most 12;
  unsupported reference forms are quarantined for review before generation.
- Normalize explicit radix numerals in questions establishing base-number context.
  `10010_2` and `(10010)_{2}` both become decimal `18`; invalid radix digits fail.
  Original answer text remains in the grading audit.
- Store junk/unparseable final answers as empty strings. Preserve recognized LaTeX
  unit labels for Math-Verify parsing. Degree stripping still requires angular context.

Final boxes take priority; otherwise the scorer accepts a clear explicit conclusion
or final math line. It never searches the reasoning for the reference value.
The notebook runs 12 CPU regression checks before preparation/generation.
Benchmark grading elsewhere in this repository is unchanged.

## Source screening and its limits

The pinned source contains 28,905 rows. The v4 full-source preflight retains 19,885,
quarantines 9,020 for review, and applies six mathematically reviewed gold corrections.
These counts are screening outcomes, not a claim that all quarantined rows are wrong
or that every retained reference is correct.

The quality registry now contains 17 individually reviewed cases across both smoke
files: six corrections and 11 quarantines. New corrections fix the degenerate hexagon
area ratio to 0 and the circle/tangent maximum area to `3*sqrt(3)*r^2/8`. Nonunique,
corrupted, inconsistent, or incomplete-reference questions are quarantined.
Each reviewed action checks the exact question hash and original gold before applying.

Beyond individual patches, unreviewed `synthetic_math` rows are quarantined by default
pending source review. This conservative source-level hold follows multiple newly
observed semantic errors; it does not establish an error rate for that entire subset.
Individually reviewed corrected rows can remain eligible. All source rows also receive
reference parseability and supported-factorization checks, alongside existing rules
for image/diagram dependence, proofs, multipart questions, and corrupted fragments.
The rules do not exhaustively establish uniqueness or mathematical truth for every row.

`quality_summary.json` reconciles counts. `quality_decisions.jsonl` preserves every
excluded/corrected original and its reason. Eligible rows retain source columns;
corrected answers retain `gold_answer_original` and `quality_action`. Original Hub
files remain stored when the complete annotated dataset is eventually published.

## Controlled smoke comparison

`REPLAY_LAST_SMOKE=True` is the notebook default. It selects the 32 still-eligible IDs
from `smoke_results (1).jsonl`, then fills the 18 excluded slots with seeded random
eligible rows without replacement. `smoke_selection_summary.json` lists shared,
excluded, and replacement IDs. Compare only matching question IDs; new questions do
not form a controlled before/after sample. Set the toggle to False for 50 fresh random
eligible rows. Changing this setting requires a new RUN_NAME.

An optional CPU-only cell accepts either original 50-row smoke file, regrades its
saved responses, and downloads `smoke_results_regraded.jsonl`, `grading_changes.jsonl`,
and `quality_decisions.jsonl`. It also prints `summary.json`. This diagnostic may have
fewer than 50 retained rows. It preserves response strings, token counts, and finish
reasons; old exports are never imported into the new generation journal because they
lack full runtime/stop provenance.

The second saved smoke retains 32 rows (256 responses). On those identical retained
responses, recorded correctness changes from 46 to 48: four flags change, including
the equation, factorization, binary, and an explicit unit-bearing final answer.
There are zero grading exceptions. This is CPU regrading, not a new model generation
or an estimate of full-pool accuracy. Detailed outputs are under
`reports/sft_pool_smoke_v4_second/regrade/`.

## Generation and publication

The model is pinned `Qwen/Qwen2.5-3B` base with the repository's English instruction
and native chat template. Defaults remain eight samples, temperature/top-p 1.0,
8,192 output tokens, 32,768 context, seed 42. Prompts are never silently truncated.
Both Qwen `<|endoftext|>` (151643) and `<|im_end|>` (151645) stop generation with EOS
enabled. Actual stop diagnostics are saved in `grading_audit.jsonl`.

The six requested columns remain `responses`, `extracted_answers`, `correct`,
`response_tokens`, `finish_reasons`, and `num_correct`. Smoke JSONL contains all 400
new responses. Review source gold quality as well as grading and truncation before
enabling RUN_FULL and then UPLOAD_FULL. Full generation reuses the new smoke journal;
per-problem checkpoints allow resume and full outputs use sharded Parquet.

Publication requires a complete full run, checks row/vector integrity and checksums,
refuses to overwrite a newer Hub head, and verifies the uploaded files. The default
HF train split becomes the eligible annotated rows in
`Seungjun/clean-math-sft-pool-30k`. No upload or new GPU inference was performed during
the local notebook repair; fresh inference must run in GPU Colab.

## Validation

Run `python -m pytest tests/test_sft_pool_quality.py tests/test_sft_pool_sampling.py
tests/test_batched_eval.py -q` and `python -m eval.pool_grading_checks`.
Tests cover equation/counterexample handling, complete factorization, radix notation,
junk and unit answers, mathematical gold checks, source holds, replay/replacement
selection, regrading preservation, checkpoint/export/upload guards, and exact embedded
notebook payload integrity. Rebuild using
`python scripts/build_sft_pool_sampling_notebook.py`.
