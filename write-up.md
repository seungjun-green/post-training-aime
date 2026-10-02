# EXAONE math reasoning: progress

The project uses **LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct** with the original English math datasets. Data preparation, Hugging Face publication, and the full stage-0 baseline evaluation are complete. Fine-tuning has not started.

## Direction and data preparation

We initially explored Korean translation through hosted APIs and local models. API refusals and translation errors that changed mathematical notation or meaning made that approach unreliable. We switched to English while keeping the same base model.

Training data was checked against the five evaluation sets using normalization v2 and distinct 8-gram coverage. A training row is removed when it covers at least 70% of an individual evaluation problem's distinct 8-grams; evaluation problems shorter than eight tokens use exact normalized substring matching. Matches are not pooled across evaluation problems. This replaced the overly aggressive rule that removed a row for any single shared 8-gram.

| Training dataset | Original rows | Removed | Retained |
|---|---:|---:|---:|
| s1K-1.1 | 1,000 | 4 | 996 |
| DAPO-Math-17k-Processed | 14,116 | 48 | 14,068 |

This removes **training-versus-evaluation overlap only**, not duplicates within or between training datasets. All 630 evaluation problems remain unchanged. The seven datasets were uploaded to Hugging Face under `Seungjun/dp_removed_{original_name}`; exact evaluation revisions are recorded in [the dataset suite](configs/english_eval_suite.json).

## Stage-0 English baseline

The full run evaluated **630 problems with 6,160 sampled responses** on an NVIDIA RTX PRO 6000 Blackwell 96GB using vLLM. These are the base model's results before project fine-tuning under the original 32-sample AIME/AMC and four-sample MATH-500 protocol. They remain historical results; the new `greedy` and `sample8` profiles require their own matching baseline runs.

| Dataset | Problems | Samples per problem (n) | Pass@1 / avg@n | Pass@4 | Pass@n | Solved at least once |
|---|---:|---:|---:|---:|---:|---:|
| AIME 2024 | 30 | 32 | 3.02% | 7.45% | 13.33% | 4/30 |
| AIME 2025 | 30 | 32 | 3.44% | 9.52% | 23.33% | 7/30 |
| AIME 2026 | 30 | 32 | 1.46% | 4.28% | 13.33% | 4/30 |
| AMC 2023 | 40 | 32 | 36.56% | 58.26% | 80.00% | 32/40 |
| MATH-500 | 500 | 4 | 63.20% | 77.60% | 77.60% | 388/500 |

Pass@1 is estimated from all sampled responses and equals avg@n here; it is not a separate greedy-decoding run. Pass@k estimates the chance of at least one correct answer among k samples. At k = n, it is the fraction of problems solved at least once. It does not measure whether the model can select its correct answer.

Evaluation uses temperature **1.0**, top-p **0.7**, a **20,480-token output limit**, and a **32,768-token context limit**. Original English problems are passed through the model's native chat template with an English step-by-step instruction. The last balanced `\boxed{...}` answer is graded with `math-verify`; reasoning quality is not separately scored. Settings are in [eval_english.yaml](configs/eval_english.yaml).

The supplied `stage0.zip` was checked for complete, unique samples and consistent metadata. Metrics were recomputed and saved answer pairs rescored with no discrepancies. Seven responses reached the output limit, and six lacked a boxed answer. Low AIME accuracy therefore is not explained by widespread truncation or missing answer extraction. The baseline performs substantially better on MATH-500 and AMC23 than on AIME under this protocol.

Run provenance:

- Evaluation Git commit: `2e0d65aaf8803aee532eb38c3335736b9e66a18f`.
- Model revision: `e949c91dec92095908d34e6b560af77dd0c993f8` (loader compatible with the pinned runtime).
- Runtime: vLLM 0.14.1, PyTorch 2.9.1, Transformers 4.57.6, math-verify 0.9.0; BF16.
- Results: `/content/drive/MyDrive/LG-AIME-English-Eval-compatible/full/results/stage0`.

## Current workflow and next step

The additional [temperature-1.0 SFT evaluator](notebooks/evaluate_stage1_sft_temp1.ipynb)
uses `sample1`: one answer per problem on all five benchmarks, top-p 0.7, and the same
20,480-token output limit. It selects epoch 5 by default and writes separate results under
`profiles/sample1/`. This permits checking sampled decoding after the greedy evaluation;
GPU results from this new profile have not yet been collected. Existing evaluation notebooks
retain their defaults; the new profile is also available through the shared baseline/SFT CLI.

A separate [DeepSeek regeneration notebook](notebooks/regenerate_s1_deepseek.ipynb) now prepares
alternative reasoning targets from the same 996 decontaminated s1K questions. It uses
`deepseek-v4-pro` with thinking enabled. Following review of the first smoke CSV, the final response
now explicitly requires Planning, Evaluation (the full worked derivation), Reflection, and
Exploration sections followed by the final answer. It imposes no prompt-level brevity target or
answer-correctness filter. `deepseek-v4-pro_answer` is the intended structured training target;
`deepseek-v4-pro_reasoning` stores raw API thinking separately for inspection.
The 20-example random smoke cell exports an old/new comparison CSV. The separate full-run cell
preserves all source columns and adds `deepseek-v4-pro_reasoning` and `deepseek-v4-pro_answer`.
Concurrent requests, retry journals, and resumable outputs are saved to Drive. Technical failures
and section-format failures remain visible separately; partial outputs contain all rows with null
generated fields for failures. The revised prompt starts a fresh run on the same seeded smoke sample.
This is data generation only: the existing SFT and evaluation implementations are unchanged.
Local mocked-API tests passed; no live DeepSeek generation has been run during implementation.

The subsequent full export used the original generation approach, and the selected training target
is now **both Pro thinking and answer**, preserving the existing `<think>…</think>` format.
`train_stage1_sft.ipynb` defaults to `configs/stage1_sft_deepseek_pro.yaml` and starts from the
original EXAONE base checkpoint with the same five-epoch hyperparameters. The uploaded export
contains 990 complete pairs and six missing pairs. After excluding missing pairs and full native
training sequences longer than 20,480 tokens, 718 examples remain (272 overlength exclusions).
This gives 45 optimizer steps per epoch and 225 across five epochs. New checkpoints/logs use
`sft_s1k_deepseek_pro`; the old config and outputs remain separate. The dataset's current HF
revision is resolved to an immutable commit and its exact content is checked and recorded.
`evaluate_stage1_sft_temp1.ipynb` selects this run's epoch-5 checkpoint: temperature 1.0,
top-p 0.7, one answer per problem on all five benchmarks. The `sample1_budget` profile
adds s1-style early-exit budget forcing with an editable thinking cap (default 18,432, minimum
zero) and the remaining 2,048 tokens including any injected `</think>` and final-answer cue.
The notebook saves a resolved budget YAML on Drive, and each budget split has its own output
directory; the earlier fixed 16,384/4,096 profile is still available. Total continuation tokens
stay within 20,480. Natural thinking termination is allowed; "Wait" extension is disabled.
Two-phase requests use the existing continuous vLLM scheduler interface and scoring code.
Results are isolated by profile, with phase counts and forced-transition flags in raw records.
An optional five-problem smoke cell precedes the full-run cell. The old R1 checkpoint can be
tested by selecting its original SFT config. No accuracy gain has been measured yet; local
mock tests validate the mechanism, and the Colab GPU smoke test remains to be run.

The active notebooks are:

1. [Prepare English datasets](notebooks/prepare_english_datasets.ipynb): CPU decontamination and Hugging Face upload.
2. [Evaluate the English baseline](notebooks/evaluate_baseline_english.ipynb): clone/update the project, install the GPU runtime, run a five-problem smoke test, then explicitly enable full evaluation. Includes tqdm progress and saved results on Drive.
3. [Stage 1 SFT](notebooks/train_stage1_sft.ipynb): prepare and inspect assistant-only loss masks, then explicitly enable five-epoch training and save every epoch checkpoint. No evaluation runs here.
4. [Stage 1 evaluation](notebooks/evaluate_stage1_sft.ipynb): evaluate epoch 5 on all five benchmarks by default, with epochs 1–4 optional afterward. Both this notebook and the baseline notebook offer `greedy` (temperature 0, one answer, 630 responses/checkpoint) and `sample8` (temperature 1.0/top-p 0.7, eight AIME/AMC answers and four MATH-500 answers, 3,040 responses/checkpoint). Use the same profile for base and SFT comparisons. Continuous batching and the original evaluation environment are retained.

Evaluation logic lives in the project rather than the notebook. The preparation notebook still bundles its preparation code. The translation notebooks and old Korean baseline notebook have been removed; `README.md` is retained as historical documentation and contains references to those retired notebooks.

Stage 1 code is implemented, with settings and execution instructions in the [run guide](docs/stage1_sft.md). CPU preparation using the actual pinned EXAONE tokenizer drops 21 examples exceeding 20,480 tokens and keeps 975, resulting in 61 optimizer steps per epoch and 305 total. The original 996 rows contain 370 `deepseek_grade`-incorrect examples; this grade does not filter training. The [data report](docs/stage1_data_report.json) records the exact dropped IDs and token-length statistics.

Next is GPU validation and fine-tuning, followed by evaluation with the same datasets, prompts, sampling settings, scoring, and runtime. The continuous-batching runner supplies all five benchmarks for each selected epoch. The greedy profile queues up to 64 problems, and sample8 queues up to 16, for the existing 32 GPU response slots. Prompts, seeds, token limits and scoring remain shared. Profiles use isolated `profiles/<profile>/` directories and allow either model to run first, with matching settings required for comparisons; old 32-sample results are preserved without importing them. Settings are recorded in manifests; exact outputs may still vary with GPU numerics and batching. CPU tests cover scoring equivalence on identical responses, out-of-order completion, interruption recovery and source validation; GPU throughput has not yet been measured. Preserve the stage-0 archive and its parent protocol/runtime manifests for comparisons; no fine-tuned results are reported yet.
