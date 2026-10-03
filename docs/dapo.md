# EXAONE DAPO pilot

## Evaluate the completed minibatch run at checkpoint 300

Open `notebooks/evaluate_dapo_checkpoint300_amc_math.ipynb` with
`configs/dapo_eval_300_amc_math.yaml`. Default checkpoint:
`/content/drive/MyDrive/LG-AIME-DAPO-MiniBatch300/checkpoints/dapo_exaone_base/checkpoint-300`.
This selects the fresh base-started minibatch run, not the earlier 100-to-300 continuation.
It evaluates AMC 2023 (40) and MATH-500 (500) only, using the existing batched evaluator,
temperature 0, top-p 1, seed 42, one answer per problem, response cap 20,480, no budget forcing.
No baseline run or optimizer state is required. The optional smoke evaluates two problems.

Output root: `/content/drive/MyDrive/LG-AIME-DAPO-MiniBatch300-Eval-AMC-MATH-temp0`.
Summary: `full/results/dapo_checkpoint300.json`; raw answers:
`full/results/dapo_checkpoint300_generations.jsonl`; resumable per-problem journal:
`full/results/dapo_checkpoint300_problems.jsonl`. The two-row results table is saved as
`full/summary_table.json` and console output is under `logs/`. Inference reads checkpoint
weights without copying or saving them. The notebook validates completed checkpoint identity
and displays correct counts, accuracy and mean response length after the full run. Local
tests exercise model selection, subset/settings, smoke/full commands and table validation;
actual GPU evaluation runs in Colab. No evaluator implementation changed.

## Training notebooks

Open `notebooks/train_dapo_exaone.ipynb` in Colab on the RTX PRO 6000 Blackwell 96GB.
Settings are plain Python variables. `MODEL_KIND = "base"` starts from the pinned original
`LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`; `"sft"` starts from
`SFT_ROOT/checkpoints/stage1/sft_s1k/epoch_5/`, with default
`SFT_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"`. The SFT manifest is checked for the
original DeepSeek columns, model revision and run identity. No Llama or Pro-column checkpoint
is selected by this notebook. Push the implementation to GitHub main before setup.

## Training and data

`configs/dapo_minibatch.yaml` defines G=8, 16 retained mixed-correctness questions per rollout,
128 generated responses split into disjoint 64-response optimizer minibatches, microbatch 1, LR 1e-6 with 20-update warmup then constant LR,
temperature/top-p 1, clip 0.20/0.28, group-normalized shaped rewards, token-normalized loss,
and KL coefficient zero. The rule-based shared scorer gives +1/-1 for correct/incorrect final
boxed answers. A linear penalty starts at 16,384 completion tokens and reaches -1 at the
20,480-token hard limit. The reference verl soft-shaping recipe keeps truncated responses in
the loss; this runner does likewise. It does not inject thinking delimiters or answer cues.

The published English dataset is `Seungjun/dp_removed_DAPO-Math-17k-Processed`, config `en`,
split `train`. HF `main` is resolved to a commit; the contents and preparation IDs must match
the existing 14,068-row decontamination record. Source prompt wrappers are stripped before
applying the unchanged shared English instruction and EXAONE chat template. Prompt overflow
is excluded and reported, never silently truncated. Local preparation against the actual pinned
EXAONE tokenizer retained all 14,068 questions, with prompt lengths 52–1,662 tokens.

Each fresh rollout shuffles the candidate dataset deterministically from seed 42 and its starting update index.
Groups with all-correct or all-incorrect answers are rejected based on correctness, not shaped
reward variance. Refill continues until 16 mixed groups are collected, or ten candidate batches
are exhausted. Exhaustion stops without an update for that batch and saves the generated text
and filtering diagnostics. This can happen if the selected policy cannot solve enough questions;
the runner does not invent rewards or disable filtering to force a successful run.

The run uses **300 optimizer updates**, with **150 fresh rollout batches**. Each generated
batch of 128 responses is shuffled and divided into two disjoint 64-response minibatches.
`algorithm.num_iterations: 1` means each response is used exactly once. TRL caches original
log-probabilities, advantages and sampling correction for both minibatches. The second
minibatch is evaluated under the already-updated policy, so clipping can engage without
repeating the same answers. A zero clip fraction is still possible for small changes.

The loss denominator is calculated **after TRL's shuffle** and equals the active completion
tokens in that optimizer minibatch (including all 64 accumulated microbatches), not the full
128-response rollout. Checkpoints save every 20 optimizer updates through 300. This follows
the paper's disjoint-minibatch structure at a smaller scale, not its exact 8192/512 batch
sizes or 16 updates per rollout. Warmup remains 20 optimizer updates, not the paper's 20
rollout steps. No benchmark improvement is promised and no evaluation is launched.

## Runtime and precision

`requirements-dapo.lock` combines the existing compatible evaluation and training locks:
PyTorch 2.9.1, Transformers 4.57.6, TRL 0.24.0 and vLLM 0.14.1. A separate environment is
installed under `/content/lg-dapo-env`. TRL's DAPO loss and accumulation machinery are extended
with explicit dynamic sampling and a separate colocated vLLM interface. An import-only alias
bridges TRL's old `GuidedDecodingParams` name to vLLM's `StructuredOutputsParams`; the stock
TRL generation path and structured decoding are not used.

Training uses full FP32 parameters/Adam state with BF16 autocast. This preserves small updates
at LR 1e-6; the rollout model uses BF16. vLLM runs at most 16 concurrent responses with 35%
GPU allocation and sleeps during optimization. A fresh BF16 weight snapshot on **local SSD**
is loaded through a named worker-extension RPC before each fresh rollout's sampling; prefix cache
is reset. This transport adds local disk traffic but avoids fragile cross-process parameter
references. RPC carries only the method name and file path, without pickle/callable serialization.
Only one temporary snapshot is retained. The stock TRL train-vs-rollout importance
correction (cap 2) remains enabled and its mismatch metrics are logged.

The policy log-prob path trims per-response padding and chunks vocabulary calculations.
Every microbatch within one optimizer minibatch uses that minibatch's total active-token denominator.
No learned reward/critic/reference model is loaded, and no LoRA or quantization is used.

## Smoke, outputs and resume

Run preparation, then enable `RUN_SMOKE`. The smoke performs two real optimizer updates using
one rollout of two retained groups (16 responses), split into disjoint 8-response minibatches,
G=8 and the full completion cap. Warmup is disabled
to exercise a nonzero update. Each smoke has a timestamped directory; full training loads fresh
source weights and never inherits smoke weights. A smoke can stop for insufficient mixed groups.
Enable `RUN_TRAINING` for the separate pilot.

Default output root: `/content/drive/MyDrive/LG-AIME-DAPO-MiniBatch300`.
The notebook starts a fresh base-model run by default; MODEL_KIND can still select SFT.
The original `configs/dapo.yaml` and `configs/dapo_continue_300.yaml` keep one update per
rollout. Do not resume those checkpoints with the new minibatch config; their original
continuation notebook remains separate.

- Checkpoints: `checkpoints/dapo_exaone_<base|sft>/checkpoint-<step>/`.
- Per-update metrics: `logs/dapo_exaone_<base|sft>/steps.jsonl`.
  `rollout_first_update` and `minibatch_index` identify each optimizer update.
  `policy_iteration` stays 1 and `reused_rollout` stays false; `cached_rollout` is true
  on the second minibatch, which uses different responses from the same rollout.
  Rollout statistics repeat across minibatches; sum `new_generated_tokens` to count fresh
  generation without double-counting. Accuracy on a reused batch is not a new measurement.
- Raw rollout text and reward/filtering diagnostics: `logs/<run>/rollouts/attempt_*/update_*/`.
- Console output, resolved settings, data report, manifest and attempt timing/memory: `logs/<run>/`.
- Smoke logs/checkpoint: `smoke_attempts/<run>/<timestamp>/`.

Checkpoints include full model/tokenizer, optimizer, scheduler, RNG and Trainer state. Allow
approximately 500GB of Drive space for fifteen full checkpoints and a smoke checkpoint.
max_steps and save_steps must be multiples of updates_per_rollout (2). Checkpoint saves/resume
are restricted to complete rollout cycles because TRL does not save its rollout cache.
Resume with `RESUME_CHECKPOINT` pointing at the latest completed checkpoint. Code, settings,
source, runtime and data must match. Interrupted work after that checkpoint is replayed; abandoned
step logs and incomplete checkpoint directories are preserved separately. Raw attempts are
always kept. GPU memory summaries cover only the training process, not a separate vLLM process.

Local validation covers actual-tokenizer preparation, reward/shaping, dynamic refill/exhaustion,
token-normalized asymmetric clipping, CPU optimizer updates, checkpoint reload/resume, notebook
commands, and the weight-sync API contract with a fake worker. **The actual vLLM/EXAONE GPU
integration and 96GB memory fit have not been run locally.** Use the included GPU smoke in Colab.

References: [DAPO paper](https://arxiv.org/html/2503.14476v1),
[verl recipe](https://verl.readthedocs.io/en/latest/algo/dapo.html),
[TRL 0.24 GRPO trainer](https://huggingface.co/docs/trl/v0.24.0/grpo_trainer).

## Llama 3.2 3B Instruct

Use `notebooks/train_dapo_llama32_3b.ipynb` and `configs/dapo_llama32_3b_minibatch.yaml`.
The source is `meta-llama/Llama-3.2-3B-Instruct`, pinned to
`0cb88a4f764b7a12671c53f0838cd831a0843b95` ([official model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)).
The notebook starts from that instruction model and does full-parameter DAPO, using the same
algorithm, data, response budgets, learning rate, seeds and 300-update/disjoint-minibatch schedule as
`dapo_minibatch.yaml`. It has no SFT-source selector. The existing EXAONE notebooks remain separate.

`train/dapo_model.py` configures the existing native vocabulary: EOS `<|eot_id|>` and padding
`<|finetune_right_pad_id|>`, without adding tokens or resizing embeddings. The shared English
instruction is rendered with Llama's native chat template and `date_string: 26 Jul 2024` to
avoid date-dependent changes on resume. Generation explicitly stops at `<|end_of_text|>`,
`<|eom_id|>` or `<|eot_id|>`; IDs are resolved from the pinned vocabulary. All stop tokens
remain in the token sequences used for the loss. Saved generation configs retain these stops.
TRL's single-EOS termination metrics are corrected from actual vLLM finish reasons so EOM
and end-of-text are not counted as truncations. Prompt-template settings and resolved stop IDs
are included in preparation provenance. No shared evaluation modules were changed.

Default Drive root: `/content/drive/MyDrive/LG-AIME-DAPO-Llama32-3B-MiniBatch300`.
Checkpoints: `checkpoints/dapo_llama32_3b_base/checkpoint-20/` through `checkpoint-300/`.
Logs: `logs/dapo_llama32_3b_base/steps.jsonl`, console logs, manifest and raw rollouts.
Smoke outputs are isolated in `smoke_attempts/<timestamp>/`. Approximately 650 GB of Drive
space covers fifteen full model/Adam checkpoints and smoke. GPU code/model downloads also need
local runtime disk. No weights are zipped for sharing. An HF account with model access is
required; the notebook reads `HF_TOKEN` from Colab secrets.

Local tests cover model selection, fixed-date template arguments, native token setup,
vLLM stop-parameter forwarding, EOS/EOM termination metrics, tied-embedding Llama CPU
updates with actual second-minibatch clipping, and exact checkpoint resume. Regression tests also preserve the older whole-batch
one-update and two-pass schedules. The public pinned
native template was additionally rendered with a synthetic vocabulary to inspect headers
and the fixed date. Full-size weights, native tokenizer token counts, vLLM execution and
96GB GPU fit still require the notebook's preparation/GPU smoke on Colab.

## Continue from 100 to 300 updates

For the **original one-pass experiment**, use `notebooks/continue_dapo_exaone_100_to_300.ipynb` with
`configs/dapo_continue_300.yaml`. It defaults to `MODEL_KIND = "base"`, restores
`LG-AIME-DAPO/checkpoints/dapo_exaone_base/checkpoint-100`, and writes new artifacts to
`LG-AIME-DAPO-100to300`. Original files remain unchanged. Checkpoints are saved at
120/140/.../300; logs use absolute steps 101–300. Ten additional full checkpoints require
approximately 300 GB of Drive space. There is no smoke run or evaluation in this notebook.

`--extend-from-checkpoint` explicitly permits an increased `training.max_steps` and a new
code revision. All other configuration fields, source identity, dataset/tokenizer report,
package versions and precision must match the parent manifest. The parent dataset revision
is reused. Parent checkpoint identity and both code revisions are recorded in the new manifest.
The scheduler must be `constant_with_warmup`, so extending the horizon does not change its
schedule. Model weights, Adam state, scheduler and RNG restore through Trainer's normal resume
path; warmup is not repeated. The validator checks resume files and shard lengths without
loading multi-gigabyte weights into the notebook.

The training cell automatically resumes the latest completed checkpoint in the continuation
folder after interruption, while keeping the parent provenance. Once 300 is complete, it skips
training. If no continuation checkpoint has completed after a failed attempt, select a new
OUTPUT_ROOT. Local tests verify extension from a finished run, restored optimizer step counts
and learning rate, rejection of changed hyperparameters, incomplete checkpoint detection,
and notebook commands. The full 200-update continuation must run on the user's GPU.

## Evaluate checkpoint 100

`notebooks/evaluate_dapo_checkpoint100.ipynb` loads the completed original DAPO checkpoint
from `LG-AIME-DAPO/checkpoints/dapo_exaone_base/checkpoint-100/` by default. `MODEL_KIND`
selects which trained DAPO run to evaluate (`base` or `sft`), not the untrained starting model.
`configs/dapo_eval.yaml` selects the existing single-answer profile and English suite/runtime.
The notebook reuses `eval.run_batched_eval` unchanged for AIME 2024/2025/2026, AMC 2023 and
MATH-500: 630 problems in total. Default temperature/top-p are 0/1, the response cap is
20,480, and budget forcing is off. Sampling can be edited as plain variables and is recorded
in YAML; temperature/top-p combinations have isolated output directories.

Optional smoke evaluates five problems and full evaluation covers all 630. Neither saves
weights or needs optimizer state or a completed baseline. Run on a free GPU after training
has finished. Results are written below
`LG-AIME-DAPO/evaluation_checkpoint100/profiles/sample1/temperature_0.0_top_p_1.0/full/results/dapo/`.
The summary is `dapo_exaone_base_checkpoint100.json`; raw answers use `_generations.jsonl`,
and `_problems.jsonl` supports interruption recovery with unchanged code/model/settings.
Console logs are under the same profile's `logs/`. Use matching settings for baseline comparisons.

## Base vs checkpoint 100 on AMC and MATH only

`notebooks/compare_base_dapo100_amc_math.ipynb` evaluates the pinned original EXAONE model
and `dapo_exaone_base/checkpoint-100` sequentially with temperature 0, top-p 1, one answer,
the shared 20,480-token cap and no budget forcing. `configs/dapo_compare_eval.yaml` selects
AMC 2023 (40 problems) and MATH-500 (500). AIME 2024/2025/2026 are excluded before loading.
The separate `eval.run_amc_math_eval` entry point reuses the existing dataset checks, prompts,
batched generation, scoring and resume journal without modifying the full evaluator.

Optional smoke generates two answers per model in an isolated folder. Full evaluation
generates 540 per model and automatically displays a two-row table with accuracy percentages,
DAPO-minus-base differences in percentage points and average token lengths. The table rejects
incomplete, smoke or mismatched-protocol results. Default Drive folder:
`LG-AIME-DAPO-Compare-100-AMC-MATH-temp0`. Summaries and raw answers are under `full/results/`,
the table is saved as `full/comparison.json`, and console output is under `logs/`.
No weights are copied or saved. Tests cover excluded datasets, greedy settings, shared-scoring
execution/resume, model selection and comparison arithmetic. Actual model evaluation runs in Colab.

## Checkpoint 140 on AMC and MATH only

`notebooks/evaluate_dapo_checkpoint140_amc_math.ipynb` evaluates the base-started EXAONE
continuation checkpoint from
`LG-AIME-DAPO-100to300/checkpoints/dapo_exaone_base/checkpoint-140/`.
`configs/dapo_eval_140_amc_math.yaml` keeps the same greedy AMC/MATH settings as the
checkpoint-100 comparison. It reuses `eval.run_amc_math_eval` without modifying evaluation
code. Only checkpoint 140 runs: 40 AMC 2023 and 500 MATH-500 problems, temperature 0,
top-p 1, one answer each, cap 20,480, no budget forcing, no AIME and no baseline prerequisite.

The optional smoke generates two answers in a separate directory. Full evaluation displays
a two-row table with correct counts, accuracy and mean response length. The default Drive root
is `LG-AIME-DAPO-Eval-140-AMC-MATH-temp0`. Summary and raw outputs are in
`full/results/dapo_checkpoint140.json` and `_generations.jsonl`; the resume journal is
`_problems.jsonl`. The table is `full/summary_table.json`, and console logs are under `logs/`.
The notebook checks the completed checkpoint marker, training manifest and model identity;
optimizer state is not needed. Run on a free GPU after training releases it.
