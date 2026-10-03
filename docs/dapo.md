# EXAONE DAPO pilot

Open `notebooks/train_dapo_exaone.ipynb` in Colab on the RTX PRO 6000 Blackwell 96GB.
Settings are plain Python variables. `MODEL_KIND = "base"` starts from the pinned original
`LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`; `"sft"` starts from
`SFT_ROOT/checkpoints/stage1/sft_s1k/epoch_5/`, with default
`SFT_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"`. The SFT manifest is checked for the
original DeepSeek columns, model revision and run identity. No Llama or Pro-column checkpoint
is selected by this notebook. Push the implementation to GitHub main before setup.

## Training and data

`configs/dapo.yaml` defines G=8, 16 retained mixed-correctness questions per update,
128 accumulated responses, microbatch 1, LR 1e-6 with 20-update warmup then constant LR,
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

Each update shuffles the candidate dataset deterministically from seed 42 and the update index.
Groups with all-correct or all-incorrect answers are rejected based on correctness, not shaped
reward variance. Refill continues until 16 mixed groups are collected, or ten candidate batches
are exhausted. Exhaustion stops without an update for that batch and saves the generated text
and filtering diagnostics. This can happen if the selected policy cannot solve enough questions;
the runner does not invent rewards or disable filtering to force a successful run.

The 100-update pilot uses fresh rollouts once per optimizer update, one pass over each retained
batch, and saves at updates 20/40/60/80/100. This is an adaptation for one GPU, not an exact
reproduction of the paper's large-batch training schedule. No benchmark evaluation is launched.

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
is loaded through a named worker-extension RPC before each update's sampling; prefix cache
is reset. This transport adds local disk traffic but avoids fragile cross-process parameter
references. RPC carries only the method name and file path, without pickle/callable serialization.
Only one temporary snapshot is retained. The stock TRL train-vs-rollout importance
correction (cap 2) remains enabled and its mismatch metrics are logged.

The policy log-prob path trims per-response padding and chunks vocabulary calculations.
Every microbatch uses the same total completion-token denominator for its accumulated loss.
No learned reward/critic/reference model is loaded, and no LoRA or quantization is used.

## Smoke, outputs and resume

Run preparation, then enable `RUN_SMOKE`. The smoke performs one real optimizer update using
two retained groups, G=8 and the full completion cap. Warmup is disabled for this single step
to exercise a nonzero update. Each smoke has a timestamped directory; full training loads fresh
source weights and never inherits smoke weights. A smoke can stop for insufficient mixed groups.
Enable `RUN_TRAINING` for the separate pilot.

Default output root: `/content/drive/MyDrive/LG-AIME-DAPO`.

- Checkpoints: `checkpoints/dapo_exaone_<base|sft>/checkpoint-<step>/`.
- Per-update metrics: `logs/dapo_exaone_<base|sft>/steps.jsonl`.
- Raw rollout text and reward/filtering diagnostics: `logs/<run>/rollouts/attempt_*/update_*/`.
- Console output, resolved settings, data report, manifest and attempt timing/memory: `logs/<run>/`.
- Smoke logs/checkpoint: `smoke_attempts/<run>/<timestamp>/`.

Checkpoints include full model/tokenizer, optimizer, scheduler, RNG and Trainer state. Allow
approximately 200GB of Drive space for five full checkpoints and a smoke checkpoint.
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

## Continue from 100 to 300 updates

Use `notebooks/continue_dapo_exaone_100_to_300.ipynb` with
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
