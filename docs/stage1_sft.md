# Stage 1: English s1-style SFT

The implementation uses the existing English prompt and the native EXAONE chat template.
Training, English prompts, dataset pins, scoring and the dependency lock are unchanged.
Both baseline and SFT evaluation now offer matching `greedy` and `sample8` profiles.
Epoch 5 is evaluated first on all five benchmarks using `eval.run_batched_eval`.
This runner changes scheduling while reusing the original prompts, model loader and scoring.
Epochs 1–4 are optional and use the same full protocol when enabled.
The earlier `eval.run_stage1_amc` utility remains available, but the current notebook
workflow runs full evaluations only.

## Colab

Push this implementation to the repository, then open
[`train_stage1_sft.ipynb`](../notebooks/train_stage1_sft.ipynb) on the RTX PRO 6000 Blackwell
96GB runtime. Enable the `HF_TOKEN` secret. Run setup and preparation, inspect the mask
example and data report, then enable `RUN_TRAINING`. This notebook only trains and saves
checkpoints. After the final checkpoint exists, open the separate
[`evaluate_stage1_sft.ipynb`](../notebooks/evaluate_stage1_sft.ipynb), run setup and checkpoint
checks, then enable `RUN_STAGE1_EVAL`. Leave `INCLUDE_EARLIER_EPOCHS = False` in the check
cell to evaluate epoch 5 only (630 responses for `greedy`, 3,040 for `sample8`). To evaluate epochs 1–4 too, set it to
`True` and rerun that cell; the order becomes 5, 1, 2, 3, 4. All selected checkpoints
use all five benchmarks and the original evaluation environment. Do not run setup while
training or evaluation is active.

Select the same `EVAL_PROFILE` in the baseline and SFT notebooks:

| Profile | Temperature | AIME/AMC answers | MATH-500 answers | Responses/checkpoint | Metrics |
| --- | --- | --- | --- | --- | --- |
| `greedy` (default) | 0 | 1 | 1 | 630 | pass@1 / accuracy |
| `sample8` | 1.0 | 8 | 4 | 3,040 | AIME/AMC pass@1/4/8; MATH-500 pass@1/4; avg@n |

`sample8` retains top-p 0.7; greedy sets top-p 1.0 (sampling truncation is unused).
Both keep the 20,480-token generation limit and all 630 benchmark questions. Profile
settings live in [`eval_profiles.yaml`](../configs/eval_profiles.yaml), as overrides
of the original English config. Training settings remain in `stage1_sft.yaml`.

To switch an active Colab session, stop evaluation, push the updated code, and run the
updated notebook from setup. Either model can be evaluated first. The baseline notebook
has a separate smoke check: its default Run all performs a five-problem smoke
check; enable `RUN_FULL_EVAL` for the selected full baseline. The SFT evaluation
notebook uses the same profile; it can run before or after the baseline. It defaults to setup/checks only; enable `RUN_STAGE1_EVAL`.
To run both options, complete the baseline and SFT workflow once for each profile.

New profiles start fresh, including the base model. They do not reuse results generated
under the old 32-sample protocol. Old sequential and `_batched` files remain intact.
The root contains separate `profiles/greedy/` and `profiles/sample8/` directories, each
with isolated smoke/full protocols and runtime manifests. Whichever model runs first
creates these manifests; later runs validate them. SFT requires no baseline files. Within a run, keep the
same code commit, profile and settings to resume completed problems. Switching profiles
requires rerunning the notebook's setup/check cells before evaluation.

## CLI

Run from a committed, clean repository checkout on the GPU host:

```bash
python -m pip install uv==0.11.22
python scripts/setup_stage1_runtime.py --venv /content/lg-sft-env

/content/lg-sft-env/bin/python train/stage1_sft.py \
  --config configs/stage1_sft.yaml --prepare-only \
  --output_root /content/drive/MyDrive/LG-AIME-Stage1

/content/lg-sft-env/bin/python train/stage1_sft.py \
  --config configs/stage1_sft.yaml \
  --output_root /content/drive/MyDrive/LG-AIME-Stage1
```

The config pins the base model to the same loader/weights revision as stage 0 and the
published s1K dataset to `6d177bf60a195dc3c2b43de6a8eff12405a1c942`. Loading verifies all
original columns by content digest and verifies IDs against the pinned provenance.
Only `question`, `deepseek_thinking_trajectory` and `deepseek_attempt` enter training.
`deepseek_grade` is counted for reporting, never used to filter or select examples.

Full-parameter training uses five epochs, batch 1 × accumulation 16, AdamW, LR 1e-5,
5% linear warmup followed by cosine decay, BF16 and gradient checkpointing. All settings,
including framework overrides, are in the config. A short FlashAttention-2 BF16
forward/backward probe selects it if installed and working; otherwise SDPA is selected
and the reason is recorded. The supplied training lock uses SDPA without requiring a
FlashAttention build. It never falls back to eager attention. CUDA OOM is fatal.

Formatting produces `<think>\n{reasoning}\n</think>\n\n{attempt}`. These are ordinary
text tokens. There is no tokenizer/template replacement or embedding resize. Labels
mask the prompt, assistant header, padding and trailing template newline; only assistant
content and its existing end-of-turn token are supervised. Token boundaries are checked
against the frozen inference prompt. EXAONE's own tokenizer performs NFKC normalization;
the source text itself is never edited. Overlength examples are dropped before TRL.

The pinned TRL 0.24.0 `SFTTrainer` receives pretokenized IDs, explicit labels and a custom
padding collator. Dataset preparation, packing and truncation are disabled. A small
subclass uses the native model cross-entropy through Transformers' Trainer, avoiding
TRL's extra full-vocabulary entropy/accuracy allocations on long sequences. Accumulation
uses the mean loss per example, including the final partial accumulation group each epoch.
No validation split, early stopping, learned reward, budget forcing or best-epoch selection
is used.

## Artifacts and resumption

Relative to `--output_root` (default `.`):

- `checkpoints/stage1/sft_s1k/epoch_1/` through `epoch_5/`: full model, native tokenizer,
  EXAONE loader Python files, optimizer/scheduler/RNG and Trainer state. Epoch 5 is the
  Stage 1 result. All five are retained; allow at least 100GB free storage.
- `logs/stage1/sft_s1k/data_report.json`: kept/dropped IDs, lengths, grade counts,
  tokenizer identity, data pins and expected step count.
- `logs/stage1/sft_s1k/sanity_check.txt`: full formatted example and visible loss region.
- `logs/stage1/sft_s1k/steps.jsonl`: loss, learning rate, pre-clipping gradient norm,
  nonpadding input tokens/second, token count and elapsed time at every optimizer step.
- `run_manifest.json`, `config.yaml`, `attempts.jsonl`, `training_summary.json` and
  `train_metrics.json` in that log directory record provenance, runtime, status,
  training time and peak allocated/reserved GPU memory. Throughput covers accumulation
  and the optimizer step; total wall time also includes model loading and checkpoint saves.

Preparation-only reports live in `logs/stage1/sft_s1k/preparation/`, separate from run logs.
Interrupted training can resume explicitly from the latest complete epoch:

```bash
/content/lg-sft-env/bin/python train/stage1_sft.py \
  --config configs/stage1_sft.yaml \
  --output_root /content/drive/MyDrive/LG-AIME-Stage1 \
  --resume_from_checkpoint /content/drive/MyDrive/LG-AIME-Stage1/checkpoints/stage1/sft_s1k/epoch_2
```

Keep the same Git commit, config and runtime. Epoch saves are first written into a
temporary `checkpoint-<step>` directory and renamed after all files finish. Logs from
an interrupted, unsaved epoch are archived and that epoch is replayed. Starting again
without an explicit resume refuses to overwrite an existing run. A hard process/runtime
kill may prevent final timing telemetry from being written; completed epoch state remains
the recovery point. The fixed seed records reproducibility settings but does not guarantee
bitwise equality across different GPU kernels or hardware.

## Evaluation

Use the existing evaluation environment, never the training environment. These commands
show the default greedy profile; use `--profile sample8` for the second option on **both**
base and SFT runs.

```bash
python scripts/setup_eval_runtime.py --venv /content/lg-eval-env

# Base model: add --smoke for a separate five-problem loading/generation check.
/content/lg-eval-env/bin/python -m eval.run_batched_eval \
  --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
  --stage stage0 --run_name baseline_english --profile greedy \
  --output_root /content/drive/MyDrive/LG-AIME-English-Eval-compatible

# SFT epoch 5, using the same profile.
/content/lg-eval-env/bin/python -m eval.run_batched_eval \
  --model /content/drive/MyDrive/LG-AIME-Stage1/checkpoints/stage1/sft_s1k/epoch_5 \
  --stage stage1 --run_name sft_s1k --profile greedy \
  --output_root /content/drive/MyDrive/LG-AIME-English-Eval-compatible
```

The CLI puts profile runs under `<output_root>/profiles/<profile>/full/results/`.
Baseline results are `stage0/baseline_english.json`; epoch 5 is `stage1/sft_s1k.json`;
optional earlier epochs use `stage1/sft_s1k_epoch{1..4}.json`. Smoke runs live under the
profile's `smoke/` directory and never substitute for a full baseline.
SFT console logs use `full_eval_epoch{epoch}_{profile}_console.log` in the training
Drive root. Epoch 5 remains the Stage 1 result; earlier epochs are descriptive.

Continuous batching uses vLLM 0.14.1's `LLMEngine.add_request`/`step` interface. The GPU
still runs up to 32 responses at once with a 90% memory budget. The greedy profile
queues up to 64 problems so its single-answer requests can fill the GPU; `sample8`
queues up to 16. The queue is refilled as problems finish. Status is printed every
30 seconds while generation advances. The notebook bar counts completed problems,
which can finish out of order within each benchmark; datasets keep their original order.

The `*_problems.jsonl` journal saves all scored responses for a completed problem in one
durable append. On interruption, complete problems are skipped and a torn final append
is discarded before regenerating that problem. The standard `*_generations.jsonl` export
is written at completion, ordered by dataset/problem/sample. Manifests record the
profile, resolved scientific/execution settings, runner digest, Git commit and runtime.
Baseline and SFT protocols must match within a profile. Exact text can still vary with
GPU numerics and batching; no bitwise equivalence or measured speedup is promised.

The original `eval.run_english_eval` and the no-profile `eval.run_batched_eval` path remain
available for legacy settings. Existing batched runs must use their recorded Git commit
to resume. Named profiles reject legacy response imports instead of mixing protocols.
Rebuild the Stage 1 notebooks with `python scripts/build_stage1_notebook.py`; the baseline launcher source lives in `scripts/english_eval_notebook.py`.
The SFT training notebook is unchanged.

## Local validation and remaining GPU work

The exact published s1K provenance was checked against the local English export. CPU
preparation with the real pinned EXAONE tokenizer retained **975** examples and dropped
**21** over 20,480 tokens. Kept lengths: minimum 1,035; median 10,241; p90 17,458.4;
p99 19,344.86; maximum 20,341. The original 996 rows include **370** incorrect grades.
The resulting schedule is **61 steps/epoch, 305 steps total** (the last accumulation
group has 15 examples). See [`stage1_data_report.json`](stage1_data_report.json).

Profile tests cover temperature-0 generation settings, exact sample counts, baseline/SFT
protocol matching, isolated smoke/full outputs, and interruption/resume behavior with a
simulated vLLM engine. Real GPU smoke tests for the new profiles remain to be run in Colab.

Local tests cover label boundaries, strict overlength dropping, grade retention, padding,
AMC protocol preservation and evaluation resumption. A tiny CPU training test exercises
the pinned SFTTrainer, partial accumulation, epoch exports, model/tokenizer reload and
checkpoint resumption. A separate reduced-size EXAONE CPU check with the actual pinned
remote model code passed forward/backward, gradient checkpointing, epoch export and
offline reload with an unchanged tokenizer. These checks do not establish 20K-token GPU
memory use or training quality.

GPU training, full-size model memory/throughput validation and Stage 1 evaluation results
have not been run on this Mac. The preflight data report is real tokenizer output;
it is not a training or accuracy result. Do not report any Stage 1 gain until the GPU
run and evaluations complete.

Implementation references: [TRL 0.24 SFTTrainer](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer),
[Transformers 4.57.6 Trainer source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/trainer.py),
[pinned EXAONE model files](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct/tree/e949c91dec92095908d34e6b560af77dd0c993f8).
