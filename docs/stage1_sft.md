# Stage 1: English s1-style SFT

The implementation uses the existing English prompt and the native EXAONE chat template.
It leaves every frozen evaluation source, config, suite and dependency lock unchanged.
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
cell to evaluate epoch 5 only (6,160 responses). To evaluate epochs 1–4 too, set it to
`True` and rerun that cell; the order becomes 5, 1, 2, 3, 4. All selected checkpoints
use all five benchmarks and the original evaluation environment. Do not run setup while
training or evaluation is active.

To switch an already-running Colab session to continuous batching, stop the evaluation
cell first. Push the updated repository, reopen the updated evaluation notebook, and run
setup and checks before enabling evaluation. `REUSE_COMPLETED_LEGACY = True` copies every
fully completed problem from the matching earlier sequential run into a separate
`_batched` run. It checks model contents, scientific protocol, datasets, runtime,
hardware and prompts; a different Git commit is expected for this explicit migration.
Incomplete problems are regenerated in full. Original results are not edited or deleted.
Do not resume the old writer while its results are being reused. After starting a batched
run, retain its code commit, execution config and legacy source files for resume.
Set `REUSE_COMPLETED_LEGACY = False` before starting a new run to generate everything anew.

The training notebook defaults to preparation only; the evaluation notebook defaults to
setup and checks only. Training settings live in
[`stage1_sft.yaml`](../configs/stage1_sft.yaml), not notebook cells. The baseline Drive root
must still contain `full/results/eval_protocol.json` and `eval_runtime.json` from the real
stage-0 run. A ZIP containing only `results/stage0/` is insufficient: preserve the two
parent manifests as well. Do not edit/recreate them to bypass a compatibility failure.

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

Use the existing evaluation environment, never the training environment:

```bash
python scripts/setup_eval_runtime.py --venv /content/lg-eval-env

# Main result only. Optionally use "5 1 2 3 4" to include earlier checkpoints afterward.
for epoch in 5; do
  run_name="sft_s1k_epoch${epoch}"
  if [ "$epoch" = 5 ]; then
    run_name="sft_s1k"
  fi
  /content/lg-eval-env/bin/python -m eval.run_batched_eval \
    --model "/content/drive/MyDrive/LG-AIME-Stage1/checkpoints/stage1/sft_s1k/epoch_${epoch}" \
    --stage stage1 --run_name "${run_name}_batched" --reuse_run_name "$run_name" \
    --execution_config configs/eval_execution.yaml \
    --output_root /content/drive/MyDrive/LG-AIME-English-Eval-compatible
done
```

The batched CLI adds `/full/` to the supplied root. Results live in
`<baseline_root>/full/results/stage1/sft_s1k_batched.json` for epoch 5 and
`sft_s1k_epoch{1..4}_batched.json` for optional earlier epochs. The original sequential
files remain intact. Each full evaluation requires 6,160 responses across AIME 2024,
AIME 2025, AIME 2026, AMC 2023 and MATH-500, minus any responses reused from completed
problems. Console logs are `full_eval_epoch{1..5}_batched_console.log` in the training
Drive root. Epoch 5 remains the Stage 1 result; earlier epochs are descriptive.

[`eval_execution.yaml`](../configs/eval_execution.yaml) queues up to 16 problems using
vLLM 0.14.1's `LLMEngine.add_request`/`step` interface. The existing 32 concurrent GPU
response limit and 90% memory budget remain unchanged. When one response finishes,
vLLM can fill its slot from another problem instead of waiting for all responses to
the current problem. This also lets multiple four-response MATH-500 problems run together.
The queue is refilled as whole problems finish. Status is printed every 30 seconds
while generation advances; the notebook bar counts completed problems even when they
finish out of order. Datasets still run in their original order.

The new `*_problems.jsonl` journal saves all scored responses for a completed problem
in one durable append. On interruption, complete problems are skipped and a torn final
append is discarded before regenerating that problem. This avoids mixing old partial
samples with a changed stochastic replay. The familiar flat `*_generations.jsonl`
export is written at completion, ordered by dataset/problem/sample. Each run manifest
records the new runner digest, execution config, and any reused source provenance.
The baseline's protocol/runtime manifests remain untouched.

All runs retain temperature 1.0, top-p 0.7, sample counts, token limits, prompts, per-problem
seeds, native n-way child seeding, answer extraction and scoring. Batch composition can
change exact sampled text; this is an execution change, not a claim of bitwise identity
with the sequential baseline. No GPU speedup has been measured locally. Generation
lengths, scoring, Drive I/O and the tail of each dataset can still limit throughput.

Rebuild both Stage 1 notebooks with `python scripts/build_stage1_notebook.py` after
changing the launcher source. The training notebook is unaffected by this optimization.

## Local validation and remaining GPU work

The exact published s1K provenance was checked against the local English export. CPU
preparation with the real pinned EXAONE tokenizer retained **975** examples and dropped
**21** over 20,480 tokens. Kept lengths: minimum 1,035; median 10,241; p90 17,458.4;
p99 19,344.86; maximum 20,341. The original 996 rows include **370** incorrect grades.
The resulting schedule is **61 steps/epoch, 305 steps total** (the last accumulation
group has 15 examples). See [`stage1_data_report.json`](stage1_data_report.json).

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
