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
is loaded through vLLM's public `apply_model` API before each update's sampling; prefix cache
is reset. This transport adds local disk traffic but avoids fragile cross-process parameter
references. Only one temporary snapshot is retained. The stock TRL train-vs-rollout importance
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
