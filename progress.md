# Qwen experiment roadmap

Updated: 2026-10-06. This is the current roadmap provided by the user; previous plans in `README.md` and `write-up.md` do not define this experiment list.

Model family: **Qwen2.5-3B**, matching the current instruct + RL run. Base revision is now pinned to `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`; future training settings remain to be decided. Status reflects reported progress, not live monitoring of Drive.

## Progress

Legend: — = not required; ☐ = pending; ◐ = running; ☑ = complete.

| ID | Experiment | Training | Evaluation | Next dependency |
| --- | --- | --- | --- | --- |
| B0 | Qwen base | — | ☑ Complete (user reported) | — |
| B-RL | Qwen base + RL | ☐ Planned | ☐ Pending | Base RL notebook/settings |
| B-SFT1 | Qwen base + SFT-v1 (long: OpenR1-Math-220k) | ☐ Planned | ☐ Pending | User will provide exact HF dataset and columns |
| B-SFT2 | Qwen base + SFT-v2 (short: Kimi-style DeepSeek) | ☐ Planned | ☐ Pending | Dataset and target column supplied; pin revision/split, confirm question column, and prepare training notebook |
| B-SFT-RL | Qwen base + SFT + RL | ☐ Planned | ☐ Pending | Decide SFT-v1 or SFT-v2 starting checkpoint |
| I0 | Qwen instruct | — | ☑ Complete (user reported) | — |
| I-RL | Qwen instruct + RL | ☑ Complete (user reported) | ☑ Complete (user reported) | — |

## Evaluation scores

**Implemented B0/I0/I-RL protocol:** AMC 2023 (40 problems) and MATH-500 (500), temperature 0, top-p 1, one response per problem, 20,480-token response cap. Same English question/instruction and existing final-box/math-equivalence scorer; each model uses its shipped native chat template/system message and EOS token. Base uses zero-shot chat formatting, not a few-shot completion protocol. Instruct/RL templates must match. Exact model/data/code/protocol records accompany the results.

**Results received 2026-10-06:** scores and output paths below were supplied by the user from the completed evaluation. Drive artifacts have not been independently rechecked here; HF revisions are the notebook's pinned selections.

Notebook: [evaluate_qwen25_3b_base_instruct_rl_amc_math.ipynb](notebooks/evaluate_qwen25_3b_base_instruct_rl_amc_math.ipynb). Self-contained, sequential evaluation of three models, optional six-response smoke, then `RUN_EVAL = True` for 1,620 responses. Resumes completed problems separately for each model.

| ID | Evaluated checkpoint / HF revision | AMC correct / 40 | AMC accuracy | AMC mean tokens | MATH correct / 500 | MATH accuracy | MATH mean tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | `Qwen/Qwen2.5-3B` @ `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | 15 | 37.5% | 1745.6 | 274 | 54.8% | 1444.2 |
| B-RL | TBD | — | — | — | — | — | — |
| B-SFT1 | TBD | — | — | — | — | — | — |
| B-SFT2 | TBD | — | — | — | — | — | — |
| B-SFT-RL | TBD | — | — | — | — | — | — |
| I0 | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 21 | 52.5% | 1780.1 | 350 | 70.0% | 780.4 |
| I-RL | MemoryFix `checkpoint-300` | 20 | 50.0% | 1810.4 | 329 | 65.8% | 1012.7 |

## Training information

| ID | Starting model | Training data / target | Epochs or optimizer steps |
| --- | --- | --- | --- |
| B0 | `Qwen/Qwen2.5-3B` @ `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | None | No training |
| B-RL | Same base revision as B0 | RL dataset/revision and settings TBD | TBD |
| B-SFT1 | Same base revision as B0 | OpenR1-Math-220k; exact HF repo, revision, split, question and target columns pending | TBD |
| B-SFT2 | Same base revision as B0 | [Seungjun/dp_removed_s1K-1.1](https://huggingface.co/datasets/Seungjun/dp_removed_s1K-1.1); target column: `kimi-style-reasoning-answer`; revision, split, and question column pending confirmation | TBD |
| B-SFT-RL | B-SFT1 or B-SFT2 checkpoint: TBD | RL dataset/revision and settings TBD | TBD |
| I0 | `Qwen/Qwen2.5-3B-Instruct` | None; use the same starting revision as I-RL | No training |
| I-RL | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `Seungjun/dp_removed_DAPO-Math-17k-Processed`, `en` / `train`; 14,068 rows; exact resolved revision in run manifest | 300 total; recovered from step 40 |

**Current I-RL settings:** full-parameter DAPO, G=8, 16 retained questions / 128 responses per rollout, two disjoint 64-response optimizer updates, LR 1e-6, warmup 20 updates, seed 42. Response cap 20,480; soft length penalty above 16,384; no KL or budget forcing. FP32 parameters/Adam state, BF16 compute, RTX PRO 6000 96GB. Checkpoints every 20 updates. Across 300 updates: 2,400 retained question selections, not necessarily unique.

**I-RL completion:** user reported training finished on 2026-10-06. The configured final checkpoint is step 300; the evaluation notebook verifies its completion marker, model shards, and training identity before evaluating. Recovery history: update 43 completed, then backward OOM occurred during update 44. Chunked vocabulary projection allowed restarting from checkpoint 40 in the `MemoryFix` folder. Updates 41 onward were replayed; the total remains 300. Training notebook: [train_dapo_qwen25_3b.ipynb](notebooks/train_dapo_qwen25_3b.ipynb). Config: [dapo_qwen25_3b.yaml](configs/dapo_qwen25_3b.yaml).

## Google Drive paths

These are absolute **Colab-mounted Drive paths** (`/content/drive/MyDrive/` = My Drive). B0/I0 evaluation destinations and the I-RL training location are user reported; future training roots remain planned.

| ID | Run root | Path status |
| --- | --- | --- |
| B0 | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/eval-base-instruct-rl-amc-math-temp0/base` | Completed evaluation outputs (user reported) |
| B-RL | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl` | Planned |
| B-SFT1 | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v1-long` | Planned |
| B-SFT2 | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short` | Planned |
| B-SFT-RL | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-rl` | Planned; record SFT variant before launch |
| I0 | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/eval-base-instruct-rl-amc-math-temp0/instruct` | Completed evaluation outputs (user reported) |
| I-RL | `/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix` | Current configured recovery run |

**Planned layout for new runs:** `<run root>/checkpoints/`, `<run root>/logs/steps.jsonl`, and `<run root>/logs/run_manifest.json`. B0/I0 need evaluation outputs only. Before each training launch, record the notebook's actual paths here if its layout differs.

**Completed B0/I0/I-RL comparison output root (user reported):**

```text
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/eval-base-instruct-rl-amc-math-temp0/
```

Under that root:

| ID | Metrics file | Generated responses file |
| --- | --- | --- |
| B0 | `base/full/results/qwen_base.json` | `base/full/results/qwen_base_generations.jsonl` |
| I0 | `instruct/full/results/qwen_instruct.json` | `instruct/full/results/qwen_instruct_generations.jsonl` |
| I-RL | `instruct_rl/full/results/qwen_instruct_rl_step300.json` | `instruct_rl/full/results/qwen_instruct_rl_step300_generations.jsonl` |

`comparison_table.csv` and `comparison_table.json` contain all six benchmark/model rows.
`model_selection.json` records chosen models/checkpoint. Each model's `full/results/` also holds
`<run_name>_manifest.json`, `<run_name>_engine.json`, `<run_name>_problems.jsonl` (resume journal),
and protocol/runtime records. Smoke files use `smoke/` instead of `full/`; console logs are in
each model's `logs/`. The user reported `comparison_table.csv` and all six metrics/generations
paths in the table above; the remaining supporting filenames describe the notebook's output layout.

**Planned evaluation directory for future trained rows:** `<run root>/eval/amc2023_math500_temp0/`.

**I-RL exact current locations:**

```text
# New loss / reward / entropy / length history (step 41 onward)
/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix/logs/dapo_qwen25_3b_base/steps.jsonl
# Console log
/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix/logs/dapo_qwen25_3b_base/training_console.log
# Resolved training settings, runtime and recovery provenance
/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix/logs/dapo_qwen25_3b_base/run_manifest.json
# Final checkpoint; completion reported by user, validated on Drive by the eval notebook
/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix/checkpoints/dapo_qwen25_3b_base/checkpoint-300/
# Generated responses and retained-question records
/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix/logs/dapo_qwen25_3b_base/rollouts/
# Original history, including the abandoned updates after step 40
/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300/logs/dapo_qwen25_3b_base/steps.jsonl
# Recovery parent
/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300/checkpoints/dapo_qwen25_3b_base/checkpoint-40/
```

For a combined I-RL training curve, use original steps **1–40** and recovered steps **41 onward**; do not double-count the original abandoned steps 41–43.
