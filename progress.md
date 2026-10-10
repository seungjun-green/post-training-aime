# Qwen experiment roadmap

Updated: 2026-10-10. This is the current roadmap provided by the user; previous plans in `README.md` and `write-up.md` do not define this experiment list.

Model family: **Qwen2.5-3B**, matching the current instruct + RL run. Base revision is now pinned to `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`; future training settings remain to be decided. Status reflects reported progress, not live monitoring of Drive.

## Progress

Legend: — = not required; ☐ = pending; ◐ = running; ☑ = complete.

| ID | Experiment | Training | Evaluation | Next dependency |
| --- | --- | --- | --- | --- |
| B0 | Qwen base | — | ☑ Complete (user reported) | — |
| B-RL | Qwen base + RL | ☑ Complete: step 300 (user reported) | ☑ Complete (user reported) | — |
| B-s1(kimi style) SFT | Qwen base + s1 (Kimi-style) SFT | ☑ Complete (user reported) | ☑ Complete (user reported) | — |
| B-rejection sampling SFT | Qwen base + rejection sampling SFT | ☑ Complete: 5 epochs (user confirmed) | ☑ Complete; epoch 4 best (user confirmed) | — |
| B-7B knowledge distillation SFT | Qwen base + 7B knowledge distillation SFT | ☐ Notebook ready (5 epochs) | ☐ Automatic after each epoch | Run preparation, GPU smoke, then training/evaluation loop |
| B-rejection sampling SFT + RL | Qwen base + rejection sampling SFT + RL | ☐ Planned | ☐ Pending | Complete rejection sampling SFT; choose checkpoint and RL settings |
| B-7B knowledge distillation SFT + RL | Qwen base + 7B knowledge distillation SFT + RL | ☐ Planned | ☐ Pending | Complete 7B distillation SFT; choose checkpoint and RL settings |
| I0 | Qwen instruct | — | ☑ Complete (user reported) | — |
| I-RL | Qwen instruct + RL | ☑ Complete (user reported) | ☑ Complete (user reported) | — |

## Evaluation scores

**Implemented B0/I0/I-RL protocol:** AMC 2023 (40 problems) and MATH-500 (500), temperature 0, top-p 1, one response per problem, 20,480-token response cap. Same English question/instruction and existing final-box/math-equivalence scorer; each model uses its shipped native chat template/system message and EOS token. Base uses zero-shot chat formatting, not a few-shot completion protocol. Instruct/RL templates must match. Exact model/data/code/protocol records accompany the results.

**Results received 2026-10-06; B-RL added 2026-10-07; rejection sampling SFT added 2026-10-10:** scores and output paths below were supplied by the user from the completed evaluation. Drive artifacts have not been independently rechecked here; HF revisions are the notebook's pinned selections.

Notebook: [evaluate_qwen25_3b_base_instruct_rl_amc_math.ipynb](notebooks/evaluate_qwen25_3b_base_instruct_rl_amc_math.ipynb). Self-contained, sequential evaluation of three models, optional six-response smoke, then `RUN_EVAL = True` for 1,620 responses. Resumes completed problems separately for each model.

| ID | Evaluated checkpoint / HF revision | AMC correct / 40 | AMC accuracy | AMC mean tokens | MATH correct / 500 | MATH accuracy | MATH mean tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | `Qwen/Qwen2.5-3B` @ `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | 15 | 37.5% | 1745.6 | 274 | 54.8% | 1444.2 |
| B-RL | `dapo_qwen25_3b_pretrained_base/checkpoint-300` | 17 | 42.5% | 2340.3 | 321 | 64.2% | 895.1 |
| B-s1(kimi style) SFT | `sft_qwen25_3b_base_s1_kimi/epoch_5` | 10 | 25.0% | 8556.8 | 257 | 51.4% | 6709.7 |
| B-rejection sampling SFT | `sft_qwen25_3b_base_self_rft/epoch_4` (user-reported best) | 15 | 37.5% | 2168.2 | 307 | 61.4% | 1032.2 |
| B-7B knowledge distillation SFT | TBD | — | — | — | — | — | — |
| B-rejection sampling SFT + RL | TBD | — | — | — | — | — | — |
| B-7B knowledge distillation SFT + RL | TBD | — | — | — | — | — | — |
| I0 | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 21 | 52.5% | 1780.1 | 350 | 70.0% | 780.4 |
| I-RL | MemoryFix `checkpoint-300` | 20 | 50.0% | 1810.4 | 329 | 65.8% | 1012.7 |

## Training information

| ID | Starting model | Training data / target | Epochs or optimizer steps |
| --- | --- | --- | --- |
| B0 | `Qwen/Qwen2.5-3B` @ `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | None | No training |
| B-RL | Same base revision as B0 | `Seungjun/dp_removed_DAPO-Math-17k-Processed`, `en` / `train`; 14,068 verified rows; resolved revision saved in run manifest | 300 optimizer updates; 150 rollout batches |
| B-s1(kimi style) SFT | Same base revision as B0 | [Seungjun/dp_removed_s1K-1.1](https://huggingface.co/datasets/Seungjun/dp_removed_s1K-1.1) @ `636ecf409774771afb0bf10a436f4b1e608b5f29`; `default` / `train`; input `question`, full target `kimi-style-reasoning-answer`; 996 source rows, 989 retained | 5 epochs; 62 updates/epoch, 310 total |
| B-rejection sampling SFT | Same base revision as B0 | [Seungjun/qwen2.5-3b-self-rft-math](https://huggingface.co/datasets/Seungjun/qwen2.5-3b-self-rft-math) @ `96c5f70e37098a9f41f5e44adc4e906ee3f5ef15`; `default` / `train`; input `problem`, target full `response`; 28,267 rows, 12,863 unique problem IDs | 5 epochs; 1,767 updates/epoch, 8,835 total |
| B-7B knowledge distillation SFT | Same base revision as B0 | [Seungjun/qwen2.5-3b-self-rft-7b-distill-math](https://huggingface.co/datasets/Seungjun/qwen2.5-3b-self-rft-7b-distill-math) @ `d13f567cd42e7f7e182ea5125a288d86cbefbe67`; input `problem`, target full `response`; 37,691 rows / 14,853 unique problems; both 3B and 7B response sources | 5 epochs; 2,356 updates/epoch, 11,780 total |
| B-rejection sampling SFT + RL | B-rejection sampling SFT checkpoint (TBD) | RL dataset and settings TBD | TBD |
| B-7B knowledge distillation SFT + RL | B-7B knowledge distillation SFT checkpoint (TBD) | RL dataset and settings TBD | TBD |
| I0 | `Qwen/Qwen2.5-3B-Instruct` | None; use the same starting revision as I-RL | No training |
| I-RL | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `Seungjun/dp_removed_DAPO-Math-17k-Processed`, `en` / `train`; 14,068 rows; exact resolved revision in run manifest | 300 total; recovered from step 40 |

**Current I-RL settings:** full-parameter DAPO, G=8, 16 retained questions / 128 responses per rollout, two disjoint 64-response optimizer updates, LR 1e-6, warmup 20 updates, seed 42. Response cap 20,480; soft length penalty above 16,384; no KL or budget forcing. FP32 parameters/Adam state, BF16 compute, RTX PRO 6000 96GB. Checkpoints every 20 updates. Across 300 updates: 2,400 retained question selections, not necessarily unique.

**B-RL notebook ready:** [train_dapo_qwen25_3b_base.ipynb](notebooks/train_dapo_qwen25_3b_base.ipynb), config [dapo_qwen25_3b_base.yaml](configs/dapo_qwen25_3b_base.yaml). Starts directly from `Qwen/Qwen2.5-3B` @ `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. Same DAPO data, optimization, sampling, rollout, and smoke settings as I-RL; includes the chunked Qwen vocabulary-projection memory fix. Preserves the base model's shipped chat template/default system message and original EOS/PAD: `<|endoftext|>` (151643). No tokenizer EOS override, extra stop tokens, or suppression of model EOS. Completion masks use actual response lengths so generated EOS receives loss while padding does not. No SFT starting checkpoint or recovery parent. Run name: `dapo_qwen25_3b_pretrained_base`.

Preparation with the real pinned base tokenizer verified all 14,068 local dataset rows and provenance digests; all prompts fit (49–1,525 tokens), with no truncation. Full training starts from the base weights, saves every 20 updates, and automatically resumes the latest completed checkpoint in its own folder. Optional two-update GPU smoke has separate timestamped outputs. GPU training has not been run here. The 300-update budget covers 2,400 retained question selections, potentially with repeats; dynamic sampling may generate additional candidates.

**B-RL restart history:** the previous attempt used an EOS override and the last supplied log showed 0/300 optimizer updates. Its files remain under `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl/`. Stop that runtime and use the rebuilt notebook in a fresh Colab runtime. The corrected run uses `base-rl-native-eos/` and must start from the original base weights, not the previous attempt. Resume validation rejects manifests with the old tokenizer/generation settings. Local checks passed (23 tests, including saved EOS/PAD and resume behavior); GPU speed has not been verified.

**B-RL completion (user reported):** the native-EOS base + RL run has completed all 300 optimizer updates. [evaluate_qwen25_3b_base_rl_checkpoint300_amc_math.ipynb](notebooks/evaluate_qwen25_3b_base_rl_checkpoint300_amc_math.ipynb) evaluates only `checkpoint-300` on AMC 2023 (40) and MATH-500 (500), temperature 0, one response per problem, 20,480-token cap. Preserves original `<|endoftext|>` EOS and the saved chat template, including `chat_template.jinja`. Self-contained notebook with optional two-problem smoke, full evaluation via `RUN_EVAL = True`, and resume. Evaluation completed (user reported 2026-10-07): AMC 2023 **17/40 (42.5%)**, mean **2340.3 tokens**; MATH-500 **321/500 (64.2%)**, mean **895.1 tokens**. Relative to B0, accuracy increased by **5.0** and **9.4 percentage points**, respectively. Scores and output paths were supplied by the user; Drive artifacts have not been independently inspected here.

```text
# Completed step-300 evaluation output root (user reported; separate from checkpoint-140 results)
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/eval/amc2023_math500_temp0/dapo_qwen25_3b_pretrained_base/checkpoint-300/
```

Under that root, `full/results/qwen_base_rl_step300.json` stores metrics, `full/results/qwen_base_rl_step300_generations.jsonl` stores answers and scores, and `full/summary_table.json` / `.csv` contain the benchmark table. Console logs are under `logs/` and smoke results under `smoke/`.

**Exact step-300 evaluation files (user reported):**

```text
# Metrics
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/eval/amc2023_math500_temp0/dapo_qwen25_3b_pretrained_base/checkpoint-300/full/results/qwen_base_rl_step300.json
# Benchmark table
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/eval/amc2023_math500_temp0/dapo_qwen25_3b_pretrained_base/checkpoint-300/full/summary_table.json
# Generated answers and scores
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/eval/amc2023_math500_temp0/dapo_qwen25_3b_pretrained_base/checkpoint-300/full/results/qwen_base_rl_step300_generations.jsonl
```

**B-RL intermediate checkpoint history (2026-10-07, user reported):** training had completed step 140 of 300 (46.7%). [evaluate_qwen25_3b_base_rl_checkpoint140_amc_math.ipynb](notebooks/evaluate_qwen25_3b_base_rl_checkpoint140_amc_math.ipynb) evaluates only the saved `checkpoint-140` on AMC 2023 (40) and MATH-500 (500), temperature 0, one response per problem, 20,480-token cap. Preserves the checkpoint's native `<|endoftext|>` EOS and saved chat template. Self-contained bundle, optional two-problem smoke, separate full evaluation with resume. No step-140 scores reported yet; Drive checkpoint validation runs in Colab. Use a separate GPU runtime if training continues.

```text
# Checkpoint evaluated
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/checkpoints/dapo_qwen25_3b_pretrained_base/checkpoint-140/
# Evaluation output root
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/eval/amc2023_math500_temp0/dapo_qwen25_3b_pretrained_base/checkpoint-140/
```

Under that evaluation root, `full/results/qwen_base_rl_step140.json` stores metrics, `full/results/qwen_base_rl_step140_generations.jsonl` stores responses and scores, and `full/summary_table.json` / `.csv` hold the benchmark table. Evaluation console logs are under `logs/`; smoke results are under `smoke/`.

**I-RL completion:** user reported training finished on 2026-10-06. The configured final checkpoint is step 300; the evaluation notebook verifies its completion marker, model shards, and training identity before evaluating. Recovery history: update 43 completed, then backward OOM occurred during update 44. Chunked vocabulary projection allowed restarting from checkpoint 40 in the `MemoryFix` folder. Updates 41 onward were replayed; the total remains 300. Training notebook: [train_dapo_qwen25_3b.ipynb](notebooks/train_dapo_qwen25_3b.ipynb). Config: [dapo_qwen25_3b.yaml](configs/dapo_qwen25_3b.yaml).

**B-s1(kimi style) SFT completion (user reported):** status `complete`, five epochs / 310 optimizer updates, no resume, total training time 1,676.619 seconds (27m 56.6s). Peak GPU allocated/reserved: 31,371,308,032 / 32,682,016,768 bytes. Run identity: `5644b316cb49344bea968be58871bc87a6bae072f50309bcfba18a1ce6e577ef`. Drive artifacts have not been independently inspected here.

**B-s1(kimi style) SFT evaluation notebook:** [evaluate_qwen25_3b_s1_kimi_amc_math.ipynb](notebooks/evaluate_qwen25_3b_s1_kimi_amc_math.ipynb). Defaults to this run’s epoch-5 checkpoint, validates its identity and saved EOS, then uses the same AMC 2023 / MATH-500 greedy protocol as the earlier Qwen comparison. Optional two-problem smoke; `RUN_EVAL = True` evaluates all 540 problems with resume. User reported completed evaluation: AMC 10/40 (25.0%), MATH-500 257/500 (51.4%). Relative to B0, accuracy decreased by 12.5 and 3.4 percentage points, while mean output tokens increased approximately 4.9× and 4.6×, respectively. Generated responses have not been inspected to determine the cause.

**B-s1(kimi style) SFT training:** [train_qwen25_3b_base_s1_kimi.ipynb](notebooks/train_qwen25_3b_base_s1_kimi.ipynb), config [sft_qwen25_3b_s1_kimi.yaml](configs/sft_qwen25_3b_s1_kimi.yaml). Full-parameter Qwen base SFT, BF16, batch 1, accumulation 16, LR 1e-5, cosine schedule, 5% warmup, seed 42, gradient checkpointing, and chunked vocabulary projection. Uses the base model's native chat template and the evaluation English instruction. Entire target text is preserved; only assistant target and native `<|im_end|>` receive loss. No added reasoning tags. EOS is set to the existing `<|im_end|>` for chat training/generation; padding is `<|endoftext|>`. No vocabulary additions.

Actual-data preparation verified 989 nonempty targets, 7 missing targets, and 0 rows above the 20,480-token full-sequence cap. Full-sequence median/max: 1,860 / 4,261 Qwen tokens; mean supervised target length including EOT: 1,732.8. No truncation or correctness filter. Five epochs cover all 989 retained examples each epoch (4,945 example presentations). The user reported GPU training complete at step 310. The notebook includes an optional one-update smoke on the longest example, epoch saves, and automatic resume from the latest completed epoch.

**B-rejection sampling SFT notebook:** [train_qwen25_3b_base_self_rft.ipynb](notebooks/train_qwen25_3b_base_self_rft.ipynb), config [sft_qwen25_3b_self_rft.yaml](configs/sft_qwen25_3b_self_rft.yaml). Full-parameter Qwen base SFT for five epochs, BF16, batch 1, accumulation 16, LR 1e-5, cosine schedule, 5% warmup, seed 42, gradient checkpointing, and chunked vocabulary projection. Preserves the base model's shipped chat template for the prompt and native EOS/PAD `<|endoftext|>`. The supervised completion is the full unmodified `response` plus native EOS; no assistant `<|im_end|>` suffix is appended. Prompt and padding labels are masked by position. Multiple solutions per problem remain separate training examples; no new correctness filter or deduplication.

Local preparation on the pinned dataset and real base tokenizer retained all **28,267** rows, with **0** missing/overlength targets and no truncation. Full-sequence median/max: **460 / 2,839** tokens. Mean supervised response including EOS: **429.9** tokens. Five epochs present **141,335** rows and take **8,835** optimizer updates. The notebook embeds its code/config, saves checkpoints every epoch, and resumes from the latest completed epoch. Optional one-update smoke uses the longest example in a separate run. Local validation: **17 focused tests passed**, covering native-EOS loss masking, full-model save/reload, exact CPU equivalence between interrupted/resumed and uninterrupted training, isolated notebook bundles, epoch evaluation ordering/resume, and compact progress output. GPU training has not been performed locally.

**Evaluation after every SFT epoch:** each epoch checkpoint is evaluated on AMC 2023 (40 problems) and MATH-500 (500) at temperature 0, one response per problem, max 20,480 tokens, using the existing scorer. This is **2,700 responses** across five epochs. The training subprocess exits after its durable epoch save so vLLM can use the GPU; the next training process resumes optimizer, scheduler, RNG, and Trainer state with the original five-epoch schedule. A failed/interrupted evaluation must finish before the next epoch starts. Results do not alter training or select a best checkpoint. The notebook displays one tqdm bar per stage (training loss/LR in the postfix) and prints the two-benchmark results after each epoch; detailed diagnostics remain in the console logs.

**B-rejection sampling SFT results (2026-10-10, user reported):** the user identifies **epoch 4** as the best checkpoint. At temperature 0: AMC 2023 **15/40 (37.5%)**, mean **2168.2 tokens**; MATH-500 **307/500 (61.4%)**, mean **1032.2 tokens**. Relative to B0, AMC accuracy is unchanged and MATH-500 increases by **6.6 percentage points**. The best-checkpoint designation is user reported; other epoch scores and Drive artifacts have not been independently compared here. The user confirmed training and evaluation are complete through epoch 5, with epoch 4 selected as the best checkpoint.

```text
# Reported best model checkpoint (configured path)
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/checkpoints/stage1/sft_qwen25_3b_base_self_rft/epoch_4/
# Epoch-4 metrics (user-reported path)
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/eval/amc2023_math500_temp0/sft_qwen25_3b_base_self_rft/epoch_4/full/results/sft_qwen25_3b_base_self_rft_epoch4.json
```

**B-rejection sampling SFT configured locations:**

```text
# Final checkpoint; earlier epochs use epoch_1 through epoch_4
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/checkpoints/stage1/sft_qwen25_3b_base_self_rft/epoch_5/
# Loss history
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/logs/stage1/sft_qwen25_3b_base_self_rft/steps.jsonl
# Run manifest
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/logs/stage1/sft_qwen25_3b_base_self_rft/run_manifest.json
# Pinned dataset snapshot and resolved training config
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/inputs/sft_qwen25_3b_base_self_rft/
# Training console log
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/sft_qwen25_3b_base_self_rft/train_console.log
# Epoch evaluation outputs: epoch_1 through epoch_5; full/results includes metrics and generations
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/eval/amc2023_math500_temp0/sft_qwen25_3b_base_self_rft/epoch_1/
# Combined results for all completed epoch evaluations (also epoch_summary.csv)
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft/eval/amc2023_math500_temp0/sft_qwen25_3b_base_self_rft/epoch_summary.json
```

**B-7B knowledge distillation SFT notebook:** [train_qwen25_3b_base_7b_distill.ipynb](notebooks/train_qwen25_3b_base_7b_distill.ipynb), config [sft_qwen25_3b_7b_distill.yaml](configs/sft_qwen25_3b_7b_distill.yaml). Student: original `Qwen/Qwen2.5-3B` base. The pinned dataset's manifest identifies the teacher as `Qwen/Qwen2.5-Math-7B-Instruct`; its published rows combine **28,415 3B responses** and **9,276 7B responses**. All source rows are used, including multiple solutions per problem; no source-based filtering or deduplication. The full `response` plus original `<|endoftext|>` EOS receives loss; native chat template, EOS/PAD, and vocabulary are preserved.

Same five-epoch SFT settings as the self-RFT experiment: full parameters, BF16, batch 1, accumulation 16, LR 1e-5, cosine schedule, 5% warmup, seed 42, gradient checkpointing, 20,480-token full-sequence cap without truncation. After each epoch, save and exit training, evaluate AMC 2023 (40) and MATH-500 (500) at temperature 0 with a 20,480-token generation cap, print the result table, then resume optimizer/scheduler/RNG state. Display uses tqdm with loss/LR; full logs remain on Drive. All five epochs are evaluated (2,700 responses); benchmark scores do not change training. Self-contained bundle, optional GPU smoke, automatic checkpoint/evaluation resume. GPU training/evaluation has not been run locally.

Local preparation verified **37,691 retained rows**, **0** missing/overlength targets, and no truncation. Full-sequence median/max: **511 / 2,839** tokens. Mean supervised target including EOS: **468.0** tokens. Five epochs present **188,455** rows and take **11,780** optimizer updates. **7 focused tests passed**, covering pinned column selection, native-EOS loss masking, checkpoint save/reload and exact CPU resume, the isolated bundle, epoch evaluation recovery, tqdm output, and evaluator scoring/resume.

**B-7B knowledge distillation SFT configured locations:**

```text
# Final model; earlier checkpoints use epoch_1 through epoch_4
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft/checkpoints/stage1/sft_qwen25_3b_base_7b_distill/epoch_5/
# Loss history
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft/logs/stage1/sft_qwen25_3b_base_7b_distill/steps.jsonl
# Run manifest
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft/logs/stage1/sft_qwen25_3b_base_7b_distill/run_manifest.json
# Training console log
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft/sft_qwen25_3b_base_7b_distill/train_console.log
# Epoch evaluations; epoch_1 through epoch_5 include metrics, generated answers, and logs
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft/eval/amc2023_math500_temp0/sft_qwen25_3b_base_7b_distill/epoch_1/
# Combined epoch results (also epoch_summary.csv)
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft/eval/amc2023_math500_temp0/sft_qwen25_3b_base_7b_distill/epoch_summary.json
```

## Google Drive paths

These are absolute **Colab-mounted Drive paths** (`/content/drive/MyDrive/` = My Drive). B0/I0 evaluation destinations and the I-RL/B-s1(kimi style) SFT training locations are user reported; future training roots remain planned.

| ID | Run root | Path status |
| --- | --- | --- |
| B0 | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/eval-base-instruct-rl-amc-math-temp0/base` | Completed evaluation outputs (user reported) |
| B-RL | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos` | Completed step-300 training and evaluation (user reported) |
| B-s1(kimi style) SFT | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short` | Completed training and evaluation (user reported) |
| B-rejection sampling SFT | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft` | Five-epoch training/evaluation complete; epoch 4 best (user confirmed) |
| B-7B knowledge distillation SFT | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft` | Configured in SFT notebook; training pending |
| B-rejection sampling SFT + RL | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft-rl` | Planned; not configured yet |
| B-7B knowledge distillation SFT + RL | `/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft-rl` | Planned; not configured yet |
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

**B-RL exact configured locations (step 300 complete, user reported):**

```text
# Final full-model checkpoint; intermediate checkpoints every 20 updates
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/checkpoints/dapo_qwen25_3b_pretrained_base/checkpoint-300/
# Loss, reward, entropy, length, learning rate, and throughput history
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/logs/dapo_qwen25_3b_pretrained_base/steps.jsonl
# Run manifest with resolved dataset revision and code identity
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/logs/dapo_qwen25_3b_pretrained_base/run_manifest.json
# Console log
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/logs/dapo_qwen25_3b_pretrained_base/training_console.log
# Generated rollout responses and scoring records
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/logs/dapo_qwen25_3b_pretrained_base/rollouts/
# Separate GPU smoke attempts
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos/smoke_attempts/
```

**B-s1(kimi style) SFT completed evaluation output root (user reported):**

```text
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short/eval/amc2023_math500_temp0/sft_qwen25_3b_base_s1_kimi/epoch_5/
```

Under that root, `full/results/sft_qwen25_3b_base_s1_kimi_epoch5.json` stores metrics;
`full/results/sft_qwen25_3b_base_s1_kimi_epoch5_generations.jsonl` stores full responses and scores.
`full/summary_table.json` and `full/summary_table.csv` contain the two-benchmark table.
The same results directory holds the `_problems.jsonl` resume journal and model/code/data manifests.
Smoke files use `smoke/` instead of `full/`; console logs are under `logs/`.

**B-s1(kimi style) SFT exact training locations (user reported):**

```text
# Final full-model checkpoint (five completed epochs)
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short/checkpoints/stage1/sft_qwen25_3b_base_s1_kimi/epoch_5/
# Loss, learning rate, gradient norm, throughput, step, and epoch history
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short/logs/stage1/sft_qwen25_3b_base_s1_kimi/steps.jsonl
# Training manifest
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short/logs/stage1/sft_qwen25_3b_base_s1_kimi/run_manifest.json
# Preparation report and loss preview
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short/logs/stage1/sft_qwen25_3b_base_s1_kimi/preparation/
# Pinned two-column dataset snapshot and training.yaml
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short/inputs/sft_qwen25_3b_base_s1_kimi/
# Console log
/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short/sft_qwen25_3b_base_s1_kimi/train_console.log
```

Epochs 1–4 use the same checkpoint directory with `epoch_1` through `epoch_4`. Optional smoke
uses run name `sft_qwen25_3b_base_s1_kimi_smoke` in separate checkpoint/log directories.

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
