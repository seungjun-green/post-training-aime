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

The full run evaluated **630 problems with 6,160 sampled responses** on an NVIDIA RTX PRO 6000 Blackwell 96GB using vLLM. These are the base model's results before project fine-tuning.

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

The active notebooks are:

1. [Prepare English datasets](notebooks/prepare_english_datasets.ipynb): CPU decontamination and Hugging Face upload.
2. [Evaluate the English baseline](notebooks/evaluate_baseline_english.ipynb): clone/update the project, install the GPU runtime, run a five-problem smoke test, then explicitly enable full evaluation. Includes tqdm progress and saved results on Drive.
3. [Stage 1 SFT](notebooks/train_stage1_sft.ipynb): prepare and inspect assistant-only loss masks, then explicitly enable five-epoch training and save every epoch checkpoint. No evaluation runs here.
4. [Stage 1 evaluation](notebooks/evaluate_stage1_sft.ipynb): evaluate epoch 5 on all five benchmarks by default (6,160 responses). Optionally enable epochs 1–4 afterward, giving the order 5, 1, 2, 3, 4 and 30,800 responses in total. Uses the original evaluation environment.

Evaluation logic lives in the project rather than the notebook. The preparation notebook still bundles its preparation code. The translation notebooks and old Korean baseline notebook have been removed; `README.md` is retained as historical documentation and contains references to those retired notebooks.

Stage 1 code is implemented, with settings and execution instructions in the [run guide](docs/stage1_sft.md). CPU preparation using the actual pinned EXAONE tokenizer drops 21 examples exceeding 20,480 tokens and keeps 975, resulting in 61 optimizer steps per epoch and 305 total. The original 996 rows contain 370 `deepseek_grade`-incorrect examples; this grade does not filter training. The [data report](docs/stage1_data_report.json) records the exact dropped IDs and token-length statistics.

Next is GPU validation and fine-tuning, followed by evaluation with the same datasets, prompts, sampling settings, scoring, and runtime. The full evaluator is unchanged and now supplies all five benchmarks for each epoch. Preserve the stage-0 archive and its parent protocol/runtime manifests for comparisons; no fine-tuned results are reported yet.
