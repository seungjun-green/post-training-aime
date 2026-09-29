# Spec 1 — Datasets, Translation, and Evaluation Protocol

This spec covers everything before training:

1. Download all datasets.
2. Decontaminate the training sets against the eval sets (8-gram matching on English).
3. Translate all datasets to Korean in a Google Colab notebook, gated by a smoke test.
4. Upload the translated datasets to Hugging Face.
5. Build the evaluation harness (defined **once** here and reused by every later spec).
6. Run the baseline evaluation on the untouched base model.

Follow the shared rules in Spec 0.

---

## 1. Datasets

### Training

| Name | Source | Split / config | Used in |
|---|---|---|---|
| s1K-1.1 | `simplescaling/s1K-1.1` | `train` | Stage 1 (SFT) |
| DAPO-Math-17K | `open-r1/DAPO-Math-17k-Processed` | `en` config | Stage 2 (RL) |

### Evaluation

| Name | Source | Size |
|---|---|---|
| AIME 2024 | `HuggingFaceH4/aime_2024` | 30 |
| AIME 2025 | `math-ai/aime25` | 30 |
| AIME 2026 | `MathArena/aime_2026` | 30 |
| AMC 2023 | `math-ai/amc23` | 40 |
| MATH-500 | `HuggingFaceH4/MATH-500` | 500 |

### Schema handling

Column names differ across these datasets. **Inspect each schema first** and map every dataset to a common internal view with at least:

- `id` — a stable unique id (use the existing id column if there is one; otherwise create one from the row index and record it).
- `problem` — the problem statement in English.
- `answer` — the ground-truth final answer.

Keep **all original columns** unchanged. The internal view is for processing only.

For DAPO-Math-17K, if the prompt wraps the problem in instruction text (e.g. "Solve the following math problem step by step…"), extract only the core problem statement into `problem`. Our own prompt format is defined in Section 5 and must be the only instruction wrapper used in training and eval.

---

## 2. Decontamination (English, before translation)

Run decontamination **before** translation so removed rows are never translated.

### Procedure

1. **Normalize** every problem text (training and eval):
   - lowercase
   - remove LaTeX delimiters and spacing/sizing commands: `$`, `\(`, `\)`, `\[`, `\]`, `\left`, `\right`, `\,`, `\;`, `\!`, `\quad`
   - separate punctuation from words, then collapse all whitespace to single spaces
   - split into word tokens on whitespace
2. **Build the eval index:** collect every 8-gram from all eval problems (all five eval sets combined).
3. **Check each training example** (s1K-1.1 `question`; DAPO-Math-17K `problem`):
   - If it shares **any** 8-gram with any eval problem → remove it.
   - For eval problems with **fewer than 8 tokens**: remove any training example whose normalized text contains that eval problem's normalized text as an exact substring.
4. Remove from **training sets only**. Eval sets are never modified.

### Outputs

- Decontaminated English training sets (saved to Drive, see Section 3).
- `decontamination_report.json` with, per training set: original row count, removed count, remaining count, and the list of removed `id`s together with the eval set and problem they matched.

---

## 3. Translation notebook (Google Colab)

Write a single notebook: `notebooks/translate_datasets.ipynb`. It runs on Colab (CPU is enough).

### 3.1 Setup

- **Drive:** mount Google Drive. Project root: `/content/drive/MyDrive/LG-Korea-AIME`. All outputs go under this root.
- **Secrets:** read from Colab Secrets with `google.colab.userdata.get(...)`:
  - `HF_TOKEN` (must have **write** permission for the upload step)
  - `ANTHROPIC_API_KEY`
- **Config cell** at the top, with every setting in one place:

```python
PROJECT_ROOT = "/content/drive/MyDrive/LG-Korea-AIME"
TRANSLATION_MODEL = "TODO"   # set before running; options:
                             # "claude-fable-5-1", "claude-opus-5-5",
                             # "claude-sonnet-5", "claude-haiku-4-5-20251001"
MAX_CONCURRENCY = 16         # max in-flight API requests
MAX_OUTPUT_TOKENS = 16000    # per request
CHUNK_CHARS = 12000          # max characters per chunk for long texts
MAX_RETRIES = 6
SMOKE_TEST = True            # True = smoke test only; False = full run
SMOKE_TEST_N = 5             # samples per dataset in smoke test
HF_USERNAME = "Seungjun"
HF_PRIVATE = True            # visibility of uploaded datasets
```

### 3.2 Notebook sections (in order)

1. Setup (install packages, mount Drive, load secrets, config)
2. Download all datasets
3. Decontamination (Section 2) — skip if its outputs already exist in Drive
4. Translation (smoke test or full run, depending on `SMOKE_TEST`)
5. Automated checks (Section 3.6)
6. Assemble and upload to Hugging Face (full run only)

### 3.3 What to translate

Each translated field is added as a new column named `ko_{original_col_name}`. Original columns are never modified.

| Dataset | Columns to translate | Not translated |
|---|---|---|
| s1K-1.1 | `question`, `deepseek_thinking_trajectory`, `deepseek_attempt` | `solution` (kept for answer checking), all Gemini columns, metadata |
| DAPO-Math-17K | `problem` (the extracted core problem) | the answer |
| All 5 eval sets | the problem column | the answer |

### 3.4 Translation requests

- Use the **async** Anthropic Python client (`AsyncAnthropic`) with an `asyncio.Semaphore(MAX_CONCURRENCY)`.
- Retry on rate-limit and server errors with exponential backoff and jitter, up to `MAX_RETRIES`.
- **Long texts** (reasoning traces): split at paragraph boundaries (`\n\n`) into chunks of at most `CHUNK_CHARS` characters, translate each chunk, and join with `\n\n`. Never split inside a LaTeX block.
- If a response has `stop_reason == "max_tokens"`, treat it as a failure: re-split that chunk smaller and retry.

System prompt for every request:

```
You are a professional translator of mathematics from English to Korean.
Translate the user's text into natural, fluent Korean.

Rules:
- Preserve all LaTeX, math expressions, numbers, variable names, and code exactly as written.
- Do not solve, simplify, correct, summarize, add, or remove anything.
- Keep the original structure: line breaks, paragraphs, lists, and \boxed{...} expressions.
- For reasoning text, translate faithfully, including hesitations and self-corrections
  (e.g. "Wait" → "잠깐").
- Output only the Korean translation, with no preface or notes.
```

### 3.5 Persistence and resuming

Colab runtimes can disconnect, so the notebook must be resumable:

- Write each finished row immediately to a JSONL file in Drive (one file per dataset), keyed by `id`.
- On restart, load existing JSONL files and skip `id`s that are already done.
- At the end of each run, zip the outputs and also trigger a browser download (`google.colab.files.download`).

Drive layout:

```
LG-Korea-AIME/
  decontamination/
    decontamination_report.json
    s1k_1.1_decontaminated.jsonl
    dapo_math_17k_decontaminated.jsonl
  smoke_test/
    <dataset_name>.jsonl
  translations/
    <dataset_name>.jsonl
  checks/
    <dataset_name>_flagged.jsonl
    checks_report.json
```

### 3.6 Automated checks (run on every translated row)

Flag a row if any of these fail:

1. **Hangul present:** each translated field contains Korean characters.
2. **Length ratio:** translated length / original length (in characters) is within `[0.3, 2.0]`.
3. **Math preserved:** the count of `\boxed{` is identical between original and translation.
4. **Final answer preserved (s1K-1.1 only):** the content of the last `\boxed{...}` in `ko_deepseek_attempt` exactly matches that in `deepseek_attempt`.
5. **Not truncated:** no chunk ended with `stop_reason == "max_tokens"`.

Flagged rows are retried once automatically. Rows that still fail are written to `checks/<dataset_name>_flagged.jsonl` and **excluded** from the uploaded dataset. `checks_report.json` records the flagged count per dataset and the reason.

### 3.7 Smoke test gate

When `SMOKE_TEST = True`:

- Translate `SMOKE_TEST_N` rows from **each** of the 7 datasets (2 training + 5 eval), chosen with a fixed seed.
- For s1K-1.1, include at least one row with a very long `deepseek_thinking_trajectory`, so chunking is exercised.
- Run the automated checks on these rows.
- Save to `smoke_test/`, zip, download, and **stop**. Do not start the full run.

The full run starts only when `SMOKE_TEST` is manually set to `False` after the smoke test output has been reviewed.

### 3.8 Upload to Hugging Face (full run only)

For each of the 7 datasets:

- Build a `datasets.Dataset` with all original columns plus the `ko_` columns (flagged rows excluded).
- Push to `Seungjun/<name>-ko` with `private=HF_PRIVATE`, using these names:

| Source | Uploaded as |
|---|---|
| s1K-1.1 | `Seungjun/s1K-1.1-ko` |
| DAPO-Math-17K | `Seungjun/DAPO-Math-17k-ko` |
| AIME 2024 | `Seungjun/aime_2024-ko` |
| AIME 2025 | `Seungjun/aime_2025-ko` |
| AIME 2026 | `Seungjun/aime_2026-ko` |
| AMC 2023 | `Seungjun/amc23-ko` |
| MATH-500 | `Seungjun/MATH-500-ko` |

- Write a dataset card for each with: the source dataset link, the **same license as the source** (AIME 2026 is CC BY-NC-SA 4.0 and must stay that way), the translation model used, a note that `ko_` columns are machine-translated, and (for training sets) the decontamination summary.

---

## 4. Where evaluation runs

Evaluation runs on the training machine (single RTX PRO 6000, 96GB), **not** in Colab. It loads the Korean eval sets from the uploaded Hugging Face datasets.

---

## 5. Evaluation protocol (defined once, reused by every stage)

Implement as a reusable module, e.g. `eval/run_eval.py`, callable as:

```
python eval/run_eval.py --model <path_or_hf_id> --stage <stage> --run_name <run_name>
```

### 5.1 Prompt format

Define the prompt in one shared module (e.g. `common/prompts.py`) that training code will also import. Use the model's own chat template with the Korean problem (`ko_` column) as the user message, followed by this instruction:

```
문제를 단계별로 풀고, 최종 답을 \boxed{} 안에 쓰세요.
```

No system prompt. The same format is used in every stage and for the baseline.

### 5.2 Generation settings

| Setting | Value |
|---|---|
| Engine | vLLM (fall back to HF `generate` only if vLLM does not support the checkpoint) |
| Temperature | 1.0 |
| Top-p | 0.7 |
| Max new tokens | 20,480 |
| Samples per problem (n) | 32 for AIME 2024/2025/2026 and AMC 2023; 4 for MATH-500 |
| Seed | fixed and recorded |

Temperature, top-p, max length, and n = 32 for AIME follow DAPO's evaluation setup. All values live in `configs/eval.yaml`.

### 5.3 Answer extraction and scoring

- Extract the content of the **last** `\boxed{...}` in the response (handle nested braces).
- No `\boxed{}` → incorrect.
- Compare against ground truth with the `math-verify` library.

### 5.4 Metrics (per eval set)

- **avg@n:** mean accuracy over all n samples (the headline number).
- **pass@k** for k ∈ {1, 4, 8, 16, 32} (only k ≤ n), using the unbiased estimator from Chen et al. (2021).
- **Response length:** mean tokens over all responses, and separately over correct and incorrect responses.
- **Korean response ratio:** fraction of Hangul characters among all letter characters in the response (excluding LaTeX), averaged over responses.

### 5.5 Outputs

- `results/<stage>/<run_name>.json` — all metrics per eval set, plus the model path, config values, and seed.
- `results/<stage>/<run_name>_generations.jsonl` — every raw generation with its extracted answer and correctness.

---

## 6. Baseline evaluation

Run the evaluation from Section 5 on `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` with no changes:

```
python eval/run_eval.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --stage stage0 --run_name baseline
```

---

## Deliverables

1. `notebooks/translate_datasets.ipynb` (decontamination, smoke test, full translation, checks, upload)
2. `decontamination_report.json` and `checks_report.json`
3. Seven translated datasets on Hugging Face under `Seungjun/`
4. `eval/run_eval.py`, `common/prompts.py`, `configs/eval.yaml`
5. `results/stage0/baseline.json` and its generations file
