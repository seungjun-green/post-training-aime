# Math reasoning on EXAONE

The current direction keeps `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` and uses the original English datasets. Translation is no longer part of the active data pipeline; earlier translation notebooks remain available as historical experiments.

## English preparation notebook (current)

Upload [`notebooks/prepare_english_datasets.ipynb`](notebooks/prepare_english_datasets.ipynb) to **Colab CPU**, add a write-capable `HF_TOKEN`, and run the cells in order. It applies the existing normalization-v2, 8-gram, 70% per-eval-problem coverage rule to the two training datasets. It does not deduplicate within/between training datasets. All five eval sets remain unchanged, as do every retained original column and text value.

Running all cells uploads seven private datasets under `Seungjun/dp_removed_{source_repo_basename}` (change `HF_USERNAME`/`HF_PRIVATE` in settings as needed). For example, `simplescaling/s1K-1.1` becomes `Seungjun/dp_removed_s1K-1.1`, and `math-ai/aime25` becomes `Seungjun/dp_removed_aime25`. Data, original cards, provenance and the overlap report are staged and checked before upload. Exact uploaded commits are saved in `english_dataset_suite.json`; reruns republish the same staged content. No translation model or GPU is needed.

The old Korean baseline notebook remains unchanged. Use the English evaluation entry point below for this new dataset suite.

## English baseline evaluation (current)

The model is **`LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`**. Use the thin [`notebooks/evaluate_baseline_english.ipynb`](notebooks/evaluate_baseline_english.ipynb) on the RTX PRO 6000 Blackwell 96GB Colab runtime. The launcher is preconfigured for `https://github.com/seungjun-green/post-training-aime.git` and clones `main`. Enable the `HF_TOKEN` Colab secret for the private datasets. The notebook records the checked-out commit automatically, installs a separate locked Python 3.12 GPU environment, and invokes the project CLI. Rerunning setup keeps the existing checkout; it does not pull new code into an ongoing run. Settings are plain Python assignments, with no Colab forms or commit field to fill in. No project code is embedded in the notebook. A private Git repository requires Git authentication in Colab before cloning; never put credentials in the URL or notebook.

All five eval datasets are pinned to the actual upload commits from `english_preparation_reports.zip` in `configs/english_eval_suite.json`. Loading checks all 630 original English rows by content digest, row count and native IDs before generation. AIME 2026 uses `problem_idx`, MATH-500 uses `unique_id`, and AMC23 uses the English `question` column. Training datasets are excluded from evaluation.

```bash
# Install uv in the host Python, then create the independent GPU environment.
python -m pip install uv==0.11.22
python scripts/setup_eval_runtime.py --venv /content/lg-eval-env

# HF_TOKEN must be available in the environment; do not store it in source files.
# Optional dataset-only validation (no GPU generation).
/content/lg-eval-env/bin/python -m eval.run_english_eval \
  --output_root /content/drive/MyDrive/LG-AIME-English-Eval-compatible --validate-only

# Five problems, one response each, written to output_root/smoke/.
/content/lg-eval-env/bin/python -m eval.run_english_eval \
  --output_root /content/drive/MyDrive/LG-AIME-English-Eval-compatible --smoke

# Full baseline: 630 problems / 6,160 responses, written to output_root/full/.
/content/lg-eval-env/bin/python -m eval.run_english_eval \
  --output_root /content/drive/MyDrive/LG-AIME-English-Eval-compatible
```

The base-model default is pinned in `eval/run_english_eval.py` to revision `e949c91dec92095908d34e6b560af77dd0c993f8`, whose loader supports the locked Transformers 4.57.6 environment. LG's newer `ccce25bd39c141fe053e0bc75818a8f5fe962802` loader imports Transformers-5-only `RopeParameters`. The official Hub file hashes match for both weight shards, weight index, model/generation config, tokenizer files, vocabulary and merges; only loader Python files differ. Explicit `--revision` and saved run revisions are still respected. The launcher uses a fresh `LG-AIME-English-Eval-compatible` output root to preserve metadata from failed attempts with the incompatible loader. Do not reuse incompatible run manifests as new baseline results.

`configs/eval_english.yaml` retains temperature 1.0, top-p 0.7, maximum 20,480 completion tokens, n=32 on AIME/AMC and n=4 on MATH-500. `common/english_prompts.py` supplies one English user message with a step-by-step/boxed-answer instruction through the checkpoint's native chat template. No system message is added. The last balanced boxed answer is scored using the existing math-verify scorer. Reports include avg@n, unbiased pass@k for k ≤ n, response lengths and the retained Korean-letter-ratio diagnostic (which never affects correctness).

Results are saved after every response. Repeating the same command resumes completed problems, preserving the recorded model revision even if upstream main moves. Partially completed problems are regenerated with their original seed and must reproduce saved responses before resuming. Frozen code/config/suite and tokenizer/runtime manifests guard later comparisons. Keep the same Git commit when resuming a run; a changed run identity requires a new run name. Smoke results cannot become full benchmark results. The smoke cell downloads `smoke_test_result.zip`, containing raw prompts/responses, scores, token counts, finish reasons, model/dataset revisions, Git commit and runtime settings. Share that ZIP for review before enabling `RUN_FULL_EVAL`; low accuracy on five questions alone does not mean the pipeline is broken.

The run manifest records the Git commit, exact model and dataset revisions, package versions, hardware, English protocol and selected IDs. The Git checkout must be committed and clean. The old Korean evaluator and its notebook are retained separately. No training code is added by this evaluation update.

## Earlier translation notebooks

### OpenAI GPT-5.6 Sol full-text translation

Upload [`notebooks/translate_datasets_openai.ipynb`](notebooks/translate_datasets_openai.ipynb) to a fresh **Colab CPU** runtime. Add `OPENAI_API_KEY` and `HF_TOKEN` to Colab Secrets. Run setup/decontamination, then the separate **35-row smoke cell**. Download the ZIP and open `smoke_test/human_review.html`. After review, enable `RUN_FULL_TRANSLATION` in the full-run cell. Upload remains a separate, disabled-by-default cell.

The Responses API receives the exact original text of each paragraph chunk with `model='gpt-5.6-sol'`, `reasoning.effort='none'` initially, and `store=False`. There are no placeholders or automatic model fallbacks. Long fields retain the existing 16,000-character paragraph chunk limit; “full text” means unmasked text, not one request for an entire long trace. This translation-only version does not make a separate semantic-review model call.

Post-translation hard checks compare numeric literals as multisets, LaTeX/code spans verbatim in source order, and the final boxed answer. A failed chunk is retried once with failure details; successful chunks are durably cached across interruptions and smoke/full runs. Persistent hard failures exclude rows. Refusals are recorded and excluded without repeated refusal attempts. Length, Hangul, paragraph and undelimited-symbol warnings are advisory and never automatically exclude rows. The exact-order rule can reject natural reordering, and written number words must remain words. These checks cannot certify mathematical meaning.

`failed_chunks.jsonl` saves rejected responses and source chunks for diagnosis. `api_usage.jsonl` and `usage_summary.json` track returned responses, token usage, cache hits and reasoning tokens. Full-run `human_review.html` puts all 130 AIME/AMC source problems first, followed by other flagged rows. Review soft warnings before publication: those rows remain upload-eligible. Results use a separate `LG-Korea-AIME-OpenAI/models/gpt-5.6-sol/<settings-id>/` folder; code or generation-setting changes require new smoke evidence. Existing notebooks remain unchanged.

Local verification uses mocked HTTP responses; no live OpenAI smoke or full translation has been run here. API access and translation quality must be checked in Colab. References: [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

### DeepSeek API comparison (new notebook)

Upload [`notebooks/translate_datasets_deepseek.ipynb`](notebooks/translate_datasets_deepseek.ipynb) to **Colab CPU**. Add `DEEPSEEK_API_KEY` and `HF_TOKEN` to Colab Secrets. No GPU/model download or other model-provider key is needed.

- Run setup/source download/decontamination, then the **smoke cell**. It calls both `deepseek-flash` and `deepseek-v4-pro` on the same 35 rows (70 translated rows total), with thinking disabled and identical translation settings. Download the comparison ZIP and open `smoke_comparison.html` for English/Flash/Pro columns, including flagged rows. The ZIP also includes each model's checks, row journals and token usage.
- After review, use the separate **full-run cell**: choose `FULL_MODEL` from the dropdown and set `RUN_FULL_TRANSLATION=True`. The selected model must have a completed matching smoke test. This cell reuses that model's smoke rows, translates/checks the remaining cleaned data and downloads the full archive. There is no global smoke/full setting to toggle. After a runtime restart, rerun setup/decontamination and this cell to resume.
- Enable `UPLOAD_DATASETS` in the final cell to publish the checked full run. The existing unspecified-source-license policy still applies. Publication updates the same seven configured HF repositories. Copy the printed model-specific run folder into the baseline notebook's `PROJECT_ROOT` to use the resulting `eval_suite.json`.

A default Run all executes the smoke comparison only; full translation and publication default to off. Both smoke and full translation incur DeepSeek API charges. Outputs use `/content/drive/MyDrive/LG-Korea-AIME-DeepSeek/models/<model>/<settings-id>/`; incompatible generation settings receive a fresh folder. The original Anthropic, Qwen and baseline notebooks are unchanged. Coverage decontamination and row quality checks use the existing shared implementation.

Hosted model names are mutable aliases. The API adapter records returned model names, fingerprints, response IDs, timestamps and raw usage, without claiming to pin hosted weights. `api_usage.jsonl` includes returned retries and truncated outputs; lost responses can still be billable, so the provider dashboard remains the billing reference. Authentication, insufficient balance and persistent service errors stop the run with resumable row journals. Local tests use mocked HTTP responses; no live DeepSeek requests have been made here.

DeepSeek requests now shield recognized LaTeX/code blocks and numeric literals with unique placeholders, then restore the original strings exactly. Missing, duplicated, unknown or damaged placeholders, changed numeric literals, altered protected blocks and changed math-symbol counts fail validation. The existing one-time quality retry and exclusion apply. Reordering literals for Korean grammar is allowed; these checks cannot prove meaning or catch every change to undelimited mathematics. The prompt also distinguishes diameter/radius and ordered pairs/triples. Review semantics and omissions in the new smoke outputs.

For this preservation update, open the rebuilt notebook in a fresh Colab runtime and rerun setup and both-model smoke comparison. The implementation hash creates fresh model run folders automatically; old translations are left untouched and do not qualify as smoke evidence. No full API run has been performed to validate the revised prompt's translation quality.

API references: [thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/), [chat completions](https://api-docs.deepseek.com/api/create-chat-completion/), [models/pricing](https://api-docs.deepseek.com/quick_start/pricing/).

### Local Qwen translation (new notebook)

Upload [`notebooks/translate_datasets_qwen.ipynb`](notebooks/translate_datasets_qwen.ipynb) to Colab and connect the **RTX PRO 6000 Blackwell 96GB** runtime. Only `HF_TOKEN` is needed; translation runs locally with `Qwen/Qwen3-32B`, BF16, thinking disabled. The notebook creates an isolated Python 3.12 GPU environment even when the notebook kernel uses Python 3.13. It installs the existing pinned GPU lock and starts a loopback-only vLLM server. Allow roughly 70GB for model weights plus additional package/download space.

The default is **35 smoke rows**, including the longest retained s1K trace. Download the results ZIP and open `smoke_review.html` to compare every English/Korean field side by side. Inspect `checks_report.json` too. After reviewing, set **`SMOKE_TEST=False` and `SMOKE_REVIEWED=True`** and rerun. The completed matching smoke manifest is also required; flipping the review flag alone cannot start a full run. Each completed row is saved immediately, including the pinned model revision. Quality retries use a different reproducible seed. No GPU inference or real Qwen translation has been run from this Mac; local tests exercise the adapter using mock HTTP responses.

Outputs are isolated under `/content/drive/MyDrive/LG-Korea-AIME-Qwen/runs/<settings-id>/`. The model commit is pinned on first use. Different generation settings/code create a new run folder; smoke/full modes share compatible Qwen translations. The original Anthropic notebook and its outputs are unchanged. The Qwen notebook uses the same v2 coverage decontamination, quality checks, source-license upload gate and publisher. Full mode updates the same seven configured Hugging Face repositories. After upload, copy the printed run folder into **`PROJECT_ROOT` in the existing baseline notebook** so it reads this run's `eval_suite.json`.

The Qwen notebook raises the paragraph chunk cap to 16,000 characters to accommodate the known 13,453-character s1K paragraph, and checks input tokens plus the output budget against the 32K context before each request. Truncated output is discarded and split only at safe paragraph boundaries. Server failures stop the run instead of marking all pending rows as bad. The final cell stops the notebook's server to release GPU memory; end the Colab runtime to stop rental charges.

Model/runtime references: [Qwen3-32B model card](https://huggingface.co/Qwen/Qwen3-32B), [Qwen vLLM non-thinking configuration](https://qwen.readthedocs.io/en/latest/deployment/vllm.html), [vLLM GPU installation](https://docs.vllm.ai/en/v0.14.1/getting_started/installation/gpu/).

### Original Anthropic translation

1. Upload `notebooks/translate_datasets.ipynb` to [Google Colab](https://colab.research.google.com/). The notebook is self-contained; no GitHub repository or local package upload is needed.
2. Use a CPU runtime. Add `HF_TOKEN` with write access and `ANTHROPIC_API_KEY` to Colab Secrets, then grant the notebook access. The configured translation model is **`claude-opus-5-5`**, as requested; API access to that exact model is checked by the first request.
3. Run with **`SMOKE_TEST = True`**. Outputs are saved under `/content/drive/MyDrive/LG-Korea-AIME`; the notebook downloads a ZIP containing all 35 smoke examples and their checks. The longest retained s1K reasoning trace is always included.
4. Review translations, math, paragraph/chunk joins, boxed answers and flagged rows. Only after review, manually set `SMOKE_TEST = False` and rerun. Completed rows are skipped on restart. Full mode uploads the seven checked datasets and writes `eval_suite.json` containing their immutable revisions and exact evaluation IDs.
5. Open `notebooks/evaluate_baseline.ipynb` and connect it to the **RTX PRO 6000 Blackwell 96GB runtime through Colab**, per your clarification. It installs a pinned Linux/Python 3.12 GPU environment and runs the shared evaluator. A normal Colab CPU/T4/A100 session will be rejected.

The full translation uses a paid Anthropic API. The smoke notebook never starts a full translation or uploads datasets automatically. Run only one notebook instance per project root.

## Execution status

All seven real English sources were downloaded and their schemas inspected. The English decontamination was actually run:

| Training set | Original | Removed | Retained |
|---|---:|---:|---:|
| s1K-1.1 | 1,000 | 4 | 996 |
| DAPO English | 14,116 | 48 | 14,068 |

These are the normalization-v2 results at 70% coverage. All 630 evaluation rows are unchanged. Source revisions, counts and deterministic smoke selections are recorded in `docs/data_preparation_summary.json`. The current report is at `data/decontamination_v2/decontamination_report.json`. It records each removed ID, every qualifying evaluation match and its distinct-8-gram coverage, per-eval removal counts, and 18 retained near misses. s1K has zero removals against AIME 2025 and AIME 2026.

The original one-shared-8-gram run removed 315 s1K and 5,610 DAPO rows. Its report and datasets remain untouched in `data/decontamination/`. Large downloaded/generated files are ignored by Git.

The notebook config exposes `DECONTAM_NGRAM_SIZE = 8`, `DECONTAM_COVERAGE_THRESHOLD = 0.7` and `DECONTAM_NEAR_MISS_MIN = 0.5`. Normalization drops punctuation-only tokens. Coverage is the fraction of one evaluation problem's distinct 8-grams present in one training problem; matches are never pooled across evaluation problems. The exact-substring rule remains for evaluation problems shorter than 8 tokens. The notebook prints per-eval counts, future-contest warnings and the ten removed rows with the lowest best-pair coverage, with the texts side by side.

New outputs use `decontamination_v2/`; additional settings/input changes use a signature-named subfolder there. The signature includes normalization version, n-gram size, threshold and near-miss minimum. Translation cache migration verifies the prior source cohort and translation settings, archives previous selections under each mode's `decontamination_history/`, and reuses compatible completed rows by ID. A changed smoke selection must be checked and reviewed again. Translation requests, quality checks, upload logic and the evaluation notebook are unchanged. The current report is also copied into the active smoke/translation output folder so the existing ZIP download includes it.

Live translation, its `checks_report.json`, Hugging Face publication, and GPU baseline results have **not** been run from this local Mac. They require the Colab secrets/runtime and the user's smoke review. No synthetic results are presented as real baseline or translation results.

Two source-policy questions remain explicit:

- Four source cards do not declare a license: processed DAPO, AIME 2024, AMC 2023 and MATH-500. `UNSPECIFIED_LICENSE_POLICY = "block"` stops publication until resolved. Setting it to `"retain-unspecified"` copies that status and source links without inventing a license. AIME 2026 always retains **CC BY-NC-SA 4.0**.
- Retained s1K row `s1k_1.1:train:869` contains a **13,453-character paragraph**. Strict paragraph-only splitting at a 12,000-character cap is impossible for that field. The current code flags this explicitly rather than splitting inside a paragraph or LaTeX. The handling choice is awaiting clarification.

## Evaluation protocol

`common/prompts.py` is the only prompt definition. It uses the checkpoint's own chat template, one Korean user message and the specified boxed-answer instruction, with no system message. The native template itself emits an empty system marker; no system text or message is supplied. `configs/eval.yaml` contains all sampling and runtime settings.

```bash
python eval/run_eval.py \
  --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
  --stage stage0 --run_name baseline \
  --output_root /content/drive/MyDrive/LG-Korea-AIME
```

With the default output root, the exact command from Spec 1 also works when `eval_suite.json` is in the working directory. The Colab notebook sets the Drive output root explicitly.

The evaluator uses vLLM, with HF generation only for unsupported checkpoint architectures. Missing installations, OOMs and other runtime errors do not trigger silent fallback. It refuses context truncation. Temperature is 1.0, top-p 0.7, completion limit 20,480 tokens, n=32 for AIME/AMC and n=4 for MATH-500. A full 630-problem evaluation generates 6,160 responses.

It extracts the last balanced `\boxed{...}`, scores with `math-verify`, and reports avg@n, unbiased pass@k, token lengths and the Korean letter ratio outside LaTeX. Missing/malformed final boxes are incorrect; empty correct/incorrect groups have `null` mean length.

Results are written to `results/<stage>/<run_name>.json` and `<run_name>_generations.jsonl`. Companion manifests record model commits, packages, seeds, GPU and dataset revisions. The baseline freezes configuration, evaluation code, tokenizer, chat template and package versions in `results/eval_protocol.json` and `results/eval_runtime.json`; later stages must reuse them unchanged. Evaluation sizes and translation exclusions are explicitly retained, so a reduced benchmark is never silently reported as a complete one.

Evaluation resumes complete problems from durable raw-generation records. A partially written problem is regenerated with its stable problem seed; existing samples must match exactly before missing samples are appended. A mismatch stops the run and requires a new run name. Different engines may produce different samples even with the same seed; the actual engine is recorded.

## Local EXAONE full-text translation

Open `notebooks/translate_datasets_exaone.ipynb` on the **RTX PRO 6000 Blackwell 96GB** Colab runtime. It runs `LGAI-EXAONE/EXAONE-4.5-33B` in BF16 with thinking disabled and pins the model commit in Drive. Only `HF_TOKEN` is required; translation calls go to the local GPU server. GPU rental still incurs costs.

Run setup and server startup, then the separate **35-row smoke cell**. Download its ZIP and inspect `smoke_test/human_review.html`. After review, enable `RUN_FULL_TRANSLATION` in the full-run cell. Publication remains separate and off by default. If you ran the cleanup cell, restart the server before running full translation.

The model receives original text without placeholders. Long fields use safe paragraph chunks. Numeric literals, ordered LaTeX/code spans and boxed answers receive hard checks, with one retry per failed chunk. Soft wording/length warnings require review. Completed chunks persist across interruptions and are reused by full translation. No second-model semantic review is performed.

The separate Python 3.12 GPU environment uses `requirements-exaone.lock` (vLLM 0.20.2 and Transformers 5.8.0); it does not replace the frozen baseline environment. A CUDA/driver check runs before weight loading. Allow at least 100GB free runtime disk. Outputs live in `LG-Korea-AIME-EXAONE/runs/<settings-id>/`; after publication, use that printed folder as the baseline notebook's `PROJECT_ROOT`.

The adapter, failure/retry handling, resume behavior, and notebook bundle are locally tested with mocked GPU responses. Actual model loading, GPU memory use, throughput and translation quality require the Colab smoke run.

## Local development

Use Python 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[translation,dev]'
.venv/bin/python scripts/prepare_data.py --root data
.venv/bin/python scripts/validate_data.py --root data
.venv/bin/python scripts/build_notebooks.py --notebook translate_datasets.ipynb
.venv/bin/python scripts/build_notebooks.py --notebook translate_datasets_qwen.ipynb
.venv/bin/python scripts/build_notebooks.py --notebook translate_datasets_deepseek.ipynb
.venv/bin/python scripts/build_notebooks.py --notebook translate_datasets_openai.ipynb
.venv/bin/python scripts/build_notebooks.py --notebook translate_datasets_exaone.ipynb
.venv/bin/python scripts/build_notebooks.py --notebook evaluate_baseline_english.ipynb
.venv/bin/python -m pytest -q
.venv/bin/ruff check common pipeline eval scripts tests
```

The generated notebooks embed the same tested Python modules. Rebuild them after changing code or configs. Notebook tests validate syntax, notebook structure and byte-for-byte bundle freshness. API concurrency/retry, truncation recovery, crash resumption, quality exclusion and upload gates are tested with isolated fake clients; scorer tests use the actual math-verify package.

For the GPU machine, use a separate Python 3.12 environment and `pip install -r requirements-eval.lock`. The lock resolves vLLM 0.14.1, PyTorch 2.9.1 and Transformers 4.57.6. The translation SDK is intentionally separate because this vLLM version pins another Anthropic SDK. Dependency resolution and tokenizer behavior are locally checked; CUDA model execution still requires the target runtime.

## Data provenance and source documentation

`configs/datasets.yaml` pins each source commit and explicitly maps schema columns/splits. Missing IDs use recorded original row indices; DAPO uses `extra_info.index`, AIME 2026 uses `problem_idx`, and MATH-500 uses `unique_id`. Internal envelopes contain `id`, `problem`, `answer` and the untouched original row. Uploads contain all original columns plus Korean fields and stable IDs; DAPO also gains its extracted English `problem`.

Source metadata was inspected directly through the [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api). Implementation references include [Math-Verify](https://github.com/huggingface/Math-Verify), [Anthropic's async SDK](https://github.com/anthropics/anthropic-sdk-python), [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/), and the [EXAONE model](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct). The original specifications are retained in `docs/`.
