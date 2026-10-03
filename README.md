# Math reasoning on EXAONE

The current direction keeps `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` and uses the original English datasets. Translation is no longer part of the active data pipeline; earlier translation notebooks remain available as historical experiments.

## DeepSeek reasoning regeneration notebook

Open [`regenerate_s1_deepseek.ipynb`](notebooks/regenerate_s1_deepseek.ipynb) on **Colab CPU**
with `HF_TOKEN` (read access) and `DEEPSEEK_API_KEY` in Colab Secrets. The notebook bundles its
code and config, so no Git push is required. It loads all 996 pinned rows from
`Seungjun/dp_removed_s1K-1.1` and asks `deepseek-v4-pro` to solve each question independently
with thinking enabled. The final response must be a self-contained worked solution with four
named sections: Planning, Evaluation (including the full derivation), Reflection, and Exploration,
followed by the final answer. There is no brevity target or correctness-based filtering.
The intended new SFT target is `deepseek-v4-pro_answer`; `deepseek-v4-pro_reasoning` retains the
raw API thinking for inspection. The existing SFT loader is not changed by this notebook.

- The separate **smoke cell** samples 20 rows with seed 42 and saves/downloads a five-column
  `comparison.csv`: question, original reasoning, original answer, new reasoning, new answer.
- The **full-run cell** defaults off. Enable `RUN_FULL_GENERATION` to process all source rows,
  reusing completed smoke results. Its JSONL export retains every original column and appends
  `deepseek-v4-pro_reasoning` and `deepseek-v4-pro_answer`.
- Defaults are 16 concurrent requests, pacing, transient-error retries, and per-response journals
  on Drive under `LG-AIME-S1-DeepSeek/runs/<model>/<settings-id>/`. Rerunning resumes successful
  rows and retries failures. Incomplete outputs and final responses with missing, empty, duplicate,
  or out-of-order sections are flagged separately; partial exports retain
  all source rows with null generated fields for failures. Settings/code changes create a new run.

Settings and the full prompt are in [`regenerate_s1_deepseek.yaml`](configs/regenerate_s1_deepseek.yaml).
`max_tokens=131072` is an API output ceiling, not a requested length; reaching it is flagged.
No SFT length filter is applied to the source or regenerated data. This notebook does not change
SFT/evaluation code or publish to Hugging Face. Local tests exercise mock API calls, including
the actual smoke/full notebook cells; live generation must be checked in the Colab smoke run.
Rebuild with `python scripts/build_regeneration_notebook.py` after changing its bundled code.
For the structured-final-response update, open the rebuilt notebook in a **fresh Colab runtime**
and rerun the 20-row smoke test. A new output folder prevents reuse of the previous unstructured
answers, while the seed keeps the same 20 questions. Raw failed responses remain in the journal.
Section checks only enforce structure; use the smoke CSV to inspect the actual solution quality.

## Upload existing DeepSeek results to Hugging Face

Open [`upload_s1_deepseek_to_huggingface.ipynb`](notebooks/upload_s1_deepseek_to_huggingface.ipynb)
on **Colab CPU**, add a write-capable `HF_TOKEN` in Secrets, and run cells in order.
The self-contained notebook reads the existing Drive export and `status.jsonl`, verifies them
against the pinned source and current HF dataset, and updates `Seungjun/dp_removed_s1K-1.1`
with `deepseek-v4-pro_reasoning` and `deepseek-v4-pro_answer`. Both columns are retained for
later combined thinking-plus-answer training. All 996 source rows and original columns stay
intact; unsuccessful rows keep null Pro fields. No length or correctness filter is applied.

The upload cell publishes the dataset and a ZIP containing both original JSONL files and a
provenance manifest. The verification cell reloads the uploaded revision, compares all values,
checks the ZIP checksum, and saves a receipt under `LG-AIME-S1-DeepSeek/hf_uploads/` on Drive.
The existing repository visibility is retained. Settings and default input paths are in
[`publish_s1_regeneration.yaml`](configs/publish_s1_regeneration.yaml); change the paths in the
notebook if needed. Identical reruns and filling missing outputs are supported; replacing
existing non-null Pro text is rejected. Existing training configs still pin the old source
revision and columns. This upload does not switch them to Pro data.

Rebuild with `python scripts/build_s1_upload_notebook.py`.

## Stage 1 SFT implementation

### Llama 3.1 8B LoRA experiment

Open [`train_stage1_llama31_lora.ipynb`](notebooks/train_stage1_llama31_lora.ipynb) for the
separate Llama experiment. It starts from pinned `meta-llama/Llama-3.1-8B-Instruct` and uses
the original `deepseek_thinking_trajectory` plus `deepseek_attempt` columns at the original
decontaminated dataset revision. Your Colab `HF_TOKEN` needs access to the gated Llama model.
Push the new implementation to GitHub main before running notebook setup.

[`configs/stage1_sft_llama31_lora.yaml`](configs/stage1_sft_llama31_lora.yaml) specifies rank 32,
alpha 64, dropout 0.05, all attention/MLP projections, LR 5e-5, five epochs, batch 1,
accumulation 16, cosine decay, 5% warmup and seed 42. Frozen base weights use BF16 without
quantization. Native PyTorch SDPA FlashAttention and nonreentrant gradient checkpointing
limit memory use. Only the LoRA matrices are trainable. The tokenizer uses its native EOT
and existing padding tokens; no vocabulary expansion is performed. Full formatted sequences
over 20,480 tokens are dropped using Llama tokenization, so the retained count is recomputed.

The notebook separates preparation, a GPU smoke run on the longest retained example, and full
training. Each smoke uses a separate timestamped directory and cannot alter full-run weights.
The default Drive root is `LG-AIME-Llama31-LoRA`. The full run saves
`checkpoints/stage1/sft_llama31_8b_lora_s1k/epoch_1/` through `epoch_5/` and per-step metrics in
`logs/stage1/sft_llama31_8b_lora_s1k/steps.jsonl`. Checkpoints contain **adapters**, tokenizer,
optimizer/scheduler/RNG state, and the pinned base revision. Evaluation must load the pinned
Llama base with its adapter or merge them; these are not standalone EXAONE checkpoints.
No evaluation runs in this notebook. Existing EXAONE training and evaluation code is unchanged.

Rebuild with `python scripts/build_llama_sft_notebook.py`. Local CPU tests cover loss masks,
strict filtering, frozen base weights, adapter reload equivalence and epoch-checkpoint resume.
The actual gated tokenizer and 8B/20K-token GPU memory check run in Colab; they have not been
verified locally.

### EXAONE runs

For an explicitly named Pro training launcher, open
[`train_stage1_sft_deepseek_pro.ipynb`](notebooks/train_stage1_sft_deepseek_pro.ipynb).
It shares the existing training workflow: Pro reasoning plus answer, five epochs from the
original base model, full-sequence filtering at 20,480 tokens, and Drive checkpoints/loss logs.
Run setup and preparation, then enable `RUN_TRAINING` in the last cell. It does not run evaluation.

Evaluate that run with [`evaluate_stage1_sft_deepseek_pro.ipynb`](notebooks/evaluate_stage1_sft_deepseek_pro.ipynb).
Set `TRAIN_ROOT` to its actual Drive folder (including a fresh-run timestamp, if used).
Blank selects a single discovered completed Pro epoch-5 run; multiple matches require selection.
Defaults are temperature 0, one answer per problem, 16,384 thinking tokens and 4,096 answer tokens,
with continuous batching across all five benchmarks. Smoke and full runs are separate cells.
Results live under that training folder's `evaluation_deepseek_pro/` directory.

`train_stage1_sft.ipynb` now defaults to `configs/stage1_sft_deepseek_pro.yaml`: the original
EXAONE base model trains on `deepseek-v4-pro_reasoning` plus `deepseek-v4-pro_answer`, with
all previous training hyperparameters unchanged. Publish the Pro export first. The loader resolves
the HF `main` revision to a commit, verifies the expected full dataset digest and original
decontamination IDs, skips missing outputs, and drops full formatted sequences over 20,480 tokens.
The supplied export retains 718 examples (six missing and 272 overlength exclusions).
Checkpoints and logs use `sft_s1k_deepseek_pro`, preserving the earlier `sft_s1k` run.
`evaluate_stage1_sft_temp1.ipynb` selects `configs/stage1_sft.yaml` and the **previous SFT run's**
epoch-5 checkpoint to test whether budget forcing improves its results.
The old `configs/stage1_sft.yaml`
continues to reproduce the original R1-column run. Preparation reports record both exclusion counts.
Push these code changes to GitHub before running the Colab training notebook.

For **selectable base/SFT evaluation and temperature**, use
[`evaluate_stage1_sft_temp1.ipynb`](notebooks/evaluate_stage1_sft_temp1.ipynb).
Set `MODEL_KIND` to `sft` (default) or `base`, and set `TEMPERATURE` (default 1.0).
SFT uses the previous `sft_s1k/epoch_5` checkpoint with budget forcing; base loads the pinned
original EXAONE model with ordinary generation and needs no SFT files. One answer per problem
is generated on all five benchmarks (630 responses). Temperature zero uses greedy decoding
with top-p 1.0; other temperatures retain top-p 0.7. The profile is selected automatically.
For SFT, earlier epochs remain optional. Set `MAX_THINKING_TOKENS` in
the notebook (default **18,432**). Minimum reasoning stays zero, and the answer allowance is
computed as **20,480 minus the thinking cap** (default **2,048**). At the reasoning cap it
inserts `</think>` and `Final Answer:` and continues generation. Natural `</think>` starts the
answer phase earlier; natural EOS is respected, with no forced "Wait" extension. Injected tokens
count against the answer allowance, so total continuation tokens remain at most 20,480.
Both phases retain continuous batching, sampling settings, and the existing scoring code.

The default result is `profiles/sample1_budget/thinking_18432_answer_2048/temperature_1.0_top_p_0.7/full/results/stage1/sft_s1k.json` under
the evaluation Drive root. Raw records log forced transitions and reasoning/answer/injected token
counts. The token-length metric includes inserted tokens. Console logs include the run name and
temperature and budget. Each setting gets a separate results folder. Rerun preparation after
editing the model, temperature, or cap; that cell saves resolved sampling/budget YAML files to
`TRAIN_ROOT/eval_configs/` and passes them to both smoke/full runs using `--sampling_config`
and, only for SFT, `--budget_config`. Base results use
`profiles/sample1/temperature_<temperature>_top_p_<top_p>/full/results/stage0/baseline_english.json`.
The original `sample1_budget16k` CLI profile
remains available with its fixed 16,384/4,096 split.
CONFIG already selects `configs/stage1_sft.yaml` for the old R1 checkpoint.
Base mode selects `sample1` without budget forcing; SFT always selects `sample1_budget`.
Existing protocols bind evaluator
code: use the original code to resume an old run, or a fresh BASELINE_ROOT for new plain-sample1
results. Old responses are not imported into the new budget-forcing profile.

No baseline run is required. Push all updated code/configs to GitHub before Colab setup.
Enable `RUN_SMOKE_EVAL` for a separate five-problem smoke test, then `RUN_STAGE1_EVAL` for
the full run. Both default off. Local tests cover forced/natural transitions, total budgets,
scoring/resume, cancellation, and notebook commands; live GPU inference must be checked in Colab.

The English SFT implementation and separate Colab launchers are ready for GPU validation:
[`train_stage1_sft.ipynb`](notebooks/train_stage1_sft.ipynb) trains for five epochs and saves
the checkpoints; [`evaluate_stage1_sft.ipynb`](notebooks/evaluate_stage1_sft.ipynb) evaluates
all five benchmarks on epoch 5 first, with epochs 1–4 optional. See
[`stage1_sft.yaml`](configs/stage1_sft.yaml), and the [run guide](docs/stage1_sft.md).
Real-tokenizer preparation retains 975 of the 996 decontaminated s1K examples after
dropping 21 over the 20,480-token limit, giving 305 optimizer steps over five epochs.
Both evaluation notebooks offer `greedy` (temperature 0, one answer) and `sample8`
(temperature 1.0, eight AIME/AMC answers and four MATH-500 answers). They use continuous
batching and separate profile results. See [`eval_profiles.yaml`](configs/eval_profiles.yaml)
and the [migration instructions](docs/stage1_sft.md#colab).
GPU training and Stage 1 accuracy results are still pending.

## DAPO pilot

For the separate **EXAONE DAPO pilot**, see
[`train_dapo_exaone.ipynb`](notebooks/train_dapo_exaone.ipynb) and
[the DAPO run guide](docs/dapo.md). Its plain `MODEL_KIND = "base"` / `"sft"`
setting selects the original EXAONE model or the original `sft_s1k/epoch_5` checkpoint.
Preparation, GPU smoke and the 100-update training pilot are separate cells; all run outputs
are saved to Drive. Push the new implementation to GitHub main before Colab setup.
The notebook now uses `configs/dapo_reuse.yaml`: two optimizer updates per generated batch,
with fixed old-policy probabilities, and a two-update smoke. It defaults to the base model
and saves the fresh experiment under `LG-AIME-DAPO-Reuse2`. The 100-update budget therefore
uses 50 fresh rollout batches. Other training hyperparameters are unchanged.

For **Llama 3.2 3B Instruct DAPO**, use
[`train_dapo_llama32_3b.ipynb`](notebooks/train_dapo_llama32_3b.ipynb), with
`configs/dapo_llama32_3b.yaml`. It starts from the pinned instruction model, uses full-parameter
training and the same two-update rollout schedule, and saves under
`LG-AIME-DAPO-Llama32-3B`. Preparation, two-update GPU smoke and full training are separate cells.
The native Llama template has a fixed date, and all three native stop tokens are configured.
Use an HF token with access to the gated model. No EXAONE/SFT files are needed.

For the **original one-pass run**, after completing step 100, use
[`continue_dapo_exaone_100_to_300.ipynb`](notebooks/continue_dapo_exaone_100_to_300.ipynb)
to restore the full training state and add 200 updates. It defaults to the base-model run,
preserves original outputs, and writes steps 101–300 under `LG-AIME-DAPO-100to300` on Drive.
Evaluate the original run's checkpoint 100 with
[`evaluate_dapo_checkpoint100.ipynb`](notebooks/evaluate_dapo_checkpoint100.ipynb): all five
benchmarks, one answer each, default temperature 0, editable sampling and no budget forcing.
For the **temperature-0 base vs DAPO-100 comparison on AMC 2023 and MATH-500 only**, use
[`compare_base_dapo100_amc_math.ipynb`](notebooks/compare_base_dapo100_amc_math.ipynb).
It evaluates 540 problems per model sequentially and displays/saves an accuracy and response-length table.
To evaluate **checkpoint 140 alone** from the continuation run on those same two benchmarks,
use [`evaluate_dapo_checkpoint140_amc_math.ipynb`](notebooks/evaluate_dapo_checkpoint140_amc_math.ipynb).
It defaults to `LG-AIME-DAPO-100to300/checkpoints/dapo_exaone_base/checkpoint-140`,
temperature 0, one answer per problem and no budget forcing. Its 540-answer run writes a
two-row summary to `LG-AIME-DAPO-Eval-140-AMC-MATH-temp0` without rerunning the baseline.

## English preparation notebook (current)

Upload [`notebooks/prepare_english_datasets.ipynb`](notebooks/prepare_english_datasets.ipynb) to **Colab CPU**, add a write-capable `HF_TOKEN`, and run the cells in order. It applies the existing normalization-v2, 8-gram, 70% per-eval-problem coverage rule to the two training datasets. It does not deduplicate within/between training datasets. All five eval sets remain unchanged, as do every retained original column and text value.

Running all cells uploads seven private datasets under `Seungjun/dp_removed_{source_repo_basename}` (change `HF_USERNAME`/`HF_PRIVATE` in settings as needed). For example, `simplescaling/s1K-1.1` becomes `Seungjun/dp_removed_s1K-1.1`, and `math-ai/aime25` becomes `Seungjun/dp_removed_aime25`. Data, original cards, provenance and the overlap report are staged and checked before upload. Exact uploaded commits are saved in `english_dataset_suite.json`; reruns republish the same staged content. No translation model or GPU is needed.

The old Korean baseline notebook remains unchanged. Use the English evaluation entry point below for this new dataset suite.

## English baseline evaluation (current)

Use [`evaluate_baseline_english.ipynb`](notebooks/evaluate_baseline_english.ipynb) on the
RTX PRO 6000 Blackwell 96GB Colab runtime. Enable the `HF_TOKEN` secret. Push local changes
before running setup: the notebook clones/updates clean `main`, records the Git commit,
and installs the locked evaluation environment. Stop active evaluation before updating.

Choose `EVAL_PROFILE` in both the baseline and SFT notebooks:

| Option | Sampling | Responses/checkpoint | Scores |
| --- | --- | --- | --- |
| `greedy` (default) | Temperature 0; one answer per problem | 630 | Accuracy/pass@1 |
| `sample8` | Temperature 1.0, top-p 0.7; eight AIME/AMC answers, four MATH-500 answers | 3,040 | AIME/AMC pass@1/4/8; MATH-500 pass@1/4; avg@n |

Both retain the original English prompts, native chat template, 20,480 output-token limit,
seed, pinned datasets and rule-based boxed-answer scorer. The five datasets contain
630 questions: AIME 2024/2025/2026 (30 each), AMC23 (40), MATH-500 (500).
[`eval_profiles.yaml`](configs/eval_profiles.yaml) defines the differences from the shared
original config. Continuous batching feeds up to 32 GPU response slots.

The baseline notebook runs a five-problem smoke check by default and downloads its archive.
After inspecting it, enable `RUN_FULL_EVAL`. Evaluate SFT using the same profile; either model can run first.
To compare both profiles, run the base and SFT model once under each option.

```bash
python -m pip install uv==0.11.22
python scripts/setup_eval_runtime.py --venv /content/lg-eval-env
# Use --smoke for a separate five-problem check, or --validate-only for dataset checks.
/content/lg-eval-env/bin/python -m eval.run_batched_eval \
  --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
  --stage stage0 --run_name baseline_english --profile greedy \
  --output_root /content/drive/MyDrive/LG-AIME-English-Eval-compatible
```

Results are isolated under `OUTPUT_ROOT/profiles/<profile>/full/results/`, with separate
`stage0/` and `stage1/` directories. Smoke uses the sibling `smoke/` directory. Each
completed problem is durably saved in `*_problems.jsonl`; the flat generations export
and final metrics are written at completion. Repeating the same command with the same
commit/settings resumes complete problems, regenerating any incomplete problem in full.
Keep the profile's protocol/runtime manifests alongside its results for SFT comparisons.

The base model remains `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`, pinned to loader revision
`e949c91dec92095908d34e6b560af77dd0c993f8` for Transformers 4.57.6 compatibility.
The prompts, answer extraction, scorer, dataset suite and dependency lock are unchanged.

These profiles start fresh and do not import old 32-sample results. Earlier sequential
and `_batched` artifacts remain untouched; resume an old run with its recorded Git commit.
The original `eval.run_english_eval` CLI/config remain available for that legacy protocol.
See the [Stage 1 guide](docs/stage1_sft.md#evaluation) for SFT commands and result paths.

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
