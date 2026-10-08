"""Self-contained Colab: smoke-review first, then eight base-model responses per row."""
from build_notebooks import ROOT, code, markdown, payload, write_notebook

FILES = [
    'common/__init__.py', 'common/io.py', 'common/math_text.py', 'common/process.py',
    'common/prompts.py', 'common/english_prompts.py', 'eval/__init__.py',
    'eval/scoring.py', 'eval/engines.py', 'eval/batched_generation.py',
    'pipeline/__init__.py', 'pipeline/sample_sft_pool.py', 'pipeline/sft_pool_quality.py',
    'eval/final_answer.py', 'eval/other_answers.py', 'eval/pool_grading_checks.py', 'configs/sft_pool_quality.yaml',
    'configs/sft_pool_smoke_ids.json',
    'configs/sft_pool_sampling.yaml', 'scripts/setup_eval_runtime.py',
    'requirements-eval.lock',
]


def cells():
    return [
        markdown('''
        # Eight Qwen2.5-3B base responses per problem — smoke test, review, full run

        Input and upload destination: **Seungjun/clean-math-sft-pool-30k**.
        On the first run this notebook resolves **the latest HF commit**, verifies the
        published text-cleanup report, and detects the actual source row count.
        Resume uses that same pinned commit even if the Hub later changes. No manual
        commit or row-count edits are needed. Finish the cleanup-and-upload notebook first.
        The notebook applies its existing additional quality screening before generation. The full run generates eight
        responses per eligible row; exclusions and verified gold corrections are audited.
        This uses the pinned **Qwen/Qwen2.5-3B base** weights, not an SFT/RL checkpoint.
        It follows the repository's English instruction and native tokenizer template.

        **Start with the 50-row smoke test (400 responses).** It saves every response,
        parsed answer, grade, token count, finish reason, and `num_correct` to a
        downloadable JSONL and Parquet. Review the outputs before enabling the full run.
        **Run all with the defaults stops after the smoke test; full generation and upload
        remain disabled.** Smoke responses are reused in the full run.

        Use a GPU Colab runtime (L4/A100 or a larger GPU recommended) and a write-capable
        **HF_TOKEN** in Colab Secrets. The locked inference runtime is installed in a
        separate Python environment. Drive stores durable checkpoints and outputs.
        Full generation may be a long job; smoke summary token counts help estimate scale.
        No model training is performed.
        '''),
        code('''
        # Run settings. Choose a NEW RUN_NAME whenever you change model/sampling settings.
        RUN_NAME = 'qwen25-3b-base-eight-cleaned-other-v1'
        DRIVE_ROOT = '/content/drive/MyDrive/LG-SFT-Pool-Sampling'
        CODE_ROOT = '/content/lg-sft-pool-sampling'
        EVAL_ENV = '/content/lg-eval-env'
        TEMPERATURE = 1.0
        TOP_P = 1.0
        MAX_NEW_TOKENS = 8192  # Per response; includes the entire generated solution.
        MAX_NUM_SEQS = 16     # Lower this if GPU memory is tight.
        SEED = 42
        REPLAY_LAST_SMOKE = True  # Reuse eligible IDs from smoke_results (1); replace excluded rows.
        RUN_FULL = False      # Turn on only AFTER inspecting the smoke output.
        UPLOAD_FULL = False   # Turn on only when ready to publish the full annotated dataset.
        '''),
        markdown('''
        ## 1. Mount Drive and install the bundled runtime

        The code and pinned dependency lock are embedded; no repository clone is needed.
        The model/source revisions are recorded, and existing checkpoints are rejected if
        settings, code, GPU type, or software versions change.
        '''),
        code(f'''
        import base64, io, json, os, subprocess, sys, zipfile
        from pathlib import Path
        from google.colab import drive, userdata
        drive.mount('/content/drive')
        os.environ['HF_TOKEN'] = userdata.get("HF_TOKEN")
        if not os.environ['HF_TOKEN']:
            raise ValueError('Enable a write-capable HF_TOKEN in Colab Secrets')
        Path(CODE_ROOT).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload([ROOT / p for p in FILES])!r}))) as bundle:
            bundle.extractall(CODE_ROOT)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.11.22', 'PyYAML==6.0.3'])
        subprocess.check_call([sys.executable, 'scripts/setup_eval_runtime.py', '--venv', EVAL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        '''),
        markdown('''
        ## 2. Pin and validate the input; choose 50 random rows

        First, CPU regression checks exercise equation preservation, factorized form,
        radix notation, junk answers, and non-final matches. A failure stops this cell.
        Unreviewed `synthetic_math` rows are quarantined by default pending source review;
        individually reviewed corrections can remain eligible. Gold parseability and
        supported factorization forms are checked across the source before GPU work.
        Screening does not prove every remaining reference answer correct.

        With `REPLAY_LAST_SMOKE=True`, retain eligible IDs from your second smoke file
        and fill excluded slots randomly to reach 50. The selection report identifies
        shared questions and replacements; compare scores only on shared IDs.
        Set it to False for a completely fresh random selection.
        Sampling is without replacement with seed 42. Each question receives eight
        stochastic generations at temperature 1.0, top-p 1.0 by default. Per-row seeds
        make smoke/full reuse consistent; GPU scheduling can still affect exact text.
        Prompts are never silently truncated: an oversized input stops the run.
        '''),
        code('''
        run_logged([str(Path(EVAL_ENV) / 'bin/python'), '-m', 'eval.pool_grading_checks'],
                   cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'grading_checks.log')
        import yaml
        CONFIG = yaml.safe_load((Path(CODE_ROOT) / 'configs/sft_pool_sampling.yaml').read_text())
        CONFIG['data'].update(revision='main', expected_rows=None, require_text_cleanup=True)
        CONFIG['sampling'].update(temperature=TEMPERATURE, top_p=TOP_P,
                                  max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        CONFIG['engine']['max_num_seqs'] = MAX_NUM_SEQS
        CONFIG['smoke']['seed'] = SEED
        if REPLAY_LAST_SMOKE:
            CONFIG['smoke']['preferred_ids'] = json.loads(
                (Path(CODE_ROOT) / 'configs/sft_pool_smoke_ids.json').read_text())
        RUN_DIR = Path(DRIVE_ROOT) / RUN_NAME
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH = RUN_DIR / 'requested_config.yaml'
        CONFIG_PATH.write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        COMMAND = [str(Path(EVAL_ENV) / 'bin/python'), '-m', 'pipeline.sample_sft_pool',
                   '--config', str(CONFIG_PATH), '--run-dir', str(RUN_DIR)]
        run_logged(COMMAND + ['--mode', 'prepare'], cwd=CODE_ROOT, log_path=RUN_DIR / 'prepare_console.log')
        selection = [json.loads(line) for line in (RUN_DIR / 'smoke_selection.jsonl').read_text().splitlines()]
        manifest = json.loads((RUN_DIR / 'manifest.json').read_text())
        print('Pinned cleaned HF commit:', manifest['dataset_revision'])
        print('Rows in cleaned HF source:', manifest['source_rows'])
        print('Quality screening:', (RUN_DIR / 'quality_summary.json').read_text())
        print('Every exclusion/correction:', RUN_DIR / 'quality_decisions.jsonl')
        print('Selection/replacement report:', (RUN_DIR / 'smoke_selection_summary.json').read_text())
        print('Eligible smoke rows:', len(selection))
        for row in selection[:5]:
            print(row['id'], row['problem'][:160])
        print('Run directory:', RUN_DIR)
        '''),
        markdown('''
        ## Optional: regrade your previous smoke file without new generation

        Enable this cell to upload either of your original 50-row smoke JSONL files. It applies reviewed
        corrections, quarantines invalid rows, and saves per-response before/after grades.
        This diagnostic sample can contain fewer than 50 rows. It is separate from the
        fresh smoke and never mixed into its generation journal.
        '''),
        code('''
        REGRADE_PREVIOUS_SMOKE = False
        if REGRADE_PREVIOUS_SMOKE:
            from google.colab import files
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Upload exactly one original smoke_results.jsonl')
            previous = RUN_DIR / 'previous_smoke.jsonl'
            previous.write_bytes(next(iter(uploaded.values())))
            run_logged(COMMAND + ['--mode', 'regrade', '--input-smoke', str(previous)],
                       cwd=CODE_ROOT, log_path=RUN_DIR / 'regrade_console.log')
            print((RUN_DIR / 'regrade/summary.json').read_text())
            files.download(str(RUN_DIR / 'regrade/smoke_results_regraded.jsonl'))
            files.download(str(RUN_DIR / 'regrade/grading_changes.jsonl'))
            files.download(str(RUN_DIR / 'regrade/quality_decisions.jsonl'))
        '''),
        markdown('''
        ## 3. Smoke test — 50 rows × 8 responses

        This is real GPU generation and Math-Verify grading. Each completed problem is
        flushed to Drive. If the runtime disconnects, rerun with the same settings to
        resume. The generation subprocess exits afterward, releasing GPU memory.

        The scorer prefers the **last `\\boxed{...}`**, then accepts a clear explicit
        final-answer statement or unambiguous final math line. It never searches the
        working for the gold value. Missing/ambiguous answers are stored as `""` and false.
        For `answer_type="other"` only, clock times, ratios, explicit base numerals,
        percentages, singleton numeric-string lists, and consistent numeric equality
        chains are normalized before Math-Verify compares them. Colon answers require
        question context to distinguish clock times from ratios; ambiguous formats fail
        with an audit reason. Other answer types keep their previous grading behavior.
        Complete equations are preserved. Rational univariate polynomial equations
        (degree at most 12) are compared up to a nonzero constant; variable renaming
        is allowed only when the question asks to construct an equation. Factorization
        requires complete polynomial factors over the rationals in this supported domain;
        unsupported reference forms are quarantined before generation. Explicit radix
        numerals are normalized only in base-number questions. Unparseable/junk final
        answers are stored as empty strings. Degree labels are normalized only for
        explicitly angular questions. Math-Verify remains the parser/equivalence engine:
        https://github.com/huggingface/Math-Verify (locked to math-verify 0.9.0).
        `grading_audit.jsonl` records extraction methods and exact stop diagnostics. `response_tokens` counts
        output token IDs, excluding the prompt. `finish_reasons` records `stop` or `length`;
        truncated responses are still graded, with truncation visible for review.
        Generation stops on Qwen's `<|endoftext|>` (151643) or `<|im_end|>` (151645),
        with EOS enabled. The tokenizer mappings are checked at startup. The 8,192-token
        cap is recorded as `length`; an EOS/end-of-turn stop is recorded as `stop`.
        '''),
        code('''
        run_logged(COMMAND + ['--mode', 'smoke'], cwd=CODE_ROOT,
                   log_path=RUN_DIR / 'smoke_console.log')
        SMOKE_DIR = RUN_DIR / 'smoke'
        SMOKE_FILE = SMOKE_DIR / 'smoke_results.jsonl'
        smoke_summary = json.loads((SMOKE_DIR / 'summary.json').read_text())
        print(json.dumps(smoke_summary, indent=2))
        print('All 400 responses:', SMOKE_FILE)
        print('Parquet:', SMOKE_DIR / 'data/train-00000.parquet')
        '''),
        markdown('''
        ## 4. Review and download the smoke result

        The JSONL contains 50 complete source rows with the six new columns.
        Review source gold quality, extraction failures, truncation, and the 0–8 pass-count histogram.
        `num_correct` is a count, not a percentage. The full run remains disabled below.
        '''),
        code('''
        import pandas as pd
        smoke_rows = [json.loads(line) for line in SMOKE_FILE.read_text().splitlines()]
        display(pd.DataFrame({
            'num_correct': list(range(9)),
            'rows': [smoke_summary['num_correct_histogram'][str(i)] for i in range(9)]}))
        display(pd.DataFrame(smoke_rows)[['id', 'problem', 'gold_answer', 'num_correct']])
        EXAMPLE_ROW = 0
        example = smoke_rows[EXAMPLE_ROW]
        print('QUESTION:', example['problem'])
        print('GOLD:', example['gold_answer'])
        for i in range(8):
            print(f"\\nRESPONSE {i+1}: correct={example['correct'][i]}, "
                  f"tokens={example['response_tokens'][i]}, finish={example['finish_reasons'][i]}")
            print('EXTRACTED:', example['extracted_answers'][i])
            print(example['responses'][i])
        '''),
        code('''
        DOWNLOAD_SMOKE = True
        if DOWNLOAD_SMOKE:
            from google.colab import files
            files.download(str(SMOKE_FILE))
        '''),
        markdown('''
        ## 5. Full run — enable only after reviewing the smoke test

        Set `RUN_FULL = True` in the following cell to continue. All eligible rows receive
        eight responses after quality screening. The 50 completed smoke rows are reused. Checkpoints are stored
        per problem, so a disconnect loses only currently unfinished problems.
        Full outputs are sharded Parquet to avoid holding all response strings in memory.
        To change generation settings, start a new RUN_NAME and rerun preparation/smoke.
        '''),
        code('''
        # RUN_FULL = True  # Uncomment after you decide to proceed.
        if RUN_FULL:
            run_logged(COMMAND + ['--mode', 'full'], cwd=CODE_ROOT,
                       log_path=RUN_DIR / 'full_console.log')
            print((RUN_DIR / 'full/summary.json').read_text())
        else:
            print('Full run disabled. Review smoke_results.jsonl first.')
        '''),
        markdown('''
        ## 6. Re-upload the full annotated dataset

        Set `UPLOAD_FULL = True` below after the full run finishes. This updates
        **Seungjun/clean-math-sft-pool-30k**, preserving its visibility and retaining
        original source files. The default split contains eligible rows. Verified gold
        fixes retain the prior value in `gold_answer_original`; `quality_action` identifies
        corrected rows. All other source columns remain unchanged.
        The dataset card is updated to load the annotated Parquet files by default.
        Publication validates every prepared row, checks vector lengths/types and checksums,
        refuses to overwrite a newer Hub revision, and verifies the uploaded files.
        Smoke-only or incomplete runs cannot be uploaded through this cell.
        '''),
        code('''
        # UPLOAD_FULL = True  # Uncomment when ready to publish the complete result.
        if UPLOAD_FULL:
            run_logged(COMMAND + ['--mode', 'upload'], cwd=CODE_ROOT,
                       log_path=RUN_DIR / 'upload_console.log')
            receipt = json.loads((RUN_DIR / 'upload_receipt.json').read_text())
            print('Verified dataset:', receipt['url'])
        else:
            print('Upload disabled. Full results remain on Drive until you enable it.')
        '''),
    ]


if __name__ == '__main__':
    write_notebook('sample_clean_sft_pool_qwen3b_8.ipynb', cells())
