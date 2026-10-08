"""CPU-only Colab for other-answer regrading, preserving all generated responses."""
from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_sft_pool_sampling_notebook import FILES


def cells():
    files = FILES + ['pipeline/regrade_other.py']
    return [
        markdown('''
        # Regrade saved `other` answers — no new model generation

        Only rows with `answer_type="other"` are regraded. Clock times, ratios,
        explicit base numerals, percentages, single numeric-string lists, and valid
        arithmetic equality chains are normalized before Math-Verify compares them.
        Powers, factorials, and normal expressions remain Math-Verify's responsibility.
        Ambiguous notation is not guessed. Other answer types stay unchanged.

        **CPU runtime is sufficient; no HF token is needed.** Use saved smoke JSONL
        or full-run Parquet files containing the six generated columns. All rows,
        source answers, generated responses, token counts, and finish reasons are kept.
        Only `extracted_answers`, `correct`, and `num_correct` can change on other rows.
        The notebook writes separate outputs and per-response comparison reports.
        It does not alter a running generation job, filter zero-score rows, or upload to HF.
        '''),
        code('''
        MOUNT_DRIVE = True
        # For a full run, enter one or more saved Parquet paths on Drive.
        # Leave empty to select JSONL/Parquet files using the upload picker instead.
        INPUT_FILES = []
        OUTPUT_DIR = '/content/drive/MyDrive/LG-SFT-Pool-Other-Regraded'
        CODE_ROOT = '/content/other-answer-regrade-code'
        '''),
        code(f'''
        import subprocess, sys, base64, io, zipfile, json
        from pathlib import Path
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
            'math-verify[antlr4_13_2]==0.9.0', 'latex2sympy2-extended==1.11.0',
            'sympy==1.14.0', 'pyarrow==25.0.1', 'PyYAML==6.0.3'])
        if MOUNT_DRIVE:
            from google.colab import drive
            drive.mount('/content/drive')
        elif OUTPUT_DIR.startswith('/content/drive/'):
            OUTPUT_DIR = '/content/other-answer-regraded'
        Path(CODE_ROOT).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload([ROOT / p for p in files])!r}))) as bundle:
            bundle.extractall(CODE_ROOT)
        subprocess.check_call([sys.executable, '-m', 'eval.pool_grading_checks'], cwd=CODE_ROOT)
        '''),
        code('''
        if not INPUT_FILES:
            from google.colab import files
            uploaded = files.upload()
            input_dir = Path('/content/other-answer-inputs')
            input_dir.mkdir(exist_ok=True)
            INPUT_FILES = []
            for name, content in uploaded.items():
                local = input_dir / Path(name).name
                local.write_bytes(content)
                INPUT_FILES.append(str(local))
            del uploaded
        if not INPUT_FILES:
            raise ValueError('Select at least one annotated JSONL or Parquet file.')
        if any(not Path(p).is_file() or Path(p).suffix.lower() not in {'.jsonl','.parquet'} for p in INPUT_FILES):
            raise ValueError('Every input must be an existing JSONL or Parquet file.')
        print('Input files:', INPUT_FILES)
        '''),
        markdown('''
        ## Regrade and save

        The output directory has one subdirectory per input. Parquet files are processed
        in small batches. An `.audit.jsonl` shows old/new grades and normalization details;
        a `.summary.json` marks successful completion and reports grading exceptions.
        The input files are never overwritten. Keep the summary with each output file.
        '''),
        code('''
        summaries = []
        for index, input_file in enumerate(INPUT_FILES):
            destination = Path(OUTPUT_DIR) / f'file-{index:04d}'
            subprocess.check_call([sys.executable, '-m', 'pipeline.regrade_other',
                '--input', str(input_file), '--output-dir', str(destination)], cwd=CODE_ROOT)
            summary_file = destination / (Path(input_file).stem + '.other-regraded.summary.json')
            summary = json.loads(summary_file.read_text())
            summaries.append(summary)
        print(json.dumps(summaries, indent=2))
        print('Saved under:', OUTPUT_DIR)
        '''),
        code('''
        DOWNLOAD_OUTPUTS = False  # Optional; Drive already retains the output files.
        if DOWNLOAD_OUTPUTS:
            import shutil
            from google.colab import files
            archive = shutil.make_archive('/content/other-answer-regraded', 'zip', OUTPUT_DIR)
            files.download(archive)
        '''),
    ]


if __name__ == '__main__':
    write_notebook('regrade_other_answers.ipynb', cells())
