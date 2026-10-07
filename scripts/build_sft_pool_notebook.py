"""Build the self-contained Colab Stage 1 pool notebook, including publication."""
import hashlib
import json
import subprocess

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

FILES = [
    'common/__init__.py', 'common/io.py', 'common/math_text.py',
    'pipeline/__init__.py', 'pipeline/datasets.py', 'pipeline/sft_pool.py',
    'pipeline/sft_pool_hub.py', 'configs/sft_pool.yaml',
    'configs/english_eval_suite.json', 'configs/dapo.yaml', 'tests/test_sft_pool.py',
]


def cells():
    manifest = {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'includes_uncommitted_implementation': True,
        'files_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES},
    }
    boot = bootstrap(payload([ROOT / name for name in FILES]))
    boot.source = boot.source.replace('/content/lg-korean-aime', '/content/lg-sft-pool-code')
    boot.source += '\n(CODE_ROOT / "sft_pool_bundle_manifest.json").write_text(' + repr(json.dumps(manifest, sort_keys=True)) + ')'
    return [
        markdown('''
        # Stage 1: build and upload the clean math SFT problem pool

        Run all cells in a **CPU Colab runtime**. No GPU, model calls, or training.
        Enable a write-capable **HF_TOKEN** in Colab Secrets. The final cell uploads
        the cleaned dataset and audit reports to **Seungjun/clean-math-sft-pool**,
        creating a **private** dataset by default. Change `hf_private` to publish publicly.

        Inputs: the original 7,500 MATH training problems (EleutherAI mirror),
        NuminaMath-1.5 (896,215 rows), and the original authors' s1K (1,000 rows).
        All training sources are pinned to verified commits. The exact five evaluation
        references are bundled from this repository's `english_eval_suite.json`.
        The RL reference is the content-checked export in `configs/dapo.yaml`.

        This is a substantial CPU/disk job: allow hours, several GB of free local disk,
        and a persistent runtime. Runtime depends on symbolic parsing and the number
        of near-duplicate candidates; there is no measured full-corpus runtime yet.
        SQLite keeps the deduplication index on local disk. Google Drive stores the
        completed artifact and upload receipt. An interrupted build reruns from the
        start; cached Hugging Face downloads can be reused within the runtime.
        '''),
        code('''
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
            'datasets==5.0.1', 'huggingface-hub==0.36.2', 'PyYAML==6.0.3',
            'math-verify[antlr4_13_2]==0.9.0', 'sympy==1.14.0',
            'latex2sympy2-extended==1.11.0', 'antlr4-python3-runtime==4.13.2',
            'mpmath==1.3.0', 'pytest==9.1.1', 'nbformat==5.11.1'])
        from google.colab import userdata, drive
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError('Enable HF_TOKEN in Colab Secrets and allow notebook access')
        drive.mount('/content/drive')
        '''),
        boot,
        code('''
        import json, yaml
        from pathlib import Path
        from huggingface_hub import HfApi
        CONFIG = yaml.safe_load((CODE_ROOT / 'configs/sft_pool.yaml').read_text())
        CONFIG['hf_repo'] = 'Seungjun/clean-math-sft-pool'
        CONFIG['hf_private'] = True
        # Optional changes; all values are recorded in provenance.json.
        CONFIG['dedup_jaccard'] = 0.8
        CONFIG['coverage_threshold'] = 0.70
        CONFIG['near_miss_lower'] = 0.50
        CONFIG['reference_ngram'] = 8
        CONFIG['exclude_numina_sources'] = ['amc_aime']
        OUTPUT = Path('/content/lg-sft-pool-output')
        DRIVE_ROOT = Path('/content/drive/MyDrive/lg-korean-aime/data/sft_pool')
        # Authenticated identity only; never display or save the token.
        print('Authenticated as:', HfApi(token=HF_TOKEN).whoami()['name'])
        print('Upload destination:', CONFIG['hf_repo'])
        print('Private:', CONFIG['hf_private'])
        '''),
        markdown('''
        ## Acceptance checks before the large build

        Tests cover LaTeX normalization, embedded and short evaluation matches,
        distinct/per-reference overlap counts, numeric variants, cross-source priority,
        conflicting/equivalent answers, exact prefix matching, and count reconciliation.
        The optional local MATH-500 fixture test is skipped here; the build also checks
        verbatim and embedded matches against an actual pinned MATH-500 input.
        '''),
        code('''
        subprocess.check_call([sys.executable, '-m', 'pytest', '-q',
                               str(CODE_ROOT / 'tests/test_sft_pool.py')], cwd=CODE_ROOT)
        '''),
        markdown('''
        ## Build the pool and reports

        Filters are conservative and auditable. s1 uses both type and source/domain
        metadata, because some physics questions are labeled `math`. Unknown domains
        are dropped. Only gold answers are extracted; reasoning traces are discarded.
        Multiple required answers and free text are excluded.

        Deduplication uses an exact, disk-backed Jaccard prefix join, an alternative to
        MinHash/LSH without probabilistic candidate misses. Conflicting clusters are
        quarantined in full. Normalization is lexical, not symbolic equivalence.

        Evaluation and RL exclusion use separate inverted indexes over distinct
        8-grams with coverage measured against one reference at a time. A supplemental
        numeric-template check flags number-swapped copies even below 0.50 coverage;
        the actual coverage is never inflated. These remain manual-review candidates.
        Ordinary coverage near misses remain in the pool unless another stage removes them.
        '''),
        code('''
        import shutil
        from pipeline.sft_pool_hub import run, sha256_file
        from common.io import write_json
        SUMMARY = run(CONFIG, CODE_ROOT, OUTPUT, HF_TOKEN)
        print(json.dumps(SUMMARY, indent=2))
        # Content-addressed Drive export, separate from the local SQLite work index.
        run_id = sha256_file(OUTPUT / 'COMPLETE.json')[:16]
        SAVED_OUTPUT = DRIVE_ROOT / run_id
        shutil.copytree(OUTPUT, SAVED_OUTPUT, dirs_exist_ok=True)
        print('Completed artifacts saved to:', SAVED_OUTPUT)
        '''),
        code('''
        import pandas as pd
        from pipeline.sft_pool import iter_jsonl
        from itertools import islice
        display(pd.DataFrame(SUMMARY['counts']).T)
        preview = list(islice(iter_jsonl(OUTPUT / 'pool.jsonl'), 5))
        display(pd.DataFrame(preview)[['dataset_source', 'problem', 'gold_answer',
                                       'eval_max_coverage', 'rl_max_coverage']] if preview else 'Empty pool')
        for kind in ['eval', 'rl']:
            print(kind, 'near-miss examples:')
            display(pd.DataFrame(list(islice(iter_jsonl(OUTPUT / f'{kind}_near_misses.jsonl'), 10))))
        '''),
        markdown('''
        ## Upload the completed artifacts to Hugging Face

        Running this cell publishes `pool.jsonl`, the summary, all audit reports,
        and provenance to **Seungjun/clean-math-sft-pool**. It verifies local checksums,
        publishes one Hub commit, then verifies every uploaded file at that revision.
        Existing visibility must match `hf_private`. Re-running updates this dataset;
        source datasets and evaluation/RL references are never written.
        '''),
        code('''
        from pipeline.sft_pool_hub import upload
        RECEIPT = upload(OUTPUT, CONFIG['hf_repo'], HF_TOKEN, private=CONFIG['hf_private'])
        write_json(SAVED_OUTPUT / 'upload_receipt.json', RECEIPT)
        print('Verified upload:', RECEIPT['url'])
        print('Pinned revision:', RECEIPT['revision'])
        print('Drive receipt:', SAVED_OUTPUT / 'upload_receipt.json')
        '''),
    ]


if __name__ == '__main__':
    write_notebook('build_clean_sft_pool.ipynb', cells())
