"""Build a standalone RunPod notebook for four independent GPU samplers."""
from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_sft_pool_sampling_notebook import FILES
from build_sft_pool_full_run_notebook import cells as colab_cells

NAME = 'full_run_clean_sft_pool_qwen3b_8_runpod_4gpu.ipynb'


def cells():
    original = colab_cells()
    prepare = original[5].source.replace('Path(DRIVE_ROOT)', 'Path(STORAGE_ROOT)')
    prepare = prepare.replace("revision='main'", 'revision=SOURCE_REVISION')
    prepare = prepare.replace("'pipeline.sample_sft_pool'", "'pipeline.sample_sft_pool_runpod'")
    prepare = prepare.replace("'--config', str(CONFIG_PATH), '--run-dir', str(RUN_DIR)]",
                              "'--config', str(CONFIG_PATH), '--run-dir', str(RUN_DIR), '--gpus', ','.join(GPU_IDS)]")
    return [
        markdown('''
        # Qwen2.5-3B — RunPod · 4 GPUs · eight answers per problem

        Use **one Linux RunPod Pod with four RTX PRO 6000 GPUs (96 GB each)** and JupyterLab.
        Each GPU loads its own pinned Qwen2.5-3B base model and processes a disjoint subset
        of problems. Each problem still receives eight responses. This is generation and
        grading, not model training. No multi-node networking or tensor parallelism is used.

        Before running:
        - Attach persistent storage at `/workspace`. A **network volume** survives Pod
          deletion; a Pod volume disk does not. Check available storage for the runtime,
          model cache, response journals, merged journal, Parquet and grading audit.
        - Supply a write-capable `HF_TOKEN` environment variable, or enter it into the hidden prompt.
        - Run all cells. The notebook installs its embedded code and locked runtime;
          no repository clone is needed. GPU preflight checks precede full generation.

        Sampling remains temperature=1, top-p=1, maximum 8,192 new tokens per answer,
        with the same prompt, grading and quality filters as the Colab notebook.
        Rows with 0/8 correct remain in the output. There is no generation smoke gate.
        **UPLOAD_FULL=True publishes automatically to Seungjun/clean-math-sft-pool-30k
        after all four workers finish and the merged result validates.**

        Re-run using the same settings, code and RUN_NAME to resume completed problems.
        Each worker has its own journal and log. Changed settings/code or GPU type are rejected.
        Keep the Pod and Jupyter kernel running; this is not a detached background job.
        Interrupting the generation cell stops workers; unfinished problems are regenerated on resume.
        Do not run two copies against the same RUN_NAME or overwrite the code during generation.

        Storage reference: [RunPod storage guide](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage).
        '''),
        code('''
        RUN_NAME = 'qwen25-3b-base-eight-runpod4-other-v1'
        STORAGE_ROOT = '/workspace/LG-SFT-Pool-Sampling'
        CODE_ROOT = '/workspace/lg-sft-pool-sampling-runpod4'
        EVAL_ENV = '/workspace/lg-eval-env'
        GPU_IDS = ['0', '1', '2', '3']  # CUDA GPU IDs or UUIDs visible to this Pod.
        SOURCE_REVISION = 'main'  # First run pins this; resume reuses the saved commit.
        # After an earlier upload, use that run's ORIGINAL dataset_revision instead of main.
        TEMPERATURE = 1.0
        TOP_P = 1.0
        MAX_NEW_TOKENS = 8192
        MAX_NUM_SEQS = 16  # Per GPU: conservative original setting, 64 active sequences total.
        SEED = 42
        UPLOAD_FULL = True
        '''),
        markdown('## 1. Install the isolated runtime and check all four GPUs'),
        code(f'''
        import base64, getpass, io, json, os, subprocess, sys, zipfile
        from pathlib import Path
        if sys.platform != 'linux':
            raise RuntimeError('Run this notebook in the Linux RunPod Pod')
        if len(GPU_IDS) != 4 or len(set(GPU_IDS)) != 4:
            raise ValueError('Four distinct GPUs are required')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_IDS)
        os.environ['HF_HOME'] = '/workspace/huggingface'
        os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN') or getpass.getpass('Hugging Face write token: ')
        if not os.environ['HF_TOKEN']:
            raise ValueError('HF_TOKEN is required')
        Path(CODE_ROOT).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload([ROOT / p for p in FILES + ['pipeline/sample_sft_pool_runpod.py', 'common/pool_progress.py']])!r}))) as bundle:
            bundle.extractall(CODE_ROOT)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.11.22', 'PyYAML==6.0.3'])
        subprocess.check_call([sys.executable, 'scripts/setup_eval_runtime.py', '--venv', EVAL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.pool_progress import run_pool_logged as run_logged
        gpu_check = """
        import torch
        assert torch.cuda.device_count() == 4, 'The four selected GPUs must all be visible'
        for i in range(4):
            props = torch.cuda.get_device_properties(i)
            print(i, props.name, round(props.total_memory / 2**30, 1), 'GiB')
            assert props.total_memory >= 90 * 2**30, 'Expected a 96 GB GPU'
            x = torch.ones(1, device=f'cuda:{{i}}')
            assert x.item() == 1
        """
        import textwrap
        run_logged([str(Path(EVAL_ENV) / 'bin/python'), '-c', textwrap.dedent(gpu_check)],
                   cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'gpu_preflight.log')
        '''),
        markdown('''
        ## 2. Check grading and pin the cleaned input

        Only the coordinator prepares the input and writes the shared manifest. The original
        text cleanup and additional quality filters are unchanged. Inspect the printed row
        count and quality report below. The full run partitions eligible rows by index modulo 4.
        '''),
        code(prepare),
        markdown('''
        ## 3. Run all four GPU workers, then merge

        Each worker uses tensor_parallel_size=1 and its own CUDA_VISIBLE_DEVICES.
        The model is downloaded once to the shared cache before workers launch.
        One progress panel updates every 30 seconds with per-GPU counts; detailed logs are in
        `RUN_DIR/workers/0/console.log` through `RUN_DIR/workers/3/console.log`.
        Startup compilation can take several minutes. CPU grading runs separately in each worker.

        Each completed problem is synced to its worker journal. The coordinator stops the
        other workers if one fails. Fix the cause and rerun with unchanged settings to resume;
        tuning sampling/batching requires a new RUN_NAME. A merge is allowed only when every
        expected row occurs exactly once in the correct worker, with valid annotations and
        matching runtime/prompt records. Export keeps original input order.
        '''),
        original[7],
        markdown(original[8].source.replace('Drive', 'persistent storage')),
        code(original[9].source.replace('Drive', 'persistent storage')),
        markdown('''
        ## Files and restart notes

        - `workers/<rank>/responses.jsonl`: durable per-GPU checkpoints; retain all four.
        - `workers/<rank>/console.log`: startup, generation progress and failure details.
        - `parallel_protocol.json`: fixed worker count, assignment rule and driver hash.
        - `responses.jsonl`: merged journal, rebuilt only after workers complete.
        - `full/data/*.parquet`, `full/summary.json`, `full/grading_audit.jsonl`: final outputs.
        - `full/provenance.json`: original experiment metadata plus the four-worker protocol.

        Upload replaces the destination's default train split, preserves original files, and
        refuses to overwrite a newer Hub commit. After a successful upload, a NEW run must
        set SOURCE_REVISION to the original cleaned commit printed in section 2; the now
        annotated `main` is intentionally rejected as input. Existing runs resume their pinned input.

        Generation is parallel but exact 4x speedup is not guaranteed. Per-problem seeds remain
        unchanged; different GPU scheduling can change exact sampled text. Settings preserve
        the original 16-sequence limit per GPU; benchmark before raising it in a new run.
        '''),
    ]


if __name__ == '__main__':
    write_notebook(NAME, cells())
