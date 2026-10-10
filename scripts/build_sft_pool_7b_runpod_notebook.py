"""Standalone Runpod Jupyter notebooks: Math-7B, three or four GPU workers."""
from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_sft_pool_sampling_notebook import FILES

NAME = 'sample_sft_pool_math7b_8_runpod_3gpu.ipynb'


def cells(gpu_count=3):
    if gpu_count not in (3, 4):
        raise ValueError('Choose three or four GPUs')
    files = FILES + ['pipeline/sample_sft_pool_7b.py', 'pipeline/sample_sft_pool_7b_runpod.py',
                     'pipeline/sample_sft_pool_runpod.py', 'configs/sft_pool_sampling_7b.yaml',
                     'common/pool_progress.py']
    return [
        markdown(f'''
        # Math-7B-Instruct · Runpod · {gpu_count} × RTX PRO 6000

        Open this notebook in your Pod's **JupyterLab** and run the cells in order.
        Designed for **Runpod PyTorch 2.8.0, {gpu_count} × 96 GB RTX PRO 6000 GPUs**.
        It installs a separate pinned Python 3.12 / PyTorch 2.9.1 / vLLM 0.14.1 runtime
        under `/workspace`; the notebook kernel can keep the template's Python/PyTorch.
        The GPU preflight checks CUDA operation on all {gpu_count} cards before generation.

        **Only rows whose original 3B `num_correct` is 0, 1, or 2 receive eight
        Qwen/Qwen2.5-Math-7B-Instruct responses.** Rows with 3–8 retain `null` in every
        new `7B_` column. Original rows, references, and all 3B results stay unchanged.
        The existing Math-Verify grader, including `other` notation handling, is reused.
        New columns: `7B_responses`, `7B_extracted_answers`, `7B_correct`,
        `7B_response_tokens`, `7B_finish_reasons`, `7B_num_correct`.

        Selected rows are divided into {gpu_count} disjoint groups, one per GPU.
        Each GPU loads one full model. Defaults allow 64 active responses per GPU,
        up to {64 * gpu_count} across the Pod.
        Completed problems are checkpointed separately per GPU. The coordinator merges
        the results and automatically updates **Seungjun/clean-math-sft-pool-30k**.

        Set a write-capable **HF_TOKEN** in the Pod environment, or enter it in the hidden
        prompt below. Attach storage at **/workspace**. A network volume survives Pod
        deletion; a regular Pod volume lasts only until that Pod is deleted.
        Keep enough disk space for the runtime, model/dataset cache, prepared dataset,
        journals, merged journal, and output Parquet files; free space is printed below.

        Keep the Pod and kernel running. This is not a detached job. On interruption,
        rerun the same notebook with the same RUN_NAME/settings to resume. Use a new name
        for a new experiment or changed batching settings. Keep all {gpu_count} worker journals.
        This notebook uses its own RUN_NAME; do not reuse a run from a different GPU count.
        '''),
        code(f'''
        RUN_NAME = 'qwen25-math-7b-eight-3b-012-runpod{gpu_count}-v1'
        STORAGE_ROOT = '/workspace/LG-SFT-Pool-7B'
        CODE_ROOT = '/workspace/lg-sft-pool-math7b-runpod{gpu_count}'
        EVAL_ENV = '/workspace/lg-math7b-eval-env'
        GPU_IDS = {[str(i) for i in range(gpu_count)]!r}
        TEMPERATURE = 1.0
        TOP_P = 1.0
        MAX_NEW_TOKENS = 3072
        MAX_NUM_SEQS = 64  # Per GPU.
        MAX_NUM_BATCHED_TOKENS = 8192  # Scheduling budget; not the model context length.
        MAX_PENDING_PROBLEMS = 16  # Per GPU, eight samples per problem.
        SEED = 42
        UPLOAD_TO_HF = True
        '''),
        markdown('''
        ## 1. Token, persistent storage, and isolated runtime

        CUDA 13.2 availability is not a requirement to install a CUDA 13.2 toolkit.
        This notebook uses its locked PyTorch/CUDA dependencies and tests them on the
        actual driver. It does not replace the template's global PyTorch installation.
        Initial package installation and model compilation can take several minutes.
        '''),
        code(f'''
        import base64, getpass, io, json, os, shutil, subprocess, sys, textwrap, zipfile
        from pathlib import Path
        if sys.platform != 'linux':
            raise RuntimeError('Run this notebook in your Linux Runpod Pod')
        if len(GPU_IDS) != {gpu_count} or len(set(GPU_IDS)) != {gpu_count} or any(not x.strip() for x in GPU_IDS):
            raise ValueError('Select {gpu_count} distinct GPU IDs or UUIDs')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_IDS)
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        # Runpod can inherit this legacy accelerator flag, but the isolated runtime
        # does not install hf_transfer. Use the supported default download path.
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        os.environ['HF_HOME'] = '/workspace/huggingface'
        os.environ['UV_CACHE_DIR'] = '/workspace/uv-cache'
        os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN') or getpass.getpass('Hugging Face write token: ')
        if not os.environ['HF_TOKEN'].strip():
            raise ValueError('HF_TOKEN is required')
        Path(STORAGE_ROOT).mkdir(parents=True, exist_ok=True)
        print('Free space on storage:', round(shutil.disk_usage(STORAGE_ROOT).free / 2**30, 1), 'GiB')
        Path(CODE_ROOT).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload([ROOT / p for p in files])!r}))) as bundle:
            bundle.extractall(CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged as run_setup_logged
        run_setup_logged([sys.executable, '-m', 'pip', 'install', 'uv==0.11.22', 'PyYAML==6.0.3', 'tqdm'],
                         cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'bootstrap.log')
        run_setup_logged([sys.executable, 'scripts/setup_eval_runtime.py', '--venv', EVAL_ENV],
                         cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'runtime_setup.log')
        from common.pool_progress import run_pool_logged
        run_setup_logged(['nvidia-smi'], cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'nvidia-smi.log')
        gpu_check = """
        import torch, vllm
        print('Isolated runtime:', torch.__version__, 'CUDA', torch.version.cuda, 'vLLM', vllm.__version__)
        assert torch.cuda.device_count() == {gpu_count}, 'All {gpu_count} GPUs must be visible'
        assert tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8), 'Blackwell requires CUDA 12.8+'
        for i in range({gpu_count}):
            props = torch.cuda.get_device_properties(i)
            print(i, props.name, round(props.total_memory / 2**30, 1), 'GiB', 'capability', (props.major, props.minor))
            assert props.total_memory >= 90 * 2**30, 'Expected the selected 96 GB GPUs'
            with torch.cuda.device(i):
                a = torch.ones((256, 256), dtype=torch.bfloat16, device=f'cuda:{{i}}')
                b = a @ a
                torch.cuda.synchronize()
                assert b[0,0].item() == 256
                del a, b
        print('All {gpu_count} GPUs passed BF16 computation checks')
        """
        run_setup_logged([str(Path(EVAL_ENV) / 'bin/python'), '-c', textwrap.dedent(gpu_check)],
                         cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'gpu_preflight.log')
        '''),
        markdown('''
        ## 2. Pin the source and show the exact 0/1/2 selection

        The first execution resolves the latest HF source commit; resume reuses that
        commit. No extra source filtering or reference corrections are applied.
        Source/model versions, code identity, settings, and the GPU-worker assignment
        rule are recorded. CPU grading checks run before generation.
        '''),
        code('''
        import yaml
        # Also set this here so rerunning preparation in an existing kernel fixes
        # the inherited flag without repeating installation. Children inherit it.
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        run_setup_logged([str(Path(EVAL_ENV) / 'bin/python'), '-m', 'eval.pool_grading_checks'],
                         cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'grading_checks.log')
        CONFIG = yaml.safe_load((Path(CODE_ROOT) / 'configs/sft_pool_sampling_7b.yaml').read_text())
        CONFIG['sampling'].update(temperature=TEMPERATURE, top_p=TOP_P,
                                  max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        CONFIG['engine'].update(max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS)
        CONFIG['execution']['max_pending_problems'] = MAX_PENDING_PROBLEMS
        RUN_DIR = Path(STORAGE_ROOT) / RUN_NAME
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH = RUN_DIR / 'requested_config.yaml'
        CONFIG_PATH.write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        COMMAND = [str(Path(EVAL_ENV) / 'bin/python'), '-m', 'pipeline.sample_sft_pool_7b_runpod',
                   '--config', str(CONFIG_PATH), '--run-dir', str(RUN_DIR), '--gpus', ','.join(GPU_IDS)]
        run_setup_logged(COMMAND + ['--mode', 'prepare'], cwd=CODE_ROOT,
                         log_path=RUN_DIR / 'prepare_console.log')
        manifest = json.loads((RUN_DIR / 'manifest.json').read_text())
        print('Original rows:', manifest['source_rows'])
        print('Selected 3B 0/1/2 rows:', len(manifest['selected_indices']))
        print('New 7B responses:', len(manifest['selected_indices']) * 8)
        print('Selected rows per GPU:', [len(manifest['selected_indices'][r::len(GPU_IDS)]) for r in range(len(GPU_IDS))])
        print('Saved under:', RUN_DIR)
        '''),
        markdown(f'''
        ## 3. Generate on all {gpu_count} GPUs, grade, and merge

        Each GPU uses the same pinned instruct model and native chat template.
        Generation stops on `<|im_end|>` (151645) or `<|endoftext|>` (151643).
        The model's native context is 4,096 tokens, including the prompt. Each response
        gets at most 3,072 new tokens, reduced to the remaining space for long prompts.
        Input is never silently truncated. A token-limit finish is recorded as `length`.

        The shared cache downloads the model once. {gpu_count} workers then run independently;
        per-GPU progress is updated every 30 seconds. Worker logs are in
        `workers/<rank>/console.log`; their resumable journals are `responses_7b.jsonl`.
        If a worker fails, the coordinator stops the others. Completed problems remain
        saved. Rerun this cell to resume after resolving the cause.

        The merge requires every selected row exactly once, in its assigned worker,
        with matching runtime/prompt records. The final dataset keeps every original
        row in original order. Rows with 3B scores 3–8 have null in all six new columns.
        '''),
        code('''
        run_pool_logged(COMMAND + ['--mode', 'full'], cwd=CODE_ROOT,
                        log_path=RUN_DIR / 'full_console.log', title=f'Math-7B · {len(GPU_IDS)} GPUs')
        summary = json.loads((RUN_DIR / 'full/summary.json').read_text())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('Full merged dataset:', RUN_DIR / 'full/data')
        '''),
        markdown('''
        ## 4. Update Hugging Face

        With the default `UPLOAD_TO_HF=True`, the completed dataset is published to
        **Seungjun/clean-math-sft-pool-30k**. All original values are validated before
        upload. Existing source files remain stored; the default train split points to
        the new annotated Parquet files. If HF changed during generation, the notebook
        refuses to overwrite the newer version. Local completed results remain available.
        '''),
        code('''
        if UPLOAD_TO_HF:
            run_setup_logged(COMMAND + ['--mode', 'upload'], cwd=CODE_ROOT,
                             log_path=RUN_DIR / 'upload_console.log')
            receipt = json.loads((RUN_DIR / 'upload_receipt.json').read_text())
            print('Verified HF update:', receipt['url'])
        else:
            print('Results saved:', RUN_DIR / 'full/data')
        '''),
        markdown('''
        References: [Runpod storage](https://docs.runpod.io/pods/storage/types),
        [vLLM GPU installation](https://docs.vllm.ai/en/v0.14.1/getting_started/installation/gpu/),
        [Qwen model](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct).

        The actual Pod GPU execution is checked by the setup and generation cells.
        This notebook does not stop or delete the Pod when finished.
        '''),
    ]


if __name__ == '__main__':
    for count in (3, 4):
        write_notebook(f'sample_sft_pool_math7b_8_runpod_{count}gpu.ipynb', cells(count))
