"""Self-contained RunPod transfer evaluation: two- and four-GPU variants."""

import json

from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_qwen_transfer_eval_notebook import cells as colab_cells

from eval.run_transfer_eval import bundle_files

NAME = 'evaluate_qwen25_3b_all9_transfer_runpod_2gpu.ipynb'
FOUR_GPU_NAME = 'evaluate_qwen25_3b_all9_transfer_runpod_4gpu.ipynb'
EXTRA_FILES = ['eval/run_transfer_runpod.py', 'eval/runpod_transfer_storage.py',
               'eval/export_transfer_results.py']


def cells(gpu_count=2, storage_root=None):
    if gpu_count not in (2, 4):
        raise ValueError('Build the two- or four-GPU notebook')
    storage_root = storage_root or ('/root/qwen-transfer' if gpu_count == 2 else '/workspace/qwen-transfer')
    storage_note = (
        '**Storage: this variant targets your 200 GB container disk**, using '
        '`/root/qwen-transfer` so a separate, smaller `/workspace` volume does not receive the downloads. '
        '**Container files are lost when the Pod stops/restarts.** Finish the Hugging Face export '
        '(or copy results elsewhere) before stopping the Pod. To use a persistent volume instead, '
        'set `STORAGE_ROOT` to a directory on that volume.' if gpu_count == 2 else
        '**Storage: use about 200 GB of persistent storage at `/workspace`.** '
        'A network volume survives Pod deletion; a Pod volume survives stops but is deleted with the Pod.')
    original = colab_cells()
    display_code = next(c.source for c in original if 'def show_results():' in c.source)
    display_code = display_code[display_code.index('def show_results():'):]
    encoded = payload([ROOT / name for name in bundle_files() + EXTRA_FILES])
    return [
        markdown(f'''
        # All nine Qwen2.5-3B variants — RunPod · {gpu_count} × RTX PRO 6000 96 GB

        **ARC-Challenge, MMLU-Pro subject breakdowns, non-math BBH, IFEval**.
        Temperature **0**, one response, **4,096 new-token cap**. Same datasets,
        prompts, scoring and selected checkpoints as the Colab transfer notebook.
        **18,360 prompts/model; 165,240 across nine models**.

        A dynamic queue runs **one complete model evaluation per GPU**, up to {gpu_count}
        models at once. Every worker uses tensor parallel size 1; a free GPU takes the
        next model. Each process exits before its GPU is reused. Per-response resume,
        native saved EOS/chat templates, and all subject/task breakdowns are preserved.

        {storage_note}
        Models, downloads, caches, environment, temporary files and results live under
        `{storage_root}`. The notebook checks actual free space on this filesystem before
        downloads. See [RunPod storage](https://docs.runpod.io/pods/storage/types).

        The **RunPod PyTorch 2.8.0 template** provides Jupyter/Linux. This notebook installs
        the existing evaluation lock in a separate Python 3.12 environment: **PyTorch
        2.9.1, vLLM 0.14.1**, with its own CUDA runtime packages. It does not replace the
        template's PyTorch. The advertised CUDA 13.2 driver capability is not a request
        to install CUDA 13.2 wheels; setup validates actual CUDA execution on every selected GPU.

        Code and official scoring resources are embedded. No GitHub checkout or training.
        No Pod is created, stopped or terminated automatically.
        The final cell exports completed inference records to Parquet, score tables to CSV,
        and publishes them with a dataset README to Hugging Face. Set a write-enabled
        `HF_TOKEN` at setup; repository ID, visibility and upload switch are editable below.
        '''),
        original[1],
        code(f'''
        from pathlib import Path
        STORAGE_ROOT = Path({storage_root!r})
        EXPERIMENT_ROOT = str(STORAGE_ROOT / "models/LG-AIME-Qwen25-3B-Experiments")
        INSTRUCT_RL_ROOT = str(STORAGE_ROOT / "models/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix")
        OUTPUT_ROOT = str(STORAGE_ROOT / "results/eval-all9-transfer-temp0")
        EVAL_ENV = str(STORAGE_ROOT / "envs/eval")
        GPU_IDS = {json.dumps([str(i) for i in range(gpu_count)])}
        MODEL_KEYS = None  # Default: all nine. Or e.g. ["base", "instruct"].
        CONFIG = "configs/qwen_transfer_eval.yaml"
        RCLONE_REMOTE = "gdrive:"  # Named remote rooted at My Drive.
        RCLONE_CONFIG = str(STORAGE_ROOT / "secrets/rclone.conf")
        HF_REPO_ID = "Seungjun/qwen25-3b-transfer-eval-results"
        HF_PRIVATE = True
        UPLOAD_TO_HF = True  # Final cell uploads only after all selected full runs finish.
        '''),
        code('''
        import os, sys, subprocess, json, shutil, getpass
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        # Keep all large package/model downloads on the selected 200 GB filesystem.
        for variable, directory in {
            "HF_HOME": "cache/huggingface", "UV_CACHE_DIR": "cache/uv",
            "UV_PYTHON_INSTALL_DIR": "cache/python", "PIP_CACHE_DIR": "cache/pip",
            "XDG_CACHE_HOME": "cache/xdg", "VLLM_CACHE_ROOT": "cache/vllm", "TMPDIR": "tmp",
        }.items():
            path = STORAGE_ROOT / directory
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)
        os.environ["RCLONE_CONFIG"] = RCLONE_CONFIG
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if not os.environ.get("HF_TOKEN"):
            token = getpass.getpass("HF token (hidden; write access required for final HF upload): ").strip()
            if token:
                os.environ["HF_TOKEN"] = token
            del token
        if UPLOAD_TO_HF and not os.environ.get("HF_TOKEN"):
            raise ValueError("Supply a write-enabled HF_TOKEN, or set UPLOAD_TO_HF = False for local export only.")
        # Small bootstrap only; torch/vLLM are installed into EVAL_ENV later.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22", "PyYAML==6.0.3"])
        print("Workspace:", STORAGE_ROOT)
        '''),
        code(f'''
        import base64, io, zipfile
        CODE_ROOT = STORAGE_ROOT / "code"
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        BUNDLE = {encoded!r}
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
            archive.extractall(CODE_ROOT)
        sys.path.insert(0, str(CODE_ROOT))
        from common.process import run_logged
        from eval.run_transfer_eval import load_config
        from eval.runpod_transfer_storage import check_space, gpu_inventory, drive_copy_plan, copy_checkpoints
        config = load_config(CODE_ROOT / CONFIG)
        print("GPUs:", gpu_inventory(GPU_IDS))
        print("Storage:", check_space(STORAGE_ROOT))
        '''),
        markdown('''
        ## Bring the seven selected checkpoints from Drive

        Two pretrained models download directly from their pinned Hugging Face revisions.
        The seven trained checkpoints must be copied from your Drive (or uploaded manually).
        The next cells generate the exact copy plan, including each required training
        `run_manifest.json`. They exclude optimizer, scheduler, RNG files and earlier epochs.

        **One-time Drive authorization:** the setup cell installs `rclone` from Ubuntu's
        signed package repository if needed. In a **Jupyter terminal**, run the command
        it prints, create a remote named `gdrive`, choose Google Drive and read-only scope.
        On a headless Pod, choose browser authentication on another machine using the
        [rclone remote-setup instructions](https://rclone.org/remote_setup/).
        Store OAuth credentials only in `RCLONE_CONFIG`; do not paste them into notebook cells.
        An existing authorized rclone config can also be uploaded to that path.

        If checkpoints are already copied into the displayed local paths, leave
        `COPY_FROM_DRIVE = False`. Otherwise finish authorization, set it to `True`, and
        run the copy cell. Copies are one-way and do not alter/delete Drive originals.
        '''),
        code('''
        plan = drive_copy_plan(config, STORAGE_ROOT, RCLONE_REMOTE, MODEL_KEYS)
        for item in plan:
            print(item["key"], "=>", item["destination"])
            print("  manifest:", item["manifest_destination"])
        # Optional utility; skip this cell if rclone is installed or files were uploaded manually.
        if shutil.which("rclone") is None:
            subprocess.check_call(["apt-get", "update", "-qq"])
            subprocess.check_call(["apt-get", "install", "-y", "--no-install-recommends", "rclone"])
        Path(RCLONE_CONFIG).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        import shlex
        print("Authorize in a Jupyter terminal:")
        print("rclone config --config " + shlex.quote(RCLONE_CONFIG))
        '''),
        code('''
        COPY_FROM_DRIVE = False
        if COPY_FROM_DRIVE:
            if not Path(RCLONE_CONFIG).is_file():
                raise FileNotFoundError("Configure the gdrive remote first: " + RCLONE_CONFIG)
            Path(RCLONE_CONFIG).chmod(0o600)
            copy_checkpoints(config, STORAGE_ROOT, RCLONE_REMOTE, selected=MODEL_KEYS)
        else:
            print("Drive copy disabled. The next cell validates the required local checkpoint files.")
        '''),
        markdown('''
        ## Validate local checkpoints and install the isolated evaluation runtime

        Checks all selected checkpoint identities, epochs/steps, full weight shards,
        EOS and saved templates. A missing file reports its path before GPU inference.
        Installation uses persistent caches; first setup may take several minutes.
        '''),
        code('''
        from eval.transfer_models import prepare_models
        runs = prepare_models(config, EXPERIMENT_ROOT, INSTRUCT_RL_ROOT, OUTPUT_ROOT, MODEL_KEYS)
        for run in runs:
            print(run["key"], "=>", run["model"], run["revision"] or "saved checkpoint")
        run_logged([sys.executable, str(CODE_ROOT / "scripts/setup_transfer_runtime.py"), "--venv", EVAL_ENV],
                   cwd=CODE_ROOT, log_path=Path(OUTPUT_ROOT) / "logs/setup_console.log")
        # Verify every physical GPU in its own process, without changing the kernel environment.
        for gpu in GPU_IDS:
            subprocess.check_call([str(Path(EVAL_ENV) / "bin/python"), "-c",
                "import torch, vllm; "
                "assert torch.cuda.device_count()==1; "
                "assert torch.__version__.startswith('2.9.1'); "
                "assert vllm.__version__=='0.14.1'; "
                "x=torch.ones((64,64),device='cuda',dtype=torch.bfloat16); "
                "assert (x@x)[0,0].item()==64; "
                "print('GPU ready:',torch.cuda.get_device_name(0),'torch',torch.__version__,'CUDA',torch.version.cuda)"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, cwd=CODE_ROOT)

        def parallel_command(*extra):
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_transfer_runpod",
                       "--config", CONFIG, "--storage-root", str(STORAGE_ROOT), "--output-root", OUTPUT_ROOT,
                       "--experiment-root", EXPERIMENT_ROOT, "--instruct-rl-root", INSTRUCT_RL_ROOT,
                       "--gpus", ",".join(GPU_IDS)]
            if MODEL_KEYS is not None:
                command += ["--models", ",".join(MODEL_KEYS)]
            return command + list(extra)
        '''),
        markdown('''
        ## Prepare benchmark data and prefetch pretrained model weights

        Runs once before parallel workers start. Fixed source hashes/counts are checked;
        cached data and pretrained weights are reused. Repeated setup is safe with the
        same code, protocol and checkpoint paths. RunPod outputs use a separate root
        from Colab outputs; do not copy Colab resume journals into this folder.
        '''),
        code('''
        run_logged(parallel_command("--prepare-only"), cwd=CODE_ROOT,
                   log_path=Path(OUTPUT_ROOT) / "logs/prepare_console.log")
        data_manifest = json.loads((Path(OUTPUT_ROOT) / "data/manifest.json").read_text())
        prompts_per_model = sum(data_manifest["counts"].values())
        print("Prompts:", prompts_per_model, "per model;", prompts_per_model * len(runs), "selected total")
        '''),
        code(display_code),
        markdown('''
        ## Optional parallel smoke — 35 prompts/model

        Exercises model loading, all four benchmark types, native templates/EOS, and scoring.
        Smoke is separate from full results. Worker logs are under `models/<key>/logs/`.
        '''),
        code('''
        RUN_SMOKE = False
        if RUN_SMOKE:
            run_logged(parallel_command("--smoke"), cwd=CODE_ROOT,
                       log_path=Path(OUTPUT_ROOT) / "logs/runpod_smoke_console.log",
                       progress_totals={"transfer": 35 * len(runs)}, compact_progress=True)
        else:
            print("Set RUN_SMOKE = True for the optional parallel smoke run.")
        '''),
        markdown('''
        ## Run/resume full evaluation on the selected GPUs

        Set `RUN_EVAL = True`. The next model starts as soon as a GPU is free.
        Progress shows aggregate completed prompts and each active GPU/model/benchmark.
        The coordinator alone writes comparison tables after every completed model.
        Model outputs are isolated, so parallel workers never write the same journal.

        On error/interruption, all launched worker process groups are stopped and saved
        responses remain. Rerun this cell to resume. Locks prevent duplicate coordinators
        or orphaned workers from writing the same output. After a browser disconnect,
        check the running kernel/status before starting another copy. The Pod continues
        to be billed while running; this notebook does not automatically stop it.
        '''),
        code('''
        RUN_EVAL = False
        if RUN_EVAL:
            run_logged(parallel_command(), cwd=CODE_ROOT,
                       log_path=Path(OUTPUT_ROOT) / "logs/runpod_eval_console.log",
                       progress_totals={"transfer": prompts_per_model * len(runs)}, compact_progress=True)
            show_results()
        else:
            print("Set RUN_EVAL = True to start/resume the GPU queue.")
        '''),
        markdown(f'''
        ## Read saved results and export

        Results: `{storage_root}/results/eval-all9-transfer-temp0/`.
        `comparison.csv`/`.json`, `subject_task_breakdowns.csv`, and per-model
        `full/results.json`, `full/responses.jsonl`, and benchmark `generations.jsonl`
        follow the Colab layout. `runpod_status.json` shows queue/GPU progress;
        `models/<key>/logs/runpod_eval_console.log` contains detailed worker diagnostics.
        `logs/runpod_eval_console.log` contains coordinator output.

        The next cell only reads results. After full evaluation is finished or stopped,
        optionally copy results back to a **new** Drive folder. Copying never deletes
        Drive files, and the source contains only evaluation artifacts, not model weights
        or OAuth credentials. A read-only Drive remote must be reauthorized with write
        access if you choose to enable backup. Alternatively download artifacts through
        Jupyter or use your persistent network volume. Copy container-disk results out
        before stopping/restarting the Pod; preserve Pod-volume results before deleting it.
        '''),
        code('''
        show_results()
        status_path = Path(OUTPUT_ROOT) / "runpod_status.json"
        if status_path.is_file():
            print(json.dumps(json.loads(status_path.read_text()), indent=2))
        '''),
        code('''
        BACKUP_TO_DRIVE = False
        DRIVE_RESULT_FOLDER = RCLONE_REMOTE + "LG-AIME-Qwen25-3B-Experiments/eval-all9-transfer-runpod-temp0"
        if BACKUP_TO_DRIVE:
            subprocess.check_call(["rclone", "copy", OUTPUT_ROOT, DRIVE_RESULT_FOLDER,
                                   "--exclude", "*.lock", "--exclude", "*.tmp", "--stats", "30s"])
            print("Results copied to:", DRIVE_RESULT_FOLDER)
        else:
            print("Optional Drive backup disabled. Local results:", OUTPUT_ROOT)
        '''),
        markdown('''
        ## Publish inference outputs and result tables to Hugging Face

        **Automatic at the end when `UPLOAD_TO_HF = True`** (default), after all selected
        full evaluations complete. Default destination:
        `Seungjun/qwen25-3b-transfer-eval-results` (**private**). Change `HF_REPO_ID` and
        `HF_PRIVATE` in the settings cell. An existing repository must have matching
        visibility and be empty or an earlier export from this notebook. Visibility
        is never changed automatically.

        - **Parquet inference outputs:** one configuration per model; all four benchmarks
          in its `test` split. Includes prompts, full responses, gold answers, grading
          details, subject/task, token counts and finish reasons.
        - **CSV tables:** overall comparison plus every subject/task/IFEval metric.
        - **README dataset card:** rendered overall, ARC, MMLU-Pro subject, BBH task,
          and IFEval tables, checkpoint list, evaluation settings, source attribution,
          and a `load_dataset` example.
        - **Provenance:** configuration, data hashes, model/runtime manifests and checksums.

        The exporter checks record completeness and table consistency before uploading.
        Smoke and incomplete runs are excluded. Earlier snapshots remain under `runs/`;
        the README points to the latest export. Only the prepared export files are sent,
        with no model weights, credentials or console logs. A write-enabled HF token
        must have access to the destination namespace/repository.

        If upload fails, the local export remains under `OUTPUT_ROOT/hf_exports/`.
        Correct the token/settings and rerun **only this cell**; inference is not repeated.
        With `UPLOAD_TO_HF = False`, the same files and README are still built locally.
        [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards)
        describe how the README and dataset configurations appear on the Hub.
        '''),
        code('''
        selected_keys = MODEL_KEYS if MODEL_KEYS is not None else list(config["models"])
        incomplete = [key for key in selected_keys
                      if not (Path(OUTPUT_ROOT) / "models" / key / "full/results.json").is_file()]
        if incomplete:
            print("HF export waits for completed full evaluations:", ", ".join(incomplete))
        else:
            export_command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.export_transfer_results",
                              "--config", CONFIG, "--output-root", OUTPUT_ROOT, "--repo-id", HF_REPO_ID,
                              "--models", ",".join(selected_keys)]
            if UPLOAD_TO_HF:
                if not os.environ.get("HF_TOKEN"):
                    raise ValueError("Set a write-enabled HF_TOKEN, then rerun this cell. Inference is already saved.")
                export_command.append("--upload")
            if not HF_PRIVATE:
                export_command.append("--public")
            run_logged(export_command, cwd=CODE_ROOT,
                       log_path=Path(OUTPUT_ROOT) / "logs/hf_export_console.log")
            if UPLOAD_TO_HF:
                from IPython.display import Markdown, display
                display(Markdown("[Open Hugging Face results and README](https://huggingface.co/datasets/" + HF_REPO_ID + ")"))
        '''),
    ]


if __name__ == '__main__':
    write_notebook(NAME, cells())
    write_notebook(FOUR_GPU_NAME, cells(4))
