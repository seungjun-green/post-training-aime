"""Build the self-contained Qwen BASE + rejection sampling SFT notebook."""

import sys

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

sys.path.insert(0, str(ROOT))
from train.qwen_self_rft_data import BUNDLE_FILES  # noqa: E402


def cells():
    boot = bootstrap(payload([ROOT / name for name in BUNDLE_FILES]))
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-qwen-self-rft-sft")
    return [
        markdown(r"""
        # Qwen2.5-3B BASE + self-RFT SFT — 5 epochs

        Roadmap **B-rejection sampling SFT**: full-parameter SFT from **`Qwen/Qwen2.5-3B` base**,
        revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.
        Dataset: [Seungjun/qwen2.5-3b-self-rft-math](https://huggingface.co/datasets/Seungjun/qwen2.5-3b-self-rft-math),
        revision `96c5f70e37098a9f41f5e44adc4e906ee3f5ef15`, `default` / `train`.
        Input: **`problem`**. Assistant target: the **entire `response`**.

        All **28,267 rows** are used, including multiple retained solutions for the same
        problem (**12,863 unique problem IDs**). No deduplication, new correctness filter,
        or reasoning tags. The response text is preserved exactly. Prompts use the same
        English step-by-step / boxed-answer instruction as evaluation and the base model's
        shipped chat template/default system message. The target is `response` followed
        by the original **`<|endoftext|>` EOS**, with no appended assistant `<|im_end|>`.
        Loss covers only the response and EOS; system/user text and padding are masked.
        The tokenizer vocabulary, chat template, and native EOS/PAD remain unchanged.

        After **each epoch**, save the checkpoint and evaluate **AMC 2023 (40 problems)**
        and **MATH-500 (500)** at **temperature 0**, one response per problem, with the
        existing 20,480-token cap and scorer. Print accuracy and mean response tokens,
        then resume training. This adds **2,700 evaluation responses** across five epochs.
        All epochs are evaluated; scores do not select checkpoints or change training.

        **5 epochs**, batch size **1**, gradient accumulation **16**, LR **1e-5**, cosine schedule,
        warmup **5%**, seed **42**, BF16, gradient checkpointing. No LoRA or quantization.
        Full-sequence cap **20,480 tokens**; overlength rows are reported and excluded, never truncated.
        Local preparation verified all **28,267 rows retained**, with no missing or overlength
        targets; full-sequence maximum **2,839 tokens**. With all rows: **1,767 optimizer updates/epoch, 8,835 total**
        (batch 1 × accumulation 16, including the last partial accumulation).

        Use a **fresh RTX PRO 6000 Blackwell 96GB Colab runtime**, enable `HF_TOKEN` in Secrets,
        and reserve about **120 GB Drive space** for five full model/optimizer checkpoints.
        Code and configs are embedded; no GitHub checkout is required. Run setup/preparation,
        inspect the loss preview, optionally run smoke, then set **`RUN_TRAINING = True`**.
        """),
        code('''
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rejection-sampling-sft"
        RUN_NAME = "sft_qwen25_3b_base_self_rft"
        TRAIN_ENV = "/content/lg-qwen-self-rft-sft-env"
        EVAL_ENV = "/content/lg-qwen-self-rft-eval-env"
        TEMPLATE_CONFIG = "configs/sft_qwen25_3b_self_rft.yaml"
        '''),
        code('''
        import os, subprocess, sys, json
        from pathlib import Path
        from google.colab import drive, userdata

        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        if not os.environ["HF_TOKEN"]:
            raise ValueError("Enable HF_TOKEN access in Colab Secrets")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "uv==0.11.22", "PyYAML==6.0.3"])
        '''),
        boot,
        markdown('''
        ## Install the locked training and evaluation environments

        Separate Python 3.12 environment: PyTorch 2.9.1, Transformers 4.57.6, TRL 0.24.0.
        Native PyTorch SDPA FlashAttention; no external flash-attn build. The vocabulary
        projection is checkpointed in chunks to limit memory use on long answers.
        Evaluation uses a separate locked vLLM environment. Training exits after saving
        each epoch, so training and evaluation never occupy the GPU simultaneously.
        '''),
        code('''
        from common.process import run_logged

        run_logged([sys.executable, str(CODE_ROOT / "scripts/setup_stage1_runtime.py"),
                    "--venv", TRAIN_ENV], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / "setup_console.log")
        TRAIN_PYTHON = str(Path(TRAIN_ENV) / "bin/python")
        run_logged([sys.executable, str(CODE_ROOT / "scripts/setup_eval_runtime.py"),
                    "--venv", EVAL_ENV], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / "setup_eval_console.log")
        EVAL_PYTHON = str(Path(EVAL_ENV) / "bin/python")
        '''),
        markdown('''
        ## Freeze the dataset and inspect the exact loss region

        Download the pinned HF revision once; save only the question and target columns to
        `TRAIN_ROOT/inputs/RUN_NAME/dataset.jsonl`. Reruns reuse this snapshot and validate
        its digest. Config, source revision, column names, retained IDs, exclusions, token
        lengths, and update counts are recorded. Changed data/settings require a new run name.

        The original EOS and padding are both `<|endoftext|>`. The shipped chat template
        formats the prompt through the assistant header; the full response and original EOS
        are appended as a completion. The template itself is never rewritten, and EOS is
        never changed to `<|im_end|>`. Padding labels are masked by position so the actual
        EOS still receives loss. Saved checkpoints retain these settings.
        '''),
        code('''
        run_logged([TRAIN_PYTHON, "-m", "train.qwen_self_rft_data",
                    "--config", TEMPLATE_CONFIG, "--output-root", TRAIN_ROOT, "--run-name", RUN_NAME],
                   cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / RUN_NAME / "snapshot_console.log")
        CONFIG_PATH = Path(TRAIN_ROOT) / "inputs" / RUN_NAME / "training.yaml"
        TRAIN_COMMAND = [TRAIN_PYTHON, "-m", "train.stage1_qwen_self_rft", "--config", str(CONFIG_PATH)]
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / RUN_NAME / "prepare_console.log")

        log_dir = Path(TRAIN_ROOT) / "logs/stage1" / RUN_NAME
        checkpoint_dir = Path(TRAIN_ROOT) / "checkpoints/stage1" / RUN_NAME
        report = json.loads((log_dir / "preparation/data_report.json").read_text())
        print("Kept:", report["kept_rows"], "Missing:", report["dropped_missing_outputs"],
              "Overlength:", report["dropped_overlength"])
        print("Full-sequence tokens:", report["token_lengths"])
        print("Target tokens including EOT:", report["supervised_tokens_including_eot"])
        print("Optimizer updates:", report["optimizer_steps_per_epoch"], "per epoch;",
              report["total_optimizer_steps"], "total")
        print("Loss preview:", log_dir / "preparation/sanity_check.txt")
        '''),
        markdown('''
        ## Optional GPU smoke — one update on the longest retained example

        Uses a separate `RUN_NAME_smoke` checkpoint/log directory, the same base weights,
        and accumulation 1. Confirms forward/backward and checkpoint save on this GPU.
        Full training starts from the original base model, independent of smoke.
        '''),
        code('''
        RUN_SMOKE = False
        if RUN_SMOKE:
            from train.self_rft_display import run_progress
            run_progress(TRAIN_COMMAND + ["--smoke", "--auto-resume"], cwd=CODE_ROOT,
                       log_path=Path(TRAIN_ROOT) / (RUN_NAME + "_smoke") / "train_console.log",
                       description="SFT smoke", total=1, mode="training")
        else:
            print("Set RUN_SMOKE = True for the optional longest-example GPU smoke.")
        '''),
        markdown('''
        ## Train and evaluate after each of five epochs

        Enable `RUN_TRAINING` below. Reruns automatically resume the latest completed
        `epoch_N`, restoring optimizer, scheduler, RNG, and Trainer state. Training totals
        five epochs including completed epochs, using the same full five-epoch LR schedule.
        Work after the latest epoch save is replayed; abandoned step logs are archived.
        Saved epochs are not retrained. Evaluation resumes saved problems, including an
        interrupted epoch-5 evaluation, before training advances to the next epoch.

        The display shows one tqdm bar per stage, with training loss/LR in the postfix,
        followed by that epoch's two-row benchmark table. Full diagnostics remain in the
        Drive console logs; loss history remains in `steps.jsonl`.

        After disconnecting, use the same notebook, rerun setup/preparation, and enable this
        cell again. Resume requires matching data, code, settings, and runtime. If a run fails
        before saving its first complete epoch, there is no resumable checkpoint; use a new
        `RUN_NAME` to restart and preserve its failure logs.
        '''),
        code('''
        RUN_TRAINING = False
        if RUN_TRAINING:
            from train.self_rft_epochs import run_epochs
            epoch_results = run_epochs(CONFIG_PATH, code_root=CODE_ROOT,
                                       train_python=TRAIN_PYTHON, eval_python=EVAL_PYTHON)
        else:
            print("Set RUN_TRAINING = True for five epochs, each followed by AMC/MATH evaluation.")
        '''),
        markdown('''
        ## Output locations

        All paths are under `TRAIN_ROOT`:
        - `inputs/RUN_NAME/`: immutable two-column dataset snapshot and resolved `training.yaml`.
        - `checkpoints/stage1/RUN_NAME/epoch_1` through `epoch_5`: full HF model/tokenizer,
          optimizer, scheduler, RNG, Trainer state, and completion marker.
        - `logs/stage1/RUN_NAME/steps.jsonl`: loss, learning rate, gradient norm, speed, step, epoch.
        - `logs/stage1/RUN_NAME/`: manifest, data report, loss preview, metrics, training summary,
          and attempt history.
        - `RUN_NAME/train_console.log`: streamed training console output.
        - `eval/amc2023_math500_temp0/RUN_NAME/epoch_N/`: each epoch's metrics,
          generated answers, per-problem resume journal, and evaluation console log.
        - `eval/amc2023_math500_temp0/RUN_NAME/epoch_summary.json` and `.csv`:
          combined epoch-by-epoch results (ten benchmark rows when complete).

        No Hugging Face upload is performed by this notebook.
        '''),
        code('''
        print("Final model:", checkpoint_dir / "epoch_5")
        print("Loss history:", log_dir / "steps.jsonl")
        print("Run manifest:", log_dir / "run_manifest.json")
        summary = log_dir / "training_summary.json"
        if summary.exists():
            print(json.dumps(json.loads(summary.read_text()), indent=2))
        else:
            print("Training has not run yet.")
        '''),
    ]


if __name__ == "__main__":
    write_notebook("train_qwen25_3b_base_self_rft.ipynb", cells())
