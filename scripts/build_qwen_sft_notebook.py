"""Build the self-contained Qwen BASE + s1 Kimi-style SFT notebook."""

import sys

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

sys.path.insert(0, str(ROOT))
from train.qwen_sft_data import BUNDLE_FILES  # noqa: E402


def cells():
    boot = bootstrap(payload([ROOT / name for name in BUNDLE_FILES]))
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-qwen-kimi-sft")
    return [
        markdown(r"""
        # Qwen2.5-3B BASE + s1 Kimi-style SFT — 5 epochs

        Roadmap **B-SFT2**: full-parameter SFT from **`Qwen/Qwen2.5-3B` base**,
        revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.
        Dataset: [Seungjun/dp_removed_s1K-1.1](https://huggingface.co/datasets/Seungjun/dp_removed_s1K-1.1),
        revision `636ecf409774771afb0bf10a436f4b1e608b5f29`, `default` / `train`.
        Input: **`question`**. Assistant target: the **entire `kimi-style-reasoning-answer`**.

        Planning, Solution and Evaluation, Reflection, Exploration, and Final answer are
        preserved exactly as stored. No `<think>` or `<answer>` tags are added; raw API
        reasoning and original s1 answers are excluded. Prompts use the same English
        step-by-step / boxed-answer instruction as evaluation and the base model's native
        chat template/default system message. Loss covers only the answer and `<|im_end|>`;
        system/user text and padding are masked. No mathematical correctness filter is applied.

        **5 epochs**, batch size **1**, gradient accumulation **16**, LR **1e-5**, cosine schedule,
        warmup **5%**, seed **42**, BF16, gradient checkpointing. No LoRA or quantization.
        Full-sequence cap **20,480 tokens**; overlength rows are reported and excluded, never truncated.
        Local preparation verified **996 source rows → 989 retained**, 7 empty targets,
        0 overlength rows; **62 optimizer updates/epoch, 310 total**.

        Use a **fresh RTX PRO 6000 Blackwell 96GB Colab runtime**, enable `HF_TOKEN` in Secrets,
        and reserve about **120 GB Drive space** for five full model/optimizer checkpoints.
        Code and configs are embedded; no GitHub checkout is required. Run setup/preparation,
        inspect the loss preview, optionally run smoke, then set **`RUN_TRAINING = True`**.
        """),
        code('''
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short"
        RUN_NAME = "sft_qwen25_3b_base_s1_kimi"
        TRAIN_ENV = "/content/lg-qwen-kimi-sft-env"
        TEMPLATE_CONFIG = "configs/sft_qwen25_3b_s1_kimi.yaml"
        '''),
        code('''
        import os, subprocess, sys, json
        from pathlib import Path
        from google.colab import drive, userdata

        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        if not os.environ["HF_TOKEN"]:
            raise ValueError("Enable HF_TOKEN access in Colab Secrets")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        '''),
        boot,
        markdown('''
        ## Install the locked training environment

        Separate Python 3.12 environment: PyTorch 2.9.1, Transformers 4.57.6, TRL 0.24.0.
        Native PyTorch SDPA FlashAttention; no external flash-attn build. The vocabulary
        projection is checkpointed in chunks to limit memory use on long answers.
        '''),
        code('''
        from common.process import run_logged

        run_logged([sys.executable, str(CODE_ROOT / "scripts/setup_stage1_runtime.py"),
                    "--venv", TRAIN_ENV], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / "setup_console.log")
        TRAIN_PYTHON = str(Path(TRAIN_ENV) / "bin/python")
        '''),
        markdown('''
        ## Freeze the dataset and inspect the exact loss region

        Download the pinned HF revision once; save only the question and target columns to
        `TRAIN_ROOT/inputs/RUN_NAME/dataset.jsonl`. Reruns reuse this snapshot and validate
        its digest. Config, source revision, column names, retained IDs, exclusions, token
        lengths, and update counts are recorded. Changed data/settings require a new run name.

        The base tokenizer's original EOS is `<|endoftext|>`, while its native chat template
        ends assistant turns with `<|im_end|>`. This run uses the existing `<|im_end|>` as
        supervised EOS and generation stop, with `<|endoftext|>` as padding. No tokens are
        added and the shipped template remains unchanged. Both settings are saved with checkpoints.
        '''),
        code('''
        run_logged([TRAIN_PYTHON, "-m", "train.qwen_sft_data",
                    "--config", TEMPLATE_CONFIG, "--output-root", TRAIN_ROOT, "--run-name", RUN_NAME],
                   cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / RUN_NAME / "snapshot_console.log")
        CONFIG_PATH = Path(TRAIN_ROOT) / "inputs" / RUN_NAME / "training.yaml"
        TRAIN_COMMAND = [TRAIN_PYTHON, "-m", "train.stage1_qwen_sft", "--config", str(CONFIG_PATH)]
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
            run_logged(TRAIN_COMMAND + ["--smoke", "--auto-resume"], cwd=CODE_ROOT,
                       log_path=Path(TRAIN_ROOT) / (RUN_NAME + "_smoke") / "train_console.log",
                       compact_progress=True)
        else:
            print("Set RUN_SMOKE = True for the optional longest-example GPU smoke.")
        '''),
        markdown('''
        ## Train for five total epochs

        Enable `RUN_TRAINING` below. Reruns automatically resume the latest completed
        `epoch_N`, restoring optimizer, scheduler, RNG, and Trainer state. Training totals
        five epochs including completed epochs. Work after the latest epoch save is replayed;
        abandoned step logs are archived. A completed epoch-5 run is detected and skipped.

        After disconnecting, use the same notebook, rerun setup/preparation, and enable this
        cell again. Resume requires matching data, code, settings, and runtime. If a run fails
        before saving its first complete epoch, there is no resumable checkpoint; use a new
        `RUN_NAME` to restart and preserve its failure logs.
        '''),
        code('''
        RUN_TRAINING = False
        if RUN_TRAINING:
            run_logged(TRAIN_COMMAND + ["--auto-resume"], cwd=CODE_ROOT,
                       log_path=Path(TRAIN_ROOT) / RUN_NAME / "train_console.log",
                       compact_progress=True)
        else:
            print("Preparation complete. Set RUN_TRAINING = True for five-epoch Qwen base SFT.")
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

        No evaluation or Hugging Face upload is performed by this notebook.
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
    write_notebook("train_qwen25_3b_base_s1_kimi.ipynb", cells())
