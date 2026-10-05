"""Build the self-contained EXAONE answer-only SFT notebook."""

import sys

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

sys.path.insert(0, str(ROOT))
from train.answer_only import BUNDLE_FILES  # noqa: E402


def cells():
    boot = bootstrap(payload([ROOT / p for p in BUNDLE_FILES]))
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-answer-only-sft")
    return [
        markdown(r"""
        # EXAONE-3.5-2.4B-Instruct — answer-only SFT, 5 epochs

        Full-parameter fine-tuning starts from **LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct**.
        The assistant target is the entire **`deepseek-v4-pro_answer`**: Planning,
        Solution and Evaluation, Reflection, Exploration, and Final answer.
        Raw API reasoning and original s1 answers are excluded. No `<think>` or `<answer>`
        tags are added. The native model chat template and tokenizer are preserved.
        Only assistant answer tokens and native EOT receive loss; prompts/padding are masked.

        Use the project's **RTX PRO 6000 Blackwell 96GB GPU** Colab runtime and enable
        `HF_TOKEN` in Colab Secrets. Allow at least **100 GB free Drive space** for five
        complete model/optimizer checkpoints. Code and config are bundled; no GitHub checkout
        is needed. Open updated notebook versions in a fresh runtime.

        Run setup and data preparation, inspect the token counts and loss preview, then set
        `RUN_TRAINING = True` in the separate training cell. Training runs for **5 epochs**
        with batch size 1, gradient accumulation 16, learning rate 1e-5, BF16, and gradient
        checkpointing. Empty answers are skipped. Sequences above 20,480 tokens are reported
        and excluded, never truncated. No correctness/grade filter is applied.
        """),
        code('''
        # Change DATASET_PATH if you moved the export or finished its remaining rows.
        DATASET_PATH = "/content/drive/MyDrive/LG-AIME-S1-DeepSeek/runs/deepseek-v4-pro/63d2e34487188421/full/dataset.partial.jsonl"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1-AnswerOnly"
        RUN_NAME = "sft_s1k_deepseek_pro_answer_only"
        TRAIN_ENV = "/content/lg-answer-only-env"
        '''),
        code('''
        import os, subprocess, sys
        from pathlib import Path
        from google.colab import drive, userdata

        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        if not os.environ["HF_TOKEN"]:
            raise ValueError("Enable HF_TOKEN access in Colab Secrets")
        if not Path(DATASET_PATH).is_file():
            raise FileNotFoundError(f"Set DATASET_PATH to your generated JSONL: {DATASET_PATH}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        '''),
        boot,
        code('''
        from common.process import run_logged

        run_logged([sys.executable, str(CODE_ROOT / "scripts/setup_stage1_runtime.py"),
                    "--venv", TRAIN_ENV], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / "setup_console.log")
        TRAIN_PYTHON = str(Path(TRAIN_ENV) / "bin/python")
        '''),
        markdown('''
        ## Freeze the data and inspect training labels

        This saves a question/answer snapshot on Drive, including original row indices for
        tracking excluded answers. A later retry of generation cannot silently change the
        training data. For updated data/settings, choose a new `RUN_NAME`.

        The current 996-row partial export has 989 nonempty answers. The preparation report
        gives the actual retained count and tokens after applying the sequence-length limit.
        The user prompt uses the same English step-by-step/boxed-answer instruction as the
        project's evaluation. Answer text is retained exactly, including proof conclusions.
        '''),
        code('''
        run_logged([TRAIN_PYTHON, "-m", "train.answer_only",
                    "--dataset", DATASET_PATH,
                    "--config", "configs/stage1_sft_deepseek_pro_answer_only.yaml",
                    "--output-root", TRAIN_ROOT, "--run-name", RUN_NAME],
                   cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / RUN_NAME / "snapshot_console.log")
        CONFIG_PATH = Path(TRAIN_ROOT) / "inputs" / RUN_NAME / "training.yaml"
        TRAIN_COMMAND = [TRAIN_PYTHON, "train/stage1_sft.py", "--config", str(CONFIG_PATH)]
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / RUN_NAME / "prepare_console.log")
        '''),
        markdown('''
        ## Train for five epochs

        Checkpoints: `TRAIN_ROOT/checkpoints/stage1/RUN_NAME/epoch_1` through `epoch_5`.
        Each contains model, tokenizer, optimizer, scheduler, RNG, and trainer state.
        Logs and the data/code/runtime manifest are under `TRAIN_ROOT/logs/stage1/RUN_NAME`.

        To resume, rerun setup/preparation with the same notebook and settings, then set
        `RESUME_CHECKPOINT` to the latest completed `epoch_N` directory. Five epochs is the
        total, including completed epochs. Work since the last completed epoch is replayed.
        A completed five-epoch run needs no further training.
        '''),
        code('''
        RUN_TRAINING = False
        RESUME_CHECKPOINT = None
        # Example: str(Path(TRAIN_ROOT) / "checkpoints/stage1" / RUN_NAME / "epoch_2")

        if RUN_TRAINING:
            command = list(TRAIN_COMMAND)
            if RESUME_CHECKPOINT is not None:
                command += ["--resume_from_checkpoint", str(RESUME_CHECKPOINT)]
            run_logged(command, cwd=CODE_ROOT,
                       log_path=Path(TRAIN_ROOT) / RUN_NAME / "train_console.log",
                       compact_progress=True)
        else:
            print("Data preparation complete. Set RUN_TRAINING = True to start five-epoch SFT.")
        '''),
    ]


if __name__ == "__main__":
    write_notebook("train_stage1_sft_deepseek_pro_answer_only.ipynb", cells())
