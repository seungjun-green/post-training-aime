"""Build the self-contained distillation continuation notebook for epochs 6–7."""

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook
from build_qwen_self_rft_sft_notebook import cells as original_cells
from train.qwen_distill_continuation import BUNDLE_FILES


def cells():
    original = original_cells()
    boot = bootstrap(payload([ROOT / name for name in BUNDLE_FILES]))
    boot.source = boot.source.replace('/content/lg-korean-aime', '/content/lg-qwen-distill-continue')
    return [
        markdown('''
        # Qwen2.5-3B 7B-distillation SFT — continue epoch 5 → epochs 6–7

        Continue the completed **B-7B knowledge distillation SFT** run for **two additional
        epochs**, with **AMC 2023 + MATH-500 evaluation after each epoch**. The student is
        **Qwen2.5-3B base**. Restore epoch-5 model weights, AdamW moments, RNG and Trainer
        state. Continue global epoch/step numbering: epochs **6 and 7**, steps **11,781–16,492**.

        The original five-epoch learning rate has decayed to zero. Start a **new cosine
        schedule at `1e-6`**, without warmup, decaying to zero over **4,712 additional
        updates** (2,356 per epoch). Subsequent epoch-6 resumes restore this new schedule;
        they do not restart it. All other training settings are inherited and validated.

        Reuse the parent's exact pinned **37,691 `problem` / `response` rows**, in the same
        order, from `Seungjun/qwen2.5-3b-self-rft-7b-distill-math`. Supervise the entire
        response plus native **`<|endoftext|>` EOS**; prompt/padding remain masked. The saved
        native chat template, vocabulary, EOS and PAD are preserved. No new tags.

        Evaluation: **temperature 0**, **20,480 maximum generated tokens**, 40 AMC 2023
        and 500 MATH-500 problems per epoch, using the existing prompts and scorer.
        Training exits before evaluation to free GPU memory. Compact tqdm bars show
        training loss/LR and evaluation progress; each epoch prints its benchmark table.

        Use a **fresh RTX PRO 6000 Blackwell 96GB Colab runtime** and `HF_TOKEN` in Secrets.
        Keep the original run folder on Drive: its epoch-5 model, optimizer, scheduler,
        RNG, Trainer state, tokenizer, manifest, data report and input snapshot are required.
        This notebook writes to a **separate continuation folder**. Allow about **50 GB**
        additional Drive space for two full checkpoints. Code is embedded; no GitHub download.
        Run setup/preparation, then set **`RUN_TRAINING = True`**.
        '''),
        code('''
        PARENT_CHECKPOINT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft/checkpoints/stage1/sft_qwen25_3b_base_7b_distill/epoch_5"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft-epoch5-to7"
        RUN_NAME = "sft_qwen25_3b_base_7b_distill_continue"
        TRAIN_ENV = "/content/lg-qwen-distill-continue-train-env"
        EVAL_ENV = "/content/lg-qwen-distill-continue-eval-env"
        '''),
        original[2], boot, original[4], original[5],
        markdown('''
        ## Validate epoch 5 and prepare the continuation

        Verify the completed parent checkpoint, its manifest and exact data settings.
        Copy the small dataset snapshot into the continuation folder; model/optimizer
        files remain in the original folder. Validate tokenization and loss masking
        against the parent's report. Changed data/settings require a new run name.
        '''),
        code('''
        run_logged([TRAIN_PYTHON, "-m", "train.qwen_distill_continuation",
                    "--parent-checkpoint", PARENT_CHECKPOINT,
                    "--output-root", TRAIN_ROOT, "--run-name", RUN_NAME],
                   cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / RUN_NAME / "snapshot_console.log")
        CONFIG_PATH = Path(TRAIN_ROOT) / "inputs" / RUN_NAME / "training.yaml"
        run_logged([TRAIN_PYTHON, "-m", "train.stage1_qwen_distill_continue",
                    "--config", str(CONFIG_PATH), "--prepare-only"],
                   cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / RUN_NAME / "prepare_console.log")
        log_dir = Path(TRAIN_ROOT) / "logs/stage1" / RUN_NAME
        checkpoint_dir = Path(TRAIN_ROOT) / "checkpoints/stage1" / RUN_NAME
        report = json.loads((log_dir / "preparation/data_report.json").read_text())
        print("Resume source:", PARENT_CHECKPOINT)
        print("Retained rows:", report["kept_rows"])
        print("Additional epochs: 6 and 7; additional updates:",
              report["optimizer_steps_per_epoch"] * 2)
        print("New learning rate: 1e-6, cosine decay to zero over both additional epochs")
        print("Loss preview:", log_dir / "preparation/sanity_check.txt")
        '''),
        markdown('''
        ## Continue training and evaluate epochs 6–7

        On rerun, restore the latest completed continuation epoch. If epoch 6 has not
        finished saving, retry from parent epoch 5 and archive abandoned loss records.
        Interrupted evaluations reuse saved problems before proceeding to the next epoch.
        Epochs 1–5 are neither retrained nor reevaluated. Keep the same code, settings and
        locked runtime when resuming. After disconnecting, rerun setup/preparation and
        enable this cell again. Results remain unconfirmed until this runs on Colab.
        '''),
        code('''
        RUN_TRAINING = False
        if RUN_TRAINING:
            from train.qwen_distill_continue_epochs import run_epochs
            epoch_results = run_epochs(CONFIG_PATH, code_root=CODE_ROOT,
                                       train_python=TRAIN_PYTHON, eval_python=EVAL_PYTHON)
        else:
            print("Set RUN_TRAINING = True to train and evaluate epochs 6 and 7.")
        '''),
        markdown('''
        ## Continuation output locations

        Under `TRAIN_ROOT`:
        - `checkpoints/stage1/RUN_NAME/epoch_6/` and `epoch_7/`: model, tokenizer,
          optimizer, scheduler, RNG and Trainer state.
        - `logs/stage1/RUN_NAME/steps.jsonl`: additional training loss/LR history.
        - `logs/stage1/RUN_NAME/run_manifest.json`: parent identity and continuation settings.
        - `RUN_NAME/train_console.log`: full training diagnostics.
        - `eval/amc2023_math500_temp0/RUN_NAME/epoch_6/` and `epoch_7/`: metrics and generations.
        - `eval/amc2023_math500_temp0/RUN_NAME/epoch_summary.csv`: four benchmark rows.

        Original epoch-1–5 checkpoints, logs and scores remain in the parent folder.
        '''),
        code('''
        print("Final model:", checkpoint_dir / "epoch_7")
        print("Loss history:", log_dir / "steps.jsonl")
        print("Run manifest:", log_dir / "run_manifest.json")
        print("Evaluation summary:", Path(TRAIN_ROOT) / "eval/amc2023_math500_temp0" / RUN_NAME / "epoch_summary.csv")
        summary = log_dir / "training_summary.json"
        if summary.exists():
            print(json.dumps(json.loads(summary.read_text()), indent=2))
        '''),
    ]


if __name__ == '__main__':
    write_notebook('continue_qwen25_3b_7b_distill_epoch5_to7.ipynb', cells())
