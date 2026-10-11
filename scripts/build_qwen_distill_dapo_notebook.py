"""Build DAPO from the original 7B-distillation SFT epoch-5 checkpoint."""

from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_qwen_base_dapo_notebook import BUNDLE_FILES as BASE_FILES, cells as base_cells

BUNDLE_FILES = BASE_FILES + [
    'train/run_qwen_distill_dapo.py', 'configs/dapo_qwen25_3b_7b_distill.yaml',
]


def cells():
    result = base_cells()
    result[0] = markdown('''
        # Qwen2.5-3B 7B-distillation SFT epoch 5 + DAPO — 300 updates

        Roadmap **B-7B knowledge distillation SFT + RL**. Initialize from the original
        **`sft_qwen25_3b_base_7b_distill/epoch_5`** on Drive (AMC 47.5%, MATH-500 64.6%).
        Epochs 6–7 from the SFT continuation are separate experiments.

        Start **fresh RL steps 1–300**, with a new AdamW optimizer, scheduler and RNG seed.
        Load the SFT **model weights and tokenizer**. SFT optimizer/scheduler/RNG files
        are not restored. Later RL resumes restore this RL run's own complete training state.

        Same DAPO settings as the completed base + RL experiment:
        - Verified **14,068-row English DAPO dataset**, exact resolved revision recorded.
        - G=8; 16 retained mixed-correctness questions / 128 responses per rollout.
        - Two disjoint 64-response updates per rollout; each retained response used once.
        - **300 optimizer updates**, 150 fresh rollout batches, checkpoint every **20 updates**.
        - LR **1e-6**, **20-update warmup** then constant; sampling temperature/top-p **1**.
        - +1/-1 correctness reward, group normalization, clip 0.20/0.28, token loss, no KL.
        - **20,480-token response cap**, soft length penalty above **16,384**; no budget forcing.

        Preserve the saved Qwen chat template/default system message, vocabulary and native
        **EOS = PAD = `<|endoftext|>` (151643)**. Use the existing English math instruction.
        No additional tags or stop tokens. Completion masks distinguish real EOS from padding.

        **RTX PRO 6000 Blackwell 96GB**; FP32 parameters/Adam state, BF16 compute/rollouts,
        gradient checkpointing, microbatch 1, chunked vocabulary-projection memory fix.
        vLLM sleeps during optimization. Fifteen resumable checkpoints plus smoke need
        roughly **650 GB** of Drive space. Training logs use compact progress display;
        full loss/reward/length records remain on Drive.

        Self-contained code/config/locks, no GitHub download. Use a fresh Colab GPU runtime
        with `HF_TOKEN` in Secrets. Run setup and preparation, optionally enable the separate
        smoke, then **`RUN_TRAINING = True`**. Training only; no automatic benchmark evaluation
        or Hugging Face upload. The SFT artifacts remain in their original folder.
        Across 300 updates: 2,400 retained question selections, potentially with repeats;
        dynamic sampling may inspect additional candidates.
        ''')
    result[1] = code('''
        SFT_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-7b-distillation-sft-rl"
        CODE_PARENT = "/content/lg-aime-dapo-qwen25-distill"
        RL_ENV = "/content/lg-dapo-distill-env"
        WORK_DIR = "/content/qwen25-distill-dapo-work"
        CONFIG = "configs/dapo_qwen25_3b_7b_distill.yaml"
        ''')
    result[2] = markdown('''
        ## Setup

        Mount Drive and enable `HF_TOKEN` for the RL dataset. Keep the original SFT epoch-5
        checkpoint, its tokenizer, Trainer state, completion marker, run manifest and data
        report available under `SFT_ROOT`. Code is extracted into a versioned local folder
        with a reproducible Git snapshot for resume validation.
        ''')
    encoded = payload([ROOT / name for name in BUNDLE_FILES])
    result[3].source = '\n'.join(
        f'BUNDLE = {encoded!r}' if line.startswith('BUNDLE = ') else line
        for line in result[3].source.splitlines())
    result[4] = markdown('''
        ## Validate the SFT source and configure paths

        Verify the original five-epoch distillation run, completed epoch 5, source dataset,
        native tokenizer/template and complete weight shards. This reads checkpoint metadata;
        it does not load model weights into RAM/GPU. A separate output root preserves SFT files.
        ''')
    result[5] = code('''
        from train.dapo_data import load_config, batch_schedule
        from train.run_qwen_distill_dapo import select_distill_model

        cfg = load_config(Path(CODE_ROOT) / CONFIG)
        SFT_CHECKPOINT, SFT_IDENTITY = select_distill_model(cfg, "sft", SFT_ROOT, OUTPUT_ROOT)
        RUN_NAME = cfg["run_name_prefix"] + "_sft"
        run_logs = Path(OUTPUT_ROOT) / "logs" / RUN_NAME
        checkpoint_root = Path(OUTPUT_ROOT) / "checkpoints" / RUN_NAME
        TRAIN_COMMAND = [str(Path(RL_ENV) / "bin/python"), "-m", "train.run_qwen_distill_dapo",
                         "--config", CONFIG, "--model-kind", "sft", "--sft-root", SFT_ROOT,
                         "--output-root", OUTPUT_ROOT, "--work-dir", WORK_DIR]
        print("Starting SFT checkpoint:", SFT_CHECKPOINT)
        print("RL updates:", cfg["training"]["max_steps"], "; updates per rollout:",
              batch_schedule(cfg)["updates_per_rollout"])
        print("Checkpoints:", checkpoint_root)
        print("Training logs:", run_logs)
        ''')
    result[10] = markdown('''
        ## Full DAPO training — 300 new RL updates

        Set `RUN_TRAINING = True`. Starts from SFT epoch-5 weights with a fresh RL optimizer.
        Save steps 20/40/.../300 with full model, tokenizer, optimizer, scheduler and RNG.
        Smoke is a separate run and its weights are not used for full training.

        Reruns automatically select the latest completed **RL** checkpoint in this output
        folder. `RESUME_CHECKPOINT` can explicitly select that same checkpoint. Resume keeps
        the original SFT source identity and restores RL training state at a rollout boundary.
        Work since the last save is replayed; abandoned logs are archived. Keep the same
        notebook/config/runtime. After disconnecting, rerun setup/preparation in a fresh GPU
        runtime. If no RL checkpoint was saved before failure, choose a new `OUTPUT_ROOT`
        to restart from SFT epoch 5 while preserving failure logs. Completed step 300 is skipped.
        ''')
    result[11] = code('''
        RUN_TRAINING = False
        RESUME_CHECKPOINT = ""
        if RUN_TRAINING:
            from train.dapo_resume import check_checkpoint
            command = list(TRAIN_COMMAND)
            complete = sorted(
                (p for p in checkpoint_root.glob("checkpoint-*") if (p / "dapo_checkpoint.json").is_file()),
                key=lambda p: int(p.name.split("-")[-1]))
            latest = RESUME_CHECKPOINT or (str(complete[-1]) if complete else "")
            done = False
            if latest:
                if not complete or Path(latest).resolve() != complete[-1].resolve():
                    raise ValueError("Resume only this RL run's latest completed checkpoint")
                marker, state = check_checkpoint(latest)
                previous = json.loads((run_logs / "run_manifest.json").read_text())
                expected = json.loads(json.dumps(cfg))
                expected["model_kind"] = "sft"
                if expected["data"]["revision"] == "main":
                    expected["data"]["revision"] = previous["config"]["data"]["revision"]
                if (marker["run_identity"] != previous["identity"]
                        or marker.get("model_kind") != "sft"
                        or previous["config"] != expected or previous["source"] != SFT_IDENTITY
                        or previous["git_commit"] != git("rev-parse", "HEAD")
                        or state["max_steps"] != 300 or not 0 < marker["global_step"] <= 300):
                    raise ValueError("Checkpoint source/settings/code do not match this SFT + RL run")
                command += ["--resume-from-checkpoint", latest]
                done = marker["global_step"] == 300
            if done:
                print("The 300-update SFT + RL run is already complete:", latest)
            else:
                print("Start/resume from:", latest or SFT_CHECKPOINT)
                run_logged(command, cwd=CODE_ROOT, log_path=run_logs / "training_console.log",
                           compact_progress=True)
        else:
            print("Set RUN_TRAINING = True to start/resume 300 RL updates from SFT epoch 5.")
        ''')
    for cell in result[12:]:
        cell.source = cell.source.replace('dapo_qwen25_3b_pretrained_base', 'dapo_qwen25_3b_7b_distill_epoch5_sft')
    return result


if __name__ == '__main__':
    write_notebook('train_dapo_qwen25_3b_7b_distill_epoch5.ipynb', cells())
