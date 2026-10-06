"""Build the separate B-RL experiment from the pinned Qwen2.5-3B base model."""

from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_qwen_dapo_notebook import cells as instruct_cells

BUNDLE_FILES = [
    ".gitignore", "common/__init__.py", "common/io.py", "common/process.py",
    "common/english_prompts.py", "common/math_text.py",
    "pipeline/__init__.py", "pipeline/datasets.py", "eval/__init__.py",
    "eval/run_english_eval.py", "eval/run_eval.py", "eval/scoring.py",
    "train/__init__.py", "train/run_dapo.py", "train/dapo_data.py", "train/dapo_model.py",
    "train/dapo_resume.py", "train/dapo_rollout.py", "train/dapo_sampling.py",
    "train/dapo_trainer.py", "train/dapo_logps.py", "train/stage1_sft.py", "train/sft_data.py",
    "configs/dapo_qwen25_3b_base.yaml", "scripts/setup_dapo_runtime.py",
    "requirements-dapo.lock", "requirements-eval.lock", "requirements-stage1.lock",
]


def cells():
    result = instruct_cells()
    for cell in result:
        cell.source = cell.source.replace("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B")
        cell.source = cell.source.replace("Qwen2.5-3B-Instruct", "Qwen2.5-3B base")
        cell.source = cell.source.replace("configs/dapo_qwen25_3b.yaml", "configs/dapo_qwen25_3b_base.yaml")
        cell.source = cell.source.replace("dapo_qwen25_3b_base/", "dapo_qwen25_3b_pretrained_base/")
        cell.source = cell.source.replace("repeat on the second pass", "repeat on the second minibatch")
    result[0] = markdown('''
        # Qwen2.5-3B BASE + RL — DAPO, 300 updates

        Roadmap **B-RL**. Starts from **Qwen/Qwen2.5-3B**, revision
        `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, the same base checkpoint as B0.
        Full-parameter RL from the original base weights. No SFT checkpoint is loaded.

        Matches the instruct + RL experiment's DAPO data and optimization settings:
        - Verified 14,068-row English DAPO dataset; dataset commit is recorded when loaded.
        - G=8, 16 retained mixed-correctness questions / 128 responses per rollout.
        - Two disjoint 64-response updates per rollout; each retained response is used once.
        - **300 updates**, 150 fresh rollout batches, checkpoints every **20 updates**.
        - LR **1e-6**, 20-update warmup then constant; sampling temperature/top-p **1**.
        - Rule-based +1/-1 reward, clip 0.20/0.28, group normalization, token-level loss, no KL.
        - Response cap **20,480 tokens**; soft length penalty above **16,384**; no budget forcing.

        Uses the base tokenizer's shipped chat template/default system message and the same
        English math instruction. Preserves the original base **EOS `<|endoftext|>` (151643)**
        and original padding token (also `<|endoftext|>`). Generation stops on the native EOS;
        no EOS override, EOS suppression, extra stop tokens, or vocabulary additions.
        Completion masks are based on actual response lengths, so generated EOS receives loss
        while added padding does not, even though EOS and padding share the same token ID.

        **RTX PRO 6000 Blackwell 96GB**, FP32 parameters/Adam state, BF16 compute/rollouts,
        gradient checkpointing, and microbatch 1. Includes the Qwen memory fix: checkpointed
        vocabulary projection in 128-token chunks. vLLM sleeps during optimization.
        Fifteen full resumable checkpoints plus smoke need roughly **650 GB** of Drive space.

        Code/config/locks are embedded. Use a fresh GPU runtime, run setup and preparation,
        then enable the separate smoke and full-training cells. Training is independent of
        the completed instruct + RL and Kimi SFT runs. No evaluation or HF upload is performed.
        Across 300 updates: 2,400 retained question selections (not necessarily unique).
        Dynamic sampling can inspect additional questions to find mixed-correctness groups.

        **Updated EOS behavior:** stop the old Colab run and start this notebook in a fresh
        runtime. The default output is now `base-rl-native-eos`, separate from the earlier
        `base-rl` attempt. Start again from the original base weights; do not resume that
        attempt. Its files are preserved. Check smoke response lengths before full training;
        correcting EOS does not guarantee short responses or establish the earlier slowdown's cause.
        ''')
    result[1] = code('''
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-rl-native-eos"
        CODE_PARENT = "/content/lg-aime-dapo-qwen25-base"
        RL_ENV = "/content/lg-dapo-base-env"
        WORK_DIR = "/content/qwen25-base-dapo-work"
        CONFIG = "configs/dapo_qwen25_3b_base.yaml"
        ''')
    result[2] = markdown('''
        ## Setup

        Enable `HF_TOKEN` in Colab Secrets. The pinned Qwen base model and verified dataset
        are loaded from Hugging Face. The embedded code is extracted into a versioned local
        folder with a reproducible Git snapshot for strict resume checks; no GitHub clone.
        ''')
    encoded = payload([ROOT / name for name in BUNDLE_FILES])
    result[3].source = "\n".join(
        f"BUNDLE = {encoded!r}" if line.startswith("BUNDLE = ") else line
        for line in result[3].source.splitlines()
    )
    result[3].source = result[3].source.replace(
        'os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")',
        'os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")\n'
        'if not os.environ["HF_TOKEN"]:\n    raise ValueError("Enable HF_TOKEN in Colab Secrets")',
    )
    result[4] = markdown('''
        ## Paths and training command

        Dedicated base + RL checkpoints and logs. The runner's `--model-kind base` means
        starting from the HF model specified by this config: **Qwen2.5-3B base**.
        ''')
    result[10] = markdown('''
        ## Full training — 300 total optimizer updates

        Set `RUN_TRAINING = True`. The first run starts from the pinned Qwen base model.
        Saves checkpoints 20/40/.../300 with model, tokenizer, optimizer, scheduler and RNG.
        Dynamic sampling may generate extra groups to find 16 mixed-correctness questions.

        On rerun, the latest completed checkpoint in this run is selected automatically.
        `RESUME_CHECKPOINT` can explicitly select that same latest checkpoint. Resume restores
        training state and starts a fresh rollout at a complete cycle boundary. Work after
        the latest saved checkpoint is replayed; abandoned logs are archived by the runner.
        After a disconnect, rerun setup/preparation in a fresh GPU runtime with the same notebook.

        This run has no recovery parent. If it fails before checkpoint 20, no saved optimizer
        state exists; choose a new `OUTPUT_ROOT` to restart and preserve the failure logs.
        Keep code/config/runtime unchanged for resume. A completed checkpoint-300 run is skipped.
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
                    raise ValueError("Resume only the latest completed checkpoint of this base + RL run")
                marker, state = check_checkpoint(latest)
                previous = json.loads((run_logs / "run_manifest.json").read_text())
                if (marker["run_identity"] != previous["identity"]
                        or previous["config"]["model"] != cfg["model"]
                        or previous["config"]["run_name_prefix"] != cfg["run_name_prefix"]
                        or previous["config"].get("tokenizer") != cfg.get("tokenizer")
                        or previous["config"].get("generation") != cfg.get("generation")):
                    raise ValueError("Checkpoint belongs to another training run")
                command += ["--resume-from-checkpoint", latest]
                done = marker["global_step"] == cfg["training"]["max_steps"]
            if done:
                print("The 300-update base + RL run is already complete:", latest)
            else:
                print("Start/resume from:", latest or cfg["model"]["repo"])
                run_logged(command, cwd=CODE_ROOT, log_path=run_logs / "training_console.log",
                           compact_progress=True)
        else:
            print("Set RUN_TRAINING = True to start/resume the base + RL run.")
        ''')
    result.append(code('''
        print("Final model:", checkpoint_root / "checkpoint-300")
        print("Loss / reward / length history:", run_logs / "steps.jsonl")
        print("Run manifest:", run_logs / "run_manifest.json")
        print("Console log:", run_logs / "training_console.log")
        '''))
    return result


if __name__ == "__main__":
    write_notebook("train_dapo_qwen25_3b_base.ipynb", cells())
