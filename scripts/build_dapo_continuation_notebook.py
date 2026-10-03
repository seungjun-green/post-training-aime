"""Build the notebook that continues a completed DAPO checkpoint to step 300."""

from build_dapo_notebook import cells as training_cells
from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # EXAONE DAPO — continue from step 100 to step 300
        Restore the completed checkpoint at **step 100** and train **200 additional updates**.
        Model weights, optimizer, scheduler, RNG and Trainer state are restored. The existing
        warmup is not repeated; learning rate stays at 1e-6. G=8, 16 mixed groups, response
        budget 20,480, length shaping and all other training settings stay unchanged.

        Settings are ordinary Python variables. Default MODEL_KIND is **base**, matching the
        completed run. Use the same RTX PRO 6000 Blackwell 96GB and HF_TOKEN secret.
        Push the continuation code/config to GitHub main before setup.

        Original checkpoints and logs remain under ORIGINAL_ROOT. New checkpoints and logs
        go under the separate OUTPUT_ROOT. Saving every 20 updates creates ten additional
        full resumable checkpoints (120, 140, ..., 300); allow roughly **300 GB additional
        Drive space**. There is no new smoke run or benchmark evaluation in this notebook.
        """),
        code("""
        MODEL_KIND = "base"  # "base" or "sft"; match the completed DAPO run
        ORIGINAL_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-100to300"
        SFT_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-dapo"
        RL_ENV = "/content/lg-dapo-env"
        WORK_DIR = "/content/lg-dapo-work"
        CONFIG = "configs/dapo_continue_300.yaml"
        """),
        markdown("""
        ## Setup
        Mount Drive, update the clean checkout and use the same pinned runtime.
        """),
        training_cells()[3],
        markdown("""
        ## Check the completed checkpoint and continuation paths
        Checks the completion marker, Trainer step, required resume files and model shard
        sizes without loading weights into the notebook. Only the total step budget may
        change. The original dataset revision is reused. The new code commit is recorded
        alongside the original run identity; this explicitly allows the continuation code
        update without disabling source/data/runtime checks.
        """),
        code("""
        from train.dapo_data import load_config
        from train.dapo_resume import extension_source
        cfg = load_config(Path(CODE_ROOT) / CONFIG)
        cfg["model_kind"] = MODEL_KIND
        RUN_NAME = cfg["run_name_prefix"] + "_" + MODEL_KIND
        parent_checkpoint = Path(ORIGINAL_ROOT) / "checkpoints" / RUN_NAME / "checkpoint-100"
        parent_manifest, continuation = extension_source(cfg, parent_checkpoint, OUTPUT_ROOT)
        run_logs = Path(OUTPUT_ROOT) / "logs" / RUN_NAME
        checkpoint_root = Path(OUTPUT_ROOT) / "checkpoints" / RUN_NAME
        TRAIN_COMMAND = [str(Path(RL_ENV) / "bin/python"), "train/run_dapo.py",
                         "--config", CONFIG, "--model-kind", MODEL_KIND, "--sft-root", SFT_ROOT,
                         "--output-root", OUTPUT_ROOT, "--work-dir", str(Path(WORK_DIR) / (RUN_NAME + "_100to300")),
                         "--extend-from-checkpoint", str(parent_checkpoint)]
        print("Restore:", parent_checkpoint)
        print("Updates:", continuation["from_step"] + 1, "through", continuation["target_step"])
        print("New checkpoints:", checkpoint_root)
        print("New logs:", run_logs)
        """),
        markdown("""
        ## Verify data and prompt
        Reuses the pinned decontaminated dataset and tokenizer from checkpoint 100.
        This only prepares data; it does not start training or rewrite original files.
        """),
        code("""
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=run_logs / "prepare_console.log", compact_progress=True)
        """),
        markdown("""
        ## Continue training
        Set RUN_TRAINING to True. On the first run, resume from the original checkpoint 100.
        If interrupted later, rerun this cell: it automatically selects the latest completed
        checkpoint in the continuation folder, restoring its optimizer and RNG state too.
        Incomplete checkpoint folders are preserved separately by the runner.

        If an attempt failed before creating any continuation checkpoint, choose a new
        OUTPUT_ROOT and rerun the path/preparation cells. Logs after the last completed
        checkpoint are archived when resuming. Keep code/config/runtime unchanged while
        resuming this continuation. Once checkpoint 300 is complete, the cell skips training.
        """),
        code("""
        RUN_TRAINING = False
        if RUN_TRAINING:
            completed = sorted(checkpoint_root.glob("checkpoint-*/dapo_checkpoint.json"),
                               key=lambda p: int(p.parent.name.split("-")[-1]))
            resume = completed[-1].parent if completed else None
            last_step = json.loads(completed[-1].read_text())["global_step"] if completed else continuation["from_step"]
            if last_step >= cfg["training"]["max_steps"]:
                print("Already complete:", resume)
            else:
                command = TRAIN_COMMAND + (["--resume-from-checkpoint", str(resume)] if resume else [])
                print("Resuming from:", resume or parent_checkpoint)
                run_logged(command, cwd=CODE_ROOT, log_path=run_logs / "training_console.log",
                           compact_progress=True)
        else:
            print("Set RUN_TRAINING = True to continue to step 300.")
        """),
        markdown("""
        ## Verify completion and find results
        New steps.jsonl records use absolute step numbers 101–300. The original 1–100 log is
        preserved under ORIGINAL_ROOT. dapo_checkpoint.json is written after saving finishes.
        Console progress refreshes in place; full console output and rollouts are saved to Drive.
        """),
        code("""
        from train.dapo_resume import check_checkpoint
        final_checkpoint = checkpoint_root / f"checkpoint-{cfg['training']['max_steps']}"
        if (final_checkpoint / "dapo_checkpoint.json").is_file():
            marker, state = check_checkpoint(final_checkpoint)
            print("Completed step:", state["global_step"])
            print("Model and resume state:", final_checkpoint)
        else:
            print("Step 300 has not finished saving yet.")
        print("Training metrics:", run_logs / "steps.jsonl")
        print("Console:", run_logs / "training_console.log")
        """),
    ]


if __name__ == "__main__":
    write_notebook("continue_dapo_exaone_100_to_300.ipynb", cells())
