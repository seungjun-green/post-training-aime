"""Build evaluation-only notebook for the completed DAPO checkpoint 100."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Evaluate EXAONE DAPO — checkpoint 100
        Evaluates the RL checkpoint on **AIME 2024, AIME 2025, AIME 2026, AMC 2023,
        and MATH-500**: **630 problems, one answer per problem**. Uses the existing
        English prompt, final-answer scorer and continuous GPU batching without changes.
        Defaults: temperature 0, top-p 1, a 20,480-token response limit, **no budget forcing**.
        For a fair comparison, evaluate the original model with the same sampling settings.

        MODEL_KIND identifies the starting model of your DAPO run; "base" loads the trained
        `dapo_exaone_base/checkpoint-100`, not the original untrained model. No completed
        baseline run is required. This notebook only evaluates; it never trains or saves
        model weights. It writes metrics, generated answers and logs to Drive.

        Use the RTX PRO 6000 Blackwell 96GB and HF_TOKEN Colab secret. Run evaluation after
        any training process on that GPU has finished. Push the new notebook/code/config to
        GitHub main before setup. Settings are plain Python variables, without Colab forms.
        """),
        code("""
        MODEL_KIND = "base"  # "base" or "sft": starting model used by the DAPO run
        RL_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO"
        TEMPERATURE = 0.0
        TOP_P = 1.0
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-dapo-eval"
        EVAL_ENV = "/content/lg-eval-env"
        CONFIG = "configs/dapo_eval.yaml"
        """),
        markdown("""
        ## Setup
        Mount Drive and install the existing pinned evaluation runtime.
        """),
        code("""
        import os, subprocess, sys, json
        from pathlib import Path
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        if not Path(CODE_ROOT).exists():
            subprocess.check_call(["git", "clone", "--branch", "main", REPO_URL, CODE_ROOT])
        def git(*args):
            return subprocess.check_output(["git", "-C", CODE_ROOT, *args], text=True).strip()
        if git("remote", "get-url", "origin") != REPO_URL or git("status", "--porcelain"):
            raise ValueError("Use a clean checkout of the configured repository")
        if git("branch", "--show-current") != "main":
            raise ValueError("Use a main checkout")
        subprocess.check_call(["git", "-C", CODE_ROOT, "pull", "--ff-only", "origin", "main"])
        print("Evaluation code commit:", git("rev-parse", "HEAD"))
        for required in [CONFIG, "train/dapo_resume.py", "eval/run_batched_eval.py", "scripts/setup_eval_runtime.py"]:
            if not (Path(CODE_ROOT) / required).is_file():
                raise FileNotFoundError(f"Push the DAPO evaluation update to GitHub main first; missing {required}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", EVAL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        """),
        markdown("""
        ## Check checkpoint 100 and prepare evaluation settings
        Checks the completion marker, training-run identity and weight shard sizes. Optimizer
        files are not needed for evaluation. This reads small headers, not full tensors.
        Rerun this cell after changing settings. Sampling settings are recorded in YAML on
        Drive; different temperatures/top-p values have separate output directories.
        """),
        code("""
        import yaml
        from eval.profiles import load_profile, profile_root
        from train.dapo_resume import check_checkpoint
        if MODEL_KIND not in {"base", "sft"}:
            raise ValueError("MODEL_KIND must be base or sft")
        cfg = yaml.safe_load((Path(CODE_ROOT) / CONFIG).read_text())
        step = cfg["checkpoint_step"]
        training_run = "dapo_exaone_" + MODEL_KIND
        checkpoint = Path(RL_ROOT) / "checkpoints" / training_run / f"checkpoint-{step}"
        marker, _ = check_checkpoint(checkpoint, require_training_state=False)
        manifest = json.loads((Path(RL_ROOT) / "logs" / training_run / "run_manifest.json").read_text())
        if (marker["global_step"] != step or marker["model_kind"] != MODEL_KIND
                or marker["run_identity"] != manifest["identity"]):
            raise ValueError("Checkpoint step/model/run identity does not match the selected DAPO run")
        EVAL_PROFILE = cfg["profile"]
        sampling = {**cfg["sampling"], "temperature": TEMPERATURE, "top_p": TOP_P}
        eval_config = yaml.safe_load((Path(CODE_ROOT) / cfg["evaluation_config"]).read_text())
        execution = yaml.safe_load((Path(CODE_ROOT) / cfg["execution_config"]).read_text())
        resolved, _ = load_profile(EVAL_PROFILE, eval_config, execution, sampling_override=sampling)
        if "budget_forcing" in resolved or any(d["n"] != 1 for d in resolved["datasets"].values()):
            raise ValueError("Expected one answer per problem without budget forcing")
        EVAL_ROOT = Path(RL_ROOT) / f"evaluation_checkpoint{step}"
        output_profile = profile_root(EVAL_ROOT, EVAL_PROFILE, sampling=sampling)
        config_dir = output_profile / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        sampling_path = config_dir / "sampling.yaml"
        sampling_path.write_text(yaml.safe_dump(sampling, sort_keys=False))
        (config_dir / "resolved_eval.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
        run_name = training_run + f"_checkpoint{step}"
        suite = json.loads((Path(CODE_ROOT) / cfg["suite"]).read_text())
        progress_totals = {name: entry["rows"] for name, entry in suite["datasets"].items()}
        result_path = output_profile / "full/results/dapo" / (run_name + ".json")
        def evaluation_command(smoke=False):
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_batched_eval",
                       "--model", str(checkpoint), "--stage", "dapo", "--run_name", run_name,
                       "--output_root", str(EVAL_ROOT), "--profile", EVAL_PROFILE,
                       "--config", cfg["evaluation_config"], "--suite", cfg["suite"],
                       "--execution_config", cfg["execution_config"], "--sampling_config", str(sampling_path)]
            return command + (["--smoke"] if smoke else [])
        print("RL checkpoint:", checkpoint)
        print("Benchmarks:", progress_totals, "Total:", sum(progress_totals.values()))
        print("Sampling:", sampling, "Answers per problem: 1")
        print("Response limit:", resolved["max_new_tokens"], "Budget forcing: off")
        print("Results:", result_path)
        """),
        markdown("""
        ## Optional smoke evaluation — five problems
        One problem per benchmark. Saves only evaluation outputs in a separate smoke folder;
        it does not copy or save checkpoint weights.
        """),
        code("""
        RUN_SMOKE_EVAL = False
        if RUN_SMOKE_EVAL:
            run_logged(evaluation_command(smoke=True), cwd=CODE_ROOT,
                       log_path=output_profile / "logs" / (run_name + "_smoke_console.log"),
                       progress_totals={name: 1 for name in progress_totals}, compact_progress=True)
        else:
            print("Optional smoke is off. Set RUN_SMOKE_EVAL = True to evaluate five problems.")
        """),
        markdown("""
        ## Full evaluation — all five benchmarks
        Enable RUN_EVAL. Uses the same continuous batching as the existing evaluator.
        Completed problems are saved on Drive and reused if you rerun with identical
        checkpoint/code/settings. The 630-problem progress bar spans all five benchmarks.
        """),
        code("""
        RUN_EVAL = False
        if RUN_EVAL:
            run_logged(evaluation_command(), cwd=CODE_ROOT,
                       log_path=output_profile / "logs" / (run_name + "_console.log"),
                       progress_totals=progress_totals, compact_progress=True)
        else:
            print("Set RUN_EVAL = True to evaluate all 630 problems.")
        """),
        markdown("""
        ## Results on Drive
        Summary JSON, raw generations and a resumable per-problem journal are saved together.
        The original RL checkpoint remains in its training folder.
        """),
        code("""
        if result_path.is_file():
            results = json.loads(result_path.read_text())
            print(json.dumps(results["metrics"], indent=2))
        else:
            print("Full evaluation is not complete yet.")
        print("Summary:", result_path)
        print("Generated answers:", result_path.with_name(run_name + "_generations.jsonl"))
        print("Resume journal:", result_path.with_name(run_name + "_problems.jsonl"))
        print("Console logs:", output_profile / "logs")
        """),
    ]


if __name__ == "__main__":
    write_notebook("evaluate_dapo_checkpoint100.ipynb", cells())
