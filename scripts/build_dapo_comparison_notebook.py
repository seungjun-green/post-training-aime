"""Build the base-vs-DAPO-100 greedy AMC/MATH comparison notebook."""

from build_dapo_eval_notebook import cells as eval_cells
from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Base vs DAPO checkpoint 100 — AMC 2023 and MATH-500
        Compare the original **EXAONE-3.5-2.4B-Instruct** with the trained
        **dapo_exaone_base/checkpoint-100**. Evaluate **AMC 2023 (40 problems)** and
        **MATH-500 (500 problems)** only. AIME 2024, 2025 and 2026 are excluded before
        dataset loading. Each model generates **540 answers**, one per problem.

        Both models use **temperature 0, top-p 1, seed 42**, the same English prompt,
        answer checker, continuous batching and **20,480-token response limit**.
        **No budget forcing** for either model. The two models run sequentially on one GPU.
        The final table shows accuracy, DAPO-minus-base difference in percentage points,
        and mean response length for each benchmark. Results and raw answers are saved on
        Drive; checkpoint weights are only read and are never copied into evaluation outputs.

        Use the RTX PRO 6000 Blackwell 96GB and HF_TOKEN secret, after any training process
        on that GPU has finished. Push the comparison code/config to GitHub main first.
        Plain Python settings below; no Colab forms. No previous baseline evaluation required.
        """),
        code("""
        RL_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Compare-100-AMC-MATH-temp0"
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-dapo-compare"
        EVAL_ENV = "/content/lg-eval-env"
        CONFIG = "configs/dapo_compare_eval.yaml"
        """),
        markdown("""
        ## Setup
        Mount Drive, update the repository and install the existing evaluation environment.
        """),
        eval_cells()[3],
        markdown("""
        ## Check models and prepare the comparison
        Validate checkpoint 100 and its training provenance. Inference files are enough;
        optimizer files are not required. Confirm the two printed model paths before running.
        """),
        code("""
        from eval.run_amc_math_eval import comparison_settings
        from eval.dapo_comparison import comparison_rows
        from common.io import write_json
        from train.dapo_resume import check_checkpoint
        spec, resolved, execution, suite = comparison_settings(Path(CODE_ROOT) / CONFIG)
        checkpoint = Path(RL_ROOT) / "checkpoints/dapo_exaone_base" / f"checkpoint-{spec['checkpoint_step']}"
        marker, _ = check_checkpoint(checkpoint, require_training_state=False)
        parent = json.loads((Path(RL_ROOT) / "logs/dapo_exaone_base/run_manifest.json").read_text())
        if (marker["global_step"] != 100 or marker["model_kind"] != "base"
                or marker["run_identity"] != parent["identity"]):
            raise ValueError("Select the completed base-started DAPO checkpoint 100")
        if any(parent["config"]["model"][key] != spec["base_model"][key] for key in ["repo", "revision"]):
            raise ValueError("Comparison baseline must match the original model used for DAPO")
        eval_runs = [("base", spec["base_model"]["repo"], spec["base_model"]["revision"]),
                     ("dapo_checkpoint100", str(checkpoint), None)]
        progress_totals = {name: entry["rows"] for name, entry in suite["datasets"].items()}
        result_root = Path(OUTPUT_ROOT) / "full/results"
        write_json(Path(OUTPUT_ROOT) / "comparison_settings.json",
                   {"spec": spec, "config": resolved, "execution": execution, "checkpoint": str(checkpoint),
                    "checkpoint_marker": marker})
        def evaluation_command(run, smoke=False):
            run_name, model, revision = run
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_amc_math_eval",
                       "--model", model, "--run-name", run_name, "--config", CONFIG,
                       "--output-root", OUTPUT_ROOT]
            if revision is not None:
                command += ["--revision", revision]
            return command + (["--smoke"] if smoke else [])
        def show_comparison():
            import pandas as pd
            from IPython.display import display
            paths = [result_root / (run[0] + ".json") for run in eval_runs]
            if not all(path.is_file() for path in paths):
                print("Complete both full evaluations before creating the comparison table.")
                return
            base, rl = [json.loads(path.read_text()) for path in paths]
            if (base["model"] != eval_runs[0][1] or base["model_revision"] != eval_runs[0][2]
                    or rl["model"] != str(checkpoint)):
                raise ValueError("Saved results refer to different models")
            rows = comparison_rows(base, rl)
            destination = Path(OUTPUT_ROOT) / "full/comparison.json"
            write_json(destination, {"models": {"base": base["model"], "dapo": rl["model"]},
                                    "temperature": 0.0, "rows": rows})
            display(pd.DataFrame(rows).style.format({
                "Base accuracy (%)": "{:.1f}", "DAPO-100 accuracy (%)": "{:.1f}",
                "Difference (pp)": "{:+.1f}", "Base mean tokens": "{:.1f}",
                "DAPO-100 mean tokens": "{:.1f}"}))
            print("Difference = DAPO minus base (percentage points). Saved:", destination)
        print("Benchmarks:", progress_totals, "Problems per model:", sum(progress_totals.values()))
        print("Temperature:", resolved["temperature"], "Answers per problem: 1; budget forcing: off")
        for name, model, revision in eval_runs:
            print(name, "->", model, "revision:", revision)
        print("Results:", result_root)
        """),
        markdown("""
        ## Optional smoke — two problems per model
        One AMC problem and one MATH-500 problem for each model, four answers in total.
        These results go into a separate smoke folder and are not used in the comparison table.
        """),
        code("""
        RUN_SMOKE = False
        if RUN_SMOKE:
            for run in eval_runs:
                run_logged(evaluation_command(run, smoke=True), cwd=CODE_ROOT,
                           log_path=Path(OUTPUT_ROOT) / "logs" / (run[0] + "_smoke_console.log"),
                           progress_totals={name: 1 for name in progress_totals}, compact_progress=True)
        else:
            print("Optional smoke is off. Set RUN_SMOKE = True to test both models.")
        """),
        markdown("""
        ## Evaluate both models and create the table
        Set RUN_COMPARISON to True. Base runs first, then checkpoint 100. Each model has
        its own 540-problem progress bar. Completed problems are saved and reused after an
        interruption with unchanged code/settings/models; rerun this cell to continue.
        The comparison table is shown automatically when both models finish.
        """),
        code("""
        RUN_COMPARISON = False
        if RUN_COMPARISON:
            for run in eval_runs:
                print("Evaluating:", run[0])
                run_logged(evaluation_command(run), cwd=CODE_ROOT,
                           log_path=Path(OUTPUT_ROOT) / "logs" / (run[0] + "_console.log"),
                           progress_totals=progress_totals, compact_progress=True)
            show_comparison()
        else:
            print("Set RUN_COMPARISON = True to evaluate both models and display the table.")
        """),
        markdown("""
        ## Show the saved comparison again
        This cell reads completed results without loading models or generating answers.
        Under OUTPUT_ROOT, `full/results/` contains each model's summary JSON,
        `_generations.jsonl` raw answers and `_problems.jsonl` resume journal.
        `full/comparison.json` contains the comparison table; `logs/` contains console output.
        """),
        code("""
        show_comparison()
        """),
    ]


if __name__ == "__main__":
    write_notebook("compare_base_dapo100_amc_math.ipynb", cells())
