"""Evaluate the EXAONE continuation checkpoint 140 on AMC and MATH only."""

from build_dapo_eval_notebook import cells as eval_cells
from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # EXAONE DAPO checkpoint 140 — AMC 2023 and MATH-500
        Evaluates **dapo_exaone_base/checkpoint-140** from the 100-to-300 continuation.
        **AMC 2023: 40 problems. MATH-500: 500 problems. Total: 540 answers.**
        AIME is excluded before loading datasets. Only checkpoint 140 is evaluated.

        Same settings as the previous base/checkpoint-100 comparison: **temperature 0,
        top-p 1, seed 42, one answer per problem**, 20,480-token response limit and no
        budget forcing. Reuses the existing English prompt, scorer and continuously batched
        evaluation entry point. No training or model-weight copying/saving.

        Results, raw answers and console logs are saved to Drive in a separate folder.
        Use the RTX PRO 6000 Blackwell 96GB and HF_TOKEN secret. Run on a free GPU after
        training releases it. Push the new notebook/config to GitHub main before setup.
        Plain Python variables below, with no Colab forms or baseline prerequisite.
        """),
        code("""
        RL_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-100to300"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Eval-140-AMC-MATH-temp0"
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-dapo140-eval"
        EVAL_ENV = "/content/lg-eval-env"
        CONFIG = "configs/dapo_eval_140_amc_math.yaml"
        """),
        markdown("""
        ## Setup
        Mount Drive and install the existing pinned evaluation runtime.
        """),
        eval_cells()[3],
        markdown("""
        ## Validate checkpoint 140 and prepare evaluation
        RL_ROOT is the continuation output folder containing `checkpoints/` and `logs/`.
        Checks the completion marker, training identity and weight shard sizes. Optimizer
        files are not required. Confirm the checkpoint path printed below before running.
        """),
        code("""
        from eval.run_amc_math_eval import comparison_settings
        from train.dapo_resume import check_checkpoint
        from common.io import write_json
        spec, resolved, execution, suite = comparison_settings(Path(CODE_ROOT) / CONFIG)
        step = spec["checkpoint_step"]
        training_run = "dapo_exaone_base"
        checkpoint = Path(RL_ROOT) / "checkpoints" / training_run / f"checkpoint-{step}"
        marker, _ = check_checkpoint(checkpoint, require_training_state=False)
        manifest = json.loads((Path(RL_ROOT) / "logs" / training_run / "run_manifest.json").read_text())
        if (marker["global_step"] != step or marker["model_kind"] != "base"
                or marker["run_identity"] != manifest["identity"]):
            raise ValueError("Select the completed base-started EXAONE DAPO checkpoint 140")
        if any(manifest["config"]["model"][key] != spec["base_model"][key] for key in ["repo", "revision"]):
            raise ValueError("Checkpoint must come from the original EXAONE model")
        run_name = f"dapo_checkpoint{step}"
        progress_totals = {name: entry["rows"] for name, entry in suite["datasets"].items()}
        result_path = Path(OUTPUT_ROOT) / "full/results" / (run_name + ".json")
        write_json(Path(OUTPUT_ROOT) / "evaluation_settings.json",
                   {"spec": spec, "config": resolved, "execution": execution,
                    "checkpoint": str(checkpoint), "checkpoint_marker": marker})
        def evaluation_command(smoke=False):
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_amc_math_eval",
                       "--model", str(checkpoint), "--run-name", run_name, "--config", CONFIG,
                       "--output-root", OUTPUT_ROOT]
            return command + (["--smoke"] if smoke else [])
        def show_results():
            if not result_path.is_file():
                print("Full evaluation is not complete yet:", result_path)
                return
            import pandas as pd
            from IPython.display import display
            result = json.loads(result_path.read_text())
            if result["model"] != str(checkpoint) or result["mode"] != "full" or result["config"] != resolved:
                raise ValueError("Saved results do not match this checkpoint/evaluation config")
            names = {"amc23": "AMC 2023", "math_500": "MATH-500"}
            rows = []
            for name, total in progress_totals.items():
                metric = result["metrics"][name]
                if (metric["problems"] != total or metric["samples_per_problem"] != 1
                        or metric["responses"] != total):
                    raise ValueError(f"Incomplete full evaluation: {name}")
                rows.append({"Benchmark": names[name], "Problems": total,
                             "Correct": round(metric["avg@1"] * total),
                             "DAPO-140 accuracy (%)": 100 * metric["avg@1"],
                             "Mean tokens": metric["response_length_tokens"]["all"]})
            table_path = Path(OUTPUT_ROOT) / "full/summary_table.json"
            write_json(table_path, {"model": str(checkpoint), "temperature": resolved["temperature"], "rows": rows})
            display(pd.DataFrame(rows).style.format({"DAPO-140 accuracy (%)": "{:.1f}", "Mean tokens": "{:.1f}"}))
            print("Summary:", result_path)
            print("Table:", table_path)
            print("Raw answers:", result_path.with_name(run_name + "_generations.jsonl"))
        print("Checkpoint:", checkpoint)
        print("Benchmarks:", progress_totals, "Total:", sum(progress_totals.values()))
        print("Temperature:", resolved["temperature"], "Answers per problem: 1; budget forcing: off")
        print("Results:", result_path)
        """),
        markdown("""
        ## Optional smoke — two problems
        One AMC problem and one MATH-500 problem. Smoke results go in a separate folder
        and are excluded from the full summary table. No weights are saved.
        """),
        code("""
        RUN_SMOKE = False
        if RUN_SMOKE:
            run_logged(evaluation_command(smoke=True), cwd=CODE_ROOT,
                       log_path=Path(OUTPUT_ROOT) / "logs/smoke_console.log",
                       progress_totals={name: 1 for name in progress_totals}, compact_progress=True)
        else:
            print("Set RUN_SMOKE = True for the optional two-problem smoke.")
        """),
        markdown("""
        ## Evaluate checkpoint 140 — 540 problems
        Set RUN_EVAL to True. The progress bar spans AMC 2023 and MATH-500 together.
        Completed problems are saved and reused after interruption with unchanged checkpoint,
        code and settings. The two-row results table appears automatically after completion.
        """),
        code("""
        RUN_EVAL = False
        if RUN_EVAL:
            run_logged(evaluation_command(), cwd=CODE_ROOT,
                       log_path=Path(OUTPUT_ROOT) / "logs/eval_console.log",
                       progress_totals=progress_totals, compact_progress=True)
            show_results()
        else:
            print("Set RUN_EVAL = True to evaluate checkpoint 140 on all 540 problems.")
        """),
        markdown("""
        ## Show saved results again
        Reads the completed summary without running inference. Under OUTPUT_ROOT,
        `full/results/dapo_checkpoint140.json` contains metrics and settings;
        `_generations.jsonl` contains answers, and `_problems.jsonl` is the resume journal.
        `full/summary_table.json` contains the displayed table. Console output is under `logs/`.
        """),
        code("""
        show_results()
        """),
    ]


if __name__ == "__main__":
    write_notebook("evaluate_dapo_checkpoint140_amc_math.ipynb", cells())
