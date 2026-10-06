"""Build self-contained Qwen base / instruct / instruct + RL evaluation."""

import sys

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

sys.path.insert(0, str(ROOT))
from eval.qwen_comparison import BUNDLE_FILES  # noqa: E402


def cells():
    boot = bootstrap(payload([ROOT / path for path in BUNDLE_FILES]))
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-qwen-comparison-eval")
    return [
        markdown(r"""
        # Qwen2.5-3B: base vs instruct vs instruct + RL

        **AMC 2023 (40) and MATH-500 (500)** for each of three models: **1,620 responses**.
        Temperature **0**, top-p **1**, one response per question, seed **42**,
        response cap **20,480 tokens**, context **32,768 tokens**. No budget forcing.

        Models: pinned `Qwen/Qwen2.5-3B`, pinned `Qwen/Qwen2.5-3B-Instruct`, and the
        completed instruct DAPO `checkpoint-300` in the `MemoryFix` Drive folder.
        All receive the same English question and step-by-step / `\boxed{}` instruction.
        Each uses its shipped **native chat template**, including its default system message,
        and model EOS token. Base evaluation is zero-shot with its shipped chat template;
        this is not a few-shot base-model benchmark protocol. Instruct and RL templates must match.
        Full generated responses and the existing final-box/math-equivalence scores are saved.

        Use a **fresh RTX PRO 6000 Blackwell 96GB Colab runtime** and enable `HF_TOKEN`
        in Secrets. Code and configs are embedded; no GitHub checkout is needed.
        Models run sequentially in separate processes to release GPU memory between runs.
        Run setup, optionally enable `RUN_SMOKE`, then set **`RUN_EVAL = True`**.
        """),
        code('''
        RL_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix"
        RL_RUN_NAME = "dapo_qwen25_3b_base"  # Training folder name; this model is INSTRUCT + RL.
        RL_STEP = 300
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/eval-base-instruct-rl-amc-math-temp0"
        EVAL_ENV = "/content/lg-qwen-comparison-eval-env"
        CONFIG = "configs/qwen_compare_eval.yaml"
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
        ## Validate the final RL checkpoint and prepare the comparison

        Checks the completion marker, full model shards, and training manifest.
        The starting model must be the pinned instruct checkpoint. No optimizer files are needed.
        If you changed the training destination, edit `RL_ROOT` / `RL_RUN_NAME` above.
        The first run records model selection; reruns refuse to mix different checkpoints.
        '''),
        code('''
        from common.io import write_json
        from common.process import run_logged
        from eval.qwen_comparison import prepare_runs, comparison_rows, result_path, LABELS
        from eval.run_amc_math_eval import comparison_settings

        spec, resolved, execution, suite = comparison_settings(CODE_ROOT / CONFIG)
        eval_root = Path(OUTPUT_ROOT)
        runs = prepare_runs(spec, RL_ROOT, RL_RUN_NAME, RL_STEP, eval_root)
        progress_totals = {name: entry["rows"] for name, entry in suite["datasets"].items()}

        def evaluation_command(run, smoke=False):
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_amc_math_eval",
                       "--model", run["model"], "--run-name", run["run_name"], "--config", CONFIG,
                       "--output-root", run["output_root"], "--bundled-code"]
            if run["revision"] is not None:
                command += ["--revision", run["revision"]]
            return command + (["--smoke"] if smoke else [])

        def show_comparison():
            missing = [LABELS[run["key"]] for run in runs if not result_path(run).is_file()]
            if missing:
                print("Full results pending:", ", ".join(missing))
                return
            import pandas as pd
            from IPython.display import display
            rows = comparison_rows(runs, resolved)
            table = pd.DataFrame(rows)
            write_json(eval_root / "comparison_table.json", {"models": runs, "rows": rows})
            table.to_csv(eval_root / "comparison_table.csv", index=False)
            display(table.style.format({"Accuracy (%)": "{:.1f}", "Mean tokens": "{:.1f}"}))
            print("Comparison:", eval_root / "comparison_table.csv")
            for run in runs:
                print(LABELS[run["key"]], "metrics:", result_path(run))
                print("Generated answers:", result_path(run).with_name(run["run_name"] + "_generations.jsonl"))

        for run in runs:
            print(LABELS[run["key"]], run["model"], run["revision"] or "local checkpoint")
        print("Problems per model:", progress_totals, "Temperature:", resolved["temperature"])
        print("Output:", eval_root)
        '''),
        markdown('''
        ## Install the locked GPU environment

        Python 3.12 / vLLM in a separate environment. Installation may take several minutes.
        '''),
        code('''
        run_logged([sys.executable, str(CODE_ROOT / "scripts/setup_eval_runtime.py"),
                    "--venv", EVAL_ENV], cwd=CODE_ROOT,
                   log_path=eval_root / "logs/setup_console.log")
        run_logged([str(Path(EVAL_ENV) / "bin/python"), "-c",
                    "from vllm import ModelRegistry; "
                    "assert 'Qwen2ForCausalLM' in ModelRegistry.get_supported_archs()"],
                   cwd=CODE_ROOT, log_path=eval_root / "logs/qwen_runtime_check.log")
        '''),
        markdown('''
        ## Optional smoke test — 2 questions per model, 6 responses total

        Saved separately under each model's `smoke/`; excluded from full metrics.
        '''),
        code('''
        RUN_SMOKE = False
        if RUN_SMOKE:
            for run in runs:
                print("Smoke:", LABELS[run["key"]])
                run_logged(evaluation_command(run, smoke=True), cwd=CODE_ROOT,
                           log_path=Path(run["output_root"]) / "logs/smoke_console.log",
                           progress_totals={name: 1 for name in progress_totals}, compact_progress=True)
        else:
            print("Set RUN_SMOKE = True for the optional six-response smoke test.")
        '''),
        markdown('''
        ## Full evaluation — 540 questions per model

        Set `RUN_EVAL = True`. Rerunning resumes completed problems with unchanged code,
        settings, model files, and runtime. Each model has a separate resume journal.
        A failure stops the loop; fix it and rerun this cell. Checkpoint hashing reads the
        model weights from Drive and can take several minutes. The six-row table appears
        after all three models finish. No training occurs.
        '''),
        code('''
        RUN_EVAL = False
        if RUN_EVAL:
            for run in runs:
                print("Full evaluation:", LABELS[run["key"]])
                run_logged(evaluation_command(run), cwd=CODE_ROOT,
                           log_path=Path(run["output_root"]) / "logs/eval_console.log",
                           progress_totals=progress_totals, compact_progress=True)
            show_comparison()
        else:
            print("Set RUN_EVAL = True to evaluate all three models (1,620 responses).")
        '''),
        markdown('''
        ## Read saved results without inference

        `OUTPUT_ROOT/comparison_table.csv` and `.json`: accuracy, correct counts, mean tokens.
        For each of `base`, `instruct`, `instruct_rl`, `<model>/full/results/` contains:
        - `<run_name>.json`: metrics and model / data / decoding metadata.
        - `<run_name>_generations.jsonl`: full prompts, responses, token counts, and scores.
        - `<run_name>_problems.jsonl`: resumable per-problem journal.
        - `<run_name>_manifest.json`, `<run_name>_engine.json`, and protocol/runtime records.

        Scoring uses the last final box and the existing math-equivalence scorer;
        a missing/invalid final box counts as incorrect. Review saved responses for grading issues.
        Native chat templates can differ for base vs instruct; their digests are recorded.
        '''),
        code("show_comparison()"),
    ]


if __name__ == "__main__":
    write_notebook("evaluate_qwen25_3b_base_instruct_rl_amc_math.ipynb", cells())
