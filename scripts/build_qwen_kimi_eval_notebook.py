"""Build self-contained greedy AMC/MATH evaluation for Qwen base + Kimi-style SFT."""

import sys

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

sys.path.insert(0, str(ROOT))
from eval.qwen_kimi_sft import BUNDLE_FILES  # noqa: E402


def cells():
    boot = bootstrap(payload([ROOT / path for path in BUNDLE_FILES]))
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-qwen-kimi-eval")
    return [
        markdown(r"""
        # Qwen base + Kimi-style SFT — AMC 2023 and MATH-500, temperature 0

        Evaluates the completed **epoch 5** checkpoint from **Qwen2.5-3B base**, trained for five epochs
        on the complete **`kimi-style-reasoning-answer`** column (roadmap B-SFT2). **AMC 2023: 40 problems; MATH-500: 500 problems.**
        Greedy decoding: **temperature 0, top-p 1, one answer per problem**, seed 42.
        The response limit is 20,480 tokens. Uses the training/evaluation English prompt,
        the checkpoint's saved chat template and `<|im_end|>` stop token,
        and the same final-box/math-equivalence scorer as the earlier Qwen comparison.
        Generation is a single response with no think-tag parsing or budget forcing.

        Use a **fresh RTX PRO 6000 Blackwell 96GB Colab runtime** and enable `HF_TOKEN`
        in Secrets. Code/config are embedded in this notebook; no GitHub checkout or
        prior baseline evaluation is required. Model weights are read directly from Drive.
        Only AMC/MATH are loaded and evaluated. Outputs include accuracy, average response
        tokens, full generated answers, and per-problem scores.

        Run setup and checkpoint validation, optionally enable the two-problem smoke test,
        then set `RUN_EVAL = True` for the full 540-problem evaluation.
        """),
        code('''
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments/base-sft-v2-short"
        TRAIN_RUN_NAME = "sft_qwen25_3b_base_s1_kimi"
        EPOCH = 5
        EXPECTED_RUN_IDENTITY = "5644b316cb49344bea968be58871bc87a6bae072f50309bcfba18a1ce6e577ef"
        EVAL_ENV = "/content/lg-qwen-kimi-eval-env"
        # None saves under TRAIN_ROOT/eval/amc2023_math500_temp0/TRAIN_RUN_NAME/epoch_N.
        OUTPUT_ROOT = None
        CONFIG = "configs/qwen_kimi_sft_eval.yaml"
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
        ## Select the trained checkpoint

        Checks its epoch marker and training manifest to confirm it belongs to the
        Qwen base + Kimi-style SFT run and the training identity you reported.
        Optimizer files are not required for evaluation. If your output folder differs from
        the defaults, edit the settings cell and rerun this cell.
        '''),
        code('''
        from common.io import write_json
        from common.process import run_logged
        from eval.qwen_kimi_sft import checkpoint_source
        from eval.run_amc_math_eval import comparison_settings

        spec, resolved, execution, suite = comparison_settings(CODE_ROOT / CONFIG)
        checkpoint, marker = checkpoint_source(
            TRAIN_ROOT, TRAIN_RUN_NAME, EPOCH, spec, EXPECTED_RUN_IDENTITY)
        eval_root = (Path(OUTPUT_ROOT) if OUTPUT_ROOT is not None else
                     Path(TRAIN_ROOT) / "eval/amc2023_math500_temp0" / TRAIN_RUN_NAME / f"epoch_{EPOCH}")
        run_name = f"{TRAIN_RUN_NAME}_epoch{EPOCH}"
        progress_totals = {name: entry["rows"] for name, entry in suite["datasets"].items()}
        result_path = eval_root / "full/results" / (run_name + ".json")
        write_json(eval_root / "evaluation_settings.json",
                   {"spec": spec, "config": resolved, "execution": execution,
                    "checkpoint": str(checkpoint), "checkpoint_marker": marker,
                    "expected_run_identity": EXPECTED_RUN_IDENTITY})

        def evaluation_command(smoke=False):
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_amc_math_eval",
                       "--model", str(checkpoint), "--run-name", run_name, "--config", CONFIG,
                       "--output-root", str(eval_root), "--bundled-code"]
            return command + (["--smoke"] if smoke else [])

        def show_results():
            if not result_path.is_file():
                print("Full evaluation is not complete yet:", result_path)
                return
            import pandas as pd
            from IPython.display import display
            result = json.loads(result_path.read_text())
            if (result["model"] != str(checkpoint) or result["mode"] != "full"
                    or result["config"] != resolved or result["run_name"] != run_name):
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
                             "Accuracy (%)": 100 * metric["avg@1"],
                             "Mean tokens": metric["response_length_tokens"]["all"]})
            table_path = eval_root / "full/summary_table.json"
            write_json(table_path, {"model": str(checkpoint), "temperature": 0, "rows": rows})
            table = pd.DataFrame(rows)
            table.to_csv(eval_root / "full/summary_table.csv", index=False)
            display(table.style.format({"Accuracy (%)": "{:.1f}", "Mean tokens": "{:.1f}"}))
            print("Metrics:", result_path)
            print("Table:", table_path)
            print("Generated answers:", result_path.with_name(run_name + "_generations.jsonl"))

        print("Checkpoint:", checkpoint)
        print("Benchmarks:", progress_totals, "Total:", sum(progress_totals.values()))
        print("Temperature:", resolved["temperature"], "One answer per problem; no budget forcing")
        print("Results:", result_path)
        '''),
        markdown('''
        ## Install the GPU evaluation environment

        Uses the project's locked Python 3.12 / vLLM environment in a separate directory.
        This can take several minutes on the first run.
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
        ## Optional smoke test — 2 problems

        One AMC and one MATH-500 problem, saved separately under `smoke/`.
        Smoke results are excluded from the full metrics.
        '''),
        code('''
        RUN_SMOKE = False
        if RUN_SMOKE:
            run_logged(evaluation_command(smoke=True), cwd=CODE_ROOT,
                       log_path=eval_root / "logs/smoke_console.log",
                       progress_totals={name: 1 for name in progress_totals}, compact_progress=True)
        else:
            print("Set RUN_SMOKE = True for the optional two-problem smoke test.")
        '''),
        markdown('''
        ## Full evaluation — 540 problems

        Set `RUN_EVAL = True`. Completed problems are saved to Drive and reused after
        interruption with unchanged model, code, runtime, and settings. To resume in a fresh
        runtime, rerun setup/validation and this cell. Initial model hashing may take a few
        minutes while reading Drive. The two-benchmark summary appears after completion.
        '''),
        code('''
        RUN_EVAL = False
        if RUN_EVAL:
            run_logged(evaluation_command(), cwd=CODE_ROOT,
                       log_path=eval_root / "logs/eval_console.log",
                       progress_totals=progress_totals, compact_progress=True)
            show_results()
        else:
            print("Set RUN_EVAL = True to evaluate the trained checkpoint on all 540 problems.")
        '''),
        markdown('''
        ## Show saved results

        Reads metrics without inference. `full/results/` contains the metrics JSON,
        `_generations.jsonl` (full responses and scores), `_problems.jsonl` (resume journal),
        and model/code/data manifests. `full/summary_table.json` and `.csv` contain the displayed table.
        Accuracy uses the last `\\boxed{...}` and the project's math-equivalence scorer;
        responses without a valid final box count as incorrect.
        '''),
        code("show_results()"),
    ]


if __name__ == "__main__":
    write_notebook("evaluate_qwen25_3b_s1_kimi_amc_math.ipynb", cells())
