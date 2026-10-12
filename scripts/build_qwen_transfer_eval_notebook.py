"""Build the self-contained nine-model cross-domain Colab evaluation notebook."""

import sys

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

sys.path.insert(0, str(ROOT))
from eval.run_transfer_eval import bundle_files  # noqa: E402

DISPLAY_FILES = ['common/eval_progress.py']


def cells():
    # Display-only helpers do not change the scientific code digest or invalidate resumes.
    boot = bootstrap(payload([ROOT / name for name in bundle_files() + DISPLAY_FILES]))
    boot.source = boot.source.replace('/content/lg-korean-aime', '/content/lg-qwen-transfer-eval')
    return [
        markdown('''
        # All nine Qwen2.5-3B variants: cross-domain evaluation

        **ARC-Challenge · MMLU-Pro · non-math BBH · IFEval**, temperature **0**.
        All nine models are selected by default: B0, B-RL, Kimi-style SFT epoch 5,
        rejection SFT epoch 4, 7B-distillation SFT epoch 5, both SFT + RL step 300
        models, I0, and I-RL step 300. The distilled student is still a **3B** model.

        **18,360 prompts/model; 165,240 total.** This full run can take substantial GPU
        time. Models run sequentially in separate processes, loading each model once
        for all four benchmarks. Completed responses are saved immediately to Drive;
        interrupted runs resume independently by model. No training or upload occurs.

        Use a fresh **RTX PRO 6000 96GB Colab GPU runtime**, with `HF_TOKEN` in Secrets.
        Code, prompts, official IFEval scoring, and sentence-tokenizer resources are
        embedded. Benchmark files download from immutable Hugging Face revisions with
        SHA-256 and row-count checks; no GitHub checkout is needed.
        '''),
        markdown('''
        ## Protocol

        | Benchmark | Test prompts/model | Prompting | Main metrics |
        | --- | ---: | --- | --- |
        | ARC-Challenge | 1,172 | Zero-shot, reason then give option letter | Generated-answer accuracy |
        | MMLU-Pro | 12,032 | Five official CoT validation examples from the same subject | Overall, all 14 subjects, and non-math aggregate |
        | Non-math BBH | 4,615 | Three official CoT examples per selected task | Per-task accuracy, task macro and example-weighted aggregate |
        | IFEval | 541 | Zero-shot; original prompt unchanged | Prompt/instruction strict and loose accuracy |

        All generation: **temperature 0, top-p 1, seed 42, one response, max 4,096 new
        tokens, context 32,768**. Truncation and unparsed-answer rates are reported.
        Context overflow raises an error; no silent prompt truncation or few-shot removal.
        Each model uses its saved chat template and EOS/PAD. The historical Kimi SFT model
        retains its saved `<|im_end|>` EOS; the other base-trained checkpoints retain
        `<|endoftext|>`. No tokenizer or token changes are made.

        This is an explicitly **generative chat-formatted comparison**. ARC scores are
        not likelihood-based `acc_norm`. MMLU/BBH use official demonstrations with a
        deterministic final-answer extractor; missing answers are incorrect, never
        randomly guessed. IFEval uses the unmodified official checks on the full response;
        no reasoning prefix, math instruction, answer tag, or stop string is added.

        **BBH is a project-defined non-math subset**, not the full BBH benchmark.
        Included: boolean expressions, causal judgement, disambiguation QA, Dyck languages,
        formal fallacies, hyperbaton, logical deduction (3/5/7 objects), movie recommendation,
        ruin names, translation-error detection, snarks, sports understanding, shuffled
        objects (3/5/7), web of lies, and word sorting (19 configurations).
        Excluded: arithmetic, object counting, geometric shapes, date understanding,
        navigation, penguin tables, colored-object reasoning, and temporal sequences.
        These omit quantitative, spatial and temporal tasks; the macro weights each of
        the 19 included configurations equally. MMLU's math subject is retained but shown
        separately, with an additional aggregate over the other 13 subjects.

        Sources: [ARC](https://huggingface.co/datasets/allenai/ai2_arc),
        [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro),
        [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard),
        [official IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval).
        Exact revisions, file hashes, licenses and extraction rules are bundled.
        '''),
        code('''
        EXPERIMENT_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments"
        INSTRUCT_RL_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix"
        OUTPUT_ROOT = EXPERIMENT_ROOT + "/eval-all9-transfer-temp0"
        EVAL_ENV = "/content/lg-qwen-transfer-eval-env"
        CONFIG = "configs/qwen_transfer_eval.yaml"
        MODEL_KEYS = None  # None = all nine, in the recorded roadmap order.
        # To run a subset now, e.g.: MODEL_KEYS = ["base", "instruct"]
        # Others: base_rl, kimi_sft, rejection_sft, distill_sft,
        #         rejection_rl, distill_rl, instruct_rl
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
        ## Check the selected checkpoints

        Verifies experiment/epoch/step, training manifest identities, full weight shards,
        saved EOS and chat templates. Uses only the seven selected checkpoints and their
        training manifests; optimizer files and older parent weights are not required.
        Prints the exact model paths before any inference. A missing checkpoint stops
        this cell with its path; use `MODEL_KEYS` above to run available models first.
        '''),
        code('''
        from common.process import run_logged
        from common.eval_progress import run_eval_logged
        from eval.run_transfer_eval import load_config
        from eval.transfer_models import prepare_models

        config = load_config(CODE_ROOT / CONFIG)
        runs = prepare_models(config, EXPERIMENT_ROOT, INSTRUCT_RL_ROOT, OUTPUT_ROOT, MODEL_KEYS)
        for run in runs:
            print(run["key"], "=>", run["model"], run["revision"] or "saved checkpoint")
        print("Results:", OUTPUT_ROOT)

        def command(*extra):
            return [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_transfer_eval",
                    "--config", CONFIG, "--output-root", OUTPUT_ROOT,
                    "--experiment-root", EXPERIMENT_ROOT, "--instruct-rl-root", INSTRUCT_RL_ROOT,
                    *extra]
        '''),
        markdown('''
        ## Install the pinned GPU/scoring environment

        Uses a separate Python 3.12 environment and the existing vLLM 0.14.1 GPU lock,
        with pinned IFEval dependencies. Verifies CUDA and runs an offline scorer check.
        Installation can take several minutes.
        '''),
        code('''
        run_logged([sys.executable, str(CODE_ROOT / "scripts/setup_transfer_runtime.py"),
                    "--venv", EVAL_ENV], cwd=CODE_ROOT,
                   log_path=Path(OUTPUT_ROOT) / "logs/setup_console.log")
        '''),
        markdown('''
        ## Prepare the fixed benchmark data

        Downloads only the pinned test data and MMLU validation demonstrations.
        Writes `data/documents.jsonl` and `data/manifest.json`. Subsequent runs reuse the
        verified prepared data. Different protocol/code/data requires a new output root.
        '''),
        code('''
        run_logged(command("--prepare-only"), cwd=CODE_ROOT,
                   log_path=Path(OUTPUT_ROOT) / "logs/data_console.log")
        data_manifest = json.loads((Path(OUTPUT_ROOT) / "data/manifest.json").read_text())
        totals = data_manifest["counts"]
        print("Full prompts per model:", totals, "Total:", sum(totals.values()))

        def show_results():
            import pandas as pd
            from IPython.display import display
            path = Path(OUTPUT_ROOT) / "comparison.json"
            if not path.is_file():
                print("No full results yet.")
                return
            result = json.loads(path.read_text())
            print("Completed models:", result["completed_models"], "/ 9")
            print("Pending:", result["pending_models"])
            if not result["rows"]:
                return
            display(pd.DataFrame(result["rows"]).set_index("Model").drop(columns=["Key"]).round(2))
            details = pd.DataFrame(result["breakdowns"])
            for benchmark in ["mmlu_pro", "bbh_nonmath"]:
                part = details[(details["Benchmark"] == benchmark) &
                               (~details["Subject/task"].isin(["overall", "non_math_overall", "task_macro"]))]
                print(benchmark, "subject/task accuracy (%)")
                display(part.pivot(index="Subject/task", columns="Model", values="Accuracy (%)").round(2))
            print("Comparison:", Path(OUTPUT_ROOT) / "comparison.csv")
            print("Breakdowns/tokens/truncation:", Path(OUTPUT_ROOT) / "subject_task_breakdowns.csv")
        '''),
        markdown('''
        ## Optional smoke — 35 prompts per selected model

        One ARC and IFEval prompt, plus one per MMLU subject and selected BBH task.
        Uses the full 4,096-token cap. Smoke results live separately and never enter
        full benchmark scores. Set `RUN_SMOKE = True` to check loading/generation/scoring.
        '''),
        code('''
        RUN_SMOKE = False
        if RUN_SMOKE:
            for index, run in enumerate(runs, 1):
                run_eval_logged(command("--model-key", run["key"], "--smoke"), cwd=CODE_ROOT,
                           log_path=Path(OUTPUT_ROOT) / "models" / run["key"] / "logs/smoke_console.log",
                           progress_totals={"arc_challenge": 1, "mmlu_pro": 14, "bbh_nonmath": 19, "ifeval": 1},
                           label=f"Smoke {index}/{len(runs)} · {run['label']}")
        else:
            print("Set RUN_SMOKE = True for the optional smoke run.")
        '''),
        markdown('''
        ## Full evaluation — run/resume selected models

        Set `RUN_EVAL = True`. Each model gets one compact tqdm bar showing its current
        benchmark, completed/total problems, remaining problems, elapsed time, estimated
        time remaining and processing speed. Startup/compilation messages and repeated
        generation heartbeats are saved to `models/<key>/logs/eval_console.log` instead
        of flooding the cell. Errors still include their diagnostic tail and log path.
        ETA starts after model loading, measures newly generated work, and can vary as
        the model moves between benchmarks. Previously saved responses count as completed.

        Rerunning skips saved responses; a completed model is checked and re-summarized
        without loading its weights onto the GPU. Reading/hashing model files from Drive
        can take several minutes. A failure stops the loop; fix it and rerun this cell.
        Completed earlier models and individual responses are retained.

        The saved comparison tables update after every finished model; the notebook
        displays the tables once after the selected models finish. This display-only
        update preserves compatibility with existing saved evaluation responses.
        Full results do not
        mix with smoke, and changed code/settings/model/data/runtime cannot silently resume.
        '''),
        code('''
        RUN_EVAL = False
        if RUN_EVAL:
            for index, run in enumerate(runs, 1):
                run_eval_logged(command("--model-key", run["key"]), cwd=CODE_ROOT,
                           log_path=Path(OUTPUT_ROOT) / "models" / run["key"] / "logs/eval_console.log",
                           progress_totals=totals, label=f"Model {index}/{len(runs)} · {run['label']}")
                run_logged(command("--summarize"), cwd=CODE_ROOT,
                           log_path=Path(OUTPUT_ROOT) / "logs/summary_console.log")
            show_results()
        else:
            print("Set RUN_EVAL = True to evaluate the selected models.")
        '''),
        markdown('''
        ## Read/export saved results

        Under `OUTPUT_ROOT`:

        - `comparison.csv` / `.json`: overall metrics across completed models, with pending models listed.
        - `subject_task_breakdowns.csv`: every MMLU subject and BBH task, IFEval metrics,
          mean output tokens, truncation and parse-failure rates.
        - `models/<key>/selection.json`: exact selected checkpoint/revision.
        - `models/<key>/full/results.json` and `metrics.csv`: model results.
        - `models/<key>/full/responses.jsonl`: durable raw-response resume journal.
        - `models/<key>/full/<benchmark>/generations.jsonl`: original task prompt,
          chat-formatted prompt, response, token count, finish reason and scores.
        - `models/<key>/full/<benchmark>/metrics.json`: benchmark breakdowns.
        - `models/<key>/full/manifest.json` and `tokenizer_identity.json`: protocol,
          model/data/code/runtime and tokenizer identities.
        - `models/<key>/logs/`: evaluation/smoke console logs. Smoke uses `smoke/` in place of `full/`.

        IFEval instruction accuracy uses the number of individual constraints as its
        denominator, not 541 prompts. Missing/unparseable MC answers count as incorrect.
        Review truncation and parse-failure rates alongside accuracy before drawing conclusions.
        '''),
        code('''
        run_logged(command("--summarize"), cwd=CODE_ROOT,
                   log_path=Path(OUTPUT_ROOT) / "logs/summary_console.log")
        show_results()
        '''),
    ]


if __name__ == '__main__':
    write_notebook('evaluate_qwen25_3b_all9_transfer.ipynb', cells())
