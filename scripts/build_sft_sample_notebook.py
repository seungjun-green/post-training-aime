"""Build a self-contained Colab notebook for SFT/Qwen output inspection."""

from build_notebooks import ROOT, code, markdown, payload, write_notebook


def cells():
    paths = sorted({
        *ROOT.glob("common/*.py"), *ROOT.glob("eval/*.py"),
        ROOT / "train/__init__.py", ROOT / "train/sft_data.py",
        ROOT / "scripts/setup_eval_runtime.py", ROOT / "requirements-eval.lock",
        *(ROOT / name for name in [
            "configs/sft_sample15.yaml", "configs/stage1_sft.yaml",
            "configs/eval_english.yaml", "configs/eval_execution.yaml",
            "configs/english_eval_suite.json", "configs/eval_profiles.yaml",
        ]),
    })
    return [
        markdown("""
        # Inspect EXAONE s1 SFT and Qwen2.5-3B-Instruct — the same 15 random problems

        Automatically runs **both models in sequence**: **sft_s1k/epoch_5**, trained
        on the original DeepSeek reasoning + answer columns, followed by the published
        **Qwen/Qwen2.5-3B-Instruct** model.
        Samples **5 AIME problems from 2024–2026 combined**, **5 AMC 2023**,
        and **5 MATH-500**, without replacement, with seed 42. The AIME years need not
        be equally represented. Each problem gets one response. With the same seed,
        both models receive the same questions and English instruction, rendered with
        each model's native chat template. Only one model is loaded on the GPU at a time;
        its subprocess exits before the next starts. Saves 15 responses per model and
        a combined 30-response JSONL. No model selection or second manual run is needed.

        Defaults: **temperature 0, 20,480 output tokens, no budget forcing**.
        This records natural model behavior, including unfinished thinking at the token
        limit. It uses the existing English prompt, continuous vLLM batching and scorer.
        This is a small qualitative sample, not a full benchmark evaluation.

        Use the **RTX PRO 6000 96GB** Colab runtime and an **HF_TOKEN** secret.
        Code and configs are bundled: no GitHub update is needed. No training or model
        uploads occur. Responses, selected questions, settings and console logs go to Drive.
        """),
        code('''
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"  # Used only for "sft"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-Sample15"
        TEMPERATURE = 0.0
        TOP_P = 1.0
        SEED = 42
        MAX_NEW_TOKENS = 20480  # Reasoning + answer combined; no forced answer phase
        CODE_ROOT = "/content/lg-aime-sft-sample15"
        EVAL_ENV = "/content/lg-eval-env"
        '''),
        markdown("## 1. Mount Drive and install the bundled evaluation runtime"),
        code(f'''
        import base64, io, json, os, subprocess, sys, zipfile
        from pathlib import Path
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        Path(CODE_ROOT).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload(paths)!r}))) as bundle:
            bundle.extractall(CODE_ROOT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22", "PyYAML==6.0.3"])
        subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", EVAL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        '''),
        markdown("""
        ## 2. Prepare both models and preview the same 15 questions

        For `"sft"`, TRAIN_ROOT must contain `checkpoints/stage1/sft_s1k/epoch_5` and its original
        `logs/stage1/sft_s1k/run_manifest.json`. This is the original EXAONE SFT,
        not the Pro-column, Llama or DAPO model. For `"qwen"`, TRAIN_ROOT is ignored;
        the pinned Hugging Face model is downloaded when generation starts.
        No baseline results are required. Results go to `OUTPUT_ROOT/<model_kind>/<run-id>/`.
        This cell downloads the small benchmark datasets but does not load the model.
        """),
        code('''
        import yaml
        from copy import deepcopy
        base_settings = yaml.safe_load((Path(CODE_ROOT) / "configs/sft_sample15.yaml").read_text())
        base_settings.update(seed=SEED, max_new_tokens=MAX_NEW_TOKENS)
        base_settings["sampling"] = {"temperature": TEMPERATURE, "top_p": TOP_P}
        Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
        runs = []
        selection = None
        for model_kind in ["sft", "qwen"]:
            settings = deepcopy(base_settings)
            settings["model_kind"] = model_kind
            settings_path = Path(OUTPUT_ROOT) / f"requested_settings_{model_kind}.yaml"
            settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.sample_sft_outputs",
                       "--config", str(settings_path), "--train-root", TRAIN_ROOT,
                       "--output-root", OUTPUT_ROOT]
            run_logged(command + ["--prepare-only"], cwd=CODE_ROOT,
                       log_path=Path(OUTPUT_ROOT) / f"prepare_{model_kind}_console.log")
            run_dir = Path(json.loads((Path(OUTPUT_ROOT) / "latest_run.json").read_text())["directory"])
            selected = json.loads((run_dir / "selection.json").read_text())
            if selection is not None and selected != selection:
                raise ValueError("Both models must receive exactly the same questions")
            selection = selected
            runs.append({"model_kind": model_kind, "command": command, "run_dir": run_dir})
            print("Model:", json.loads((run_dir / "settings.json").read_text())["model"])
            print("JSONL destination:", run_dir / "generations.jsonl")
        progress_totals = {}
        for row in selection:
            name = row["dataset"]
            progress_totals[name] = progress_totals.get(name, 0) + 1
            print(f"{name} / {row['id']}: {row['question'][:160]}")
        print("Counts per model:", progress_totals, "Total per model:", len(selection))
        print("Sampling:", base_settings["sampling"], "Budget forcing: off")
        '''),
        markdown("""
        ## 3. Run both models — 15 problems each, 30 responses total

        Runs EXAONE SFT first, then Qwen automatically. Completed problems are flushed to
        `problems.jsonl` on Drive; rerunning resumes with the same settings/checkpoint.
        Settings changes create a separate run folder. The process exports a flat
        `generations.jsonl` when it finishes (also on a normal Python exception).
        A hard runtime disconnect leaves the durable journal; rerun to recover the export.
        """),
        code('''
        from common.io import digest, write_json, write_jsonl
        records_by_model = {}
        for run in runs:
            model_kind, run_dir = run["model_kind"], run["run_dir"]
            print("Starting:", model_kind)
            run_logged(run["command"], cwd=CODE_ROOT, log_path=run_dir / "generation_console.log",
                       progress_totals=progress_totals, compact_progress=True)
            generation_path = run_dir / "generations.jsonl"
            records = [json.loads(line) for line in generation_path.read_text().splitlines()]
            assert len(records) == len(selection) == 15
            expected = {(r["dataset"], str(r["id"])) for r in selection}
            assert {(r["dataset"], str(r["id"])) for r in records} == expected
            assert all(r["source"]["model_kind"] == model_kind for r in records)
            records_by_model[model_kind] = records
            for row in records:
                print(model_kind, row["dataset"], row["id"], "correct:", row["correct"],
                      "tokens:", row["token_count"], "finish:", row["finish_reason"])
            print("Saved:", generation_path)
        sources = {run["model_kind"]: str(run["run_dir"]) for run in runs}
        combined_dir = Path(OUTPUT_ROOT) / "both" / digest(sources)[:16]
        combined_path = combined_dir / "both_models_generations.jsonl"
        combined = [row for records in records_by_model.values() for row in records]
        write_jsonl(combined_path, combined)
        write_json(combined_dir / "sources.json", sources)
        print("Both models finished. Saved", len(combined), "responses:", combined_path)
        '''),
        markdown("""
        ## 4. Read both outputs for one question and download the combined JSONL

        `response` is the exact full generation; `reasoning` is the text inside `<think>`;
        `answer_text` is everything after `</think>`; `gold_answer` is the reference answer.
        Missing delimiters are flagged with `parse_status`, with unsplittable fields null.
        Qwen responses without think tags use `parse_status="plain_response"`,
        `reasoning=null` and `answer_text=response` (the complete worked solution).
        They are not artificially split into hidden reasoning and final answer.
        `finish_reason` / `truncated` identify output-limit hits. The original response is
        never shortened in the JSONL. Correctness uses the existing boxed-answer scorer.
        """),
        code('''
        EXAMPLE_INDEX = 0
        selected = selection[EXAMPLE_INDEX]
        print("QUESTION:", selected["question"])
        print("REFERENCE:", selected["gold_answer"])
        for model_kind, records in records_by_model.items():
            row = next(r for r in records if r["dataset"] == selected["dataset"]
                       and str(r["id"]) == str(selected["id"]))
            print(f"\\nFULL MODEL RESPONSE ({model_kind}):")
            print(row["response"])
        '''),
        code('''
        DOWNLOAD_JSONL = True
        if DOWNLOAD_JSONL:
            from google.colab import files
            files.download(str(combined_path))
        '''),
    ]


if __name__ == "__main__":
    write_notebook("sample_original_sft_15.ipynb", cells())
