"""Build separate Stage 1 training and evaluation launchers."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Stage 1 — English s1-style SFT on EXAONE
        Use the **RTX PRO 6000 Blackwell 96GB** runtime and enable the `HF_TOKEN` secret.
        Push the Stage 1 implementation to your GitHub repository before running setup.
        This notebook uses a separate training environment and preserves the frozen evaluation
        environment. All training settings are in `configs/stage1_sft_deepseek_pro.yaml`.
        This starts from the original EXAONE base model and trains on the new Pro thinking
        **plus** answer. Hyperparameters are unchanged from the original five-epoch SFT run.
        The published HF dataset must contain the two Pro columns before you run preparation.
        Its current revision is resolved to an immutable commit and recorded in the run manifest.
        Checkpoints use the separate run name `sft_s1k_deepseek_pro`.
        This notebook performs SFT and saves all five epoch checkpoints. It never runs evaluation.
        A default Run all prepares and inspects data only; enable `RUN_TRAINING` to train.
        Keep enough Drive space for five full model/optimizer checkpoints (allow at least 100GB).
        For later evaluation, set the evaluation notebook's CONFIG to
        `configs/stage1_sft_deepseek_pro.yaml` to select these new checkpoints.
        """),
        code("""
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-stage1"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"
        TRAIN_ENV = "/content/lg-sft-env"
        CONFIG = "configs/stage1_sft_deepseek_pro.yaml"
        """),
        code("""
        import os, subprocess, sys
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
        print("Code commit:", git("rev-parse", "HEAD"))
        if not (Path(CODE_ROOT) / CONFIG).is_file():
            raise FileNotFoundError(
                f"The GitHub checkout does not contain {CONFIG}. Commit and push the Pro SFT "
                "update (including configs/stage1_sft_deepseek_pro.yaml, train/sft_data.py, "
                "and train/stage1_sft.py) to main, then rerun this setup cell. "
                "Uploading the notebook alone does not update the cloned training code."
            )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_stage1_runtime.py", "--venv", TRAIN_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        TRAIN_COMMAND = [str(Path(TRAIN_ENV) / "bin/python"), "train/stage1_sft.py",
                         "--config", CONFIG, "--output_root", TRAIN_ROOT]
        """),
        markdown("""
        ## Data and loss-mask inspection
        This reads all 996 published rows without loading model weights. It skips missing Pro
        outputs and tokenizes the new reasoning plus answer with the existing English prompt
        and native chat template. Review the printed full text and masked text. Full sequences
        longer than 20,480 tokens are dropped, never truncated. For this uploaded export, expect
        718 retained rows, six missing-output exclusions, and 272 overlength exclusions.
        Old grades describe the old answers and never filter training. Training repeats these
        checks before the first optimizer step.
        """),
        code("""
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / "prepare_console.log")
        """),
        markdown("""
        ## Five-epoch full fine-tuning
        Each complete epoch is saved with its tokenizer, model code, optimizer, scheduler and RNG.
        Set `RESUME_CHECKPOINT` to the latest complete epoch directory to resume an interrupted run.
        Keep the same code commit and config. An incomplete epoch is replayed from its start.
        An existing run will not be silently overwritten. Do not rerun setup during training.
        """),
        code("""
        RUN_TRAINING = False
        RESUME_CHECKPOINT = None
        if RUN_TRAINING:
            command = TRAIN_COMMAND + (["--resume_from_checkpoint", RESUME_CHECKPOINT] if RESUME_CHECKPOINT else [])
            run_logged(command, cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / "training_console.log")
        else:
            print("Training is off. Review preparation, then enable RUN_TRAINING.")
        """),
    ]


def eval_cells():
    return [
        markdown("""
        # Stage 1 — Evaluate epoch 5 first
        This notebook evaluates existing checkpoints; it never trains a model.
        Epoch 5 is evaluated on **AIME 2024, AIME 2025, AIME 2026, AMC 2023,
        and MATH-500** using continuous batching. Choose the same `EVAL_PROFILE` as your baseline:
        - **greedy** (default): temperature 0, one answer per problem, **630 responses**, pass@1.
        - **sample8**: temperature 1.0/top-p 0.7, eight answers per AIME/AMC problem and
          four per MATH-500 problem, **3,040 responses**. Reports pass@1/4/8 on AIME/AMC
          and pass@1/4 on MATH-500.
        The default checkpoint selection is **epoch 5 only**. Optionally enable
        `INCLUDE_EARLIER_EPOCHS` to run epochs 1–4 after epoch 5.
        Run this after the SFT notebook has saved the final checkpoint to Drive.
        Use the **RTX PRO 6000 Blackwell 96GB** runtime and enable the `HF_TOKEN` secret.
        Push the notebook changes to your GitHub repository before running setup.
        Evaluation runs sequentially in the original locked evaluation environment, releasing
        GPU memory after each checkpoint. No training environment is installed here.
        """),
        code("""
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-stage1-eval"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"
        BASELINE_ROOT = "/content/drive/MyDrive/LG-AIME-English-Eval-compatible"
        EVAL_ENV = "/content/lg-eval-env"
        CONFIG = "configs/stage1_sft.yaml"
        EVAL_PROFILE = "greedy" # @param ["greedy", "sample8"]
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", EVAL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        """),
        markdown("""
        ## Check saved checkpoints
        Keep `TRAIN_ROOT` the same as in the training notebook. SFT evaluation can run before
        the baseline. The first run creates the selected profile's protocol/runtime manifests
        under `BASELINE_ROOT/profiles/<profile>/full/results/`. Later runs validate the same
        sampling, prompts, datasets, scoring and runtime for comparisons within that profile.
        To switch from an older notebook, stop evaluation, push the updated code, and run setup.
        New profiles start fresh and leave previous 32-sample results intact; old responses are
        not imported. Once a run starts, keep its code commit and settings unchanged for resume.
        """),
        code("""
        import yaml
        from eval.profiles import profile_root
        INCLUDE_EARLIER_EPOCHS = False
        cfg = yaml.safe_load((Path(CODE_ROOT) / CONFIG).read_text())
        epochs = int(cfg["training"]["num_train_epochs"])
        selected_epochs = [epochs]
        if INCLUDE_EARLIER_EPOCHS:
            selected_epochs.extend(range(1, epochs))
        checkpoint_root = Path(TRAIN_ROOT) / "checkpoints" / cfg["stage"] / cfg["run_name"]
        result_root = profile_root(BASELINE_ROOT, EVAL_PROFILE) / "full/results"
        run_identities = set()
        eval_runs = []
        for epoch in selected_epochs:
            checkpoint = checkpoint_root / f"epoch_{epoch}"
            marker = json.loads((checkpoint / "stage1_checkpoint.json").read_text())
            if marker["epoch"] != epoch:
                raise ValueError(f"Checkpoint epoch mismatch: {checkpoint}")
            run_identities.add(marker["run_identity"])
            # Profiles use separate directories for baseline and SFT comparisons.
            run_name = cfg["run_name"] if epoch == epochs else f"{cfg['run_name']}_epoch{epoch}"
            eval_runs.append((epoch, checkpoint, run_name))
        if len(run_identities) != 1:
            raise ValueError("Selected epoch checkpoints must belong to the same training run")
        suite = json.loads((Path(CODE_ROOT) / "configs/english_eval_suite.json").read_text())
        progress_totals = {name: entry["rows"] for name, entry in suite["datasets"].items()}
        print("Benchmarks:", ", ".join(progress_totals))
        for epoch, checkpoint, run_name in eval_runs:
            print(f"Epoch {epoch}: {checkpoint} -> {result_root / cfg['stage'] / (run_name + '.json')}")
        """),
        markdown("""
        ## Full evaluation — epoch 5, then optional earlier checkpoints
        Enable `RUN_STAGE1_EVAL` to evaluate epoch 5. Leave `INCLUDE_EARLIER_EPOCHS = False`
        above for the main result only. To include the earlier checkpoints, set it to `True`
        and rerun the check cell; the evaluation order will be **5, 1, 2, 3, 4**.
        A default Run all performs setup and checks only. Every selected epoch uses all five benchmarks, with the same
        prompts, sampling, seeds and scoring as the baseline. Epoch 5 remains the Stage 1 result.
        Reports, raw generations and manifests are saved under `BASELINE_ROOT/profiles/<profile>/full/results/stage1/`:
        `sft_s1k_epoch1.json` through `sft_s1k_epoch4.json`, and `sft_s1k.json` for epoch 5. Completed problems are durably saved during the run
        in `*_problems.jsonl`; the standard `*_generations.jsonl` export is written at completion.
        Console logs are saved under `TRAIN_ROOT`. Repeating the same commands resumes completed
        problems. `configs/eval_profiles.yaml` defines both options and queues up to 64 greedy
        or 16 sampled problems to feed the existing 32 GPU response slots. Exact sampled text can
        change with batching even though seeds and sampling settings are unchanged.
        Progress counts completed problems, which can finish out of order within each benchmark.
        No AMC-only runs are launched by this notebook.
        """),
        code("""
        RUN_STAGE1_EVAL = False
        if RUN_STAGE1_EVAL:
            eval_python = str(Path(EVAL_ENV) / "bin/python")
            for epoch, checkpoint, run_name in eval_runs:
                command = [eval_python, "-m", "eval.run_batched_eval", "--model", str(checkpoint),
                           "--stage", cfg["stage"], "--run_name", run_name, "--output_root", BASELINE_ROOT,
                           "--profile", EVAL_PROFILE,
                           "--execution_config", "configs/eval_execution.yaml"]
                run_logged(command,
                           cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / f"full_eval_epoch{epoch}_{EVAL_PROFILE}_console.log",
                           progress_totals=progress_totals)
        else:
            print("Evaluation is off. Enable RUN_STAGE1_EVAL to run the selected checkpoints, starting with epoch 5.")
        """),
    ]


def sample1_eval_cells():
    """Reuse the checkpoint/evaluation workflow with single-sample stochastic decoding."""
    notebook_cells = eval_cells()
    notebook_cells[0] = markdown("""
        # Stage 1 — DeepSeek Pro SFT epoch 5 evaluation
        Uses `configs/stage1_sft_deepseek_pro.yaml` and loads the checkpoint from
        `TRAIN_ROOT/checkpoints/stage1/sft_s1k_deepseek_pro/epoch_5/`.
        Evaluate **epoch 5** on **AIME 2024, AIME 2025, AIME 2026, AMC 2023, and MATH-500**.
        The `sample1` profile generates exactly **one answer per problem** at **temperature 1.0**
        and **top-p 0.7**: **630 responses per checkpoint**, reporting accuracy/pass@1.
        Top-p matches the original sampled baseline; the output limit remains **20,480 tokens**.
        This means one sampled answer, not one few-shot demonstration in the prompt.

        Epoch 5 is selected by default. Enable `INCLUDE_EARLIER_EPOCHS` to evaluate epochs 1–4
        afterward. Existing checkpoints are loaded from Drive; this notebook does not train.
        Use the **RTX PRO 6000 Blackwell 96GB** runtime and enable the `HF_TOKEN` secret.
        Push the new notebook, profile config, and profile-loader changes to the configured
        GitHub repository before running setup. Setup clones/updates that repository.

        Results are saved as `profiles/sample1/full/results/stage1/sft_s1k_deepseek_pro.json`,
        separately from the earlier SFT run. Console logs also use the new run name.
        A baseline run is **not required** to start. The same `sample1` profile is also available
        to the baseline CLI when you want a matched comparison. Default Run all performs setup
        and checkpoint checks; enable `RUN_STAGE1_EVAL` in the final cell to generate answers.
        """)
    for cell in notebook_cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace(
                'CONFIG = "configs/stage1_sft.yaml"',
                'CONFIG = "configs/stage1_sft_deepseek_pro.yaml"',
            ).replace(
                'f"full_eval_epoch{epoch}_{EVAL_PROFILE}_console.log"',
                'f"{run_name}_epoch{epoch}_{EVAL_PROFILE}_console.log"',
            ).replace(
                'print("Evaluation code commit:", git("rev-parse", "HEAD"))',
                'print("Evaluation code commit:", git("rev-parse", "HEAD"))\n'
                'if not (Path(CODE_ROOT) / CONFIG).is_file():\n'
                '    raise FileNotFoundError(f"Missing {CONFIG}. Push the Pro SFT updates to GitHub main, then rerun setup.")',
            )
            cell.source = cell.source.replace(
                'EVAL_PROFILE = "greedy" # @param ["greedy", "sample8"]',
                'EVAL_PROFILE = "sample1"  # temperature 1.0; one answer on every benchmark',
            )
        elif cell.source.startswith("## Full evaluation"):
            cell.source = cell.source.replace("sft_s1k", "sft_s1k_deepseek_pro")
            cell.source = cell.source.replace(
                "`configs/eval_profiles.yaml` defines both options and queues up to 64 greedy\n"
                "or 16 sampled problems to feed the existing 32 GPU response slots.",
                "`configs/eval_profiles.yaml` defines `sample1`: temperature 1.0, top-p 0.7,\n"
                "one answer per problem, and up to 64 pending problems for 32 GPU response slots.",
            )
    return notebook_cells


def budget_eval_cells():
    notebook_cells = sample1_eval_cells()
    notebook_cells[0] = markdown("""
        # Stage 1 — Epoch 5 evaluation with budget forcing
        Default checkpoint: `TRAIN_ROOT/checkpoints/stage1/sft_s1k_deepseek_pro/epoch_5/`.
        Evaluate **AIME 2024, AIME 2025, AIME 2026, AMC 2023, and MATH-500** with
        **temperature 1.0, top-p 0.7, one answer per problem** (630 responses).

        Set **MAX_THINKING_TOKENS** in the settings cell. Its default is **18,432**;
        the answer allowance is automatically **20,480 − MAX_THINKING_TOKENS** (default **2,048**).
        Minimum reasoning stays zero. At the cap, append `</think>`
        and `Final Answer:` and continue the same assistant response. A natural `</think>`
        starts the answer phase earlier; natural EOS is respected. No "Wait" extension is used.
        The answer allowance includes injected delimiter/cue tokens, keeping the full continuation
        within **20,480 tokens**. The first-phase count includes the generated opening think marker.

        Each budget gets a separate folder, by default
        `profiles/sample1_budget/thinking_18432_answer_2048/`. Changing the cap starts a separate
        evaluation; rerun the checkpoint-check cell after editing settings. The resolved budget
        YAML is saved to Drive under `TRAIN_ROOT/eval_configs/` and used by both smoke and full runs.
        Raw records include whether the transition was forced and each phase's token counts.
        Training, prompts, benchmarks, and answer scoring are unchanged. No baseline is required.

        Use the **RTX PRO 6000 Blackwell 96GB**, enable `HF_TOKEN`, and push the updated Python
        files/configs to GitHub main before setup. Run the optional five-problem smoke cell first,
        then enable the separate full evaluation cell. Default Run all performs setup/checks only.
        Epoch 5 is selected by default; earlier epochs remain optional.

        To test the **previous R1 SFT checkpoint** with the same budget forcing, change CONFIG
        below to `configs/stage1_sft.yaml`; this selects `sft_s1k/epoch_5` and a separate result name.
        To run without budget forcing, select `sample1` and use a fresh BASELINE_ROOT if that
        profile already has manifests from an older evaluator version. Existing runs require
        their original code/settings to resume; old generations are never imported here.
        """)
    for cell in notebook_cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace(
                'EVAL_PROFILE = "sample1"  # temperature 1.0; one answer on every benchmark',
                'EVAL_PROFILE = "sample1_budget" # @param ["sample1_budget", "sample1"]\n'
                'MAX_THINKING_TOKENS = 18432 # @param {type:"integer"}',
            )
            cell.source = cell.source.replace(
                'if not (Path(CODE_ROOT) / CONFIG).is_file():',
                'if (not (Path(CODE_ROOT) / CONFIG).is_file()\n'
                '        or not (Path(CODE_ROOT) / "eval/budget_generation.py").is_file()):',
            ).replace(
                'f"Missing {CONFIG}. Push the Pro SFT updates to GitHub main, then rerun setup."',
                'f"Missing {CONFIG} or budget-forcing code. Push all evaluation updates to GitHub main, then rerun setup."',
            )
            if "progress_totals =" in cell.source:
                cell.source = cell.source.replace(
                    "from eval.profiles import profile_root",
                    "from eval.profiles import profile_root, load_profile\n"
                    "eval_config = yaml.safe_load((Path(CODE_ROOT) / 'configs/eval_english.yaml').read_text())\n"
                    "execution = yaml.safe_load((Path(CODE_ROOT) / 'configs/eval_execution.yaml').read_text())\n"
                    "budget = None\n"
                    "BUDGET_ARGS = []\n"
                    "EVAL_LABEL = EVAL_PROFILE\n"
                    "if EVAL_PROFILE == 'sample1_budget':\n"
                    "    if type(MAX_THINKING_TOKENS) is not int or not 0 < MAX_THINKING_TOKENS < eval_config['max_new_tokens']:\n"
                    "        raise ValueError('MAX_THINKING_TOKENS must be a positive integer below the total token budget')\n"
                    "    budget = yaml.safe_load((Path(CODE_ROOT) / 'configs/eval_profiles.yaml').read_text())[EVAL_PROFILE]['budget_forcing']\n"
                    "    budget['max_reasoning_tokens'] = MAX_THINKING_TOKENS\n"
                    "    budget['answer_budget_tokens'] = eval_config['max_new_tokens'] - MAX_THINKING_TOKENS\n"
                    "    load_profile(EVAL_PROFILE, eval_config, execution, budget_override=budget)\n"
                    "    EVAL_LABEL += f\"_thinking_{MAX_THINKING_TOKENS}_answer_{budget['answer_budget_tokens']}\"\n"
                    "    budget_path = Path(TRAIN_ROOT) / 'eval_configs' / (EVAL_LABEL + '.yaml')\n"
                    "    budget_path.parent.mkdir(parents=True, exist_ok=True)\n"
                    "    budget_path.write_text(yaml.safe_dump(budget, sort_keys=False))\n"
                    "    BUDGET_ARGS = ['--budget_config', str(budget_path)]\n"
                    "    print('Budget config saved:', budget_path)",
                ).replace(
                    "profile_root(BASELINE_ROOT, EVAL_PROFILE)",
                    "profile_root(BASELINE_ROOT, EVAL_PROFILE, budget)",
                )
                cell.source += '\nprint("Evaluation profile:", EVAL_PROFILE)\nprint("Budget settings:", budget or "off")'
            if "RUN_STAGE1_EVAL = False" in cell.source:
                cell.source = cell.source.replace(
                    "        run_logged(command,", "        command.extend(BUDGET_ARGS)\n        run_logged(command,"
                ).replace("_{EVAL_PROFILE}_console.log", "_{EVAL_LABEL}_console.log")
        elif cell.source.startswith("## Full evaluation"):
            cell.source = cell.source.replace(
                "`configs/eval_profiles.yaml` defines `sample1`: temperature 1.0, top-p 0.7,",
                "`configs/eval_profiles.yaml` defines the selected profile: temperature 1.0, top-p 0.7,",
            )
    smoke = [markdown("""
        ## Optional smoke test — five problems, epoch 5
        Enable this to test one problem from each benchmark before the full run. It uses the same
        checkpoint, sampling, and budget settings. Smoke results and raw generations are isolated
        under `profiles/<profile>/smoke/`; they are not counted as full evaluation results.
        Check `budget_forcing` in the generations JSONL for phase lengths and `forced` transitions.
        A short smoke response may finish naturally without exercising the reasoning cap.
        """), code("""
        RUN_SMOKE_EVAL = False
        if RUN_SMOKE_EVAL:
            epoch, checkpoint, run_name = eval_runs[0]
            command = [str(Path(EVAL_ENV) / "bin/python"), "-m", "eval.run_batched_eval",
                       "--model", str(checkpoint), "--stage", cfg["stage"], "--run_name", run_name,
                       "--output_root", BASELINE_ROOT, "--profile", EVAL_PROFILE,
                       "--execution_config", "configs/eval_execution.yaml", "--smoke"]
            command.extend(BUDGET_ARGS)
            run_logged(command, cwd=CODE_ROOT,
                       log_path=Path(TRAIN_ROOT) / f"{run_name}_{EVAL_LABEL}_smoke_console.log",
                       progress_totals={name: 1 for name in progress_totals})
        else:
            print("Smoke evaluation is off. Enable RUN_SMOKE_EVAL to test five problems.")
        """)]
    index = next(i for i, cell in enumerate(notebook_cells) if cell.source.startswith("## Full evaluation"))
    notebook_cells[index:index] = smoke
    return notebook_cells


if __name__ == "__main__":
    write_notebook("train_stage1_sft.ipynb", cells())
    write_notebook("evaluate_stage1_sft.ipynb", eval_cells())
    write_notebook("evaluate_stage1_sft_temp1.ipynb", budget_eval_cells())
