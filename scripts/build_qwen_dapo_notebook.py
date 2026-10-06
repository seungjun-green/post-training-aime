"""Build a bundled Qwen2.5-3B-Instruct DAPO notebook using the tested launcher flow."""

from build_llama_dapo_notebook import cells as llama_cells
from build_notebooks import ROOT, code, markdown, payload, write_notebook


def cells():
    result = llama_cells()
    replacements = {
        "meta-llama/Llama-3.2-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
        "Llama 3.2 3B Instruct": "Qwen2.5-3B-Instruct",
        "dapo_llama32_3b_minibatch.yaml": "dapo_qwen25_3b.yaml",
        "dapo_llama32_3b": "dapo_qwen25_3b",
        "LG-AIME-DAPO-Llama32-3B-MiniBatch300": "LG-AIME-DAPO-Qwen25-3B-MiniBatch300",
        "Llama SFT": "Qwen SFT",
        "llama32-dapo-work": "qwen25-dapo-work",
    }
    for cell in result:
        for old, new in replacements.items():
            cell.source = cell.source.replace(old, new)
    result[0].source = result[0].source.replace(
        "Llama's native chat template uses a fixed date for reproducibility.\n"
        "Native EOT, end-of-text and end-of-message tokens stop generation; padding is separate.",
        "Qwen's native chat template is used. `<|im_end|>` stops generation;\n"
        "`<|endoftext|>` is padding and is excluded from the configured stop list.",
    )
    result[0].source += (
        "\n\nCode, config and dependency locks are bundled. No GitHub push or clone is needed.\n"
        "The memory fix checkpoints the vocabulary projection together with log-probabilities\n"
        "in 128-token chunks, avoiding full 20K-token vocabulary logits during backward.\n"
        "Open this updated notebook in a fresh GPU runtime. GPU smoke still verifies runtime fit.\n"
        "Defaults recover the interrupted run from checkpoint 40 into a separate Drive folder."
    )
    result[1] = code('''
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix"
        RECOVER_FROM_CHECKPOINT = "/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300/checkpoints/dapo_qwen25_3b_base/checkpoint-40"
        # For a new run from the original model, set RECOVER_FROM_CHECKPOINT = "".
        CODE_PARENT = "/content/lg-aime-dapo-qwen25"
        RL_ENV = "/content/lg-dapo-env"
        WORK_DIR = "/content/qwen25-dapo-work"
        CONFIG = "configs/dapo_qwen25_3b.yaml"
    ''')
    result[2] = markdown('''
        ## Setup
        Enable HF_TOKEN in Colab Secrets for the dataset. The published Qwen model is
        loaded directly; no prior SFT or EXAONE checkpoint is needed. Code is installed
        in a versioned local folder and recorded as a reproducible Git snapshot so that
        the shared trainer can verify exact checkpoint-resume identity.
    ''')
    files = sorted({
        *(path for folder in ["common", "pipeline", "eval", "train"] for path in (ROOT / folder).glob("*.py")),
        ROOT / ".gitignore", ROOT / "configs/dapo_qwen25_3b.yaml",
        ROOT / "scripts/setup_dapo_runtime.py",
        ROOT / "requirements-dapo.lock", ROOT / "requirements-eval.lock", ROOT / "requirements-stage1.lock",
    })
    result[3] = code(f'''
        import base64, hashlib, io, json, os, subprocess, sys, zipfile
        from pathlib import Path
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        BUNDLE = {payload(files)!r}
        bundle_bytes = base64.b64decode(BUNDLE)
        bundle_id = hashlib.sha256(bundle_bytes).hexdigest()
        CODE_ROOT = str(Path(CODE_PARENT) / bundle_id[:16])
        Path(CODE_ROOT).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:
            for name in bundle.namelist():
                path = Path(CODE_ROOT) / name
                data = bundle.read(name)
                if path.exists() and path.read_bytes() != data:
                    raise ValueError(f"Bundled file changed: {{path}}. Use a fresh CODE_PARENT.")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        def git(*args, **kwargs):
            return subprocess.check_output(["git", "-C", CODE_ROOT, *args], text=True, **kwargs).strip()
        if not (Path(CODE_ROOT) / ".git").exists():
            git("init", "--quiet")
            git("config", "core.autocrlf", "false")
            git("config", "core.filemode", "false")
            git("add", ".")
            commit_env = dict(os.environ, GIT_AUTHOR_NAME="Notebook bundle", GIT_COMMITTER_NAME="Notebook bundle",
                              GIT_AUTHOR_EMAIL="bundle@localhost", GIT_COMMITTER_EMAIL="bundle@localhost",
                              GIT_AUTHOR_DATE="2026-01-01T00:00:00+00:00", GIT_COMMITTER_DATE="2026-01-01T00:00:00+00:00")
            git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Bundled DAPO " + bundle_id,
                env=commit_env)
        if git("status", "--porcelain"):
            raise ValueError("Bundled code was modified; use a fresh CODE_PARENT.")
        print("Bundled code:", CODE_ROOT, "commit:", git("rev-parse", "HEAD"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_dapo_runtime.py", "--venv", RL_ENV,
                               "--architecture", "Qwen2ForCausalLM"], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
    ''')
    result[10] = markdown('''
        ## Recover training — 300 total optimizer updates

        Defaults load the old **checkpoint 40**, including optimizer, scheduler and RNG,
        and continue with update 41 under the new `OUTPUT_ROOT`. Updates 41–43 from the
        interrupted run are replayed. The original checkpoints/logs are preserved.
        The total remains 300 updates, with the same data, learning rate, rewards,
        64-response minibatches, and 20,480-token cap. The implementation changes, so the
        recovered run records its new code identity and the parent checkpoint explicitly.

        Leave `RECOVER_FROM_CHECKPOINT` set when resuming this recovery run. A completed
        checkpoint in the new folder is selected automatically on rerun. `RESUME_CHECKPOINT`
        can explicitly select the latest complete checkpoint instead. If the runtime stops
        before saving checkpoint 60, rerun from parent checkpoint 40 in this same recovery folder.
        Use a fresh GPU runtime after the CUDA error, then run setup and preparation again.
    ''')
    result[11] = code('''
        RUN_TRAINING = False
        RESUME_CHECKPOINT = ""
        if RUN_TRAINING:
            command = list(TRAIN_COMMAND)
            if RECOVER_FROM_CHECKPOINT:
                command += ["--recover-from-checkpoint", RECOVER_FROM_CHECKPOINT]
            complete = sorted(
                (p for p in checkpoint_root.glob("checkpoint-*") if (p / "dapo_checkpoint.json").is_file()),
                key=lambda p: int(p.name.split("-")[-1]))
            latest = RESUME_CHECKPOINT or (str(complete[-1]) if complete else "")
            if latest:
                command += ["--resume-from-checkpoint", latest]
            if complete and int(complete[-1].name.split("-")[-1]) >= cfg["training"]["max_steps"]:
                print("The full 300-update run is already complete:", complete[-1])
            else:
                print("Continue from:", latest or RECOVER_FROM_CHECKPOINT or "original model")
                run_logged(command, cwd=CODE_ROOT, log_path=run_logs / "training_console.log",
                           compact_progress=True)
        else:
            print("Set RUN_TRAINING = True to recover/continue training.")
    ''')
    return result


if __name__ == "__main__":
    write_notebook("train_dapo_qwen25_3b.ipynb", cells())
