"""Build the CPU Colab notebook publishing seven trained Qwen checkpoints to HF."""

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook

NAME = 'upload_qwen25_3b_7_checkpoints_to_hf.ipynb'
FILES = ['common/__init__.py', 'common/io.py', 'eval/__init__.py', 'eval/transfer_models.py',
         'eval/runpod_transfer_storage.py', 'eval/transfer_hub.py',
         'configs/qwen_transfer_eval.yaml', 'configs/qwen_transfer_hub.yaml']


def cells():
    boot = bootstrap(payload([ROOT / name for name in FILES]))
    boot.source = boot.source.replace('/content/lg-korean-aime', '/content/lg-qwen-checkpoint-upload')
    return [
        markdown('''
        # Upload the seven trained Qwen2.5-3B checkpoints to Hugging Face

        **CPU Colab is sufficient.** Run cells in order with a **write-enabled `HF_TOKEN`**
        in Colab Secrets. Colab mounts Google Drive normally; no rclone or separate Google
        OAuth client setup. Uploads are enabled in the last cell.

        Seven **private model repositories** under `Seungjun` by default:
        base + RL step 300, Kimi SFT epoch 5, rejection SFT epoch 4, 7B-distillation SFT
        epoch 5, rejection SFT epoch 4 + RL step 300, distillation SFT epoch 5 + RL step
        300, and instruct + RL step 300. Original B0 and I0 already exist on HF.

        Copies the selected full weights, saved tokenizer/chat template, EOS/PAD and
        generation settings, checkpoint marker, and original training manifest.
        **No conversion, quantization, merging, tokenizer modification or training.**
        Optimizer/scheduler states and other epochs are excluded. Weights are read
        directly from Drive; the notebook does not duplicate seven models on Colab disk.

        Each upload is checked against Hub file checksums. Completed models are reused
        on rerun; failed uploads can be retried. After all seven succeed, a small
        `transfer_catalog.json` is published in the base + RL model repository. It pins
        each checkpoint's exact commit and lets RunPod download everything automatically.
        Source Drive files are not modified. Existing unrelated repositories or different
        checkpoint contents are rejected; use a different repository name in that case.
        '''),
        code('''
        import os, sys, subprocess, json
        from pathlib import Path
        EXPERIMENT_ROOT = "/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments"
        INSTRUCT_RL_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix"
        HF_OWNER = "Seungjun"
        HF_PRIVATE = True
        RECEIPT_DIR = str(Path(EXPERIMENT_ROOT) / "hf-checkpoint-upload")

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "huggingface-hub==0.36.2", "PyYAML==6.0.3"])
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Enable a write-enabled HF_TOKEN in Colab Secrets")
        os.environ["HF_TOKEN"] = HF_TOKEN
        '''),
        boot,
        markdown('''
        ## Validate all seven checkpoints and show upload destinations

        Checks saved epoch/step, source model, run identity, weight shards, tokenizer
        and EOS before creating any HF repository. These are the same checks used by
        the evaluation notebook. Edit `hub["repositories"]` here if you want different
        repository names; the displayed catalog repository is the only value RunPod needs.
        HF_PRIVATE controls new repositories; an existing repository must already match.
        '''),
        code('''
        import yaml
        from eval.transfer_hub import upload_plan, upload_all
        config = yaml.safe_load((CODE_ROOT / "configs/qwen_transfer_eval.yaml").read_text())
        hub = yaml.safe_load((CODE_ROOT / "configs/qwen_transfer_hub.yaml").read_text())
        hub["repositories"] = {key: HF_OWNER + "/" + repo.split("/", 1)[1]
                               for key, repo in hub["repositories"].items()}
        hub["private"] = HF_PRIVATE
        plan = upload_plan(config, hub, EXPERIMENT_ROOT, INSTRUCT_RL_ROOT)
        for item in plan:
            print(f"{item['key']}: {item['bytes'] / 2**30:.2f} GiB → {item['repo_id']}")
            print("  source:", item["path"])
        print("Total:", round(sum(item["bytes"] for item in plan) / 2**30, 2), "GiB")
        print("RunPod HF_CATALOG_REPO:", hub["repositories"][hub["catalog_key"]])
        print("Visibility:", "private" if HF_PRIVATE else "public")
        print("Upload receipts:", RECEIPT_DIR)
        '''),
        markdown('''
        ## Upload and verify

        Hashing and transferring tens of GB can take time; leave the Colab session
        connected. The upload library shows transfer progress. A receipt is written to
        Drive after every verified model. A rerun verifies already-uploaded checkpoints
        and continues; it does not start training or overwrite different checkpoints.

        After **“All seven checkpoints verified”**, open the updated RunPod notebook
        and run its setup/download cells. The default catalog repository already matches.
        If you changed HF_OWNER or repository names, copy the printed HF_CATALOG_REPO
        into RunPod. It resolves the catalog once and pins all seven model revisions.
        '''),
        code('''
        RUN_UPLOAD = True
        if RUN_UPLOAD:
            receipt = upload_all(config, hub, EXPERIMENT_ROOT, INSTRUCT_RL_ROOT,
                                 RECEIPT_DIR, token=HF_TOKEN)
            from IPython.display import Markdown, display
            table = "| Variant | Model repository | Pinned revision |\\n| --- | --- | --- |\\n"
            for key, record in receipt["catalog"]["models"].items():
                repo, revision = record["repo_id"], record["revision"]
                table += f"| {key} | [{repo}](https://huggingface.co/{repo}) | `{revision}` |\\n"
            display(Markdown(table))
            print("HF_CATALOG_REPO =", repr(receipt["repo_id"]))
            print("HF_CATALOG_REVISION =", repr(receipt["revision"]), "# optional exact pin; main also works")
            print("Completion receipt:", Path(RECEIPT_DIR) / "upload_complete.json")
        else:
            print("Set RUN_UPLOAD = True to publish the seven validated checkpoints.")
        '''),
    ]


if __name__ == '__main__':
    write_notebook(NAME, cells())
