import ast
import base64
import io
import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import yaml

from common.io import write_json
from eval.qwen_kimi_sft import BUNDLE_FILES, checkpoint_source
from eval.run_amc_math_eval import comparison_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen_kimi_sft_eval.yaml"


def make_checkpoint(root):
    config = yaml.safe_load((ROOT / "configs/sft_qwen25_3b_s1_kimi.yaml").read_text())
    name = config["run_name"]
    path = root / "checkpoints/stage1" / name / "epoch_5"
    for file in ["config.json", "tokenizer_config.json", "tokenizer.json"]:
        write_json(path / file, {})
    write_json(path / "stage1_checkpoint.json", {"epoch": 5, "global_step": 310, "run_identity": "trained"})
    write_json(root / "logs/stage1" / name / "run_manifest.json", {"identity": "trained", "config": config})
    write_json(path / "config.json", {"model_type": "qwen2", "eos_token_id": 151645})
    write_json(path / "generation_config.json", {"eos_token_id": [151645]})
    write_json(path / "tokenizer_config.json", {"eos_token": "<|im_end|>"})
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    return path, config


def test_notebook_commands_checkpoint_and_complete_summary(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_qwen_kimi_eval_notebook import cells

    notebook = nbformat.read(ROOT / "notebooks/evaluate_qwen25_3b_s1_kimi_amc_math.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "answer_only_eval", "exec")
        assert "@param" not in source and '"git", "clone"' not in source
    checkpoint, _ = make_checkpoint(tmp_path / "training")
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    assert context["TRAIN_ROOT"].endswith("base-sft-v2-short")
    assert context["EPOCH"] == 5
    assert context["EXPECTED_RUN_IDENTITY"] == "5644b316cb49344bea968be58871bc87a6bae072f50309bcfba18a1ce6e577ef"
    context.update(TRAIN_ROOT=str(tmp_path / "training"), CODE_ROOT=ROOT, EXPECTED_RUN_IDENTITY="trained")
    prepare = next(s for s in sources if "def evaluation_command" in s)
    exec(prepare, context)
    assert context["checkpoint"] == checkpoint
    assert context["eval_root"] == tmp_path / "training/eval/amc2023_math500_temp0/sft_qwen25_3b_base_s1_kimi/epoch_5"
    assert context["progress_totals"] == {"amc23": 40, "math_500": 500}
    config = context["resolved"]
    assert config["temperature"] == 0 and config["top_p"] == 1 and config["pass_k"] == [1]
    assert "budget_forcing" not in config
    assert config["max_new_tokens"] == 20480
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append((command, kwargs))
    for flag in ["RUN_SMOKE", "RUN_EVAL"]:
        source = next(s for s in sources if f"{flag} = False" in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f"{flag} = False", f"{flag} = True"), context)
    assert len(calls) == 2
    assert "--smoke" in calls[0][0] and "--smoke" not in calls[1][0]
    for (command, kwargs), total in zip(calls, [2, 540], strict=True):
        assert command[command.index("--model") + 1] == str(checkpoint)
        assert "--bundled-code" in command
        assert sum(kwargs["progress_totals"].values()) == total
    metrics = {name: {"problems": n, "responses": n, "samples_per_problem": 1,
                      "avg@1": 0.5, "response_length_tokens": {"all": 1900.}}
               for name, n in context["progress_totals"].items()}
    result = {"model": str(checkpoint), "mode": "full", "config": config, "metrics": metrics, "run_name": context["run_name"]}
    write_json(context["result_path"], result)
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(
        DataFrame=lambda rows: SimpleNamespace(style=SimpleNamespace(format=lambda _: rows),
                                               to_csv=lambda path, **kw: None)))
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(display=lambda _: None))
    context["show_results"]()
    table = json.loads((context["eval_root"] / "full/summary_table.json").read_text())
    assert [r["Correct"] for r in table["rows"]] == [20, 250]
    assert [r["Accuracy (%)"] for r in table["rows"]] == [50, 50]
    result["metrics"]["math_500"]["responses"] = 499
    write_json(context["result_path"], result)
    with pytest.raises(ValueError, match="Incomplete full evaluation"):
        context["show_results"]()


def test_checkpoint_rejects_wrong_training_or_incomplete_model(tmp_path):
    checkpoint, config = make_checkpoint(tmp_path)
    name = config["run_name"]
    spec, _, _, _ = comparison_settings(CONFIG)
    assert checkpoint_source(tmp_path, name, 5, spec)[0] == checkpoint
    with pytest.raises(ValueError, match="EXPECTED_RUN_IDENTITY"):
        checkpoint_source(tmp_path, name, 5, spec, "another-run")
    shard = checkpoint / "model.safetensors"
    original = shard.read_bytes()
    shard.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="Incomplete model shard data"):
        checkpoint_source(tmp_path, name, 5, spec)
    shard.write_bytes(original)
    # Sharded and single-file exports are both supported without optimizer files.
    write_json(checkpoint / "model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        checkpoint_source(tmp_path, name, 5, spec)
    (checkpoint / "model.safetensors.index.json").unlink()
    config["data"]["response_format"] = "reasoning_and_answer"
    write_json(tmp_path / "logs/stage1" / name / "run_manifest.json", {"identity": "trained", "config": config})
    with pytest.raises(ValueError, match="Kimi-style"):
        checkpoint_source(tmp_path, name, 5, spec)
    write_json(checkpoint / "stage1_checkpoint.json", {"epoch": 5, "run_identity": "other"})
    with pytest.raises(ValueError, match="identity"):
        checkpoint_source(tmp_path, name, 5, spec)


@pytest.mark.parametrize("field", ["base_model", "dataset", "eos"])
def test_rejects_wrong_base_data_and_stop_token(tmp_path, field):
    checkpoint, config = make_checkpoint(tmp_path)
    spec, _, _, _ = comparison_settings(CONFIG)
    if field == "base_model":
        config["model"]["repo"] = "Qwen/Qwen2.5-3B-Instruct"
    elif field == "dataset":
        config["data"]["hf_source"]["revision"] = "wrong"
    else:
        write_json(checkpoint / "generation_config.json", {"eos_token_id": 151643})
    write_json(tmp_path / "logs/stage1" / config["run_name"] / "run_manifest.json",
               {"identity": "trained", "config": config})
    with pytest.raises(ValueError):
        checkpoint_source(tmp_path, config["run_name"], 5, spec)


def test_bundle_is_current_and_works_without_repository(tmp_path):
    notebook = nbformat.read(ROOT / "notebooks/evaluate_qwen25_3b_s1_kimi_amc_math.ipynb", as_version=4)
    boot = next(c.source for c in notebook.cells if c.cell_type == "code" and "BUNDLE =" in c.source)
    assignment = next(n for n in ast.parse(boot).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "BUNDLE" for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assignment.value)))) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        for name in BUNDLE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(tmp_path)
    subprocess.run([sys.executable, "-m", "eval.run_amc_math_eval", "--help"],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([sys.executable, "-c", "from eval.answer_only import evaluation_code_digest; "
                    "from eval.run_amc_math_eval import comparison_settings; "
                    "from eval.run_english_eval import code_fingerprint; "
                    "from eval.run_batched_eval import runner_digest; "
                    "s,c,e,d=comparison_settings('configs/qwen_kimi_sft_eval.yaml'); "
                    "assert list(d['datasets'])==['amc23','math_500']; "
                    "assert c['temperature']==0; "
                    "assert len(evaluation_code_digest())==len(code_fingerprint())==len(runner_digest())==64"],
                   cwd=tmp_path, check=True, capture_output=True)
