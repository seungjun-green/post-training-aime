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
from eval.qwen_base_rl import BUNDLE_FILES, checkpoint_source
from eval.run_amc_math_eval import comparison_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen_base_rl_eval.yaml"


def make_checkpoint(root, step=140):
    config = yaml.safe_load((ROOT / "configs/dapo_qwen25_3b_base.yaml").read_text())
    config["model_kind"] = "base"
    name = config["run_name_prefix"] + "_base"
    path = root / "checkpoints" / name / f"checkpoint-{step}"
    write_json(path / "dapo_checkpoint.json", {
        "global_step": step, "model_kind": "base", "run_identity": "trained"})
    write_json(root / "logs" / name / "run_manifest.json", {"identity": "trained", "config": config})
    write_json(path / "config.json", {"model_type": "qwen2", "eos_token_id": 151643})
    write_json(path / "generation_config.json", {"eos_token_id": [151643]})
    write_json(path / "tokenizer_config.json", {
        "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "chat_template": "native"})
    write_json(path / "tokenizer.json", {})
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    return path, config


@pytest.mark.parametrize("step", [140, 300])
def test_notebook_commands_checkpoint_and_complete_summary(tmp_path, monkeypatch, step):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_qwen_base_rl_eval_notebook import cells

    notebook = nbformat.read(ROOT / f"notebooks/evaluate_qwen25_3b_base_rl_checkpoint{step}_amc_math.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells(step)]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "answer_only_eval", "exec")
        assert "@param" not in source and '"git", "clone"' not in source
    checkpoint, _ = make_checkpoint(tmp_path / "training", step)
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    assert context["TRAIN_ROOT"].endswith("base-rl-native-eos")
    assert context["STEP"] == step
    context.update(TRAIN_ROOT=str(tmp_path / "training"), CODE_ROOT=ROOT)
    prepare = next(s for s in sources if "def evaluation_command" in s)
    exec(prepare, context)
    assert context["checkpoint"] == checkpoint
    assert context["eval_root"] == tmp_path / f"training/eval/amc2023_math500_temp0/dapo_qwen25_3b_pretrained_base/checkpoint-{step}"
    assert context["run_name"] == f"qwen_base_rl_step{step}"
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


@pytest.mark.parametrize("step", [140, 300])
def test_bundle_is_current_and_works_without_repository(tmp_path, step):
    notebook = nbformat.read(ROOT / f"notebooks/evaluate_qwen25_3b_base_rl_checkpoint{step}_amc_math.ipynb", as_version=4)
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
                    "s,c,e,d=comparison_settings('configs/qwen_base_rl_eval.yaml'); "
                    "assert list(d['datasets'])==['amc23','math_500']; "
                    "assert c['temperature']==0; "
                    "assert len(evaluation_code_digest())==len(code_fingerprint())==len(runner_digest())==64"],
                   cwd=tmp_path, check=True, capture_output=True)


@pytest.mark.parametrize("change", ["identity", "model", "eos", "override", "incomplete", "step"])
def test_rejects_wrong_or_incomplete_checkpoint(tmp_path, change):
    checkpoint, config = make_checkpoint(tmp_path)
    name = checkpoint.parent.name
    spec, _, _, _ = comparison_settings(CONFIG)
    # A completed step 140 is valid even with a 300-step training budget.
    assert config["training"]["max_steps"] == 300
    assert checkpoint_source(tmp_path, name, 140, spec)[0] == checkpoint
    manifest = {"identity": "trained", "config": config}
    if change == "identity":
        manifest["identity"] = "another"
    elif change == "model":
        config["model"]["repo"] = "Qwen/Qwen2.5-3B-Instruct"
    elif change == "eos":
        write_json(checkpoint / "generation_config.json", {"eos_token_id": 151645})
    elif change == "override":
        config["generation"] = {"stop_tokens": ["<|im_end|>"]}
    elif change == "step":
        write_json(checkpoint / "dapo_checkpoint.json", {
            "global_step": 160, "run_identity": "trained", "model_kind": "base"})
    else:
        shard = checkpoint / "model.safetensors"
        shard.write_bytes(shard.read_bytes()[:-1])
    write_json(tmp_path / "logs" / name / "run_manifest.json", manifest)
    with pytest.raises(ValueError):
        checkpoint_source(tmp_path, name, 140, spec)


def test_saved_transformers_tokenizer_with_separate_template(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    checkpoint, _ = make_checkpoint(tmp_path)
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<|endoftext|>": 0}, unk_token="<|endoftext|>")),
        eos_token="<|endoftext|>", pad_token="<|endoftext|>", chat_template=template)
    tokenizer.save_pretrained(checkpoint)
    assert "chat_template" not in json.loads((checkpoint / "tokenizer_config.json").read_text())
    assert (checkpoint / "chat_template.jinja").read_text() == template
    spec, _, _, _ = comparison_settings(CONFIG)
    assert checkpoint_source(tmp_path, checkpoint.parent.name, 140, spec)[0] == checkpoint
    assert PreTrainedTokenizerFast.from_pretrained(checkpoint).chat_template == template
    # Separate template support must not bypass EOS validation.
    write_json(checkpoint / "generation_config.json", {"eos_token_id": 151645})
    with pytest.raises(ValueError, match="generation_config.json eos_token_id"):
        checkpoint_source(tmp_path, checkpoint.parent.name, 140, spec)


@pytest.mark.parametrize("empty_file", [False, True])
def test_missing_or_empty_template_is_rejected(tmp_path, empty_file):
    checkpoint, _ = make_checkpoint(tmp_path)
    path = checkpoint / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config.pop("chat_template")
    write_json(path, config)
    if empty_file:
        (checkpoint / "chat_template.jinja").write_text("  \n")
    spec, _, _, _ = comparison_settings(CONFIG)
    with pytest.raises(ValueError, match="Missing nonempty default chat template"):
        checkpoint_source(tmp_path, checkpoint.parent.name, 140, spec)
