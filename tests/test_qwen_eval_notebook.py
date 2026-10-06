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
from eval.qwen_comparison import BUNDLE_FILES, checkpoint_source, comparison_rows, prepare_runs, result_path
from eval.run_amc_math_eval import comparison_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen_compare_eval.yaml"
NOTEBOOK = ROOT / "notebooks/evaluate_qwen25_3b_base_instruct_rl_amc_math.ipynb"
RUN_NAME = "dapo_qwen25_3b_base"


def make_checkpoint(root):
    config = yaml.safe_load((ROOT / "configs/dapo_qwen25_3b.yaml").read_text())
    config["model_kind"] = "base"
    checkpoint = root / "checkpoints" / RUN_NAME / "checkpoint-300"
    write_json(checkpoint / "dapo_checkpoint.json", {"global_step": 300, "run_identity": "trained"})
    write_json(checkpoint / "config.json", {"model_type": "qwen2"})
    write_json(checkpoint / "tokenizer_config.json", {"chat_template": "qwen"})
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (checkpoint / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    manifest = {"identity": "trained", "config": config}
    write_json(root / "logs" / RUN_NAME / "run_manifest.json", manifest)
    return checkpoint, manifest


def test_checkpoint_validation_and_selection_resume(tmp_path):
    checkpoint, manifest = make_checkpoint(tmp_path / "train")
    spec, _, _, _ = comparison_settings(CONFIG)
    args = (tmp_path / "train", RUN_NAME, 300, spec["models"]["instruct"])
    assert checkpoint_source(*args)[0] == checkpoint
    runs = prepare_runs(spec, *args[:3], tmp_path / "eval")
    assert prepare_runs(spec, *args[:3], tmp_path / "eval") == runs
    assert [r["key"] for r in runs] == ["base", "instruct", "instruct_rl"]
    assert runs[-1]["model"] == str(checkpoint) and runs[-1]["revision"] is None
    assert len({r["output_root"] for r in runs}) == 3
    selection = tmp_path / "eval/model_selection.json"
    write_json(selection, [])
    with pytest.raises(ValueError, match="Selected models changed"):
        prepare_runs(spec, *args[:3], tmp_path / "eval")
    manifest["config"]["model"]["repo"] = "Qwen/Qwen2.5-3B"
    manifest_path = tmp_path / "train/logs" / RUN_NAME / "run_manifest.json"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="pinned Qwen instruct"):
        checkpoint_source(*args)
    manifest["identity"] = "different"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="identity"):
        checkpoint_source(*args)
    shard = checkpoint / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Incomplete model shard"):
        checkpoint_source(*args)


def write_results(runs, config):
    for i, run in enumerate(runs):
        result = {key: "same" for key in ["execution", "protocol_digest", "dataset_suite", "selected_ids",
                                         "packages", "hardware", "runner_digest", "bundle_digest"]}
        result.update(model=run["model"], model_revision=run["revision"], run_name=run["run_name"],
                      mode="full", config=config, chat_template_digest="base" if i == 0 else "instruct")
        result["metrics"] = {name: {"problems": n, "responses": n, "samples_per_problem": 1,
                                    "avg@1": .1 * i, "response_length_tokens": {"all": 100 * (i + 1)}}
                             for name, n in [("amc23", 40), ("math_500", 500)]}
        write_json(result_path(run), result)


def test_notebook_runs_three_models_with_pins_and_displays_table(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_qwen_eval_notebook import cells

    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    sources = [c.source for c in nb.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "qwen_eval", "exec")
        assert "@param" not in source
    make_checkpoint(tmp_path / "training")
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    assert context["RL_ROOT"].endswith("-MemoryFix") and context["RL_STEP"] == 300
    context.update(CODE_ROOT=ROOT, RL_ROOT=str(tmp_path / "training"), OUTPUT_ROOT=str(tmp_path / "eval"))
    exec(next(s for s in sources if "def evaluation_command" in s), context)
    config, runs = context["resolved"], context["runs"]
    assert config["temperature"] == 0 and config["top_p"] == 1
    assert config["max_new_tokens"] == 20480 and "budget_forcing" not in config
    assert context["progress_totals"] == {"amc23": 40, "math_500": 500}
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append((command, kwargs))
    for flag in ["RUN_SMOKE", "RUN_EVAL"]:
        source = next(s for s in sources if f"{flag} = False" in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f"{flag} = False", f"{flag} = True"), context)
    assert len(calls) == 6
    for i, (command, kwargs) in enumerate(calls):
        run = runs[i % 3]
        assert command[command.index("--model") + 1] == run["model"]
        assert "--bundled-code" in command
        assert ("--revision" in command) == (i % 3 != 2)
        assert ("--smoke" in command) == (i < 3)
        assert sum(kwargs["progress_totals"].values()) == (2 if i < 3 else 540)
    write_results(runs, config)
    displayed, csvs = [], []
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(DataFrame=lambda rows: SimpleNamespace(
        style=SimpleNamespace(format=lambda _: rows), to_csv=lambda path, **kw: csvs.append(path))))
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(display=displayed.append))
    context["show_comparison"]()
    table = json.loads((tmp_path / "eval/comparison_table.json").read_text())
    assert len(table["rows"]) == 6
    assert [r["Correct"] for r in table["rows"]] == [0, 0, 4, 50, 8, 100]
    assert table["rows"] == displayed[0] and len(csvs) == 1


@pytest.mark.parametrize("change,error", [
    (("mode", "smoke"), "selected model"),
    (("model_revision", "different"), "selected model"),
    (("selected_ids", "different"), "selected_ids"),
    (("chat_template_digest", "different"), "same native chat"),
])
def test_comparison_rejects_mismatched_runs(tmp_path, change, error):
    make_checkpoint(tmp_path / "train")
    spec, config, _, _ = comparison_settings(CONFIG)
    runs = prepare_runs(spec, tmp_path / "train", RUN_NAME, 300, tmp_path / "eval")
    write_results(runs, config)
    assert len(comparison_rows(runs, config)) == 6
    path = result_path(runs[-1])
    result = json.loads(path.read_text())
    result[change[0]] = change[1]
    write_json(path, result)
    with pytest.raises(ValueError, match=error):
        comparison_rows(runs, config)


def test_three_model_evaluation_scores_and_resumes(tmp_path, monkeypatch):
    from test_batched_eval import Tokenizer, install_core
    from eval import engines, english_engines, run_amc_math_eval as runner

    make_checkpoint(tmp_path / "train")
    spec, _, _, _ = comparison_settings(CONFIG)
    runs = prepare_runs(spec, tmp_path / "train", RUN_NAME, 300, tmp_path / "eval")
    cores = []
    def loader(config, suite, token):
        assert list(config["datasets"]) == ["amc23", "math_500"]
        return {name: [{"id": name, entry["problem_column"]: "Question", entry["answer_column"]: "0"}]
                for name, entry in config["datasets"].items()}
    def create(model, config, revision):
        tokenizer = Tokenizer()
        tokenizer.chat_template = "base" if model == runs[0]["model"] else "instruct"
        engine = SimpleNamespace(name="vllm", config=config, tokenizer=tokenizer)
        cores.append(install_core(monkeypatch, engine))
        return engine, None
    monkeypatch.setattr(runner, "load_eval_sets", loader)
    monkeypatch.setattr(runner, "resolve_model", lambda model, revision, token: (revision, "digest"))
    monkeypatch.setattr(runner, "package_versions", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(engines, "check_hardware", lambda config: {"gpu": "test"})
    monkeypatch.setattr(english_engines, "create_engine", create)
    for attempt in range(2):
        for run in runs:
            argv = ["eval", "--model", run["model"], "--run-name", run["run_name"],
                    "--config", str(CONFIG), "--output-root", run["output_root"], "--smoke", "--bundled-code"]
            if run["revision"]:
                argv += ["--revision", run["revision"]]
            monkeypatch.setattr(sys, "argv", argv)
            runner.main()
            assert len(cores[-1].calls) == (2 if attempt == 0 else 0)
            assert all(p.temperature == 0 and p.n == 1 for _, _, p in cores[-1].calls)
            result = json.loads(result_path(run, smoke=True).read_text())
            assert all(m["avg@1"] == 1 and m["responses"] == 1 for m in result["metrics"].values())


def test_bundle_runs_in_isolation_and_matches_source(tmp_path):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    boot = next(c.source for c in nb.cells if c.cell_type == "code" and "BUNDLE =" in c.source)
    assignment = next(n for n in ast.parse(boot).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "BUNDLE" for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assignment.value)))) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        for name in BUNDLE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(tmp_path)
    subprocess.run([sys.executable, "-m", "eval.run_amc_math_eval", "--help"],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([sys.executable, "-c",
                    "from eval.qwen_comparison import prepare_runs; "
                    "from eval.answer_only import evaluation_code_digest; "
                    "from eval.run_amc_math_eval import comparison_settings; "
                    "s,c,e,d=comparison_settings('configs/qwen_compare_eval.yaml'); "
                    "assert len(evaluation_code_digest())==64; "
                    "assert c['temperature']==0; "
                    "assert sum(v['rows'] for v in d['datasets'].values())==540"],
                   cwd=tmp_path, check=True, capture_output=True)
