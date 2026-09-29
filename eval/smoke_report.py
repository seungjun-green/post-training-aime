"""Export the completed smoke run for review without bundling unrelated files."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def write_smoke_archive(root, stage, run_name):
    root = Path(root)
    run = root / "results" / stage
    result = run / f"{run_name}.json"
    if json.loads(result.read_text()).get("mode") != "smoke":
        raise ValueError("Only a completed smoke run can be exported as smoke_test_result")
    members = {
        "smoke_test_result.json": result,
        "generations.jsonl": run / f"{run_name}_generations.jsonl",
        "run_manifest.json": run / f"{run_name}_manifest.json",
        "engine.json": run / f"{run_name}_engine.json",
        "eval_protocol.json": root / "results/eval_protocol.json",
        "eval_runtime.json": root / "results/eval_runtime.json",
    }
    for source in members.values():
        if not source.is_file():
            raise FileNotFoundError(f"Incomplete smoke artifacts: {source}")
    destination = root / "archives" / stage / run_name / "smoke_test_result.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name, source in members.items():
            archive.write(source, name)
        archive.writestr(
            "README.txt",
            "English evaluation smoke test: one problem and one response per evaluation set.\n"
            "generations.jsonl contains rendered prompts, responses, reference/extracted answers,\n"
            "correctness, token counts and finish reasons. The JSON files record metrics, model\n"
            "revision, Git commit, dataset pins and environment. Review pipeline behavior, not\n"
            "benchmark accuracy: five problems are insufficient to estimate performance.\n"
            "Keep RUN_FULL_EVAL=False until this smoke result has been reviewed.\n",
        )
    temporary.replace(destination)
    return destination
