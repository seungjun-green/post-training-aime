"""Explicit, checked source adapters; original rows are retained without mutation."""

import json
from copy import deepcopy
from pathlib import Path

import yaml

from common.io import digest, read_jsonl, write_json, write_jsonl

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)


def load_registry(path="configs/datasets.yaml"):
    return yaml.safe_load(Path(path).read_text())


def nested(row, key):
    for part in key.split("."):
        row = row[part]
    return row


def core_dapo_problem(value):
    if isinstance(value, list):
        if len(value) != 1 or value[0].get("role") != "user":
            raise ValueError("Unrecognized DAPO message wrapper; inspect source before proceeding")
        value = value[0]["content"]
    if not isinstance(value, str):
        raise ValueError("DAPO prompt must be text or one user message")
    if value.startswith(DAPO_PREFIX):
        value = value[len(DAPO_PREFIX) :]
    elif value.startswith("Solve the following math problem"):
        raise ValueError("Unknown DAPO instruction wrapper; refusing to guess")
    return value


def adapt_rows(name, rows, spec):
    adapted = []
    seen = set()
    for i, original in enumerate(rows):
        original = deepcopy(original)
        identifier = (
            nested(original, spec["id_column"])
            if spec["id_column"]
            else f"{name}:{spec['split']}:{i}"
        )
        if identifier is None or str(identifier) in seen:
            raise ValueError(f"Missing or duplicate source id in {name}: {identifier}")
        seen.add(str(identifier))
        problem = original[spec["problem_column"]]
        if name == "dapo_math_17k":
            problem = core_dapo_problem(problem)
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"Missing problem in {name}/{identifier}")
        answer = original[spec["answer_column"]]
        if answer is None or not str(answer).strip():
            raise ValueError(f"Missing answer in {name}/{identifier}")
        adapted.append(
            {"id": identifier, "problem": problem, "answer": answer, "original": original}
        )
    return adapted


def source_fields(row, spec):
    return {
        column: row["problem"] if column == "problem" else row["original"][column]
        for column in spec["translate"]
    }


def cache_source_card(name, spec, cache, token):
    from huggingface_hub import hf_hub_download

    destination = cache / f"{name}_source_card.md"
    if destination.exists():
        return
    downloaded = hf_hub_download(
        spec["repo"], "README.md", repo_type="dataset", revision=spec["revision"], token=token
    )
    text = Path(downloaded).read_text(encoding="utf-8")
    # Re-establish the destination immediately before writing, after network work.
    # The initial mkdir alone is insufficient if the mounted folder disappears.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def download_sources(registry, root, token=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    cache = Path(root) / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    data, manifest = {}, {}
    for name, spec in registry.items():
        signature = digest(spec)
        metadata_path = cache / f"{name}_metadata.json"
        rows_path = cache / f"{name}.jsonl"
        if rows_path.exists() and metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
            if meta["signature"] != signature:
                raise ValueError(f"Source config changed for {name}; use a new project root")
            rows = read_jsonl(rows_path)
            if digest(rows) != meta["content_digest"]:
                raise ValueError(f"Source cache checksum failed for {name}")
        else:
            info = api.dataset_info(spec["repo"], revision=spec["revision"])
            ds = load_dataset(
                spec["repo"],
                name=spec["config"],
                split=spec["split"],
                revision=spec["revision"],
                token=token,
            )
            print(f"{name}: {ds.features}; rows={len(ds)}")
            if len(ds) != spec["expected_rows"]:
                raise ValueError(f"Unexpected source size for {name}: {len(ds)}")
            # Validate the actual schema before applying the mapping.
            needed = {spec["problem_column"], spec["answer_column"]}
            needed |= set(spec["translate"]) - {"problem"}
            if spec["id_column"]:
                needed.add(spec["id_column"].split(".")[0])
            if not needed <= set(ds.column_names):
                raise ValueError(
                    f"Schema mismatch for {name}: missing {needed - set(ds.column_names)}"
                )
            rows = adapt_rows(name, ds, spec)
            card = info.card_data.to_dict() if info.card_data else {}
            if card.get("license") != spec["license"]:
                raise ValueError(f"Source license changed for {name}; inspect before proceeding")
            cache_source_card(name, spec, cache, token)
            meta = {
                "signature": signature,
                "repo": spec["repo"],
                "revision": info.sha,
                "config": spec["config"],
                "split": spec["split"],
                "features": ds.features.to_dict(),
                "rows": len(rows),
                "id_column": spec["id_column"],
                "generated_id_rule": None
                if spec["id_column"]
                else f"{name}:{spec['split']}:<zero-based-row-index>",
                "license": card.get("license"),
                "content_digest": digest(rows),
            }
            write_jsonl(rows_path, rows)
            write_json(metadata_path, meta)
        if len(rows) != spec["expected_rows"]:
            raise ValueError(f"Incomplete source cache for {name}")
        # Repair a missing card even when rows and metadata were already cached.
        cache_source_card(name, spec, cache, token)
        data[name], manifest[name] = rows, meta
    write_json(cache / "manifest.json", manifest)
    return data, manifest
