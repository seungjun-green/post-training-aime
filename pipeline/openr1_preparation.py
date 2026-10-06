"""CPU preparation of correct OpenR1 traces, eval decontamination, and Hub publication."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from common.io import digest, read_jsonl, write_json, write_jsonl
from pipeline.decontamination import NORMALIZATION_VERSION, decontaminate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "open-r1/OpenR1-Math-220k"
TARGET_NAME = "dp_removed_OpenR1-Math-220k"
COLUMNS = [
    "problem",
    "generations",
    "correctness_math_verify",
    "finish_reasons",
    "is_reasoning_complete",
    "source",
    "uuid",
]


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def initialize_run(
    root,
    token,
    *,
    tokenizer="Qwen/Qwen2.5-3B",
    max_tokens=20480,
    missing_finish="drop",
    batch_size=128,
):
    """Freeze revisions once; changed settings/code get a separate resumable folder."""
    from huggingface_hub import HfApi, hf_hub_download

    if missing_finish not in {"drop", "reasoning_complete"}:
        raise ValueError("missing_finish must be drop or reasoning_complete")
    if max_tokens < 1 or batch_size < 1:
        raise ValueError("Positive token limit and batch size required")
    suite = json.loads((ROOT / "configs/english_eval_suite.json").read_text())
    settings = {
        "source_repo": SOURCE,
        "subset": "default",
        "split": "train",
        "tokenizer": tokenizer,
        "max_tokens": max_tokens,
        "token_count_rule": "encode(problem) + encode(generation), add_special_tokens=False",
        "missing_finish": missing_finish,
        "batch_size": batch_size,
        "normalization_version": NORMALIZATION_VERSION,
        "ngram_size": 8,
        "coverage_threshold": 0.7,
        "near_miss_min": 0.5,
        "eval_suite": suite,
        "code_digest": digest(
            {
                p: (ROOT / p).read_text()
                for p in [
                    "pipeline/openr1_preparation.py",
                    "pipeline/decontamination.py",
                    "common/io.py",
                ]
            }
        ),
    }
    folder = Path(root) / "runs" / digest(settings)[:20]
    path = folder / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != settings:
            raise ValueError("Preparation settings differ from saved manifest")
    else:
        api = HfApi(token=token)
        source_revision = api.dataset_info(SOURCE).sha
        tokenizer_revision = api.model_info(tokenizer).sha
        card = Path(
            hf_hub_download(
                SOURCE, "README.md", repo_type="dataset", revision=source_revision, token=token
            )
        ).read_text()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source_card.md").write_text(card)
        manifest = {
            "settings": settings,
            "source_revision": source_revision,
            "tokenizer_revision": tokenizer_revision,
        }
        manifest["signature"] = digest(manifest)
        write_json(path, manifest)
    return folder


def filter_problem(row, source_index, tokenizer, max_tokens, missing_finish):
    """Filter parallel trace arrays together; never truncate or alter training text."""
    counts = Counter(source_problems=1)
    generations = row.get("generations")
    correctness = row.get("correctness_math_verify")
    if not isinstance(generations, list) or not isinstance(correctness, list):
        raise ValueError(f"Missing trace arrays at source row {source_index}")
    size = len(generations)
    if len(correctness) != size:
        raise ValueError(f"Misaligned correctness array at source row {source_index}")
    counts["source_traces"] = size
    arrays = {}
    for name in ["finish_reasons", "is_reasoning_complete"]:
        value = row.get(name)
        if value is not None and (not isinstance(value, list) or len(value) != size):
            raise ValueError(f"Misaligned {name} at source row {source_index}")
        arrays[name] = value if value is not None else [None] * size
    problem = row.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        counts["invalid_problem_traces"] = size
        return None, counts
    indices, texts, bases = [], [], []
    for i, text in enumerate(generations):
        if correctness[i] is not True:
            counts["incorrect_or_unverified_traces"] += 1
            continue
        if not isinstance(text, str) or not text.strip():
            counts["empty_generation_traces"] += 1
            continue
        reason, complete = arrays["finish_reasons"][i], arrays["is_reasoning_complete"][i]
        if reason == "length":
            counts["truncated_traces"] += 1
            continue
        if reason not in {None, "stop"}:
            counts["other_finish_reason_traces"] += 1
            continue
        if complete is False:
            counts["incomplete_reasoning_traces"] += 1
            continue
        if reason is None and (missing_finish == "drop" or complete is not True):
            counts["missing_completion_evidence_traces"] += 1
            continue
        indices.append(i)
        texts.append(text)
        bases.append("finish_reason_stop" if reason == "stop" else "reasoning_complete_fallback")
    if not indices:
        return None, counts
    encoded = tokenizer(
        [problem, *texts], add_special_tokens=False, truncation=False, return_attention_mask=False
    )["input_ids"]
    problem_tokens = len(encoded[0])
    kept, lengths, evidence = [], [], []
    for i, tokens, basis in zip(indices, encoded[1:], bases, strict=True):
        length = problem_tokens + len(tokens)
        if length > max_tokens:
            counts["overlength_traces"] += 1
        else:
            kept.append(i)
            lengths.append(length)
            evidence.append(basis)
            if basis == "reasoning_complete_fallback":
                counts["kept_missing_finish_fallback_traces"] += 1
    if not kept:
        return None, counts
    counts.update(kept_problems=1, kept_traces=len(kept))
    return {
        "id": f"openr1:default:train:{source_index}",
        "source_index": source_index,
        "uuid": row.get("uuid"),
        "source": row.get("source"),
        "problem": problem,
        "generations": [generations[i] for i in kept],
        "correctness_math_verify": [True] * len(kept),
        "finish_reasons": [arrays["finish_reasons"][i] for i in kept],
        "generation_indices": kept,
        "problem_generation_tokens": lengths,
        "completion_evidence": evidence,
    }, counts


def validated_marker(folder, name):
    path = folder / name
    if not path.exists():
        return None
    marker = json.loads(path.read_text())
    for filename, expected in marker["files"].items():
        if checksum(folder / filename) != expected:
            raise ValueError(f"Cached output changed: {folder / filename}")
    return marker


def filter_traces(folder, token):
    """Stream source rows; resume at completed batches without re-tokenizing them."""
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    folder = Path(folder)
    done = validated_marker(folder, "filter_complete.json")
    if done:
        print("Reusing completed trace filtering:", done["counts"])
        return done
    manifest = json.loads((folder / "manifest.json").read_text())
    settings = manifest["settings"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["tokenizer"],
        revision=manifest["tokenizer_revision"],
        token=token,
        trust_remote_code=False,
    )
    stream = load_dataset(
        SOURCE,
        name="default",
        split="train",
        streaming=True,
        revision=manifest["source_revision"],
        token=token,
    ).select_columns(COLUMNS)
    split_info = (stream.info.splits or {}).get("train")
    expected_rows = split_info.num_examples if split_info is not None else None
    batch_dir = folder / "filtered"
    batch_dir.mkdir(exist_ok=True)
    counts, by_source, files = Counter(), {}, {}
    start, batch = 0, 0
    while (batch_dir / f"{batch:06d}.json").exists():
        marker = validated_marker(batch_dir, f"{batch:06d}.json")
        if marker["start"] != start:
            raise ValueError("Non-contiguous filter checkpoint")
        counts.update(marker["counts"])
        for source, values in marker["by_source"].items():
            by_source.setdefault(source, Counter()).update(values)
        files.update({f"filtered/{n}": h for n, h in marker["files"].items()})
        start += marker["source_rows"]
        batch += 1
    # IterableDataset.skip may reread preceding source shards, but their tokenization is cached.
    iterator = iter(stream.skip(start))
    with tqdm(
        total=expected_rows, desc="Filter source problems", unit="problem", initial=start
    ) as progress:
        exhausted = False
        while not exhausted:
            output, local, local_sources = [], Counter(), {}
            consumed = 0
            for offset in range(settings["batch_size"]):
                try:
                    row = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                kept, stats = filter_problem(
                    row,
                    start + offset,
                    tokenizer,
                    settings["max_tokens"],
                    settings["missing_finish"],
                )
                local.update(stats)
                source = str(row.get("source"))
                local_sources.setdefault(source, Counter()).update(stats)
                if kept:
                    output.append(kept)
                consumed += 1
                progress.update()
            if not consumed:
                break
            filename = f"{batch:06d}.jsonl"
            write_jsonl(batch_dir / filename, output)
            marker = {
                "start": start,
                "source_rows": consumed,
                "counts": dict(local),
                "by_source": local_sources,
                "files": {filename: checksum(batch_dir / filename)},
            }
            write_json(batch_dir / f"{batch:06d}.json", marker)
            counts.update(local)
            for source, stats in local_sources.items():
                by_source.setdefault(source, Counter()).update(stats)
            files[f"filtered/{filename}"] = marker["files"][filename]
            start += consumed
            batch += 1
    if not start:
        raise ValueError("Source dataset was empty")
    if expected_rows is not None and start != expected_rows:
        raise ValueError(f"Source row count differs: {start} != {expected_rows}")
    result = {"counts": dict(counts), "by_source": by_source, "files": files}
    write_json(folder / "filter_complete.json", result)
    print(json.dumps({"counts": counts, "by_source": by_source}, indent=2))
    return result


def load_references(suite, token):
    from datasets import load_dataset

    references = {}
    for name, spec in suite["datasets"].items():
        original = list(
            load_dataset(
                spec["repo"],
                name=spec["config"],
                split=spec["split"],
                revision=spec["revision"],
                token=token,
            )
        )
        if len(original) != spec["rows"] or digest(original) != spec["content_digest"]:
            raise ValueError(f"Evaluation dataset changed: {name}")
        ids = []
        references[name] = []
        for row in original:
            identifier = row
            for part in spec["id_column"].split("."):
                identifier = identifier[part]
            ids.append(identifier)
            references[name].append({"id": identifier, "problem": row[spec["problem_column"]]})
        if ids != spec["ids"]:
            raise ValueError(f"Evaluation IDs changed: {name}")
    return references


def output_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("source_index", pa.int64()),
            ("uuid", pa.string()),
            ("source", pa.string()),
            ("problem", pa.string()),
            ("generations", pa.list_(pa.string())),
            ("correctness_math_verify", pa.list_(pa.bool_())),
            ("finish_reasons", pa.list_(pa.string())),
            ("generation_indices", pa.list_(pa.int64())),
            ("problem_generation_tokens", pa.list_(pa.int64())),
            ("completion_evidence", pa.list_(pa.string())),
        ]
    )


def remove_overlap(rows, references):
    registry = {name: {"role": "eval"} for name in references}
    registry["openr1"] = {"role": "train"}
    clean, reports, near = decontaminate(
        {**references, "openr1": rows},
        registry,
        ngram_size=8,
        coverage_threshold=0.7,
        near_miss_min=0.5,
    )
    return clean["openr1"], reports["openr1"], near


def decontaminate_filtered(folder, token):
    """Apply the existing pairwise rule to problems; remove all traces of matched rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    folder = Path(folder)
    filtered = validated_marker(folder, "filter_complete.json")
    if filtered is None:
        raise ValueError("Complete step 1 first")
    done = validated_marker(folder, "decontamination_complete.json")
    if done:
        print("Reusing decontaminated output:", done["counts"])
        return done
    manifest = json.loads((folder / "manifest.json").read_text())
    references = load_references(manifest["settings"]["eval_suite"], token)
    export = folder / "export"
    (export / "data").mkdir(parents=True, exist_ok=True)
    removed, near_misses, counts, per_eval = [], [], Counter(), Counter()
    by_source, shards = {}, []
    for filename in tqdm(sorted(filtered["files"]), desc="Decontaminate batches", unit="batch"):
        rows = read_jsonl(folder / filename)
        clean, report, near = remove_overlap(rows, references)
        removed.extend(report["removed"])
        near_misses.extend(near)
        per_eval.update(report["removal_counts_by_eval_set"])
        kept_ids = {r["id"] for r in clean}
        for row in rows:
            status = "kept" if row["id"] in kept_ids else "removed"
            values = {f"{status}_problems": 1, f"{status}_traces": len(row["generations"])}
            counts.update(values)
            by_source.setdefault(str(row["source"]), Counter()).update(values)
        if clean:
            name = f"data/train-{len(shards):05d}.parquet"
            tmp = export / (name + ".tmp")
            pq.write_table(
                pa.Table.from_pylist(clean, schema=output_schema()), tmp, compression="zstd"
            )
            tmp.replace(export / name)
            shards.append(name)
    if not counts["kept_problems"]:
        raise ValueError("No training examples remain; inspect filter counts before uploading")
    report = {
        "normalization_version": NORMALIZATION_VERSION,
        "ngram_size": 8,
        "coverage_threshold": 0.7,
        "near_miss_min": 0.5,
        "counts": dict(counts),
        "removal_counts_by_eval_set": dict(per_eval),
        "by_source": by_source,
        "removed": removed,
        "near_misses": near_misses,
    }
    write_json(export / "decontamination_report.json", report)
    write_json(
        export / "filter_report.json",
        {"counts": filtered["counts"], "by_source": filtered["by_source"]},
    )
    write_json(export / "provenance.json", manifest)
    (export / "source_card.md").write_text((folder / "source_card.md").read_text())
    settings = manifest["settings"]
    (export / "README.md").write_text(
        "---\nlicense: apache-2.0\nlanguage:\n- en\nconfigs:\n- config_name: default\n"
        "  data_files:\n  - split: train\n    path: data/train-*.parquet\n---\n\n"
        f"# {TARGET_NAME}\n\nDerived from [{SOURCE}](https://huggingface.co/datasets/{SOURCE}), "
        f"default/train at `{manifest['source_revision']}`. Original source card: [source_card.md](source_card.md).\n\n"
        f"{counts['kept_problems']} problems, {counts['kept_traces']} retained traces. "
        "One row per problem; `generations` contains ALL retained traces. Parallel arrays refer to "
        "those traces; `generation_indices` records original positions. Text is unchanged. "
        "Only math-verify-correct traces are included. Explicit length stops and incomplete reasoning "
        f"are excluded. Missing finish reason policy: `{settings['missing_finish']}`; "
        "fallback does not prove the final answer was untruncated and is identified in `completion_evidence`.\n\n"
        f"Length <= {settings['max_tokens']} using `{settings['tokenizer']}` at "
        f"`{manifest['tokenizer_revision']}`: tokens(problem) + tokens(generation), "
        "without special tokens, chat formatting, padding or truncation. Training templates add overhead.\n\n"
        "Train/eval overlap removal: normalization v2, distinct 8-grams, coverage >= 0.7 per "
        "evaluation problem. Short eval problems use normalized exact substring matching. "
        "References: AIME 2024/2025/2026, AMC23, MATH-500; exact pins in provenance.json. "
        "No within/between-training deduplication or blanket source-label exclusions. "
        "See filter_report.json and decontamination_report.json for counts, matches and near misses.\n"
    )
    names = shards + [
        "README.md",
        "source_card.md",
        "provenance.json",
        "filter_report.json",
        "decontamination_report.json",
    ]
    result = {"counts": dict(counts), "files": {"export/" + n: checksum(export / n) for n in names}}
    write_json(folder / "decontamination_complete.json", result)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"removed", "near_misses"}}, indent=2
        )
    )
    print("Near misses retained:", len(near_misses))
    return result


def publish(folder, username, private, token):
    """Upload only verified staged files in one commit; preserve unrelated Hub datasets."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    folder = Path(folder)
    ready = validated_marker(folder, "decontamination_complete.json")
    if not ready:
        raise ValueError("Complete decontamination before upload")
    if not username or "/" in username:
        raise ValueError("Set HF_USERNAME to your Hugging Face user or organization")
    repo = f"{username}/{TARGET_NAME}"
    api = HfApi(token=token)
    exists = api.repo_exists(repo, repo_type="dataset")
    if exists:
        try:
            old = json.loads(
                Path(
                    hf_hub_download(repo, "provenance.json", repo_type="dataset", token=token)
                ).read_text()
            )
        except EntryNotFoundError as e:
            raise ValueError(
                "Existing target has no preparation provenance; refusing to overwrite"
            ) from e
        current = json.loads((folder / "manifest.json").read_text())
        if old["signature"] != current["signature"]:
            raise ValueError(
                "Target contains a different preparation run; choose a different HF_USERNAME or review it first"
            )
    else:
        api.create_repo(repo, repo_type="dataset", private=private)
    names = [str(Path(n).relative_to("export")) for n in ready["files"]]
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(folder / "export"),
        allow_patterns=names,
        commit_message="Prepare correct, length-filtered, decontaminated OpenR1 default",
    )
    receipt = {
        "repo": repo,
        "revision": commit.oid,
        "url": f"https://huggingface.co/datasets/{repo}",
        "counts": ready["counts"],
        "files": ready["files"],
    }
    write_json(folder / "upload_receipt.json", receipt)
    print(receipt["url"], "revision:", receipt["revision"])
    return receipt
