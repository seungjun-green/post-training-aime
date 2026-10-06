"""Add a regenerated answer column without replacing existing s1 data."""

from datasets import Value

from common.io import digest


def merge_answer(source, current, exported, statuses, config):
    source_rows = list(source)
    spec = config["source"]
    count = len(source_rows)
    if count != spec["expected_rows"] or digest(source_rows) != spec["content_digest"]:
        raise ValueError("Pinned source differs from the expected dataset")
    if any(len(rows) != count for rows in (current, exported, statuses)):
        raise ValueError("Source, current, export, and status row counts must match")
    column, answer = config["new_column"], config["answer_column"]
    if column in source.column_names:
        raise ValueError("New column must not replace an original source column")
    values = []
    for i, (old, live, new, status) in enumerate(zip(source_rows, current, exported, statuses, strict=True)):
        if any(key not in live or key not in new or live[key] != value or new[key] != value
               for key, value in old.items()):
            raise ValueError(f"Original fields or row order differ at row {i}")
        if status.get("source_index") != i:
            raise ValueError(f"Status row alignment differs at row {i}")
        if answer not in new:
            raise ValueError(f"Missing answer column at row {i}")
        value = new[answer]
        if status.get("status") == "complete":
            if (not isinstance(value, str) or not value.strip()
                    or status.get("finish_reason") != "stop"
                    or status.get("response_model") != config["model"]):
                raise ValueError(f"Invalid completed answer at row {i}")
        elif value is not None:
            raise ValueError(f"Incomplete row {i} must have a null answer")
        if live.get(column) is not None and live[column] != value:
            raise ValueError(f"Would overwrite an existing answer at row {i}")
        values.append(value)
    if not any(value is not None for value in values):
        raise ValueError("Export contains no completed answers")
    merged = current.remove_columns(column) if column in current.column_names else current
    merged = merged.add_column(column, values, feature=Value("string"))
    return merged, {"rows": count, "answers": sum(v is not None for v in values),
                    "null_answers": sum(v is None for v in values), "new_column": column,
                    "dataset_digest": digest(list(merged))}


def train_parquet_files(repo_files):
    """Accept only the known complete default/train shard layout."""
    import re

    paths = sorted(p for p in repo_files if p.startswith("data/") and p.endswith(".parquet"))
    parsed = [re.fullmatch(r"data/train-(\d{5})-of-(\d{5})\.parquet", p) for p in paths]
    if not paths or not all(parsed):
        raise ValueError("Expected only default train Parquet shards under data/")
    if (any(int(m[2]) != len(paths) for m in parsed)
            or [int(m[1]) for m in parsed] != list(range(len(paths)))):
        raise ValueError("Incomplete or duplicate train Parquet shard layout")
    return paths
