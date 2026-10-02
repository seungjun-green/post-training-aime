"""Validate regenerated s1 exports and merge only the new model columns."""

from datasets import Value

from common.io import digest


def merge_export(source, current, exported, statuses, config):
    """Preserve source/current fields; reject misalignment and output downgrades."""
    spec = config["source"]
    columns = config["new_columns"]
    source_rows = list(source)
    count = len(source_rows)
    if count != spec["expected_rows"] or digest(source_rows) != spec["content_digest"]:
        raise ValueError("Pinned source does not match the expected dataset")
    if any(len(rows) != count for rows in (current, exported, statuses)):
        raise ValueError("Source, current dataset, export, and status must have equal row counts")
    if len(columns) != 2 or len(set(columns)) != 2:
        raise ValueError("Exactly two distinct generated columns are required")
    if set(columns) & set(source.column_names):
        raise ValueError("Generated columns must not replace original columns")
    if [row.get("source_index") for row in statuses] != list(range(count)):
        raise ValueError("Status rows must be in source order without missing/duplicate indices")
    complete = 0
    for index, (old, live, new, status) in enumerate(
        zip(source_rows, current, exported, statuses, strict=True)
    ):
        if set(new) != set(old) | set(columns):
            raise ValueError(f"Unexpected or missing export columns at row {index}")
        if any(key not in live or live[key] != value or new[key] != value
               for key, value in old.items()):
            raise ValueError(f"Original data or row order changed at row {index}")
        values = [new[name] for name in columns]
        valid = all(isinstance(value, str) and value.strip() for value in values)
        if status.get("status") == "complete":
            if not valid or status.get("finish_reason") != "stop":
                raise ValueError(f"Completed row {index} has missing or truncated output")
            if status.get("response_model") != config["model"]:
                raise ValueError(f"Unexpected model at row {index}")
            complete += 1
        elif any(value is not None for value in values):
            raise ValueError(f"Unsuccessful row {index} must have null generated fields")
        for name in columns:
            # Safe to rerun an identical upload or fill previously missing results.
            if live.get(name) is not None and live[name] != new[name]:
                raise ValueError(f"Would replace an existing generated value at row {index}: {name}")
    merged = current
    for name in columns:
        if name in merged.column_names:
            merged = merged.remove_columns(name)
        merged = merged.add_column(name, [row[name] for row in exported], feature=Value("string"))
    return merged, {"rows": count, "complete": complete, "incomplete": count - complete,
                    "new_columns": columns, "dataset_digest": digest(list(merged)),
                    "correctness_filter": False, "length_filter": False}
