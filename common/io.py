"""Durable JSON files and append-only journals with recoverable interrupted tails."""

import hashlib
import json
import os
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path, *, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    mode = "r+b" if repair_tail else "rb"
    with path.open(mode) as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if repair_tail and not f.read(1) and not line.endswith(b"\n"):
                    f.truncate(start)
                    break
                raise ValueError(f"Corrupt JSONL at byte {start}: {path}") from None
            rows.append(row)
            if repair_tail and not line.endswith(b"\n"):
                f.seek(0, 2)
                f.write(b"\n")
    return rows


def latest_by_id(path):
    return {str(row["id"]): row for row in read_jsonl(path, repair_tail=True)}
