"""Local JSONL I/O. A failed run never publishes a partially written file."""
import json
import os
import tempfile
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as tmp:
        name = tmp.name
    try:
        with open(name, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
