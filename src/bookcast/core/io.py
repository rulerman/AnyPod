from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path, ensure_ascii: bool = False) -> None:
    path = Path(path)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=ensure_ascii, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> List[Any]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(data_list: List[Any], path: str | Path) -> None:
    path = Path(path)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(item: Any, path: str | Path) -> None:
    path = Path(path)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
