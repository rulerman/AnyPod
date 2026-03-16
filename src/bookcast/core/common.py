from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def normalize_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text.strip())
    return title[:120] if title else "未命名章节"


def sanitize_filename(text: str) -> str:
    name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "untitled"


def build_book_id(path: Path) -> str:
    candidate = re.sub(r"[^\w]+", "_", path.stem.strip()).strip("_").lower()
    return candidate or "book_demo"


def section_type_from_title(title: str) -> str:
    if re.search(r"前言|序言|引言|导论|preface|introduction", title, re.I):
        return "preface"
    if re.search(r"附录|后记|appendix", title, re.I):
        return "appendix"
    return "chapter"


def ensure_parent(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def to_json_string(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def char_to_page(char_pos: int, page_maps: List[Dict[str, int]]) -> int:
    if not page_maps:
        return 1
    for item in page_maps:
        if item["char_start"] <= char_pos <= item["char_end"]:
            return item["page_no"]
    if char_pos < page_maps[0]["char_start"]:
        return page_maps[0]["page_no"]
    return page_maps[-1]["page_no"]


def page_start_to_char(page_no: int, page_maps: List[Dict[str, int]]) -> int:
    for item in page_maps:
        if item["page_no"] == page_no:
            return item["char_start"]
    return 0


def page_end_to_char(page_no: int, page_maps: List[Dict[str, int]]) -> int:
    for item in page_maps:
        if item["page_no"] == page_no:
            return item["char_end"]
    return page_maps[-1]["char_end"] if page_maps else 0
