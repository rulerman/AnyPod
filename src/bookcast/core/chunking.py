from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from .common import char_to_page


WORD_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
INDENT_PATTERN = re.compile(r"^(?:　+|[ \t]{2,})")


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def extract_paragraphs(full_text: str) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    cursor = 0
    for line in full_text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(line)
        cursor = end
        lines.append(
            {
                "raw_text": line,
                "text": line.strip(),
                "char_start": start,
                "char_end": end,
                "is_blank": not line.strip(),
                "starts_with_indent": bool(INDENT_PATTERN.match(line)),
            }
        )

    paragraphs: List[Dict[str, Any]] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush_current() -> None:
        nonlocal current_start, current_end
        if current_start is None or current_end is None:
            return
        text = full_text[current_start:current_end].strip()
        if text:
            paragraphs.append(
                {
                    "paragraph_index": len(paragraphs) + 1,
                    "text": text,
                    "char_start": current_start,
                    "char_end": current_end,
                }
            )
        current_start = None
        current_end = None

    for line in lines:
        if line["is_blank"]:
            flush_current()
            continue

        if current_start is None:
            current_start = int(line["char_start"])
            current_end = int(line["char_end"])
            continue

        if line["starts_with_indent"]:
            flush_current()
            current_start = int(line["char_start"])
            current_end = int(line["char_end"])
            continue

        current_end = int(line["char_end"])

    flush_current()
    return paragraphs


def split_large_text(text: str, max_words: int) -> List[str]:
    if count_words(text) <= max_words:
        return [text]

    matches = list(WORD_PATTERN.finditer(text))
    if not matches:
        return [text]

    pieces: List[str] = []
    chunk_start = 0
    for index in range(max_words, len(matches), max_words):
        split_pos = matches[index].start()
        pieces.append(text[chunk_start:split_pos])
        chunk_start = split_pos
    pieces.append(text[chunk_start:])
    return [piece for piece in pieces if piece]


def build_chunks_for_range(
    section_id: str,
    start_char: int,
    end_char: int,
    page_maps: List[Dict[str, int]],
    paragraphs: List[Dict[str, Any]],
    full_text: str,
    max_words: int,
) -> List[Dict[str, Any]]:
    relevant = [
        p for p in paragraphs
        if not (p["char_end"] <= start_char or p["char_start"] >= end_char)
    ]

    if not relevant:
        text = full_text[start_char:end_char]
        if not text.strip():
            return []
        relevant = [
            {
                "paragraph_index": 1,
                "text": text.strip(),
                "char_start": start_char,
                "char_end": end_char,
            }
        ]

    chunk_maps: List[Dict[str, Any]] = []
    current_units: List[Dict[str, Any]] = []
    current_words = 0
    chunk_index = 1

    def flush_current() -> None:
        nonlocal current_units, current_words, chunk_index
        if not current_units:
            return

        chunk_start = int(current_units[0]["char_start"])
        chunk_end = int(current_units[-1]["char_end"])
        raw_text = full_text[chunk_start:chunk_end]
        if not raw_text.strip():
            current_units = []
            current_words = 0
            return

        parts = split_large_text(raw_text, max_words)
        part_offset = 0
        for part_text in parts:
            if not part_text.strip():
                part_offset += len(part_text)
                continue
            local_start = raw_text.find(part_text, part_offset)
            if local_start < 0:
                local_start = part_offset
            local_end = local_start + len(part_text)
            abs_start = chunk_start + local_start
            abs_end = chunk_start + local_end
            page_start = char_to_page(abs_start, page_maps)
            page_end = char_to_page(max(abs_start, abs_end - 1), page_maps)

            overlapping = [
                u for u in current_units
                if not (u["char_end"] <= abs_start or u["char_start"] >= abs_end)
            ]
            paragraph_start = (
                int(overlapping[0]["paragraph_index"])
                if overlapping else int(current_units[0]["paragraph_index"])
            )
            paragraph_end = (
                int(overlapping[-1]["paragraph_index"])
                if overlapping else int(current_units[-1]["paragraph_index"])
            )

            chunk_id = f"{section_id}_CK{chunk_index:04d}"
            chunk_maps.append(
                {
                    "chunk_id": chunk_id,
                    "parent_section_id": section_id,
                    "chunk_index": chunk_index,
                    "page_start": page_start,
                    "page_end": page_end,
                    "char_start": abs_start,
                    "char_end": abs_end,
                    "paragraph_start": paragraph_start,
                    "paragraph_end": paragraph_end,
                    "char_count": len(part_text),
                }
            )
            chunk_index += 1
            part_offset = local_end

        current_units = []
        current_words = 0

    for unit in relevant:
        unit_text = full_text[unit["char_start"]:unit["char_end"]]
        word_count = count_words(unit_text)
        if current_units and current_words + word_count > max_words:
            flush_current()
        current_units.append(unit)
        current_words += word_count
    flush_current()

    return chunk_maps


def iter_chunk_source_records(chunk_maps: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for item in chunk_maps:
        yield {
            "chunk_id": item["chunk_id"],
            "parent_section_id": item["parent_section_id"],
            "chunk_index": item["chunk_index"],
            "page_start": item["page_start"],
            "page_end": item["page_end"],
            "char_start": item["char_start"],
            "char_end": item["char_end"],
            "paragraph_start": item["paragraph_start"],
            "paragraph_end": item["paragraph_end"],
            "char_count": item["char_count"],
        }
