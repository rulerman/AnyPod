from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pymupdf
import pymupdf4llm

from ..core.io import write_json, write_jsonl
from ..core.chunking import build_chunks_for_range, extract_paragraphs, iter_chunk_source_records
from ..core.common import (
    build_book_id,
    normalize_title,
    page_end_to_char,
    page_start_to_char,
    sanitize_filename,
    save_text,
    section_type_from_title,
)


logger = logging.getLogger("anypod.step1")


def extract_page_texts(pdf_path: Path) -> Tuple[str, List[Dict[str, int]], List[List[Any]]]:
    with pymupdf.open(str(pdf_path)) as doc:
        toc = doc.get_toc(simple=True)
        page_maps: List[Dict[str, int]] = []
        parts: List[str] = []
        cursor = 0

        for page_index in range(len(doc)):
            page_text = doc[page_index].get_text("text").strip()
            if page_text:
                start = cursor
                parts.append(page_text)
                cursor += len(page_text)
                page_maps.append(
                    {
                        "page_no": page_index + 1,
                        "char_start": start,
                        "char_end": cursor,
                    }
                )
                parts.append("\n\n")
                cursor += 2
            else:
                page_maps.append(
                    {
                        "page_no": page_index + 1,
                        "char_start": cursor,
                        "char_end": cursor,
                    }
                )

    full_text = "".join(parts)
    if not full_text.strip():
        full_text = pymupdf4llm.to_markdown(str(pdf_path))
        page_maps = [{"page_no": 1, "char_start": 0, "char_end": len(full_text)}]

    return full_text, page_maps, toc


def detect_sections_from_bookmarks(
    toc: List[List[Any]],
    page_maps: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    if not toc:
        return []

    valid_items = [item for item in toc if len(item) >= 3 and int(item[2]) > 0]
    if not valid_items:
        return []

    min_level = min(int(item[0]) for item in valid_items)
    top_items = [item for item in valid_items if int(item[0]) == min_level]
    if not top_items:
        return []

    total_pages = page_maps[-1]["page_no"] if page_maps else 1
    sections: List[Dict[str, Any]] = []
    for index, item in enumerate(top_items, start=1):
        title = normalize_title(str(item[1]))
        page_start = max(1, min(int(item[2]), total_pages))
        if index < len(top_items):
            next_page = max(1, min(int(top_items[index][2]), total_pages))
            page_end = max(page_start, next_page)
        else:
            page_end = total_pages

        sections.append(
            {
                "section_id": f"CH{index:02d}",
                "section_type": section_type_from_title(title),
                "title": title,
                "page_start": page_start,
                "page_end": page_end,
                "char_start": page_start_to_char(page_start, page_maps),
                "char_end": page_end_to_char(page_end, page_maps),
            }
        )
    return sections


class Step1Parse:
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        chunk_max_words: int = 3000,
        force: bool = False,
        ignore_bookmarks: bool = False,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.chunk_max_words = chunk_max_words
        self.force = force
        self.ignore_bookmarks = ignore_bookmarks
        self.book_structure_path = self.output_dir / "book_structure.json"
        self.chunk_source_map_path = self.output_dir / "chunk_source_map.jsonl"
        self.full_text_path = self.output_dir / "full_text.txt"
        self.raw_text_dir = self.output_dir / "raw_text"
        self.raw_chunk_dir = self.raw_text_dir / "chunks"
        self.input_copy_dir = self.output_dir / "input"

    def run(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_text_dir.mkdir(parents=True, exist_ok=True)
        self.raw_chunk_dir.mkdir(parents=True, exist_ok=True)
        self.input_copy_dir.mkdir(parents=True, exist_ok=True)

        dest_input = self.input_copy_dir / self.input_path.name
        if self.force or not dest_input.exists():
            shutil.copy2(self.input_path, dest_input)

        logger.info("Parsing source file: %s", self.input_path)
        source_type = self.input_path.suffix.lower().lstrip(".")
        if source_type == "pdf":
            full_text, page_maps, toc = extract_page_texts(self.input_path)
        elif source_type in {"txt", "md"}:
            full_text = self.input_path.read_text(encoding="utf-8")
            page_maps = [{"page_no": 1, "char_start": 0, "char_end": len(full_text)}]
            toc = []
        else:
            raise ValueError(f"Unsupported input format: {self.input_path.suffix}")

        save_text(self.full_text_path, full_text)

        logger.info("Detecting section structure")
        bookmark_sections = [] if self.ignore_bookmarks else detect_sections_from_bookmarks(toc, page_maps)
        sections = bookmark_sections
        sections_detected = bool(sections)
        structure_mode = "bookmarks" if sections_detected else "none"
        toc_detected = bool(sections_detected)

        book_structure = {
            "book_id": build_book_id(self.input_path),
            "book_title": self.input_path.stem,
            "source_type": source_type,
            "has_bookmarks": bool(toc),
            "toc_detected": toc_detected,
            "structure_mode": structure_mode,
            "sections_detected": sections_detected,
            "sections": sections,
        }
        write_json(book_structure, self.book_structure_path)

        logger.info("Building chunks")
        paragraphs = extract_paragraphs(full_text)
        chunk_maps: List[Dict[str, Any]] = []

        if sections_detected:
            for section in sections:
                section_text = full_text[section["char_start"]:section["char_end"]]
                save_text(
                    self.raw_text_dir / f"{section['section_id']}_{sanitize_filename(section['title'])}.txt",
                    section_text,
                )
                section_chunks = build_chunks_for_range(
                    section_id=section["section_id"],
                    start_char=int(section["char_start"]),
                    end_char=int(section["char_end"]),
                    page_maps=page_maps,
                    paragraphs=paragraphs,
                    full_text=full_text,
                    max_words=self.chunk_max_words,
                )
                chunk_maps.extend(section_chunks)
                for chunk in section_chunks:
                    chunk_text = full_text[int(chunk["char_start"]):int(chunk["char_end"])]
                    save_text(self.raw_chunk_dir / f"{chunk['chunk_id']}.txt", chunk_text)
        else:
            save_text(self.raw_text_dir / "CH00_full_text.txt", full_text)
            whole_chunks = build_chunks_for_range(
                section_id="CH00",
                start_char=0,
                end_char=len(full_text),
                page_maps=page_maps,
                paragraphs=paragraphs,
                full_text=full_text,
                max_words=self.chunk_max_words,
            )
            chunk_maps.extend(whole_chunks)
            for chunk in whole_chunks:
                chunk_text = full_text[int(chunk["char_start"]):int(chunk["char_end"])]
                save_text(self.raw_chunk_dir / f"{chunk['chunk_id']}.txt", chunk_text)

        write_jsonl(list(iter_chunk_source_records(chunk_maps)), self.chunk_source_map_path)

        return {
            "book_structure": str(self.book_structure_path),
            "chunk_source_map": str(self.chunk_source_map_path),
            "full_text": str(self.full_text_path),
            "raw_chunk_dir": str(self.raw_chunk_dir),
            "chunk_max_words": self.chunk_max_words,
            "ignore_bookmarks": self.ignore_bookmarks,
            "sections_detected": sections_detected,
            "sections_count": len(sections),
            "chunks_count": len(chunk_maps),
        }
