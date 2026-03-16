from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..core.io import read_json, write_json
from ..core.llm_client import SimpleLLMClient
from ..core.prompts import build_book_charter_prompt


logger = logging.getLogger("anypod.step3")


BOOK_CHARTER_REQUIRED_SCHEMA = {
    "book_summary": None,
    "global_theme": None,
    "core_argument_or_mainline": None,
    "global_terms": [
        {
            "term": None,
            "explanation": None,
        }
    ],
    "planning_notes": None,
}


def ensure_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def ensure_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = ensure_string(item)
        if text:
            result.append(text)
    return result


def ensure_core_terms(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        term = ensure_string(item.get("term"))
        explanation = ensure_string(
            item.get("explanation")
            or item.get("Explanation")
            or item.get("definition")
            or item.get("desc")
        )
        if term:
            result.append(
                {
                    "term": term,
                    "explanation": explanation,
                }
            )
    return result


def list_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


class Step3BookCharter:
    def __init__(self, output_dir: Path, llm_client: SimpleLLMClient, max_retries: int = 5) -> None:
        self.output_dir = output_dir
        self.llm_client = llm_client
        self.max_retries = max(1, int(max_retries))
        self.book_structure_path = self.output_dir / "book_structure.json"
        self.chunk_card_dir = self.output_dir / "chunk_cards"
        self.section_card_dir = self.output_dir / "section_cards"
        self.book_charter_path = self.output_dir / "book_charter.json"
        self.llm_raw_dir = self.output_dir / "llm_raw" / "book_charter"

    def build_attempt_output_path(self, raw_output_path: Path, attempt_index: int) -> Path:
        if attempt_index == 1:
            return raw_output_path
        return raw_output_path.with_name(
            f"{raw_output_path.stem}_attempt_{attempt_index:02d}{raw_output_path.suffix}"
        )

    def safe_generate_json(
        self,
        prompt: str,
        raw_output_path: Path,
        item_label: str,
        required_schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        last_exception: Exception | None = None
        for attempt_index in range(1, self.max_retries + 1):
            attempt_output_path = self.build_attempt_output_path(raw_output_path, attempt_index)
            try:
                return self.llm_client.generate_json(
                    prompt=prompt,
                    raw_output_path=attempt_output_path,
                    required_schema=required_schema,
                )
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "%s generation failed on attempt %s/%s. Raw output: %s. Error: %s",
                    item_label,
                    attempt_index,
                    self.max_retries,
                    attempt_output_path,
                    exc,
                )

        logger.warning(
            "%s failed %s times in a row and fell back to the default empty result. Last error: %s",
            item_label,
            self.max_retries,
            last_exception,
        )
        return {}

    def load_chunk_cards(self) -> List[Dict[str, Any]]:
        return [read_json(path) for path in list_json_files(self.chunk_card_dir)]

    def load_section_cards(self) -> List[Dict[str, Any]]:
        return [read_json(path) for path in list_json_files(self.section_card_dir)]

    def build_chunk_briefs(self, chunk_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        briefs: List[Dict[str, Any]] = []
        for card in chunk_cards:
            briefs.append(
                {
                    "chunk_id": card.get("chunk_id"),
                    "parent_section_id": card.get("parent_section_id"),
                    "chunk_index": card.get("chunk_index"),
                    "summary": card.get("summary", ""),
                    "key_points": card.get("key_points", []),
                    "core_terms": card.get("core_terms", []),
                }
            )
        return briefs

    def build_section_briefs(self, section_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        briefs: List[Dict[str, Any]] = []
        for card in section_cards:
            briefs.append(
                {
                    "section_id": card.get("section_id"),
                    "section_type": card.get("section_type"),
                    "title": card.get("title", ""),
                    "source_chunk_ids": card.get("source_chunk_ids", []),
                    "summary": card.get("summary", ""),
                    "thesis_or_function": card.get("thesis_or_function", ""),
                    "key_points": card.get("key_points", []),
                    "core_terms": card.get("core_terms", []),
                }
            )
        return briefs

    def normalize_book_charter(
        self,
        llm_result: Dict[str, Any],
        book_structure: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "book_id": ensure_string(book_structure.get("book_id"), "book_demo"),
            "book_title": ensure_string(book_structure.get("book_title"), "未命名图书"),
            "book_summary": ensure_string(llm_result.get("book_summary")),
            "global_theme": ensure_string(llm_result.get("global_theme")),
            "core_argument_or_mainline": ensure_string_list(llm_result.get("core_argument_or_mainline")),
            "global_terms": ensure_core_terms(llm_result.get("global_terms")),
            "planning_notes": ensure_string_list(llm_result.get("planning_notes")),
        }

    def run(self) -> Dict[str, Any]:
        if not self.book_structure_path.exists():
            raise FileNotFoundError(f"Missing understanding input: {self.book_structure_path}")
        if not self.chunk_card_dir.exists():
            raise FileNotFoundError(f"Missing understanding input: {self.chunk_card_dir}")
        if not self.section_card_dir.exists():
            raise FileNotFoundError(f"Missing understanding input: {self.section_card_dir}")

        chunk_cards = self.load_chunk_cards()
        section_cards = self.load_section_cards()
        if not chunk_cards:
            raise FileNotFoundError(f"No chunk cards found: {self.chunk_card_dir}")
        if not section_cards:
            raise FileNotFoundError(f"No section cards found: {self.section_card_dir}")

        self.llm_raw_dir.mkdir(parents=True, exist_ok=True)

        book_structure = read_json(self.book_structure_path)
        prompt = build_book_charter_prompt(
            book_title=ensure_string(book_structure.get("book_title"), "未命名图书"),
            all_section_cards_json=json.dumps(
                self.build_section_briefs(section_cards),
                ensure_ascii=False,
                indent=2,
            ),
            all_chunk_cards_json=json.dumps(
                self.build_chunk_briefs(chunk_cards),
                ensure_ascii=False,
                indent=2,
            ),
        )
        llm_result = self.safe_generate_json(
            prompt=prompt,
            raw_output_path=self.llm_raw_dir / "book_charter.txt",
            item_label="book_charter",
            required_schema=BOOK_CHARTER_REQUIRED_SCHEMA,
        )
        book_charter = self.normalize_book_charter(llm_result, book_structure)
        write_json(book_charter, self.book_charter_path)

        return {
            "book_charter": str(self.book_charter_path),
            "llm_raw_dir": str(self.llm_raw_dir),
            "chunk_card_count": len(chunk_cards),
            "section_card_count": len(section_cards),
            "max_retries": self.max_retries,
        }
