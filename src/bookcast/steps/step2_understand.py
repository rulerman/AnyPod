from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from ..core.io import read_json, read_jsonl, write_json
from ..core.llm_client import SimpleLLMClient
from ..core.prompts import build_chunk_card_prompt, build_pseudo_section_prompt, build_section_card_prompt


logger = logging.getLogger("anypod.step2")


CHUNK_CARD_REQUIRED_SCHEMA = {
    "summary": None,
    "key_points": None,
    "core_terms": [
        {
            "term": None,
            "explanation": None,
        }
    ],
}

SECTION_CARD_REQUIRED_SCHEMA = {
    "title": None,
    "summary": None,
    "thesis_or_function": None,
    "key_points": None,
    "core_terms": [
        {
            "term": None,
            "explanation": None,
        }
    ],
}

PSEUDO_SECTION_REQUIRED_SCHEMA = {
    "pseudo_sections": [
        {
            "source_chunk_ids": None,
            "title": None,
            "summary": None,
            "thesis_or_function": None,
            "key_points": None,
            "core_terms": [
                {
                    "term": None,
                    "explanation": None,
                }
            ],
        }
    ],
}


def build_attempt_output_path(raw_output_path: Path, attempt_index: int) -> Path:
    if attempt_index == 1:
        return raw_output_path
    return raw_output_path.with_name(
        f"{raw_output_path.stem}_attempt_{attempt_index:02d}{raw_output_path.suffix}"
    )


def run_generate_json_task(task: Dict[str, Any]) -> Dict[str, Any]:
    llm_client = SimpleLLMClient(**dict(task["llm_client_config"]))
    prompt = str(task["prompt"])
    raw_output_path = Path(str(task["raw_output_path"]))
    item_label = str(task["item_label"])
    required_schema = task.get("required_schema")
    max_retries = max(1, int(task["max_retries"]))

    last_exception: Exception | None = None
    for attempt_index in range(1, max_retries + 1):
        attempt_output_path = build_attempt_output_path(raw_output_path, attempt_index)
        try:
            llm_result = llm_client.generate_json(
                prompt=prompt,
                raw_output_path=attempt_output_path,
                required_schema=required_schema,
            )
            return {
                "task_index": int(task["task_index"]),
                "item_key": str(task["item_key"]),
                "llm_result": llm_result,
                "failed": False,
            }
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "%s generation failed on attempt %s/%s. Raw output: %s. Error: %s",
                item_label,
                attempt_index,
                max_retries,
                attempt_output_path,
                exc,
            )

    logger.warning(
        "%s failed %s times in a row and fell back to an empty JSON object. Last error: %s",
        item_label,
        max_retries,
        last_exception,
    )
    return {
        "task_index": int(task["task_index"]),
        "item_key": str(task["item_key"]),
        "llm_result": {},
        "failed": True,
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
            or item.get("ex Explanation")
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


def ensure_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


class Step2Understand:
    def __init__(
        self,
        output_dir: Path,
        llm_client: SimpleLLMClient,
        max_retries: int = 5,
        num_workers: int = 4,
    ) -> None:
        self.output_dir = output_dir
        self.llm_client = llm_client
        self.max_retries = max(1, int(max_retries))
        self.num_workers = max(1, int(num_workers))
        self.book_structure_path = self.output_dir / "book_structure.json"
        self.chunk_source_map_path = self.output_dir / "chunk_source_map.jsonl"
        self.full_text_path = self.output_dir / "full_text.txt"
        self.chunk_card_dir = self.output_dir / "chunk_cards"
        self.section_card_dir = self.output_dir / "section_cards"
        self.raw_chunk_dir = self.output_dir / "raw_text" / "chunks"
        self.llm_raw_chunk_dir = self.output_dir / "llm_raw" / "chunk_cards"
        self.llm_raw_section_dir = self.output_dir / "llm_raw" / "section_cards"

    def build_attempt_output_path(self, raw_output_path: Path, attempt_index: int) -> Path:
        return build_attempt_output_path(raw_output_path, attempt_index)

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
            "%s failed %s times in a row and fell back to an empty JSON object. Last error: %s",
            item_label,
            self.max_retries,
            last_exception,
        )
        return {}

    def build_llm_task(
        self,
        task_index: int,
        item_key: str,
        prompt: str,
        raw_output_path: Path,
        item_label: str,
        required_schema: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        return {
            "task_index": task_index,
            "item_key": item_key,
            "prompt": prompt,
            "raw_output_path": str(raw_output_path),
            "item_label": item_label,
            "required_schema": required_schema,
            "max_retries": self.max_retries,
            "llm_client_config": self.llm_client.to_runtime_config(),
        }

    def run_json_tasks_in_parallel(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not tasks:
            return []
        if self.num_workers <= 1 or len(tasks) == 1:
            return [run_generate_json_task(task) for task in tasks]

        results: List[Dict[str, Any]] = []
        max_workers = min(self.num_workers, len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(run_generate_json_task, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning(
                        "Parallel task %s failed and fell back to an empty JSON object. Error: %s",
                        task["item_label"],
                        exc,
                    )
                    results.append(
                        {
                            "task_index": int(task["task_index"]),
                            "item_key": str(task["item_key"]),
                            "llm_result": {},
                            "failed": True,
                        }
                    )

        results.sort(key=lambda item: int(item["task_index"]))
        return results

    def load_chunk_text(self, chunk_record: Dict[str, Any], full_text: str) -> str:
        chunk_file = self.raw_chunk_dir / f"{chunk_record['chunk_id']}.txt"
        if chunk_file.exists():
            return chunk_file.read_text(encoding="utf-8")
        return full_text[int(chunk_record["char_start"]):int(chunk_record["char_end"])]

    def build_chunk_card(self, chunk_record: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        paragraph_count = max(
            0,
            int(chunk_record["paragraph_end"]) - int(chunk_record["paragraph_start"]) + 1,
        )
        return {
            "chunk_id": chunk_record["chunk_id"],
            "parent_section_id": chunk_record["parent_section_id"],
            "chunk_index": int(chunk_record["chunk_index"]),
            "source_locator": {
                "page_start": int(chunk_record["page_start"]),
                "page_end": int(chunk_record["page_end"]),
                "char_start": int(chunk_record["char_start"]),
                "char_end": int(chunk_record["char_end"]),
                "paragraph_start": int(chunk_record["paragraph_start"]),
                "paragraph_end": int(chunk_record["paragraph_end"]),
            },
            "length_stats": {
                "char_count": int(chunk_record["char_count"]),
                "paragraph_count": paragraph_count,
            },
            "summary": ensure_string(llm_result.get("summary")),
            "key_points": ensure_string_list(llm_result.get("key_points")),
            "core_terms": ensure_core_terms(llm_result.get("core_terms")),
        }

    def build_section_card(
        self,
        section_record: Dict[str, Any],
        chunk_cards: List[Dict[str, Any]],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_chunk_ids = [card["chunk_id"] for card in chunk_cards]
        first_locator = chunk_cards[0]["source_locator"]
        last_locator = chunk_cards[-1]["source_locator"]
        return {
            "section_id": section_record["section_id"],
            "section_type": section_record["section_type"],
            "title": ensure_string(llm_result.get("title"), section_record["title"]),
            "source_chunk_ids": source_chunk_ids,
            "source_locator": {
                "page_start": int(first_locator["page_start"]),
                "page_end": int(last_locator["page_end"]),
                "char_start": int(first_locator["char_start"]),
                "char_end": int(last_locator["char_end"]),
            },
            "summary": ensure_string(llm_result.get("summary")),
            "thesis_or_function": ensure_string(llm_result.get("thesis_or_function")),
            "key_points": ensure_string_list(llm_result.get("key_points")),
            "core_terms": ensure_core_terms(llm_result.get("core_terms")),
        }

    def build_pseudo_section_card(
        self,
        pseudo_section_id: str,
        chunk_cards: List[Dict[str, Any]],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_chunk_ids = [card["chunk_id"] for card in chunk_cards]
        first_locator = chunk_cards[0]["source_locator"]
        last_locator = chunk_cards[-1]["source_locator"]
        return {
            "section_id": pseudo_section_id,
            "section_type": "pseudo-section",
            "title": ensure_string(llm_result.get("title"), pseudo_section_id),
            "source_chunk_ids": source_chunk_ids,
            "source_locator": {
                "page_start": int(first_locator["page_start"]),
                "page_end": int(last_locator["page_end"]),
                "char_start": int(first_locator["char_start"]),
                "char_end": int(last_locator["char_end"]),
            },
            "summary": ensure_string(llm_result.get("summary")),
            "thesis_or_function": ensure_string(llm_result.get("thesis_or_function")),
            "key_points": ensure_string_list(llm_result.get("key_points")),
            "core_terms": ensure_core_terms(llm_result.get("core_terms")),
        }

    def run(self) -> Dict[str, Any]:
        if not self.book_structure_path.exists():
            raise FileNotFoundError(f"Missing parse artifact: {self.book_structure_path}")
        if not self.chunk_source_map_path.exists():
            raise FileNotFoundError(f"Missing parse artifact: {self.chunk_source_map_path}")
        if not self.full_text_path.exists():
            raise FileNotFoundError(f"Missing parse artifact: {self.full_text_path}")

        self.llm_raw_chunk_dir.mkdir(parents=True, exist_ok=True)
        self.llm_raw_section_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_card_dir.mkdir(parents=True, exist_ok=True)
        self.section_card_dir.mkdir(parents=True, exist_ok=True)

        book_structure = read_json(self.book_structure_path)
        chunk_records = read_jsonl(self.chunk_source_map_path)
        full_text = self.full_text_path.read_text(encoding="utf-8")
        sections = ensure_dict_list(book_structure.get("sections"))
        sections_detected = bool(book_structure.get("sections_detected")) and bool(sections)
        section_title_by_id = {
            section["section_id"]: section["title"]
            for section in sections
        }

        logger.info("Generating chunk cards in parallel, workers=%s", self.num_workers)
        chunk_tasks: List[Dict[str, Any]] = []
        for chunk_record in chunk_records:
            chunk_text = self.load_chunk_text(chunk_record, full_text)
            prompt = build_chunk_card_prompt(
                chunk_text=chunk_text,
                section_title=section_title_by_id.get(chunk_record["parent_section_id"], ""),
            )
            chunk_tasks.append(
                self.build_llm_task(
                    task_index=len(chunk_tasks),
                    item_key=str(chunk_record["chunk_id"]),
                    prompt=prompt,
                    raw_output_path=self.llm_raw_chunk_dir / f"{chunk_record['chunk_id']}.txt",
                    item_label=f"chunk {chunk_record['chunk_id']}",
                    required_schema=CHUNK_CARD_REQUIRED_SCHEMA,
                )
            )

        chunk_results = self.run_json_tasks_in_parallel(chunk_tasks)
        chunk_cards: List[Dict[str, Any]] = []
        failed_chunk_count = 0
        for chunk_record, task_result in zip(chunk_records, chunk_results):
            if task_result["failed"]:
                failed_chunk_count += 1
            chunk_card = self.build_chunk_card(chunk_record, task_result["llm_result"])
            chunk_cards.append(chunk_card)
            write_json(chunk_card, self.chunk_card_dir / f"{chunk_card['chunk_id']}.json")

        logger.info("Generating section cards in parallel, workers=%s", self.num_workers)
        section_cards: List[Dict[str, Any]] = []
        failed_section_count = 0
        if sections_detected:
            chunk_cards_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for chunk_card in chunk_cards:
                chunk_cards_by_section[chunk_card["parent_section_id"]].append(chunk_card)

            section_tasks: List[Dict[str, Any]] = []
            section_build_inputs: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
            for section_record in sections:
                section_id = section_record["section_id"]
                current_chunk_cards = chunk_cards_by_section.get(section_id, [])
                if not current_chunk_cards:
                    logger.warning(
                        "Section %s has no matching chunks. Skipping section card generation.",
                        section_id,
                    )
                    continue

                current_chunk_cards.sort(key=lambda item: item["chunk_index"])
                prompt = build_section_card_prompt(
                    section_id=section_id,
                    section_title=section_record["title"],
                    chunk_cards_json=json.dumps(current_chunk_cards, ensure_ascii=False, indent=2),
                )
                section_tasks.append(
                    self.build_llm_task(
                        task_index=len(section_tasks),
                        item_key=str(section_id),
                        prompt=prompt,
                        raw_output_path=self.llm_raw_section_dir / f"{section_id}.txt",
                        item_label=f"section {section_id}",
                        required_schema=SECTION_CARD_REQUIRED_SCHEMA,
                    )
                )
                section_build_inputs.append((section_record, current_chunk_cards))

            section_results = self.run_json_tasks_in_parallel(section_tasks)
            for (section_record, current_chunk_cards), task_result in zip(section_build_inputs, section_results):
                if task_result["failed"]:
                    failed_section_count += 1
                section_card = self.build_section_card(
                    section_record,
                    current_chunk_cards,
                    task_result["llm_result"],
                )
                section_cards.append(section_card)
                write_json(section_card, self.section_card_dir / f"{section_card['section_id']}.json")
        else:
            chunk_card_by_id = {card["chunk_id"]: card for card in chunk_cards}
            ordered_chunk_ids = [card["chunk_id"] for card in sorted(chunk_cards, key=lambda item: item["chunk_index"])]
            prompt = build_pseudo_section_prompt(
                all_chunk_cards_json=json.dumps(chunk_cards, ensure_ascii=False, indent=2),
            )
            llm_result = self.safe_generate_json(
                prompt=prompt,
                raw_output_path=self.llm_raw_section_dir / "pseudo_sections.txt",
                item_label="pseudo sections",
                required_schema=PSEUDO_SECTION_REQUIRED_SCHEMA,
            )
            pseudo_sections = ensure_dict_list(llm_result.get("pseudo_sections"))
            if not pseudo_sections:
                failed_section_count += 1
                pseudo_sections = [
                    {
                        "source_chunk_ids": ordered_chunk_ids,
                        "title": "全书内容概览",
                        "summary": "",
                        "thesis_or_function": "",
                        "key_points": [],
                        "core_terms": [],
                    }
                ]

            for index, pseudo_section in enumerate(pseudo_sections, start=1):
                source_chunk_ids = [
                    chunk_id for chunk_id in ensure_string_list(pseudo_section.get("source_chunk_ids"))
                    if chunk_id in chunk_card_by_id
                ]
                if not source_chunk_ids:
                    continue
                current_chunk_cards = [chunk_card_by_id[chunk_id] for chunk_id in source_chunk_ids]
                section_card = self.build_pseudo_section_card(
                    pseudo_section_id=f"PS{index:02d}",
                    chunk_cards=current_chunk_cards,
                    llm_result=pseudo_section,
                )
                section_cards.append(section_card)
                write_json(section_card, self.section_card_dir / f"{section_card['section_id']}.json")

        return {
            "chunk_card_dir": str(self.chunk_card_dir),
            "section_card_dir": str(self.section_card_dir),
            "llm_raw_chunk_dir": str(self.llm_raw_chunk_dir),
            "llm_raw_section_dir": str(self.llm_raw_section_dir),
            "chunk_card_count": len(chunk_cards),
            "section_card_count": len(section_cards),
            "failed_chunk_count": failed_chunk_count,
            "failed_section_count": failed_section_count,
            "max_retries": self.max_retries,
            "num_workers": self.num_workers,
            "sections_detected": sections_detected,
        }
