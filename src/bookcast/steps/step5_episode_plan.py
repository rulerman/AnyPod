from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ..core.io import read_json, write_json

from ..core.llm_client import SimpleLLMClient
from ..core.prompts import build_episode_plan_prompt


logger = logging.getLogger("anypod.step5")


EPISODE_PLAN_REQUIRED_SCHEMA = {
    "total_episode_count": None,
    "episodes": [
        {
            "title": None,
            "section_ids": None,
            "covers": None,
            "neighbor_context": None,
            "must_cover": None,
            "can_skip": None,
            "forbidden_to_introduce": None,
            "hook": None,
            "recap_focus": None,
            "teaser_goal": None,
            "tone_target": None,
        }
    ],
}


def ensure_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def ensure_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text.strip().lower()).strip("_")
    return slug or "untitled"


def build_episode_plan_id(
    book_id: str,
    mode: str,
    primary_language: str,
    dialogue_mode: str,
) -> str:
    return (
        f"plan_{sanitize_slug(book_id)}_"
        f"{sanitize_slug(mode)}_"
        f"{sanitize_slug(primary_language)}_"
        f"{sanitize_slug(dialogue_mode)}"
    )


def normalize_episode_plan_from_llm_result(
    llm_result: Dict[str, Any],
    book_charter: Dict[str, Any],
    program_config: Dict[str, Any],
    user_preferences: Dict[str, Any],
) -> Dict[str, Any]:
    book_id = ensure_string(book_charter.get("book_id"), "book_demo")
    mode = ensure_string(program_config.get("mode"))
    primary_language = ensure_string(program_config.get("primary_language"))
    dialogue_mode = ensure_string(user_preferences.get("dialogue_mode"))
    target_episode_minutes = ensure_int(program_config.get("target_episode_minutes"), 0)
    target_script_chars = ensure_int(program_config.get("target_script_chars"), 0)
    target_duration_sec = target_episode_minutes * 60 if target_episode_minutes > 0 else 0

    episodes_raw = llm_result.get("episodes")
    if not isinstance(episodes_raw, list):
        episodes_raw = []

    episodes: List[Dict[str, Any]] = []
    for index, raw_episode in enumerate(episodes_raw, start=1):
        if not isinstance(raw_episode, dict):
            raw_episode = {}
        episodes.append(
            {
                "episode_id": f"E{index:03d}",
                "episode_index": index,
                "title": ensure_string(raw_episode.get("title")),
                "mode": mode,
                "target_duration_sec": target_duration_sec,
                "target_script_chars": target_script_chars,
                "section_ids": ensure_string_list(raw_episode.get("section_ids")),
                "covers": ensure_string_list(raw_episode.get("covers")),
                "neighbor_context": ensure_string_list(raw_episode.get("neighbor_context")),
                "must_cover": ensure_string_list(raw_episode.get("must_cover")),
                "can_skip": ensure_string_list(raw_episode.get("can_skip")),
                "forbidden_to_introduce": ensure_string_list(raw_episode.get("forbidden_to_introduce")),
                "hook": ensure_string(raw_episode.get("hook")),
                "recap_focus": ensure_string(raw_episode.get("recap_focus")),
                "teaser_goal": ensure_string(raw_episode.get("teaser_goal")),
                "tone_target": ensure_string(raw_episode.get("tone_target")),
            }
        )

    total_episode_count = ensure_int(llm_result.get("total_episode_count"), 0)
    if total_episode_count <= 0:
        total_episode_count = len(episodes)

    return {
        "plan_id": build_episode_plan_id(book_id, mode, primary_language, dialogue_mode),
        "book_id": book_id,
        "mode": mode,
        "total_episode_count": total_episode_count,
        "episodes": episodes,
    }


def sort_by_char_start(card: Dict[str, Any], fallback_index_key: str = "") -> tuple[int, int, str]:
    locator = card.get("source_locator")
    if not isinstance(locator, dict):
        locator = {}
    fallback_index = ensure_int(card.get(fallback_index_key), 0) if fallback_index_key else 0
    return (
        ensure_int(locator.get("char_start"), 0),
        fallback_index,
        ensure_string(card.get("chunk_id") or card.get("section_id")),
    )


class Step5EpisodePlan:
    def __init__(self, output_dir: Path, llm_client: SimpleLLMClient, max_retries: int = 5) -> None:
        self.output_dir = output_dir
        self.llm_client = llm_client
        self.max_retries = max(1, int(max_retries))
        self.book_charter_path = self.output_dir / "book_charter.json"
        self.program_config_path = self.output_dir / "program_config.json"
        self.user_preferences_path = self.output_dir / "user_preferences.json"
        self.chunk_card_dir = self.output_dir / "chunk_cards"
        self.section_card_dir = self.output_dir / "section_cards"
        self.all_cards_summary_path = self.output_dir / "all_cards_summary.json"
        self.episode_plan_path = self.output_dir / "episode_plan.json"
        self.llm_raw_dir = self.output_dir / "llm_raw" / "episode_plan"

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
            "%s failed %s times in a row and fell back to an empty JSON object. Last error: %s",
            item_label,
            self.max_retries,
            last_exception,
        )
        return {}

    def load_chunk_cards(self) -> List[Dict[str, Any]]:
        cards = [read_json(path) for path in list_json_files(self.chunk_card_dir)]
        cards.sort(key=lambda item: sort_by_char_start(item, fallback_index_key="chunk_index"))
        return cards

    def load_section_cards(self) -> List[Dict[str, Any]]:
        cards = [read_json(path) for path in list_json_files(self.section_card_dir)]
        cards.sort(key=lambda item: sort_by_char_start(item))
        return cards

    def build_all_cards_summary(
        self,
        section_cards: List[Dict[str, Any]],
        chunk_cards: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        chunk_card_by_id: Dict[str, Dict[str, Any]] = {}
        chunk_cards_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for chunk_card in chunk_cards:
            chunk_id = ensure_string(chunk_card.get("chunk_id"))
            if not chunk_id:
                continue
            chunk_card_by_id[chunk_id] = chunk_card
            parent_section_id = ensure_string(chunk_card.get("parent_section_id"))
            if parent_section_id:
                chunk_cards_by_section[parent_section_id].append(chunk_card)

        summary_list: List[Dict[str, Any]] = []
        for section_card in section_cards:
            section_id = ensure_string(section_card.get("section_id"))
            if not section_id:
                continue

            source_chunk_ids = ensure_string_list(section_card.get("source_chunk_ids"))
            section_chunk_cards: List[Dict[str, Any]] = []
            if source_chunk_ids:
                for chunk_id in source_chunk_ids:
                    chunk_card = chunk_card_by_id.get(chunk_id)
                    if chunk_card is not None:
                        section_chunk_cards.append(chunk_card)
            else:
                section_chunk_cards = list(chunk_cards_by_section.get(section_id, []))

            section_chunk_cards.sort(key=lambda item: sort_by_char_start(item, fallback_index_key="chunk_index"))

            chunk_summaries: List[Dict[str, Any]] = []
            for chunk_card in section_chunk_cards:
                length_stats = chunk_card.get("length_stats")
                if not isinstance(length_stats, dict):
                    length_stats = {}
                chunk_summaries.append(
                    {
                        "chunk_id": ensure_string(chunk_card.get("chunk_id")),
                        "length_stats": {
                            "char_count": ensure_int(length_stats.get("char_count"), 0),
                            "paragraph_count": ensure_int(length_stats.get("paragraph_count"), 0),
                        },
                        "summary": ensure_string(chunk_card.get("summary")),
                        "key_points": ensure_string_list(chunk_card.get("key_points")),
                        "core_terms": ensure_core_terms(chunk_card.get("core_terms")),
                    }
                )

            summary_list.append(
                {
                    "section_id": section_id,
                    "section_type": ensure_string(section_card.get("section_type")),
                    "title": ensure_string(section_card.get("title")),
                    "summary": ensure_string(section_card.get("summary")),
                    "thesis_or_function": ensure_string(section_card.get("thesis_or_function")),
                    "key_points": ensure_string_list(section_card.get("key_points")),
                    "core_terms": ensure_core_terms(section_card.get("core_terms")),
                    "chunk": chunk_summaries,
                }
            )

        return summary_list

    def build_plan_id(
        self,
        book_id: str,
        mode: str,
        primary_language: str,
        dialogue_mode: str,
    ) -> str:
        return build_episode_plan_id(book_id, mode, primary_language, dialogue_mode)

    def normalize_episode_plan(
        self,
        llm_result: Dict[str, Any],
        book_charter: Dict[str, Any],
        program_config: Dict[str, Any],
        user_preferences: Dict[str, Any],
    ) -> Dict[str, Any]:
        return normalize_episode_plan_from_llm_result(
            llm_result=llm_result,
            book_charter=book_charter,
            program_config=program_config,
            user_preferences=user_preferences,
        )

    def run(self) -> Dict[str, Any]:
        if not self.book_charter_path.exists():
            raise FileNotFoundError(f"Missing book charter input: {self.book_charter_path}")
        if not self.program_config_path.exists():
            raise FileNotFoundError(f"Missing program config input: {self.program_config_path}")
        if not self.user_preferences_path.exists():
            raise FileNotFoundError(f"Missing user preferences input: {self.user_preferences_path}")
        if not self.chunk_card_dir.exists():
            raise FileNotFoundError(f"Missing chunk card directory: {self.chunk_card_dir}")
        if not self.section_card_dir.exists():
            raise FileNotFoundError(f"Missing section card directory: {self.section_card_dir}")

        chunk_cards = self.load_chunk_cards()
        section_cards = self.load_section_cards()
        if not chunk_cards:
            raise FileNotFoundError(f"No chunk cards found: {self.chunk_card_dir}")
        if not section_cards:
            raise FileNotFoundError(f"No section cards found: {self.section_card_dir}")

        self.llm_raw_dir.mkdir(parents=True, exist_ok=True)

        book_charter = read_json(self.book_charter_path)
        program_config = read_json(self.program_config_path)
        user_preferences = read_json(self.user_preferences_path)
        all_cards_summary = self.build_all_cards_summary(section_cards, chunk_cards)
        write_json(all_cards_summary, self.all_cards_summary_path)

        prompt = build_episode_plan_prompt(
            target_script_chars=ensure_int(program_config.get("target_script_chars"), 0),
            target_input_chars=ensure_int(program_config.get("target_input_chars"), 0),
            mode=ensure_string(program_config.get("mode")),
            primary_language=ensure_string(program_config.get("primary_language"), ensure_string(user_preferences.get("primary_language"), "zh")),
            positioning=ensure_string(program_config.get("positioning")),
            target_audience=ensure_string(program_config.get("target_audience")),
            core_argument_or_mainline_json=json.dumps(
                ensure_string_list(book_charter.get("core_argument_or_mainline")),
                ensure_ascii=False,
                indent=2,
            ),
            all_cards_summary_json=json.dumps(
                all_cards_summary,
                ensure_ascii=False,
                indent=2,
            ),
        )
        llm_result = self.safe_generate_json(
            prompt=prompt,
            raw_output_path=self.llm_raw_dir / "episode_plan.txt",
            item_label="episode_plan",
            required_schema=EPISODE_PLAN_REQUIRED_SCHEMA,
        )
        episode_plan = self.normalize_episode_plan(
            llm_result=llm_result,
            book_charter=book_charter,
            program_config=program_config,
            user_preferences=user_preferences,
        )
        write_json(episode_plan, self.episode_plan_path)

        return {
            "episode_plan": str(self.episode_plan_path),
            "all_cards_summary": str(self.all_cards_summary_path),
            "llm_raw_dir": str(self.llm_raw_dir),
            "chunk_card_count": len(chunk_cards),
            "section_card_count": len(section_cards),
            "max_retries": self.max_retries,
        }
