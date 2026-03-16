from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..core.io import read_json, read_jsonl, write_json

from ..core.common import save_text
from ..core.llm_client import SimpleLLMClient
from ..core.prompts import build_episode_script_prompt
from ..tts_text import normalize_text


logger = logging.getLogger("anypod.step6")


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


def list_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def episode_index_from_id(episode_id: str) -> int:
    match = re.search(r"(\d+)", episode_id)
    if not match:
        return 0
    return int(match.group(1))


class Step6SourcePackScript:
    def __init__(
        self,
        output_dir: Path,
        llm_client: SimpleLLMClient,
        episode_ids: List[str] | None = None,
        max_retries: int = 5,
    ) -> None:
        self.output_dir = output_dir
        self.llm_client = llm_client
        self.episode_ids = episode_ids or []
        self.max_retries = max(1, int(max_retries))
        self.program_config_path = self.output_dir / "program_config.json"
        self.speaker_bible_path = self.output_dir / "speaker_bible.json"
        self.episode_plan_path = self.output_dir / "episode_plan.json"
        self.chunk_card_dir = self.output_dir / "chunk_cards"
        self.section_card_dir = self.output_dir / "section_cards"
        self.raw_chunk_dir = self.output_dir / "raw_text" / "chunks"
        self.chunk_source_map_path = self.output_dir / "chunk_source_map.jsonl"
        self.full_text_path = self.output_dir / "full_text.txt"
        self.series_memory_summary_path = self.output_dir / "series_memory_summary.json"
        self.episode_card_dir = self.output_dir / "episode_cards"
        self.source_pack_dir = self.output_dir / "source_packs"
        self.script_dir = self.output_dir / "scripts"
        self.llm_raw_dir = self.output_dir / "llm_raw" / "scripts"

    def build_attempt_output_path(self, raw_output_path: Path, attempt_index: int) -> Path:
        if attempt_index == 1:
            return raw_output_path
        return raw_output_path.with_name(
            f"{raw_output_path.stem}_attempt_{attempt_index:02d}{raw_output_path.suffix}"
        )

    def safe_generate_text(self, prompt: str, raw_output_path: Path, item_label: str) -> str:
        last_exception: Exception | None = None
        for attempt_index in range(1, self.max_retries + 1):
            attempt_output_path = self.build_attempt_output_path(raw_output_path, attempt_index)
            try:
                return self.llm_client.generate_text(
                    prompt=prompt,
                    raw_output_path=attempt_output_path,
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
            "%s failed %s times in a row and fell back to an empty string. Last error: %s",
            item_label,
            self.max_retries,
            last_exception,
        )
        return ""

    def load_json_directory(self, directory: Path) -> List[Dict[str, Any]]:
        return [read_json(path) for path in list_json_files(directory)]

    def build_chunk_map(self) -> Dict[str, Dict[str, Any]]:
        chunk_cards = self.load_json_directory(self.chunk_card_dir)
        result: Dict[str, Dict[str, Any]] = {}
        for card in chunk_cards:
            chunk_id = ensure_string(card.get("chunk_id"))
            if chunk_id:
                result[chunk_id] = card
        return result

    def build_section_map(self) -> Dict[str, Dict[str, Any]]:
        section_cards = self.load_json_directory(self.section_card_dir)
        result: Dict[str, Dict[str, Any]] = {}
        for card in section_cards:
            section_id = ensure_string(card.get("section_id"))
            if section_id:
                result[section_id] = card
        return result

    def build_chunk_source_map(self) -> Dict[str, Dict[str, Any]]:
        if not self.chunk_source_map_path.exists():
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for item in read_jsonl(self.chunk_source_map_path):
            if not isinstance(item, dict):
                continue
            chunk_id = ensure_string(item.get("chunk_id"))
            if chunk_id:
                result[chunk_id] = item
        return result

    def load_series_memory_summary(self) -> Dict[str, Any]:
        if not self.series_memory_summary_path.exists():
            return {}
        data = read_json(self.series_memory_summary_path)
        return data if isinstance(data, dict) else {}

    def load_recent_episode_cards(self, current_episode_index: int) -> List[Dict[str, Any]]:
        if not self.episode_card_dir.exists():
            return []
        cards = self.load_json_directory(self.episode_card_dir)
        cards.sort(
            key=lambda item: (
                ensure_int(item.get("episode_index"), episode_index_from_id(ensure_string(item.get("episode_id")))),
                ensure_string(item.get("episode_id")),
            )
        )
        filtered = [
            card
            for card in cards
            if ensure_int(card.get("episode_index"), episode_index_from_id(ensure_string(card.get("episode_id"))))
            < current_episode_index
        ]
        return filtered[-2:]

    def load_last_episode_tail_excerpt(self, current_episode_index: int) -> str:
        if current_episode_index <= 1:
            return ""
        previous_episode_id = f"E{current_episode_index - 1:03d}"
        previous_script_path = self.script_dir / f"{previous_episode_id}.txt"
        if not previous_script_path.exists():
            return ""
        text = previous_script_path.read_text(encoding="utf-8")
        return text[-200:].strip()

    def load_chunk_text(
        self,
        chunk_id: str,
        chunk_source_map_by_id: Dict[str, Dict[str, Any]],
        full_text: str,
    ) -> str:
        chunk_file = self.raw_chunk_dir / f"{chunk_id}.txt"
        if chunk_file.exists():
            return chunk_file.read_text(encoding="utf-8")

        chunk_source = chunk_source_map_by_id.get(chunk_id)
        if not chunk_source:
            return ""

        char_start = ensure_int(chunk_source.get("char_start"), 0)
        char_end = ensure_int(chunk_source.get("char_end"), 0)
        if char_end <= char_start:
            return ""
        return full_text[char_start:char_end]

    def select_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.episode_ids:
            return episodes
        wanted = {episode_id.strip() for episode_id in self.episode_ids if episode_id.strip()}
        return [episode for episode in episodes if ensure_string(episode.get("episode_id")) in wanted]

    def build_source_pack(
        self,
        episode: Dict[str, Any],
        program_config: Dict[str, Any],
        speaker_bible: Dict[str, Any],
        section_card_by_id: Dict[str, Dict[str, Any]],
        chunk_card_by_id: Dict[str, Dict[str, Any]],
        chunk_source_map_by_id: Dict[str, Dict[str, Any]],
        full_text: str,
        series_memory_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        episode_id = ensure_string(episode.get("episode_id"))
        episode_index = ensure_int(episode.get("episode_index"), episode_index_from_id(episode_id))

        section_ids = ensure_string_list(episode.get("section_ids"))
        covers = ensure_string_list(episode.get("covers"))

        if not section_ids:
            inferred_section_ids: List[str] = []
            for chunk_id in covers:
                chunk_card = chunk_card_by_id.get(chunk_id)
                if not chunk_card:
                    continue
                parent_section_id = ensure_string(chunk_card.get("parent_section_id"))
                if parent_section_id and parent_section_id not in inferred_section_ids:
                    inferred_section_ids.append(parent_section_id)
            section_ids = inferred_section_ids

        section_cards = [
            section_card_by_id[section_id]
            for section_id in section_ids
            if section_id in section_card_by_id
        ]
        chunk_cards = [
            chunk_card_by_id[chunk_id]
            for chunk_id in covers
            if chunk_id in chunk_card_by_id
        ]
        raw_text_chunks = [
            {
                "chunk_id": chunk_id,
                "text": self.load_chunk_text(
                    chunk_id=chunk_id,
                    chunk_source_map_by_id=chunk_source_map_by_id,
                    full_text=full_text,
                ),
            }
            for chunk_id in covers
        ]

        return {
            "episode_id": episode_id,
            "program_config": program_config,
            "speaker_bible": speaker_bible,
            "episode_plan": episode,
            "section_cards": section_cards,
            "chunk_cards": chunk_cards,
            "raw_text_chunks": raw_text_chunks,
            "recent_episode_cards": self.load_recent_episode_cards(episode_index),
            "last_episode_tail_excerpt": self.load_last_episode_tail_excerpt(episode_index),
            "series_memory_summary": series_memory_summary,
        }

    def run(self) -> Dict[str, Any]:
        if not self.program_config_path.exists():
            raise FileNotFoundError(f"Missing program config input: {self.program_config_path}")
        if not self.speaker_bible_path.exists():
            raise FileNotFoundError(f"Missing speaker bible input: {self.speaker_bible_path}")
        if not self.episode_plan_path.exists():
            raise FileNotFoundError(f"Missing episode plan input: {self.episode_plan_path}")
        if not self.chunk_card_dir.exists():
            raise FileNotFoundError(f"Missing chunk card directory: {self.chunk_card_dir}")
        if not self.section_card_dir.exists():
            raise FileNotFoundError(f"Missing section card directory: {self.section_card_dir}")

        program_config = read_json(self.program_config_path)
        speaker_bible = read_json(self.speaker_bible_path)
        episode_plan = read_json(self.episode_plan_path)
        full_text = self.full_text_path.read_text(encoding="utf-8") if self.full_text_path.exists() else ""

        episodes = episode_plan.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise FileNotFoundError(f"No usable episode plan found: {self.episode_plan_path}")

        selected_episodes = self.select_episodes(episodes)
        if not selected_episodes:
            raise FileNotFoundError("No requested episode_id matched the episode plan")

        self.source_pack_dir.mkdir(parents=True, exist_ok=True)
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.llm_raw_dir.mkdir(parents=True, exist_ok=True)

        section_card_by_id = self.build_section_map()
        chunk_card_by_id = self.build_chunk_map()
        chunk_source_map_by_id = self.build_chunk_source_map()
        series_memory_summary = self.load_series_memory_summary()

        generated_episode_ids: List[str] = []
        for episode in sorted(
            selected_episodes,
            key=lambda item: ensure_int(item.get("episode_index"), episode_index_from_id(ensure_string(item.get("episode_id")))),
        ):
            episode_id = ensure_string(episode.get("episode_id"))
            if not episode_id:
                continue

            source_pack = self.build_source_pack(
                episode=episode,
                program_config=program_config,
                speaker_bible=speaker_bible,
                section_card_by_id=section_card_by_id,
                chunk_card_by_id=chunk_card_by_id,
                chunk_source_map_by_id=chunk_source_map_by_id,
                full_text=full_text,
                series_memory_summary=series_memory_summary,
            )
            write_json(source_pack, self.source_pack_dir / f"{episode_id}.json")

            prompt = build_episode_script_prompt(
                speaker_bible_json=json.dumps(source_pack["speaker_bible"], ensure_ascii=False, indent=2),
                program_config_json=json.dumps(source_pack["program_config"], ensure_ascii=False, indent=2),
                episode_plan_json=json.dumps(source_pack["episode_plan"], ensure_ascii=False, indent=2),
                recent_episode_cards_json=json.dumps(source_pack["recent_episode_cards"], ensure_ascii=False, indent=2),
                series_memory_summary_json=json.dumps(source_pack["series_memory_summary"], ensure_ascii=False, indent=2),
                last_episode_tail_excerpt=ensure_string(source_pack["last_episode_tail_excerpt"]),
                raw_text_chunks_json=json.dumps(source_pack["raw_text_chunks"], ensure_ascii=False, indent=2),
                mode=ensure_string(source_pack["episode_plan"].get("mode")),
                target_script_chars=ensure_int(source_pack["episode_plan"].get("target_script_chars"), 0),
                primary_language=ensure_string(
                    source_pack["program_config"].get("primary_language"),
                    ensure_string(source_pack["program_config"].get("language_output_rules", {}).get("script_language"), "zh"),
                ),
            )
            script_text = self.safe_generate_text(
                prompt=prompt,
                raw_output_path=self.llm_raw_dir / f"{episode_id}.txt",
                item_label=f"script:{episode_id}",
            )
            normalized_script_text = normalize_text(script_text)
            save_text(self.script_dir / f"{episode_id}.txt", normalized_script_text)
            generated_episode_ids.append(episode_id)

        return {
            "source_pack_dir": str(self.source_pack_dir),
            "script_dir": str(self.script_dir),
            "llm_raw_dir": str(self.llm_raw_dir),
            "generated_episode_ids": generated_episode_ids,
            "max_retries": self.max_retries,
        }
