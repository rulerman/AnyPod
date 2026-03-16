from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..core.io import read_json, write_json

from ..core.llm_client import SimpleLLMClient
from ..core.prompts import build_episode_memory_prompt


logger = logging.getLogger("anypod.step8")


EPISODE_MEMORY_REQUIRED_SCHEMA = {
    "episode_card": {
        "title": None,
        "summary": None,
        "covered_sections": None,
        "covered_chunks": None,
        "introduced_terms": None,
        "callbacks": None,
        "resolved_loops": None,
        "open_loops": None,
        "tone_notes": None,
        "do_not_repeat_next_episode": None,
    },
    "updated_series_memory": {
        "cumulative_summary": None,
        "section_coverage_status": [
            {
                "section_id": None,
                "status": None,
                "episode_ids": None,
            }
        ],
        "introduced_terms_global": [
            {
                "term": None,
                "first_episode_id": None,
            }
        ],
        "callbacks_global": None,
        "open_loops_global": None,
        "resolved_loops_global": None,
        "repetition_watchlist": None,
    },
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


def episode_index_from_id(episode_id: str) -> int:
    match = re.search(r"(\d+)", episode_id)
    if not match:
        return 0
    return int(match.group(1))


def list_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def ensure_section_coverage_status(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        section_id = ensure_string(item.get("section_id"))
        status = ensure_string(item.get("status"))
        episode_ids = ensure_string_list(item.get("episode_ids"))
        if section_id:
            result.append(
                {
                    "section_id": section_id,
                    "status": status,
                    "episode_ids": episode_ids,
                }
            )
    return result


def ensure_terms_global(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        term = ensure_string(item.get("term"))
        first_episode_id = ensure_string(item.get("first_episode_id"))
        if term:
            result.append(
                {
                    "term": term,
                    "first_episode_id": first_episode_id,
                }
            )
    return result


class Step8EpisodeMemory:
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
        self.source_pack_dir = self.output_dir / "source_packs"
        self.script_dir = self.output_dir / "scripts"
        self.episode_card_dir = self.output_dir / "episode_cards"
        self.series_memory_summary_path = self.output_dir / "series_memory_summary.json"
        self.llm_raw_dir = self.output_dir / "llm_raw" / "episode_memory"

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

    def select_source_pack_files(self) -> List[Path]:
        source_pack_files = list_json_files(self.source_pack_dir)
        if not self.episode_ids:
            return source_pack_files
        wanted = {episode_id.strip() for episode_id in self.episode_ids if episode_id.strip()}
        return [path for path in source_pack_files if path.stem in wanted]

    def load_old_memory(self, program_config: Dict[str, Any]) -> Dict[str, Any]:
        if self.series_memory_summary_path.exists():
            data = read_json(self.series_memory_summary_path)
            if isinstance(data, dict):
                return data
        return {
            "series_id": ensure_string(program_config.get("series_id")),
            "completed_episode_count": 0,
            "completed_episode_ids": [],
            "cumulative_summary": "",
            "section_coverage_status": [],
            "introduced_terms_global": [],
            "callbacks_global": [],
            "open_loops_global": [],
            "resolved_loops_global": [],
            "repetition_watchlist": [],
        }

    def normalize_episode_card(
        self,
        llm_result: Dict[str, Any],
        source_pack: Dict[str, Any],
        script_text: str,
    ) -> Dict[str, Any]:
        episode_id = ensure_string(source_pack.get("episode_id"))
        episode_plan = source_pack.get("episode_plan")
        if not isinstance(episode_plan, dict):
            episode_plan = {}
        return {
            "episode_id": episode_id,
            "episode_index": ensure_int(episode_plan.get("episode_index"), episode_index_from_id(episode_id)),
            "title": ensure_string(llm_result.get("title")),
            "summary": ensure_string(llm_result.get("summary")),
            "covered_sections": ensure_string_list(llm_result.get("covered_sections")),
            "covered_chunks": ensure_string_list(llm_result.get("covered_chunks")),
            "introduced_terms": ensure_string_list(llm_result.get("introduced_terms")),
            "callbacks": ensure_string_list(llm_result.get("callbacks")),
            "resolved_loops": ensure_string_list(llm_result.get("resolved_loops")),
            "open_loops": ensure_string_list(llm_result.get("open_loops")),
            "tone_notes": ensure_string(llm_result.get("tone_notes")),
            "actual_script_chars": len(script_text),
            "do_not_repeat_next_episode": ensure_string_list(llm_result.get("do_not_repeat_next_episode")),
        }

    def normalize_series_memory(
        self,
        llm_result: Dict[str, Any],
        old_memory: Dict[str, Any],
        episode_card: Dict[str, Any],
        program_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        completed_episode_ids = ensure_string_list(old_memory.get("completed_episode_ids"))
        current_episode_id = ensure_string(episode_card.get("episode_id"))
        if current_episode_id and current_episode_id not in completed_episode_ids:
            completed_episode_ids.append(current_episode_id)

        return {
            "series_id": ensure_string(old_memory.get("series_id"), ensure_string(program_config.get("series_id"))),
            "completed_episode_count": len(completed_episode_ids),
            "completed_episode_ids": completed_episode_ids,
            "cumulative_summary": ensure_string(llm_result.get("cumulative_summary")),
            "section_coverage_status": ensure_section_coverage_status(llm_result.get("section_coverage_status")),
            "introduced_terms_global": ensure_terms_global(llm_result.get("introduced_terms_global")),
            "callbacks_global": ensure_string_list(llm_result.get("callbacks_global")),
            "open_loops_global": ensure_string_list(llm_result.get("open_loops_global")),
            "resolved_loops_global": ensure_string_list(llm_result.get("resolved_loops_global")),
            "repetition_watchlist": ensure_string_list(llm_result.get("repetition_watchlist")),
        }

    def run(self) -> Dict[str, Any]:
        if not self.program_config_path.exists():
            raise FileNotFoundError(f"Missing program config input: {self.program_config_path}")
        if not self.source_pack_dir.exists():
            raise FileNotFoundError(f"Missing source pack directory: {self.source_pack_dir}")
        if not self.script_dir.exists():
            raise FileNotFoundError(f"Missing script directory: {self.script_dir}")

        program_config = read_json(self.program_config_path)
        old_memory = self.load_old_memory(program_config)
        source_pack_files = self.select_source_pack_files()
        if not source_pack_files:
            raise FileNotFoundError(f"No usable source packs found: {self.source_pack_dir}")

        self.episode_card_dir.mkdir(parents=True, exist_ok=True)
        self.llm_raw_dir.mkdir(parents=True, exist_ok=True)

        generated_episode_ids: List[str] = []
        for source_pack_path in sorted(source_pack_files, key=lambda path: episode_index_from_id(path.stem)):
            episode_id = source_pack_path.stem
            script_path = self.script_dir / f"{episode_id}.txt"
            if not script_path.exists():
                raise FileNotFoundError(f"Missing script file: {script_path}")

            source_pack = read_json(source_pack_path)
            script_text = script_path.read_text(encoding="utf-8")

            prompt = build_episode_memory_prompt(
                episode_plan_json=json.dumps(source_pack.get("episode_plan", {}), ensure_ascii=False, indent=2),
                generated_script=script_text,
                old_memory_json=json.dumps(old_memory, ensure_ascii=False, indent=2),
            )
            llm_result = self.safe_generate_json(
                prompt=prompt,
                raw_output_path=self.llm_raw_dir / f"{episode_id}.txt",
                item_label=f"episode_memory:{episode_id}",
                required_schema=EPISODE_MEMORY_REQUIRED_SCHEMA,
            )
            episode_card_result = llm_result.get("episode_card")
            if not isinstance(episode_card_result, dict):
                episode_card_result = {}
            updated_series_memory_result = llm_result.get("updated_series_memory")
            if not isinstance(updated_series_memory_result, dict):
                updated_series_memory_result = {}

            episode_card = self.normalize_episode_card(
                llm_result=episode_card_result,
                source_pack=source_pack,
                script_text=script_text,
            )
            write_json(episode_card, self.episode_card_dir / f"{episode_id}.json")

            old_memory = self.normalize_series_memory(
                llm_result=updated_series_memory_result,
                old_memory=old_memory,
                episode_card=episode_card,
                program_config=program_config,
            )
            write_json(old_memory, self.series_memory_summary_path)
            generated_episode_ids.append(episode_id)

        return {
            "episode_card_dir": str(self.episode_card_dir),
            "series_memory_summary": str(self.series_memory_summary_path),
            "llm_raw_dir": str(self.llm_raw_dir),
            "generated_episode_ids": generated_episode_ids,
            "max_retries": self.max_retries,
        }
