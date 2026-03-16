from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..core.io import read_json, write_json
from ..core.llm_client import SimpleLLMClient
from ..core.prompts import build_program_config_prompt, build_speaker_bible_prompt


logger = logging.getLogger("anypod.step4")


PROGRAM_CONFIG_REQUIRED_SCHEMA = {
    "show_title": None,
    "positioning": None,
    "target_audience": None,
    "language_output_rules": {
        "script_language": None,
        "term_policy": None,
    },
    "pace_style": None,
    "target_script_chars": None,
    "target_input_chars": None,
    "tone_guardrails": None,
    "content_guardrails": None,
}

SPEAKER_BIBLE_REQUIRED_SCHEMA = {
    "fixed_opening": None,
    "speakers": [
        {
            "display_name": None,
            "role": None,
            "persona_summary": None,
            "tone": None,
            "vocabulary_preferences": None,
            "sentence_length_tendency": None,
            "transition_patterns": None,
            "banned_catchphrases": None,
            "allowed_show_phrases": None,
            "target_share_percent": None,
        }
    ],
    "interaction_rules": None,
    "consistency_rules": None,
    "forbidden_behaviors": None,
}


def ensure_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def ensure_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
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


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text.strip().lower()).strip("_")
    return slug or "untitled"


def map_episode_minutes(level: str) -> int:
    level_map = {
        "low": 10,
        "mid": 20,
        "high": 30,
    }
    return level_map.get(level, 20)


class Step4GlobalPlan:
    def __init__(
        self,
        output_dir: Path,
        llm_client: SimpleLLMClient,
        user_preferences: Dict[str, Any],
        max_retries: int = 5,
    ) -> None:
        self.output_dir = output_dir
        self.llm_client = llm_client
        self.user_preferences = user_preferences
        self.max_retries = max(1, int(max_retries))
        self.book_charter_path = self.output_dir / "book_charter.json"
        self.user_preferences_path = self.output_dir / "user_preferences.json"
        self.program_config_path = self.output_dir / "program_config.json"
        self.speaker_bible_path = self.output_dir / "speaker_bible.json"
        self.llm_raw_dir = self.output_dir / "llm_raw" / "global_plan"

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

    def build_series_id(self, book_id: str, mode: str, primary_language: str, dialogue_mode: str) -> str:
        return (
            f"series_{sanitize_slug(book_id)}_"
            f"{sanitize_slug(mode)}_"
            f"{sanitize_slug(primary_language)}_"
            f"{sanitize_slug(dialogue_mode)}"
        )

    def build_default_program_config(self, book_charter: Dict[str, Any]) -> Dict[str, Any]:
        primary_language = ensure_string(self.user_preferences.get("primary_language"), "zh")
        mode = ensure_string(self.user_preferences.get("mode"), "deep_dive")
        dialogue_mode = ensure_string(self.user_preferences.get("dialogue_mode"), "dual")
        allow_external_knowledge = ensure_bool(
            self.user_preferences.get("allow_external_knowledge"),
            True,
        )
        target_episode_minutes = ensure_int(
            self.user_preferences.get("target_episode_minutes"),
            20,
        )
        book_id = ensure_string(book_charter.get("book_id"), "book_demo")

        return {
            "series_id": self.build_series_id(book_id, mode, primary_language, dialogue_mode),
            "primary_language": primary_language,
            "mode": mode,
            "allow_external_knowledge": allow_external_knowledge,
            "target_episode_minutes": target_episode_minutes,
            "show_title": "",
            "positioning": "",
            "target_audience": "",
            "language_output_rules": {
                "script_language": "",
                "term_policy": "",
            },
            "pace_style": "",
            "target_script_chars": 0,
            "target_input_chars": 0,
            "tone_guardrails": [],
            "content_guardrails": [],
        }

    def build_program_customization(self) -> Dict[str, Any]:
        customization: Dict[str, Any] = {}

        show_title = ensure_string(self.user_preferences.get("show_title"))
        if show_title:
            customization["show_title_preference"] = show_title

        positioning = ensure_string(self.user_preferences.get("positioning"))
        if positioning:
            customization["positioning_preference"] = positioning

        target_audience = ensure_string(self.user_preferences.get("target_audience"))
        if target_audience:
            customization["target_audience_preference"] = target_audience

        return customization

    def normalize_program_config(self, llm_result: Dict[str, Any], book_charter: Dict[str, Any]) -> Dict[str, Any]:
        program_config = self.build_default_program_config(book_charter)
        language_output_rules = llm_result.get("language_output_rules")
        if isinstance(language_output_rules, dict):
            program_config["language_output_rules"] = {
                "script_language": ensure_string(
                    language_output_rules.get("script_language"),
                    program_config["language_output_rules"]["script_language"],
                ),
                "term_policy": ensure_string(
                    language_output_rules.get("term_policy"),
                    program_config["language_output_rules"]["term_policy"],
                ),
            }

        program_config.update(
            {
                "show_title": ensure_string(llm_result.get("show_title"), program_config["show_title"]),
                "positioning": ensure_string(llm_result.get("positioning"), program_config["positioning"]),
                "target_audience": ensure_string(
                    llm_result.get("target_audience"),
                    program_config["target_audience"],
                ),
                "pace_style": ensure_string(llm_result.get("pace_style"), program_config["pace_style"]),
                "target_script_chars": ensure_int(
                    llm_result.get("target_script_chars"),
                    program_config["target_script_chars"],
                ),
                "target_input_chars": ensure_int(
                    llm_result.get("target_input_chars"),
                    program_config["target_input_chars"],
                ),
                "tone_guardrails": ensure_string_list(llm_result.get("tone_guardrails"))
                or program_config["tone_guardrails"],
                "content_guardrails": ensure_string_list(llm_result.get("content_guardrails"))
                or program_config["content_guardrails"],
            }
        )
        return program_config

    def build_default_speaker(self, speaker_id: str, user_speaker: Dict[str, Any], share: int, role: str) -> Dict[str, Any]:
        return {
            "speaker_id": speaker_id,
            "tts_voice_id": ensure_string(user_speaker.get("tts_voice_id"), f"voice_{speaker_id.lower()}"),
            "language": ensure_string(user_speaker.get("language"), "zh"),
            "display_name": "",
            "role": "",
            "persona_summary": "",
            "tone": "",
            "vocabulary_preferences": [],
            "sentence_length_tendency": "",
            "transition_patterns": [],
            "banned_catchphrases": [],
            "allowed_show_phrases": [],
            "target_share_percent": 0,
        }

    def build_speaker_customization(self) -> Dict[str, Any]:
        customization: Dict[str, Any] = {}

        fixed_opening = ensure_string(self.user_preferences.get("fixed_opening"))
        if fixed_opening:
            customization["fixed_opening_preference"] = fixed_opening

        requested_speakers = self.user_preferences.get("speakers")
        if not isinstance(requested_speakers, list):
            requested_speakers = []

        speaker_preferences: List[Dict[str, Any]] = []
        for index, speaker in enumerate(requested_speakers, start=1):
            if not isinstance(speaker, dict):
                continue

            speaker_preference: Dict[str, Any] = {
                "speaker_id": ensure_string(speaker.get("speaker_id"), f"S{index}"),
            }
            display_name = ensure_string(speaker.get("display_name"))
            if display_name:
                speaker_preference["display_name_preference"] = display_name

            persona_style = ensure_string(speaker.get("persona_style"))
            if persona_style:
                speaker_preference["persona_style_preference"] = persona_style

            if len(speaker_preference) > 1:
                speaker_preferences.append(speaker_preference)

        if speaker_preferences:
            customization["speakers"] = speaker_preferences

        return customization

    def normalize_speaker_bible(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        dialogue_mode = ensure_string(self.user_preferences.get("dialogue_mode"), "dual")
        requested_speakers = self.user_preferences.get("speakers")
        if not isinstance(requested_speakers, list):
            requested_speakers = []
        requested_count = 1 if dialogue_mode == "single" else 2
        llm_speakers = llm_result.get("speakers")
        if not isinstance(llm_speakers, list):
            llm_speakers = []

        speakers: List[Dict[str, Any]] = []
        for index in range(requested_count):
            speaker_id = f"S{index + 1}"
            user_speaker = requested_speakers[index] if index < len(requested_speakers) else {}
            llm_speaker = llm_speakers[index] if index < len(llm_speakers) else {}
            speaker = self.build_default_speaker(
                speaker_id=speaker_id,
                user_speaker=user_speaker,
                share=0,
                role="",
            )
            speaker.update(
                {
                    "display_name": ensure_string(llm_speaker.get("display_name"), speaker["display_name"]),
                    "role": ensure_string(llm_speaker.get("role"), speaker["role"]),
                    "persona_summary": ensure_string(
                        llm_speaker.get("persona_summary"),
                        speaker["persona_summary"],
                    ),
                    "tone": ensure_string(llm_speaker.get("tone"), speaker["tone"]),
                    "vocabulary_preferences": ensure_string_list(llm_speaker.get("vocabulary_preferences")),
                    "sentence_length_tendency": ensure_string(
                        llm_speaker.get("sentence_length_tendency"),
                        speaker["sentence_length_tendency"],
                    ),
                    "transition_patterns": ensure_string_list(llm_speaker.get("transition_patterns")),
                    "banned_catchphrases": ensure_string_list(llm_speaker.get("banned_catchphrases")),
                    "allowed_show_phrases": ensure_string_list(llm_speaker.get("allowed_show_phrases")),
                    "target_share_percent": ensure_int(
                        llm_speaker.get("target_share_percent"),
                        speaker["target_share_percent"],
                    ),
                }
            )
            speakers.append(speaker)

        interaction_rules = ensure_string_list(llm_result.get("interaction_rules"))
        if dialogue_mode == "single":
            interaction_rules = []
        consistency_rules = ensure_string_list(llm_result.get("consistency_rules"))
        forbidden_behaviors = ensure_string_list(llm_result.get("forbidden_behaviors"))

        return {
            "dialogue_mode": dialogue_mode,
            "fixed_opening": ensure_string(llm_result.get("fixed_opening")),
            "speakers": speakers,
            "interaction_rules": interaction_rules,
            "consistency_rules": consistency_rules,
            "forbidden_behaviors": forbidden_behaviors,
        }

    def run(self) -> Dict[str, Any]:
        if not self.book_charter_path.exists():
            raise FileNotFoundError(f"Missing book charter input: {self.book_charter_path}")

        self.llm_raw_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.user_preferences, self.user_preferences_path)

        book_charter = read_json(self.book_charter_path)
        book_title = ensure_string(book_charter.get("book_title"), "未命名图书")
        primary_language = ensure_string(self.user_preferences.get("primary_language"), "zh")
        mode = ensure_string(self.user_preferences.get("mode"), "deep_dive")
        dialogue_mode = ensure_string(self.user_preferences.get("dialogue_mode"), "dual")
        target_episode_minutes = ensure_int(self.user_preferences.get("target_episode_minutes"), 20)
        allow_external_knowledge = ensure_bool(self.user_preferences.get("allow_external_knowledge"), True)
        program_customization = self.build_program_customization()
        speaker_customization = self.build_speaker_customization()

        program_prompt = build_program_config_prompt(
            book_title=book_title,
            primary_language=primary_language,
            mode=mode,
            target_episode_minutes=target_episode_minutes,
            allow_external_knowledge=allow_external_knowledge,
            dialogue_mode=dialogue_mode,
            book_charter_json=json.dumps(book_charter, ensure_ascii=False, indent=2),
            user_customization_json=json.dumps(program_customization, ensure_ascii=False, indent=2),
        )
        program_result = self.safe_generate_json(
            prompt=program_prompt,
            raw_output_path=self.llm_raw_dir / "program_config.txt",
            item_label="program_config",
            required_schema=PROGRAM_CONFIG_REQUIRED_SCHEMA,
        )
        program_config = self.normalize_program_config(program_result, book_charter)
        write_json(program_config, self.program_config_path)

        speaker_prompt = build_speaker_bible_prompt(
            dialogue_mode=dialogue_mode,
            primary_language=primary_language,
            show_title=program_config["show_title"],
            positioning=program_config["positioning"],
            target_audience=program_config["target_audience"],
            user_customization_json=json.dumps(speaker_customization, ensure_ascii=False, indent=2),
        )
        speaker_result = self.safe_generate_json(
            prompt=speaker_prompt,
            raw_output_path=self.llm_raw_dir / "speaker_bible.txt",
            item_label="speaker_bible",
            required_schema=SPEAKER_BIBLE_REQUIRED_SCHEMA,
        )
        speaker_bible = self.normalize_speaker_bible(speaker_result)
        write_json(speaker_bible, self.speaker_bible_path)

        return {
            "user_preferences": str(self.user_preferences_path),
            "program_config": str(self.program_config_path),
            "speaker_bible": str(self.speaker_bible_path),
            "llm_raw_dir": str(self.llm_raw_dir),
            "target_episode_minutes": target_episode_minutes,
            "max_retries": self.max_retries,
        }
