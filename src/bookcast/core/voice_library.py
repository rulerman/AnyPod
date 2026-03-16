from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .common import sanitize_filename
from .io import write_jsonl


BOOKCAST_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
VOICE_LIBRARY_PATH = REPO_ROOT / "config" / "voice_library.jsonl"
VOICE_ASSET_DIR = REPO_ROOT / "voice_library_assets"
DEFAULT_VOICE_ID_S1 = "voice_a"
DEFAULT_VOICE_ID_S2 = "voice_b"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _strip_speaker_prefix(prompt_text: Any) -> str:
    text = _clean_text(prompt_text)
    if not text:
        return ""
    return re.sub(r"^\[S\d+\]\s*", "", text, count=1).strip()


def _apply_speaker_prefix(prompt_text: Any, speaker_index: int | None) -> str:
    text = _strip_speaker_prefix(prompt_text)
    if not text:
        return ""
    if speaker_index is None:
        return text
    return f"[S{speaker_index}] {text}"


def normalize_voice_entry(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None

    voice_id = _clean_text(value.get("voice_id"))
    if not voice_id:
        return None

    return {
        "voice_id": voice_id,
        "speaker_name": _clean_text(value.get("speaker_name")),
        "prompt_audio": _clean_text(value.get("prompt_audio")),
        "prompt_text": _strip_speaker_prefix(value.get("prompt_text")),
    }


def load_voice_library(path: str | Path = VOICE_LIBRARY_PATH) -> list[dict[str, str]]:
    library_path = Path(path)
    if not library_path.exists():
        return []

    deduped: dict[str, dict[str, str]] = {}
    for raw_line in library_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry = normalize_voice_entry(parsed)
        if entry is None:
            continue
        deduped[entry["voice_id"]] = entry

    return sorted(deduped.values(), key=lambda item: item["voice_id"])


def save_voice_library(entries: list[dict[str, str]], path: str | Path = VOICE_LIBRARY_PATH) -> list[dict[str, str]]:
    normalized_entries: list[dict[str, str]] = []
    deduped: dict[str, dict[str, str]] = {}
    for item in entries:
        entry = normalize_voice_entry(item)
        if entry is None:
            continue
        deduped[entry["voice_id"]] = entry

    normalized_entries = sorted(deduped.values(), key=lambda item: item["voice_id"])
    write_jsonl(normalized_entries, path)
    return normalized_entries


def find_voice_entry(voice_id: str | None, path: str | Path = VOICE_LIBRARY_PATH) -> dict[str, str] | None:
    normalized_voice_id = _clean_text(voice_id)
    if not normalized_voice_id:
        return None

    for entry in load_voice_library(path):
        if entry["voice_id"] == normalized_voice_id:
            return entry
    return None


def load_required_voice_entry(voice_id: str, path: str | Path = VOICE_LIBRARY_PATH) -> dict[str, str]:
    entry = find_voice_entry(voice_id, path=path)
    if entry is None:
        raise FileNotFoundError(f"Voice entry was not found in the voice library: {voice_id}")
    return entry


def get_default_voice_id(speaker_index: int) -> str:
    if speaker_index == 1:
        return DEFAULT_VOICE_ID_S1
    if speaker_index == 2:
        return DEFAULT_VOICE_ID_S2
    raise ValueError(f"Unsupported speaker index: {speaker_index}")


def resolve_prompt_audio_path(prompt_audio: str | None) -> Path | None:
    prompt_audio_text = _clean_text(prompt_audio)
    if not prompt_audio_text:
        return None

    candidate = Path(prompt_audio_text).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def serialize_prompt_audio_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def store_prompt_audio_file(voice_id: str, source_path: str | Path) -> str:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Prompt audio file does not exist: {source}")

    suffix = source.suffix or ".wav"
    target_name = f"{sanitize_filename(voice_id)}{suffix}"
    target_path = VOICE_ASSET_DIR / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)
    return serialize_prompt_audio_path(target_path)


def upsert_voice_entry(
    voice_id: str,
    speaker_name: str,
    prompt_text: str,
    prompt_audio_path: str | None = None,
    uploaded_prompt_audio_path: str | None = None,
    previous_voice_id: str | None = None,
    path: str | Path = VOICE_LIBRARY_PATH,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    normalized_voice_id = _clean_text(voice_id)
    normalized_speaker_name = _clean_text(speaker_name)
    normalized_prompt_text = _strip_speaker_prefix(prompt_text)
    normalized_previous_voice_id = _clean_text(previous_voice_id)

    if not normalized_voice_id:
        raise ValueError("voice_id is required")
    if not normalized_speaker_name:
        raise ValueError("speaker_name is required")
    if not normalized_prompt_text:
        raise ValueError("prompt_text is required")

    current_entries = load_voice_library(path)
    entry_by_id = {item["voice_id"]: item for item in current_entries}

    if (
        normalized_previous_voice_id
        and normalized_previous_voice_id != normalized_voice_id
        and normalized_voice_id in entry_by_id
    ):
        raise ValueError(f"voice_id already exists: {normalized_voice_id}")

    existing_entry = entry_by_id.get(normalized_previous_voice_id) or entry_by_id.get(normalized_voice_id)
    resolved_prompt_audio: str
    if _clean_text(uploaded_prompt_audio_path):
        resolved_prompt_audio = store_prompt_audio_file(normalized_voice_id, str(uploaded_prompt_audio_path))
    else:
        candidate_prompt_audio = _clean_text(prompt_audio_path)
        if not candidate_prompt_audio and existing_entry is not None:
            candidate_prompt_audio = existing_entry.get("prompt_audio", "")
        resolved_audio_path = resolve_prompt_audio_path(candidate_prompt_audio)
        if resolved_audio_path is None or not resolved_audio_path.exists():
            raise FileNotFoundError("prompt_audio is required and must point to an existing file")
        resolved_prompt_audio = serialize_prompt_audio_path(resolved_audio_path)

    saved_entry = {
        "voice_id": normalized_voice_id,
        "speaker_name": normalized_speaker_name,
        "prompt_audio": resolved_prompt_audio,
        "prompt_text": normalized_prompt_text,
    }

    if normalized_previous_voice_id and normalized_previous_voice_id in entry_by_id:
        del entry_by_id[normalized_previous_voice_id]
    entry_by_id[normalized_voice_id] = saved_entry
    updated_entries = save_voice_library(list(entry_by_id.values()), path=path)
    return updated_entries, saved_entry


def resolve_voice_prompt_overrides(
    voice_id: str | None,
    speaker_index: int | None = None,
    path: str | Path = VOICE_LIBRARY_PATH,
) -> tuple[str | None, str | None]:
    entry = find_voice_entry(voice_id, path=path)
    if entry is None:
        return None, None

    resolved_audio_path = resolve_prompt_audio_path(entry.get("prompt_audio"))
    if resolved_audio_path is not None and not resolved_audio_path.exists():
        raise FileNotFoundError(f"Prompt audio file does not exist: {resolved_audio_path}")
    prompt_audio = str(resolved_audio_path) if resolved_audio_path is not None else None
    prompt_text = _apply_speaker_prefix(entry.get("prompt_text"), speaker_index) or None
    return prompt_audio, prompt_text


def resolve_default_voice_prompt(
    speaker_index: int,
    path: str | Path = VOICE_LIBRARY_PATH,
) -> tuple[str, str]:
    voice_id = get_default_voice_id(speaker_index)
    prompt_audio, prompt_text = resolve_voice_prompt_overrides(voice_id, speaker_index=speaker_index, path=path)
    if not prompt_audio:
        raise FileNotFoundError(f"Default prompt audio is missing for voice_id: {voice_id}")
    if not prompt_text:
        raise ValueError(f"Default prompt text is missing for voice_id: {voice_id}")
    return prompt_audio, prompt_text
