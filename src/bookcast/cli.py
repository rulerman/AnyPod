from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookcast.core.io import read_json, read_jsonl
from bookcast.core.llm_client import (
    PLAN_AGENT_NAME,
    SimpleLLMClient,
    UNDERSTANDING_AGENT_NAME,
    WRITING_AGENT_NAME,
)
from bookcast.core.runtime_log import console_print
from bookcast.core.voice_library import resolve_voice_prompt_overrides
from bookcast.steps import (
    AsyncTTSWorker,
    Step1Parse,
    Step2Understand,
    Step3BookCharter,
    Step4GlobalPlan,
    Step5EpisodePlan,
    Step6SourcePackScript,
    Step7TTS,
    Step8EpisodeMemory,
)
from bookcast.steps.step4_global_plan import map_episode_minutes


TOTAL_STEPS = 8
CONFIG_JSON_DEST = "config_json"
print = console_print


# ── Argument helpers ──


def parse_episode_ids(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped or stripped.lower() == "all":
        return []
    return [item.strip() for item in stripped.split(",") if item.strip()]


def parse_bool_arg(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean arguments only accept true or false")


def episode_index_from_id(episode_id: str) -> int:
    match = re.search(r"(\d+)", episode_id)
    if not match:
        return 0
    return int(match.group(1))


def _parser_action_map(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if action.dest and action.dest != "help"
    }


def _parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        defaults[action.dest] = action.default
    return defaults


def _parse_bool_config_value(value: Any, dest: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        return parse_bool_arg(value)
        raise ValueError(f"Config field {dest} must be a boolean or a true/false string")


def _parse_episode_ids_config_value(value: Any) -> list[str]:
    if isinstance(value, list):
        normalized = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    if isinstance(value, str):
        return parse_episode_ids(value)
        raise ValueError("Config field episode_ids must be a string or a list of strings")


def _normalize_config_value(action: argparse.Action, value: Any) -> Any:
    if isinstance(action, argparse._StoreTrueAction):
        return _parse_bool_config_value(value, action.dest)

    if value is None and action.default is None:
        return None
    if value == "" and action.default is None:
        return None

    if action.dest == "episode_ids":
        normalized = _parse_episode_ids_config_value(value)
    elif action.type is Path:
        normalized = Path(str(value))
    elif action.type is parse_bool_arg:
        normalized = _parse_bool_config_value(value, action.dest)
    elif action.type is not None:
        try:
            normalized = action.type(value)
        except Exception as exc:
            raise ValueError(f"Invalid value for config field {action.dest}: {value}") from exc
    else:
        normalized = value

    if action.choices and normalized not in action.choices:
        raise ValueError(f"Config field {action.dest} must be one of: {', '.join(map(str, action.choices))}")
    return normalized


def _load_config_json(config_path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    raw_config = read_json(config_path)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must be a JSON object: {config_path}")

    actions = _parser_action_map(parser)
    normalized: dict[str, Any] = {}
    for key, value in raw_config.items():
        if key == CONFIG_JSON_DEST:
            continue
        action = actions.get(key)
        if action is None:
            raise ValueError(f"Config file contains an unknown field: {key}")
        normalized[key] = _normalize_config_value(action, value)
    return normalized


def _collect_explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit_dests: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest and dest != "help":
            explicit_dests.add(dest)
    return explicit_dests


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    cli_args = parser.parse_args(argv)

    merged = _parser_defaults(parser)
    config_path = getattr(cli_args, CONFIG_JSON_DEST, None)
    if config_path is not None:
        merged.update(_load_config_json(config_path.resolve(), parser))

    # 显式命令行参数优先于 JSON 配置；未显式给出的参数才回退到 JSON 或默认值。
    for dest in _collect_explicit_cli_dests(parser, argv):
        merged[dest] = getattr(cli_args, dest)

    args = argparse.Namespace(**merged)
    if args.input_path is None:
        parser.error("Missing input_path; provide it via the command line or --config_json")
    if args.output_dir is None:
        parser.error("Missing output_dir; provide it via the command line or --config_json")
    return args


def resolve_episode_ids(output_dir: Path, requested_episode_ids: list[str]) -> list[str]:
    episode_plan_path = output_dir / "episode_plan.json"
    if not episode_plan_path.exists():
        raise FileNotFoundError(f"Missing episode plan input: {episode_plan_path}")

    episode_plan = read_json(episode_plan_path)
    episodes = episode_plan.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise FileNotFoundError(f"No usable episode plan was found: {episode_plan_path}")

    planned_ids = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        episode_id = str(episode.get("episode_id", "")).strip()
        if episode_id:
            planned_ids.append(episode_id)

    if not requested_episode_ids:
        return sorted(planned_ids, key=episode_index_from_id)

    wanted = {episode_id.strip() for episode_id in requested_episode_ids if episode_id.strip()}
    return [episode_id for episode_id in sorted(planned_ids, key=episode_index_from_id) if episode_id in wanted]


def build_user_preferences(args: argparse.Namespace) -> dict:
    speakers = [
        {
            "speaker_id": "S1",
            "display_name": args.speaker_1_name,
            "persona_style": args.speaker_1_style,
            "tts_voice_id": args.speaker_1_voice_id,
            "language": args.speaker_1_language or args.primary_language,
        }
    ]
    if args.dialogue_mode == "dual":
        speakers.append(
            {
                "speaker_id": "S2",
                "display_name": args.speaker_2_name,
                "persona_style": args.speaker_2_style,
                "tts_voice_id": args.speaker_2_voice_id,
                "language": args.speaker_2_language or args.primary_language,
            }
        )

    return {
        "dialogue_mode": args.dialogue_mode,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "primary_language": args.primary_language,
        "mode": args.mode,
        "target_episode_level": args.target_episode_level,
        "target_episode_minutes": map_episode_minutes(args.target_episode_level),
        "allow_external_knowledge": args.allow_external_knowledge,
        "show_title": args.show_title,
        "positioning": args.positioning,
        "target_audience": args.target_audience,
        "fixed_opening": args.fixed_opening,
    }


def build_tts_options(args: argparse.Namespace) -> dict[str, Any]:
    options = {
        "tts_model_path": args.tts_model_path,
        "tts_audio_tokenizer_path": args.tts_audio_tokenizer_path,
        "tts_prompt_audio_s1": args.tts_prompt_audio_s1,
        "tts_prompt_audio_s2": args.tts_prompt_audio_s2,
        "tts_prompt_text_s1": args.tts_prompt_text_s1,
        "tts_prompt_text_s2": args.tts_prompt_text_s2,
        "tts_device": args.tts_device,
        "tts_torch_dtype": args.tts_torch_dtype,
        "tts_attn_implementation": args.tts_attn_implementation,
        "tts_cfg_scale": args.tts_cfg_scale,
        "tts_ddpm_steps": args.tts_ddpm_steps,
        "tts_language_model_path": args.tts_language_model_path,
    }

    speaker_1_prompt_audio, speaker_1_prompt_text = resolve_voice_prompt_overrides(args.speaker_1_voice_id, speaker_index=1)
    if options["tts_prompt_audio_s1"] is None and speaker_1_prompt_audio:
        options["tts_prompt_audio_s1"] = speaker_1_prompt_audio
    if options["tts_prompt_text_s1"] is None and speaker_1_prompt_text:
        options["tts_prompt_text_s1"] = speaker_1_prompt_text

    if args.dialogue_mode == "dual":
        speaker_2_prompt_audio, speaker_2_prompt_text = resolve_voice_prompt_overrides(args.speaker_2_voice_id, speaker_index=2)
        if options["tts_prompt_audio_s2"] is None and speaker_2_prompt_audio:
            options["tts_prompt_audio_s2"] = speaker_2_prompt_audio
        if options["tts_prompt_text_s2"] is None and speaker_2_prompt_text:
            options["tts_prompt_text_s2"] = speaker_2_prompt_text

    return options


def safe_read_json(path: Path) -> Any | None:
    try:
        return read_json(path)
    except Exception:
        return None


def safe_read_jsonl(path: Path) -> list[Any]:
    try:
        return read_jsonl(path)
    except Exception:
        return []


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def list_json_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def is_complete_chunk_card(path: Path, chunk_id: str) -> bool:
    data = safe_read_json(path)
    return (
        isinstance(data, dict)
        and str(data.get("chunk_id", "")).strip() == chunk_id
        and "summary" in data
        and isinstance(data.get("key_points"), list)
        and isinstance(data.get("core_terms"), list)
    )


def is_complete_section_card(path: Path, section_id: str | None = None) -> bool:
    data = safe_read_json(path)
    if not isinstance(data, dict):
        return False
    if section_id is not None and str(data.get("section_id", "")).strip() != section_id:
        return False
    return (
        "summary" in data
        and "thesis_or_function" in data
        and isinstance(data.get("source_chunk_ids"), list)
        and isinstance(data.get("key_points"), list)
        and isinstance(data.get("core_terms"), list)
    )


def is_complete_book_charter(path: Path) -> bool:
    data = safe_read_json(path)
    return isinstance(data, dict) and all(
        key in data
        for key in (
            "book_id",
            "book_title",
            "book_summary",
            "global_theme",
            "core_argument_or_mainline",
            "global_terms",
            "planning_notes",
        )
    )


def is_complete_program_config(path: Path) -> bool:
    data = safe_read_json(path)
    return isinstance(data, dict) and all(
        key in data
        for key in (
            "series_id",
            "primary_language",
            "mode",
            "allow_external_knowledge",
            "target_episode_minutes",
            "show_title",
            "positioning",
            "target_audience",
            "language_output_rules",
            "pace_style",
            "target_script_chars",
            "target_input_chars",
            "tone_guardrails",
            "content_guardrails",
        )
    )


def is_complete_speaker_bible(path: Path, expected_speaker_count: int) -> bool:
    data = safe_read_json(path)
    if not isinstance(data, dict):
        return False
    speakers = data.get("speakers")
    if not isinstance(speakers, list) or len(speakers) < expected_speaker_count:
        return False
    return all(
        key in data
        for key in (
            "dialogue_mode",
            "fixed_opening",
            "speakers",
            "interaction_rules",
            "consistency_rules",
            "forbidden_behaviors",
        )
    )


def is_complete_episode_plan(path: Path) -> bool:
    data = safe_read_json(path)
    episodes = data.get("episodes") if isinstance(data, dict) else None
    if not isinstance(episodes, list) or not episodes:
        return False
    return all(isinstance(item, dict) and str(item.get("episode_id", "")).strip() for item in episodes)


def is_complete_source_pack(path: Path, episode_id: str) -> bool:
    data = safe_read_json(path)
    return (
        isinstance(data, dict)
        and str(data.get("episode_id", "")).strip() == episode_id
        and isinstance(data.get("episode_plan"), dict)
        and isinstance(data.get("program_config"), dict)
        and isinstance(data.get("speaker_bible"), dict)
        and isinstance(data.get("raw_text_chunks"), list)
    )


def is_complete_script(path: Path) -> bool:
    if not is_nonempty_file(path):
        return False
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def is_complete_audio_meta(path: Path, episode_id: str, expected_tts_backend: str | None = None) -> bool:
    data = safe_read_json(path)
    if not isinstance(data, dict):
        return False
    if str(data.get("episode_id", "")).strip() != episode_id:
        return False
    if expected_tts_backend is not None and str(data.get("tts_backend", "")).strip() != expected_tts_backend:
        return False

    output_path = Path(str(data.get("output_path", "")).strip())
    if not output_path or not is_nonempty_file(output_path):
        return False

    segment_paths = data.get("segment_paths")
    if not isinstance(segment_paths, list):
        return False
    for segment_path in segment_paths:
        segment_text = str(segment_path).strip()
        if not segment_text:
            return False
        if not is_nonempty_file(Path(segment_text)):
            return False
    return True


def is_complete_episode_card(path: Path, episode_id: str) -> bool:
    data = safe_read_json(path)
    return (
        isinstance(data, dict)
        and str(data.get("episode_id", "")).strip() == episode_id
        and "summary" in data
        and isinstance(data.get("covered_sections"), list)
        and isinstance(data.get("covered_chunks"), list)
    )


def inspect_step1_outputs(output_dir: Path, input_path: Path) -> dict[str, Any]:
    book_structure_path = output_dir / "book_structure.json"
    chunk_source_map_path = output_dir / "chunk_source_map.jsonl"
    full_text_path = output_dir / "full_text.txt"
    input_copy_path = output_dir / "input" / input_path.name
    raw_chunk_dir = output_dir / "raw_text" / "chunks"

    if not (
        is_nonempty_file(book_structure_path)
        and is_nonempty_file(chunk_source_map_path)
        and is_nonempty_file(full_text_path)
        and is_nonempty_file(input_copy_path)
    ):
        return {"complete": False}

    book_structure = safe_read_json(book_structure_path)
    chunk_records = safe_read_jsonl(chunk_source_map_path)
    if not isinstance(book_structure, dict) or not chunk_records:
        return {"complete": False}

    for chunk_record in chunk_records:
        if not isinstance(chunk_record, dict):
            return {"complete": False}
        chunk_id = str(chunk_record.get("chunk_id", "")).strip()
        if not chunk_id or not is_nonempty_file(raw_chunk_dir / f"{chunk_id}.txt"):
            return {"complete": False}

    sections = book_structure.get("sections")
    sections_count = len(sections) if isinstance(sections, list) else 0
    return {
        "complete": True,
        "sections_count": sections_count,
        "chunks_count": len(chunk_records),
    }


def inspect_step2_outputs(output_dir: Path) -> dict[str, Any]:
    book_structure = safe_read_json(output_dir / "book_structure.json")
    chunk_records = safe_read_jsonl(output_dir / "chunk_source_map.jsonl")
    chunk_card_dir = output_dir / "chunk_cards"
    section_card_dir = output_dir / "section_cards"
    if not isinstance(book_structure, dict) or not chunk_records:
        return {"complete": False}

    for chunk_record in chunk_records:
        if not isinstance(chunk_record, dict):
            return {"complete": False}
        chunk_id = str(chunk_record.get("chunk_id", "")).strip()
        if not chunk_id or not is_complete_chunk_card(chunk_card_dir / f"{chunk_id}.json", chunk_id):
            return {"complete": False}

    sections = book_structure.get("sections")
    sections_detected = bool(book_structure.get("sections_detected")) and isinstance(sections, list) and bool(sections)
    section_cards = list_json_files(section_card_dir)
    if sections_detected:
        assert isinstance(sections, list)
        for section in sections:
            if not isinstance(section, dict):
                return {"complete": False}
            section_id = str(section.get("section_id", "")).strip()
            if not section_id or not is_complete_section_card(section_card_dir / f"{section_id}.json", section_id):
                return {"complete": False}
        section_card_count = len(sections)
    else:
        if not section_cards:
            return {"complete": False}
        if any(not is_complete_section_card(path) for path in section_cards):
            return {"complete": False}
        section_card_count = len(section_cards)

    return {
        "complete": True,
        "chunk_card_count": len(chunk_records),
        "section_card_count": section_card_count,
    }


def inspect_step3_outputs(output_dir: Path) -> dict[str, Any]:
    complete = is_complete_book_charter(output_dir / "book_charter.json")
    return {"complete": complete}


def inspect_step4_outputs(output_dir: Path, expected_user_preferences: dict) -> dict[str, Any]:
    user_preferences_path = output_dir / "user_preferences.json"
    stored_user_preferences = safe_read_json(user_preferences_path)
    if stored_user_preferences != expected_user_preferences:
        return {"complete": False}

    expected_speaker_count = int(expected_user_preferences.get("speaker_count", 1))
    complete = (
        is_nonempty_file(user_preferences_path)
        and is_complete_program_config(output_dir / "program_config.json")
        and is_complete_speaker_bible(output_dir / "speaker_bible.json", expected_speaker_count)
    )
    return {"complete": complete}


def inspect_step5_outputs(output_dir: Path) -> dict[str, Any]:
    all_cards_summary = safe_read_json(output_dir / "all_cards_summary.json")
    if not isinstance(all_cards_summary, list) or not all_cards_summary:
        return {"complete": False}
    return {
        "complete": is_complete_episode_plan(output_dir / "episode_plan.json"),
    }


def inspect_step6_episode(output_dir: Path, episode_id: str) -> bool:
    return is_complete_source_pack(output_dir / "source_packs" / f"{episode_id}.json", episode_id) and is_complete_script(
        output_dir / "scripts" / f"{episode_id}.txt"
    )


def inspect_step7_episode(output_dir: Path, episode_id: str, expected_tts_backend: str | None = None) -> bool:
    audio_path = output_dir / "audios" / f"{episode_id}.wav"
    audio_meta_path = output_dir / "audio_meta" / f"{episode_id}.json"
    return is_nonempty_file(audio_path) and is_complete_audio_meta(audio_meta_path, episode_id, expected_tts_backend)


def inspect_step8_completed_prefix(output_dir: Path, episode_ids: list[str]) -> list[str]:
    memory_summary = safe_read_json(output_dir / "series_memory_summary.json")
    if not isinstance(memory_summary, dict):
        return []
    completed_ids = memory_summary.get("completed_episode_ids")
    if not isinstance(completed_ids, list):
        return []

    completed_set = {str(item).strip() for item in completed_ids if str(item).strip()}
    completed_prefix: list[str] = []
    for episode_id in episode_ids:
        episode_card_path = output_dir / "episode_cards" / f"{episode_id}.json"
        if episode_id in completed_set and inspect_step6_episode(output_dir, episode_id) and is_complete_episode_card(
            episode_card_path, episode_id
        ):
            completed_prefix.append(episode_id)
            continue
        break
    return completed_prefix


# ── Progress printing ──


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.0f}s"


def _step_header(step: int, msg: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  [Step {step}/{TOTAL_STEPS}] {msg}")
    print(f"{'=' * 50}")


def _step_done(step: int, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"  [Step {step}/{TOTAL_STEPS}] Done.{suffix}")


def _step_skipped(step: int, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"  [Step {step}/{TOTAL_STEPS}] Skip.{suffix}")


def _episode_header(idx: int, total: int, episode_id: str) -> None:
    print(f"\n  --- [Episode {idx}/{total}] {episode_id} ---")


# ── Pipeline ──


def run_full_pipeline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    pipeline_start = time.time()
    resume_enabled = not args.force

    print(f"\nAnyPod Pipeline")
    print(f"  Input:  {args.input_path}")
    print(f"  Output: {output_dir}")

    # ── LLM clients ──
    llm_kwargs = dict(
        url=args.llm_url,
        model=args.llm_model,
        timeout=args.llm_timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    understanding_llm = SimpleLLMClient(agent_name=UNDERSTANDING_AGENT_NAME, **llm_kwargs)
    plan_llm = SimpleLLMClient(agent_name=PLAN_AGENT_NAME, **llm_kwargs)
    writing_llm = SimpleLLMClient(agent_name=WRITING_AGENT_NAME, **llm_kwargs)
    user_preferences = build_user_preferences(args)
    tts_options = build_tts_options(args)

    # ── Cache check ──
    from bookcast.step_cache import find_cache, restore_from_cache, save_to_cache

    cache_hit_dir = find_cache(args.input_path)
    if cache_hit_dir is not None:
        restore_from_cache(cache_hit_dir, output_dir)
        print("  Cache hit — 跳过 Step 1~3")

    # ── Step 1: Parse ──
    _step_header(1, f"Parse source file: {args.input_path.name}")
    step1_reused = False
    step1_state = inspect_step1_outputs(output_dir, args.input_path.resolve())
    if resume_enabled and step1_state.get("complete"):
        step1_reused = True
        _step_skipped(
            1,
            f"Reuse existing parse outputs ({step1_state['sections_count']} sections, {step1_state['chunks_count']} chunks)",
        )
    else:
        t0 = time.time()
        step1 = Step1Parse(
            input_path=args.input_path.resolve(),
            output_dir=output_dir,
            chunk_max_words=args.chunk_max_words,
            force=args.force,
            ignore_bookmarks=args.ignore_bookmarks,
        )
        r1 = step1.run()
        _step_done(1, f"{r1['sections_count']} sections, {r1['chunks_count']} chunks ({_fmt_elapsed(time.time() - t0)})")
    resume_chain = resume_enabled and step1_reused

    # ── Step 2: Understand ──
    _step_header(2, f"Understand content (workers={args.step2_num_workers})")
    step2_reused = False
    step2_state = inspect_step2_outputs(output_dir)
    if resume_chain and step2_state.get("complete"):
        step2_reused = True
        _step_skipped(
            2,
            f"Reuse existing understanding outputs ({step2_state['chunk_card_count']} chunk cards, {step2_state['section_card_count']} section cards)",
        )
    else:
        t0 = time.time()
        step2 = Step2Understand(
            output_dir=output_dir,
            llm_client=understanding_llm,
            max_retries=args.max_retries,
            num_workers=args.step2_num_workers,
        )
        r2 = step2.run()
        _step_done(2, f"{r2['chunk_card_count']} chunk cards, {r2['section_card_count']} section cards ({_fmt_elapsed(time.time() - t0)})")
    resume_chain = resume_chain and step2_reused

    # ── Step 3: Book Charter ──
    _step_header(3, "Generate book charter")
    step3_reused = False
    step3_state = inspect_step3_outputs(output_dir)
    if resume_chain and step3_state.get("complete"):
        step3_reused = True
        _step_skipped(3, "Reuse existing book charter")
    else:
        t0 = time.time()
        step3 = Step3BookCharter(
            output_dir=output_dir,
            llm_client=plan_llm,
            max_retries=args.max_retries,
        )
        step3.run()
        _step_done(3, f"({_fmt_elapsed(time.time() - t0)})")
    resume_chain = resume_chain and step3_reused

    # ── Save cache (仅在非缓存命中时) ──
    if cache_hit_dir is None:
        save_to_cache(args.input_path, output_dir)

    # ── Step 4: Global Plan ──
    _step_header(4, "Generate show config and speaker bible")
    step4_reused = False
    step4_state = inspect_step4_outputs(output_dir, user_preferences)
    if resume_chain and step4_state.get("complete"):
        step4_reused = True
        _step_skipped(4, "Reuse existing show config and speaker bible")
    else:
        t0 = time.time()
        step4 = Step4GlobalPlan(
            output_dir=output_dir,
            llm_client=plan_llm,
            user_preferences=user_preferences,
            max_retries=args.max_retries,
        )
        step4.run()
        _step_done(4, f"({_fmt_elapsed(time.time() - t0)})")
    resume_chain = resume_chain and step4_reused

    # ── Step 5: Episode Plan ──
    _step_header(5, "Plan episodes")
    step5_reused = False
    step5_state = inspect_step5_outputs(output_dir)
    if resume_chain and step5_state.get("complete"):
        step5_reused = True
    else:
        t0 = time.time()
        step5 = Step5EpisodePlan(
            output_dir=output_dir,
            llm_client=plan_llm,
            max_retries=args.max_retries,
        )
        step5.run()
    episode_ids = resolve_episode_ids(output_dir, args.episode_ids)
    if not episode_ids:
        raise FileNotFoundError("No matching episode_id was found for processing")
    if step5_reused:
        _step_skipped(5, f"Reuse existing episode plan ({len(episode_ids)} episodes)")
    else:
        _step_done(5, f"{len(episode_ids)} episodes ({_fmt_elapsed(time.time() - t0)})")
    resume_chain = resume_chain and step5_reused

    step6_complete_ids = (
        {episode_id for episode_id in episode_ids if inspect_step6_episode(output_dir, episode_id)}
        if resume_chain
        else set()
    )
    step7_complete_ids = (
        {episode_id for episode_id in episode_ids if inspect_step7_episode(output_dir, episode_id, args.tts_backend)}
        if resume_chain and not args.skip_tts
        else set()
    )
    step8_completed_prefix_ids = set(inspect_step8_completed_prefix(output_dir, episode_ids)) if resume_chain else set()

    # ── TTS worker (async) ──
    if args.tts_backend == "moss-tts" and args.dialogue_mode == "dual":
        raise ValueError("moss-tts 后端仅支持单人模式 (dialogue_mode=single)，不支持双人模式")

    tts_worker = None
    pending_tts_episode_ids = []
    if not args.skip_tts:
        if resume_chain:
            pending_tts_episode_ids = [
                episode_id
                for episode_id in episode_ids
                if episode_id not in step6_complete_ids or episode_id not in step7_complete_ids
            ]
        else:
            pending_tts_episode_ids = list(episode_ids)

    if pending_tts_episode_ids:
        tts_worker = AsyncTTSWorker(
            Step7TTS(
                output_dir=output_dir,
                tts_backend=args.tts_backend,
                tts_options=tts_options,
            ),
            total_episode_count=len(pending_tts_episode_ids),
        )
        tts_worker.start()
    elif not args.skip_tts and resume_chain:
        print(f"\n  [Step 7/{TOTAL_STEPS}] Skip. All episode-level TTS outputs already exist")

    # ── Episode loop: Step 6 → 7(enqueue) → 8 ──
    total_episodes = len(episode_ids)
    try:
        for idx, episode_id in enumerate(episode_ids, start=1):
            _episode_header(idx, total_episodes, episode_id)
            step6_reused = resume_chain and episode_id in step6_complete_ids

            # Step 6: Script
            if step6_reused:
                print(f"    [Step 6/{TOTAL_STEPS}] Skip. Reuse existing source pack and script")
            else:
                print(f"    [Step 6/{TOTAL_STEPS}] Generate script ...")
                t0 = time.time()
                step6 = Step6SourcePackScript(
                    output_dir=output_dir,
                    llm_client=writing_llm,
                    episode_ids=[episode_id],
                    max_retries=args.max_retries,
                )
                step6.run()
                print(f"    [Step 6/{TOTAL_STEPS}] Done. ({_fmt_elapsed(time.time() - t0)})")

            # Step 7: TTS enqueue
            step7_reused = resume_chain and step6_reused and episode_id in step7_complete_ids
            if step7_reused:
                print(f"    [Step 7/{TOTAL_STEPS}] Skip. Reuse existing audio")
            elif tts_worker is not None:
                tts_worker.enqueue_episode(episode_id)
                print(f"    [Step 7/{TOTAL_STEPS}] TTS queued")

            # Step 8: Episode Memory
            step8_reused = resume_chain and step6_reused and episode_id in step8_completed_prefix_ids
            if step8_reused:
                print(f"    [Step 8/{TOTAL_STEPS}] Skip. Reuse existing episode memory")
            else:
                print(f"    [Step 8/{TOTAL_STEPS}] Update memory ...")
                t0 = time.time()
                step8 = Step8EpisodeMemory(
                    output_dir=output_dir,
                    llm_client=writing_llm,
                    episode_ids=[episode_id],
                    max_retries=args.max_retries,
                )
                step8.run()
                print(f"    [Step 8/{TOTAL_STEPS}] Done. ({_fmt_elapsed(time.time() - t0)})")
    except Exception:
        if tts_worker is not None:
            tts_worker.close_and_wait()
        raise

    # ── Wait for TTS ──
    if tts_worker is not None:
        print(f"\n  [Step 7/{TOTAL_STEPS}] Waiting for TTS to finish ...")
        t0 = time.time()
        tts_result = tts_worker.close_and_wait()
        ok = len(tts_result.get("generated_episode_ids", []))
        fail = len(tts_result.get("failed_episode_ids", []))
        detail = f"{ok} succeeded"
        if fail:
            detail += f", {fail} failed"
        print(f"  [Step 7/{TOTAL_STEPS}] Done. {detail} ({_fmt_elapsed(time.time() - t0)})")

    elapsed = _fmt_elapsed(time.time() - pipeline_start)
    print(f"\n{'=' * 50}")
    print(f"  Pipeline completed ({elapsed})")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 50}\n")


# ── Argparse ──


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anypod",
        description="AnyPod - 将任意书籍转化为播客系列，全流程一键运行",
    )
    parser.add_argument(
        "--config_json",
        type=Path,
        default=None,
        help="JSON 配置文件路径；可包含任意命令行参数字段，命令行显式传参优先级更高",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="输入文件绝对路径，支持 pdf/txt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="输出目录绝对路径",
    )
    parser.add_argument(
        "--chunk_max_words",
        type=int,
        default=3000,
        help="chunk 最大字数/词数，中文一个字算一个，英文一个单词算一个 (默认: 3000)",
    )
    parser.add_argument(
        "--ignore_bookmarks",
        action="store_true",
        help="强制忽略 PDF 自带书签，用于测试无书签分支",
    )
    parser.add_argument("--force", action="store_true", help="强制覆盖已有输入副本")
    parser.add_argument(
        "--dialogue_mode",
        type=str,
        default="dual",
        choices=["single", "dual"],
        help="主持人模式 (默认: dual)",
    )
    parser.add_argument(
        "--primary_language",
        type=str,
        default="zh",
        help="节目主输出语种，如 zh/en (默认: zh)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="deep_dive",
        choices=["faithful", "concise", "deep_dive"],
        help="播客模式 (默认: deep_dive)",
    )
    parser.add_argument(
        "--target_episode_level",
        type=str,
        default="mid",
        choices=["low", "mid", "high"],
        help="每集目标时长等级，low/mid/high 映射 10/20/30 分钟 (默认: mid)",
    )
    parser.add_argument(
        "--allow_external_knowledge",
        type=parse_bool_arg,
        default=True,
        help="是否允许补充外部知识 (默认: true)",
    )
    parser.add_argument("--show_title", type=str, default="", help="节目名称偏好，留空则由 LLM 自行设计")
    parser.add_argument("--positioning", type=str, default="", help="节目定位偏好，留空则由 LLM 自行设计")
    parser.add_argument("--target_audience", type=str, default="", help="目标听众偏好，留空则由 LLM 自行设计")
    parser.add_argument("--fixed_opening", type=str, default="", help="固定开场白偏好，留空则由 LLM 自行设计")
    parser.add_argument("--speaker_1_name", type=str, default="", help="主持人一名称偏好")
    parser.add_argument("--speaker_1_style", type=str, default="", help="主持人一性格特点/风格提示")
    parser.add_argument("--speaker_1_voice_id", type=str, default="voice_a", help="说话人一的 TTS 音色 ID")
    parser.add_argument("--speaker_1_language", type=str, default="zh", help="说话人一语种")
    parser.add_argument("--speaker_2_name", type=str, default="", help="主持人二名称偏好")
    parser.add_argument("--speaker_2_style", type=str, default="", help="主持人二性格特点/风格提示")
    parser.add_argument("--speaker_2_voice_id", type=str, default="voice_b", help="说话人二的 TTS 音色 ID")
    parser.add_argument("--speaker_2_language", type=str, default="zh", help="说话人二语种")
    parser.add_argument(
        "--episode_ids",
        type=parse_episode_ids,
        default=[],
        help="要处理的集数 ID，逗号分隔，如 E001,E002；默认 all",
    )
    parser.add_argument("--skip_tts", type=parse_bool_arg, default=False, help="是否跳过 TTS (默认: false)")
    parser.add_argument(
        "--tts_backend",
        type=str,
        default="moss-ttsd",
        choices=["moss-ttsd", "vibevoice", "moss-tts", "moss-tts(api)"],
        help="TTS 后端选择 (默认: moss-ttsd)",
    )
    parser.add_argument("--tts_model_path", type=str, default=None, help="TTS 模型目录；不同后端按各自格式解释")
    parser.add_argument(
        "--tts_audio_tokenizer_path",
        type=str,
        default=None,
        help="仅 moss-ttsd 使用的 audio tokenizer 目录",
    )
    parser.add_argument(
        "--tts_prompt_audio_s1",
        type=str,
        default=None,
        help="说话人一参考音频绝对路径；两种 TTS 后端都可用",
    )
    parser.add_argument(
        "--tts_prompt_audio_s2",
        type=str,
        default=None,
        help="说话人二参考音频绝对路径；双人模式下推荐提供",
    )
    parser.add_argument(
        "--tts_prompt_text_s1",
        type=str,
        default=None,
        help="仅 moss-ttsd 使用的说话人一参考文本",
    )
    parser.add_argument(
        "--tts_prompt_text_s2",
        type=str,
        default=None,
        help="仅 moss-ttsd 使用的说话人二参考文本",
    )
    parser.add_argument("--tts_device", type=str, default=None, help="TTS 推理设备，如 cuda、cuda:0、cpu")
    parser.add_argument(
        "--tts_torch_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="TTS 加载模型时使用的 torch dtype",
    )
    parser.add_argument(
        "--tts_attn_implementation",
        type=str,
        default=None,
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="TTS attention backend；vibevoice 支持 auto，moss-ttsd 主要使用显式实现",
    )
    parser.add_argument("--tts_cfg_scale", type=float, default=None, help="仅 vibevoice 使用的 CFG scale")
    parser.add_argument("--tts_ddpm_steps", type=int, default=None, help="仅 vibevoice 使用的 DDPM 步数")
    parser.add_argument(
        "--tts_language_model_path",
        type=str,
        default=None,
        help="仅 vibevoice 使用的 tokenizer 路径覆盖",
    )
    parser.add_argument("--llm_url", type=str, default=None, help="全局覆盖 LLM 接口地址")
    parser.add_argument("--llm_model", type=str, default=None, help="全局覆盖 LLM 模型名")
    parser.add_argument("--llm_timeout", type=int, default=None, help="全局覆盖 LLM 超时秒数")
    parser.add_argument("--temperature", type=float, default=None, help="采样温度")
    parser.add_argument("--top_p", type=float, default=None, help="top_p")
    parser.add_argument("--top_k", type=int, default=None, help="top_k")
    parser.add_argument("--max_retries", type=int, default=5, help="单个 LLM 请求的最大重试次数 (默认: 5)")
    parser.add_argument("--step2_num_workers", type=int, default=4, help="第二步多进程并发数 (默认: 4)")
    parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: WARNING)",
    )
    return parser


# ── Entry ──


def main() -> None:
    args = parse_args_with_config()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input_path}")

    run_full_pipeline(args)


if __name__ == "__main__":
    main()
