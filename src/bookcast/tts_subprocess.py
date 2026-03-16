from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bookcast.core.io import write_json


TTS_MODULE_FILES = {
    "moss-ttsd": "demo_tts.py",
    "vibevoice": "vibevoice_tts.py",
    "moss-tts": "moss_tts.py",
    "moss-tts(api)": "moss_tts_api.py",
}
TTS_OPTION_ENV_VARS = {
    "tts_model_path": "ANYPOD_TTS_MODEL_PATH",
    "tts_audio_tokenizer_path": "ANYPOD_TTS_AUDIO_TOKENIZER_PATH",
    "tts_prompt_audio_s1": "ANYPOD_TTS_PROMPT_AUDIO_S1",
    "tts_prompt_audio_s2": "ANYPOD_TTS_PROMPT_AUDIO_S2",
    "tts_prompt_text_s1": "ANYPOD_TTS_PROMPT_TEXT_S1",
    "tts_prompt_text_s2": "ANYPOD_TTS_PROMPT_TEXT_S2",
    "tts_device": "ANYPOD_TTS_DEVICE",
    "tts_torch_dtype": "ANYPOD_TTS_TORCH_DTYPE",
    "tts_attn_implementation": "ANYPOD_TTS_ATTN_IMPLEMENTATION",
    "tts_cfg_scale": "ANYPOD_TTS_CFG_SCALE",
    "tts_ddpm_steps": "ANYPOD_TTS_DDPM_STEPS",
    "tts_language_model_path": "ANYPOD_TTS_LANGUAGE_MODEL_PATH",
}
TTS_READY_MARKER = "__ANYPOD_TTS_READY__ "
TTS_RESULT_MARKER = "__ANYPOD_TTS_RESULT__ "


def ensure_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single AnyPod TTS episode in an isolated runtime.")
    parser.add_argument("--output_dir", type=Path, required=True, help="AnyPod output directory")
    parser.add_argument("--episode_id", type=str, default=None, help="Episode ID")
    parser.add_argument(
        "--tts_backend",
        type=str,
        required=True,
        choices=["moss-ttsd", "vibevoice", "moss-tts", "moss-tts(api)"],
        help="TTS backend",
    )
    parser.add_argument("--result_path", type=Path, default=None, help="Path to the subprocess result JSON")
    parser.add_argument("--serve", action="store_true", help="Run as a persistent TTS service")
    parser.add_argument("--tts_model_path", type=str, default=None)
    parser.add_argument("--tts_audio_tokenizer_path", type=str, default=None)
    parser.add_argument("--tts_prompt_audio_s1", type=str, default=None)
    parser.add_argument("--tts_prompt_audio_s2", type=str, default=None)
    parser.add_argument("--tts_prompt_text_s1", type=str, default=None)
    parser.add_argument("--tts_prompt_text_s2", type=str, default=None)
    parser.add_argument("--tts_device", type=str, default=None)
    parser.add_argument("--tts_torch_dtype", type=str, default=None)
    parser.add_argument("--tts_attn_implementation", type=str, default=None)
    parser.add_argument("--tts_cfg_scale", type=float, default=None)
    parser.add_argument("--tts_ddpm_steps", type=int, default=None)
    parser.add_argument("--tts_language_model_path", type=str, default=None)
    return parser


def apply_tts_env_overrides(args: argparse.Namespace) -> None:
    os.environ["ANYPOD_TTS_BACKEND"] = args.tts_backend
    for option_name, env_name in TTS_OPTION_ENV_VARS.items():
        option_value = getattr(args, option_name)
        if option_value is None:
            continue
        if isinstance(option_value, str) and not option_value.strip():
            continue
        os.environ[env_name] = str(option_value)


def load_tts_module(tts_backend: str) -> ModuleType:
    module_path = Path(__file__).resolve().parent / TTS_MODULE_FILES[tts_backend]
    if not module_path.exists():
        raise FileNotFoundError(f"Missing TTS module file: {module_path}")

    module_name = f"bookcast_tts_runtime_{tts_backend.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TTS module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_tts_episode(
    output_dir: Path,
    episode_id: str,
    tts_backend: str,
    tts_module: ModuleType,
) -> dict[str, Any]:
    script_path = output_dir / "scripts" / f"{episode_id}.txt"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script file: {script_path}")

    script_text = script_path.read_text(encoding="utf-8")
    segment_dir = output_dir / "audio_segments" / episode_id
    output_path = output_dir / "audios" / f"{episode_id}.wav"
    result = tts_module.synthesize_podcast(
        text_to_generate=script_text,
        output_dir=segment_dir,
        output_path=output_path,
    )

    normalized_output_path = ensure_string(result.get("output_path"), str(output_path))
    raw_segment_paths = result.get("segment_paths")
    if not isinstance(raw_segment_paths, list):
        raw_segment_paths = [normalized_output_path]
    segment_paths = [ensure_string(item) for item in raw_segment_paths if ensure_string(item)]

    return {
        "episode_id": episode_id,
        "tts_backend": tts_backend,
        "tts_mode": ensure_string(getattr(tts_module, "TTS_MODE", None), tts_backend),
        "tts_module_name": ensure_string(getattr(tts_module, "__name__", None), tts_backend),
        "output_path": normalized_output_path,
        "segment_paths": segment_paths,
    }


def emit_marker(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix}{json.dumps(payload, ensure_ascii=False)}", flush=True)


def serve_requests(
    output_dir: Path,
    tts_backend: str,
    tts_module: ModuleType,
) -> None:
    emit_marker(
        TTS_READY_MARKER,
        {
            "tts_backend": tts_backend,
            "tts_mode": ensure_string(getattr(tts_module, "TTS_MODE", None), tts_backend),
            "tts_module_name": ensure_string(getattr(tts_module, "__name__", None), tts_backend),
        },
    )

    for raw_line in sys.stdin:
        command_text = raw_line.strip()
        if not command_text:
            continue

        command = json.loads(command_text)
        action = ensure_string(command.get("action"), "synthesize")
        if action == "shutdown":
            break

        episode_id = ensure_string(command.get("episode_id"))
        if not episode_id:
            emit_marker(
                TTS_RESULT_MARKER,
                {
                    "status": "error",
                    "episode_id": "",
                    "error": "Missing episode_id",
                },
            )
            continue

        try:
            result = run_tts_episode(
                output_dir=output_dir,
                episode_id=episode_id,
                tts_backend=tts_backend,
                tts_module=tts_module,
            )
            emit_marker(
                TTS_RESULT_MARKER,
                {
                    "status": "ok",
                    "episode_id": episode_id,
                    "result": result,
                },
            )
        except Exception as exc:  # noqa: BLE001
            emit_marker(
                TTS_RESULT_MARKER,
                {
                    "status": "error",
                    "episode_id": episode_id,
                    "error": str(exc),
                },
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    apply_tts_env_overrides(args)
    tts_module = load_tts_module(args.tts_backend)
    module_prepare_runtime = getattr(tts_module, "prepare_runtime", None)
    if callable(module_prepare_runtime):
        module_prepare_runtime()

    if args.serve:
        serve_requests(
            output_dir=output_dir,
            tts_backend=args.tts_backend,
            tts_module=tts_module,
        )
        return

    if not args.episode_id:
        parser.error("--episode_id is required when not running in service mode")
    if args.result_path is None:
        parser.error("--result_path is required when not running in service mode")

    result = run_tts_episode(
        output_dir=output_dir,
        episode_id=args.episode_id,
        tts_backend=args.tts_backend,
        tts_module=tts_module,
    )

    write_json(
        result,
        args.result_path.resolve(),
    )


if __name__ == "__main__":
    main()
