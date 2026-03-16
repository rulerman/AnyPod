from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bookcast.core.voice_library import resolve_default_voice_prompt


DEFAULT_MODEL_PATH = str(SRC_DIR.parent / "model" / "VibeVoice-1.5B")
DEFAULT_TEXT_TO_GENERATE = "[S1]你好。[S2]你好。"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TORCH_DTYPE = "bfloat16" if DEFAULT_DEVICE.startswith("cuda") else "float32"
DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_CFG_SCALE = 1.3
DEFAULT_DDPM_STEPS = 10
TTS_BACKEND = "vibevoice"
TTS_MODE = "vibevoice_minimal_inference"

SPEAKER_TAG_PATTERN = re.compile(r"\[S(\d+)\]", re.IGNORECASE)
_RUNTIME_CACHE: tuple[Any, Any, dict[str, Any]] | None = None
DEFAULT_PROMPT_AUDIO_SPEAKER1, _DEFAULT_PROMPT_TEXT_UNUSED_S1 = resolve_default_voice_prompt(1)
DEFAULT_PROMPT_AUDIO_SPEAKER2, _DEFAULT_PROMPT_TEXT_UNUSED_S2 = resolve_default_voice_prompt(2)


def env_text(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def env_optional_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def resolve_existing_path(path_text: str, label: str) -> str:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} points to a path that does not exist: {path}")
    return str(path.resolve())


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported TTS dtype: {dtype_name}") from exc


def resolve_attn_candidates(attn_implementation: str, device: str) -> list[str]:
    if attn_implementation != "auto":
        return [attn_implementation]
    if device.startswith("cuda"):
        return ["flash_attention_2", "sdpa", "eager"]
    return ["sdpa", "eager"]


def convert_tagged_text_to_script(raw_text: str) -> tuple[str, list[str], dict[str, int]]:
    segments: list[tuple[str, str]] = []
    cleaned_segments: list[tuple[str, str]] = []
    last_end = 0
    current_speaker: str | None = None

    for match in SPEAKER_TAG_PATTERN.finditer(raw_text):
        if current_speaker is not None:
            segments.append((current_speaker, raw_text[last_end:match.start()]))
        current_speaker = match.group(1)
        last_end = match.end()

    if current_speaker is not None:
        segments.append((current_speaker, raw_text[last_end:]))

    for speaker_id, segment_text in segments:
        normalized_text = re.sub(r"\s+", " ", segment_text).strip()
        if normalized_text:
            cleaned_segments.append((speaker_id, normalized_text))

    if not cleaned_segments:
        raise ValueError("No valid [S1]/[S2] speaker tags were detected in the script")

    source_speaker_ids = sorted({int(speaker_id) for speaker_id, _ in cleaned_segments})
    speaker_mapping = {
        str(source_speaker_id): normalized_index + 1
        for normalized_index, source_speaker_id in enumerate(source_speaker_ids)
    }
    script_lines = [
        f"Speaker {speaker_mapping[speaker_id]}: {segment_text}"
        for speaker_id, segment_text in cleaned_segments
    ]
    return "\n".join(script_lines), [str(speaker_id) for speaker_id in source_speaker_ids], speaker_mapping


def prepare_voice_samples(source_speaker_ids: list[str]) -> list[str]:
    voice_samples: list[str] = []
    for speaker_id in source_speaker_ids:
        if speaker_id == "1":
            resolved_audio = resolve_existing_path(
                env_text("ANYPOD_TTS_PROMPT_AUDIO_S1", DEFAULT_PROMPT_AUDIO_SPEAKER1),
                "ANYPOD_TTS_PROMPT_AUDIO_S1",
            )
        elif speaker_id == "2":
            resolved_audio = resolve_existing_path(
                env_text("ANYPOD_TTS_PROMPT_AUDIO_S2", DEFAULT_PROMPT_AUDIO_SPEAKER2),
                "ANYPOD_TTS_PROMPT_AUDIO_S2",
            )
        else:
            raise ValueError(f"VibeVoice currently supports only 1 or 2 reference speakers, but received: S{speaker_id}")
        voice_samples.append(resolved_audio)
    return voice_samples


def load_runtime_config() -> dict[str, Any]:
    device = env_text("ANYPOD_TTS_DEVICE", DEFAULT_DEVICE)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable, but ANYPOD_TTS_DEVICE is set to {device}")

    dtype_name = env_text("ANYPOD_TTS_TORCH_DTYPE", DEFAULT_TORCH_DTYPE)
    return {
        "model_path": resolve_existing_path(env_text("ANYPOD_TTS_MODEL_PATH", DEFAULT_MODEL_PATH), "ANYPOD_TTS_MODEL_PATH"),
        "device": device,
        "dtype_name": dtype_name,
        "attn_implementation": env_text("ANYPOD_TTS_ATTN_IMPLEMENTATION", DEFAULT_ATTN_IMPLEMENTATION),
        "cfg_scale": float(env_optional_text("ANYPOD_TTS_CFG_SCALE") or DEFAULT_CFG_SCALE),
        "ddpm_steps": int(env_optional_text("ANYPOD_TTS_DDPM_STEPS") or DEFAULT_DDPM_STEPS),
        "language_model_path": env_optional_text("ANYPOD_TTS_LANGUAGE_MODEL_PATH"),
    }


def load_processor_and_model(config: dict[str, Any]) -> tuple[Any, Any, str, str]:
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    processor_kwargs = {}
    language_model_path = config.get("language_model_path")
    if language_model_path:
        processor_kwargs["language_model_pretrained_name"] = resolve_existing_path(
            str(language_model_path),
            "ANYPOD_TTS_LANGUAGE_MODEL_PATH",
        )

    processor = VibeVoiceProcessor.from_pretrained(str(config["model_path"]), **processor_kwargs)

    used_dtype_name = str(config["dtype_name"])
    if not str(config["device"]).startswith("cuda") and used_dtype_name != "float32":
        used_dtype_name = "float32"

    torch_dtype = get_torch_dtype(used_dtype_name)
    attn_candidates = resolve_attn_candidates(str(config["attn_implementation"]), str(config["device"]))

    last_error: Exception | None = None
    for attn_impl in attn_candidates:
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(config["model_path"]),
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            )
            model = model.to(str(config["device"]))
            model.eval()
            model.set_ddpm_inference_steps(num_steps=int(config["ddpm_steps"]))
            return processor, model, attn_impl, used_dtype_name
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("VibeVoice could not be loaded with the current attention backend") from last_error


def get_runtime() -> tuple[Any, Any, dict[str, Any]]:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    config = load_runtime_config()
    processor, model, used_attn_implementation, used_dtype_name = load_processor_and_model(config)
    runtime_info = {
        **config,
        "used_attn_implementation": used_attn_implementation,
        "used_dtype_name": used_dtype_name,
    }
    _RUNTIME_CACHE = (processor, model, runtime_info)
    return _RUNTIME_CACHE


def prepare_runtime() -> None:
    get_runtime()


def move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)

    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def synthesize_podcast(
    text_to_generate: str,
    output_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    processor, model, runtime_info = get_runtime()
    normalized_script, source_speaker_ids, speaker_mapping = convert_tagged_text_to_script(text_to_generate)
    voice_samples = prepare_voice_samples(source_speaker_ids)

    batch = processor(
        text=[normalized_script],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    batch = move_batch_to_device(batch, str(runtime_info["device"]))

    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=None,
            cfg_scale=float(runtime_info["cfg_scale"]),
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise RuntimeError("VibeVoice did not return valid audio")

    output_dir = Path(output_dir)
    output_path = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))

    return {
        "output_path": str(output_path),
        "segment_paths": [str(output_path)],
        "speaker_mapping": speaker_mapping,
        "voice_samples": voice_samples,
    }


def main() -> None:
    result = synthesize_podcast(
        text_to_generate=DEFAULT_TEXT_TO_GENERATE,
        output_dir=Path(__file__).resolve().parents[2] / "demo" / "output" / "vibevoice_tts",
        output_path=Path(__file__).resolve().parents[2] / "demo" / "output" / "vibevoice_tts" / "final_podcast.wav",
    )
    print(result["output_path"])


if __name__ == "__main__":
    main()
