from __future__ import annotations

import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bookcast.core.voice_library import resolve_default_voice_prompt
from bookcast.moss_processor_patch import patch_moss_processor_audio_loading


DEFAULT_MODEL_PATH = str(SRC_DIR.parent / "model" / "MOSS-TTSD-v1.0")
DEFAULT_AUDIO_TOKENIZER_PATH = str(SRC_DIR.parent / "model" / "MOSS-Audio-Tokenizer")
DEFAULT_TEXT_TO_GENERATE = "[S1]你好。[S2]你好。"
TTS_BACKEND = "moss-ttsd"
TTS_MODE = "moss_ttsd_default_reference"

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR.parent / "demo" / "output" / "tts"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "final_podcast.wav"


def env_text(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def env_existing_path(name: str, default: str) -> str:
    path = Path(env_text(name, default)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} points to a path that does not exist: {path}")
    return str(path)


PRETRAINED_MODEL_NAME_OR_PATH = env_existing_path("ANYPOD_TTS_MODEL_PATH", DEFAULT_MODEL_PATH)
AUDIO_TOKENIZER_NAME_OR_PATH = env_existing_path("ANYPOD_TTS_AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH)
DEFAULT_PROMPT_AUDIO_SPEAKER1, DEFAULT_PROMPT_TEXT_SPEAKER1 = resolve_default_voice_prompt(1)
DEFAULT_PROMPT_AUDIO_SPEAKER2, DEFAULT_PROMPT_TEXT_SPEAKER2 = resolve_default_voice_prompt(2)
PROMPT_AUDIO_SPEAKER1 = env_existing_path("ANYPOD_TTS_PROMPT_AUDIO_S1", DEFAULT_PROMPT_AUDIO_SPEAKER1)
PROMPT_AUDIO_SPEAKER2 = env_existing_path("ANYPOD_TTS_PROMPT_AUDIO_S2", DEFAULT_PROMPT_AUDIO_SPEAKER2)
PROMPT_TEXT_SPEAKER1 = env_text("ANYPOD_TTS_PROMPT_TEXT_S1", DEFAULT_PROMPT_TEXT_SPEAKER1)
PROMPT_TEXT_SPEAKER2 = env_text("ANYPOD_TTS_PROMPT_TEXT_S2", DEFAULT_PROMPT_TEXT_SPEAKER2)

device = env_text("ANYPOD_TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
dtype_name = env_text("ANYPOD_TTS_TORCH_DTYPE", "bfloat16" if device.startswith("cuda") else "float32")
dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}.get(dtype_name)
if dtype is None:
    raise ValueError(f"Unsupported ANYPOD_TTS_TORCH_DTYPE: {dtype_name}")

processor = AutoProcessor.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    trust_remote_code=True,
    codec_path=AUDIO_TOKENIZER_NAME_OR_PATH,
)
processor = patch_moss_processor_audio_loading(processor)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)
processor.audio_tokenizer.eval()

def resolve_attn_implementation() -> str:
    if not device.startswith("cuda"):
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"

attn_implementation = env_text("ANYPOD_TTS_ATTN_IMPLEMENTATION", resolve_attn_implementation())
model = AutoModel.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()


def load_and_resample_audio(audio_path: str | Path, target_sr: int) -> torch.Tensor:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio).transpose(0, 1).contiguous()

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sr)

    return wav


def build_conversations(
    text_to_generate: str,
    prompt_audio_speaker1: str = PROMPT_AUDIO_SPEAKER1,
    prompt_audio_speaker2: str = PROMPT_AUDIO_SPEAKER2,
    prompt_text_speaker1: str = PROMPT_TEXT_SPEAKER1,
    prompt_text_speaker2: str = PROMPT_TEXT_SPEAKER2,
) -> list[list[object]]:
    target_sr = int(processor.model_config.sampling_rate)
    wav1 = load_and_resample_audio(prompt_audio_speaker1, target_sr)
    has_speaker_2 = "[S2]" in text_to_generate

    reference_wavs = [wav1]
    prompt_text_parts = [prompt_text_speaker1]
    if has_speaker_2:
        wav2 = load_and_resample_audio(prompt_audio_speaker2, target_sr)
        reference_wavs.append(wav2)
        prompt_text_parts.append(prompt_text_speaker2)

    reference_audio_codes = processor.encode_audios_from_wav(reference_wavs, sampling_rate=target_sr)
    concat_prompt_wav = torch.cat(reference_wavs, dim=-1)
    prompt_audio = processor.encode_audios_from_wav([concat_prompt_wav], sampling_rate=target_sr)[0]

    full_text = f"{' '.join(prompt_text_parts)} {text_to_generate}"
    return [
        [
            processor.build_user_message(
                text=full_text,
                reference=reference_audio_codes,
            ),
            processor.build_assistant_message(audio_codes_list=[prompt_audio]),
        ],
    ]


def synthesize_podcast(
    text_to_generate: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    conversations = build_conversations(text_to_generate=text_to_generate)
    output_dir = Path(output_dir)
    output_path = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    batch = processor(conversations, mode="continuation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    generated_paths: list[Path] = []
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32000,
        )

        sample_idx = 0
        for message in processor.decode(outputs):
            for seg_idx, audio in enumerate(message.audio_codes_list):
                segment_path = output_dir / f"{sample_idx}_{seg_idx}.wav"
                sf.write(
                    segment_path,
                    audio.detach().cpu().to(torch.float32).numpy(),
                    int(processor.model_config.sampling_rate),
                )
                generated_paths.append(segment_path)
            sample_idx += 1

    if generated_paths and output_path != generated_paths[0]:
        generated_paths[0].replace(output_path)
        generated_paths[0] = output_path

    return {
        "output_path": str(output_path),
        "segment_paths": [str(path) for path in generated_paths],
    }


def main() -> None:
    result = synthesize_podcast(DEFAULT_TEXT_TO_GENERATE)
    print(result["output_path"])


if __name__ == "__main__":
    main()
