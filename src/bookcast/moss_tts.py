from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bookcast.core.voice_library import resolve_default_voice_prompt
from bookcast.moss_processor_patch import patch_moss_processor_audio_loading


DEFAULT_MODEL_PATH = str(SRC_DIR.parent / "model" / "MOSS-TTS")
DEFAULT_AUDIO_TOKENIZER_PATH = str(SRC_DIR.parent / "model" / "MOSS-Audio-Tokenizer")
TTS_BACKEND = "moss-tts"
TTS_MODE = "moss_tts_voice_cloning"

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


def resolve_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# ── cuDNN SDPA 设置 ──
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

PRETRAINED_MODEL_NAME_OR_PATH = env_existing_path("ANYPOD_TTS_MODEL_PATH", DEFAULT_MODEL_PATH)
AUDIO_TOKENIZER_NAME_OR_PATH = env_existing_path("ANYPOD_TTS_AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH)
DEFAULT_PROMPT_AUDIO_SPEAKER1, _ = resolve_default_voice_prompt(1)
PROMPT_AUDIO_SPEAKER1 = env_existing_path("ANYPOD_TTS_PROMPT_AUDIO_S1", DEFAULT_PROMPT_AUDIO_SPEAKER1)

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

attn_implementation = env_text("ANYPOD_TTS_ATTN_IMPLEMENTATION", resolve_attn_implementation())
model = AutoModel.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()


def synthesize_podcast(
    text_to_generate: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    if "[S2]" in text_to_generate:
        raise ValueError("moss-tts 后端仅支持单人模式，脚本中不能包含 [S2] 标签")

    # 去除 [S1] 标签，保留纯文本
    text = re.sub(r"\[S1\]", "", text_to_generate).strip()

    ref_audio_path = PROMPT_AUDIO_SPEAKER1
    conversations = [
        processor.build_user_message(text=text, reference=[ref_audio_path]),
    ]

    output_dir = Path(output_dir)
    output_path = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    batch = processor([conversations], mode="generation")
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
    result = synthesize_podcast("[S1]你好，这是一段测试语音。")
    print(result["output_path"])


if __name__ == "__main__":
    main()
