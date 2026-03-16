from __future__ import annotations

import base64
import re
import sys
from pathlib import Path
from typing import Any

import requests

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bookcast.core.io import read_json


TTS_BACKEND = "moss-tts(api)"
TTS_MODE = "moss_tts_api_remote"

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR.parent.parent / "config" / "llm_api_config.json"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR.parent / "demo" / "output" / "tts"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "final_podcast.wav"

API_TIMEOUT = 600


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")
    config = read_json(CONFIG_PATH)
    if "moss_tts_api" not in config:
        raise KeyError("配置文件中缺少 'moss_tts_api' 字段，请在 config/llm_api_config.json 中添加")
    return config["moss_tts_api"]


def prepare_runtime() -> None:
    _load_config()


def get_runtime() -> tuple[None, None, dict[str, Any]]:
    return None, None, _load_config()


def synthesize_podcast(
    text_to_generate: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    config = _load_config()

    if "[S2]" in text_to_generate:
        raise ValueError("moss-tts(api) 后端仅支持单人模式，脚本中不能包含 [S2] 标签")

    # 去除 [S1] 标签，保留纯文本
    text = re.sub(r"\[S1\]", "", text_to_generate).strip()

    output_dir = Path(output_dir)
    output_path = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    base_url = config["base_url"]
    api_key = config["api_key"]
    if not api_key:
        raise ValueError("配置文件中 moss_tts_api.api_key 为空，请填入有效的 API Key")

    payload = {
        "model": config.get("model", "moss-tts"),
        "text": text,
        "voice_id": config["voice_id"],
        "meta_info": True,
        "sampling_params": {
            "max_new_tokens": config.get("max_new_tokens", 512),
            "temperature": config.get("temperature", 1.7),
            "top_p": config.get("top_p", 0.8),
            "top_k": config.get("top_k", 25),
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(base_url, json=payload, headers=headers, timeout=API_TIMEOUT)

    if response.status_code == 429:
        raise RuntimeError(
            f"moss-tts API 速率限制 (HTTP 429)，请稍后重试。响应: {response.text}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"moss-tts API 调用失败 (HTTP {response.status_code}): {response.text}"
        )

    result = response.json()

    if "code" in result and result["code"] != 0:
        raise RuntimeError(f"moss-tts API 返回错误 (code={result['code']}): {result}")

    audio_b64 = result.get("audio_data")
    if not audio_b64:
        raise RuntimeError(f"moss-tts API 返回中缺少 audio_data 字段: {result}")

    audio_bytes = base64.b64decode(audio_b64)
    output_path.write_bytes(audio_bytes)

    return {
        "output_path": str(output_path),
        "segment_paths": [str(output_path)],
    }
