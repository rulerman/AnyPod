from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import requests

from .common import save_text


UNDERSTANDING_AGENT_NAME = "understanding_agent"
PLAN_AGENT_NAME = "plan_agent"
WRITING_AGENT_NAME = "writing_agent"
DEFAULT_AGENT_NAME = UNDERSTANDING_AGENT_NAME

REQUIRED_LLM_CONFIG_KEYS = (
    "base_url",
    "model",
    "api_key",
)

OPTIONAL_LLM_CONFIG_DEFAULTS = {
    "timeout": 600,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "system_prompt": None,
    "api_key_env": "",
    "extra_headers": {},
    "extra_body": {},
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "llm_api_config.json"


def extract_text_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "".join(text_parts)

    return str(message_content)


def extract_json_candidate_text(text: str) -> str:
    marker = "</think>"
    if marker in text:
        text = text.split(marker, 1)[1]

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return cleaned


def extract_text_candidate_text(text: str) -> str:
    marker = "</think>"
    if marker in text:
        text = text.split(marker, 1)[1]

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return cleaned


def parse_json_object_from_text(
    text: str,
    required_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not normalize_text(text):
        raise ValueError("Model output is empty")

    decoder = json.JSONDecoder()
    candidate_sources = [text]

    think_cleaned = extract_json_candidate_text(text)
    if think_cleaned and think_cleaned != text:
        candidate_sources.insert(0, think_cleaned)

    first_object: dict[str, Any] | None = None
    found_json_object = False
    for candidate_text in candidate_sources:
        for start_index, char in enumerate(candidate_text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate_text[start_index:])
            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            found_json_object = True
            if first_object is None:
                first_object = parsed

            if required_schema:
                try:
                    validate_required_json_keys(parsed, required_schema)
                except ValueError:
                    continue
            return parsed

    if required_schema:
        if found_json_object:
            raise ValueError("Model output contained JSON objects, but none matched the required schema")
        raise ValueError("No valid JSON object was found in the model output")

    if first_object is not None:
        return first_object
    raise ValueError("No valid JSON object was found in the model output")


def collect_missing_required_keys(data: Any, required_schema: dict[str, Any], prefix: str = "") -> list[str]:
    if not isinstance(data, dict):
        return [prefix or "<root>"]

    missing: list[str] = []
    for key, child_schema in required_schema.items():
        key_path = f"{prefix}.{key}" if prefix else key
        if key not in data:
            missing.append(key_path)
            continue

        value = data[key]
        if isinstance(child_schema, dict):
            if not isinstance(value, dict):
                missing.append(key_path)
                continue
            missing.extend(collect_missing_required_keys(value, child_schema, key_path))
            continue

        if isinstance(child_schema, list):
            if not isinstance(value, list):
                missing.append(key_path)
                continue
            if not child_schema or not value:
                continue

            item_schema = child_schema[0]
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    item_path = f"{key_path}[{index}]"
                    if not isinstance(item, dict):
                        missing.append(item_path)
                        continue
                    missing.extend(collect_missing_required_keys(item, item_schema, item_path))

    return missing


def validate_required_json_keys(data: dict[str, Any], required_schema: dict[str, Any]) -> None:
    missing = collect_missing_required_keys(data, required_schema)
    if not missing:
        return

    preview = ", ".join(missing[:20])
    if len(missing) > 20:
        preview = f"{preview} ..."
    raise ValueError(f"Model output is missing required keys: {preview}")


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_base_url(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    suffix = "/chat/completions"
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    return text.rstrip("/")


def build_chat_completions_url(value: Any) -> str:
    base_url = normalize_base_url(value)
    if not base_url:
        return ""
    return f"{base_url}/chat/completions"


def ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def ensure_string_dict(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        text_key = normalize_text(key)
        if not text_key:
            continue
        result[text_key] = str(item)
    return result


def ensure_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def ensure_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def ensure_positive_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return max(1, int(value))


def validate_required_config_keys(config_data: dict[str, Any], config_path: Path, agent_name: str) -> None:
    missing = [key for key in REQUIRED_LLM_CONFIG_KEYS if key not in config_data]
    if missing:
        raise ValueError(
            f"LLM config {agent_name}[0] is missing required fields: {', '.join(missing)}. File: {config_path}"
        )


def extract_agent_config(raw_config: dict[str, Any], agent_name: str, config_path: Path) -> dict[str, Any]:
    if agent_name not in raw_config:
        raise ValueError(f"LLM config is missing agent: {agent_name}. File: {config_path}")

    agent_items = raw_config[agent_name]
    if not isinstance(agent_items, list) or not agent_items:
        raise ValueError(f"LLM config field {agent_name} must be a non-empty array. File: {config_path}")

    agent_config = agent_items[0]
    if not isinstance(agent_config, dict):
        raise ValueError(f"LLM config field {agent_name}[0] must be a JSON object. File: {config_path}")

    return agent_config


def load_llm_api_config(
    agent_name: str = DEFAULT_AGENT_NAME,
    config_path: Path | None = None,
) -> dict[str, Any]:
    resolved_config_path = config_path or DEFAULT_CONFIG_PATH
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Missing LLM config file: {resolved_config_path}")

    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"LLM config must be a JSON object: {resolved_config_path}")
    loaded = extract_agent_config(raw_config, agent_name, resolved_config_path)
    validate_required_config_keys(loaded, resolved_config_path, agent_name)

    merged = copy.deepcopy(OPTIONAL_LLM_CONFIG_DEFAULTS)
    merged.update(loaded)

    configured_base_url = normalize_base_url(loaded.get("base_url"))
    if not configured_base_url:
        raise ValueError(f"LLM config field base_url cannot be empty: {resolved_config_path}")

    model = normalize_text(loaded.get("model"))
    if not model:
        raise ValueError(f"LLM config field model cannot be empty: {resolved_config_path}")

    api_key = normalize_text(loaded.get("api_key"))
    api_key_env = normalize_text(loaded.get("api_key_env"))
    if not api_key and api_key_env:
        api_key = normalize_text(os.getenv(api_key_env))

    merged["base_url"] = configured_base_url
    merged["url"] = build_chat_completions_url(configured_base_url)
    merged["model"] = model
    merged["api_key"] = api_key
    merged["api_key_env"] = api_key_env
    merged["timeout"] = ensure_positive_int(
        merged.get("timeout"),
        int(OPTIONAL_LLM_CONFIG_DEFAULTS["timeout"]),
    )
    merged["temperature"] = ensure_optional_float(merged.get("temperature"))
    merged["top_p"] = ensure_optional_float(merged.get("top_p"))
    merged["top_k"] = ensure_optional_int(merged.get("top_k"))
    merged["system_prompt"] = normalize_text(merged.get("system_prompt")) or None
    merged["extra_headers"] = ensure_string_dict(merged.get("extra_headers"))
    merged["extra_body"] = ensure_dict(merged.get("extra_body"))
    return merged


DEFAULT_LLM_CONFIG = load_llm_api_config(DEFAULT_AGENT_NAME)
DEFAULT_BASE_URL = str(DEFAULT_LLM_CONFIG["base_url"])
DEFAULT_URL = str(DEFAULT_LLM_CONFIG["url"])
DEFAULT_MODEL = str(DEFAULT_LLM_CONFIG["model"])
DEFAULT_TIMEOUT = int(DEFAULT_LLM_CONFIG["timeout"])
DEFAULT_TEMPERATURE = ensure_optional_float(DEFAULT_LLM_CONFIG.get("temperature"))
DEFAULT_TOP_P = ensure_optional_float(DEFAULT_LLM_CONFIG.get("top_p"))
DEFAULT_TOP_K = ensure_optional_int(DEFAULT_LLM_CONFIG.get("top_k"))
DEFAULT_SYSTEM_PROMPT = DEFAULT_LLM_CONFIG.get("system_prompt")


def build_request_output_path(raw_output_path: Path) -> Path:
    return raw_output_path.with_name(f"{raw_output_path.stem}_request.json")


def mask_headers(headers: dict[str, str]) -> dict[str, str]:
    masked: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "authorization" and value:
            masked[key] = "Bearer ***"
        else:
            masked[key] = value
    return masked


def save_request_payload(
    raw_output_path: Path,
    url: str,
    base_url: str,
    timeout: int,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> None:
    request_record = {
        "url": url,
        "base_url": base_url,
        "timeout": timeout,
        "headers": mask_headers(headers),
        "payload": payload,
    }
    save_text(
        build_request_output_path(raw_output_path),
        json.dumps(request_record, ensure_ascii=False, indent=2),
    )


class SimpleLLMClient:
    def __init__(
        self,
        agent_name: str = DEFAULT_AGENT_NAME,
        url: str | None = DEFAULT_URL,
        model: str | None = DEFAULT_MODEL,
        timeout: int | None = DEFAULT_TIMEOUT,
        temperature: float | None = DEFAULT_TEMPERATURE,
        top_p: float | None = DEFAULT_TOP_P,
        top_k: int | None = DEFAULT_TOP_K,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        config = load_llm_api_config(
            agent_name=agent_name,
            config_path=Path(config_path) if config_path else None,
        )

        resolved_base_url = normalize_base_url(base_url or url)
        if not resolved_base_url:
            resolved_base_url = str(config["base_url"])

        resolved_api_key = normalize_text(api_key)
        if not resolved_api_key:
            resolved_api_key = str(config["api_key"])

        self.base_url = resolved_base_url
        self.url = build_chat_completions_url(resolved_base_url)
        self.agent_name = agent_name
        self.model = normalize_text(model) or str(config["model"])
        self.api_key = resolved_api_key
        self.timeout = ensure_positive_int(timeout, int(config["timeout"]))
        self.temperature = temperature if temperature is not None else config.get("temperature")
        self.top_p = top_p if top_p is not None else config.get("top_p")
        self.top_k = top_k if top_k is not None else config.get("top_k")
        self.system_prompt = (
            normalize_text(system_prompt) or None
            if system_prompt is not None
            else config.get("system_prompt")
        )
        self.extra_headers = dict(config.get("extra_headers", {}))
        self.extra_headers.update(ensure_string_dict(extra_headers))
        self.extra_body = dict(config.get("extra_body", {}))
        self.extra_body.update(ensure_dict(extra_body))

    def to_runtime_config(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "system_prompt": self.system_prompt,
            "extra_headers": dict(self.extra_headers),
            "extra_body": dict(self.extra_body),
        }

    def build_messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        return messages

    def build_payload(self, prompt: str, system_prompt: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.build_messages(prompt, system_prompt),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        payload.update(self.extra_body)
        return payload

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def request_chat_completion(
        self,
        prompt: str,
        raw_output_path: Path,
        system_prompt: str | None,
    ) -> str:
        resolved_system_prompt = (
            normalize_text(system_prompt) or None
            if system_prompt is not None
            else self.system_prompt
        )
        payload = self.build_payload(prompt=prompt, system_prompt=resolved_system_prompt)
        headers = self.build_headers()
        save_request_payload(
            raw_output_path=raw_output_path,
            url=self.url,
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
            payload=payload,
        )

        response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        message_content = result["choices"][0]["message"]["content"]
        raw_text = extract_text_content(message_content)
        save_text(raw_output_path, raw_text)
        return raw_text

    def generate_json(
        self,
        prompt: str,
        raw_output_path: Path,
        system_prompt: str | None = None,
        required_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw_text = self.request_chat_completion(
            prompt=prompt,
            raw_output_path=raw_output_path,
            system_prompt=system_prompt,
        )

        try:
            parsed = parse_json_object_from_text(raw_text, required_schema=required_schema)
        except Exception as exc:
            raise ValueError(f"Failed to parse model output JSON. Raw output: {raw_output_path}") from exc
        if required_schema:
            validate_required_json_keys(parsed, required_schema)
        return parsed

    def generate_text(
        self,
        prompt: str,
        raw_output_path: Path,
        system_prompt: str | None = None,
    ) -> str:
        raw_text = self.request_chat_completion(
            prompt=prompt,
            raw_output_path=raw_output_path,
            system_prompt=system_prompt,
        )

        cleaned = extract_text_candidate_text(raw_text)
        if not cleaned:
            raise ValueError(f"Model output text is empty. Raw output: {raw_output_path}")
        return cleaned
