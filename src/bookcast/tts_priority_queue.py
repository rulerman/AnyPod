from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import fcntl


QUEUE_FILE_NAME = ".anypod_tts_priority_queue.json"


def get_tts_priority_queue_path(output_dir: Path) -> Path:
    return output_dir.resolve() / QUEUE_FILE_NAME


@contextmanager
def _locked_queue_file(queue_path: Path) -> Iterator[TextIO]:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_queue(handle: TextIO) -> list[str]:
    handle.seek(0)
    raw_text = handle.read().strip()
    if not raw_text:
        return []
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    queue: list[str] = []
    for item in data:
        text = str(item).strip()
        if text:
            queue.append(text)
    return queue


def _write_queue(handle: TextIO, episode_ids: list[str]) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(episode_ids, ensure_ascii=False, indent=2))
    handle.flush()


def clear_tts_priority_queue(output_dir: Path) -> None:
    queue_path = get_tts_priority_queue_path(output_dir)
    with _locked_queue_file(queue_path) as handle:
        _write_queue(handle, [])


def enqueue_tts_priority_episode(output_dir: Path, episode_id: str) -> bool:
    normalized_episode_id = str(episode_id).strip()
    if not normalized_episode_id:
        return False
    queue_path = get_tts_priority_queue_path(output_dir)
    with _locked_queue_file(queue_path) as handle:
        queue = _read_queue(handle)
        if normalized_episode_id in queue:
            return False
        queue.append(normalized_episode_id)
        _write_queue(handle, queue)
        return True


def pop_tts_priority_episode(output_dir: Path) -> str | None:
    queue_path = get_tts_priority_queue_path(output_dir)
    with _locked_queue_file(queue_path) as handle:
        queue = _read_queue(handle)
        if not queue:
            return None
        episode_id = queue.pop(0)
        _write_queue(handle, queue)
        return episode_id
