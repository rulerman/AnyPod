"""前 3 步（Parse → Understand → BookCharter）产物的跨 output_dir 缓存。

缓存判定规则：文件名相同 + 文件大小相同 → 视为同一文件。
缓存目录：{项目根目录}/cache/{key}/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bookcast.core.runtime_log import console_print

print = console_print

# 项目根目录（AnyPod/）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "cache"

# 需要缓存的文件和目录（相对于 output_dir）
CACHED_ITEMS: list[str] = [
    "book_structure.json",
    "chunk_source_map.jsonl",
    "full_text.txt",
    "input",
    "raw_text",
    "chunk_cards",
    "section_cards",
    "book_charter.json",
    "llm_raw/chunk_cards",
    "llm_raw/section_cards",
    "llm_raw/book_charter",
]


def compute_cache_key(input_path: Path) -> str:
    """根据文件名和文件大小生成缓存 key。"""
    resolved = Path(input_path).resolve()
    size = resolved.stat().st_size
    return f"{resolved.name}_{size}"


def find_cache(input_path: Path) -> Path | None:
    """检查缓存是否存在且完整（book_charter.json 存在），返回缓存目录路径或 None。"""
    key = compute_cache_key(input_path)
    cache_path = CACHE_DIR / key
    if cache_path.is_dir() and (cache_path / "book_charter.json").is_file():
        return cache_path
    return None


def restore_from_cache(cache_dir: Path, output_dir: Path) -> None:
    """将缓存内容复制到 output_dir。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for item_name in CACHED_ITEMS:
        src = cache_dir / item_name
        dst = output_dir / item_name
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"  Cache restored from: {cache_dir}")


def save_to_cache(input_path: Path, output_dir: Path) -> None:
    """将 output_dir 中前 3 步的产物复制到缓存目录。"""
    key = compute_cache_key(input_path)
    cache_path = CACHE_DIR / key
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    for item_name in CACHED_ITEMS:
        src = output_dir / item_name
        dst = cache_path / item_name
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"  Cache saved to: {cache_path}")
