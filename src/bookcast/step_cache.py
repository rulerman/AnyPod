"""Cross-output_dir cache for artifacts from Steps 1–3 (Parse → Understand → BookCharter).

Cache key: same filename + same file size → treated as the same input.
Cache location: {project_root}/cache/{key}/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bookcast.core.runtime_log import console_print

print = console_print

# Project root (AnyPod/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "cache"

# Files and directories to cache (relative to output_dir)
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
    """Build a cache key from filename and file size."""
    resolved = Path(input_path).resolve()
    size = resolved.stat().st_size
    return f"{resolved.name}_{size}"


def find_cache(input_path: Path) -> Path | None:
    """Return the cache dir if it exists and looks complete (book_charter.json exists)."""
    key = compute_cache_key(input_path)
    cache_path = CACHE_DIR / key
    if cache_path.is_dir() and (cache_path / "book_charter.json").is_file():
        return cache_path
    return None


def restore_from_cache(cache_dir: Path, output_dir: Path) -> None:
    """Copy cached artifacts into output_dir."""
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
    """Copy Step 1–3 artifacts from output_dir into the cache directory."""
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
