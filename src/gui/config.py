"""
Thin wrapper around ~/.3p-admm/config.json.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path.home() / ".3p-admm" / "config.json"

_cache: dict = {}


def load() -> dict:
    """Load config from disk. Returns {} if file is missing or corrupt."""
    global _cache
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _cache = {}
    return _cache


def save(data: dict) -> None:
    """Atomically write data to config file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CONFIG_PATH)


def get(key: str, default=None):
    """Get a value from the in-memory cache (call load() first)."""
    return _cache.get(key, default)


def set(key: str, value) -> None:
    """Set a value and immediately persist to disk."""
    _cache[key] = value
    save(_cache)
