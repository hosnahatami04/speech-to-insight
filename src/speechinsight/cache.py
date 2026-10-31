from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def cache_key(audio_hash: str, stage: str, model: str, params: dict[str, Any]) -> str:
    """Deterministic key over the inputs that fully determine a stage's output.
    Any change to audio, stage, model or params yields a different key so stale
    results are never reused."""

    payload = json.dumps(
        {"audio": audio_hash, "stage": stage, "model": model, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audio_hash(samples: np.ndarray) -> str:
    """Content hash of a sample array, stable across runs. Rounds to int16 scale
    so bit-level float noise does not change the key while audible content does."""

    quantized = np.ascontiguousarray(np.round(samples * 32768.0).astype(np.int32))
    return hashlib.sha256(quantized.tobytes()).hexdigest()


class StageCache:
    """Content-addressed cache. A key maps to exactly one JSON file; the same key
    always returns the same file and any input change produces a different key."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> Any:
        with self._path(key).open(encoding="utf-8") as handle:
            return json.load(handle)

    def store(self, key: str, value: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        tmp.replace(path)
