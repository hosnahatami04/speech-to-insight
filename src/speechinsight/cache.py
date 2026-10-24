from __future__ import annotations

import hashlib
import json
from typing import Any


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
