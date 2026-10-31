from __future__ import annotations

from pathlib import Path

import numpy as np

from speechinsight.cache import StageCache, audio_hash, cache_key


def test_same_key_same_file(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    key = cache_key("h", "asr", "base.en", {"beam": 5})
    cache.store(key, {"value": 1})
    assert cache.has(key)
    assert cache.load(key) == {"value": 1}


def test_param_change_misses(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    key_a = cache_key("h", "asr", "base.en", {"beam": 5})
    key_b = cache_key("h", "asr", "base.en", {"beam": 1})
    cache.store(key_a, {"value": 1})
    assert not cache.has(key_b)


def test_audio_hash_is_stable_and_content_sensitive() -> None:
    a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    c = np.array([0.1, 0.2, 0.4], dtype=np.float32)
    assert audio_hash(a) == audio_hash(b)
    assert audio_hash(a) != audio_hash(c)
