from __future__ import annotations

from speechinsight.cache import cache_key


def test_same_inputs_same_key() -> None:
    a = cache_key("abc", "asr", "base.en", {"beam": 5})
    b = cache_key("abc", "asr", "base.en", {"beam": 5})
    assert a == b


def test_param_change_changes_key() -> None:
    a = cache_key("abc", "asr", "base.en", {"beam": 5})
    b = cache_key("abc", "asr", "base.en", {"beam": 1})
    assert a != b


def test_param_order_does_not_matter() -> None:
    a = cache_key("abc", "asr", "base.en", {"beam": 5, "temp": 0})
    b = cache_key("abc", "asr", "base.en", {"temp": 0, "beam": 5})
    assert a == b
