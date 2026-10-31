from __future__ import annotations

from speechinsight.asr.normalize import NORMALIZER_NAME, normalize


def test_lowercases_and_strips_punctuation() -> None:
    assert normalize("Hello, World!") == "hello world"


def test_keeps_apostrophe_inside_words() -> None:
    assert normalize("don't") == "don't"


def test_unifies_curly_apostrophe() -> None:
    assert normalize("don" + chr(0x2019) + "t") == "don't"


def test_collapses_whitespace() -> None:
    assert normalize("a   b\tc\n d") == "a b c d"


def test_expands_abbreviations_and_small_numbers() -> None:
    assert normalize("Mr Smith had 3 cats") == "mister smith had three cats"


def test_leaves_larger_numbers_as_written() -> None:
    assert normalize("room 42") == "room 42"


def test_normalizer_name_is_stable() -> None:
    assert NORMALIZER_NAME == "speechinsight-v1"
