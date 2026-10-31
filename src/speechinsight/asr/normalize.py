from __future__ import annotations

import re

# Normalizer name reported alongside every WER so results are comparable only
# against numbers produced with the same normalization. See ADR 0004.
NORMALIZER_NAME = "speechinsight-v1"

# Apostrophe variants unified to a straight quote: left/right single quotation
# marks and the modifier-letter apostrophe. Built from code points so the source
# contains no ambiguous literals.
_APOSTROPHE_CHARS = "".join(chr(c) for c in (0x2018, 0x2019, 0x02BC)) + "`"
_APOSTROPHE = re.compile(f"[{re.escape(_APOSTROPHE_CHARS)}]")
_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")

# Spelled-out forms for the small integers that recognizers and references
# disagree on most. Documented and deliberately limited; larger numbers are left
# as written so the rule stays predictable.
_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}

# Abbreviations expanded to their spoken form.
_ABBREVIATIONS = {
    "mr": "mister",
    "mrs": "missus",
    "dr": "doctor",
    "st": "saint",
    "vs": "versus",
    "etc": "etcetera",
}


def normalize(text: str) -> str:
    """Normalize text for WER scoring.

    Rules, in order:
    1. Lowercase.
    2. Unify apostrophe variants to a straight quote.
    3. Remove punctuation except the apostrophe inside words (so "don't" stays).
    4. Expand the documented abbreviations and single-digit numbers.
    5. Collapse whitespace.
    """

    text = text.lower()
    text = _APOSTROPHE.sub("'", text)
    text = _PUNCT.sub(" ", text)
    tokens = []
    for token in _WS.sub(" ", text).strip().split():
        token = token.strip("'")
        if not token:
            continue
        token = _ABBREVIATIONS.get(token, token)
        token = _NUMBER_WORDS.get(token, token)
        tokens.append(token)
    return " ".join(tokens)
