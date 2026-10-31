from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from speechinsight.extract.client import OllamaClient, OllamaConfig, OllamaError


class _FakeResp:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *a: object) -> None:
        return None


def test_generate_json_parses_envelope_response() -> None:
    envelope = {"response": json.dumps({"decisions": [], "actions": []})}
    with patch("urllib.request.urlopen", return_value=_FakeResp(json.dumps(envelope))):
        client = OllamaClient(OllamaConfig(timeout_s=1.0, retries=0))
        result = client.generate_json("x")
    assert result == {"decisions": [], "actions": []}


def test_generate_json_retries_then_raises_on_bad_json() -> None:
    bad = _FakeResp("not json")
    with patch("urllib.request.urlopen", return_value=bad):
        client = OllamaClient(OllamaConfig(timeout_s=1.0, retries=1))
        with pytest.raises(OllamaError, match="failed after 2 attempts"):
            client.generate_json("x")


def test_generate_json_requires_json_object_not_array() -> None:
    envelope = {"response": json.dumps([1, 2, 3])}
    with patch("urllib.request.urlopen", return_value=_FakeResp(json.dumps(envelope))):
        client = OllamaClient(OllamaConfig(timeout_s=1.0, retries=0))
        with pytest.raises(OllamaError, match="JSON object"):
            client.generate_json("x")
