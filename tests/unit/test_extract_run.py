from __future__ import annotations

from speechinsight.extract.run import extract, load_prompt_template
from speechinsight.types import Transcript, Utterance


class _FakeClient:
    """Returns a canned response per chunk invocation."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_json(self, prompt: str) -> dict:
        self.calls += 1
        if not self._responses:
            raise RuntimeError("no more canned responses")
        return self._responses.pop(0)


def _ratio(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def _transcript() -> Transcript:
    return Transcript(
        utterances=(
            Utterance("u0001", "A", 0.0, 1.0, "We'll pick blue."),
            Utterance("u0002", "B", 1.0, 2.0, "Build prototype by Friday."),
        )
    )


def test_extract_runs_end_to_end_with_fake_client(tmp_path) -> None:
    prompt = "Transcript:\n{transcript}\n"
    chunk_response = {
        "decisions": [{"text": "Pick blue casing.", "evidence": ["u0001"]}],
        "actions": [{"text": "Build prototype.", "owner": "B", "evidence": ["u0002"]}],
    }
    client = _FakeClient([chunk_response])

    report = extract(_transcript(), client, prompt, _ratio)

    assert report.schema_errors == 0
    assert len(report.decisions.kept) == 1
    assert len(report.actions.kept) == 1
    assert report.actions.kept[0].item.owner == "B"
    assert report.prompt_version == "v1"


def test_schema_error_is_counted_not_fatal() -> None:
    prompt = "{transcript}"

    class _BrokenClient:
        def generate_json(self, prompt: str) -> dict:
            return {"decisions": "not a list", "actions": []}  # schema violation

    report = extract(_transcript(), _BrokenClient(), prompt, _ratio)
    assert report.schema_errors == 1
    assert report.decisions.kept == []


def test_item_without_valid_evidence_is_dropped() -> None:
    prompt = "{transcript}"
    bad = {
        "decisions": [{"text": "Floating item", "evidence": ["u9999"]}],
        "actions": [],
    }
    client = _FakeClient([bad])
    report = extract(_transcript(), client, prompt, _ratio)
    assert report.decisions.dropped_no_evidence == 1
    assert report.decisions.kept == []


def test_load_prompt_template(tmp_path) -> None:
    p = tmp_path / "prompt.txt"
    p.write_text("hello {transcript}", encoding="utf-8")
    assert load_prompt_template(p) == "hello {transcript}"


def test_prompt_with_literal_json_braces_substitutes_correctly() -> None:
    """The real production prompt (``prompts/extract_v1.txt``) shows a JSON
    example containing ``{"text": ...}``. Naive ``str.format`` interprets that
    as a format field and raises KeyError. The runner must substitute the
    ``{transcript}`` placeholder without parsing other braces."""

    prompt = (
        'Example of a valid item:\n'
        '{"text": "demo", "owner": null, "due": null, "evidence": ["u0001"]}\n'
        "Transcript:\n{transcript}\n"
    )
    captured: dict[str, str] = {}

    class _Capture:
        def generate_json(self, prompt: str) -> dict:
            captured["prompt"] = prompt
            return {"decisions": [], "actions": []}

    report = extract(_transcript(), _Capture(), prompt, _ratio)
    assert report.schema_errors == 0
    # The template body is kept; only {transcript} is replaced.
    assert '{"text": "demo"' in captured["prompt"]
    assert "u0001" in captured["prompt"]
    assert "We'll pick blue." in captured["prompt"]
