from __future__ import annotations

from pathlib import Path

from speechinsight.data.ami import AmiMeeting, SummaryItem
from speechinsight.data.data_card import render_data_card, write_data_card
from speechinsight.data.manifests import Manifest, UtteranceEntry
from speechinsight.types import Segment


def _ami() -> dict[str, list[AmiMeeting]]:
    meeting = AmiMeeting(
        meeting_id="ES2011a",
        split="dev",
        segments=(Segment("A", 0.0, 60.0), Segment("B", 60.0, 120.0)),
        decisions=(SummaryItem("decision", "Use a yellow case."),),
        actions=(SummaryItem("action", "Build a prototype."),),
    )
    return {"dev": [meeting]}


def test_render_includes_real_counts_and_notes() -> None:
    libri = {
        "dev-clean": Manifest(
            "librispeech",
            "dev-clean",
            (UtteranceEntry("u1", "a.flac", 120.0, "hi", "84"),),
        )
    }
    card = render_data_card(libri, None, _ami())
    assert "scenario meetings are acted" in card
    assert "ES2011a" in card
    assert "Not yet populated" in card  # accents absent
    assert "| dev | 1 |" in card  # one AMI dev meeting


def test_accent_group_under_100_marked_report_only() -> None:
    accents = Manifest(
        "accents",
        "dev",
        tuple(UtteranceEntry(f"u{i}", f"{i}.wav", 1.0, "x", f"s{i}", "Welsh") for i in range(5)),
    )
    card = render_data_card({}, accents, {})
    assert "report only, <100" in card


def test_write_data_card(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "data_card.md"
    write_data_card("# Data card\n", out)
    assert out.read_text(encoding="utf-8") == "# Data card\n"
