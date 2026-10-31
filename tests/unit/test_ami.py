from __future__ import annotations

from pathlib import Path

from speechinsight.data.ami import load_meeting, parse_rttm, parse_summary

_RTTM = """\
SPEAKER meet 1 0.00 2.00 <NA> <NA> A <NA> <NA>
SPEAKER meet 1 1.50 2.00 <NA> <NA> B <NA> <NA>
SPEAKER meet 1 4.00 1.00 <NA> <NA> A <NA> <NA>
"""

_SUMMARY = """\
<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
<abstract nite:id="a.1"><sentence nite:id="s.1">Intro.</sentence></abstract>
<decisions nite:id="d.1">
<sentence nite:id="s.2">The team will use a yellow case.</sentence>
<sentence nite:id="s.3">*NA*</sentence>
</decisions>
<actions nite:id="ac.1">
<sentence nite:id="s.4">The designer will build a prototype.</sentence>
</actions>
</nite:root>
"""


def test_parse_rttm_reads_overlapping_segments(tmp_path: Path) -> None:
    rttm = tmp_path / "meet.rttm"
    rttm.write_text(_RTTM, encoding="utf-8")
    segments = parse_rttm(rttm)
    assert len(segments) == 3
    # The first two overlap (0-2 and 1.5-3.5) and are kept as written.
    assert segments[0].speaker == "A"
    assert segments[1].start == 1.5
    assert segments[0].end == 2.0 and segments[1].start < segments[0].end


def test_parse_summary_filters_placeholders(tmp_path: Path) -> None:
    summary = tmp_path / "meet.abssumm.xml"
    summary.write_text(_SUMMARY, encoding="utf-8")
    decisions, actions = parse_summary(summary)
    assert [d.text for d in decisions] == ["The team will use a yellow case."]
    assert [a.text for a in actions] == ["The designer will build a prototype."]
    assert all(d.kind == "decision" for d in decisions)
    assert all(a.kind == "action" for a in actions)


def test_load_meeting_without_summary(tmp_path: Path) -> None:
    rttm = tmp_path / "meet.rttm"
    rttm.write_text(_RTTM, encoding="utf-8")
    meeting = load_meeting("meet", "dev", rttm, None)
    assert meeting.decisions == () and meeting.actions == ()
    assert set(meeting.speakers) == {"A", "B"}
    assert meeting.duration == 5.0
