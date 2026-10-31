from __future__ import annotations

from collections import Counter
from pathlib import Path

from speechinsight.data.ami import AmiMeeting
from speechinsight.data.manifests import Manifest


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.1f} min"


def render_data_card(
    librispeech: dict[str, Manifest],
    accents: Manifest | None,
    ami: dict[str, list[AmiMeeting]],
) -> str:
    """Render the data card from real manifest and meeting statistics. Every
    number here is computed from the built manifests, never hand-entered."""

    lines: list[str] = ["# Data card", ""]

    lines.append("## LibriSpeech (clean read speech)")
    lines.append("")
    lines.append("Source: https://www.openslr.org/12 · License: CC BY 4.0")
    lines.append("")
    lines.append("| split | files | total duration | speakers |")
    lines.append("| --- | --- | --- | --- |")
    for split, manifest in sorted(librispeech.items()):
        lines.append(
            f"| {split} | {len(manifest.entries)} | "
            f"{_fmt_minutes(manifest.total_duration)} | {len(manifest.speakers)} |"
        )
    lines.append("")

    lines.append("## Accented English")
    lines.append("")
    if accents is None or not accents.entries:
        lines.append(
            "Not yet populated. Requires manual download under the publisher's "
            "terms; see the repository data instructions. Per-accent WER is "
            "reported only once the clips are in place."
        )
    else:
        by_accent = Counter(e.accent for e in accents.entries)
        lines.append(f"Files: {len(accents.entries)} · speakers: {len(accents.speakers)}")
        lines.append("")
        lines.append("| accent | files |")
        lines.append("| --- | --- |")
        for accent, count in sorted(by_accent.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            note = " (report only, <100)" if count < 100 else ""
            lines.append(f"| {accent} | {count}{note} |")
    lines.append("")

    lines.append("## AMI Meeting Corpus (meeting speech)")
    lines.append("")
    lines.append("Source: https://groups.inf.ed.ac.uk/ami/corpus/ · License: CC BY 4.0")
    lines.append("")
    lines.append(
        "Known limitation: AMI scenario meetings are acted — participants play "
        "roles on a product-design team, not a real meeting."
    )
    lines.append("")
    lines.append("| split | meetings | total duration | decisions | actions |")
    lines.append("| --- | --- | --- | --- | --- |")
    for split, meetings in sorted(ami.items()):
        dur = sum(m.duration for m in meetings)
        dec = sum(len(m.decisions) for m in meetings)
        act = sum(len(m.actions) for m in meetings)
        lines.append(f"| {split} | {len(meetings)} | {_fmt_minutes(dur)} | {dec} | {act} |")
    lines.append("")

    lines.append("### AMI meetings (selected subset)")
    lines.append("")
    lines.append("| meeting | split | speakers | duration | decisions | actions |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for split, meetings in sorted(ami.items()):
        for m in meetings:
            lines.append(
                f"| {m.meeting_id} | {split} | {len(m.speakers)} | "
                f"{_fmt_minutes(m.duration)} | {len(m.decisions)} | {len(m.actions)} |"
            )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_data_card(content: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
