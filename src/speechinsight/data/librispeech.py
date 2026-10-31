from __future__ import annotations

from pathlib import Path

import soundfile as sf

from speechinsight.data.manifests import Manifest, UtteranceEntry


def _read_transcripts(trans_file: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with trans_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            mapping[utt_id] = text
    return mapping


def build_manifest(root: str | Path, split: str) -> Manifest:
    """Scan a LibriSpeech split directory (e.g. ``LibriSpeech/dev-clean``) and
    build a manifest. The speaker id is the LibriSpeech speaker directory, which
    keeps the dev/test speaker sets disjoint by construction."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"LibriSpeech split directory not found: {root}")

    entries: list[UtteranceEntry] = []
    for flac in sorted(root.rglob("*.flac")):
        speaker = flac.parent.parent.name
        chapter = flac.parent.name
        trans_file = flac.parent / f"{speaker}-{chapter}.trans.txt"
        transcripts = _read_transcripts(trans_file)
        text = transcripts.get(flac.stem)
        if text is None:
            raise ValueError(f"no transcript for {flac.name} in {trans_file}")
        info = sf.info(str(flac))
        entries.append(
            UtteranceEntry(
                id=flac.stem,
                path=str(flac),
                duration=info.frames / info.samplerate,
                text=text,
                speaker=speaker,
            )
        )
    return Manifest(corpus="librispeech", split=split, entries=tuple(entries))
