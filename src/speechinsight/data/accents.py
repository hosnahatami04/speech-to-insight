from __future__ import annotations

import csv
from pathlib import Path

import soundfile as sf

from speechinsight.data.manifests import Manifest, UtteranceEntry


def build_manifest(tsv_path: str | Path, clips_dir: str | Path, split: str) -> Manifest:
    """Build an accented-speech manifest from a Common Voice-style TSV plus its
    clips directory. The TSV must have ``path``, ``sentence``, ``accents`` and
    ``client_id`` columns. Rows without an accent label or a readable clip are
    skipped, so only labelled, decodable utterances enter the manifest."""

    tsv_path = Path(tsv_path)
    clips_dir = Path(clips_dir)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"accent TSV not found: {tsv_path}")

    entries: list[UtteranceEntry] = []
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accent = (row.get("accents") or row.get("accent") or "").strip()
            if not accent:
                continue
            clip = clips_dir / row["path"]
            if not clip.is_file():
                continue
            info = sf.info(str(clip))
            entries.append(
                UtteranceEntry(
                    id=Path(row["path"]).stem,
                    path=str(clip),
                    duration=info.frames / info.samplerate,
                    text=row["sentence"],
                    speaker=row.get("client_id", "unknown"),
                    accent=accent,
                )
            )
    return Manifest(corpus="accents", split=split, entries=tuple(entries))
