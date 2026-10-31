from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UtteranceEntry:
    """One single-speaker recording with a reference transcript. Used for the
    read-speech corpora (LibriSpeech, accented speech)."""

    id: str
    path: str
    duration: float
    text: str
    speaker: str
    accent: str | None = None


@dataclass(frozen=True, slots=True)
class Manifest:
    corpus: str
    split: str
    entries: tuple[UtteranceEntry, ...]

    @property
    def total_duration(self) -> float:
        return sum(e.duration for e in self.entries)

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(sorted({e.speaker for e in self.entries}))


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": manifest.corpus,
        "split": manifest.split,
        "entries": [asdict(e) for e in manifest.entries],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def read_manifest(path: str | Path) -> Manifest:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = tuple(UtteranceEntry(**item) for item in payload["entries"])
    return Manifest(corpus=payload["corpus"], split=payload["split"], entries=entries)
