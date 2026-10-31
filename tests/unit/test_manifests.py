from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from speechinsight.data import librispeech as libri
from speechinsight.data.ami_splits import DEV, TEST, TRAIN, split_for
from speechinsight.data.manifests import Manifest, UtteranceEntry, read_manifest, write_manifest


def _make_librispeech(root: Path, split: str, speaker: str, chapter: str) -> None:
    d = root / split / speaker / chapter
    d.mkdir(parents=True)
    lines = []
    for i in range(2):
        uid = f"{speaker}-{chapter}-{i:04d}"
        sf.write(str(d / f"{uid}.flac"), np.zeros(1600, dtype=np.float32), 16_000)
        lines.append(f"{uid} HELLO WORLD")
    (d / f"{speaker}-{chapter}.trans.txt").write_text("\n".join(lines), encoding="utf-8")


def test_librispeech_manifest_uses_speaker_dir(tmp_path: Path) -> None:
    _make_librispeech(tmp_path, "dev-clean", "84", "121123")
    manifest = libri.build_manifest(tmp_path / "dev-clean", "dev-clean")
    assert len(manifest.entries) == 2
    assert manifest.speakers == ("84",)
    assert manifest.entries[0].text == "HELLO WORLD"


def test_librispeech_dev_test_speakers_disjoint(tmp_path: Path) -> None:
    _make_librispeech(tmp_path, "dev-clean", "84", "121123")
    _make_librispeech(tmp_path, "test-clean", "1089", "134686")
    dev = libri.build_manifest(tmp_path / "dev-clean", "dev-clean")
    test = libri.build_manifest(tmp_path / "test-clean", "test-clean")
    assert set(dev.speakers).isdisjoint(test.speakers)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = Manifest(
        corpus="librispeech",
        split="dev-clean",
        entries=(
            UtteranceEntry("u1", "a.flac", 1.0, "hi", "84"),
            UtteranceEntry("u2", "b.flac", 2.0, "bye", "84"),
        ),
    )
    path = tmp_path / "m.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


def test_ami_splits_are_disjoint_and_match_official_counts() -> None:
    assert len(TRAIN) == 136
    assert len(DEV) == 18
    assert len(TEST) == 16
    assert set(TRAIN).isdisjoint(DEV)
    assert set(TRAIN).isdisjoint(TEST)
    assert set(DEV).isdisjoint(TEST)
    assert split_for("ES2011a") == "dev"
    assert split_for("ES2004a") == "test"
    assert split_for("unknown") is None
