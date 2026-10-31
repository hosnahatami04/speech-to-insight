from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from speechinsight.data.accents import build_manifest


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("a.wav", "b.wav"):
        sf.write(str(clips / name), np.zeros(1600, dtype=np.float32), 16_000)
    tsv = tmp_path / "validated.tsv"
    tsv.write_text(
        "client_id\tpath\tsentence\taccents\n"
        "spk1\ta.wav\tHello there\tIndian\n"
        "spk2\tb.wav\tNo accent here\t\n"  # no accent -> skipped
        "spk3\tmissing.wav\tMissing clip\tScottish\n",  # missing clip -> skipped
        encoding="utf-8",
    )
    return tsv, clips


def test_build_manifest_keeps_only_labelled_present_clips(tmp_path: Path) -> None:
    tsv, clips = _setup(tmp_path)
    manifest = build_manifest(tsv, clips, "dev")
    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.accent == "Indian"
    assert entry.text == "Hello there"
    assert entry.speaker == "spk1"
