from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speechinsight.data.build import build_all

_RTTM = "SPEAKER ES2011a 1 0.00 10.00 <NA> <NA> A <NA> <NA>\n"
_SUMMARY = (
    '<?xml version="1.0"?>\n'
    '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
    '<decisions nite:id="d"><sentence nite:id="s1">Pick blue.</sentence></decisions>\n'
    '<actions nite:id="a"><sentence nite:id="s2">Ship it.</sentence></actions>\n'
    "</nite:root>\n"
)


def _make_librispeech(root: Path) -> None:
    d = root / "data" / "LibriSpeech" / "dev-clean" / "84" / "121123"
    d.mkdir(parents=True)
    uid = "84-121123-0000"
    sf.write(str(d / f"{uid}.flac"), np.zeros(1600, dtype=np.float32), 16_000)
    (d / "84-121123.trans.txt").write_text(f"{uid} HELLO\n", encoding="utf-8")


def _make_ami(root: Path) -> None:
    rttm = root / "data" / "ami" / "rttms"
    summ = root / "data" / "ami" / "abstractive"
    rttm.mkdir(parents=True)
    summ.mkdir(parents=True)
    (rttm / "ES2011a.rttm").write_text(_RTTM, encoding="utf-8")
    (summ / "ES2011a.abssumm.xml").write_text(_SUMMARY, encoding="utf-8")


def _cfg() -> dict:
    return {
        "librispeech": {"dev_clean": "data/LibriSpeech/dev-clean"},
        "accents": {},
        "ami": {
            "rttm_dir": "data/ami/rttms",
            "summary_dir": "data/ami/abstractive",
            "dev_meetings": ["ES2011a"],
            "test_meetings": [],
        },
    }


def test_build_all_writes_manifests_and_card(tmp_path: Path) -> None:
    _make_librispeech(tmp_path)
    _make_ami(tmp_path)
    reports = tmp_path / "reports"

    build_all(tmp_path, _cfg(), reports)

    assert (tmp_path / "artifacts" / "manifests" / "librispeech_dev-clean.json").is_file()
    card = (reports / "data_card.md").read_text(encoding="utf-8")
    assert "ES2011a" in card
    assert "Pick blue" not in card  # summary text is counted, not dumped verbatim
    assert "| dev | 1 |" in card


def test_build_all_missing_librispeech_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="LibriSpeech"):
        build_all(tmp_path, _cfg(), tmp_path / "reports")
