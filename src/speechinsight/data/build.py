from __future__ import annotations

from pathlib import Path
from typing import Any

from speechinsight.data import accents as accents_mod
from speechinsight.data import ami as ami_mod
from speechinsight.data import librispeech as libri_mod
from speechinsight.data.ami import AmiMeeting
from speechinsight.data.data_card import render_data_card, write_data_card
from speechinsight.data.manifests import Manifest, write_manifest
from speechinsight.logging import get_logger

logger = get_logger("speechinsight.data.build")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_librispeech(root: Path, cfg: dict[str, Any], out_dir: Path) -> dict[str, Manifest]:
    manifests: dict[str, Manifest] = {}
    mapping = {"dev-clean": cfg.get("dev_clean"), "test-clean": cfg.get("test_clean")}
    for split, loc in mapping.items():
        if loc is None:
            continue
        split_dir = _resolve(root, loc)
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"LibriSpeech {split} not found at {split_dir}. "
                "Download it from https://www.openslr.org/12 and extract it there."
            )
        manifest = libri_mod.build_manifest(split_dir, split)
        write_manifest(manifest, out_dir / f"librispeech_{split}.json")
        manifests[split] = manifest
        logger.info(
            "built librispeech manifest",
            extra={"fields": {"split": split, "files": len(manifest.entries)}},
        )
    return manifests


def build_accents(root: Path, cfg: dict[str, Any], out_dir: Path) -> Manifest | None:
    tsv = cfg.get("tsv")
    clips = cfg.get("clips")
    if tsv is None or clips is None:
        return None
    tsv_path = _resolve(root, tsv)
    clips_dir = _resolve(root, clips)
    if not tsv_path.is_file():
        logger.warning(
            "accent corpus not present; skipping",
            extra={"fields": {"expected": str(tsv_path)}},
        )
        return None
    manifest = accents_mod.build_manifest(tsv_path, clips_dir, "dev")
    write_manifest(manifest, out_dir / "accents.json")
    return manifest


def build_ami(root: Path, cfg: dict[str, Any]) -> dict[str, list[AmiMeeting]]:
    rttm_dir = _resolve(root, cfg["rttm_dir"])
    summary_dir = _resolve(root, cfg["summary_dir"])
    out: dict[str, list[AmiMeeting]] = {}
    for split in ("dev", "test"):
        meetings: list[AmiMeeting] = []
        for meeting_id in cfg.get(f"{split}_meetings", []):
            rttm = rttm_dir / f"{meeting_id}.rttm"
            if not rttm.is_file():
                logger.warning(
                    "AMI rttm not present; skipping meeting",
                    extra={"fields": {"meeting": meeting_id, "expected": str(rttm)}},
                )
                continue
            summary = summary_dir / f"{meeting_id}.abssumm.xml"
            meetings.append(
                ami_mod.load_meeting(
                    meeting_id, split, rttm, summary if summary.is_file() else None
                )
            )
        if meetings:
            out[split] = meetings
    return out


def build_all(root: Path, data_cfg: dict[str, Any], reports_dir: Path) -> None:
    """Build every available manifest and write the data card. LibriSpeech is
    required; the accent and AMI corpora warn and are omitted when absent."""

    manifest_dir = root / "artifacts" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    librispeech = build_librispeech(root, data_cfg.get("librispeech", {}), manifest_dir)
    accents = build_accents(root, data_cfg.get("accents", {}), manifest_dir)
    ami = build_ami(root, data_cfg.get("ami", {}))

    card = render_data_card(librispeech, accents, ami)
    write_data_card(card, reports_dir / "data_card.md")
    logger.info("wrote data card", extra={"fields": {"path": str(reports_dir / "data_card.md")}})
