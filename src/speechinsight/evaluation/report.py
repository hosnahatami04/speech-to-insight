from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from speechinsight.evaluation.attribution import AttributionTable
from speechinsight.evaluation.noise_sweep import SweepPoint
from speechinsight.evaluation.propagation import PropagationFit


@dataclass(frozen=True, slots=True)
class ModelPins:
    """The exact resolved identities frozen into one evaluation run. Reporting
    these next to the numbers is required by the spec (every number carries the
    configuration that produced it)."""

    asr_tiny: dict[str, str]
    asr_base: dict[str, str]
    asr_small: dict[str, str]
    speaker_embedding: dict[str, str]
    nli: dict[str, str]
    matcher_embedding: dict[str, str]
    extractor_model_tag: str
    extractor_model_digest: str
    prompt_version: str


@dataclass
class EvaluationResults:
    """Top-level container that becomes results.json."""

    git_commit: str
    model_pins: ModelPins
    wer: dict[str, Any] = field(default_factory=dict)
    der: dict[str, Any] = field(default_factory=dict)
    noise_sweep: list[dict[str, Any]] = field(default_factory=list)
    per_accent_wer: list[dict[str, Any]] = field(default_factory=list)
    extraction: dict[str, Any] = field(default_factory=dict)
    propagation: dict[str, Any] = field(default_factory=dict)
    attribution: list[dict[str, Any]] = field(default_factory=list)
    what_didnt_work: list[str] = field(default_factory=list)


def current_git_commit() -> str:
    """SHA of HEAD, or 'unknown' if git is unavailable (e.g. inside a slim
    container). The commit travels with the results so numbers are traceable."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def model_pins_from_settings(raw: dict[str, Any]) -> ModelPins:
    models = raw["models"]
    extraction = raw.get("extraction", {})
    return ModelPins(
        asr_tiny=models["asr_tiny"],
        asr_base=models["asr_base"],
        asr_small=models["asr_small"],
        speaker_embedding=models["speaker_embedding"],
        nli=models["nli"],
        matcher_embedding=models["matcher_embedding"],
        extractor_model_tag=extraction.get("model_tag", ""),
        extractor_model_digest=extraction.get("model_digest", ""),
        prompt_version=extraction.get("prompt_version", "v1"),
    )


def _asdict(obj: Any) -> Any:
    """Recursively serialize a dataclass-of-dataclasses to a JSON-safe tree."""

    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list | tuple):
        return [_asdict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


def write_results_json(results: EvaluationResults, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_asdict(results), f, indent=2, sort_keys=True)


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.2f}%" if value is not None else "—"


def render_results_md(results: EvaluationResults) -> str:
    """Render results.md from the JSON structure. Every number here comes from
    the ``results`` object; nothing is interpolated from outside."""

    lines: list[str] = ["# Evaluation results", ""]
    lines.append(f"Commit: `{results.git_commit}`")
    lines.append("")
    lines.append("Model pins:")
    lines.append("")
    pins = results.model_pins
    lines.append(
        f"- faster-whisper tiny.en: `{pins.asr_tiny['repo']}@{pins.asr_tiny['revision'][:12]}`"
    )
    lines.append(
        f"- faster-whisper base.en: `{pins.asr_base['repo']}@{pins.asr_base['revision'][:12]}`"
    )
    lines.append(
        f"- faster-whisper small.en: `{pins.asr_small['repo']}@{pins.asr_small['revision'][:12]}`"
    )
    sp = pins.speaker_embedding
    lines.append(f"- speaker embedding: `{sp['repo']}@{sp['revision'][:12]}`")
    lines.append(f"- NLI: `{pins.nli['repo']}@{pins.nli['revision'][:12]}`")
    mp = pins.matcher_embedding
    lines.append(f"- matcher embedding: `{mp['repo']}@{mp['revision'][:12]}`")
    lines.append(
        f"- extractor: `{pins.extractor_model_tag}` ({pins.extractor_model_digest[:19]}...)"
    )
    lines.append(f"- prompt version: `{pins.prompt_version}`")
    lines.append("")

    if results.wer:
        lines.append("## Speech recognition (WER)")
        lines.append("")
        lines.append("| split | model | WER | CI low | CI high | normalizer |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in results.wer.get("points", []):
            lines.append(
                f"| {entry['split']} | {entry['model']} | {_fmt_pct(entry['wer'])} | "
                f"{_fmt_pct(entry['ci_low'])} | {_fmt_pct(entry['ci_high'])} | "
                f"{entry['normalizer']} |"
            )
        lines.append("")

    if results.der:
        lines.append("## Diarization (DER)")
        lines.append("")
        lines.append(f"- Collar: {results.der.get('collar_s', 0.25)} s")
        lines.append(f"- Overlap skipped: {results.der.get('skip_overlap', False)}")
        lines.append(f"- DER: {_fmt_pct(results.der.get('der'))}")
        lines.append(f"- Miss: {_fmt_pct(results.der.get('miss'))}")
        lines.append(f"- False alarm: {_fmt_pct(results.der.get('false_alarm'))}")
        lines.append(f"- Confusion: {_fmt_pct(results.der.get('confusion'))}")
        lines.append("")

    if results.noise_sweep:
        lines.append("## Noise robustness")
        lines.append("")
        lines.append("| model | noise | SNR (dB) | WER | CI |")
        lines.append("| --- | --- | --- | --- | --- |")
        for p in results.noise_sweep:
            lines.append(
                f"| {p['model']} | {p['noise_kind']} | {p['snr_db']} | "
                f"{_fmt_pct(p['wer'])} | ±{_fmt_pct((p['ci_high'] - p['ci_low']) / 2)} |"
            )
        lines.append("")

    if results.per_accent_wer:
        lines.append("## Per-accent WER")
        lines.append("")
        lines.append("| accent | utterances | WER | CI | note |")
        lines.append("| --- | --- | --- | --- | --- |")
        for a in results.per_accent_wer:
            lines.append(
                f"| {a['accent']} | {a['utterances']} | {_fmt_pct(a['wer'])} | "
                f"±{_fmt_pct((a['ci_high'] - a['ci_low']) / 2)} | {a.get('note', '')} |"
            )
        lines.append("")

    if results.extraction:
        lines.append("## Extraction (grounding and matching)")
        lines.append("")
        ex = results.extraction
        lines.append(f"- Schema validity: {_fmt_pct(ex.get('schema_validity'))}")
        lines.append(f"- Dropped (no evidence): {ex.get('dropped_no_evidence', 0)}")
        lines.append(f"- Unsupported entities: {ex.get('unsupported_entities', 0)}")
        lines.append(f"- Unsupported by NLI: {ex.get('unsupported_nli', 0)}")
        lines.append(f"- F1 vs reference transcript: {_fmt_pct(ex.get('f1_reference'))}")
        lines.append(f"- F1 vs system transcript: {_fmt_pct(ex.get('f1_system'))}")
        lines.append("")

    if results.propagation:
        lines.append("## Error propagation")
        lines.append("")
        prop = results.propagation
        lines.append(f"- F1 slope per 10 WER points: {prop.get('f1_slope_per_10wer', 'n/a')}")
        lines.append(f"- Fit intercept: {prop.get('f1_intercept', 'n/a')}")
        lines.append(f"- Points fitted: {prop.get('n_points', 0)}")
        lines.append("")

    if results.attribution:
        lines.append("## Stage attribution")
        lines.append("")
        lines.append("| ASR | Diarization | F1 | utterances |")
        lines.append("| --- | --- | --- | --- |")
        for c in results.attribution:
            lines.append(
                f"| {c['asr']} | {c['diarization']} | {_fmt_pct(c['extraction_f1'])} | "
                f"{c['utterances']} |"
            )
        lines.append("")

    lines.append("## What didn't work")
    lines.append("")
    if results.what_didnt_work:
        for item in results.what_didnt_work:
            lines.append(f"- {item}")
    else:
        lines.append("_(no entries recorded)_")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_results_md(results: EvaluationResults, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_results_md(results), encoding="utf-8")
    return path


def noise_sweep_as_dicts(points: Iterable[SweepPoint]) -> list[dict[str, Any]]:
    return [asdict(p) for p in points]


def attribution_as_dicts(table: AttributionTable) -> list[dict[str, Any]]:
    return [asdict(c) for c in table.cells]


def propagation_as_dict(fit: PropagationFit) -> dict[str, Any]:
    """Report slope rescaled per 10 WER points (readable in the markdown)."""

    return {
        "f1_slope_per_10wer": round(fit.slope * 0.1, 4),
        "f1_intercept": round(fit.intercept, 4),
        "n_points": fit.n_points,
    }
