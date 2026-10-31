from __future__ import annotations

import json
from pathlib import Path

from speechinsight.evaluation.attribution import AttributionCell, build_attribution
from speechinsight.evaluation.noise_sweep import SweepPoint
from speechinsight.evaluation.propagation import PropagationFit
from speechinsight.evaluation.report import (
    EvaluationResults,
    ModelPins,
    attribution_as_dicts,
    model_pins_from_settings,
    noise_sweep_as_dicts,
    propagation_as_dict,
    render_results_md,
    write_results_json,
    write_results_md,
)


def _pins() -> ModelPins:
    return ModelPins(
        asr_tiny={"repo": "r/tiny", "revision": "abc1234567890def"},
        asr_base={"repo": "r/base", "revision": "aaa0000000000000"},
        asr_small={"repo": "r/small", "revision": "bbb0000000000000"},
        speaker_embedding={"repo": "r/ecapa", "revision": "ccc0000000000000"},
        nli={"repo": "r/nli", "revision": "ddd0000000000000"},
        matcher_embedding={"repo": "r/match", "revision": "eee0000000000000"},
        extractor_model_tag="qwen2.5:3b-instruct",
        extractor_model_digest="sha256:fff0000000000000000000000000000000000000000000000000000000000000",
        prompt_version="v1",
    )


def test_json_round_trip(tmp_path: Path) -> None:
    r = EvaluationResults(git_commit="deadbeef", model_pins=_pins())
    out = tmp_path / "results.json"
    write_results_json(r, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["git_commit"] == "deadbeef"
    assert data["model_pins"]["prompt_version"] == "v1"


def test_markdown_includes_pins_and_sections() -> None:
    r = EvaluationResults(
        git_commit="deadbeef",
        model_pins=_pins(),
        wer={
            "points": [
                {
                    "split": "test-clean",
                    "model": "base.en",
                    "wer": 0.07,
                    "ci_low": 0.065,
                    "ci_high": 0.075,
                    "normalizer": "speechinsight-v1",
                }
            ]
        },
        der={"collar_s": 0.25, "der": 0.30, "miss": 0.10, "false_alarm": 0.05, "confusion": 0.15},
        what_didnt_work=["AMI subset is small"],
    )
    md = render_results_md(r)
    assert "deadbeef" in md
    assert "qwen2.5:3b-instruct" in md
    assert "7.00%" in md
    assert "AMI subset is small" in md


def test_write_results_md(tmp_path: Path) -> None:
    r = EvaluationResults(git_commit="x", model_pins=_pins())
    out = tmp_path / "results.md"
    returned = write_results_md(r, out)
    assert returned == out
    assert out.exists() and out.read_text(encoding="utf-8").startswith("# Evaluation")


def test_model_pins_from_settings() -> None:
    raw = {
        "models": {
            "asr_tiny": {"repo": "r/tiny", "revision": "a"},
            "asr_base": {"repo": "r/base", "revision": "b"},
            "asr_small": {"repo": "r/small", "revision": "c"},
            "speaker_embedding": {"repo": "r/sp", "revision": "d"},
            "nli": {"repo": "r/nli", "revision": "e"},
            "matcher_embedding": {"repo": "r/m", "revision": "f"},
        },
        "extraction": {
            "model_tag": "qwen2.5:3b-instruct",
            "model_digest": "sha256:x",
            "prompt_version": "v2",
        },
    }
    pins = model_pins_from_settings(raw)
    assert pins.asr_base["repo"] == "r/base"
    assert pins.extractor_model_tag == "qwen2.5:3b-instruct"
    assert pins.prompt_version == "v2"


def test_sweep_and_attribution_serialization() -> None:
    pts = [SweepPoint("base.en", "white", 10.0, 0.15, 0.12, 0.18, 100, "speechinsight-v1")]
    dicts = noise_sweep_as_dicts(pts)
    assert dicts[0]["wer"] == 0.15

    table = build_attribution([AttributionCell("reference", "reference", 0.8, None, None, 50)])
    assert attribution_as_dicts(table)[0]["extraction_f1"] == 0.8


def test_propagation_as_dict_rescales_slope_per_10wer() -> None:
    fit = PropagationFit(slope=-0.5, intercept=0.8, n_points=5)
    out = propagation_as_dict(fit)
    # slope -0.5 per 1 WER => -0.05 per 10 WER points
    assert out["f1_slope_per_10wer"] == -0.05
    assert out["n_points"] == 5
