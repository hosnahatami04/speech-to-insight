from __future__ import annotations

from typing import Any

from speechinsight.asr.whisper import WhisperConfig, WhisperTranscriber


def whisper_config(settings_raw: dict[str, Any], model_key: str) -> WhisperConfig:
    """Build a WhisperConfig from the loaded settings. ``model_key`` names an
    entry under ``models`` (e.g. ``asr_base``); ASR hyperparameters come from the
    ``asr`` config block."""

    pin = settings_raw["models"][model_key]
    asr = settings_raw.get("asr", {})
    return WhisperConfig(
        repo=pin["repo"],
        revision=pin["revision"],
        beam_size=asr.get("beam_size", 5),
        temperature=asr.get("temperature", 0.0),
        compute_type=asr.get("compute_type", "int8"),
        cpu_threads=asr.get("cpu_threads", 4),
        word_timestamps=asr.get("word_timestamps", True),
    )


def build_transcriber(settings_raw: dict[str, Any], model_key: str) -> WhisperTranscriber:
    return WhisperTranscriber(whisper_config(settings_raw, model_key))
