"""Measure the real-time factor of each ASR model on a short audio file.

Run with a 16 kHz mono WAV path and one or more model keys:

    uv run python benchmarks/rtf.py path/to/audio.wav asr_tiny asr_base asr_small
"""

from __future__ import annotations

import sys

from speechinsight.asr.factory import build_transcriber
from speechinsight.audio.io import load_audio
from speechinsight.config import load_settings
from speechinsight.evaluation.rtf import measure_rtf


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: rtf.py <audio.wav> <model_key> [<model_key> ...]", file=sys.stderr)
        return 2
    audio_path, model_keys = argv[0], argv[1:]
    settings = load_settings()
    audio = load_audio(audio_path)
    for model_key in model_keys:
        transcriber = build_transcriber(settings.raw, model_key)
        rtf = measure_rtf(transcriber.transcribe, audio.samples, audio.sample_rate)
        print(f"{model_key}\t{rtf:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
