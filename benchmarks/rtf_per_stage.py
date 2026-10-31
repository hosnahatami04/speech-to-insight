"""Measure the real-time factor of each pipeline stage on a short audio file.

Run with:
    uv run python benchmarks/rtf_per_stage.py path/to/audio.wav
"""

from __future__ import annotations

import sys
import time

from speechinsight.asr.factory import build_transcriber
from speechinsight.audio.io import load_audio
from speechinsight.config import load_settings
from speechinsight.diarize.cluster import cluster_windows, labels_to_segments
from speechinsight.diarize.embed import EcapaEmbedder, iter_windows
from speechinsight.diarize.vad import SileroVad


def _rtf(label: str, elapsed_s: float, duration_s: float) -> str:
    return f"{label:<20} {elapsed_s:7.2f}s  (RTF {elapsed_s / duration_s:.3f})"


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("usage: rtf_per_stage.py <audio.wav>", file=sys.stderr)
        return 2
    audio_path = argv[0]
    audio = load_audio(audio_path)
    duration = audio.duration

    settings = load_settings()
    print(f"audio: {audio_path}  ({duration:.1f}s)")

    t = time.perf_counter()
    transcriber = build_transcriber(settings.raw, "asr_base")
    transcriber.transcribe(audio.samples, audio.sample_rate)
    print(_rtf("ASR (base.en)", time.perf_counter() - t, duration))

    t = time.perf_counter()
    vad = SileroVad()
    regions = vad.detect(audio.samples, audio.sample_rate)
    pin = settings.raw["models"]["speaker_embedding"]
    embedder = EcapaEmbedder(repo=pin["repo"], revision=pin["revision"])
    windows = embedder.embed_windows(
        audio.samples, audio.sample_rate, iter_windows(regions)
    )
    labels_to_segments(windows, cluster_windows(windows))
    print(_rtf("Diarization", time.perf_counter() - t, duration))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
