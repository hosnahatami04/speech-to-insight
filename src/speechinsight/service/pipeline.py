"""Default pipeline implementation used by the service.

Each stage is loaded lazily so a missing optional dependency (for example
Ollama when the extractor is down) does not stop the API from booting. The
compose stack brings up ollama/api/ui together.
"""

from __future__ import annotations

import time
from typing import Any

from speechinsight.service.api import Pipeline


class DefaultPipeline(Pipeline):
    """Real pipeline that wires the ASR, diarization and extraction stages
    together on first use. All heavy objects are cached per-process."""

    def __init__(self) -> None:
        self._settings: Any | None = None
        self._transcriber: Any | None = None
        self._vad: Any | None = None
        self._embedder: Any | None = None
        self._extract_client: Any | None = None
        self._prompt: str | None = None

    def _ensure(self) -> None:
        if self._settings is not None:
            return
        from speechinsight.asr.factory import build_transcriber
        from speechinsight.config import load_settings
        from speechinsight.diarize.embed import EcapaEmbedder
        from speechinsight.diarize.vad import SileroVad
        from speechinsight.extract.client import OllamaClient, OllamaConfig
        from speechinsight.extract.run import load_prompt_template

        self._settings = load_settings()
        self._transcriber = build_transcriber(self._settings.raw, "asr_base")
        self._vad = SileroVad()
        pin = self._settings.raw["models"]["speaker_embedding"]
        self._embedder = EcapaEmbedder(repo=pin["repo"], revision=pin["revision"])

        ex = self._settings.raw.get("extraction", {})
        self._extract_client = OllamaClient(
            OllamaConfig(
                host=ex.get("host", "http://localhost:11434"),
                model_tag=ex.get("model_tag", "qwen2.5:3b-instruct"),
                timeout_s=ex.get("timeout_s", 120.0),
                retries=ex.get("retries", 1),
                seed=ex.get("seed", 0),
            )
        )
        self._prompt = load_prompt_template("prompts/extract_v1.txt")

    def transcribe_and_diarize(self, audio_path: str) -> tuple[Any, dict[str, float]]:
        from speechinsight.audio.io import load_audio
        from speechinsight.diarize.align import build_transcript
        from speechinsight.diarize.cluster import cluster_windows, labels_to_segments
        from speechinsight.diarize.embed import iter_windows

        self._ensure()
        assert self._transcriber is not None and self._vad is not None
        assert self._embedder is not None

        audio = load_audio(audio_path)
        t0 = time.perf_counter()
        words = self._transcriber.transcribe(audio.samples, audio.sample_rate)
        asr_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        regions = self._vad.detect(audio.samples, audio.sample_rate)
        windows = iter_windows(regions)
        embedded = self._embedder.embed_windows(audio.samples, audio.sample_rate, windows)
        labels = cluster_windows(embedded)
        segments = labels_to_segments(embedded, labels)
        diar_s = time.perf_counter() - t1

        transcript = build_transcript(words, segments)
        return transcript, {"asr_s": asr_s, "diarization_s": diar_s}

    def extract(self, transcript: Any) -> Any:
        from speechinsight.extract.client import OllamaError
        from speechinsight.extract.run import extract

        self._ensure()
        assert self._extract_client is not None and self._prompt is not None

        def similarity(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            return len(sa & sb) / max(1, len(sa | sb))

        try:
            return extract(transcript, self._extract_client, self._prompt, similarity)
        except OllamaError as exc:
            raise Pipeline.ExtractorUnavailable(str(exc)) from exc

    def health(self) -> dict[str, bool]:
        import urllib.error
        import urllib.request

        self._ensure()
        assert self._extract_client is not None
        extractor_ok = True
        try:
            req = urllib.request.Request(
                f"{self._extract_client._config.host}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2.0):
                pass
        except (urllib.error.URLError, OSError):
            extractor_ok = False
        return {"asr": True, "diarization": True, "extractor": extractor_ok}


def build_default_app() -> Any:
    """Entry point for `uvicorn --factory`."""

    from speechinsight.service.api import create_app

    return create_app(DefaultPipeline())
