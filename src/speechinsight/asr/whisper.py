from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from speechinsight.types import Word


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    repo: str
    revision: str
    beam_size: int = 5
    temperature: float = 0.0
    compute_type: str = "int8"
    cpu_threads: int = 4
    word_timestamps: bool = True
    language: str = "en"


class WhisperTranscriber:
    """faster-whisper backend. Deterministic (beam search, temperature 0) and
    pinned to an exact model revision so runs are reproducible. The model is
    loaded lazily on first use."""

    def __init__(self, config: WhisperConfig) -> None:
        self._config = config
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._config.repo,
                revision=self._config.revision,
                device="cpu",
                compute_type=self._config.compute_type,
                cpu_threads=self._config.cpu_threads,
            )
        return self._model

    def transcribe(self, samples: npt.NDArray[np.float32], sample_rate: int) -> list[Word]:
        if sample_rate != 16_000:
            raise ValueError(f"expected 16 kHz audio, got {sample_rate}")
        model = self._ensure_model()
        segments, _ = model.transcribe(  # type: ignore[attr-defined]
            samples,
            beam_size=self._config.beam_size,
            temperature=self._config.temperature,
            word_timestamps=self._config.word_timestamps,
            language=self._config.language,
        )
        words: list[Word] = []
        for segment in segments:
            segment_words = getattr(segment, "words", None)
            if segment_words:
                for w in segment_words:
                    words.append(Word(text=w.word.strip(), start=w.start, end=w.end))
            else:
                words.append(Word(text=segment.text.strip(), start=segment.start, end=segment.end))
        return words
