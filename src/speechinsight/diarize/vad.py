from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class VadConfig:
    threshold: float = 0.5
    min_speech_s: float = 0.25
    min_silence_s: float = 0.1


class SileroVad:
    """Silero VAD wrapper. The model is loaded lazily and reused across calls."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self._config = config or VadConfig()
        self._model: object | None = None
        self._utils: tuple | None = None  # type: ignore[type-arg]

    def _ensure_model(self) -> tuple[object, object]:
        if self._model is None:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._model = load_silero_vad()
            self._utils = (get_speech_timestamps,)
        assert self._utils is not None
        return self._model, self._utils[0]

    def detect(self, samples: FloatArray, sample_rate: int) -> list[SpeechRegion]:
        if sample_rate != 16_000:
            raise ValueError(f"Silero VAD expects 16 kHz, got {sample_rate}")
        model, get_speech_timestamps = self._ensure_model()
        import torch

        tensor = torch.from_numpy(np.asarray(samples, dtype=np.float32))
        timestamps = get_speech_timestamps(  # type: ignore[operator]
            tensor,
            model,
            threshold=self._config.threshold,
            min_speech_duration_ms=int(self._config.min_speech_s * 1000),
            min_silence_duration_ms=int(self._config.min_silence_s * 1000),
            return_seconds=True,
        )
        return [SpeechRegion(start=float(t["start"]), end=float(t["end"])) for t in timestamps]
