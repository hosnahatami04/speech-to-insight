from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from speechinsight.diarize.vad import SpeechRegion

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Window:
    """A fixed-length chunk of audio inside a speech region. ``embedding`` is
    kept in memory only and never persisted — speaker embeddings are biometric
    data (spec §1.2)."""

    start: float
    end: float
    embedding: FloatArray


@dataclass(frozen=True, slots=True)
class WindowingConfig:
    window_s: float = 1.5
    hop_s: float = 0.75


def iter_windows(
    regions: list[SpeechRegion], config: WindowingConfig | None = None
) -> list[tuple[float, float]]:
    """Yield (start, end) windows that lie entirely within some speech region.
    A region shorter than one window still gets one window centered on it."""

    cfg = config or WindowingConfig()
    out: list[tuple[float, float]] = []
    for region in regions:
        duration = region.end - region.start
        if duration <= 0:
            continue
        if duration < cfg.window_s:
            out.append((region.start, region.end))
            continue
        t = region.start
        while t + cfg.window_s <= region.end + 1e-9:
            out.append((t, t + cfg.window_s))
            t += cfg.hop_s
    return out


class EcapaEmbedder:
    """Thin wrapper around SpeechBrain's ECAPA-TDNN speaker embedding. Loaded
    lazily; the resulting embeddings are returned, never cached."""

    def __init__(self, repo: str, revision: str) -> None:
        self._repo = repo
        self._revision = revision
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=self._repo,
                savedir=f"artifacts/models/{self._repo.replace('/', '_')}",
                run_opts={"device": "cpu"},
            )
        return self._model

    def embed(self, samples: FloatArray, sample_rate: int) -> FloatArray:
        if sample_rate != 16_000:
            raise ValueError(f"ECAPA expects 16 kHz, got {sample_rate}")
        import torch

        model = self._ensure_model()
        tensor = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(tensor)  # type: ignore[attr-defined]
        return np.asarray(emb.squeeze().cpu().numpy(), dtype=np.float32)

    def embed_windows(
        self, samples: FloatArray, sample_rate: int, windows: list[tuple[float, float]]
    ) -> list[Window]:
        """Compute an embedding for each window. The embeddings are only in the
        returned ``Window`` objects; nothing is written to disk here."""

        out: list[Window] = []
        for start_s, end_s in windows:
            lo = int(start_s * sample_rate)
            hi = min(len(samples), int(end_s * sample_rate))
            if hi <= lo:
                continue
            chunk = samples[lo:hi]
            emb = self.embed(chunk, sample_rate)
            out.append(Window(start=start_s, end=end_s, embedding=emb))
        return out
