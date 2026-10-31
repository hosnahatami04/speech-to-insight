from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from speechinsight.types import Word


class Transcriber(Protocol):
    """A speech recognizer. Takes 16 kHz mono float32 audio and returns words
    with timestamps. Keeping this behind a Protocol lets the stage be evaluated
    in isolation and swapped for a reference transcript in the propagation study."""

    def transcribe(self, samples: npt.NDArray[np.float32], sample_rate: int) -> list[Word]: ...
