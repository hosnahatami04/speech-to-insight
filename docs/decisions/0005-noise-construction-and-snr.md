# 0005. Noise construction and SNR measurement

## Context

The robustness study mixes additive noise into clean speech at a controlled
signal-to-noise ratio. For the WER curves to be comparable across models and
noise types, the SNR has to mean the same thing everywhere: the ratio of
*speech* power to *noise* power, not of overall waveform powers (silence
would otherwise inflate the apparent SNR). Three noise types are needed to
probe different failure modes.

## Decision

- **White noise** is Gaussian samples with unit standard deviation.
- **Pink noise** is white noise shaped in the frequency domain so the amplitude
  spectrum is scaled by `1/sqrt(f)`; power then falls as `1/f`, which a
  slope test confirms is near -1 in the log-log spectrum.
- **Babble noise** mixes 4–6 utterances from distinct speakers (ids different
  from the target utterance's speaker) at random offsets, with each source
  tiled to cover the target length. The mix is normalized to unit standard
  deviation so the SNR step scales it predictably.
- **Speech-active power** is estimated per utterance as the mean square of
  frames whose RMS is above the median RMS over the file. This is a simple,
  parameter-free proxy that avoids depending on a VAD for the mixing step and
  is deterministic.
- The noise waveform is scaled so `measured_speech_power / measured_noise_power`
  hits the target SNR; a tolerance of 0.3 dB is enforced in tests.
- If the mixed waveform peaks above 1.0 the whole signal is divided by its
  peak. This preserves the ratio of speech to noise (both are scaled by the
  same factor), so the SNR is unchanged. The alternative — clipping — would
  distort both signals and corrupt the SNR.
- Every noise generator takes a seed; identical seeds produce identical
  waveforms.

## Rejected options

- **A fixed VAD for speech-power estimation**: adds a dependency and a model
  decision inside the measurement, which should stay simple and auditable.
- **Normalizing noise to a fixed level and ignoring per-file speech power**:
  loses the per-utterance SNR guarantee and makes the curves hard to interpret.
- **Clipping instead of peak-normalizing**: destroys the SNR we just set.

## Consequence

Every reported `(noise kind, SNR)` point is a controlled, reproducible mix.
The spec limits SNRs to `{20, 10, 5, 0}` dB and noise kinds to
`{white, pink, babble}`, and the sweep code enumerates that grid.
