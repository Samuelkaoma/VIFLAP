"""The sample-rate decision, and the split that keeps it testable.

``viflap.analysis.speaker.neural`` holds the science — what rate a VoxCeleb2
checkpoint is defined against and how a signal is put into it — and imports
nothing heavier than scipy. The adapter that loads an 89 MB checkpoint and calls
a deep learning framework lives in ``viflap.infrastructure``. That is not
tidiness: it is what lets the decision below be tested in milliseconds, without
torch installed, on a machine that has never downloaded a checkpoint.

The decision itself was measured on 40 held-out recordings through
``amr12.2_clean``. At the extractor's native 16 kHz the cosine EER was 0.00% with
a mean different-source similarity of 0.386; at 8 kHz it was 5.79% and 0.621 —
the embeddings collapse toward one another and stop separating people. So
resampling is load-bearing rather than cosmetic, and these tests exist because a
future change that quietly skipped it would produce embeddings, not an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from viflap.analysis.speaker.neural import (
    NEURAL_EXTRACTOR_RATE,
    prepare_for_extractor,
    resample_to,
)


def _tone(seconds: float, rate: int, frequency: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return np.sin(2 * np.pi * frequency * t)


class TestResampleTo:
    def test_the_rate_is_what_the_checkpoint_expects(self) -> None:
        assert NEURAL_EXTRACTOR_RATE == 16_000

    @pytest.mark.parametrize(
        "source_rate,target_rate", [(8_000, 16_000), (16_000, 8_000), (44_100, 16_000)]
    )
    def test_duration_is_preserved_across_a_rate_change(
        self, source_rate, target_rate
    ) -> None:
        signal = _tone(2.0, source_rate)
        out = resample_to(signal, source_rate, target_rate)
        assert out.size / target_rate == pytest.approx(2.0, abs=0.01)

    def test_a_signal_already_at_the_target_rate_is_untouched(self) -> None:
        signal = _tone(1.0, 16_000)
        assert np.array_equal(resample_to(signal, 16_000, 16_000), signal)

    def test_upsampling_preserves_the_content_rather_than_inventing_it(self) -> None:
        """8 kHz to 16 kHz restores the rate, not the band.

        A 440 Hz tone must still be a 440 Hz tone. If it lands anywhere else the
        resampler is wrong in the way that matters: the extractor's filterbank
        would then be reading every formant at the wrong frequency, which is
        exactly the failure the native-8 kHz arm of the probe measured.
        """
        upsampled = resample_to(_tone(1.0, 8_000), 8_000, 16_000)
        spectrum = np.abs(np.fft.rfft(upsampled * np.hanning(upsampled.size)))
        peak_hz = float(np.fft.rfftfreq(upsampled.size, 1 / 16_000)[np.argmax(spectrum)])
        assert peak_hz == pytest.approx(440.0, abs=5.0)

    def test_upsampling_leaves_the_upper_band_empty(self) -> None:
        """The band above 4 kHz stays empty, and that is the honest outcome.

        Upsampling adds no information. Anything appreciable above the original
        Nyquist frequency would be an imaging artefact the extractor would read
        as speech.
        """
        upsampled = resample_to(_tone(1.0, 8_000, frequency=1000.0), 8_000, 16_000)
        spectrum = np.abs(np.fft.rfft(upsampled))
        frequencies = np.fft.rfftfreq(upsampled.size, 1 / 16_000)
        upper = spectrum[frequencies > 4_200.0]
        assert float(upper.max()) < 0.01 * float(spectrum.max())


class TestPrepareForExtractor:
    def test_narrowband_audio_arrives_at_the_extractor_rate(self) -> None:
        prepared = prepare_for_extractor(_tone(1.0, 8_000), 8_000)
        assert prepared.size == pytest.approx(NEURAL_EXTRACTOR_RATE, rel=0.01)

    def test_wideband_audio_is_left_where_it_is(self) -> None:
        signal = _tone(1.0, 16_000)
        assert np.array_equal(prepare_for_extractor(signal, 16_000), signal)

    def test_every_input_rate_converges_on_one_output_rate(self) -> None:
        """The whole point of routing through one function.

        Two call sites disagreeing about the rate would produce embeddings that
        are not comparable, and nothing downstream can detect that by looking at
        the vectors.
        """
        sizes = {
            prepare_for_extractor(_tone(1.0, rate), rate).size
            for rate in (8_000, 16_000, 22_050, 44_100)
        }
        assert max(sizes) - min(sizes) <= 1
