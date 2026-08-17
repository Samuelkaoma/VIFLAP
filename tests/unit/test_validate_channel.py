"""The channel validation measures rather than checks, so its own arithmetic
must be checked here instead.

Two things in ``scripts/validate_channel.py`` can be wrong without anything
noticing. The delay estimate is one: a codec need not return a signal aligned
with its input, and comparing frame *i* against frame *i* under even one frame of
offset produces a distortion dominated by the misalignment rather than by the
coding — a plausible-looking number that is measuring nothing. The other is the
probe, which runs on a remote machine whose log this project's environment
cannot read, so a probe that raised instead of reporting would lose the run.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate_channel import (
    compare,
    estimate_delay,
    probe_ffmpeg,
)
from viflap.analysis.channel.codec import NARROWBAND_RATE


def _speechlike(rng: np.random.Generator, seconds: float = 1.5) -> np.ndarray:
    """A signal with speech's broad character: a resonant excitation, amplitude
    modulated. Not speech, but neither white nor periodic, so cross-correlation
    has a genuine peak to find rather than a plateau or a comb of equal ones.
    """
    n = int(seconds * NARROWBAND_RATE)
    t = np.arange(n) / NARROWBAND_RATE
    excitation = rng.normal(0.0, 1.0, n)
    carrier = np.sin(2 * np.pi * 130.0 * t) + 0.4 * np.sin(2 * np.pi * 780.0 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (carrier + 0.3 * excitation) * envelope


class TestEstimateDelay:
    @pytest.mark.parametrize("delay", [0, 1, 37, 240])
    def test_recovers_a_known_delay(self, rng, delay) -> None:
        source = _speechlike(rng)
        lagged = np.concatenate([np.zeros(delay), source])
        assert estimate_delay(source, lagged) == delay

    def test_never_reports_a_codec_anticipating_its_input(self, rng) -> None:
        """Negative lags are excluded on physical grounds, not for convenience.

        A near-symmetric correlation lets noise pick the wrong side, and a
        negative delay would then be applied as a positive one.
        """
        source = _speechlike(rng)
        advanced = source[100:]
        assert estimate_delay(source, advanced) >= 0

    def test_a_signal_too_short_to_align_reports_no_delay(self) -> None:
        assert estimate_delay(np.zeros(10), np.zeros(10)) == 0


class TestCompare:
    def test_a_signal_against_itself_shows_no_distortion(self, rng) -> None:
        """The control. Any non-zero reading here is the measurement's own error."""
        signals = [_speechlike(rng), _speechlike(rng)]
        result = compare("identity", signals, signals, "a", "b", 12.20)

        assert result.mean_spectral_distortion_db == pytest.approx(0.0, abs=1e-6)
        assert result.fraction_frames_above_2db == 0.0
        assert result.mean_segmental_snr_db > 100.0
        assert result.n_frames > 0

    def test_misalignment_is_removed_rather_than_measured(self, rng) -> None:
        """The whole reason the delay estimate exists.

        The same signal, delayed, must read as the same signal. Without the
        alignment step this comparison reports several decibels of spectral
        distortion that no coder introduced.
        """
        signals = [_speechlike(rng)]
        delayed = [np.concatenate([np.zeros(160), signals[0]])]
        result = compare("delayed", signals, delayed, "a", "b", None)

        assert result.mean_delay_samples == pytest.approx(160.0)
        assert result.mean_spectral_distortion_db == pytest.approx(0.0, abs=1e-6)

    def test_a_noisier_subject_reads_as_more_distorted(self, rng) -> None:
        signals = [_speechlike(rng)]
        mild = [signals[0] + rng.normal(0.0, 0.02, signals[0].size)]
        severe = [signals[0] + rng.normal(0.0, 0.30, signals[0].size)]

        gentle = compare("mild", signals, mild, "a", "b", None)
        harsh = compare("severe", signals, severe, "a", "b", None)

        assert harsh.mean_spectral_distortion_db > gentle.mean_spectral_distortion_db
        assert harsh.mean_segmental_snr_db < gentle.mean_segmental_snr_db

    def test_refuses_to_report_a_figure_from_no_frames(self) -> None:
        with pytest.raises(ValueError, match="no comparable frames"):
            compare("empty", [np.zeros(8)], [np.zeros(8)], "a", "b", None)


class TestProbeFfmpeg:
    """The probe runs where nobody can read the log, so it must always report.

    The first workflow run came back with ``available: false`` and no way to
    tell whether ffmpeg was missing, built without the codec, or carrying it
    under an unexpected name. The probe exists to close that, which it can only
    do by never raising.
    """

    def test_reports_rather_than_raises_wherever_it_runs(self) -> None:
        probe = probe_ffmpeg()
        if probe.ffmpeg_path is None:
            assert probe.version is None
            assert probe.amr_encoder_lines == []
        else:
            assert isinstance(probe.version, str)

    def test_survives_ffmpeg_being_absent(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        probe = probe_ffmpeg()
        assert probe.ffmpeg_path is None
        assert probe.configuration is None

    def test_survives_an_ffmpeg_that_cannot_be_run(self, monkeypatch) -> None:
        """A path that exists but fails to execute is not a crash, it is a fact."""
        monkeypatch.setattr("shutil.which", lambda _: "/nonexistent/ffmpeg")
        probe = probe_ffmpeg()
        assert probe.ffmpeg_path == "/nonexistent/ffmpeg"
        assert probe.amr_encoder_lines == []
