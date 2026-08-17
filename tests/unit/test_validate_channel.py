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
    NO_AUDIO,
    NO_REFERENCE_CODER,
    align_to,
    compare,
    estimate_delay,
    main,
    probe_ffmpeg,
    select_recordings,
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


class TestAlignToACommonSource:
    """One coder against another is where the non-negative lag rule breaks.

    The rule is right for a coder against its source — a codec delays its
    output, it does not anticipate its input. It is wrong for two coders
    compared with each other: the parametric model returns its output aligned
    with the input while AMR delays it, so the parametric signal leads and the
    estimator has no way to say so. The third workflow run reported a mean
    delay of 36 samples and a maximum of 106 on a comparison whose true offset
    is a constant minus 40.
    """

    def test_a_leading_subject_defeats_the_direct_comparison(self, rng) -> None:
        """The defect, asserted, so the fix cannot be undone silently."""
        source = _speechlike(rng)
        undelayed = source
        delayed = np.concatenate([np.zeros(40), source])

        # Comparing the delayed one against the undelayed one directly: the
        # subject leads, so the estimate cannot be right.
        assert estimate_delay(delayed, undelayed) != 0

    def test_aligning_both_to_the_source_leaves_them_aligned_with_each_other(
        self, rng
    ) -> None:
        source = [_speechlike(rng)]
        undelayed = [source[0].copy()]
        delayed = [np.concatenate([np.zeros(40), source[0]])]

        result = compare(
            "coder vs coder",
            align_to(source, delayed),
            align_to(source, undelayed),
            "a",
            "b",
            None,
        )
        assert result.mean_delay_samples == pytest.approx(0.0)
        assert result.mean_spectral_distortion_db == pytest.approx(0.0, abs=1e-6)

    def test_leaves_an_already_aligned_signal_untouched(self, rng) -> None:
        source = [_speechlike(rng)]
        assert np.array_equal(align_to(source, [source[0].copy()])[0], source[0])

    def test_removes_exactly_the_delay_it_was_given(self, rng) -> None:
        source = [_speechlike(rng)]
        delayed = [np.concatenate([np.zeros(73), source[0]])]
        assert np.allclose(align_to(source, delayed)[0], source[0])


class TestSelectRecordings:
    """Six recordings from one chapter of one speaker is not six recordings.

    That is what the first run of this measurement actually sampled: a sorted
    glob puts one speaker's files consecutively, so taking the first N of it
    took one voice. Coding distortion varies with the voice, so the sample has
    to span speakers or the figure is narrower than it reads.
    """

    @staticmethod
    def _corpus(root, speakers: dict[str, int]):
        for speaker, count in speakers.items():
            directory = root / speaker / "chapter"
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / f"{speaker}-{index:03d}.flac").write_bytes(b"")
        return root

    def test_spreads_across_speakers_before_taking_a_second_from_any(
        self, tmp_path
    ) -> None:
        root = self._corpus(tmp_path, {"101": 6, "202": 6, "303": 6})
        chosen = select_recordings(root, "*.flac", 6)
        speakers = [path.relative_to(root).parts[0] for path in chosen]
        assert sorted(set(speakers)) == ["101", "202", "303"]
        assert speakers.count("101") == 2

    def test_falls_back_to_the_speakers_that_have_material(self, tmp_path) -> None:
        """An uneven corpus must still fill the request rather than stopping."""
        root = self._corpus(tmp_path, {"101": 1, "202": 5})
        chosen = select_recordings(root, "*.flac", 4)
        assert len(chosen) == 4

    def test_never_returns_more_than_the_corpus_holds(self, tmp_path) -> None:
        root = self._corpus(tmp_path, {"101": 2, "202": 1})
        assert len(select_recordings(root, "*.flac", 50)) == 3

    def test_is_the_same_selection_on_every_run(self, tmp_path) -> None:
        root = self._corpus(tmp_path, {"101": 4, "202": 4, "303": 4})
        assert select_recordings(root, "*.flac", 7) == select_recordings(root, "*.flac", 7)

    def test_an_empty_corpus_selects_nothing_rather_than_failing(self, tmp_path) -> None:
        assert select_recordings(tmp_path, "*.flac", 6) == []


class TestExitStatus:
    """ "Ran and found no encoder" and "did not run" must not share a code.

    They are opposite outcomes and Python exits 1 for an uncaught exception, so
    the script cannot also use 1 for the first. Two workflow runs crashed inside
    the measurement and were indistinguishable from a clean run that found no
    encoder, on a machine that cannot read the job log.
    """

    def test_no_reference_coder_is_not_the_interpreter_failure_code(self) -> None:
        assert NO_REFERENCE_CODER != 1
        assert NO_AUDIO != 1
        assert NO_REFERENCE_CODER != NO_AUDIO
        assert NO_REFERENCE_CODER != 0

    def test_an_empty_corpus_reports_having_nothing_to_measure(self, tmp_path) -> None:
        assert main(["--audio", str(tmp_path), "--pattern", "*.flac"]) == NO_AUDIO


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
