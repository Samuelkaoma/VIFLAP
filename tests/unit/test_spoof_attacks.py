"""Synthetic attacks, and the properties that make them worth training on.

A countermeasure is only as honest as the spoofed material it learned from. Two
ways to get a flattering number out of a bad attack set, both tested against
here.

**An attack detectable by loudness.** If resynthesis changes the level, the
detector separates the classes on energy and reports an equal error rate that
measures this module rather than synthetic speech. Every attack matches the
energy of what it replaces.

**Attacks that all fail the same way.** Leaving one family out of training tests
nothing if the remaining three carry the same artefact. The four here attack
different things — three replace the excitation, one leaves the excitation
genuine and smooths the filter — and the tests assert that separation rather
than assuming it.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from viflap.analysis.spoof.attacks import (
    ATTACKS,
    apply_attack,
    lpc_noise,
    lpc_pulse,
    oversmoothed,
    phase_randomised,
)
from viflap.domain.errors import InsufficientDataError

SAMPLE_RATE = 16_000


@pytest.fixture
def speech() -> NDArray[np.float64]:
    """A voice-like signal: harmonic source through fixed formants.

    Structured rather than noise, because the attacks are all-pole analysis and
    resynthesis — white noise has no vocal tract to model and would make every
    attack look like every other.
    """
    duration = 1.5
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE

    source = np.zeros(n)
    for harmonic in range(1, 30):
        if harmonic * 110.0 >= SAMPLE_RATE / 2:
            break
        source += np.sin(2.0 * np.pi * harmonic * 110.0 * t) / harmonic

    impulse = np.zeros(512)
    steps = np.arange(512)
    for frequency in (700.0, 1600.0, 2600.0):
        decay = np.exp(-np.pi * 90.0 * steps / SAMPLE_RATE)
        impulse += decay * np.sin(2.0 * np.pi * frequency * steps / SAMPLE_RATE)

    voiced = np.convolve(source, impulse, mode="same")
    envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)) ** 2
    signal = voiced * envelope
    return signal / (np.max(np.abs(signal)) + 1e-12)


def _rms(signal: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(signal**2)))


class TestRegistry:
    def test_four_families_are_registered(self) -> None:
        """Cross-attack evaluation needs more than one, and more than two makes
        the held-out estimate less dependent on which one was held out."""
        assert set(ATTACKS) == {
            "lpc_noise",
            "lpc_pulse",
            "phase_randomised",
            "oversmoothed",
        }

    def test_every_attack_describes_itself(self) -> None:
        for attack in ATTACKS.values():
            assert attack.description
            assert attack.attack_id

    def test_an_unknown_attack_is_refused(self, speech: NDArray[np.float64]) -> None:
        with pytest.raises(InsufficientDataError):
            apply_attack("neural_tts", speech, SAMPLE_RATE, np.random.default_rng(0))


class TestSharedProperties:
    @pytest.mark.parametrize("attack_id", sorted(ATTACKS))
    def test_output_matches_the_input_length(
        self, speech: NDArray[np.float64], attack_id: str
    ) -> None:
        produced = apply_attack(attack_id, speech, SAMPLE_RATE, np.random.default_rng(1))
        assert produced.shape == speech.shape

    @pytest.mark.parametrize("attack_id", sorted(ATTACKS))
    def test_output_is_finite(self, speech: NDArray[np.float64], attack_id: str) -> None:
        """An all-pole filter can be unstable; a NaN would poison the training set."""
        produced = apply_attack(attack_id, speech, SAMPLE_RATE, np.random.default_rng(1))
        assert np.isfinite(produced).all()

    @pytest.mark.parametrize("attack_id", sorted(ATTACKS))
    def test_energy_is_preserved(self, speech: NDArray[np.float64], attack_id: str) -> None:
        """The anti-cheat property. An attack that changes the level would let a
        detector separate the classes without learning anything about synthesis."""
        produced = apply_attack(attack_id, speech, SAMPLE_RATE, np.random.default_rng(1))
        assert _rms(produced) == pytest.approx(_rms(speech), rel=0.35)

    @pytest.mark.parametrize("attack_id", sorted(ATTACKS))
    def test_the_signal_is_actually_changed(
        self, speech: NDArray[np.float64], attack_id: str
    ) -> None:
        """A pass-through would be labelled spoofed and train the detector to
        call genuine speech synthetic."""
        produced = apply_attack(attack_id, speech, SAMPLE_RATE, np.random.default_rng(1))
        assert not np.allclose(produced, speech, atol=1e-6)

    @pytest.mark.parametrize("attack_id", sorted(ATTACKS))
    def test_generation_is_reproducible(
        self, speech: NDArray[np.float64], attack_id: str
    ) -> None:
        """Same seed, same attack: otherwise the model changes on every rerun."""
        first = apply_attack(attack_id, speech, SAMPLE_RATE, np.random.default_rng(5))
        second = apply_attack(attack_id, speech, SAMPLE_RATE, np.random.default_rng(5))
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("attack_id", sorted(ATTACKS))
    def test_a_signal_shorter_than_a_frame_is_refused(self, attack_id: str) -> None:
        with pytest.raises(InsufficientDataError):
            apply_attack(attack_id, np.zeros(100), SAMPLE_RATE, np.random.default_rng(0))


class TestTheAttacksDifferFromEachOther:
    """Cross-attack evaluation is meaningless if they all do the same thing."""

    def test_replacing_the_excitation_destroys_the_waveform(
        self, speech: NDArray[np.float64]
    ) -> None:
        for attack in (lpc_noise, lpc_pulse, phase_randomised):
            produced = attack(speech, SAMPLE_RATE, np.random.default_rng(3))
            correlation = float(np.corrcoef(speech, produced)[0, 1])
            assert abs(correlation) < 0.3, attack.__name__

    def test_smoothing_the_filter_preserves_the_waveform(
        self, speech: NDArray[np.float64]
    ) -> None:
        """The attack on a different axis. It keeps the recording's own residual,
        so what it is detectable by cannot be the excitation — which is what
        stops a detector passing the cross-attack test on one trick."""
        produced = oversmoothed(speech, SAMPLE_RATE)
        correlation = float(np.corrcoef(speech, produced)[0, 1])
        assert correlation > 0.5

    def test_the_attacks_are_not_each_other(self, speech: NDArray[np.float64]) -> None:
        produced = {
            attack_id: apply_attack(
                attack_id, speech, SAMPLE_RATE, np.random.default_rng(4)
            )
            for attack_id in sorted(ATTACKS)
        }
        identifiers = sorted(produced)
        for index, first in enumerate(identifiers):
            for second in identifiers[index + 1 :]:
                assert not np.allclose(produced[first], produced[second], atol=1e-6)


class TestPhaseRandomisation:
    def test_the_frame_magnitude_spectrum_is_preserved(
        self, speech: NDArray[np.float64]
    ) -> None:
        """The point of this attack, asserted at the level features are computed.

        Comparison is per frame rather than over the whole signal. Overlap-adding
        frames whose phase has been randomised makes them sum incoherently, so
        the *global* spectrum of the output differs a great deal — correlation
        around 0.25 — while the frame spectra a front-end actually sees are
        nearly identical. Testing the whole-signal transform would assert
        something false about a signal that behaves exactly as intended.
        """
        produced = phase_randomised(speech, SAMPLE_RATE, np.random.default_rng(2))

        def mean_frame_spectrum(signal: NDArray[np.float64]) -> NDArray[np.float64]:
            length, hop = 400, 200
            n_frames = 1 + (signal.size - length) // hop
            window = np.hanning(length)
            frames = np.stack(
                [signal[i * hop : i * hop + length] * window for i in range(n_frames)]
            )
            return np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)

        correlation = float(
            np.corrcoef(mean_frame_spectrum(speech), mean_frame_spectrum(produced))[0, 1]
        )
        assert correlation > 0.9

    def test_it_produces_a_real_signal(self, speech: NDArray[np.float64]) -> None:
        """DC and Nyquist must keep zero phase, or the inverse transform is
        complex and the imaginary part is silently discarded."""
        produced = phase_randomised(speech, SAMPLE_RATE, np.random.default_rng(2))
        assert np.isrealobj(produced)


class TestDeterministicAttacks:
    """Two of the four draw nothing, and take a generator only for uniformity."""

    def test_pulse_excitation_needs_no_generator(self, speech: NDArray[np.float64]) -> None:
        assert np.array_equal(
            lpc_pulse(speech, SAMPLE_RATE), lpc_pulse(speech, SAMPLE_RATE)
        )

    def test_oversmoothing_needs_no_generator(self, speech: NDArray[np.float64]) -> None:
        assert np.array_equal(
            oversmoothed(speech, SAMPLE_RATE), oversmoothed(speech, SAMPLE_RATE)
        )

    def test_a_generator_does_not_change_them(self, speech: NDArray[np.float64]) -> None:
        assert np.array_equal(
            lpc_pulse(speech, SAMPLE_RATE, np.random.default_rng(1)),
            lpc_pulse(speech, SAMPLE_RATE, np.random.default_rng(999)),
        )
