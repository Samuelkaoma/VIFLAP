"""Training the countermeasure, and the memory bound that makes it possible.

Training scores every pooled training frame against both class models to fix the
out-of-domain threshold. That call holds an ``n_frames x n_components`` array and
``logsumexp`` keeps several copies of it, so at realistic scale — 1.8 million
frames at 64 components — it asks for 877 MB per copy and the run dies part-way
through with an allocation error. It did exactly that here before the work was
chunked.

The property that makes chunking safe is that the quantity is a percentile over
frames and frames are independent, so block size cannot change the answer. That
is what these tests pin: the same data trained at wildly different block sizes
must produce the *same model*, not merely a similar one.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from viflap.analysis.spoof.attacks import apply_attack
from viflap.analysis.spoof.countermeasure import (
    CountermeasureConfig,
    SpoofingCountermeasure,
    TrainingExample,
)
from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError

SAMPLE_RATE = 16_000


def _voice(seed: int, seconds: float = 2.0) -> NDArray[np.float64]:
    """A voice-like signal, distinct per seed."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = 95.0 + 7.0 * (seed % 11)

    source = np.zeros(n)
    for harmonic in range(1, 26):
        if harmonic * f0 >= SAMPLE_RATE / 2:
            break
        source += np.sin(2.0 * np.pi * harmonic * f0 * t) / harmonic

    impulse = np.zeros(512)
    steps = np.arange(512)
    for offset, frequency in enumerate((640.0, 1500.0, 2500.0)):
        shifted = frequency + 40.0 * (seed % 7) * (offset + 1)
        impulse += np.exp(-np.pi * 95.0 * steps / SAMPLE_RATE) * np.sin(
            2.0 * np.pi * shifted * steps / SAMPLE_RATE
        )

    voiced = np.convolve(source, impulse, mode="same")
    envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.2 * t)) ** 2
    signal = voiced * envelope + rng.normal(0.0, 1e-4, n)
    return signal / (np.max(np.abs(signal)) + 1e-12)


@pytest.fixture(scope="module")
def examples() -> list[TrainingExample]:
    """Genuine speech plus two attack families derived from it."""
    built: list[TrainingExample] = []
    for seed in range(6):
        signal = _voice(seed)
        built.append(
            TrainingExample(
                signal=signal,
                sample_rate=SAMPLE_RATE,
                is_bona_fide=True,
                attack_id="none",
            )
        )
        for attack_id in ("lpc_noise", "lpc_pulse"):
            built.append(
                TrainingExample(
                    signal=apply_attack(
                        attack_id, signal, SAMPLE_RATE, np.random.default_rng(seed)
                    ),
                    sample_rate=SAMPLE_RATE,
                    is_bona_fide=False,
                    attack_id=attack_id,
                )
            )
    return built


def _config(chunk_frames: int) -> CountermeasureConfig:
    return CountermeasureConfig(
        n_components=4, max_iterations=8, min_frames=20, chunk_frames=chunk_frames
    )


class TestChunkingDoesNotChangeTheModel:
    def test_block_size_leaves_the_model_identical(
        self, examples: list[TrainingExample]
    ) -> None:
        """The detector id is a content hash, so equality here is exact.

        Chunking bounds memory over a percentile computed across independent
        frames. If block size moved the answer at all, the quantity would depend
        on how the work happened to be divided, and every trained model would be
        an artefact of the machine it was trained on.
        """
        small = SpoofingCountermeasure.train(examples, _config(64))
        large = SpoofingCountermeasure.train(examples, _config(10_000_000))

        assert small.detector_id == large.detector_id

    def test_a_block_size_of_one_still_works(self, examples: list[TrainingExample]) -> None:
        """The degenerate bound. Slow, but it must not be a special case."""
        single = SpoofingCountermeasure.train(examples, _config(1))
        default = SpoofingCountermeasure.train(examples, _config(10_000_000))

        assert single.detector_id == default.detector_id

    def test_a_block_size_below_one_is_refused(self) -> None:
        with pytest.raises(InvalidEvidenceError):
            CountermeasureConfig(chunk_frames=0)


class TestTrainingRequirements:
    def test_both_classes_are_required(self, examples: list[TrainingExample]) -> None:
        """A one-class model ranks recordings by how ordinary they are, which is
        not the same question and would still return a number."""
        genuine_only = [e for e in examples if e.is_bona_fide]

        with pytest.raises(InsufficientDataError):
            SpoofingCountermeasure.train(genuine_only, _config(64))

    def test_an_empty_training_set_is_refused(self) -> None:
        with pytest.raises(InsufficientDataError):
            SpoofingCountermeasure.train([], _config(64))

    def test_the_attack_count_is_recorded(self, examples: list[TrainingExample]) -> None:
        """Cross-attack evaluation is impossible without knowing how many
        families the model saw, and a model trained on one is not one whose
        generalisation has been tested."""
        trained = SpoofingCountermeasure.train(examples, _config(64))
        assert trained.training_summary["n_attack_types"] == 2.0

    def test_a_mel_filterbank_is_refused(self) -> None:
        """Mel spacing averages away the upper-spectrum artefacts that
        distinguish synthetic speech, so the configuration forbids it."""
        from viflap.analysis.dsp.spectral import CepstralConfig

        with pytest.raises(InvalidEvidenceError):
            CountermeasureConfig(cepstral=CepstralConfig.for_speaker_recognition())


class TestScoring:
    def test_a_trained_model_separates_its_own_training_classes(
        self, examples: list[TrainingExample]
    ) -> None:
        """Not a performance claim — a wiring check. If the sign were inverted
        every downstream verdict would be backwards, and the gate would admit
        synthetic speech and exclude genuine."""
        trained = SpoofingCountermeasure.train(examples, _config(64))

        genuine = [
            trained.score(e.signal, e.sample_rate).log_likelihood_ratio
            for e in examples
            if e.is_bona_fide
        ]
        spoofed = [
            trained.score(e.signal, e.sample_rate).log_likelihood_ratio
            for e in examples
            if not e.is_bona_fide
        ]

        assert np.mean(genuine) > np.mean(spoofed)

    def test_a_recording_too_short_to_judge_is_refused(
        self, examples: list[TrainingExample]
    ) -> None:
        trained = SpoofingCountermeasure.train(examples, _config(64))

        with pytest.raises(InsufficientDataError):
            trained.score(np.zeros(2000), SAMPLE_RATE)
