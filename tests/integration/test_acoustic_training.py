"""Training the acoustic stack, including from two separate corpora.

The stack's unsupervised stages — the UBM and the total variability matrix —
need audio, and its supervised stages — LDA and PLDA — need speakers. Those are
different requirements and :meth:`SpeakerComparisonSystem.train` accepts a
separate corpus for each, so the front-end can be estimated on material matched
to the deployment population while the back-end borrows its speaker subspace
from wherever enough speakers exist.

The fixtures synthesise voices rather than loading audio. Noise will not do:
the front-end's voice activity detector rejects frames flatter than a threshold
precisely so that stationary noise cannot be mistaken for speech, so a corpus of
white noise trains nothing because every frame is discarded before the UBM sees
it. What is synthesised here is a harmonic source through formant resonances,
under a syllable-rate envelope with pauses — peaked enough in frequency and
varied enough in time to survive the front-end for the reason real speech does.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from viflap.analysis.speaker.gmm import GmmConfig
from viflap.analysis.speaker.ivector import TotalVariabilityConfig
from viflap.analysis.speaker.pipeline import (
    FrontEndConfig,
    SpeakerComparisonSystem,
    SpeakerSystemConfig,
    TrainingRecording,
    _evenly_spaced,
)
from viflap.analysis.speaker.plda import PldaConfig
from viflap.domain.errors import InsufficientDataError

SAMPLE_RATE = 16_000
SECONDS = 4.0


def _resonance(
    frequency: float, bandwidth: float, length: int = 512
) -> NDArray[np.float64]:
    """Impulse response of a single formant, as a decaying sinusoid."""
    n = np.arange(length)
    decay = np.exp(-np.pi * bandwidth * n / SAMPLE_RATE)
    return decay * np.sin(2.0 * np.pi * frequency * n / SAMPLE_RATE)


def _synthetic_voice(
    f0: float, formants: Sequence[float], rng: np.random.Generator
) -> NDArray[np.float64]:
    """A voice-like signal: harmonic source, formant filter, syllabic envelope."""
    n_samples = int(SECONDS * SAMPLE_RATE)
    t = np.arange(n_samples) / SAMPLE_RATE

    # Harmonic source. Falling harmonic amplitudes keep the spectrum peaked at
    # low frequencies, which is what makes the flatness measure accept it.
    source = np.zeros(n_samples)
    for harmonic in range(1, 30):
        if harmonic * f0 >= SAMPLE_RATE / 2:
            break
        source += np.sin(2.0 * np.pi * harmonic * f0 * t) / harmonic

    impulse = np.zeros(512)
    for frequency in formants:
        impulse += _resonance(frequency, bandwidth=90.0)
    voiced = np.convolve(source, impulse, mode="same")

    # Syllable-rate envelope with two pauses. The pauses matter: the detector
    # estimates its noise floor from a low percentile of frame energy, and a
    # signal with no quiet frames gives it nothing to estimate from.
    envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.5 * t)) ** 2
    envelope[int(0.20 * n_samples) : int(0.28 * n_samples)] = 0.0
    envelope[int(0.62 * n_samples) : int(0.70 * n_samples)] = 0.0

    signal = voiced * envelope
    signal += rng.normal(0.0, 1e-3, n_samples)
    return signal / (np.max(np.abs(signal)) + 1e-12)


def _speaker_voice(speaker: int, take: int) -> NDArray[np.float64]:
    """One recording of one speaker. Timbre follows the speaker, not the take."""
    rng = np.random.default_rng(1000 * speaker + take)
    f0 = 95.0 + 11.0 * speaker
    formants = (
        520.0 + 45.0 * speaker,
        1420.0 + 80.0 * speaker,
        2530.0 + 60.0 * speaker,
    )
    # Small per-take jitter so repeats of a speaker are not identical signals,
    # which would leave the within-speaker covariance singular.
    jitter = 1.0 + 0.02 * ((take % 3) - 1)
    return _synthetic_voice(f0 * jitter, formants, rng)


def _labelled_corpus(n_speakers: int = 8, per_speaker: int = 3) -> list[TrainingRecording]:
    return [
        TrainingRecording(
            signal=_speaker_voice(speaker, take),
            sample_rate=SAMPLE_RATE,
            speaker_id=f"spk{speaker:02d}",
            recording_id=f"spk{speaker:02d}-{take}",
        )
        for speaker in range(n_speakers)
        for take in range(per_speaker)
    ]


def _background_corpus(offset: int, size: int = 12) -> list[TrainingRecording]:
    """Unlabelled material. The speaker id is deliberately uninformative."""
    return [
        TrainingRecording(
            signal=_speaker_voice(offset + index, take=7),
            sample_rate=SAMPLE_RATE,
            speaker_id="unlabelled",
            recording_id=f"bg{offset}-{index}",
        )
        for index in range(size)
    ]


class _LazyCorpus(Sequence[TrainingRecording]):
    """A background corpus that loads on access and counts the loads.

    Stands in for the real case, where the pool is far too large to materialise
    and the sequence reads audio from disk per item.
    """

    def __init__(self, builders: Sequence[Callable[[], TrainingRecording]]) -> None:
        self._builders = list(builders)
        self.loads: Counter[int] = Counter()

    def __len__(self) -> int:
        return len(self._builders)

    def __getitem__(self, index: int) -> TrainingRecording:  # type: ignore[override]
        # Counted after the access, not before: the Sequence mixin's __iter__
        # walks indices until one raises IndexError, so incrementing first
        # would record a load for the index one past the end.
        item = self._builders[index]()
        self.loads[index] += 1
        return item


def _tiny_config(**overrides: object) -> SpeakerSystemConfig:
    """Small enough to train in seconds, large enough to exercise every stage."""
    defaults: dict[str, object] = {
        "ubm": GmmConfig(n_components=4, max_iterations=5),
        "total_variability": TotalVariabilityConfig(rank=8, max_iterations=3),
        "plda": PldaConfig(min_speakers=6, max_iterations=5),
    }
    defaults.update(overrides)
    return SpeakerSystemConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def labelled() -> list[TrainingRecording]:
    return _labelled_corpus()


class TestSyntheticCorpus:
    """The fixture has to survive the front-end or every test below is vacuous."""

    def test_synthetic_voice_survives_voice_activity_detection(
        self, labelled: list[TrainingRecording]
    ) -> None:
        config = _tiny_config()
        features, _ = config.front_end.extract_features(
            labelled[0].signal, labelled[0].sample_rate
        )
        assert features.shape[0] > 100
        assert features.shape[1] == 60


class TestSingleCorpusTraining:
    def test_trains_and_reports_no_background(
        self, labelled: list[TrainingRecording]
    ) -> None:
        system = SpeakerComparisonSystem.train(labelled, _tiny_config())

        summary = dict(system.describe())
        assert summary["n_training_speakers"] == 8.0
        assert summary["n_training_recordings"] == 24.0
        assert summary["n_background_recordings"] == 0.0

    def test_training_is_deterministic(self, labelled: list[TrainingRecording]) -> None:
        first = SpeakerComparisonSystem.train(labelled, _tiny_config())
        second = SpeakerComparisonSystem.train(labelled, _tiny_config())
        assert first.model_id == second.model_id


class TestBackgroundCorpusTraining:
    def test_background_changes_the_model(self, labelled: list[TrainingRecording]) -> None:
        """The background corpus must actually reach the unsupervised stages."""
        without = SpeakerComparisonSystem.train(labelled, _tiny_config())
        with_background = SpeakerComparisonSystem.train(
            labelled, _tiny_config(), background=_background_corpus(offset=50)
        )
        assert without.model_id != with_background.model_id

    def test_different_backgrounds_give_different_ubms(
        self, labelled: list[TrainingRecording]
    ) -> None:
        """Same labelled corpus, different background, different front-end."""
        first = SpeakerComparisonSystem.train(
            labelled, _tiny_config(), background=_background_corpus(offset=50)
        )
        second = SpeakerComparisonSystem.train(
            labelled, _tiny_config(), background=_background_corpus(offset=200)
        )
        assert first.model_id != second.model_id

    def test_background_speaker_labels_are_ignored(
        self, labelled: list[TrainingRecording]
    ) -> None:
        """Every background recording shares one id; the speaker count is unaffected."""
        system = SpeakerComparisonSystem.train(
            labelled, _tiny_config(), background=_background_corpus(offset=50)
        )
        summary = dict(system.describe())

        assert summary["n_training_speakers"] == 8.0
        assert summary["plda_n_training_speakers"] == 8.0
        assert summary["n_background_recordings"] == 12.0

    def test_background_training_is_deterministic(
        self, labelled: list[TrainingRecording]
    ) -> None:
        first = SpeakerComparisonSystem.train(
            labelled, _tiny_config(), background=_background_corpus(offset=50)
        )
        second = SpeakerComparisonSystem.train(
            labelled, _tiny_config(), background=_background_corpus(offset=50)
        )
        assert first.model_id == second.model_id

    def test_recording_budget_caps_statistics(
        self, labelled: list[TrainingRecording]
    ) -> None:
        system = SpeakerComparisonSystem.train(
            labelled,
            _tiny_config(
                # Below the rank the total variability matrix refuses to train,
                # so the budget has to clear it for this test to be about the
                # budget rather than about that guard.
                total_variability=TotalVariabilityConfig(rank=3, max_iterations=3),
                background_recording_budget=5,
            ),
            background=_background_corpus(offset=50, size=12),
        )
        assert dict(system.describe())["n_background_recordings"] == 5.0

    def test_recording_budget_below_the_rank_is_refused(
        self, labelled: list[TrainingRecording]
    ) -> None:
        """Capping below the subspace dimension is an error, not a silent fit."""
        with pytest.raises(InsufficientDataError):
            SpeakerComparisonSystem.train(
                labelled,
                _tiny_config(background_recording_budget=5),
                background=_background_corpus(offset=50, size=12),
            )

    def test_background_is_read_twice_and_audio_is_not_retained(
        self, labelled: list[TrainingRecording]
    ) -> None:
        """Two passes, by design: statistics need a UBM that pass one produces.

        Guards the memory contract. Holding the features from the first pass
        would make the second unnecessary, and would also make the pool's size
        the peak memory of a training run — which is the thing this API exists
        to avoid.
        """
        corpus = _background_corpus(offset=50, size=6)
        lazy = _LazyCorpus([lambda item=item: item for item in corpus])  # type: ignore[misc]

        SpeakerComparisonSystem.train(
            labelled,
            _tiny_config(
                total_variability=TotalVariabilityConfig(rank=3, max_iterations=3)
            ),
            background=lazy,
        )

        assert set(lazy.loads) == set(range(6))
        assert set(lazy.loads.values()) == {2}

    def test_embedding_uses_the_background_trained_front_end(
        self, labelled: list[TrainingRecording]
    ) -> None:
        """A trained system still scores, and stamps its own id on the result."""
        system = SpeakerComparisonSystem.train(
            labelled, _tiny_config(), background=_background_corpus(offset=50)
        )
        embedding = system.embed(labelled[0].signal, SAMPLE_RATE)

        assert embedding.model_id == system.model_id
        assert np.isfinite(embedding.vector).all()


class TestArchiveRoundTrip:
    """A model that cannot be reloaded is not a model, and this failed silently.

    ``training_speakers`` was written as a numpy object array, which numpy will
    only read back by unpickling — and :meth:`load` reads with
    ``allow_pickle=False``, as anything loading a file from disk should. So
    every archive that recorded its training speakers refused to load, and the
    leakage check those speakers exist for could not run on any model that had
    them. Nothing caught it because no test saved a model and loaded it again.
    """

    def test_a_saved_model_loads_and_keeps_its_training_speakers(
        self, labelled: list[TrainingRecording], tmp_path: Path
    ) -> None:
        system = SpeakerComparisonSystem.train(labelled, _tiny_config())
        assert system.training_speakers, "the fixture should have labelled speakers"

        path = tmp_path / "model.npz"
        system.save(path)
        reloaded = SpeakerComparisonSystem.load(path)

        assert reloaded.model_id == system.model_id
        assert reloaded.training_speakers == system.training_speakers
        assert reloaded.config.front_end.sliding_cmvn_frames == (
            system.config.front_end.sliding_cmvn_frames
        )

    def test_a_non_default_normalisation_window_survives_the_archive(
        self, labelled: list[TrainingRecording], tmp_path: Path
    ) -> None:
        """The window is part of the front-end a model was fitted to.

        Losing it on save would leave the model scoring through a front-end it
        never saw, which produces numbers rather than an error.
        """
        system = SpeakerComparisonSystem.train(
            labelled, _tiny_config(front_end=FrontEndConfig(sliding_cmvn_frames=0))
        )
        path = tmp_path / "utterance-cmvn.npz"
        system.save(path)

        reloaded = SpeakerComparisonSystem.load(path)
        assert reloaded.config.front_end.sliding_cmvn_frames == 0
        assert reloaded.model_id == system.model_id


class TestEvenlySpaced:
    def test_returns_everything_when_under_budget(self) -> None:
        assert _evenly_spaced(5, 10) == [0, 1, 2, 3, 4]
        assert _evenly_spaced(5, None) == [0, 1, 2, 3, 4]

    def test_spreads_across_the_range_rather_than_taking_a_prefix(self) -> None:
        selected = _evenly_spaced(100, 5)

        assert len(selected) == 5
        assert selected[0] == 0
        assert selected[-1] == 99
        # A prefix would put everything in the first tenth. The point of the
        # even spacing is that a corpus grouped by speaker or language does not
        # contribute only its opening group.
        assert max(selected) > 50

    def test_is_deterministic(self) -> None:
        assert _evenly_spaced(100, 7) == _evenly_spaced(100, 7)
