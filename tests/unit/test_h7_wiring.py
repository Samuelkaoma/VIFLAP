"""Carrying the speaker through to H7, which is what the interval rests on.

H7 asks whether a countermeasure detects synthesis it has never seen well
enough to gate evidence, and its decision rule is on the **upper bound** of
``C_llr``. An upper bound needs a resampling unit, and the unit is the speaker:
two examples from one person are not independent evidence about how the
detector generalises, and a bootstrap that treats them as though they were
reports an interval that is too narrow in the direction that supports the
hypothesis.

``TrainingExample`` carried ``attack_id`` for exactly this reason and did not
carry ``speaker_id``, so the cross-attack evaluation could report an equal error
rate and nothing with an interval on it. These tests hold the thread from the
corpus to the bootstrap: the speaker must survive attack generation, survive
channel degradation, and stay aligned with the scores when the detector refuses
a recording — which is the one place it could go wrong invisibly, because a
refusal drops a score without dropping anything else unless they are built
together.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.train_countermeasure import (
    build_examples,
    degrade_examples,
    score_examples,
)
from viflap.analysis.channel.degradation import DegradationCondition
from viflap.analysis.spoof.countermeasure import TrainingExample
from viflap.domain.errors import InsufficientDataError
from viflap.evaluation.hypotheses import H7SyntheticGating

RATE = 16000


class _Recording:
    """Enough of a ``Recording`` for example building."""

    def __init__(self, index: int) -> None:
        rng = np.random.default_rng(index)
        self.signal = rng.normal(0.0, 0.1, RATE)
        self.sample_rate = RATE
        self.speaker_id = f"spk{index:03d}"
        self.recording_id = f"spk{index:03d}-0-r0"


class _Detector:
    """Scores by speaker so the alignment is checkable, and refuses on demand."""

    def __init__(self, refuse: set[str] | None = None) -> None:
        self.refuse = refuse or set()

    def score(self, signal: np.ndarray, sample_rate: int):
        marker = float(signal[0])
        if marker in {float(x) for x in self.refuse}:
            raise InsufficientDataError("refused")

        class _R:
            log_likelihood_ratio = marker

        return _R()


def _marked(examples, value_of):
    """Rewrite each signal so its first sample identifies the example."""
    return [
        TrainingExample(
            signal=np.full(RATE, float(value_of(e))),
            sample_rate=e.sample_rate,
            is_bona_fide=e.is_bona_fide,
            attack_id=e.attack_id,
            speaker_id=e.speaker_id,
            condition=e.condition,
        )
        for e in examples
    ]


class TestTheSpeakerSurvivesTheJourney:
    def test_build_examples_puts_the_speaker_on_both_classes(self) -> None:
        """Genuine and spoofed derive from one recording and one person."""
        examples = build_examples([_Recording(1), _Recording(2)], ["lpc_noise"])
        assert examples
        assert all(e.speaker_id for e in examples)
        genuine = {e.speaker_id for e in examples if e.is_bona_fide}
        spoofed = {e.speaker_id for e in examples if not e.is_bona_fide}
        assert genuine == spoofed == {"spk001", "spk002"}

    def test_degradation_does_not_drop_the_speaker(self) -> None:
        """``degrade_examples`` rebuilds each example, so it can lose fields."""
        examples = build_examples([_Recording(3)], ["lpc_noise"])
        degraded = degrade_examples(
            examples, [DegradationCondition(bitrate_kbps=12.20)], seed=1
        )
        assert [e.speaker_id for e in degraded] == [e.speaker_id for e in examples]
        assert all(e.condition for e in degraded)


class TestScoresAndSpeakersStayAligned:
    def test_speakers_come_back_one_per_score(self) -> None:
        examples = _marked(
            build_examples([_Recording(i) for i in range(3)], ["lpc_noise"]),
            lambda e: int(e.speaker_id[-3:]),
        )
        scores, labels, speakers = score_examples(_Detector(), examples)
        assert len(speakers) == scores.size == labels.size == len(examples)

    def test_a_refusal_drops_the_speaker_with_its_score(self) -> None:
        """The failure this guards against is silent: a speaker list built
        separately would keep an entry the score list no longer has, shifting
        every speaker after the refused recording onto the wrong trial."""
        examples = _marked(
            build_examples([_Recording(i) for i in range(4)], ["lpc_noise"]),
            lambda e: int(e.speaker_id[-3:]),
        )
        detector = _Detector(refuse={2})
        scores, labels, speakers = score_examples(detector, examples)

        assert labels.size == scores.size
        assert len(speakers) == scores.size < len(examples)
        assert "spk002" not in speakers
        # Every surviving score still identifies the speaker it is filed under.
        for score, speaker in zip(scores, speakers, strict=True):
            assert int(speaker[-3:]) == int(score)


class TestTheDecisionRule:
    def test_h7_refuses_to_run_without_held_out_attacks(self) -> None:
        """Evaluating on seen attacks reports the flattering number, and the
        rule says so rather than letting a caller ask for it."""
        rng = np.random.default_rng(0)
        scores = rng.normal(0.0, 1.0, 40)
        labels = np.array([1, 0] * 20, dtype=np.int64)
        speakers = [f"s{i // 2}" for i in range(40)]
        with pytest.raises(InsufficientDataError, match="excluded from training"):
            H7SyntheticGating().run(scores, labels, speakers, [], n_resamples=20)

    def test_a_detector_at_chance_is_not_supported(self) -> None:
        """§10's phase-randomised family sits at chance; the rule must not
        return `supported` for it however the interval falls."""
        rng = np.random.default_rng(1)
        scores = rng.normal(0.0, 0.01, 120)
        labels = np.array([1, 0] * 60, dtype=np.int64)
        speakers = [f"s{i // 2}" for i in range(120)]
        outcome = H7SyntheticGating().run(
            scores, labels, speakers, ["phase_randomised"], n_resamples=50
        )
        assert not outcome.supported
        assert outcome.hypothesis == "H7"
        assert "phase_randomised" in outcome.statement
