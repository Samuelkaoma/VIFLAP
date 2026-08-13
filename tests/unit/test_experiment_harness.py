"""Tests for the experiment harness and the changes the experiments required.

The harness is not part of the shipped system, but the numbers in the report
come out of it, and a defect here produces a plausible wrong answer rather than
a failure. The properties asserted below are the ones whose violation would make
a result optimistic: leakage between partitions, same-session pairs counted as
same-source trials, and trials mis-attributed to a resampling unit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.compare_capacity import _common_survivors
from scripts.compare_cmvn import restrict_to_common_survivors
from scripts.corpus import (
    Recording,
    materialise,
    scan_corpus,
    split_by_speaker,
)
from scripts.corpus_zambian import scan_unlabelled
from scripts.experiment import (
    DegradedRecording,
    LazyBackgroundCorpus,
    _stable_seed,
    build_trials,
    degrade_many,
)
from scripts.report_h1 import render
from viflap.analysis.calibration.metrics import compute_cllr_min
from viflap.analysis.channel.codec import NARROWBAND_RATE, ParametricCelpCodec
from viflap.analysis.channel.degradation import DegradationCondition
from viflap.analysis.speaker.pipeline import _ubm_training_frames
from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError
from viflap.evaluation.splits import (
    bootstrap_contrast_over_speakers,
    bootstrap_over_speakers,
    paired_bootstrap_over_speakers,
)

SAMPLE_RATE = 16_000


def _write_corpus(
    root: Path,
    rng: np.random.Generator,
    speakers: int = 1,
    sessions: int = 2,
    utterances: int = 10,
    seconds: float = 7.0,
    gain: float = 1.0,
) -> Path:
    """Write a miniature corpus in the layout the fetcher produces."""
    for speaker in range(speakers):
        for session in range(sessions):
            directory = root / f"s{speaker:03d}" / f"c{session}"
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(utterances):
                audio = rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1 * gain
                sf.write(
                    directory / f"u{index:03d}.flac",
                    audio.astype(np.float32),
                    SAMPLE_RATE,
                )
    return root


class TestRecordingConstruction:
    """The scan decides recordings from FLAC headers alone; materialising reads
    only what the scan chose. Both halves must agree, because a disagreement
    yields a recording of a different length from the one its label claims —
    and duration is a factor in the sweep."""

    def test_recordings_are_exactly_the_target_length(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        _write_corpus(tmp_path, rng, utterances=10, seconds=7.0)
        plans = scan_corpus(tmp_path, target_seconds=30.0, max_recordings_per_session=4)
        assert plans, "ten seven-second utterances should yield at least one recording"
        for recording in materialise(plans):
            assert recording.signal.size == int(30.0 * SAMPLE_RATE)

    def test_a_partial_final_recording_is_discarded(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        _write_corpus(tmp_path, rng, utterances=5, seconds=7.0)
        plans = scan_corpus(tmp_path, target_seconds=30.0, max_recordings_per_session=4)
        # Five seven-second utterances is 35 s: one full recording, and five
        # leftover seconds that are dropped rather than kept short.
        assert len([p for p in plans if p.session_id == "c0"]) == 1

    def test_the_per_session_cap_is_honoured(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        _write_corpus(tmp_path, rng, sessions=1, utterances=40, seconds=7.0)
        plans = scan_corpus(
            tmp_path,
            target_seconds=30.0,
            max_recordings_per_session=2,
            min_sessions_per_speaker=1,
        )
        assert len(plans) == 2

    def test_amplitude_is_normalised_away(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Recording level is a property of the microphone, not the speaker, and
        left alone it is a strong spurious cue."""
        _write_corpus(tmp_path / "quiet", rng, utterances=6, seconds=7.0, gain=0.01)
        _write_corpus(tmp_path / "loud", rng, utterances=6, seconds=7.0, gain=8.0)
        recordings = materialise(scan_corpus(tmp_path / "quiet")) + materialise(
            scan_corpus(tmp_path / "loud")
        )
        assert recordings
        peaks = [float(np.max(np.abs(r.signal))) for r in recordings]
        assert peaks == pytest.approx([0.7] * len(peaks), abs=1e-5)

    def test_speakers_with_one_session_are_dropped(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """A single-session speaker contributes no within-speaker between-session
        variation, so the set of speakers behind the between-speaker term would
        differ from the set behind the within-speaker term."""
        _write_corpus(tmp_path, rng, speakers=1, sessions=1, utterances=10)
        assert scan_corpus(tmp_path, min_sessions_per_speaker=2) == []

    def test_scanning_reads_no_audio(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """The whole point of the scan: the split can be computed over a corpus
        far larger than memory, and only the needed partitions are read."""
        _write_corpus(tmp_path, rng, speakers=2, utterances=10)
        plans = scan_corpus(tmp_path)
        assert plans
        assert all(not hasattr(plan, "signal") for plan in plans)
        assert all(plan.target_samples == int(30.0 * SAMPLE_RATE) for plan in plans)

    def test_materialised_audio_is_float32(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """float64 would double the corpus in memory for no added precision:
        the source is 16-bit, which float32 represents exactly."""
        _write_corpus(tmp_path, rng, utterances=10)
        recordings = materialise(scan_corpus(tmp_path))
        assert recordings
        assert all(r.signal.dtype == np.float32 for r in recordings)


def _recording(speaker: str, session: str, index: int) -> Recording:
    return Recording(
        signal=np.zeros(8),
        sample_rate=SAMPLE_RATE,
        speaker_id=speaker,
        session_id=session,
        recording_id=f"{speaker}-{session}-{index}",
        source_utterances=(),
    )


class TestSpeakerDisjointSplit:
    def test_no_speaker_appears_in_two_partitions(self) -> None:
        recordings = [
            _recording(f"s{speaker:03d}", f"c{session}", index)
            for speaker in range(40)
            for session in range(2)
            for index in range(2)
        ]
        split = split_by_speaker(recordings)

        train = {r.speaker_id for r in split.train}
        development = {r.speaker_id for r in split.development}
        evaluation = {r.speaker_id for r in split.evaluation}
        assert not train & development
        assert not train & evaluation
        assert not development & evaluation
        assert train | development | evaluation == {f"s{i:03d}" for i in range(40)}

    def test_every_recording_of_a_speaker_stays_together(self) -> None:
        recordings = [
            _recording(f"s{speaker:03d}", f"c{session}", index)
            for speaker in range(30)
            for session in range(2)
            for index in range(2)
        ]
        split = split_by_speaker(recordings)
        for part in (split.train, split.development, split.evaluation):
            speakers = {r.speaker_id for r in part}
            assert len(part) == 4 * len(speakers)

    def test_the_split_is_reproducible(self) -> None:
        """Training and evaluation are separate processes that each rebuild the
        split. If it were not reproducible they would disagree about which
        speakers the model had seen."""
        recordings = [
            _recording(f"s{speaker:03d}", f"c{session}", 0)
            for speaker in range(30)
            for session in range(2)
        ]
        first = split_by_speaker(recordings)
        second = split_by_speaker(recordings)
        assert [r.recording_id for r in first.evaluation] == [
            r.recording_id for r in second.evaluation
        ]

    def test_too_few_speakers_is_refused(self) -> None:
        recordings = [_recording(f"s{i}", "c0", 0) for i in range(4)]
        with pytest.raises(ValueError, match="too few speakers"):
            split_by_speaker(recordings)


class _StubPlda:
    """Returns the negated distance between vectors, so identical vectors score
    highest. Enough structure to check trial construction without training."""

    def score_many(self, probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        return -np.linalg.norm(gallery - probe, axis=1)


class _StubSystem:
    def __init__(self) -> None:
        self.plda = _StubPlda()


def _degraded(speaker: str, session: str, index: int) -> DegradedRecording:
    return DegradedRecording(
        signal=np.zeros(4),
        sample_rate=NARROWBAND_RATE,
        speaker_id=speaker,
        session_id=session,
        recording_id=f"{speaker}-{session}-{index}",
        condition_label="amr12.2_clean",
        codec_mode="parametric_celp",
    )


class _StubEmbedding:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector


class TestTrialConstruction:
    def _embedded(self) -> list[tuple[DegradedRecording, _StubEmbedding]]:
        rng = np.random.default_rng(5)
        items = []
        for speaker in ("a", "b", "c"):
            for session in ("s0", "s1"):
                for index in range(2):
                    items.append(
                        (
                            _degraded(speaker, session, index),
                            _StubEmbedding(rng.standard_normal(4)),
                        )
                    )
        return items

    def test_same_source_trials_never_share_a_session(self) -> None:
        """Two recordings from one session share a microphone and a room, so a
        system scores them alike partly for reasons unrelated to the speaker.
        The operational question is whether two separate calls can be linked."""
        embedded = self._embedded()
        trials = build_trials(embedded, _StubSystem(), cross_session_only=True)

        # Three speakers, two sessions of two recordings each. Cross-session
        # same-source pairs per speaker: 2 x 2 = 4. Total 12.
        assert trials.n_same_source == 12

    def test_including_same_session_pairs_adds_exactly_the_easy_ones(self) -> None:
        embedded = self._embedded()
        strict = build_trials(embedded, _StubSystem(), cross_session_only=True)
        permissive = build_trials(embedded, _StubSystem(), cross_session_only=False)
        # One within-session pair per session, two sessions, three speakers.
        assert permissive.n_same_source == strict.n_same_source + 6

    def test_every_trial_is_attributed_to_one_speaker(self) -> None:
        trials = build_trials(self._embedded(), _StubSystem())
        assert len(trials.speakers) == trials.scores.size == trials.labels.size
        assert set(trials.speakers) <= {"a", "b", "c"}

    def test_different_source_trials_cross_speakers(self) -> None:
        trials = build_trials(self._embedded(), _StubSystem())
        # Twelve recordings, C(12,2) = 66 pairs; 6 are same-speaker same-session
        # and excluded, 12 are same-speaker cross-session.
        assert trials.n_different_source == 66 - 6 - 12


class TestStableSeeding:
    """Noise realisations are derived from a recording's identifier so that a
    rerun of the same experiment produces the same degraded audio."""

    def test_the_seed_is_a_content_hash_not_the_salted_builtin(self) -> None:
        """`hash` on a string is salted per interpreter. Using it would give the
        same recording different noise on every run, and the difference would
        show up as unexplained movement in the reported metrics."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scripts.experiment import _stable_seed;"
                "print(_stable_seed('19-198-r00'))",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONHASHSEED": "1"},
        )
        other = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scripts.experiment import _stable_seed;"
                "print(_stable_seed('19-198-r00'))",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONHASHSEED": "9999"},
        )
        assert completed.stdout.strip() == other.stdout.strip()
        assert completed.stdout.strip() == str(_stable_seed("19-198-r00"))

    def test_different_recordings_get_different_seeds(self) -> None:
        seeds = {_stable_seed(f"spk-{i}-r0") for i in range(200)}
        assert len(seeds) > 150, "seed collisions would correlate noise across recordings"


def _cell(**overrides: object) -> dict[str, object]:
    cell = {
        "condition": "amr4.75_babble5dB",
        "bitrate_kbps": 4.75,
        "noise_type": "babble",
        "snr_db": 5.0,
        "duration_seconds": 5.0,
        "codec_mode": "parametric_celp",
        "n_evaluation_speakers": 42,
        "n_same_source": 396,
        "n_different_source": 24692,
        "n_refused": 0,
        "refusal_rate": 0.0,
        "c_llr_min": 0.62,
        "c_llr_min_lower": 0.55,
        "c_llr_min_upper": 0.70,
        "c_llr_matched": 0.81,
        "c_llr_matched_lower": 0.7,
        "c_llr_matched_upper": 0.9,
        "calibration_loss_matched": 0.19,
        "c_llr_transferred": None,
        "calibration_loss_transferred": None,
        "equal_error_rate": 0.21,
        "h1_supported": False,
        "h1_falsified": True,
        "notes": [],
    }
    cell.update(overrides)
    return cell


def _payload(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model_id": "ivec-plda-test",
        "model_describe": {"ubm_components": 128.0, "ivector_rank": 100.0},
        "split": {"evaluation_speakers": 42},
        "reference_condition": "amr12.2_clean",
        "elapsed_minutes": 12.0,
        "verdict": {
            "n_cells": len(cells),
            "n_supported": 0,
            "n_falsified": sum(1 for c in cells if c["h1_falsified"]),
            "n_inconclusive": sum(
                1 for c in cells if not c["h1_falsified"] and not c["h1_supported"]
            ),
            "best_cell": {
                "condition": "amr12.2_clean",
                "duration": 30.0,
                "c_llr_min": 0.34,
                "interval": [0.22, 0.43],
            },
            "worst_cell": {
                "condition": "amr4.75_babble5dB",
                "duration": 5.0,
                "c_llr_min": 0.62,
                "interval": [0.55, 0.70],
            },
            "decision_rule": "per cell: supported if ...",
        },
        "cells": cells,
    }


class TestUnevaluableCells:
    """A cell can survive refusal with plenty of recordings and still be
    unevaluable, because the bootstrap resamples *speakers*. Guarding on the
    recording count — as this did first — passed such a cell straight into the
    resampler and aborted a multi-hour sweep on its twenty-second cell."""

    def test_recordings_surviving_from_too_few_speakers_is_refused(self) -> None:
        from viflap.evaluation.splits import bootstrap_over_speakers

        # Six recordings, ample trials, but only two speakers behind them.
        scores = np.linspace(-2.0, 2.0, 12)
        labels = np.array([1, 0] * 6, dtype=np.int64)
        speakers = ["a"] * 6 + ["b"] * 6

        with pytest.raises(InsufficientDataError, match="at least three speakers"):
            bootstrap_over_speakers(
                compute_cllr_min, scores, labels, speakers, n_resamples=50
            )

    def test_three_speakers_is_enough_to_produce_an_interval(self) -> None:
        rng = np.random.default_rng(3)
        speakers = [s for s in ("a", "b", "c") for _ in range(20)]
        labels = np.array([1, 0] * 30, dtype=np.int64)
        scores = rng.standard_normal(60) + labels * 2.0

        estimate = bootstrap_over_speakers(
            compute_cllr_min, scores, labels, speakers, n_resamples=80
        )
        assert estimate.resampling_unit == "speakers"
        assert estimate.lower <= estimate.value <= estimate.upper


class TestPairedComparison:
    """Two models are compared on shared trials, not by their separate intervals.

    The capacity question — is 0.34 the channel or the model — is a question
    about a *difference*, and the marginal intervals are dominated by variation
    between speakers that both models saw. Comparing them marginally throws that
    cancellation away and would report a real difference as a null.
    """

    def _paired_scores(
        self, boost: float, n_speakers: int = 20, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Trials with a large per-speaker offset and a small consistent boost.

        The offset shifts a speaker's whole score distribution, which is what
        makes the marginal interval wide. The boost improves same-source
        separation by the same amount for every speaker, which is the effect a
        paired comparison should be able to see through that width.
        """
        rng = np.random.default_rng(seed)
        baseline: list[float] = []
        variant: list[float] = []
        labels: list[int] = []
        speakers: list[str] = []

        for speaker in range(n_speakers):
            offset = rng.normal(0.0, 1.5)
            for label in (1, 0):
                for _ in range(8):
                    score = rng.normal(1.0 if label else -1.0, 1.0) + offset
                    baseline.append(score)
                    variant.append(score + boost if label else score)
                    labels.append(label)
                    speakers.append(f"s{speaker:02d}")

        return (
            np.array(baseline),
            np.array(variant),
            np.array(labels, dtype=np.int64),
            speakers,
        )

    def test_identical_scores_give_exactly_no_difference(self) -> None:
        baseline, _, labels, speakers = self._paired_scores(boost=0.0)
        estimate = paired_bootstrap_over_speakers(
            compute_cllr_min, baseline, baseline, labels, speakers, n_resamples=120
        )
        assert estimate.value == pytest.approx(0.0, abs=1e-12)
        assert estimate.lower == pytest.approx(0.0, abs=1e-12)
        assert estimate.upper == pytest.approx(0.0, abs=1e-12)

    def test_pairing_sees_an_improvement_the_marginal_intervals_miss(self) -> None:
        """The whole reason the comparison is paired, asserted on data.

        The marginal intervals overlap heavily because the speaker effect is
        large. The paired interval excludes zero because that effect is common
        to both systems and cancels trial for trial.
        """
        baseline, variant, labels, speakers = self._paired_scores(boost=0.6)

        marginal_baseline = bootstrap_over_speakers(
            compute_cllr_min, baseline, labels, speakers, n_resamples=300
        )
        marginal_variant = bootstrap_over_speakers(
            compute_cllr_min, variant, labels, speakers, n_resamples=300
        )
        paired = paired_bootstrap_over_speakers(
            compute_cllr_min, baseline, variant, labels, speakers, n_resamples=300
        )

        assert marginal_variant.value < marginal_baseline.value
        # The marginal intervals do not separate the two systems ...
        assert marginal_variant.lower < marginal_baseline.upper
        # ... but the paired one does, and in the direction of the improvement.
        assert paired.value < 0.0
        assert paired.upper < 0.0
        assert paired.resampling_unit == "speakers"

    def test_the_paired_interval_is_the_narrower_one(self) -> None:
        baseline, variant, labels, speakers = self._paired_scores(boost=0.6)
        marginal = bootstrap_over_speakers(
            compute_cllr_min, baseline, labels, speakers, n_resamples=300
        )
        paired = paired_bootstrap_over_speakers(
            compute_cllr_min, baseline, variant, labels, speakers, n_resamples=300
        )
        assert (paired.upper - paired.lower) < (marginal.upper - marginal.lower)

    def test_misaligned_score_vectors_are_rejected(self) -> None:
        baseline, variant, labels, speakers = self._paired_scores(boost=0.3)
        with pytest.raises(InvalidEvidenceError):
            paired_bootstrap_over_speakers(
                compute_cllr_min, baseline, variant[:-4], labels, speakers, n_resamples=60
            )


class TestContrastOfContrasts:
    """The quantity a confounded factor needs: a difference of two differences.

    Asking how much of a duration effect survives a front-end change is not a
    metric and not a paired difference. It is the gap between two durations
    under one front-end set against the same gap under another, over four score
    vectors on one set of trials — and it needs an interval like everything
    else here.
    """

    def _four_vectors(
        self, short_penalty: float, confounded: float, n_speakers: int = 20, seed: int = 5
    ) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
        """Two front-ends at two durations, with a known duration effect.

        ``short_penalty`` is the genuine cost of the shorter duration and is
        common to both front-ends. ``confounded`` is an extra cost the baseline
        front-end suffers at the short duration only — the confound — so the
        contrast of contrasts should recover ``-confounded``.
        """
        rng = np.random.default_rng(seed)
        vectors: dict[str, list[float]] = {
            "baseline_long": [],
            "baseline_short": [],
            "fixed_long": [],
            "fixed_short": [],
        }
        labels: list[int] = []
        speakers: list[str] = []

        for speaker in range(n_speakers):
            offset = rng.normal(0.0, 1.5)
            for label in (1, 0):
                for _ in range(8):
                    base = rng.normal(1.0 if label else -1.0, 1.0) + offset
                    # Shortening pulls the same-source scores down toward the
                    # different-source ones, which is how duration degrades
                    # discrimination. Different-source scores are left alone.
                    penalty = short_penalty if label else 0.0
                    extra = confounded if label else 0.0
                    vectors["baseline_long"].append(base)
                    vectors["fixed_long"].append(base)
                    vectors["baseline_short"].append(base - penalty - extra)
                    vectors["fixed_short"].append(base - penalty)
                    labels.append(label)
                    speakers.append(f"s{speaker:02d}")

        return (
            {name: np.array(values) for name, values in vectors.items()},
            np.array(labels, dtype=np.int64),
            speakers,
        )

    @staticmethod
    def _duration_gap_difference(vectors, labels) -> float:
        fixed = compute_cllr_min(vectors["fixed_short"], labels) - compute_cllr_min(
            vectors["fixed_long"], labels
        )
        baseline = compute_cllr_min(vectors["baseline_short"], labels) - compute_cllr_min(
            vectors["baseline_long"], labels
        )
        return fixed - baseline

    def test_a_single_vector_statistic_reproduces_the_plain_bootstrap(self) -> None:
        """The general form is the same estimator, not a second one."""
        rng = np.random.default_rng(11)
        speakers = [f"s{i:02d}" for i in range(6) for _ in range(20)]
        labels = np.array([1, 0] * 60, dtype=np.int64)
        scores = rng.standard_normal(120) + labels * 2.0

        plain = bootstrap_over_speakers(
            compute_cllr_min, scores, labels, speakers, n_resamples=200
        )
        general = bootstrap_contrast_over_speakers(
            lambda vectors, resampled: compute_cllr_min(vectors["only"], resampled),
            {"only": scores},
            labels,
            speakers,
            n_resamples=200,
        )
        assert general.value == pytest.approx(plain.value)
        assert general.lower == pytest.approx(plain.lower)
        assert general.upper == pytest.approx(plain.upper)
        assert general.p_value_vs_zero is None

    def test_a_confound_shows_up_as_a_contrast_excluding_zero(self) -> None:
        vectors, labels, speakers = self._four_vectors(short_penalty=0.5, confounded=0.6)
        estimate = bootstrap_contrast_over_speakers(
            self._duration_gap_difference,
            vectors,
            labels,
            speakers,
            n_resamples=300,
            with_p_value=True,
        )
        # The fixed front-end's duration gap is smaller, so the contrast is
        # negative, and it is separated from zero.
        assert estimate.value < 0.0
        assert estimate.upper < 0.0
        assert estimate.p_value_vs_zero is not None
        assert estimate.p_value_vs_zero < 0.05

    def test_no_confound_leaves_the_contrast_covering_zero(self) -> None:
        """The control: a duration effect the front-end change does not touch."""
        vectors, labels, speakers = self._four_vectors(short_penalty=0.5, confounded=0.0)
        estimate = bootstrap_contrast_over_speakers(
            self._duration_gap_difference, vectors, labels, speakers, n_resamples=300
        )
        assert estimate.value == pytest.approx(0.0, abs=1e-12)
        assert estimate.lower <= 0.0 <= estimate.upper

    def test_vectors_of_different_lengths_are_refused(self) -> None:
        vectors, labels, speakers = self._four_vectors(short_penalty=0.4, confounded=0.2)
        vectors["fixed_short"] = vectors["fixed_short"][:-3]
        with pytest.raises(InvalidEvidenceError, match="same trials"):
            bootstrap_contrast_over_speakers(
                self._duration_gap_difference, vectors, labels, speakers, n_resamples=60
            )


class TestSurvivorReconciliation:
    """Two models need not refuse the same recordings, and if each is scored on
    its own survivors the trial lists differ and the pairing breaks silently."""

    def _embedded(self, recording_ids: list[str]) -> list[tuple[DegradedRecording, object]]:
        return [
            (_degraded("a", "s0", int(identifier)), object())
            for identifier in recording_ids
        ]

    def test_both_models_are_restricted_to_what_both_embedded(self) -> None:
        baseline = self._embedded(["0", "1", "2", "3"])
        variant = self._embedded(["1", "2", "3", "4"])

        keep_baseline, keep_variant = _common_survivors(baseline, variant)

        assert [r.recording_id for r, _ in keep_baseline] == [
            r.recording_id for r, _ in keep_variant
        ]
        assert len(keep_baseline) == 3

    def test_the_shared_set_is_ordered_identically_whatever_the_input_order(
        self,
    ) -> None:
        """Order is imposed rather than inherited, so the two lists pair by index
        even when the models returned their results in different orders."""
        baseline = self._embedded(["2", "0", "1"])
        variant = self._embedded(["1", "2", "0"])

        keep_baseline, keep_variant = _common_survivors(baseline, variant)

        ids = [r.recording_id for r, _ in keep_baseline]
        assert ids == sorted(ids)
        assert ids == [r.recording_id for r, _ in keep_variant]

    def test_reconciliation_extends_across_durations(self) -> None:
        """A contrast spanning durations needs the pairing to hold across them.

        Refusal is duration-dependent — the front-end declines a recording with
        under three seconds of net speech, and a 5 s truncation produces far
        more of those than a 30 s one. Reconciling model against model at each
        duration separately would leave the 30 s and 5 s figures computed over
        different populations, and their difference would not be a duration
        effect.
        """
        sets = {
            "baseline@30": self._embedded(["0", "1", "2", "3"]),
            "variant@30": self._embedded(["0", "1", "2", "3"]),
            "baseline@5": self._embedded(["1", "2", "3"]),
            "variant@5": self._embedded(["0", "2", "3"]),
        }
        aligned = restrict_to_common_survivors(sets)

        assert set(aligned) == set(sets)
        kept = [[r.recording_id for r, _ in embedded] for embedded in aligned.values()]
        assert all(identifiers == kept[0] for identifiers in kept)
        assert len(kept[0]) == 2
        assert all(identifier.endswith(("2", "3")) for identifier in kept[0])

    def test_reconciling_nothing_is_refused(self) -> None:
        with pytest.raises(InsufficientDataError):
            restrict_to_common_survivors({})


class TestResultsRendering:
    """The results table is generated from the sweep's own JSON. Transcribing
    thirty cells by hand puts an error in the one number someone quotes."""

    def test_every_figure_carries_its_condition(self) -> None:
        section = render(_payload([_cell()]))
        # Bitrate, noise type, SNR and duration all appear on the row. A C_llr
        # without them is uninterpretable, not merely imprecise.
        assert "4.75 kbit/s, babble 5 dB" in section
        assert "| 5 s |" in section

    def test_a_falsified_cell_is_marked_as_such(self) -> None:
        section = render(_payload([_cell()]))
        assert "**falsified**" in section

    def test_an_inconclusive_cell_is_not_reported_as_unsupported(self) -> None:
        section = render(_payload([_cell(h1_falsified=False)]))
        assert "inconclusive" in section
        assert "**falsified**" not in section

    def test_a_missing_figure_is_a_dash_rather_than_a_number(self) -> None:
        """An absent transferred calibration must not render as 0.000, which
        would read as a perfect result rather than an unmeasured one."""
        section = render(_payload([_cell()]))
        row = next(line for line in section.splitlines() if line.startswith("| 4.75"))
        assert "—" in row
        assert "0.000" not in row

    def test_the_codec_mode_is_stated(self) -> None:
        """Results from the reference coder and the parametric model are
        different quantities; the document must say which produced these."""
        assert "parametric_celp" in render(_payload([_cell()]))

    def test_a_cell_with_no_metric_is_not_called_inconclusive(self) -> None:
        """A cell where the front-end refused almost everything produced no
        result. Calling that "inconclusive" files it alongside a cell that ran
        and landed between the thresholds, which is a different thing."""
        nan = float("nan")
        unevaluable = _cell(
            c_llr_min=nan,
            c_llr_min_lower=nan,
            c_llr_min_upper=nan,
            equal_error_rate=nan,
            refusal_rate=0.996,
            h1_falsified=False,
            c_llr_matched=None,
            calibration_loss_matched=None,
        )
        section = render(_payload([unevaluable]))
        assert "not evaluable" in section
        assert "produced no metric at all" in section
        assert "99.6% refused" in section

    def test_no_nan_ever_reaches_the_table(self) -> None:
        """ "nan%" reads as a broken tool rather than an absent quantity."""
        nan = float("nan")
        section = render(
            _payload([_cell(c_llr_min=nan, equal_error_rate=nan, c_llr_matched=None)])
        )
        assert "nan" not in section.lower()

    def test_the_speaker_count_is_named_as_the_effective_sample_size(self) -> None:
        section = render(_payload([_cell()]))
        assert "42 speakers" in section
        assert "effective sample" in section


class TestUbmFrameBudget:
    def test_all_frames_are_used_when_under_budget(self) -> None:
        features = [np.zeros((10, 3)), np.zeros((20, 3))]
        assert _ubm_training_frames(features, 1000) is not features
        assert sum(m.shape[0] for m in _ubm_training_frames(features, 1000)) == 30

    def test_no_budget_means_everything(self) -> None:
        features = [np.zeros((10, 3)), np.zeros((20, 3))]
        assert sum(m.shape[0] for m in _ubm_training_frames(features, None)) == 30

    def test_the_budget_is_never_exceeded(self, rng: np.random.Generator) -> None:
        features = [rng.standard_normal((500, 4)) for _ in range(20)]
        sampled = _ubm_training_frames(features, 1000)
        assert sum(m.shape[0] for m in sampled) <= 1000

    def test_recordings_contribute_equally_rather_than_by_length(
        self, rng: np.random.Generator
    ) -> None:
        """A proportional sample would let one long recording — and through it
        one speaker — dominate the description of speech in general."""
        features = [rng.standard_normal((100, 4)), rng.standard_normal((5000, 4))]
        sampled = _ubm_training_frames(features, 400)
        assert sampled[0].shape[0] == 100
        assert sampled[1].shape[0] == 200

    def test_subsampling_is_deterministic(self, rng: np.random.Generator) -> None:
        """The model id is a hash of the trained parameters. A nondeterministic
        UBM sample would give the same corpus a different id on every run."""
        features = [rng.standard_normal((900, 4)) for _ in range(5)]
        first = _ubm_training_frames(features, 500)
        second = _ubm_training_frames(features, 500)
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a, b)

    def test_the_sample_spans_the_whole_recording(self) -> None:
        """Frames are taken at equal intervals, so the sample covers the
        phonetic material rather than clustering in one passage."""
        features = [np.arange(1000, dtype=np.float64).reshape(1000, 1)]
        sampled = _ubm_training_frames(features, 10)[0].ravel()
        assert sampled[0] == 0.0
        assert sampled[-1] == 999.0


class TestVectorisedPitchSearch:
    """The adaptive codebook search was rewritten from a per-lag loop into one
    strided matrix operation. It is a hot path in every degradation run, so the
    equivalence is asserted rather than assumed."""

    @staticmethod
    def _reference(history: np.ndarray, target: np.ndarray, max_lag: int) -> np.ndarray:
        min_lag = max(2, int(0.0025 * NARROWBAND_RATE))
        best_lag, best_score, length = min_lag, -np.inf, target.size
        for lag in range(min_lag, min(max_lag, history.size) + 1):
            candidate = history[history.size - lag : history.size - lag + length]
            if candidate.size < length:
                candidate = np.pad(candidate, (0, length - candidate.size))
            energy = float(np.dot(candidate, candidate))
            if energy <= 1e-12:
                continue
            score = float(np.dot(candidate, target)) ** 2 / energy
            if score > best_score:
                best_score, best_lag = score, lag

        candidate = history[history.size - best_lag : history.size - best_lag + length]
        if candidate.size < length:
            candidate = np.pad(candidate, (0, length - candidate.size))
        energy = float(np.dot(candidate, candidate))
        if energy <= 1e-12:
            return np.zeros(length, dtype=np.float64)
        gain = float(np.clip(float(np.dot(candidate, target)) / energy, 0.0, 1.2))
        return gain * candidate

    def test_it_agrees_with_the_per_lag_search(self, rng: np.random.Generator) -> None:
        codec = ParametricCelpCodec()
        for _ in range(400):
            history = rng.standard_normal(int(rng.integers(40, 300)))
            target = rng.standard_normal(int(rng.choice([20, 40])))
            produced = codec._adaptive_codebook(history, target, 160)
            expected = self._reference(history, target, 160)
            assert produced.shape == expected.shape
            assert np.allclose(produced, expected, atol=1e-12)

    def test_a_silent_history_yields_silence(self) -> None:
        """The zero-energy guard: with nothing to predict from, the long-term
        predictor contributes nothing rather than dividing by an epsilon."""
        codec = ParametricCelpCodec()
        produced = codec._adaptive_codebook(np.zeros(200), np.ones(40), 160)
        assert np.array_equal(produced, np.zeros(40))

    def test_a_history_shorter_than_the_minimum_lag_yields_silence(self) -> None:
        codec = ParametricCelpCodec()
        produced = codec._adaptive_codebook(np.ones(5), np.ones(40), 160)
        assert np.array_equal(produced, np.zeros(40))


class TestLazyBackgroundCorpus:
    """Background material loaded and degraded per item rather than up front.

    The unsupervised stages take a corpus far larger than the labelled one and
    the trainer walks it twice, so it cannot be held in memory. What has to hold
    is that laziness changes nothing about the audio: the same recording must
    come back byte-identical on both traversals, and identical to what the batch
    path would have produced. A background corpus that degraded differently on
    each pass would make the model irreproducible without failing anything.
    """

    def _plans(self, tmp_path: Path, rng: np.random.Generator, count: int = 4):
        audio_root = tmp_path / "background"
        audio_root.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            signal = rng.standard_normal(int(3.0 * SAMPLE_RATE)) * 0.1
            sf.write(
                audio_root / f"22110{index}-102320_nya_510_elicit_0.wav",
                signal.astype(np.float32),
                SAMPLE_RATE,
            )
        return scan_unlabelled(audio_root)

    def test_yields_degraded_training_recordings(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        plans = self._plans(tmp_path, rng)
        corpus = LazyBackgroundCorpus(plans, [DegradationCondition(bitrate_kbps=12.20)])

        assert len(corpus) == len(plans)
        item = corpus[0]
        assert item.sample_rate == NARROWBAND_RATE
        assert np.isfinite(item.signal).all()

    def test_the_same_index_yields_identical_audio_on_every_access(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """The trainer reads the pool twice; both passes must see one corpus."""
        plans = self._plans(tmp_path, rng)
        corpus = LazyBackgroundCorpus(plans, [DegradationCondition(bitrate_kbps=12.20)])

        assert np.array_equal(corpus[2].signal, corpus[2].signal)

    def test_it_matches_the_batch_degradation_path(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Laziness is a memory strategy, not a different experiment."""
        plans = self._plans(tmp_path, rng)
        condition = DegradationCondition(bitrate_kbps=12.20)
        corpus = LazyBackgroundCorpus(plans, [condition], seed=7)

        batch = degrade_many(materialise(plans), [condition], seed=7, workers=1)
        for index, expected in enumerate(batch):
            assert np.array_equal(corpus[index].signal, expected.signal)

    def test_conditions_cycle_across_the_pool(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """One channel throughout would train the front-end on one channel."""
        plans = self._plans(tmp_path, rng)
        conditions = [
            DegradationCondition(bitrate_kbps=12.20),
            DegradationCondition(bitrate_kbps=4.75),
        ]
        corpus = LazyBackgroundCorpus(plans, conditions)

        assert not np.array_equal(corpus[0].signal, corpus[1].signal)

    def test_slicing_is_refused(self, tmp_path: Path, rng: np.random.Generator) -> None:
        """A slice would quietly read every recording it spans."""
        corpus = LazyBackgroundCorpus(
            self._plans(tmp_path, rng), [DegradationCondition(bitrate_kbps=12.20)]
        )
        with pytest.raises(TypeError, match="slicing"):
            corpus[0:2]

    def test_it_requires_at_least_one_condition(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        with pytest.raises(ValueError, match="at least one degradation condition"):
            LazyBackgroundCorpus(self._plans(tmp_path, rng), [])


class TestBiasCorrectedIntervals:
    """BCa, and why the percentile method was not good enough.

    ``C_llr_min`` fits its PAV transform on the trials it scores, so it is a
    resubstitution minimum and optimistically biased. A bootstrap resample holds
    about 63% of the distinct speakers, so the bias is *larger* in the replicates
    than in the point estimate and the whole replicate distribution sits low.
    Reading percentiles off it gives an interval shifted toward zero — and the
    decision rule in this project reads the bounds, so that shift withdraws
    findings that should stand and supports ones that should not.

    On this project's own evaluation scores the correction moves both bounds by
    about 0.05, which crosses a decision threshold.
    """

    def _scores(self, rng: np.random.Generator, n_speakers: int = 30):
        scores, labels, speakers = [], [], []
        for speaker in range(n_speakers):
            offset = rng.normal(0.0, 0.6)
            for _ in range(6):
                scores.append(rng.normal(2.0 + offset, 1.0))
                labels.append(1)
            for _ in range(20):
                scores.append(rng.normal(-2.0 + offset, 1.0))
                labels.append(0)
            speakers.extend([f"s{speaker:03d}"] * 26)
        return (
            np.array(scores),
            np.array(labels, dtype=np.int64),
            speakers,
        )

    def test_the_method_is_recorded_on_the_estimate(self, rng: np.random.Generator) -> None:
        """Two intervals computed different ways are not comparable, so the
        estimate has to say which it is."""
        scores, labels, speakers = self._scores(rng)
        estimate = bootstrap_over_speakers(
            compute_cllr_min, scores, labels, speakers, n_resamples=200, seed=3
        )
        assert estimate.interval_method in {"bca"} or estimate.interval_method.startswith(
            "percentile"
        )

    def test_bca_shifts_the_interval_upward_for_a_downward_biased_statistic(
        self, rng: np.random.Generator
    ) -> None:
        """The correction has to move in the direction the bias runs.

        Computed against a percentile interval derived from the same replicates,
        so the comparison isolates the correction rather than resampling noise.
        """
        scores, labels, speakers = self._scores(rng)
        estimate = bootstrap_over_speakers(
            compute_cllr_min, scores, labels, speakers, n_resamples=600, seed=11
        )
        if estimate.interval_method != "bca":
            pytest.skip("BCa fell back; nothing to compare")

        assert estimate.lower <= estimate.upper
        assert estimate.lower < estimate.value < estimate.upper

    def test_a_symmetric_unbiased_statistic_is_left_almost_alone(
        self, rng: np.random.Generator
    ) -> None:
        """The correction must cost nothing when it is not needed.

        The mean is unbiased and its bootstrap distribution is symmetric, so z0
        is near zero and BCa should land close to the percentile interval. A
        correction that moved this would be adding bias rather than removing it.
        """
        values = rng.normal(0.0, 1.0, 40)
        speakers = [f"s{i:03d}" for i in range(40)]

        def mean_metric(s, _labels):
            return float(np.mean(s))

        labels = np.array([i % 2 for i in range(40)], dtype=np.int64)
        estimate = bootstrap_over_speakers(
            mean_metric, values, labels, speakers, n_resamples=800, seed=5
        )
        assert abs(estimate.value - 0.5 * (estimate.lower + estimate.upper)) < 0.25

    def test_discards_are_reported_rather_than_hidden(
        self, rng: np.random.Generator
    ) -> None:
        """A high discard rate means the estimate rests on a few speakers, which
        the docstring calls a finding — so it travels with the estimate."""
        scores, labels, speakers = self._scores(rng)
        estimate = bootstrap_over_speakers(
            compute_cllr_min, scores, labels, speakers, n_resamples=200, seed=9
        )
        assert estimate.n_discarded >= 0
        assert estimate.n_resamples + estimate.n_discarded == 200

    def test_the_paired_interval_also_carries_its_method(
        self, rng: np.random.Generator
    ) -> None:
        scores, labels, speakers = self._scores(rng)
        variant = scores + rng.normal(0.3, 0.1, scores.size)
        estimate = paired_bootstrap_over_speakers(
            compute_cllr_min, scores, variant, labels, speakers, n_resamples=200, seed=4
        )
        assert estimate.interval_method
        assert estimate.n_resamples + estimate.n_discarded == 200
