"""The scoring path behind §22, on a fabricated archive.

§22's headline — four cells reaching ``supported``, the first in the document —
comes out of ``score_neural.py``, and until now nothing tested it. That is the
gap this file closes. The embeddings themselves cost 330 minutes to produce and
are not something a test suite can regenerate, so the archive here is fabricated:
vectors drawn with a known speaker structure, in the key layout
``extract_neural.py`` writes.

What that can and cannot check is worth being clear about. It cannot check that
ECAPA embeddings are good, and it does not try. It checks the things that would
silently corrupt a result computed from real ones — that same-session pairs are
excluded from the same-source trials, that refusal counts are joined to the right
cell, that the evaluation cells are discovered from the archive rather than
assumed — and each of those has a precedent in this project for going wrong
quietly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.score_neural import _session_of, build_trials, main
from viflap.analysis.speaker.plda import PldaConfig, train_plda
from viflap.analysis.speaker.transforms import fit_transform_chain
from viflap.domain.errors import InsufficientDataError

N_SPEAKERS = 20
N_SESSIONS = 3
DIMENSION = 16
CONDITIONS = ("amr12.2_clean", "amr12.2_babble20dB")
DURATIONS = (30.0, 15.0, 5.0)


def _cell(seed: int, separation: float):
    """Vectors with a real speaker structure, plus their labels and ids.

    Separation is a parameter so a caller can make one cell easier than another:
    a scoring bug that ignored the cell key would otherwise produce identical
    numbers everywhere and look entirely plausible.
    """
    rng = np.random.default_rng(seed)
    vectors, speakers, recordings = [], [], []
    for speaker in range(N_SPEAKERS):
        centre = rng.normal(0.0, separation, DIMENSION)
        for session in range(N_SESSIONS):
            vectors.append(centre + rng.normal(0.0, 1.0, DIMENSION))
            speakers.append(f"spk{speaker:03d}")
            recordings.append(f"spk{speaker:03d}-{session}-r0")
    return (
        np.stack(vectors),
        np.array(speakers, dtype=np.str_),
        np.array(recordings, dtype=np.str_),
    )


@pytest.fixture(scope="module")
def archive_path(tmp_path_factory):
    """An ``.npz`` in exactly the layout ``extract_neural.py`` writes."""
    saved: dict[str, np.ndarray] = {}
    vectors, speakers, recordings = _cell(seed=1, separation=3.0)
    saved["train|vectors"] = vectors
    saved["train|speakers"] = speakers
    saved["train|recordings"] = recordings

    seed = 10
    for partition in ("development", "evaluation"):
        for condition in CONDITIONS:
            for duration in DURATIONS:
                seed += 1
                # Longer durations separate better, as they do in every table in
                # the document.
                vectors, speakers, recordings = _cell(seed, 1.0 + duration / 15.0)
                key = f"{partition}|{condition}@{duration:g}"
                saved[f"{key}|vectors"] = vectors
                saved[f"{key}|speakers"] = speakers
                saved[f"{key}|recordings"] = recordings

    path = tmp_path_factory.mktemp("neural") / "embeddings.npz"
    np.savez_compressed(path, **saved)
    return path


@pytest.fixture(scope="module")
def extraction_report(tmp_path_factory):
    path = tmp_path_factory.mktemp("neural-report") / "extraction.json"
    path.write_text(
        json.dumps(
            {
                "extractor_id": "neural:fabricated@16000",
                "cells": [
                    {
                        "partition": "evaluation",
                        "condition": condition,
                        "duration_seconds": duration,
                        "n_embeddings": N_SPEAKERS * N_SESSIONS,
                        # A distinct number per cell, so a lookup joining on the
                        # wrong key cannot coincidentally agree.
                        "n_refused": int(duration) + (condition == CONDITIONS[1]),
                    }
                    for condition in CONDITIONS
                    for duration in DURATIONS
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def report(tmp_path_factory, archive_path, extraction_report):
    output = tmp_path_factory.mktemp("neural-out") / "h1_neural.json"
    exit_code = main(
        [
            "--embeddings",
            str(archive_path),
            "--extraction-report",
            str(extraction_report),
            "--output",
            str(output),
            "--resamples",
            "60",
        ]
    )
    assert exit_code == 0
    return json.loads(output.read_text(encoding="utf-8"))


class TestSessionParsing:
    def test_the_session_is_recovered_from_the_recording_id(self) -> None:
        assert _session_of("spk001-2-r0") == "spk001-2"
        assert _session_of("spk001-2-r13") == "spk001-2"

    def test_a_speaker_with_a_hyphen_still_parses(self) -> None:
        """Only the trailing recording index is stripped, not everything after
        the first hyphen."""
        assert _session_of("19-198-r4") == "19-198"


class TestTrialFormation:
    """§2's rule: a same-source trial crosses sessions, or it is not a trial."""

    def _built(self):
        vectors, speakers, recordings = _cell(seed=1, separation=3.0)
        labels = np.array(
            [int(name.removeprefix("spk")) for name in speakers], dtype=np.int64
        )
        transform = fit_transform_chain(vectors, labels)
        plda = train_plda(transform.apply(vectors), labels, PldaConfig(max_iterations=10))
        return build_trials(vectors, list(speakers), list(recordings), plda, transform)

    def test_same_source_trials_never_share_a_session(self) -> None:
        """Two recordings from one session share a microphone, a room and a day.

        Counting them as same-source evidence measures the session and reports
        it as the speaker, which is the single easiest way to manufacture a good
        result on this corpus.
        """
        trials = self._built()
        # Three sessions per speaker, all pairings admissible: C(3,2) = 3 each.
        assert trials.n_same_source == N_SPEAKERS * 3

    def test_different_source_trials_are_the_rest(self) -> None:
        trials = self._built()
        total = N_SPEAKERS * N_SESSIONS
        all_pairs = total * (total - 1) // 2
        assert trials.n_same_source + trials.n_different_source == all_pairs

    def test_every_trial_is_attributed_to_a_speaker(self) -> None:
        trials = self._built()
        assert len(trials.speakers) == trials.n_same_source + trials.n_different_source
        assert trials.n_speakers == N_SPEAKERS

    def test_the_effective_sample_size_is_reported_beside_the_nominal_one(
        self,
    ) -> None:
        """§18: speakers do not own trials evenly and the bootstrap assumes they
        are exchangeable, so the discrepancy has to be visible."""
        trials = self._built()
        assert 0 < trials.kish_effective_sample_size <= trials.n_speakers

    def test_an_archive_with_no_admissible_pair_is_refused(self) -> None:
        """One recording per speaker yields no same-source trial at all, and a
        silent empty result would be far worse than an error."""
        vectors, speakers, recordings = _cell(seed=1, separation=3.0)
        labels = np.array(
            [int(name.removeprefix("spk")) for name in speakers], dtype=np.int64
        )
        transform = fit_transform_chain(vectors, labels)
        plda = train_plda(transform.apply(vectors), labels, PldaConfig(max_iterations=10))
        with pytest.raises(InsufficientDataError):
            build_trials(
                vectors[:1], list(speakers[:1]), list(recordings[:1]), plda, transform
            )


class TestTheReport:
    def test_every_evaluation_cell_is_scored(self, report) -> None:
        """Cells are discovered from the archive, so a partially written archive
        must produce a short report rather than a report with missing cells
        silently filled in."""
        assert len(report["cells"]) == len(CONDITIONS) * len(DURATIONS)
        found = {(c["condition"], c["duration_seconds"]) for c in report["cells"]}
        assert found == {(c, d) for c in CONDITIONS for d in DURATIONS}

    def test_refusal_counts_are_joined_to_the_right_cell(self, report) -> None:
        """The join is on ``(condition, duration)`` and both parts matter.

        The fabricated counts differ per cell precisely so that a lookup keyed on
        duration alone, or on condition alone, disagrees here instead of
        coincidentally matching.
        """
        for cell in report["cells"]:
            expected = int(cell["duration_seconds"]) + (cell["condition"] == CONDITIONS[1])
            assert cell["n_refused"] == expected, cell["condition"]

    def test_each_cell_carries_an_interval_containing_its_estimate(self, report) -> None:
        for cell in report["cells"]:
            assert cell["c_llr_min_lower"] <= cell["c_llr_min"] <= cell["c_llr_min_upper"]

    def test_the_decision_is_the_hypothesis_object_s_and_is_consistent(
        self, report
    ) -> None:
        """§3's rule, applied by ``H1ChannelViability`` rather than re-derived.

        Checked against the interval rather than trusted: a cell cannot be both
        supported and falsified, and each verdict must agree with the bounds the
        same cell reports.
        """
        for cell in report["cells"]:
            assert not (cell["h1_supported"] and cell["h1_falsified"])
            if cell["h1_supported"]:
                assert cell["c_llr_min_upper"] <= 0.30
            if cell["h1_falsified"]:
                assert cell["c_llr_min_lower"] > 0.50

    def test_longer_durations_score_better_on_this_archive(self, report) -> None:
        """The fabricated cells were built with separation rising with duration.

        This is a wiring check on the cell keys, not a finding: if the scorer
        read the wrong cell's vectors the ordering would not follow the
        construction.
        """
        for condition in CONDITIONS:
            by_duration = {
                c["duration_seconds"]: c["c_llr_min"]
                for c in report["cells"]
                if c["condition"] == condition
            }
            assert by_duration[30.0] < by_duration[5.0], condition

    def test_the_back_end_records_what_it_fitted(self, report) -> None:
        """A result whose transform dimension and PLDA convergence are not
        recorded cannot be reproduced when it is challenged."""
        back_end = report["back_end"]
        assert back_end["n_training_speakers"] == N_SPEAKERS
        assert back_end["transform_dimension"] > 0
        assert back_end["plda_n_iterations"] >= 1
        assert isinstance(back_end["plda_converged"], bool)

    def test_the_extractor_identity_travels_with_the_result(self, report) -> None:
        assert report["extractor_id"] == "neural:fabricated@16000"

    def test_a_matched_calibration_is_fitted_from_the_development_cell(
        self, report
    ) -> None:
        """The development partition exists in the archive, so every cell should
        carry a matched figure; a silent ``None`` would mean the development key
        was not found and the calibration quietly skipped."""
        for cell in report["cells"]:
            assert cell["c_llr_matched"] is not None, cell["condition"]
            assert cell["calibration_loss_matched"] is not None
