"""Joining two systems' trials, and the ways that join can be silently wrong.

§22's claim rested on marginal intervals because nothing recorded what a *trial*
was — only its score. Both scorers now persist the recording-id pair behind each
trial, and this is the machinery that joins on it.

Every test here is about a failure that produces numbers rather than an error.
An index join on two differently ordered archives returns a full-length vector of
differences between unrelated trials. A join that silently drops the
non-overlapping part reports a paired result on a subset without saying so. An
ownership rule that has drifted between the two scripts gives the bootstrap
different resampling units for the two arms while both still look like speaker
identifiers. None of these raise on their own.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.compare_extractors import load_cells, pair_cell
from scripts.experiment import _pair_key
from viflap.domain.errors import InsufficientDataError

N_OWNERS = 30


def _archive(order: list[int], scores: dict[str, float], owners: dict[str, str], labels):
    """One system's cell, laid out in a caller-chosen trial order."""
    keys = [f"r{i:03d}\tr{i + 500:03d}" for i in range(len(scores))]
    ordered_keys = [keys[i] for i in order]
    return {
        "pairs": np.array(ordered_keys, dtype=np.str_),
        "scores": np.array([scores[k] for k in ordered_keys], dtype=np.float64),
        "labels": np.array([labels[k] for k in ordered_keys], dtype=np.int64),
        "owners": np.array([owners[k] for k in ordered_keys], dtype=np.str_),
    }


@pytest.fixture
def two_systems():
    """The same 300 trials, scored by two systems, stored in different orders."""
    rng = np.random.default_rng(5)
    keys = [f"r{i:03d}\tr{i + 500:03d}" for i in range(300)]
    labels = {k: int(i % 5 == 0) for i, k in enumerate(keys)}
    owners = {k: f"spk{i % N_OWNERS:02d}" for i, k in enumerate(keys)}
    # The variant separates the classes better, which is the thing the paired
    # difference should detect.
    baseline_scores = {
        k: float(rng.normal(2.0 if labels[k] else -2.0, 3.0)) for k in keys
    }
    variant_scores = {
        k: float(rng.normal(4.0 if labels[k] else -4.0, 1.5)) for k in keys
    }

    forward = list(range(300))
    shuffled = list(rng.permutation(300))
    return (
        _archive(forward, baseline_scores, owners, labels),
        _archive(shuffled, variant_scores, owners, labels),
        baseline_scores,
        variant_scores,
    )


class TestThePairKey:
    def test_the_key_does_not_depend_on_order(self) -> None:
        """Which recording is "first" is an artefact of enumeration order."""
        assert _pair_key("a", "b") == _pair_key("b", "a")

    def test_the_key_keeps_both_identifiers(self) -> None:
        assert set(_pair_key("z", "a")) == {"a", "z"}


class TestLoadingCells:
    def test_keys_split_into_cell_and_field(self, tmp_path) -> None:
        path = tmp_path / "scores.npz"
        np.savez_compressed(
            path,
            **{
                "amr12.2_clean@30|scores": np.zeros(3),
                "amr12.2_clean@30|labels": np.zeros(3, dtype=np.int64),
                "amr12.2_babble20dB@5|scores": np.zeros(2),
            },
        )
        cells = load_cells(path)
        assert set(cells) == {"amr12.2_clean@30", "amr12.2_babble20dB@5"}
        assert set(cells["amr12.2_clean@30"]) == {"scores", "labels"}

    def test_a_condition_containing_an_at_sign_still_splits_on_the_last_bar(
        self, tmp_path
    ) -> None:
        """``rpartition`` on the separator, so only the field name is taken."""
        path = tmp_path / "scores.npz"
        np.savez_compressed(path, **{"odd|name@30|scores": np.zeros(1)})
        assert set(load_cells(path)) == {"odd|name@30"}


class TestTheJoin:
    def test_trials_are_matched_by_key_not_by_row(self, two_systems) -> None:
        """The defect this whole mechanism exists to prevent.

        The variant archive is stored in a shuffled order. An index join would
        difference unrelated trials and return a full-length result that looks
        entirely normal. Here the paired scores are checked back against the
        dictionaries they were built from.
        """
        baseline, variant, baseline_scores, variant_scores = two_systems
        result = pair_cell("amr12.2_clean", 30.0, baseline, variant, 200, 1)

        assert result.n_trials_paired == 300
        assert result.trial_sets_identical
        # If the join had gone by row index, these two would disagree, because
        # C_llr_min on mismatched pairings is not the C_llr_min of either system.
        expected_variant = np.array(
            [variant_scores[k] for k in sorted(baseline_scores)], dtype=np.float64
        )
        assert result.variant_c_llr_min == pytest.approx(
            _cllr_min_of(expected_variant, sorted(baseline_scores), baseline)
        )

    def test_the_better_system_gives_a_negative_difference(self, two_systems) -> None:
        """Sign convention: negative means the variant discriminates better."""
        baseline, variant, _, _ = two_systems
        result = pair_cell("amr12.2_clean", 30.0, baseline, variant, 400, 1)
        assert result.difference < 0.0
        assert result.variant_c_llr_min < result.baseline_c_llr_min

    def test_a_shuffled_variant_gives_the_same_answer_as_an_unshuffled_one(
        self, two_systems
    ) -> None:
        """The join must make storage order irrelevant, which is its whole job."""
        baseline, variant, _, variant_scores = two_systems
        keys = [str(k) for k in variant["pairs"]]
        reordered = {
            "pairs": np.array(sorted(keys), dtype=np.str_),
            "scores": np.array(
                [variant_scores[k] for k in sorted(keys)], dtype=np.float64
            ),
            "labels": np.array(
                [
                    int(variant["labels"][keys.index(k)])
                    for k in sorted(keys)
                ],
                dtype=np.int64,
            ),
            "owners": np.array(
                [str(variant["owners"][keys.index(k)]) for k in sorted(keys)],
                dtype=np.str_,
            ),
        }
        first = pair_cell("c", 30.0, baseline, variant, 200, 3)
        second = pair_cell("c", 30.0, baseline, reordered, 200, 3)
        assert first.variant_c_llr_min == pytest.approx(second.variant_c_llr_min)
        assert first.difference == pytest.approx(second.difference)


class TestPartialOverlap:
    def test_only_the_intersection_is_paired_and_it_is_recorded(
        self, two_systems
    ) -> None:
        """The five-second case: the baseline front-end refused some recordings.

        Pairing the intersection is what ``paired_bootstrap_over_speakers``
        requires of its caller, but doing it silently would report a paired
        result on a subset without saying which subset.
        """
        baseline, variant, _, _ = two_systems
        trimmed = {key: value[:250] for key, value in baseline.items()}
        result = pair_cell("amr12.2_clean", 5.0, trimmed, variant, 200, 1)

        assert result.n_trials_paired == 250
        assert not result.trial_sets_identical
        assert result.n_trials_variant_only == 50
        assert result.notes and "intersection" in result.notes[0]

    def test_no_overlap_at_all_is_refused(self, two_systems) -> None:
        """Archives of different corpora share no key, and a zero-length paired
        result would be a number rather than an error."""
        baseline, variant, _, _ = two_systems
        alien = dict(variant)
        alien["pairs"] = np.array(
            [f"x{i}\ty{i}" for i in range(len(variant["pairs"]))], dtype=np.str_
        )
        with pytest.raises(InsufficientDataError, match="share no trial"):
            pair_cell("c", 30.0, baseline, alien, 100, 1)


class TestDisagreementIsRefused:
    def test_disagreeing_labels_are_refused(self, two_systems) -> None:
        """Truth is a property of the corpus, not of the system that scored it."""
        baseline, variant, _, _ = two_systems
        corrupted = dict(variant)
        flipped = np.asarray(variant["labels"]).copy()
        flipped[0] = 1 - flipped[0]
        corrupted["labels"] = flipped
        with pytest.raises(InsufficientDataError, match="truth"):
            pair_cell("c", 30.0, baseline, corrupted, 100, 1)

    def test_disagreeing_owners_are_refused(self, two_systems) -> None:
        """§18's rule is shared code; drift between the scripts would give the
        two arms different resampling units while both still look like speaker
        identifiers."""
        baseline, variant, _, _ = two_systems
        corrupted = dict(variant)
        owners = np.asarray(variant["owners"]).copy()
        owners[0] = "somebody-else"
        corrupted["owners"] = owners
        with pytest.raises(InsufficientDataError, match="ownership rule"):
            pair_cell("c", 30.0, baseline, corrupted, 100, 1)


def _cllr_min_of(scores, keys, baseline):
    from viflap.analysis.calibration.metrics import compute_cllr_min

    order = {str(k): i for i, k in enumerate(baseline["pairs"])}
    labels = np.array(
        [int(baseline["labels"][order[k]]) for k in keys], dtype=np.int64
    )
    return compute_cllr_min(scores, labels)
