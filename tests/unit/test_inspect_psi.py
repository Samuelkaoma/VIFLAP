"""The ψ spectrum diagnostic, and the two ways its sweep could mislead.

§23 refutes two of §21's three candidates using this script, so the arithmetic it
does has to be right and the sweep it runs has to vary one thing.

Two failures would be invisible in the output. A subsample that relabelled
speakers wrongly would train PLDA against scrambled classes and return a
perfectly plausible spectrum. And a sweep that let the LDA ceiling move with the
speaker count would confound the count with the transform dimension — the
comparison §23 draws its conclusion from would then be between two things at
once, which is the defect §9 had to restrict its own comparison to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.inspect_psi import fit, spectrum, subsample

#: Wider than the largest speaker count used below, so that ``n_speakers - 1``
#: is what binds the LDA ceiling rather than the vector dimension. With a
#: narrower fixture the ceiling would be pinned by the dimension for every
#: subsample and the sweep test could not show the confound it exists to rule
#: out — which is also why the real 192-dimensional ECAPA embeddings pass
#: through untruncated at 306 speakers.
DIMENSION = 50


@pytest.fixture
def labelled():
    """Vectors with a deliberately dominant leading between-speaker axis."""
    rng = np.random.default_rng(11)
    vectors, labels = [], []
    for speaker in range(40):
        centre = rng.normal(0.0, 1.0, DIMENSION)
        centre[0] *= 6.0  # one axis carries far more between-speaker variance
        for _ in range(6):
            vectors.append(centre + rng.normal(0.0, 1.0, DIMENSION))
            labels.append(speaker)
    return np.stack(vectors), np.array(labels, dtype=np.int64)


class TestSpectrum:
    def test_the_ratio_is_the_first_over_the_second(self) -> None:
        summary = spectrum(np.array([2.0, 10.0, 5.0, 0.05]))
        assert summary["psi_1"] == 10.0
        assert summary["psi_2"] == 5.0
        assert summary["ratio"] == pytest.approx(2.0)

    def test_the_spectrum_is_sorted_regardless_of_input_order(self) -> None:
        """PLDA returns psi in its own order; §1's claim is about the largest two."""
        ascending = spectrum(np.array([0.1, 1.0, 20.0]))
        descending = spectrum(np.array([20.0, 1.0, 0.1]))
        assert ascending == descending

    def test_the_share_and_the_ratio_say_different_things(self) -> None:
        """A ratio alone cannot tell one enormous axis from a steep decay.

        Both spectra below have ψ₁/ψ₂ = 2, and they are very different claims
        about what the model learned.
        """
        steep = spectrum(np.array([10.0, 5.0, 2.5, 1.25]))
        flat = spectrum(np.array([10.0, 5.0, 5.0, 5.0]))
        assert steep["ratio"] == pytest.approx(flat["ratio"])
        assert steep["psi_1_share_of_total"] > flat["psi_1_share_of_total"]

    def test_inert_dimensions_are_counted_at_the_named_threshold(self) -> None:
        """0.1 is ``INERT_PSI``: below it a dimension carries almost nothing."""
        summary = spectrum(np.array([5.0, 1.0, 0.05, 0.01]))
        assert summary["n_above_inert"] == 2


class TestSubsampling:
    def test_all_of_a_drawn_speaker_s_recordings_come_with_them(self, labelled) -> None:
        vectors, labels = labelled
        _, subset_labels = subsample(vectors, labels, 10, np.random.default_rng(1))
        counts = np.unique(subset_labels, return_counts=True)[1]
        assert set(counts.tolist()) == {6}

    def test_labels_are_relabelled_contiguously_from_zero(self, labelled) -> None:
        """A gap in the label space would make the class count wrong downstream."""
        vectors, labels = labelled
        _, subset_labels = subsample(vectors, labels, 10, np.random.default_rng(2))
        assert sorted(set(subset_labels.tolist())) == list(range(10))

    def test_relabelling_does_not_scramble_which_vectors_share_a_speaker(
        self, labelled
    ) -> None:
        """The failure that returns a plausible spectrum from scrambled classes."""
        vectors, labels = labelled
        subset_vectors, subset_labels = subsample(
            vectors, labels, 12, np.random.default_rng(3)
        )
        # Two rows share a new label if and only if they shared an old one.
        for new in np.unique(subset_labels):
            rows = subset_vectors[subset_labels == new]
            originals = {
                int(labels[np.all(np.isclose(vectors, row), axis=1)][0]) for row in rows
            }
            assert len(originals) == 1, originals

    def test_asking_for_every_speaker_returns_everything(self, labelled) -> None:
        vectors, labels = labelled
        subset_vectors, subset_labels = subsample(
            vectors, labels, 40, np.random.default_rng(4)
        )
        assert subset_vectors.shape == vectors.shape
        assert len(set(subset_labels.tolist())) == 40


class TestFitting:
    def test_the_dominant_axis_is_found(self, labelled) -> None:
        summary = spectrum(fit(*labelled).psi)
        assert summary["ratio"] > 1.5

    def test_pinning_the_dimension_makes_the_sweep_vary_one_thing(self, labelled) -> None:
        """The sweep's whole validity.

        Without ``lda_dimension``, the ceiling is ``n_speakers - 1``, so a
        subsample changes the transform width as well as the sample size and the
        comparison confounds them. §23 pins it at the smallest count's ceiling.
        """
        vectors, labels = labelled
        # Both above train_plda's twenty-speaker floor; the point is that they
        # differ, not that either is small.
        small_v, small_l = subsample(vectors, labels, 24, np.random.default_rng(5))
        big_v, big_l = subsample(vectors, labels, 40, np.random.default_rng(6))

        floating_small = fit(small_v, small_l)
        floating_big = fit(big_v, big_l)
        assert floating_small.psi.size != floating_big.psi.size

        pinned_small = fit(small_v, small_l, lda_dimension=23)
        pinned_big = fit(big_v, big_l, lda_dimension=23)
        assert pinned_small.psi.size == pinned_big.psi.size == 23

    def test_length_normalisation_is_switchable(self, labelled) -> None:
        """§23's second arm needs this to actually change the fit."""
        vectors, labels = labelled
        with_norm = fit(vectors, labels).psi
        without = fit(vectors, labels, length_normalise=False).psi
        assert not np.allclose(np.sort(with_norm), np.sort(without))
