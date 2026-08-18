"""The overstatement simulation, and whether its new arms do what they claim.

§11 was withdrawn once for reporting simulation output as measurement, and its
remaining criticism was that it reported point estimates from one seed in a
document that insists on intervals. Two things were added in response —
replication with the acoustic marginal resampled, and a t-copula arm to break the
correct-specification-by-construction that made the dependence model's defeat
uninformative.

Both are the kind of addition that can silently be a no-op. A resampling routine
that draws trials instead of speakers still returns an array; a "t-copula" that
does not actually share its scale factor across streams still returns numbers with
the right marginals and the right linear correlation, and simply has no tail
dependence. These tests exist because neither failure is visible in the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.measure_overstatement import (
    _STREAMS,
    _from_positions,
    _interval,
    _summarise,
    resample_marginal,
    simulate,
)


@pytest.fixture
def marginal():
    """A skewed, bounded marginal in the shape the real one has."""
    rng = np.random.default_rng(7)
    same = rng.normal(2.0, 1.0, 400)
    different = rng.normal(-4.0, 2.0, 4000)
    return same, different


class TestQuantileMapping:
    def test_positions_map_monotonically_onto_the_sample(self) -> None:
        sample = np.array([-3.0, -1.0, 0.5, 2.0, 9.0])
        mapped = _from_positions(np.array([0.01, 0.3, 0.6, 0.99]), sample)
        assert list(mapped) == sorted(mapped)
        assert mapped.min() >= sample.min()
        assert mapped.max() <= sample.max()

    def test_the_empirical_shape_survives_the_mapping(self, marginal) -> None:
        """The point of quantile mapping: the marginal stays the measured one.

        Fitting a normal instead would smooth away the asymmetry and the tails,
        which is the region an overstatement study is about.
        """
        same, _ = marginal
        positions = np.random.default_rng(1).uniform(size=20000)
        mapped = _from_positions(positions, same)
        assert mapped.mean() == pytest.approx(same.mean(), abs=0.05)
        assert mapped.std() == pytest.approx(same.std(), abs=0.05)


class TestMarginalResampling:
    def _fixture(self):
        speakers = np.array([f"s{i // 10:02d}" for i in range(200)])
        labels = np.array([i % 2 for i in range(200)], dtype=np.int64)
        log_lrs = np.arange(200, dtype=np.float64)
        return log_lrs, labels, speakers

    def test_whole_speakers_are_drawn_not_individual_trials(self) -> None:
        """§2's rule. Resampling trials treats correlated observations as
        independent and produces intervals several times too narrow."""
        log_lrs, labels, speakers = self._fixture()
        same, different = resample_marginal(
            log_lrs, labels, speakers, np.random.default_rng(3)
        )
        # Each speaker owns exactly 10 consecutive values, 5 of each class, so
        # any draw must be a multiple of that block.
        assert same.size % 5 == 0
        assert different.size % 5 == 0
        assert same.size + different.size == 200

    def test_different_seeds_give_different_replicates(self) -> None:
        log_lrs, labels, speakers = self._fixture()
        first, _ = resample_marginal(
            log_lrs, labels, speakers, np.random.default_rng(1)
        )
        second, _ = resample_marginal(
            log_lrs, labels, speakers, np.random.default_rng(2)
        )
        assert not np.array_equal(np.sort(first), np.sort(second))

    def test_resampling_is_reproducible_for_one_seed(self) -> None:
        log_lrs, labels, speakers = self._fixture()
        args = (log_lrs, labels, speakers)
        first, _ = resample_marginal(*args, np.random.default_rng(11))
        second, _ = resample_marginal(*args, np.random.default_rng(11))
        assert np.array_equal(first, second)


class TestTheCopulaArm:
    def _streams_matrix(self, copula: str, correlation: float, seed: int, marginal):
        same, different = marginal
        training = simulate(
            same,
            different,
            correlation,
            4000,
            np.random.default_rng(seed),
            copula=copula,
        )
        rows = [
            [observation.log_lrs[stream] for stream in _STREAMS]
            for observation in training.observations
            if observation.is_same_source
        ]
        return np.array(rows)

    def test_an_unknown_copula_is_refused(self, marginal) -> None:
        """Silently falling back to Gaussian would make the misspecification arm
        report the correctly-specified result under a different name."""
        same, different = marginal
        with pytest.raises(ValueError, match="unknown copula"):
            simulate(
                same, different, 0.5, 10, np.random.default_rng(0), copula="clayton"
            )

    def test_both_copulas_preserve_the_measured_marginal(self, marginal) -> None:
        """A copula changes the dependence and must not touch the marginals.

        If this failed, the t arm would be comparing two things at once and its
        result could not be attributed to misspecification.
        """
        same, _ = marginal
        for copula in ("gaussian", "t"):
            values = self._streams_matrix(copula, 0.6, 5, marginal)[:, 0]
            assert values.mean() == pytest.approx(same.mean(), abs=0.25), copula

    def test_the_t_copula_adds_joint_tail_dependence(self, marginal) -> None:
        """The mechanism, asserted rather than assumed.

        A t-copula with a *shared* scale factor makes extreme values arrive
        together. Sharing it is the whole point — divide each stream by its own
        chi-square draw instead and the linear correlation is unchanged while the
        tail dependence disappears, which would make this arm an expensive
        no-op that still produced plausible numbers.
        """
        def joint_tail_rate(copula: str) -> float:
            matrix = self._streams_matrix(copula, 0.5, 21, marginal)
            thresholds = np.percentile(matrix, 90, axis=0)
            return float(np.mean(np.all(matrix >= thresholds, axis=1)))

        gaussian_rate = joint_tail_rate("gaussian")
        t_rate = joint_tail_rate("t")
        assert t_rate > gaussian_rate, (gaussian_rate, t_rate)

    def test_the_two_copulas_impose_a_similar_linear_correlation(
        self, marginal
    ) -> None:
        """Otherwise the t arm would differ in strength of dependence as well as
        in its shape, and the comparison would confound the two."""
        def mean_correlation(copula: str) -> float:
            matrix = self._streams_matrix(copula, 0.6, 33, marginal)
            corr = np.corrcoef(matrix, rowvar=False)
            off = corr[np.triu_indices_from(corr, k=1)]
            return float(np.mean(off))

        assert mean_correlation("t") == pytest.approx(
            mean_correlation("gaussian"), abs=0.15
        )


class TestSummaries:
    def test_the_interval_brackets_the_mean(self) -> None:
        summary = _interval([0.1, 0.2, 0.3, 0.4, 0.5])
        assert summary["lower"] <= summary["mean"] <= summary["upper"]
        assert summary["n_replicates"] == 5

    def test_a_single_replicate_yields_a_degenerate_interval(self) -> None:
        """One seed gives a point estimate wearing an interval's clothes, which
        is precisely what §11 was criticised for. It should be visibly so."""
        summary = _interval([0.42])
        assert summary["lower"] == summary["upper"] == summary["mean"]

    def test_the_model_difference_is_taken_within_each_replicate(self) -> None:
        """§7's argument, applied here.

        Both arms see the same simulated incidents and the same resampled
        marginal, so most of the replicate-to-replicate variation is common and
        cancels in a difference. Differencing the summaries instead of
        summarising the differences would throw that away — and the constructed
        case below has a perfectly stable difference inside wildly varying
        marginals, so the two approaches disagree completely.
        """
        per_replicate = {
            "naive_sum": [0.5, 0.9],
            "linear_logistic": [0.2, 0.6],
            "gaussian_latent": [0.25, 0.65],
        }
        summary = _summarise(0.4, per_replicate, [0.1, 0.2], [0.3, 0.4])
        difference = summary["latent_minus_linear"]
        assert difference["mean"] == pytest.approx(0.05)
        assert difference["lower"] == pytest.approx(0.05)
        assert difference["upper"] == pytest.approx(0.05)
        # The marginals, by contrast, span a wide range.
        assert summary["c_llr"]["linear_logistic"]["upper"] > 0.5

    def test_every_arm_is_summarised(self) -> None:
        per_replicate = {
            "naive_sum": [0.5],
            "linear_logistic": [0.2],
            "gaussian_latent": [0.25],
        }
        summary = _summarise(0.0, per_replicate, [0.1], [0.2])
        assert set(summary["c_llr"]) == set(per_replicate)
        assert summary["correlation"] == 0.0
