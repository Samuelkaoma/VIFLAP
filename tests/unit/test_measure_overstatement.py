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


class TestStreamStrengthIsNotStreamQuality:
    """``_ASSUMED_STRENGTH`` does not do what §11 said it did.

    The constant is documented as making the unmeasured streams *weaker* than
    the measured one, and the comment at the multiplication says it "reduces
    separation". It does not. Multiplying a stream's log-LRs by a positive
    constant is monotonic, so the ranking of trials is untouched and
    discrimination is exactly unchanged; what moves is calibration, and the
    streams become under-confident rather than less informative.

    That matters because the three fusion arms respond to it completely
    differently, and only one of them is affected at all. These tests pin the
    property so the constant cannot quietly be read as a quality knob again.
    """

    def test_scaling_leaves_discrimination_exactly_unchanged(self) -> None:
        """The claim in one line: same ``C_llr_min``, different ``C_llr``."""
        from viflap.analysis.calibration.metrics import compute_cllr, compute_cllr_min

        rng = np.random.default_rng(0)
        scores = np.concatenate(
            [rng.normal(2.2, 2.1, 2000), rng.normal(-3.9, 2.3, 20000)]
        )
        labels = np.concatenate([np.ones(2000), np.zeros(20000)]).astype(np.int64)

        baseline_min = compute_cllr_min(scores, labels)
        baseline_cllr = compute_cllr(scores, labels)
        for strength in (0.75, 0.5, 0.1):
            scaled = scores * strength
            assert compute_cllr_min(scaled, labels) == pytest.approx(baseline_min)
            assert compute_cllr(scaled, labels) != pytest.approx(baseline_cllr)

    def test_a_fitted_linear_fusion_absorbs_the_scaling_exactly(
        self, marginal
    ) -> None:
        """``w0 + sum wi*li`` under ``li -> si*li`` is the same model at
        ``wi/si``, so refitting recovers it and the constant cannot reach the
        result.

        Exact in the algebra, and equal to about 5e-6 in practice — the residual
        is where the optimiser stopped, not the property failing. The tolerance
        below is still two orders of magnitude tighter than the movement the
        naive sum shows on the same inputs, which is the comparison that matters.
        """
        from viflap.analysis.fusion.models import LinearLogisticFusion

        same, different = marginal
        values = [
            self._arm(LinearLogisticFusion, same, different, strength)
            for strength in (1.0, 0.75, 0.25)
        ]
        assert values[0] == pytest.approx(values[1], abs=1e-4)
        assert values[0] == pytest.approx(values[2], abs=1e-4)

    def test_only_the_naive_sum_can_feel_it(self, marginal) -> None:
        """The one arm with nothing to refit.

        ``NaiveIndependentFusion`` adds the log-LRs as given, so it is the only
        arm that has to take a rescaled input at face value. Any §11 statement
        about the naive sum is therefore partly a statement about an arbitrary
        constant, and that is exactly what the section had to withdraw.
        """
        from viflap.analysis.fusion.models import NaiveIndependentFusion

        same, different = marginal
        costs = [
            self._arm(NaiveIndependentFusion, same, different, strength)
            for strength in (1.0, 0.25)
        ]
        assert costs[0] != pytest.approx(costs[1], abs=1e-3)

    @staticmethod
    def _arm(model_class, same, different, strength: float) -> float:
        import scripts.measure_overstatement as mo
        from viflap.domain.evidence import EvidenceStream

        original = dict(mo._ASSUMED_STRENGTH)
        mo._ASSUMED_STRENGTH.clear()
        mo._ASSUMED_STRENGTH.update(
            {
                EvidenceStream.BEHAVIOURAL: strength,
                EvidenceStream.TEMPORAL: strength,
            }
        )
        try:
            training = mo.simulate(
                same, different, 0.4, 1200, np.random.default_rng(3)
            )
            evaluation = mo.simulate(
                same, different, 0.4, 1200, np.random.default_rng(4)
            )
            model = model_class()
            fitted = model.fit(training) if hasattr(model, "fit") else model
            return mo._fused_cllr(fitted, evaluation)
        finally:
            mo._ASSUMED_STRENGTH.clear()
            mo._ASSUMED_STRENGTH.update(original)


class TestWeakeningAStreamProperly:
    """``weaken`` has to do what ``_ASSUMED_STRENGTH`` provably does not.

    Scaling log-LRs is monotonic and leaves ``C_llr_min`` exactly where it was.
    Making a stream genuinely less informative means increasing the *overlap*
    between its two class distributions, which is what sliding the same-source
    marginal toward the different-source one does. The distinguishing test is
    the first one: discrimination must actually move.
    """

    def test_discrimination_actually_degrades(self, marginal) -> None:
        """The property scaling could not deliver."""
        from scripts.measure_overstatement import weaken
        from viflap.analysis.calibration.metrics import compute_cllr_min

        same, different = marginal
        labels = np.concatenate(
            [np.ones(same.size), np.zeros(different.size)]
        ).astype(np.int64)

        costs = []
        for factor in (1.0, 0.7, 0.4):
            weakened = weaken(same, different, factor)
            costs.append(
                compute_cllr_min(np.concatenate([weakened, different]), labels)
            )
        assert costs[0] < costs[1] < costs[2], costs

    def test_one_leaves_the_marginal_untouched(self, marginal) -> None:
        """Every §11 figure predating this option was produced at 1.0, so it has
        to be exactly the identity rather than approximately so."""
        from scripts.measure_overstatement import weaken

        same, different = marginal
        assert np.array_equal(weaken(same, different, 1.0), same)

    def test_zero_puts_the_class_means_together(self, marginal) -> None:
        """A stream carrying no information at all is the far end of the range."""
        from scripts.measure_overstatement import weaken

        same, different = marginal
        collapsed = weaken(same, different, 0.0)
        assert float(np.mean(collapsed)) == pytest.approx(
            float(np.mean(different)), abs=1e-9
        )

    def test_the_spread_is_left_alone(self, marginal) -> None:
        """Only the gap between the classes moves.

        Shrinking the spread as well would change the distribution's shape and
        the comparison across streams would stop being like-for-like — the
        property the quantile mapping exists to preserve.
        """
        from scripts.measure_overstatement import weaken

        same, different = marginal
        assert float(np.std(weaken(same, different, 0.5))) == pytest.approx(
            float(np.std(same))
        )

    def test_it_reaches_only_the_stream_it_is_asked_for(self, marginal) -> None:
        """Per-stream, so one stream can be weak while another stays measured."""
        from scripts.measure_overstatement import simulate
        from viflap.domain.evidence import EvidenceStream

        same, different = marginal
        training = simulate(
            same,
            different,
            0.0,
            3000,
            np.random.default_rng(9),
            discriminability={EvidenceStream.BEHAVIOURAL: 0.3},
        )
        rows = np.array(
            [
                [o.log_lrs[s] for s in _STREAMS]
                for o in training.observations
                if o.is_same_source
            ]
        )
        acoustic, behavioural = rows[:, 0].mean(), rows[:, 1].mean()
        # The behavioural column was slid toward the different-source mean; the
        # acoustic one was not, so their same-source means must now differ.
        assert behavioural < acoustic
