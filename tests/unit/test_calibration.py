"""Calibration and forensic metrics, against analytically known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import isotonic_regression
from scipy.stats import norm

from viflap.analysis.calibration.calibrators import (
    IsotonicCalibrator,
    KernelDensityCalibrator,
    LogisticCalibrator,
    _gaussian_kde_log_density,
    as_reported,
)
from viflap.analysis.calibration.elub import apply_bounds, empirical_bounds
from viflap.analysis.calibration.metrics import (
    compute_cllr,
    compute_cllr_min,
    compute_eer,
    evaluate,
)
from viflap.analysis.calibration.pav import pav_calibrate, pool_adjacent_violators
from viflap.domain.errors import (
    CalibrationError,
    InsufficientDataError,
    ModelNotTrainedError,
)


class TestCllrReferencePoints:
    """C_llr has absolute reference points; they are not approximate."""

    def test_uninformative_system_scores_exactly_one(self) -> None:
        labels = np.concatenate([np.ones(500), np.zeros(500)]).astype(int)
        assert compute_cllr(np.zeros(1000), labels) == pytest.approx(1.0, abs=1e-12)

    def test_perfect_confident_system_approaches_zero(self) -> None:
        labels = np.concatenate([np.ones(500), np.zeros(500)]).astype(int)
        scores = np.where(labels == 1, 40.0, -40.0)
        assert compute_cllr(scores, labels) < 1e-15

    def test_confidently_wrong_system_exceeds_one(self) -> None:
        """Worse than useless is a state accuracy metrics cannot express."""
        labels = np.concatenate([np.ones(500), np.zeros(500)]).astype(int)
        scores = np.where(labels == 1, -10.0, 10.0)
        assert compute_cllr(scores, labels) > 1.0

    @pytest.mark.parametrize("factor", [0.25, 0.5, 2.0, 4.0])
    def test_minimised_only_at_the_truth(self, factor: float, rng) -> None:
        """C_llr is strictly proper: mis-stating in either direction costs.

        For scores drawn from N(+2,1) against N(-2,1), the correct log-LR of
        observing s is exactly 4s. Anything else is penalised, which is the
        property that makes exaggeration unprofitable.
        """
        labels = np.concatenate([np.ones(2000), np.zeros(2000)]).astype(int)
        scores = rng.standard_normal(4000) + np.where(labels == 1, 2.0, -2.0)
        truth = 4.0 * scores
        assert compute_cllr(truth * factor, labels) > compute_cllr(truth, labels)

    def test_requires_both_trial_types(self) -> None:
        with pytest.raises(InsufficientDataError):
            compute_cllr(np.array([1.0, 2.0]), np.array([1, 1]))

    def test_survives_extreme_log_lrs_without_overflow(self) -> None:
        """The naive form overflows below about -745, on exactly the strong
        results whose cost matters most."""
        labels = np.array([1, 1, 0, 0])
        scores = np.array([900.0, -900.0, -900.0, 900.0])
        value = compute_cllr(scores, labels)
        assert math.isfinite(value)


class TestCllrMin:
    def test_invariant_to_monotonic_transformation(self, rng) -> None:
        """C_llr_min measures discrimination alone, so calibration cannot move it."""
        labels = np.concatenate([np.ones(800), np.zeros(800)]).astype(int)
        scores = rng.standard_normal(1600) + np.where(labels == 1, 1.5, 0.0)
        baseline = compute_cllr_min(scores, labels)

        for transform in (
            lambda s: s * 7.0 + 3.0,
            np.exp,
            lambda s: s**3,
            np.arctan,
        ):
            assert compute_cllr_min(transform(scores), labels) == pytest.approx(
                baseline, abs=1e-9
            )

    def test_never_exceeds_cllr(self, rng) -> None:
        """C_llr_min minimises C_llr over monotonic recalibrations, by definition."""
        for _ in range(50):
            n = int(rng.integers(80, 400))
            labels = rng.integers(0, 2, n)
            if labels.sum() < 5 or (1 - labels).sum() < 5:
                continue
            scores = rng.standard_normal(n) * rng.uniform(0.1, 8.0)
            assert compute_cllr(scores, labels) >= compute_cllr_min(scores, labels) - 1e-9


class TestPav:
    def test_agrees_with_scipy_isotonic_regression(self, rng) -> None:
        for _ in range(25):
            n = int(rng.integers(20, 300))
            scores = rng.standard_normal(n)
            labels = rng.integers(0, 2, n)
            ours = pool_adjacent_violators(scores, labels)
            order = np.argsort(scores, kind="stable")
            theirs = isotonic_regression(labels[order].astype(float)).x
            assert np.allclose(ours.fitted[order], theirs, atol=1e-10)

    def test_fitted_values_are_non_decreasing_in_score(self, rng) -> None:
        scores = rng.standard_normal(200)
        labels = rng.integers(0, 2, 200)
        result = pool_adjacent_violators(scores, labels)
        order = np.argsort(scores, kind="stable")
        assert np.all(np.diff(result.fitted[order]) >= -1e-12)

    def test_prior_correction_makes_it_a_likelihood_ratio(self, rng) -> None:
        """A likelihood ratio must not shift with the trial set's class balance.

        Without dividing out the empirical prior, the same evidence evaluated on
        a balanced and an unbalanced set would report different strengths.
        """

        def calibrated_at(n_same: int, n_different: int, probe: float) -> float:
            gen = np.random.default_rng(11)
            scores = np.concatenate(
                [gen.normal(2.0, 1.0, n_same), gen.normal(0.0, 1.0, n_different)]
            )
            labels = np.concatenate([np.ones(n_same), np.zeros(n_different)]).astype(int)
            calibrated = pav_calibrate(scores, labels)
            order = np.argsort(scores)
            return float(np.interp(probe, scores[order], calibrated[order]))

        balanced = calibrated_at(3000, 3000, 2.0)
        unbalanced = calibrated_at(3000, 30000, 2.0)
        assert abs(balanced - unbalanced) / math.log(10.0) < 0.4


class TestEer:
    @pytest.mark.parametrize("separation", [1.0, 2.0, 4.0])
    def test_matches_the_analytic_value(self, separation: float) -> None:
        """Two unit-variance Gaussians separated by d have EER = Phi(-d/2)."""
        gen = np.random.default_rng(7)
        n = 120_000
        scores = np.concatenate(
            [
                gen.normal(separation / 2, 1.0, n),
                gen.normal(-separation / 2, 1.0, n),
            ]
        )
        labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
        assert compute_eer(scores, labels) == pytest.approx(
            float(norm.cdf(-separation / 2)), abs=0.005
        )


class TestCalibrators:
    @pytest.fixture
    def development(self, rng):
        labels = np.concatenate([np.ones(2000), np.zeros(2000)]).astype(int)
        # Badly scaled and offset: right ordering, meaningless numbers.
        scores = np.concatenate(
            [rng.normal(60.0, 20.0, 2000), rng.normal(10.0, 20.0, 2000)]
        )
        return scores, labels

    def test_logistic_removes_calibration_loss(self, development) -> None:
        scores, labels = development
        before = evaluate(scores, labels, include_bounds=False)
        calibrator = LogisticCalibrator().fit(scores, labels)
        after = evaluate(calibrator.transform(scores), labels, include_bounds=False)

        assert before.calibration_loss > 1.0
        assert after.calibration_loss < 0.05
        # Calibration is a monotonic transform, so discrimination is untouched.
        assert after.c_llr_min == pytest.approx(before.c_llr_min, abs=1e-9)

    def test_unfitted_calibrator_refuses(self) -> None:
        with pytest.raises(ModelNotTrainedError):
            LogisticCalibrator().calibrate(1.0)

    def test_refuses_too_few_development_trials(self) -> None:
        with pytest.raises(InsufficientDataError):
            LogisticCalibrator().fit(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 1]))

    def test_logistic_slope_is_non_negative(self, development) -> None:
        """A negative slope would invert the score's meaning — always a symptom."""
        scores, labels = development
        slope, _ = LogisticCalibrator().fit(scores, labels).parameters
        assert slope >= 0.0

    @pytest.mark.parametrize(
        "calibrator_class",
        [LogisticCalibrator, IsotonicCalibrator, KernelDensityCalibrator],
    )
    def test_all_calibrators_reduce_calibration_loss(
        self, calibrator_class, development
    ) -> None:
        scores, labels = development
        calibrator = calibrator_class().fit(scores, labels)
        after = evaluate(calibrator.transform(scores), labels, include_bounds=False)
        assert after.calibration_loss < 0.1

    def test_calibration_is_reproducible(self, development) -> None:
        """The objective is convex, so refitting must give the same answer."""
        scores, labels = development
        first = LogisticCalibrator().fit(scores, labels).parameters
        second = LogisticCalibrator().fit(scores, labels).parameters
        assert first == pytest.approx(second, rel=1e-10)


class TestKernelDensityScaling:
    """The kernel matrix is ``n_query x n_sample`` and both grow with the
    validation set. Held whole, twenty thousand of each is 3.5 GB — so this
    calibrator used to fail on exactly the validation sets large enough to
    justify choosing it. It is evaluated in blocks over the query axis instead.
    """

    @staticmethod
    def _unchunked(query: np.ndarray, sample: np.ndarray, bandwidth: float) -> np.ndarray:
        from scipy.special import logsumexp

        z = (query[:, None] - sample[None, :]) / bandwidth
        kernels = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - np.log(bandwidth)
        return logsumexp(kernels, axis=1) - np.log(sample.size)

    def test_blocking_does_not_change_the_result(self) -> None:
        rng = np.random.default_rng(0)
        query = rng.standard_normal(4000)
        sample = rng.standard_normal(3000)
        produced = _gaussian_kde_log_density(query, sample, 0.3)
        assert np.allclose(produced, self._unchunked(query, sample, 0.3), atol=1e-12)

    def test_a_query_smaller_than_one_block_is_unaffected(self) -> None:
        rng = np.random.default_rng(1)
        query = rng.standard_normal(7)
        sample = rng.standard_normal(11)
        produced = _gaussian_kde_log_density(query, sample, 0.5)
        assert np.allclose(produced, self._unchunked(query, sample, 0.5), atol=1e-12)

    def test_a_realistic_validation_set_is_evaluable(self) -> None:
        """Twenty thousand trials of each class is an ordinary evaluation for
        this system, and used to raise MemoryError while trying to allocate
        3.5 GB to produce twenty thousand numbers."""
        rng = np.random.default_rng(2)
        scores = np.concatenate(
            [rng.normal(2.0, 1.0, 11_000), rng.normal(-2.0, 1.0, 11_000)]
        )
        labels = np.array([1] * 11_000 + [0] * 11_000, dtype=np.int64)

        calibrator = KernelDensityCalibrator().fit(scores, labels)
        calibrated = calibrator.transform(scores)

        assert calibrated.shape == scores.shape
        assert np.all(np.isfinite(calibrated))
        # Well-separated classes: the mapping must still order them correctly.
        assert calibrated[labels == 1].mean() > calibrated[labels == 0].mean()

    def test_an_empty_sample_is_refused_rather_than_dividing_by_zero(self) -> None:
        with pytest.raises(CalibrationError, match="empty sample"):
            _gaussian_kde_log_density(np.array([0.0]), np.array([]), 1.0)


class TestEmpiricalBounds:
    def test_bound_widens_with_evidence(self) -> None:
        """More trials support stronger claims; the bound must reflect that."""
        previous = 0.0
        for n in (200, 2000, 20000):
            gen = np.random.default_rng(3)
            labels = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)
            scores = np.where(labels == 1, gen.normal(3, 1, n), gen.normal(-3, 1, n))
            bounds = empirical_bounds(scores, labels)
            assert bounds.upper_log10 > previous
            previous = bounds.upper_log10

    def test_extrapolation_is_bounded_not_reported(self, rng) -> None:
        """A calibration asked for a strength beyond its data will supply one."""
        labels = np.concatenate([np.ones(2000), np.zeros(2000)]).astype(int)
        scores = np.concatenate(
            [rng.normal(60.0, 20.0, 2000), rng.normal(10.0, 20.0, 2000)]
        )
        calibrator = LogisticCalibrator().fit(scores, labels)

        # A score far beyond anything in development.
        result = calibrator.calibrate(400.0)
        assert result.was_bounded
        assert result.extrapolation_log10 > 5.0
        assert abs(result.log_lr.log10) <= abs(result.unbounded_log_lr / math.log(10.0))

    def test_bounds_are_applied_elementwise(self) -> None:
        gen = np.random.default_rng(5)
        labels = np.concatenate([np.ones(500), np.zeros(500)]).astype(int)
        scores = np.where(labels == 1, gen.normal(3, 1, 1000), gen.normal(-3, 1, 1000))
        bounds = empirical_bounds(scores, labels)
        clipped = apply_bounds(np.array([-100.0, 0.0, 100.0]), bounds)
        assert clipped[0] == pytest.approx(bounds.lower_log_lr)
        assert clipped[2] == pytest.approx(bounds.upper_log_lr)


class TestAsReported:
    """The array form of ``calibrate``, and the reason it is not written twice.

    ``data/reports/calibrator_comparison.json`` came to hold unbounded values
    under a schema the results document reads as bounded ones, because one
    script clipped and another did not. Both now call this function.
    """

    @pytest.fixture
    def fitted(self, rng):
        labels = np.concatenate([np.ones(2000), np.zeros(2000)]).astype(int)
        scores = np.concatenate(
            [rng.normal(60.0, 20.0, 2000), rng.normal(10.0, 20.0, 2000)]
        )
        return LogisticCalibrator().fit(scores, labels), scores, labels

    def test_agrees_with_the_scalar_path_elementwise(self, fitted) -> None:
        """The anti-drift check: the two routes to a reported LR are one route."""
        calibrator, scores, _ = fitted
        probe = np.concatenate([scores[:50], np.array([-500.0, 0.0, 500.0])])
        expected = np.array([calibrator.calibrate(s).log_lr.value for s in probe])
        assert as_reported(calibrator, probe) == pytest.approx(expected, abs=1e-12)

    def test_differs_from_transform_where_the_clip_bites(self, fitted) -> None:
        calibrator, _, _ = fitted
        far = np.array([-500.0, 500.0])
        assert not np.allclose(as_reported(calibrator, far), calibrator.transform(far))

    def test_never_leaves_the_supported_range(self, fitted) -> None:
        calibrator, scores, _ = fitted
        bounds = calibrator.bounds
        reported = as_reported(calibrator, np.concatenate([scores, [-1e6, 1e6]]))
        assert reported.min() >= bounds.lower_log_lr - 1e-12
        assert reported.max() <= bounds.upper_log_lr + 1e-12

    def test_order_preserving_calibrators_share_their_bounds(self, fitted) -> None:
        """§15's point, as a test rather than as prose.

        ``empirical_bounds`` is computed from a PAV fit, which reads only the
        rank order of the scores. A logistic map is affine-increasing and an
        isotonic map is monotone, so neither changes that order and both must
        arrive at the same bounds. Three calibrator families landing within 0.01
        of one another after clipping is therefore close to a necessity rather
        than a finding about calibration, which is what §5 read it as.
        """
        _, scores, labels = fitted
        logistic = LogisticCalibrator().fit(scores, labels).bounds
        isotonic = IsotonicCalibrator().fit(scores, labels).bounds
        assert logistic.lower_log_lr == pytest.approx(isotonic.lower_log_lr, abs=1e-9)
        assert logistic.upper_log_lr == pytest.approx(isotonic.upper_log_lr, abs=1e-9)
