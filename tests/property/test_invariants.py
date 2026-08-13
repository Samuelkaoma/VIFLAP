"""Property-based tests over the mathematical invariants.

Example-based tests check the cases somebody thought of. These check properties
that must hold for *every* input, which is the right shape for the claims this
system makes: "C_llr is never below C_llr_min" is not a statement about a
fixture, and testing it on one is testing something weaker than the claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from viflap.analysis.calibration.metrics import compute_cllr, compute_cllr_min
from viflap.analysis.calibration.pav import pool_adjacent_violators
from viflap.domain.hypotheses import PosteriorAssessment, PriorOdds
from viflap.domain.values import (
    LogLikelihoodRatio,
    LogOdds,
    UncertaintyInterval,
    logistic,
)

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

finite_log_lrs = st.floats(
    min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False
)


@SETTINGS
@given(finite_log_lrs)
def test_log_lr_negation_is_an_involution(value: float) -> None:
    llr = LogLikelihoodRatio(value)
    negated = -llr
    assert (-negated).value == pytest.approx(llr.value)


@SETTINGS
@given(finite_log_lrs)
def test_strength_band_depends_only_on_magnitude(value: float) -> None:
    """Which is why direction must always be rendered alongside it."""
    assert LogLikelihoodRatio(value).strength is LogLikelihoodRatio(-value).strength


@SETTINGS
@given(finite_log_lrs, st.integers(min_value=2, max_value=10_000_000))
def test_posterior_probability_always_valid(log_lr: float, database_size: int) -> None:
    """No combination of evidence and prior may produce an invalid probability.

    This is the property the log-odds formulation exists to guarantee. The
    linear form underflows for small priors and overflows for strong evidence,
    and a national-scale search occupies exactly that corner.
    """
    prior = PriorOdds.uniform_over_database(database_size)
    posterior = PosteriorAssessment.from_evidence(prior, LogLikelihoodRatio(log_lr))
    probability = posterior.probability.value
    assert 0.0 <= probability <= 1.0
    assert math.isfinite(probability)


@SETTINGS
@given(finite_log_lrs, st.integers(min_value=2, max_value=1_000_000))
def test_bayes_rule_is_exactly_addition(log_lr: float, database_size: int) -> None:
    prior = PriorOdds.uniform_over_database(database_size)
    posterior = PosteriorAssessment.from_evidence(prior, LogLikelihoodRatio(log_lr))
    assert posterior.posterior_log_odds.value == pytest.approx(
        prior.log_odds.value + log_lr, abs=1e-9
    )


@SETTINGS
@given(st.floats(min_value=-16.0, max_value=16.0, allow_nan=False, allow_infinity=False))
def test_log_odds_probability_round_trip(value: float) -> None:
    """Log-odds survive a round trip through probability within the useful range.

    Bounded at +/-16 because the round trip is inherently lossy beyond it, and
    the loss is a property of binary floating point rather than of this code: at
    log-odds 25 the probability is ``1 - 1.4e-11``, and a double has no bits
    left to distinguish it from neighbouring values.

    This is exactly why posteriors are carried as log-odds throughout and
    converted to a probability only for display. A system that stored the
    probability would have discarded the difference between "overwhelming" and
    "certain", which is a distinction this system must never lose.
    """
    probability = LogOdds(value).to_probability()
    assume(0.0 < probability.value < 1.0)
    assert probability.to_log_odds().value == pytest.approx(value, abs=1e-6)


@SETTINGS
@given(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
)
def test_interval_shift_preserves_width(centre: float, half: float, shift: float) -> None:
    interval = UncertaintyInterval(centre - half, centre + half)
    shifted = interval.shifted_by(shift)
    assert shifted.width == pytest.approx(interval.width)
    assert shifted.midpoint == pytest.approx(interval.midpoint + shift)


scores_and_labels = st.integers(min_value=40, max_value=300).flatmap(
    lambda n: st.tuples(
        arrays(
            np.float64,
            n,
            elements=st.floats(
                min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False
            ),
        ),
        arrays(np.int64, n, elements=st.integers(min_value=0, max_value=1)),
    )
)


@SETTINGS
@given(scores_and_labels)
def test_cllr_never_below_cllr_min(data) -> None:
    """C_llr_min minimises C_llr over monotonic recalibrations, by definition.

    A violation would mean the metric implementation is wrong, never that the
    system under evaluation has an unusual property.
    """
    scores, labels = data
    assume(int(np.count_nonzero(labels == 1)) >= 5)
    assume(int(np.count_nonzero(labels == 0)) >= 5)
    assert compute_cllr(scores, labels) >= compute_cllr_min(scores, labels) - 1e-9


@SETTINGS
@given(scores_and_labels)
def test_cllr_is_non_negative(data) -> None:
    scores, labels = data
    assume(int(np.count_nonzero(labels == 1)) >= 5)
    assume(int(np.count_nonzero(labels == 0)) >= 5)
    assert compute_cllr(scores, labels) >= 0.0


# Scores on a coarse grid, so an affine map within the bounds below cannot
# collapse distinct values into ties. Generating unconstrained floats lets
# hypothesis produce arrays spanning fifty orders of magnitude, where any shift
# absorbs the small entries — a real effect, asserted in its own test below, but
# one that would cause this property to discard almost every case and quietly
# stop testing the domain it names.
gridded_scores_and_labels = st.integers(min_value=40, max_value=200).flatmap(
    lambda n: st.tuples(
        arrays(
            np.float64,
            n,
            elements=st.integers(min_value=-3000, max_value=3000).map(
                lambda value: value / 100.0
            ),
        ),
        arrays(np.int64, n, elements=st.integers(min_value=0, max_value=1)),
    )
)


@SETTINGS
@given(
    gridded_scores_and_labels,
    st.floats(min_value=0.5, max_value=20.0, allow_nan=False),
    st.floats(min_value=-20.0, max_value=20.0, allow_nan=False),
)
def test_cllr_min_invariant_under_affine_transformation(data, scale, shift) -> None:
    """Discrimination cannot be changed by rescaling the scores.

    Conditioned on the transformation preserving the **ordering relation**,
    including which scores are tied. That is the actual mathematical
    precondition, and in floating point an affine map does not always satisfy
    it: adding 1.0 to a score of ``2.3e-58`` absorbs it entirely, so a score
    that was strictly greatest becomes tied with the rest.

    Checking the permutation is not enough, and the difference is the kind a
    property test exists to find. ``argsort`` is stable, so it returns the same
    index order whether or not the transform created new ties — the permutation
    is unchanged while the tie structure is not. Pool-adjacent-violators pools
    tied scores into one block, so a transform that creates ties genuinely
    changes ``C_llr_min``. The precondition is therefore stated on the *dense
    rank*, which encodes both the order and the ties.
    """
    from scipy.stats import rankdata

    scores, labels = data
    assume(int(np.count_nonzero(labels == 1)) >= 5)
    assume(int(np.count_nonzero(labels == 0)) >= 5)

    transformed_scores = scores * scale + shift
    assume(
        np.array_equal(
            rankdata(scores, method="dense"),
            rankdata(transformed_scores, method="dense"),
        )
    )

    baseline = compute_cllr_min(scores, labels)
    transformed = compute_cllr_min(transformed_scores, labels)
    assert transformed == pytest.approx(baseline, abs=1e-9)


def test_absorption_changes_discrimination_and_that_is_correct() -> None:
    """The complement of the property above, asserted rather than assumed away.

    An affine map that collapses distinct scores into ties does change
    ``C_llr_min``, and it should: pool-adjacent-violators pools tied scores, so
    a system whose scores have become indistinguishable genuinely discriminates
    less. Adding 1.0 to a score of ``1e-58`` absorbs it entirely.

    Recorded as a test so that the exclusion in the property above is documented
    behaviour rather than a convenient filter.
    """
    scores = np.concatenate([np.full(20, 1e-58), np.zeros(20)])
    labels = np.concatenate([np.ones(20), np.zeros(20)]).astype(np.int64)

    separated = compute_cllr_min(scores, labels)
    absorbed = compute_cllr_min(scores + 1.0, labels)

    # Before absorption the classes are perfectly separable; after, every score
    # is identical and the system can no longer distinguish them at all.
    assert separated < 0.01
    assert absorbed == pytest.approx(1.0, abs=1e-9)


@SETTINGS
@given(scores_and_labels)
def test_pav_output_is_monotonic_in_score(data) -> None:
    scores, labels = data
    assume(scores.size >= 5)
    result = pool_adjacent_violators(scores, labels)
    order = np.argsort(scores, kind="stable")
    assert np.all(np.diff(result.fitted[order]) >= -1e-12)


@SETTINGS
@given(scores_and_labels)
def test_pav_preserves_the_total(data) -> None:
    """Isotonic regression is a projection: it redistributes without adding."""
    scores, labels = data
    assume(scores.size >= 5)
    result = pool_adjacent_violators(scores, labels)
    assert float(result.fitted.sum()) == pytest.approx(float(labels.sum()), abs=1e-6)


@SETTINGS
@given(st.floats(min_value=-700.0, max_value=700.0, allow_nan=False))
def test_logistic_stays_within_the_unit_interval(value: float) -> None:
    result = logistic(value)
    assert 0.0 <= result <= 1.0
    assert math.isfinite(result)
