"""The hypothesis tests, as executable protocols.

Each of the proposal's hypotheses H1 to H7 states a falsification condition.
This module turns those conditions into code, so that a hypothesis is decided by
a procedure fixed in advance rather than by inspection of the results.

That ordering is the point. A falsification condition written after seeing the
data is not a falsification condition. Encoding them here — with their
thresholds as constructor arguments recorded in the output — means the decision
rule is committed before the experiment and travels with its result.

Every protocol reports three things: the estimate with its speaker-level
interval, whether the hypothesis is supported, and whether it is **falsified**.
Those are not complements. A result can be neither — the interval spanning both
thresholds, meaning the experiment was underpowered — and reporting that as
"not supported" would confuse an absence of evidence with evidence of absence.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.calibration.metrics import compute_cllr, compute_cllr_min
from viflap.domain.errors import InsufficientDataError
from viflap.domain.metrics import Estimate
from viflap.evaluation.splits import bootstrap_over_speakers

__all__ = [
    "H1ChannelViability",
    "H2DisguiseResistance",
    "H3DisguiseDecay",
    "H4CrossLingualPenalty",
    "H5FusionSuperiority",
    "H7SyntheticGating",
    "HypothesisOutcome",
    "holm_bonferroni",
]


@dataclass(frozen=True, slots=True)
class HypothesisOutcome:
    """The decision on one hypothesis, with the rule that produced it."""

    hypothesis: str
    statement: str
    estimate: Estimate
    supported: bool
    falsified: bool
    decision_rule: str
    detail: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.supported and self.falsified:
            raise ValueError(
                "a hypothesis cannot be both supported and falsified; the "
                "decision rule is inconsistent"
            )

    @property
    def is_inconclusive(self) -> bool:
        """Neither supported nor falsified.

        The honest outcome of an underpowered experiment, and one that must be
        reported as itself. Collapsing it into "not supported" turns a
        statement about the experiment into a statement about the world.
        """
        return not self.supported and not self.falsified

    def describe(self) -> str:
        if self.supported:
            verdict = "SUPPORTED"
        elif self.falsified:
            verdict = "FALSIFIED"
        else:
            verdict = "INCONCLUSIVE — the interval spans both decision thresholds"
        return (
            f"{self.hypothesis}: {verdict}\n"
            f"  {self.statement}\n"
            f"  Estimate: {self.estimate}\n"
            f"  Rule: {self.decision_rule}"
        )


class H1ChannelViability:
    """H1 — speaker embeddings survive narrowband coding well enough to be used.

    Supported if ``C_llr_min <= 0.30``; falsified if ``C_llr_min > 0.50``, at
    which point the acoustic stream contributes too little to justify its
    inclusion and the thesis proceeds on non-acoustic evidence.

    The decision is made on the **interval**, not the point estimate. A point
    estimate of 0.29 with an interval reaching 0.55 does not support the
    hypothesis; it means the experiment cannot decide it.
    """

    def __init__(self, support_below: float = 0.30, falsify_above: float = 0.50) -> None:
        self._support = support_below
        self._falsify = falsify_above

    @property
    def support_below(self) -> float:
        """The threshold the upper bound must fall below for support."""
        return self._support

    @property
    def falsify_above(self) -> float:
        """The threshold the lower bound must exceed for falsification.

        Exposed, along with :attr:`support_below`, so that a report aggregating
        many cells can state the rule it applied without reaching into the
        object. A report that describes its own decision rule from a hardcoded
        string is a report that can disagree with the code that produced it.
        """
        return self._falsify

    def run(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int64],
        speakers: Sequence[Hashable],
        condition: str = "unspecified",
        n_resamples: int = 1000,
    ) -> HypothesisOutcome:
        estimate = bootstrap_over_speakers(
            compute_cllr_min, scores, labels, speakers, n_resamples=n_resamples
        )
        return HypothesisOutcome(
            hypothesis="H1",
            statement=(
                f"Speaker embeddings from narrowband-coded telephony retain "
                f"sufficient discriminative information under condition "
                f"'{condition}'."
            ),
            estimate=estimate,
            supported=estimate.upper <= self._support,
            falsified=estimate.lower > self._falsify,
            decision_rule=(
                f"supported if the upper bound of C_llr_min <= {self._support}; "
                f"falsified if the lower bound > {self._falsify}"
            ),
            detail={"condition_count": 1.0},
        )


class H2DisguiseResistance:
    """H2 — feature classes degrade unequally under deliberate disguise.

    Supported if the anatomically constrained classes (nasal resonance,
    articulatory timing) retain measurably more discriminative information under
    disguise than the globally spectral and F0-derived classes. Falsified if
    degradation is statistically indistinguishable across classes, which would
    remove the basis for disguise-aware weighting.
    """

    def __init__(self, min_separation: float = 0.05) -> None:
        self._min_separation = min_separation

    def run(
        self,
        by_feature_class: Mapping[str, NDArray[np.float64]],
        labels: NDArray[np.int64],
        speakers: Sequence[Hashable],
        resistant_classes: Sequence[str],
        n_resamples: int = 1000,
    ) -> HypothesisOutcome:
        if len(by_feature_class) < 2:
            raise InsufficientDataError(
                "at least two feature classes are needed to compare their degradation"
            )

        estimates = {
            name: bootstrap_over_speakers(
                compute_cllr_min, scores, labels, speakers, n_resamples=n_resamples
            )
            for name, scores in by_feature_class.items()
        }

        resistant = [estimates[n] for n in resistant_classes if n in estimates]
        fragile = [e for n, e in estimates.items() if n not in resistant_classes]
        if not resistant or not fragile:
            raise InsufficientDataError(
                "both resistant and fragile feature classes must be present"
            )

        best_resistant = min(resistant, key=lambda e: e.value)
        best_fragile = min(fragile, key=lambda e: e.value)
        separation = best_fragile.value - best_resistant.value

        return HypothesisOutcome(
            hypothesis="H2",
            statement=(
                "Under deliberate disguise, anatomically constrained feature "
                "classes retain more discriminative information than globally "
                "spectral and F0-derived classes."
            ),
            estimate=Estimate(
                value=separation,
                lower=best_fragile.lower - best_resistant.upper,
                upper=best_fragile.upper - best_resistant.lower,
                confidence_level=best_resistant.confidence_level,
                resampling_unit="speakers",
                n_resamples=min(best_resistant.n_resamples, best_fragile.n_resamples),
            ),
            # Intervals must not overlap zero: a separation whose interval
            # includes zero is consistent with the classes degrading equally,
            # which is the falsification condition rather than weak support.
            supported=(best_fragile.lower - best_resistant.upper) > self._min_separation,
            falsified=abs(separation) < self._min_separation
            and (best_fragile.lower - best_resistant.upper) < 0,
            decision_rule=(
                f"supported if the resistant classes' C_llr_min interval lies "
                f"below the fragile classes' by more than {self._min_separation}"
            ),
            detail={f"{name}_c_llr_min": e.value for name, e in estimates.items()},
        )


class H3DisguiseDecay:
    """H3 — disguise consistency degrades over the course of a call.

    Supported if the final third of a disguised call yields materially more
    discriminative information than the first third. The prediction rests on
    attention being a depleting resource, and an offender executing a deception
    spending it on the deception.
    """

    def __init__(self, min_improvement: float = 0.03) -> None:
        self._min_improvement = min_improvement

    def run(
        self,
        first_third_scores: NDArray[np.float64],
        final_third_scores: NDArray[np.float64],
        labels: NDArray[np.int64],
        speakers: Sequence[Hashable],
        n_resamples: int = 1000,
    ) -> HypothesisOutcome:
        # Paired by construction: the same trials measured on two segments of
        # the same calls. Resampling speakers keeps the pairing, so the
        # difference is a within-speaker contrast and the between-speaker
        # variance — which is large — cancels rather than obscuring the effect.
        def paired_difference(
            indices_scores: NDArray[np.float64], resampled_labels: NDArray[np.int64]
        ) -> float:
            half = indices_scores.shape[0] // 2
            return compute_cllr_min(
                indices_scores[:half], resampled_labels[:half]
            ) - compute_cllr_min(indices_scores[half:], resampled_labels[half:])

        stacked = np.concatenate([first_third_scores, final_third_scores])
        stacked_labels = np.concatenate([labels, labels])
        stacked_speakers = list(speakers) + list(speakers)

        estimate = bootstrap_over_speakers(
            paired_difference,
            stacked,
            stacked_labels,
            stacked_speakers,
            n_resamples=n_resamples,
        )
        return HypothesisOutcome(
            hypothesis="H3",
            statement=(
                "Discriminative information recoverable from the final third of "
                "a disguised call exceeds that from the first third."
            ),
            estimate=estimate,
            supported=estimate.lower > self._min_improvement,
            falsified=estimate.upper < 0.0,
            decision_rule=(
                f"supported if the lower bound of (first-third C_llr_min minus "
                f"final-third C_llr_min) exceeds {self._min_improvement}"
            ),
        )


class H4CrossLingualPenalty:
    """H4 — English-trained extractors incur a penalty on code-switched speech.

    A negative result here is valuable and is reported as such: if no penalty
    exists, that is a finding about the transferability of these models to
    Bantu-language telephony, which the literature does not currently establish
    either way.
    """

    def __init__(self, min_penalty: float = 0.02) -> None:
        self._min_penalty = min_penalty

    def run(
        self,
        baseline_scores: NDArray[np.float64],
        target_scores: NDArray[np.float64],
        labels: NDArray[np.int64],
        speakers: Sequence[Hashable],
        target_description: str,
        n_resamples: int = 1000,
    ) -> HypothesisOutcome:
        baseline = bootstrap_over_speakers(
            compute_cllr_min, baseline_scores, labels, speakers, n_resamples=n_resamples
        )
        target = bootstrap_over_speakers(
            compute_cllr_min, target_scores, labels, speakers, n_resamples=n_resamples
        )
        penalty = target.value - baseline.value

        return HypothesisOutcome(
            hypothesis="H4",
            statement=(
                f"Embedding extractors trained on English-dominant corpora incur "
                f"a performance penalty on {target_description}."
            ),
            estimate=Estimate(
                value=penalty,
                lower=target.lower - baseline.upper,
                upper=target.upper - baseline.lower,
                confidence_level=baseline.confidence_level,
                resampling_unit="speakers",
                n_resamples=min(baseline.n_resamples, target.n_resamples),
            ),
            supported=(target.lower - baseline.upper) > self._min_penalty,
            falsified=(target.upper - baseline.lower) < self._min_penalty,
            decision_rule=(
                f"supported if the penalty interval lies above "
                f"{self._min_penalty}; falsified if it lies below — a genuine "
                f"negative result worth reporting"
            ),
            detail={"baseline_c_llr_min": baseline.value, "target_c_llr_min": target.value},
        )


class H5FusionSuperiority:
    """H5 — dependence-corrected fusion beats the best single stream.

    The central hypothesis, and the one whose falsification would be most
    damaging — and most publishable. If fusion gains vanish once dependence is
    properly modelled, an entire class of multimodal forensic systems rests on
    an unexamined assumption, and the field needs to know.

    Two comparisons, both required. Fusion must beat the best single stream, and
    it must differ from naive summation. Beating the best stream while matching
    naive summation would mean the dependence modelling contributed nothing.
    """

    def __init__(self, min_improvement: float = 0.02) -> None:
        self._min_improvement = min_improvement

    def run(
        self,
        single_stream_scores: Mapping[str, NDArray[np.float64]],
        fused_scores: NDArray[np.float64],
        naive_scores: NDArray[np.float64],
        labels: NDArray[np.int64],
        speakers: Sequence[Hashable],
        n_resamples: int = 1000,
    ) -> HypothesisOutcome:
        if not single_stream_scores:
            raise InsufficientDataError("at least one single-stream result is required")

        per_stream = {
            name: bootstrap_over_speakers(
                compute_cllr, scores, labels, speakers, n_resamples=n_resamples
            )
            for name, scores in single_stream_scores.items()
        }
        best_name = min(per_stream, key=lambda n: per_stream[n].value)
        best = per_stream[best_name]

        fused = bootstrap_over_speakers(
            compute_cllr, fused_scores, labels, speakers, n_resamples=n_resamples
        )
        naive = bootstrap_over_speakers(
            compute_cllr, naive_scores, labels, speakers, n_resamples=n_resamples
        )

        improvement = best.value - fused.value
        beats_best_stream = (best.lower - fused.upper) > self._min_improvement
        differs_from_naive = abs(naive.value - fused.value) > self._min_improvement

        return HypothesisOutcome(
            hypothesis="H5",
            statement=(
                "Dependence-corrected fusion of the evidence streams achieves a "
                "lower C_llr than the best single stream, and differs from naive "
                "conditional-independence summation."
            ),
            estimate=Estimate(
                value=improvement,
                lower=best.lower - fused.upper,
                upper=best.upper - fused.lower,
                confidence_level=fused.confidence_level,
                resampling_unit="speakers",
                n_resamples=min(best.n_resamples, fused.n_resamples),
            ),
            supported=beats_best_stream and differs_from_naive,
            falsified=(best.upper - fused.lower) < 0.0,
            decision_rule=(
                f"supported if fusion's C_llr interval lies below the best single "
                f"stream's by more than {self._min_improvement} AND differs from "
                f"naive summation by more than the same margin. "
                f"Beats best single stream: {beats_best_stream} "
                f"(margin {best.lower - fused.upper:+.4f}). "
                f"Differs from naive summation: {differs_from_naive} "
                f"(margin {abs(naive.value - fused.value):+.4f}). "
                f"Both are required: beating the best stream while performing no "
                f"differently from naive summation would mean the dependence "
                f"modelling contributed nothing."
            ),
            detail={
                "best_single_stream_c_llr": best.value,
                "fused_c_llr": fused.value,
                "naive_c_llr": naive.value,
                "margin_over_best_stream": best.lower - fused.upper,
                "margin_over_naive": abs(naive.value - fused.value),
                **{f"{name}_c_llr": e.value for name, e in per_stream.items()},
            },
        )


class H7SyntheticGating:
    """H7 — synthetic speech is detectable well enough to gate acoustic evidence.

    Evaluated on **held-out attack types only**. Performance against synthesis
    methods present in training is an upper bound that will not be seen in
    service, because an offender using a synthesiser the training set contains
    is not the offender this system will meet.
    """

    def __init__(self, max_cllr: float = 0.40) -> None:
        self._max_cllr = max_cllr

    def run(
        self,
        detector_scores: NDArray[np.float64],
        labels: NDArray[np.int64],
        speakers: Sequence[Hashable],
        held_out_attacks: Sequence[str],
        n_resamples: int = 1000,
    ) -> HypothesisOutcome:
        if not held_out_attacks:
            raise InsufficientDataError(
                "H7 must be evaluated on attack types excluded from training; "
                "evaluating on seen attacks reports a number with no bearing on "
                "how the countermeasure will behave in service"
            )
        estimate = bootstrap_over_speakers(
            compute_cllr, detector_scores, labels, speakers, n_resamples=n_resamples
        )
        return HypothesisOutcome(
            hypothesis="H7",
            statement=(
                f"Synthetic and converted speech generated by methods absent "
                f"from training ({', '.join(held_out_attacks)}) is detected well "
                f"enough to gate acoustic evidence."
            ),
            estimate=estimate,
            supported=estimate.upper <= self._max_cllr,
            falsified=estimate.lower > 1.0,
            decision_rule=(
                f"supported if the upper bound of C_llr on held-out attacks "
                f"<= {self._max_cllr}; falsified if the lower bound exceeds 1.0, "
                f"i.e. the detector is worse than uninformative on unseen attacks"
            ),
            detail={"n_held_out_attacks": float(len(held_out_attacks))},
        )


def holm_bonferroni(
    p_values: Mapping[str, float], family_alpha: float = 0.05
) -> dict[str, bool]:
    """Holm-Bonferroni correction over a family of tests.

    Required wherever several hypotheses are tested on one dataset. Testing
    seven hypotheses at the five percent level gives roughly a thirty percent
    chance of at least one spurious result, and a thesis reporting seven tests
    without correction is reporting that chance as a finding.

    Holm rather than plain Bonferroni: it controls the same family-wise error
    rate while rejecting strictly more often, so it costs no rigour and less
    power.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    n = len(ordered)
    decisions: dict[str, bool] = {}
    for rank, (name, p_value) in enumerate(ordered):
        threshold = family_alpha / (n - rank)
        if p_value <= threshold:
            decisions[name] = True
        else:
            # Once one test fails, every less significant test fails too.
            for remaining, _ in ordered[rank:]:
                decisions[remaining] = False
            break
    return decisions
