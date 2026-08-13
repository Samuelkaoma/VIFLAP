"""Numerical value objects for evidential reasoning.

Design notes
------------
**The log domain is canonical.** :class:`LogLikelihoodRatio` — not
:class:`LikelihoodRatio` — is the representation used everywhere inside the
system. Three reasons:

1. Fused evidence routinely exceeds the dynamic range of a 64-bit float in the
   linear domain. A fused ``LR`` of ``10^400`` is arithmetically unremarkable in
   logs and simply does not exist as a ``float``. A system that stores linear
   ratios silently converts strong evidence into ``inf``.
2. Every operation that matters — fusion, applying a prior, propagating along a
   graph path — is addition in the log domain and multiplication in the linear
   one. Addition does not overflow.
3. Calibration is defined, trained and evaluated on log-likelihood-ratios.
   Storing anything else forces a conversion at every boundary.

Conversion to a linear ratio is available but deliberately fallible: it raises
rather than saturating. A likelihood ratio too large to represent is not a very
strong result, it is a symptom that calibration has failed, and rounding it to
``1.8e308`` would conceal that.

**Probabilities of hypotheses are computed in log-odds.** ``posterior_prob =
LR * prior / (1 + LR * prior)`` loses all precision, and then overflows, exactly
where the answer matters. The logistic function applied to summed log-odds does
neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from viflap.domain.errors import InvalidEvidenceError

__all__ = [
    "LOG_FLOAT_MAX",
    "EvidentialStrength",
    "LikelihoodRatio",
    "LogLikelihoodRatio",
    "LogOdds",
    "Probability",
    "UncertaintyInterval",
    "log_logistic",
    "logistic",
]

#: Largest natural logarithm whose exponential is a finite ``float``.
LOG_FLOAT_MAX: Final[float] = 709.782712893384


def logistic(x: float, /) -> float:
    """Numerically stable logistic function ``1 / (1 + exp(-x))``.

    The naive expression overflows for ``x`` below about ``-745``; this branches
    so that :func:`math.exp` is only ever applied to a non-positive argument.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def log_logistic(x: float, /) -> float:
    """Natural logarithm of :func:`logistic`, stable in both tails."""
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


class EvidentialStrength(Enum):
    """Verbal equivalents for likelihood-ratio magnitude.

    The bands follow the convention used in European forensic practice
    (ENFSI guideline, Annex on verbal scales): successive powers of ten, with
    the verbal label attaching to the *evidence*, never to the hypothesis.

    Two properties of this scale are load-bearing:

    - It is symmetric. Evidence supporting the different-source proposition gets
      the same vocabulary with the direction reversed, because ``LR = 1/100`` is
      exactly as informative as ``LR = 100``.
    - The strongest band is open-ended and named for what it is. There is no
      band called "conclusive", because no likelihood ratio is conclusive; the
      posterior depends on the prior, which the system does not hold.
    """

    NO_SUPPORT = "provides no assistance"
    WEAK = "provides weak support for"
    MODERATE = "provides moderate support for"
    MODERATELY_STRONG = "provides moderately strong support for"
    STRONG = "provides strong support for"
    VERY_STRONG = "provides very strong support for"
    EXTREMELY_STRONG = "provides extremely strong support for"

    @classmethod
    def for_log10_lr(cls, log10_lr: float, /) -> EvidentialStrength:
        """Return the band for a base-10 log likelihood ratio.

        The magnitude alone determines the band; the sign determines which
        proposition is supported and is reported separately, never folded into
        the verbal label.
        """
        magnitude = abs(log10_lr)
        if magnitude < 1.0:
            return cls.NO_SUPPORT if magnitude < 0.3 else cls.WEAK
        if magnitude < 2.0:
            return cls.MODERATE
        if magnitude < 3.0:
            return cls.MODERATELY_STRONG
        if magnitude < 4.0:
            return cls.STRONG
        if magnitude < 6.0:
            return cls.VERY_STRONG
        return cls.EXTREMELY_STRONG


@dataclass(frozen=True, slots=True, order=True)
class LogLikelihoodRatio:
    """A log-likelihood-ratio, ``ln[ p(E | H_ss) / p(E | H_ds) ]``.

    This is the canonical evidential quantity in VIFLAP. It is *not* a
    probability, and it does not become one without a prior that the system does
    not possess.

    Invariant: the value is finite. ``NaN`` means a computation failed and must
    not propagate silently; ``+/-inf`` means the model claims one proposition is
    impossible, which no finite quantity of evidence can establish.
    """

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InvalidEvidenceError(
                "log-likelihood-ratio must be finite; an infinite value asserts "
                "that one proposition is impossible, which evidence cannot establish",
                value=self.value,
            )

    # -- Alternative bases ------------------------------------------------

    @property
    def log10(self) -> float:
        """Base-10 log-LR. The reporting scale in forensic practice."""
        return self.value / math.log(10.0)

    @property
    def log2(self) -> float:
        """Base-2 log-LR, used in the ``C_llr`` integrand."""
        return self.value / math.log(2.0)

    # -- Interpretation ---------------------------------------------------

    @property
    def supports_same_source(self) -> bool:
        return self.value > 0.0

    @property
    def supports_different_source(self) -> bool:
        return self.value < 0.0

    @property
    def strength(self) -> EvidentialStrength:
        """Verbal band for this magnitude of evidence."""
        return EvidentialStrength.for_log10_lr(self.log10)

    @property
    def is_representable_as_ratio(self) -> bool:
        """Whether :meth:`to_likelihood_ratio` will succeed."""
        return -LOG_FLOAT_MAX < self.value < LOG_FLOAT_MAX

    # -- Conversions ------------------------------------------------------

    def to_likelihood_ratio(self) -> LikelihoodRatio:
        """Convert to the linear domain.

        Raises
        ------
        InvalidEvidenceError
            If the ratio is not representable as a finite ``float``. This is not
            defensive pedantry: a log-LR beyond +/-709 corresponds to evidence
            far outside anything a calibrated forensic system can support, and
            saturating it to the largest float would present a calibration
            failure as an extremely strong result.
        """
        if not self.is_representable_as_ratio:
            raise InvalidEvidenceError(
                "log-likelihood-ratio is outside the range representable as a "
                "linear ratio; a value this extreme indicates a calibration "
                "failure rather than very strong evidence — report log10 LR instead",
                log_lr=self.value,
                log10_lr=self.log10,
            )
        return LikelihoodRatio(math.exp(self.value))

    def __add__(self, other: LogLikelihoodRatio) -> LogLikelihoodRatio:
        """Sum two log-LRs.

        Note that this is *only* the correct combination rule when the two
        pieces of evidence are conditionally independent given each proposition.
        It is exposed because that assumption is the explicit baseline against
        which dependence-corrected fusion is measured, not because it is
        generally valid here. See :mod:`viflap.analysis.fusion`.
        """
        return LogLikelihoodRatio(self.value + other.value)

    def __neg__(self) -> LogLikelihoodRatio:
        """Invert the propositions: ``ln(1/LR) = -ln(LR)``."""
        return LogLikelihoodRatio(-self.value)

    @classmethod
    def neutral(cls) -> LogLikelihoodRatio:
        """Evidence equally probable under both propositions (``LR = 1``).

        Constructing this deliberately is legitimate. Substituting it for a
        stream that produced no evidence is not: absent data and uninformative
        data are different states, and conflating them fabricates evidence.
        """
        return cls(0.0)

    @classmethod
    def from_log10(cls, log10_lr: float, /) -> LogLikelihoodRatio:
        return cls(log10_lr * math.log(10.0))

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"log10 LR = {self.log10:+.3f}"


@dataclass(frozen=True, slots=True, order=True)
class LikelihoodRatio:
    """A likelihood ratio in the linear domain, ``p(E | H_ss) / p(E | H_ds)``.

    Used at system boundaries where a human or an external format expects a
    ratio. Internal computation uses :class:`LogLikelihoodRatio`.

    Invariants: strictly positive and finite. A ratio of probabilities cannot be
    negative, and zero or infinity would assert that one proposition is
    impossible.
    """

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InvalidEvidenceError("likelihood ratio must be finite", value=self.value)
        if self.value <= 0.0:
            raise InvalidEvidenceError(
                "likelihood ratio must be strictly positive; it is a ratio of "
                "probability densities",
                value=self.value,
            )

    @property
    def log_lr(self) -> LogLikelihoodRatio:
        return LogLikelihoodRatio(math.log(self.value))

    @property
    def log10(self) -> float:
        return math.log10(self.value)

    @property
    def strength(self) -> EvidentialStrength:
        return EvidentialStrength.for_log10_lr(self.log10)

    @property
    def supports_same_source(self) -> bool:
        return self.value > 1.0

    @property
    def supports_different_source(self) -> bool:
        return self.value < 1.0

    @property
    def is_uninformative(self) -> bool:
        """Whether the evidence is equally probable under both propositions."""
        return math.isclose(self.value, 1.0, rel_tol=1e-12, abs_tol=1e-12)

    def inverted(self) -> LikelihoodRatio:
        """The likelihood ratio with the propositions exchanged."""
        return LikelihoodRatio(1.0 / self.value)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        if self.value >= 1.0:
            return f"LR = {self.value:,.4g}"
        return f"LR = 1 / {1.0 / self.value:,.4g}"


@dataclass(frozen=True, slots=True, order=True)
class Probability:
    """A probability in ``[0, 1]``.

    Exists so that a probability cannot be passed where odds are expected, and
    so that the bounds are checked once at construction rather than assumed at
    every use.
    """

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InvalidEvidenceError("probability must be finite", value=self.value)
        if not 0.0 <= self.value <= 1.0:
            raise InvalidEvidenceError("probability must lie in [0, 1]", value=self.value)

    @property
    def percent(self) -> float:
        return self.value * 100.0

    @property
    def complement(self) -> Probability:
        return Probability(1.0 - self.value)

    def to_log_odds(self) -> LogOdds:
        """Convert to log-odds.

        Raises
        ------
        InvalidEvidenceError
            For probabilities of exactly zero or one, whose log-odds are
            infinite. Certainty is not representable as odds, and a system that
            silently clamped it would be asserting near-certainty it was never
            given.
        """
        if self.value <= 0.0 or self.value >= 1.0:
            raise InvalidEvidenceError(
                "probabilities of exactly 0 or 1 have infinite log-odds",
                value=self.value,
            )
        return LogOdds(math.log(self.value / (1.0 - self.value)))

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.percent:.4g}%"


@dataclass(frozen=True, slots=True, order=True)
class LogOdds:
    """Odds on the natural-log scale.

    Both prior and posterior odds are carried in this form. Bayes' rule is then
    addition — ``posterior = prior + log_lr`` — which is exact for any magnitude
    of evidence and any size of database. The linear form of the same
    calculation loses precision for small priors and overflows for strong
    evidence, which is precisely the regime a national-scale database search
    operates in.
    """

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InvalidEvidenceError("log-odds must be finite", value=self.value)

    @property
    def odds(self) -> float:
        """Odds in the linear domain, or ``inf`` if not representable.

        Unlike :meth:`LogLikelihoodRatio.to_likelihood_ratio` this saturates
        rather than raising, because it is used only for display alongside the
        log value, which is always exact.
        """
        if self.value >= LOG_FLOAT_MAX:
            return math.inf
        if self.value <= -LOG_FLOAT_MAX:
            return 0.0
        return math.exp(self.value)

    def to_probability(self) -> Probability:
        """Convert to a probability via the logistic function.

        Exact in both tails: a log-odds of ``-30`` yields ``9.36e-14`` rather
        than underflowing, and ``+800`` yields ``1.0`` rather than ``nan``.
        """
        return Probability(logistic(self.value))

    def updated_with(self, log_lr: LogLikelihoodRatio) -> LogOdds:
        """Apply Bayes' rule: posterior log-odds = prior log-odds + log-LR."""
        return LogOdds(self.value + log_lr.value)

    @classmethod
    def from_odds(cls, odds: float, /) -> LogOdds:
        if not math.isfinite(odds) or odds <= 0.0:
            raise InvalidEvidenceError(
                "odds must be finite and strictly positive", odds=odds
            )
        return cls(math.log(odds))

    @classmethod
    def from_probability(cls, probability: float, /) -> LogOdds:
        return Probability(probability).to_log_odds()

    def __str__(self) -> str:  # pragma: no cover - presentation only
        odds = self.odds
        if odds >= 1.0:
            return f"{odds:,.4g} : 1"
        return f"1 : {1.0 / odds:,.4g}" if odds > 0.0 else "vanishingly small"


@dataclass(frozen=True, slots=True)
class UncertaintyInterval:
    """An interval on a log-likelihood-ratio.

    Interpretation is fixed by construction: these are the bounds of a
    ``confidence_level`` interval on the *log* likelihood ratio, estimated by
    resampling over **speakers** rather than over trials. Trials sharing a
    speaker are not independent, so resampling trials understates variance and
    produces intervals that are tight, symmetric, and wrong.

    The interval is a statement about estimation uncertainty in the system's own
    output. It is not the uncertainty in the proposition, which is a matter of
    the posterior and therefore of the prior.
    """

    lower: float
    upper: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        for name, bound in (("lower", self.lower), ("upper", self.upper)):
            if not math.isfinite(bound):
                raise InvalidEvidenceError(
                    f"uncertainty interval {name} bound must be finite", **{name: bound}
                )
        if self.lower > self.upper:
            raise InvalidEvidenceError(
                "uncertainty interval lower bound exceeds upper bound",
                lower=self.lower,
                upper=self.upper,
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise InvalidEvidenceError(
                "confidence level must lie strictly within (0, 1)",
                confidence_level=self.confidence_level,
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def spans_neutral(self) -> bool:
        """Whether the interval includes a log-LR of zero.

        An interval spanning zero means the evidence does not reliably
        distinguish the propositions in the direction the point estimate
        suggests. Interfaces are required to make this visually explicit; a
        point estimate of ``log10 LR = 2`` with an interval crossing zero must
        not be presented the same way as one that does not.
        """
        return self.lower <= 0.0 <= self.upper

    def contains(self, log_lr: float, /) -> bool:
        return self.lower <= log_lr <= self.upper

    def shifted_by(self, offset: float, /) -> UncertaintyInterval:
        """Translate both bounds, e.g. when applying a prior in log-odds."""
        return UncertaintyInterval(
            lower=self.lower + offset,
            upper=self.upper + offset,
            confidence_level=self.confidence_level,
        )

    def clipped_to(self, lower: float, upper: float) -> UncertaintyInterval:
        """Intersect with hard bounds, e.g. an empirical lower/upper bound (ELUB).

        Used where the interval would otherwise extend past the strongest
        likelihood ratio the calibration data can support.
        """
        if lower > upper:
            raise InvalidEvidenceError(
                "clipping bounds are inverted", lower=lower, upper=upper
            )
        return UncertaintyInterval(
            lower=min(max(self.lower, lower), upper),
            upper=max(min(self.upper, upper), lower),
            confidence_level=self.confidence_level,
        )

    def __str__(self) -> str:  # pragma: no cover - presentation only
        ln10 = math.log(10.0)
        pct = int(round(self.confidence_level * 100))
        return f"log10 LR in [{self.lower / ln10:+.3f}, {self.upper / ln10:+.3f}] ({pct}%)"
