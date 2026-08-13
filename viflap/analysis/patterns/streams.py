"""Temporal, transactional and device evidence streams.

Each stream reduces an incident to a set of observations and compares two
incidents with the conjugate models in
:mod:`viflap.analysis.patterns.conjugate`. Because all three use the same
machinery, their outputs are on the same footing — genuine likelihood ratios
under stated models — and the dependence *between* them is left to the fusion
layer rather than being obscured by each having its own scoring scheme.

Circular time is treated as circular
------------------------------------
Hour of day is not a number on a line. The difference between 23:00 and 01:00 is
two hours, not twenty-two, and any statistic that treats the hour as a real
number gets this wrong for exactly the operators who work at night — which, for
telephony fraud targeting people at home, is a substantial share of them. Hour of
day is therefore binned and the background smoothed with a circular (von Mises)
kernel, so that adjacent hours are treated as adjacent and 23 is adjacent to 0.

Identifier sharing is weighted by rarity
----------------------------------------
Two incidents cashing out through the same agent is evidence in proportion to
how unusual that agent is. The Dirichlet-multinomial handles this without a
tuning parameter: the concentration parameters are proportional to background
frequency, so a rare agent contributes far more than a busy one. A set-overlap
index — the obvious alternative — gives both the same value and discards the
distinction permanently.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.patterns.conjugate import (
    BackgroundPopulation,
    DirichletMultinomialComparator,
    NormalInverseGammaComparator,
)
from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError
from viflap.domain.evidence import EvidenceStream

__all__ = [
    "CallRecord",
    "DeviceComparator",
    "DeviceObservation",
    "TemporalComparator",
    "TemporalProfile",
    "Transaction",
    "TransactionalComparator",
    "TransactionalProfile",
    "circular_background",
]

HOURS = tuple(str(hour) for hour in range(24))
WEEKDAYS = tuple(str(day) for day in range(7))


def circular_background(
    counts: Mapping[str, int],
    n_bins: int,
    concentration: float = 4.0,
    description: str = "",
) -> BackgroundPopulation:
    """Background over circular bins, smoothed with a von Mises kernel.

    Smoothing matters for two reasons. It stops a bin that happened to be empty
    in the background sample from producing an enormous likelihood ratio on the
    strength of a sampling gap. And it encodes the fact that adjacent hours are
    substantively similar, so an operator active at 22:00 and one active at
    23:00 are not treated as having nothing in common.

    ``concentration`` is the von Mises concentration: higher is less smoothing.
    Four corresponds to a kernel with most of its mass within about two hours.
    """
    if n_bins < 2:
        raise InvalidEvidenceError("a circular background needs at least two bins")

    observed = np.array([float(counts.get(str(index), 0)) for index in range(n_bins)])
    if observed.sum() <= 0.0:
        raise InsufficientDataError("no observations to build a background from")

    angles = 2.0 * np.pi * np.arange(n_bins) / n_bins
    smoothed = np.zeros(n_bins)
    for index in range(n_bins):
        # Kernel weight depends only on angular separation, so bin 23 is as
        # close to bin 0 as bin 1 is.
        weights = np.exp(concentration * np.cos(angles - angles[index]))
        weights /= weights.sum()
        smoothed += observed[index] * weights

    smoothed /= smoothed.sum()
    return BackgroundPopulation(
        frequencies={str(index): float(smoothed[index]) for index in range(n_bins)},
        total_observations=int(observed.sum()),
        description=description or f"Circular background over {n_bins} bins",
    )


def _as_datetime(value: datetime | str | int | float) -> datetime:
    """Coerce a timestamp, requiring timezone awareness.

    A naive timestamp cannot be placed on a clock, and hour-of-day analysis on
    timestamps of unknown zone measures the record-keeping of the operator that
    supplied them rather than the behaviour of the person.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    else:
        raise InvalidEvidenceError("unsupported timestamp type", type=type(value).__name__)

    if parsed.tzinfo is None:
        raise InvalidEvidenceError(
            "timestamps must carry a timezone; hour-of-day evidence computed "
            "from timestamps of unknown zone measures the record-keeping rather "
            "than the behaviour",
            value=str(value),
        )
    return parsed


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One call in an incident's call detail records."""

    timestamp: datetime
    duration_seconds: float
    direction: str = "outbound"


@dataclass(frozen=True, slots=True)
class TemporalProfile:
    """An incident reduced to its temporal observations."""

    hour_counts: Mapping[str, int]
    weekday_counts: Mapping[str, int]
    log_durations: NDArray[np.float64]
    log_inter_call_seconds: NDArray[np.float64]
    n_calls: int

    @classmethod
    def from_records(cls, records: Sequence[CallRecord]) -> TemporalProfile:
        if not records:
            raise InsufficientDataError("no call records supplied")

        timestamps = sorted(_as_datetime(record.timestamp) for record in records)
        hours = Counter(str(stamp.hour) for stamp in timestamps)
        weekdays = Counter(str(stamp.weekday()) for stamp in timestamps)

        durations = np.array(
            [max(record.duration_seconds, 1.0) for record in records], dtype=np.float64
        )
        intervals = np.array(
            [
                max((later - earlier).total_seconds(), 1.0)
                for earlier, later in zip(timestamps, timestamps[1:], strict=False)
            ],
            dtype=np.float64,
        )

        return cls(
            hour_counts=dict(hours),
            weekday_counts=dict(weekdays),
            # Log scale: durations and intervals span several orders of
            # magnitude and are close to log-normal, so the normal model is
            # applied where it is approximately true rather than where it is
            # convenient.
            log_durations=np.log(durations),
            log_inter_call_seconds=np.log(intervals) if intervals.size else np.zeros(0),
            n_calls=len(records),
        )


class TemporalComparator:
    """Likelihood ratio from an incident pair's calling rhythm.

    Combines hour-of-day, day-of-week and call-duration evidence. The
    components are summed, which assumes them conditionally independent *within*
    this stream — an assumption that is imperfect and is stated rather than
    hidden. It is tolerable here because the fusion layer models dependence
    across streams and this residual within-stream dependence is small relative
    to it; the alternative, a joint model over all three, needs more data than a
    single stream's development set provides.
    """

    def __init__(
        self,
        hour_background: BackgroundPopulation,
        weekday_background: BackgroundPopulation,
        duration_model: NormalInverseGammaComparator,
        concentration: float = 30.0,
        min_calls: int = 3,
    ) -> None:
        self._hour = DirichletMultinomialComparator(hour_background, concentration)
        self._weekday = DirichletMultinomialComparator(weekday_background, concentration)
        self._duration = duration_model
        self._min_calls = min_calls

    @property
    def stream(self) -> EvidenceStream:
        return EvidenceStream.TEMPORAL

    def score(self, first: TemporalProfile, second: TemporalProfile) -> float:
        """Uncalibrated log-likelihood ratio, to be calibrated before reporting."""
        if first.n_calls < self._min_calls or second.n_calls < self._min_calls:
            raise InsufficientDataError(
                "too few calls for a temporal comparison; a rhythm cannot be "
                "estimated from a handful of records",
                n_first=first.n_calls,
                n_second=second.n_calls,
                required=self._min_calls,
            )

        total = self._hour.log_likelihood_ratio(first.hour_counts, second.hour_counts)
        total += self._weekday.log_likelihood_ratio(
            first.weekday_counts, second.weekday_counts
        )
        total += self._duration.log_likelihood_ratio(
            first.log_durations, second.log_durations
        )
        return float(total)

    def diagnostics(
        self, first: TemporalProfile, second: TemporalProfile
    ) -> dict[str, float]:
        return {
            "n_calls_first": float(first.n_calls),
            "n_calls_second": float(second.n_calls),
            "hour_component": self._hour.log_likelihood_ratio(
                first.hour_counts, second.hour_counts
            ),
            "weekday_component": self._weekday.log_likelihood_ratio(
                first.weekday_counts, second.weekday_counts
            ),
            "duration_component": self._duration.log_likelihood_ratio(
                first.log_durations, second.log_durations
            ),
        }


# ---------------------------------------------------------------------------
# Transactional
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transaction:
    """One mobile-money transaction associated with an incident."""

    timestamp: datetime
    amount: float
    transaction_type: str
    counterparty_wallet: str = ""
    agent_id: str = ""


@dataclass(frozen=True, slots=True)
class TransactionalProfile:
    """An incident reduced to its transactional observations."""

    agent_counts: Mapping[str, int]
    counterparty_counts: Mapping[str, int]
    type_counts: Mapping[str, int]
    log_amounts: NDArray[np.float64]
    n_transactions: int

    @classmethod
    def from_transactions(cls, transactions: Sequence[Transaction]) -> TransactionalProfile:
        if not transactions:
            raise InsufficientDataError("no transactions supplied")

        agents = Counter(
            transaction.agent_id for transaction in transactions if transaction.agent_id
        )
        counterparties = Counter(
            transaction.counterparty_wallet
            for transaction in transactions
            if transaction.counterparty_wallet
        )
        types = Counter(transaction.transaction_type for transaction in transactions)
        amounts = np.array(
            [max(transaction.amount, 0.01) for transaction in transactions],
            dtype=np.float64,
        )

        return cls(
            agent_counts=dict(agents),
            counterparty_counts=dict(counterparties),
            type_counts=dict(types),
            log_amounts=np.log(amounts),
            n_transactions=len(transactions),
        )


class TransactionalComparator:
    """Likelihood ratio from mobile-money behaviour.

    The agent component is the strongest and is the clearest illustration of why
    rarity must be modelled. Cashing out through the same agent is powerful
    evidence when the agent is small and nearly none when the agent handles a
    large share of the corridor's volume. The Dirichlet-multinomial derives that
    difference from the background frequencies; a set-overlap index cannot
    express it at all.

    Amounts are compared on the log scale, where their distribution is
    approximately normal. A preference for round numbers — 500, 1000, 2000 — is
    a real and quite individuating habit, and it survives the log transform as a
    concentration of mass at particular values.
    """

    def __init__(
        self,
        agent_background: BackgroundPopulation,
        counterparty_background: BackgroundPopulation,
        type_background: BackgroundPopulation,
        amount_model: NormalInverseGammaComparator,
        concentration: float = 20.0,
        min_transactions: int = 2,
    ) -> None:
        self._agents = DirichletMultinomialComparator(agent_background, concentration)
        self._counterparties = DirichletMultinomialComparator(
            counterparty_background, concentration
        )
        self._types = DirichletMultinomialComparator(type_background, concentration * 5)
        self._amounts = amount_model
        self._min_transactions = min_transactions

    @property
    def stream(self) -> EvidenceStream:
        return EvidenceStream.TRANSACTIONAL

    def score(self, first: TransactionalProfile, second: TransactionalProfile) -> float:
        if (
            first.n_transactions < self._min_transactions
            or second.n_transactions < self._min_transactions
        ):
            raise InsufficientDataError(
                "too few transactions for a transactional comparison",
                n_first=first.n_transactions,
                n_second=second.n_transactions,
                required=self._min_transactions,
            )

        total = self._amounts.log_likelihood_ratio(first.log_amounts, second.log_amounts)
        total += self._types.log_likelihood_ratio(first.type_counts, second.type_counts)

        # Agent and counterparty evidence is only available when both incidents
        # recorded it. Absent identifiers contribute nothing rather than being
        # treated as an empty set, which would read as positive evidence of
        # *dis*-similarity.
        if first.agent_counts and second.agent_counts:
            total += self._agents.log_likelihood_ratio(
                first.agent_counts, second.agent_counts
            )
        if first.counterparty_counts and second.counterparty_counts:
            total += self._counterparties.log_likelihood_ratio(
                first.counterparty_counts, second.counterparty_counts
            )
        return float(total)

    def diagnostics(
        self, first: TransactionalProfile, second: TransactionalProfile
    ) -> dict[str, float]:
        shared_agents = set(first.agent_counts) & set(second.agent_counts)
        rarest = min(
            (self._agents.background.frequency_of(agent) for agent in shared_agents),
            default=float("nan"),
        )
        return {
            "n_transactions_first": float(first.n_transactions),
            "n_transactions_second": float(second.n_transactions),
            "n_shared_agents": float(len(shared_agents)),
            "rarest_shared_agent_frequency": float(rarest),
            "n_shared_counterparties": float(
                len(set(first.counterparty_counts) & set(second.counterparty_counts))
            ),
        }


# ---------------------------------------------------------------------------
# Device and network
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeviceObservation:
    """Device and network attributes recorded for an incident."""

    imei_counts: Mapping[str, int] = field(default_factory=dict)
    handset_model_counts: Mapping[str, int] = field(default_factory=dict)
    cell_site_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.imei_counts or self.handset_model_counts or self.cell_site_counts)


class DeviceComparator:
    """Likelihood ratio from device and network identifiers.

    A shared IMEI is the strongest single piece of evidence this system handles,
    and it is worth being explicit about why it still yields a likelihood ratio
    rather than a conclusion. An IMEI identifies a handset, not a person.
    Handsets are sold, lent, shared within a household and stolen. The
    likelihood ratio for a shared IMEI is large — the background frequency of any
    particular one is minute — but it bears on the proposition "the same handset
    was used", from which "the same person spoke" follows only with further
    argument that belongs to the investigator.

    The system reports the first. It has no vocabulary for the second.
    """

    def __init__(
        self,
        imei_background: BackgroundPopulation,
        handset_background: BackgroundPopulation,
        cell_site_background: BackgroundPopulation,
        concentration: float = 5.0,
    ) -> None:
        # Low concentration for IMEI: handsets are strongly individuating, so an
        # actor's distribution over them is very far from the population's.
        self._imei = DirichletMultinomialComparator(imei_background, concentration)
        # High concentration for handset model: most people own a common model,
        # so agreement is weak evidence and the prior should say so.
        self._handset = DirichletMultinomialComparator(
            handset_background, concentration * 40
        )
        self._cell_sites = DirichletMultinomialComparator(
            cell_site_background, concentration * 6
        )

    @property
    def stream(self) -> EvidenceStream:
        return EvidenceStream.DEVICE

    def score(self, first: DeviceObservation, second: DeviceObservation) -> float:
        if first.is_empty or second.is_empty:
            raise InsufficientDataError(
                "device evidence requires attributes recorded for both incidents"
            )

        total = 0.0
        contributions = 0
        for comparator, left, right in (
            (self._imei, first.imei_counts, second.imei_counts),
            (self._handset, first.handset_model_counts, second.handset_model_counts),
            (self._cell_sites, first.cell_site_counts, second.cell_site_counts),
        ):
            if left and right:
                total += comparator.log_likelihood_ratio(left, right)
                contributions += 1

        if contributions == 0:
            raise InsufficientDataError(
                "the two incidents record no attribute in common; there is "
                "nothing to compare, which is not the same as evidence of "
                "difference"
            )
        return float(total)

    def diagnostics(
        self, first: DeviceObservation, second: DeviceObservation
    ) -> dict[str, float]:
        shared_imei = set(first.imei_counts) & set(second.imei_counts)
        return {
            "n_shared_imei": float(len(shared_imei)),
            "n_shared_cell_sites": float(
                len(set(first.cell_site_counts) & set(second.cell_site_counts))
            ),
            "shares_handset_model": float(
                bool(set(first.handset_model_counts) & set(second.handset_model_counts))
            ),
            "rarest_shared_imei_frequency": float(
                min(
                    (self._imei.background.frequency_of(imei) for imei in shared_imei),
                    default=float("nan"),
                )
            ),
        }
