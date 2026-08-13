"""Propagating evidence along paths, and the reasons not to.

A two-hop path is not a link
----------------------------
Given ``A -- B`` at ``LR = 30`` and ``B -- C`` at ``LR = 30``, what is the
evidence that ``A`` and ``C`` share an actor? It is emphatically not ``900``.
Three separate problems compound:

**The propositions are different.** Each edge compares two incidents. The chain
asserts something about a third pair that was never compared, and it does so
through an intermediate whose own attribution is uncertain.

**The edges are not independent.** They frequently rest on the same streams, the
same trained models, and the same channel conditions. If the acoustic model has
a systematic bias for a particular handset, both edges inherit it, and
multiplying them squares the bias rather than accumulating independent evidence.

**Transitivity does not hold for evidence.** ``A`` resembling ``B`` and ``B``
resembling ``C`` does not make ``A`` resemble ``C`` — that is a property of
equivalence relations, and evidential support is not one.

What is offered instead
-----------------------
Three functions with honestly different guarantees, ordered by how much they
assume:

:func:`bound_path_evidence`
    A conservative bound: the weakest link. Assumes nothing beyond the edges
    being what they say. The chain cannot be stronger than its weakest member,
    and this is the only one of the three that is safe to report without
    qualification.

:func:`propagate_analytic`
    Assumes edge errors are jointly Gaussian with a stated correlation, and
    propagates mean and variance. Fast, and only as good as the correlation
    estimate.

:func:`propagate_monte_carlo`
    Resamples each edge from its interval, with correlation induced between
    edges sharing evidence streams, and reports the distribution of the path
    total. The most defensible of the three, and the most expensive.

All three return a distribution or a bound. None returns a bare number, because
a bare number is what invites the reader to treat a chain of inferences as a
measurement.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from viflap.analysis.graph.model import LinkageEdge, LinkageGraph
from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError
from viflap.domain.values import UncertaintyInterval

__all__ = [
    "PathEvidence",
    "bound_path_evidence",
    "find_linkage_paths",
    "propagate_analytic",
    "propagate_monte_carlo",
]

_LN10 = math.log(10.0)


@dataclass(frozen=True, slots=True)
class PathEvidence:
    """What a path through linkage edges supports, and how confidently."""

    path: tuple[tuple[str, str], ...]
    n_hops: int
    point_log_lr: float
    interval: UncertaintyInterval
    method: str
    shared_stream_fraction: float
    """How much the edges along the path rest on the same evidence streams. At
    one, every hop rests on identical streams and the hops are close to
    perfectly correlated — the path is nearly a single piece of evidence
    repeated. Near zero, the hops draw on disjoint evidence and their
    combination is closer to genuine accumulation."""

    naive_product_log_lr: float
    """What simply multiplying the edges would have claimed. Reported so the
    gap is visible."""

    @property
    def inflation_log10(self) -> float:
        """Orders of magnitude that naive multiplication would have added."""
        return (self.naive_product_log_lr - self.point_log_lr) / _LN10

    @property
    def is_reportable(self) -> bool:
        """Whether this path should be shown as evidence at all.

        A path is investigative *guidance* — somewhere to look — rather than
        evidence about the endpoints, and beyond two hops it is not even that.
        The threshold is stated here rather than left to each caller so that it
        cannot quietly differ between the interface and the report.
        """
        return self.n_hops <= 2 and not self.interval.spans_neutral

    def describe(self) -> str:
        route = " -> ".join(identifier for _, identifier in self.path)
        return (
            f"Path {route} spans {self.n_hops} linkage hops. Propagated evidence "
            f"is log10 LR {self.point_log_lr / _LN10:+.2f} "
            f"[{self.interval.lower / _LN10:+.2f}, {self.interval.upper / _LN10:+.2f}]. "
            f"Multiplying the edges directly would have claimed "
            f"{self.inflation_log10:+.2f} orders of magnitude more. The hops share "
            f"{self.shared_stream_fraction:.0%} of their evidence streams. This "
            f"path is a direction for investigation, not evidence about its "
            f"endpoints, which were never compared."
        )


def find_linkage_paths(
    graph: LinkageGraph,
    source: tuple[str, str],
    target: tuple[str, str],
    max_hops: int = 3,
    min_log_lr: float = 0.0,
) -> list[list[LinkageEdge]]:
    """Enumerate simple paths of linkage edges between two nodes.

    Breadth-first with a hop limit. The limit is not a performance guard: paths
    beyond about three hops carry no usable evidence whatever their arithmetic,
    so enumerating them would produce results whose only effect is to be
    misread.
    """
    if max_hops < 1:
        raise InvalidEvidenceError("max_hops must be at least one")

    paths: list[list[LinkageEdge]] = []
    queue: deque[tuple[tuple[str, str], list[LinkageEdge], set[tuple[str, str]]]] = deque(
        [(source, [], {source})]
    )

    while queue:
        current, edges, visited = queue.popleft()
        if len(edges) >= max_hops:
            continue
        for neighbour, edge in graph.linkage_neighbours(current, min_log_lr=min_log_lr):
            if neighbour in visited:
                continue
            extended = [*edges, edge]
            if neighbour == target:
                paths.append(extended)
                continue
            queue.append((neighbour, extended, visited | {neighbour}))
    return paths


def _path_nodes(
    source: tuple[str, str], edges: Sequence[LinkageEdge]
) -> tuple[tuple[str, str], ...]:
    nodes = [source]
    current = source
    for edge in edges:
        current = edge.target if edge.source == current else edge.source
        nodes.append(current)
    return tuple(nodes)


def _shared_stream_fraction(edges: Sequence[LinkageEdge]) -> float:
    """Mean Jaccard overlap of contributing streams across consecutive hops."""
    if len(edges) < 2:
        return 0.0
    overlaps = []
    for first, second in zip(edges, edges[1:], strict=False):
        union = first.contributing_streams | second.contributing_streams
        if not union:
            continue
        overlaps.append(
            len(first.contributing_streams & second.contributing_streams) / len(union)
        )
    return float(np.mean(overlaps)) if overlaps else 0.0


def bound_path_evidence(
    source: tuple[str, str], edges: Sequence[LinkageEdge]
) -> PathEvidence:
    """The conservative bound: a chain is no stronger than its weakest link.

    Assumes only that each edge is what it says. No independence, no
    distributional form, no correlation estimate. Where the other two methods
    require assumptions that may not hold, this one requires none — so it is the
    figure to report when the propagation has to be defended rather than merely
    computed.

    The interval is bounded the same way: by the tightest lower bound and the
    weakest upper bound along the path.
    """
    if not edges:
        raise InsufficientDataError("a path needs at least one edge")

    values = [edge.log_lr.value for edge in edges]
    # The weakest link, by magnitude, with its sign. A chain containing an edge
    # that supports *different* sources cannot support linkage at all.
    weakest_index = int(np.argmin([abs(value) for value in values]))
    point = values[weakest_index]
    if any(value < 0.0 for value in values):
        point = min(values)

    return PathEvidence(
        path=_path_nodes(source, edges),
        n_hops=len(edges),
        point_log_lr=float(point),
        interval=UncertaintyInterval(
            lower=float(min(edge.uncertainty.lower for edge in edges)),
            upper=float(min(edge.uncertainty.upper for edge in edges)),
        ),
        method="weakest-link bound",
        shared_stream_fraction=_shared_stream_fraction(edges),
        naive_product_log_lr=float(sum(values)),
    )


def propagate_analytic(
    source: tuple[str, str],
    edges: Sequence[LinkageEdge],
    correlation: float | None = None,
) -> PathEvidence:
    """Gaussian propagation of mean and variance along the path.

    Variance accumulates as ``sum(var) + 2 * sum_{i<j} rho * sd_i * sd_j``. The
    cross term is the whole point: with ``rho = 0`` this reduces to independent
    accumulation and reports an interval far too narrow.

    ``correlation`` defaults to the fraction of evidence streams the consecutive
    hops share, which is a crude but grounded estimate — hops resting on the
    same streams err together — and is better than assuming zero.

    A **discount** is applied for each hop beyond the first. This is the
    non-transitivity of evidential support made explicit: the chain asserts
    something about a pair that was never compared, and each additional
    intermediate weakens that assertion regardless of the arithmetic. The
    discount is a modelling choice, stated here, and not derived from data the
    project does not yet have.
    """
    if not edges:
        raise InsufficientDataError("a path needs at least one edge")

    shared = _shared_stream_fraction(edges)
    rho = shared if correlation is None else float(np.clip(correlation, -0.95, 0.95))

    means = np.array([edge.log_lr.value for edge in edges])
    # Recover a standard deviation from each interval, treating it as a 95%
    # interval — which is what UncertaintyInterval documents it to be.
    spreads = np.array(
        [
            max((edge.uncertainty.upper - edge.uncertainty.lower) / (2 * 1.96), 1e-9)
            for edge in edges
        ]
    )

    total_mean = float(np.sum(means))
    variance = float(np.sum(spreads**2))
    for i in range(len(spreads)):
        for j in range(i + 1, len(spreads)):
            variance += 2.0 * rho * spreads[i] * spreads[j]
    deviation = math.sqrt(max(variance, 1e-18))

    # Each additional hop halves the evidential contribution of the ones beyond
    # the first, because the proposition the chain addresses is not the
    # proposition any edge was computed for.
    discount = 0.5 ** (len(edges) - 1)
    discounted_mean = total_mean * discount

    return PathEvidence(
        path=_path_nodes(source, edges),
        n_hops=len(edges),
        point_log_lr=discounted_mean,
        interval=UncertaintyInterval(
            lower=discounted_mean - 1.96 * deviation,
            upper=discounted_mean + 1.96 * deviation,
        ),
        method=f"analytic Gaussian (rho={rho:.2f}, hop discount={discount:.2f})",
        shared_stream_fraction=shared,
        naive_product_log_lr=total_mean,
    )


def propagate_monte_carlo(
    source: tuple[str, str],
    edges: Sequence[LinkageEdge],
    n_samples: int = 4000,
    seed: int = 0,
) -> PathEvidence:
    """Resample each edge and report the distribution of the path total.

    Correlation between hops is induced through a shared latent factor whose
    weight is the fraction of evidence streams the hops have in common. Two hops
    resting entirely on the acoustic stream move together; two resting on
    disjoint streams move independently. This is the mechanism by which
    dependence between hops enters the result, and it is the reason this method
    is preferred to the analytic one where the cost is affordable.

    The same per-hop discount as :func:`propagate_analytic` is applied, for the
    same reason.
    """
    if not edges:
        raise InsufficientDataError("a path needs at least one edge")

    rng = np.random.default_rng(seed)
    shared = _shared_stream_fraction(edges)

    means = np.array([edge.log_lr.value for edge in edges])
    spreads = np.array(
        [
            max((edge.uncertainty.upper - edge.uncertainty.lower) / (2 * 1.96), 1e-9)
            for edge in edges
        ]
    )

    common = rng.standard_normal(n_samples)
    totals = np.zeros(n_samples)
    for index in range(len(edges)):
        private = rng.standard_normal(n_samples)
        # sqrt weighting so the total variance of each hop is preserved whatever
        # the split between shared and private components.
        perturbation = (
            math.sqrt(shared) * common + math.sqrt(max(1.0 - shared, 0.0)) * private
        )
        totals += means[index] + spreads[index] * perturbation

    discount = 0.5 ** (len(edges) - 1)
    totals *= discount

    return PathEvidence(
        path=_path_nodes(source, edges),
        n_hops=len(edges),
        point_log_lr=float(np.median(totals)),
        interval=UncertaintyInterval(
            lower=float(np.percentile(totals, 2.5)),
            upper=float(np.percentile(totals, 97.5)),
        ),
        method=f"Monte Carlo ({n_samples:,} samples, shared={shared:.2f}, "
        f"hop discount={discount:.2f})",
        shared_stream_fraction=shared,
        naive_product_log_lr=float(np.sum(means)),
    )
