"""Community detection over probabilistic edges.

Clustering a linkage graph asks: which incidents form an operation? The answer
is investigatively valuable and technically treacherous, because every community
detection algorithm assumes its edges are facts.

The failure is silent. Run Louvain on likelihood-ratio edges and it returns
communities. They look like communities. Nothing in the output distinguishes a
cluster resting on five strong, mutually corroborating linkages from one resting
on three marginal edges that happened to exceed a threshold.

What is done here
-----------------
Detection is run many times over **resampled** edge weights, each draw taking
every edge from its own uncertainty interval. A node's stability is the fraction
of resamplings in which it lands with its most frequent partners. A community
that survives resampling is a finding; one that dissolves was an artefact of the
particular values the evidence happened to take, and the difference is now
visible rather than invisible.

Why the algorithm is implemented here rather than imported
----------------------------------------------------------
Label propagation, written out, is about forty lines. Taking it from a library
would add a dependency and — more to the point — would encourage passing the
graph to that library's *other* algorithms, which have no notion of edge
uncertainty and would be applied to these edges without anything to stop them.
Keeping the one algorithm that is safe here, and not holding a general graph
object at all, is what makes the unsafe operations unavailable rather than
merely inadvisable.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from viflap.analysis.graph.model import LinkageGraph
from viflap.domain.errors import InsufficientDataError

__all__ = ["Community", "CommunityReport", "detect_communities"]

_LN10 = math.log(10.0)


@dataclass(frozen=True, slots=True)
class Community:
    """A cluster of incidents, with how reliably it holds together."""

    members: tuple[str, ...]
    stability: float
    """Mean fraction of resamplings in which these members clustered together.
    One means the community appears in every resampling of the evidence; a value
    near ``1/k`` for ``k`` communities means membership is essentially arbitrary
    and the cluster should not be reported."""

    mean_internal_log_lr: float
    min_internal_log_lr: float
    """The weakest linkage holding the community together. A cluster whose
    minimum internal edge is near zero is joined by a hair, and the mean
    conceals it."""

    n_internal_edges: int

    @property
    def is_robust(self) -> bool:
        return self.stability >= 0.8 and self.min_internal_log_lr > 0.0

    def describe(self) -> str:
        return (
            f"{len(self.members)} incidents, appearing together in "
            f"{self.stability:.0%} of resamplings. Internal linkages span "
            f"log10 LR {self.min_internal_log_lr / _LN10:+.2f} to "
            f"{self.mean_internal_log_lr / _LN10:+.2f} (mean). "
            + (
                "This grouping survives resampling of the evidence."
                if self.is_robust
                else "This grouping does not survive resampling and should be "
                "treated as a suggestion rather than a finding."
            )
        )


@dataclass(frozen=True, slots=True)
class CommunityReport:
    """The outcome of a stability-assessed community detection."""

    communities: tuple[Community, ...]
    node_stability: Mapping[str, float]
    n_resamples: int
    threshold_log_lr: float

    @property
    def robust_communities(self) -> tuple[Community, ...]:
        return tuple(c for c in self.communities if c.is_robust)

    def describe(self) -> str:
        robust = len(self.robust_communities)
        return (
            f"{len(self.communities)} candidate groupings found over "
            f"{self.n_resamples:,} resamplings of the edge evidence, of which "
            f"{robust} survive resampling. Edges below log10 LR "
            f"{self.threshold_log_lr / _LN10:+.2f} were not followed. Groupings "
            f"are investigative hypotheses about which incidents form one "
            f"operation; they are not assertions about who conducted them."
        )


def detect_communities(
    graph: LinkageGraph,
    threshold_log_lr: float = math.log(10.0),
    n_resamples: int = 200,
    seed: int = 0,
    max_iterations: int = 50,
) -> CommunityReport:
    """Detect communities and assess their stability under edge resampling.

    Parameters
    ----------
    threshold_log_lr:
        Edges weaker than this are not followed. Defaults to ``LR = 10``, which
        is deliberately not zero: an edge at ``LR = 1.5`` is barely evidence,
        and following it connects unrelated components into one large cluster
        that then appears to be an organisation.
    n_resamples:
        Resamplings of the edge weights. Each draws every edge from its own
        interval, so a cluster held together by a marginal edge dissolves in the
        resamplings where that edge lands below the threshold.
    """
    edges = list(graph.linkages)
    if not edges:
        raise InsufficientDataError("the graph contains no linkage edges")

    rng = np.random.default_rng(seed)
    assignments: dict[str, list[int]] = defaultdict(list)
    co_membership: Counter[tuple[str, str]] = Counter()
    node_keys = sorted({node.identifier for node in graph.nodes})

    for _ in range(n_resamples):
        sampled: list[tuple[str, str, float]] = []
        for edge in edges:
            spread = max(
                (edge.uncertainty.upper - edge.uncertainty.lower) / (2 * 1.96), 1e-9
            )
            drawn = float(rng.normal(edge.log_lr.value, spread))
            if drawn >= threshold_log_lr:
                sampled.append((edge.source[1], edge.target[1], drawn))

        labels = _label_propagation(node_keys, sampled, rng, max_iterations)
        for node, label in labels.items():
            assignments[node].append(label)

        by_label: dict[int, list[str]] = defaultdict(list)
        for node, label in labels.items():
            by_label[label].append(node)
        for members in by_label.values():
            if len(members) < 2:
                continue
            for i, first in enumerate(sorted(members)):
                for second in sorted(members)[i + 1 :]:
                    co_membership[(first, second)] += 1

    # The reported partition uses the *unperturbed* weights; the resamplings
    # supply stability, not the answer. Reporting the modal resampled partition
    # instead would report a clustering of a fabricated dataset.
    final_edges = [
        (edge.source[1], edge.target[1], edge.log_lr.value)
        for edge in edges
        if edge.log_lr.value >= threshold_log_lr
    ]
    final_labels = _label_propagation(node_keys, final_edges, rng, max_iterations)

    node_stability = {
        node: _stability_of(node, final_labels, co_membership, n_resamples)
        for node in node_keys
    }

    grouped: dict[int, list[str]] = defaultdict(list)
    for node, label in final_labels.items():
        grouped[label].append(node)

    communities: list[Community] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        members_sorted = tuple(sorted(members))
        internal = [
            edge.log_lr.value
            for edge in edges
            if edge.source[1] in members_sorted and edge.target[1] in members_sorted
        ]
        if not internal:
            continue
        communities.append(
            Community(
                members=members_sorted,
                stability=float(np.mean([node_stability[node] for node in members_sorted])),
                mean_internal_log_lr=float(np.mean(internal)),
                min_internal_log_lr=float(np.min(internal)),
                n_internal_edges=len(internal),
            )
        )

    communities.sort(key=lambda c: (-c.stability, -len(c.members)))
    return CommunityReport(
        communities=tuple(communities),
        node_stability=node_stability,
        n_resamples=n_resamples,
        threshold_log_lr=threshold_log_lr,
    )


def _stability_of(
    node: str,
    final_labels: Mapping[str, int],
    co_membership: Counter[tuple[str, str]],
    n_resamples: int,
) -> float:
    """How often this node's final companions were also its companions."""
    companions = [
        other
        for other, label in final_labels.items()
        if other != node and label == final_labels[node]
    ]
    if not companions:
        return 1.0
    rates = [
        co_membership.get(tuple(sorted((node, other))), 0) / max(n_resamples, 1)
        for other in companions
    ]
    return float(np.mean(rates))


def _label_propagation(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    rng: np.random.Generator,
    max_iterations: int,
) -> dict[str, int]:
    """Weighted asynchronous label propagation.

    Each node adopts the label carried by the greatest total edge weight among
    its neighbours, iterating until stable. Weighted, so a node joined by one
    strong linkage and three weak ones follows the strong one.

    Node order is shuffled each iteration. Label propagation is order-dependent,
    and a fixed order produces a partition that is an artefact of the
    identifiers' alphabetical order — which is exactly the kind of stability the
    resampling above would fail to detect, because it is stable for the wrong
    reason.
    """
    labels = {node: index for index, node in enumerate(nodes)}
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for source, target, weight in edges:
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight))

    order = list(nodes)
    for _ in range(max_iterations):
        rng.shuffle(order)
        changed = False
        for node in order:
            neighbours = adjacency.get(node)
            if not neighbours:
                continue
            weights: dict[int, float] = defaultdict(float)
            for other, weight in neighbours:
                weights[labels[other]] += weight
            best = max(weights.items(), key=lambda item: (item[1], -item[0]))[0]
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break
    return labels
