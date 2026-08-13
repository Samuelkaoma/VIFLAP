"""The linkage graph: probabilistic edges, and what that costs.

The graph is a *consequence* of fusion, not a parallel system. Its nodes are
incidents, wallets, devices and subscriber records; its edges are of two kinds,
and conflating them is the error this module is arranged to prevent.

**Deterministic edges** are facts of record. This incident involved that wallet.
This SIM was registered to that subscriber. They are true or absent.

**Linkage edges** are likelihood ratios. They assert that two incidents were
probably conducted by the same actor, with a stated strength and an uncertainty
interval.

Why treating them alike is wrong
--------------------------------
Every standard graph algorithm — shortest path, connected components, community
detection, centrality — assumes edges are facts. Feed it probabilistic edges and
it will produce clusters, and those clusters will look exactly like the ones it
produces from real edges. There is no error, no warning, and no way to tell from
the output that the input was uncertain.

The compounding is severe and it is easy to underestimate. Two incidents joined
through an intermediate by two edges of ``LR = 30`` each do **not** amount to a
direct link of ``LR = 900``. The intermediate inference is itself uncertain, the
two hops are not independent — they frequently rest on the same stream, the same
model, and the same channel conditions — and the errors accumulate multiplicatively.
A four-hop path through moderate edges can be reported as overwhelming evidence
by an algorithm doing nothing wrong.

What this module does about it
------------------------------
Paths are traversed only through :mod:`viflap.analysis.graph.propagation`, which
returns a distribution rather than a value. Community detection runs many times
over resampled edge weights and reports how often each node lands in the same
cluster, so a cluster that exists only at one resampling of the evidence is
visible as such. And a path's likelihood ratio is bounded rather than multiplied,
because the honest statement about a chain of uncertain inferences is a bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from viflap.domain.errors import InvalidEvidenceError
from viflap.domain.linkage import ComparisonResult
from viflap.domain.values import LogLikelihoodRatio, UncertaintyInterval

__all__ = [
    "EdgeKind",
    "LinkageEdge",
    "LinkageGraph",
    "Node",
    "NodeKind",
]


class NodeKind(Enum):
    """What a node represents."""

    INCIDENT = "incident"
    WALLET = "wallet"
    DEVICE = "device"
    SUBSCRIBER = "subscriber"
    AGENT = "agent"
    CELL_SITE = "cell_site"


class EdgeKind(Enum):
    """What an edge asserts, and with what certainty."""

    LINKAGE = "linkage"
    """A probabilistic assertion that two incidents share an actor, carrying a
    likelihood ratio and an interval. Never traversable as a fact."""

    TRANSACTION = "transaction"
    """A record that an incident involved a wallet. Deterministic."""

    DEVICE_USE = "device_use"
    REGISTRATION = "registration"
    LOCATION = "location"

    @property
    def is_probabilistic(self) -> bool:
        return self is EdgeKind.LINKAGE


@dataclass(frozen=True, slots=True)
class Node:
    """A vertex."""

    identifier: str
    kind: NodeKind
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise InvalidEvidenceError("a node requires a non-empty identifier")

    @property
    def key(self) -> tuple[str, str]:
        """Identity, namespaced by kind.

        A wallet and an incident may legitimately share a string identifier in
        source records; without the namespace they would silently become one
        node, and every path through that node would be an artefact.
        """
        return (self.kind.value, self.identifier)


@dataclass(frozen=True, slots=True)
class LinkageEdge:
    """A probabilistic edge derived from a comparison result."""

    source: tuple[str, str]
    target: tuple[str, str]
    log_lr: LogLikelihoodRatio
    uncertainty: UncertaintyInterval
    result_id: str
    contributing_streams: frozenset[str]
    """Which streams produced this edge. Used when propagating along a path: two
    consecutive edges resting on the same single stream are far more correlated
    than two resting on disjoint streams, and the propagation accounts for it."""

    @property
    def supports_linkage(self) -> bool:
        return self.log_lr.supports_same_source

    @property
    def is_weakly_determined(self) -> bool:
        return self.uncertainty.spans_neutral

    @classmethod
    def from_result(cls, result: ComparisonResult, result_id: str) -> LinkageEdge:
        return cls(
            source=(NodeKind.INCIDENT.value, result.pair.first.value),
            target=(NodeKind.INCIDENT.value, result.pair.second.value),
            log_lr=result.fused_log_lr,
            uncertainty=result.uncertainty,
            result_id=result_id,
            contributing_streams=frozenset(stream.value for stream in result.contributing),
        )


@dataclass(frozen=True, slots=True)
class FactualEdge:
    """A deterministic edge: a fact of record."""

    source: tuple[str, str]
    target: tuple[str, str]
    kind: EdgeKind
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind.is_probabilistic:
            raise InvalidEvidenceError(
                "a linkage is not a fact of record and must be represented as a "
                "LinkageEdge carrying its likelihood ratio",
                kind=self.kind.value,
            )


class LinkageGraph:
    """A graph holding both kinds of edge, and keeping them apart.

    Deliberately not a thin wrapper over a general graph library. The point of
    the type is that it has *no* method returning a path or a component computed
    over mixed edges: the operations that would silently treat a likelihood ratio
    as a fact are not merely discouraged, they are absent. Traversal goes through
    :mod:`viflap.analysis.graph.propagation`, which returns distributions.
    """

    def __init__(self) -> None:
        self._nodes: dict[tuple[str, str], Node] = {}
        self._linkages: list[LinkageEdge] = []
        self._facts: list[FactualEdge] = []
        self._linkage_index: dict[tuple[str, str], list[int]] = {}
        self._fact_index: dict[tuple[str, str], list[int]] = {}

    # -- Construction -----------------------------------------------------

    def add_node(self, node: Node) -> Node:
        self._nodes.setdefault(node.key, node)
        return self._nodes[node.key]

    def add_linkage(self, edge: LinkageEdge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise InvalidEvidenceError(
                "both endpoints must be added before the edge joining them",
                source=edge.source,
                target=edge.target,
            )
        index = len(self._linkages)
        self._linkages.append(edge)
        self._linkage_index.setdefault(edge.source, []).append(index)
        self._linkage_index.setdefault(edge.target, []).append(index)

    def add_fact(self, edge: FactualEdge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise InvalidEvidenceError(
                "both endpoints must be added before the edge joining them",
                source=edge.source,
                target=edge.target,
            )
        index = len(self._facts)
        self._facts.append(edge)
        self._fact_index.setdefault(edge.source, []).append(index)
        self._fact_index.setdefault(edge.target, []).append(index)

    @classmethod
    def from_results(cls, results: Mapping[str, ComparisonResult]) -> LinkageGraph:
        """Build a graph from stored comparison results."""
        graph = cls()
        for result_id, result in results.items():
            for incident in (result.pair.first, result.pair.second):
                graph.add_node(Node(incident.value, NodeKind.INCIDENT))
            graph.add_linkage(LinkageEdge.from_result(result, result_id))
        return graph

    # -- Inspection -------------------------------------------------------

    @property
    def nodes(self) -> Sequence[Node]:
        return [self._nodes[key] for key in sorted(self._nodes)]

    @property
    def linkages(self) -> Sequence[LinkageEdge]:
        return list(self._linkages)

    @property
    def facts(self) -> Sequence[FactualEdge]:
        return list(self._facts)

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    def linkage_neighbours(
        self, key: tuple[str, str], min_log_lr: float | None = None
    ) -> list[tuple[tuple[str, str], LinkageEdge]]:
        """Nodes joined to ``key`` by a linkage edge.

        ``min_log_lr`` filters by strength. It has no default: the caller must
        decide what strength of evidence is worth following, because that is a
        judgement about the investigation rather than a property of the graph.
        """
        results: list[tuple[tuple[str, str], LinkageEdge]] = []
        for index in self._linkage_index.get(key, ()):
            edge = self._linkages[index]
            if min_log_lr is not None and edge.log_lr.value < min_log_lr:
                continue
            other = edge.target if edge.source == key else edge.source
            results.append((other, edge))
        return results

    def fact_neighbours(
        self, key: tuple[str, str], kind: EdgeKind | None = None
    ) -> list[tuple[tuple[str, str], FactualEdge]]:
        """Nodes joined to ``key`` by a deterministic edge."""
        results: list[tuple[tuple[str, str], FactualEdge]] = []
        for index in self._fact_index.get(key, ()):
            edge = self._facts[index]
            if kind is not None and edge.kind is not kind:
                continue
            other = edge.target if edge.source == key else edge.source
            results.append((other, edge))
        return results

    def summary(self) -> dict[str, float]:
        """Counts an analyst needs before reading anything else off the graph."""
        weak = sum(1 for edge in self._linkages if edge.is_weakly_determined)
        supporting = sum(1 for edge in self._linkages if edge.supports_linkage)
        return {
            "n_nodes": float(len(self._nodes)),
            "n_linkage_edges": float(len(self._linkages)),
            "n_factual_edges": float(len(self._facts)),
            "n_linkages_supporting": float(supporting),
            "n_linkages_weakly_determined": float(weak),
            "fraction_weakly_determined": (
                weak / len(self._linkages) if self._linkages else 0.0
            ),
        }
