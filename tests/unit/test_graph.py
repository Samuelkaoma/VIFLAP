"""The linkage graph, and the arithmetic it refuses to do.

The graph holds two kinds of edge and the distinction is the whole design.
**Factual** edges are records: this incident involved that wallet, this SIM was
registered to that subscriber. **Linkage** edges are likelihood ratios: two
incidents *probably* share an actor, with a stated strength and an interval.

Conflating them is how a graph turns a probabilistic claim into a fact by
drawing it as a line, so the type system refuses the conflation and these tests
pin the refusal.

The second half is propagation. Given ``A--B`` at 30 and ``B--C`` at 30, the
evidence linking ``A`` to ``C`` is emphatically not 900: the propositions are
different, the intermediate's own attribution is uncertain, and the edges are
not independent — they frequently rest on the same models and the same channel,
so multiplying them squares a shared bias rather than accumulating independent
evidence. The tests below assert that the code declines to multiply, and that it
records what the naive answer would have been so the gap is visible rather than
merely avoided.
"""

from __future__ import annotations

import math

import pytest

from viflap.analysis.graph.model import (
    EdgeKind,
    FactualEdge,
    LinkageEdge,
    Node,
    NodeKind,
)
from viflap.analysis.graph.propagation import (
    _shared_stream_fraction,
    bound_path_evidence,
    propagate_analytic,
)
from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError
from viflap.domain.values import LogLikelihoodRatio, UncertaintyInterval

ACOUSTIC = "acoustic"
BEHAVIOURAL = "behavioural"
TEMPORAL = "temporal"


def _edge(
    source: str,
    target: str,
    log_lr: float,
    streams: frozenset[str] = frozenset({ACOUSTIC}),
    half_width: float = 1.0,
) -> LinkageEdge:
    return LinkageEdge(
        source=(NodeKind.INCIDENT.value, source),
        target=(NodeKind.INCIDENT.value, target),
        log_lr=LogLikelihoodRatio(log_lr),
        uncertainty=UncertaintyInterval(
            lower=log_lr - half_width, upper=log_lr + half_width
        ),
        result_id=f"res-{source}-{target}",
        contributing_streams=streams,
    )


class TestNodeIdentity:
    def test_a_node_requires_an_identifier(self) -> None:
        with pytest.raises(InvalidEvidenceError):
            Node(identifier="   ", kind=NodeKind.INCIDENT)

    def test_identity_is_namespaced_by_kind(self) -> None:
        """A wallet and an incident may share a string identifier in source
        records. Without the namespace they silently become one node, and every
        path drawn through that node is an artefact of the collision."""
        wallet = Node(identifier="1047", kind=NodeKind.WALLET)
        incident = Node(identifier="1047", kind=NodeKind.INCIDENT)

        assert wallet.key != incident.key

    def test_the_same_kind_and_identifier_is_the_same_node(self) -> None:
        assert (
            Node(identifier="1047", kind=NodeKind.WALLET).key
            == Node(identifier="1047", kind=NodeKind.WALLET).key
        )


class TestEdgeKindSeparation:
    def test_a_linkage_cannot_be_recorded_as_a_fact(self) -> None:
        """The error this module is arranged to prevent: an assertion carrying a
        likelihood ratio, drawn as though it were a record."""
        with pytest.raises(InvalidEvidenceError):
            FactualEdge(
                source=(NodeKind.INCIDENT.value, "i1"),
                target=(NodeKind.INCIDENT.value, "i2"),
                kind=EdgeKind.LINKAGE,
            )

    @pytest.mark.parametrize(
        "kind",
        [
            EdgeKind.TRANSACTION,
            EdgeKind.DEVICE_USE,
            EdgeKind.REGISTRATION,
            EdgeKind.LOCATION,
        ],
    )
    def test_records_of_fact_are_admitted(self, kind: EdgeKind) -> None:
        edge = FactualEdge(
            source=(NodeKind.INCIDENT.value, "i1"),
            target=(NodeKind.WALLET.value, "w1"),
            kind=kind,
        )
        assert edge.kind is kind

    def test_only_linkage_is_probabilistic(self) -> None:
        assert EdgeKind.LINKAGE.is_probabilistic
        for kind in (EdgeKind.TRANSACTION, EdgeKind.DEVICE_USE, EdgeKind.REGISTRATION):
            assert not kind.is_probabilistic


class TestLinkageEdge:
    def test_an_edge_reports_whether_it_supports_linkage(self) -> None:
        assert _edge("i1", "i2", 3.0).supports_linkage
        assert not _edge("i1", "i2", -3.0).supports_linkage

    def test_an_interval_spanning_neutral_is_weakly_determined(self) -> None:
        """An edge whose interval covers zero has not established its direction."""
        assert _edge("i1", "i2", 0.5, half_width=2.0).is_weakly_determined
        assert not _edge("i1", "i2", 5.0, half_width=1.0).is_weakly_determined


class TestWeakestLinkBound:
    """The figure to report when the propagation has to be defended.

    It assumes only that each edge is what it says — no independence, no
    distributional form, no correlation estimate.
    """

    def test_a_chain_is_no_stronger_than_its_weakest_link(self) -> None:
        edges = [
            _edge("A", "B", math.log(30.0)),
            _edge("B", "C", math.log(30.0)),
        ]
        evidence = bound_path_evidence((NodeKind.INCIDENT.value, "A"), edges)

        assert evidence.point_log_lr == pytest.approx(math.log(30.0))

    def test_it_does_not_multiply(self) -> None:
        """900 is the answer this module exists to not give."""
        edges = [
            _edge("A", "B", math.log(30.0)),
            _edge("B", "C", math.log(30.0)),
        ]
        evidence = bound_path_evidence((NodeKind.INCIDENT.value, "A"), edges)

        assert math.exp(evidence.point_log_lr) < 100.0
        assert math.exp(evidence.naive_product_log_lr) == pytest.approx(900.0, rel=1e-6)

    def test_the_naive_answer_is_recorded_rather_than_discarded(self) -> None:
        """So the gap between the defensible figure and the tempting one is
        visible in the record instead of being quietly avoided."""
        edges = [_edge("A", "B", 2.0), _edge("B", "C", 3.0)]
        evidence = bound_path_evidence((NodeKind.INCIDENT.value, "A"), edges)

        assert evidence.naive_product_log_lr == pytest.approx(5.0)
        assert evidence.point_log_lr < evidence.naive_product_log_lr

    def test_one_edge_against_linkage_poisons_the_chain(self) -> None:
        """A chain containing evidence for *different* sources cannot support
        linkage, however strong the other hops are."""
        edges = [
            _edge("A", "B", 6.0),
            _edge("B", "C", -4.0),
            _edge("C", "D", 6.0),
        ]
        evidence = bound_path_evidence((NodeKind.INCIDENT.value, "A"), edges)

        assert evidence.point_log_lr == pytest.approx(-4.0)

    def test_a_path_needs_an_edge(self) -> None:
        with pytest.raises(InsufficientDataError):
            bound_path_evidence((NodeKind.INCIDENT.value, "A"), [])

    def test_the_path_and_hop_count_are_reported(self) -> None:
        edges = [_edge("A", "B", 2.0), _edge("B", "C", 3.0)]
        evidence = bound_path_evidence((NodeKind.INCIDENT.value, "A"), edges)

        assert evidence.n_hops == 2
        assert len(evidence.path) == 3


class TestSharedStreamFraction:
    """Hops resting on the same streams err together, and the propagation says so."""

    def test_disjoint_streams_share_nothing(self) -> None:
        edges = [
            _edge("A", "B", 2.0, streams=frozenset({ACOUSTIC})),
            _edge("B", "C", 2.0, streams=frozenset({BEHAVIOURAL})),
        ]
        assert _shared_stream_fraction(edges) == pytest.approx(0.0)

    def test_identical_streams_share_everything(self) -> None:
        edges = [
            _edge("A", "B", 2.0, streams=frozenset({ACOUSTIC})),
            _edge("B", "C", 2.0, streams=frozenset({ACOUSTIC})),
        ]
        assert _shared_stream_fraction(edges) == pytest.approx(1.0)

    def test_partial_overlap_is_the_jaccard_index(self) -> None:
        edges = [
            _edge("A", "B", 2.0, streams=frozenset({ACOUSTIC, BEHAVIOURAL})),
            _edge("B", "C", 2.0, streams=frozenset({ACOUSTIC, TEMPORAL})),
        ]
        # Intersection {acoustic} over union {acoustic, behavioural, temporal}.
        assert _shared_stream_fraction(edges) == pytest.approx(1.0 / 3.0)

    def test_a_single_hop_shares_nothing_with_itself(self) -> None:
        assert _shared_stream_fraction([_edge("A", "B", 2.0)]) == 0.0


class TestAnalyticPropagation:
    def test_shared_streams_widen_the_interval(self) -> None:
        """Correlated hops carry less independent information, and the interval
        has to show it. Assuming independence here is the error that makes a
        two-hop chain look like a measurement."""
        correlated = propagate_analytic(
            (NodeKind.INCIDENT.value, "A"),
            [
                _edge("A", "B", 3.0, streams=frozenset({ACOUSTIC})),
                _edge("B", "C", 3.0, streams=frozenset({ACOUSTIC})),
            ],
        )
        independent = propagate_analytic(
            (NodeKind.INCIDENT.value, "A"),
            [
                _edge("A", "B", 3.0, streams=frozenset({ACOUSTIC})),
                _edge("B", "C", 3.0, streams=frozenset({BEHAVIOURAL})),
            ],
        )

        correlated_width = correlated.interval.upper - correlated.interval.lower
        independent_width = independent.interval.upper - independent.interval.lower
        assert correlated_width > independent_width

    def test_it_discounts_rather_than_multiplying(self) -> None:
        edges = [
            _edge("A", "B", math.log(30.0)),
            _edge("B", "C", math.log(30.0)),
        ]
        evidence = propagate_analytic((NodeKind.INCIDENT.value, "A"), edges)

        assert evidence.point_log_lr < evidence.naive_product_log_lr

    def test_a_single_hop_is_not_discounted_into_meaninglessness(self) -> None:
        """One hop is a comparison that actually happened. Nothing is being
        propagated, so nothing should be taken away for propagating."""
        evidence = propagate_analytic(
            (NodeKind.INCIDENT.value, "A"), [_edge("A", "B", 3.0)]
        )
        assert evidence.point_log_lr == pytest.approx(3.0)
