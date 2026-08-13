"""The linkage graph, with probabilistic edges kept probabilistic.

The graph is a consequence of fusion rather than a parallel system. Its central
design property is a negative one: :class:`~viflap.analysis.graph.model.LinkageGraph`
has no method that returns a path or a component computed over mixed edges. The
operations that would silently treat a likelihood ratio as a fact are absent, not
merely discouraged.

``model``
    Nodes, and the two kinds of edge — deterministic facts of record, and
    probabilistic linkages carrying a likelihood ratio and an interval.
``propagation``
    Traversal, returning distributions and bounds rather than values. Includes
    the conservative weakest-link bound, which is the only one of the three
    methods safe to report without qualification.
``community``
    Clustering with stability assessed by resampling every edge from its own
    interval, so a community that exists only at one draw of the evidence is
    visible as such.
"""

from viflap.analysis.graph.community import (
    Community,
    CommunityReport,
    detect_communities,
)
from viflap.analysis.graph.model import (
    EdgeKind,
    FactualEdge,
    LinkageEdge,
    LinkageGraph,
    Node,
    NodeKind,
)
from viflap.analysis.graph.propagation import (
    PathEvidence,
    bound_path_evidence,
    find_linkage_paths,
    propagate_analytic,
    propagate_monte_carlo,
)

__all__ = [
    "Community",
    "CommunityReport",
    "EdgeKind",
    "FactualEdge",
    "LinkageEdge",
    "LinkageGraph",
    "Node",
    "NodeKind",
    "PathEvidence",
    "bound_path_evidence",
    "detect_communities",
    "find_linkage_paths",
    "propagate_analytic",
    "propagate_monte_carlo",
]
