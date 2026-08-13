"""VIFLAP — calibrated multi-evidence case linkage for telephony-enabled fraud.

The system evaluates how much the available evidence shifts the odds that two
reported incidents were conducted by the same actor, and reports that as a
calibrated likelihood ratio. It does not assert identity, and it has no
vocabulary for doing so.

Layering, innermost first. Dependencies point inward only.

``viflap.domain``
    Concepts and their invariants. Standard library only.
``viflap.analysis``
    The science: signal processing, statistical models, calibration, fusion,
    the linkage graph. No I/O.
``viflap.evaluation``
    Deciding the research hypotheses, with speaker-disjoint splits and
    speaker-level resampling.
``viflap.application``
    Use cases expressed against ports.
``viflap.infrastructure``
    Adapters satisfying those ports.
``viflap.interfaces``
    Delivery.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
