"""Evaluation: deciding the hypotheses, honestly.

``splits``
    Speaker-disjoint splitting and speaker-level resampling. Two rules that
    govern every number this system reports, and whose violation produces
    results that are optimistic, confident and wrong.
``hypotheses``
    H1 to H7 as executable protocols, with their falsification conditions fixed
    in advance and recorded with each result. Distinguishes "falsified" from
    "inconclusive", because an underpowered experiment is a fact about the
    experiment rather than about the world.
``ablation``
    Every subset of streams, and each stream's marginal contribution given the
    others — which is the quantity that matters when the streams share a cause.
``reporting``
    Assembling results into a document, with the conditions attached.
"""

from viflap.evaluation.ablation import AblationReport, SubsetResult, run_ablation
from viflap.evaluation.hypotheses import (
    H1ChannelViability,
    H2DisguiseResistance,
    H3DisguiseDecay,
    H4CrossLingualPenalty,
    H5FusionSuperiority,
    H7SyntheticGating,
    HypothesisOutcome,
    holm_bonferroni,
)
from viflap.evaluation.splits import (
    SpeakerDisjointSplit,
    bootstrap_over_speakers,
    make_folds,
    verify_disjoint,
)

__all__ = [
    "AblationReport",
    "H1ChannelViability",
    "H2DisguiseResistance",
    "H3DisguiseDecay",
    "H4CrossLingualPenalty",
    "H5FusionSuperiority",
    "H7SyntheticGating",
    "HypothesisOutcome",
    "SpeakerDisjointSplit",
    "SubsetResult",
    "bootstrap_over_speakers",
    "holm_bonferroni",
    "make_folds",
    "run_ablation",
    "verify_disjoint",
]
