"""Calibration and forensic evaluation.

This is the layer at which a number becomes evidence. Everything below produces
scores whose scale is an accident of construction; nothing below this package
returns a :class:`~viflap.domain.values.LogLikelihoodRatio`, and everything
above receives only calibrated values.

``pav``
    Pool-adjacent-violators. Optimal monotonic calibration, in linear time.
``metrics``
    ``C_llr``, its decomposition into discrimination and calibration loss, and
    equal error rate — the last reported for comparability with the literature
    and used for nothing.
``elub``
    Empirical bounds on what a validation set can support. Applied by default,
    because a calibration model asked for a likelihood ratio beyond its data
    will supply one.
``calibrators``
    Logistic, isotonic and kernel-density calibration behind one interface.
``plots``
    Tippett and reliability plots — the diagnostics that show *where* a system
    fails, which a summary statistic cannot.
"""

from viflap.analysis.calibration.calibrators import (
    CalibratedOutput,
    Calibrator,
    IsotonicCalibrator,
    KernelDensityCalibrator,
    LogisticCalibrator,
)
from viflap.analysis.calibration.elub import (
    EmpiricalBounds,
    apply_bounds,
    empirical_bounds,
)
from viflap.analysis.calibration.metrics import (
    compute_cllr,
    compute_cllr_min,
    compute_eer,
    evaluate,
    split_by_label,
)
from viflap.analysis.calibration.pav import (
    PavResult,
    pav_calibrate,
    pool_adjacent_violators,
)
from viflap.analysis.calibration.plots import reliability_plot, tippett_plot

__all__ = [
    "CalibratedOutput",
    "Calibrator",
    "EmpiricalBounds",
    "IsotonicCalibrator",
    "KernelDensityCalibrator",
    "LogisticCalibrator",
    "PavResult",
    "apply_bounds",
    "compute_cllr",
    "compute_cllr_min",
    "compute_eer",
    "empirical_bounds",
    "evaluate",
    "pav_calibrate",
    "pool_adjacent_violators",
    "reliability_plot",
    "split_by_label",
    "tippett_plot",
]
