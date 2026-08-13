"""Diagnostic plots for calibration.

Two plots, chosen because they answer questions that summary statistics cannot.

**Tippett plot.** The cumulative distributions of likelihood ratios for
same-source and different-source trials, on the same axes. A single ``C_llr``
value cannot show *where* a system fails, and the failures that matter here are
in the tails: how often does a different-source pair receive a strongly
supporting likelihood ratio? That question is a point on this plot, and it is
the question a court would ask.

**Reliability plot.** The fitted calibration curve — the optimal monotonic
transform from PAV — against the identity. Where the curve lies above the
identity, the system is understating; below, overstating. The direction matters
more than the magnitude, because overstatement runs against the person the
comparison concerns.

Both return figures rather than writing files, so the caller decides where the
output goes, and neither imports a plotting backend at module scope.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.calibration.metrics import split_by_label
from viflap.analysis.calibration.pav import pav_calibrate
from viflap.domain.errors import InsufficientDataError

__all__ = ["reliability_plot", "tippett_plot"]

_LN10 = math.log(10.0)


def _figure(figsize: tuple[float, float]):
    """Create a figure without requiring an interactive backend.

    ``matplotlib`` is imported here rather than at module scope: this package is
    imported by the API process, which has no display and no need to load a
    plotting stack on every request.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def tippett_plot(
    log_lrs: NDArray[np.float64],
    labels: NDArray[np.int64],
    title: str = "Tippett plot",
    *,
    comparisons: Mapping[str, tuple[NDArray[np.float64], NDArray[np.int64]]] | None = None,
):
    """Cumulative likelihood-ratio distributions by trial type.

    The same-source curve shows the proportion of same-source trials with
    ``log10 LR`` **greater than** the abscissa; the different-source curve shows
    the proportion **less than** it. Drawn this way, both curves fall from left
    to right toward their own tail, and the region where they cross is the
    region where the system confuses the two.

    Two reference marks are drawn because they are what a reader should look
    for. The vertical line at zero separates support for one proposition from
    support for the other, so the different-source curve's height there is the
    rate at which the system misleadingly supports linkage. The annotation
    reports that rate numerically, since reading it off a log axis by eye
    invites optimism.
    """
    same, different = split_by_label(log_lrs, labels)
    if same.size == 0 or different.size == 0:
        raise InsufficientDataError("a Tippett plot requires trials of both types")

    figure, axes = _figure((8.0, 5.5))
    _draw_tippett_pair(
        axes, same, different, label_prefix="", colours=("#1f77b4", "#d62728")
    )

    if comparisons:
        palette = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
        for index, (name, (other_lrs, other_labels)) in enumerate(comparisons.items()):
            other_same, other_different = split_by_label(other_lrs, other_labels)
            colour = palette[index % len(palette)]
            _draw_tippett_pair(
                axes,
                other_same,
                other_different,
                label_prefix=f"{name} ",
                colours=(colour, colour),
                linestyles=("-", "--"),
                alpha=0.65,
            )

    misleading = float(np.mean(different > 0.0))
    axes.axvline(0.0, color="black", linewidth=1.0, zorder=1)
    axes.set_xlabel(r"$\log_{10}$ likelihood ratio")
    axes.set_ylabel("Cumulative proportion of trials")
    axes.set_ylim(0.0, 1.02)
    axes.set_title(title)
    axes.grid(True, linestyle=":", alpha=0.5)
    axes.legend(loc="upper right", fontsize=9)
    axes.annotate(
        f"{misleading:.1%} of different-source trials receive\n"
        f"a likelihood ratio supporting linkage",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=9,
        color="#d62728",
    )
    figure.tight_layout()
    return figure


def _draw_tippett_pair(
    axes,
    same: NDArray[np.float64],
    different: NDArray[np.float64],
    label_prefix: str,
    colours: tuple[str, str],
    linestyles: tuple[str, str] = ("-", "-"),
    alpha: float = 1.0,
) -> None:
    """Draw one system's pair of cumulative curves."""
    same_log10 = np.sort(same / _LN10)
    same_proportion = 1.0 - np.arange(same_log10.size) / same_log10.size
    axes.step(
        same_log10,
        same_proportion,
        where="post",
        color=colours[0],
        linestyle=linestyles[0],
        alpha=alpha,
        label=f"{label_prefix}same-source (proportion above)",
    )

    different_log10 = np.sort(different / _LN10)
    different_proportion = np.arange(1, different_log10.size + 1) / different_log10.size
    axes.step(
        different_log10,
        different_proportion,
        where="post",
        color=colours[1],
        linestyle=linestyles[1],
        alpha=alpha,
        label=f"{label_prefix}different-source (proportion below)",
    )


def reliability_plot(
    log_lrs: NDArray[np.float64],
    labels: NDArray[np.int64],
    title: str = "Calibration reliability",
):
    """Reported likelihood ratio against the optimally calibrated one.

    The horizontal axis is what the system said; the vertical is what an
    optimal monotonic recalibration of the same scores would have said. Points
    on the diagonal are honest. Points below it overstate — the system claimed
    more than the ordering of its own scores supports.

    The asymmetry is deliberate in the annotation: overstatement is called out
    numerically because it is the direction that produces false accusations,
    whereas understatement merely wastes evidence.
    """
    log_lrs = np.asarray(log_lrs, dtype=np.float64)
    optimal = pav_calibrate(log_lrs, labels)

    finite = np.isfinite(optimal)
    reported_log10 = log_lrs[finite] / _LN10
    optimal_log10 = optimal[finite] / _LN10

    figure, axes = _figure((6.0, 6.0))

    limit = float(
        max(
            np.abs(reported_log10).max(initial=1.0),
            np.abs(optimal_log10).max(initial=1.0),
        )
    )
    axes.plot(
        [-limit, limit],
        [-limit, limit],
        color="black",
        linewidth=1.0,
        label="perfectly calibrated",
    )
    axes.scatter(
        reported_log10,
        optimal_log10,
        s=8,
        alpha=0.4,
        c=np.where(np.asarray(labels)[finite] == 1, "#1f77b4", "#d62728"),
    )

    overstated = float(np.mean(np.abs(reported_log10) > np.abs(optimal_log10) + 0.5))
    axes.set_xlabel(r"reported $\log_{10}$ LR")
    axes.set_ylabel(r"optimally recalibrated $\log_{10}$ LR")
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_title(title)
    axes.grid(True, linestyle=":", alpha=0.5)
    axes.legend(loc="upper left", fontsize=9)
    axes.annotate(
        f"{overstated:.1%} of trials overstated by more\nthan half an order of magnitude",
        xy=(0.03, 0.03),
        xycoords="axes fraction",
        fontsize=9,
    )
    figure.tight_layout()
    return figure
