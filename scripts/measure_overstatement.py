"""How badly does assuming independent streams overstate the evidence?

The research proposal names this an explicit deliverable, and the reason it
deserves that status is in `viflap/analysis/fusion/overstatement.py`: even if the
dependence-corrected models gain little over naive summation, "how badly does the
standard method mislead" is answered either way, and naive summation is what
deployed multimodal forensic systems actually use. A finding that it overstates
by two orders of magnitude is a safety result whether or not the correction is
worth its complexity.

Why this is a simulation, and what stops it being arbitrary
-----------------------------------------------------------
Answering the question with real data needs incidents carrying several streams
with known linkage truth. No such corpus exists for this setting, and none is
going to appear during this project. But the question is about **the method**,
not about Zambia: it asks what independence costs when streams are correlated,
and that is answerable wherever the inputs are realistic.

So the simulation invents as little as possible.

**The acoustic marginal is real.** Same-source and different-source log-LRs are
resampled from the 25,088 evaluation trials in
`data/reports/calibration_scores.npz`, calibrated exactly as the system
calibrates them — fitted on development speakers, applied to evaluation
speakers. The shape of that distribution, including its overlap and its tails, is
what this system actually produces through the parametric CELP channel.

**The dependence mechanism is the documented one.** `EvidenceStream` says the
streams "are *not* conditionally independent of one another — the same operator
running the same operation is the common cause of all of them". The simulation
induces dependence exactly that way: a per-incident latent factor standing for
the operation, shared across streams, with the streams' own idiosyncratic
variation on top. It is not a correlation knob bolted onto independent draws; it
is the causal structure the module describes.

**The correlation is swept rather than assumed.** Nobody knows the true value.
Reporting a curve over plausible values says more than reporting one number from
a guess, and it shows where the correction starts to matter.

What is still invented
----------------------
The behavioural, temporal and device marginals. They are given the same
distributional family as the acoustic one with weaker separation, which is a
stated assumption and not a measurement — the acoustic stream is the only one
this project has evaluated end to end. The overstatement curve is therefore
about the *structure* of the dependence problem, and the absolute figures would
move under different marginals.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.calibration.calibrators import LogisticCalibrator
from viflap.analysis.calibration.metrics import compute_cllr
from viflap.analysis.fusion.base import FusionObservation, FusionTrainingSet
from viflap.analysis.fusion.models import (
    GaussianLatentFusion,
    LinearLogisticFusion,
    NaiveIndependentFusion,
)
from viflap.analysis.fusion.overstatement import measure_overstatement
from viflap.domain.evidence import EvidenceStream

#: Streams simulated. The acoustic marginal is measured; the rest are assumed.
_STREAMS = (
    EvidenceStream.ACOUSTIC,
    EvidenceStream.BEHAVIOURAL,
    EvidenceStream.TEMPORAL,
)

#: Separation of the assumed marginals relative to the measured acoustic one.
#: Below one because the acoustic stream is the only one with a measured
#: discrimination figure, and assuming the unmeasured streams are stronger than
#: the measured one would flatter every result that follows.
_ASSUMED_STRENGTH = {
    EvidenceStream.BEHAVIOURAL: 0.75,
    EvidenceStream.TEMPORAL: 0.50,
}


def load_acoustic_log_lrs(
    path: Path, cell: str = "amr12.2_clean@30"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Real calibrated log-LRs for one cell, split by truth.

    Calibrated the way the system calibrates: the affine map is fitted on the
    development speakers and applied to the evaluation speakers, so the returned
    values are what a deployment would actually emit rather than raw scores on
    an arbitrary scale.
    """
    stored = np.load(path, allow_pickle=True)
    calibrator = LogisticCalibrator().fit(
        stored[f"{cell}|dev_scores"], stored[f"{cell}|dev_labels"]
    )

    # ``calibrate``, not ``transform``. The latter is the raw affine map and is
    # not what the system may report: 60.6% of this cell's trials are clipped to
    # the ELUB bounds, and the unbounded map runs to -16.1 log10 where the
    # bounds stop at -2.35. Feeding an overstatement study values twelve orders
    # of magnitude beyond anything the system can emit measures the calibrator's
    # tail rather than the cost of assuming independence. The first version of
    # this script did exactly that, which is the same error §2 of the results
    # document records the first draft making.
    log_lrs = np.array(
        [
            calibrator.calibrate(float(score)).log_lr.value
            for score in stored[f"{cell}|eval_scores"]
        ]
    )
    labels = stored[f"{cell}|eval_labels"]
    return log_lrs[labels == 1], log_lrs[labels == 0]


def _quantile_map(
    gaussian: NDArray[np.float64], sample: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map standard-normal draws onto an empirical distribution by quantile.

    This is what keeps the marginals real while the dependence is imposed: the
    latent structure decides the *rank* of each draw, and the empirical sample
    decides what value that rank corresponds to. Fitting a normal to the
    acoustic log-LRs instead would discard the asymmetry and the tails, which is
    where an overstatement study lives.
    """
    from scipy.stats import norm

    ordered = np.sort(sample)
    positions = np.clip(norm.cdf(gaussian), 1e-6, 1 - 1e-6)
    indices = (positions * (ordered.size - 1)).astype(np.int64)
    return ordered[indices]


def _fused_cllr(model: object, evaluation: FusionTrainingSet) -> float:
    """``C_llr`` of one fusion model over the evaluation comparisons.

    Reported for all three arms because it is the quantity that says which model
    a deployment should prefer. The exaggeration statistic says how far two
    models disagree; it does not say which is right, and a "correction" that
    disagrees with the baseline while scoring worse than it is not a correction.
    """
    fused: list[float] = []
    labels: list[int] = []
    for observation in evaluation.observations:
        if not observation.log_lrs:
            continue
        if not model.supports_pattern(observation.pattern):  # type: ignore[attr-defined]
            continue
        fused.append(model.fuse(observation.log_lrs))  # type: ignore[attr-defined]
        labels.append(1 if observation.is_same_source else 0)
    return float(compute_cllr(np.asarray(fused), np.asarray(labels, dtype=np.int64)))


def simulate(
    same_source: NDArray[np.float64],
    different_source: NDArray[np.float64],
    correlation: float,
    n_incidents: int,
    rng: np.random.Generator,
) -> FusionTrainingSet:
    """Generate comparisons whose streams share an operation-level cause.

    ``correlation`` is the fraction of each stream's latent variance coming from
    the shared factor. Zero makes the streams conditionally independent, which is
    the assumption naive summation makes; one makes them redundant.
    """
    observations: list[FusionObservation] = []
    truth = rng.random(n_incidents) < 0.5

    shared = rng.standard_normal(n_incidents)
    for index in range(n_incidents):
        latent = shared[index]
        log_lrs: dict[EvidenceStream, float] = {}
        for stream in _STREAMS:
            idiosyncratic = rng.standard_normal()
            combined = (
                np.sqrt(correlation) * latent + np.sqrt(1.0 - correlation) * idiosyncratic
            )

            sample = same_source if truth[index] else different_source
            value = float(_quantile_map(np.array([combined]), sample)[0])

            # The unmeasured streams are weaker than the measured one. Shrinking
            # towards zero reduces separation without changing the distribution
            # family, so the comparison across streams stays like-for-like.
            value *= _ASSUMED_STRENGTH.get(stream, 1.0)
            log_lrs[stream] = value

        observations.append(
            FusionObservation(
                log_lrs=log_lrs,
                is_same_source=bool(truth[index]),
                group_id=f"incident-{index:05d}",
            )
        )
    return FusionTrainingSet(observations)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", type=Path, default=Path("data/reports/calibration_scores.npz")
    )
    parser.add_argument("--cell", default="amr12.2_clean@30")
    parser.add_argument("--incidents", type=int, default=4000)
    parser.add_argument(
        "--correlations",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/overstatement.json")
    )
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    same, different = load_acoustic_log_lrs(arguments.scores, arguments.cell)
    print(
        f"acoustic marginal from {arguments.cell}: "
        f"{same.size} same-source, {different.size} different-source trials",
        flush=True,
    )
    print(
        f"  same-source   mean {same.mean():+.2f}  sd {same.std():.2f}\n"
        f"  different     mean {different.mean():+.2f}  sd {different.std():.2f}",
        flush=True,
    )

    results: list[dict[str, object]] = []
    for correlation in arguments.correlations:
        rng = np.random.default_rng(arguments.seed)
        training = simulate(same, different, correlation, arguments.incidents, rng)
        evaluation = simulate(
            same,
            different,
            correlation,
            arguments.incidents,
            np.random.default_rng(arguments.seed + 1),
        )

        # Three arms, not two. The first version compared naive summation
        # against a dependence model and reported the difference as the cost of
        # assuming independence. It is not: `NaiveIndependentFusion` is an
        # unweighted sum with no fitted parameters, so the difference confounds
        # dependence with calibration. `LinearLogisticFusion` still assumes
        # independence but fits weights, which separates the two — whatever it
        # gains over the naive sum is calibration, and only what the dependence
        # model gains over *it* is dependence.
        latent = GaussianLatentFusion().fit(training)
        linear = LinearLogisticFusion().fit(training)
        report = measure_overstatement(latent, evaluation)

        costs = {
            "naive_sum": _fused_cllr(NaiveIndependentFusion(), evaluation),
            "linear_logistic": _fused_cllr(linear, evaluation),
            "gaussian_latent": _fused_cllr(latent, evaluation),
        }

        exaggeration = report.exaggeration_log10
        entry = {
            "correlation": correlation,
            "n_comparisons": report.n_comparisons,
            "c_llr": costs,
            "median_exaggeration_log10": float(np.median(exaggeration)),
            "p95_exaggeration_log10": float(np.percentile(exaggeration, 95)),
            "n_band_changes": report.n_band_changes,
            "band_change_rate": report.n_band_changes / max(report.n_comparisons, 1),
        }
        results.append(entry)
        print(
            f"rho {correlation:.1f}:  C_llr  naive {costs['naive_sum']:.3f}  "
            f"linear {costs['linear_logistic']:.3f}  latent {costs['gaussian_latent']:.3f}"
            f"   | median exaggeration {entry['median_exaggeration_log10']:+.2f} log10, "
            f"band changes {entry['band_change_rate']:.1%}",
            flush=True,
        )

    artefact = {
        "cell": arguments.cell,
        "n_incidents": arguments.incidents,
        "streams": [s.value for s in _STREAMS],
        "acoustic_marginal": {
            "source": str(arguments.scores),
            "n_same_source": int(same.size),
            "n_different_source": int(different.size),
            "same_source_mean": float(same.mean()),
            "different_source_mean": float(different.mean()),
        },
        "assumed_strengths": {s.value: v for s, v in _ASSUMED_STRENGTH.items()},
        "results": results,
        "caveat": (
            "The acoustic marginal is measured; the behavioural and temporal "
            "marginals are assumed to share its distributional family with "
            "weaker separation. The curve describes the structure of the "
            "dependence problem, not the magnitudes any particular deployment "
            "would see."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    print(f"wrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
