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
from collections.abc import Mapping, Sequence
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

#: Scale applied to the assumed marginals relative to the measured acoustic one.
#:
#: **This does not make a stream weaker, and the name and the original comment
#: both said it did.** Multiplying a stream's log-LRs by a positive constant is
#: monotonic: the ranking of trials is untouched, so ``C_llr_min`` is exactly
#: unchanged and every stream here has *identical* discrimination whatever this
#: says. What moves is ``C_llr`` — the stream becomes under-confident, which is a
#: calibration defect rather than a quality one.
#:
#: The consequence is that this constant reaches exactly one of the three fusion
#: arms. ``LinearLogisticFusion`` refits ``w_i`` and absorbs the scaling to
#: optimiser tolerance; ``GaussianLatentFusion`` refits means and covariances and
#: the Jacobian cancels between numerator and denominator. Only
#: ``NaiveIndependentFusion`` adds the values as given and has nothing to refit,
#: so it is the only arm that can feel this — and at high correlation the
#: under-confidence partly offsets the double-counting, which *helps* it.
#:
#: Kept at these values rather than removed, because every published figure in
#: §11 was produced with them and changing them silently would make the section
#: unreproducible. §11 records which of its claims survive: the latent-versus-
#: linear comparison does, because both arms are invariant, and anything
#: involving the naive sum does not.
_ASSUMED_STRENGTH = {
    EvidenceStream.BEHAVIOURAL: 0.75,
    EvidenceStream.TEMPORAL: 0.50,
}


def load_acoustic_log_lrs(
    path: Path, cell: str = "amr12.2_clean@30"
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.str_]]:
    """Real calibrated log-LRs for one cell, with their labels and owners.

    Calibrated the way the system calibrates: the affine map is fitted on the
    development speakers and applied to the evaluation speakers, so the returned
    values are what a deployment would actually emit rather than raw scores on
    an arbitrary scale.

    The speaker column comes back too, because the marginal is an *estimate* from
    102 speakers and resampling it is the only way any interval here reflects
    that. §2's rule applies to this resampling as much as to any other in the
    project: speakers, never trials.
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
    return log_lrs, labels, stored[f"{cell}|eval_speakers"]


def resample_marginal(
    log_lrs: NDArray[np.float64],
    labels: NDArray[np.int64],
    speakers: NDArray[np.str_],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw a replicate of the acoustic marginal, resampling **speakers**.

    Replicating the simulation with a fresh seed measures Monte-Carlo noise in
    the simulation and nothing else — every replicate would otherwise draw from
    the same fixed 25,088 trials, treating an estimate from 102 speakers as if
    it were the population. That is the same mistake §2 rejects for the acoustic
    intervals, and it would produce intervals here that are too narrow for a
    reason having nothing to do with the number of incidents simulated.

    Speakers are drawn with replacement and all of a drawn speaker's trials come
    with them, so the correlation between one speaker's trials is preserved.
    """
    unique = np.unique(speakers)
    drawn = rng.choice(unique, size=unique.size, replace=True)
    selected = np.concatenate([np.flatnonzero(speakers == name) for name in drawn])
    values, truth = log_lrs[selected], labels[selected]
    return values[truth == 1], values[truth == 0]


def _from_positions(
    positions: NDArray[np.float64], sample: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map uniform positions onto an empirical distribution by quantile.

    This is what keeps the marginals real while the dependence is imposed: the
    latent structure decides the *rank* of each draw, and the empirical sample
    decides what value that rank corresponds to. Fitting a normal to the
    acoustic log-LRs instead would discard the asymmetry and the tails, which is
    where an overstatement study lives.

    Taking uniforms rather than latent values is what lets the copula vary
    independently of the marginal — which is the entire point of a copula, and
    the reason a t-copula arm costs one line here rather than a second generator.
    """
    ordered = np.sort(sample)
    clipped = np.clip(positions, 1e-6, 1 - 1e-6)
    indices = (clipped * (ordered.size - 1)).astype(np.int64)
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


def weaken(
    same_source: NDArray[np.float64],
    different_source: NDArray[np.float64],
    discriminability: float,
) -> NDArray[np.float64]:
    """Make a stream genuinely less discriminative, not merely less confident.

    ``_ASSUMED_STRENGTH`` scales log-LRs, which is monotonic and therefore leaves
    ``C_llr_min`` exactly unchanged — §11 records that finding and the fact that
    the constant reaches only the naive sum. Reducing *discrimination* needs the
    two class distributions to overlap more, so this slides the same-source
    marginal toward the different-source one by a fraction of the gap between
    their means, leaving both spreads alone.

    ``discriminability = 1`` is the measured marginal untouched; ``0`` puts the
    two means on top of each other, which is a stream carrying no information.
    Nothing here is calibrated afterwards, so what comes out is a stream whose
    ROC has genuinely moved.
    """
    if discriminability >= 1.0:
        return same_source
    gap = float(np.mean(same_source) - np.mean(different_source))
    return same_source - (1.0 - discriminability) * gap


def simulate(
    same_source: NDArray[np.float64],
    different_source: NDArray[np.float64],
    correlation: float,
    n_incidents: int,
    rng: np.random.Generator,
    copula: str = "gaussian",
    degrees_of_freedom: float = 4.0,
    discriminability: Mapping[EvidenceStream, float] | None = None,
) -> FusionTrainingSet:
    """Generate comparisons whose streams share an operation-level cause.

    ``correlation`` is the fraction of each stream's latent variance coming from
    the shared factor. Zero makes the streams conditionally independent, which is
    the assumption naive summation makes; one makes them redundant.

    ``copula`` selects the dependence structure and exists to break the one thing
    the original version of §11 could not test. Under ``"gaussian"`` the
    generative process is a Gaussian latent factor and
    :class:`GaussianLatentFusion` is a Gaussian latent-factor model, so the
    dependence arm is *correctly specified by construction* — the best case for
    it, which it lost anyway. Under ``"t"`` the same linear correlation is
    imposed through a Student-t copula instead, which adds joint tail dependence
    that no Gaussian copula has at any correlation. The fitted model is unchanged,
    so this is the misspecified case, and it is the operationally honest one: a
    real operation's streams fail together.
    """
    from scipy.stats import norm, t

    truth = rng.random(n_incidents) < 0.5
    shared = rng.standard_normal(n_incidents)
    idiosyncratic = rng.standard_normal((n_incidents, len(_STREAMS)))
    combined = (
        np.sqrt(correlation) * shared[:, None] + np.sqrt(1.0 - correlation) * idiosyncratic
    )

    if copula == "t":
        # One chi-square draw per incident, **shared across its streams**. The
        # sharing is the whole mechanism: it is what makes extreme values arrive
        # together rather than independently, and it is exactly the structure a
        # Gaussian latent model has no parameter for.
        scale = np.sqrt(rng.chisquare(degrees_of_freedom, n_incidents) / degrees_of_freedom)
        positions = t.cdf(combined / scale[:, None], degrees_of_freedom)
    elif copula == "gaussian":
        positions = norm.cdf(combined)
    else:
        raise ValueError(f"unknown copula {copula!r}; expected 'gaussian' or 't'")

    values = np.empty_like(positions)
    for index in range(len(_STREAMS)):
        column = positions[:, index]
        # Each stream may carry its own same-source marginal, which is what makes
        # one stream genuinely weaker than another rather than merely quieter.
        stream_same = same_source
        if discriminability is not None:
            factor = discriminability.get(_STREAMS[index], 1.0)
            stream_same = weaken(same_source, different_source, factor)
        values[truth, index] = _from_positions(column[truth], stream_same)
        values[~truth, index] = _from_positions(column[~truth], different_source)
        # Scales the stream's confidence, NOT its discrimination — this is a
        # monotonic map and leaves C_llr_min exactly unchanged. See
        # ``_ASSUMED_STRENGTH`` for what that means for each fusion arm.
        values[:, index] *= _ASSUMED_STRENGTH.get(_STREAMS[index], 1.0)

    return FusionTrainingSet(
        [
            FusionObservation(
                log_lrs={
                    stream: float(values[row, index])
                    for index, stream in enumerate(_STREAMS)
                },
                is_same_source=bool(truth[row]),
                group_id=f"incident-{row:05d}",
            )
            for row in range(n_incidents)
        ]
    )


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
    parser.add_argument(
        "--replicates",
        type=int,
        default=40,
        help=(
            "Independent replicates per correlation. One is the old behaviour "
            "and produces point estimates with no interval, which is what §11 "
            "was criticised for."
        ),
    )
    parser.add_argument("--copula", choices=("gaussian", "t"), default="gaussian")
    parser.add_argument(
        "--behavioural-discriminability",
        type=float,
        default=1.0,
        help=(
            "Slide the behavioural stream's same-source marginal toward the "
            "different-source one by (1 - this) of the gap, making it genuinely "
            "less discriminative. 1.0 leaves it equal to the acoustic stream, "
            "which is what every figure in §11 before this option was produced "
            "with. See `weaken` on why scaling log-LRs does not do this."
        ),
    )
    parser.add_argument("--degrees-of-freedom", type=float, default=4.0)
    parser.add_argument(
        "--no-resample-marginal",
        action="store_true",
        help=(
            "Hold the acoustic marginal fixed across replicates. The interval "
            "then covers simulation noise only and understates, because the "
            "marginal is itself an estimate from 102 speakers."
        ),
    )
    arguments = parser.parse_args(argv)

    log_lrs, labels, speakers = load_acoustic_log_lrs(arguments.scores, arguments.cell)
    same, different = log_lrs[labels == 1], log_lrs[labels == 0]
    print(
        f"acoustic marginal from {arguments.cell}: "
        f"{same.size} same-source, {different.size} different-source trials "
        f"over {np.unique(speakers).size} speakers",
        flush=True,
    )
    print(
        f"  same-source   mean {same.mean():+.2f}  sd {same.std():.2f}\n"
        f"  different     mean {different.mean():+.2f}  sd {different.std():.2f}",
        flush=True,
    )
    print(
        f"  {arguments.replicates} replicates, {arguments.copula} copula"
        + ("" if arguments.no_resample_marginal else ", marginal resampled over speakers"),
        flush=True,
    )

    discriminability = {EvidenceStream.BEHAVIOURAL: arguments.behavioural_discriminability}
    results: list[dict[str, object]] = []
    for correlation in arguments.correlations:
        per_replicate: dict[str, list[float]] = {
            "naive_sum": [],
            "linear_logistic": [],
            "gaussian_latent": [],
        }
        exaggerations: list[float] = []
        band_rates: list[float] = []

        for replicate in range(arguments.replicates):
            # Every replicate gets its own stream, and the marginal is redrawn
            # with it, so the spread below carries both sources of variation.
            base = arguments.seed + 1000 * replicate
            if arguments.no_resample_marginal:
                marginal_same, marginal_different = same, different
            else:
                marginal_same, marginal_different = resample_marginal(
                    log_lrs, labels, speakers, np.random.default_rng(base + 7)
                )

            training = simulate(
                marginal_same,
                marginal_different,
                correlation,
                arguments.incidents,
                np.random.default_rng(base),
                copula=arguments.copula,
                degrees_of_freedom=arguments.degrees_of_freedom,
                discriminability=discriminability,
            )
            evaluation = simulate(
                marginal_same,
                marginal_different,
                correlation,
                arguments.incidents,
                np.random.default_rng(base + 1),
                copula=arguments.copula,
                degrees_of_freedom=arguments.degrees_of_freedom,
                discriminability=discriminability,
            )
            _accumulate(training, evaluation, per_replicate, exaggerations, band_rates)

        entry = _summarise(correlation, per_replicate, exaggerations, band_rates)
        results.append(entry)
        costs = {name: entry["c_llr"][name]["mean"] for name in per_replicate}  # type: ignore[index]
        print(
            f"rho {correlation:.1f}:  C_llr  naive {costs['naive_sum']:.3f}  "
            f"linear {costs['linear_logistic']:.3f}  latent {costs['gaussian_latent']:.3f}"
            f"   | latent-minus-linear "
            f"{entry['latent_minus_linear']['mean']:+.3f} "  # type: ignore[index]
            f"[{entry['latent_minus_linear']['lower']:+.3f}, "  # type: ignore[index]
            f"{entry['latent_minus_linear']['upper']:+.3f}]",  # type: ignore[index]
            flush=True,
        )

    _write_artefact(arguments, same, different, speakers, results)
    print(f"wrote {arguments.output}", flush=True)
    return 0


def _accumulate(
    training: FusionTrainingSet,
    evaluation: FusionTrainingSet,
    per_replicate: dict[str, list[float]],
    exaggerations: list[float],
    band_rates: list[float],
) -> None:
    """Fit the three arms on one replicate and record what they cost.

    Three arms, not two. The first version compared naive summation against a
    dependence model and reported the difference as the cost of assuming
    independence. It is not: ``NaiveIndependentFusion`` is an unweighted sum with
    no fitted parameters, so the difference confounds dependence with
    calibration. ``LinearLogisticFusion`` still assumes independence but fits
    weights, which separates the two — whatever it gains over the naive sum is
    calibration, and only what the dependence model gains over *it* is dependence.
    """
    latent = GaussianLatentFusion().fit(training)
    linear = LinearLogisticFusion().fit(training)
    report = measure_overstatement(latent, evaluation)

    per_replicate["naive_sum"].append(_fused_cllr(NaiveIndependentFusion(), evaluation))
    per_replicate["linear_logistic"].append(_fused_cllr(linear, evaluation))
    per_replicate["gaussian_latent"].append(_fused_cllr(latent, evaluation))
    exaggerations.append(float(np.median(report.exaggeration_log10)))
    band_rates.append(report.n_band_changes / max(report.n_comparisons, 1))


def _interval(values: Sequence[float]) -> dict[str, float]:
    """Mean and a 95% Monte-Carlo interval over replicates.

    A percentile interval over independent replicates, **not** a bootstrap: the
    replicates are genuinely independent draws, so their spread is the sampling
    distribution directly and resampling them would only add noise.
    """
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "lower": float(np.percentile(array, 2.5)),
        "upper": float(np.percentile(array, 97.5)),
        "n_replicates": int(array.size),
    }


def _summarise(
    correlation: float,
    per_replicate: dict[str, list[float]],
    exaggerations: list[float],
    band_rates: list[float],
) -> dict[str, object]:
    """One correlation level's replicates, reduced to intervals.

    ``latent_minus_linear`` is differenced **within** each replicate before being
    summarised, which is the same argument §7 makes for pairing: both arms saw
    the same simulated incidents and the same resampled marginal, so nearly all
    the replicate-to-replicate variation is common to them and cancels. Comparing
    two marginal intervals instead would hide a difference this one resolves.
    """
    difference = [
        latent - linear
        for latent, linear in zip(
            per_replicate["gaussian_latent"], per_replicate["linear_logistic"], strict=True
        )
    ]
    return {
        "correlation": correlation,
        "c_llr": {name: _interval(values) for name, values in per_replicate.items()},
        "latent_minus_linear": _interval(difference),
        "median_exaggeration_log10": _interval(exaggerations),
        "band_change_rate": _interval(band_rates),
    }


def _write_artefact(
    arguments: argparse.Namespace,
    same: NDArray[np.float64],
    different: NDArray[np.float64],
    speakers: NDArray[np.str_],
    results: list[dict[str, object]],
) -> None:
    artefact = {
        "cell": arguments.cell,
        "n_incidents": arguments.incidents,
        "n_replicates": arguments.replicates,
        "copula": arguments.copula,
        "degrees_of_freedom": (
            arguments.degrees_of_freedom if arguments.copula == "t" else None
        ),
        "marginal_resampled_over_speakers": not arguments.no_resample_marginal,
        "streams": [s.value for s in _STREAMS],
        "acoustic_marginal": {
            "source": str(arguments.scores),
            "n_same_source": int(same.size),
            "n_different_source": int(different.size),
            "n_speakers": int(np.unique(speakers).size),
            "same_source_mean": float(same.mean()),
            "different_source_mean": float(different.mean()),
            "units": "nats",
        },
        "assumed_strengths": {s.value: v for s, v in _ASSUMED_STRENGTH.items()},
        "results": results,
        "caveat": (
            "The acoustic marginal is measured; the behavioural and temporal "
            "marginals are assumed to share its distributional family with "
            "weaker separation. The curve describes the structure of the "
            "dependence problem, not the magnitudes any particular deployment "
            "would see. Intervals are Monte-Carlo over independent replicates "
            "and, unless disabled, include resampling of the acoustic marginal "
            "over the speakers it was estimated from. Note that "
            "calibration_scores.npz is the 42-speaker baseline evaluation of "
            "sections 4 and 5, not the 306-speaker pooled model of section 9 or "
            "the borrowed extractor of section 22, so the marginal fed to this "
            "simulation is weaker than the system's current best."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artefact, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
