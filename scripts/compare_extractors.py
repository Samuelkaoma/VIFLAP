"""Pair the i-vector and borrowed-extractor systems trial for trial.

§22 reported both systems' marginal intervals side by side and said plainly that
the comparison was not paired, so the direction was not established at a stated
confidence. This closes that. It is the same instrument §7 and §9 use, applied to
the same question one stage earlier in the pipeline.

Why marginal intervals were not enough
--------------------------------------
Both systems' intervals are wide because *speakers differ from one another*, and
both systems saw the same speakers — so most of that width is shared and cancels
in a difference. §22's clean 30 s cell has [0.212, 0.383] against [0.031, 0.230],
which overlap; reading that as "indistinguishable" is precisely the error
``paired_bootstrap_over_speakers`` was written to prevent, and §7 records that
reading it the other way is an error too. Neither marginal interval answers the
question. The paired difference does.

What makes this possible, and what nearly made it impossible
------------------------------------------------------------
Both scripts now persist their per-trial scores with the **recording-id pair**
behind each trial. That key is what the join runs on. Joining on row index would
appear to work and be silently wrong: both scripts iterate ``i < j``, but over
inputs ordered differently, and at five seconds the i-vector front-end refuses
recordings the neural extractor does not — so the two trial lists differ in
length *and* in content, and an index join would align unrelated pairs.

The intersection is what gets paired
------------------------------------
Where the two systems scored different trial sets, only their intersection can be
paired, which is what ``paired_bootstrap_over_speakers`` requires of its caller.
At 30 s and 15 s the sets are identical and nothing is dropped. At 5 s the
intersection is the i-vector system's survivor subset, so the paired figure there
compares the two systems **on the recordings the i-vector front-end was willing to
embed** — an easier subset, per §6, because refusals remove the recordings
carrying least speech. That is stated per cell rather than hidden, and it is why
the 5 s rows are reported separately from the four clean ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.calibration.metrics import compute_cllr_min, compute_eer
from viflap.domain.errors import InsufficientDataError
from viflap.evaluation.hypotheses import holm_bonferroni
from viflap.evaluation.splits import paired_bootstrap_over_speakers


@dataclass(slots=True)
class PairedCell:
    """One cell's paired comparison, and how much of it could be paired."""

    condition: str
    duration_seconds: float

    n_trials_paired: int
    n_trials_baseline_only: int
    n_trials_variant_only: int
    trial_sets_identical: bool

    n_owners: int
    baseline_c_llr_min: float
    variant_c_llr_min: float
    baseline_eer: float
    variant_eer: float

    difference: float
    difference_lower: float
    difference_upper: float
    p_value: float
    difference_excludes_zero: bool
    survives_multiplicity: bool = False
    notes: list[str] = field(default_factory=list)


def load_cells(path: Path) -> dict[str, dict[str, NDArray[np.generic]]]:
    """Per-cell score vectors from one system's persisted archive."""
    stored = np.load(path, allow_pickle=False)
    cells: dict[str, dict[str, NDArray[np.generic]]] = {}
    for key in stored.files:
        cell, _, field_name = key.rpartition("|")
        cells.setdefault(cell, {})[field_name] = stored[key]
    return cells


def pair_cell(
    condition: str,
    duration: float,
    baseline: dict[str, NDArray[np.generic]],
    variant: dict[str, NDArray[np.generic]],
    n_resamples: int,
    seed: int,
) -> PairedCell:
    """Join two systems' trials on the recording-id pair and difference them."""
    notes: list[str] = []

    baseline_index = {str(key): i for i, key in enumerate(baseline["pairs"])}
    variant_index = {str(key): i for i, key in enumerate(variant["pairs"])}
    shared = sorted(set(baseline_index) & set(variant_index))
    if not shared:
        raise InsufficientDataError(
            "the two systems share no trial; the recording identifiers do not "
            "correspond, which means these archives are not of the same corpus",
            condition=condition,
        )

    left = np.array([baseline_index[key] for key in shared])
    right = np.array([variant_index[key] for key in shared])

    baseline_labels = np.asarray(baseline["labels"])[left].astype(np.int64)
    variant_labels = np.asarray(variant["labels"])[right].astype(np.int64)
    if not np.array_equal(baseline_labels, variant_labels):
        # The truth of a trial is a property of the corpus, not of the system.
        # Disagreement means the two archives disagree about who spoke, and no
        # difference computed across them would mean anything.
        raise InsufficientDataError(
            "the two systems disagree about the truth of a shared trial",
            condition=condition,
        )

    baseline_owners = [str(o) for o in np.asarray(baseline["owners"])[left]]
    variant_owners = [str(o) for o in np.asarray(variant["owners"])[right]]
    if baseline_owners != variant_owners:
        # Both derive the owner from the same hashed unordered recording pair,
        # so they must agree. If they do not, §18's attribution rule has drifted
        # between the two scripts and the bootstrap units are not comparable.
        raise InsufficientDataError(
            "the two systems attribute a shared trial to different owners; "
            "§18's ownership rule has drifted between them",
            condition=condition,
        )

    baseline_scores = np.asarray(baseline["scores"], dtype=np.float64)[left]
    variant_scores = np.asarray(variant["scores"], dtype=np.float64)[right]

    identical = len(shared) == len(baseline_index) == len(variant_index)
    if not identical:
        notes.append(
            f"trial sets differ: {len(baseline_index)} baseline and "
            f"{len(variant_index)} variant trials, {len(shared)} paired. The "
            f"paired figure is computed on the intersection, which is the "
            f"subset the baseline front-end was willing to embed."
        )

    estimate = paired_bootstrap_over_speakers(
        compute_cllr_min,
        baseline_scores,
        variant_scores,
        baseline_labels,
        baseline_owners,
        n_resamples=n_resamples,
        seed=seed,
    )
    return PairedCell(
        condition=condition,
        duration_seconds=duration,
        n_trials_paired=len(shared),
        n_trials_baseline_only=len(baseline_index) - len(shared),
        n_trials_variant_only=len(variant_index) - len(shared),
        trial_sets_identical=identical,
        n_owners=len(set(baseline_owners)),
        baseline_c_llr_min=compute_cllr_min(baseline_scores, baseline_labels),
        variant_c_llr_min=compute_cllr_min(variant_scores, baseline_labels),
        baseline_eer=compute_eer(baseline_scores, baseline_labels),
        variant_eer=compute_eer(variant_scores, baseline_labels),
        difference=estimate.value,
        difference_lower=estimate.lower,
        difference_upper=estimate.upper,
        p_value=(
            estimate.p_value_vs_zero
            if estimate.p_value_vs_zero is not None
            else float("nan")
        ),
        difference_excludes_zero=estimate.upper < 0.0 or estimate.lower > 0.0,
        notes=notes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/reports/h1_pooled_scores.npz"),
        help="Per-trial scores from evaluate_h1.py --scores (the i-vector system).",
    )
    parser.add_argument(
        "--variant",
        type=Path,
        default=Path("data/reports/h1_neural_scores.npz"),
        help="Per-trial scores from score_neural.py --scores (the borrowed extractor).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/h1_extractor_paired.json")
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    baseline = load_cells(arguments.baseline)
    variant = load_cells(arguments.variant)
    shared_cells = sorted(set(baseline) & set(variant))
    print(
        f"{len(baseline)} baseline cells, {len(variant)} variant cells, "
        f"{len(shared_cells)} in common",
        flush=True,
    )
    if not shared_cells:
        raise InsufficientDataError("the two archives share no cell")

    cells: list[PairedCell] = []
    for cell in shared_cells:
        condition, _, duration_text = cell.partition("@")
        paired = pair_cell(
            condition,
            float(duration_text),
            baseline[cell],
            variant[cell],
            arguments.resamples,
            arguments.seed,
        )
        cells.append(paired)
        marker = "identical" if paired.trial_sets_identical else "INTERSECTION"
        print(
            f"  {cell:<28} {paired.baseline_c_llr_min:.3f} -> "
            f"{paired.variant_c_llr_min:.3f}   diff {paired.difference:+.3f} "
            f"[{paired.difference_lower:+.3f}, {paired.difference_upper:+.3f}] "
            f"p={paired.p_value:.4f}  {paired.n_trials_paired} trials ({marker})",
            flush=True,
        )

    # Holm over this run's cells. They are one family tested on one dataset, and
    # reporting six uncorrected verdicts reports the chance of a spurious one as
    # a finding.
    decisions = holm_bonferroni(
        {f"{c.condition}@{c.duration_seconds:g}": c.p_value for c in cells}
    )
    for cell_result in cells:
        cell_result.survives_multiplicity = decisions.get(
            f"{cell_result.condition}@{cell_result.duration_seconds:g}", False
        )

    identical_cells = [c for c in cells if c.trial_sets_identical]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "baseline": str(arguments.baseline),
                "variant": str(arguments.variant),
                "n_resamples": arguments.resamples,
                "sign_convention": (
                    "difference = variant - baseline on C_llr_min, so negative "
                    "means the variant (borrowed extractor) discriminates better"
                ),
                "n_cells": len(cells),
                "n_cells_with_identical_trial_sets": len(identical_cells),
                "n_excluding_zero": sum(1 for c in cells if c.difference_excludes_zero),
                "n_surviving_holm": sum(1 for c in cells if c.survives_multiplicity),
                "cells": [asdict(c) for c in cells],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"\n{sum(1 for c in cells if c.difference_excludes_zero)} of {len(cells)} "
        f"exclude zero, {sum(1 for c in cells if c.survives_multiplicity)} survive "
        f"Holm; {len(identical_cells)} cells paired on identical trial sets",
        flush=True,
    )
    print(f"wrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
