"""Fit the back-end to borrowed embeddings and score the same held-out speakers.

`extract_neural.py` produces vectors and stops. This turns them into the numbers
§9 reports, on the same 102 evaluation speakers, through the same channel, with
the same trial rules and the same bias-corrected bootstrap — so that the only
thing differing between this and the i-vector system is the extractor.

That constraint is the point. §7 trained a larger i-vector model on the same
corpus and it was significantly worse in all six cells, which is the standing
reminder that a plausible improvement has to be measured rather than assumed.
An extractor swapped in alongside a changed trial rule, a changed bootstrap or a
changed split would produce a number that cannot be attributed to anything.

What is reused rather than reimplemented
-----------------------------------------
``_different_source_owner`` for trial attribution, so the symmetric hashed rule
of §18 applies here too; ``bootstrap_over_speakers`` for BCa intervals, so §14's
bias correction applies; ``compute_cllr_min`` and ``compute_eer`` unchanged.
Reimplementing any of them would be a second opportunity to get them wrong, and
the differences would be invisible in the output.

Same-source trials still cross sessions
----------------------------------------
§2's rule holds: a same-source trial is a pair from two *different* sessions,
because two recordings from one session share a microphone, a room and a day.
The session is recovered from the recording identifier, which is built as
``{speaker}-{session}-r{index}``, so no rescan of the corpus is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.experiment import _different_source_owner, _pair_key
from viflap.analysis.calibration.calibrators import LogisticCalibrator, as_reported
from viflap.analysis.calibration.metrics import compute_cllr, compute_eer
from viflap.analysis.speaker.plda import PldaConfig, PldaModel, train_plda
from viflap.analysis.speaker.transforms import IVectorTransform, fit_transform_chain
from viflap.domain.errors import CalibrationError, InsufficientDataError
from viflap.evaluation.hypotheses import H1ChannelViability


def _session_of(recording_id: str) -> str:
    """``{speaker}-{session}-r{index}`` back to ``{speaker}-{session}``."""
    return recording_id.rsplit("-", 1)[0]


@dataclass(slots=True)
class TrialSet:
    scores: NDArray[np.float64]
    labels: NDArray[np.int64]
    speakers: list[str]

    pairs: tuple[tuple[str, str], ...] = ()
    """The two recording identifiers behind each trial, sorted. Same purpose and
    same key function as ``experiment.TrialSet.pairs`` — it is what lets §22's
    comparison be joined trial for trial against the i-vector system rather than
    compared through two marginal intervals."""

    @property
    def n_same_source(self) -> int:
        return int(np.count_nonzero(self.labels == 1))

    @property
    def n_different_source(self) -> int:
        return int(np.count_nonzero(self.labels == 0))

    @property
    def n_speakers(self) -> int:
        return len(set(self.speakers))

    @property
    def kish_effective_sample_size(self) -> float:
        """Effective resampling units, given how unevenly speakers own trials.

        Reported beside the nominal count because the bootstrap treats speakers
        as exchangeable and they are not exactly so. §18 records the attribution
        defect this number was introduced to make visible.
        """
        counts = np.array(
            [self.speakers.count(speaker) for speaker in sorted(set(self.speakers))],
            dtype=float,
        )
        if counts.size == 0 or counts.sum() == 0:
            return 0.0
        return float(counts.sum() ** 2 / np.sum(counts**2))


def build_trials(
    vectors: NDArray[np.float64],
    speakers: Sequence[str],
    recording_ids: Sequence[str],
    plda: PldaModel,
    transform: IVectorTransform,
) -> TrialSet:
    """Score every admissible pair of embeddings."""
    projected = transform.apply(vectors)
    scores: list[float] = []
    labels: list[int] = []
    owners: list[str] = []
    pairs: list[tuple[str, str]] = []

    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            same_speaker = speakers[i] == speakers[j]
            if same_speaker and _session_of(recording_ids[i]) == _session_of(
                recording_ids[j]
            ):
                continue
            scores.append(plda.score(projected[i], projected[j]))
            labels.append(int(same_speaker))
            owners.append(
                speakers[i]
                if same_speaker
                else _different_source_owner(
                    speakers[i], speakers[j], recording_ids[i], recording_ids[j]
                )
            )
            pairs.append(_pair_key(recording_ids[i], recording_ids[j]))

    if not scores:
        raise InsufficientDataError("no admissible trials were formed")
    return TrialSet(
        np.array(scores), np.array(labels, dtype=np.int64), owners, tuple(pairs)
    )


@dataclass(slots=True)
class CellResult:
    condition: str
    duration_seconds: float
    n_evaluation_speakers: int
    kish_effective_speakers: float
    n_same_source: int
    n_different_source: int
    n_refused: int

    c_llr_min: float
    c_llr_min_lower: float
    c_llr_min_upper: float
    c_llr_matched: float | None
    c_llr_matched_unbounded: float | None
    calibration_loss_matched: float | None
    equal_error_rate: float
    h1_supported: bool
    h1_falsified: bool
    notes: list[str] = field(default_factory=list)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/reports/neural_embeddings.npz")
    )
    parser.add_argument(
        "--extraction-report",
        type=Path,
        default=Path("data/reports/neural_extraction.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/h1_neural.json"))
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help=(
            "Persist per-trial scores, labels, owners and recording-id pairs to "
            "this .npz. Without it the only way to pair this system against "
            "another is to rescore both, which for the i-vector side means "
            "re-degrading and re-embedding the corpus."
        ),
    )
    parser.add_argument("--lda-dimension", type=int, default=None)
    parser.add_argument("--resamples", type=int, default=2000)
    arguments = parser.parse_args(argv)

    archive = np.load(arguments.embeddings, allow_pickle=False)
    keys = set(archive.files)
    started = time.monotonic()

    train_vectors = archive["train|vectors"]
    train_speakers = [str(s) for s in archive["train|speakers"]]
    print(
        f"training back-end on {train_vectors.shape[0]} embeddings of "
        f"{train_vectors.shape[1]} dimensions from "
        f"{len(set(train_speakers))} speakers",
        flush=True,
    )

    codes = {speaker: index for index, speaker in enumerate(sorted(set(train_speakers)))}
    labels = np.array([codes[s] for s in train_speakers], dtype=np.int64)
    transform = fit_transform_chain(
        train_vectors, labels, lda_dimension=arguments.lda_dimension
    )
    plda = train_plda(transform.apply(train_vectors), labels, PldaConfig(max_iterations=40))
    print(
        f"  transform to {transform.projection.shape[1]}d, PLDA psi1 "
        f"{np.max(plda.psi):.2f}, {plda.n_iterations} iterations, "
        f"converged={plda.converged}",
        flush=True,
    )

    hypothesis = H1ChannelViability()
    cells: list[CellResult] = []
    persisted: dict[str, NDArray[np.generic]] = {}
    extraction = (
        json.loads(arguments.extraction_report.read_text(encoding="utf-8"))
        if arguments.extraction_report.exists()
        else {"cells": []}
    )
    refusals = {
        (entry["condition"], entry["duration_seconds"]): entry["n_refused"]
        for entry in extraction.get("cells", [])
        if entry.get("partition") == "evaluation"
    }

    evaluation_keys = sorted(
        {
            key.split("|", 1)[1].rsplit("|", 1)[0]
            for key in keys
            if key.startswith("evaluation|")
        }
    )
    for cell in evaluation_keys:
        condition, _, duration_text = cell.partition("@")
        duration = float(duration_text)
        notes: list[str] = []

        evaluation = build_trials(
            archive[f"evaluation|{cell}|vectors"],
            [str(s) for s in archive[f"evaluation|{cell}|speakers"]],
            [str(r) for r in archive[f"evaluation|{cell}|recordings"]],
            plda,
            transform,
        )
        # Through H1ChannelViability.run rather than bootstrapping here and
        # applying the rule afterwards. That method is what evaluate_h1.py calls,
        # so the interval, the decision and the threshold all come from the same
        # place for both extractors — and the rule cannot drift between them.
        outcome = hypothesis.run(
            evaluation.scores,
            evaluation.labels,
            evaluation.speakers,
            condition=cell,
            n_resamples=arguments.resamples,
        )
        estimate = outcome.estimate

        if arguments.scores is not None:
            persisted[f"{cell}|scores"] = evaluation.scores
            persisted[f"{cell}|labels"] = evaluation.labels
            persisted[f"{cell}|owners"] = np.array(evaluation.speakers, dtype=np.str_)
            # Stored as one string per trial rather than a 2-D array, so the
            # join key survives a round trip through ``np.savez`` without the
            # caller having to know it was ever a tuple.
            persisted[f"{cell}|pairs"] = np.array(
                ["\t".join(pair) for pair in evaluation.pairs], dtype=np.str_
            )
            # Also under the names `compare_calibrators.py` writes and
            # `measure_overstatement.py` reads. The duplication is deliberate:
            # §11 needs a *calibrated* marginal, which needs the development
            # trials to fit the calibrator on, and reproducing that key
            # convention here is what lets §11 be re-run against this system
            # without a loader that knows about two archive layouts.
            persisted[f"{cell}|eval_scores"] = evaluation.scores
            persisted[f"{cell}|eval_labels"] = evaluation.labels
            persisted[f"{cell}|eval_speakers"] = np.array(
                evaluation.speakers, dtype=np.str_
            )

        c_llr_matched = c_llr_unbounded = calibration_loss = None
        development_key = f"development|{cell}"
        if f"{development_key}|vectors" in keys:
            development = build_trials(
                archive[f"{development_key}|vectors"],
                [str(s) for s in archive[f"{development_key}|speakers"]],
                [str(r) for r in archive[f"{development_key}|recordings"]],
                plda,
                transform,
            )
            if arguments.scores is not None:
                persisted[f"{cell}|dev_scores"] = development.scores
                persisted[f"{cell}|dev_labels"] = development.labels
            try:
                calibrator = LogisticCalibrator().fit(
                    development.scores, development.labels
                )
            except (CalibrationError, InsufficientDataError) as exc:
                notes.append(f"no calibrator could be fitted: {exc.message}")
            else:
                c_llr_matched = compute_cllr(
                    as_reported(calibrator, evaluation.scores), evaluation.labels
                )
                c_llr_unbounded = compute_cllr(
                    calibrator.transform(evaluation.scores), evaluation.labels
                )
                calibration_loss = c_llr_matched - estimate.value

        cells.append(
            CellResult(
                condition=condition,
                duration_seconds=duration,
                n_evaluation_speakers=evaluation.n_speakers,
                kish_effective_speakers=round(evaluation.kish_effective_sample_size, 2),
                n_same_source=evaluation.n_same_source,
                n_different_source=evaluation.n_different_source,
                n_refused=int(refusals.get((condition, duration), 0)),
                c_llr_min=estimate.value,
                c_llr_min_lower=estimate.lower,
                c_llr_min_upper=estimate.upper,
                c_llr_matched=c_llr_matched,
                c_llr_matched_unbounded=c_llr_unbounded,
                calibration_loss_matched=calibration_loss,
                equal_error_rate=compute_eer(evaluation.scores, evaluation.labels),
                h1_supported=outcome.supported,
                h1_falsified=outcome.falsified,
                notes=notes,
            )
        )
        print(
            f"  {cell:<28} C_llr_min {estimate.value:.3f} "
            f"[{estimate.lower:.3f}, {estimate.upper:.3f}]  "
            f"EER {compute_eer(evaluation.scores, evaluation.labels) * 100:.2f}%  "
            f"ESS {evaluation.kish_effective_sample_size:.1f}/{evaluation.n_speakers}",
            flush=True,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "extractor_id": extraction.get("extractor_id", "unknown"),
                "back_end": {
                    "transform_dimension": int(transform.projection.shape[1]),
                    "plda_psi_1": float(np.max(plda.psi)),
                    "plda_n_iterations": plda.n_iterations,
                    "plda_converged": plda.converged,
                    "n_training_speakers": len(set(train_speakers)),
                },
                "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
                "cells": [asdict(cell) for cell in cells],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {arguments.output}", flush=True)
    if arguments.scores is not None:
        arguments.scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.scores, **persisted)
        print(f"wrote {arguments.scores}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
