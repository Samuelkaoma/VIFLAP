"""Serve the API backed by trained artefacts, so the interface shows real output.

``build_demonstration_container`` returns absence for every stream by design, so
an interface pointed at it renders an empty breakdown and cannot be judged. This
script assembles the other kind of container: a real acoustic comparator, a
calibrator fitted on development speakers, a fitted fusion model, and a
population of incidents enrolled from evaluation recordings.

What is real here and what is not
---------------------------------
Real: the acoustic likelihood ratios. They come from the trained i-vector/PLDA
system, through a calibrator fitted on speakers it never saw, applied to
recordings from speakers neither of them saw.

Not real: the incidents. A LibriSpeech reader is not a fraud suspect, and the
case references are fabricated identifiers for recordings. Nothing here supports
a claim about any person, and the enrolled population is a corpus partition
wearing incident numbers so that the interface has something structural to
render.

Not present: the other four streams. This corpus carries no transcripts,
timings, transactions or handset identifiers, and no calibrated model is
deployed for any of them. They are registered anyway, as comparators with no
fitted calibration, so that each reports its own absence with a reason instead
of vanishing from the result.

That distinction is the whole point of registering them. A stream that is simply
missing from the payload is indexed identically to one that was never attempted,
and an interface cannot tell the reader which happened. It also means the
interface is exercised on the case it will actually meet — one stream carrying
evidence and four explaining themselves — rather than on a result where every
stream happens to be present.

Startup degrades and embeds several hundred recordings and therefore takes a few
minutes before the port opens.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.corpus import materialise, scan_corpus, split_by_speaker
from scripts.experiment import build_trials, degrade_many, embed_many
from viflap.analysis.calibration.calibrators import LogisticCalibrator
from viflap.analysis.channel.degradation import DegradationCondition
from viflap.analysis.fusion.base import FusionObservation, FusionTrainingSet
from viflap.analysis.fusion.models import LinearLogisticFusion
from viflap.analysis.speaker.pipeline import SpeakerComparisonSystem
from viflap.application.ports import EvidenceBundle, IncidentRecord
from viflap.domain.evidence import EvidenceStream
from viflap.domain.governance import CaseReference
from viflap.domain.linkage import IncidentId
from viflap.infrastructure.comparators import (
    AcousticStreamComparator,
    CalibratedStreamComparator,
)
from viflap.infrastructure.fusion_provider import FittedFusionProvider
from viflap.infrastructure.memory_repositories import InMemoryStore
from viflap.interfaces.bootstrap import build_container

#: The channel every enrolled recording is passed through. One condition, not a
#: mixture: the enrolled population must be comparable with itself, and mixing
#: channels within a database silently makes some pairs easier than others.
SERVING_CONDITION = DegradationCondition(bitrate_kbps=12.20)


def _residual_spread(calibrator: LogisticCalibrator, trials) -> float:
    """Spread of calibrated log-LRs within a class, in nats.

    Sets the width of the interval the interface displays. Measured rather than
    assumed: a fixed half-width would state a precision nobody established, and
    the comparator's contract treats zero as "never measured" and falls back to
    something deliberately wide.
    """
    calibrated = calibrator.transform(trials.scores)
    same = calibrated[trials.labels == 1]
    different = calibrated[trials.labels == 0]
    spreads = [float(np.std(part)) for part in (same, different) if part.size > 1]
    return float(np.mean(spreads)) if spreads else 0.0


def prepare(
    corpus: Path,
    model_path: Path,
    *,
    n_enrolled: int,
    workers: int | None,
) -> tuple[SpeakerComparisonSystem, LogisticCalibrator, float, list, list]:
    """Train-free preparation: degrade, embed, fit the calibrator and fusion."""
    system = SpeakerComparisonSystem.load(model_path)
    # Split on plans and read only the development and evaluation audio. The
    # training partition is never touched here, and materialising it would cost
    # gigabytes this process has no use for.
    split = split_by_speaker(scan_corpus(corpus))

    development = materialise(split.development)
    evaluation = materialise(split.evaluation[:n_enrolled])

    print(
        f"degrading {len(development) + len(evaluation)} recordings through "
        f"{SERVING_CONDITION.label}",
        flush=True,
    )
    degraded_dev = degrade_many(development, [SERVING_CONDITION], workers=workers)
    degraded_eval = degrade_many(evaluation, [SERVING_CONDITION], workers=workers)

    print("embedding", flush=True)
    dev_embedded, _ = embed_many(degraded_dev, model_path, workers=workers)
    eval_embedded, refused = embed_many(degraded_eval, model_path, workers=workers)
    print(
        f"  {len(eval_embedded)} enrolled, {len(refused)} refused by the front-end",
        flush=True,
    )

    dev_trials = build_trials(dev_embedded, system, seed=11)
    calibrator = LogisticCalibrator().fit(dev_trials.scores, dev_trials.labels)
    spread = _residual_spread(calibrator, dev_trials)
    print(
        f"  calibrator fitted on {dev_trials.n_speakers} development speakers "
        f"({dev_trials.n_same_source} same-source, "
        f"{dev_trials.n_different_source} different-source trials); "
        f"residual spread {spread:.2f} nats",
        flush=True,
    )
    return system, calibrator, spread, dev_trials, eval_embedded


def fit_fusion(calibrator: LogisticCalibrator, dev_trials) -> FittedFusionProvider:
    """Fit the fusion model on development comparisons.

    Only the acoustic stream is present, so what is fitted is the single-stream
    case: an affine map from one calibrated log-LR to the fused value. That is
    the honest thing to fit against this corpus. It is not a demonstration of
    the dependence correction, which needs several streams and is hypothesis H5's
    business rather than H1's.
    """
    calibrated = calibrator.transform(dev_trials.scores)
    observations = [
        FusionObservation(
            log_lrs={EvidenceStream.ACOUSTIC: float(value)},
            is_same_source=bool(label == 1),
            group_id=speaker,
        )
        for value, label, speaker in zip(
            calibrated, dev_trials.labels, dev_trials.speakers, strict=True
        )
    ]
    model = LinearLogisticFusion().fit(FusionTrainingSet(observations))
    return FittedFusionProvider(
        model,
        stream_spreads={EvidenceStream.ACOUSTIC: _spread_of(calibrated, dev_trials)},
    )


def _spread_of(calibrated, trials) -> float:
    same = calibrated[trials.labels == 1]
    different = calibrated[trials.labels == 0]
    parts = [float(np.std(p)) for p in (same, different) if p.size > 1]
    return float(np.mean(parts)) if parts else 0.0


class _UndeployedComparator(CalibratedStreamComparator):
    """A stream with no calibrated model, which therefore reports absence.

    The base class already refuses before scoring when its calibrator is not
    fitted, so ``_score`` is unreachable. It is overridden to assert rather than
    to return something, because a subclass that quietly produced a number here
    would be the exact failure the calibration boundary exists to prevent.
    """

    def _score(self, first, second):
        raise AssertionError("an unfitted comparator must refuse before scoring")


def undeployed_comparators() -> list[CalibratedStreamComparator]:
    """One comparator per stream that has no trained model in this deployment."""
    return [
        _UndeployedComparator(stream, LogisticCalibrator(), f"undeployed-{stream.value}")
        for stream in EvidenceStream.ordered()
        if stream is not EvidenceStream.ACOUSTIC
    ]


def enrol(container, embedded: Sequence[tuple], case_reference: str) -> list[str]:
    """Enrol embedded recordings as incidents.

    One incident per recording, identified by the recording rather than by the
    speaker. Two recordings of one reader become two incidents, which is the
    structure the search is meant to reason about: the question is whether two
    incidents share an actor, and the answer is not carried in the identifiers.
    """
    reference = CaseReference.parse(case_reference, container.case_format)
    enrolled: list[str] = []

    for index, (recording, embedding) in enumerate(embedded):
        incident_id = IncidentId(f"ZP-2025-{index + 1:05d}")
        container.incidents.add(
            IncidentRecord(
                incident_id=incident_id,
                case_reference=reference,
                enrolled_at=datetime.now(UTC),
                enrolled_by="scripts.serve_trained",
                metadata={
                    "source_recording": recording.recording_id,
                    "channel_condition": recording.condition_label,
                    "codec_mode": recording.codec_mode,
                },
            )
        )
        container.evidence.store_bundle(
            EvidenceBundle(
                incident_id=incident_id,
                payloads={EvidenceStream.ACOUSTIC: embedding},
                validity=None,
                metadata={"source_recording": recording.recording_id},
            )
        )
        enrolled.append(incident_id.value)

    container.incidents.apply()
    container.evidence.apply()
    return enrolled


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/librispeech"))
    parser.add_argument("--model", type=Path, default=Path("models/acoustic.npz"))
    parser.add_argument("--audit", type=Path, default=Path("data/serving/audit.jsonl"))
    parser.add_argument("--case-reference", default="ZP-2025-01847")
    parser.add_argument("--enrolled", type=int, default=60)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--allow-origin",
        default="http://127.0.0.1:3000",
        help=(
            "Origin permitted to call this API. Needed only because the Vite "
            "development server runs on a different port; a deployment serves "
            "the built assets from the same origin and needs no such header."
        ),
    )
    arguments = parser.parse_args(argv)

    system, calibrator, spread, dev_trials, eval_embedded = prepare(
        arguments.corpus,
        arguments.model,
        n_enrolled=arguments.enrolled,
        workers=arguments.workers,
    )

    comparator = AcousticStreamComparator(system, calibrator, residual_spread=spread)
    fusion = fit_fusion(calibrator, dev_trials)
    container = build_container(
        comparators=[comparator, *undeployed_comparators()],
        fusion=fusion,
        audit_path=arguments.audit,
        store=InMemoryStore(),
    )

    enrolled = enrol(container, eval_embedded, arguments.case_reference)
    print(
        f"enrolled {len(enrolled)} incidents: {enrolled[0]} .. {enrolled[-1]}", flush=True
    )

    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware

    from viflap.interfaces.api.app import create_app

    app = create_app(container)
    # Development only, and narrow: one named origin, not a wildcard. The API
    # returns evidence about identified individuals, and a permissive CORS
    # policy on such a service is a disclosure waiting for a page to be opened.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[arguments.allow_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    print(
        f"\nmodel {system.model_id}\n"
        f"calibrator {calibrator.calibrator_id}\n"
        f"serving on http://{arguments.host}:{arguments.port}\n",
        flush=True,
    )
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
