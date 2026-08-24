"""Put real speech on the synthetic pipeline, and reach the validity gate.

`synthetic_pipeline.py` drives four streams end to end with the real comparators
and says plainly what it cannot do: the acoustic stream is absent, so no
acoustic comparator is registered, and `_validity_absence` fires only on streams
for which `is_gated_by_validity` holds. **The validity gate — the thing that
decides whether acoustic evidence reaches fusion at all — has never been
exercised end to end.** This module closes that.

The hook was left in the corpus from the start: `Operator.acoustic_speaker_id`
records which real corpus speaker supplies an operator's audio. It was never
filled in, because the synthetic corpus deliberately does not synthesise speech —
the acoustic stream is the one with real data behind it, and inventing audio for
it would have been the one place this corpus could do genuine harm.

What is real and what is not
----------------------------
**Real**: the audio, the channel, the i-vector system, the countermeasure and
the gate policy. An operator *is* a LibriSpeech speaker; two incidents by one
operator are two genuine recordings of one person through the modelled channel,
and the acoustic likelihood ratios are the system's own.

**Not real**: everything else about the incident — the transcript, the handset,
the transactions, the timings. The binding between an operator and a speaker is
arbitrary.

So the acoustic stream here is a measurement and the rest is a simulation, which
means the *fused* number is a simulation. §11's boundary governs: no figure from
this is a result. What this establishes is that the parts compose — that a real
embedding, a real countermeasure verdict and three simulated streams reach
fusion, the gate and the audit chain without anything in between silently
dropping evidence or admitting what it should refuse.

Admission on genuine speech is half the question
------------------------------------------------
§24 measured how often the gate admits real speech — 71 of 80 — and recorded
plainly that the other half was untouched: "How often this gate *correctly
excludes* a spoofed recording through the channel is a separate experiment, and
the EERs above suggest the answer is 'not reliably'." ``--spoof`` runs that
experiment. Every recording is replaced by a spoofed version of itself before
the channel, and the arms are reported beside the genuine one, because an
exclusion rate with nothing to compare it to cannot distinguish a gate that
detects synthesis from a gate that refuses everything.

**These are seen attacks.** The deployed detector trained on all four families,
so the exclusion rates this produces are the optimistic case and are not a
forecast for an attack it has not met. The honest generalisation figure remains
§24's leave-one-family-out mean unseen-attack EER of 29.37%.

Why the gate needs audio to be exercised at all
-----------------------------------------------
`CompareIncidents._validity_absence` consults the assessment only for streams
where `is_gated_by_validity` is true, which is the acoustic stream alone. With no
acoustic payload the method returns early every time, so every existing test of
the gate drives it through a hand-built `ValidityAssessment` rather than through
a countermeasure that looked at a signal. Here the verdict comes from
`ValidityGate.assess` running the trained detector over the same audio the
embedding came from.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.corpus import (
    Recording,
    RecordingPlan,
    materialise,
    scan_corpora,
    split_by_speaker,
)
from scripts.experiment import degrade_many
from scripts.synthesise_incidents import Incident, Operator
from viflap.analysis.channel.degradation import DegradationCondition
from viflap.analysis.speaker.pipeline import AcousticEmbedding, SpeakerComparisonSystem
from viflap.analysis.spoof.attacks import ATTACKS, apply_attack
from viflap.analysis.spoof.countermeasure import SpoofingCountermeasure
from viflap.analysis.spoof.gate import ValidityGate
from viflap.domain.errors import (
    ConvergenceError,
    InsufficientDataError,
    InvalidEvidenceError,
)
from viflap.domain.evidence import ValidityAssessment

#: The channel every synthetic incident is heard through. One condition, because
#: the point here is composition rather than a duration or bitrate sweep — §4 and
#: §22 do that properly on the real corpus.
CONDITION = DegradationCondition(bitrate_kbps=12.20)

#: Base seed for the attack realisations generated here. Deliberately *not*
#: `train_countermeasure.ATTACK_SEED`: a recording that appeared in both places
#: would otherwise be spoofed identically, and the detector would be scored on
#: the exact waveform it was fitted to. Different base, different realisation,
#: no possibility of that.
EVALUATION_ATTACK_SEED = 20260824

#: The label used for the unspoofed arm, so genuine and spoofed results sit in
#: one table under the same keys.
GENUINE = "none"


@dataclass(frozen=True, slots=True)
class AcousticEvidence:
    """One incident's audio, reduced to what the pipeline consumes."""

    embedding: AcousticEmbedding
    validity: ValidityAssessment


def bind_operators(
    operators: Sequence[Operator], plans: Sequence[RecordingPlan]
) -> dict[str, list[RecordingPlan]]:
    """Give each operator a real speaker, and their recordings.

    One speaker per operator, taken in sorted order so the binding is a function
    of the corpus rather than of iteration order. Speakers with fewer recordings
    than the operator has incidents are still used — an operator simply reuses
    sessions, which is realistic and is handled by the trial rules downstream.

    Refuses rather than truncates when there are too few speakers. Silently
    binding two operators to one speaker would make two different people
    acoustically identical, which is the one confound this whole corpus exists to
    keep separable.
    """
    by_speaker: dict[str, list[RecordingPlan]] = {}
    for plan in plans:
        by_speaker.setdefault(plan.speaker_id, []).append(plan)

    speakers = sorted(by_speaker)
    if len(speakers) < len(operators):
        raise InsufficientDataError(
            "too few corpus speakers to give every operator a distinct one; "
            "binding two operators to one speaker would make two different "
            "people acoustically identical",
            n_operators=len(operators),
            n_speakers=len(speakers),
        )
    return {
        operator.operator_id: by_speaker[speakers[index]]
        for index, operator in enumerate(operators)
    }


def plan_for(
    incident: Incident, bound: Mapping[str, Sequence[RecordingPlan]]
) -> RecordingPlan:
    """Which recording an incident is heard as.

    Incidents of one operator are spread across that speaker's recordings by the
    incident's index within the operation, so two incidents by one operator are
    usually two *different* sessions. §2's rule is what makes that matter: a
    same-source trial has to cross sessions or it measures the room.
    """
    recordings = bound[incident.operator_id]
    index = int(incident.incident_id.rsplit("-", 1)[1])
    return recordings[index % len(recordings)]


def spoof(
    recordings: Sequence[Recording], attack_id: str
) -> tuple[list[Recording | None], list[str]]:
    """Replace each recording with a spoofed version of itself, before the channel.

    **Attack first, channel second**, which is both the order
    `train_countermeasure.degrade_examples` uses and the order the threat runs
    in: an attacker synthesises a waveform and then puts it through a telephone.
    Spoofing after the coder would model an adversary who can inject audio into
    the network, which is a different and much stronger threat than the one this
    detector is for.

    Attacks that fail are reported rather than skipped. A family that cannot be
    generated on a fifth of the recordings has an exclusion rate computed over a
    subset selected by whatever made the generation fail, and that is not the
    same quantity as an exclusion rate.
    """
    spoofed: list[Recording | None] = []
    failures: list[str] = []
    for recording in recordings:
        digest = hashlib.sha256(
            f"{recording.recording_id}:{attack_id}".encode()
        ).digest()
        rng = np.random.default_rng(
            EVALUATION_ATTACK_SEED + int.from_bytes(digest[:4], "big")
        )
        try:
            signal = apply_attack(
                attack_id, recording.signal, recording.sample_rate, rng
            )
        except (ConvergenceError, InsufficientDataError, InvalidEvidenceError) as error:
            failures.append(f"{recording.recording_id}: {type(error).__name__}")
            spoofed.append(None)
            continue
        spoofed.append(dataclasses.replace(recording, signal=signal))
    return spoofed, failures


def build_acoustic(
    incidents: Sequence[Incident],
    bound: Mapping[str, Sequence[RecordingPlan]],
    system: SpeakerComparisonSystem,
    gate: ValidityGate,
    *,
    workers: int | None = None,
    seed: int = 20250601,
    attack_id: str = GENUINE,
) -> tuple[dict[str, AcousticEvidence], list[str]]:
    """Optionally spoof, then degrade, embed and judge one recording per incident.

    The countermeasure sees the **same degraded signal** the embedding came from.
    Judging clean audio and embedding degraded audio would make the gate's
    verdict describe a recording the system never scored, which is exactly the
    mismatch a validity gate exists to prevent elsewhere.

    With ``attack_id`` set, every recording is spoofed before it reaches the
    channel, and the question the run answers inverts: not how often the gate
    admits genuine speech, but how often it **excludes** speech that is not.
    """
    plans = [plan_for(incident, bound) for incident in incidents]
    recordings = materialise(plans)
    failures: list[str] = []
    if attack_id != GENUINE:
        outcomes, failures = spoof(recordings, attack_id)
        # Kept positionally rather than by recording id: one operator's
        # incidents reuse recordings, so ids are not unique across the list and
        # matching on them would drop the wrong incidents.
        surviving = [
            (incident, recording)
            for incident, recording in zip(incidents, outcomes, strict=True)
            if recording is not None
        ]
        incidents = [incident for incident, _ in surviving]
        recordings = [recording for _, recording in surviving]
    degraded = degrade_many(recordings, [CONDITION], seed=seed, workers=workers)

    evidence: dict[str, AcousticEvidence] = {}
    for incident, item in zip(incidents, degraded, strict=True):
        try:
            embedding = system.embed(item.signal, item.sample_rate)
        except InsufficientDataError:
            # A refused recording is an incident with no acoustic stream, which
            # the comparison layer already knows how to report. Dropping the
            # incident instead would quietly select for easy audio.
            continue
        evidence[incident.incident_id] = AcousticEvidence(
            embedding=embedding,
            validity=gate.assess(item.recording_id, item.signal, item.sample_rate),
        )
    return evidence, failures


def summarise(evidence: Mapping[str, AcousticEvidence]) -> dict[str, object]:
    """What the gate did, which is the point of the exercise."""
    verdicts: dict[str, int] = {}
    for item in evidence.values():
        name = item.validity.verdict.value
        verdicts[name] = verdicts.get(name, 0) + 1
    scores = np.array(
        [item.validity.countermeasure_log_lr for item in evidence.values()]
    )
    thresholds = evidence and next(iter(evidence.values())).validity.threshold
    return {
        "n_incidents_with_audio": len(evidence),
        "verdicts": verdicts,
        # The whole distribution, not just its centre. Whether the detector's
        # scores ever *reach* the policy's admit threshold is the question this
        # run exists to answer, and a median cannot answer it.
        "countermeasure_log_lr": (
            {
                "min": round(float(scores.min()), 4),
                "median": round(float(np.median(scores)), 4),
                "max": round(float(scores.max()), 4),
            }
            if scores.size
            else None
        ),
        "admit_threshold": thresholds or None,
        "n_reaching_admit_threshold": (
            int(np.count_nonzero(scores >= thresholds)) if scores.size else 0
        ),
        "n_admitting_acoustic": sum(
            1
            for item in evidence.values()
            if item.validity.verdict.permits_acoustic_evidence
        ),
    }


def summarise_arm(
    attack_id: str, evidence: Mapping[str, AcousticEvidence], failures: Sequence[str]
) -> dict[str, object]:
    """One arm of the spoof sweep, named and with its failures carried.

    ``n_excluded`` is what a spoofed arm is asking about and ``n_admitted`` is
    what it is afraid of. Both are reported for the genuine arm too, because an
    exclusion rate is only interpretable beside the rate on speech that is real:
    a gate excluding every spoof and every genuine recording alike has not
    detected anything.
    """
    verdicts = summarise(evidence)["verdicts"]
    assert isinstance(verdicts, dict)
    return {
        "attack_id": attack_id,
        "seen_in_training": attack_id in ATTACKS,
        "n_generation_failures": len(failures),
        "generation_failures": list(failures[:10]),
        "n_excluded": verdicts.get("excluded", 0),
        "n_indeterminate": verdicts.get("indeterminate", 0),
        "n_admitted": verdicts.get("admitted", 0),
        **summarise(evidence),
    }


def load_corpus(
    roots: Sequence[Path], target_seconds: float = 30.0
) -> list[RecordingPlan]:
    """Evaluation-partition plans only.

    The acoustic model was trained on the training partition, so binding
    operators to speakers it memorised would make this exercise report the
    model's training performance. `split_by_speaker` is the same function the
    real evaluations use, so the partition is the same one.
    """
    plans = scan_corpora(roots, target_seconds=target_seconds, max_recordings_per_session=2)
    return list(split_by_speaker(plans).evaluation)


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.synthesise_incidents import generate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/acoustic_pooled.npz"))
    parser.add_argument(
        "--countermeasure", type=Path, default=Path("models/countermeasure_english.npz")
    )
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--operators", type=int, default=18)
    parser.add_argument("--operations", type=int, default=8)
    parser.add_argument("--incidents-per-operation", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20250601)
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/synthetic_acoustic.json")
    )
    parser.add_argument(
        "--spoof",
        nargs="+",
        default=None,
        choices=[*sorted(ATTACKS), "all"],
        help=(
            "also run the sweep with every recording spoofed by these attack "
            "families before the channel, and report how often the gate "
            "excludes them. The genuine arm always runs alongside, because an "
            "exclusion rate means nothing without it."
        ),
    )
    arguments = parser.parse_args(argv)

    roots = arguments.corpus or [
        Path("data/corpus/librispeech"),
        Path("data/corpus/librispeech-360"),
    ]
    operators, _, incidents = generate(
        arguments.operators,
        arguments.operations,
        arguments.incidents_per_operation,
        arguments.seed,
    )
    print(f"{len(operators)} operators, {len(incidents)} incidents", flush=True)

    plans = load_corpus(roots)
    bound = bind_operators(operators, plans)
    print(
        f"bound {len(bound)} operators to corpus speakers from "
        f"{len({p.speaker_id for p in plans})} available",
        flush=True,
    )

    system = SpeakerComparisonSystem.load(arguments.model)
    gate = ValidityGate(SpoofingCountermeasure.load(arguments.countermeasure))
    print(f"model {system.model_id}, detector {gate.detector_id}", flush=True)

    families = (
        sorted(ATTACKS)
        if arguments.spoof and "all" in arguments.spoof
        else list(arguments.spoof or [])
    )
    arms: list[dict[str, object]] = []
    evidence: dict[str, AcousticEvidence] = {}
    for attack_id in [GENUINE, *families]:
        print(f"[{attack_id}]", flush=True)
        arm_evidence, failures = build_acoustic(
            incidents,
            bound,
            system,
            gate,
            workers=arguments.workers,
            seed=arguments.seed,
            attack_id=attack_id,
        )
        if attack_id == GENUINE:
            evidence = arm_evidence
        arm = summarise_arm(attack_id, arm_evidence, failures)
        print(json.dumps(arm, indent=2), flush=True)
        arms.append(arm)

    summary = summarise(evidence)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "warning": (
                    "The acoustic stream here is a measurement and every other "
                    "stream is a simulation, so any fused figure is a "
                    "simulation. No number from this is a result."
                ),
                "model_id": system.model_id,
                "detector_id": gate.detector_id,
                "condition": CONDITION.label,
                "n_operators": len(operators),
                "n_incidents": len(incidents),
                **summary,
                # Present only when --spoof was asked for, so a run without it
                # reproduces the artefact §24 quotes rather than a superset of
                # it under the same keys.
                **({"arms": arms} if families else {}),
                **(
                    {
                        "spoof_caveat": (
                            "The deployed detector trained on all four attack "
                            "families, so these exclusion rates are the "
                            "seen-attack case and an upper bound. The "
                            "leave-one-family-out figure for this detector is "
                            "a mean unseen-attack EER of 29.37%."
                        )
                    }
                    if families
                    else {}
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
