"""Drive the whole system on the synthetic corpus, with the real comparators.

What this closes
----------------
``tests/integration/test_pipeline.py`` drives enrolment, comparison, search,
audit and retention through a ``SignatureComparator`` — a scalar per-incident
number with a genuine calibrator wrapped round it. That proves the orchestration
and the governance compose. It cannot show that the *analysis* code composes with
them, because none of it is on the path: no profile is built, no background is
estimated, no conjugate model is evaluated, and no fusion model ever sees four
streams that share a cause.

This module puts the real comparators on that path. ``synthesise_incidents.py``
supplies incidents with the documented dependence structure; this turns each into
the four payloads the real stream comparators expect, fits their backgrounds,
calibrators and fusion model on held-out operators, and hands the result to the
same use cases the integration tests drive.

**Hard boundary: no figure produced here is a result.** Every likelihood ratio
below is a property of the generative parameters in ``synthesise_incidents.py``,
most of which are marked ASSUMED in its provenance table. §11 of the results
document was withdrawn for reporting simulation output as measurement, and this
module exists downstream of that correction. Its output is an interface test with
numbers attached, not evidence about anything.

Three partitions, split by operator
-----------------------------------
Backgrounds, calibration and evaluation each get their own operators, disjoint by
identifier, for the same reason §1 splits speakers three ways: a calibrator fitted
on the incidents it later scores describes those incidents rather than the system.

Operations deliberately **straddle** the split, and that is not an oversight. An
operator is one person; an operation employs several. Splitting by operation would
make script evidence unable to cross the boundary, which is precisely the linkage
the behavioural stream exists to find. Splitting by operator keeps the delegation
pattern reachable from the evaluation side.

Which streams this reaches, and which it does not
-------------------------------------------------
Behavioural, temporal, transactional and device: all four, with real backgrounds
and real calibrators.

**The acoustic stream and therefore the validity gate are not reached.** This
corpus synthesises no speech — deliberately, since the acoustic stream is the one
with real data behind it — so no acoustic payload exists, no acoustic comparator
is registered, and ``_validity_absence`` is never consulted because it fires only
on streams for which ``is_gated_by_validity`` holds. Saying the gate is exercised
here would be false. The route to reaching it is already sketched in the corpus:
``Operator.acoustic_speaker_id`` binds an operator to a real corpus speaker, and
filling it in with LibriSpeech material and the trained i-vector system is what
puts audio on this path.

A coupling worth knowing about
------------------------------
Transcripts here are generated at 600-1000 words specifically to clear
:data:`~viflap.analysis.behaviour.profile.MIN_WORDS_IDIOLECT`, which is 500. They
were about a hundred words when this module was first written, and at that length
the behavioural stream would have run with its idiolect term withheld throughout
— producing script evidence only, and doing so silently as far as any assertion
here was concerned. The two move together: shortening the corpus or raising the
floor turns half this stream off, and the way to notice is
``idiolect_was_withheld`` in the behavioural diagnostics rather than a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.fit_behaviour_backgrounds import derive_function_words
from scripts.synthesise_incidents import Incident, generate
from viflap.analysis.behaviour.profile import (
    BehaviouralComparator,
    DisfluencyInventory,
    build_profile,
)
from viflap.analysis.calibration.calibrators import Calibrator, LogisticCalibrator
from viflap.analysis.fusion.base import FusionObservation, FusionTrainingSet
from viflap.analysis.fusion.models import GaussianLatentFusion
from viflap.analysis.patterns.conjugate import (
    BackgroundPopulation,
    NormalInverseGammaComparator,
)
from viflap.analysis.patterns.streams import (
    CallRecord,
    DeviceComparator,
    DeviceObservation,
    TemporalComparator,
    TemporalProfile,
    Transaction,
    TransactionalComparator,
    TransactionalProfile,
    circular_background,
)
from viflap.application.comparison import CompareIncidents, ComparisonRequest
from viflap.application.ingestion import IngestIncident, IngestionRequest
from viflap.application.search import SearchDatabase, SearchRequest
from viflap.domain.errors import InsufficientDataError
from viflap.domain.evidence import EvidenceStream
from viflap.domain.governance import AnalystRole, CaseReference, Principal
from viflap.domain.hypotheses import PriorOdds
from viflap.domain.linkage import IncidentId, IncidentPair
from viflap.infrastructure.audit import FileAuditLog
from viflap.infrastructure.clock import FixedClock
from viflap.infrastructure.comparators import (
    BehaviouralStreamComparator,
    DeviceStreamComparator,
    TemporalStreamComparator,
    TransactionalStreamComparator,
)
from viflap.infrastructure.fusion_provider import FittedFusionProvider
from viflap.infrastructure.memory_repositories import InMemoryUnitOfWork

#: The streams this corpus can fill. Acoustic is absent by construction — see the
#: module docstring on what that costs.
STREAMS: tuple[EvidenceStream, ...] = (
    EvidenceStream.BEHAVIOURAL,
    EvidenceStream.TEMPORAL,
    EvidenceStream.TRANSACTIONAL,
    EvidenceStream.DEVICE,
)

#: Tokens taken as the function-word inventory, by frequency in the background
#: partition. Small because these transcripts are short and built from a
#: nine-move stem inventory; a larger list would reach into hapaxes.
FUNCTION_WORD_COUNT = 60

#: von Mises concentration for the weekday background. Higher than the hour
#: default because seven bins smoothed at the 24-bin setting are very nearly
#: uniform, and a background that is uniform by accident of smoothing says
#: nothing about the population.
WEEKDAY_CONCENTRATION = 12.0

WARNING = (
    "SIMULATION. No figure in this file is a result. Every likelihood ratio is a "
    "property of the generative parameters in scripts/synthesise_incidents.py, "
    "most of which are marked ASSUMED in its provenance table. This artefact "
    "records that the pipeline runs end to end on structured incidents; it "
    "records nothing about Zambia, about fraud, or about any person."
)


# ---------------------------------------------------------------------------
# Incidents to payloads
# ---------------------------------------------------------------------------


def payloads_for(
    incident: Incident,
    function_words: frozenset[str],
    disfluencies: DisfluencyInventory,
) -> dict[EvidenceStream, Any]:
    """Reduce one incident to the payload each stream comparator expects.

    Built through the same constructors the comparators consume, rather than by
    assembling count dictionaries here. ``fit_behaviour_backgrounds.pool_counts``
    gives the reason: a payload built by one code path and a background estimated
    by another drift the moment either changes its tokenisation, and the drift is
    silent because both still look like counts.
    """
    return {
        EvidenceStream.BEHAVIOURAL: build_profile(
            incident.transcript, function_words, disfluencies
        ),
        EvidenceStream.TEMPORAL: TemporalProfile.from_records(
            [
                CallRecord(
                    timestamp=datetime.fromisoformat(call.timestamp),
                    duration_seconds=call.duration_seconds,
                    direction=call.direction,
                )
                for call in incident.calls
            ]
        ),
        EvidenceStream.TRANSACTIONAL: TransactionalProfile.from_transactions(
            [
                Transaction(
                    timestamp=datetime.fromisoformat(transaction.timestamp),
                    amount=transaction.amount_zmw,
                    transaction_type=transaction.transaction_type,
                    counterparty_wallet=transaction.counterparty_wallet,
                    agent_id=transaction.agent_id,
                )
                for transaction in incident.transactions
            ]
        ),
        # One incident, so every count is one. The Dirichlet-multinomial handles
        # singleton observations; what makes them evidence is the background's
        # opinion of how rare the category is, not how often it was seen here.
        EvidenceStream.DEVICE: DeviceObservation(
            imei_counts={incident.imei: 1},
            handset_model_counts={incident.handset_model: 1},
            cell_site_counts={incident.cell_site: 1},
        ),
    }


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Partitions:
    """Incidents split three ways by operator, with the split recorded."""

    background: tuple[Incident, ...]
    development: tuple[Incident, ...]
    evaluation: tuple[Incident, ...]
    operators: Mapping[str, tuple[str, ...]]

    def summary(self) -> dict[str, int]:
        return {
            "background_incidents": len(self.background),
            "development_incidents": len(self.development),
            "evaluation_incidents": len(self.evaluation),
            "background_operators": len(self.operators["background"]),
            "development_operators": len(self.operators["development"]),
            "evaluation_operators": len(self.operators["evaluation"]),
        }


def split_by_operator(incidents: Sequence[Incident]) -> Partitions:
    """Assign each operator to one partition, balancing incident counts.

    Greedy largest-first rather than a third of the sorted identifiers. Operators
    are assigned to operations at random and appear in wildly different numbers of
    incidents — 2 to 17 at the default settings — so slicing a sorted list can put
    most of the corpus in one partition and leave another with too few same-source
    pairs to fit a calibrator at all.
    """
    counts = Counter(incident.operator_id for incident in incidents)
    names = ("background", "development", "evaluation")
    assigned: dict[str, list[str]] = {name: [] for name in names}
    load = dict.fromkeys(names, 0)

    # Ties broken on the identifier so the split is a function of the corpus and
    # not of dictionary ordering.
    for operator, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        target = min(names, key=lambda name: (load[name], name))
        assigned[target].append(operator)
        load[target] += count

    membership = {
        operator: name for name, operators in assigned.items() for operator in operators
    }
    grouped: dict[str, list[Incident]] = {name: [] for name in names}
    for incident in incidents:
        grouped[membership[incident.operator_id]].append(incident)

    return Partitions(
        background=tuple(grouped["background"]),
        development=tuple(grouped["development"]),
        evaluation=tuple(grouped["evaluation"]),
        operators={name: tuple(sorted(assigned[name])) for name in names},
    )


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawComparators:
    """The analysis-layer comparators, before calibration wraps them."""

    behavioural: BehaviouralComparator
    temporal: TemporalComparator
    transactional: TransactionalComparator
    device: DeviceComparator
    function_words: frozenset[str]
    disfluencies: DisfluencyInventory

    def by_stream(self) -> dict[EvidenceStream, Any]:
        return {
            EvidenceStream.BEHAVIOURAL: self.behavioural,
            EvidenceStream.TEMPORAL: self.temporal,
            EvidenceStream.TRANSACTIONAL: self.transactional,
            EvidenceStream.DEVICE: self.device,
        }


def fit_raw_comparators(background: Sequence[Incident]) -> RawComparators:
    """Estimate every background this system needs from the background partition.

    Each background is the denominator of its stream's likelihood ratio: how
    surprising is it that two incidents share this feature, given how often it
    occurs in the population? Estimating them from incidents no calibrator or
    evaluation will see is what keeps that denominator a statement about the
    population rather than about the pair being scored.
    """
    if not background:
        raise InsufficientDataError("the background partition is empty")

    transcripts = [incident.transcript for incident in background]
    function_words = frozenset(derive_function_words(transcripts, FUNCTION_WORD_COUNT))
    disfluencies = DisfluencyInventory()

    pooled: dict[str, Counter[str]] = {
        key: Counter()
        for key in (
            "function_words",
            "disfluencies",
            "moves",
            "ngrams",
            "hours",
            "weekdays",
            "agents",
            "counterparties",
            "types",
            "imeis",
            "handsets",
            "cell_sites",
        )
    }
    log_durations: list[float] = []
    log_amounts: list[float] = []

    for incident in background:
        payloads = payloads_for(incident, function_words, disfluencies)

        behaviour = payloads[EvidenceStream.BEHAVIOURAL]
        pooled["function_words"].update(behaviour.function_word_counts)
        pooled["disfluencies"].update(behaviour.disfluency_counts)
        pooled["moves"].update(behaviour.move_counts)
        pooled["ngrams"].update(behaviour.character_ngram_counts)

        temporal = payloads[EvidenceStream.TEMPORAL]
        pooled["hours"].update(temporal.hour_counts)
        pooled["weekdays"].update(temporal.weekday_counts)
        log_durations.extend(temporal.log_durations.tolist())

        transactional = payloads[EvidenceStream.TRANSACTIONAL]
        pooled["agents"].update(transactional.agent_counts)
        pooled["counterparties"].update(transactional.counterparty_counts)
        pooled["types"].update(transactional.type_counts)
        log_amounts.extend(transactional.log_amounts.tolist())

        device = payloads[EvidenceStream.DEVICE]
        pooled["imeis"].update(device.imei_counts)
        pooled["handsets"].update(device.handset_model_counts)
        pooled["cell_sites"].update(device.cell_site_counts)

    def population(key: str, what: str) -> BackgroundPopulation:
        if not pooled[key]:
            raise InsufficientDataError(
                f"the background partition produced no {what}; a background of "
                f"nothing cannot be normalised, and substituting a uniform one "
                f"would put every likelihood ratio for this component on a "
                f"distribution the corpus never supported",
                component=key,
            )
        return BackgroundPopulation.from_counts(pooled[key], f"synthetic corpus: {what}")

    return RawComparators(
        behavioural=BehaviouralComparator(
            function_word_background=population("function_words", "function words"),
            disfluency_background=population("disfluencies", "disfluencies"),
            move_background=population("moves", "script moves"),
            ngram_background=population("ngrams", "character n-grams"),
            # No language identifier is supplied, so no span is ever labelled and
            # every switch count is empty. Passing a switch background anyway
            # would advertise a component that can never fire.
            switch_background=None,
        ),
        temporal=TemporalComparator(
            hour_background=circular_background(
                pooled["hours"], 24, description="synthetic corpus: hour of day"
            ),
            weekday_background=circular_background(
                pooled["weekdays"],
                7,
                concentration=WEEKDAY_CONCENTRATION,
                description="synthetic corpus: day of week",
            ),
            duration_model=NormalInverseGammaComparator.from_background(
                np.array(log_durations, dtype=np.float64),
                description="synthetic corpus: log call duration",
            ),
        ),
        transactional=TransactionalComparator(
            agent_background=population("agents", "mobile-money agents"),
            counterparty_background=population("counterparties", "counterparty wallets"),
            type_background=population("types", "transaction types"),
            amount_model=NormalInverseGammaComparator.from_background(
                np.array(log_amounts, dtype=np.float64),
                description="synthetic corpus: log transaction amount",
            ),
        ),
        device=DeviceComparator(
            imei_background=population("imeis", "handset identifiers"),
            handset_background=population("handsets", "handset models"),
            cell_site_background=population("cell_sites", "cell sites"),
        ),
        function_words=function_words,
        disfluencies=disfluencies,
    )


# ---------------------------------------------------------------------------
# Calibration and fusion
# ---------------------------------------------------------------------------


def _pairs(incidents: Sequence[Incident]) -> Iterator[tuple[Incident, Incident]]:
    for index, first in enumerate(incidents):
        for second in incidents[index + 1 :]:
            yield first, second


def raw_scores(
    raw: RawComparators,
    first: Mapping[EvidenceStream, Any],
    second: Mapping[EvidenceStream, Any],
) -> dict[EvidenceStream, float]:
    """Uncalibrated log-LR per stream, skipping streams that decline the pair.

    A stream that raises is omitted rather than recorded as zero. Zero is a valid
    log-likelihood ratio meaning "uninformative", and the fusion layer's own
    design note says using it as a missing marker makes absent evidence and
    neutral evidence indistinguishable everywhere downstream.
    """
    scored: dict[EvidenceStream, float] = {}
    for stream, comparator in raw.by_stream().items():
        try:
            outcome = comparator.score(first[stream], second[stream])
        except InsufficientDataError:
            continue
        scored[stream] = (
            outcome.total_log_lr
            if stream is EvidenceStream.BEHAVIOURAL
            else float(outcome)
        )
    return scored


@dataclass(frozen=True, slots=True)
class FittedBackEnd:
    """Calibrators, their measured spreads, and the fusion model over them."""

    calibrators: Mapping[EvidenceStream, Calibrator]
    spreads: Mapping[EvidenceStream, float]
    fusion: GaussianLatentFusion
    n_development_pairs: int
    n_same_operator_pairs: int


def fit_back_end(
    raw: RawComparators, development: Sequence[Incident]
) -> FittedBackEnd:
    """Fit one calibrator per stream and a fusion model over their output.

    Same-source means **same operator**, not same operation. That choice is the
    one that makes this corpus worth running: three of the four streams here carry
    the operation as much as the person — the handset pool, the cell sites, the
    agents and the script are all the operation's property — so calibrating
    against operator identity is what forces the delegation confound through the
    machinery instead of around it.
    """
    payloads = {
        incident.incident_id: payloads_for(incident, raw.function_words, raw.disfluencies)
        for incident in development
    }

    per_stream: dict[EvidenceStream, list[tuple[float, int]]] = {
        stream: [] for stream in STREAMS
    }
    observations: list[FusionObservation] = []
    raw_by_pair: list[tuple[dict[EvidenceStream, float], bool, str]] = []
    n_same = 0

    for first, second in _pairs(development):
        same = first.operator_id == second.operator_id
        n_same += int(same)
        scored = raw_scores(
            raw, payloads[first.incident_id], payloads[second.incident_id]
        )
        for stream, value in scored.items():
            per_stream[stream].append((value, int(same)))
        owner = "|".join(sorted((first.operator_id, second.operator_id)))
        raw_by_pair.append((scored, same, owner))

    calibrators: dict[EvidenceStream, Calibrator] = {}
    spreads: dict[EvidenceStream, float] = {}
    for stream, samples in per_stream.items():
        if not samples:
            continue
        scores = np.array([value for value, _ in samples], dtype=np.float64)
        labels = np.array([label for _, label in samples], dtype=np.int64)
        calibrator = LogisticCalibrator().fit(scores, labels)
        calibrators[stream] = calibrator
        # Spread of the calibrated log-LR on same-source development trials, in
        # nats, which is what CalibratedStreamComparator documents its interval
        # as being built from. Measured rather than defaulted: the default widens
        # the interval to the full empirical bound, honestly but uselessly.
        same_source = scores[labels == 1]
        spreads[stream] = float(
            np.std([calibrator.calibrate(float(s)).log_lr.value for s in same_source])
        )

    for scored, same, owner in raw_by_pair:
        calibrated = {
            stream: calibrators[stream].calibrate(value).log_lr.value
            for stream, value in scored.items()
            if stream in calibrators
        }
        if calibrated:
            observations.append(
                FusionObservation(calibrated, is_same_source=same, group_id=owner)
            )

    fusion = GaussianLatentFusion().fit(FusionTrainingSet(observations))
    return FittedBackEnd(
        calibrators=calibrators,
        spreads=spreads,
        fusion=fusion,
        n_development_pairs=len(raw_by_pair),
        n_same_operator_pairs=n_same,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssembledSystem:
    """Everything needed to run the use cases, plus what it was built from."""

    ingest: IngestIncident
    compare: CompareIncidents
    search: SearchDatabase
    audit: FileAuditLog
    unit_of_work: InMemoryUnitOfWork
    clock: FixedClock
    raw: RawComparators
    back_end: FittedBackEnd
    partitions: Partitions


def assemble(incidents: Sequence[Incident], audit_path: Path) -> AssembledSystem:
    """Fit everything and wire the use cases, without enrolling anything yet.

    Takes no seed. Every stochastic choice belongs to corpus generation, which
    happened before this was called; fitting backgrounds, calibrators and the
    fusion model is deterministic given the incidents, and offering a seed here
    would suggest otherwise.
    """
    partitions = split_by_operator(incidents)
    raw = fit_raw_comparators(partitions.background)
    back_end = fit_back_end(raw, partitions.development)

    comparators = [
        BehaviouralStreamComparator(
            raw.behavioural,
            back_end.calibrators[EvidenceStream.BEHAVIOURAL],
            residual_spread=back_end.spreads[EvidenceStream.BEHAVIOURAL],
        ),
        TemporalStreamComparator(
            raw.temporal,
            back_end.calibrators[EvidenceStream.TEMPORAL],
            residual_spread=back_end.spreads[EvidenceStream.TEMPORAL],
        ),
        TransactionalStreamComparator(
            raw.transactional,
            back_end.calibrators[EvidenceStream.TRANSACTIONAL],
            residual_spread=back_end.spreads[EvidenceStream.TRANSACTIONAL],
        ),
        DeviceStreamComparator(
            raw.device,
            back_end.calibrators[EvidenceStream.DEVICE],
            residual_spread=back_end.spreads[EvidenceStream.DEVICE],
        ),
    ]

    clock = FixedClock(datetime(2026, 6, 1, 9, 0, tzinfo=UTC))
    unit_of_work = InMemoryUnitOfWork()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = FileAuditLog(audit_path)

    compare = CompareIncidents(
        comparators,
        FittedFusionProvider(back_end.fusion, stream_spreads=dict(back_end.spreads)),
        audit,
        clock,
    )
    return AssembledSystem(
        ingest=IngestIncident(
            extractors={
                stream: _extractor_for(stream) for stream in STREAMS
            },
            unit_of_work=unit_of_work,
            audit=audit,
            clock=clock,
        ),
        compare=compare,
        search=SearchDatabase(
            compare, unit_of_work.incidents, unit_of_work.evidence, audit, clock
        ),
        audit=audit,
        unit_of_work=unit_of_work,
        clock=clock,
        raw=raw,
        back_end=back_end,
        partitions=partitions,
    )


def _extractor_for(stream: EvidenceStream) -> Any:
    """Per-stream extractor reading the payload prepared on the request.

    The payloads are computed once per incident and carried on the ingestion
    request's metadata rather than recomputed per stream: ``build_profile``
    tokenises the whole transcript, and doing that four times per incident is
    three times more work for an identical answer.
    """

    def extract(request: IngestionRequest) -> Any:
        return request.metadata["payloads"][stream]

    return extract


def enrol(
    system: AssembledSystem,
    incidents: Sequence[Incident],
    case_reference: CaseReference,
    officer: Principal,
) -> list[IncidentId]:
    """Enrol a partition through the real ingestion use case."""
    enrolled: list[IncidentId] = []
    for incident in incidents:
        payloads = payloads_for(
            incident, system.raw.function_words, system.raw.disfluencies
        )
        incident_id = IncidentId(f"ZP-2026-{incident.incident_id}")
        system.ingest.execute(
            IngestionRequest(
                incident_id=incident_id,
                case_reference=case_reference,
                submitted_by=officer,
                metadata={
                    "payloads": payloads,
                    "operator_id": incident.operator_id,
                    "operation_id": incident.operation_id,
                },
            )
        )
        enrolled.append(incident_id)
    return enrolled


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", type=int, default=24)
    parser.add_argument("--operations", type=int, default=10)
    parser.add_argument("--incidents-per-operation", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20250601)
    parser.add_argument(
        "--audit", type=Path, default=Path("data/reports/synthetic_pipeline_audit.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/synthetic_pipeline.json")
    )
    arguments = parser.parse_args(argv)

    _, _, incidents = generate(
        arguments.operators,
        arguments.operations,
        arguments.incidents_per_operation,
        arguments.seed,
    )
    print(f"{len(incidents)} synthetic incidents", flush=True)

    if arguments.audit.exists():
        # The audit log is append-only and hash-chained, so a rerun appending to
        # the previous run's chain would verify happily and report a session that
        # never happened.
        arguments.audit.unlink()

    system = assemble(incidents, arguments.audit)
    print(f"  split: {json.dumps(system.partitions.summary())}", flush=True)
    print(
        f"  back-end: {system.back_end.n_development_pairs} development pairs, "
        f"{system.back_end.n_same_operator_pairs} same-operator",
        flush=True,
    )
    for stream in STREAMS:
        print(
            f"    {stream.value:<14} spread {system.back_end.spreads[stream]:.3f} nats",
            flush=True,
        )

    case_reference = CaseReference.parse("ZP-2026-00001")
    officer = Principal("enrol-sim", frozenset({AnalystRole.ENROLMENT_OFFICER}))
    investigator = Principal("inv-sim", frozenset({AnalystRole.INVESTIGATOR}))

    enrolled = enrol(system, system.partitions.evaluation, case_reference, officer)
    print(f"  enrolled {len(enrolled)} evaluation incidents", flush=True)

    operator_of = {
        f"ZP-2026-{incident.incident_id}": incident.operator_id
        for incident in system.partitions.evaluation
    }
    operation_of = {
        f"ZP-2026-{incident.incident_id}": incident.operation_id
        for incident in system.partitions.evaluation
    }

    # Three groups, because the interesting one is the middle. Same operation with
    # a different operator is the delegation case the behavioural stream's whole
    # decomposition exists for, and a run that only contrasted "same person" with
    # "unrelated" would never put a number on it.
    groups: dict[str, list[float]] = {
        "same_operator": [],
        "same_operation_different_operator": [],
        "different_operation": [],
    }
    delegation_flags = dict.fromkeys(groups, 0)
    store = system.unit_of_work.store.evidence
    for first, second in _pairs(enrolled):
        result = system.compare.execute(
            ComparisonRequest(
                pair=IncidentPair(first, second, case_reference),
                prior=PriorOdds.uniform_over_database(len(enrolled)),
                requested_by=investigator,
            ),
            store[first.value],
            store[second.value],
        )
        if operator_of[first.value] == operator_of[second.value]:
            name = "same_operator"
        elif operation_of[first.value] == operation_of[second.value]:
            name = "same_operation_different_operator"
        else:
            name = "different_operation"
        groups[name].append(result.fused_log_lr.log10)

        behavioural = result.outcomes[EvidenceStream.BEHAVIOURAL]
        diagnostics = getattr(behavioural, "diagnostics", {}) or {}
        delegation_flags[name] += int(diagnostics.get("suggests_delegation", 0.0) > 0.0)

    results = system.search.execute(
        SearchRequest(enrolled[0], case_reference, investigator, max_results=5)
    )
    verification = system.audit.verify()

    report = {
        "warning": WARNING,
        "seed": arguments.seed,
        "corpus": {
            "n_incidents": len(incidents),
            "n_operators": arguments.operators,
            "n_operations": arguments.operations,
        },
        "split": system.partitions.summary(),
        "back_end": {
            "n_development_pairs": system.back_end.n_development_pairs,
            "n_same_operator_pairs": system.back_end.n_same_operator_pairs,
            "fusion_model_id": system.back_end.fusion.model_id,
            "streams_calibrated": sorted(
                stream.value for stream in system.back_end.calibrators
            ),
            "residual_spread_nats": {
                stream.value: round(value, 4)
                for stream, value in system.back_end.spreads.items()
            },
        },
        "comparisons": {
            name: {
                "n_pairs": len(values),
                "median_fused_log10_lr": (
                    round(float(np.median(values)), 4) if values else None
                ),
                "n_flagged_as_delegation": delegation_flags[name],
            }
            for name, values in groups.items()
        },
        "search": {
            "n_results": len(results.results),
            "prior_population_size": results.prior.population_size,
            "prior_supplied_by": results.prior.supplied_by,
        },
        "audit": {
            "chain_intact": verification.is_intact,
            "n_entries": verification.n_entries,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, values in groups.items():
        median = f"{float(np.median(values)):+.2f}" if values else "n/a"
        print(f"  {name:<38} {len(values):>5} pairs, median log10 LR {median}")
    print(
        f"  audit chain intact={verification.is_intact} over "
        f"{verification.n_entries} entries",
        flush=True,
    )
    print(f"\nwrote {arguments.output}", flush=True)
    print(WARNING, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
