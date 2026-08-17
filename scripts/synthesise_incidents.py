"""A synthetic Zambian incident corpus, for testing the system rather than measuring it.

What this is for, and the boundary that governs it
---------------------------------------------------
Three of VIFLAP's five evidence streams — temporal, transactional and device —
have never seen data resembling their deployment population. They are
implemented and unit-tested against invented fixtures, and the end-to-end tests
drive the pipeline with a scalar test double. That proves the machinery
composes; it proves nothing about evidence.

This generator closes the first gap and **not the second**. It exists so that
the whole system can be run end to end on incidents shaped like the real thing:
so that fusion meets realistic dependence, the validity gate meets a real
refusal, the audit chain meets a real session, and interface defects surface
before a deployment finds them.

**No figure derived from this data is a result.** A ``C_llr`` computed here is a
property of these generative parameters and nothing else. §11 of the results
document made precisely that error — it fed a fusion model unbounded calibrator
output and called the result "the cost of assuming independence", when most of
what it measured was the cost of not fitting anything — and had to be withdrawn
after review. Anything this module produces is labelled simulation, or it should
not be reported.

The dependence structure is the documented one
-----------------------------------------------
``EvidenceStream`` states the streams "are *not* conditionally independent of one
another — the same operator running the same operation is the common cause of
all of them". So incidents are generated that way rather than by attaching a
correlation knob to independent draws: an **operation** supplies the script, the
handset pool and the transaction profile; an **operator** supplies the voice and
the idiolect; and an incident is one operator running one operation.

That structure is not invented for convenience. It is what the 2024 Lusaka call
centre case looked like: a single operation employing Zambian operators aged
roughly 20 to 25, reading scripted dialogues, behind a pool of more than 13,000
SIM cards driven through simboxes. Delegation — one operation, many operators —
is the pattern ``BehaviouralScore.suggests_shared_operation_not_speaker`` was
written to detect, and it is documented in Zambian casework rather than
hypothesised.

Which parameters are grounded, and which are not
-------------------------------------------------
Stated explicitly, because a generator whose provenance is unrecorded invites
its outputs to be quoted as though they were measurements.

**Grounded in published sources** (see ``PARAMETER_PROVENANCE``):
mobile money penetration; the three mobile money operators and the dominance of
two of them; the suspected-fraud rate among digital transaction attempts; the
operator age band, the scripted-dialogue working method and the scale of the SIM
pool, all from the Lusaka case.

**Assumed, and marked as such**: transaction amount distributions, call
durations, inter-call timing, code-switching rates, and every idiolect
parameter. No published Zambian figures for these were found. They are chosen to
be plausible and are *not* evidence about Zambia; a result that depends on one
of them is a result about this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

#: Where each generative parameter comes from. Carried in the output so that a
#: figure derived from this corpus arrives with its provenance attached rather
#: than needing to be traced back to a docstring.
PARAMETER_PROVENANCE: dict[str, str] = {
    "mobile_money_operators": (
        "grounded: Airtel Money, MTN MoMo and Zamtel Kwacha are Zambia's three "
        "mobile money providers, with Airtel and MTN dominant."
    ),
    "mobile_money_penetration": (
        "grounded: FinScope 2025 reports financial inclusion at 80.1% and "
        "mobile money usage at 76.2% of the population."
    ),
    "background_fraud_rate": (
        "grounded: TransUnion reports 1.1% of Zambian digital transaction "
        "attempts suspected fraudulent in 2025, 1.9% in financial services."
    ),
    "operator_age_band": (
        "grounded: the 2024 Lusaka call centre case recruited Zambian operators "
        "aged roughly 20-25."
    ),
    "scripted_dialogue": (
        "grounded: operators in that case worked from scripted dialogues, which "
        "is the delegation pattern the behavioural stream models."
    ),
    "sim_pool_size": (
        "grounded in scale only: over 13,000 SIM cards were seized in that case, "
        "driven through simboxes. The per-operation pool here is a small "
        "fraction of that and its size is assumed."
    ),
    "transaction_amounts_zmw": "ASSUMED. No published Zambian distribution found.",
    "call_durations": "ASSUMED. No published Zambian distribution found.",
    "inter_call_timing": "ASSUMED, beyond call-centre working hours.",
    "code_switching_rate": "ASSUMED. Bemba/Nyanja/English mixing is real; the rate is not measured.",
    "idiolect_parameters": "ASSUMED throughout. Nothing here is evidence about any person.",
}

MOBILE_MONEY_OPERATORS: tuple[tuple[str, float], ...] = (
    ("airtel_money", 0.45),
    ("mtn_momo", 0.42),
    ("zamtel_kwacha", 0.13),
)

#: Rhetorical moves an advance-fee or agent-impersonation pretext runs through.
#: The inventory is the operation's property, not the operator's — which is the
#: whole point of separating script from idiolect.
SCRIPT_MOVES: tuple[str, ...] = (
    "greet",
    "establish_authority",
    "state_problem",
    "create_urgency",
    "request_identifier",
    "request_pin",
    "instruct_transfer",
    "reassure",
    "close",
)

#: Filler and discourse markers. Bemba and Nyanja markers alongside English ones
#: because Zambian conversational speech mixes them; which markers a given
#: operator favours is an idiolect property and is drawn per operator.
DISCOURSE_MARKERS: tuple[str, ...] = (
    "eeh",
    "ati",
    "sha",
    "iwe",
    "nomba",
    "bwanji",
    "you know",
    "so",
    "actually",
    "please",
)


@dataclass(slots=True)
class Operator:
    """A person. Supplies the voice and the idiolect, never the script."""

    operator_id: str
    age: int
    marker_weights: dict[str, float]
    words_per_utterance: float
    disfluency_rate: float
    acoustic_speaker_id: str | None = None
    """Which real corpus speaker supplies this operator's audio, when the
    acoustic stream is filled from real recordings rather than simulated. Left
    unset here: this module does not synthesise speech, because the acoustic
    stream is the one stream that already has real data behind it."""


@dataclass(slots=True)
class Operation:
    """A campaign. Supplies the script, the handset pool and the money profile."""

    operation_id: str
    move_order: tuple[str, ...]
    mobile_money_operator: str
    sim_pool: tuple[str, ...]
    handset_models: tuple[str, ...]
    typical_amount_zmw: float
    operator_ids: tuple[str, ...]


@dataclass(slots=True)
class Incident:
    """One call, by one operator, running one operation."""

    incident_id: str
    operation_id: str
    operator_id: str
    started_at: str
    duration_seconds: float
    transcript: str
    msisdn: str
    imei: str
    handset_model: str
    mobile_money_operator: str
    transactions_zmw: list[float]
    is_fraudulent: bool
    notes: list[str] = field(default_factory=list)


def _draw_operator(rng: np.random.Generator, index: int) -> Operator:
    weights = rng.dirichlet(np.full(len(DISCOURSE_MARKERS), 0.6))
    return Operator(
        operator_id=f"op{index:03d}",
        # Grounded band; the distribution within it is assumed.
        age=int(rng.integers(20, 26)),
        marker_weights=dict(zip(DISCOURSE_MARKERS, weights.tolist(), strict=True)),
        words_per_utterance=float(rng.normal(11.0, 2.5)),
        disfluency_rate=float(rng.uniform(0.02, 0.09)),
    )


def _draw_operation(
    rng: np.random.Generator, index: int, operators: Sequence[Operator]
) -> Operation:
    # An operation's script is a stable ordering of the move inventory with a
    # little local variation, which is what makes the sequence evidence in
    # _sequence_log_lr informative rather than constant.
    order = list(SCRIPT_MOVES)
    if rng.random() < 0.5:
        cut = int(rng.integers(2, len(order) - 2))
        order[cut], order[cut + 1] = order[cut + 1], order[cut]

    labels = [name for name, _ in MOBILE_MONEY_OPERATORS]
    shares = np.array([share for _, share in MOBILE_MONEY_OPERATORS])
    chosen = str(rng.choice(labels, p=shares / shares.sum()))

    # Delegation: several operators share one operation. This is the structure
    # the Lusaka case documents and the pattern the behavioural stream's
    # script/idiolect split exists to separate.
    n_operators = int(rng.integers(2, min(5, len(operators)) + 1))
    assigned = rng.choice(len(operators), size=n_operators, replace=False)

    return Operation(
        operation_id=f"opn{index:03d}",
        move_order=tuple(order),
        mobile_money_operator=chosen,
        sim_pool=tuple(
            f"26097{int(rng.integers(1000000, 9999999))}"
            for _ in range(int(rng.integers(8, 25)))
        ),
        handset_models=tuple(
            rng.choice(
                ["itel A56", "Tecno Spark 8", "Samsung A04", "Nokia 105", "Infinix Hot 11"],
                size=int(rng.integers(1, 4)),
                replace=False,
            ).tolist()
        ),
        typical_amount_zmw=float(rng.lognormal(mean=5.6, sigma=0.7)),
        operator_ids=tuple(operators[int(i)].operator_id for i in assigned),
    )


def _utterance(rng: np.random.Generator, operator: Operator, move: str) -> str:
    """One line of dialogue: the move's content in the operator's habits."""
    stems = {
        "greet": "good morning madam this is calling from",
        "establish_authority": "i am the agent handling your account today",
        "state_problem": "there is a problem with your wallet it has been suspended",
        "create_urgency": "if we do not fix it now the money will be reversed",
        "request_identifier": "can you confirm the number registered on the account",
        "request_pin": "now enter the code i am sending and tell me what it says",
        "instruct_transfer": "send it to the number i give you for verification",
        "reassure": "do not worry madam this is the normal procedure",
        "close": "thank you for your cooperation have a good day",
    }
    words = stems.get(move, "yes madam").split()
    markers = list(operator.marker_weights)
    probabilities = np.array([operator.marker_weights[m] for m in markers])
    probabilities = probabilities / probabilities.sum()

    out: list[str] = []
    for word in words:
        out.append(word)
        if rng.random() < operator.disfluency_rate:
            out.append(str(rng.choice(markers, p=probabilities)))
    return " ".join(out)


def generate(
    n_operators: int = 12,
    n_operations: int = 6,
    incidents_per_operation: int = 8,
    seed: int = 20250601,
) -> tuple[list[Operator], list[Operation], list[Incident]]:
    """Build a corpus of operators, operations and the incidents that link them."""
    rng = np.random.default_rng(seed)
    operators = [_draw_operator(rng, i) for i in range(n_operators)]
    by_id = {operator.operator_id: operator for operator in operators}
    operations = [_draw_operation(rng, i, operators) for i in range(n_operations)]

    incidents: list[Incident] = []
    start = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)
    for operation in operations:
        for index in range(incidents_per_operation):
            operator = by_id[str(rng.choice(list(operation.operator_ids)))]
            # Call-centre working hours: the one temporal feature with any
            # grounding. The within-day pattern is assumed.
            offset = timedelta(
                days=int(rng.integers(0, 21)),
                hours=int(rng.integers(0, 9)),
                minutes=int(rng.integers(0, 60)),
            )
            transcript = " . ".join(
                _utterance(rng, operator, move) for move in operation.move_order
            )
            incidents.append(
                Incident(
                    incident_id=f"{operation.operation_id}-{index:02d}",
                    operation_id=operation.operation_id,
                    operator_id=operator.operator_id,
                    started_at=(start + offset).isoformat(),
                    duration_seconds=float(rng.lognormal(mean=4.9, sigma=0.5)),
                    transcript=transcript,
                    msisdn=str(rng.choice(list(operation.sim_pool))),
                    imei=f"35{int(rng.integers(10**12, 10**13 - 1))}",
                    handset_model=str(rng.choice(list(operation.handset_models))),
                    mobile_money_operator=operation.mobile_money_operator,
                    transactions_zmw=[
                        round(
                            float(rng.lognormal(np.log(operation.typical_amount_zmw), 0.5)),
                            2,
                        )
                        for _ in range(int(rng.integers(1, 4)))
                    ],
                    is_fraudulent=True,
                )
            )
    return operators, operations, incidents


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", type=int, default=12)
    parser.add_argument("--operations", type=int, default=6)
    parser.add_argument("--incidents-per-operation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20250601)
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/synthetic_incidents.json")
    )
    arguments = parser.parse_args(argv)

    operators, operations, incidents = generate(
        arguments.operators,
        arguments.operations,
        arguments.incidents_per_operation,
        arguments.seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "warning": (
                    "SIMULATION. No figure derived from this corpus is a result; "
                    "it is a property of the generative parameters below. See "
                    "parameter_provenance for which of those are grounded in "
                    "published sources and which are assumed."
                ),
                "parameter_provenance": PARAMETER_PROVENANCE,
                "seed": arguments.seed,
                "operators": [asdict(o) for o in operators],
                "operations": [asdict(o) for o in operations],
                "incidents": [asdict(i) for i in incidents],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"{len(operators)} operators, {len(operations)} operations, "
        f"{len(incidents)} incidents -> {arguments.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
