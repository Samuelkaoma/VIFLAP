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
    "calls_per_incident": (
        "ASSUMED. An incident arrives as a set of call detail records rather "
        "than one call, because a rhythm cannot be estimated from one call and "
        "TemporalComparator refuses below three. The number is not measured."
    ),
    "operator_shift_hour": (
        "ASSUMED. Call-centre working hours are grounded; that an individual "
        "operator keeps to a preferred hour within them is not, and it is what "
        "gives the temporal stream any operator-level signal at all."
    ),
    "imei_pool_and_handset_binding": (
        "ASSUMED. The SIM pool's scale is grounded; that operators keep to a "
        "preferred handset within an operation's pool is not."
    ),
    "transcript_length": (
        "ASSUMED. Drawn at 600-1000 words so transcripts clear the idiolect "
        "floor of MIN_WORDS_IDIOLECT (500), which is itself from Ishihara "
        "(2017). A three-minute call at conversational rate is roughly 450 "
        "words, so this is call-length rather than arbitrary, but no Zambian "
        "distribution of fraud-call durations was found to fit it to."
    ),
    "cell_sites": (
        "ASSUMED. That the Lusaka operation worked from one premises is "
        "grounded, so few cells is the right shape; the identifiers and their "
        "number are invented."
    ),
    "mobile_money_agent_popularity": (
        "ASSUMED. That agent volumes are heavily skewed is the premise the "
        "transactional stream's rarity weighting rests on and is plausible, but "
        "no Zambian agent-volume distribution was found to fit it to."
    ),
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
    shift_start_hour: int = 8
    """The hour this operator tends to start work. Grounded only in that the
    operation keeps call-centre hours; the individual preference within them is
    assumed, and it is the sole reason the temporal stream carries any
    operator-level signal here rather than being purely operation-level."""

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
    imei_pool: tuple[str, ...] = ()
    """Handsets the operation drives its SIM pool through. Held at operation
    level because that is where the Lusaka case puts it — the simboxes were the
    operation's, not any operator's."""

    handset_by_imei: dict[str, str] = field(default_factory=dict)
    """Model of each handset in the pool. A handset has one model, so drawing
    the two independently would produce a corpus in which the same IMEI reports
    different models across incidents and the device stream's two components
    contradict each other."""

    cell_sites: tuple[str, ...] = ()
    """Cells the operation's traffic originates from. Small, because the Lusaka
    case was one building: location is operation-level evidence here, and the
    device stream should be seen to treat it that way."""

    agent_ids: tuple[str, ...] = ()
    agent_weights: tuple[float, ...] = ()
    """Mobile-money agents the operation cashes out through, and how its volume
    is spread over them. Heavily skewed, because a uniform spread would make
    every shared agent equally informative and the rarity weighting the
    transactional stream is built on would have nothing to weigh."""


@dataclass(slots=True)
class CallDetail:
    """One call record. An incident carries several; a rhythm needs more than one."""

    timestamp: str
    duration_seconds: float
    direction: str = "outbound"


@dataclass(slots=True)
class TransactionDetail:
    """One mobile-money movement, with the counterparties that make it evidence."""

    timestamp: str
    amount_zmw: float
    transaction_type: str
    agent_id: str
    counterparty_wallet: str


@dataclass(slots=True)
class Incident:
    """One episode, by one operator, running one operation."""

    incident_id: str
    operation_id: str
    operator_id: str
    started_at: str
    duration_seconds: float
    transcript: str
    msisdn: str
    imei: str
    handset_model: str
    cell_site: str
    mobile_money_operator: str
    calls: list[CallDetail]
    transactions: list[TransactionDetail]
    is_fraudulent: bool
    notes: list[str] = field(default_factory=list)


#: Movement types a cash-out sequence runs through.
TRANSACTION_TYPES: tuple[str, ...] = (
    "deposit",
    "transfer_out",
    "cash_out",
    "airtime_purchase",
)

HANDSET_MODELS: tuple[str, ...] = (
    "itel A56",
    "Tecno Spark 8",
    "Samsung A04",
    "Nokia 105",
    "Infinix Hot 11",
)


def _draw_operator(rng: np.random.Generator, index: int) -> Operator:
    weights = rng.dirichlet(np.full(len(DISCOURSE_MARKERS), 0.6))
    return Operator(
        operator_id=f"op{index:03d}",
        # Grounded band; the distribution within it is assumed.
        age=int(rng.integers(20, 26)),
        marker_weights=dict(zip(DISCOURSE_MARKERS, weights.tolist(), strict=True)),
        words_per_utterance=float(rng.normal(11.0, 2.5)),
        disfluency_rate=float(rng.uniform(0.02, 0.09)),
        shift_start_hour=int(rng.integers(7, 15)),
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

    models = tuple(
        rng.choice(HANDSET_MODELS, size=int(rng.integers(1, 4)), replace=False).tolist()
    )
    imei_pool = tuple(
        f"35{int(rng.integers(10**12, 10**13 - 1))}" for _ in range(int(rng.integers(3, 8)))
    )
    # One model per handset, fixed here rather than redrawn per incident. A
    # corpus in which one IMEI reports several models would let the device
    # stream's two components disagree about the same object.
    handset_by_imei = {imei: str(rng.choice(models)) for imei in imei_pool}

    n_agents = int(rng.integers(4, 10))
    # Concentration well below one: most volume through one or two agents, with
    # a long tail of rare ones. The rare ones are what the Dirichlet-multinomial
    # is supposed to reward, and a flat spread would never exercise it.
    weights = rng.dirichlet(np.full(n_agents, 0.35))

    return Operation(
        operation_id=f"opn{index:03d}",
        move_order=tuple(order),
        mobile_money_operator=chosen,
        sim_pool=tuple(
            f"26097{int(rng.integers(1000000, 9999999))}"
            for _ in range(int(rng.integers(8, 25)))
        ),
        handset_models=models,
        typical_amount_zmw=float(rng.lognormal(mean=5.6, sigma=0.7)),
        operator_ids=tuple(operators[int(i)].operator_id for i in assigned),
        imei_pool=imei_pool,
        handset_by_imei=handset_by_imei,
        cell_sites=tuple(
            f"LSK-{int(rng.integers(1000, 9999))}" for _ in range(int(rng.integers(1, 4)))
        ),
        agent_ids=tuple(f"AG{index:02d}{slot:03d}" for slot in range(n_agents)),
        agent_weights=tuple(weights.tolist()),
    )


#: What a caller says while working a move — re-asking, pressing, filling.
#: Generic across moves deliberately: the move's content is in its stem, and
#: what surrounds it is conversational padding, which is exactly where the
#: function words and disfluencies carrying the idiolect live. Keeping it out of
#: the stems is what stops the idiolect features from becoming a second copy of
#: the script.
CONTINUATIONS: tuple[str, ...] = (
    "are you there madam",
    "can you hear me properly on this line",
    "i will wait while you check the phone",
    "it is very important that you do it now",
    "no there is no problem with doing it this way",
    "yes that is correct just continue like that",
    "let me explain it again so that it is clear to you",
    "do not put the phone down please",
    "we have many other customers waiting on this same issue",
    "the system is showing me your details here on my screen",
)


def _utterance(rng: np.random.Generator, operator: Operator, move: str) -> str:
    """One turn of dialogue: the move's content, at the operator's usual length.

    ``words_per_utterance`` was drawn per operator from the beginning and never
    read, so the one idiolect parameter with an obvious surface consequence had
    none. It sets the target length here, and turns are only ever *extended* to
    reach it — truncating could cut the cue phrase that makes the move
    detectable, which would silently remove moves from the script term.
    """
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
    target = max(int(round(operator.words_per_utterance)), 6)
    while len(words) < target:
        words.extend(str(rng.choice(CONTINUATIONS)).split())

    markers = list(operator.marker_weights)
    probabilities = np.array([operator.marker_weights[m] for m in markers])
    probabilities = probabilities / probabilities.sum()

    out: list[str] = []
    for word in words:
        out.append(word)
        if rng.random() < operator.disfluency_rate:
            out.append(str(rng.choice(markers, p=probabilities)))
    return " ".join(out)


def _transcript(rng: np.random.Generator, operator: Operator, operation: Operation) -> str:
    """A whole call: every move once, then more turns until it is call-length.

    A nine-move script at one turn per move ran to about a hundred words, which
    is not a phone call — it is a summary of one. Three minutes of conversational
    speech is roughly 450 words, and
    :data:`~viflap.analysis.behaviour.profile.MIN_WORDS_IDIOLECT` is 500, so a
    corpus of hundred-word transcripts could only ever exercise the script term
    and would have made the idiolect half of this stream untestable end to end.

    Extra turns repeat moves rather than inventing new ones, which is what a
    caller working a resistant victim actually does: the move inventory is the
    operation's and does not grow because the call ran long.
    """
    turns = [_utterance(rng, operator, move) for move in operation.move_order]
    n_words = sum(len(turn.split()) for turn in turns)
    target = int(rng.integers(600, 1000))

    while n_words < target:
        for move in operation.move_order:
            turn = _utterance(rng, operator, move)
            turns.append(turn)
            n_words += len(turn.split())
            if n_words >= target:
                break
    return " . ".join(turns)


def _draw_calls(
    rng: np.random.Generator, operator: Operator, started: datetime, duration: float
) -> list[CallDetail]:
    """The incident's own call, plus the operator's activity around it.

    An incident arrives as call detail records rather than as one call, and it
    has to: ``TemporalComparator`` refuses below three calls, on the stated
    ground that a rhythm cannot be estimated from a handful of records. A
    generator emitting one call per incident can only ever exercise that
    refusal.

    The surrounding calls cluster on the operator's preferred hour, which is
    what makes hour-of-day evidence discriminate between two people working the
    same operation. Grounding for that preference is thin — see
    ``operator_shift_hour`` in :data:`PARAMETER_PROVENANCE`.
    """
    calls = [CallDetail(timestamp=started.isoformat(), duration_seconds=round(duration, 1))]
    for _ in range(int(rng.integers(3, 9))):
        hour = (operator.shift_start_hour + int(rng.integers(0, 4))) % 24
        stamp = started.replace(hour=hour, minute=int(rng.integers(0, 60))) + timedelta(
            days=int(rng.integers(-2, 3))
        )
        calls.append(
            CallDetail(
                timestamp=stamp.isoformat(),
                duration_seconds=round(float(rng.lognormal(mean=4.9, sigma=0.5)), 1),
            )
        )
    calls.sort(key=lambda call: call.timestamp)
    return calls


def _draw_transactions(
    rng: np.random.Generator, operation: Operation, started: datetime
) -> list[TransactionDetail]:
    """The cash-out sequence, through agents drawn on the operation's skew."""
    weights = np.array(operation.agent_weights)
    n_transactions = int(rng.integers(2, 6))
    transactions: list[TransactionDetail] = []
    for slot in range(n_transactions):
        agent = operation.agent_ids[int(rng.choice(len(operation.agent_ids), p=weights))]
        transactions.append(
            TransactionDetail(
                timestamp=(
                    started + timedelta(minutes=int(rng.integers(2, 90)) * (slot + 1))
                ).isoformat(),
                amount_zmw=round(
                    float(rng.lognormal(np.log(operation.typical_amount_zmw), 0.5)), 2
                ),
                transaction_type=str(rng.choice(TRANSACTION_TYPES)),
                agent_id=agent,
                # Money moves onto the operation's own numbers, so counterparty
                # wallets recur across its incidents and are evidence about the
                # operation rather than noise.
                counterparty_wallet=str(rng.choice(list(operation.sim_pool))),
            )
        )
    return transactions


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

    # Which handset each operator habitually picks up within each operation.
    # Fixed once per pairing rather than redrawn per incident: a handset that
    # changed hands every call would leave the device stream with no repeated
    # IMEI to find, which is the one observation it treats as strong.
    preferred: dict[tuple[str, str], str] = {
        (operation.operation_id, operator_id): str(rng.choice(list(operation.imei_pool)))
        for operation in operations
        for operator_id in operation.operator_ids
    }

    incidents: list[Incident] = []
    start = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)
    for operation in operations:
        for index in range(incidents_per_operation):
            operator = by_id[str(rng.choice(list(operation.operator_ids)))]
            # Call-centre working hours: the one temporal feature with any
            # grounding. The hour within them is the operator's habit and is not.
            started = (start + timedelta(days=int(rng.integers(0, 21)))).replace(
                hour=(operator.shift_start_hour + int(rng.integers(0, 4))) % 24,
                minute=int(rng.integers(0, 60)),
            )
            duration = float(rng.lognormal(mean=4.9, sigma=0.5))
            transcript = _transcript(rng, operator, operation)
            imei = preferred[(operation.operation_id, operator.operator_id)]
            if rng.random() < 0.2:
                imei = str(rng.choice(list(operation.imei_pool)))
            incidents.append(
                Incident(
                    incident_id=f"{operation.operation_id}-{index:02d}",
                    operation_id=operation.operation_id,
                    operator_id=operator.operator_id,
                    started_at=started.isoformat(),
                    duration_seconds=duration,
                    transcript=transcript,
                    msisdn=str(rng.choice(list(operation.sim_pool))),
                    imei=imei,
                    handset_model=operation.handset_by_imei[imei],
                    cell_site=str(rng.choice(list(operation.cell_sites))),
                    mobile_money_operator=operation.mobile_money_operator,
                    calls=_draw_calls(rng, operator, started, duration),
                    transactions=_draw_transactions(rng, operation, started),
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
