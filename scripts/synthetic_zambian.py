"""A synthetic Zambian incident corpus, and what it may never be used for.

Three of this system's five evidence streams — temporal, transactional and
device — have never seen anything resembling their deployment population. They
are implemented, unit-tested against invented fixtures, and that is all. The
end-to-end tests in ``tests/integration`` use a scalar test double for every
comparator, so they establish that the machinery composes, the validity gate
fires and the audit chain holds. They establish nothing about evidence.

This module closes the first gap and deliberately not the second.

**What this data is for**: exercising the whole pipeline at once — five streams
through fusion, through the gate, through calibration, into a report — on inputs
shaped like the deployment population rather than like a unit test. That finds
interface defects, scaling problems, and assumptions that only break when
streams meet. It is worth doing and nothing else supplies it.

**What this data may never be used for**: any reported figure about how well
VIFLAP works. A ``C_llr`` computed on generated incidents is a property of the
generator, and reporting it as a property of the system is precisely the error
§11 of the results document made and had to withdraw after external review —
the first version fed the fusion model unbounded calibrator output and called
the result "the cost of assuming independence" when most of it was the cost of
not fitting anything. Every quantity derived from this module carries
``synthetic=True`` into its report for that reason.

Provenance is attached to every parameter
------------------------------------------
Each value below is marked ``SOURCED`` or ``ASSUMED``. That distinction is the
point of the module: a generator whose parameters are all guesses is a
simulation of the author's expectations, and one whose parameters are silently
mixed is worse, because it reads as though it were grounded throughout. Where a
figure could not be found it is assumed, said to be assumed, and given a
rationale that can be argued with.

The sourced figures come from Zambian and pan-African material:

- Zambian mobile money reached 8.6 million users by the end of 2021, against
  about 4.85 million in 2019, with roughly 47,000 agents by the end of 2018
  (Bank of Zambia, reported via development press).
- Zambian agent withdrawal fee tables for 2026 are banded at K500, K1,000,
  K2,000 and K5,000 across Airtel Money, MTN MoMo and Zamtel, which brackets the
  ordinary transaction range.
- Between 58% and 72% of African mobile money fraud is social engineering, of
  which SIM-swap accounts for about 43% and agent-assisted fraud about 38%; the
  dominant script is impersonation of network staff or police.
- Lusaka and the Copperbelt are strongly multilingual, and code-switching
  between Bemba, Nyanja and English is ordinary rather than exceptional; "Town
  Bemba" is a documented urban hybrid.

The assumed figures — call durations, inter-call intervals, handset model
distribution, operators per group — could not be sourced for Zambia and are
reasoned from the sourced ones or from the mechanics of the fraud itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

import numpy as np

from viflap.analysis.patterns.streams import (
    CallRecord,
    DeviceObservation,
    Transaction,
)


class Provenance(Enum):
    """Whether a parameter was found or reasoned."""

    SOURCED = "sourced"
    ASSUMED = "assumed"


@dataclass(frozen=True, slots=True)
class Parameter:
    """One generator parameter, with where it came from.

    Carried rather than documented in prose because the distinction has to
    survive into the report. A reader of generated data must be able to ask
    which of its properties are grounded, and get an answer from the artefact.
    """

    name: str
    value: float | tuple[float, ...] | tuple[str, ...]
    provenance: Provenance
    basis: str


#: Ordinary mobile-money transaction values in kwacha. The Zambian agent
#: withdrawal fee tables for 2026 band at K500, K1,000, K2,000 and K5,000 across
#: all three operators, which is where ordinary usage sits; a lognormal centred
#: in that range reproduces the long right tail without a hard ceiling.
TRANSACTION_KWACHA = Parameter(
    name="transaction_amount_kwacha",
    value=(1200.0, 1.1),
    provenance=Provenance.SOURCED,
    basis="Zambian agent withdrawal fee bands, K500-K5,000, Airtel/MTN/Zamtel 2026",
)

#: Fraction of incidents whose script is impersonation of network staff or
#: police, as against agent-assisted diversion.
IMPERSONATION_SHARE = Parameter(
    name="impersonation_share",
    value=0.62,
    provenance=Provenance.SOURCED,
    basis="58-72% of African mobile money fraud is social engineering; "
    "SIM-swap 43% and agent-assisted 38% of that",
)

#: Probability that a given utterance switches language. Lusaka and Copperbelt
#: speech is routinely mixed rather than monolingual, so a low per-utterance
#: switch rate would misrepresent the material the behavioural stream must read.
CODE_SWITCH_RATE = Parameter(
    name="code_switch_rate",
    value=0.35,
    provenance=Provenance.SOURCED,
    basis="documented Bemba/Nyanja/English mixing in Lusaka and the Copperbelt; "
    "Town Bemba as an urban hybrid variety",
)

#: Call duration in seconds, lognormal. **Assumed.** No Zambian figure was
#: found. Reasoned from the script: an impersonation call must establish a
#: pretext, extract an identifier and talk a victim through a handset
#: interaction, which does not happen in under a minute and rarely holds a
#: victim beyond about ten.
CALL_DURATION_SECONDS = Parameter(
    name="call_duration_seconds",
    value=(180.0, 0.6),
    provenance=Provenance.ASSUMED,
    basis="reasoned from the mechanics of an impersonation script; no Zambian "
    "call-duration distribution was found",
)

#: Seconds between successive calls within one incident. **Assumed.** An
#: operation working a list dials again quickly; the spread is wide because
#: failed calls end fast and successful ones do not.
INTER_CALL_SECONDS = Parameter(
    name="inter_call_seconds",
    value=(900.0, 1.0),
    provenance=Provenance.ASSUMED,
    basis="reasoned from an operation working a call list; no Zambian figure found",
)

#: Handset models, most-common first. **Assumed** in its proportions: Zambian
#: handset share was not found. The names are real budget Android devices
#: prevalent in the region, and the shape is a steep Zipf, which is what handset
#: populations generally look like — a few models dominating a long tail.
HANDSET_MODELS = Parameter(
    name="handset_models",
    value=(
        "itel A56",
        "TECNO Spark 8C",
        "Samsung Galaxy A04",
        "TECNO Pop 5",
        "Infinix Hot 11",
        "Nokia 105",
        "Huawei Y6p",
    ),
    provenance=Provenance.ASSUMED,
    basis="regionally prevalent budget devices; proportions assumed Zipf, no "
    "Zambian handset-share data found",
)

#: Cell sites, named for the two urban areas the sourced linguistic material
#: describes. **Assumed** as a set; the concentration is the point rather than
#: the names.
CELL_SITES = Parameter(
    name="cell_sites",
    value=(
        "LSK-Kabwata",
        "LSK-Matero",
        "LSK-Chilenje",
        "LSK-Kanyama",
        "CB-Kitwe-Central",
        "CB-Ndola-Masala",
    ),
    provenance=Provenance.ASSUMED,
    basis="Lusaka and Copperbelt siting, from where the multilingual material is "
    "documented; individual sites are illustrative",
)

PARAMETERS: tuple[Parameter, ...] = (
    TRANSACTION_KWACHA,
    IMPERSONATION_SHARE,
    CODE_SWITCH_RATE,
    CALL_DURATION_SECONDS,
    INTER_CALL_SECONDS,
    HANDSET_MODELS,
    CELL_SITES,
)


@dataclass(frozen=True, slots=True)
class SyntheticIncident:
    """One generated incident, with the truth that generated it.

    ``operator_id`` and ``operation_id`` are the ground truth a fusion
    experiment needs and a real corpus does not have. They are the reason to
    generate at all, and the reason nothing measured on them may be reported as
    a property of the system.
    """

    incident_id: str
    operator_id: str
    operation_id: str

    calls: tuple[CallRecord, ...]
    transactions: tuple[Transaction, ...]
    device: DeviceObservation
    transcript: str

    synthetic: bool = field(default=True, init=False)
    """Travels into every report derived from this incident. A ``C_llr`` from
    generated data is a property of the generator; the flag is what stops it
    being read as a property of the system."""


def _stable_seed(*parts: str) -> int:
    """Seed from a hash of the parts, never from the built-in ``hash``.

    ``hash`` is salted per interpreter, so a corpus generated today and
    regenerated tomorrow would differ while claiming the same seed. §18 records
    the same rule for trial attribution, and for the same reason.
    """
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _zipf_choice(
    rng: np.random.Generator, options: tuple[str, ...], size: int, exponent: float = 1.2
) -> list[str]:
    weights = np.array([1.0 / (i + 1) ** exponent for i in range(len(options))])
    weights /= weights.sum()
    # str() on each: numpy returns np.str_, which is a str subclass but
    # serialises as a numpy scalar and shows up as np.str_('...') in any
    # report that round-trips through repr. Dict keys built from these reach
    # JSON.
    return [str(item) for item in rng.choice(options, size=size, p=weights)]


#: Move labels the behavioural stream's script segmenter recognises, in the
#: order an impersonation pretext runs them. Ordering is evidence in its own
#: right — `_sequence_log_lr` scores it separately — so a generator that emitted
#: moves in a random order would make the script component untestable.
IMPERSONATION_SCRIPT: tuple[str, ...] = (
    "greeting",
    "authority_claim",
    "problem_statement",
    "urgency",
    "credential_request",
    "instruction",
    "closing",
)

AGENT_DIVERSION_SCRIPT: tuple[str, ...] = (
    "greeting",
    "transaction_reference",
    "problem_statement",
    "instruction",
    "credential_request",
    "closing",
)

#: Phrasing per move, per language. Three languages because the sourced material
#: says Lusaka speech is mixed rather than monolingual, and a generator that
#: emitted only English would exercise the behavioural stream on material the
#: deployment population does not produce.
PHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "greeting": {
        "en": ("good afternoon sir", "hello madam how are you"),
        "bem": ("mwapoleni mukwai", "shani mukwai"),
        "nya": ("muli bwanji", "moni bambo"),
    },
    "authority_claim": {
        "en": ("i am calling from the network security team",),
        "bem": ("ndefuma ku network security",),
        "nya": ("ndikuyimba kuchokera ku network",),
    },
    "transaction_reference": {
        "en": ("there is a transaction of two thousand on your wallet",),
        "bem": ("kwaliba transaction ya two thousand pa wallet yenu",),
        "nya": ("pali transaction ya two thousand pa wallet yanu",),
    },
    "problem_statement": {
        "en": ("your account has been flagged for suspicious activity",),
        "bem": ("account yenu naibikwa pa suspicious activity",),
        "nya": ("account yanu yaikidwa pa suspicious activity",),
    },
    "urgency": {
        "en": ("if you do not act now the account will be locked",),
        "bem": ("nga tamucita nomba account ikashibwa",),
        "nya": ("ngati simuchita tsopano account izatsekedwa",),
    },
    "credential_request": {
        "en": ("read me the code that has come to your phone",),
        "bem": ("mumbelenge code iyafika pa phone yenu",),
        "nya": ("mundiwerengere code yomwe yafika pa foni yanu",),
    },
    "instruction": {
        "en": ("press star three zero five hash and confirm",),
        "bem": ("tinyeni star three zero five hash mukonfeme",),
        "nya": ("dinani star three zero five hash mutsimikize",),
    },
    "closing": {
        "en": ("thank you for your cooperation sir",),
        "bem": ("natotela sana mukwai",),
        "nya": ("zikomo kwambiri bambo",),
    },
}

#: Disfluencies the inventory already recognises, so generated speech carries
#: the markers the idiolect component is built to count.
DISFLUENCIES: dict[str, tuple[str, ...]] = {
    "en": ("eh", "you know"),
    "bem": ("eeh", "ati", "nomba"),
    "nya": ("eeh", "bwanji", "sha"),
}


def _utterance(
    rng: np.random.Generator, move: str, base_language: str, switch_rate: float
) -> str:
    """One move, in the operator's base language or switched out of it.

    The switch is per utterance rather than per incident, because that is what
    the sourced material describes: Lusaka speech mixes within a conversation
    rather than choosing a language for it.
    """
    languages = tuple(PHRASES[move].keys())
    language = base_language
    if rng.random() < switch_rate:
        alternatives = tuple(item for item in languages if item != base_language)
        if alternatives:
            language = str(rng.choice(alternatives))
    if language not in PHRASES[move]:
        language = "en"
    phrase = str(rng.choice(PHRASES[move][language]))
    if rng.random() < 0.4:
        marker = str(rng.choice(DISFLUENCIES.get(language, DISFLUENCIES["en"])))
        phrase = f"{marker} {phrase}"
    return phrase


def generate_incident(
    incident_id: str,
    operator_id: str,
    operation_id: str,
    *,
    seed: int = 20250601,
    n_calls: int = 6,
    n_transactions: int = 5,
) -> SyntheticIncident:
    """One incident, with all five streams and the truth that produced it.

    The dependence structure is the whole point and it is not a correlation
    knob. ``operator_id`` drives the idiolect — base language, disfluency habit,
    handset — and ``operation_id`` drives the script, the transaction pattern
    and the working hours. An incident is the two together, which is exactly the
    common cause ``EvidenceStream`` describes: *the same operator running the
    same operation*. Delegation, one operation with more than one operator, is
    then expressible by holding ``operation_id`` and varying ``operator_id``,
    and it is the pattern the behavioural stream's flag exists to catch.
    """
    rng = np.random.default_rng(_stable_seed(str(seed), incident_id))
    operator_rng = np.random.default_rng(_stable_seed(str(seed), "operator", operator_id))
    operation_rng = np.random.default_rng(
        _stable_seed(str(seed), "operation", operation_id)
    )

    base_language = str(operator_rng.choice(("en", "bem", "nya")))
    handset = _zipf_choice(operator_rng, HANDSET_MODELS.value, 1)[0]  # type: ignore[arg-type]
    imei = f"35{int(operator_rng.integers(10**12, 10**13 - 1)):013d}"

    script = (
        IMPERSONATION_SCRIPT
        if operation_rng.random() < float(IMPERSONATION_SHARE.value)
        else AGENT_DIVERSION_SCRIPT
    )
    # Working hours belong to the operation: a team keeps its own shift.
    start_hour = int(operation_rng.integers(8, 19))
    site_pool = _zipf_choice(operation_rng, CELL_SITES.value, 3)  # type: ignore[arg-type]

    switch_rate = float(CODE_SWITCH_RATE.value)
    transcript = " . ".join(
        _utterance(rng, move, base_language, switch_rate) for move in script
    )

    day = datetime(2026, 3, int(rng.integers(1, 28)), start_hour, tzinfo=UTC)
    duration_median, duration_sigma = CALL_DURATION_SECONDS.value  # type: ignore[misc]
    gap_median, gap_sigma = INTER_CALL_SECONDS.value  # type: ignore[misc]

    calls: list[CallRecord] = []
    stamp = day
    for _ in range(n_calls):
        calls.append(
            CallRecord(
                timestamp=stamp,
                duration_seconds=float(
                    rng.lognormal(np.log(duration_median), duration_sigma)
                ),
                direction="outbound",
            )
        )
        stamp = stamp + timedelta(
            seconds=float(rng.lognormal(np.log(gap_median), gap_sigma))
        )

    amount_median, amount_sigma = TRANSACTION_KWACHA.value  # type: ignore[misc]
    agents = [f"AG-{int(operation_rng.integers(10000, 99999))}" for _ in range(2)]
    transactions = tuple(
        Transaction(
            timestamp=calls[min(index, len(calls) - 1)].timestamp
            + timedelta(seconds=float(rng.integers(30, 600))),
            amount=float(rng.lognormal(np.log(amount_median), amount_sigma)),
            transaction_type=str(rng.choice(("withdrawal", "transfer", "deposit"))),
            counterparty_wallet=f"09{int(rng.integers(10**7, 10**8 - 1)):08d}",
            agent_id=str(rng.choice(agents)),
        )
        for index in range(n_transactions)
    )

    sites = _zipf_choice(rng, tuple(site_pool), n_calls)
    device = DeviceObservation(
        imei_counts={imei: n_calls},
        handset_model_counts={handset: n_calls},
        cell_site_counts={site: sites.count(site) for site in set(sites)},
    )

    return SyntheticIncident(
        incident_id=incident_id,
        operator_id=operator_id,
        operation_id=operation_id,
        calls=tuple(calls),
        transactions=transactions,
        device=device,
        transcript=transcript,
    )


def generate_corpus(
    n_operations: int = 12,
    operators_per_operation: int = 3,
    incidents_per_operator: int = 3,
    *,
    seed: int = 20250601,
) -> list[SyntheticIncident]:
    """A corpus with delegation built into its structure.

    ``operators_per_operation`` above one is what makes the corpus interesting:
    it produces pairs sharing an operation but not a speaker, which is the
    pattern §13 says the behavioural stream's decomposition exists to detect and
    which no real corpus available to this project contains with known truth.
    """
    incidents: list[SyntheticIncident] = []
    for operation in range(n_operations):
        operation_id = f"op-{operation:03d}"
        for operator in range(operators_per_operation):
            operator_id = f"spk-{operation:03d}-{operator:02d}"
            for repeat in range(incidents_per_operator):
                incidents.append(
                    generate_incident(
                        incident_id=f"inc-{operation:03d}-{operator:02d}-{repeat:02d}",
                        operator_id=operator_id,
                        operation_id=operation_id,
                        seed=seed,
                    )
                )
    return incidents


def describe_parameters() -> list[dict[str, object]]:
    """The parameter set with its provenance, for a report to carry."""
    return [
        {
            "name": parameter.name,
            "value": parameter.value,
            "provenance": parameter.provenance.value,
            "basis": parameter.basis,
        }
        for parameter in PARAMETERS
    ]
