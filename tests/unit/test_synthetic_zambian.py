"""The generated corpus, and the properties that make it worth generating.

Three of five evidence streams have never seen material resembling their
deployment population. This corpus supplies it — for exercising the pipeline,
never for a reported figure — so what has to hold is that the *structure* is
right: the dependence is the one ``EvidenceStream`` describes, delegation is
expressible, the outputs feed the real comparators, and the whole thing is
reproducible.

The last of those is not a nicety. A generator seeded from the built-in ``hash``
would produce a different corpus on every interpreter while claiming the same
seed, which is the defect §18 records for trial attribution.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from scripts.synthetic_zambian import (
    PARAMETERS,
    Provenance,
    describe_parameters,
    generate_corpus,
    generate_incident,
)
from viflap.analysis.behaviour.profile import build_profile
from viflap.analysis.patterns.streams import TemporalProfile, TransactionalProfile


@pytest.fixture
def corpus():
    return generate_corpus(
        n_operations=6, operators_per_operation=3, incidents_per_operator=2
    )


class TestStructure:
    def test_delegation_is_expressible(self, corpus) -> None:
        """One operation, several operators — the pattern with no real corpus.

        §13 says the behavioural decomposition exists to separate the operation
        from the person, and that establishing it properly needs labelled
        same-operation-different-speaker pairs the project does not have. This
        is where they come from.
        """
        by_operation: dict[str, set[str]] = {}
        for incident in corpus:
            by_operation.setdefault(incident.operation_id, set()).add(incident.operator_id)
        assert all(len(speakers) >= 2 for speakers in by_operation.values())

    def test_operators_recur_so_same_source_pairs_exist(self, corpus) -> None:
        counts = Counter(incident.operator_id for incident in corpus)
        assert min(counts.values()) >= 2

    def test_an_operator_belongs_to_one_operation(self, corpus) -> None:
        """Otherwise the two truth labels are not separable and the corpus
        cannot distinguish a shared script from a shared speaker."""
        for operator in {incident.operator_id for incident in corpus}:
            operations = {
                incident.operation_id
                for incident in corpus
                if incident.operator_id == operator
            }
            assert len(operations) == 1


class TestDependenceIsCausalNotBolted:
    """``EvidenceStream`` states the streams share a common cause: the same
    operator running the same operation. The generator has to induce dependence
    that way rather than by correlating independent draws, or a fusion
    experiment on it tests the wrong thing.
    """

    def test_the_operator_drives_the_device(self) -> None:
        first = generate_incident("a", "spk-1", "op-1")
        second = generate_incident("b", "spk-1", "op-2")
        assert first.device.imei_counts.keys() == second.device.imei_counts.keys()

    def test_the_operation_drives_the_script(self) -> None:
        first = generate_incident("a", "spk-1", "op-1")
        second = generate_incident("b", "spk-2", "op-1")
        moves_first = build_profile(first.transcript, frozenset({"the"}), min_words=1)
        moves_second = build_profile(second.transcript, frozenset({"the"}), min_words=1)
        assert moves_first.move_sequence == moves_second.move_sequence

    def test_different_operators_carry_different_handsets(self) -> None:
        devices = {
            tuple(generate_incident(f"i{n}", f"spk-{n}", "op-1").device.imei_counts)
            for n in range(8)
        }
        assert len(devices) > 1


class TestFeedsTheRealComparators:
    """Generated data that the production profiles refuse is worthless."""

    def test_calls_build_a_temporal_profile(self, corpus) -> None:
        profile = TemporalProfile.from_records(corpus[0].calls)
        assert profile.n_calls == len(corpus[0].calls)

    def test_transactions_build_a_transactional_profile(self, corpus) -> None:
        profile = TransactionalProfile.from_transactions(corpus[0].transactions)
        assert profile.n_transactions == len(corpus[0].transactions)

    def test_transcripts_build_a_behavioural_profile(self, corpus) -> None:
        profile = build_profile(
            corpus[0].transcript, frozenset({"the", "your"}), min_words=1
        )
        assert profile.n_words > 0
        assert profile.move_sequence

    def test_device_counts_are_plain_strings(self, corpus) -> None:
        """numpy returns ``np.str_``, which is a str subclass that serialises as
        a numpy scalar. These become JSON keys in a report."""
        keys = {
            **corpus[0].device.handset_model_counts,
            **corpus[0].device.cell_site_counts,
            **corpus[0].device.imei_counts,
        }
        assert all(type(key) is str for key in keys)


class TestGroundedWhereItClaimsToBe:
    def test_amounts_sit_in_the_sourced_band(self, corpus) -> None:
        """K500 to K5,000 is where the Zambian agent fee tables band, so the
        bulk of generated transactions must land there rather than merely
        being positive."""
        amounts = np.array([t.amount for incident in corpus for t in incident.transactions])
        within = float(np.mean((amounts >= 300.0) & (amounts <= 8000.0)))
        assert within > 0.8, f"only {within:.0%} of amounts in the ordinary range"

    def test_transcripts_are_multilingual(self, corpus) -> None:
        """Monolingual English would misrepresent the material the behavioural
        stream must read: the sourced work describes Lusaka speech as routinely
        mixed rather than as choosing a language per conversation."""
        joined = " ".join(incident.transcript for incident in corpus)
        assert any(word in joined for word in ("mukwai", "muli", "moni", "zikomo"))
        assert "account" in joined

    def test_every_parameter_declares_its_provenance(self) -> None:
        assert all(isinstance(p.provenance, Provenance) for p in PARAMETERS)
        assert all(p.basis for p in PARAMETERS)

    def test_the_provenance_split_is_reported_rather_than_blurred(self) -> None:
        """A generator whose parameters are silently mixed reads as grounded
        throughout. Both kinds must be present and both must be visible."""
        described = describe_parameters()
        kinds = {entry["provenance"] for entry in described}
        assert kinds == {"sourced", "assumed"}


class TestReproducibility:
    def test_the_same_seed_gives_the_same_corpus(self) -> None:
        first = generate_corpus(n_operations=3, seed=7)
        second = generate_corpus(n_operations=3, seed=7)
        assert [i.transcript for i in first] == [i.transcript for i in second]
        assert [t.amount for i in first for t in i.transactions] == [
            t.amount for i in second for t in i.transactions
        ]

    def test_a_different_seed_gives_a_different_corpus(self) -> None:
        first = generate_corpus(n_operations=3, seed=7)
        second = generate_corpus(n_operations=3, seed=8)
        assert [t.amount for i in first for t in i.transactions] != [
            t.amount for i in second for t in i.transactions
        ]

    def test_every_incident_is_marked_synthetic(self, corpus) -> None:
        """The flag travels into any report derived from this data. A C_llr
        from generated incidents is a property of the generator."""
        assert all(incident.synthetic for incident in corpus)
