"""The generative structure has to be the one the system claims to detect.

A synthetic corpus is only useful for testing VIFLAP if it contains the pattern
VIFLAP exists to find. The behavioural stream splits evidence into *script* —
which survives a change of speaker — and *idiolect* — which does not; the whole
point is detecting one operation run by several operators. A generator that gave
every incident its own script and its own speaker would exercise the plumbing
and could never exercise the claim.

So these tests assert the structure rather than the numbers: scripts belong to
operations, idiolect belongs to operators, and the two vary independently. If a
future change breaks that, the corpus keeps generating and quietly stops being
worth testing against, which is the failure mode worth catching.

They also assert that the provenance table stays honest, because the one thing
this corpus must never do is have a figure quoted off it as a measurement.
"""

from __future__ import annotations

from collections import Counter

import pytest

from scripts.synthesise_incidents import (
    DISCOURSE_MARKERS,
    PARAMETER_PROVENANCE,
    SCRIPT_MOVES,
    generate,
)
from viflap.analysis.behaviour.profile import MIN_WORDS_IDIOLECT, build_profile


@pytest.fixture
def corpus():
    return generate(n_operators=12, n_operations=6, incidents_per_operation=8)


class TestGenerativeStructure:
    def test_delegation_exists_or_the_corpus_is_pointless(self, corpus) -> None:
        """At least one operation must be run by more than one operator.

        This is the pattern ``suggests_shared_operation_not_speaker`` detects.
        Without it the corpus cannot exercise the behavioural stream's central
        claim, however realistic it looks otherwise.
        """
        _, operations, _ = corpus
        assert any(len(operation.operator_ids) > 1 for operation in operations)

    def test_every_incident_is_run_by_an_operator_of_its_operation(self, corpus) -> None:
        _, operations, incidents = corpus
        roster = {o.operation_id: set(o.operator_ids) for o in operations}
        for incident in incidents:
            assert incident.operator_id in roster[incident.operation_id]

    def test_one_operator_appears_across_several_incidents(self, corpus) -> None:
        """Same-source comparisons need a person to appear more than once."""
        _, _, incidents = corpus
        counts = Counter(incident.operator_id for incident in incidents)
        assert max(counts.values()) >= 2

    def test_the_script_belongs_to_the_operation(self, corpus) -> None:
        """Every incident of one operation runs the same move order."""
        _, operations, _ = corpus
        for operation in operations:
            assert set(operation.move_order) == set(SCRIPT_MOVES)
            assert len(operation.move_order) == len(SCRIPT_MOVES)

    def test_operations_differ_in_their_scripts(self, corpus) -> None:
        """Otherwise script evidence is constant and cannot discriminate."""
        _, operations, _ = corpus
        assert len({operation.move_order for operation in operations}) > 1

    def test_idiolect_belongs_to_the_operator_not_the_operation(self, corpus) -> None:
        """Two operators must differ in their marker habits.

        Identical idiolects would make the idiolect term uninformative, and a
        test suite that passed on such a corpus would be testing nothing.
        """
        operators, _, _ = corpus
        first, second = operators[0], operators[1]
        assert first.marker_weights != second.marker_weights
        assert set(first.marker_weights) == set(DISCOURSE_MARKERS)

    def test_the_handset_pool_belongs_to_the_operation(self, corpus) -> None:
        """The device stream's signal is the operation's, per the Lusaka case."""
        _, operations, incidents = corpus
        pools = {o.operation_id: set(o.sim_pool) for o in operations}
        for incident in incidents:
            assert incident.msisdn in pools[incident.operation_id]

    def test_transcripts_clear_the_idiolect_floor(self, corpus) -> None:
        """Otherwise half the behavioural stream is switched off, silently.

        ``BehaviouralComparator`` withholds the idiolect term below
        ``MIN_WORDS_IDIOLECT`` and still returns a score — the script term alone.
        A corpus of short transcripts therefore exercises the stream's plumbing
        while never touching the component the authorship literature is about,
        and nothing fails to say so. This is the guard on that.
        """
        _, _, incidents = corpus
        lengths = [
            build_profile(incident.transcript, frozenset({"the", "you", "i", "is"})).n_words
            for incident in incidents
        ]
        assert min(lengths) >= MIN_WORDS_IDIOLECT, (
            f"shortest transcript is {min(lengths)} words against a floor of "
            f"{MIN_WORDS_IDIOLECT}"
        )

    def test_the_operator_length_habit_reaches_the_transcript(self, corpus) -> None:
        """``words_per_utterance`` was drawn per operator and never read.

        A declared idiolect parameter with no consequence is worse than none: it
        reads as though the corpus varies along an axis it does not.
        """
        operators, _, incidents = corpus
        habit = {o.operator_id: o.words_per_utterance for o in operators}
        by_operator: dict[str, list[float]] = {}
        for incident in incidents:
            turns = incident.transcript.split(" . ")
            mean_length = sum(len(t.split()) for t in turns) / len(turns)
            by_operator.setdefault(incident.operator_id, []).append(mean_length)

        # Two operators with clearly different habits must differ in the
        # transcripts they produce.
        ranked = sorted(by_operator, key=lambda name: habit[name])
        shortest, longest = ranked[0], ranked[-1]
        assert habit[shortest] < habit[longest]
        mean = lambda values: sum(values) / len(values)  # noqa: E731
        assert mean(by_operator[shortest]) < mean(by_operator[longest])

    def test_generation_is_reproducible(self, corpus) -> None:
        _, _, incidents = corpus
        _, _, again = generate(n_operators=12, n_operations=6, incidents_per_operation=8)
        assert [i.incident_id for i in incidents] == [i.incident_id for i in again]
        assert [i.transcript for i in incidents] == [i.transcript for i in again]

    def test_a_different_seed_gives_a_different_corpus(self) -> None:
        _, _, first = generate(seed=1)
        _, _, second = generate(seed=2)
        assert [i.transcript for i in first] != [i.transcript for i in second]


class TestProvenanceStaysHonest:
    """The corpus must never let a figure be quoted off it as a measurement."""

    def test_every_assumed_parameter_says_so_in_capitals(self) -> None:
        assumed = [
            key
            for key, note in PARAMETER_PROVENANCE.items()
            if not note.startswith("grounded")
        ]
        assert assumed, "a generator with nothing assumed is not being honest"
        for key in assumed:
            assert "ASSUMED" in PARAMETER_PROVENANCE[key], key

    def test_grounded_parameters_are_marked_as_grounded(self) -> None:
        grounded = [
            key for key, note in PARAMETER_PROVENANCE.items() if note.startswith("grounded")
        ]
        assert len(grounded) >= 4

    def test_no_parameter_is_left_undocumented(self) -> None:
        """Silence about a parameter is the failure this table exists to stop."""
        for key, note in PARAMETER_PROVENANCE.items():
            assert note.strip(), key
            assert note.startswith("grounded") or "ASSUMED" in note, key
