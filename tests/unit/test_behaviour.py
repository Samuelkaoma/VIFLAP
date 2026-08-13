"""The behavioural stream: idiolect, script structure, and telling them apart.

This stream exists because of an attack the acoustic stream cannot survive. A
group that rotates who makes the calls defeats voice linkage completely — two
incidents run by two people are two different voices, and no speaker recognition
system links them. What persists is the *operation*: the same pretext, the same
sequence of moves, the same escalation when the victim resists.

So the module scores two things separately and the tests follow that division:

**Idiolect** — function words, disfluencies, code-switching. Belongs to the
person. Destroyed by delegation.

**Script** — move inventory, ordering, characteristic phrasing. Belongs to the
operation. Survives delegation.

The transcripts below are mobile money agent-impersonation scripts, which is the
setting this system is aimed at, and they are constructed so that the pairs
differ in exactly one respect at a time: same script with a different operator,
or same operator running a different script. That is what makes it possible to
assert which component should move.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from viflap.analysis.behaviour.profile import (
    BehaviouralComparator,
    BehaviouralProfile,
    BehaviouralScore,
    DisfluencyInventory,
    _sequence_log_lr,
    build_profile,
)
from viflap.analysis.patterns.conjugate import (
    BackgroundPopulation,
    DirichletMultinomialComparator,
)
from viflap.domain.errors import InsufficientDataError
from viflap.domain.evidence import EvidenceStream

FUNCTION_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "must",
        "your",
        "you",
        "we",
        "i",
        "it",
        "this",
        "that",
        "there",
        "now",
        "just",
        "please",
        "sir",
        "madam",
        "not",
        "no",
        "yes",
        "from",
        "by",
        "at",
        "as",
    }
)


# -- Transcripts ---------------------------------------------------------
#
# OPERATOR_A and OPERATOR_B run the SAME script: authority claim, then urgency,
# then a request for the PIN, then a threat when the victim hesitates. They are
# different people — different function word habits, different filled pauses,
# different code-switching.

OPERATOR_A_SCRIPT_ONE = """
uh good afternoon madam this is calling from the head office of the mobile money
department um we have detected that there is a problem with your account and it
must be corrected now otherwise the account will be blocked today. eeh please
listen carefully madam because this is urgent. i need you to confirm the pin that
you are using on the handset so that we can verify that you are the owner of the
wallet. um if you do not confirm the pin right now the money that is in the
account will be frozen and you will have to travel to the head office in town to
recover it. eeh madam are you there. i am waiting for the pin now please.
"""

OPERATOR_B_SCRIPT_ONE = """
ah hello sir, so, this is the manager from the bank mobile money section and
there is something wrong with the wallet which we must fix immediately, right.
so the system is showing an error and the account will be closed if it is not
corrected. mmm i want you to give me the pin code from your phone so that we can
confirm ownership on our side, okay. so if the pin is not given the funds will
be locked and then, ah, you must go yourself to town to sort it out with the
office. mmm sir, are you still on the line, i need that pin now, okay, quickly.
"""

# OPERATOR_A again, but a different operation: a lottery pretext with no
# authority claim and no threat. Same person, different script.
OPERATOR_A_SCRIPT_TWO = """
uh good afternoon madam i am calling with very good news for you today um your
number has been selected in the promotion that we are running this month and you
have won a prize. eeh congratulations madam this is a big thing. um to receive
the prize i just need you to confirm a few details so that we can send the money
to the correct wallet. eeh it is not a problem it is just the normal procedure
that everyone follows. um so madam are you happy with the news. i will wait
while you get a pen and paper to write the reference number down please.
"""


def _background(counts: Mapping[str, int], description: str) -> BackgroundPopulation:
    """A background population from pooled counts, normalised to sum to one."""
    total = sum(counts.values())
    return BackgroundPopulation(
        frequencies={key: value / total for key, value in counts.items()},
        total_observations=total,
        description=description,
    )


def _pool(*mappings: Mapping[str, int]) -> dict[str, int]:
    pooled: dict[str, int] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            pooled[key] = pooled.get(key, 0) + value
    return pooled


@pytest.fixture
def profiles() -> dict[str, BehaviouralProfile]:
    return {
        name: build_profile(text, FUNCTION_WORDS)
        for name, text in (
            ("a_one", OPERATOR_A_SCRIPT_ONE),
            ("b_one", OPERATOR_B_SCRIPT_ONE),
            ("a_two", OPERATOR_A_SCRIPT_TWO),
        )
    }


@pytest.fixture
def comparator(profiles: dict[str, BehaviouralProfile]) -> BehaviouralComparator:
    """Backgrounds pooled from the three transcripts.

    A real deployment estimates these from the case corpus. Pooling the material
    under test is the honest small-scale stand-in: it gives every category a
    non-zero background frequency, so no likelihood ratio here is an artefact of
    a category the background happened never to have seen.
    """
    every = list(profiles.values())
    return BehaviouralComparator(
        function_word_background=_background(
            _pool(*(p.function_word_counts for p in every)), "pooled function words"
        ),
        disfluency_background=_background(
            _pool(*(p.disfluency_counts for p in every)), "pooled disfluencies"
        ),
        move_background=_background(_pool(*(p.move_counts for p in every)), "pooled moves"),
        ngram_background=_background(
            _pool(*(p.character_ngram_counts for p in every)), "pooled character n-grams"
        ),
    )


class TestDisfluencyInventory:
    def test_recognises_english_markers(self) -> None:
        inventory = DisfluencyInventory()
        assert inventory.contains("um")
        assert inventory.contains("uh")

    def test_recognises_zambian_discourse_markers(self) -> None:
        """The inventory is language-specific, and the deployment is Zambian."""
        inventory = DisfluencyInventory()
        for marker in ("eeh", "ati", "sha", "iwe", "nomba", "bwanji"):
            assert inventory.contains(marker), marker

    def test_matching_is_case_insensitive(self) -> None:
        assert DisfluencyInventory().contains("EEH")

    def test_an_ordinary_word_is_not_a_disfluency(self) -> None:
        assert not DisfluencyInventory().contains("account")

    def test_the_inventory_is_replaceable(self) -> None:
        """Supplied as configuration because a different deployment needs different
        markers, and a hard-coded English list would silently under-count."""
        inventory = DisfluencyInventory(markers=frozenset({"zzz"}))
        assert inventory.contains("zzz")
        assert not inventory.contains("um")


class TestBuildProfile:
    def test_counts_function_words(self, profiles: dict[str, BehaviouralProfile]) -> None:
        counts = profiles["a_one"].function_word_counts
        assert counts["the"] > 0
        assert sum(counts.values()) > 0

    def test_counts_disfluencies(self, profiles: dict[str, BehaviouralProfile]) -> None:
        assert profiles["a_one"].disfluency_counts.get("eeh", 0) > 0
        assert profiles["b_one"].disfluency_counts.get("mmm", 0) > 0

    def test_detects_script_moves(self, profiles: dict[str, BehaviouralProfile]) -> None:
        assert profiles["a_one"].move_counts
        assert len(profiles["a_one"].move_sequence) > 0

    def test_refuses_a_transcript_too_short_to_profile(self) -> None:
        """Function word profiles from short texts are noise wearing a number."""
        with pytest.raises(InsufficientDataError):
            build_profile("um hello there sir", FUNCTION_WORDS)

    def test_disfluency_rate_is_per_hundred_words(self) -> None:
        profile = build_profile(OPERATOR_A_SCRIPT_ONE, FUNCTION_WORDS)
        expected = 100.0 * sum(profile.disfluency_counts.values()) / profile.n_words
        assert profile.disfluency_rate == pytest.approx(expected)

    def test_the_two_operators_have_different_disfluency_habits(
        self, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        """The premise of the fixtures, asserted rather than assumed."""
        a_markers = set(profiles["a_one"].disfluency_counts)
        b_markers = set(profiles["b_one"].disfluency_counts)
        assert a_markers != b_markers

    def test_one_operator_keeps_their_habits_across_scripts(
        self, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        """Idiolect follows the person, which is what makes it evidence."""
        shared = set(profiles["a_one"].disfluency_counts) & set(
            profiles["a_two"].disfluency_counts
        )
        assert {"um", "eeh"} <= shared


class TestBehaviouralComparator:
    def test_reports_its_stream(self, comparator: BehaviouralComparator) -> None:
        assert comparator.stream is EvidenceStream.BEHAVIOURAL

    def test_a_transcript_compared_with_itself_supports_linkage(
        self, comparator: BehaviouralComparator, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        score = comparator.score(profiles["a_one"], profiles["a_one"])
        assert score.total_log_lr > 0.0

    def test_refuses_transcripts_below_the_minimum(
        self, comparator: BehaviouralComparator, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        short = build_profile(OPERATOR_A_SCRIPT_ONE, FUNCTION_WORDS, min_words=1)
        object.__setattr__(short, "n_words", 5)

        with pytest.raises(InsufficientDataError):
            comparator.score(short, profiles["a_one"])

    def test_the_total_is_the_sum_of_its_parts(
        self, comparator: BehaviouralComparator, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        """The decomposition must not drift from the number reported."""
        score = comparator.score(profiles["a_one"], profiles["b_one"])
        assert score.total_log_lr == pytest.approx(
            score.idiolect_log_lr + score.script_log_lr
        )

    def test_diagnostics_record_what_was_compared(
        self, comparator: BehaviouralComparator, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        score = comparator.score(profiles["a_one"], profiles["b_one"])
        assert score.diagnostics["n_words_first"] == float(profiles["a_one"].n_words)
        assert score.diagnostics["n_words_second"] == float(profiles["b_one"].n_words)


class TestDelegationSignature:
    """Same operation, different operator — the finding voice cannot produce.

    ``suggests_shared_operation_not_speaker`` is a relative comparison, not a
    threshold on either component, because both are uncalibrated and sit on
    scales set by their own inventory sizes. These tests therefore pin the
    *relationship*, which is the part that carries meaning.
    """

    def _score(self, idiolect: float, script: float) -> BehaviouralScore:
        return BehaviouralScore(
            idiolect_log_lr=idiolect,
            script_log_lr=script,
            total_log_lr=idiolect + script,
            diagnostics={},
        )

    def test_strong_script_and_weak_idiolect_is_the_delegation_pattern(self) -> None:
        assert self._score(idiolect=0.2, script=5.0).suggests_shared_operation_not_speaker

    def test_both_strong_is_not_delegation(self) -> None:
        """One person running their own script. Nothing to flag."""
        assert not self._score(
            idiolect=5.0, script=5.0
        ).suggests_shared_operation_not_speaker

    def test_weak_script_is_not_delegation_however_weak_the_idiolect(self) -> None:
        """Below a factor of ten the script evidence is not worth remarking on."""
        assert not self._score(
            idiolect=0.0, script=math.log(9.0)
        ).suggests_shared_operation_not_speaker

    def test_negative_idiolect_does_not_inflate_the_comparison(self) -> None:
        """Evidence *against* a shared speaker must not make delegation easier.

        The guard is the ``max(idiolect, 0)``: without it a strongly negative
        idiolect term would clear any ratio test trivially, and the flag would
        fire hardest exactly where the two transcripts look least alike.
        """
        flagged = self._score(idiolect=-50.0, script=math.log(10.0) + 0.01)
        assert flagged.suggests_shared_operation_not_speaker

        # ... but the script term still has to stand on its own.
        assert not self._score(
            idiolect=-50.0, script=math.log(10.0) - 0.01
        ).suggests_shared_operation_not_speaker

    def test_move_evidence_favours_the_shared_script(
        self, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        """The script machinery proper, isolated from the n-gram term.

        Moves and their ordering are what "script structure" means. Both
        separate the pairs the right way round: positive for two operators
        running one script, negative for one operator running two.
        """
        moves = DirichletMultinomialComparator(
            _background(_pool(*(p.move_counts for p in profiles.values())), "pooled moves"),
            concentration=40.0,
        )

        delegated = moves.log_likelihood_ratio(
            profiles["a_one"].move_counts, profiles["b_one"].move_counts
        ) + _sequence_log_lr(
            profiles["a_one"].move_sequence, profiles["b_one"].move_sequence
        )
        same_operator = moves.log_likelihood_ratio(
            profiles["a_one"].move_counts, profiles["a_two"].move_counts
        ) + _sequence_log_lr(
            profiles["a_one"].move_sequence, profiles["a_two"].move_sequence
        )

        assert delegated > 0.0
        assert same_operator < 0.0
        assert delegated > same_operator

    def test_character_ngrams_carry_speaker_evidence_into_the_script_term(
        self, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        """A characterisation test for a real problem, not an endorsement of it.

        ``script_log_lr`` sums move evidence, sequence evidence and a character
        n-gram term. Character n-grams are the standard instrument of authorship
        attribution — they measure who is writing. Here they are counted as
        *script* evidence, the component whose entire purpose is to survive a
        change of speaker.

        The consequence is measurable: the n-gram term prefers two transcripts
        by one operator over two transcripts sharing a script, and it is large
        enough to reverse the sign of the whole script component. The move and
        sequence terms, which do separate the pairs correctly, are swamped.

        This is pinned rather than fixed because the fix is a modelling
        decision, not a bug fix: either n-grams move to the idiolect term, or
        they are restricted to script-bearing spans, or the components are
        calibrated separately so their magnitudes become comparable. That choice
        needs labelled same-operation-different-speaker data, which the corpus
        does not yet contain. Until then, this test fails loudly if the
        behaviour changes, and documents why the delegation flag should not be
        trusted on ``script_log_lr`` alone.
        """
        ngrams = DirichletMultinomialComparator(
            _background(
                _pool(*(p.character_ngram_counts for p in profiles.values())),
                "pooled character n-grams",
            ),
            concentration=15.0,
        )

        delegated = ngrams.log_likelihood_ratio(
            profiles["a_one"].character_ngram_counts,
            profiles["b_one"].character_ngram_counts,
        )
        same_operator = ngrams.log_likelihood_ratio(
            profiles["a_one"].character_ngram_counts,
            profiles["a_two"].character_ngram_counts,
        )

        # It runs the wrong way for a script term: the same operator scores
        # higher than the same script.
        assert same_operator > delegated

        # And it is large enough to decide the sum on its own.
        assert abs(delegated) > 20.0 * abs(
            _sequence_log_lr(
                profiles["a_one"].move_sequence, profiles["b_one"].move_sequence
            )
        )

    def test_one_operator_across_two_scripts_scores_idiolect_above_the_delegated_pair(
        self, comparator: BehaviouralComparator, profiles: dict[str, BehaviouralProfile]
    ) -> None:
        """The mirror image, and the reason the two components are reported apart."""
        delegated = comparator.score(profiles["a_one"], profiles["b_one"])
        same_operator = comparator.score(profiles["a_one"], profiles["a_two"])

        assert same_operator.idiolect_log_lr > delegated.idiolect_log_lr
