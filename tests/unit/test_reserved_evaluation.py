"""A reserved evaluation set, and the failure it exists to prevent.

`split_by_speaker` orders speakers by a seeded permutation and takes fractions,
so the split is a function of the *whole speaker set*. Adding speakers reshuffles
everyone, and a speaker held out at one corpus size becomes training material at
the next. Nothing detects it: both splits are internally valid and
`verify_disjoint` checks a split against itself.

§9 lost all but 35 of its baseline evaluation speakers that way, and §25 lost all
but 19 — too few to compare the 306- and 562-speaker models at all. These tests
pin the mechanism that stops it happening a third time.
"""

from __future__ import annotations

import pytest

from scripts.corpus import split_by_speaker
from viflap.evaluation.reserved import RESERVED_EVALUATION_SPEAKERS


class _Item:
    def __init__(self, speaker: str, index: int) -> None:
        self.speaker_id = speaker
        self.session_id = f"{speaker}-{index}"
        self.recording_id = f"{speaker}-{index}-r0"


def _corpus(n_speakers: int, per_speaker: int = 3) -> list[_Item]:
    return [
        _Item(f"{s:04d}", i) for s in range(n_speakers) for i in range(per_speaker)
    ]


class TestTheProblem:
    def test_growing_the_corpus_reshuffles_the_evaluation_set(self) -> None:
        """The defect, demonstrated. This is why §25 could not be paired."""
        small = {p.speaker_id for p in split_by_speaker(_corpus(60)).evaluation}
        large = {p.speaker_id for p in split_by_speaker(_corpus(120)).evaluation}
        # Most of the smaller corpus's held-out speakers do not survive.
        assert len(small & large) < len(small) * 0.6, len(small & large)


class TestReserving:
    def test_reserved_speakers_always_land_in_evaluation(self) -> None:
        reserved = {"0001", "0002", "0003", "0004", "0005"}
        split = split_by_speaker(_corpus(80), reserved_evaluation=reserved)
        held = {p.speaker_id for p in split.evaluation}
        assert reserved <= held

    def test_reserved_speakers_never_reach_training_or_development(self) -> None:
        """The whole point: a reserved speaker the model trained on is worse
        than no reservation, because it looks held out and is not."""
        reserved = {f"{i:04d}" for i in range(8)}
        split = split_by_speaker(_corpus(80), reserved_evaluation=reserved)
        assert not reserved & {p.speaker_id for p in split.train}
        assert not reserved & {p.speaker_id for p in split.development}

    def test_two_corpus_sizes_share_their_reserved_speakers_exactly(self) -> None:
        """The guarantee. Both corpora contain the reserved speakers, so both
        hold them out, so a paired comparison is available at any size."""
        reserved = {f"{i:04d}" for i in range(12)}
        small = {
            p.speaker_id
            for p in split_by_speaker(_corpus(60), reserved_evaluation=reserved).evaluation
        }
        large = {
            p.speaker_id
            for p in split_by_speaker(_corpus(140), reserved_evaluation=reserved).evaluation
        }
        assert reserved <= small & large

    def test_reserving_does_not_inflate_the_evaluation_partition(self) -> None:
        """Reserved speakers move from the drawn part of evaluation into the
        pinned part; they are not added on top, or the split would drift toward
        evaluation every time the list grew."""
        plain = split_by_speaker(_corpus(100))
        pinned = split_by_speaker(
            _corpus(100), reserved_evaluation={f"{i:04d}" for i in range(10)}
        )
        assert pinned.summary()["evaluation_speakers"] == pytest.approx(
            plain.summary()["evaluation_speakers"], abs=1
        )

    def test_identifiers_absent_from_the_corpus_are_ignored(self) -> None:
        """The list is a superset covering corpora a given call may not hold, so
        refusing on an absent identifier would make it unusable on any subset."""
        split = split_by_speaker(
            _corpus(60), reserved_evaluation={"0001", "9999", "nonexistent"}
        )
        assert "0001" in {p.speaker_id for p in split.evaluation}

    def test_the_split_stays_speaker_disjoint(self) -> None:
        reserved = {f"{i:04d}" for i in range(10)}
        split = split_by_speaker(_corpus(90), reserved_evaluation=reserved)
        train = {p.speaker_id for p in split.train}
        dev = {p.speaker_id for p in split.development}
        held = {p.speaker_id for p in split.evaluation}
        assert not train & dev and not train & held and not dev & held


class TestTheProjectList:
    def test_it_is_a_hundred_speakers(self) -> None:
        assert len(RESERVED_EVALUATION_SPEAKERS) == 100

    def test_it_is_immutable(self) -> None:
        """A protocol decision that can be mutated at run time is not one."""
        assert isinstance(RESERVED_EVALUATION_SPEAKERS, frozenset)

    def test_every_entry_is_a_librispeech_identifier(self) -> None:
        for speaker in RESERVED_EVALUATION_SPEAKERS:
            assert speaker.isdigit(), speaker
