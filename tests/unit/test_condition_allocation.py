"""Dealing channel conditions out to training recordings.

Corpus-wide balance was the property the original allocation was written for,
and it has it. The property it lacks is balance *within* a speaker, and that is
the one PLDA is sensitive to: a speaker's recordings are the only evidence the
model has about within-speaker variability, so whatever is common to them is
indistinguishable from the speaker. A speaker who happens to draw mostly
4.75 kbit/s in babble carries a mean channel offset, and LDA and PLDA absorb it
as between-speaker variance — the model learns a channel and reports a person.

These are the properties the stratified allocation has to hold, stated so that
a future change to the allocation cannot quietly lose one.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from scripts.train_acoustic import (
    TRAINING_CONDITIONS,
    assign_conditions,
    assign_conditions_globally,
    condition_balance,
)


def _corpus(n_speakers: int, per_speaker: int) -> list[str]:
    return [f"spk{s:04d}" for s in range(n_speakers) for _ in range(per_speaker)]


def _uneven_corpus(n_speakers: int = 306, seed: int = 0) -> list[str]:
    """Speaker sizes like the pooled LibriSpeech corpus: two to eight each."""
    rng = np.random.default_rng(seed)
    speakers: list[str] = []
    for s in range(n_speakers):
        speakers.extend([f"spk{s:04d}"] * int(rng.integers(2, 9)))
    return speakers


class TestStratifiedAllocation:
    def test_every_recording_receives_exactly_one_condition(self) -> None:
        speakers = _uneven_corpus(40)
        assert len(assign_conditions(speakers)) == len(speakers)

    @pytest.mark.parametrize("per_speaker", [2, 3, 5, 8])
    def test_a_speaker_never_repeats_a_condition_it_could_have_avoided(
        self, per_speaker
    ) -> None:
        """The Latin square's within-block property, on blocks of equal size."""
        speakers = _corpus(50, per_speaker)
        conditions = assign_conditions(speakers)

        for speaker in set(speakers):
            theirs = [
                c.label for s, c in zip(speakers, conditions, strict=True) if s == speaker
            ]
            assert len(set(theirs)) == min(per_speaker, len(TRAINING_CONDITIONS))

    def test_a_speaker_with_more_recordings_than_conditions_repeats_evenly(self) -> None:
        speakers = _corpus(10, len(TRAINING_CONDITIONS) * 2)
        conditions = assign_conditions(speakers)

        for speaker in set(speakers):
            counts = Counter(
                c.label for s, c in zip(speakers, conditions, strict=True) if s == speaker
            )
            assert set(counts.values()) == {2}

    def test_corpus_wide_balance_is_not_sacrificed_for_it(self) -> None:
        """The property the old allocation had must survive the new one."""
        speakers = _uneven_corpus()
        counts = Counter(c.label for c in assign_conditions(speakers))
        assert len(counts) == len(TRAINING_CONDITIONS)
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_no_condition_attaches_to_the_speakers_that_sort_early(self) -> None:
        """The cycle continues across speaker boundaries rather than restarting.

        Restarting at condition zero for every speaker would give every speaker
        the same first condition, which is a worse confound than the one being
        removed: it would make condition zero a property of speakers with few
        recordings.
        """
        speakers = _corpus(40, 3)
        conditions = assign_conditions(speakers)
        firsts = Counter()
        seen: set[str] = set()
        for speaker, condition in zip(speakers, conditions, strict=True):
            if speaker not in seen:
                seen.add(speaker)
                firsts[condition.label] += 1
        assert len(firsts) > 1

    def test_the_allocation_is_the_same_on_every_run(self) -> None:
        speakers = _uneven_corpus(60)
        first = [c.label for c in assign_conditions(speakers)]
        second = [c.label for c in assign_conditions(speakers)]
        assert first == second

    def test_it_does_not_depend_on_the_order_the_corpus_was_scanned_in(self) -> None:
        """Sorted speaker order, so a filesystem walk cannot change the design."""
        grouped = _corpus(20, 4)
        interleaved = [grouped[i] for i in np.argsort([i % 4 for i in range(len(grouped))])]

        def per_speaker(speakers):
            conditions = assign_conditions(speakers)
            return {
                speaker: Counter(
                    c.label
                    for s, c in zip(speakers, conditions, strict=True)
                    if s == speaker
                )
                for speaker in set(speakers)
            }

        assert per_speaker(grouped) == per_speaker(interleaved)

    def test_a_different_seed_moves_the_assignment_but_not_the_design(self) -> None:
        speakers = _corpus(30, 4)
        first = assign_conditions(speakers, seed=1)
        second = assign_conditions(speakers, seed=2)

        assert [c.label for c in first] != [c.label for c in second]
        assert Counter(c.label for c in first) == Counter(c.label for c in second)


class TestAgainstTheGlobalAllocation:
    """The comparison that says why the change was made.

    Both allocations are exactly balanced corpus-wide, so the corpus-level
    summary cannot tell them apart. The per-speaker offset can.
    """

    def test_the_global_allocation_leaves_speakers_with_repeated_conditions(
        self,
    ) -> None:
        speakers = _uneven_corpus()
        balance = condition_balance(speakers, assign_conditions_globally(len(speakers)))
        assert balance["speakers_with_a_repeated_condition"] > 0.5

    def test_the_stratified_allocation_leaves_none(self) -> None:
        speakers = _uneven_corpus()
        balance = condition_balance(speakers, assign_conditions(speakers))
        assert balance["speakers_with_a_repeated_condition"] == 0.0
        assert (
            balance["mean_distinct_conditions_per_speaker"]
            == balance["mean_reachable_conditions_per_speaker"]
        )

    def test_the_per_speaker_channel_offset_is_roughly_halved(self) -> None:
        """The quantity LDA and PLDA absorb as between-speaker variance.

        Not driven to zero: speakers with fewer recordings than there are
        conditions cannot see the whole design, so some offset is unavoidable
        without unbalancing the corpus. Halving it is what the allocation buys.
        """
        speakers = _uneven_corpus()
        stratified = condition_balance(speakers, assign_conditions(speakers))
        globally = condition_balance(speakers, assign_conditions_globally(len(speakers)))

        assert (
            stratified["speaker_mean_bitrate_sd"]
            < 0.6 * (globally["speaker_mean_bitrate_sd"])
        )

    def test_both_are_balanced_corpus_wide_so_that_alone_proves_nothing(self) -> None:
        speakers = _uneven_corpus()
        for conditions in (
            assign_conditions(speakers),
            assign_conditions_globally(len(speakers)),
        ):
            balance = condition_balance(speakers, conditions)
            spread = (
                balance["corpus_condition_count_max"]
                - balance["corpus_condition_count_min"]
            )
            assert spread <= 1.0
