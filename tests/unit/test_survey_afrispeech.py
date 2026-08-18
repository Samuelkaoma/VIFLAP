"""Counting speakers in a corpus survey, and the ways a count can flatter.

§8's whole argument is that hour counts overstate what a corpus offers a speaker
recognition system, so a survey that miscounted speakers would corrupt exactly
the claim it exists to support. Two miscounts are plausible and neither shows in
the output: counting *utterances* where speakers were meant, and counting a
speaker once per split so anyone appearing in both train and test is counted
twice.

The absence tests matter as much as the presence ones. A survey that silently
returned nothing for a language nobody has recorded would be indistinguishable
from one that never looked, and §8's finding is precisely an absence.
"""

from __future__ import annotations

import pytest

from scripts.survey_afrispeech import USABLE_SECONDS, ZAMBIAN_LANGUAGES, summarise


def _row(speaker: str, country: str, accent: str, duration: float) -> dict[str, str]:
    return {
        "user_ids": speaker,
        "country": country,
        "accent": accent,
        "duration": str(duration),
    }


class TestSpeakerCounting:
    def test_speakers_are_counted_not_utterances(self) -> None:
        """One person with fifty recordings is one speaker."""
        rows = [_row("a", "ZM", "bemba", 10.0) for _ in range(50)]
        summary = summarise(rows)
        assert summary["n_utterances"] == 50
        assert summary["n_speakers"] == 1

    def test_a_speaker_appearing_in_two_splits_is_counted_once(self) -> None:
        """``summarise`` takes the splits already concatenated, which is exactly
        where double counting would happen if the caller pooled naively."""
        rows = [_row("a", "NG", "igbo", 30.0), _row("a", "NG", "igbo", 40.0)]
        summary = summarise(rows)
        assert summary["n_speakers"] == 1
        assert summary["by_country"]["NG"]["speakers"] == 1

    def test_duration_accumulates_across_a_speaker_s_utterances(self) -> None:
        """The usable threshold is on *total* speech, so per-utterance testing
        would refuse everyone in a corpus of short clips."""
        rows = [_row("a", "NG", "igbo", 21.0) for _ in range(3)]
        summary = summarise(rows)
        assert summary["by_country"]["NG"]["usable_speakers"] == 1

    def test_a_speaker_below_the_threshold_is_counted_but_not_usable(self) -> None:
        rows = [_row("a", "NG", "igbo", USABLE_SECONDS - 1.0)]
        summary = summarise(rows)
        assert summary["by_country"]["NG"]["speakers"] == 1
        assert summary["by_country"]["NG"]["usable_speakers"] == 0

    def test_the_threshold_is_inclusive(self) -> None:
        rows = [_row("a", "NG", "igbo", USABLE_SECONDS)]
        assert summarise(rows)["by_country"]["NG"]["usable_speakers"] == 1


class TestNormalisation:
    def test_country_codes_are_matched_case_and_space_insensitively(self) -> None:
        """A corpus assembled from several sources spells its fields several
        ways, and a country that failed to match would read as an absence."""
        rows = [_row("a", " zm ", "bemba", 90.0), _row("b", "ZM", "bemba", 90.0)]
        summary = summarise(rows)
        assert summary["zambia_present"]
        assert summary["n_zambian_speakers"] == 2

    def test_accent_labels_are_matched_case_insensitively(self) -> None:
        rows = [_row("a", "MW", "Chichewa", 90.0)]
        assert summarise(rows)["zambian_languages"]["chichewa"]["speakers"] == 1


class TestAbsenceIsReported:
    def test_every_named_language_appears_even_at_zero(self) -> None:
        """§8's finding is an absence, so the absence has to be in the output.

        A survey reporting only the languages it found cannot be told apart from
        one that never asked about the others.
        """
        summary = summarise([_row("a", "NG", "igbo", 90.0)])
        assert set(summary["zambian_languages"]) == set(ZAMBIAN_LANGUAGES)
        for name in ZAMBIAN_LANGUAGES:
            assert summary["zambian_languages"][name]["speakers"] == 0

    def test_a_corpus_with_no_zambian_speakers_says_so_explicitly(self) -> None:
        summary = summarise([_row("a", "NG", "igbo", 90.0)])
        assert summary["zambia_present"] is False
        assert summary["n_zambian_speakers"] == 0

    def test_a_language_present_under_another_country_still_counts(self) -> None:
        """Chichewa is Zambian and Malawian; the real corpus has one speaker of
        it and they are in Malawi. Language and country are separate questions
        and the survey must not collapse them."""
        summary = summarise([_row("a", "MW", "chichewa", 545.0)])
        assert summary["zambian_languages"]["chichewa"]["speakers"] == 1
        assert summary["zambia_present"] is False


class TestReportedHours:
    def test_hours_are_seconds_over_thirty_six_hundred(self) -> None:
        rows = [_row("a", "NG", "igbo", 1800.0), _row("b", "NG", "igbo", 1800.0)]
        assert summarise(rows)["by_country"]["NG"]["hours"] == pytest.approx(1.0)

    def test_countries_are_ordered_by_speaker_count(self) -> None:
        rows = [_row(f"n{i}", "NG", "igbo", 90.0) for i in range(5)]
        rows += [_row("k0", "KE", "luo", 90.0)]
        assert list(summarise(rows)["by_country"]) == ["NG", "KE"]
