"""Fitting the behavioural stream's backgrounds from a real corpus.

The background is the denominator of every likelihood ratio the stream produces,
so a fault here does not raise — it shifts every number the stream will ever
report, in a direction nobody can see from the output.

One of these tests exists because of a fault exactly like that.
:func:`build_profile` takes an *optional* language identifier, and without one it
cannot label spans, so ``switch_counts`` comes back empty and code-switching —
the most Zambia-specific signal the stream has — is silently never measured. The
first fit written here passed every check and produced an artefact with no switch
data at all. ``test_switches_are_empty_without_an_identifier`` pins the trap and
``test_switches_are_populated_with_an_identifier`` pins the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fit_behaviour_backgrounds import (
    build_identifier,
    chunk,
    derive_function_words,
    interleave,
    load_comparator,
    main,
    pool_counts,
    read_bigc,
    to_background,
)
from viflap.analysis.behaviour.profile import DisfluencyInventory

#: Transcribed speech, so it carries filled pauses. A fixture without them would
#: leave the disfluency component empty and the fit would refuse — correctly, but
#: for a reason that has nothing to do with what most of these tests check.
BEMBA = [
    "eeh Ibumba lyabashitata nabeminina mumbali mumusebo elyo kwati bali na imbwa",
    "Uyu muntu ali na amano kabili mmm elyo bushe pali abantu bonse pa musebo",
    "ati Aba bantu bali mu ng'anda kabili elyo na imbwa yabo iyi ili pa nse",
    "Bushe uyu mwaice ali kwati alefwaya ukulya eeh elyo kuli bamayo bakwe",
]
#: Some sentences carry move cues — an officer, a bank, a hurry. Ordinary
#: description does contain them, which is why the real corpus yields a usable
#: move background from image captions, and a fixture without any would fail the
#: fit for a reason unrelated to what is being tested.
ENGLISH = [
    "uh A group of men is standing on the roadside and a police officer is there",
    "This person is clever and um there are people from the bank who are watching",
    "well These people are in the house and they must hurry outside immediately",
    "Is this child hungry and uh does he want food from his mother over there",
]


def _write_corpus(path: Path, rows: int = 60) -> Path:
    """A BIG-C shaped TSV: the columns the reader looks for, plus noise."""
    lines = ["pair_id\timage\tsentence_id\tbem_transcription\ten_translation"]
    for index in range(rows):
        lines.append(
            f"p{index}\timg{index}.jpg\t{index}\t"
            f"{BEMBA[index % len(BEMBA)]}\t{ENGLISH[index % len(ENGLISH)]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path / "bigc.tsv")


class TestReadingTheCorpus:
    def test_reads_both_languages(self, corpus: Path) -> None:
        bemba, english = read_bigc(corpus)
        assert len(bemba) == 60
        assert len(english) == 60

    def test_refuses_a_corpus_with_no_transcripts(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.tsv"
        path.write_text("pair_id\tbem_transcription\n1\t\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no Bemba transcripts"):
            read_bigc(path)


class TestFunctionWordDerivation:
    def test_takes_the_most_frequent_tokens(self) -> None:
        """There is no published Bemba function word list, so frequency stands in."""
        words = derive_function_words(BEMBA, count=5)

        assert len(words) == 5
        assert "elyo" in words

    def test_pooling_both_languages_covers_both(self) -> None:
        """A Bemba-only inventory counts nothing at all in an English transcript,
        and Zambia banks and polices in English."""
        words = set(derive_function_words([*BEMBA, *ENGLISH], count=40))

        assert words & {"the", "and", "is", "are"}
        assert words & {"elyo", "na", "kabili"}

    def test_is_case_insensitive(self) -> None:
        words = derive_function_words(["The THE the"], count=1)
        assert words == ["the"]


class TestChunking:
    def test_concatenates_to_roughly_the_target_length(self) -> None:
        documents = list(chunk(BEMBA * 20, words_per_chunk=50))

        assert documents
        for document in documents:
            assert len(document.split()) >= 50

    def test_a_short_tail_is_dropped(self) -> None:
        """A stub document would be profiled as though it were an incident."""
        assert list(chunk(["one two three"], words_per_chunk=100)) == []

    def test_a_substantial_tail_is_kept(self) -> None:
        assert len(list(chunk(BEMBA, words_per_chunk=12))) >= 1


class TestInterleaving:
    def test_alternates_between_the_languages(self) -> None:
        mixed = list(interleave(["a", "b"], ["x", "y"]))
        assert mixed == ["a", "x", "b", "y"]

    def test_stops_at_the_shorter_corpus(self) -> None:
        assert list(interleave(["a", "b", "c"], ["x"])) == ["a", "x"]


class TestCodeSwitchDetection:
    """The fault that produced an artefact with no switch data."""

    def _documents(self) -> list[str]:
        return list(chunk(interleave(BEMBA * 20, ENGLISH * 20), words_per_chunk=120))

    def test_switches_are_empty_without_an_identifier(self) -> None:
        """The trap: no identifier, no language labels, no switches, no error.

        This is the behaviour, not the intent — it is pinned so that the absence
        of switch data can never again look like a property of the corpus.
        """
        counts = pool_counts(
            self._documents(),
            frozenset(derive_function_words([*BEMBA, *ENGLISH], 40)),
            DisfluencyInventory(),
            identifier=None,
        )
        assert not counts["switches"]

    def test_switches_are_populated_with_an_identifier(self) -> None:
        counts = pool_counts(
            self._documents(),
            frozenset(derive_function_words([*BEMBA, *ENGLISH], 40)),
            DisfluencyInventory(),
            identifier=build_identifier(BEMBA * 20, ENGLISH * 20),
        )
        assert counts["switches"]

    def test_the_identifier_learns_both_languages(self) -> None:
        identifier = build_identifier(BEMBA * 20, ENGLISH * 20)
        assert identifier.languages == ("bemba", "english")


class TestBackgroundConstruction:
    def test_frequencies_sum_to_one(self) -> None:
        background = to_background({"a": 3, "b": 1}, "test")

        assert sum(background["frequencies"].values()) == pytest.approx(1.0)
        assert background["total_observations"] == 4

    def test_unseen_mass_is_reserved(self) -> None:
        """Without it, a category absent from the background has frequency zero
        and yields an infinite likelihood ratio on a sampling gap."""
        assert to_background({"a": 1}, "test")["unseen_mass"] > 0.0

    def test_an_empty_background_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no observations"):
            to_background({}, "test")


class TestArtefactRoundTrip:
    @pytest.fixture
    def artefact(self, corpus: Path, tmp_path: Path) -> Path:
        output = tmp_path / "backgrounds.json"
        main(
            [
                "--corpus",
                str(corpus),
                "--output",
                str(output),
                "--chunk-words",
                "60",
                "--function-words",
                "40",
            ]
        )
        return output

    def test_all_three_languages_are_fitted(self, artefact: Path) -> None:
        stored = json.loads(artefact.read_text(encoding="utf-8"))
        assert set(stored["languages"]) == {"bemba", "english", "mixed"}

    def test_provenance_is_recorded(self, artefact: Path) -> None:
        """A background whose origin is unrecorded cannot be defended later."""
        provenance = json.loads(artefact.read_text(encoding="utf-8"))["provenance"]

        assert provenance["n_bemba_transcripts"] == 60
        assert "per_language" in provenance
        assert "inter-sentential" in provenance["note"]

    def test_a_comparator_can_be_rebuilt(self, artefact: Path) -> None:
        comparator, function_words = load_comparator(artefact)
        assert function_words
        assert comparator.stream.value == "behavioural"

    def test_the_inventory_travels_with_the_comparator(self, artefact: Path) -> None:
        """Profiling with a different inventory than the background was fitted on
        puts every count into a category the background has never seen, and each
        likelihood ratio then rests on the unseen-mass reservation rather than on
        evidence. Returning them together is what stops that being easy."""
        _, function_words = load_comparator(artefact)
        stored = json.loads(artefact.read_text(encoding="utf-8"))

        assert function_words == frozenset(stored["function_words"])

    def test_a_specific_language_can_be_selected(self, artefact: Path) -> None:
        for language in ("bemba", "english", "mixed"):
            comparator, _ = load_comparator(artefact, language)
            assert comparator is not None

    def test_an_unknown_language_is_refused(self, artefact: Path) -> None:
        with pytest.raises(ValueError, match="no background fitted"):
            load_comparator(artefact, "nyanja")

    def test_the_default_language_is_the_mixed_one(self, artefact: Path) -> None:
        """Zambian speech code-switches, so the mixed background is the one that
        matches the deployment unless the caller knows otherwise."""
        stored = json.loads(artefact.read_text(encoding="utf-8"))
        assert stored["default_language"] == "mixed"
