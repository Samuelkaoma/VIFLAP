"""Estimate the behavioural stream's background populations from real Bemba speech.

Every likelihood ratio the behavioural comparator produces is a ratio against a
background: how surprising is it that two incidents share this feature, given how
often it occurs in the relevant population? Without a background estimated from
real material, the stream cannot produce a number at all, and with a background
estimated from the wrong population it produces a confident wrong one.

The corpus
----------
BIG-C: 92,116 transcribed utterances of conversational Bemba, about 867,000
tokens. It is used here for the one thing it is unimpeachable at. Its
``speaker_id`` field is not a person — 37 of its 74 values carry both male and
female labels in near-equal proportion — which disqualifies it as labelled
speech data. **Background estimation needs text, not identity**, so that defect
is irrelevant to this use, and what remains is the largest body of real Zambian
conversational Bemba available.

It is ordinary conversation rather than fraud calls, and that is the correct
shape for a background. The background answers "how often does this appear in
speech generally", so a rhetorical move that never occurs in ordinary talk gets a
low frequency and counts heavily when two incidents share it. A background
estimated from fraud calls would say those moves are common and would throw away
the evidence.

Three backgrounds, because Zambia is not monolingual
----------------------------------------------------
English is Zambia's official language and the language of banking, policing and
officialdom — which is to say, the language of every pretext an
agent-impersonation script relies on. A background estimated only from Bemba
would be the wrong denominator for material that is substantially English, and
most Zambian speech is neither: it code-switches, constantly.

So three background sets are fitted and the caller chooses:

``bemba``
    Idiolect from the Bemba transcripts.
``english``
    Idiolect from the English translations. These are translations rather than
    natural speech, so their function word statistics carry translationese —
    but they were produced by Zambian annotators, so the register is closer to
    Zambian English than any British or American corpus would be.
``mixed``
    Utterances alternated between the two, which is what the deployment actually
    faces.

Script moves are estimated from the **English translations** in every case,
because the move inventory is defined by semantic cues (``"from the bank"``,
``"officer"``) that are English strings. Moves are semantic and translate;
idiolect is surface form and does not. A Bemba cue inventory would be better and
needs a Bemba speaker to write it.

What the mixed background does and does not represent
-----------------------------------------------------
It alternates whole utterances, so it represents **inter-sentential** switching:
one turn in Bemba, the next in English. That is real and common.

It does **not** represent intra-sentential switching — English nouns inside
Bemba clause structure, which is the more frequent form in Zambian speech and the
one the sliding-window language identifier exists to catch. Synthesising that
faithfully requires the grammatical constraints of real code-switching, and
approximating it by shuffling words would produce a background for a language
nobody speaks. The switch counts here are therefore a floor, and a genuinely
code-switched Zambian corpus would raise them.

Function words are derived, not assumed
---------------------------------------
There is no published Bemba function word list. They are taken here as the most
frequent tokens in the corpus, which is standard practice for a low-resource
language and is what the frequency distribution supports: the top of the Bemba
list is ``na``, ``elyo``, ``kwati``, ``kuli``, ``uyu``, ``kabili``, ``bushe`` —
conjunctions, demonstratives, locatives and a question particle, which is exactly
the class wanted.

Bemba is agglutinative, so grammatical information that English carries in
separate words is carried here in prefixes and infixes bound to the stem. The
corpus has 90,128 types over 867,000 tokens, a ratio far above English, and much
of that is morphology rather than vocabulary. Whole-word function-word lists
therefore capture less of the grammar than they would in English, and the
character n-gram inventory carries correspondingly more of it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from viflap.analysis.behaviour.profile import (
    BehaviouralComparator,
    DisfluencyInventory,
    build_profile,
)
from viflap.analysis.behaviour.text import LanguageIdentifier
from viflap.analysis.patterns.conjugate import BackgroundPopulation

#: Tokens taken as function words, by corpus frequency. Large enough to reach
#: past the content words that top any conversational corpus — BIG-C is grounded
#: on images, so "imbwa" (dog) and "abantu" (people) rank high — and small enough
#: that the tail stays frequent enough to estimate.
DEFAULT_FUNCTION_WORD_COUNT = 150

#: Words per pooled document. The comparator refuses transcripts below forty
#: words, and the background must be estimated from documents of the same shape
#: as the ones it will judge; single BIG-C utterances are far shorter than an
#: incident transcript, so they are concatenated to a comparable length.
DEFAULT_CHUNK_WORDS = 250


def read_bigc(path: Path) -> tuple[list[str], list[str]]:
    """Return the Bemba transcripts and their English translations."""
    bemba: list[str] = []
    english: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source = (row.get("bem_transcription") or "").strip()
            translation = (row.get("en_translation") or "").strip()
            if source:
                bemba.append(source)
            if translation:
                english.append(translation)
    if not bemba:
        raise ValueError(f"no Bemba transcripts found in {path}")
    return bemba, english


def derive_function_words(transcripts: Sequence[str], count: int) -> list[str]:
    """The most frequent tokens, taken as the function word inventory."""
    frequencies: Counter[str] = Counter()
    for transcript in transcripts:
        frequencies.update(token.lower() for token in transcript.split())
    return [token for token, _ in frequencies.most_common(count)]


def chunk(transcripts: Iterable[str], words_per_chunk: int) -> Iterator[str]:
    """Concatenate transcripts into documents of roughly equal length."""
    held: list[str] = []
    held_words = 0
    for transcript in transcripts:
        held.append(transcript)
        held_words += len(transcript.split())
        if held_words >= words_per_chunk:
            yield " ".join(held)
            held, held_words = [], 0
    if held_words >= words_per_chunk // 2:
        yield " ".join(held)


def build_identifier(bemba: Sequence[str], english: Sequence[str]) -> LanguageIdentifier:
    """Train a language identifier on both sides of the corpus.

    Without one, :func:`build_profile` cannot label spans and every switch count
    is zero — silently. The code-switching feature is the most Zambia-specific
    signal the behavioural stream has, and an identifier that was never supplied
    turns it off without any error to notice.
    """
    return (
        LanguageIdentifier().add_language("bemba", bemba).add_language("english", english)
    )


def interleave(bemba: Sequence[str], english: Sequence[str]) -> Iterator[str]:
    """Alternate utterances between languages, approximating a code-switched turn.

    Inter-sentential switching only — see the module docstring on what this does
    not represent.
    """
    for index in range(min(len(bemba), len(english))):
        yield bemba[index]
        yield english[index]


def pool_counts(
    documents: Iterable[str],
    function_words: frozenset[str],
    disfluencies: DisfluencyInventory,
    identifier: LanguageIdentifier | None = None,
) -> dict[str, Counter[str]]:
    """Pool every count the comparator's backgrounds are estimated from.

    Profiles are built with the same function the comparator uses, rather than
    counting tokens directly here. A background counted by one code path and
    consumed by another drifts the moment either changes its tokenisation, and
    the drift is silent — the numbers still look like counts.
    """
    pooled = {
        "function_words": Counter[str](),
        "disfluencies": Counter[str](),
        "moves": Counter[str](),
        "character_ngrams": Counter[str](),
        "switches": Counter[str](),
    }
    n_documents = 0
    for document in documents:
        try:
            profile = build_profile(document, function_words, disfluencies, identifier)
        except Exception:
            # Short or unparseable chunks are skipped. A background is a
            # population estimate; one document more or less does not change it,
            # and aborting the fit over one would.
            continue
        n_documents += 1
        pooled["function_words"].update(profile.function_word_counts)
        pooled["disfluencies"].update(profile.disfluency_counts)
        pooled["moves"].update(profile.move_counts)
        pooled["character_ngrams"].update(profile.character_ngram_counts)
        pooled["switches"].update(profile.switch_counts)

    if not n_documents:
        raise ValueError("no document survived profiling; is the corpus empty?")
    pooled["_meta"] = Counter({"n_documents": n_documents})
    return pooled


def to_background(
    counts: Mapping[str, int], description: str, component: str = "background"
) -> dict[str, object]:
    """Normalise pooled counts into a serialisable background population.

    An empty component is refused rather than filled in. A background of nothing
    cannot be normalised, and substituting a uniform one would put every
    likelihood ratio for that component on a distribution the corpus never
    supported — silently, since the artefact would still load and the stream
    would still return numbers.

    The usual cause is a mismatch rather than a missing corpus: the disfluency
    inventory is language-specific, so a corpus in a language it does not cover
    yields no disfluencies at all, and written material such as a translation
    yields none because filled pauses are edited out of writing.
    """
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(
            f"cannot build the {component!r} background for {description}: the "
            f"corpus produced no observations of it. If this is the disfluency "
            f"component, the marker inventory probably does not match the "
            f"corpus language, or the text is written rather than transcribed."
        )
    return {
        "frequencies": {key: value / total for key, value in counts.items()},
        "total_observations": total,
        "description": description,
        "unseen_mass": 0.01,
    }


def load_comparator(
    path: Path, language: str | None = None
) -> tuple[BehaviouralComparator, frozenset[str]]:
    """Rebuild a comparator, and the inventory to profile with, from an artefact.

    Both are returned together because they are one unit: a profile built with a
    different function word inventory than the background was estimated from
    produces counts over categories the background has never seen, and every
    likelihood ratio then rests on the unseen-mass reservation rather than on
    evidence. Handing the caller the comparator alone would make that mismatch
    easy to cause and impossible to notice.
    """
    artefact = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = language or artefact.get("default_language", "mixed")
    if chosen not in artefact["languages"]:
        raise ValueError(
            f"no background fitted for {chosen!r}; available: "
            f"{sorted(artefact['languages'])}"
        )
    backgrounds = artefact["languages"][chosen]

    def population(name: str) -> BackgroundPopulation:
        entry = backgrounds[name]
        return BackgroundPopulation(
            frequencies=entry["frequencies"],
            total_observations=entry["total_observations"],
            description=entry["description"],
            unseen_mass=entry["unseen_mass"],
        )

    comparator = BehaviouralComparator(
        function_word_background=population("function_words"),
        disfluency_background=population("disfluencies"),
        move_background=population("moves"),
        ngram_background=population("character_ngrams"),
        switch_background=population("switches") if "switches" in backgrounds else None,
    )
    return comparator, frozenset(artefact["function_words"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="BIG-C datasheet TSV")
    parser.add_argument(
        "--output", type=Path, default=Path("models/behaviour_backgrounds.json")
    )
    parser.add_argument("--function-words", type=int, default=DEFAULT_FUNCTION_WORD_COUNT)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    arguments = parser.parse_args(argv)

    print(f"reading {arguments.corpus}", flush=True)
    bemba, english = read_bigc(arguments.corpus)
    print(f"  {len(bemba)} Bemba transcripts, {len(english)} English translations")

    # Derived from both sides pooled: the inventory has to cover whichever
    # language the material turns out to be in, and a Bemba-only list would
    # count nothing at all in an English transcript.
    function_words = derive_function_words([*bemba, *english], arguments.function_words)
    print(f"  derived {len(function_words)} function words: {function_words[:10]}")

    disfluencies = DisfluencyInventory()
    inventory = frozenset(function_words)

    print("training the language identifier on both languages", flush=True)
    identifier = build_identifier(bemba, english)
    print(f"  languages: {identifier.languages}")

    corpora: dict[str, list[str]] = {
        "bemba": list(chunk(bemba, arguments.chunk_words)),
        "english": list(chunk(english, arguments.chunk_words)),
        "mixed": list(chunk(interleave(bemba, english), arguments.chunk_words)),
    }

    fitted: dict[str, dict[str, object]] = {}
    provenance: dict[str, object] = {}
    for language, documents in corpora.items():
        print(f"pooling counts for {language}", flush=True)
        counts = pool_counts(documents, inventory, disfluencies, identifier)
        n_documents = counts["_meta"]["n_documents"]
        print(f"  {n_documents} documents, {len(counts['switches'])} switch types")

        backgrounds: dict[str, object] = {
            "function_words": to_background(
                counts["function_words"], f"BIG-C {language}", "function_words"
            ),
            "disfluencies": to_background(
                counts["disfluencies"], f"BIG-C {language}", "disfluencies"
            ),
            "character_ngrams": to_background(
                counts["character_ngrams"], f"BIG-C {language}", "character_ngrams"
            ),
            # Always from the English translations: the cue inventory is English,
            # and a move is semantic rather than a property of surface form.
            "moves": to_background(
                pool_counts(corpora["english"], inventory, disfluencies, identifier)[
                    "moves"
                ]
                if language != "english"
                else counts["moves"],
                "BIG-C English translations",
                "moves",
            ),
        }
        if counts["switches"]:
            backgrounds["switches"] = to_background(
                counts["switches"], f"BIG-C {language}", "switches"
            )
        fitted[language] = backgrounds
        provenance[language] = {
            "documents": n_documents,
            "switch_types": len(counts["switches"]),
        }

    artefact: dict[str, object] = {
        "provenance": {
            "corpus": str(arguments.corpus),
            "n_bemba_transcripts": len(bemba),
            "n_english_translations": len(english),
            "chunk_words": arguments.chunk_words,
            "per_language": provenance,
            "note": (
                "Idiolect estimated per language from surface form; script moves "
                "always from the English translations, because the cue inventory "
                "is English. The mixed set alternates whole utterances, so it "
                "represents inter-sentential switching only. Speaker identity in "
                "this corpus is unreliable and is not used: background estimation "
                "needs text, not identity."
            ),
        },
        "function_words": function_words,
        "disfluency_markers": sorted(disfluencies.markers),
        "default_language": "mixed",
        "languages": fitted,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    print(f"wrote {arguments.output}", flush=True)

    for language, backgrounds in fitted.items():
        summary = ", ".join(
            f"{name} {len(entry['frequencies'])}"  # type: ignore[index]
            for name, entry in backgrounds.items()
        )
        print(f"  {language}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
