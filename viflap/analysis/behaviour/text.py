"""Tokenisation and language identification for code-switched telephony.

Zambian telephony is multilingual and code-switched *within utterances*: a
single sentence may move between English, Bemba and Nyanja and back. That is not
an inconvenience to be normalised away — the pattern of switching is itself
behavioural evidence, and where a speaker switches is characteristic of them.

Language identification by character n-grams
--------------------------------------------
Word-list identification fails on this material for two reasons. Code-switched
speech mixes vocabularies within a clause, so any window long enough to contain
several words spans a switch. And transcripts of telephony contain
non-standard spellings, truncations and transcriber variation that a word list
does not contain.

Character n-grams degrade gracefully under both. They are trained per language
from whatever text is available, they need no lexicon, and they identify a
language from a short span. Bemba and Nyanja are morphologically rich Bantu
languages whose characteristic prefixes and vowel patterns are strongly
expressed in character n-grams, so this works better here than it would for two
closely related European languages.

Nothing in this module hard-codes a word list for any language. The previous
implementation contained a handful of Bemba and Nyanja words chosen by hand,
which would identify those languages only in the sentences that happen to
contain them and would silently mislabel everything else as English.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from viflap.domain.errors import (
    InsufficientDataError,
    InvalidEvidenceError,
    ModelNotTrainedError,
)

__all__ = [
    "CharacterNGramLanguageModel",
    "LanguageIdentifier",
    "LanguageSpan",
    "Token",
    "tokenise",
]

_TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Token:
    """One token with its position in the source text."""

    text: str
    start: int
    end: int

    @property
    def is_word(self) -> bool:
        return self.text[:1].isalpha()

    @property
    def normalised(self) -> str:
        return self.text.lower()


def tokenise(text: str) -> list[Token]:
    """Split text into word, number and punctuation tokens with offsets.

    Offsets are retained because the behavioural analysis needs to know *where*
    something occurred — a filled pause at the start of a turn is a different
    behaviour from one mid-clause — and recovering position by re-searching for
    the token is ambiguous whenever it appears more than once.

    Apostrophes inside words are kept, so contractions stay single tokens.
    Splitting them creates spurious high-frequency fragments that dominate any
    function-word profile.
    """
    return [
        Token(text=found.group(0), start=found.start(), end=found.end())
        for found in _TOKEN_PATTERN.finditer(text)
    ]


class CharacterNGramLanguageModel:
    """Character n-gram model with Witten-Bell smoothing.

    Witten-Bell rather than add-one. Add-one smoothing distributes probability
    mass uniformly across every unseen continuation, which for a character model
    over a large alphabet means most of the mass goes to sequences the language
    never produces. Witten-Bell allocates it in proportion to how *productive* a
    context has been — a context that has been followed by many distinct
    characters is likely to admit another, one that has always been followed by
    the same character is not. That is the right prior for morphology, which is
    what distinguishes these languages.
    """

    def __init__(self, order: int = 4) -> None:
        if order < 2:
            raise InvalidEvidenceError(
                "character n-gram order must be at least two", order=order
            )
        self._order = order
        self._context_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._context_totals: dict[str, int] = defaultdict(int)
        self._alphabet: set[str] = set()
        self._trained = False

    @property
    def order(self) -> int:
        return self._order

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, texts: Iterable[str]) -> CharacterNGramLanguageModel:
        total_characters = 0
        for text in texts:
            padded = self._pad(text)
            self._alphabet.update(padded)
            for index in range(self._order - 1, len(padded)):
                context = padded[index - self._order + 1 : index]
                character = padded[index]
                self._context_counts[context][character] += 1
                self._context_totals[context] += 1
                total_characters += 1

        if total_characters < 200:
            raise InsufficientDataError(
                "too little text to train a character language model",
                n_characters=total_characters,
                required=200,
            )
        self._trained = True
        return self

    def _pad(self, text: str) -> str:
        return "" * (self._order - 1) + text.lower() + ""

    def log_probability(self, text: str) -> float:
        """Total log probability of ``text``, in nats."""
        if not self._trained:
            raise ModelNotTrainedError("character language model has not been trained")

        padded = self._pad(text)
        alphabet_size = max(len(self._alphabet), 2)
        total = 0.0

        for index in range(self._order - 1, len(padded)):
            context = padded[index - self._order + 1 : index]
            character = padded[index]
            total += self._character_log_probability(context, character, alphabet_size)
        return total

    def _character_log_probability(
        self, context: str, character: str, alphabet_size: int
    ) -> float:
        """Witten-Bell smoothed probability, backing off to shorter contexts."""
        counts = self._context_counts.get(context)
        if counts is None or self._context_totals.get(context, 0) == 0:
            if len(context) == 0:
                return math.log(1.0 / alphabet_size)
            return self._character_log_probability(context[1:], character, alphabet_size)

        total = self._context_totals[context]
        distinct = len(counts)
        # Witten-Bell: mass reserved for unseen continuations is proportional to
        # the number of distinct continuations already seen in this context.
        lambda_weight = total / (total + distinct)

        seen = counts.get(character, 0) / total
        if len(context) == 0:
            backoff = 1.0 / alphabet_size
        else:
            backoff = math.exp(
                self._character_log_probability(context[1:], character, alphabet_size)
            )

        probability = lambda_weight * seen + (1.0 - lambda_weight) * backoff
        return math.log(max(probability, 1e-300))


@dataclass(frozen=True, slots=True)
class LanguageSpan:
    """A contiguous span attributed to one language."""

    language: str
    start: int
    end: int
    confidence: float
    """Posterior probability of the assigned language over the alternatives.
    Low values mark spans the identifier could not resolve — common at genuine
    switch points, and reported rather than hidden."""


class LanguageIdentifier:
    """Identifies language spans within code-switched text.

    Operates over a sliding window of tokens rather than whole sentences,
    because a sentence is exactly the wrong unit here: intra-sentential
    switching is the phenomenon being measured, and a sentence-level label
    averages it away.
    """

    def __init__(self, window_tokens: int = 4) -> None:
        self._models: dict[str, CharacterNGramLanguageModel] = {}
        self._window = window_tokens

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def add_language(
        self, language: str, texts: Iterable[str], order: int = 4
    ) -> LanguageIdentifier:
        """Train and register a model for one language."""
        self._models[language] = CharacterNGramLanguageModel(order).train(texts)
        return self

    def identify_spans(self, text: str) -> list[LanguageSpan]:
        """Attribute contiguous spans of ``text`` to languages.

        Adjacent windows receiving the same label are merged, so the output is a
        span sequence rather than a per-window label sequence. The number of
        transitions in that sequence is the code-switch count, which is the
        behavioural quantity of interest.
        """
        if len(self._models) < 2:
            raise ModelNotTrainedError(
                "language identification needs at least two trained languages"
            )

        tokens = [token for token in tokenise(text) if token.is_word]
        if not tokens:
            return []

        spans: list[LanguageSpan] = []
        for start_index in range(0, len(tokens), self._window):
            window = tokens[start_index : start_index + self._window]
            if not window:
                continue
            fragment = " ".join(token.normalised for token in window)

            # Length-normalised log probability: without normalisation a longer
            # window is less probable under every model, and comparing raw
            # totals across windows of different length is meaningless.
            scores = {
                language: model.log_probability(fragment) / max(len(fragment), 1)
                for language, model in self._models.items()
            }
            best = max(scores, key=lambda key: scores[key])

            values = list(scores.values())
            peak = max(values)
            normaliser = sum(math.exp(value - peak) for value in values)
            confidence = 1.0 / normaliser if normaliser > 0 else 0.0

            spans.append(
                LanguageSpan(
                    language=best,
                    start=window[0].start,
                    end=window[-1].end,
                    confidence=float(confidence),
                )
            )

        return _merge_adjacent(spans)


def _merge_adjacent(spans: Sequence[LanguageSpan]) -> list[LanguageSpan]:
    """Merge consecutive spans sharing a language."""
    if not spans:
        return []
    merged = [spans[0]]
    for span in spans[1:]:
        previous = merged[-1]
        if span.language == previous.language:
            merged[-1] = LanguageSpan(
                language=previous.language,
                start=previous.start,
                end=span.end,
                confidence=min(previous.confidence, span.confidence),
            )
        else:
            merged.append(span)
    return merged
