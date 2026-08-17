"""Behavioural profiles and comparison.

The behavioural stream is not a supplement to the acoustic one. It answers a
different question, and it is the only defence against the attack the acoustic
stream cannot survive.

An organised group defeats voice-based linkage completely by rotating who
speaks. No acoustic system, however good, links two incidents conducted by two
different people. But the *operation* persists: the same script, the same
pretext sequence, the same escalation when the victim hesitates, the same
closing move. Script structure characterises the operation rather than the
speaker, so it links incidents across operators within one group — which is
frequently the linkage an investigator most needs.

The profile therefore separates two things that a single "behavioural score"
would merge:

**Idiolect** — function word usage, disfluency habits, code-switching patterns.
Speaker-specific. Defeated by delegation.

**Script structure** — pretext category, move sequence, characteristic phrasing.
Operation-specific. Survives delegation.

They are scored separately and reported separately, because "these two incidents
were run by the same operation but probably not the same person" is a finding,
and one that a merged score cannot express.

Function words and why they carry the idiolect
----------------------------------------------
Content words are determined by the topic; a bank impersonation script contains
"account" whoever is running it. Function words — articles, prepositions,
auxiliaries, discourse particles — are produced below the level of conscious
choice, are largely independent of topic, and are stable within a speaker. This
is the foundation of authorship attribution, and it transfers to speech
transcripts. It is also why the disfluency inventory matters: filled pauses are
over-learned motor habits, and a speaker concentrating on a deception has no
attention left to modify them.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.behaviour.text import LanguageIdentifier, tokenise
from viflap.analysis.patterns.conjugate import (
    BackgroundPopulation,
    DirichletMultinomialComparator,
    NormalInverseGammaComparator,
)
from viflap.domain.errors import InsufficientDataError
from viflap.domain.evidence import EvidenceStream

__all__ = [
    "BehaviouralComparator",
    "BehaviouralProfile",
    "BehaviouralScore",
    "DisfluencyInventory",
    "ScriptMove",
    "segment_script_moves",
]


@dataclass(frozen=True, slots=True)
class DisfluencyInventory:
    """Filled pauses and hesitation markers, per language.

    Supplied as configuration rather than hard-coded, because the inventory is
    language-specific and a system deployed elsewhere needs a different one. The
    defaults cover English and the principal Zambian languages and are a
    starting point to be replaced by an inventory derived from the corpus.
    """

    markers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                # English
                "um",
                "uh",
                "er",
                "erm",
                "ah",
                "eh",
                "hmm",
                "mm",
                "like",
                "well",
                "so",
                "right",
                "okay",
                # Bemba / Nyanja / general Zambian discourse markers
                "eeh",
                "aah",
                "mmm",
                "ati",
                "sha",
                "iwe",
                "kansi",
                "nomba",
                "bwanji",
                "shani",
            }
        )
    )

    def contains(self, token: str) -> bool:
        return token.lower() in self.markers


@dataclass(frozen=True, slots=True)
class ScriptMove:
    """One rhetorical move in a social-engineering script."""

    label: str
    position: float
    """Where the move occurs, as a fraction of the transcript. Sequence
    *and* placement matter: threatening at the outset is a different operation
    from threatening only after resistance."""


#: Rhetorical moves, with the cues that indicate them. Cues are configuration,
#: not a classifier — an honest description of what this is. A trained move
#: classifier would be better and needs labelled data that does not yet exist;
#: until then these are stated openly so their coverage can be judged, rather
#: than buried in a method that implies more than it delivers.
DEFAULT_MOVE_CUES: Mapping[str, tuple[str, ...]] = {
    "authority_claim": (
        "officer",
        "police",
        "manager",
        "head office",
        "from the bank",
        "customer care",
        "security department",
    ),
    "urgency": (
        "immediately",
        "right now",
        "urgent",
        "expire",
        "within",
        "quickly",
        "before",
        "last chance",
    ),
    "threat": ("arrest", "court", "jail", "blocked", "frozen", "suspended", "legal action"),
    "reassurance": (
        "don't worry",
        "do not worry",
        "safe",
        "secure",
        "no problem",
        "trust me",
        "standard procedure",
    ),
    "credential_request": (
        "pin",
        "password",
        "code",
        "otp",
        "one time",
        "confirm your",
        "verify your",
        "read me",
    ),
    "instruction": ("go to", "dial", "press", "send", "transfer", "deposit", "you must"),
    "reward": ("won", "prize", "lucky", "selected", "promotion", "bonus", "reward"),
    "closing": (
        "thank you",
        "goodbye",
        "call you back",
        "we will call",
        "stay on the line",
    ),
}


def segment_script_moves(
    text: str, cues: Mapping[str, tuple[str, ...]] | None = None
) -> list[ScriptMove]:
    """Identify rhetorical moves and where they occur.

    Cue-based, with the limitation named in :data:`DEFAULT_MOVE_CUES`. Matching
    is on lowercased text with position recorded as a fraction of the
    transcript, so the sequence is comparable between a ninety-second call and a
    six-minute one — which matters, because the same script run against a
    resistant victim takes longer without becoming a different script.
    """
    cues = cues or DEFAULT_MOVE_CUES
    lowered = text.lower()
    length = max(len(lowered), 1)

    moves: list[ScriptMove] = []
    for label, phrases in cues.items():
        for phrase in phrases:
            position = lowered.find(phrase)
            while position != -1:
                moves.append(ScriptMove(label=label, position=position / length))
                position = lowered.find(phrase, position + 1)

    moves.sort(key=lambda move: move.position)
    return moves


@dataclass(frozen=True, slots=True)
class BehaviouralProfile:
    """An incident's transcript reduced to behavioural observations."""

    function_word_counts: Mapping[str, int]
    disfluency_counts: Mapping[str, int]
    move_counts: Mapping[str, int]
    move_sequence: tuple[str, ...]
    switch_counts: Mapping[str, int]
    """Counts of language-pair transitions, e.g. ``"english>bemba"``."""

    character_ngram_counts: Mapping[str, int]
    log_utterance_lengths: NDArray[np.float64]
    n_tokens: int
    n_words: int

    @property
    def disfluency_rate(self) -> float:
        """Disfluencies per hundred words."""
        return 100.0 * sum(self.disfluency_counts.values()) / max(self.n_words, 1)

    @property
    def switch_rate(self) -> float:
        """Language switches per hundred words."""
        return 100.0 * sum(self.switch_counts.values()) / max(self.n_words, 1)


def build_profile(
    transcript: str,
    function_words: frozenset[str],
    disfluencies: DisfluencyInventory | None = None,
    identifier: LanguageIdentifier | None = None,
    ngram_order: int = 4,
    min_words: int = 40,
) -> BehaviouralProfile:
    """Reduce a transcript to a behavioural profile.

    Raises
    ------
    InsufficientDataError
        Below ``min_words``. Function word profiles from short texts are
        dominated by sampling noise: forty words contain perhaps fifteen
        function word tokens, and no stable distribution is recoverable from
        that.
    """
    disfluencies = disfluencies or DisfluencyInventory()
    tokens = tokenise(transcript)
    words = [token for token in tokens if token.is_word]

    if len(words) < min_words:
        raise InsufficientDataError(
            "transcript is too short for a behavioural profile",
            n_words=len(words),
            required=min_words,
        )

    normalised = [token.normalised for token in words]
    function_counts = Counter(word for word in normalised if word in function_words)
    disfluency_counts = Counter(word for word in normalised if disfluencies.contains(word))

    moves = segment_script_moves(transcript)
    move_counts = Counter(move.label for move in moves)
    move_sequence = tuple(move.label for move in moves)

    switch_counts: Counter[str] = Counter()
    if identifier is not None and len(identifier.languages) >= 2:
        spans = identifier.identify_spans(transcript)
        for earlier, later in zip(spans, spans[1:], strict=False):
            switch_counts[f"{earlier.language}>{later.language}"] += 1

    # Character n-grams capture sub-word habits — morphology, characteristic
    # spellings, phrasing independent of the transcriber — that word-level
    # counts miss entirely in agglutinative languages.
    #
    # They are taken **non-overlapping**. Sliding the window one character at a
    # time yields n times as many n-grams from the same text, and each shares
    # n-1 characters with its neighbour. Treating those as independent
    # multinomial draws counts the same characters repeatedly, and the
    # likelihood ratio inflates by roughly a factor of n in the exponent — which
    # for a 4-gram over a few hundred characters is orders of magnitude. That is
    # the same "counting one fact several times" error the fusion layer exists
    # to correct, occurring inside a single stream where fusion cannot see it.
    #
    # Non-overlapping windows are a blunt correction and they discard
    # information. The alternative — an effective-sample-size adjustment to the
    # concentration — needs the autocorrelation estimated per language, which
    # the corpus does not yet support. This errs toward understating, which is
    # the direction to err in.
    lowered = transcript.lower()
    ngram_counts = Counter(
        lowered[index : index + ngram_order]
        for index in range(0, max(len(lowered) - ngram_order + 1, 0), ngram_order)
    )

    # Utterance lengths, split on sentence-final punctuation. Length distribution
    # is a stable prosodic-syntactic habit and survives topic change.
    utterances = [
        segment.strip()
        for segment in __import__("re").split(r"[.!?]+", transcript)
        if segment.strip()
    ]
    lengths = np.array(
        [max(len(tokenise(utterance)), 1) for utterance in utterances], dtype=np.float64
    )

    return BehaviouralProfile(
        function_word_counts=dict(function_counts),
        disfluency_counts=dict(disfluency_counts),
        move_counts=dict(move_counts),
        move_sequence=move_sequence,
        switch_counts=dict(switch_counts),
        character_ngram_counts=dict(ngram_counts),
        log_utterance_lengths=np.log(lengths),
        n_tokens=len(tokens),
        n_words=len(words),
    )


@dataclass(frozen=True, slots=True)
class BehaviouralScore:
    """A behavioural comparison, decomposed by what it speaks to."""

    idiolect_log_lr: float
    """Speaker-specific evidence: function words, disfluencies, code-switching,
    utterance lengths and character n-grams. Defeated if the group rotates who
    speaks.

    Character n-grams sat in ``script_log_lr`` until they were measured. They
    are the authorship literature's strongest single feature for identifying a
    person, and here they were large enough to reverse the sign of the component
    that exists to be person-independent. They also carry a script's fixed
    wording, so their presence here is the better of two imperfect placements
    rather than a clean one — see the note in
    :meth:`BehaviouralComparator.score`."""

    script_log_lr: float
    """Operation-specific evidence: move inventory and move sequence. Survives
    delegation, and is what links incidents across operators in one group."""

    total_log_lr: float
    diagnostics: Mapping[str, float]

    @property
    def suggests_shared_operation_not_speaker(self) -> bool:
        """Whether the script links substantially more strongly than the idiolect.

        The signature of delegation: one operation, more than one operator. A
        merged score cannot express this, and it is precisely the pattern an
        investigator needs to see when a group is rotating its callers.

        Stated as a *relative* comparison between the two components rather than
        as a threshold on each. Both components are uncalibrated at this point,
        on scales set by their respective inventory sizes, so an absolute
        threshold on either is a threshold on the inventory. Their ratio is
        scale-robust and is the quantity that actually matters: script evidence
        substantially outrunning idiolect evidence is what delegation looks
        like, whatever the units.

        This remains a heuristic and is reported as an indication to be examined,
        never as a finding. Establishing it properly requires calibrating the two
        components separately against pairs known to be same-operation-different-
        speaker, which needs labelled data the corpus does not yet contain.
        """
        return self.script_log_lr > math.log(10.0) and self.script_log_lr > 2.5 * max(
            self.idiolect_log_lr, 0.0
        )


class BehaviouralComparator:
    """Compares two behavioural profiles.

    Every component uses the conjugate machinery in
    :mod:`viflap.analysis.patterns.conjugate`, so rarity is handled without
    tuned weights: two incidents sharing an unusual discourse particle counts
    for far more than two sharing "the".
    """

    def __init__(
        self,
        function_word_background: BackgroundPopulation,
        disfluency_background: BackgroundPopulation,
        move_background: BackgroundPopulation,
        ngram_background: BackgroundPopulation,
        switch_background: BackgroundPopulation | None = None,
        utterance_length_model: NormalInverseGammaComparator | None = None,
        min_words: int = 40,
    ) -> None:
        # Concentrations differ by what each inventory is like. Function words
        # are a small, universally shared inventory, so an actor's distribution
        # is close to the population's and the concentration is high. Character
        # n-grams are a huge inventory where an actor uses a tiny idiosyncratic
        # subset, so it is low.
        self._function_words = DirichletMultinomialComparator(
            function_word_background, concentration=200.0
        )
        self._disfluencies = DirichletMultinomialComparator(
            disfluency_background, concentration=25.0
        )
        self._moves = DirichletMultinomialComparator(move_background, concentration=40.0)
        self._ngrams = DirichletMultinomialComparator(ngram_background, concentration=15.0)
        self._switches = (
            DirichletMultinomialComparator(switch_background, concentration=10.0)
            if switch_background is not None
            else None
        )
        self._utterance_lengths = utterance_length_model
        self._min_words = min_words

    @property
    def stream(self) -> EvidenceStream:
        return EvidenceStream.BEHAVIOURAL

    def score(
        self, first: BehaviouralProfile, second: BehaviouralProfile
    ) -> BehaviouralScore:
        """Uncalibrated log-likelihood ratios, decomposed."""
        if first.n_words < self._min_words or second.n_words < self._min_words:
            raise InsufficientDataError(
                "transcripts are too short for a behavioural comparison",
                n_words_first=first.n_words,
                n_words_second=second.n_words,
                required=self._min_words,
            )

        idiolect = 0.0
        if first.function_word_counts and second.function_word_counts:
            idiolect += self._function_words.log_likelihood_ratio(
                first.function_word_counts, second.function_word_counts
            )
        if first.disfluency_counts and second.disfluency_counts:
            idiolect += self._disfluencies.log_likelihood_ratio(
                first.disfluency_counts, second.disfluency_counts
            )
        if self._switches is not None and first.switch_counts and second.switch_counts:
            idiolect += self._switches.log_likelihood_ratio(
                first.switch_counts, second.switch_counts
            )
        if self._utterance_lengths is not None:
            idiolect += self._utterance_lengths.log_likelihood_ratio(
                first.log_utterance_lengths, second.log_utterance_lengths
            )
        # Character n-grams are an idiolect feature, and were counted as script
        # evidence until now. The authorship attribution literature reports them
        # repeatedly as the single best-performing feature for identifying a
        # *person* — they encode morphology, characteristic spellings,
        # punctuation and capitalisation habits — so the strongest speaker
        # signal available was sitting in the one component whose entire purpose
        # is to survive a change of speaker. It was large enough to reverse the
        # sign of that component, which is how the defect was found.
        #
        # This is not a clean separation and should not be reported as one. A
        # scripted transcript's n-grams also carry the script's fixed wording,
        # so some genuinely operation-level evidence moves across with them.
        # Restricting them to script-bearing spans is the better fix and needs
        # labelled same-operation-different-speaker data the corpus does not
        # have. Placing them where the literature says they belong is the
        # defensible choice in the meantime; leaving them where they were is not.
        if first.character_ngram_counts and second.character_ngram_counts:
            idiolect += self._ngrams.log_likelihood_ratio(
                first.character_ngram_counts, second.character_ngram_counts
            )

        script = 0.0
        if first.move_counts and second.move_counts:
            script += self._moves.log_likelihood_ratio(
                first.move_counts, second.move_counts
            )
        script += _sequence_log_lr(first.move_sequence, second.move_sequence)

        return BehaviouralScore(
            idiolect_log_lr=float(idiolect),
            script_log_lr=float(script),
            total_log_lr=float(idiolect + script),
            diagnostics={
                "n_words_first": float(first.n_words),
                "n_words_second": float(second.n_words),
                "disfluency_rate_first": first.disfluency_rate,
                "disfluency_rate_second": second.disfluency_rate,
                "switch_rate_first": first.switch_rate,
                "switch_rate_second": second.switch_rate,
                "n_moves_first": float(len(first.move_sequence)),
                "n_moves_second": float(len(second.move_sequence)),
            },
        )


def _sequence_log_lr(
    first: Sequence[str], second: Sequence[str], scale: float = 2.5
) -> float:
    """Evidence from the *order* of rhetorical moves, not merely their presence.

    Two scripts may contain the same moves in different orders and be different
    operations: threatening before requesting credentials is a different pretext
    from requesting first and threatening only on refusal. A count-based
    comparison cannot see the difference, so the ordering is scored separately.

    Uses normalised longest-common-subsequence length, mapped to a log-LR by a
    fixed scale. The scale is an explicitly acknowledged parameter with no
    principled derivation — unlike everything else in this module, which is a
    marginal likelihood under a stated model. It is bounded so that ordering
    evidence cannot dominate, and it should be replaced by a properly estimated
    sequence model once labelled same-operation pairs exist.
    """
    if not first or not second:
        return 0.0

    length = _longest_common_subsequence(first, second)
    similarity = 2.0 * length / (len(first) + len(second))
    # Centred so that chance-level agreement contributes nothing rather than
    # positive evidence. Chance level is approximated at one third, the typical
    # overlap of two unrelated move sequences over this inventory.
    return float(scale * (similarity - 0.34))


def _longest_common_subsequence(first: Sequence[str], second: Sequence[str]) -> int:
    """Length of the longest common subsequence, in O(len(first) * len(second)).

    Uses a rolling row rather than a full table: the sequences here are short,
    but this is called once per candidate in a database search, and the full
    table allocates on every call.
    """
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second):
            if left == right:
                current.append(previous[index] + 1)
            else:
                current.append(max(previous[index + 1], current[index]))
        previous = current
    return previous[-1]
