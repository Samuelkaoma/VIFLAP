"""Behavioural evidence: idiolect and script structure.

The stream that survives an offender rotating who speaks. Acoustic evidence
cannot link two incidents conducted by two different people; the operation
persists across them, and script structure is what carries it.

``text``
    Tokenisation and character-n-gram language identification for
    intra-utterance code-switching. Nothing here hard-codes a word list for any
    language.
``profile``
    Behavioural profiles and their comparison, keeping idiolect (speaker-specific,
    defeated by delegation) separate from script structure (operation-specific,
    survives it) — because "same operation, different speaker" is a finding a
    merged score cannot express.
"""

from viflap.analysis.behaviour.profile import (
    BehaviouralComparator,
    BehaviouralProfile,
    BehaviouralScore,
    DisfluencyInventory,
    ScriptMove,
    build_profile,
    segment_script_moves,
)
from viflap.analysis.behaviour.text import (
    CharacterNGramLanguageModel,
    LanguageIdentifier,
    LanguageSpan,
    Token,
    tokenise,
)

__all__ = [
    "BehaviouralComparator",
    "BehaviouralProfile",
    "BehaviouralScore",
    "CharacterNGramLanguageModel",
    "DisfluencyInventory",
    "LanguageIdentifier",
    "LanguageSpan",
    "ScriptMove",
    "Token",
    "build_profile",
    "segment_script_moves",
    "tokenise",
]
