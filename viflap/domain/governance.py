"""Governance primitives: case binding, authority separation, output language.

These are architectural constraints expressed as types. The distinction that
matters is between a rule the system *has* and a capability the system *lacks*:

- A rule the system has can be waived by whoever operates it.
- A capability the system lacks cannot.

Everything in this module is written to fall in the second category. A query
without a case reference is not rejected by a check that a determined operator
could remove; it cannot be constructed, because the object that represents a
query requires a :class:`CaseReference` and that type cannot be instantiated
from an invalid string. Likewise, the system does not decline to assert
identity — it has no code path that produces the vocabulary of identity, and the
one place text crosses the boundary raises if such vocabulary appears.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from re import Pattern
from typing import Final

from viflap.domain.errors import (
    AuthorityViolation,
    CaseBindingViolation,
    OutputConstraintViolation,
    SeparationOfDutiesViolation,
)

__all__ = [
    "AnalystRole",
    "Authority",
    "CaseReference",
    "CaseReferenceFormat",
    "OutputLanguagePolicy",
    "Principal",
    "assert_separation_of_duties",
]


# ---------------------------------------------------------------------------
# Case binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseReferenceFormat:
    """The shape of a valid case reference in a given deployment.

    The *format* is deployment configuration — jurisdictions number their
    complaints differently — but the *requirement* is architectural. This type
    keeps the two separate: infrastructure supplies the pattern, and the domain
    guarantees nothing without one gets through.
    """

    pattern: Pattern[str]
    description: str

    @classmethod
    def compile(cls, pattern: str, description: str) -> CaseReferenceFormat:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:  # pragma: no cover - configuration error path
            raise CaseBindingViolation(
                "case reference format is not a valid regular expression",
                pattern=pattern,
            ) from exc
        return cls(pattern=compiled, description=description)


#: Default format: an issuing-agency prefix, a year, and a serial.
#: Example: ``ZP-2025-01847``.
DEFAULT_CASE_REFERENCE_FORMAT: Final[CaseReferenceFormat] = CaseReferenceFormat.compile(
    r"^[A-Z]{2,5}-\d{4}-\d{4,6}$",
    "Issuing agency code, year of filing, and serial number (e.g. ZP-2025-01847)",
)


@dataclass(frozen=True, slots=True, order=True)
class CaseReference:
    """A validated reference to a filed complaint.

    Every operation that touches evidence takes one of these. Because the type
    cannot hold an invalid value, "queries without a case reference are
    impossible" is a statement about what the program can represent rather than
    a policy that operations are expected to observe.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value != self.value.strip():
            raise CaseBindingViolation(
                "case reference must be non-empty and free of surrounding whitespace",
                value=self.value,
            )

    @classmethod
    def parse(
        cls,
        raw: str | None,
        fmt: CaseReferenceFormat = DEFAULT_CASE_REFERENCE_FORMAT,
    ) -> CaseReference:
        """Parse and validate a case reference.

        Raises
        ------
        CaseBindingViolation
            If the reference is absent, blank, or does not conform. This must
            not be caught and converted into a warning.
        """
        if raw is None:
            raise CaseBindingViolation(
                "a case reference is required for every operation on evidence"
            )
        candidate = raw.strip()
        if not candidate:
            raise CaseBindingViolation(
                "a case reference is required for every operation on evidence"
            )
        if fmt.pattern.match(candidate) is None:
            raise CaseBindingViolation(
                "case reference does not conform to the required format",
                value=candidate,
                expected=fmt.description,
            )
        return cls(candidate)

    @property
    def issuing_agency(self) -> str:
        """The agency prefix, where the format carries one."""
        head, _, _ = self.value.partition("-")
        return head

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return self.value


# ---------------------------------------------------------------------------
# Authority and separation of duties
# ---------------------------------------------------------------------------


class Authority(Enum):
    """An atomic permission.

    Modelled separately from roles so that separation of duties is a statement
    about *authorities*, which is where the risk lives. A deployment that
    invents a new role cannot accidentally combine incompatible powers, because
    the incompatibility is defined over authorities and checked when the role is
    assigned.
    """

    ENROL = "enrol"
    """Add recordings, transactions or identifiers to the searchable corpus.
    Whoever holds this can shape what the system will find."""

    QUERY = "query"
    """Run comparisons against enrolled data."""

    EXPORT = "export"
    """Move results outside the system boundary."""

    AUDIT = "audit"
    """Read the audit log. Held by an oversight body, not the operator."""

    ADMINISTER = "administer"
    """Change models, thresholds or retention policy. Changing an operating
    point changes every result the system will produce, so this is separated
    from the ability to run or export queries."""


class AnalystRole(Enum):
    """A named bundle of authorities.

    The roles are deliberately narrow. Breadth is achieved by assigning a
    principal more than one role, which is the point at which
    :func:`assert_separation_of_duties` applies.
    """

    ENROLMENT_OFFICER = "enrolment_officer"
    INVESTIGATOR = "investigator"
    DISCLOSURE_OFFICER = "disclosure_officer"
    OVERSIGHT_AUDITOR = "oversight_auditor"
    SYSTEM_ADMINISTRATOR = "system_administrator"

    @property
    def authorities(self) -> frozenset[Authority]:
        return _ROLE_AUTHORITIES[self]

    def grants(self, authority: Authority, /) -> bool:
        return authority in self.authorities


_ROLE_AUTHORITIES: Final[dict[AnalystRole, frozenset[Authority]]] = {
    AnalystRole.ENROLMENT_OFFICER: frozenset({Authority.ENROL}),
    AnalystRole.INVESTIGATOR: frozenset({Authority.QUERY}),
    AnalystRole.DISCLOSURE_OFFICER: frozenset({Authority.EXPORT}),
    AnalystRole.OVERSIGHT_AUDITOR: frozenset({Authority.AUDIT}),
    AnalystRole.SYSTEM_ADMINISTRATOR: frozenset({Authority.ADMINISTER}),
}


#: Authority pairs that must never be held by one principal, with the reason.
#:
#: Each entry answers "what could this person do that no one could detect?".
INCOMPATIBLE_AUTHORITIES: Final[tuple[tuple[Authority, Authority, str], ...]] = (
    (
        Authority.ENROL,
        Authority.QUERY,
        "a principal who can both enrol and query can plant a reference and "
        "then produce a linkage to it",
    ),
    (
        Authority.ENROL,
        Authority.EXPORT,
        "a principal who can both enrol and export can introduce evidence and "
        "remove the result in one unobserved sequence",
    ),
    (
        Authority.QUERY,
        Authority.EXPORT,
        "a principal who can both query and export can remove results without "
        "a second party observing which queries were run",
    ),
    (
        Authority.AUDIT,
        Authority.ENROL,
        "audit custody must be independent of the operating agency; an auditor "
        "who can enrol is auditing their own work",
    ),
    (
        Authority.AUDIT,
        Authority.QUERY,
        "audit custody must be independent of the operating agency",
    ),
    (
        Authority.AUDIT,
        Authority.EXPORT,
        "audit custody must be independent of the operating agency",
    ),
    (
        Authority.ADMINISTER,
        Authority.QUERY,
        "a principal who can change the operating point and run queries can "
        "tune the system until it returns a desired result",
    ),
    (
        Authority.ADMINISTER,
        Authority.AUDIT,
        "an administrator with audit custody can alter the system and hold the "
        "only record of having done so",
    ),
)


def assert_separation_of_duties(roles: Iterable[AnalystRole]) -> None:
    """Reject an assignment of roles that concentrates incompatible authority.

    Raises
    ------
    SeparationOfDutiesViolation
        Naming both authorities and the specific risk, so the message is usable
        by whoever has to decide how to split the work instead.
    """
    held: set[Authority] = set()
    for role in roles:
        held |= role.authorities

    for first, second, rationale in INCOMPATIBLE_AUTHORITIES:
        if first in held and second in held:
            raise SeparationOfDutiesViolation(
                "incompatible authorities assigned to a single principal",
                first=first.value,
                second=second.value,
                rationale=rationale,
            )


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated actor, with the roles they hold.

    Separation of duties is checked at construction. An assignment that violates
    it cannot be represented, so no downstream code needs to re-check.
    """

    identifier: str
    roles: frozenset[AnalystRole]

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise AuthorityViolation("a principal must have a non-empty identifier")
        if not self.roles:
            raise AuthorityViolation(
                "a principal must hold at least one role", principal=self.identifier
            )
        assert_separation_of_duties(self.roles)

    @property
    def authorities(self) -> frozenset[Authority]:
        return frozenset().union(*(role.authorities for role in self.roles))

    def holds(self, authority: Authority, /) -> bool:
        return authority in self.authorities

    def require(self, authority: Authority, /) -> None:
        """Assert that this principal may perform an action.

        Raises
        ------
        AuthorityViolation
        """
        if not self.holds(authority):
            raise AuthorityViolation(
                "principal does not hold the authority required for this action",
                principal=self.identifier,
                required=authority.value,
                held=sorted(item.value for item in self.authorities),
            )


# ---------------------------------------------------------------------------
# Output language
# ---------------------------------------------------------------------------


#: Vocabulary that asserts identity or a conclusion. Inflections are listed
#: explicitly rather than stemmed, so the policy can be audited by reading it.
PROHIBITED_PHRASES: Final[tuple[str, ...]] = (
    "match",
    "matches",
    "matched",
    "matching",
    "identified",
    "identification",
    "identify",
    "identifies",
    "voiceprint",
    "voice print",
    "voice fingerprint",
    "is the same person",
    "are the same person",
    "same individual",
    "positively identified",
    "conclusively",
    "proves",
    "proven",
    "guilty",
    "the suspect is",
)

#: Framings that correctly express evidential weight. Advisory: offered as a
#: replacement when a violation is raised, never used to whitelist text.
PERMITTED_FRAMINGS: Final[tuple[str, ...]] = (
    "the evidence provides ... support for",
    "the observed evidence is ... times more probable if",
    "linkage hypothesis",
    "likelihood ratio",
    "the evidence is consistent with",
)


def _compile_prohibited(phrases: Iterable[str]) -> Pattern[str]:
    """Build the prohibited-language pattern.

    Two details are load-bearing.

    Word boundaries are expressed as ``(?<![\\w-])`` / ``(?![\\w-])`` rather than
    ``\\b``, so that a hyphenated compound such as ``re-match`` is treated as one
    word and does not trip the policy on its second half.

    Multi-word phrases match across arbitrary whitespace, so a double space or a
    line break between "voice" and "print" does not evade the policy. The
    substitution target is derived from :func:`re.escape` itself rather than
    assumed: whether that function escapes a space has varied between Python
    versions, and hard-coding either form silently disables every multi-word
    entry on the other.

    Built as a module-level function rather than in the class body because a
    comprehension inside a class body cannot see names in the class namespace.
    """
    escaped_space = re.escape(" ")
    alternatives = "|".join(
        re.escape(phrase).replace(escaped_space, r"\s+") for phrase in phrases
    )
    return re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])", re.IGNORECASE)


class OutputLanguagePolicy:
    """Constrains the vocabulary of anything the system says.

    The system reports the weight of evidence. It does not report conclusions
    about identity, and the vocabulary of identity is therefore not available to
    it.

    Matching is on **word boundaries**, not substrings. Substring matching is
    both too weak and too strong: it misses ``"matches"`` written as
    ``"match-es"`` while rejecting ``"dispatch"``, ``"rematch"`` and the
    ordinary English of an unrelated log line. Rejecting valid text trains
    operators to route around the policy, which defeats it more thoroughly than
    a gap would.

    Violations raise. They are not silently rewritten, because the phrasing is a
    symptom: some code path formed a conclusion it had no basis to form, and
    quietly editing the wording would leave that path in place.
    """

    #: Re-exported on the class so the policy reads as one unit at the call
    #: site; the module-level tuples remain the single definition.
    PROHIBITED_PHRASES: Final[tuple[str, ...]] = PROHIBITED_PHRASES
    PERMITTED_FRAMINGS: Final[tuple[str, ...]] = PERMITTED_FRAMINGS

    _PATTERN: Final[Pattern[str]] = _compile_prohibited(PROHIBITED_PHRASES)

    @classmethod
    def find_violations(cls, text: str, /) -> list[str]:
        """Return every prohibited phrase occurring in ``text``."""
        return [found.group(0) for found in cls._PATTERN.finditer(text)]

    @classmethod
    def assert_permitted(cls, text: str, /, *, origin: str = "output") -> str:
        """Return ``text`` unchanged, or raise if it asserts identity.

        Parameters
        ----------
        text:
            The string about to cross the system boundary.
        origin:
            Where the text was produced, so a violation identifies the code path
            that needs correcting rather than only the string.

        Raises
        ------
        OutputConstraintViolation
        """
        violations = cls.find_violations(text)
        if violations:
            raise OutputConstraintViolation(
                "text leaving the system asserts identity or conclusion; the "
                "system reports the weight of evidence and nothing further",
                origin=origin,
                prohibited=sorted({violation.lower() for violation in violations}),
                permitted_framings=list(cls.PERMITTED_FRAMINGS),
            )
        return text

    @classmethod
    def assert_structure_permitted(
        cls, value: object, /, *, origin: str = "output"
    ) -> None:
        """Recursively check every string reachable from ``value``.

        Applied to whole response bodies so that a prohibited phrase cannot
        reach an operator by being nested inside a diagnostic field that nobody
        thought to check.
        """
        if isinstance(value, str):
            cls.assert_permitted(value, origin=origin)
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str):
                    cls.assert_permitted(key, origin=f"{origin}.key")
                cls.assert_structure_permitted(item, origin=f"{origin}.{key}")
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                cls.assert_structure_permitted(item, origin=f"{origin}[{index}]")
