"""Speakers permanently reserved for evaluation, whatever the corpus becomes.

Every corpus-size comparison this project has made has had to be restricted
after the fact to whatever overlap survived, and each time the overlap was
smaller. §9 compared 125 against 306 speakers and found only **35** of its
baseline evaluation speakers still held out by both. §25 compared 306 against
562 and found **19** — too few for an interval to describe anything but those
nineteen people, so the comparison could not be made at all.

The cause is structural rather than careless. ``split_by_speaker`` orders
speakers by a seeded permutation and takes fractions, so *adding* speakers
reshuffles everyone: a speaker held out at one corpus size is very likely
training material at the next. Nothing detects this, because both splits are
internally valid and ``verify_disjoint`` checks a split against itself.

The fix has to be decided once and then left alone. These hundred speakers go to
the evaluation partition of every split that is given this list, at every corpus
size, forever. Two models trained on any two corpus versions then share an
evaluation set exactly, and the paired instrument §7 and §22 rely on applies
without restriction.

Chosen from the 936 usable speakers of ``librispeech`` + ``librispeech-360-full``
(§25), ordered by a stable hash of the identifier rather than by the identifier
itself — LibriSpeech numbers correlate with when a reader joined, so a contiguous
block would be a cohort rather than a sample. Every one has at least two sessions,
because a reserved speaker who cannot form a cross-session same-source trial is
reserved capacity nobody can use.

**Do not add to this list, and do not remove from it.** Growing it later
re-creates the problem for every model trained before the change; shrinking it
invalidates every comparison that used the larger set. If it must ever change,
the new list is a *different* list under a different name and the two are not
comparable — which is exactly the situation this module exists to prevent.

It lives in the package rather than under ``data/`` because ``data/`` is not
version controlled, and a protocol decision that is not version controlled is
not a protocol.
"""

from __future__ import annotations

__all__ = ["RESERVED_EVALUATION_SPEAKERS"]

#: LibriSpeech identifiers, held out of training at every corpus size.
RESERVED_EVALUATION_SPEAKERS: frozenset[str] = frozenset(
    {
        "39",
        "55",
        "101",
        "125",
        "211",
        "216",
        "225",
        "231",
        "242",
        "329",
        "374",
        "398",
        "439",
        "445",
        "501",
        "543",
        "559",
        "561",
        "636",
        "671",
        "830",
        "1116",
        "1222",
        "1289",
        "1392",
        "1445",
        "1498",
        "1748",
        "2110",
        "2156",
        "2196",
        "2229",
        "2254",
        "2272",
        "2416",
        "2436",
        "2498",
        "2518",
        "2673",
        "2999",
        "3118",
        "3157",
        "3340",
        "3448",
        "3482",
        "3513",
        "3615",
        "3638",
        "3664",
        "3686",
        "3816",
        "3947",
        "4064",
        "4152",
        "4434",
        "5002",
        "5029",
        "5062",
        "5093",
        "5339",
        "5386",
        "5604",
        "5656",
        "5703",
        "5723",
        "5810",
        "6037",
        "6060",
        "6115",
        "6167",
        "6189",
        "6233",
        "6269",
        "6426",
        "6476",
        "6499",
        "6575",
        "6701",
        "6763",
        "6865",
        "6877",
        "7061",
        "7069",
        "7286",
        "7402",
        "7437",
        "7447",
        "7553",
        "7766",
        "7789",
        "7802",
        "7909",
        "8152",
        "8494",
        "8498",
        "8591",
        "8742",
        "8838",
        "8848",
        "8855",
    }
)
