"""Shared fixtures.

Deterministic throughout. Every random draw is seeded, because a forensic system
whose test suite passes intermittently is a system whose failures will be
dismissed as flakiness — and one of them will eventually be real.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from viflap.domain.governance import AnalystRole, CaseReference, Principal
from viflap.infrastructure.clock import FixedClock


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20250601)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(datetime(2025, 6, 1, 12, 0, tzinfo=UTC))


@pytest.fixture
def case_reference() -> CaseReference:
    return CaseReference.parse("ZP-2025-01847")


@pytest.fixture
def investigator() -> Principal:
    return Principal("inv-001", frozenset({AnalystRole.INVESTIGATOR}))


@pytest.fixture
def enrolment_officer() -> Principal:
    return Principal("enrol-001", frozenset({AnalystRole.ENROLMENT_OFFICER}))


@pytest.fixture
def auditor() -> Principal:
    return Principal("audit-001", frozenset({AnalystRole.OVERSIGHT_AUDITOR}))


@pytest.fixture
def labelled_scores(rng: np.random.Generator):
    """A well-behaved score set: 600 same-source, 3000 different-source.

    Unbalanced on purpose. Pairs combine quadratically, so an evaluation set
    always has far more different-source trials, and a metric that only works on
    balanced data would pass a balanced fixture and fail in service.
    """
    same = rng.normal(2.0, 1.0, 600)
    different = rng.normal(-2.0, 1.0, 3000)
    scores = np.concatenate([same, different])
    labels = np.concatenate([np.ones(600), np.zeros(3000)]).astype(int)
    speakers = np.concatenate(
        [
            np.array([f"spk{i % 40:02d}" for i in range(600)]),
            np.array([f"spk{i % 40:02d}" for i in range(3000)]),
        ]
    )
    return scores, labels, list(speakers)
