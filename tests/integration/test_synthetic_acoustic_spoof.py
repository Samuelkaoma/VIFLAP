"""The other half of §24's question: what the gate does when the speech is fake.

§24 measured admission on genuine speech and said plainly what it had not
measured — how often the gate *excludes* a spoofed recording through the same
channel. ``--spoof`` runs that, and these tests hold the parts of it that would
be wrong quietly rather than loudly.

Three of them matter more than the rest. The attack must land **before** the
channel, because spoofing after the coder models an adversary who can inject
audio into the network rather than one who makes a phone call. The incident-to-
recording alignment must survive a failed attack, because one operator's
incidents reuse recordings and so recording ids are not unique — matching on
them drops the wrong incidents and nothing downstream would notice. And the
attack realisations must not be the ones the detector trained on, which is a
property of a constant and is therefore exactly the kind of thing that gets
changed by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.synthetic_acoustic as synthetic_acoustic
from scripts.corpus import Recording
from scripts.synthetic_acoustic import (
    EVALUATION_ATTACK_SEED,
    GENUINE,
    build_acoustic,
    spoof,
    summarise_arm,
)
from scripts.train_countermeasure import ATTACK_SEED
from viflap.analysis.spoof.attacks import ATTACKS
from viflap.domain.evidence import ValidityAssessment, ValidityVerdict

RATE = 16000


def voiced(seed: int, seconds: float = 1.0) -> np.ndarray:
    """A crude voiced signal: a pulse train through a two-formant filter.

    Not speech, but periodic and resonant enough that LPC analysis converges,
    which is what the attacks need in order to be applied at all.
    """
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    excitation = np.zeros(n)
    excitation[:: RATE // 120] = 1.0
    time = np.arange(n) / RATE
    signal = np.zeros(n)
    for formant, gain in ((700.0, 1.0), (1220.0, 0.6), (2600.0, 0.3)):
        signal += gain * np.convolve(
            excitation, np.exp(-time[:400] * 300) * np.sin(2 * np.pi * formant * time[:400])
        )[:n]
    return signal + rng.normal(0.0, 1e-3, n)


def recording(index: int, recording_id: str | None = None) -> Recording:
    return Recording(
        signal=voiced(index),
        sample_rate=RATE,
        speaker_id=f"spk{index:03d}",
        session_id=f"spk{index:03d}-0",
        recording_id=recording_id or f"spk{index:03d}-0-r0",
        source_utterances=(),
    )


class TestTheAttackRealisations:
    def test_the_evaluation_seed_is_not_the_training_seed(self) -> None:
        """A shared base would spoof a shared recording identically, and the
        detector would be scored on the exact waveform it was fitted to."""
        assert EVALUATION_ATTACK_SEED != ATTACK_SEED

    @pytest.mark.parametrize("attack_id", sorted(ATTACKS))
    def test_every_family_changes_the_signal_and_keeps_the_identity(
        self, attack_id: str
    ) -> None:
        original = recording(1)
        spoofed, failures = spoof([original], attack_id)
        assert failures == []
        assert spoofed[0] is not None
        assert spoofed[0].recording_id == original.recording_id
        assert spoofed[0].sample_rate == original.sample_rate
        assert not np.allclose(spoofed[0].signal, original.signal)

    def test_the_same_recording_spoofs_the_same_way_twice(self) -> None:
        """Two runs of the same experiment must produce the same audio, or the
        arms are not comparable to each other or to a rerun."""
        first, _ = spoof([recording(2)], "lpc_pulse")
        second, _ = spoof([recording(2)], "lpc_pulse")
        assert first[0] is not None and second[0] is not None
        np.testing.assert_array_equal(first[0].signal, second[0].signal)

    def test_two_recordings_do_not_share_a_realisation(self) -> None:
        spoofed, _ = spoof([recording(3), recording(4)], "lpc_noise")
        assert spoofed[0] is not None and spoofed[1] is not None
        assert not np.allclose(spoofed[0].signal, spoofed[1].signal)

    def test_a_failed_attack_leaves_a_hole_rather_than_shortening_the_list(
        self, monkeypatch
    ) -> None:
        """Alignment against the incident list is positional; a silently
        shortened list would shift every incident after the failure."""

        def explode(attack_id, signal, sample_rate, rng):
            from viflap.domain.errors import ConvergenceError

            if signal[0] == 0.0:
                raise ConvergenceError("no")
            return signal * 0.5

        monkeypatch.setattr(synthetic_acoustic, "apply_attack", explode)
        broken = recording(5)
        broken.signal[0] = 0.0
        spoofed, failures = spoof([broken, recording(6)], "lpc_noise")

        assert len(spoofed) == 2
        assert spoofed[0] is None and spoofed[1] is not None
        assert len(failures) == 1 and "ConvergenceError" in failures[0]


class _System:
    def embed(self, signal, sample_rate):
        return object()


class _Gate:
    """Admits loud recordings and excludes quiet ones, deterministically."""

    detector_id = "test"

    def assess(self, recording_id, signal, sample_rate) -> ValidityAssessment:
        score = float(np.max(np.abs(signal)))
        verdict = (
            ValidityVerdict.ADMITTED if score > 1.0 else ValidityVerdict.EXCLUDED
        )
        return ValidityAssessment(
            recording_id=recording_id,
            verdict=verdict,
            countermeasure_log_lr=score,
            threshold=1.0,
            detector_id="test",
        )


@pytest.fixture
def incidents_and_binding():
    from scripts.synthesise_incidents import generate

    operators, _, incidents = generate(
        n_operators=4, n_operations=2, incidents_per_operation=3
    )
    # Two recordings per operator and three incidents each, so at least one
    # recording is reused -- which is what makes recording ids non-unique.
    bound = {
        operator.operator_id: [
            _plan(f"spk{index:03d}", session) for session in range(2)
        ]
        for index, operator in enumerate(operators)
    }
    return incidents, bound


def _plan(speaker: str, session: int):
    class _P:
        speaker_id = speaker
        session_id = f"{speaker}-{session}"
        recording_id = f"{speaker}-{session}-r0"

    return _P()


class TestTheSpoofedArm:
    def test_the_attack_lands_before_the_channel(
        self, monkeypatch, incidents_and_binding
    ) -> None:
        """Spoofing after the coder would be a different, stronger threat model.

        The order is recorded by watching which of the two steps sees the
        attacked signal: degradation must receive it, not produce it.
        """
        incidents, bound = incidents_and_binding
        order: list[str] = []

        def fake_materialise(plans):
            return [recording(i, plan.recording_id) for i, plan in enumerate(plans)]

        def fake_apply_attack(attack_id, signal, sample_rate, rng):
            order.append("attack")
            return signal * 2.0

        def fake_degrade_many(recordings, conditions, *, seed=0, workers=None):
            order.append("degrade")
            return list(recordings)

        monkeypatch.setattr(synthetic_acoustic, "materialise", fake_materialise)
        monkeypatch.setattr(synthetic_acoustic, "apply_attack", fake_apply_attack)
        monkeypatch.setattr(synthetic_acoustic, "degrade_many", fake_degrade_many)

        build_acoustic(
            incidents, bound, _System(), _Gate(), attack_id="lpc_noise", workers=1
        )
        assert order.count("degrade") == 1
        assert order.index("degrade") == len(order) - 1

    def test_a_failed_attack_drops_its_own_incident_and_no_other(
        self, monkeypatch, incidents_and_binding
    ) -> None:
        incidents, bound = incidents_and_binding

        def fake_materialise(plans):
            return [recording(i, plan.recording_id) for i, plan in enumerate(plans)]

        def fail_the_first(attack_id, signal, sample_rate, rng):
            from viflap.domain.errors import ConvergenceError

            if not getattr(fail_the_first, "done", False):
                fail_the_first.done = True
                raise ConvergenceError("no")
            return signal * 2.0

        monkeypatch.setattr(synthetic_acoustic, "materialise", fake_materialise)
        monkeypatch.setattr(synthetic_acoustic, "apply_attack", fail_the_first)
        monkeypatch.setattr(
            synthetic_acoustic,
            "degrade_many",
            lambda recordings, conditions, *, seed=0, workers=None: list(recordings),
        )

        evidence, failures = build_acoustic(
            incidents, bound, _System(), _Gate(), attack_id="lpc_noise", workers=1
        )
        assert len(failures) == 1
        assert len(evidence) == len(incidents) - 1
        assert set(evidence) == {i.incident_id for i in incidents[1:]}

    def test_the_genuine_arm_applies_no_attack(
        self, monkeypatch, incidents_and_binding
    ) -> None:
        incidents, bound = incidents_and_binding
        calls: list[str] = []

        monkeypatch.setattr(
            synthetic_acoustic,
            "materialise",
            lambda plans: [recording(i, p.recording_id) for i, p in enumerate(plans)],
        )
        monkeypatch.setattr(
            synthetic_acoustic,
            "apply_attack",
            lambda *a, **k: calls.append("x") or np.zeros(RATE),
        )
        monkeypatch.setattr(
            synthetic_acoustic,
            "degrade_many",
            lambda recordings, conditions, *, seed=0, workers=None: list(recordings),
        )

        evidence, failures = build_acoustic(
            incidents, bound, _System(), _Gate(), attack_id=GENUINE, workers=1
        )
        assert calls == []
        assert failures == []
        assert len(evidence) == len(incidents)


class TestTheArmSummary:
    @staticmethod
    def _evidence(verdicts):
        class _E:
            def __init__(self, verdict: ValidityVerdict) -> None:
                self.validity = ValidityAssessment(
                    recording_id="r",
                    verdict=verdict,
                    countermeasure_log_lr=0.0,
                    threshold=2.3,
                    detector_id="test",
                )

        return {str(i): _E(v) for i, v in enumerate(verdicts)}

    def test_exclusions_and_admissions_are_both_named(self) -> None:
        """A spoofed arm is asking about exclusions and afraid of admissions, so
        neither may be left to be inferred from the verdict dictionary."""
        arm = summarise_arm(
            "lpc_noise",
            self._evidence(
                [ValidityVerdict.EXCLUDED] * 3
                + [ValidityVerdict.INDETERMINATE] * 2
                + [ValidityVerdict.ADMITTED]
            ),
            [],
        )
        assert arm["attack_id"] == "lpc_noise"
        assert arm["n_excluded"] == 3
        assert arm["n_indeterminate"] == 2
        assert arm["n_admitted"] == 1

    def test_a_verdict_never_reached_reads_as_zero_not_missing(self) -> None:
        arm = summarise_arm("lpc_pulse", self._evidence([ValidityVerdict.EXCLUDED]), [])
        assert arm["n_admitted"] == 0
        assert arm["n_indeterminate"] == 0

    def test_the_genuine_arm_is_marked_as_not_an_attack(self) -> None:
        arm = summarise_arm(GENUINE, self._evidence([ValidityVerdict.ADMITTED]), [])
        assert arm["seen_in_training"] is False
        assert summarise_arm("lpc_noise", self._evidence([]), [])["seen_in_training"]

    def test_generation_failures_are_carried_not_swallowed(self) -> None:
        arm = summarise_arm(
            "oversmoothed", self._evidence([ValidityVerdict.EXCLUDED]), ["a", "b"]
        )
        assert arm["n_generation_failures"] == 2
        assert arm["generation_failures"] == ["a", "b"]
