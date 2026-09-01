"""The validity gate, reached through a real signal rather than a fixture.

§24's defect — a policy demanding ±2.3 from a detector whose scores on genuine
speech span -2.33 to +1.60 — is invisible to every test of either component. The
policy is coherent, the detector works as documented, and only putting one to the
other shows that the gate admits nothing.

These tests do not repeat that measurement, which needs the corpus and two
trained models and belongs in a script. They assert the thing that made the
measurement possible and would silently stop being true: that an operator binds
to a real speaker, that the recording judged is the recording embedded, and that
the gate is reachable at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.synthesise_incidents import generate
from scripts.synthetic_acoustic import bind_operators, plan_for, summarise
from viflap.domain.errors import InsufficientDataError
from viflap.domain.evidence import ValidityAssessment, ValidityVerdict


class _Plan:
    """Enough of a ``RecordingPlan`` to bind against."""

    def __init__(self, speaker: str, session: str, index: int) -> None:
        self.speaker_id = speaker
        self.session_id = session
        self.recording_id = f"{speaker}-{session}-r{index}"


def _plans(n_speakers: int, per_speaker: int = 4) -> list[_Plan]:
    return [
        _Plan(f"spk{s:03d}", f"{s}{c}", 0)
        for s in range(n_speakers)
        for c in range(per_speaker)
    ]


class TestBinding:
    def test_every_operator_gets_a_distinct_speaker(self) -> None:
        """Two operators sharing a speaker would make two different people
        acoustically identical, which is the confound the whole corpus is built
        to keep separable."""
        operators, _, _ = generate(
            n_operators=12, n_operations=4, incidents_per_operation=4
        )
        bound = bind_operators(operators, _plans(40))
        chosen = [plans[0].speaker_id for plans in bound.values()]
        assert len(set(chosen)) == len(operators)

    def test_too_few_speakers_is_refused_not_truncated(self) -> None:
        operators, _, _ = generate(
            n_operators=12, n_operations=4, incidents_per_operation=4
        )
        with pytest.raises(InsufficientDataError, match="acoustically identical"):
            bind_operators(operators, _plans(5))

    def test_the_binding_does_not_depend_on_iteration_order(self) -> None:
        """Sorted, so the binding is a function of the corpus. Otherwise a rerun
        could quietly rebind operators and two runs would not be comparable."""
        operators, _, _ = generate(n_operators=8, n_operations=3, incidents_per_operation=4)
        plans = _plans(30)
        first = bind_operators(operators, plans)
        shuffled = list(plans)
        np.random.default_rng(3).shuffle(shuffled)
        second = bind_operators(operators, shuffled)
        assert {k: v[0].speaker_id for k, v in first.items()} == {
            k: v[0].speaker_id for k, v in second.items()
        }


class TestIncidentToRecording:
    def test_incidents_of_one_operator_spread_across_sessions(self) -> None:
        """§2's rule needs same-source trials to cross sessions.

        If every incident of an operator mapped to the same recording, every
        same-source acoustic trial would compare a recording with itself.
        """
        operators, _, incidents = generate(
            n_operators=10, n_operations=3, incidents_per_operation=6
        )
        bound = bind_operators(operators, _plans(30, per_speaker=4))
        by_operator: dict[str, set[str]] = {}
        for incident in incidents:
            plan = plan_for(incident, bound)
            by_operator.setdefault(incident.operator_id, set()).add(plan.recording_id)
        assert max(len(v) for v in by_operator.values()) > 1

    def test_an_incident_always_maps_into_its_own_operator_s_recordings(self) -> None:
        operators, _, incidents = generate(
            n_operators=10, n_operations=3, incidents_per_operation=5
        )
        bound = bind_operators(operators, _plans(30))
        for incident in incidents:
            owned = {p.recording_id for p in bound[incident.operator_id]}
            assert plan_for(incident, bound).recording_id in owned


class TestTheSummary:
    """What §24 reads off the run."""

    @staticmethod
    def _evidence(scores, threshold=2.3):
        class _E:
            def __init__(self, score: float) -> None:
                verdict = (
                    ValidityVerdict.ADMITTED
                    if score >= threshold
                    else ValidityVerdict.INDETERMINATE
                )
                self.validity = ValidityAssessment(
                    recording_id="r",
                    verdict=verdict,
                    countermeasure_log_lr=score,
                    threshold=threshold,
                    detector_id="test",
                )

        return {str(i): _E(s) for i, s in enumerate(scores)}

    def test_the_whole_range_is_reported_not_just_the_centre(self) -> None:
        """§24's finding is that the *maximum* never reaches the threshold, and
        a median cannot show that."""
        summary = summarise(self._evidence([-2.0, -0.5, 1.6]))
        assert summary["countermeasure_log_lr"]["min"] == pytest.approx(-2.0)
        assert summary["countermeasure_log_lr"]["max"] == pytest.approx(1.6)

    def test_reaching_the_threshold_is_counted_separately_from_admission(
        self,
    ) -> None:
        """A recording can fail to be admitted for a domain reason while still
        clearing the score bar; conflating the two would hide which rule fired.
        """
        summary = summarise(self._evidence([-1.0, 3.0]))
        assert summary["n_reaching_admit_threshold"] == 1

    def test_a_run_admitting_nothing_says_so(self) -> None:
        summary = summarise(self._evidence([-2.0, -0.2, 1.6]))
        assert summary["n_admitting_acoustic"] == 0
        assert summary["n_reaching_admit_threshold"] == 0
        assert summary["verdicts"] == {"indeterminate": 3}
