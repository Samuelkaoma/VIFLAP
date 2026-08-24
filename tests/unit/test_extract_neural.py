"""Resuming the ten-hour extraction, and the guard that refuses a wrong resume.

`extract_neural.py` produced nothing for ten hours and then died in its last
block, so it now checkpoints after every batch. The value of that is entirely in
one property: **a run that dies and resumes must produce exactly what an
uninterrupted run would have produced.** A checkpoint that is merely *nearly*
right is worse than none, because the artefact would be well-formed, would load,
and would carry embeddings from a partly different configuration into a section
that describes it as one.

So the central test crashes a run part-way through, restarts it, and compares
the artefact key by key against a reference run that was never interrupted. It
uses a fake extractor over fake audio — the real one costs six seconds per
recording and this must run in the suite — but it drives the real ``main``, the
real split, the real batching and the real checkpoint format. What the fakes
give up is any claim about ECAPA; what they keep is every place the resume
machinery could lose, duplicate or reorder a recording.

The refusal counter gets its own attention. It lives outside the arrays, in the
checkpoint's metadata, and is the one piece of per-block state that a resume
could plausibly reset to zero or double. §6 records that refusals at short
durations are not random with respect to difficulty, so a lost refusal count is
a silently biased cell rather than a cosmetic defect.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import scripts.extract_neural as extract_neural
from scripts.corpus import RecordingPlan
from scripts.extract_neural import (
    CHECKPOINT_VERSION,
    META_KEY,
    cell_key,
    checkpoint_path_for,
    load_checkpoint,
    restore_block,
    save_checkpoint,
)
from viflap.domain.errors import InsufficientDataError

#: A fake "sample rate" so that 300 samples is 30 seconds and truncation to 5
#: seconds leaves 50. Nothing decodes these signals, so the units are free.
RATE = 10
SAMPLES = 300

N_SPEAKERS = 30
N_SESSIONS = 3


@dataclasses.dataclass(frozen=True)
class FakePiece:
    """Stands in for a degraded recording: identity, a signal, and truncation."""

    speaker_id: str
    recording_id: str
    signal: np.ndarray
    sample_rate: int = RATE

    def truncated(self, seconds: float) -> FakePiece:
        return dataclasses.replace(
            self, signal=self.signal[: int(seconds * self.sample_rate)]
        )


class FakeExtractor:
    """Embeds identity, refuses some short pieces, and can be made to die.

    The embedding is ``[value, length]`` where the value carries both which
    recording this is and which condition it went through, so a bug that mixed
    two conditions or reordered a batch changes the vectors rather than leaving
    them coincidentally equal.
    """

    extractor_id = "fake-extractor/v1"

    def __init__(self, crash_after: int | None = None) -> None:
        self.calls = 0
        self.crash_after = crash_after

    def embed(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        if self.crash_after is not None and self.calls > self.crash_after:
            raise RuntimeError("simulated crash")
        # Refuse a seventh of the recordings, but only below six seconds, which
        # is the shape of the real gate: short pieces carrying little speech.
        if signal.size < 6 * RATE and int(signal[0]) % 7 == 0:
            raise InsufficientDataError("too little speech")
        return np.array([float(signal[0]), float(signal.size)], dtype=np.float64)


def fake_plans() -> list[RecordingPlan]:
    return [
        RecordingPlan(
            speaker_id=f"spk{speaker:03d}",
            session_id=f"spk{speaker:03d}-{session}",
            recording_id=f"spk{speaker:03d}-{session}-r0",
            sources=(),
            sample_rate=RATE,
            target_samples=SAMPLES,
        )
        for speaker in range(N_SPEAKERS)
        for session in range(N_SESSIONS)
    ]


def fake_materialise(plans):
    """Give each plan a constant signal whose value is its index in the corpus."""
    index = {plan.recording_id: i for i, plan in enumerate(fake_plans())}
    return [
        FakePiece(
            speaker_id=plan.speaker_id,
            recording_id=plan.recording_id,
            signal=np.full(SAMPLES, float(index[plan.recording_id])),
        )
        for plan in plans
    ]


def _code(label: str) -> int:
    """A stable small integer per condition. Never the built-in ``hash``: it is
    salted per interpreter, and this project has been bitten by that twice."""
    return sum(ord(character) for character in label)


def fake_degrade_many(recordings, conditions, *, seed=0, workers=None):
    """Nudge the signal by the condition, so conditions are distinguishable."""
    chosen = (
        list(conditions) * len(recordings) if len(conditions) == 1 else list(conditions)
    )
    return [
        dataclasses.replace(
            recording,
            signal=recording.signal + 0.001 * (_code(condition.label) % 97),
        )
        for recording, condition in zip(recordings, chosen, strict=True)
    ]


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Point ``extract_neural`` at fake audio, with small batches."""
    monkeypatch.setattr(extract_neural, "BATCH", 4)
    monkeypatch.setattr(extract_neural, "materialise", fake_materialise)
    monkeypatch.setattr(extract_neural, "degrade_many", fake_degrade_many)
    monkeypatch.setattr(extract_neural, "scan_corpora", lambda *a, **k: fake_plans())


def run_main(tmp_path, name, *, crash_after=None, extra=()):
    """Drive ``main`` with a fake extractor, returning it and the exit status."""
    extractor = FakeExtractor(crash_after=crash_after)
    import viflap.infrastructure.neural_extractor as neural_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            neural_module, "NeuralEmbeddingExtractor", lambda *a, **k: extractor
        )
        status = extract_neural.main(
            [
                "--corpus",
                str(tmp_path / "corpus"),
                "--output",
                str(tmp_path / f"{name}.npz"),
                "--report",
                str(tmp_path / f"{name}.json"),
                *extra,
            ]
        )
    return extractor, status


def test_cell_key_matches_the_layout_downstream_reads():
    """``score_neural`` discovers cells by this naming; it is not free to change."""
    assert cell_key("train", None) == "train"
    assert cell_key("evaluation|amr12.2_clean", 30.0) == "evaluation|amr12.2_clean@30"
    assert cell_key("evaluation|amr12.2_clean", 5.0) == "evaluation|amr12.2_clean@5"


def test_checkpoint_path_sits_beside_the_artefact(tmp_path):
    output = tmp_path / "neural_embeddings_pinned.npz"
    assert checkpoint_path_for(output, None) == (
        tmp_path / "neural_embeddings_pinned.checkpoint.npz"
    )
    override = tmp_path / "elsewhere.npz"
    assert checkpoint_path_for(output, override) == override


def test_checkpoint_round_trips_everything_a_resume_needs(tmp_path):
    path = tmp_path / "c.npz"
    fingerprint = {"seed": 7, "durations": [30.0, 5.0]}
    arrays = {
        "train|vectors": np.arange(6, dtype=np.float64).reshape(3, 2),
        "train|speakers": np.array(["a", "b", "c"], dtype=np.str_),
        "train|recordings": np.array(["a-0-r0", "b-0-r0", "c-0-r0"], dtype=np.str_),
    }
    summary = [{"partition": "train", "n_embeddings": 3}]
    partial = {"block": "train", "done": 4, "refused": {"train": 2}}

    save_checkpoint(
        path,
        fingerprint=fingerprint,
        arrays=arrays,
        summary=summary,
        completed=["nothing"],
        partial=partial,
        elapsed_minutes=12.5,
    )
    loaded, loaded_summary, completed, loaded_partial, elapsed = load_checkpoint(
        path, fingerprint
    )

    assert set(loaded) == set(arrays)
    np.testing.assert_array_equal(loaded["train|vectors"], arrays["train|vectors"])
    assert loaded_summary == summary
    assert completed == {"nothing"}
    assert loaded_partial == partial
    assert elapsed == pytest.approx(12.5)


def test_checkpoint_write_is_atomic(tmp_path):
    """A crash mid-write must leave the previous checkpoint intact, not a stub."""
    path = tmp_path / "c.npz"
    fingerprint = {"seed": 7}
    save_checkpoint(
        path,
        fingerprint=fingerprint,
        arrays={"a|vectors": np.zeros((1, 2))},
        summary=[],
        completed=[],
        partial=None,
        elapsed_minutes=0.0,
    )
    first = path.read_bytes()

    # The temporary file is the only thing that exists before the rename, so it
    # must not be the destination and must not survive a completed write.
    save_checkpoint(
        path,
        fingerprint=fingerprint,
        arrays={"a|vectors": np.zeros((2, 2))},
        summary=[],
        completed=[],
        partial=None,
        elapsed_minutes=1.0,
    )
    assert not path.with_name(path.name + ".tmp").exists()
    assert path.read_bytes() != first
    arrays, _, _, _, _ = load_checkpoint(path, fingerprint)
    assert arrays["a|vectors"].shape == (2, 2)


def test_a_checkpoint_from_another_configuration_is_refused(tmp_path):
    """Naming the field that differs, because the usual cause is a wrong flag."""
    path = tmp_path / "c.npz"
    save_checkpoint(
        path,
        fingerprint={"seed": 7, "durations": [30.0]},
        arrays={},
        summary=[],
        completed=[],
        partial=None,
        elapsed_minutes=0.0,
    )
    with pytest.raises(ValueError, match="durations"):
        load_checkpoint(path, {"seed": 7, "durations": [30.0, 5.0]})


def test_a_checkpoint_from_another_version_is_refused(tmp_path):
    path = tmp_path / "c.npz"
    meta = json.dumps(
        {
            "version": CHECKPOINT_VERSION + 1,
            "fingerprint": {},
            "summary": [],
            "completed": [],
            "partial": None,
            "elapsed_minutes": 0.0,
        }
    )
    np.savez(path, **{META_KEY: np.array(meta, dtype=np.str_)})
    with pytest.raises(ValueError, match="checkpoint version"):
        load_checkpoint(path, {})


def test_restore_block_returns_live_accumulators():
    """What comes back must be appendable, not the finished arrays."""
    arrays = {
        "train|vectors": np.arange(6, dtype=np.float64).reshape(3, 2),
        "train|speakers": np.array(["a", "b", "c"], dtype=np.str_),
        "train|recordings": np.array(["a-0-r0", "b-0-r0", "c-0-r0"], dtype=np.str_),
    }
    collected = restore_block(arrays, "train", [None], {"train": 5})
    vectors, speakers, ids, refused = collected[None]

    assert len(vectors) == 3 and len(speakers) == 3 and len(ids) == 3
    assert refused == [5]
    vectors.append(np.array([6.0, 7.0]))
    assert extract_neural._finalise(collected)[None][0].shape == (4, 2)


def test_restore_block_handles_a_block_with_nothing_yet():
    collected = restore_block({}, "train", [None, 5.0], {})
    assert collected[None] == ([], [], [], [0])
    assert collected[5.0] == ([], [], [], [0])


@pytest.mark.usefixtures("fake_pipeline")
def test_a_resumed_run_reproduces_an_uninterrupted_one_exactly(tmp_path):
    """The whole point: crash, restart, and land on the same artefact.

    The reference run and the crashed one differ in nothing but where the
    process ended, so every key, every vector and every label must match. An
    equality that held only on the training block would pass a weaker test and
    still lose the evaluation cells, so this compares the full key set.
    """
    reference, status = run_main(tmp_path, "reference")
    assert status == 0
    assert not (tmp_path / "reference.checkpoint.npz").exists()

    # Die two thirds of the way through, then resume with a healthy extractor.
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_main(tmp_path, "resumed", crash_after=reference.calls * 2 // 3)
    assert (tmp_path / "resumed.checkpoint.npz").exists()

    second, status = run_main(tmp_path, "resumed")
    assert status == 0
    assert not (tmp_path / "resumed.checkpoint.npz").exists()

    with np.load(tmp_path / "reference.npz") as expected, np.load(
        tmp_path / "resumed.npz"
    ) as actual:
        assert sorted(expected.files) == sorted(actual.files)
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    # The resumed run must not have re-embedded what the checkpoint already held.
    assert second.calls < reference.calls


@pytest.mark.usefixtures("fake_pipeline")
def test_a_resumed_run_reproduces_the_refusal_counts(tmp_path):
    """Refusals live in metadata rather than arrays, so they resume separately.

    §6 records that short-duration refusals fall on the recordings carrying
    least speech, so a count reset by a resume would understate exactly the
    cells where it matters most.
    """
    reference, _ = run_main(tmp_path, "reference")
    with pytest.raises(RuntimeError):
        run_main(tmp_path, "resumed", crash_after=reference.calls * 2 // 3)
    run_main(tmp_path, "resumed")

    expected = json.loads((tmp_path / "reference.json").read_text(encoding="utf-8"))
    actual = json.loads((tmp_path / "resumed.json").read_text(encoding="utf-8"))
    assert actual["cells"] == expected["cells"]
    assert sum(cell["n_refused"] for cell in expected["cells"]) > 0


@pytest.mark.usefixtures("fake_pipeline")
def test_restart_discards_the_checkpoint(tmp_path):
    reference, _ = run_main(tmp_path, "reference")
    with pytest.raises(RuntimeError):
        run_main(tmp_path, "resumed", crash_after=reference.calls * 2 // 3)

    fresh, status = run_main(tmp_path, "resumed", extra=("--restart",))
    assert status == 0
    assert fresh.calls == reference.calls


@pytest.mark.usefixtures("fake_pipeline")
def test_a_resumed_run_reports_the_total_cost_not_the_last_attempt(tmp_path):
    """``elapsed_minutes`` is quoted when a run is written up, so it must not reset.

    The fakes finish in milliseconds, so the accrued time is forced to a value
    no second attempt could reach on its own. Without that the assertion would
    compare 0.0 against 0.0 and hold whatever the code did.
    """
    reference, _ = run_main(tmp_path, "reference")
    with pytest.raises(RuntimeError):
        run_main(tmp_path, "resumed", crash_after=reference.calls * 2 // 3)

    checkpoint = tmp_path / "resumed.checkpoint.npz"
    with np.load(checkpoint, allow_pickle=False) as loaded:
        meta = json.loads(str(loaded[META_KEY]))
        arrays = {key: loaded[key] for key in loaded.files if key != META_KEY}
    meta["elapsed_minutes"] = 999.0
    np.savez(checkpoint, **arrays, **{META_KEY: np.array(json.dumps(meta), np.str_)})

    run_main(tmp_path, "resumed")
    report = json.loads((tmp_path / "resumed.json").read_text(encoding="utf-8"))
    assert report["elapsed_minutes"] >= 999.0
