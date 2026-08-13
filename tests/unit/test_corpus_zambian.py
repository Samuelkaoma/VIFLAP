"""Scanning the Zambian corpora, and refusing to label what has no labels.

The corpora differ in whether they identify their speakers, and the difference
decides which training stages each may feed. These tests pin that down: the
roster is the authority on who exists, material the roster does not know is
background rather than a new speaker, and background material cannot reach a
speaker split without raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.corpus import split_by_speaker
from scripts.corpus_zambian import (
    UNLABELLED_SPEAKER,
    parse_labelled_name,
    read_speaker_roster,
    reject_unlabelled,
    scan_labelled,
    scan_unlabelled,
)

SAMPLE_RATE = 16_000

#: The real layout, names included. They are present precisely so the test can
#: show the parser leaves them alone.
ROSTER_TEXT = """The meaning of the fields in left-to-right order is as follows:

* ID   - the ID of the speaker
 ---------------------------------------------------------------------------
 | ID   |SEX| \tUTTERANCES\t| \tDURATION\t| \tNAME       | NATIVE LANGUAGE |
 ---------------------------------------------------------------------------
 | 01   | M | \t5723\t\t|\t10:36:30 \t| \tBrian      |      Bemba      |
 | 02   | F | \t954\t\t|\t01:15:20 \t| \tRichard    |      Bemba      |
 | 03   | F | \t3117\t\t|\t05:03:44 \t| \tVictoria   |      Nsenga     |
"""


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    sf.write(path, samples, SAMPLE_RATE)


@pytest.fixture
def roster_file(tmp_path: Path) -> Path:
    path = tmp_path / "speaker_info.txt"
    path.write_text(ROSTER_TEXT, encoding="utf-8")
    return path


class TestSpeakerRoster:
    def test_parses_the_published_layout(self, roster_file: Path) -> None:
        roster = read_speaker_roster(roster_file)

        assert set(roster) == {"01", "02", "03"}
        assert roster["01"].sex == "M"
        assert roster["01"].utterances == 5723
        assert roster["01"].duration_seconds == 10 * 3600 + 36 * 60 + 30

    def test_does_not_read_speaker_names(self, roster_file: Path) -> None:
        """Names are personal data the pipeline has no use for."""
        roster = read_speaker_roster(roster_file)

        serialised = repr(roster)
        for name in ("Brian", "Richard", "Victoria"):
            assert name not in serialised

    def test_rejects_a_file_that_is_not_a_roster(self, tmp_path: Path) -> None:
        path = tmp_path / "not_a_roster.txt"
        path.write_text("audio\tsentence\nfoo.wav\thello\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no speaker rows"):
            read_speaker_roster(path)


class TestFilenameParsing:
    def test_splits_speaker_from_session(self) -> None:
        assert parse_labelled_name("01-200921-192247_bem_d31_elicit_16.wav") == (
            "01",
            "200921-192247",
        )

    def test_returns_none_for_bembaspeech_unattributed_recordings(self) -> None:
        """A date then a time is not a speaker then a session.

        These are the corpus's own 1,051 later additions. Requiring the full
        ``yymmdd-hhmmss`` after the speaker is what tells the two apart.
        """
        assert parse_labelled_name("200701-160335_bem_d16_elicit_40.wav") is None

    def test_returns_none_for_the_zambezi_voice_layout(self) -> None:
        """Nyanja, Tonga and Lozi begin with the session and name no speaker."""
        assert parse_labelled_name("221102-102320_nya_510_elicit_0.wav") is None


class TestScanLabelled:
    def _corpus(self, tmp_path: Path) -> Path:
        audio = tmp_path / "audio"
        for session in ("200921-192247", "200922-101500"):
            for index in range(4):
                _write_wav(audio / f"01-{session}_bem_d31_elicit_{index}.wav")
        for session in ("200923-084500", "200924-093000"):
            for index in range(4):
                _write_wav(audio / f"02-{session}_bem_a02_elicit_{index}.wav")
        return audio

    def test_builds_plans_for_roster_speakers(
        self, tmp_path: Path, roster_file: Path
    ) -> None:
        audio = self._corpus(tmp_path)
        plans = scan_labelled(audio, read_speaker_roster(roster_file), target_seconds=2.0)

        assert {p.speaker_id for p in plans} == {"01", "02"}
        assert all(p.sample_rate == SAMPLE_RATE for p in plans)
        assert all(p.target_samples == 2 * SAMPLE_RATE for p in plans)

    def test_excludes_unattributed_session_first_files(
        self, tmp_path: Path, roster_file: Path
    ) -> None:
        """The corpus gained 1,051 recordings after the roster was written."""
        audio = self._corpus(tmp_path)
        for index in range(4):
            _write_wav(audio / f"200701-160335_bem_d16_elicit_{index}.wav")

        plans = scan_labelled(audio, read_speaker_roster(roster_file), target_seconds=2.0)

        assert {p.speaker_id for p in plans} == {"01", "02"}
        assert "200701" not in {p.speaker_id for p in plans}

    def test_excludes_a_well_formed_prefix_the_roster_does_not_know(
        self, tmp_path: Path, roster_file: Path
    ) -> None:
        """The second gate: parses as a speaker, but the corpus never documents it.

        The filename pattern cannot catch this one — ``99`` is shaped exactly
        like a speaker — so the roster has to.
        """
        audio = self._corpus(tmp_path)
        for session in ("200925-120000", "200926-130000"):
            for index in range(4):
                _write_wav(audio / f"99-{session}_bem_zzz_elicit_{index}.wav")

        plans = scan_labelled(audio, read_speaker_roster(roster_file), target_seconds=2.0)

        assert "99" not in {p.speaker_id for p in plans}
        assert {p.speaker_id for p in plans} == {"01", "02"}

    def test_drops_speakers_with_too_few_sessions(
        self, tmp_path: Path, roster_file: Path
    ) -> None:
        audio = tmp_path / "audio"
        for index in range(4):
            _write_wav(audio / f"01-200921-192247_bem_d31_elicit_{index}.wav")

        plans = scan_labelled(
            audio,
            read_speaker_roster(roster_file),
            target_seconds=2.0,
            min_sessions_per_speaker=2,
        )
        assert plans == []

    def test_keeps_sessions_separable_for_within_speaker_covariance(
        self, tmp_path: Path, roster_file: Path
    ) -> None:
        audio = self._corpus(tmp_path)
        plans = scan_labelled(audio, read_speaker_roster(roster_file), target_seconds=2.0)

        for speaker in ("01", "02"):
            sessions = {p.session_id for p in plans if p.speaker_id == speaker}
            assert len(sessions) == 2

    def test_is_deterministic(self, tmp_path: Path, roster_file: Path) -> None:
        audio = self._corpus(tmp_path)
        roster = read_speaker_roster(roster_file)

        first = scan_labelled(audio, roster, target_seconds=2.0)
        second = scan_labelled(audio, roster, target_seconds=2.0)
        assert [p.recording_id for p in first] == [p.recording_id for p in second]

    def test_missing_root_is_an_error(self, tmp_path: Path, roster_file: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan_labelled(tmp_path / "absent", read_speaker_roster(roster_file))


class TestScanUnlabelled:
    def test_one_plan_per_file_marked_unlabelled(self, tmp_path: Path) -> None:
        audio = tmp_path / "nya"
        for index in range(3):
            _write_wav(audio / f"22110{index}-102320_nya_510_elicit_0.wav", seconds=1.5)

        plans = scan_unlabelled(audio)

        assert len(plans) == 3
        assert {p.speaker_id for p in plans} == {UNLABELLED_SPEAKER}
        assert all(p.target_samples == int(1.5 * SAMPLE_RATE) for p in plans)

    def test_accepts_several_roots(self, tmp_path: Path) -> None:
        for language in ("nya", "toi", "loz"):
            _write_wav(tmp_path / language / f"221102-102320_{language}_510_elicit_0.wav")

        plans = scan_unlabelled([tmp_path / lang for lang in ("nya", "toi", "loz")])

        assert len(plans) == 3
        assert len({p.recording_id for p in plans}) == 3

    def test_max_files_bounds_the_scan(self, tmp_path: Path) -> None:
        """Disk, not preference: the full pool is far larger than it needs to be."""
        audio = tmp_path / "nya"
        for index in range(10):
            _write_wav(audio / f"2211{index:02d}-102320_nya_510_elicit_0.wav")

        assert len(scan_unlabelled(audio, max_files=4)) == 4


class TestSplittabilityFloor:
    """Why BembaSpeech can only ever be an evaluation set.

    Not a test of this module, but of the constraint that decides how this
    module's output may be used. BembaSpeech has twelve speakers with two or
    more sessions and a three-way split needs thirteen, so the corpus cannot be
    partitioned into training, development and evaluation parts at all. That
    conclusion is recorded in the H1 results document, and this pins the number
    it rests on so the two cannot drift apart.
    """

    @dataclass(frozen=True)
    class _Item:
        speaker_id: str

    def _speakers(self, count: int) -> list[TestSplittabilityFloor._Item]:
        return [self._Item(f"s{index:02d}") for index in range(count)]

    def test_twelve_speakers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="too few speakers"):
            split_by_speaker(self._speakers(12))

    def test_thirteen_speakers_is_the_floor(self) -> None:
        split = split_by_speaker(self._speakers(13))

        assert len(split.evaluation) == 3
        assert len(split.development) == 3
        assert len(split.train) == 7


class TestRejectUnlabelled:
    def test_passes_labelled_plans_through(self, tmp_path: Path, roster_file: Path) -> None:
        audio = tmp_path / "audio"
        for session in ("200921-192247", "200922-101500"):
            for index in range(4):
                _write_wav(audio / f"01-{session}_bem_d31_elicit_{index}.wav")

        plans = scan_labelled(audio, read_speaker_roster(roster_file), target_seconds=2.0)
        assert reject_unlabelled(plans) == plans

    def test_raises_on_unlabelled_plans(self, tmp_path: Path) -> None:
        """The failure this prevents is silent: one fake speaker, many sessions."""
        audio = tmp_path / "nya"
        _write_wav(audio / "221102-102320_nya_510_elicit_0.wav")

        with pytest.raises(ValueError, match="no speaker identity"):
            reject_unlabelled(scan_unlabelled(audio))

    def test_raises_when_unlabelled_plans_are_mixed_in(self, tmp_path: Path) -> None:
        """The realistic mistake is a concatenation, not a wholesale swap."""
        labelled_audio = tmp_path / "bem"
        for session in ("200921-192247", "200922-101500"):
            for index in range(4):
                _write_wav(labelled_audio / f"01-{session}_bem_d31_elicit_{index}.wav")
        background_audio = tmp_path / "nya"
        _write_wav(background_audio / "221102-102320_nya_510_elicit_0.wav")

        roster_path = tmp_path / "speaker_info.txt"
        roster_path.write_text(ROSTER_TEXT, encoding="utf-8")

        mixed = [
            *scan_labelled(
                labelled_audio, read_speaker_roster(roster_path), target_seconds=2.0
            ),
            *scan_unlabelled(background_audio),
        ]
        with pytest.raises(ValueError, match="no speaker identity"):
            reject_unlabelled(mixed)
