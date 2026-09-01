"""Count what AfriSpeech-200 actually offers a Zambian speaker-recognition system.

§8 surveyed the corpora available for the target population and left AfriSpeech-200
unresolved, recording it as blocked behind account creation and terms acceptance.
**It is not blocked.** The dataset reports ``gated=False`` and its per-utterance
manifests are served without authentication, so the question could have been
answered at any time by reading three CSVs totalling 22 MB — no audio, no account,
nothing to agree to.

What is counted, and why it is speakers rather than hours
---------------------------------------------------------
§8's finding is that the resource position for Zambian forensic speech is far
worse than the hour counts suggest, because speaker recognition needs *labelled
speakers* and the corpora were built for speech recognition, where transcribed
hours is the figure of merit. So this counts speakers, and counts them by country
as well as by accent, because an accent label is a claim about how someone speaks
and a country field is a claim about where they are.

The usable-speaker threshold is 60 seconds of total speech, which is the minimum
for two 30-second recordings — one to enrol and one to test. It is a floor on
being *splittable at all*, not a claim that 60 seconds is sufficient: §5 measures
what this system does at 30 seconds and §22 does the same, and neither suggests a
speaker with exactly two such recordings contributes much.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

#: Where the manifests live. Public, ungated, and small.
BASE = (
    "https://huggingface.co/datasets/intronhealth/afrispeech-200/resolve/main/transcripts"
)
SPLITS = ("train.csv", "dev.csv", "test.csv")

#: Seconds of total speech below which a speaker cannot supply two 30-second
#: recordings and therefore cannot appear on both sides of a trial.
USABLE_SECONDS = 60.0

#: Zambia's ISO code, and the languages §8 is asking about. Kept explicit so the
#: absence of a result is a recorded absence rather than a query nobody ran.
ZAMBIA = "ZM"
ZAMBIAN_LANGUAGES = ("bemba", "nyanja", "chichewa", "tonga", "lozi", "lunda", "luvale")


def summarise(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Speaker counts by country and by accent, and the Zambian answer.

    Takes parsed rows rather than a path so the arithmetic is testable without a
    network round trip — the fetch is the part that cannot be tested here, and
    it is also the part that cannot be got wrong quietly.
    """
    seconds: defaultdict[str, float] = defaultdict(float)
    utterances: Counter[str] = Counter()
    countries: defaultdict[str, set[str]] = defaultdict(set)
    accents: defaultdict[str, set[str]] = defaultdict(set)

    for row in rows:
        speaker = row["user_ids"]
        seconds[speaker] += float(row["duration"])
        utterances[speaker] += 1
        countries[row["country"].strip().upper()].add(speaker)
        accents[row["accent"].strip().lower()].add(speaker)

    def usable(speakers: Iterable[str]) -> int:
        return sum(1 for s in speakers if seconds[s] >= USABLE_SECONDS)

    by_country = {
        code: {
            "speakers": len(speakers),
            "usable_speakers": usable(speakers),
            "hours": round(sum(seconds[s] for s in speakers) / 3600.0, 2),
        }
        for code, speakers in sorted(countries.items(), key=lambda item: -len(item[1]))
    }
    by_language = {
        name: {
            "speakers": len(accents.get(name, set())),
            "usable_speakers": usable(accents.get(name, set())),
            "hours": round(sum(seconds[s] for s in accents.get(name, set())) / 3600.0, 2),
        }
        for name in ZAMBIAN_LANGUAGES
    }

    return {
        "n_utterances": len(rows),
        "n_speakers": len(seconds),
        "usable_seconds_threshold": USABLE_SECONDS,
        "by_country": by_country,
        "zambian_languages": by_language,
        "zambia_present": ZAMBIA in countries,
        "n_zambian_speakers": len(countries.get(ZAMBIA, set())),
    }


def load(directory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in SPLITS:
        with (directory / name).open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests",
        type=Path,
        required=True,
        help=f"Directory holding {', '.join(SPLITS)} fetched from {BASE}",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/afrispeech_survey.json")
    )
    arguments = parser.parse_args(argv)

    rows = load(arguments.manifests)
    summary = summarise(rows)
    summary["source"] = BASE

    print(
        f"{summary['n_utterances']} utterances, {summary['n_speakers']} speakers",
        flush=True,
    )
    print(f"Zambia present: {summary['zambia_present']}", flush=True)
    for code, entry in list(summary["by_country"].items())[:8]:
        print(
            f"  {code or '(blank)':<8} {entry['speakers']:>5} speakers, "
            f"{entry['usable_speakers']:>5} usable, {entry['hours']:>7.2f} h",
            flush=True,
        )
    print("Zambian languages:", flush=True)
    for name, entry in summary["zambian_languages"].items():
        print(
            f"  {name:<10} {entry['speakers']:>4} speakers, "
            f"{entry['usable_speakers']:>4} usable, {entry['hours']:>6.2f} h",
            flush=True,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
