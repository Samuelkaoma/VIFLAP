"""Render the H1 sweep into the results document.

Generated rather than transcribed. A table of thirty cells copied by hand into
prose is a table with a transcription error in it, and the one number that gets
mistyped will be the one someone quotes. This reads the sweep's own JSON and
writes the section, so the document and the data cannot disagree.

Every figure is printed with its condition attached. A ``C_llr`` without its
bitrate, noise type, SNR and duration is not a weak claim but an
uninterpretable one — this system spans a wide range across the design, and a
single number drawn from it means nothing without the cell it came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MARKER = "<!-- Filled from data/reports/h1_sweep.json. -->"


def _fmt(value: float | None, places: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value != value:  # NaN
        return "—"
    return f"{value:.{places}f}"


def _percent(value: float | None) -> str:
    """A rate as a percentage, or a dash where none was computed.

    "nan%" in a results table reads as a defect in the tooling rather than as a
    quantity that does not exist, and invites the reader to discount the row.
    """
    if value is None or value != value:
        return "—"
    return f"{value * 100:.2f}%"


def _is_evaluable(cell: dict[str, Any]) -> bool:
    value = cell["c_llr_min"]
    return isinstance(value, int | float) and value == value  # not NaN


def _verdict(cell: dict[str, Any]) -> str:
    # "Not evaluable" before anything else. A cell where the front-end refused
    # almost every recording did not produce a weak result; it produced no
    # result, and calling that "inconclusive" would put it in the same category
    # as a cell that ran and landed between the thresholds. The first is a
    # statement about the recordings, the second about the experiment.
    if not _is_evaluable(cell):
        return "not evaluable"
    if cell["h1_supported"]:
        return "supported"
    if cell["h1_falsified"]:
        return "**falsified**"
    return "inconclusive"


def _condition_label(cell: dict[str, Any]) -> str:
    if cell["noise_type"] is None:
        return f"{cell['bitrate_kbps']:g} kbit/s, clean"
    return f"{cell['bitrate_kbps']:g} kbit/s, {cell['noise_type']} {cell['snr_db']:g} dB"


def render(payload: dict[str, Any]) -> str:
    """Build the results section from the sweep payload."""
    cells: list[dict[str, Any]] = payload["cells"]
    verdict = payload["verdict"]
    describe = payload.get("model_describe", {})

    codec_modes = sorted({c["codec_mode"] for c in cells})
    lines: list[str] = []

    lines.append(
        f"All figures below come from model `{payload['model_id']}` "
        f"({int(describe.get('ubm_components', 0))} UBM components, rank "
        f"{int(describe.get('ivector_rank', 0))}), evaluated on "
        f"{payload['split']['evaluation_speakers']} speakers disjoint from both "
        f"training and calibration. Channel: **{', '.join(codec_modes)}**."
    )
    lines.append("")
    lines.append(
        "Intervals are 95% percentile bootstraps resampling **speakers**, not "
        "trials. `C_llr_min` is discrimination and decides H1; `C_llr (matched)` "
        "is the cost of the reported likelihood ratio when a calibrator was "
        "fitted on development speakers under the same condition; "
        "`C_llr (transferred)` applies one calibrator fitted under "
        f"`{payload['reference_condition']}` to every cell, which is the "
        "operational case."
    )
    lines.append("")
    lines.append(
        f"The transferred column is empty for `{payload['reference_condition']}` "
        f"at full duration: that is the cell the transferred calibrator was "
        f"fitted from, so a figure there would be the matched one reported twice "
        f"under a name suggesting it had been validated somewhere else."
    )
    lines.append("")

    lines.append(
        "| Condition | Dur. | C_llr_min [95% CI] | C_llr matched | C_llr transf. "
        "| Calib. loss | EER | Refused | H1 |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---|")

    for cell in sorted(
        cells,
        key=lambda c: (
            c["bitrate_kbps"] == 4.75,
            str(c["noise_type"]),
            -(c["snr_db"] or 99),
            -c["duration_seconds"],
        ),
    ):
        interval = (
            f"{_fmt(cell['c_llr_min'])} "
            f"[{_fmt(cell['c_llr_min_lower'])}, {_fmt(cell['c_llr_min_upper'])}]"
        )
        lines.append(
            f"| {_condition_label(cell)} "
            f"| {cell['duration_seconds']:g} s "
            f"| {interval} "
            f"| {_fmt(cell['c_llr_matched'])} "
            f"| {_fmt(cell['c_llr_transferred'])} "
            f"| {_fmt(cell['calibration_loss_matched'])} "
            f"| {_percent(cell['equal_error_rate'])} "
            f"| {cell['refusal_rate'] * 100:.1f}% "
            f"| {_verdict(cell)} |"
        )

    unevaluable = [c for c in cells if not _is_evaluable(c)]
    if unevaluable:
        lines.append("")
        lines.append(
            f"**{len(unevaluable)} of {len(cells)} cells produced no metric at "
            f"all**, because the front-end refused almost every recording in "
            f"them: "
            + ", ".join(
                f"`{c['condition']}` at {c['duration_seconds']:g} s "
                f"({c['refusal_rate'] * 100:.1f}% refused)"
                for c in unevaluable
            )
            + ". These are excluded from the counts below rather than scored as "
            "failures. Refusing is the designed behaviour — an i-vector from "
            "under three seconds of speech reflects the model's prior rather "
            "than the recording — but it means the system has no acoustic "
            "opinion whatever at these operating points, which is a stronger "
            "statement than a poor one."
        )

    lines.append("")
    lines.append("### Verdict on H1")
    lines.append("")
    lines.append(f"Decision rule, fixed in advance: {verdict['decision_rule']}.")
    lines.append("")
    lines.append(
        f"Of {verdict['n_cells']} evaluable cells: **{verdict['n_supported']} "
        f"supported, {verdict['n_falsified']} falsified, "
        f"{verdict['n_inconclusive']} inconclusive**."
    )
    lines.append("")
    best, worst = verdict["best_cell"], verdict["worst_cell"]
    lines.append(
        f"- Best cell: `{best['condition']}` at {best['duration']:g} s — "
        f"C_llr_min {best['c_llr_min']:.3f} "
        f"[{best['interval'][0]:.3f}, {best['interval'][1]:.3f}]"
    )
    lines.append(
        f"- Worst cell: `{worst['condition']}` at {worst['duration']:g} s — "
        f"C_llr_min {worst['c_llr_min']:.3f} "
        f"[{worst['interval'][0]:.3f}, {worst['interval'][1]:.3f}]"
    )
    lines.append("")

    trials = next((c for c in cells if c["n_same_source"]), None)
    if trials is not None:
        lines.append(
            f"Trial counts at full duration: {trials['n_same_source']} same-source "
            f"(cross-session only) and {trials['n_different_source']} "
            f"different-source, over {trials['n_evaluation_speakers']} speakers. "
            f"The speaker count, not the trial count, is the effective sample "
            f"size, and it is what the intervals reflect."
        )
        lines.append("")

    lines.append(
        f"Sweep completed in {payload['elapsed_minutes']:.0f} minutes over "
        f"{len(cells)} cells."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("data/reports/h1_sweep.json"))
    parser.add_argument(
        "--document", type=Path, default=Path("docs/H1-acoustic-results.md")
    )
    arguments = parser.parse_args(argv)

    # Explicit UTF-8 on both ends. Python's default is the locale encoding,
    # which on this platform is cp1252: the em dashes and the "<=" glyphs in the
    # rendered section would be written as cp1252 bytes into a file every other
    # tool reads as UTF-8, and the corruption would appear in the middle of the
    # results table rather than as a failure.
    payload = json.loads(arguments.sweep.read_text(encoding="utf-8"))
    section = render(payload)

    document = arguments.document.read_text(encoding="utf-8")
    if MARKER not in document:
        raise SystemExit(
            f"{arguments.document} has no results placeholder; refusing to guess "
            f"where the table belongs"
        )
    arguments.document.write_text(document.replace(MARKER, section), encoding="utf-8")
    print(f"wrote results into {arguments.document}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
