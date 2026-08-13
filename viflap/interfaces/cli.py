"""Command line for training, evaluation and audit verification.

Kept deliberately narrow. There is no command that runs a comparison, because a
comparison requires a case reference, an authenticated principal, and an audit
sink — and a shell invocation that carries all three by flag is an invitation to
put them in a script and lose the accountability they exist to provide. Queries
go through the API, where the principal is resolved and the audit entry is
written by the same code path that produces the result.

What the command line is for is the work that is legitimately batch: training
models, running an evaluation, and — the important one — letting an oversight
body verify the audit chain without going through the service that wrote it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]


def _verify_audit(arguments: argparse.Namespace) -> int:
    """Verify an audit log's hash chain.

    Deliberately does not import the application layer. An oversight body should
    be able to verify the chain with the smallest possible amount of the
    operator's software, and ideally with an independent implementation of
    SHA-256 and the canonical serialisation — which the output of this command
    documents well enough to write.
    """
    from viflap.infrastructure.audit import FileAuditLog

    path = Path(arguments.path)
    if not path.exists():
        print(f"No audit log at {path}", file=sys.stderr)
        return 2

    verification = FileAuditLog(path).verify()
    report = {
        "path": str(path),
        "intact": verification.is_intact,
        "entries": verification.n_entries,
        "first_broken_index": verification.first_broken_index,
        "detail": verification.detail,
    }
    print(json.dumps(report, indent=2))

    if not verification.is_intact:
        print(
            "\nThe chain does not verify. Entries before index "
            f"{verification.first_broken_index} remain trustworthy; entries from "
            f"that point on cannot be relied upon. This is a matter for "
            f"investigation, not for re-running the command.",
            file=sys.stderr,
        )
        return 1
    return 0


def _describe_model(arguments: argparse.Namespace) -> int:
    """Print a trained model's identity and properties."""
    from viflap.analysis.speaker.pipeline import SpeakerComparisonSystem

    path = Path(arguments.path)
    if not path.exists():
        print(f"No model at {path}", file=sys.stderr)
        return 2

    system = SpeakerComparisonSystem.load(path)
    print(json.dumps(system.describe(), indent=2, default=str))
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    """Evaluate scores against ground truth and report forensic metrics.

    Takes a JSON Lines file of ``{"score": …, "label": …, "speaker": …}``. The
    speaker field is required rather than optional: without it the only interval
    obtainable is a trial-level one, which is too narrow and would be reported
    as though it were not.
    """
    import numpy as np

    from viflap.analysis.calibration.metrics import compute_cllr, compute_cllr_min, evaluate
    from viflap.evaluation.splits import bootstrap_over_speakers

    path = Path(arguments.path)
    if not path.exists():
        print(f"No score file at {path}", file=sys.stderr)
        return 2

    scores, labels, speakers = [], [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            scores.append(float(record["score"]))
            labels.append(int(record["label"]))
            speakers.append(str(record["speaker"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"Line {line_number} is unusable: {exc}", file=sys.stderr)
            return 2

    score_array = np.array(scores)
    label_array = np.array(labels)

    summary = evaluate(score_array, label_array)
    print(summary.describe())
    print()

    for name, metric in (("C_llr", compute_cllr), ("C_llr_min", compute_cllr_min)):
        estimate = bootstrap_over_speakers(
            metric, score_array, label_array, speakers, n_resamples=arguments.resamples
        )
        print(f"{name:12s} {estimate}")

    if summary.is_misleading:
        print(
            "\nC_llr exceeds 1.0. This system is worse than uninformative: an "
            "investigator relying on it would decide less well than one ignoring "
            "it.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viflap",
        description=(
            "VIFLAP command line. Comparisons are not available here: they "
            "require a case reference, an authenticated principal and an audit "
            "entry, and go through the API where all three are enforced."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-audit", help="verify an audit log's hash chain")
    verify.add_argument("path", help="path to the audit log")
    verify.set_defaults(handler=_verify_audit)

    describe = subparsers.add_parser(
        "describe-model", help="print a trained model's identity and properties"
    )
    describe.add_argument("path", help="path to the model archive")
    describe.set_defaults(handler=_describe_model)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="compute forensic metrics from a score file"
    )
    evaluate_parser.add_argument("path", help="JSON Lines: score, label, speaker")
    evaluate_parser.add_argument(
        "--resamples", type=int, default=1000, help="bootstrap resamples over speakers"
    )
    evaluate_parser.set_defaults(handler=_evaluate)

    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
