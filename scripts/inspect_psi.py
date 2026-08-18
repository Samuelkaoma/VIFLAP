"""What produces the ψ₁ spike, tested against an extractor that shares none of it.

§1 recorded that the leading PLDA eigenvalue is five to seven times the second
across every i-vector model this project has trained, and that one dominant axis
of between-speaker variation is what a nuisance factor absorbed into the speaker
subspace looks like. §21 refuted the most promising explanation — condition
stratification moved ψ₁ only 44.951 → 43.967 — and left three candidates:
LibriSpeech session or environment effects, length normalisation, and upward bias
in a leading eigenvalue estimated from a few hundred speakers.

§22 supplies an instrument none of those sections had. ECAPA embeddings share the
corpus, the channel and the back-end with the i-vector system and share *nothing*
of its front-end: no MFCCs, no GMM-UBM, no total-variability subspace. So the
ratio either survives the change of extractor or it does not, and that alone
partitions the candidates.

What this can and cannot separate
---------------------------------
It cannot clear the back-end. Length normalisation, LDA, WCCN and the PLDA
implementation are *shared* between the two systems, so a ratio that survives is
consistent with any of them being the cause. That is why the length-normalisation
arm below exists: it is the one shared component cheap enough to switch off.

The speaker-count sweep addresses the third candidate directly. If the spike were
upward bias in a leading eigenvalue estimated from few speakers, fewer speakers
should give a *larger* ratio. §21 already noted the 125-speaker i-vector model
shows a smaller one, which is the wrong direction; this repeats that as a
controlled sweep rather than a comparison between two models that differ in other
ways too.

Several draws per speaker count, because a single subsample confounds the count
with which speakers happened to be drawn — the error §11 was criticised for.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.speaker.pipeline import SpeakerComparisonSystem
from viflap.analysis.speaker.plda import PldaConfig, PldaModel, train_plda
from viflap.analysis.speaker.transforms import fit_transform_chain

#: Models whose stored spectra are reported for comparison. §1's claim is about
#: the range across all of them, so quoting one would understate it.
IVECTOR_MODELS: tuple[tuple[str, str], ...] = (
    ("acoustic", "models/acoustic.npz"),
    ("acoustic_large", "models/acoustic_large.npz"),
    ("acoustic_pooled", "models/acoustic_pooled.npz"),
    ("acoustic_pooled_cmvn_utt", "models/acoustic_pooled_cmvn_utt.npz"),
    ("acoustic_pooled_cmvn100", "models/acoustic_pooled_cmvn100.npz"),
)


def spectrum(psi: NDArray[np.float64]) -> dict[str, float]:
    """The shape of the between-speaker spectrum, not just its head.

    ``ratio`` is ψ₁/ψ₂ because that is the quantity §1 states and tracks. The
    share and the effective dimension are here because a ratio alone cannot
    distinguish "one enormous axis over a flat tail" from "a steeply decaying
    spectrum", and those are different claims about what the model learned.
    """
    ordered = np.sort(np.asarray(psi, dtype=np.float64))[::-1]
    total = float(ordered.sum())
    return {
        "dimension": int(ordered.size),
        "psi_1": float(ordered[0]),
        "psi_2": float(ordered[1]),
        "ratio": float(ordered[0] / ordered[1]),
        "psi_1_share_of_total": float(ordered[0] / total) if total > 0 else float("nan"),
        "top_five": [round(float(v), 3) for v in ordered[:5]],
        "n_above_inert": int(np.count_nonzero(ordered > 0.1)),
    }


def fit(
    vectors: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    length_normalise: bool = True,
    lda_dimension: int | None = None,
) -> PldaModel:
    transform = fit_transform_chain(
        vectors, labels, lda_dimension=lda_dimension, length_normalise=length_normalise
    )
    return train_plda(transform.apply(vectors), labels, PldaConfig(max_iterations=40))


def load_embeddings(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    archive = np.load(path, allow_pickle=False)
    vectors = archive["train|vectors"]
    names = [str(name) for name in archive["train|speakers"]]
    codes = {name: index for index, name in enumerate(sorted(set(names)))}
    return vectors, np.array([codes[n] for n in names], dtype=np.int64), len(codes)


def subsample(
    vectors: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_speakers: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """All recordings of a random subset of speakers, relabelled contiguously."""
    unique = np.unique(labels)
    drawn = rng.choice(unique, size=n_speakers, replace=False)
    keep = np.isin(labels, drawn)
    remap = {old: new for new, old in enumerate(sorted(drawn.tolist()))}
    return vectors[keep], np.array([remap[int(v)] for v in labels[keep]], dtype=np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/reports/neural_embeddings.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/psi_spectrum.json")
    )
    parser.add_argument(
        "--speaker-counts", type=int, nargs="+", default=[75, 150, 225, 306]
    )
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    report: dict[str, object] = {}

    # -- The i-vector models, from what they already stored ------------------
    stored: dict[str, dict[str, float]] = {}
    for name, path in IVECTOR_MODELS:
        if not Path(path).exists():
            continue
        system = SpeakerComparisonSystem.load(path)
        stored[name] = spectrum(system.plda.psi)
        print(
            f"{name:<28} dim {stored[name]['dimension']:>3}  "
            f"psi1 {stored[name]['psi_1']:7.3f}  ratio {stored[name]['ratio']:.3f}",
            flush=True,
        )
    report["ivector_models"] = stored

    # -- The borrowed extractor, same back-end -------------------------------
    vectors, labels, n_speakers = load_embeddings(arguments.embeddings)
    print(
        f"\nECAPA embeddings: {vectors.shape[0]} recordings, {vectors.shape[1]} dims, "
        f"{n_speakers} speakers",
        flush=True,
    )

    neural = spectrum(fit(vectors, labels).psi)
    print(
        f"{'ecapa (length-normalised)':<28} dim {neural['dimension']:>3}  "
        f"psi1 {neural['psi_1']:7.3f}  ratio {neural['ratio']:.3f}",
        flush=True,
    )

    # Length normalisation is the one component shared with the i-vector system
    # that can be switched off without changing anything else.
    unnormalised = spectrum(fit(vectors, labels, length_normalise=False).psi)
    print(
        f"{'ecapa (no length norm)':<28} dim {unnormalised['dimension']:>3}  "
        f"psi1 {unnormalised['psi_1']:7.3f}  ratio {unnormalised['ratio']:.3f}",
        flush=True,
    )
    report["ecapa"] = {"length_normalised": neural, "unnormalised": unnormalised}

    # -- Does the ratio grow when speakers are removed? ----------------------
    # The estimation-bias hypothesis predicts it should. The LDA ceiling moves
    # with the speaker count, so the transform dimension is pinned to the
    # smallest count's ceiling; otherwise this would sweep two things at once.
    ceiling = min(arguments.speaker_counts) - 1
    print(
        f"\nspeaker sweep at a fixed transform dimension of {ceiling}, "
        f"{arguments.draws} draws each",
        flush=True,
    )
    sweep: list[dict[str, object]] = []
    for count in arguments.speaker_counts:
        ratios: list[float] = []
        psi_ones: list[float] = []
        for draw in range(arguments.draws):
            rng = np.random.default_rng(arguments.seed + 1000 * draw + count)
            if count >= n_speakers:
                subset_vectors, subset_labels = vectors, labels
            else:
                subset_vectors, subset_labels = subsample(vectors, labels, count, rng)
            measured = spectrum(
                fit(subset_vectors, subset_labels, lda_dimension=ceiling).psi
            )
            ratios.append(measured["ratio"])
            psi_ones.append(measured["psi_1"])
            if count >= n_speakers:
                # Every draw is the same full set; one is enough.
                break
        entry = {
            "n_speakers": count,
            "n_draws": len(ratios),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_lower": float(np.min(ratios)),
            "ratio_upper": float(np.max(ratios)),
            "psi_1_mean": float(np.mean(psi_ones)),
        }
        sweep.append(entry)
        print(
            f"  {count:>4} speakers  ratio {entry['ratio_mean']:.3f} "
            f"[{entry['ratio_lower']:.3f}, {entry['ratio_upper']:.3f}]  "
            f"psi1 {entry['psi_1_mean']:7.3f}  ({entry['n_draws']} draws)",
            flush=True,
        )
    report["speaker_sweep"] = sweep
    report["speaker_sweep_transform_dimension"] = ceiling

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
