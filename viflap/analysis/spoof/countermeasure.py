"""Synthetic and converted speech detection.

Architecturally this is not an evidence stream. It contributes no likelihood
ratio and it does not vote. It conditions the **admissibility** of the acoustic
stream, and the distinction is the point: a synthetic recording does not carry
weak information about a human speaker's anatomy, it carries none. Down-weighting
it would assert that a text-to-speech rendering is slightly informative about the
vocal tract of the person who typed the text.

The detector
------------
LFCC front-end into two Gaussian mixtures — one for ``bona fide`` speech, one for
spoofed — scored as the mean per-frame log-likelihood ratio. This is the ASVspoof
baseline, chosen deliberately over something more elaborate:

- It is a **generative** model of both classes, so its score is already a
  log-likelihood ratio in form, and calibrating it is a one-dimensional problem
  with a clear interpretation.
- Its failure mode is legible. A discriminative deep model that fails on an
  unseen attack fails opaquely; a mixture model that assigns low likelihood
  under *both* classes is visibly out of its domain, and that condition can be
  detected and reported rather than silently producing a confident score.
- It does not need the training data that this deployment does not have.

Linear rather than mel spacing is essential and not a preference. Synthesis
artefacts — vocoder phase discontinuities, harmonic structure terminating
abruptly at the synthesiser's cutoff, unnaturally regular spectral fine
structure — concentrate in the upper spectrum. A mel filterbank averages that
region into a few wide channels, which is auditorily appropriate and destroys
precisely the evidence being looked for.

The generalisation problem, stated plainly
------------------------------------------
The consistent finding across every edition of the ASVspoof challenge is that
countermeasure performance degrades severely against synthesis methods absent
from training. That is the central limitation of this component and it is not
solved here. What is done instead:

- The operating point is deliberately conservative, and the asymmetry is stated:
  admitting synthetic speech contaminates evidence presented to a court, whereas
  excluding genuine speech only weakens a case.
- An out-of-domain indicator is computed. When a recording is improbable under
  both models, the verdict is ``INDETERMINATE`` rather than a confident call,
  and ``INDETERMINATE`` does not admit acoustic evidence.
- Cross-attack evaluation is a first-class operation
  (:meth:`SpoofingCountermeasure.evaluate_cross_attack`), because a
  countermeasure evaluated only on attacks it was trained against reports a
  number with no bearing on how it will behave in service.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.dsp.framing import FrameConfig
from viflap.analysis.dsp.spectral import CepstralConfig, add_deltas, compute_cepstra
from viflap.analysis.speaker.gmm import GaussianMixture, GmmConfig, train_ubm
from viflap.domain.errors import (
    InsufficientDataError,
    InvalidEvidenceError,
)

__all__ = [
    "CountermeasureConfig",
    "CountermeasureScore",
    "SpoofingCountermeasure",
    "TrainingExample",
]


@dataclass(frozen=True, slots=True)
class CountermeasureConfig:
    """Training and scoring parameters for the countermeasure."""

    frame: FrameConfig = FrameConfig()
    cepstral: CepstralConfig = CepstralConfig.for_spoofing_countermeasure()
    delta_order: int = 2
    delta_window: int = 2
    n_components: int = 128
    max_iterations: int = 60
    seed: int = 0

    min_frames: int = 100
    """Minimum frames for a score. One second of audio does not distinguish a
    vocoder from a bad line."""

    chunk_frames: int = 16_384
    """Frames per block when scoring pooled training material.

    Scoring holds an ``n_frames x n_components`` array and ``logsumexp`` keeps
    several copies of it. At 64 components a training set of 1.8 million frames
    needs 877 MB per copy, which is an allocation failure part-way through a
    long run rather than anything informative. Chunking bounds the peak without
    changing the result: the quantity computed is a percentile over frames, and
    frames are independent."""

    out_of_domain_percentile: float = 1.0
    """Frames whose likelihood under *both* models falls below this percentile
    of the training distribution are counted as out of domain. A recording
    dominated by such frames is one the detector has no basis to judge, which is
    the expected condition for a synthesis method it has never seen — the
    failure this component is known to have."""

    def __post_init__(self) -> None:
        if self.chunk_frames < 1:
            raise InvalidEvidenceError(
                "chunk_frames must be at least one frame",
                chunk_frames=self.chunk_frames,
            )
        if self.cepstral.scale.value != "linear":
            raise InvalidEvidenceError(
                "the countermeasure requires a linearly spaced filterbank; mel "
                "spacing averages away the upper-spectrum artefacts that "
                "distinguish synthetic speech",
                scale=self.cepstral.scale.value,
            )

    @classmethod
    def compact(cls) -> CountermeasureConfig:
        """Small configuration for tests, exercising the same code paths."""
        return cls(n_components=8, max_iterations=25)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One labelled recording for countermeasure training."""

    signal: NDArray[np.float64]
    sample_rate: int
    is_bona_fide: bool
    attack_id: str = "none"
    """Identifier of the synthesis or conversion method, ``"none"`` for genuine
    speech. Required for cross-attack evaluation: without it, a held-out split
    cannot be constructed and the only number obtainable is the one that
    overstates performance."""


@dataclass(frozen=True, slots=True)
class CountermeasureScore:
    """The detector's assessment of one recording.

    ``log_likelihood_ratio`` is **calibrated**. This matters because the gate
    that consumes it expresses its operating point as a strength of evidence —
    "admit above ten to one" — and that phrasing is only meaningful on a
    calibrated scale. The raw mean per-frame log-ratio of two Gaussian mixtures
    is on an arbitrary scale that grows with the feature dimension and the frame
    count; comparing it against a threshold chosen as a strength of evidence
    silently compares a number against an unrelated one. The uncalibrated value
    is retained for diagnosis.
    """

    log_likelihood_ratio: float
    """Calibrated log-likelihood ratio of bona fide against spoofed. Positive
    supports genuine speech."""

    raw_score: float
    """Mean per-frame log-ratio before calibration, on the detector's own
    arbitrary scale. Diagnostic only."""

    n_frames: int
    out_of_domain_fraction: float
    """Fraction of frames improbable under both models. High values mean the
    recording is unlike anything in training, and the score should not be
    trusted regardless of its magnitude."""

    frame_score_std: float
    """Dispersion of per-frame scores, on the calibrated scale."""

    dispersion_ratio: float
    """This recording's per-frame dispersion relative to what training recordings
    typically show. A partially manipulated recording — a synthetic segment
    spliced into genuine audio — has a bimodal frame score distribution that the
    mean conceals and this exposes.

    Expressed as a ratio rather than an absolute figure because the absolute
    dispersion depends on the feature dimension, the calibration slope and the
    frame count, none of which say anything about the recording. A threshold on
    the raw figure is a threshold on the configuration; a threshold on the ratio
    is a threshold on "unusually inconsistent for this detector"."""

    detector_id: str

    @property
    def is_out_of_domain(self) -> bool:
        return self.out_of_domain_fraction > 0.25

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the recording looks internally inconsistent."""
        return self.dispersion_ratio > 2.5


class SpoofingCountermeasure:
    """A trained two-class LFCC-GMM countermeasure."""

    _FORMAT_VERSION = 1

    def __init__(
        self,
        config: CountermeasureConfig,
        bona_fide: GaussianMixture,
        spoofed: GaussianMixture,
        out_of_domain_threshold: float,
        training_summary: Mapping[str, float],
        calibration: tuple[float, float] = (1.0, 0.0),
        typical_frame_std: float = 1.0,
    ) -> None:
        self._config = config
        self._bona_fide = bona_fide
        self._spoofed = spoofed
        self._out_of_domain_threshold = out_of_domain_threshold
        self._training_summary = dict(training_summary)
        # Affine calibration (slope, intercept) mapping the raw mean per-frame
        # log-ratio onto a log-likelihood-ratio scale. Fitted by
        # cross-validation during training; the identity is a deliberate,
        # clearly-wrong default so that an uncalibrated detector is obvious
        # rather than plausible.
        self._calibration = calibration
        # Median per-frame dispersion across training recordings, on the raw
        # scale. The reference against which a recording's own dispersion is
        # judged unusual.
        self._typical_frame_std = max(typical_frame_std, 1e-9)
        self._detector_id = self._compute_detector_id()

    def _compute_detector_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self._FORMAT_VERSION).encode())
        for model in (self._bona_fide, self._spoofed):
            for array in (model.weights, model.means, model.variances):
                digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return f"lfcc-gmm-{digest.hexdigest()[:16]}"

    @property
    def detector_id(self) -> str:
        return self._detector_id

    @property
    def training_summary(self) -> dict[str, float]:
        return dict(self._training_summary)

    # -- Feature extraction ------------------------------------------------

    def _features(
        self, signal: NDArray[np.float64], sample_rate: int
    ) -> NDArray[np.float64]:
        """LFCC with deltas.

        No voice activity detection and no cepstral mean normalisation, unlike
        the speaker front-end. Both would be actively harmful here.

        Silence is *evidence* for this task: a synthesiser's silence is not a
        room's silence, it is a numerically clean absence of signal, or comfort
        noise with the wrong spectrum. Discarding non-speech frames throws away
        one of the more reliable cues.

        Cepstral mean normalisation removes any time-invariant linear filter.
        The systematic spectral signature of a vocoder is exactly such a filter,
        so normalising it away removes the artefact being detected. This is the
        same operation that is essential for the speaker back-end and
        counterproductive here, which is why the two front-ends are separate
        rather than shared.
        """
        frame_config = (
            self._config.frame
            if self._config.frame.sample_rate == sample_rate
            else self._config.frame.with_sample_rate(sample_rate)
        )
        cepstra, _ = compute_cepstra(signal, frame_config, self._config.cepstral)
        return add_deltas(
            cepstra, window=self._config.delta_window, order=self._config.delta_order
        )

    # -- Training ----------------------------------------------------------

    @classmethod
    def train(
        cls,
        examples: Sequence[TrainingExample],
        config: CountermeasureConfig | None = None,
    ) -> SpoofingCountermeasure:
        """Train both class models.

        Raises
        ------
        InsufficientDataError
            If either class is absent or too small. A one-class model would score
            everything against a single distribution, and the resulting number
            would rank recordings by how ordinary they are rather than by
            whether they are synthetic.
        """
        config = config or CountermeasureConfig()
        if not examples:
            raise InsufficientDataError("no examples supplied for training")

        instance = cls.__new__(cls)
        instance._config = config

        bona_fide_features: list[NDArray[np.float64]] = []
        spoofed_features: list[NDArray[np.float64]] = []
        attacks: set[str] = set()

        for example in examples:
            features = instance._features(example.signal, example.sample_rate)
            if features.shape[0] < config.min_frames:
                continue
            if example.is_bona_fide:
                bona_fide_features.append(features)
            else:
                spoofed_features.append(features)
                attacks.add(example.attack_id)

        if not bona_fide_features or not spoofed_features:
            raise InsufficientDataError(
                "both genuine and spoofed examples are required; a single-class "
                "model cannot distinguish synthesis from unusual speech",
                n_bona_fide=len(bona_fide_features),
                n_spoofed=len(spoofed_features),
            )

        gmm_config = GmmConfig(
            n_components=config.n_components,
            max_iterations=config.max_iterations,
            seed=config.seed,
        )
        bona_fide = train_ubm(bona_fide_features, gmm_config)
        spoofed = train_ubm(spoofed_features, gmm_config)

        # The out-of-domain threshold is the low percentile of the best-of-both
        # frame likelihood across training data. Frames below it at scoring time
        # are unlike anything either model was fitted to.
        #
        # Computed in chunks. ``log_likelihood`` forms an
        # ``n_frames x n_components`` array and ``logsumexp`` holds several
        # copies of it, so pooling every frame of a realistic training set into
        # one call allocates gigabytes for a quantity that is one number. The
        # per-chunk results are one-dimensional and cost a few megabytes in
        # total, and the percentile over the concatenation is identical to the
        # percentile over a single pass.
        pooled = np.concatenate(bona_fide_features + spoofed_features)
        chunk = config.chunk_frames
        best_of_both = np.concatenate(
            [
                np.maximum(
                    bona_fide.log_likelihood(pooled[start : start + chunk]),
                    spoofed.log_likelihood(pooled[start : start + chunk]),
                )
                for start in range(0, pooled.shape[0], chunk)
            ]
        )
        threshold = float(np.percentile(best_of_both, config.out_of_domain_percentile))

        calibration = _fit_calibration_by_cross_validation(
            bona_fide_features, spoofed_features, gmm_config
        )

        # Reference within-recording dispersion, from training recordings.
        #
        # The ninetieth percentile, not the median. The check this feeds exists
        # to catch a *spliced* recording — genuine audio with a synthetic
        # segment inserted — whose per-frame scores are bimodal. Spoofed
        # recordings are legitimately more dispersed than genuine ones across
        # the board, so a reference at the median of the pooled classes flags
        # ordinary spoofed material as "inconsistent" and downgrades a confident
        # exclusion to "cannot tell". The upper percentile places the reference
        # above normal within-class variation, so only genuinely anomalous
        # dispersion trips it.
        per_recording_std = [
            float(
                np.std(
                    bona_fide.log_likelihood(features) - spoofed.log_likelihood(features)
                )
            )
            for features in (*bona_fide_features, *spoofed_features)
        ]
        typical_frame_std = float(np.percentile(per_recording_std, 90))

        summary = {
            "n_bona_fide_recordings": float(len(bona_fide_features)),
            "n_spoofed_recordings": float(len(spoofed_features)),
            "n_attack_types": float(len(attacks)),
            "n_frames": float(pooled.shape[0]),
            "calibration_slope": calibration[0],
            "calibration_intercept": calibration[1],
            "typical_frame_std": typical_frame_std,
        }
        return cls(
            config,
            bona_fide,
            spoofed,
            threshold,
            summary,
            calibration,
            typical_frame_std,
        )

    # -- Scoring -----------------------------------------------------------

    def score(self, signal: NDArray[np.float64], sample_rate: int) -> CountermeasureScore:
        """Score one recording."""
        features = self._features(signal, sample_rate)
        if features.shape[0] < self._config.min_frames:
            raise InsufficientDataError(
                "recording is too short for the countermeasure to assess",
                n_frames=int(features.shape[0]),
                required=self._config.min_frames,
            )

        bona_fide_frames = self._bona_fide.log_likelihood(features)
        spoofed_frames = self._spoofed.log_likelihood(features)
        frame_scores = bona_fide_frames - spoofed_frames

        best_of_both = np.maximum(bona_fide_frames, spoofed_frames)
        out_of_domain = float(np.mean(best_of_both < self._out_of_domain_threshold))

        raw = float(np.mean(frame_scores))
        raw_std = float(np.std(frame_scores))
        slope, intercept = self._calibration

        return CountermeasureScore(
            log_likelihood_ratio=float(slope * raw + intercept),
            raw_score=raw,
            n_frames=int(features.shape[0]),
            out_of_domain_fraction=out_of_domain,
            frame_score_std=float(abs(slope) * raw_std),
            dispersion_ratio=float(raw_std / self._typical_frame_std),
            detector_id=self._detector_id,
        )

    @property
    def calibration(self) -> tuple[float, float]:
        """Fitted affine calibration (slope, intercept)."""
        return self._calibration

    def evaluate_cross_attack(
        self,
        examples: Sequence[TrainingExample],
        held_out_attacks: Sequence[str],
    ) -> dict[str, float]:
        """Evaluate on attack types excluded from training.

        This is the experiment for hypothesis H7, and it is the only evaluation
        of a countermeasure that means anything operationally. Performance on
        attacks present in training is an upper bound that will not be seen in
        service, because an offender who has a synthesiser the training set
        contains is not the offender this system will meet.

        Returns per-attack mean scores and separation from the genuine
        distribution, so that a method the detector fails on is identifiable
        individually rather than averaged into an aggregate that conceals it.
        """
        held_out = set(held_out_attacks)
        genuine: list[float] = []
        by_attack: dict[str, list[float]] = {}

        for example in examples:
            try:
                score = self.score(example.signal, example.sample_rate)
            except InsufficientDataError:
                continue
            if example.is_bona_fide:
                genuine.append(score.log_likelihood_ratio)
            elif example.attack_id in held_out:
                by_attack.setdefault(example.attack_id, []).append(
                    score.log_likelihood_ratio
                )

        if not genuine or not by_attack:
            raise InsufficientDataError(
                "cross-attack evaluation needs genuine examples and at least one "
                "held-out attack type",
                n_genuine=len(genuine),
                n_held_out_attacks=len(by_attack),
            )

        genuine_array = np.array(genuine)
        results: dict[str, float] = {
            "genuine_mean": float(np.mean(genuine_array)),
            "genuine_std": float(np.std(genuine_array)),
        }
        for attack, scores in sorted(by_attack.items()):
            attack_array = np.array(scores)
            pooled_std = np.sqrt(0.5 * (genuine_array.var() + attack_array.var()))
            results[f"{attack}_mean"] = float(np.mean(attack_array))
            results[f"{attack}_separation"] = float(
                (genuine_array.mean() - attack_array.mean()) / max(pooled_std, 1e-9)
            )
        return results

    # -- Persistence -------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array([self._FORMAT_VERSION]),
            detector_id=np.array([self._detector_id]),
            calibration=np.array(self._calibration),
            typical_frame_std=np.array([self._typical_frame_std]),
            bona_fide_weights=self._bona_fide.weights,
            bona_fide_means=self._bona_fide.means,
            bona_fide_variances=self._bona_fide.variances,
            spoofed_weights=self._spoofed.weights,
            spoofed_means=self._spoofed.means,
            spoofed_variances=self._spoofed.variances,
            out_of_domain_threshold=np.array([self._out_of_domain_threshold]),
            training_summary=np.array([json.dumps(self._training_summary)]),
        )

    @classmethod
    def load(
        cls, path: Path | str, config: CountermeasureConfig | None = None
    ) -> SpoofingCountermeasure:
        with np.load(Path(path), allow_pickle=False) as archive:
            version = int(archive["format_version"][0])
            if version != cls._FORMAT_VERSION:
                raise InvalidEvidenceError(
                    "countermeasure archive format version is not supported",
                    found=version,
                    supported=cls._FORMAT_VERSION,
                )
            instance = cls(
                config=config or CountermeasureConfig(),
                bona_fide=GaussianMixture(
                    weights=archive["bona_fide_weights"],
                    means=archive["bona_fide_means"],
                    variances=archive["bona_fide_variances"],
                ),
                spoofed=GaussianMixture(
                    weights=archive["spoofed_weights"],
                    means=archive["spoofed_means"],
                    variances=archive["spoofed_variances"],
                ),
                out_of_domain_threshold=float(archive["out_of_domain_threshold"][0]),
                training_summary=json.loads(str(archive["training_summary"][0])),
                calibration=(
                    float(archive["calibration"][0]),
                    float(archive["calibration"][1]),
                ),
                typical_frame_std=float(archive["typical_frame_std"][0]),
            )
            stored = str(archive["detector_id"][0])

        if instance.detector_id != stored:
            raise InvalidEvidenceError(
                "countermeasure parameters do not hash to the stored identifier",
                stored=stored,
                computed=instance.detector_id,
            )
        return instance


def _fit_calibration_by_cross_validation(
    bona_fide_features: Sequence[NDArray[np.float64]],
    spoofed_features: Sequence[NDArray[np.float64]],
    gmm_config: GmmConfig,
    n_folds: int = 3,
) -> tuple[float, float]:
    """Fit an affine calibration of the detector score, by cross-validation.

    Why this is necessary rather than a refinement: the raw score is the mean
    per-frame log-ratio of two Gaussian mixtures, whose magnitude scales with
    the feature dimension and has no interpretation as a strength of evidence.
    The gate that consumes it expresses its operating point *as* a strength of
    evidence. Applying a threshold of "ten to one" to a number whose natural
    range is in the hundreds is a comparison between unrelated quantities, and
    the resulting verdicts are governed by the arbitrary scale rather than by
    the policy.

    Cross-validation rather than in-sample fitting: a calibration fitted on the
    scores the models were trained on is fitted to their over-confidence, and it
    transfers that over-confidence to new recordings while appearing well
    calibrated on the training set.

    Falls back to the identity when there is too little data to cross-validate,
    which leaves the detector visibly uncalibrated rather than silently
    miscalibrated.
    """

    n_bona_fide = len(bona_fide_features)
    n_spoofed = len(spoofed_features)
    if min(n_bona_fide, n_spoofed) < n_folds:
        return (1.0, 0.0)

    scores: list[float] = []
    labels: list[int] = []

    for fold in range(n_folds):
        bona_fide_test = [
            features
            for index, features in enumerate(bona_fide_features)
            if index % n_folds == fold
        ]
        bona_fide_train = [
            features
            for index, features in enumerate(bona_fide_features)
            if index % n_folds != fold
        ]
        spoofed_test = [
            features
            for index, features in enumerate(spoofed_features)
            if index % n_folds == fold
        ]
        spoofed_train = [
            features
            for index, features in enumerate(spoofed_features)
            if index % n_folds != fold
        ]
        if not (bona_fide_train and spoofed_train and bona_fide_test and spoofed_test):
            continue

        try:
            fold_bona_fide = train_ubm(bona_fide_train, gmm_config)
            fold_spoofed = train_ubm(spoofed_train, gmm_config)
        except (InsufficientDataError, InvalidEvidenceError):
            continue

        for features in bona_fide_test:
            scores.append(
                float(
                    np.mean(
                        fold_bona_fide.log_likelihood(features)
                        - fold_spoofed.log_likelihood(features)
                    )
                )
            )
            labels.append(1)
        for features in spoofed_test:
            scores.append(
                float(
                    np.mean(
                        fold_bona_fide.log_likelihood(features)
                        - fold_spoofed.log_likelihood(features)
                    )
                )
            )
            labels.append(0)

    score_array = np.array(scores)
    label_array = np.array(labels)
    if (
        score_array.size == 0
        or int(np.count_nonzero(label_array == 1)) < 2
        or int(np.count_nonzero(label_array == 0)) < 2
        or float(np.std(score_array)) <= 0.0
    ):
        return (1.0, 0.0)

    # The general-purpose calibrator enforces a minimum trial count that a
    # countermeasure development set will not usually meet, so the two-parameter
    # fit is done directly here. The objective is the same C_llr; what is
    # relaxed is only the guard against fitting a calibration to too few points,
    # and the consequence — a calibration that describes this development set
    # more closely than the population — is recorded in the training summary.
    from scipy.optimize import minimize

    from viflap.analysis.calibration.calibrators import _cllr_objective_and_gradient

    centre = float(np.mean(score_array))
    scale = float(np.std(score_array))
    same = (score_array[label_array == 1] - centre) / scale
    different = (score_array[label_array == 0] - centre) / scale

    result = minimize(
        _cllr_objective_and_gradient,
        x0=np.array([1.0, 0.0]),
        args=(same, different),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None), (None, None)],
    )
    slope_z, intercept_z = float(result.x[0]), float(result.x[1])
    return (slope_z / scale, intercept_z - slope_z * centre / scale)
