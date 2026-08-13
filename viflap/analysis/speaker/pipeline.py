"""The acoustic comparison system, assembled and serialisable.

Composes the front-end, UBM, total variability matrix, session compensation and
PLDA back-end into one object that can be trained, saved, loaded, and asked for
the score of a pair of recordings.

Two properties are load-bearing for a forensic setting.

**Every artefact carries an identity.** A score is meaningless without knowing
which system produced it, and two scores from different systems are not
comparable. :attr:`SpeakerComparisonSystem.model_id` is derived from the
parameters themselves, so a model cannot be relabelled without changing what it
computes, and a result recorded with a model id can be traced to the exact
system that produced it.

**The output is a score, not evidence.** :meth:`score` returns an uncalibrated
PLDA log-likelihood ratio. It is not a likelihood ratio the system may report,
because the model's Gaussian assumptions are approximations and its training
population is not the relevant population. Converting a score into evidence is
the calibration layer's job, and the type signature keeps that boundary visible:
this module deals in floats, and only the calibration layer produces a
:class:`~viflap.domain.values.LogLikelihoodRatio`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.dsp.framing import FrameConfig
from viflap.analysis.dsp.spectral import (
    CepstralConfig,
    add_deltas,
    compute_cepstra,
    sliding_cmvn,
)
from viflap.analysis.dsp.vad import VadConfig, detect_voice_activity
from viflap.analysis.speaker.gmm import (
    BaumWelchStatistics,
    GaussianMixture,
    GmmConfig,
    train_ubm,
)
from viflap.analysis.speaker.ivector import (
    IVectorExtractor,
    IVectorPosterior,
    TotalVariabilityConfig,
    train_total_variability,
)
from viflap.analysis.speaker.plda import PldaConfig, PldaModel, train_plda
from viflap.analysis.speaker.transforms import IVectorTransform, fit_transform_chain
from viflap.domain.errors import (
    InsufficientDataError,
    InvalidEvidenceError,
)

__all__ = [
    "AcousticEmbedding",
    "FrontEndConfig",
    "SpeakerComparisonSystem",
    "SpeakerSystemConfig",
    "TrainingRecording",
]


@dataclass(frozen=True, slots=True)
class FrontEndConfig:
    """Feature extraction settings, fixed for the lifetime of a trained system.

    Bundled and serialised with the model because a front-end mismatch between
    training and scoring is silent: features of the right shape are produced,
    the model returns a number, and the number is wrong. Storing the
    configuration with the model makes the mismatch impossible rather than
    merely unlikely.
    """

    frame: FrameConfig = FrameConfig()
    cepstral: CepstralConfig = CepstralConfig.for_speaker_recognition()
    vad: VadConfig = VadConfig()
    delta_order: int = 2
    delta_window: int = 2
    sliding_cmvn_frames: int = 300
    """Window for cepstral mean and variance normalisation, in frames; zero or
    below normalises over the whole utterance.

    Three hundred frames is three seconds, chosen to track AMR rate adaptation
    within a call. It is a *fixed* window, so its character changes with the
    length of what it is given: over a 30 s recording (~2,300 speech frames) it
    is a local estimate spanning some 8% of the utterance, and over a 5 s
    recording (~350 frames) it spans some 85% and is very nearly the utterance
    mean. A duration sweep at a fixed window therefore varies the front-end
    alongside the duration, and the two effects cannot be separated after the
    fact. Setting this to zero, or to a window short enough to stay local at
    every duration, makes the front-end duration-invariant and is what the
    control arms in the results document do."""

    min_speech_seconds: float = 3.0
    """Minimum net speech below which no embedding is produced. Three seconds is
    already marginal for narrowband telephony; below it the i-vector is
    dominated by its prior and the resulting score reflects the model more than
    the recording."""

    def extract_features(
        self, signal: NDArray[np.float64], sample_rate: int
    ) -> tuple[NDArray[np.float64], dict[str, float]]:
        """Signal to speech-only, normalised feature matrix, with diagnostics.

        Voice activity detection is applied *before* normalisation. Normalising
        over all frames including silence estimates the mean partly from comfort
        noise, which is a property of the network rather than of the speaker,
        and then subtracts that from the speech.
        """
        frame_config = (
            self.frame
            if self.frame.sample_rate == sample_rate
            else self.frame.with_sample_rate(sample_rate)
        )

        activity = detect_voice_activity(signal, frame_config, self.vad)
        if activity.speech_duration_seconds < self.min_speech_seconds:
            raise InsufficientDataError(
                "insufficient net speech for a reliable acoustic comparison",
                speech_seconds=round(activity.speech_duration_seconds, 2),
                required_seconds=self.min_speech_seconds,
            )

        cepstra, _ = compute_cepstra(signal, frame_config, self.cepstral)
        speech_cepstra = cepstra[activity.is_speech[: cepstra.shape[0]]]

        normalised = sliding_cmvn(
            speech_cepstra, self.sliding_cmvn_frames, normalise_variance=True
        )
        features = add_deltas(normalised, window=self.delta_window, order=self.delta_order)

        diagnostics = {
            "speech_seconds": activity.speech_duration_seconds,
            "speech_fraction": activity.speech_fraction,
            "total_seconds": activity.geometry.duration_seconds,
            "noise_floor_db": activity.noise_floor_db,
            "n_feature_frames": float(features.shape[0]),
        }
        return features, diagnostics

    @property
    def feature_dimension(self) -> int:
        return self.cepstral.n_cepstra * (1 + self.delta_order)


@dataclass(frozen=True, slots=True)
class SpeakerSystemConfig:
    """Everything needed to train a speaker comparison system."""

    front_end: FrontEndConfig = FrontEndConfig()
    ubm: GmmConfig = GmmConfig()
    total_variability: TotalVariabilityConfig = TotalVariabilityConfig()
    plda: PldaConfig = PldaConfig()
    lda_dimension: int | None = None

    ubm_frame_budget: int | None = None
    """Cap on the frames used to estimate the UBM, or ``None`` for all of them.

    EM holds an ``n_frames x n_components`` responsibility matrix. At 256
    components a million frames is two gigabytes, and the corpora this system is
    meant for produce several million — so the constraint binds in exactly the
    case that matters, and the failure is an allocation error part-way through a
    long run rather than anything informative.

    Capping costs very little. A UBM describes the distribution of speech in
    general; it converges on a broad sample and gains almost nothing from the
    tail. The subsample is taken evenly *across recordings* rather than as a
    random draw from the pooled frames, because a random draw over-represents
    long recordings and, through them, whichever speakers happen to be verbose.

    The cap applies to UBM estimation alone. Baum-Welch statistics, the total
    variability matrix and PLDA all still see every frame of every recording:
    those stages are where per-recording and per-speaker structure is learned,
    and subsampling them would discard the thing being estimated.
    """

    background_recording_budget: int | None = None
    """Cap on background recordings contributing statistics to the total
    variability matrix, or ``None`` for all of them.

    Only consulted when :meth:`SpeakerComparisonSystem.train` is given a
    ``background`` corpus. Baum-Welch statistics are small per recording but not
    free — at 256 components and 60 dimensions roughly 120 kB each — so a
    background pool of tens of thousands of recordings costs gigabytes to hold
    while the total variability matrix is estimated. The cap bounds that.

    Recordings are selected at even intervals across the pool rather than by
    taking the first *n*, so a corpus that arrives grouped by speaker, language
    or session does not contribute only its opening group.

    Must exceed :attr:`TotalVariabilityConfig.rank`, or the matrix it is used to
    estimate has more dimensions than observations and training refuses. The
    refusal is correct and the budget is the wrong place to economise: set it
    well above the rank, since a subspace fitted from barely more recordings
    than it has dimensions is determined by the few it happened to see.
    """

    @classmethod
    def compact(cls) -> SpeakerSystemConfig:
        """A small configuration for tests and development.

        Not a degraded version of the operational system: the same code paths
        with fewer components and a lower-rank subspace, so that behaviour
        verified here is behaviour of the system that ships.
        """
        return cls(
            ubm=GmmConfig(n_components=16, max_iterations=25),
            total_variability=TotalVariabilityConfig(rank=20, max_iterations=8),
            plda=PldaConfig(max_iterations=20, min_speakers=5),
            lda_dimension=10,
        )


@dataclass(frozen=True, slots=True)
class TrainingRecording:
    """One labelled recording for system training."""

    signal: NDArray[np.float64]
    sample_rate: int
    speaker_id: str
    recording_id: str


@dataclass(frozen=True, slots=True)
class AcousticEmbedding:
    """A recording reduced to a comparable representation, with its provenance."""

    vector: NDArray[np.float64]
    """Post-transform i-vector, ready for the PLDA back-end."""

    raw_ivector: NDArray[np.float64]
    posterior_shrinkage: float
    diagnostics: dict[str, float]
    model_id: str

    @property
    def is_well_determined(self) -> bool:
        """Whether the recording determined the embedding, or the prior did."""
        return self.posterior_shrinkage < 0.5


class SpeakerComparisonSystem:
    """A trained acoustic comparison system.

    Construct via :meth:`train`, or :meth:`load` a saved one. There is no
    constructor that produces an untrained instance capable of scoring: an
    untrained system that returns numbers is worse than one that refuses,
    because the numbers are indistinguishable from real ones.
    """

    _FORMAT_VERSION = 1

    def __init__(
        self,
        config: SpeakerSystemConfig,
        ubm: GaussianMixture,
        total_variability: NDArray[np.float64],
        transform: IVectorTransform,
        plda: PldaModel,
        training_summary: dict[str, float],
        training_speakers: frozenset[str] = frozenset(),
    ) -> None:
        self._config = config
        self._ubm = ubm
        self._extractor = IVectorExtractor(ubm, total_variability)
        self._transform = transform
        self._plda = plda
        self._training_summary = dict(training_summary)
        self._training_speakers = frozenset(training_speakers)
        self._model_id = self._compute_model_id()

    @property
    def training_speakers(self) -> frozenset[str]:
        """Which speakers trained the back-end, so leakage can be *checked*.

        The count alone is not enough. An evaluation is only meaningful on
        speakers this model has never seen, and the usual way that breaks is not
        a bad split — it is a correct split computed over the *wrong corpus*.
        A model trained on two pooled partitions, evaluated with the corpus
        argument left at its default, produces a split that is internally
        disjoint and passes every check while scoring speakers the model
        memorised. That has happened three times in this project's history, and
        counts cannot detect it because the counts look plausible.

        Empty for models trained before this was recorded. An empty set means
        the check is unavailable, not that it passed, and callers should say so
        rather than proceed quietly.
        """
        return self._training_speakers

    # -- Identity ---------------------------------------------------------

    def _compute_model_id(self) -> str:
        """Derive an identifier from the model's own parameters.

        A content hash, not a version string. Two systems with the same id
        compute the same function, and a system whose parameters changed cannot
        keep its id. This is what makes a recorded ``model_id`` on a historical
        result actually mean something a year later.
        """
        digest = hashlib.sha256()
        digest.update(str(self._FORMAT_VERSION).encode())
        for array in (
            self._ubm.weights,
            self._ubm.means,
            self._ubm.variances,
            self._extractor.matrix,
            self._transform.mean,
            self._transform.projection,
            self._plda.mean,
            self._plda.transform,
            self._plda.psi,
        ):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return f"ivec-plda-{digest.hexdigest()[:16]}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def config(self) -> SpeakerSystemConfig:
        return self._config

    @property
    def plda(self) -> PldaModel:
        return self._plda

    def describe(self) -> dict[str, float | str]:
        """Model properties an analyst needs in order to interpret a score."""
        return {
            "model_id": self._model_id,
            "ubm_components": float(self._ubm.n_components),
            "feature_dimension": float(self._ubm.n_dimensions),
            "ivector_rank": float(self._extractor.rank),
            "transform_dimension": float(self._transform.output_dimension),
            # Part of the front-end rather than the back-end, but reported here
            # because it changes what the model computes and two models that
            # differ only in it are not comparable. Zero means utterance-level.
            "cmvn_frames": float(self._config.front_end.sliding_cmvn_frames),
            **{f"plda_{key}": value for key, value in self._plda.diagnostics().items()},
            **self._training_summary,
        }

    # -- Training ---------------------------------------------------------

    @classmethod
    def train(
        cls,
        recordings: Sequence[TrainingRecording],
        config: SpeakerSystemConfig | None = None,
        *,
        background: Sequence[TrainingRecording] | None = None,
    ) -> SpeakerComparisonSystem:
        """Train the whole stack from labelled recordings.

        Stages, in order, each consuming the previous one's output:

        1. Front-end features for every recording.
        2. UBM by EM over pooled features.
        3. Baum-Welch statistics per recording against the UBM.
        4. Total variability matrix by EM over those statistics (unsupervised).
        5. i-vector per recording.
        6. Centring, LDA and WCCN from the labelled i-vectors.
        7. PLDA on the transformed i-vectors.

        Recordings whose front-end stage fails — too little speech, most often —
        are skipped with the reason recorded, rather than aborting the run. A
        corpus assembled from operational material always contains some
        unusable files, and discovering that at the end of a long training run
        is worse than skipping them and reporting how many.

        Parameters
        ----------
        recordings:
            Labelled material. Stages 5 to 7 use these, and their speaker count
            is what bounds the system: LDA can extract at most ``n_speakers - 1``
            discriminative directions no matter how much audio is supplied.
        background:
            Optional unlabelled corpus for stages 2 and 4, which are
            unsupervised and need audio rather than speaker labels. Any
            ``speaker_id`` on these recordings is ignored.

            Separating the two lets the acoustic front-end be estimated on
            material matched to the deployment population while the back-end
            borrows its speaker subspace from wherever enough speakers exist.
            The two requirements are genuinely different — stages 2 and 4 want
            hours of representative audio, stages 6 and 7 want many distinct
            speakers — and no rule says one corpus must satisfy both.

            The pool is traversed **twice**, once for UBM frames and once for
            Baum-Welch statistics, because the statistics cannot be computed
            until the UBM exists. Audio is not retained between passes, so a
            lazily-loading sequence keeps peak memory bounded at the cost of
            reading the audio twice. Passing ``None`` estimates every stage from
            ``recordings``, which is the single-corpus behaviour and reads the
            audio once.
        """
        config = config or SpeakerSystemConfig()
        if not recordings:
            raise InsufficientDataError("no recordings supplied for training")

        features: list[NDArray[np.float64]] = []
        labels: list[str] = []
        skipped: list[tuple[str, str]] = []

        for recording in recordings:
            try:
                extracted, _ = config.front_end.extract_features(
                    recording.signal, recording.sample_rate
                )
            except InsufficientDataError as exc:
                skipped.append((recording.recording_id, exc.message))
                continue
            features.append(extracted)
            labels.append(recording.speaker_id)

        if len(features) < config.plda.min_speakers:
            raise InsufficientDataError(
                "too few usable recordings survived feature extraction",
                usable=len(features),
                skipped=len(skipped),
                required=config.plda.min_speakers,
            )

        if background is None:
            ubm_frames = _ubm_training_frames(features, config.ubm_frame_budget)
            ubm = train_ubm(ubm_frames, config.ubm)
            n_frames_ubm = sum(int(matrix.shape[0]) for matrix in ubm_frames)
            statistics: list[BaumWelchStatistics] = [
                ubm.baum_welch(matrix) for matrix in features
            ]
            # One corpus, so the same statistics serve both the total
            # variability matrix and the i-vectors and are computed once.
            variability_statistics = statistics
            n_background = 0
        else:
            background_frames, n_frames_ubm = _background_ubm_frames(background, config)
            ubm = train_ubm(background_frames, config.ubm)
            # Released before the second pass: the frame sample is the largest
            # thing held here, and nothing after this point reads it.
            del background_frames
            variability_statistics = _background_statistics(background, ubm, config)
            n_background = len(variability_statistics)
            statistics = [ubm.baum_welch(matrix) for matrix in features]

        total_variability = train_total_variability(
            variability_statistics, ubm, config.total_variability
        )
        extractor = IVectorExtractor(ubm, total_variability)
        ivectors = np.stack([extractor.extract(item).mean for item in statistics])

        unique_labels = sorted(set(labels))
        label_index = {label: index for index, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_index[label] for label in labels])

        transform = fit_transform_chain(
            ivectors, numeric_labels, lda_dimension=config.lda_dimension
        )
        projected = transform.apply(ivectors)
        plda = train_plda(projected, numeric_labels, config.plda)

        summary = {
            "n_training_recordings": float(len(features)),
            "n_training_speakers": float(len(unique_labels)),
            "n_skipped_recordings": float(len(skipped)),
            "mean_frames_per_recording": float(
                np.mean([matrix.shape[0] for matrix in features])
            ),
            "n_frames_total": float(sum(matrix.shape[0] for matrix in features)),
            # Recorded because it is the one training quantity a reader cannot
            # recover from the others. Two models with identical component
            # counts, one estimated from every frame and one from a tenth of
            # them, are different models and this is what distinguishes them.
            "n_frames_ubm": float(n_frames_ubm),
            # Zero when a single corpus trained every stage. Non-zero records
            # that the front-end and the back-end saw different material, which
            # changes what the model is and is not recoverable from the rest.
            "n_background_recordings": float(n_background),
        }
        return cls(
            config,
            ubm,
            total_variability,
            transform,
            plda,
            summary,
            frozenset(unique_labels),
        )

    # -- Inference --------------------------------------------------------

    def embed(self, signal: NDArray[np.float64], sample_rate: int) -> AcousticEmbedding:
        """Reduce a recording to a comparable representation."""
        features, diagnostics = self._config.front_end.extract_features(signal, sample_rate)
        posterior: IVectorPosterior = self._extractor.extract_from_features(features)
        vector = self._transform.apply(posterior.mean)

        return AcousticEmbedding(
            vector=vector,
            raw_ivector=posterior.mean,
            posterior_shrinkage=posterior.shrinkage,
            diagnostics={
                **diagnostics,
                "ivector_shrinkage": posterior.shrinkage,
                "ubm_occupancy": posterior.occupancy,
            },
            model_id=self._model_id,
        )

    def score(self, first: AcousticEmbedding, second: AcousticEmbedding) -> float:
        """Uncalibrated PLDA log-likelihood ratio for a pair of embeddings.

        Raises
        ------
        InvalidEvidenceError
            If the embeddings came from different models. Comparing them would
            produce a number, and the number would be meaningless: the two
            vectors live in bases fitted to different data.
        """
        if first.model_id != second.model_id:
            raise InvalidEvidenceError(
                "embeddings were produced by different models and are not comparable",
                first_model=first.model_id,
                second_model=second.model_id,
            )
        if first.model_id != self._model_id:
            raise InvalidEvidenceError(
                "embeddings were produced by a different model than the one scoring them",
                embedding_model=first.model_id,
                scoring_model=self._model_id,
            )
        return self._plda.score(first.vector, second.vector)

    def score_against_gallery(
        self, probe: AcousticEmbedding, gallery: Sequence[AcousticEmbedding]
    ) -> NDArray[np.float64]:
        """Score one embedding against many, for a database search."""
        for candidate in gallery:
            if candidate.model_id != self._model_id:
                raise InvalidEvidenceError(
                    "gallery contains embeddings from a different model",
                    embedding_model=candidate.model_id,
                    scoring_model=self._model_id,
                )
        if not gallery:
            return np.zeros(0, dtype=np.float64)
        matrix = np.stack([candidate.vector for candidate in gallery])
        return self._plda.score_many(probe.vector, matrix)

    # -- Persistence ------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Write the model to a single compressed archive.

        A single file rather than a directory: a model split across files can be
        partially copied, and a system that loads five of six arrays and scores
        with a stale sixth produces plausible wrong answers.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array([self._FORMAT_VERSION]),
            model_id=np.array([self._model_id]),
            ubm_weights=self._ubm.weights,
            ubm_means=self._ubm.means,
            ubm_variances=self._ubm.variances,
            total_variability=self._extractor.matrix,
            transform_mean=self._transform.mean,
            transform_projection=self._transform.projection,
            transform_length_normalise=np.array([self._transform.length_normalise]),
            plda_mean=self._plda.mean,
            plda_transform=self._plda.transform,
            plda_psi=self._plda.psi,
            plda_speakers=np.array([self._plda.n_training_speakers]),
            plda_recordings=np.array([self._plda.n_training_recordings]),
            training_summary=np.array([json.dumps(self._training_summary)]),
            config=np.array([json.dumps(_config_to_dict(self._config))]),
            # A unicode array, not an object array. numpy will only unpickle an
            # object array, :meth:`load` reads with ``allow_pickle=False`` as it
            # must, and the two together made every archive that recorded its
            # training speakers unloadable — the leakage check this exists for
            # could not run on any model that had it.
            training_speakers=np.array(sorted(self._training_speakers), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: Path | str) -> SpeakerComparisonSystem:
        """Load a model, verifying that its parameters still hash to its id.

        The check catches silent corruption and, more usefully, catches a model
        whose arrays were edited after training. A model that no longer computes
        what its id says it computes cannot be allowed to score, because every
        result it produced would be traceable to an identifier that describes a
        different function.
        """
        with np.load(Path(path), allow_pickle=False) as archive:
            version = int(archive["format_version"][0])
            if version != cls._FORMAT_VERSION:
                raise InvalidEvidenceError(
                    "model archive format version is not supported",
                    found=version,
                    supported=cls._FORMAT_VERSION,
                )

            ubm = GaussianMixture(
                weights=archive["ubm_weights"],
                means=archive["ubm_means"],
                variances=archive["ubm_variances"],
            )
            transform = IVectorTransform(
                mean=archive["transform_mean"],
                projection=archive["transform_projection"],
                length_normalise=bool(archive["transform_length_normalise"][0]),
            )
            plda = PldaModel(
                mean=archive["plda_mean"],
                transform=archive["plda_transform"],
                psi=archive["plda_psi"],
                n_training_speakers=int(archive["plda_speakers"][0]),
                n_training_recordings=int(archive["plda_recordings"][0]),
            )
            system = cls(
                config=_config_from_dict(json.loads(str(archive["config"][0]))),
                ubm=ubm,
                total_variability=archive["total_variability"],
                transform=transform,
                plda=plda,
                training_summary=json.loads(str(archive["training_summary"][0])),
                # Absent from archives written before this was recorded. An
                # empty set disables the leakage check rather than passing it.
                training_speakers=frozenset(
                    str(name) for name in archive.get("training_speakers", [])
                ),
            )
            stored_id = str(archive["model_id"][0])

        if system.model_id != stored_id:
            raise InvalidEvidenceError(
                "model parameters do not hash to the stored identifier; the "
                "archive has been modified or corrupted since it was written",
                stored=stored_id,
                computed=system.model_id,
            )
        return system


def _ubm_training_frames(
    features: Sequence[NDArray[np.float64]], budget: int | None
) -> list[NDArray[np.float64]]:
    """Subsample frames for UBM estimation, evenly across recordings.

    Each recording contributes the same number of frames, taken at equal
    intervals along it rather than drawn at random. Two reasons for the even
    interval: it covers the whole recording, so the sample spans the phonetic
    material rather than clustering in whichever passage the draw favoured; and
    it is deterministic, so a UBM trained twice on the same corpus is the same
    UBM and its content-derived model id is stable.

    Recordings shorter than their allocation contribute everything they have.
    The result is therefore at or below the budget, never above it.
    """
    if budget is None or not features:
        return list(features)

    total = sum(int(matrix.shape[0]) for matrix in features)
    if total <= budget:
        return list(features)

    per_recording = max(1, budget // len(features))
    sampled: list[NDArray[np.float64]] = []
    for matrix in features:
        n_frames = int(matrix.shape[0])
        if n_frames <= per_recording:
            sampled.append(matrix)
            continue
        indices = np.linspace(0, n_frames - 1, per_recording).astype(np.int64)
        sampled.append(matrix[indices])
    return sampled


def _background_ubm_frames(
    background: Sequence[TrainingRecording], config: SpeakerSystemConfig
) -> tuple[list[NDArray[np.float64]], int]:
    """Stream a background corpus, keeping only the frames the UBM will see.

    :func:`_ubm_training_frames` cannot serve here. It needs every feature
    matrix in memory at once to decide whether the budget binds at all, and a
    background corpus is exactly the case where they do not fit — a hundred and
    sixty hours of audio is tens of gigabytes of features. The per-recording
    allocation is therefore fixed from the recording count up front and applied
    as each recording is read, so peak memory is the sample rather than the
    corpus.

    One consequence is worth stating: because the total is unknown in advance,
    the allocation is applied even when the pool would have fitted inside the
    budget, so a small background corpus is trimmed slightly where passing it as
    ``recordings`` would not have trimmed it. The sample stays at or below
    budget either way.

    Recordings too short for the front-end are skipped rather than aborting the
    pass, as everywhere else in training.
    """
    budget = config.ubm_frame_budget
    per_recording = None if budget is None else max(1, budget // max(1, len(background)))

    sampled: list[NDArray[np.float64]] = []
    total = 0
    for item in background:
        try:
            matrix, _ = config.front_end.extract_features(item.signal, item.sample_rate)
        except InsufficientDataError:
            continue
        n_frames = int(matrix.shape[0])
        if per_recording is not None and n_frames > per_recording:
            indices = np.linspace(0, n_frames - 1, per_recording).astype(np.int64)
            matrix = matrix[indices]
        sampled.append(matrix)
        total += int(matrix.shape[0])

    if not sampled:
        raise InsufficientDataError(
            "no background recording survived feature extraction",
            n_background=len(background),
        )
    return sampled, total


def _background_statistics(
    background: Sequence[TrainingRecording],
    ubm: GaussianMixture,
    config: SpeakerSystemConfig,
) -> list[BaumWelchStatistics]:
    """Second pass over the background corpus, for the total variability matrix.

    The audio is read again rather than carried over from the first pass. These
    statistics depend on the UBM, which does not exist while that pass is
    running, and holding the features until it does would defeat the point of
    streaming them.
    """
    statistics: list[BaumWelchStatistics] = []
    for index in _evenly_spaced(len(background), config.background_recording_budget):
        item = background[index]
        try:
            matrix, _ = config.front_end.extract_features(item.signal, item.sample_rate)
        except InsufficientDataError:
            continue
        statistics.append(ubm.baum_welch(matrix))

    if not statistics:
        raise InsufficientDataError(
            "no background recording survived feature extraction",
            n_background=len(background),
        )
    return statistics


def _evenly_spaced(count: int, budget: int | None) -> list[int]:
    """Indices of at most ``budget`` items, spread evenly across ``count``.

    Even spacing rather than the first ``budget``, because a corpus that arrives
    grouped by speaker, language or session would otherwise contribute only
    whichever group happens to come first.
    """
    if budget is None or count <= budget:
        return list(range(count))
    return sorted({int(index) for index in np.linspace(0, count - 1, budget)})


def _config_to_dict(config: SpeakerSystemConfig) -> dict[str, object]:
    """Serialise the parts of the configuration that affect scoring.

    Only what changes the computed function is stored. Training-time settings
    such as iteration counts and seeds have already had their effect and are
    recorded in the training summary rather than here, where they would suggest
    they still matter.
    """
    front_end = config.front_end
    return {
        "sample_rate": front_end.frame.sample_rate,
        "frame_length_ms": front_end.frame.frame_length_ms,
        "frame_shift_ms": front_end.frame.frame_shift_ms,
        "window": front_end.frame.window.value,
        "pre_emphasis": front_end.frame.pre_emphasis,
        "n_filters": front_end.cepstral.n_filters,
        "n_cepstra": front_end.cepstral.n_cepstra,
        "filterbank_scale": front_end.cepstral.scale.value,
        "low_hz": front_end.cepstral.low_hz,
        "high_hz": front_end.cepstral.high_hz,
        "lifter": front_end.cepstral.lifter,
        "delta_order": front_end.delta_order,
        "delta_window": front_end.delta_window,
        "sliding_cmvn_frames": front_end.sliding_cmvn_frames,
        "min_speech_seconds": front_end.min_speech_seconds,
    }


def _config_from_dict(payload: dict[str, object]) -> SpeakerSystemConfig:
    """Rebuild the scoring-relevant configuration from a saved model."""
    from viflap.analysis.dsp.framing import WindowType
    from viflap.analysis.dsp.spectral import FilterbankScale

    frame = FrameConfig(
        sample_rate=int(payload["sample_rate"]),  # type: ignore[arg-type]
        frame_length_ms=float(payload["frame_length_ms"]),  # type: ignore[arg-type]
        frame_shift_ms=float(payload["frame_shift_ms"]),  # type: ignore[arg-type]
        window=WindowType(payload["window"]),
        pre_emphasis=float(payload["pre_emphasis"]),  # type: ignore[arg-type]
    )
    cepstral = CepstralConfig(
        n_filters=int(payload["n_filters"]),  # type: ignore[arg-type]
        n_cepstra=int(payload["n_cepstra"]),  # type: ignore[arg-type]
        scale=FilterbankScale(payload["filterbank_scale"]),
        low_hz=float(payload["low_hz"]),  # type: ignore[arg-type]
        high_hz=float(payload["high_hz"]),  # type: ignore[arg-type]
        lifter=float(payload["lifter"]),  # type: ignore[arg-type]
    )
    front_end = FrontEndConfig(
        frame=frame,
        cepstral=cepstral,
        delta_order=int(payload["delta_order"]),  # type: ignore[arg-type]
        delta_window=int(payload["delta_window"]),  # type: ignore[arg-type]
        sliding_cmvn_frames=int(payload["sliding_cmvn_frames"]),  # type: ignore[arg-type]
        min_speech_seconds=float(payload["min_speech_seconds"]),  # type: ignore[arg-type]
    )
    return SpeakerSystemConfig(front_end=front_end)
