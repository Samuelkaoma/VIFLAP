"""Adapter for a pre-trained speaker embedding network.

The science of this lives in :mod:`viflap.analysis.speaker.neural` — the sample
rate a VoxCeleb2 checkpoint is defined against, and what feeding it anything
else costs, both measured. What lives *here* is everything that made it
unwelcome there: loading a checkpoint from disk, and depending on a deep
learning framework.

That split is enforced rather than preferred. ``tests/architecture`` refuses
third-party imports the analysis layer has no business holding and refuses file
I/O in the analysis layer at all, and it caught this module's first draft
sitting in the wrong place on both counts. Infrastructure is where adapters to
external artefacts belong.

Why a borrowed extractor at all
-------------------------------
§12 of the results document benchmarks this project's i-vector system against
E3FS3 and finds the gap is not architecture: E3FS3's ResNet saw roughly 6,000
VoxCeleb2 speakers where this system's extractor and back-end both see 306. Six
thousand speakers cannot be collected here — §8 established that twelve Zambian
speakers have recoverable identity — so the reachable move is the one E3FS3
itself made, and take a pre-trained extractor while retraining only the
back-end. This class is that extractor.

It is an alternative, not a replacement. The i-vector stack remains the system
of record until a paired comparison on the same held-out speakers says
otherwise, because §7 is a standing reminder that a plausible improvement can
be significantly worse in all six cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.dsp.framing import FrameConfig
from viflap.analysis.dsp.vad import VadConfig, detect_voice_activity
from viflap.analysis.speaker.neural import (
    NEURAL_EXTRACTOR_RATE,
    prepare_for_extractor,
)
from viflap.domain.errors import ConfigurationError, InsufficientDataError

__all__ = ["NeuralEmbeddingConfig", "NeuralEmbeddingExtractor"]


@dataclass(frozen=True, slots=True)
class NeuralEmbeddingConfig:
    """Where the checkpoint comes from and how it is run."""

    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    """A HuggingFace model id. It reaches every embedding's provenance, because
    two embeddings from different checkpoints are not comparable and nothing
    downstream can tell them apart by looking at the vectors."""

    savedir: Path = Path("models/pretrained/spkrec-ecapa-voxceleb")
    device: str = "cpu"

    min_speech_seconds: float = 3.0
    """Minimum **net speech**, measured after voice activity detection.

    Kept from the i-vector front-end, and kept for a *different* reason. There
    the justification is that an i-vector from under three seconds is dominated
    by its prior, and ``posterior_shrinkage`` measures exactly that. A neural
    embedding has no prior and no posterior, so that argument does not transfer
    and must not be quoted as though it did. What remains is empirical:
    published verification systems degrade sharply below a few seconds of net
    speech, and §5 measures the same shape here. The gate is retained mainly so
    that both extractors refuse the same recordings and the comparison between
    them stays paired. Its basis is weaker in this module and is stated rather
    than inherited.

    > **This measured the wrong quantity until now, and the name was the giveaway.**
    > It compared ``signal.size / sample_rate`` — wall-clock length — against a
    > threshold called ``min_speech_seconds``, so a five-second clip that was
    > four seconds of silence passed a gate named for speech. The i-vector
    > front-end has always run VAD first and tested
    > ``activity.speech_duration_seconds``. Same parameter name, same default,
    > different quantity.
    >
    > The consequence was measured before the cause was found. §22's five-second
    > cells could not be paired on identical trial sets because the i-vector
    > front-end refused 12 and 73 recordings there and this one refused none —
    > 6,126 and 35,050 trials that had to be dropped from the pairing. The
    > docstring's own stated purpose, that both extractors refuse the same
    > recordings, was not being served by the code under it."""

    vad: VadConfig = VadConfig()
    """The same detector the i-vector front-end uses, so that "three seconds of
    speech" means one thing across the two systems rather than two."""

    n_threads: int = 8


def assert_sufficient_speech(
    signal: NDArray[np.float64], sample_rate: int, config: NeuralEmbeddingConfig
) -> float:
    """Refuse a recording carrying too little **speech**, and return how much.

    A free function rather than a method because the gate is the part worth
    testing and the extractor around it costs an 89 MB checkpoint to construct.
    A test that has to load a network to find out whether a threshold fires is a
    test nobody runs, and this gate went unexamined for exactly that reason
    while it measured the wrong quantity.

    ``detect_voice_activity`` raises ``InsufficientDataError`` itself when there
    is too little signal to frame at all. That is the same refusal by a
    different route and is deliberately not caught: converting it to a pass
    would admit a recording shorter than one analysis window.
    """
    frame_config = FrameConfig().with_sample_rate(sample_rate)
    activity = detect_voice_activity(signal, frame_config, config.vad)
    if activity.speech_duration_seconds < config.min_speech_seconds:
        raise InsufficientDataError(
            "insufficient net speech for a reliable neural embedding",
            speech_seconds=round(activity.speech_duration_seconds, 2),
            seconds=round(signal.size / float(sample_rate), 2),
            required_seconds=config.min_speech_seconds,
        )
    return float(activity.speech_duration_seconds)


class NeuralEmbeddingExtractor:
    """Wraps a pre-trained speaker embedding network.

    Torch and SpeechBrain are imported inside ``__init__`` rather than at module
    scope. They are an optional extra — ``pip install 'viflap[neural]'`` — and a
    deployment that only ever scores through the i-vector stack should not carry
    a deep learning framework it never calls.
    """

    def __init__(self, config: NeuralEmbeddingConfig | None = None) -> None:
        self._config = config or NeuralEmbeddingConfig()
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigurationError(
                "the neural extractor needs optional dependencies; install them "
                "with pip install 'viflap[neural]'",
                missing=str(exc),
            ) from exc

        self._torch = torch
        torch.set_num_threads(self._config.n_threads)

        # COPY rather than the default SYMLINK. Creating a symlink on Windows
        # needs a privilege an ordinary account does not hold, so the default
        # downloads every file successfully into the HuggingFace cache and then
        # fails placing them, surfacing as WinError 1314. That reads like a
        # download failure and is not one; the distinction cost a run.
        self._classifier = EncoderClassifier.from_hparams(
            source=self._config.source,
            savedir=str(self._config.savedir),
            run_opts={"device": self._config.device},
            local_strategy=LocalStrategy.COPY,
        )

    @property
    def config(self) -> NeuralEmbeddingConfig:
        return self._config

    @property
    def extractor_id(self) -> str:
        """Identifies which network produced an embedding."""
        return f"neural:{self._config.source}@{NEURAL_EXTRACTOR_RATE}"

    def embed(self, signal: NDArray[np.float64], sample_rate: int) -> NDArray[np.float64]:
        """One recording to one embedding.

        The signal is put into the extractor's own rate first, whatever it
        arrives at. That is not a convenience: at 8 kHz this checkpoint's
        different-source similarity rises from 0.386 to 0.621 and it stops
        separating speakers.
        """
        signal = np.asarray(signal, dtype=np.float64).ravel()
        assert_sufficient_speech(signal, sample_rate, self._config)

        prepared = prepare_for_extractor(signal, sample_rate)
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(prepared, dtype=np.float32)
        ).unsqueeze(0)
        with self._torch.no_grad():
            vector = self._classifier.encode_batch(tensor).squeeze().cpu().numpy()
        return np.asarray(vector, dtype=np.float64).ravel()

    def embed_many(
        self, signals: list[tuple[NDArray[np.float64], int]]
    ) -> NDArray[np.float64]:
        """Embed a batch, one recording at a time.

        Not vectorised across the batch on purpose. Recordings differ in length,
        so a batched call needs padding, and padding changes the statistics the
        network pools over — a short recording padded to a long one's length is
        not the same input as the short recording. The cost is modest: about
        3.4 seconds per 30-second recording on eight cores, roughly nine times
        faster than real time.
        """
        return np.stack([self.embed(signal, rate) for signal, rate in signals])
