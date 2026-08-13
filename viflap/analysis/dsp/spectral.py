"""Filterbanks and cepstral features.

Two cepstral families are implemented, and the difference between them is not
cosmetic.

**MFCC** (mel-spaced filterbank) compresses the upper spectrum in proportion to
auditory resolution. That is the right prior for speaker and speech recognition,
where the information is concentrated where the ear is sensitive.

**LFCC** (linearly spaced filterbank) does not. It preserves uniform resolution
across the whole band, including the upper region where synthesis and voice
conversion artefacts concentrate — vocoder phase discontinuities, harmonic
structure that stops abruptly at the synthesiser's cutoff, unnaturally regular
spectral fine structure. This is why the ASVspoof baselines use LFCC for the
countermeasure and not MFCC: a mel filterbank averages the evidence of synthesis
away in the act of being auditorily plausible.

Both are built on a shared triangular-filterbank construction so that the only
difference between them is the frequency warping, which makes their comparison
in an ablation meaningful.

Band limiting
-------------
The default lower and upper edges are 300 Hz and 3400 Hz, the AMR narrowband
passband. Placing filters outside it would produce channels whose energy is
determined by the codec's stopband rather than by the speaker, and those
channels are pure nuisance variance: they vary with the network and not with the
person, so a back-end trained on them learns the channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct

from viflap.analysis.dsp.framing import FrameConfig, FrameGeometry, frame_signal
from viflap.domain.errors import InvalidEvidenceError

__all__ = [
    "CepstralConfig",
    "FilterbankScale",
    "add_deltas",
    "cepstral_mean_variance_normalise",
    "compute_cepstra",
    "filterbank_matrix",
    "hz_to_mel",
    "mel_to_hz",
    "power_spectrum",
    "sliding_cmvn",
]

#: Telephony passband, 3GPP TS 26.090.
NARROWBAND_LOW_HZ: Final[float] = 300.0
NARROWBAND_HIGH_HZ: Final[float] = 3400.0


def hz_to_mel(hz: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """HTK mel scale: ``2595 log10(1 + f / 700)``.

    The HTK formulation rather than Slaney's, because the pretrained speaker
    embedding extractors this system is benchmarked against are trained with it,
    and a mismatched warping is a silent domain shift.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (np.power(10.0, np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


class FilterbankScale(Enum):
    """Frequency warping applied to the filterbank."""

    MEL = "mel"
    LINEAR = "linear"


def filterbank_matrix(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    low_hz: float = NARROWBAND_LOW_HZ,
    high_hz: float = NARROWBAND_HIGH_HZ,
    scale: FilterbankScale = FilterbankScale.MEL,
    *,
    area_normalise: bool = True,
) -> NDArray[np.float64]:
    """Build a triangular filterbank as a ``(n_filters, n_fft // 2 + 1)`` matrix.

    Filters are laid out with 50% overlap: the peak of filter ``i`` is the lower
    edge of filter ``i + 1`` and the upper edge of filter ``i - 1``, so every
    frequency in the band contributes to exactly two filters with weights
    summing to one.

    Parameters
    ----------
    area_normalise:
        Divide each filter by its bandwidth so that all filters have unit area.
        Without it, wide high-frequency mel filters accumulate more energy than
        narrow low-frequency ones purely because they are wider, and the
        resulting log-energies carry a systematic tilt that the DCT then spreads
        across every coefficient. Disable only to reproduce a system that
        omitted it.
    """
    if n_filters < 2:
        raise InvalidEvidenceError(
            "a filterbank needs at least two filters", n_filters=n_filters
        )
    nyquist = sample_rate / 2.0
    if not 0.0 <= low_hz < high_hz <= nyquist:
        raise InvalidEvidenceError(
            "filterbank band must satisfy 0 <= low < high <= Nyquist",
            low_hz=low_hz,
            high_hz=high_hz,
            nyquist=nyquist,
        )

    if scale is FilterbankScale.MEL:
        edges_warped = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_filters + 2)
        edges_hz = np.asarray(mel_to_hz(edges_warped), dtype=np.float64)
    else:
        edges_hz = np.linspace(low_hz, high_hz, n_filters + 2)

    n_bins = n_fft // 2 + 1
    bin_hz = np.linspace(0.0, nyquist, n_bins)
    bank = np.zeros((n_filters, n_bins), dtype=np.float64)

    for index in range(n_filters):
        left, centre, right = edges_hz[index], edges_hz[index + 1], edges_hz[index + 2]
        # Guard against degenerate triangles, which occur when the requested
        # filter count is too high for the band and adjacent edges collapse into
        # the same FFT bin.
        if not (left < centre < right):
            raise InvalidEvidenceError(
                "filterbank triangles collapsed: too many filters for this band "
                "and FFT size",
                n_filters=n_filters,
                n_fft=n_fft,
                band=(low_hz, high_hz),
            )
        rising = (bin_hz - left) / (centre - left)
        falling = (right - bin_hz) / (right - centre)
        bank[index] = np.clip(np.minimum(rising, falling), 0.0, None)

        if area_normalise:
            width = right - left
            if width > 0.0:
                bank[index] *= 2.0 / width

    if not np.any(bank.sum(axis=1) > 0.0):
        raise InvalidEvidenceError(
            "every filter is empty; the FFT resolution is too coarse for this filterbank",
            n_fft=n_fft,
            n_filters=n_filters,
        )
    return bank


def power_spectrum(frames: NDArray[np.float64], n_fft: int) -> NDArray[np.float64]:
    """One-sided power spectrum of each frame.

    Uses ``rfft``: the input is real, so the negative-frequency half is the
    conjugate mirror of the positive half and computing it wastes half the work
    and half the memory.
    """
    spectrum = np.fft.rfft(frames, n=n_fft, axis=-1)
    return np.abs(spectrum) ** 2


@dataclass(frozen=True, slots=True)
class CepstralConfig:
    """Cepstral front-end parameters."""

    n_filters: int = 24
    n_cepstra: int = 20
    scale: FilterbankScale = FilterbankScale.MEL
    low_hz: float = NARROWBAND_LOW_HZ
    high_hz: float = NARROWBAND_HIGH_HZ
    include_energy: bool = True
    """Replace the zeroth cepstral coefficient with the log frame energy. C0 is
    the mean of the log filterbank, which is close to but not the same as log
    energy; using energy directly keeps the coefficient interpretable and makes
    it removable by normalisation without disturbing the rest."""

    lifter: float = 22.0
    """Sinusoidal liftering coefficient. Higher cepstral coefficients have small
    magnitude and would otherwise be ignored by any subsequent model that is
    sensitive to scale, including the diagonal-covariance Gaussians used later.
    Set to zero to disable."""

    log_floor: float = 1e-10
    """Floor applied before the logarithm. Prevents ``-inf`` on filterbank
    channels that fall entirely inside a codec stopband, which happens routinely
    at low AMR rates."""

    def __post_init__(self) -> None:
        if self.n_cepstra > self.n_filters:
            raise InvalidEvidenceError(
                "cannot extract more cepstral coefficients than filterbank "
                "channels; the additional coefficients would be identically zero",
                n_cepstra=self.n_cepstra,
                n_filters=self.n_filters,
            )
        if self.n_cepstra < 1:
            raise InvalidEvidenceError("n_cepstra must be positive")
        if self.log_floor <= 0.0:
            raise InvalidEvidenceError("log floor must be positive")

    @classmethod
    def for_speaker_recognition(cls) -> CepstralConfig:
        """MFCC configuration for the speaker back-end."""
        return cls(n_filters=24, n_cepstra=20, scale=FilterbankScale.MEL)

    @classmethod
    def for_spoofing_countermeasure(cls) -> CepstralConfig:
        """LFCC configuration for the synthetic-speech countermeasure.

        Linear spacing with more filters than the MFCC front-end, and the band
        extended to Nyquist. Synthesis artefacts sit at the top of the band; the
        countermeasure is the one component that must look there, and it is
        applied before the channel simulation in evaluation so that the
        artefacts have not already been filtered away.
        """
        return cls(
            n_filters=20,
            n_cepstra=20,
            scale=FilterbankScale.LINEAR,
            low_hz=0.0,
            high_hz=4000.0,
            include_energy=False,
            lifter=0.0,
        )


def compute_cepstra(
    signal: NDArray[np.float64],
    frame_config: FrameConfig,
    cepstral_config: CepstralConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], FrameGeometry]:
    """Compute cepstral coefficients for a signal.

    Returns an array of shape ``(n_frames, n_cepstra)`` and the frame geometry.

    The chain is: frame and window, power spectrum, triangular filterbank, log,
    DCT-II (orthonormal), optional energy substitution and liftering. That is
    the standard construction; it is written out here rather than delegated so
    that every choice in it is visible and testable, which matters because a
    front-end mismatch between enrolment and test is one of the more common
    silent causes of degraded forensic performance.
    """
    frames, geometry = frame_signal(signal, frame_config, rng=rng)

    spectra = power_spectrum(frames, frame_config.n_fft)
    bank = filterbank_matrix(
        n_filters=cepstral_config.n_filters,
        n_fft=frame_config.n_fft,
        sample_rate=frame_config.sample_rate,
        low_hz=cepstral_config.low_hz,
        high_hz=min(cepstral_config.high_hz, frame_config.sample_rate / 2.0),
        scale=cepstral_config.scale,
    )

    filterbank_energies = spectra @ bank.T
    log_energies = np.log(np.maximum(filterbank_energies, cepstral_config.log_floor))

    # DCT-II with orthonormal scaling, so the transform is its own inverse up to
    # transposition and the coefficients are on a comparable scale.
    cepstra = dct(log_energies, type=2, axis=-1, norm="ortho")[
        :, : cepstral_config.n_cepstra
    ]

    if cepstral_config.include_energy:
        frame_energy = np.sum(frames**2, axis=1)
        cepstra[:, 0] = np.log(np.maximum(frame_energy, cepstral_config.log_floor))

    if cepstral_config.lifter > 0.0:
        index = np.arange(cepstral_config.n_cepstra)
        weights = 1.0 + 0.5 * cepstral_config.lifter * np.sin(
            np.pi * index / cepstral_config.lifter
        )
        cepstra = cepstra * weights

    return cepstra, geometry


def add_deltas(
    features: NDArray[np.float64], window: int = 2, order: int = 2
) -> NDArray[np.float64]:
    """Append delta and delta-delta coefficients.

    Uses the standard regression estimator

    .. code-block:: text

        d[t] = sum_{n=1..N} n (c[t+n] - c[t-n]) / (2 sum_{n=1..N} n^2)

    which is a least-squares slope over a window of ``2N + 1`` frames rather
    than a first difference. The distinction matters at 8 kHz with 10 ms shifts:
    a first difference is dominated by frame-to-frame estimation noise, whereas
    the regression slope averages it down.

    Edges are handled by replicating the first and last frames. Zero-padding
    would assert that the signal had zero energy immediately outside the
    analysed region, producing a large spurious derivative at both ends.
    """
    if window < 1:
        raise InvalidEvidenceError("delta window must be at least one frame")
    if order < 0:
        raise InvalidEvidenceError("delta order cannot be negative")

    features = np.asarray(features, dtype=np.float64)
    outputs = [features]
    current = features
    denominator = 2.0 * sum(n * n for n in range(1, window + 1))

    for _ in range(order):
        padded = np.pad(current, ((window, window), (0, 0)), mode="edge")
        delta = np.zeros_like(current)
        for n in range(1, window + 1):
            ahead = padded[window + n : window + n + current.shape[0]]
            behind = padded[window - n : window - n + current.shape[0]]
            delta += n * (ahead - behind)
        delta /= denominator
        outputs.append(delta)
        current = delta

    return np.concatenate(outputs, axis=1)


def cepstral_mean_variance_normalise(
    features: NDArray[np.float64], *, normalise_variance: bool = True
) -> NDArray[np.float64]:
    """Normalise features to zero mean and, optionally, unit variance.

    Cepstral mean subtraction removes any time-invariant linear filtering: a
    convolution in the time domain is an addition in the log-spectral domain and
    therefore a constant offset in the cepstral domain. Handset transducer
    response, the codec's average spectral shaping and the line response are all
    of that form, so subtracting the utterance mean removes the bulk of the
    channel — which, in this application, is the dominant nuisance factor.

    Variance normalisation additionally equalises the dynamic range across
    recordings of differing level and noise.

    The cost is real and should be stated: mean subtraction over the whole
    utterance also removes any genuinely time-invariant *speaker* characteristic
    that happens to be spectrally flat. It is applied because the channel varies
    more across the recordings this system compares than the speaker's
    long-term average spectrum does, but that is an empirical judgement, and the
    ablation in the evaluation layer tests it rather than assuming it.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] == 0:
        return features
    centred = features - features.mean(axis=0, keepdims=True)
    if not normalise_variance:
        return centred
    deviation = centred.std(axis=0, keepdims=True)
    # A dimension with no variance carries no information; dividing by its
    # standard deviation would amplify numerical noise into a dominant feature.
    deviation = np.where(deviation < 1e-8, 1.0, deviation)
    return centred / deviation


def sliding_cmvn(
    features: NDArray[np.float64],
    window_frames: int = 300,
    *,
    normalise_variance: bool = False,
) -> NDArray[np.float64]:
    """Cepstral normalisation over a sliding window rather than the whole utterance.

    AMR rate adaptation changes the effective channel *within* a single call as
    radio conditions vary, so the utterance-level mean is an average over
    conditions rather than an estimate of one. A sliding window tracks the
    change, at the cost of a noisier estimate from fewer frames.

    Defaults to three seconds of frames, which is long enough for the estimate
    to be stable and short enough to follow rate adaptation.

    A non-positive ``window_frames`` selects utterance-level normalisation
    explicitly, at every length. That is not the same as passing a large number
    and relying on the window exceeding the frame count: it says so in the
    configuration rather than leaving the front-end's behaviour to be inferred
    from the duration of whatever was fed to it. The distinction matters because
    a fixed window silently changes character with duration — three seconds of
    window over a thirty-second recording is a local estimate, and over a
    five-second recording it is very nearly the utterance mean — so a duration
    sweep run at a fixed window varies the front-end as well as the duration.
    """
    features = np.asarray(features, dtype=np.float64)
    n_frames = features.shape[0]
    if n_frames == 0:
        return features
    if window_frames <= 0 or window_frames >= n_frames:
        return cepstral_mean_variance_normalise(
            features, normalise_variance=normalise_variance
        )

    half = window_frames // 2
    # Cumulative sums give an O(n) sliding mean; the naive per-frame slice is
    # O(n * window) and becomes the dominant cost on long calls.
    padded = np.pad(features, ((half, half), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    cumulative = np.vstack([np.zeros((1, features.shape[1])), cumulative])
    span = 2 * half + 1
    windowed_sum = cumulative[span : span + n_frames] - cumulative[:n_frames]
    means = windowed_sum / span
    centred = features - means

    if not normalise_variance:
        return centred

    padded_sq = np.pad(features**2, ((half, half), (0, 0)), mode="edge")
    cumulative_sq = np.cumsum(padded_sq, axis=0)
    cumulative_sq = np.vstack([np.zeros((1, features.shape[1])), cumulative_sq])
    windowed_sq = cumulative_sq[span : span + n_frames] - cumulative_sq[:n_frames]
    variance = np.maximum(windowed_sq / span - means**2, 0.0)
    deviation = np.sqrt(variance)
    deviation = np.where(deviation < 1e-8, 1.0, deviation)
    return centred / deviation
