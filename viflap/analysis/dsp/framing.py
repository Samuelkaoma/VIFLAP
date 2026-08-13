"""Framing, windowing and pre-emphasis.

Everything downstream — filterbanks, LPC, pitch, voice quality — consumes frames
produced here, so the framing convention is defined once and shared. Two
consequences worth stating:

- Frame timing is exact and recoverable. :meth:`FrameGeometry.frame_times`
  returns the centre time of every frame, which is what within-call temporal
  analysis (hypothesis H3, whether disguise decays over a call) needs in order
  to segment by position without re-deriving the arithmetic.
- Pre-emphasis is applied *before* framing, not per frame. Applying it per frame
  discards the sample preceding each frame boundary and introduces a
  discontinuity at every frame start, which shows up as spurious high-frequency
  energy in exactly the band the narrowband channel has already made scarce.

Defaults target 8 kHz narrowband telephony rather than 16 kHz studio speech,
because that is the operating condition, not a degraded special case of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError

__all__ = [
    "FrameConfig",
    "FrameGeometry",
    "WindowType",
    "frame_signal",
    "make_window",
    "pre_emphasise",
]

#: Telephony sampling rate. Present as a named constant because a great deal of
#: the design follows from it rather than from a general-purpose default.
NARROWBAND_RATE: Final[int] = 8000


class WindowType(Enum):
    """Analysis window.

    ``HAMMING`` is the default for cepstral analysis: its first sidelobe is
    roughly 43 dB down, which is sufficient to keep formant peaks from smearing
    into one another, and its non-zero endpoints matter less than that once the
    signal is pre-emphasised.

    ``POVEY`` is Hamming raised to the power 0.85 and tapered to zero at the
    edges. It is the Kaldi default and is included because published i-vector
    and x-vector systems are trained with it; reproducing their front-end
    exactly removes one source of unexplained difference when comparing against
    the literature.
    """

    HAMMING = "hamming"
    HANN = "hann"
    POVEY = "povey"
    RECTANGULAR = "rectangular"


def make_window(window: WindowType, length: int) -> NDArray[np.float64]:
    """Return a window of ``length`` samples.

    Periodic rather than symmetric definitions are used, which is the correct
    convention for spectral analysis: a symmetric window of length ``N`` has
    period ``N - 1`` and introduces a small bias in the estimated spectrum.
    """
    if length < 1:
        raise InvalidEvidenceError("window length must be positive", length=length)
    if window is WindowType.RECTANGULAR:
        return np.ones(length, dtype=np.float64)

    n = np.arange(length, dtype=np.float64)
    if window is WindowType.HAMMING:
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / length)
    if window is WindowType.HANN:
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)
    if window is WindowType.POVEY:
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)
        return np.power(hann, 0.85)
    raise InvalidEvidenceError("unknown window type", window=window)


@dataclass(frozen=True, slots=True)
class FrameConfig:
    """Framing parameters.

    Durations are given in milliseconds rather than samples so that a
    configuration is meaningful independently of sampling rate, and so that a
    corpus resampled from 16 kHz to 8 kHz is analysed with the same *physical*
    window rather than one half the duration.
    """

    sample_rate: int = NARROWBAND_RATE
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    window: WindowType = WindowType.HAMMING
    pre_emphasis: float = 0.97
    """First-order high-pass coefficient. Compensates the roughly -6 dB/octave
    spectral tilt of voiced speech, so that the higher formants are not
    swamped by the first when the spectral envelope is estimated. Set to zero to
    disable, which is appropriate when the glottal source itself is the object
    of analysis rather than the vocal tract filter."""

    dither: float = 1e-5
    """Amplitude of uniform noise added before analysis. Guards against
    ``log(0)`` on digitally silent frames — common in telephony, where comfort
    noise generation and packet loss concealment both produce exact zeros. The
    magnitude is far below the quantisation floor of the recording."""

    remove_dc: bool = True
    """Subtract the per-frame mean. Handset and interface DC offsets otherwise
    appear as spurious energy in the lowest filterbank channel."""

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise InvalidEvidenceError(
                "sample rate must be positive", sample_rate=self.sample_rate
            )
        if self.frame_length_ms <= 0.0 or self.frame_shift_ms <= 0.0:
            raise InvalidEvidenceError(
                "frame length and shift must be positive",
                frame_length_ms=self.frame_length_ms,
                frame_shift_ms=self.frame_shift_ms,
            )
        if self.frame_shift_ms > self.frame_length_ms:
            raise InvalidEvidenceError(
                "frame shift exceeds frame length, which would leave gaps in the "
                "analysis and silently discard speech",
                frame_length_ms=self.frame_length_ms,
                frame_shift_ms=self.frame_shift_ms,
            )
        if not 0.0 <= self.pre_emphasis < 1.0:
            raise InvalidEvidenceError(
                "pre-emphasis coefficient must lie in [0, 1)",
                pre_emphasis=self.pre_emphasis,
            )
        if self.dither < 0.0:
            raise InvalidEvidenceError("dither amplitude cannot be negative")

    @property
    def frame_length(self) -> int:
        """Frame length in samples."""
        return int(round(self.frame_length_ms * self.sample_rate / 1000.0))

    @property
    def frame_shift(self) -> int:
        """Frame shift (hop) in samples."""
        return int(round(self.frame_shift_ms * self.sample_rate / 1000.0))

    @property
    def n_fft(self) -> int:
        """FFT size: the next power of two at or above the frame length.

        Powers of two are not merely faster; keeping the size fixed for a given
        configuration means the filterbank matrix is built once and reused, and
        that bin centre frequencies are identical across every recording in a
        corpus.
        """
        return int(2 ** np.ceil(np.log2(max(self.frame_length, 1))))

    @property
    def n_freq_bins(self) -> int:
        """Number of bins in a one-sided spectrum."""
        return self.n_fft // 2 + 1

    def with_sample_rate(self, sample_rate: int) -> FrameConfig:
        """Return the same physical configuration at a different rate."""
        return FrameConfig(
            sample_rate=sample_rate,
            frame_length_ms=self.frame_length_ms,
            frame_shift_ms=self.frame_shift_ms,
            window=self.window,
            pre_emphasis=self.pre_emphasis,
            dither=self.dither,
            remove_dc=self.remove_dc,
        )


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """Where frames sit in time, so that results can be located in the signal."""

    n_frames: int
    frame_length: int
    frame_shift: int
    sample_rate: int

    @property
    def frame_times(self) -> NDArray[np.float64]:
        """Centre time of each frame, in seconds."""
        centres = np.arange(self.n_frames) * self.frame_shift + self.frame_length / 2.0
        return centres / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        """Duration spanned by the frames."""
        if self.n_frames == 0:
            return 0.0
        span = (self.n_frames - 1) * self.frame_shift + self.frame_length
        return span / self.sample_rate

    def segment_mask(self, start_fraction: float, end_fraction: float) -> NDArray[np.bool_]:
        """Boolean mask selecting frames within a fractional span of the signal.

        Used to compare the first third of a call against the final third
        (hypothesis H3: disguise consistency degrades as attention is spent).
        Expressed in fractions so the comparison is meaningful across calls of
        different length.
        """
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise InvalidEvidenceError(
                "segment bounds must satisfy 0 <= start < end <= 1",
                start_fraction=start_fraction,
                end_fraction=end_fraction,
            )
        index = np.arange(self.n_frames)
        lower = start_fraction * self.n_frames
        upper = end_fraction * self.n_frames
        return (index >= lower) & (index < upper)


def pre_emphasise(signal: NDArray[np.float64], coefficient: float) -> NDArray[np.float64]:
    """Apply a first-order high-pass filter ``y[n] = x[n] - a * x[n-1]``.

    The first sample is filtered against a replica of itself rather than against
    zero. Filtering against zero produces a spurious transient with amplitude
    equal to the first sample, which is then windowed into the first frame and
    inflates its energy.
    """
    if coefficient == 0.0:
        return np.asarray(signal, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return signal
    emphasised = np.empty_like(signal)
    emphasised[0] = signal[0] * (1.0 - coefficient)
    emphasised[1:] = signal[1:] - coefficient * signal[:-1]
    return emphasised


def frame_signal(
    signal: NDArray[np.float64],
    config: FrameConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], FrameGeometry]:
    """Split a signal into overlapping, windowed frames.

    Returns
    -------
    frames:
        Array of shape ``(n_frames, frame_length)``, pre-emphasised, dithered,
        DC-removed and windowed, in that order.
    geometry:
        Where the frames sit in time.

    Notes
    -----
    Only whole frames are produced. A trailing partial frame is discarded rather
    than zero-padded: padding fabricates silence, which then registers as a
    low-energy speech frame and drags the estimated noise floor down.

    The frame matrix is built with a strided view and copied once. The obvious
    Python loop is roughly two orders of magnitude slower on a three-minute
    call, and a database search runs this over every enrolled recording.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise InvalidEvidenceError(
            "framing expects a single-channel signal; mix down or select a "
            "channel before analysis",
            shape=signal.shape,
        )

    length = config.frame_length
    shift = config.frame_shift
    if signal.size < length:
        raise InsufficientDataError(
            "signal is shorter than a single analysis frame",
            n_samples=int(signal.size),
            required=length,
            duration_ms=float(signal.size) / config.sample_rate * 1000.0,
        )

    working = pre_emphasise(signal, config.pre_emphasis)

    if config.dither > 0.0:
        generator = rng if rng is not None else np.random.default_rng(0)
        working = working + generator.uniform(-config.dither, config.dither, working.size)

    n_frames = 1 + (working.size - length) // shift
    strided = np.lib.stride_tricks.sliding_window_view(working, length)[::shift][:n_frames]
    frames = np.array(strided, dtype=np.float64, copy=True)

    if config.remove_dc:
        frames -= frames.mean(axis=1, keepdims=True)

    frames *= make_window(config.window, length)

    geometry = FrameGeometry(
        n_frames=n_frames,
        frame_length=length,
        frame_shift=shift,
        sample_rate=config.sample_rate,
    )
    return frames, geometry
