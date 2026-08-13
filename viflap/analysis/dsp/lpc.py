"""Linear prediction: spectral envelope, formants, and the glottal residual.

Linear predictive analysis models a frame of speech as the output of an
all-pole filter driven by an excitation signal. The poles approximate the vocal
tract resonances; the residual after inverse filtering approximates the
excitation, and hence the glottal source.

This is the machinery behind the anatomical feature classes the proposal
identifies as most resistant to disguise (section 4.2). It is worth being
precise about what survives the channel, because the answer is not encouraging
and pretending otherwise would be the central error this system exists to avoid:

- **Formant frequencies and bandwidths** are estimated from the pole locations.
  These survive narrowband coding reasonably well within the passband, because
  the codec is itself built on a linear-prediction model and preserves the
  spectral envelope by design. F1 and F2 are usually recoverable; F3 sits near
  the upper band edge at 3.4 kHz and is often unreliable; F4 and above are gone.

- **Antiformants** — spectral zeros introduced by the side branch of the nasal
  cavity during nasal consonants — are the most speaker-specific and the most
  disguise-resistant acoustic resource available, since the sinuses have no
  motor pathway to alter. An all-pole model cannot represent a zero directly.
  They are estimated here by inverse-filtering and analysing the residual
  spectrum, which is an approximation and is documented as one.

- **Glottal source measures** derived from the residual are the most
  channel-fragile. A CELP coder does not transmit the residual; it transmits an
  index into a codebook of excitation vectors chosen to minimise perceptual
  error. At 12.2 kbit/s the reconstruction retains some periodicity structure;
  at 4.75 kbit/s the fine structure is essentially synthetic. Jitter and shimmer
  measured on decoded speech at low rates characterise the codec, not the
  larynx. :mod:`viflap.analysis.dsp.voice_quality` therefore reports a
  reliability figure alongside every such measurement, and the channel
  characterisation experiment (H1) is what determines where the cutoff lies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.domain.errors import ConvergenceError, InvalidEvidenceError

__all__ = [
    "Formant",
    "FormantTrack",
    "autocorrelation",
    "formants_from_lpc",
    "levinson_durbin",
    "lpc_analysis",
    "lpc_residual",
    "lpc_spectrum",
    "recommended_order",
]


def recommended_order(sample_rate: int, extra_poles: int = 4) -> int:
    """Rule-of-thumb LPC order: one pole pair per kilohertz, plus a few.

    The vocal tract produces roughly one resonance per kilohertz of bandwidth,
    each requiring a pole pair. The additional poles absorb the spectral tilt of
    the glottal source and the radiation characteristic, which are not
    resonances but do shape the envelope.

    At 8 kHz this gives order 12, which is also what AMR itself uses — not a
    coincidence, and convenient, because it means the analysis order matches the
    order at which the signal was coded.
    """
    return int(sample_rate / 1000) + extra_poles


def autocorrelation(frame: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """Biased autocorrelation estimate for lags ``0 .. order``.

    The biased estimator (dividing by the frame length rather than by the number
    of overlapping samples at each lag) is used deliberately. It guarantees a
    positive semi-definite autocorrelation sequence, which in turn guarantees
    that Levinson-Durbin produces a *stable* filter with all poles inside the
    unit circle. The unbiased estimator has lower bias and no such guarantee,
    and an unstable synthesis filter produces formant estimates outside the
    Nyquist range.

    Computed by FFT: direct evaluation is O(N * order), this is O(N log N), and
    it is called once per frame for every frame of every recording.
    """
    frame = np.asarray(frame, dtype=np.float64)
    n = frame.size
    if order >= n:
        raise InvalidEvidenceError(
            "LPC order must be smaller than the frame length",
            order=order,
            frame_length=n,
        )
    n_fft = int(2 ** np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(frame, n_fft)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n_fft)[: order + 1]
    return correlation / n


def levinson_durbin(
    r: NDArray[np.float64], order: int
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Solve the Yule-Walker equations by Levinson-Durbin recursion.

    Returns
    -------
    coefficients:
        Taps of the prediction-error filter ``A(z) = 1 + sum_{k=1..p} a_k z^-k``,
        with ``a[0] = 1``.

        This is the convention used throughout: the returned array is the
        filter numerator that whitens the signal, so the residual is
        ``convolve(x, a)``, synthesis is ``lfilter([1], a, e)``, and the poles of
        the vocal tract model are ``roots(a)``. It is also the convention of
        ``scipy.signal.lfilter`` and of the LSF conversion in the codec model.
        The predictor coefficients are the negation of ``a[1:]``; conflating the
        two flips the sign of every formant estimate's polynomial and is the
        single easiest mistake to make here, so the sign is asserted directly
        against a known autoregressive process in the test suite.
    error:
        Final prediction error power.
    reflection:
        Reflection (PARCOR) coefficients. Their magnitudes are bounded by one
        for a stable filter, which gives a cheap stability check, and they are
        the quantities a CELP coder actually quantises.

    The recursion is O(p^2) rather than the O(p^3) of a general Toeplitz solve,
    and it is numerically better behaved because it never forms the Toeplitz
    matrix explicitly.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.size < order + 1:
        raise InvalidEvidenceError(
            "autocorrelation sequence is too short for the requested order",
            available=int(r.size),
            required=order + 1,
        )
    if r[0] <= 0.0:
        raise ConvergenceError(
            "frame has zero energy; linear prediction is undefined", r0=float(r[0])
        )

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    reflection = np.zeros(order, dtype=np.float64)
    error = float(r[0])

    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1]) if i > 1 else r[i]
        k = -acc / error
        reflection[i - 1] = k

        previous = a[1:i].copy()
        a[1:i] = previous + k * previous[::-1]
        a[i] = k

        error *= 1.0 - k * k
        if error <= 0.0:
            # The recursion has become numerically degenerate, which happens on
            # near-periodic frames where the model can predict perfectly. Stop
            # and return the order reached rather than producing NaNs.
            error = float(np.finfo(np.float64).tiny)
            break

    # The recursion already produces the prediction-error filter taps in the
    # A(z) = 1 + sum a_k z^-k convention; return them unchanged.
    coefficients = np.concatenate([[1.0], a[1 : order + 1]])
    return coefficients, error, reflection


@dataclass(frozen=True, slots=True)
class Formant:
    """One estimated vocal tract resonance."""

    frequency_hz: float
    bandwidth_hz: float

    @property
    def is_plausible(self) -> bool:
        """Whether the resonance could plausibly come from a vocal tract.

        Wide-bandwidth poles are not resonances. They arise where the all-pole
        model is fitting spectral tilt or noise, and admitting them into a
        formant track corrupts the very features chosen for their anatomical
        constraint.
        """
        return 90.0 <= self.frequency_hz <= 5000.0 and self.bandwidth_hz <= 700.0


@dataclass(frozen=True, slots=True)
class FormantTrack:
    """Formants across frames, with the frames where estimation failed marked."""

    frequencies: NDArray[np.float64]
    """Shape ``(n_frames, n_formants)``. ``NaN`` where no plausible pole was
    found at that index — a real occurrence in narrowband speech, and better
    represented than interpolated over."""

    bandwidths: NDArray[np.float64]

    @property
    def n_formants(self) -> int:
        return self.frequencies.shape[1]

    def statistics(self) -> dict[str, float]:
        """Per-formant central tendency and spread, ignoring failed frames.

        The median is used rather than the mean: formant estimation produces
        occasional gross outliers when the root-finder locks onto a spurious
        pole, and a single such value at 4 kHz shifts a mean materially while
        leaving a median untouched.
        """
        stats: dict[str, float] = {}
        for index in range(self.n_formants):
            values = self.frequencies[:, index]
            valid = values[~np.isnan(values)]
            label = f"f{index + 1}"
            if valid.size == 0:
                stats[f"{label}_median_hz"] = float("nan")
                stats[f"{label}_iqr_hz"] = float("nan")
                stats[f"{label}_coverage"] = 0.0
                continue
            stats[f"{label}_median_hz"] = float(np.median(valid))
            stats[f"{label}_iqr_hz"] = float(
                np.percentile(valid, 75) - np.percentile(valid, 25)
            )
            stats[f"{label}_coverage"] = float(valid.size / max(values.size, 1))

        f1 = self.frequencies[:, 0]
        f2 = self.frequencies[:, 1] if self.n_formants > 1 else None
        if f2 is not None:
            both = ~np.isnan(f1) & ~np.isnan(f2)
            if np.any(both):
                # F2/F1 is a coarse proxy for vocal tract length, which is the
                # anatomical parameter formant scaling mostly reflects.
                stats["f2_f1_ratio_median"] = float(np.median(f2[both] / f1[both]))
        return stats


def lpc_analysis(
    frames: NDArray[np.float64], order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit an all-pole model to every frame.

    Returns the prediction coefficients, shape ``(n_frames, order + 1)``, and
    the residual error power per frame.

    Frames on which the recursion fails — digital silence, most often — receive
    a pass-through filter and zero error power, and are identifiable by that
    zero. Failing the whole recording because one frame is silent would be
    wrong; silently substituting a plausible-looking filter would be worse.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    n_frames = frames.shape[0]
    coefficients = np.zeros((n_frames, order + 1), dtype=np.float64)
    coefficients[:, 0] = 1.0
    errors = np.zeros(n_frames, dtype=np.float64)

    for index in range(n_frames):
        try:
            r = autocorrelation(frames[index], order)
            a, error, _ = levinson_durbin(r, order)
        except (ConvergenceError, InvalidEvidenceError):
            continue
        coefficients[index] = a
        errors[index] = error

    return coefficients, errors


def formants_from_lpc(
    coefficients: NDArray[np.float64],
    sample_rate: int,
    n_formants: int = 4,
    min_frequency_hz: float = 90.0,
    max_bandwidth_hz: float = 700.0,
    min_separation_hz: float = 180.0,
) -> FormantTrack:
    """Extract formants from LPC coefficients by root-finding.

    Each complex-conjugate pole pair ``r e^{j theta}`` corresponds to a
    resonance at ``f = theta * fs / (2 pi)`` with bandwidth
    ``B = -ln(r) * fs / pi``. The bandwidth follows from the pole radius: a pole
    close to the unit circle is a sharp resonance, one further inside is broad.

    Selection is by **prominence, then order** — not by taking the lowest ``N``
    poles that pass a bandwidth filter. The difference is not subtle. An
    order-12 fit to a synthesised vowel with formants at 700, 1220 and 2600 Hz
    typically yields poles at 708/44, 1204/104, 2617/130 — the true resonances,
    all sharp — *plus* spurious wide poles at around 1240/654 and 2222/869 where
    the model is fitting spectral tilt. Taking the three lowest survivors of a
    700 Hz bandwidth filter returns 708, 1204 and 1240: F3 is reported as
    1240 Hz, off by a factor of two, with every appearance of confidence.

    So: filter to the plausible frequency and bandwidth range, rank the
    survivors by bandwidth (a sharp pole is a real resonance, a broad one is the
    model absorbing tilt), keep the ``n_formants`` sharpest, enforce a minimum
    separation so that two poles fitting the same resonance are not reported as
    two formants, and only then sort by frequency.

    Missing entries are left as ``NaN`` rather than shifting the remaining
    formants up an index, which would silently relabel F3 as F2 and destroy the
    frame-to-frame correspondence the statistics depend on.

    This is the classical method with the standard robustness constraints. It is
    not the most accurate available — tracking with continuity constraints
    across frames does better — but it has the property that matters here: it
    fails visibly, by returning ``NaN``, rather than producing a confident wrong
    answer.
    """
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    n_frames = coefficients.shape[0]
    frequencies = np.full((n_frames, n_formants), np.nan, dtype=np.float64)
    bandwidths = np.full((n_frames, n_formants), np.nan, dtype=np.float64)

    for index in range(n_frames):
        row = coefficients[index]
        if row.size < 2 or not np.all(np.isfinite(row)) or np.allclose(row[1:], 0.0):
            continue

        roots = np.roots(row)
        # Each conjugate pair appears twice; keep the upper half-plane only.
        roots = roots[np.imag(roots) > 1e-9]
        if roots.size == 0:
            continue
        # Poles outside the unit circle correspond to an unstable filter and
        # cannot be resonances of a physical system.
        roots = roots[np.abs(roots) < 1.0]
        if roots.size == 0:
            continue

        angles = np.angle(roots)
        radii = np.abs(roots)
        candidate_hz = angles * sample_rate / (2.0 * np.pi)
        candidate_bw = -np.log(np.maximum(radii, 1e-12)) * sample_rate / np.pi

        keep = (candidate_hz >= min_frequency_hz) & (candidate_bw <= max_bandwidth_hz)
        candidate_hz = candidate_hz[keep]
        candidate_bw = candidate_bw[keep]
        if candidate_hz.size == 0:
            continue

        selected_hz, selected_bw = _select_by_prominence(
            candidate_hz, candidate_bw, n_formants, min_separation_hz
        )
        if selected_hz.size == 0:
            continue

        frequencies[index, : selected_hz.size] = selected_hz
        bandwidths[index, : selected_bw.size] = selected_bw

    return FormantTrack(frequencies=frequencies, bandwidths=bandwidths)


def _select_by_prominence(
    frequencies: NDArray[np.float64],
    bandwidths: NDArray[np.float64],
    n_formants: int,
    min_separation_hz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Choose the ``n_formants`` sharpest well-separated poles, ordered by frequency.

    Sharpness (narrow bandwidth) is the discriminator between a genuine vocal
    tract resonance and a pole the all-pole model has spent on spectral tilt or
    noise. The separation constraint prevents two poles that have converged on
    the same resonance — common when the model order exceeds what the spectrum
    supports — from being reported as two distinct formants.
    """
    sharpest_first = np.argsort(bandwidths)
    chosen_hz: list[float] = []
    chosen_bw: list[float] = []

    for position in sharpest_first:
        frequency = float(frequencies[position])
        if any(abs(frequency - taken) < min_separation_hz for taken in chosen_hz):
            continue
        chosen_hz.append(frequency)
        chosen_bw.append(float(bandwidths[position]))
        if len(chosen_hz) == n_formants:
            break

    if not chosen_hz:
        return np.zeros(0), np.zeros(0)

    by_frequency = np.argsort(chosen_hz)
    return (
        np.array(chosen_hz, dtype=np.float64)[by_frequency],
        np.array(chosen_bw, dtype=np.float64)[by_frequency],
    )


def lpc_residual(
    frames: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Inverse-filter each frame to recover the excitation estimate.

    Applying ``A(z)`` to the speech removes the vocal tract contribution,
    leaving an approximation of the glottal source: a train of sharp pulses at
    the fundamental period during voiced speech, and noise during unvoiced.

    Implemented as a direct convolution, truncated to the frame length. The
    frame is already windowed, so the leading samples of the residual are
    affected by the filter's startup transient; the voice-quality measures that
    consume this discard the first ``order`` samples for that reason.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    if frames.shape[0] != coefficients.shape[0]:
        raise InvalidEvidenceError(
            "frame count and coefficient row count differ",
            n_frames=frames.shape[0],
            n_coefficient_rows=coefficients.shape[0],
        )

    residual = np.zeros_like(frames)
    for index in range(frames.shape[0]):
        filtered = np.convolve(frames[index], coefficients[index], mode="full")
        residual[index] = filtered[: frames.shape[1]]
    return residual


def lpc_spectrum(
    coefficients: NDArray[np.float64],
    error_power: NDArray[np.float64],
    n_fft: int,
) -> NDArray[np.float64]:
    """All-pole spectral envelope, ``|G / A(e^{jw})|^2``.

    The envelope with the harmonic fine structure removed. Used to locate
    spectral prominences and, by comparison with the observed spectrum, to find
    the regions of spectral *cancellation* that indicate an antiformant.
    """
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    error_power = np.atleast_1d(np.asarray(error_power, dtype=np.float64))
    response = np.fft.rfft(coefficients, n=n_fft, axis=-1)
    denominator = np.maximum(np.abs(response) ** 2, 1e-20)
    return error_power[:, None] / denominator
