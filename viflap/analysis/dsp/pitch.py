"""Fundamental frequency estimation by the YIN algorithm.

YIN (de Cheveigné and Kawahara, 2002) is used rather than plain autocorrelation
because of one property that matters disproportionately here: its cumulative
mean normalised difference function suppresses the zero-lag maximum that causes
autocorrelation methods to report octave errors. An octave error is not a small
error in this application — it moves a male speaker's estimated F0 into the
female range, and F0 statistics feed both the prosodic feature set and the
disguise analysis.

A caution that belongs with the implementation rather than in a footnote. F0 is
among the *least* useful speaker-discriminative features under the conditions
this system operates in, for two independent reasons:

1. It is trivially and voluntarily alterable. Raising or lowering pitch is the
   first thing an untrained speaker does when disguising their voice, requires
   no equipment and no practice.
2. It varies within a speaker with affect, fatigue, health and interlocutor,
   often by more than it varies between speakers of the same sex.

It is computed because prosodic *dynamics* — contour shape, the variability of
F0 rather than its mean, the relationship between F0 and phrase position — are
considerably more stable than the mean, and because measuring how F0-derived
features degrade relative to nasal and articulatory-timing features is the
substance of hypothesis H2. It is not computed because pitch identifies people.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.domain.errors import InvalidEvidenceError

__all__ = ["F0Track", "PitchConfig", "estimate_f0"]


@dataclass(frozen=True, slots=True)
class PitchConfig:
    """Pitch estimation parameters.

    The default range spans adult male and female speakers with margin. It is
    deliberately not narrowed by an assumed sex: doing so would embed an
    assumption about the speaker into the measurement of the speaker, and would
    make the octave errors it was meant to prevent invisible.
    """

    min_f0_hz: float = 60.0
    max_f0_hz: float = 400.0
    frame_length_ms: float = 40.0
    """Longer than the cepstral analysis window. YIN needs at least two periods
    of the lowest frequency it must resolve; at 60 Hz that is 33 ms."""

    frame_shift_ms: float = 10.0
    threshold: float = 0.15
    """YIN's absolute threshold on the normalised difference function. Lower
    values demand stronger periodicity before declaring a frame voiced. 0.15 is
    the value from the original paper and is appropriate for band-limited
    speech, where periodicity is partially obscured."""

    def __post_init__(self) -> None:
        if not 0.0 < self.min_f0_hz < self.max_f0_hz:
            raise InvalidEvidenceError(
                "pitch range must satisfy 0 < min < max",
                min_f0_hz=self.min_f0_hz,
                max_f0_hz=self.max_f0_hz,
            )
        if not 0.0 < self.threshold < 1.0:
            raise InvalidEvidenceError(
                "YIN threshold must lie in (0, 1)", threshold=self.threshold
            )


@dataclass(frozen=True, slots=True)
class F0Track:
    """Fundamental frequency across frames."""

    f0_hz: NDArray[np.float64]
    """``NaN`` on unvoiced frames. Unvoiced frames have no fundamental; filling
    them with zero or an interpolated value would corrupt every statistic
    computed from the track."""

    periodicity: NDArray[np.float64]
    """``1 - d'(tau)`` at the chosen lag: how strongly periodic the frame is.
    Used as a confidence weight and to gate the voice-quality measures."""

    frame_shift_seconds: float
    period_samples: NDArray[np.float64]
    """Estimated period in samples, with sub-sample refinement. Retained
    because the voice-quality measures need period *marks*, and re-deriving
    them from the frequency would discard the interpolation."""

    sample_rate: int

    @property
    def voiced(self) -> NDArray[np.bool_]:
        return ~np.isnan(self.f0_hz)

    @property
    def voiced_fraction(self) -> float:
        if self.f0_hz.size == 0:
            return 0.0
        return float(np.count_nonzero(self.voiced) / self.f0_hz.size)

    def statistics(self) -> dict[str, float]:
        """Summary statistics over voiced frames.

        Reported in semitones as well as hertz. Hertz is a linear scale and
        pitch perception and production are approximately logarithmic, so a
        standard deviation in hertz is not comparable between a low-pitched and
        a high-pitched speaker: the same *proportional* variability yields a
        larger figure for the higher-pitched voice. Semitones remove that
        artefact, which matters when comparing variability across speakers.
        """
        voiced = self.f0_hz[self.voiced]
        if voiced.size < 2:
            return {
                "f0_median_hz": float("nan"),
                "f0_iqr_hz": float("nan"),
                "f0_std_semitones": float("nan"),
                "f0_range_semitones": float("nan"),
                "voiced_fraction": self.voiced_fraction,
            }
        semitones = 12.0 * np.log2(voiced / np.median(voiced))
        return {
            "f0_median_hz": float(np.median(voiced)),
            "f0_iqr_hz": float(np.percentile(voiced, 75) - np.percentile(voiced, 25)),
            "f0_std_semitones": float(np.std(semitones)),
            "f0_range_semitones": float(
                np.percentile(semitones, 95) - np.percentile(semitones, 5)
            ),
            "voiced_fraction": self.voiced_fraction,
        }


def estimate_f0(
    signal: NDArray[np.float64], sample_rate: int, config: PitchConfig | None = None
) -> F0Track:
    """Estimate F0 frame by frame using YIN.

    The four steps of the algorithm, in order:

    1. **Difference function.** ``d(tau) = sum_j (x_j - x_{j+tau})^2``, computed
       via autocorrelation so the cost is O(N log N) rather than O(N * tau_max).
    2. **Cumulative mean normalisation.** ``d'(tau) = d(tau) / [(1/tau) sum_{j<=tau}
       d(j)]``, with ``d'(0) = 1``. This is the step that removes the zero-lag
       maximum and with it the dominant source of octave errors.
    3. **Absolute threshold.** Take the first local minimum below the threshold
       rather than the global minimum. Taking the global minimum reintroduces
       octave errors, because ``d'`` at twice the true period is often slightly
       lower than at the true period.
    4. **Parabolic interpolation.** Refine the lag to sub-sample resolution. At
       8 kHz an integer lag quantises a 200 Hz estimate to steps of about 5 Hz,
       which is coarse enough to distort the variability statistics that are the
       point of computing F0 at all.
    """
    config = config or PitchConfig()
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise InvalidEvidenceError("pitch estimation expects a single channel")

    frame_length = int(round(config.frame_length_ms * sample_rate / 1000.0))
    frame_shift = int(round(config.frame_shift_ms * sample_rate / 1000.0))
    min_lag = max(2, int(np.floor(sample_rate / config.max_f0_hz)))
    max_lag = int(np.ceil(sample_rate / config.min_f0_hz))

    if frame_length <= max_lag:
        raise InvalidEvidenceError(
            "pitch analysis window is too short to resolve the lowest requested "
            "frequency; at least two periods are required",
            frame_length=frame_length,
            required=max_lag * 2,
            min_f0_hz=config.min_f0_hz,
        )
    if signal.size < frame_length:
        return F0Track(
            f0_hz=np.zeros(0),
            periodicity=np.zeros(0),
            frame_shift_seconds=frame_shift / sample_rate,
            period_samples=np.zeros(0),
            sample_rate=sample_rate,
        )

    n_frames = 1 + (signal.size - frame_length) // frame_shift
    f0 = np.full(n_frames, np.nan, dtype=np.float64)
    periodicity = np.zeros(n_frames, dtype=np.float64)
    periods = np.full(n_frames, np.nan, dtype=np.float64)

    for index in range(n_frames):
        start = index * frame_shift
        frame = signal[start : start + frame_length]
        cmndf = _cumulative_mean_normalised_difference(frame, max_lag)

        lag = _absolute_threshold(cmndf, min_lag, max_lag, config.threshold)
        if lag is None:
            continue

        refined_lag, refined_value = _parabolic_refine(cmndf, lag)
        if refined_lag <= 0.0:
            continue

        f0[index] = sample_rate / refined_lag
        periods[index] = refined_lag
        periodicity[index] = float(np.clip(1.0 - refined_value, 0.0, 1.0))

    # Reject estimates that escaped the requested range through interpolation.
    out_of_range = (f0 < config.min_f0_hz) | (f0 > config.max_f0_hz)
    f0[out_of_range] = np.nan
    periods[out_of_range] = np.nan
    periodicity[out_of_range] = 0.0

    return F0Track(
        f0_hz=f0,
        periodicity=periodicity,
        frame_shift_seconds=frame_shift / sample_rate,
        period_samples=periods,
        sample_rate=sample_rate,
    )


def _cumulative_mean_normalised_difference(
    frame: NDArray[np.float64], max_lag: int
) -> NDArray[np.float64]:
    """YIN steps 1 and 2 for one frame."""
    n = frame.size
    max_lag = min(max_lag, n - 1)

    # d(tau) = r(0) + r_tail(tau) - 2 r_xx(tau), where the running power terms
    # are obtained from cumulative sums and the correlation from an FFT.
    power = frame**2
    cumulative_power = np.concatenate([[0.0], np.cumsum(power)])
    total = cumulative_power[n]

    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(frame, n_fft)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n_fft)[: max_lag + 1]

    lags = np.arange(max_lag + 1)
    # d(tau) = sum_{j=0}^{n-tau-1} (x[j] - x[j+tau])^2, expanded into three
    # terms so that all of them come from cumulative sums and one FFT:
    #   leading  = sum_{j=0}^{n-tau-1} x[j]^2      -> cumulative_power[n - tau]
    #   trailing = sum_{j=tau}^{n-1}   x[j]^2      -> total - cumulative_power[tau]
    #   cross    = sum_{j} x[j] x[j+tau]           -> the autocorrelation
    # Getting `leading` wrong is easy and near-silent: d(0) still comes out at
    # zero because the normalisation fixes d'(0) = 1 by definition, so the
    # defect only shows as octave errors at some fundamentals and not others.
    leading = cumulative_power[n - lags]
    trailing = total - cumulative_power[lags]
    difference = np.maximum(leading + trailing - 2.0 * correlation, 0.0)

    cmndf = np.ones(max_lag + 1, dtype=np.float64)
    running = np.cumsum(difference[1:])
    denominators = running / np.arange(1, max_lag + 1)
    nonzero = denominators > 1e-12
    cmndf[1:][nonzero] = difference[1:][nonzero] / denominators[nonzero]
    return cmndf


def _absolute_threshold(
    cmndf: NDArray[np.float64], min_lag: int, max_lag: int, threshold: float
) -> int | None:
    """YIN step 3: first local minimum of ``d'`` below ``threshold``."""
    max_lag = min(max_lag, cmndf.size - 2)
    lag = min_lag
    while lag < max_lag:
        if cmndf[lag] < threshold:
            # Descend to the bottom of this dip rather than taking its first
            # point, which sits on the shoulder.
            while lag + 1 < max_lag and cmndf[lag + 1] < cmndf[lag]:
                lag += 1
            return lag
        lag += 1
    return None


def _parabolic_refine(cmndf: NDArray[np.float64], lag: int) -> tuple[float, float]:
    """YIN step 4: fit a parabola through the minimum and its neighbours."""
    if lag <= 0 or lag >= cmndf.size - 1:
        return float(lag), float(cmndf[lag])
    left, centre, right = cmndf[lag - 1], cmndf[lag], cmndf[lag + 1]
    denominator = 2.0 * (2.0 * centre - left - right)
    if abs(denominator) < 1e-12:
        return float(lag), float(centre)
    offset = (right - left) / denominator
    # A vertex further than half a sample away means the three points are not a
    # local minimum; the interpolation is not applicable.
    if abs(offset) > 1.0:
        return float(lag), float(centre)
    refined_value = centre - 0.25 * (left - right) * offset
    return float(lag) + offset, float(refined_value)
