"""Glottal source measures: jitter, shimmer, harmonics-to-noise ratio.

These characterise the excitation signal rather than the vocal tract filter, and
they are the features most damaged by the channel this system operates over.
That damage is not incidental to the design; it is the reason every measurement
in this module is returned alongside an explicit reliability figure.

Why the channel destroys them
-----------------------------
AMR narrowband is an ACELP coder: algebraic code-excited linear prediction. It
does not transmit the speech waveform. It transmits a quantised spectral
envelope, a pitch lag, and an *index* into an algebraic codebook of excitation
vectors, chosen so that the resynthesised signal minimises a perceptually
weighted error. The decoder then constructs an excitation that sounds right.

Jitter is cycle-to-cycle variation in the fundamental period, typically under
one percent. Shimmer is cycle-to-cycle variation in amplitude. Both are
properties of the fine structure of the excitation — exactly the part the coder
replaces with a codebook entry. At 12.2 kbit/s enough structure survives for the
measurements to retain some relationship to the source; at 4.75 kbit/s they
substantially measure the codebook.

Consequently this module reports :class:`VoiceQualityMeasures.reliability`, a
figure derived from the observable properties of the signal, and the acoustic
stream is required to attach it to any evidence derived from these features. A
jitter measurement from a 4.75 kbit/s recording is not weak evidence about the
speaker. It is a measurement of something else, and the calibration layer must
be able to tell the difference.

Definitions follow the conventions used in the phonetics literature (as
implemented in Praat), so that values are comparable with published work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.dsp.pitch import F0Track
from viflap.domain.errors import InsufficientDataError

__all__ = ["VoiceQualityMeasures", "measure_voice_quality"]


@dataclass(frozen=True, slots=True)
class VoiceQualityMeasures:
    """Glottal source measurements with an explicit reliability figure."""

    jitter_local: float
    """Mean absolute difference between consecutive periods, divided by the mean
    period. Dimensionless; healthy modal voice is typically below 0.01."""

    jitter_rap: float
    """Relative average perturbation: the same quantity smoothed over three
    periods. Less sensitive to isolated period-detection errors than
    ``jitter_local``, which matters on band-limited speech where such errors are
    common."""

    jitter_ppq5: float
    """Five-point period perturbation quotient. Smoother again; the difference
    between this and ``jitter_local`` indicates how much of the apparent jitter
    is short-term measurement noise."""

    shimmer_local: float
    """Mean absolute difference between consecutive peak amplitudes, divided by
    the mean amplitude."""

    shimmer_apq3: float
    """Three-point amplitude perturbation quotient."""

    harmonics_to_noise_db: float
    """Ratio of periodic to aperiodic energy, in decibels. Higher values mean a
    more purely periodic voice. Breathiness, hoarseness and background noise all
    lower it, which is why it must be read together with the recording SNR
    rather than as a property of the speaker alone."""

    n_periods: int
    reliability: float
    """In ``[0, 1]``. How far these measurements can be attributed to the
    speaker rather than to the channel. Combines the number of usable periods,
    the periodicity strength reported by the pitch tracker, and the harmonics-to-
    noise ratio. Below roughly 0.3 the measurements should be treated as
    uninformative about the speaker and the features excluded rather than
    down-weighted."""

    @property
    def is_reliable(self) -> bool:
        return self.reliability >= 0.3

    def as_features(self) -> dict[str, float]:
        """Feature dictionary for the acoustic stream.

        Reliability travels with the features. A consumer that ignores it will
        at least have been handed it.
        """
        return {
            "jitter_local": self.jitter_local,
            "jitter_rap": self.jitter_rap,
            "jitter_ppq5": self.jitter_ppq5,
            "shimmer_local": self.shimmer_local,
            "shimmer_apq3": self.shimmer_apq3,
            "hnr_db": self.harmonics_to_noise_db,
            "voice_quality_reliability": self.reliability,
        }


def measure_voice_quality(
    signal: NDArray[np.float64],
    f0_track: F0Track,
    *,
    min_periods: int = 20,
    min_periodicity: float = 0.5,
) -> VoiceQualityMeasures:
    """Measure jitter, shimmer and HNR from a signal and its pitch track.

    Period marks are placed by following the pitch track and locating the
    amplitude peak within each expected period. This is a simplification of
    proper glottal closure instant detection, and it is chosen deliberately:
    closure instant detection on ACELP-decoded speech locates the codebook
    pulses, which is a precise measurement of the wrong thing. Peak-picking
    within a period is cruder but degrades gracefully rather than producing
    confidently wrong marks.

    Raises
    ------
    InsufficientDataError
        If fewer than ``min_periods`` usable periods are found. Perturbation
        measures over a handful of periods are dominated by their own estimation
        variance.
    """
    signal = np.asarray(signal, dtype=np.float64)
    voiced = f0_track.voiced & (f0_track.periodicity >= min_periodicity)

    if np.count_nonzero(voiced) < 3:
        raise InsufficientDataError(
            "too few reliably voiced frames for glottal source measurement",
            voiced_frames=int(np.count_nonzero(voiced)),
        )

    runs = _place_period_marks(signal, f0_track, voiced)

    # Periods are differences between consecutive marks *within* a voiced run.
    # Differencing across a run boundary would measure the length of a silence
    # and enter it into the jitter statistic as an enormous perturbation.
    period_chunks: list[NDArray[np.float64]] = []
    amplitude_chunks: list[NDArray[np.float64]] = []
    for marks, amplitudes in runs:
        if marks.size < 2:
            continue
        period_chunks.append(np.diff(marks).astype(np.float64) / f0_track.sample_rate)
        amplitude_chunks.append(amplitudes[:-1])

    if not period_chunks:
        raise InsufficientDataError(
            "no usable glottal periods for perturbation measurement"
        )

    periods = np.concatenate(period_chunks)
    amplitudes = np.concatenate(amplitude_chunks)

    # Discard periods implausibly far from the median: these are almost always a
    # missed or doubled mark, and a single one dominates a jitter figure
    # computed over a few hundred periods.
    median_period = float(np.median(periods))
    keep = np.abs(periods - median_period) < 0.35 * median_period
    periods = periods[keep]
    amplitudes = amplitudes[keep]

    if periods.size < min_periods:
        raise InsufficientDataError(
            "too few plausible glottal periods after outlier rejection",
            n_periods=int(periods.size),
            required=min_periods,
        )

    jitter_local = _perturbation(periods, order=1)
    jitter_rap = _perturbation(periods, order=3)
    jitter_ppq5 = _perturbation(periods, order=5)
    shimmer_local = _perturbation(amplitudes, order=1)
    shimmer_apq3 = _perturbation(amplitudes, order=3)

    hnr_db = _harmonics_to_noise(signal, f0_track, voiced)

    reliability = _reliability(
        n_periods=periods.size,
        mean_periodicity=float(np.mean(f0_track.periodicity[voiced])),
        hnr_db=hnr_db,
    )

    return VoiceQualityMeasures(
        jitter_local=jitter_local,
        jitter_rap=jitter_rap,
        jitter_ppq5=jitter_ppq5,
        shimmer_local=shimmer_local,
        shimmer_apq3=shimmer_apq3,
        harmonics_to_noise_db=hnr_db,
        n_periods=int(periods.size),
        reliability=reliability,
    )


def _place_period_marks(
    signal: NDArray[np.float64], f0_track: F0Track, voiced: NDArray[np.bool_]
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Locate one amplitude peak per glottal period, grouped into voiced runs.

    Marks are placed by **waveform cross-correlation**, pitch-synchronously.
    From the current mark, a template window of roughly one period is matched
    against every candidate position between 0.7 and 1.3 periods ahead, and the
    next mark is placed where normalised cross-correlation peaks, refined to
    sub-sample resolution by parabolic interpolation. This is the method used in
    the phonetics literature (Praat's periodic cross-correlation point process),
    and both of its properties are necessary here.

    **Why not step by a fixed period.** The true period is fractional — 66.7
    samples at 120 Hz and 8 kHz — so an integer step drifts a whole period every
    hundred cycles. The measured "jitter" is then the drift of the analysis, a
    property of the arithmetic rather than of the larynx, and it swamps the real
    perturbation entirely.

    **Why not pick the amplitude peak.** Each glottal pulse excites the formants,
    and the response rings. With F1 near 700 Hz the ringing has a period of about
    eleven samples, so a single glottal period contains six oscillation peaks of
    comparable height. Amplitude peak-picking jumps between adjacent ones,
    producing apparent jitter on the order of a sixth of a period — again, in a
    signal that has none. Cross-correlation matches the *shape* of the waveform
    across a whole period, so the ringing is part of the template rather than a
    source of ambiguity.

    Returns one ``(marks, amplitudes)`` pair per contiguous voiced run, with
    fractional marks. Runs are kept separate so that no period is ever measured
    across a silence.
    """
    shift_samples = max(1, int(round(f0_track.frame_shift_seconds * f0_track.sample_rate)))
    magnitude = np.abs(signal)
    runs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    # Bridge one- and two-frame gaps in the voicing decision before segmenting.
    # A 10 ms unvoiced blip in the middle of a sustained vowel is a limitation of
    # frame-level voicing detection, not a pause in the speech. Left unbridged it
    # fragments a two-second vowel into dozens of runs, each too short to yield
    # the three periods this analysis needs, and the recording is then reported
    # as containing no usable voiced speech at all.
    contiguous = _bridge_short_gaps(voiced, max_gap=2)

    for start_frame, end_frame in _contiguous_runs(contiguous):
        run_start = start_frame * shift_samples
        run_end = min(end_frame * shift_samples + shift_samples, signal.size)
        period = _period_at(f0_track, start_frame)
        if period is None:
            # The run's first frame may itself be a bridged one; fall back to
            # the first frame in the run that carries a period estimate.
            for candidate_frame in range(start_frame, end_frame + 1):
                period = _period_at(f0_track, candidate_frame)
                if period is not None:
                    break
        if period is None or run_end - run_start < 3 * period:
            continue

        # Seed on the strongest peak in the first period of the run. Only the
        # seed uses amplitude; every subsequent mark is placed by correlation
        # relative to it, so a poor seed shifts all marks equally and leaves the
        # differences between them — which is what jitter measures — unaffected.
        #
        # The seed is offset by half a period from the start of the run so that
        # the first correlation template fits entirely inside the signal. Seeding
        # at the very first sample leaves no room for it, the first correlation
        # step fails, and the whole run is discarded — which looks identical to
        # "this recording has no usable voiced speech".
        seed_offset = max(int(round(0.5 * period)), 1)
        seed_start = run_start + seed_offset
        seed_end = min(seed_start + int(round(period)), run_end)
        if seed_end <= seed_start:
            continue
        current = float(seed_start + int(np.argmax(magnitude[seed_start:seed_end])))

        marks = [current]
        amplitudes = [_period_peak_amplitude(magnitude, current, period)]

        while True:
            frame_index = int(min(current // shift_samples, f0_track.f0_hz.size - 1))
            period = _period_at(f0_track, frame_index) or period
            next_mark = _correlate_next_mark(signal, current, period, run_end)
            if next_mark is None:
                break
            marks.append(next_mark)
            amplitudes.append(_period_peak_amplitude(magnitude, next_mark, period))
            current = next_mark

        if len(marks) >= 2:
            runs.append(
                (
                    np.array(marks, dtype=np.float64),
                    np.array(amplitudes, dtype=np.float64),
                )
            )
    return runs


def _correlate_next_mark(
    signal: NDArray[np.float64], current: float, period: float, limit: int
) -> float | None:
    """Find the next period mark by normalised cross-correlation."""
    half = max(2, int(round(0.5 * period)))
    centre = int(round(current))
    template_start = centre - half
    template_end = centre + half
    if template_start < 0 or template_end > signal.size:
        return None

    template = signal[template_start:template_end]
    template_norm = float(np.linalg.norm(template))
    if template_norm <= 1e-12:
        return None

    low = centre + int(round(0.7 * period))
    high = centre + int(round(1.3 * period))
    if high + half > min(limit, signal.size) or low - half < 0 or high <= low:
        return None

    # One strided view over the candidate span, so the whole search is a single
    # matrix-vector product rather than a Python loop per candidate.
    span = signal[low - half : high + half]
    windows = np.lib.stride_tricks.sliding_window_view(span, 2 * half)
    norms = np.linalg.norm(windows, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return None

    scores = np.full(windows.shape[0], -np.inf)
    scores[valid] = (windows[valid] @ template) / (norms[valid] * template_norm)

    best = int(np.argmax(scores))
    refined = _parabolic_peak(scores, best)
    return float(low + refined)


def _parabolic_peak(values: NDArray[np.float64], index: int) -> float:
    """Sub-sample refinement of a discrete maximum by parabolic interpolation.

    Sub-sample accuracy is not a refinement here, it is the measurement. Typical
    jitter is under one percent of a period; at 8 kHz that is under 0.7 samples,
    so an integer-resolution mark cannot resolve the quantity at all and would
    report the quantisation grid instead.
    """
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    if abs(offset) > 1.0:
        return float(index)
    return float(index) + float(offset)


def _period_peak_amplitude(
    magnitude: NDArray[np.float64], mark: float, period: float
) -> float:
    """Peak magnitude within the period centred on ``mark``.

    Shimmer is defined on the peak amplitude of each period, so the amplitude is
    taken over the whole period rather than at the mark itself. Sampling at the
    mark would measure whichever point of the formant ringing the correlation
    happened to align to, which varies with the template and not with the voice.
    """
    half = max(1, int(round(0.5 * period)))
    centre = int(round(mark))
    low = max(0, centre - half)
    high = min(magnitude.size, centre + half)
    if high <= low:
        return 0.0
    return float(np.max(magnitude[low:high]))


def _period_at(f0_track: F0Track, frame_index: int) -> float | None:
    """Period in samples at a frame, or ``None`` if unusable."""
    if not 0 <= frame_index < f0_track.period_samples.size:
        return None
    period = float(f0_track.period_samples[frame_index])
    if not np.isfinite(period) or period < 2.0:
        return None
    return period


def _bridge_short_gaps(mask: NDArray[np.bool_], max_gap: int) -> NDArray[np.bool_]:
    """Fill runs of ``False`` no longer than ``max_gap`` between ``True`` runs.

    Morphological closing on a boolean sequence. Only interior gaps are filled:
    leading and trailing false regions are genuine boundaries of the voiced
    region and are left alone.
    """
    if max_gap <= 0 or mask.size == 0 or not np.any(mask):
        return mask
    bridged = mask.copy()
    padded = np.concatenate([[True], mask, [True]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    # Pairs now delimit False runs, since the padding starts in the True state.
    for start, end in zip(changes[::2], changes[1::2], strict=False):
        if start == 0 or end >= mask.size:
            continue
        if end - start <= max_gap:
            bridged[start:end] = True
    return bridged


def _contiguous_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` frame indices of each contiguous ``True`` run."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end) - 1)
        for start, end in zip(changes[::2], changes[1::2], strict=False)
    ]


def _perturbation(values: NDArray[np.float64], order: int) -> float:
    """Generic perturbation quotient over a smoothing window of ``order`` points.

    For ``order = 1`` this is the mean absolute first difference over the mean,
    i.e. local jitter or shimmer. For larger odd orders it is the mean absolute
    deviation of each value from the mean of its ``order``-point neighbourhood,
    over the mean — the RAP (order 3) and PPQ5 (order 5) definitions.

    Expressing all of them through one function makes the relationship between
    them explicit: they differ only in how much short-term variation is smoothed
    away before the deviation is taken.
    """
    values = np.asarray(values, dtype=np.float64)
    mean_value = float(np.mean(np.abs(values)))
    if mean_value <= 0.0 or values.size <= order:
        return float("nan")

    if order == 1:
        deviation = float(np.mean(np.abs(np.diff(values))))
        return deviation / mean_value

    half = order // 2
    window = np.ones(order) / order
    smoothed = np.convolve(values, window, mode="valid")
    centre = values[half : values.size - half]
    deviation = float(np.mean(np.abs(centre - smoothed)))
    return deviation / mean_value


def _harmonics_to_noise(
    signal: NDArray[np.float64], f0_track: F0Track, voiced: NDArray[np.bool_]
) -> float:
    """Harmonics-to-noise ratio, following Boersma (1993).

    For a signal that is a periodic component plus additive noise, the
    normalised autocorrelation at the fundamental period is the fraction of the
    energy that is periodic, giving

    .. code-block:: text

        HNR = 10 log10[ r(T) / (1 - r(T)) ]

    Two corrections separate a working implementation from one that reports
    4 dB for a perfectly periodic signal.

    **Divide out the window's own autocorrelation.** A windowed segment of a
    periodic signal does not autocorrelate to one at the period, because the
    window tapers: the overlap between the segment and its shifted copy is
    weighted by ``r_w(tau)``, the autocorrelation of the window itself. Boersma's
    correction is ``r_x(tau) = r_xw(tau) / r_w(tau)``. Without it the reported
    ratio falls with lag, so low-pitched voices appear noisier than
    high-pitched ones purely as an artefact.

    **Interpolate the peak.** The true period is fractional. Evaluating at a
    rounded lag misaligns a 2600 Hz harmonic — period three samples — by an
    appreciable fraction of its own cycle, and the resulting correlation deficit
    is charged to noise. Parabolic interpolation of the correlation peak
    recovers the value at the true, fractional period.

    Averaged over voiced frames in the linear domain before conversion to
    decibels. Averaging decibel values gives quiet frames the same weight as
    loud ones and produces a figure dominated by the worst frames in the file.
    """
    shift_samples = int(round(f0_track.frame_shift_seconds * f0_track.sample_rate))
    window_samples = max(
        shift_samples * 4,
        3
        * int(
            np.nanmax(
                np.where(np.isfinite(f0_track.period_samples), f0_track.period_samples, 0.0)
            )
            or 1
        ),
    )
    ratios: list[float] = []

    window = np.hanning(window_samples)
    window_energy = float(np.dot(window, window))
    if window_energy <= 0.0:
        return float("nan")

    for frame_index in np.flatnonzero(voiced):
        period = f0_track.period_samples[frame_index]
        if not np.isfinite(period) or period < 2.0:
            continue
        start = int(frame_index) * shift_samples
        end = start + window_samples
        if end > signal.size:
            continue

        segment = signal[start:end]
        segment = (segment - segment.mean()) * window
        energy = float(np.dot(segment, segment))
        if energy <= 0.0:
            continue

        lag = int(round(period))
        low = max(2, lag - 3)
        high = min(lag + 4, segment.size - 1)
        if high <= low:
            continue

        lags = np.arange(low, high)
        corrected = np.empty(lags.size, dtype=np.float64)
        for position, candidate in enumerate(lags):
            raw = float(np.dot(segment[:-candidate], segment[candidate:])) / energy
            # Autocorrelation of the window at the same lag, normalised.
            window_correlation = (
                float(np.dot(window[:-candidate], window[candidate:])) / window_energy
            )
            corrected[position] = (
                raw / window_correlation if window_correlation > 1e-6 else 0.0
            )

        best_index = int(np.argmax(corrected))
        peak = _interpolated_peak_value(corrected, best_index)

        # Clip below one: values at or above it imply exactly zero noise energy,
        # which is a numerical artefact rather than a measurement.
        normalised = float(np.clip(peak, 1e-6, 1.0 - 1e-6))
        ratios.append(normalised / (1.0 - normalised))

    if not ratios:
        return float("nan")
    return float(10.0 * np.log10(np.mean(ratios)))


def _interpolated_peak_value(values: NDArray[np.float64], index: int) -> float:
    """Height of a discrete maximum after parabolic interpolation."""
    if index <= 0 or index >= values.size - 1:
        return float(values[index])
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return float(centre)
    offset = 0.5 * (left - right) / denominator
    if abs(offset) > 1.0:
        return float(centre)
    return float(centre - 0.25 * (left - right) * offset)


def _reliability(n_periods: int, mean_periodicity: float, hnr_db: float) -> float:
    """Combine the observable indicators of measurement trustworthiness.

    Three factors, multiplied because each is individually necessary:

    - **Sample size.** Perturbation estimates have variance inversely
      proportional to the period count; saturates at 200 periods, roughly two
      seconds of voiced speech.
    - **Periodicity.** How confidently the pitch tracker located each period. If
      the periods are wrong, their variation measures nothing.
    - **Harmonics-to-noise ratio.** Below about 5 dB the aperiodic component
      dominates and the periodic structure being measured is largely noise.
      This is the factor that low-bitrate coding degrades.

    The figure is a monotone summary of signal properties, not a probability. It
    is reported so that the calibration layer can condition on it and so that an
    investigator can see when a feature class was measured under conditions that
    do not support it.
    """
    size_factor = float(np.clip(n_periods / 200.0, 0.0, 1.0))
    periodicity_factor = float(np.clip(mean_periodicity, 0.0, 1.0))
    if not np.isfinite(hnr_db):
        hnr_factor = 0.0
    else:
        hnr_factor = float(np.clip((hnr_db - 2.0) / 13.0, 0.0, 1.0))
    return size_factor * periodicity_factor * hnr_factor
