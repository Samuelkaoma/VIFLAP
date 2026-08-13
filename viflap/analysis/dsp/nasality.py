"""Nasal segment detection and antiformant estimation.

Section 4.2 of the research proposal identifies nasal resonance as the most
disguise-resistant acoustic resource available, and the argument is sound. The
nasal cavity and paranasal sinuses form a fixed, non-muscular side branch. When
the velum lowers during a nasal consonant, the vocal tract couples to that
branch, and the coupling imposes both characteristic resonances and — the
diagnostically valuable part — *antiformants*: frequencies at which the side
branch cancels energy, whose positions are determined by a morphology the
speaker has no motor pathway to alter. A speaker can change pitch, rate,
register and accent. They cannot change the shape of their sinuses.

This module locates nasal segments and estimates their antiformants. Three
honest qualifications belong with it:

**Antiformant estimation from an all-pole model is indirect.** Linear prediction
fits poles. A zero cannot be represented by a pole, and a pole-zero fit from a
short band-limited frame is poorly conditioned. The method used here — fitting
an all-pole model to the *reciprocal* spectrum, whose poles are the zeros of the
original — is a standard technique and it is an approximation. The estimate is
reported with a confidence figure derived from the depth of the observed
spectral null, and shallow nulls are rejected rather than reported weakly.

**The channel removes part of the evidence.** Nasal antiformants for /m/ sit
typically between 750 and 1250 Hz and for /n/ between about 1450 and 2200 Hz,
both inside the narrowband passband, which is why this feature class is worth
pursuing at all. But the coder's spectral quantisation smooths deep spectral
nulls, and how much survives at each bitrate is an empirical question that
hypothesis H1 exists to answer rather than something to assume.

**The feature is defeated by a head cold.** Nasal occlusion — deliberate or
incidental — decouples the cavity and removes the evidence entirely. This is
listed in the proposal's threat model as a trivial-effort, severe-effect attack
on precisely the most valuable feature class, and it is the reason this stream
is one input to fusion rather than the basis of a system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.dsp.framing import FrameConfig, frame_signal
from viflap.analysis.dsp.lpc import levinson_durbin
from viflap.analysis.dsp.spectral import power_spectrum
from viflap.domain.errors import (
    ConvergenceError,
    InsufficientDataError,
    InvalidEvidenceError,
)

__all__ = [
    "Antiformant",
    "NasalConfig",
    "NasalFeatures",
    "detect_nasal_frames",
    "estimate_antiformants",
    "extract_nasal_features",
]


@dataclass(frozen=True, slots=True)
class NasalConfig:
    """Nasal analysis parameters.

    The frequency ranges are those reported for nasal murmurs in the phonetics
    literature. They are wide because the quantity being measured varies
    between speakers — which is the entire point of measuring it.
    """

    murmur_band_hz: tuple[float, float] = (200.0, 500.0)
    """Where the low nasal resonance sits. A murmur has most of its energy
    here."""

    antiformant_search_hz: tuple[float, float] = (700.0, 2500.0)
    """Where to look for the spectral null. Spans the ranges for /m/ and /n/,
    and stays clear of the codec band edges where the transition would be
    mistaken for a null."""

    min_low_frequency_ratio: float = 0.55
    """Minimum fraction of passband energy below 1 kHz for a frame to be a
    murmur candidate. Nasals are markedly low-pass relative to vowels."""

    min_null_depth_db: float = 6.0
    """Minimum depth of a spectral null, relative to the surrounding envelope,
    before it is accepted as an antiformant. Below this the null is
    indistinguishable from ordinary spectral variation, and reporting it would
    manufacture a feature."""

    relative_energy_floor_db: float = -25.0
    """How far below the recording's peak frame energy a candidate may sit.
    Nasals are quieter than adjacent vowels but not silent; this rejects
    low-level noise between words."""

    min_nasal_frames: int = 15
    """Minimum nasal frames required before features are reported, roughly
    150 ms of nasal murmur. Fewer than this and the per-speaker estimate is
    dominated by which particular tokens happened to be captured."""

    def __post_init__(self) -> None:
        for name, band in (
            ("murmur_band_hz", self.murmur_band_hz),
            ("antiformant_search_hz", self.antiformant_search_hz),
        ):
            if band[0] >= band[1]:
                raise InvalidEvidenceError(f"{name} must be an increasing range", band=band)


@dataclass(frozen=True, slots=True)
class Antiformant:
    """An estimated spectral zero."""

    frequency_hz: float
    depth_db: float
    """How far the null falls below the local spectral envelope. Serves as the
    confidence in the estimate: a deep, narrow null is unambiguous, a shallow
    one is not."""

    @property
    def is_confident(self) -> bool:
        return self.depth_db >= 10.0


@dataclass(frozen=True, slots=True)
class NasalFeatures:
    """Nasal-segment features for one recording."""

    n_nasal_frames: int
    nasal_fraction: float
    murmur_f1_median_hz: float
    """Median frequency of the low nasal resonance."""

    murmur_f1_iqr_hz: float
    antiformant_median_hz: float
    antiformant_iqr_hz: float
    antiformant_depth_median_db: float
    low_frequency_energy_ratio: float
    """Median fraction of passband energy below 1 kHz across nasal frames. A
    coarse but robust index of the degree of nasal coupling."""

    spectral_null_count: int
    """Number of nasal frames in which a confident antiformant was located.
    Together with ``n_nasal_frames`` this states how much of the feature
    actually survived, which a median alone conceals."""

    def as_features(self) -> dict[str, float]:
        return {
            "nasal_fraction": self.nasal_fraction,
            "nasal_murmur_f1_median_hz": self.murmur_f1_median_hz,
            "nasal_murmur_f1_iqr_hz": self.murmur_f1_iqr_hz,
            "nasal_antiformant_median_hz": self.antiformant_median_hz,
            "nasal_antiformant_iqr_hz": self.antiformant_iqr_hz,
            "nasal_antiformant_depth_db": self.antiformant_depth_median_db,
            "nasal_low_freq_ratio": self.low_frequency_energy_ratio,
            "nasal_null_coverage": (
                self.spectral_null_count / self.n_nasal_frames
                if self.n_nasal_frames
                else 0.0
            ),
        }


def detect_nasal_frames(
    spectra: NDArray[np.float64],
    sample_rate: int,
    config: NasalConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Identify frames whose spectral shape is consistent with a nasal murmur.

    This is a segment-type detector, not a phone recogniser. It selects frames
    with the acoustic signature of nasal coupling — energy concentrated low, a
    resonance in the murmur band, moderate level relative to the recording —
    without identifying which nasal consonant produced them. That is sufficient
    and appropriate here: the anatomical evidence is in the resonances, and a
    phone label would add a dependence on a recogniser trained on languages
    other than the ones being analysed.

    Returns the boolean mask and the per-frame low-frequency energy ratio.
    """
    bin_hz = np.linspace(0.0, sample_rate / 2.0, spectra.shape[1])

    passband = (bin_hz >= 200.0) & (bin_hz <= min(3400.0, sample_rate / 2.0))
    low_band = passband & (bin_hz <= 1000.0)

    passband_energy = spectra[:, passband].sum(axis=1)
    low_energy = spectra[:, low_band].sum(axis=1)
    ratio = np.divide(
        low_energy,
        passband_energy,
        out=np.zeros_like(low_energy),
        where=passband_energy > 0.0,
    )

    frame_db = 10.0 * np.log10(np.maximum(passband_energy, 1e-12))
    if frame_db.size == 0:
        return np.zeros(0, dtype=bool), ratio
    peak_db = float(np.percentile(frame_db, 95))

    murmur_band = (bin_hz >= config.murmur_band_hz[0]) & (
        bin_hz <= config.murmur_band_hz[1]
    )
    murmur_energy = spectra[:, murmur_band].sum(axis=1)
    murmur_ratio = np.divide(
        murmur_energy,
        passband_energy,
        out=np.zeros_like(murmur_energy),
        where=passband_energy > 0.0,
    )

    is_nasal = (
        (ratio >= config.min_low_frequency_ratio)
        & (murmur_ratio >= 0.25)
        & (frame_db >= peak_db + config.relative_energy_floor_db)
    )
    return is_nasal, ratio


def estimate_antiformants(
    spectrum: NDArray[np.float64],
    sample_rate: int,
    config: NasalConfig,
    order: int = 6,
) -> Antiformant | None:
    """Estimate the principal antiformant of one frame.

    Method: the zeros of a spectrum are the poles of its reciprocal. Taking the
    reciprocal of the power spectrum and fitting an all-pole model to *that*
    turns the zero-estimation problem into a pole-estimation problem, which
    Levinson-Durbin solves in closed form and stably.

    The estimate is then verified directly against the observed spectrum: the
    depth of the null at the estimated frequency, relative to the local envelope
    on either side, must exceed ``min_null_depth_db``. Without that check the
    method returns a frequency for every frame including those with no zero at
    all, since an all-pole fit to a smooth reciprocal spectrum still has poles.

    Returns ``None`` where no sufficiently deep null is found. That is a common
    and correct outcome, not a failure.
    """
    n_bins = spectrum.size
    bin_hz = np.linspace(0.0, sample_rate / 2.0, n_bins)
    low, high = config.antiformant_search_hz
    band = (bin_hz >= low) & (bin_hz <= min(high, sample_rate / 2.0 - 100.0))
    if np.count_nonzero(band) < 8:
        return None

    floor = float(np.maximum(spectrum.max(), 1e-20)) * 1e-8
    reciprocal = 1.0 / np.maximum(spectrum, floor)

    # Autocorrelation of the reciprocal spectrum, obtained by inverse transform.
    # The spectrum is one-sided and real, so its inverse transform is the
    # autocorrelation sequence of the corresponding time signal.
    correlation = np.fft.irfft(reciprocal)[: order + 1]
    if correlation.size < order + 1 or correlation[0] <= 0.0:
        return None
    try:
        coefficients, _, _ = levinson_durbin(correlation, order)
    except (ConvergenceError, InvalidEvidenceError):
        return None

    roots = np.roots(coefficients)
    roots = roots[(np.imag(roots) > 1e-9) & (np.abs(roots) < 1.0)]
    if roots.size == 0:
        return None

    candidates_hz = np.angle(roots) * sample_rate / (2.0 * np.pi)
    in_band = (candidates_hz >= low) & (candidates_hz <= high)
    candidates_hz = candidates_hz[in_band]
    if candidates_hz.size == 0:
        return None

    # Verify each candidate against the observed spectrum and keep the deepest.
    log_spectrum = 10.0 * np.log10(np.maximum(spectrum, floor))
    best: Antiformant | None = None
    for frequency in candidates_hz:
        depth = _null_depth_db(log_spectrum, bin_hz, float(frequency))
        if depth >= config.min_null_depth_db and (best is None or depth > best.depth_db):
            best = Antiformant(frequency_hz=float(frequency), depth_db=float(depth))
    return best


def _null_depth_db(
    log_spectrum: NDArray[np.float64], bin_hz: NDArray[np.float64], frequency: float
) -> float:
    """Depth of the spectral null at ``frequency`` below its local surroundings.

    Measured as the difference between the maximum of the log spectrum in
    flanking bands either side of the candidate and the minimum within the
    candidate's own narrow band. Using flanking maxima rather than a smoothed
    envelope avoids the circularity of comparing the spectrum with a smoothed
    version of itself, which fills the null being measured.
    """
    centre_band = np.abs(bin_hz - frequency) <= 60.0
    flank_low = (bin_hz >= frequency - 400.0) & (bin_hz <= frequency - 120.0)
    flank_high = (bin_hz >= frequency + 120.0) & (bin_hz <= frequency + 400.0)

    if not np.any(centre_band) or not (np.any(flank_low) or np.any(flank_high)):
        return 0.0

    flanks = np.concatenate([log_spectrum[flank_low], log_spectrum[flank_high]])
    if flanks.size == 0:
        return 0.0
    return float(np.max(flanks) - np.min(log_spectrum[centre_band]))


def extract_nasal_features(
    signal: NDArray[np.float64],
    frame_config: FrameConfig,
    config: NasalConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> NasalFeatures:
    """Extract nasal-segment features from a recording.

    Raises
    ------
    InsufficientDataError
        If too few nasal frames are found. Refusing is correct: nasal features
        computed from three frames measure which words the speaker happened to
        use, not the geometry of their sinuses.
    """
    config = config or NasalConfig()
    frames, _ = frame_signal(signal, frame_config, rng=rng)
    spectra = power_spectrum(frames, frame_config.n_fft)

    is_nasal, low_ratio = detect_nasal_frames(spectra, frame_config.sample_rate, config)
    nasal_indices = np.flatnonzero(is_nasal)

    if nasal_indices.size < config.min_nasal_frames:
        raise InsufficientDataError(
            "too few nasal segments for anatomical feature extraction",
            n_nasal_frames=int(nasal_indices.size),
            required=config.min_nasal_frames,
            n_frames=int(frames.shape[0]),
        )

    bin_hz = np.linspace(0.0, frame_config.sample_rate / 2.0, spectra.shape[1])
    murmur_band = (bin_hz >= config.murmur_band_hz[0]) & (
        bin_hz <= config.murmur_band_hz[1]
    )
    murmur_bins = np.flatnonzero(murmur_band)

    murmur_peaks: list[float] = []
    antiformant_hz: list[float] = []
    antiformant_depth: list[float] = []

    for index in nasal_indices:
        frame_spectrum = spectra[index]
        peak_bin = murmur_bins[int(np.argmax(frame_spectrum[murmur_band]))]
        murmur_peaks.append(float(bin_hz[peak_bin]))

        antiformant = estimate_antiformants(
            frame_spectrum, frame_config.sample_rate, config
        )
        if antiformant is not None:
            antiformant_hz.append(antiformant.frequency_hz)
            antiformant_depth.append(antiformant.depth_db)

    peaks = np.array(murmur_peaks, dtype=np.float64)
    zeros = np.array(antiformant_hz, dtype=np.float64)
    depths = np.array(antiformant_depth, dtype=np.float64)

    return NasalFeatures(
        n_nasal_frames=int(nasal_indices.size),
        nasal_fraction=float(nasal_indices.size / max(frames.shape[0], 1)),
        murmur_f1_median_hz=float(np.median(peaks)),
        murmur_f1_iqr_hz=float(np.percentile(peaks, 75) - np.percentile(peaks, 25)),
        antiformant_median_hz=float(np.median(zeros)) if zeros.size else float("nan"),
        antiformant_iqr_hz=(
            float(np.percentile(zeros, 75) - np.percentile(zeros, 25))
            if zeros.size >= 4
            else float("nan")
        ),
        antiformant_depth_median_db=(
            float(np.median(depths)) if depths.size else float("nan")
        ),
        low_frequency_energy_ratio=float(np.median(low_ratio[nasal_indices])),
        spectral_null_count=int(zeros.size),
    )
