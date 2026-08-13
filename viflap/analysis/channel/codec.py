"""Narrowband channel simulation.

The evidence this system reasons about has passed through a mobile speech codec
before anyone analyses it. Characterising what survives that passage is
hypothesis H1, and it cannot be characterised without the ability to apply the
degradation under controlled conditions.

Two implementations, with a clear preference between them
---------------------------------------------------------
:class:`FfmpegAmrCodec` invokes a real AMR-NB encoder and decoder. When ffmpeg
with ``libopencore-amrnb`` is available this is what runs, because it is the
actual coder and no model of it is as good as it is.

:class:`ParametricCelpCodec` is the fallback, and it is a genuine
analysis-by-synthesis CELP model rather than a filter and some noise. It
performs, per 20 ms frame: linear predictive analysis, conversion to line
spectral frequencies, quantisation of those frequencies at a rate-dependent
resolution, open-loop pitch analysis, adaptive-codebook (long-term predictor)
reconstruction, and algebraic fixed-codebook excitation with a rate-dependent
number of non-zero pulses. That is the structure of ACELP, and it reproduces the
*character* of the degradation — a preserved spectral envelope, a
resynthesised excitation, fine glottal structure replaced by codebook pulses —
which is exactly the degradation that matters for whether jitter and shimmer
survive.

The distinction between the two is recorded in the output. A result derived from
parametrically simulated degradation is not a result derived from AMR, and the
proposal's own limitation 4 ("simulated channel degradation is not real
degradation") applies to it. :attr:`ChannelResult.is_reference_codec` carries
that fact forward rather than leaving it to a footnote.

What the previous approach got wrong
------------------------------------
Bandpass filtering and adding white noise at a bitrate-dependent SNR is not a
model of a codec. A codec is not a noisy channel: it is a parametric analyser
and resynthesiser. Additive noise leaves the fine temporal structure of the
excitation intact, merely buried, so jitter and shimmer measured through it
survive far better than they do in reality. Any conclusion about glottal source
features drawn through such a simulation would be optimistic, and optimistic in
the direction that overstates the evidence.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from viflap.analysis.dsp.lpc import autocorrelation, levinson_durbin
from viflap.domain.errors import ConfigurationError, ConvergenceError, InvalidEvidenceError

__all__ = [
    "AMR_NB_BITRATES",
    "ChannelResult",
    "Codec",
    "CodecMode",
    "FfmpegAmrCodec",
    "ParametricCelpCodec",
    "resolve_codec",
]

#: The eight AMR narrowband modes, in kbit/s (3GPP TS 26.071).
AMR_NB_BITRATES: Final[tuple[float, ...]] = (
    4.75,
    5.15,
    5.90,
    6.70,
    7.40,
    7.95,
    10.20,
    12.20,
)

NARROWBAND_RATE: Final[int] = 8000


class CodecMode(Enum):
    """Which implementation produced a degraded signal."""

    REFERENCE_AMR = "reference_amr"
    """A real AMR-NB encoder and decoder."""

    PARAMETRIC_CELP = "parametric_celp"
    """The analysis-by-synthesis model in this module."""


@dataclass(frozen=True, slots=True)
class ChannelResult:
    """A degraded signal and the conditions that produced it."""

    signal: NDArray[np.float64]
    sample_rate: int
    bitrate_kbps: float
    mode: CodecMode
    lpc_order: int
    n_codebook_pulses: int

    @property
    def is_reference_codec(self) -> bool:
        """Whether a real codec produced this, or a model of one.

        Carried into every experimental record. A ``C_llr`` obtained under
        parametric simulation establishes a trend; it does not establish
        operational performance, and the two must not be reported as though
        they were the same quantity.
        """
        return self.mode is CodecMode.REFERENCE_AMR

    def provenance(self) -> dict[str, float | str | bool]:
        return {
            "codec_mode": self.mode.value,
            "bitrate_kbps": self.bitrate_kbps,
            "sample_rate": self.sample_rate,
            "is_reference_codec": self.is_reference_codec,
        }


class Codec(Protocol):
    """A speech codec that can be applied to a signal."""

    @property
    def is_available(self) -> bool:
        """Whether this implementation can run in the current environment."""
        ...

    def apply(
        self, signal: NDArray[np.float64], sample_rate: int, bitrate_kbps: float
    ) -> ChannelResult:
        """Encode and decode ``signal``, returning the degraded result."""
        ...


def _validate_bitrate(bitrate_kbps: float) -> float:
    """Snap to the nearest defined AMR mode, rejecting anything far from one."""
    nearest = min(AMR_NB_BITRATES, key=lambda mode: abs(mode - bitrate_kbps))
    if abs(nearest - bitrate_kbps) > 0.01:
        raise InvalidEvidenceError(
            "bitrate is not an AMR narrowband mode; the codec operates only at "
            "the eight defined rates and interpolating between them would model "
            "a codec that does not exist",
            requested=bitrate_kbps,
            available=list(AMR_NB_BITRATES),
        )
    return nearest


def _resample_to_narrowband(
    signal: NDArray[np.float64], sample_rate: int
) -> NDArray[np.float64]:
    """Resample to 8 kHz using a polyphase filter.

    ``scipy.signal.resample_poly`` rather than ``resample``: the latter is an
    FFT method that assumes the signal is periodic, which produces wrap-around
    artefacts at both ends of a speech recording. Those artefacts are loud,
    broadband, and would be analysed as speech.
    """
    if sample_rate == NARROWBAND_RATE:
        return np.asarray(signal, dtype=np.float64)
    from math import gcd

    divisor = gcd(int(sample_rate), NARROWBAND_RATE)
    return scipy.signal.resample_poly(
        np.asarray(signal, dtype=np.float64),
        NARROWBAND_RATE // divisor,
        int(sample_rate) // divisor,
    )


def _apply_passband(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the 300-3400 Hz telephony passband.

    A fourth-order Butterworth applied forwards and backwards, giving eighth-order
    magnitude response with zero phase distortion. Zero phase matters here:
    a causal filter's group delay varies across the band and would smear the
    temporal structure that the articulatory-timing features measure, adding a
    degradation the real channel does not impose.
    """
    nyquist = NARROWBAND_RATE / 2.0
    sos = scipy.signal.butter(
        4, [300.0 / nyquist, 3400.0 / nyquist], btype="band", output="sos"
    )
    return scipy.signal.sosfiltfilt(sos, signal)


class ParametricCelpCodec:
    """Analysis-by-synthesis model of an ACELP narrowband coder.

    Structure, per 20 ms frame, mirroring AMR-NB:

    1. **Short-term prediction.** Order-10 LPC over the frame.
    2. **Spectral quantisation.** Coefficients converted to line spectral
       frequencies, quantised on a uniform grid whose resolution follows the
       bitrate, and converted back. LSFs rather than raw coefficients because
       quantising the latter can produce an unstable filter, whereas LSFs remain
       stable under quantisation provided their ordering is preserved — which is
       precisely why real coders use them.
    3. **Long-term prediction.** Open-loop pitch search over the residual, giving
       an adaptive-codebook contribution: the previous excitation delayed by the
       pitch lag and scaled.
    4. **Fixed codebook.** The remaining excitation is approximated by a small
       number of unit pulses at the positions of its largest peaks — the
       algebraic codebook structure. The pulse count is the principal
       rate-dependent parameter, from 2 at 4.75 kbit/s to 10 at 12.2 kbit/s.
    5. **Synthesis.** The reconstructed excitation is passed through the
       quantised synthesis filter.

    The consequence that matters: the output waveform is *built*, not
    transmitted. Its fine structure comes from a handful of pulses per subframe,
    so cycle-to-cycle amplitude and period variation in the output is largely an
    artefact of pulse placement. That is the mechanism by which real coders
    destroy jitter and shimmer, and it is reproduced here rather than
    approximated by adding noise.
    """

    #: Non-zero excitation pulses per 5 ms subframe, by mode. Follows the
    #: algebraic codebook sizes of the corresponding AMR modes.
    PULSES_BY_BITRATE: Final[dict[float, int]] = {
        4.75: 2,
        5.15: 2,
        5.90: 2,
        6.70: 3,
        7.40: 4,
        7.95: 4,
        10.20: 8,
        12.20: 10,
    }

    #: Line spectral frequency quantiser resolution in hertz, by mode. Coarser
    #: quantisation at lower rates smooths the spectral envelope, which is what
    #: fills in narrow spectral nulls and degrades antiformant estimation.
    LSF_STEP_HZ_BY_BITRATE: Final[dict[float, float]] = {
        4.75: 60.0,
        5.15: 55.0,
        5.90: 48.0,
        6.70: 40.0,
        7.40: 34.0,
        7.95: 30.0,
        10.20: 22.0,
        12.20: 15.0,
    }

    def __init__(
        self,
        lpc_order: int = 10,
        frame_ms: float = 20.0,
        subframe_ms: float = 5.0,
        seed: int = 0,
    ) -> None:
        self.lpc_order = lpc_order
        self.frame_ms = frame_ms
        self.subframe_ms = subframe_ms
        self._rng = np.random.default_rng(seed)

    @property
    def is_available(self) -> bool:
        return True

    def apply(
        self, signal: NDArray[np.float64], sample_rate: int, bitrate_kbps: float
    ) -> ChannelResult:
        bitrate = _validate_bitrate(bitrate_kbps)
        narrowband = _resample_to_narrowband(signal, sample_rate)
        narrowband = _apply_passband(narrowband)

        frame_length = int(round(self.frame_ms * NARROWBAND_RATE / 1000.0))
        subframe_length = int(round(self.subframe_ms * NARROWBAND_RATE / 1000.0))
        n_pulses = self.PULSES_BY_BITRATE[bitrate]
        lsf_step_hz = self.LSF_STEP_HZ_BY_BITRATE[bitrate]

        if narrowband.size < frame_length:
            return ChannelResult(
                signal=narrowband,
                sample_rate=NARROWBAND_RATE,
                bitrate_kbps=bitrate,
                mode=CodecMode.PARAMETRIC_CELP,
                lpc_order=self.lpc_order,
                n_codebook_pulses=n_pulses,
            )

        n_frames = narrowband.size // frame_length
        output = np.zeros(n_frames * frame_length, dtype=np.float64)

        # Persistent state across frames: the synthesis filter memory and the
        # excitation history the adaptive codebook indexes into. Resetting these
        # per frame would introduce a discontinuity at every frame boundary that
        # the real coder does not have.
        synthesis_memory = np.zeros(self.lpc_order, dtype=np.float64)
        max_lag = int(0.020 * NARROWBAND_RATE)
        excitation_history = np.zeros(max_lag + frame_length, dtype=np.float64)

        for frame_index in range(n_frames):
            start = frame_index * frame_length
            frame = narrowband[start : start + frame_length]

            coefficients = self._quantised_lpc(frame, lsf_step_hz)
            residual = scipy.signal.lfilter(coefficients, [1.0], frame)

            excitation = np.zeros(frame_length, dtype=np.float64)
            for sub_start in range(0, frame_length, subframe_length):
                sub_end = min(sub_start + subframe_length, frame_length)
                target = residual[sub_start:sub_end]
                if target.size == 0:
                    continue

                adaptive = self._adaptive_codebook(excitation_history, target, max_lag)
                innovation = self._algebraic_codebook(target - adaptive, n_pulses)
                subframe_excitation = adaptive + innovation

                excitation[sub_start:sub_end] = subframe_excitation
                excitation_history = np.concatenate(
                    [excitation_history[subframe_excitation.size :], subframe_excitation]
                )

            synthesised, synthesis_memory = scipy.signal.lfilter(
                [1.0], coefficients, excitation, zi=synthesis_memory
            )
            output[start : start + frame_length] = synthesised

        # Guard against the synthesis filter ringing on a marginally stable
        # quantised envelope, which would otherwise produce a signal orders of
        # magnitude louder than the input.
        peak = float(np.max(np.abs(output))) if output.size else 0.0
        input_peak = float(np.max(np.abs(narrowband)))
        if peak > 0.0 and input_peak > 0.0 and peak > 4.0 * input_peak:
            output *= input_peak / peak

        return ChannelResult(
            signal=output,
            sample_rate=NARROWBAND_RATE,
            bitrate_kbps=bitrate,
            mode=CodecMode.PARAMETRIC_CELP,
            lpc_order=self.lpc_order,
            n_codebook_pulses=n_pulses,
        )

    def _quantised_lpc(
        self, frame: NDArray[np.float64], step_hz: float
    ) -> NDArray[np.float64]:
        """LPC analysis followed by LSF quantisation and reconstruction."""
        try:
            r = autocorrelation(frame * np.hamming(frame.size), self.lpc_order)
            # Lag windowing and a white-noise floor, both standard in real
            # coders: they condition the autocorrelation so that the recursion
            # does not produce a filter with poles arbitrarily close to the unit
            # circle, which would ring audibly on quantisation.
            lag_window = np.exp(
                -0.5
                * (2.0 * np.pi * 60.0 * np.arange(self.lpc_order + 1) / NARROWBAND_RATE)
                ** 2
            )
            r = r * lag_window
            r[0] *= 1.0001
            coefficients, _, _ = levinson_durbin(r, self.lpc_order)
        except (ConvergenceError, InvalidEvidenceError):
            return np.concatenate([[1.0], np.zeros(self.lpc_order)])

        lsf = self._lpc_to_lsf(coefficients)
        if lsf is None:
            return coefficients

        step = step_hz * 2.0 * np.pi / NARROWBAND_RATE
        quantised = np.round(lsf / step) * step
        # Enforce ordering and minimum separation. An LSF vector that is not
        # strictly increasing corresponds to an unstable filter; real coders
        # apply exactly this repair after quantisation.
        min_gap = 2.0 * np.pi * 50.0 / NARROWBAND_RATE
        for index in range(1, quantised.size):
            if quantised[index] - quantised[index - 1] < min_gap:
                quantised[index] = quantised[index - 1] + min_gap
        quantised = np.clip(quantised, min_gap, np.pi - min_gap)

        reconstructed = self._lsf_to_lpc(quantised)
        return reconstructed if reconstructed is not None else coefficients

    def _lpc_to_lsf(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Convert prediction coefficients to line spectral frequencies.

        The LSFs are the angles of the roots of the sum and difference
        polynomials ``P(z) = A(z) + z^-(p+1) A(z^-1)`` and
        ``Q(z) = A(z) - z^-(p+1) A(z^-1)``. Their roots lie on the unit circle
        and interlace whenever ``A(z)`` is minimum-phase, which is what makes
        the representation robust to quantisation.
        """
        order = coefficients.size - 1
        extended = np.concatenate([coefficients, [0.0]])
        reversed_extended = extended[::-1]
        p_poly = extended + reversed_extended
        q_poly = extended - reversed_extended

        angles: list[float] = []
        for polynomial in (p_poly, q_poly):
            roots = np.roots(polynomial)
            on_circle = roots[np.abs(np.abs(roots) - 1.0) < 1e-2]
            positive = np.angle(on_circle)
            angles.extend(float(a) for a in positive if 1e-6 < a < np.pi - 1e-6)

        if len(angles) < order:
            return None
        return np.sort(np.array(angles[:order], dtype=np.float64))

    def _lsf_to_lpc(self, lsf: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Reconstruct prediction coefficients from line spectral frequencies."""
        order = lsf.size
        if order % 2 != 0:
            return None
        # Odd-indexed LSFs belong to P(z), even-indexed to Q(z).
        p_roots = np.exp(1j * lsf[0::2])
        q_roots = np.exp(1j * lsf[1::2])
        p_poly = np.real(np.poly(np.concatenate([p_roots, np.conj(p_roots)])))
        q_poly = np.real(np.poly(np.concatenate([q_roots, np.conj(q_roots)])))

        p_extended = np.convolve(p_poly, [1.0, 1.0])
        q_extended = np.convolve(q_poly, [1.0, -1.0])
        length = max(p_extended.size, q_extended.size)
        p_extended = np.pad(p_extended, (0, length - p_extended.size))
        q_extended = np.pad(q_extended, (0, length - q_extended.size))

        coefficients = 0.5 * (p_extended + q_extended)[: order + 1]
        if coefficients.size != order + 1 or not np.isfinite(coefficients).all():
            return None
        coefficients = coefficients / coefficients[0]

        # Reject a reconstruction that is not minimum-phase; the caller falls
        # back to the unquantised coefficients rather than synthesising through
        # an unstable filter.
        if np.any(np.abs(np.roots(coefficients)) >= 1.0):
            return None
        return coefficients

    def _adaptive_codebook(
        self,
        history: NDArray[np.float64],
        target: NDArray[np.float64],
        max_lag: int,
    ) -> NDArray[np.float64]:
        """Long-term prediction: the past excitation at the best pitch lag.

        Searches lags from 2.5 ms to 20 ms (400 Hz down to 50 Hz) for the delay
        maximising normalised correlation with the target, then scales by the
        least-squares gain. This is the component that carries voicing
        periodicity through the coder, and its quantisation to a single gain and
        lag per subframe is one reason cycle-to-cycle period variation does not
        survive.
        """
        min_lag = max(2, int(0.0025 * NARROWBAND_RATE))
        length = target.size
        highest_lag = min(max_lag, history.size)
        if highest_lag < min_lag or length == 0:
            return np.zeros(length, dtype=np.float64)

        # The search is over every lag at once rather than one at a time. Each
        # candidate is a window of the excitation history, and successive lags
        # are the same window shifted by one sample, so the whole set is a
        # strided view of one padded array — no copying, and the scores become a
        # single matrix-vector product.
        #
        # Padding the history with `length` zeros reproduces exactly what the
        # per-lag form did when a short lag ran past the end of the history: the
        # candidate was zero-filled to the target's length.
        padded = np.concatenate([history, np.zeros(length, dtype=np.float64)])
        windows = np.lib.stride_tricks.sliding_window_view(padded, length)

        lags = np.arange(min_lag, highest_lag + 1)
        candidates = windows[history.size - lags]

        energies = np.einsum("ij,ij->i", candidates, candidates)
        correlations = candidates @ target
        # Lags whose candidate carries no energy are excluded rather than scored:
        # the normalised correlation is undefined there, and admitting them would
        # let an all-zero window win on a division by an epsilon.
        usable = energies > 1e-12
        scores = np.full(lags.size, -np.inf, dtype=np.float64)
        scores[usable] = correlations[usable] ** 2 / energies[usable]

        # argmax returns the first maximum, so ties resolve to the shortest lag,
        # as the ascending loop did. With every score at -inf — a silent history —
        # this selects min_lag, and the zero-energy guard below then returns
        # silence, which is the same outcome.
        best = int(np.argmax(scores))
        candidate = candidates[best]
        energy = float(energies[best])
        if energy <= 1e-12:
            return np.zeros(length, dtype=np.float64)
        gain = float(np.dot(candidate, target)) / energy
        # Real coders bound the adaptive gain to prevent runaway periodicity.
        gain = float(np.clip(gain, 0.0, 1.2))
        return gain * candidate

    def _algebraic_codebook(
        self, target: NDArray[np.float64], n_pulses: int
    ) -> NDArray[np.float64]:
        """Fixed codebook: approximate the target with ``n_pulses`` unit pulses.

        Positions are the ``n_pulses`` largest-magnitude samples of the target,
        with signs taken from it and a single gain fitted by least squares. This
        is the essential behaviour of an algebraic codebook: whatever the
        residual actually looked like, what gets transmitted is a handful of
        signed pulses and one gain.

        The information-theoretic consequence is the point of the whole module.
        At 4.75 kbit/s a 5 ms subframe is represented by two pulses. The
        cycle-to-cycle amplitude variation of the reconstructed signal is
        determined by where those two pulses landed, not by the speaker's
        vocal folds.
        """
        if target.size == 0 or n_pulses <= 0:
            return np.zeros_like(target)

        pulses = np.zeros_like(target)
        count = min(n_pulses, target.size)
        positions = np.argpartition(np.abs(target), -count)[-count:]
        pulses[positions] = np.sign(target[positions])

        energy = float(np.dot(pulses, pulses))
        if energy <= 0.0:
            return pulses
        gain = float(np.dot(pulses, target)) / energy
        return gain * pulses


class FfmpegAmrCodec:
    """Real AMR-NB encode/decode via ffmpeg.

    Preferred whenever available. The parametric model reproduces the character
    of the degradation; this reproduces the degradation.
    """

    def __init__(self, ffmpeg_path: str | None = None) -> None:
        self._ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
        self._encoder: str | None = None
        if self._ffmpeg is not None:
            self._encoder = self._detect_encoder()

    def _detect_encoder(self) -> str | None:
        """Find an available AMR-NB encoder.

        ffmpeg builds vary in which external codec libraries they were compiled
        against, and AMR is frequently omitted for licensing reasons. Probing
        once at construction is cheaper and clearer than discovering it on the
        first recording of a large batch.
        """
        assert self._ffmpeg is not None
        try:
            completed = subprocess.run(
                [self._ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for candidate in ("libopencore_amrnb", "amr_nb"):
            if candidate in completed.stdout:
                return candidate
        return None

    @property
    def is_available(self) -> bool:
        return self._ffmpeg is not None and self._encoder is not None

    def apply(
        self, signal: NDArray[np.float64], sample_rate: int, bitrate_kbps: float
    ) -> ChannelResult:
        if not self.is_available:
            raise ConfigurationError(
                "no AMR-NB encoder is available in this ffmpeg build",
                ffmpeg=self._ffmpeg,
            )
        bitrate = _validate_bitrate(bitrate_kbps)
        narrowband = _resample_to_narrowband(signal, sample_rate)

        with tempfile.TemporaryDirectory(prefix="viflap-codec-") as directory:
            workspace = Path(directory)
            source = workspace / "source.wav"
            coded = workspace / "coded.amr"
            decoded = workspace / "decoded.wav"

            _write_wav(source, narrowband, NARROWBAND_RATE)
            self._run(
                [
                    "-i",
                    str(source),
                    "-ar",
                    "8000",
                    "-ac",
                    "1",
                    "-ab",
                    f"{int(bitrate * 1000)}",
                    "-c:a",
                    str(self._encoder),
                    str(coded),
                ]
            )
            self._run(["-i", str(coded), "-ar", "8000", "-ac", "1", str(decoded)])
            output = _read_wav(decoded, expected_rate=NARROWBAND_RATE)

        return ChannelResult(
            signal=output,
            sample_rate=NARROWBAND_RATE,
            bitrate_kbps=bitrate,
            mode=CodecMode.REFERENCE_AMR,
            lpc_order=10,
            n_codebook_pulses=ParametricCelpCodec.PULSES_BY_BITRATE[bitrate],
        )

    def _run(self, arguments: list[str]) -> None:
        assert self._ffmpeg is not None
        completed = subprocess.run(
            [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *arguments],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            raise ConfigurationError(
                "AMR codec invocation failed",
                returncode=completed.returncode,
                stderr=completed.stderr.strip()[:500],
            )


def resolve_codec(prefer_reference: bool = True, seed: int = 0) -> Codec:
    """Return the best available codec implementation.

    Prefers the real encoder and falls back to the parametric model. The caller
    does not choose blindly: :attr:`ChannelResult.mode` on every output records
    which was used, so an experiment run partly on one and partly on the other
    is detectable rather than silently mixed.
    """
    if prefer_reference:
        reference = FfmpegAmrCodec()
        if reference.is_available:
            return reference
    return ParametricCelpCodec(seed=seed)


def _write_wav(path: Path, signal: NDArray[np.float64], sample_rate: int) -> None:
    """Write 16-bit PCM.

    Scaling is by the peak with 1 dB of headroom rather than by a fixed factor:
    a fixed factor clips loud recordings, and clipping introduces broadband
    distortion that the codec then encodes faithfully, contaminating the
    measurement of the codec's own effect.
    """
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    scale = (0.891 / peak) if peak > 0.0 else 1.0
    samples = np.clip(signal * scale * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def _read_wav(path: Path, expected_rate: int) -> NDArray[np.float64]:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != expected_rate:
            raise ConfigurationError(
                "decoded audio has an unexpected sample rate",
                expected=expected_rate,
                actual=handle.getframerate(),
            )
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
