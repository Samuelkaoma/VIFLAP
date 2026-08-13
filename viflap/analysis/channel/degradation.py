"""Additive noise, packet loss, and the controlled degradation sweep.

Two things here are done differently from the obvious approach, and in both
cases the obvious approach produces optimistic results.

**Noise is spectrally shaped, not white.** Real interference is not white.
Babble is speech, so it occupies the same band as the target and its long-term
spectrum follows the long-term average speech spectrum. Vehicle noise is
concentrated below 300 Hz, largely outside the telephony passband, and is
therefore far less harmful at the same nominal signal-to-noise ratio. Using
white noise for all of them makes the reported SNR incomparable between
conditions, and understates the damage babble does to exactly the mid-frequency
region where formants live.

**Packet loss is concealed, not zeroed.** No real network hands a decoder a
silent gap. Every telephony decoder implements packet loss concealment: it
extrapolates the missing frame from the last good one, typically by repeating a
pitch period with progressive attenuation, and mutes only after several
consecutive losses. Substituting silence produces sharp discontinuities at both
edges of every gap. Those are broadband transients that a spoofing
countermeasure will happily detect, so simulating loss by zeroing produces a
countermeasure that has learned to detect the simulation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from viflap.analysis.channel.codec import (
    AMR_NB_BITRATES,
    ChannelResult,
    Codec,
    resolve_codec,
)
from viflap.domain.errors import InvalidEvidenceError

__all__ = [
    "DegradationCondition",
    "DegradationSweep",
    "NoiseType",
    "add_shaped_noise",
    "apply_condition",
    "simulate_packet_loss",
]


class NoiseType(Enum):
    """Interference types encountered in mobile telephony recordings."""

    WHITE = "white"
    """Flat spectrum. Approximates thermal and quantisation noise. Included
    principally as a reference condition, since little real interference is
    actually white."""

    BABBLE = "babble"
    """Competing speech. The hardest case: it occupies the same band as the
    target and shares its modulation statistics, so no spectral filtering
    separates them. Common in the market and street settings where these calls
    are made and received."""

    VEHICLE = "vehicle"
    """Engine and road noise, concentrated below 300 Hz. Loud in absolute terms
    but mostly outside the telephony passband, so its effect at a given
    broadband SNR is milder than babble."""

    PINK = "pink"
    """``1/f`` spectrum. A reasonable model of general ambient noise."""


def _long_term_speech_spectrum_filter(n_taps: int = 129) -> NDArray[np.float64]:
    """FIR filter approximating the long-term average speech spectrum.

    The LTASS is roughly flat to 500 Hz and falls at about 9 dB per octave above
    it. Babble is a sum of speech signals and therefore inherits this shape;
    filtering white noise with it yields babble whose spectrum matches real
    babble far better than white noise does.

    Designed by frequency sampling, which is adequate for a smooth target
    response and avoids an optimisation dependency.
    """
    frequencies = np.linspace(0.0, 1.0, n_taps // 2 + 1)
    hz = frequencies * 4000.0
    response = np.ones_like(hz)
    above = hz > 500.0
    response[above] = 10.0 ** (-9.0 * np.log2(hz[above] / 500.0) / 20.0)
    response[hz < 80.0] = 0.05
    return scipy.signal.firwin2(n_taps, frequencies, response)


def _shape_noise(
    noise: NDArray[np.float64], noise_type: NoiseType, sample_rate: int
) -> NDArray[np.float64]:
    """Apply the spectral shape characteristic of ``noise_type``."""
    if noise_type is NoiseType.WHITE:
        return noise
    if noise_type is NoiseType.BABBLE:
        return np.convolve(noise, _long_term_speech_spectrum_filter(), mode="same")
    if noise_type is NoiseType.VEHICLE:
        nyquist = sample_rate / 2.0
        sos = scipy.signal.butter(4, 250.0 / nyquist, btype="low", output="sos")
        return scipy.signal.sosfilt(sos, noise)
    if noise_type is NoiseType.PINK:
        # Pink noise by spectral weighting: scale each bin by 1/sqrt(f), which
        # gives a power spectrum proportional to 1/f.
        spectrum = np.fft.rfft(noise)
        frequencies = np.fft.rfftfreq(noise.size, d=1.0 / sample_rate)
        weights = np.ones_like(frequencies)
        weights[1:] = 1.0 / np.sqrt(frequencies[1:])
        return np.fft.irfft(spectrum * weights, n=noise.size)
    raise InvalidEvidenceError("unknown noise type", noise_type=noise_type)


def add_shaped_noise(
    signal: NDArray[np.float64],
    noise_type: NoiseType,
    snr_db: float,
    sample_rate: int,
    *,
    rng: np.random.Generator | None = None,
    measure_in_passband: bool = True,
) -> NDArray[np.float64]:
    """Add spectrally shaped noise at a specified signal-to-noise ratio.

    Parameters
    ----------
    measure_in_passband:
        Measure both signal and noise power within 300-3400 Hz rather than
        broadband. This matters for vehicle noise, whose energy is mostly below
        the passband: a broadband SNR of 10 dB corresponds to a passband SNR
        perhaps 15 dB higher, so conditions labelled with the same SNR would not
        be comparable across noise types. Measuring in the band that survives
        the codec makes the label mean the same thing in every condition.
    """
    generator = rng if rng is not None else np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return signal

    noise = _shape_noise(generator.standard_normal(signal.size), noise_type, sample_rate)

    if measure_in_passband:
        signal_power = _passband_power(signal, sample_rate)
        noise_power = _passband_power(noise, sample_rate)
    else:
        signal_power = float(np.mean(signal**2))
        noise_power = float(np.mean(noise**2))

    if signal_power <= 0.0 or noise_power <= 0.0:
        return signal

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return signal + noise * np.sqrt(target_noise_power / noise_power)


def _passband_power(signal: NDArray[np.float64], sample_rate: int) -> float:
    """Mean power within the telephony passband."""
    nyquist = sample_rate / 2.0
    high = min(3400.0, nyquist * 0.99)
    if high <= 300.0:
        return float(np.mean(signal**2))
    sos = scipy.signal.butter(
        4, [300.0 / nyquist, high / nyquist], btype="band", output="sos"
    )
    return float(np.mean(scipy.signal.sosfiltfilt(sos, signal) ** 2))


def simulate_packet_loss(
    signal: NDArray[np.float64],
    sample_rate: int,
    loss_rate: float,
    *,
    frame_ms: float = 20.0,
    burstiness: float = 0.6,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Simulate frame loss with decoder-side concealment.

    Loss is generated by a two-state Gilbert-Elliott model rather than
    independent Bernoulli trials. Real radio loss is bursty — a fade lasts many
    frames — and independent losses of the same average rate produce many short
    gaps that concealment hides almost perfectly, understating the damage.
    ``burstiness`` sets the probability of remaining in the loss state.

    Concealment repeats the last pitch period of the preceding good frame with
    progressive attenuation, which is the standard approach (ITU-T G.711
    Appendix I and the AMR error concealment procedures). Output is muted after
    several consecutive losses, since extrapolating further produces an audible
    and measurable artificial buzz.
    """
    if not 0.0 <= loss_rate < 1.0:
        raise InvalidEvidenceError("loss rate must lie in [0, 1)", loss_rate=loss_rate)
    if not 0.0 <= burstiness < 1.0:
        raise InvalidEvidenceError("burstiness must lie in [0, 1)", burstiness=burstiness)

    generator = rng if rng is not None else np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float64)
    frame_length = int(round(frame_ms * sample_rate / 1000.0))
    if loss_rate == 0.0 or signal.size < 2 * frame_length:
        return signal.copy()

    n_frames = signal.size // frame_length
    # Gilbert-Elliott transition probabilities chosen so the stationary loss
    # probability equals loss_rate for the given persistence.
    stay_lost = burstiness
    enter_loss = loss_rate * (1.0 - stay_lost) / max(1.0 - loss_rate, 1e-9)
    enter_loss = float(np.clip(enter_loss, 0.0, 1.0))

    output = signal.copy()
    lost = False
    consecutive = 0

    for index in range(1, n_frames):
        transition = generator.random()
        lost = transition < (stay_lost if lost else enter_loss)
        if not lost:
            consecutive = 0
            continue

        consecutive += 1
        start = index * frame_length
        end = start + frame_length
        output[start:end] = _conceal_frame(
            output[:start], frame_length, consecutive, sample_rate
        )
    return output


def _conceal_frame(
    history: NDArray[np.float64],
    frame_length: int,
    consecutive_losses: int,
    sample_rate: int,
) -> NDArray[np.float64]:
    """Generate a concealment frame from the signal history.

    Estimates the pitch period of the preceding audio by autocorrelation, then
    tiles that period to fill the gap, attenuating by 6 dB per additional
    consecutive lost frame and muting after four — the behaviour of a real
    decoder, which stops extrapolating once the estimate has become unreliable.
    """
    if consecutive_losses > 4 or history.size < frame_length:
        return np.zeros(frame_length, dtype=np.float64)

    search = history[-min(history.size, int(0.030 * sample_rate)) :]
    min_lag = max(2, int(sample_rate / 400.0))
    max_lag = min(int(sample_rate / 50.0), search.size - 1)
    if max_lag <= min_lag:
        return np.zeros(frame_length, dtype=np.float64)

    centred = search - search.mean()
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1 :]
    energy = correlation[0]
    if energy <= 1e-12:
        return np.zeros(frame_length, dtype=np.float64)
    period = int(np.argmax(correlation[min_lag : max_lag + 1])) + min_lag

    template = history[-period:]
    repeats = int(np.ceil(frame_length / period))
    concealed = np.tile(template, repeats)[:frame_length]

    attenuation = 10.0 ** (-6.0 * (consecutive_losses - 1) / 20.0)
    # Taper across the frame so successive concealed frames decay smoothly
    # rather than stepping in level at each frame boundary.
    taper = np.linspace(1.0, 0.75, frame_length)
    return concealed * attenuation * taper


@dataclass(frozen=True, slots=True)
class DegradationCondition:
    """One point in the degradation design.

    A condition is a complete, labelled specification of the channel a recording
    passed through. It is stored with every experimental result, because a
    ``C_llr`` without its condition is not a weak claim, it is an
    uninterpretable one — the same system spans an order of magnitude across
    this grid.
    """

    bitrate_kbps: float
    noise_type: NoiseType | None = None
    snr_db: float | None = None
    packet_loss_rate: float = 0.0

    def __post_init__(self) -> None:
        if (self.noise_type is None) != (self.snr_db is None):
            raise InvalidEvidenceError(
                "noise type and SNR must be specified together",
                noise_type=self.noise_type,
                snr_db=self.snr_db,
            )

    @property
    def label(self) -> str:
        """Stable, sortable identifier for this condition."""
        parts = [f"amr{self.bitrate_kbps:g}"]
        if self.noise_type is not None and self.snr_db is not None:
            parts.append(f"{self.noise_type.value}{self.snr_db:g}dB")
        else:
            parts.append("clean")
        if self.packet_loss_rate > 0.0:
            parts.append(f"loss{self.packet_loss_rate * 100:g}pct")
        return "_".join(parts)

    def as_record(self) -> dict[str, float | str | None]:
        return {
            "bitrate_kbps": self.bitrate_kbps,
            "noise_type": self.noise_type.value if self.noise_type else None,
            "snr_db": self.snr_db,
            "packet_loss_rate": self.packet_loss_rate,
            "condition": self.label,
        }


def apply_condition(
    signal: NDArray[np.float64],
    sample_rate: int,
    condition: DegradationCondition,
    codec: Codec | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> ChannelResult:
    """Apply a full degradation condition to a signal.

    Order of operations matters and follows the physical signal path: noise is
    added at the microphone, *before* encoding, and packet loss occurs on the
    network, *after* it. Adding noise after the codec would produce a signal
    whose noise had never been through the coder, which is not a condition that
    can occur and which would leave the noise far more structured — and far
    easier to separate from the speech — than it really is.
    """
    generator = rng if rng is not None else np.random.default_rng()
    working = np.asarray(signal, dtype=np.float64)

    if condition.noise_type is not None and condition.snr_db is not None:
        working = add_shaped_noise(
            working,
            condition.noise_type,
            condition.snr_db,
            sample_rate,
            rng=generator,
        )

    active_codec = codec if codec is not None else resolve_codec()
    result = active_codec.apply(working, sample_rate, condition.bitrate_kbps)

    if condition.packet_loss_rate > 0.0:
        degraded = simulate_packet_loss(
            result.signal,
            result.sample_rate,
            condition.packet_loss_rate,
            rng=generator,
        )
        result = ChannelResult(
            signal=degraded,
            sample_rate=result.sample_rate,
            bitrate_kbps=result.bitrate_kbps,
            mode=result.mode,
            lpc_order=result.lpc_order,
            n_codebook_pulses=result.n_codebook_pulses,
        )
    return result


@dataclass(frozen=True, slots=True)
class DegradationSweep:
    """A factorial design over channel conditions.

    This is the experimental apparatus for hypothesis H1 — what
    speaker-discriminative information survives the African mobile telephony
    channel. Enumerating the grid as an object rather than as nested loops in a
    script means the design is inspectable, reproducible, and recorded with its
    results.
    """

    bitrates: Sequence[float] = field(default_factory=lambda: list(AMR_NB_BITRATES))
    noise_types: Sequence[NoiseType] = field(
        default_factory=lambda: [NoiseType.BABBLE, NoiseType.VEHICLE]
    )
    snrs_db: Sequence[float] = field(default_factory=lambda: [20.0, 15.0, 10.0, 5.0])
    packet_loss_rates: Sequence[float] = field(default_factory=lambda: [0.0, 0.03, 0.10])
    include_clean: bool = True

    def conditions(self) -> Iterator[DegradationCondition]:
        """Enumerate every cell of the design."""
        for bitrate in self.bitrates:
            for loss in self.packet_loss_rates:
                if self.include_clean:
                    yield DegradationCondition(bitrate_kbps=bitrate, packet_loss_rate=loss)
                for noise_type in self.noise_types:
                    for snr in self.snrs_db:
                        yield DegradationCondition(
                            bitrate_kbps=bitrate,
                            noise_type=noise_type,
                            snr_db=snr,
                            packet_loss_rate=loss,
                        )

    @property
    def size(self) -> int:
        return sum(1 for _ in self.conditions())

    @classmethod
    def compact(cls) -> DegradationSweep:
        """A reduced design for development and continuous integration.

        The full sweep is several hundred conditions; running it on every commit
        would be pointless. This retains the extremes of each factor, which is
        what a regression test needs.
        """
        return cls(
            bitrates=[4.75, 12.20],
            noise_types=[NoiseType.BABBLE],
            snrs_db=[20.0, 5.0],
            packet_loss_rates=[0.0, 0.10],
        )
