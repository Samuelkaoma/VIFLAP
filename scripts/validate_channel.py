"""How far is the parametric channel model from a real AMR-NB coder?

§16 of the results document records that every absolute figure in this project
passed through `ParametricCelpCodec` rather than a reference coder, because
ffmpeg with `libopencore-amrnb` was not available on the machine the experiments
ran on. It also records the measurement that has never been made: run both
coders over the same recordings and report the difference.

This script is that measurement. It needs no model, no corpus split and no
training — only audio and an ffmpeg build that carries an AMR-NB encoder — so it
can run anywhere one exists, which is the point. On a machine without one it
still runs, reports `available: false`, and measures nothing, rather than
quietly measuring the parametric model against itself and reporting the result
as a validation.

What is measured, and against what
----------------------------------
**Log-spectral distance between LPC envelopes**, the standard figure of merit for
spectral quantisation. The comparison is against the source *resampled to the
same 8 kHz band*, never against the original wideband signal: telephony
bandwidth reduction is a property of the channel being modelled rather than a
coding error, and counting it would inflate every figure by the same large
constant and make the coders look more alike than they are.

**Segmental SNR**, frame by frame over speech-active frames, which measures the
waveform rather than the envelope. The two disagree informatively: an
analysis-by-synthesis coder can track the envelope closely while replacing the
excitation entirely, and §16's finding is that this model's distortion comes from
the excitation rather than from the LSF quantiser that its bitrate label names.

**Coder against coder**, directly. This is the quantity §16 asks for and the one
that scopes the document: if the parametric model is harsher than the reference,
every absolute figure in §§4-17 is pessimistic, and by a stateable amount.

The alignment problem, which is not optional
--------------------------------------------
A codec need not return a signal the same length as its input, and AMR's
encoder introduces a look-ahead delay. Comparing frame *i* of the output against
frame *i* of the input under a delay of even one frame produces a
distance dominated by the misalignment. The delay is therefore estimated by
cross-correlation per recording and removed before anything is measured, and the
estimate is reported so a suspiciously large one is visible rather than absorbed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.channel.codec import (
    NARROWBAND_RATE,
    Codec,
    CodecMode,
    FfmpegAmrCodec,
    ParametricCelpCodec,
    _resample_to_narrowband,
)
from viflap.analysis.dsp.framing import FrameConfig, frame_signal
from viflap.analysis.dsp.lpc import lpc_analysis, lpc_spectrum, recommended_order

#: Frames below this fraction of the recording's peak frame energy are excluded.
#: Silence contributes a distance dominated by whatever the two coders do with
#: comfort noise, which is not what either figure is meant to describe, and it
#: would let a recording's silence fraction move its reported distortion.
SPEECH_FLOOR_FRACTION = 1e-4

#: Envelopes are compared on this many frequency bins. 256 gives about 15 Hz of
#: resolution at 8 kHz, finer than any formant bandwidth, so the figure is not
#: limited by the grid.
N_FFT = 512


@dataclass(slots=True)
class PairwiseDistortion:
    """One comparison between two versions of the same recordings."""

    label: str
    reference: str
    subject: str
    bitrate_kbps: float | None

    n_recordings: int
    n_frames: int

    mean_spectral_distortion_db: float
    median_spectral_distortion_db: float
    fraction_frames_above_2db: float

    mean_segmental_snr_db: float
    median_segmental_snr_db: float

    mean_delay_samples: float
    max_delay_samples: int


@dataclass(slots=True)
class ValidationReport:
    """Everything the run established, including that it could not run."""

    available: bool
    reference_mode: str | None
    parametric_mode: str
    ffmpeg_encoder: str | None
    n_recordings: int
    bitrates_kbps: list[float]
    sample_rate: int
    elapsed_seconds: float
    notes: list[str] = field(default_factory=list)
    comparisons: list[PairwiseDistortion] = field(default_factory=list)


def _speech_frames(frames: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Frames carrying enough energy to be worth comparing."""
    energy = np.sum(frames**2, axis=1)
    peak = float(np.max(energy)) if energy.size else 0.0
    if peak <= 0.0:
        return np.zeros(frames.shape[0], dtype=bool)
    return energy > peak * SPEECH_FLOOR_FRACTION


def estimate_delay(reference: NDArray[np.float64], subject: NDArray[np.float64]) -> int:
    """Samples by which ``subject`` lags ``reference``, by cross-correlation.

    Restricted to non-negative lags: a codec delays its output, it does not
    anticipate its input, and allowing negative lags lets noise in a
    near-symmetric correlation pick a physically impossible alignment.
    """
    length = min(reference.size, subject.size)
    if length < 64:
        return 0
    a = reference[:length] - reference[:length].mean()
    b = subject[:length] - subject[:length].mean()
    # Search a generous window: AMR's look-ahead is a few milliseconds, but a
    # resampler's group delay can add more, and an underestimate here is far
    # more damaging than a wide search is slow.
    window = min(length - 1, NARROWBAND_RATE // 10)
    correlation = np.correlate(b, a[: length - window], mode="valid")[: window + 1]
    return int(np.argmax(correlation)) if correlation.size else 0


def compare(
    label: str,
    reference_signals: Sequence[NDArray[np.float64]],
    subject_signals: Sequence[NDArray[np.float64]],
    reference_name: str,
    subject_name: str,
    bitrate_kbps: float | None,
) -> PairwiseDistortion:
    """Log-spectral distance and segmental SNR over a set of recording pairs.

    Both figures are pooled over frames rather than averaged over recordings, so
    a long recording contributes proportionally. Averaging per recording first
    would weight a five-second file equally with a thirty-second one, and the
    frame is the unit the distortion is a property of.
    """
    config = FrameConfig(sample_rate=NARROWBAND_RATE)
    order = recommended_order(NARROWBAND_RATE)

    distortions: list[NDArray[np.float64]] = []
    snrs: list[NDArray[np.float64]] = []
    delays: list[int] = []

    for source, coded in zip(reference_signals, subject_signals, strict=True):
        delay = estimate_delay(source, coded)
        delays.append(delay)
        aligned = coded[delay:]
        length = min(source.size, aligned.size)
        if length < config.frame_length * 2:
            continue
        a, b = source[:length], aligned[:length]

        frames_a, _ = frame_signal(a, config)
        frames_b, _ = frame_signal(b, config)
        n_frames = min(frames_a.shape[0], frames_b.shape[0])
        frames_a, frames_b = frames_a[:n_frames], frames_b[:n_frames]

        keep = _speech_frames(frames_a) & _speech_frames(frames_b)
        if not keep.any():
            continue
        frames_a, frames_b = frames_a[keep], frames_b[keep]

        coefficients_a, error_a = lpc_analysis(frames_a, order)
        coefficients_b, error_b = lpc_analysis(frames_b, order)
        envelope_a = lpc_spectrum(coefficients_a, error_a, N_FFT)
        envelope_b = lpc_spectrum(coefficients_b, error_b, N_FFT)

        # The conventional definition: root-mean-square difference of the log
        # envelopes in decibels, per frame. 10*log10 of a power spectrum, so the
        # 10 rather than 20 is correct and not a slip.
        log_a = 10.0 * np.log10(np.maximum(envelope_a, 1e-20))
        log_b = 10.0 * np.log10(np.maximum(envelope_b, 1e-20))
        distortions.append(np.sqrt(np.mean((log_a - log_b) ** 2, axis=1)))

        signal_power = np.sum(frames_a**2, axis=1)
        noise_power = np.sum((frames_a - frames_b) ** 2, axis=1)
        snrs.append(
            10.0
            * np.log10(np.maximum(signal_power, 1e-20) / np.maximum(noise_power, 1e-20))
        )

    if not distortions:
        raise ValueError(f"{label}: no comparable frames survived alignment")

    distortion = np.concatenate(distortions)
    snr = np.concatenate(snrs)
    return PairwiseDistortion(
        label=label,
        reference=reference_name,
        subject=subject_name,
        bitrate_kbps=bitrate_kbps,
        n_recordings=len(reference_signals),
        n_frames=int(distortion.size),
        mean_spectral_distortion_db=round(float(np.mean(distortion)), 4),
        median_spectral_distortion_db=round(float(np.median(distortion)), 4),
        fraction_frames_above_2db=round(float(np.mean(distortion > 2.0)), 4),
        mean_segmental_snr_db=round(float(np.mean(snr)), 4),
        median_segmental_snr_db=round(float(np.median(snr)), 4),
        mean_delay_samples=round(float(np.mean(delays)), 1),
        max_delay_samples=int(max(delays)) if delays else 0,
    )


def load_audio(paths: Sequence[Path], max_seconds: float) -> list[NDArray[np.float64]]:
    """Read recordings, peak-normalised, truncated to a common maximum length."""
    import soundfile as sf

    signals: list[NDArray[np.float64]] = []
    for path in paths:
        data, rate = sf.read(str(path), dtype="float64", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        keep = int(max_seconds * rate)
        data = np.asarray(data[:keep], dtype=np.float64)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak <= 0.0:
            continue
        signals.append(data / peak * 0.7)
    return signals


def run(
    signals: Sequence[NDArray[np.float64]],
    sample_rate: int,
    bitrates: Sequence[float],
    *,
    seed: int,
) -> ValidationReport:
    """Encode every recording through both coders and compare all three signals."""
    started = time.monotonic()
    reference_codec = FfmpegAmrCodec()
    parametric: Codec = ParametricCelpCodec(seed=seed)
    notes: list[str] = []

    # The band-matched source. Comparing against the wideband original would
    # charge both coders for the bandwidth reduction, which is the channel rather
    # than the coding, and would swamp the difference between them.
    band_matched = [_resample_to_narrowband(signal, sample_rate) for signal in signals]

    report = ValidationReport(
        available=reference_codec.is_available,
        reference_mode=CodecMode.REFERENCE_AMR.value
        if reference_codec.is_available
        else None,
        parametric_mode=CodecMode.PARAMETRIC_CELP.value,
        ffmpeg_encoder=getattr(reference_codec, "_encoder", None),
        n_recordings=len(signals),
        bitrates_kbps=list(bitrates),
        sample_rate=sample_rate,
        elapsed_seconds=0.0,
        notes=notes,
    )

    if not reference_codec.is_available:
        notes.append(
            "no AMR-NB encoder is available in this ffmpeg build, so the "
            "reference comparison could not be made. The parametric model's "
            "distortion against a band-matched source is still reported, but it "
            "is not a validation of anything: it is the same measurement §16 "
            "already records, and the number that scopes this project is the "
            "difference between the two coders."
        )

    for bitrate in bitrates:
        parametric_coded = [
            parametric.apply(signal, sample_rate, bitrate).signal for signal in signals
        ]
        report.comparisons.append(
            compare(
                f"parametric vs band-matched source @ {bitrate:g}",
                band_matched,
                parametric_coded,
                "band_matched_source",
                CodecMode.PARAMETRIC_CELP.value,
                bitrate,
            )
        )

        if not reference_codec.is_available:
            continue

        reference_coded = [
            reference_codec.apply(signal, sample_rate, bitrate).signal for signal in signals
        ]
        report.comparisons.append(
            compare(
                f"reference vs band-matched source @ {bitrate:g}",
                band_matched,
                reference_coded,
                "band_matched_source",
                CodecMode.REFERENCE_AMR.value,
                bitrate,
            )
        )
        report.comparisons.append(
            compare(
                f"parametric vs reference @ {bitrate:g}",
                reference_coded,
                parametric_coded,
                CodecMode.REFERENCE_AMR.value,
                CodecMode.PARAMETRIC_CELP.value,
                bitrate,
            )
        )

    report.elapsed_seconds = round(time.monotonic() - started, 1)
    return report


def render(report: ValidationReport) -> str:
    """A table an examiner can read without opening the JSON."""
    lines = [
        f"AMR-NB encoder available: {report.available}"
        + (f" ({report.ffmpeg_encoder})" if report.ffmpeg_encoder else ""),
        f"{report.n_recordings} recordings, {report.sample_rate} Hz source, "
        f"{report.elapsed_seconds:.0f}s",
        "",
        f"{'comparison':<44} {'frames':>7} {'LSD dB':>7} {'>2dB':>6} {'segSNR':>7} {'delay':>6}",
    ]
    for item in report.comparisons:
        lines.append(
            f"{item.label:<44} {item.n_frames:>7} "
            f"{item.mean_spectral_distortion_db:>7.2f} "
            f"{item.fraction_frames_above_2db * 100:>5.1f}% "
            f"{item.mean_segmental_snr_db:>7.2f} "
            f"{item.mean_delay_samples:>6.0f}"
        )
    if report.notes:
        lines.append("")
        lines.extend(f"note: {note}" for note in report.notes)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("data/corpus/bembaspeech"),
        help=(
            "directory of audio to measure. Any speech serves — the comparison "
            "is coder against coder — but §16's figures came from BembaSpeech, "
            "so using the same material makes the numbers directly comparable."
        ),
    )
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--max-recordings", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--bitrates", type=float, nargs="+", default=[12.20, 4.75])
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/channel_validation.json")
    )
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    paths = sorted(arguments.audio.rglob(arguments.pattern))[: arguments.max_recordings]
    if not paths:
        print(
            f"no audio matching {arguments.pattern} under {arguments.audio}",
            file=sys.stderr,
        )
        return 2

    import soundfile as sf

    sample_rate = int(sf.info(str(paths[0])).samplerate)
    signals = load_audio(paths, arguments.max_seconds)
    if not signals:
        print("every candidate recording was silent", file=sys.stderr)
        return 2

    print(f"measuring {len(signals)} recordings from {arguments.audio}", flush=True)
    report = run(signals, sample_rate, arguments.bitrates, seed=arguments.seed)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                **{k: v for k, v in asdict(report).items() if k != "comparisons"},
                "source_files": [str(path) for path in paths],
                "comparisons": [asdict(item) for item in report.comparisons],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(render(report))
    print(f"\nwrote {arguments.output}", flush=True)

    # A non-zero exit when no reference coder exists, so an automated run cannot
    # be mistaken for a successful validation by anything reading exit codes.
    return 0 if report.available else 1


if __name__ == "__main__":
    sys.exit(main())
