"""Controlled channel degradation.

The apparatus for hypothesis H1: what speaker-discriminative information
survives narrowband mobile telephony, and how that varies with bitrate, noise
and packet loss.

Every degraded signal carries a :class:`~viflap.analysis.channel.codec.CodecMode`
recording whether a real AMR encoder or the parametric CELP model produced it.
Results from the two are not interchangeable, and the distinction is preserved
in the data rather than left to the reader's memory of how the experiment was
configured.
"""

from viflap.analysis.channel.codec import (
    AMR_NB_BITRATES,
    ChannelResult,
    Codec,
    CodecMode,
    FfmpegAmrCodec,
    ParametricCelpCodec,
    resolve_codec,
)
from viflap.analysis.channel.degradation import (
    DegradationCondition,
    DegradationSweep,
    NoiseType,
    add_shaped_noise,
    apply_condition,
    simulate_packet_loss,
)

__all__ = [
    "AMR_NB_BITRATES",
    "ChannelResult",
    "Codec",
    "CodecMode",
    "DegradationCondition",
    "DegradationSweep",
    "FfmpegAmrCodec",
    "NoiseType",
    "ParametricCelpCodec",
    "add_shaped_noise",
    "apply_condition",
    "resolve_codec",
    "simulate_packet_loss",
]
