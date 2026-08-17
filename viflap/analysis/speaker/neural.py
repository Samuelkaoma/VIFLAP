"""A borrowed embedding extractor, and the one decision it forces.

§12 benchmarked this project's i-vector system against E3FS3 and found the gap
was not architecture. E3FS3's ResNet was trained on roughly 6,000 VoxCeleb2
speakers; only its LDA and PLDA saw 91. This system's extractor and back-end
both see 306. §9 had already measured that speaker count binds in the back-end —
125 to 306 speakers improved ``C_llr_min`` by 0.104 at the best cell — and §12
found the same mechanism operating one stage earlier.

Six thousand speakers cannot be collected for this project; §8 established that
twelve Zambian speakers have recoverable identity. What can be done is what
E3FS3 itself did: **take a publicly pre-trained extractor and retrain only the
back-end on the speakers available.** That imports the speakers into the stage
where they bind, without collecting them. It is a data-transfer strategy rather
than an architecture change, and it is the only one of the two that is reachable
here.

This module is deliberately *not* a replacement
-----------------------------------------------
The i-vector stack remains the reference implementation and the system of
record. This is a second extractor to be compared against it on the same
speakers through the same channel, because §7 is a standing reminder that a
plausible improvement can be significantly worse in all six cells. Nothing here
is reported until it has been paired against the existing system.

Sample rate is a measured decision, not a default
--------------------------------------------------
The checkpoint was trained on 16 kHz wideband speech. This project's channel
leaves audio at 8 kHz, so something has to give, and the choice was measured on
40 held-out recordings through ``amr12.2_clean`` rather than assumed:

===============================  ==========  ============================
input                            cosine EER  mean different-source score
===============================  ==========  ============================
coded, native 8 kHz              5.79%       0.621
coded, upsampled to 16 kHz       0.00%       0.386
wideband, no channel             0.00%       0.105
===============================  ==========  ============================

Feeding the network its native rate is not optional. At 8 kHz the different-
source similarity rises to 0.621 — the embeddings collapse toward one another
and stop separating people — because the model's filterbank is defined against a
16 kHz rate and everything it expects sits at the wrong frequency. Upsampling
restores the rate without restoring the band, and recovers nearly all of the
separation.

Those figures are a smoke test on twenty speakers under cosine scoring, not a
result. They establish that the extractor survives the channel and that the
resampling decision matters; they say nothing about ``C_llr``, which needs the
PLDA back-end and the full held-out evaluation.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from numpy.typing import NDArray

__all__ = [
    "NEURAL_EXTRACTOR_RATE",
    "prepare_for_extractor",
    "resample_to",
]

#: The rate the VoxCeleb2 checkpoints are defined against. Not a preference:
#: see the module docstring for what feeding 8 kHz costs.
NEURAL_EXTRACTOR_RATE = 16_000


def resample_to(
    signal: NDArray[np.float64], sample_rate: int, target_rate: int
) -> NDArray[np.float64]:
    """Resample by a polyphase filter, in either direction.

    ``scipy.signal.resample_poly`` rather than ``resample``, for the same reason
    the channel model gives: the latter is an FFT method that assumes the signal
    is periodic and produces wrap-around artefacts at both ends of a recording,
    which are loud, broadband, and would be analysed as speech.

    Upsampling adds no information — the band above 4 kHz stays empty after an
    8 kHz signal is taken to 16 — and that is the point. What it restores is the
    *rate* the extractor's filterbank is defined against.
    """
    if sample_rate == target_rate:
        return np.asarray(signal, dtype=np.float64)
    from math import gcd

    divisor = gcd(int(sample_rate), int(target_rate))
    return np.asarray(
        scipy.signal.resample_poly(
            np.asarray(signal, dtype=np.float64),
            int(target_rate) // divisor,
            int(sample_rate) // divisor,
        ),
        dtype=np.float64,
    )


def prepare_for_extractor(
    signal: NDArray[np.float64], sample_rate: int
) -> NDArray[np.float64]:
    """Put a recording into the form a VoxCeleb2 checkpoint expects.

    The whole of the sample-rate decision above, in one call, so that no caller
    has to remember it and no caller can quietly differ. Every embedding this
    project produces goes through here.
    """
    return resample_to(signal, sample_rate, NEURAL_EXTRACTOR_RATE)
