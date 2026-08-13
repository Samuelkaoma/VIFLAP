"""Signal processing front-end.

Pure functions and frozen configuration objects over numpy arrays. No I/O, no
model state, no dependency on any layer above this one — every function here can
be tested by handing it an array and inspecting what comes back.

Module map
----------
``framing``
    Pre-emphasis, framing, windowing, and the frame-to-time geometry every other
    module shares.
``spectral``
    Filterbanks and cepstra. MFCC for the speaker back-end, LFCC for the
    spoofing countermeasure, built on one filterbank construction so the
    comparison between them is meaningful.
``vad``
    Voice activity detection. Determines which frames every downstream statistic
    is computed over, which makes it the highest-leverage component in the
    front-end.
``lpc``
    Linear prediction: spectral envelope, formants by root-finding, and the
    glottal residual.
``pitch``
    F0 by YIN.
``voice_quality``
    Jitter, shimmer and harmonics-to-noise ratio, each accompanied by a
    reliability figure, because these are the measurements the codec damages
    most.
``nasality``
    Nasal segment detection and antiformant estimation — the feature class the
    proposal identifies as most resistant to deliberate disguise.
"""

from viflap.analysis.dsp.framing import (
    FrameConfig,
    FrameGeometry,
    WindowType,
    frame_signal,
    make_window,
    pre_emphasise,
)
from viflap.analysis.dsp.lpc import (
    Formant,
    FormantTrack,
    autocorrelation,
    formants_from_lpc,
    levinson_durbin,
    lpc_analysis,
    lpc_residual,
    lpc_spectrum,
    recommended_order,
)
from viflap.analysis.dsp.nasality import (
    Antiformant,
    NasalConfig,
    NasalFeatures,
    detect_nasal_frames,
    estimate_antiformants,
    extract_nasal_features,
)
from viflap.analysis.dsp.pitch import F0Track, PitchConfig, estimate_f0
from viflap.analysis.dsp.spectral import (
    CepstralConfig,
    FilterbankScale,
    add_deltas,
    cepstral_mean_variance_normalise,
    compute_cepstra,
    filterbank_matrix,
    hz_to_mel,
    mel_to_hz,
    power_spectrum,
    sliding_cmvn,
)
from viflap.analysis.dsp.vad import (
    VadConfig,
    VadResult,
    detect_voice_activity,
    spectral_flatness,
)
from viflap.analysis.dsp.voice_quality import VoiceQualityMeasures, measure_voice_quality

__all__ = [
    "Antiformant",
    "CepstralConfig",
    "F0Track",
    "FilterbankScale",
    "Formant",
    "FormantTrack",
    "FrameConfig",
    "FrameGeometry",
    "NasalConfig",
    "NasalFeatures",
    "PitchConfig",
    "VadConfig",
    "VadResult",
    "VoiceQualityMeasures",
    "WindowType",
    "add_deltas",
    "autocorrelation",
    "cepstral_mean_variance_normalise",
    "compute_cepstra",
    "detect_nasal_frames",
    "detect_voice_activity",
    "estimate_antiformants",
    "estimate_f0",
    "extract_nasal_features",
    "filterbank_matrix",
    "formants_from_lpc",
    "frame_signal",
    "hz_to_mel",
    "levinson_durbin",
    "lpc_analysis",
    "lpc_residual",
    "lpc_spectrum",
    "make_window",
    "measure_voice_quality",
    "mel_to_hz",
    "power_spectrum",
    "pre_emphasise",
    "recommended_order",
    "sliding_cmvn",
    "spectral_flatness",
]
