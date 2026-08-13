"""The science layer.

Signal processing, statistical models, calibration and fusion. Depends on the
domain layer and on numpy/scipy; performs no I/O and knows nothing of databases,
HTTP or configuration files. Every function here can be tested by handing it an
array and inspecting what comes back.

``dsp``
    Framing, filterbanks, voice activity detection, linear prediction, pitch,
    voice quality, nasality.
``channel``
    Controlled narrowband degradation — the apparatus for characterising what
    survives the telephony channel.
``speaker``
    GMM-UBM, i-vector total variability, session compensation, two-covariance
    PLDA.
``spoof``
    Synthetic speech countermeasure and the validity gate that conditions
    acoustic admissibility.
``patterns``
    Conjugate marginal likelihood ratios, and the temporal, transactional and
    device streams built on them.
``behaviour``
    Idiolect and script structure over code-switched transcripts.
``calibration``
    Where a score becomes evidence. Nothing below this package returns a
    likelihood ratio.
``fusion``
    Combination with explicit dependence modelling, and measurement of what
    independence would have overstated.
"""
