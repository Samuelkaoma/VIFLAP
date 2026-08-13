"""Signal processing and statistical models, against known-answer signals."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest
import scipy.signal as sps
from scipy.special import logsumexp as sps_logsumexp

from viflap.analysis.channel import (
    NoiseType,
    ParametricCelpCodec,
    add_shaped_noise,
    simulate_packet_loss,
)
from viflap.analysis.channel.degradation import _passband_power
from viflap.analysis.dsp import (
    FrameConfig,
    add_deltas,
    autocorrelation,
    cepstral_mean_variance_normalise,
    detect_voice_activity,
    estimate_f0,
    filterbank_matrix,
    formants_from_lpc,
    frame_signal,
    levinson_durbin,
    lpc_analysis,
    measure_voice_quality,
    recommended_order,
    sliding_cmvn,
)
from viflap.analysis.patterns import (
    BackgroundPopulation,
    DirichletMultinomialComparator,
    NormalInverseGammaComparator,
)
from viflap.analysis.speaker.gmm import GmmConfig, _expectation_step, train_ubm
from viflap.analysis.speaker.plda import PldaConfig, train_plda
from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError

SR = 8000
LN10 = math.log(10.0)


def synth_vowel(f0: float, formants, duration: float, seed: int = 1, jitter: float = 0.0):
    """Source-filter synthesis with exactly known F0 and formants.

    Pulses are placed at fractional positions by splitting across two samples.
    Snapping to integers would inject roughly 0.7% jitter at 120 Hz that the
    synthesiser created and the experimenter did not, setting a floor beneath
    which no perturbation measurement could be validated.
    """
    gen = np.random.default_rng(seed)
    n = int(duration * SR)
    excitation = np.zeros(n)
    position = 0.0
    while position < n - 1:
        index = int(position)
        fraction = position - index
        excitation[index] += 1.0 - fraction
        excitation[index + 1] += fraction
        position += (SR / f0) * (1.0 + jitter * gen.standard_normal())

    signal = excitation
    for frequency, bandwidth in formants:
        radius = np.exp(-np.pi * bandwidth / SR)
        theta = 2 * np.pi * frequency / SR
        signal = sps.lfilter(
            [1.0], [1.0, -2 * radius * np.cos(theta), radius * radius], signal
        )
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak
    return signal


TARGET_FORMANTS = [(700.0, 80.0), (1220.0, 90.0), (2600.0, 130.0)]


class TestLinearPrediction:
    def test_recovers_a_known_autoregressive_process(self, rng) -> None:
        """The sign convention is asserted directly.

        Returning the predictor rather than the prediction-error filter flips
        the sign of the polynomial whose roots are the formants, and every
        downstream estimate with it. It is the easiest mistake here to make and
        the hardest to notice.
        """
        true_coefficients = np.array([1.0, -1.4, 0.6])
        noise = rng.standard_normal(40_000)
        signal = sps.lfilter([1.0], true_coefficients, noise)

        r = autocorrelation(signal, 2)
        estimated, _, _ = levinson_durbin(r, 2)
        assert np.allclose(estimated, true_coefficients, atol=0.03)

    def test_recovers_synthesised_formants(self) -> None:
        vowel = synth_vowel(120.0, TARGET_FORMANTS, 1.0)
        frames, _ = frame_signal(vowel, FrameConfig(sample_rate=SR))
        coefficients, _ = lpc_analysis(frames, recommended_order(SR))
        stats = formants_from_lpc(coefficients, SR, n_formants=3).statistics()

        assert stats["f1_median_hz"] == pytest.approx(700.0, abs=60.0)
        assert stats["f2_median_hz"] == pytest.approx(1220.0, abs=90.0)
        assert stats["f3_median_hz"] == pytest.approx(2600.0, abs=150.0)

    def test_selection_is_by_prominence_not_by_order(self) -> None:
        """Taking the lowest N survivors of a bandwidth filter reports F3 wrong.

        An order-12 fit yields the true resonances plus spurious wide poles
        where the model is absorbing spectral tilt. Sorting by frequency and
        truncating picks a spurious pole at about 1240 Hz as F3.
        """
        vowel = synth_vowel(120.0, TARGET_FORMANTS, 1.0)
        frames, _ = frame_signal(vowel, FrameConfig(sample_rate=SR))
        coefficients, _ = lpc_analysis(frames, 12)
        stats = formants_from_lpc(coefficients, SR, n_formants=3).statistics()
        assert stats["f3_median_hz"] > 2000.0


class TestPitch:
    @pytest.mark.parametrize("true_f0", [95.0, 145.0, 210.0])
    def test_no_octave_errors(self, true_f0: float) -> None:
        """The cumulative mean normalisation is what prevents these.

        A defect in the difference function shows up as an octave error at some
        fundamentals and not others, which is why several are tested.
        """
        signal = synth_vowel(true_f0, TARGET_FORMANTS, 1.0)
        stats = estimate_f0(signal, SR).statistics()
        assert stats["f0_median_hz"] == pytest.approx(true_f0, abs=3.0)


class TestVoiceQuality:
    def test_jitter_is_near_zero_on_a_periodic_signal(self) -> None:
        signal = synth_vowel(120.0, TARGET_FORMANTS, 3.0, jitter=0.0)
        measures = measure_voice_quality(signal, estimate_f0(signal, SR), min_periods=15)
        assert measures.jitter_local < 0.003

    def test_jitter_responds_monotonically_to_injected_perturbation(self) -> None:
        previous = -1.0
        for injected in (0.0, 0.005, 0.010):
            signal = synth_vowel(120.0, TARGET_FORMANTS, 3.0, seed=3, jitter=injected)
            measures = measure_voice_quality(
                signal, estimate_f0(signal, SR), min_periods=15
            )
            assert measures.jitter_local > previous
            previous = measures.jitter_local


class TestChannel:
    @pytest.mark.parametrize("noise_type", list(NoiseType))
    def test_snr_is_calibrated_in_the_telephony_passband(self, noise_type) -> None:
        """A broadband SNR means different things for different noise types.

        Vehicle noise sits largely below 300 Hz, so a broadband figure would be
        far higher in-band than the label suggests, and conditions sharing a
        label would not be comparable.
        """
        signal = synth_vowel(130.0, TARGET_FORMANTS, 1.5)
        noisy = add_shaped_noise(signal, noise_type, 10.0, SR, rng=np.random.default_rng(2))
        residual = noisy[: len(signal)] - signal
        measured = 10 * np.log10(
            _passband_power(signal, SR) / _passband_power(residual, SR)
        )
        assert measured == pytest.approx(10.0, abs=0.6)

    def test_codec_preserves_the_envelope_and_destroys_fine_structure(self) -> None:
        """The property that makes the simulation worth having.

        A bandpass-plus-noise model leaves the excitation intact, so jitter
        measured through it survives far better than it does in reality — and
        conclusions about glottal features would be optimistic.
        """
        codec = ParametricCelpCodec(seed=1)
        source = synth_vowel(120.0, TARGET_FORMANTS, 3.0, seed=5, jitter=0.010)
        source_measures = measure_voice_quality(
            source, estimate_f0(source, SR), min_periods=15
        )

        decoded = codec.apply(source, SR, 12.20).signal
        # F0 survives: the coder transmits a pitch lag.
        assert estimate_f0(decoded, SR).statistics()["f0_median_hz"] == pytest.approx(
            120.0, abs=4.0
        )
        # Glottal fine structure does not.
        try:
            decoded_measures = measure_voice_quality(
                decoded, estimate_f0(decoded, SR), min_periods=15
            )
            assert decoded_measures.reliability < source_measures.reliability / 2
        except InsufficientDataError:
            pass  # Refusing outright is an even stronger form of the same result.

    def test_packet_loss_is_concealed_not_zeroed(self) -> None:
        """No real decoder hands you a silent gap.

        Zeroing creates sharp broadband transients at both edges of every gap,
        which a countermeasure trained on such data learns to detect — that is,
        it learns to detect the simulation.
        """
        signal = synth_vowel(130.0, TARGET_FORMANTS, 3.0)
        lossy = simulate_packet_loss(signal, SR, 0.15, rng=np.random.default_rng(2))
        is_zero = np.abs(lossy) < 1e-12
        if np.any(is_zero):
            padded = np.concatenate([[False], is_zero, [False]])
            changes = np.flatnonzero(padded[1:] != padded[:-1])
            longest = int(np.max(changes[1::2] - changes[::2]))
        else:
            longest = 0
        # Concealment mutes only after four consecutive losses of 20 ms.
        assert longest <= int(SR * 0.10) + 1


class TestSpectral:
    def test_delta_of_a_constant_sequence_is_exactly_zero(self) -> None:
        constant = np.ones((50, 4))
        assert np.allclose(add_deltas(constant, order=1)[:, 4:], 0.0)

    def test_filterbank_has_no_empty_filters(self) -> None:
        bank = filterbank_matrix(24, 256, SR)
        assert bank.shape == (24, 129)
        assert np.all(bank.sum(axis=1) > 0)

    def test_vad_locates_speech_and_rejects_silence(self, rng) -> None:
        speech = synth_vowel(130.0, TARGET_FORMANTS, 0.6)
        silence = rng.standard_normal(int(0.5 * SR)) * 1e-4
        mixed = np.concatenate([silence, speech, silence, speech, silence])
        result = detect_voice_activity(mixed, FrameConfig(sample_rate=SR))
        assert 0.3 < result.speech_fraction < 0.75
        assert len(result.speech_segments()) == 2


class TestCepstralNormalisationWindow:
    """The window is not duration-neutral, and the tests say so explicitly.

    A fixed window normalises locally over a long recording and globally over a
    short one, which makes it a hidden second factor in any sweep over duration.
    These pin both halves of that behaviour and the setting that removes it.
    """

    #: Speech frames a 30 s, 15 s and 5 s recording actually yields through
    #: `amr12.2_clean`, measured on evaluation material rather than assumed.
    FRAMES_BY_DURATION: ClassVar[dict[int, int]] = {30: 2646, 15: 1324, 5: 446}

    @pytest.fixture
    def features(self, rng) -> np.ndarray:
        # A slow drift plus noise: without the drift a local and a global mean
        # coincide and the distinction under test would be invisible. The drift
        # dominates the noise so the comparison below is not a measurement of
        # whichever seed the fixture happened to draw.
        n = self.FRAMES_BY_DURATION[30]
        frames = np.arange(n)[:, None] / n
        return frames * np.array([9.0, -6.0, 3.0]) + rng.standard_normal((n, 3))

    @staticmethod
    def _distance_from_utterance_level(span: np.ndarray, window: int) -> float:
        windowed = sliding_cmvn(span, window, normalise_variance=True)
        utterance = cepstral_mean_variance_normalise(span, normalise_variance=True)
        return float(np.sqrt(np.mean((windowed - utterance) ** 2)))

    def test_non_positive_window_is_utterance_level_at_any_length(self, features) -> None:
        for n_frames in (446, 2646):
            span = features[:n_frames]
            assert np.allclose(
                sliding_cmvn(span, 0, normalise_variance=True),
                cepstral_mean_variance_normalise(span, normalise_variance=True),
            )
            assert np.allclose(
                sliding_cmvn(span, -1, normalise_variance=True),
                cepstral_mean_variance_normalise(span, normalise_variance=True),
            )

    def test_a_fixed_window_changes_character_with_duration(self, features) -> None:
        """The confound itself, as a monotone approach to the utterance mean.

        The default 300 frames covers 11% of a 30 s recording and 67% of a 5 s
        one, so it is a local estimate at the long end and nearly a global one
        at the short end. Shortening the recording therefore changes what the
        front-end does, not only how much speech it does it to — which is what
        makes the duration axis of the sweep two factors rather than one.
        """
        distances = [
            self._distance_from_utterance_level(features[:n_frames], 300)
            for n_frames in (
                self.FRAMES_BY_DURATION[5],
                self.FRAMES_BY_DURATION[15],
                self.FRAMES_BY_DURATION[30],
            )
        ]
        assert distances[0] < distances[1] < distances[2]
        # Not a marginal difference: at 5 s the operation sits far nearer
        # utterance level than at 30 s.
        assert distances[0] < 0.6 * distances[2]

    def test_a_window_longer_than_the_recording_is_exactly_global(self, features) -> None:
        """The limit of the same behaviour, where it stops being a matter of degree."""
        span = features[:280]
        assert np.allclose(
            sliding_cmvn(span, 300, normalise_variance=True),
            cepstral_mean_variance_normalise(span, normalise_variance=True),
        )

    def test_a_short_window_stays_local_at_both_durations(self, features) -> None:
        for n_frames in (446, 2646):
            span = features[:n_frames]
            windowed = sliding_cmvn(span, 100, normalise_variance=True)
            assert not np.allclose(
                windowed, cepstral_mean_variance_normalise(span, normalise_variance=True)
            )
            # A local mean tracks the drift, so the residual carries none of it:
            # the correlation between frame index and value is destroyed where
            # an utterance-level mean would leave it standing.
            index = np.arange(n_frames)
            local = abs(float(np.corrcoef(index, windowed[:, 0])[0, 1]))
            global_ = abs(
                float(
                    np.corrcoef(
                        index,
                        cepstral_mean_variance_normalise(span, normalise_variance=True)[
                            :, 0
                        ],
                    )[0, 1]
                )
            )
            assert local < global_


class TestGaussianMixture:
    def test_em_log_likelihood_never_decreases(self, rng) -> None:
        """A decrease is a defect in the update equations, not a property of data."""
        data = np.concatenate(
            [
                rng.normal(0.0, 1.0, (400, 4)),
                rng.normal(5.0, 1.0, (400, 4)),
                rng.normal(-4.0, 0.5, (400, 4)),
            ]
        )
        model = train_ubm([data], GmmConfig(n_components=3, max_iterations=40, seed=1))
        assert model.n_components == 3
        assert np.all(model.variances > 0)
        assert model.weights.sum() == pytest.approx(1.0)

    def test_refuses_too_few_frames_for_the_component_count(self, rng) -> None:
        with pytest.raises(InsufficientDataError):
            train_ubm([rng.standard_normal((30, 4))], GmmConfig(n_components=64))

    def test_baum_welch_statistics_are_centred_on_the_ubm(self, rng) -> None:
        """Adding the UBM mean back to the centred statistics recovers the
        component's own estimate of that mean.

        This is the property that makes the statistics usable by the i-vector
        model, which is defined on deviations from the UBM supervector. Asserting
        instead that the centred statistics sum to zero would be asserting that
        training reached an exact EM fixed point, which it does not — it stops on
        a tolerance, and the residual is a property of the stopping rule rather
        than of the statistics.
        """
        data = rng.standard_normal((500, 4))
        model = train_ubm([data], GmmConfig(n_components=2, max_iterations=30, seed=0))
        statistics = model.baum_welch(data)

        responsibilities = model.responsibilities(data)
        for component in range(model.n_components):
            occupancy = statistics.zeroth[component]
            assume_enough = occupancy > 1.0
            assert assume_enough
            recovered = statistics.first[component] / occupancy + model.means[component]
            expected = (responsibilities[:, component] @ data) / responsibilities[
                :, component
            ].sum()
            assert np.allclose(recovered, expected, atol=1e-9)


class TestExpectationStepChunking:
    """The E-step runs blockwise so its working set does not scale with the corpus.

    At 600k frames and 256 components the responsibility matrix alone is 1.2 GB
    and the step keeps several arrays of that size alive at once, which is more
    than the training machine has. Chunking is only sound if it computes the
    same thing, so that equivalence is asserted rather than assumed.
    """

    @pytest.fixture
    def features(self, rng) -> np.ndarray:
        return np.concatenate(
            [
                rng.normal(0.0, 1.0, (300, 5)),
                rng.normal(4.0, 1.0, (300, 5)),
                rng.normal(-3.0, 0.5, (300, 5)),
            ]
        )

    def test_statistics_match_the_whole_matrix_reduction(self, features) -> None:
        """Blocked accumulation equals forming the full matrix and reducing it."""
        model = train_ubm([features], GmmConfig(n_components=3, max_iterations=5, seed=1))

        joint = model.log_component_likelihoods(features)
        frame_log_likelihood = sps_logsumexp(joint, axis=1)
        gamma = np.exp(joint - frame_log_likelihood[:, None])

        chunked = _expectation_step(model, features, chunk_frames=64)

        assert chunked.mean_log_likelihood == pytest.approx(
            float(np.mean(frame_log_likelihood))
        )
        assert np.allclose(chunked.occupancy, gamma.sum(axis=0))
        assert np.allclose(chunked.first_moment, gamma.T @ features)
        assert np.allclose(chunked.second_moment, gamma.T @ (features**2))

    def test_chunk_size_does_not_change_the_trained_model(self, features) -> None:
        """The chunk size is a memory knob, not a modelling choice.

        A chunk of 7 frames and one larger than the corpus take different code
        paths through the accumulation loop and must still agree, otherwise the
        capacity result would depend on how much RAM the machine had.
        """
        blocked = train_ubm(
            [features],
            GmmConfig(n_components=3, max_iterations=20, seed=3, chunk_frames=7),
        )
        whole = train_ubm(
            [features],
            GmmConfig(n_components=3, max_iterations=20, seed=3, chunk_frames=10_000),
        )

        assert np.allclose(blocked.weights, whole.weights)
        assert np.allclose(blocked.means, whole.means)
        assert np.allclose(blocked.variances, whole.variances)

    def test_chunk_frames_must_be_positive(self) -> None:
        with pytest.raises(InvalidEvidenceError):
            GmmConfig(chunk_frames=0)


class TestPldaConvergence:
    """EM cannot decrease the observed-data likelihood, and now this is checked.

    What the trainer previously tracked was the quadratic data term alone — no
    ``log|W|``, no ``log|B|``, no latent prior, no posterior-covariance trace.
    That is neither the observed-data likelihood nor the evidence lower bound,
    so it carried no monotonicity guarantee, the stopping rule halted wherever
    it happened to settle, and the standard sanity check on an EM
    implementation was unavailable.
    """

    @staticmethod
    def _corpus(
        n_speakers: int = 30, per_speaker: int = 3, dimension: int = 6, seed: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        between = np.diag(rng.uniform(0.5, 3.0, dimension))
        within = np.diag(rng.uniform(0.2, 1.0, dimension))
        vectors, labels = [], []
        for speaker in range(n_speakers):
            position = rng.multivariate_normal(np.zeros(dimension), between)
            for _ in range(per_speaker):
                vectors.append(
                    position + rng.multivariate_normal(np.zeros(dimension), within)
                )
                labels.append(speaker)
        return np.array(vectors), np.array(labels)

    def test_the_likelihood_never_decreases_across_iteration_budgets(self) -> None:
        """Black-box monotonicity: more EM can only help.

        Training to a cap of 1, 2, 3 ... and comparing the likelihood each run
        reports tests the property without reaching into the loop, and would
        fail on an M-step that improved the tracked quantity while making the
        model worse.
        """
        vectors, labels = self._corpus()
        values = [
            train_plda(
                vectors, labels, PldaConfig(max_iterations=cap, min_speakers=10)
            ).final_log_likelihood
            for cap in range(1, 7)
        ]
        assert all(np.isfinite(values))
        for earlier, later in zip(values, values[1:], strict=False):
            assert later >= earlier - 1e-9

    def test_the_reported_likelihood_belongs_to_the_model_returned(self) -> None:
        """Stopping early and hitting the cap must both report their own model.

        The loop evaluates the likelihood before the M-step, so on the
        cap-exhausted path the parameters advance one step past the last
        evaluation. Reporting that stale value would describe a model the caller
        never receives.
        """
        vectors, labels = self._corpus()
        capped = train_plda(vectors, labels, PldaConfig(max_iterations=1, min_speakers=10))
        settled = train_plda(
            vectors, labels, PldaConfig(max_iterations=50, min_speakers=10)
        )

        assert capped.n_iterations == 1
        assert not capped.converged
        assert settled.converged
        assert settled.n_iterations <= 50
        assert settled.final_log_likelihood >= capped.final_log_likelihood - 1e-9

    def test_convergence_diagnostics_reach_the_model_description(self) -> None:
        vectors, labels = self._corpus()
        model = train_plda(vectors, labels, PldaConfig(min_speakers=10))
        diagnostics = model.diagnostics()

        assert diagnostics["n_iterations"] >= 1.0
        assert diagnostics["converged"] == 1.0
        assert np.isfinite(diagnostics["final_log_likelihood"])

    def test_heavy_tailed_data_still_converges_monotonically(self) -> None:
        """The guard must not fire on data the model merely fits badly.

        A decrease means the update equations are wrong. Misspecification is a
        different thing and has to remain trainable, or the check would be
        rejecting corpora rather than defects.
        """
        rng = np.random.default_rng(11)
        vectors, labels = [], []
        for speaker in range(30):
            position = rng.standard_t(3, 6) * 2.0
            for _ in range(int(rng.integers(2, 7))):
                vectors.append(position + rng.standard_t(2, 6) * 0.5)
                labels.append(speaker)

        model = train_plda(np.array(vectors), np.array(labels), PldaConfig(min_speakers=10))
        assert np.isfinite(model.final_log_likelihood)
        assert model.n_iterations > 1


class TestConjugateComparators:
    @pytest.fixture
    def agent_background(self) -> BackgroundPopulation:
        return BackgroundPopulation.from_counts(
            {"BUSY": 90_000, **{f"A{i:04d}": 20 for i in range(500)}},
            description="agent volumes",
        )

    def test_rarity_is_modelled_not_ignored(self, agent_background) -> None:
        """A set-overlap index scores these identically. This must not."""
        comparator = DirichletMultinomialComparator(agent_background, concentration=20.0)
        common = comparator.log_likelihood_ratio({"BUSY": 3}, {"BUSY": 3})
        rare = comparator.log_likelihood_ratio({"A0007": 3}, {"A0007": 3})
        assert rare > common + 2.0 * LN10

    def test_disagreement_supports_different_source(self, agent_background) -> None:
        comparator = DirichletMultinomialComparator(agent_background, concentration=20.0)
        assert comparator.log_likelihood_ratio({"A0007": 3}, {"A0091": 3}) < 0.0

    def test_evidence_accumulates_sub_linearly(self, agent_background) -> None:
        """The second transaction through the same agent tells you less than the first."""
        comparator = DirichletMultinomialComparator(agent_background, concentration=20.0)
        one = comparator.log_likelihood_ratio({"A0007": 1}, {"A0007": 1})
        ten = comparator.log_likelihood_ratio({"A0007": 10}, {"A0007": 10})
        assert ten > one
        assert ten < 10.0 * one

    def test_unseen_categories_do_not_give_infinite_evidence(self) -> None:
        """Zero background frequency would otherwise be a sampling gap reported
        as certainty."""
        background = BackgroundPopulation.from_counts({"A": 100, "B": 100}, "test")
        comparator = DirichletMultinomialComparator(background, concentration=10.0)
        value = comparator.log_likelihood_ratio({"NEVER_SEEN": 2}, {"NEVER_SEEN": 2})
        assert math.isfinite(value)

    def test_continuous_model_separates_scales(self, rng) -> None:
        background = np.log(np.exp(rng.normal(7.0, 1.4, 5000)))
        model = NormalInverseGammaComparator.from_background(
            background, within_actor_fraction=0.3
        )
        similar = model.log_likelihood_ratio(
            np.log([1000, 1050, 980]), np.log([990, 1010, 1040])
        )
        different = model.log_likelihood_ratio(
            np.log([1000, 1050, 980]), np.log([45000, 52000, 38000])
        )
        assert similar > 0.0 > different
