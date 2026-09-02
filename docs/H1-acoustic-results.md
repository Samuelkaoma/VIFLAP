# H1 — what survives the narrowband channel

**Status: first empirical result. Superseded numbers must not be quoted from
earlier drafts of this file.**

This reports the first numbers VIFLAP has produced from real human speech. Every
figure previously associated with this codebase came from synthetic signals
constructed inside test scripts, and proved only that the code computes what it
claims to compute.

---

## 1. What was done

| | |
|---|---|
| Corpus | LibriSpeech `train-clean-100` (OpenSLR resource 12) |
| Retrieved | 251 speakers, 5,672 utterances, streamed from a 6.39 GB archive |
| Usable after session filtering | 209 speakers, 1,049 recordings |
| Recording construction | consecutive utterances from one chapter concatenated to exactly 30.0 s, peak-normalised |
| Session unit | the LibriSpeech chapter |
| Channel | **parametric CELP model, not the reference AMR-NB coder** |
| Acoustic stack | GMM-UBM → i-vector total variability → length-norm/LDA/WCCN → two-covariance PLDA |

### Speaker-disjoint three-way split

| Partition | Speakers | Recordings | Purpose |
|---|---:|---:|---|
| Train | 125 | 614 | UBM, total variability matrix, transforms, PLDA |
| Development | 42 | 210 | calibration only |
| Evaluation | 42 | 225 | every reported metric |

No speaker appears in more than one partition. The split is rebuilt
deterministically from the corpus by both the training and the evaluation
script, and re-verified with `verify_disjoint` at the start of every evaluation
rather than assumed.

### Trained model

`ivec-plda-71db6bb5073718f7`

| Parameter | Value |
|---|---:|
| UBM components | 128 |
| Feature dimension | 60 (19 MFCC + energy, Δ, ΔΔ) |
| Cepstral normalisation window | 300 frames (3 s), sliding — see §17 |
| i-vector rank | 100 |
| PLDA dimension | 100 |
| Training frames (total) | 1,440,290 |
| Training frames (UBM estimation) | 599,878 |
| Training recordings skipped | 0 |
| PLDA between-speaker variance, max / mean | 50.76 / 1.57 |
| PLDA `psi` spectrum, ψ₁ / ψ₂ / ratio | 50.76 / 9.90 / **5.13** |
| PLDA dimensions with ψ < 0.1 | **40 of 100** |

The last two rows are the back-end's health, and they are here because nothing
else in this document shows it. In the diagonalised space `W = I`, so `psi` is
the between-speaker variance expressed in units of the within-speaker variance:
a dimension at ψ = 0.1 has a between-speaker standard deviation under a third of
its within-speaker one, and two recordings of one person differ along it by more
than two different people do. Forty such dimensions of a hundred are carrying
almost nothing.

`PldaModel.effective_dimension` was supposed to report this and did not. It
tested `psi > 1e-6` — a test for exact numerical collapse — so it returned
100 of 100 for every model this project has trained, including this one. The
threshold is now 0.1 and named; the count above is what the corrected
diagnostic returns.

**ψ₁ is five times ψ₂, and that is the more interesting number.** One dominant
axis of between-speaker variation is what a nuisance factor absorbed into the
speaker subspace looks like, because a factor shared across a speaker's
recordings is indistinguishable from the speaker as far as PLDA is concerned.
The ratio runs 5.1 to 7.0 across all five models on disk and across all three
cepstral-normalisation front-ends, so whatever produces it is not a
normalisation artefact. §21 reports what it is and whether removing it helps.

Training was multi-condition: each of the 614 training recordings was passed
through one of eight channel conditions spanning 4.75–12.20 kbit/s, clean,
babble and vehicle noise. Training on clean audio and evaluating on degraded
audio would have measured a front-end mismatch rather than speaker
discriminability.

---

## 2. How the trials were formed

**Same-source trials cross sessions.** A same-source trial is a pair of
recordings from two *different* chapters of one reader. Pairs from within one
chapter were excluded: they share a microphone, a room and a day, and a system
scores them alike partly for reasons that have nothing to do with the speaker.
The operational question is whether two separate calls can be linked.

**Resampling is over speakers, never over trials.** Every interval below is a
95% percentile bootstrap over the 42 evaluation speakers, drawing speakers with
replacement and taking all of each drawn speaker's trials. A bootstrap over
trials would treat correlated observations as independent and produce intervals
several times too narrow.

**Two calibrations are reported.** *Matched* uses a calibrator fitted on the
development speakers under the same channel condition — the optimistic case.
*Transferred* fits one calibrator under the cleanest condition and applies it
everywhere, which is what happens in service, where the channel a call arrived
over is not known in advance.

**Every `C_llr` is the as-reported quantity.** Calibrated log-LRs are clipped to
the empirical bounds fitted on the development set, which is what
`Calibrator.calibrate` does and therefore what any deployment emits. The
unclipped mapping from `Calibrator.transform` is a different quantity, is not
reported as `C_llr` anywhere in this document, and is retained per cell only as
`c_llr_matched_unbounded` and `c_llr_transferred_unbounded`. Confusing the two
overstates calibration loss by roughly a factor of three; §5 records the
correction, because the first version of this document made exactly that error.

---

## 3. The decision rule, fixed in advance

From `viflap.evaluation.hypotheses.H1ChannelViability`, which encodes the
proposal's falsification condition and was written before any of this data
existed:

- **Supported** if the *upper* bound of `C_llr_min` ≤ 0.30
- **Falsified** if the *lower* bound of `C_llr_min` > 0.50
- **Inconclusive** otherwise — the interval spans both thresholds

The decision is made on the interval, not the point estimate. Inconclusive is a
statement about the experiment, not about the world, and is reported as itself.

---

## 4. Results

All figures below come from model `ivec-plda-71db6bb5073718f7` (128 UBM components, rank 100), evaluated on 42 speakers disjoint from both training and calibration. Channel: **parametric_celp**.

Intervals are 95% percentile bootstraps resampling **speakers**, not trials. `C_llr_min` is discrimination and decides H1; `C_llr (matched)` is the cost of the reported likelihood ratio when a calibrator was fitted on development speakers under the same condition; `C_llr (transferred)` applies one calibrator fitted under `amr12.2_clean` to every cell, which is the operational case.

The transferred column is empty for `amr12.2_clean` at full duration: that is the cell the transferred calibrator was fitted from, so a figure there would be the matched one reported twice under a name suggesting it had been validated somewhere else.

| Condition | Dur. | C_llr_min [95% CI] | C_llr matched | C_llr transf. | Calib. loss | EER | Refused | H1 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 12.2 kbit/s, clean | 30 s | 0.343 [0.219, 0.429] | 0.416 | — | 0.073 | 10.86% | 0.0% | inconclusive |
| 12.2 kbit/s, clean | 15 s | 0.389 [0.279, 0.472] | 0.458 | 0.494 | 0.069 | 11.98% | 0.0% | inconclusive |
| 12.2 kbit/s, clean | 5 s | 0.574 [0.485, 0.644] | 0.626 | 1.150 | 0.052 | 19.17% | 1.3% | inconclusive |
| 12.2 kbit/s, babble 20 dB | 30 s | 0.353 [0.226, 0.432] | 0.420 | 0.425 | 0.067 | 11.36% | 0.0% | inconclusive |
| 12.2 kbit/s, babble 20 dB | 15 s | 0.414 [0.299, 0.489] | 0.468 | 0.533 | 0.054 | 13.04% | 0.0% | inconclusive |
| 12.2 kbit/s, babble 20 dB | 5 s | 0.542 [0.445, 0.607] | 0.596 | 0.989 | 0.054 | 18.09% | 12.4% | inconclusive |
| 12.2 kbit/s, babble 5 dB | 30 s | 0.407 [0.289, 0.500] | 0.454 | 0.518 | 0.047 | 12.12% | 0.0% | inconclusive |
| 12.2 kbit/s, babble 5 dB | 15 s | 0.501 [0.385, 0.597] | 0.531 | 0.759 | 0.030 | 15.87% | 15.6% | inconclusive |
| 12.2 kbit/s, babble 5 dB | 5 s | — [—, —] | — | — | — | — | 99.6% | not evaluable |
| 12.2 kbit/s, vehicle 20 dB | 30 s | 0.356 [0.227, 0.441] | 0.434 | 0.436 | 0.078 | 11.11% | 0.0% | inconclusive |
| 12.2 kbit/s, vehicle 20 dB | 15 s | 0.425 [0.306, 0.494] | 0.496 | 0.562 | 0.070 | 12.91% | 0.0% | inconclusive |
| 12.2 kbit/s, vehicle 20 dB | 5 s | 0.573 [0.484, 0.643] | 0.611 | 1.155 | 0.039 | 19.78% | 0.9% | inconclusive |
| 12.2 kbit/s, vehicle 5 dB | 30 s | 0.384 [0.260, 0.465] | 0.460 | 0.462 | 0.076 | 12.37% | 0.0% | inconclusive |
| 12.2 kbit/s, vehicle 5 dB | 15 s | 0.459 [0.340, 0.537] | 0.519 | 0.589 | 0.060 | 14.39% | 0.0% | inconclusive |
| 12.2 kbit/s, vehicle 5 dB | 5 s | 0.626 [0.533, 0.690] | 0.664 | 1.214 | 0.039 | 20.88% | 0.9% | **falsified** |
| 4.75 kbit/s, clean | 30 s | 0.344 [0.214, 0.434] | 0.411 | 0.412 | 0.066 | 9.89% | 0.0% | inconclusive |
| 4.75 kbit/s, clean | 15 s | 0.399 [0.291, 0.483] | 0.457 | 0.537 | 0.058 | 12.37% | 0.0% | inconclusive |
| 4.75 kbit/s, clean | 5 s | 0.594 [0.497, 0.679] | 0.644 | 1.219 | 0.049 | 19.82% | 0.9% | inconclusive |
| 4.75 kbit/s, babble 20 dB | 30 s | 0.356 [0.236, 0.443] | 0.413 | 0.441 | 0.057 | 11.11% | 0.0% | inconclusive |
| 4.75 kbit/s, babble 20 dB | 15 s | 0.424 [0.305, 0.507] | 0.471 | 0.624 | 0.047 | 14.39% | 0.0% | inconclusive |
| 4.75 kbit/s, babble 20 dB | 5 s | 0.547 [0.462, 0.606] | 0.596 | 1.158 | 0.049 | 18.40% | 9.3% | inconclusive |
| 4.75 kbit/s, babble 5 dB | 30 s | 0.408 [0.292, 0.503] | 0.455 | 0.538 | 0.048 | 11.87% | 0.0% | inconclusive |
| 4.75 kbit/s, babble 5 dB | 15 s | 0.530 [0.415, 0.614] | 0.553 | 0.901 | 0.022 | 16.92% | 1.3% | inconclusive |
| 4.75 kbit/s, babble 5 dB | 5 s | — [—, —] | — | — | — | — | 98.2% | not evaluable |
| 4.75 kbit/s, vehicle 20 dB | 30 s | 0.379 [0.256, 0.463] | 0.451 | 0.464 | 0.072 | 11.11% | 0.0% | inconclusive |
| 4.75 kbit/s, vehicle 20 dB | 15 s | 0.440 [0.332, 0.519] | 0.497 | 0.611 | 0.057 | 13.38% | 0.0% | inconclusive |
| 4.75 kbit/s, vehicle 20 dB | 5 s | 0.610 [0.523, 0.681] | 0.646 | 1.350 | 0.035 | 20.96% | 0.0% | **falsified** |
| 4.75 kbit/s, vehicle 5 dB | 30 s | 0.406 [0.294, 0.490] | 0.473 | 0.476 | 0.066 | 13.89% | 0.0% | inconclusive |
| 4.75 kbit/s, vehicle 5 dB | 15 s | 0.475 [0.367, 0.551] | 0.525 | 0.612 | 0.050 | 15.91% | 0.0% | inconclusive |
| 4.75 kbit/s, vehicle 5 dB | 5 s | 0.632 [0.545, 0.688] | 0.655 | 1.219 | 0.023 | 21.21% | 0.0% | **falsified** |

**2 of 30 cells produced no metric at all**, because the front-end refused almost every recording in them: `amr12.2_babble5dB` at 5 s (99.6% refused), `amr4.75_babble5dB` at 5 s (98.2% refused). These are excluded from the counts below rather than scored as failures. Refusing is the designed behaviour — an i-vector from under three seconds of speech reflects the model's prior rather than the recording — but it means the system has no acoustic opinion whatever at these operating points, which is a stronger statement than a poor one.

### Verdict on H1

Decision rule, fixed in advance: per cell: supported if the upper bound of C_llr_min <= 0.3; falsified if the lower bound > 0.5.

Of 28 evaluable cells: **0 supported, 3 falsified, 25 inconclusive**.

- Best cell: `amr12.2_clean` at 30 s — C_llr_min 0.343 [0.219, 0.429]
- Worst cell: `amr4.75_vehicle5dB` at 5 s — C_llr_min 0.632 [0.545, 0.688]

Trial counts at full duration: 396 same-source (cross-session only) and 24692 different-source, over 42 speakers. The speaker count, not the trial count, is the effective sample size, and it is what the intervals reflect.

Sweep completed in 82 minutes over 30 cells.

---

## 5. What the numbers say

### H1 is not supported anywhere, and is falsified at five seconds

No cell met the support condition, and the best is not close: `C_llr_min` 0.343
with an upper bound of 0.429 against a threshold of 0.30, under the most
favourable channel available. Three cells met the falsification condition, all at
5 seconds, and two of those had **no refusals at all**, so they are not artefacts
of a survivor subset.

The honest summary is that H1 is **duration-conditional**. The acoustic stream is
not viable at 5 seconds. At 30 seconds it is inconclusive everywhere — the
experiment cannot separate "usable" from "not usable" at that length, and 42
evaluation speakers is not enough resampling base to do so.

> **Corrected by §14, and superseded in part by §9.**
>
> Under bias-corrected intervals the falsification at five seconds is **wider
> than reported here, not narrower**: 6 of the 10 five-second cells are
> falsified rather than 3. Every 30-second and 15-second cell remains
> inconclusive, and no cell anywhere is supported. The three cells that moved
> — `amr12.2_clean@5s`, `amr12.2_vehicle20dB@5s`, `amr4.75_clean@5s` — did so
> because a downward-biased percentile interval makes falsification *harder* to
> reach, so correcting the bias pushed in the direction of the finding that was
> already here.
>
> Separately, the falsification is a property of a model trained on **125
> speakers**, not of the channel. Retrained on 306 (§9), both five-second cells
> in the reference condition return to inconclusive and the best cell improves
> to 0.276 [0.215, 0.388] at 7.89% EER.
>
> The numbers in this section stand as the 125-speaker percentile result. The
> corrected sweep is `data/reports/h1_sweep_bca_full.json`; §9 is the current
> model.

### Duration dominates; bitrate is nearly free until it isn't

| Factor | Contrast | Change in `C_llr_min` |
|---|---|---:|
| Duration | 30 s → 5 s | **+0.23** |
| Noise level | clean → babble 5 dB (30 s) | +0.06 |
| Bitrate | 12.2 → 4.75 kbit/s (30 s) | +0.001 |
| Bitrate | 12.2 → 4.75 kbit/s (5 s) | +0.020 |

> **Checked for a front-end confound in §17, and it survives.** The cepstral
> normalisation window is fixed at 300 frames, which is 11% of a 30-second
> recording and 67% of a five-second one — so shortening the recording changes
> what the front-end does as well as how much speech it does it to. Retrained
> with the window held duration-invariant, **94% of the 30 s → 5 s gap remains**
> and the contrast does not exclude zero. The row below is a duration effect.

Bitrate costs almost nothing at 30 seconds and roughly twenty times more at 5.
The factors interact rather than adding: with enough frames the i-vector averages
away the coarser spectral quantisation, and with 5 seconds it cannot. Bitrate
matters only when the system is already short of speech.

This is a property of *this front-end*, not of the channel. MFCCs describe the
spectral envelope, which is what an ACELP coder protects through LSF
quantisation. The glottal-source features in the proposal's taxonomy — jitter,
shimmer — measure the excitation fine structure the codebook discards, and the
bitrate axis would matter far more for them. Nothing here licenses a claim about
those features.

### Noise type matters more than noise level

Vehicle noise at 5 dB costs less than babble at 5 dB and produces no refusals
where babble produces almost total refusal, at the same nominal in-band SNR. The
asymmetry is the passband: vehicle energy is concentrated below 300 Hz and
largely does not survive the telephony filter, while babble occupies the speech
band and shares its modulation statistics. The degradation model was built on
that premise and the sweep is consistent with it.

### Matched calibration is cheap, and is not the constraint

> **Withdrawn by §15.** These figures are measured *after* the ELUB clip, which
> replaces the reported likelihood ratio with a bound for **60.6% of trials** at
> the best operating point — 61.5% of different-source against 3.0% of
> same-source. The clip removes 0.187 of 0.260 bits of calibration loss, so what
> is reported below is the residue left after a safeguard repaired most of the
> problem. Pre-clip, matched calibration costs **0.26 bits** at the best cell,
> which is the same order as the discrimination floor rather than negligible
> beside it. The numbers below remain correct as *operational* quantities —
> bounded is what a deployment emits — but the inference drawn from them does
> not hold.

With empirical bounding applied — which is what the system reports — matched
calibration costs a mean of **0.054 bits** across the 28 evaluable cells, never
more than 0.078.

| Duration | Mean matched calibration loss | Range |
|---|---:|---|
| 30 s | 0.065 | 0.047 – 0.078 |
| 15 s | 0.052 | 0.022 – 0.070 |
| 5 s | 0.043 | 0.023 – 0.054 |

A separate experiment (`scripts/compare_calibrators.py`, results in
`data/reports/calibrator_comparison.json`) fitted logistic, isotonic and
kernel-density calibrators on identical development trials across nine cells.
After bounding, all three land within about 0.01 of each other — at 12.2 kbit/s
clean and 30 seconds, 0.416 / 0.426 / 0.420. The affine restriction of the
logistic form costs essentially nothing once the empirical bounds are applied,
and no available calibrator family recovers a meaningful part of the remaining
gap. Where the reported likelihood ratio falls short of the discrimination, the
shortfall is small and it is not the mapping's fault.

> **The artefact said something else, and the artefact was wrong.** Until now
> `calibrator_comparison.json` held 0.603 / 0.451 / 9.929 under a field named
> `c_llr` — the *unbounded* mapping. The figures in this paragraph are the
> bounded ones and were correct, having been recomputed by hand, but anyone
> opening the JSON would have concluded they had been edited rather than
> measured. The script was scoring `Calibrator.transform`; it now reaches the
> reported quantity through the same `as_reported` helper `evaluate_h1.py` uses,
> so the two cannot disagree about what a reported likelihood ratio is.
>
> Rerun, it reproduces every figure in this paragraph — 0.4164 / 0.4258 /
> 0.4200 — and every clipped fraction in §15, and now records both columns, the
> clipped fraction by class, and the bounds themselves. §15 explains why the
> conclusion drawn from these numbers does not follow even though the numbers
> hold.

**A correction, recorded because the wrong version was written down first.** An
earlier draft of this document claimed matched calibration cost ~0.26 bits and
that a non-affine calibrator would recover most of it. That was measured with
`Calibrator.transform`, the raw mapping, rather than `Calibrator.calibrate`,
which additionally clips to the empirical bounds. The bounding is not cosmetic —
it is what stops a calibration extrapolating past the strength its validation
data supports — so the unbounded figure scored a mapping that never reaches a
report. It overstated the loss roughly threefold and made the kernel-density
calibrator look catastrophic (`C_llr` 9.9) where its bounded form is competitive
(0.42). Every `C_llr` in this document is now the bounded, as-reported quantity;
the unbounded values are retained per cell as
`c_llr_matched_unbounded` / `c_llr_transferred_unbounded` for diagnosis only.

### Transferred calibration at short duration is worse than reporting nothing

A `C_llr` of 1.0 is the cost of a system that always reports a likelihood ratio
of 1 — one offering no evidence whatever. **Seven of the eight evaluable
5-second cells exceed it**, and this survives bounding.

| Duration | `C_llr` transferred, range | Cells worse than neutral |
|---|---|---:|
| 30 s | 0.412 – 0.538 | 0 of 9 |
| 15 s | 0.494 – 0.901 | 0 of 10 |
| 5 s | **0.989 – 1.350** | **7 of 8** |

A calibrator fitted on 30-second clean speech and applied to a 5-second recording
does not merely degrade: it produces likelihood ratios that are actively
misleading, worse for a decision-maker than an admission of ignorance. The
mechanism is straightforward — short-duration i-vectors are shrunk toward the
prior, and a mapping fitted on unshrunk scores reads that shrinkage as evidence.

This is the strongest operational finding in the sweep and it bears directly on
the system's design. It is empirical support for a rule the architecture already
enforces: a stream outside the domain its calibration was fitted for must report
**absence**, not a number. `AbsenceReason.OUT_OF_CALIBRATION_DOMAIN` exists for
exactly this case, and these figures are the argument for gating on duration
explicitly rather than trusting one deployed calibrator to generalise across it.

Note the asymmetry with the previous finding. Calibration is cheap when the
calibrator matches the condition and harmful when it does not, and the gap
between those two is far larger than any difference between calibrator families.
Effort spent choosing a better calibration *form* is misdirected; effort spent
ensuring a calibration is *applicable* is not.

---

## 6. What these numbers do not establish

Each of the following is a reason the figures above are **not** an estimate of
operational performance. They are listed in the order they would most change the
result.

**The channel is simulated, not real.** ffmpeg with `libopencore-amrnb` was not
available, so every recording passed through the parametric CELP model rather
than a real AMR-NB coder. The model reproduces the *character* of the
degradation — quantised spectral envelope, resynthesised excitation, fine glottal
structure replaced by codebook pulses — and is not the coder. `ChannelResult.mode`
records `parametric_celp` on every result here, and these figures must never be
pooled with any obtained through the reference coder. This is limitation 4 of
the proposal, and it applies in full.

**The speech is not telephony and the speakers are not the target population.**
LibriSpeech is 16 kHz read audiobook English recorded on good microphones by
volunteers reading prepared text. The target population is Bantu-language
conversational telephony from callers actively attempting deception. Read speech
is more fluent, more consistent in level and rate, and phonetically richer than
conversational speech, and every one of those differences flatters a speaker
comparison system. Whether the penalty for cross-lingual transfer is real is
hypothesis H4 and is untested here.

**The model is small by the standards of the literature.** 128 UBM components
and rank 100, trained on 125 speakers. Published i-vector/PLDA systems use
512–2048 components, rank 400–600, and thousands of training speakers. The
figures below therefore characterise *this* system, and are a lower bound on
what the architecture can do rather than an estimate of the channel's
information ceiling.

This was the last standing explanation for `C_llr_min` ≈ 0.34 once calibration
had been ruled out, and it has since been tested directly. **§7 reports the
result: it is not the explanation.** Doubling capacity makes the system
significantly *worse*, and the constraint is the corpus. The sentence that stood
here previously — that distinguishing "the channel destroyed the information"
from "the model was too small to find it" required a capacity sweep not yet run
— is superseded by §7.

**Sessions are not calls.** A LibriSpeech chapter is a recording session, used
here as the closest available analogue to a separate call. Two calls by one
offender differ in handset, network, background and emotional state far more
than two chapters by one volunteer differ.

**No disguise, no synthesis, no second stream.** H2, H3, H5 and H7 are untouched.
The corpus contains no deliberate disguise, no synthetic speech, and no
transcripts, timings, transactions or handset identifiers, so nothing here says
anything about the four non-acoustic streams or about fusion.

**At short durations the metric is computed on survivors, and that flatters
it.** The front-end refuses any recording with less than three seconds of net
speech. At 30 seconds almost nothing is refused; at 5 seconds the refusal rate
rises sharply, and it rises *with added noise* — noise lifts the voice activity
detector's noise floor, so marginal recordings fall below the threshold.

The refusal is therefore not random with respect to difficulty. It removes the
recordings carrying least speech, which are the hardest ones, and the metric is
then computed over what survived. This shows up directly in the table: some
noisy 5-second cells report a *lower* `C_llr_min` than the clean 5-second cell
at the same duration, which is not noise improving discrimination but the
harder recordings having been excluded before scoring.

Refusal rate is reported per cell for exactly this reason. Any 5-second cell
with a substantial refusal rate should be read as "the discrimination available
on the recordings this system was willing to score", and the refused fraction is
part of the result rather than a footnote to it. The system declining is the
correct behaviour — an i-vector from two seconds of speech is dominated by its
prior — but a cell's `C_llr_min` and its refusal rate have to be quoted
together.

**42 evaluation speakers is a small resampling base.** The intervals are honest
about that — they are wide — but a bootstrap over 42 units cannot be narrow, and
some cells will be inconclusive for that reason alone rather than because the
underlying quantity sits between the thresholds.

---

## 7. Capacity: is 0.34 the channel, or the model?

§6 left one explanation standing. Calibration is not the constraint, and the
remaining question was whether `C_llr_min` ≈ 0.34 is the channel having
destroyed the speaker information or this model having been too small to find
what survived. That question decides whether the thesis re-scopes away from the
acoustic stream, so it was tested rather than argued.

A second model was trained at double the capacity on the identical corpus,
split, channel conditions and seed:

| | 128/100 baseline | 256/200 variant |
|---|---:|---:|
| Model | `ivec-plda-71db6bb5073718f7` | `ivec-plda-b4f91e467a430fbb` |
| UBM components | 128 | 256 |
| i-vector rank | 100 | 200 |
| UBM frames | 599,878 | 599,878 |
| Training speakers / recordings | 125 / 614 | 125 / 614 |

Artefacts: `data/reports/training_large.json`, `data/reports/h1_capacity.json`,
`data/reports/h1_capacity_paired.json`.

### The comparison is paired, and it has to be

Both models were scored on the *same* evaluation speakers, through the *same*
degraded audio, on the *same* trials. Their separate bootstrap intervals overlap
almost entirely — but that width is variation between the 42 speakers, and both
models saw those same 42 speakers, so nearly all of it is common to the two and
cancels in a difference.

Reading the overlap as "no difference" would have manufactured a null. At 30 s
clean the marginal intervals are [0.219, 0.429] and [0.271, 0.486] — heavily
overlapping — while the paired difference is [+0.027, +0.085], excluding zero
decisively. The comparison below therefore bootstraps the **difference** over
speakers (`paired_bootstrap_over_speakers`), via `scripts/compare_capacity.py`.

Both models refused identically in every cell (0/0, 3/3, 28/28), so no
restriction to common survivors was needed and the trials pair exactly.

### More capacity makes it worse

`C_llr_min`, and the paired difference variant − baseline. Positive means the
larger model discriminates **worse**.

Intervals are BCa at B = 2000, with p-values derived through the same
correction and Holm-Bonferroni applied over the six cells.

| Condition | Dur. | 128/100 | 256/200 | Difference [95% CI] | p | Holm |
|---|---:|---:|---:|---|---:|:---:|
| 12.2 kbit/s, clean | 30 s | 0.343 | 0.398 | +0.055 [+0.028, +0.086] | 0.0010 | **✓** |
| 12.2 kbit/s, clean | 15 s | 0.389 | 0.459 | +0.070 [+0.049, +0.103] | 0.0010 | **✓** |
| 12.2 kbit/s, clean | 5 s | 0.574 | 0.604 | +0.030 [+0.002, +0.057] | 0.0342 | **✓** |
| 12.2 kbit/s, babble 20 dB | 30 s | 0.353 | 0.402 | +0.050 [+0.029, +0.080] | 0.0010 | **✓** |
| 12.2 kbit/s, babble 20 dB | 15 s | 0.414 | 0.471 | +0.058 [+0.032, +0.084] | 0.0010 | **✓** |
| 12.2 kbit/s, babble 20 dB | 5 s | 0.542 | 0.586 | +0.044 [+0.018, +0.069] | 0.0010 | **✓** |

**All six cells are significantly worse, and all six survive the multiplicity
correction.** EER agrees independently, degrading in all six (10.86% → 12.30% at
the best cell). Under the fixed decision rule the 5 s clean cell now
**falsifies** H1 for the larger model, where the baseline was inconclusive there.

> An earlier version of this table used percentile intervals at B = 300 and
> reported five of six, with the sixth described as missing significance "by a
> thousandth". That bound was well inside the Monte-Carlo error of a tail
> quantile at B = 300 and should not have been stated to that precision. With
> the bias correction of §14 and B = 2000 it excludes zero at p = 0.034. The
> finding is unchanged in direction and stronger in support.

The headline sweep in §4 is unaffected: the 128/100 model remains the system of
record, and its numbers are unchanged.

### Why: the corpus caps the discriminative subspace

The degradation is not a surprise once the transform chain is examined. From
`fit_transform_chain` in `viflap/analysis/speaker/transforms.py`:

```python
maximum_lda = min(dimension, n_speakers - 1)
```

The between-speaker scatter of `S` speakers has rank at most `S − 1`, so with
**125 training speakers no more than 124 discriminative dimensions exist**,
whatever the i-vector rank. The consequence is visible in the two models:

| | i-vector rank | Dimension reaching PLDA |
|---|---:|---:|
| 128/100 baseline | 100 | 100 — passes through untouched |
| 256/200 variant | 200 | **124 — truncated by the speaker count** |

The requested capacity increase was 2× in rank. What survived to PLDA was
100 → 124, a factor of 1.24, and **no rank above 124 can ever deliver more at
125 training speakers.** This is an *a priori* ceiling, true before a single
trial was scored.

The UBM half compounds it in the other direction. Holding the frame budget at
599,878 — necessary, since changing it would confound capacity with training
data — halves the data per component, from ~4,690 frames per component to
~2,343. More parameters estimated from the same evidence is the textbook recipe
for the noisier estimate the numbers show.

### What this does and does not establish

Three explanations were in play. This experiment separates them unevenly, and
saying which is which is the point of the section.

**Refuted: "the model was too small to find it."** This is settled, and settled
more strongly than a null would have settled it. Capacity did not merely fail to
help — it hurt, significantly, in **all six** cells, every one of which survives
a multiplicity correction. The 128/100 configuration is not below the useful size for this corpus;
it is at or past the optimum. Building a bigger acoustic model on this training
set is not a route to a better `C_llr_min`.

**Supported: "this corpus is the limit."** Two independent lines converge. The
structural one is the 124-dimension ceiling, which follows from the speaker
count by mathematics rather than by measurement. The empirical one is the
degradation itself, which is what over-parameterisation on insufficient data
looks like. 125 training speakers caps PLDA quality regardless of component
count, and this is the constraint that binds.

**Not established: "the channel destroyed the information."** Refuting "too
small" does *not* promote this to the answer, and it must not be reported as
though it does. The corpus ceiling means **no model trained on this corpus can
measure the channel's information content** — the channel may well be preserving
a great deal that 125 speakers cannot teach any model to read. This experiment
is silent on the channel, and the acoustic stream should not be re-scoped out of
the thesis on the strength of it.

> **Answered in §9.** Trained on 306 speakers at this same configuration, the
> system improves significantly in five of six cells — four surviving a
> multiplicity correction — and in the same direction in all six. The channel
> was *not* destroying the information in the benign
> conditions — it was preserving what 125 speakers could not teach the model to
> read. The gain shrinks as the channel worsens, so both explanations hold in
> different regimes rather than one being correct.
>
> §12 benchmarks this against a published forensic system and finds the same
> mechanism operating one stage earlier: that system's extractor was trained on
> ~6,000 speakers. Speaker count binds in the front-end as well as the back-end.

### What would separate the remaining two

More speakers, not more parameters. The 124-dimension ceiling rises directly
with the training speaker count, so a corpus with, say, 1,000 speakers lifts it
to 999 and lets capacity be tested at a point where it is not already
self-defeating.

**Common Voice** is the natural second corpus for the speaker count. It was
assumed here that it would also move the material toward the target population
and so serve H4 at the same time; §8 surveys what is actually available and
finds that it does not — Common Voice carries no Zambian language at all, and
the corpora that do carry them are the smallest on the axis that binds. The two
goals separate, and §8 sets out the architecture that follows from that.

Until a second corpus is in place, the honest statement of the acoustic result
is that H1 is duration-conditional, inconclusive at 30 s, falsified at 5 s, and
**limited by the training corpus rather than by the channel or by the model
size** — with the channel's own contribution still unmeasured.

### Reproducibility note

The two training runs realised 1,440,290 and 1,440,209 total frames — a
difference of 81 frames, 0.006%. The UBM frame count was identical (599,878),
and the degradation seeds are derived per recording, so this is floating-point
non-determinism in the codec propagating through the voice-activity boundaries
rather than a difference in material. It is immaterial at this magnitude and is
recorded rather than smoothed over. The *evaluation* comparison is unaffected in
any case: it degrades each cell once and embeds that same audio with both
models, so the trials are shared by construction.

---

## 8. Corpus strategy for the target population

§7 concluded that the training corpus is what binds, and that more speakers are
the way past it. This section surveys what is actually obtainable for Zambian
languages, and reports a defect in the largest of those corpora that decides how
it can be used.

### The stack does not need one corpus

Only two of the seven training stages are limited by speaker count:

| Stage | Needs | Speaker labels |
|---|---|:---:|
| 1. Front-end features | audio | ✗ |
| 2. UBM | lots of audio | ✗ |
| 3. Baum-Welch statistics | audio | ✗ |
| 4. Total variability matrix | lots of audio | ✗ |
| 5. i-vector extraction | audio | ✗ |
| **6. LDA** | **many speakers** | ✓ |
| **7. PLDA** | **many speakers** | ✓ |

Stages 2 and 4 want hours of representative audio; stages 6 and 7 want many
distinct speakers. Those are different requirements, and nothing obliges one
corpus to satisfy both. `SpeakerComparisonSystem.train` now accepts a separate
`background` corpus for the unsupervised stages, so the front-end can be
estimated on material matched to the deployment population while the back-end
takes its speaker subspace from wherever enough speakers exist. The pool is
traversed twice without retaining audio, so its size does not become the peak
memory of a training run.

### What exists for Zambian languages

**Common Voice 26.0** spans 294 locales and includes **no Zambian language** —
no Bemba, Nyanja, Tonga or Lozi. Its Bantu holdings are East African:

| Locale | Contributors | Validated hrs | Download | Mean clip |
|---|---:|---:|---:|---:|
| Swahili `sw` | **1,523** | 392.2 | 22.4 GB | 5.25 s |
| Kinyarwanda `rw` | 1,185 | 2,001.7 | 61.4 GB | 5.01 s |
| Luganda `lg` | 672 | 436.9 | 11.9 GB | 5.78 s |

Swahili is the better target despite holding a fifth of Kinyarwanda's audio:
contributors set the LDA ceiling, hours do not, and it is a third of the
download. The ~5 s mean clip length is common to all three and matters — a 30 s
condition cannot be cut from a 5 s clip without concatenating across sessions,
which is not the same thing as 30 s of continuous speech.

**Zambian corpora** are the mirror image — right population, wrong scale:

| Corpus | Language | Speakers *documented* | Speakers *recoverable* | Hours |
|---|---|---:|---:|---:|
| BembaSpeech / ZV Bemba | Bemba | 17 | **17** (12 usable) | 23.6 |
| Zambezi Voice | Nyanja | 12 | **0** | 21.9 |
| | Tonga | 9 | **0** | 19.6 |
| | Lozi | 6 | **0** | 4.4 |
| Zambezi Voice (unlabelled) | 5 languages | — | — | 160 |
| BIG-C | Bemba | — | **0** | 187.1 |

All are 16 kHz mono WAV, so none needs transcoding.

The gap between the documented and recoverable columns is the finding. Only
BembaSpeech publishes identity that survives contact with the released files.

**BembaSpeech** encodes the speaker in the filename —
`01-200921-192247_bem_d31_elicit_16.wav` gives speaker `01`, session
`200921-192247` — and ships a roster of sex, utterance count, duration and
native language. Parsing all 15,489 filenames reproduces the roster's
per-speaker utterance counts **exactly for all seventeen speakers**, with 1,051
files correctly identified as carrying no speaker and none misattributed. The
identity is real and checkable.

It is also thin. Seventeen speakers, steeply imbalanced — one holds 10.6 hours,
the median is 14 minutes — and **only twelve have two or more sessions**.
Within-speaker covariance cannot be estimated from a single session, so twelve
is the operative number, against the 42 evaluation speakers this project uses
now.

**Zambezi Voice's Nyanja, Tonga and Lozi** publish no speaker identity at all.
Filenames begin with the session, `221102-102320_nya_510_elicit_0.wav`; no
manifest column names a speaker; no roster is released. No field or combination
of fields reproduces the documented counts — the session timestamp yields
172/250/44 distinct values against 12/9/6 speakers, and the only low-cardinality
field takes 4–5 values that look like annotator or text-source hashes. The paper
counts speakers from the authors' records; the release does not say which
recording is whose.

### AfriSpeech-200 was never blocked, and contributes nothing

> **A correction to this project's own records.** AfriSpeech-200 was listed as
> unresolved because its usable-speaker count was thought to sit behind account
> creation and terms acceptance. It does not. The dataset reports
> `gated=False`, and its per-utterance manifests are served without
> authentication: three CSVs, 22 MB in total, no audio and nothing to agree to.
> The question was answerable at any point and was not asked. Artefact:
> `data/reports/afrispeech_survey.json`, from `scripts/survey_afrispeech.py`.

67,365 utterances from **2,463 speakers**, which on the hour count is a serious
resource. Distributed by country:

| Country | Speakers | Usable (≥60 s total) | Hours |
|---|---:|---:|---:|
| Nigeria | 1,979 | 1,285 | 141.95 |
| South Africa | 223 | 147 | 22.63 |
| Kenya | 137 | 101 | 20.82 |
| Botswana | 38 | 27 | 3.95 |
| Ghana | 37 | 31 | 5.15 |
| Uganda | 26 | 17 | 2.89 |
| Rwanda | 9 | 6 | 1.47 |

**Zambia does not appear.** Not few speakers — none: zero rows carry `ZM`, and
the field is present and populated for every other country in the corpus, so
this is an absence in the data rather than a missing column.

The language view is no better:

| Language | Speakers | Usable | Hours |
|---|---:|---:|---:|
| Bemba | **0** | 0 | 0.00 |
| Nyanja | **0** | 0 | 0.00 |
| Chichewa | **1** | 1 | 0.15 |
| Tonga | **0** | 0 | 0.00 |
| Lozi | **0** | 0 | 0.00 |

Chichewa is the one hit and it is a single speaker with nine minutes of speech,
who is **Malawian** — Malawi contributes exactly one speaker to the corpus and
Chichewa exactly one, and they are the same person. Chichewa and Nyanja are the
same language across a border, so this is the closest AfriSpeech comes to the
target population: one person, 545 seconds, from the wrong country.

**One speaker is not one speaker short of useful.** A single speaker cannot form
a same-source trial, cannot contribute to a within-speaker covariance estimate,
and cannot be split across partitions. §8's Bemba finding was that twelve usable
speakers is one short of splittable; this is eleven further short of that.

The threshold above is 60 seconds of total speech, which is the floor for two
30-second recordings — one to enrol, one to test. It is deliberately generous:
§5 and §22 both measure this system at 30 seconds, and neither suggests a
speaker contributing exactly two such recordings is worth much. It is a test of
whether a speaker is splittable at all.

**This strengthens the section's conclusion rather than qualifying it.**
AfriSpeech-200 is pan-African accented *English* built for clinical speech
recognition, and it does what it was built for: 141 hours from Nigeria, with
speaker identity that survives into the release, which is more than any Zambian
corpus in the table above manages. The identity is there. The population is not.
Two hundred hours of labelled African speech turns out to contain one speaker
of one Zambian language, and he lives in Malawi.

### BIG-C's speaker identifier does not identify a speaker

BIG-C is by far the largest Zambian speech corpus, and on paper the most
attractive: 92,117 utterances, 187 hours of spontaneous conversational Bemba,
16 kHz mono WAV, with a `speaker_id` column. It cannot be used for speaker
recognition as released.

The `speaker_id` field takes 74 distinct values, but **37 of those 74 carry both
male and female gender labels**, and not marginally — the minority gender's share
has a median of 0.468 and a maximum of 0.498. Near-perfect halves are not
annotation noise. The field is a conversation-local participant index, not a
person: it distinguishes the speakers *within* a dialogue and is reused across
dialogues by different people.

The finest identity recoverable from the released metadata is the pair
(session, `speaker_id`), taking the recording timestamp embedded in the audio
filename as the session. That yields 2,913 identities of which only 36 are
gender-inconsistent — a 1.2% rate consistent with ordinary annotation error, and
evidence that the pair *is* a coherent identity where the bare field is not.

This leaves no correct way to label the corpus for speaker recognition:

- Treating `speaker_id` as the speaker puts **different people into the same
  class**, contaminating the same-source distribution — the more damaging error,
  since it inflates apparent within-speaker variability.
- Treating (session, `speaker_id`) as the speaker puts **one person into many
  classes**, contaminating the different-source distribution with same-speaker
  pairs and depressing apparent performance.

Neither supports the speaker-disjoint split this project's method requires, and
`verify_disjoint` cannot detect the problem because it compares identifiers,
which are exactly what is unreliable here. A restricted use remains sound —
within a single session `speaker_id` is trustworthy, so within-session trials are
valid — but those are same-occasion, same-channel comparisons, the easiest
condition there is and the least like forensic casework.

### What follows

BIG-C is not lost; it is reclassified. Its 187 hours need **no speaker labels at
all** to serve stages 2 and 4, and neither do Zambezi Voice's 160 hours of
unlabelled radio. Together that is roughly 350 hours of Zambian audio available
for the unsupervised front-end, which is precisely what the `background`
parameter consumes. The defect that disqualifies BIG-C as labelled material is
irrelevant to the stages that do not read labels.

That gives a three-part architecture:

1. **UBM and total variability** — Zambian audio: BIG-C plus Zambezi Voice's
   unlabelled radio. Broadcast channel variation is multi-condition training at
   no cost.
2. **LDA and PLDA** — Common Voice Swahili, for its 1,523 contributors.
3. **Evaluation** — BembaSpeech, the only Zambian corpus with recoverable
   identity: **12 speakers** with two or more sessions.

`scripts/corpus_zambian.py` implements the split. `scan_labelled` applies the
roster and the filename pattern; `scan_unlabelled` prepares everything else and
marks it `<unlabelled>`; `reject_unlabelled` refuses that sentinel wherever
identity is required. The guard is not ceremony. Unlabelled plans share one id,
so feeding them to a speaker split produces a single fake speaker with hundreds
of sessions, and **every downstream stage accepts it quietly** — the split
succeeds, `verify_disjoint` passes because the id really does appear on one side
only, PLDA trains, and the model is meaningless.

### Twelve speakers is one short of splittable

The scanner has been run end to end against real BembaSpeech audio. It builds
30.0 s recordings at 16 kHz from concatenated utterances, and the front-end
returns 2,363–2,644 frames of 60 dimensions at speech fractions of 79–88%, so
the material passes voice activity detection comfortably. Nothing about the
audio is the problem.

The speaker count is. `split_by_speaker` requires at least three speakers in
each of the development and evaluation parts, which at the default 0.2 fractions
means **thirteen speakers minimum** — verified by construction: 12 is refused,
13 yields 7/3/3. BembaSpeech has twelve.

So the only Zambian corpus with recoverable speaker identity **cannot be used as
a standalone corpus at all**. It cannot be split into training, development and
evaluation parts, by one speaker.

This is not fatal, but it removes a choice. BembaSpeech can only serve as an
**evaluation set**, with training done elsewhere — which is the borrowed-backend
architecture above, now the only way to use Zambian data rather than merely the
preferable one. Used that way the twelve speakers are all evaluation speakers,
and no split of them is needed.

### Language is not the gap

One assumption behind this survey deserves stating, because it is wrong in a way
that matters. The search for Zambian-*language* corpora treats language as the
axis of population match. For this system it is close to the least important
one. Speaker discrimination here rests on vocal tract and channel properties
rather than lexical content, cross-lingual PLDA is routine, and §5 measured the
factors that do dominate: duration at +0.23 from 30 s to 5 s, bitrate at +0.001
to +0.020. Language mismatch is smaller than either.

**English is also Zambia's official language**, so English-language material is
realistic for casework there rather than a substitute for it. What is wrong with
LibriSpeech is not that it is English but that it is US and UK speakers reading
audiobooks in studio conditions — wrong population, wrong speaking style, wrong
bandwidth.

That reframes the requirement. What the back-end needs is many speakers whose
voices resemble the deployment population, in any language. **AfriSpeech-200**
fits better than anything else surveyed: 2,463 speakers with a genuine
`speaker_id`, 200 hours of African-accented English across 13 countries, and a
10.7 s mean clip — twice Common Voice's, which matters because it permits a 30 s
condition without concatenating across sessions. Zambia is not among its
countries, but Malawi is, and Malawian Chichewa is the same language as Zambian
Nyanja; Zimbabwe, Botswana and Tanzania are also present. It is 61.9 GB, though
served as shards, so a country subset is possible where Common Voice's single
archive allows none.

### Open questions

The count of Common Voice contributors with enough material to be usable — as
opposed to the 1,523 who submitted anything — still needs `validated.tsv`, which
ships only inside the full corpus download. The same question applies to
AfriSpeech-200's 2,463. Whether BembaSpeech overlaps Zambezi Voice's other
languages is now moot: those languages contribute no labelled speakers to
overlap with.

---

## 9. More speakers: the question in §7 answered

§7 refuted "the model was too small" and left "the channel destroyed the
information" unestablished, because no model trained on 125 speakers can measure
what the channel preserved. This section supplies the missing corpus and answers
it.

### The corpus

LibriSpeech ships `train-clean-100` and `train-clean-360` as separate
partitions. Streaming the second was stopped at 380 of its 921 speakers — 23 GB
over a 5.4 Mbit/s link is nine hours, and the partial fetch already sufficed.
Pooled with the existing partition and verified disjoint by identifier:

| | Speakers | Usable (≥2 sessions) |
|---|---:|---:|
| `train-clean-100` | 251 | 209 |
| `train-clean-360` (partial) | 380 | 301 |
| **Pooled** | 631 | **510** |

Speaker identifiers are bare numbers, unique only within the partition that
issued them, so `scan_corpora` refuses to merge two roots that reuse one rather
than silently placing two people under a single label. The two partitions
overlap in zero identifiers, checked rather than assumed.

Split 306 train / 102 development / 102 evaluation. Model
`ivec-plda-d5023efe82508a33`, trained at **128 components and rank 100** — the
baseline configuration, deliberately unchanged.

### One variable moves

Rank 100 sits below the new LDA ceiling of 305, so `min(100, 305) = 100` and the
transform dimension is 100 in both models. Nothing about the architecture, the
subspace size, the channel or the front-end differs. The only change is that
LDA and PLDA were estimated from 306 speakers instead of 125.

That makes this the cleanest available test of §7's remaining question, and a
sharper one than raising the rank would have been: any movement is attributable
to estimation quality alone.

### The comparison had to be restricted, and the reason generalises

Two models trained on different splits have different held-out sets, and the
overlap is not benign:

| | |
|---|---:|
| baseline evaluation speakers inside the pooled model's **training** set | **23 of 42** |
| pooled evaluation speakers inside the baseline's **training** set | 15 of 102 |

Scoring both models on either model's own split would have rewarded whichever
had already seen the speakers — and by a wide margin, since more than half the
baseline's evaluation set is pooled-training material. Nothing in the pipeline
would have objected: `verify_disjoint` checks a split against itself, and both
splits are internally disjoint. **The leakage is between models, which nothing
was checking.**

Both models were therefore scored on the **35 speakers held out by both**,
listed in `data/reports/common_holdout_speakers.txt` and recorded in the report.
`compare_capacity.py` takes `--evaluation-speakers` for this, which makes the
evaluation population an explicit auditable input rather than something derived
implicitly from whichever split ran last.

### Result: more speakers is better, and the gain depends on the channel

Paired over the 35 common speakers. Negative means the 306-speaker model
discriminates better.

Intervals are BCa at B = 2000; p-values are derived through the same correction,
so an interval excluding zero and a p-value below 0.05 agree by construction.
Holm-Bonferroni is applied over the six cells, which are one family tested on
one dataset.

| Condition | Dur. | 125 spk | 306 spk | Difference [95% CI] | p | Holm |
|---|---:|---:|---:|---|---:|:---:|
| 12.2 kbit/s, clean | 30 s | 0.352 | **0.248** | −0.104 [−0.168, −0.052] | 0.0010 | **✓** |
| 12.2 kbit/s, clean | 15 s | 0.400 | 0.321 | −0.079 [−0.118, −0.035] | 0.0025 | **✓** |
| 12.2 kbit/s, clean | 5 s | 0.566 | 0.494 | −0.071 [−0.112, −0.022] | 0.0036 | **✓** |
| 12.2 kbit/s, babble 20 dB | 30 s | 0.348 | 0.294 | −0.054 [−0.094, −0.021] | 0.0018 | **✓** |
| 12.2 kbit/s, babble 20 dB | 15 s | 0.412 | 0.376 | −0.036 [−0.076, −0.0003] | 0.0480 | ✗ |
| 12.2 kbit/s, babble 20 dB | 5 s | 0.517 | 0.494 | −0.023 [−0.072, +0.021] | 0.3079 | ✗ |

Five of six exclude zero, **four survive the multiplicity correction**, and
**all six point the same way**. The cell Holm drops is borderline on every
measure — its upper bound is −0.0003 and its p-value 0.048 — and nothing here
leans on it.

> Earlier versions of this table reported percentile intervals at B = 300 and no
> multiplicity correction, giving four of six excluding zero. The correction in
> §14 moved one cell into significance and Holm moved it back out. The
> conclusion is unchanged and now rests on four cells rather than four
> uncorrected ones.

Set against §7, where more parameters on the same corpus made **all six** cells
significantly worse, the pair of experiments brackets the question:

- capacity was not the constraint — adding it hurt
- the corpus was — adding speakers helped

**The channel was therefore not destroying the speaker information.** It was
preserving information the model could not reach, because 125 speakers are not
enough to learn what distinguishes people.

### The gain is conditional, and that is the more useful finding

The improvement is largest where the channel is gentlest and falls monotonically
as it worsens: −0.104 at 30 s clean down to −0.023 at 5 s in babble. Read across
the row, both explanations are true in different regimes. In clean,
long-duration conditions the corpus was the limit and more speakers recover real
information. At five seconds under babble the channel genuinely has destroyed
it, and no quantity of training data brings it back.

"Channel or corpus" was the wrong question. It is the corpus where the channel
is mild and the channel where it is harsh, and the crossover sits inside the
operating range this system is meant for.

### Standalone verdict: the falsification is withdrawn

Evaluated on its own 102 held-out speakers — a legitimate standalone evaluation,
and 102 speakers rather than 42 narrows every interval.

| Condition | Dur. | `C_llr_min` | Interval | EER | Verdict |
|---|---:|---:|---|---:|---|
| clean | 30 s | **0.276** | [0.188, 0.342] | **7.89%** | inconclusive |
| clean | 15 s | 0.349 | [0.262, 0.418] | 9.81% | inconclusive |
| clean | 5 s | 0.539 | [0.463, 0.598] | 16.87% | inconclusive |
| babble 20 dB | 30 s | 0.295 | [0.207, 0.362] | 9.15% | inconclusive |
| babble 20 dB | 15 s | 0.370 | [0.286, 0.435] | 10.95% | inconclusive |
| babble 20 dB | 5 s | 0.514 | [0.445, 0.568] | 15.96% | inconclusive |

Against §4's sweep, the best cell improves from 0.343 [0.219, 0.429] at 10.86%
EER to **0.276 [0.188, 0.342] at 7.89%** — a 27% relative reduction in equal
error rate.

**§5's falsification at five seconds no longer holds.** Both five-second cells
now have lower bounds below 0.50 (0.463 and 0.445), so under the rule fixed in
§3 they are inconclusive rather than falsified. The finding was a property of a
model trained on 125 speakers, not of the channel, and it does not survive a
better-trained one.

**No cell reaches supported.** The best upper bound is 0.342 against a threshold
of 0.30. H1 has moved from *falsified at short duration* to *inconclusive
everywhere* — the negative finding is withdrawn without the positive one being
established, and saying so plainly is the whole point of an interval-based rule.

### What follows

The ceiling is now 305 and the rank is 100, so the constraint has moved: the
subspace, not the corpus, is what caps the model. §7 showed rank 200 truncated
to 124 and hurting on 125 speakers; on 306 speakers it would pass through
untruncated. That is the natural next experiment and it is cheap — the corpus is
already on disk.

The corpus route is not exhausted either. 510 speakers came from a fetch stopped
at 41%; completing it reaches roughly 761, and the trend across §7 and §9 gives
no reason to think 306 is where the returns stop.

> **A scope-correction note stood here and has been withdrawn.** It claimed,
> on the strength of §12's first version, that architecture rather than corpus
> was the dominant constraint, because a published x-vector system reached
> `C_llr` below 0.09 "trained on 91 speakers". That system's *extractor* was
> trained on approximately 6,000 speakers; 91 trained its back-end only. The
> corrected benchmark supports this section rather than qualifying it — speaker
> count binds in the extractor too. See §12.
>
> The two intervals this section reports at five seconds are nonetheless
> affected by the bootstrap correction in §14, which is a separate matter and
> does change a verdict here.

The honest summary is that the resource position for Zambian forensic speech is
considerably worse than the hour counts suggest. Roughly 390 hours of Zambian
audio are available to the unsupervised stages — and that number is real, since
those stages need no labels — while **twelve speakers** are available to
everything that determines discrimination. Two hundred and thirty-two hours of
Bemba speech exist across BIG-C and Zambezi Voice in which nobody can say who is
speaking.

**That gap is itself a finding.** These corpora were built for speech
recognition, where transcribed hours is the figure of merit and speaker identity
is incidental metadata. Speaker recognition needs the quantity nobody was
optimising for, and for Zambian languages it has not been collected. Reporting
that is more useful than working around it quietly.


---

## 10. The spoofing countermeasure, and what it cannot see

The validity gate decides whether acoustic evidence reaches fusion at all, and
until now it applied a policy to a score nothing produced. This section supplies
a detector and measures how far it can be trusted.

### Attacks, generated rather than downloaded

No public corpus of spoofed Zambian speech exists, so the attacks are generated
from the BembaSpeech recordings themselves — genuine and spoofed material differ
in what was done to them and not in who was speaking or what language they
spoke. Four families, in `viflap/analysis/spoof/attacks.py`:

| Attack | What it discards |
|---|---|
| `lpc_noise` | excitation replaced by white noise |
| `lpc_pulse` | excitation replaced by a pulse train at fixed F0 |
| `phase_randomised` | phase discarded, frame magnitudes kept |
| `oversmoothed` | excitation kept genuine, formant trajectory smoothed |

Three attack the source and one attacks the filter, so a detector cannot pass a
cross-attack test by learning a single axis. Every attack matches the energy of
the frame it replaces: an attack detectable by loudness would let the detector
separate the classes without learning anything about synthesis, and the reported
error rate would then measure a level difference this module introduced.

These are **vocoder-class** attacks. A detector that separates them has not been
shown to detect neural synthesis, and every number below is correspondingly a
lower bound on the generalisation problem rather than an estimate of it.

### A stability trap worth recording

The first implementation of `oversmoothed` averaged the LPC *polynomial*
coefficients across frames. That does not preserve stability — two stable
all-pole filters can average to one with poles outside the unit circle, and the
synthesis filter then diverges to NaN.

The failure is **data-dependent**, which is what makes it worth writing down. It
did not appear on the Bemba recording used to check the attacks by hand; it
appeared on a synthetic test signal, after the faulty attack had already gone
into a training run. Smoothing is now done on reflection coefficients, whose
magnitudes are below one exactly when the filter is stable and whose mean is
therefore stable by construction, with the polynomial rebuilt afterwards.

### Result: it works in domain, and not much beyond it

Trained on 8 speakers, evaluated on 4 held-out speakers, 64 genuine and 64
spoofed recordings per condition.

| | EER |
|---|---:|
| unseen speakers, **seen** attacks | 16.41% |
| unseen speakers, **unseen** attacks (mean) | **34.77%** |
| **generalisation gap** | **+18.36 points** |

Held out one family at a time:

| Family held out | EER when unseen |
|---|---:|
| `lpc_pulse` | 1.56% |
| `oversmoothed` | 37.50% |
| `lpc_noise` | 50.00% |
| `phase_randomised` | 50.00% |

Two families sit exactly at chance. This is the known central weakness of
spoofing countermeasures, measured rather than asserted, and it is the empirical
justification for `GatePolicy` returning `INDETERMINATE` on out-of-domain frames
instead of trusting the score. A deployment assuming this detector generalises
would be wrong by eighteen points.

### The blindness has a mechanism

Scoring held-out speakers with the model trained on *all four* families shows
the attacks are not equally visible:

| | Mean score | Separation from genuine |
|---|---:|---:|
| genuine | +0.55 | — |
| `lpc_pulse` | −19.75 | +9.58 sd |
| `oversmoothed` | −3.85 | +5.22 sd |
| `lpc_noise` | −2.15 | +3.15 sd |
| `phase_randomised` | **+0.28** | **+0.46 sd** |

**Phase randomisation is invisible to this detector even when it trains on it.**
The cause is the feature representation, not the modelling: the countermeasure
uses linear-frequency cepstral coefficients, computed from the **magnitude**
spectrum, and phase randomisation preserves frame magnitudes almost exactly —
their correlation with the original is 0.93, and 0.996 in the log domain. The
features discard precisely the information the attack destroys.

`CountermeasureConfig` already refuses a mel filterbank, on the reasoning that
mel spacing "averages away the upper-spectrum artefacts that distinguish
synthetic speech". That reasoning is about the magnitude spectrum and is
correct as far as it goes; it does not reach an attack that leaves the magnitude
spectrum alone.

This single blindness accounts for most of the in-domain error: a quarter of the
spoofed material is indistinguishable from genuine at any threshold, which puts
a floor of roughly 12.5% under the seen-attack figure of 16.41%.

### Retrained on 300 speakers, with Bemba held out entirely

The result above was trained on **12 speakers**, which was a design error. It
reused a subset built to validate the *speaker-recognition* scanner, where two
sessions per speaker is a real requirement. A countermeasure does not model
speaker identity: it needs many voices, not many sessions of a few. And vocoder
artefacts are properties of the synthesis rather than of the language, so
restricting the genuine class to Bemba bought nothing while costing 25x the
speaker diversity available on disk.

Retrained on **300 LibriSpeech speakers, one recording each**, with the whole of
BembaSpeech held out. The evaluation is now unseen speakers, unseen **language**,
and unseen attack family.

| | 12 Bemba speakers | 300 English speakers |
|---|---:|---:|
| Seen attacks | 16.41% | 19.14% |
| Unseen attacks, mean | 34.77% | **24.61%** |
| **Generalisation gap** | +18.36 | **+5.47** |

Held out one at a time:

| Family | 12 speakers | 300 speakers |
|---|---:|---:|
| `lpc_pulse` | 1.56% | 4.17% |
| `lpc_noise` | 50.00% | **13.54%** |
| `oversmoothed` | 37.50% | 28.12% |
| `phase_randomised` | 50.00% | **52.60%** |

**Speaker diversity buys generalisation, and buys it cheaply.** The gap falls
from +18.4 points to +5.5, and `lpc_noise` moves from exactly chance to 13.5%.
Nothing about the model changed except how many different voices it saw.

**The seen-attack figure gets worse, and that is the price of a harder test.**
19.14% against 16.41% is not a regression: the earlier number was Bemba-trained
and Bemba-evaluated, while this one is trained on English and evaluated on Bemba.
Roughly three points is what cross-lingual transfer costs this detector — which
is a useful number in itself, and the question a Zambian deployment actually
asks of a detector trained on whatever data exists.

**The phase blindness is confirmed as structural, not statistical.** Twenty-five
times the training speakers moved `phase_randomised` from 50.00% to 52.60% —
that is, not at all. A representation computed from magnitude spectra cannot
detect a manipulation that preserves magnitude spectra, and no quantity of
training data changes that. The controlled comparison is stronger evidence for
§10's mechanism than the original single-condition result was.

### What follows

**A phase-sensitive feature is required, not optional.** Group delay, modified
group delay or constant-Q cepstral coefficients would give the detector access
to what LFCCs throw away. Until one is added, magnitude-preserving manipulation
is undetectable by this component at any operating point, and the gate's
uncertain band is the only thing standing between such a recording and
admission.

The absolute figures are also limited by scale — 12 speakers, 192 genuine
recordings, 32 mixture components — so a properly resourced detector would score
better in domain. The **gap** is the durable finding; the level is not.

---

## 11. What independence costs: the overstatement deliverable

The proposal names this an explicit deliverable, and `fusion/overstatement.py`
gives the reason it deserves that status: even if the dependence-corrected
models gain little over naive summation, *how badly does the standard method
mislead* is answered either way — and naive summation is what deployed
multimodal forensic systems actually use.

### A simulation, and what stops it being arbitrary

Answering this with real data needs incidents carrying several streams with
known linkage truth. No such corpus exists for this setting. But the question is
about **the method** rather than about Zambia — what does independence cost when
streams are correlated — so it is answerable wherever the inputs are realistic.

Three things keep it grounded:

**The acoustic marginal is measured, not invented.** Same-source and
different-source log-LRs are resampled from the 25,088 real evaluation trials in
`calibration_scores.npz`, calibrated as the system calibrates — fitted on
development speakers, applied to evaluation speakers. Same-source mean **+2.24**
against different-source **−3.92**, in nats. Draws are mapped onto that empirical
distribution by quantile, so its asymmetry and its tails survive; fitting a
normal instead would smooth away exactly the region an overstatement study lives
in.

> **These two numbers were wrong until now, and wrongly in a way this section
> had already diagnosed elsewhere.** They read "+1.98 against −12.44", which are
> the means of the **unbounded** calibrator output — precisely the quantity the
> correction block below says this section stopped using. When the result table
> was recomputed with `calibrate` in place of `transform`, this sentence was not,
> so the paragraph asserting that the marginal is what a deployment would emit
> was quoting a marginal running to −16.1 log₁₀ against bounds of ±[−2.35, 3.64].
> The bounded means are +2.24 and −3.92 nats (+0.97 and −1.70 log₁₀). **No
> reported figure moves**: every number in the result table was already computed
> from the bounded marginal, which is why the error survived — it was in the
> prose only.

**The dependence mechanism is the documented one.** `EvidenceStream` states the
streams "are *not* conditionally independent of one another — the same operator
running the same operation is the common cause of all of them". The simulation
induces dependence that way: a per-incident latent factor standing for the
operation, shared across streams, with each stream's idiosyncratic variation on
top. Not a correlation knob attached to independent draws — the causal structure
the module describes.

**The correlation is swept, not assumed.** Nobody knows its true value, so a
curve over plausible values says more than one number from a guess.

### Result

> **This section was rewritten after external review.** The first version had
> two defects that between them invalidated it. It fed the simulation the
> **unbounded** calibrator output — `transform` rather than `calibrate` — so the
> acoustic marginal ran to −16.1 log₁₀ where the system may report only −2.35,
> and 60.6% of the trials involved were ones the system would have clipped. And
> it compared naive summation against a dependence model only, so what it called
> "the cost of independence" confounded dependence with calibration: the naive
> model is an *unweighted sum with no fitted parameters*. Both are fixed below,
> and the conclusion changes.

4,000 incidents per level, held-out evaluation, three streams, bounded marginals.
Three fusion models rather than two — the middle one still assumes independence
but fits its weights, which is what separates the two effects.

> **This table now carries intervals, and the previous version's point estimates
> are superseded.** It reported one seed per level: 0.148 / 0.077 / 0.105 at
> ρ = 0, rising to 0.538 / 0.335 / 0.491 at ρ = 0.8. Every one of those values
> falls inside the intervals below, so nothing is retracted — but they were
> quoted to three decimals from a single draw, and the intervals show how little
> that precision meant. Artefact: `data/reports/overstatement.json`.

**40 independent replicates per level**, with the acoustic marginal **resampled
over its 42 evaluation speakers** in each. Both sources of variation matter and
only one of them is about the simulation: holding the marginal fixed would treat
an estimate from 42 speakers as the population, which is the error §2 rejects for
every other interval in this document. Intervals are 95% Monte-Carlo over
replicates — the replicates are genuinely independent, so their spread *is* the
sampling distribution and bootstrapping them would only add noise.

| ρ | naive sum | **linear-logistic** (independent, calibrated) | Gaussian latent (dependence) | Band changes |
|---:|---|---|---|---:|
| 0.0 | 0.158 [0.032, 0.316] | **0.082 [0.019, 0.154]** | 0.100 [0.023, 0.155] | 87.8% |
| 0.2 | 0.238 [0.066, 0.482] | **0.149 [0.057, 0.257]** | 0.190 [0.082, 0.273] | 81.9% |
| 0.4 | 0.331 [0.113, 0.637] | **0.214 [0.105, 0.335]** | 0.279 [0.154, 0.365] | 70.9% |
| 0.6 | 0.437 [0.171, 0.821] | **0.275 [0.154, 0.405]** | 0.363 [0.226, 0.450] | 61.6% |
| 0.8 | 0.555 [0.240, 1.000] | **0.331 [0.203, 0.463]** | 0.442 [0.300, 0.531] | 65.2% |

Values are `C_llr`; lower is better.

**The intervals are enormous, and that is the first finding.** At ρ = 0 the naive
sum spans [0.032, 0.316] — a factor of ten. Almost all of that width comes from
resampling 42 speakers, not from the 4,000 simulated incidents, and it means the
marginal figures in the old table were about as precise as their third decimal
suggested they were not. Any statement of the form "naive summation costs 0.148"
was never supportable from this experiment.

### The comparison had to be paired, exactly as in §7

Those marginal intervals overlap almost completely, and reading that as "the
models are indistinguishable" would manufacture a null — the same error §7 was
written to avoid. Every arm within one replicate sees the *same* simulated
incidents and the *same* resampled marginal, so nearly all of that width is
common to them and cancels in a difference. Differenced within replicate,
`gaussian_latent − linear_logistic`, positive meaning the dependence model is
**worse**:

| ρ | Gaussian latent − linear-logistic [95% CI] | Excludes zero |
|---:|---|:---:|
| 0.0 | +0.019 [+0.001, +0.049] | ✓ |
| 0.2 | +0.041 [+0.013, +0.075] | ✓ |
| 0.4 | +0.065 [+0.023, +0.112] | ✓ |
| 0.6 | +0.088 [+0.030, +0.157] | ✓ |
| 0.8 | +0.110 [+0.028, +0.183] | ✓ |

**All five exclude zero, and the penalty grows with dependence.** The dependence
model is significantly worse than the independence-assuming model that fits its
weights, at every level of dependence tested — including the levels where the
correction is supposed to earn its place. The previous version asserted this from
point estimates; it is now measured.

### Misspecification makes it worse, not better

The obvious objection to the above is that the generative process was a Gaussian
latent factor and the dependence model is a Gaussian latent-factor model, so the
comparison was the correction's best case. The honest operational case is
misspecification: real streams fail *together*, and a Gaussian copula has no
parameter for that at any correlation.

Rerun with the same linear correlation imposed through a **Student-t copula**
(ν = 4) — one chi-square draw per incident, shared across its streams, which is
what produces joint tail dependence. The fitted models are unchanged, so this arm
is misspecified by construction. Artefact:
`data/reports/overstatement_tcopula.json`.

| ρ | naive sum | **linear-logistic** | Gaussian latent | Latent − linear [95% CI] |
|---:|---|---|---|---|
| 0.0 | 0.190 | **0.119** | 0.166 | +0.047 [+0.012, +0.090] |
| 0.2 | 0.268 | **0.176** | 0.240 | +0.064 [+0.025, +0.120] |
| 0.4 | 0.355 | **0.231** | 0.312 | +0.081 [+0.038, +0.157] |
| 0.6 | 0.453 | **0.284** | 0.382 | +0.099 [+0.041, +0.183] |
| 0.8 | 0.562 | **0.334** | 0.454 | +0.119 [+0.043, +0.214] |

**Every arm degrades under misspecification, and the dependence model degrades
fastest.** Its penalty relative to the calibrated independence model widens at
every level — most sharply at ρ = 0, where it more than doubles from +0.019 to
+0.047. Tail dependence hurts the model that is supposed to be modelling
dependence more than it hurts the model that ignores it.

That is not the result this arm was added to find. It was added because
correct-specification-by-construction made the dependence model's defeat
uninformative; the expectation was that misspecification would narrow the gap or
reverse it. It widened it.

**Calibration buys more than the dependence correction does, at every level of
dependence.** Fitting weights while still assuming independence takes `C_llr`
from 0.148 to 0.077 at ρ = 0; modelling the dependence takes it only to 0.105.
At ρ = 0.8 — where the dependence correction should be at its most valuable —
the independence-assuming model still wins, 0.335 against 0.491.

That is the finding, and it is not the one the first version reported. What was
labelled "the cost of assuming independence" was mostly **the cost of not
fitting anything**. Against a fair baseline, the independence assumption costs
far less than the first version claimed, and the dependence machinery does not
pay for itself on this simulation.

**The dependence model does beat the naive sum** at every ρ, so it is not
useless — it is simply dominated by a simpler model. An external review of this
section, working from the unbounded marginals, found it scoring *worse* than
naive; that result was an artefact of the tail the bounding removes, and does
not survive the fix.

The band-change column is now hard to read as an indictment of independence: it
is **highest at ρ = 0**, where the streams are independent by construction and
naive summation is exactly correct. At that point the disagreement is the
dependence model's error, not the baseline's.

### The conclusion survives a marginal four times stronger

The caveat above — that `calibration_scores.npz` is the superseded 42-speaker
baseline — is now discharged rather than merely noted. §22's borrowed extractor
persists its per-trial scores, so the same simulation can be driven from the
best acoustic marginal this project has, over 102 evaluation speakers instead of
42. Artefacts: `data/reports/overstatement_ecapa.json` and
`overstatement_ecapa_tcopula.json`.

| Acoustic marginal | Same-source mean | Different-source mean | Speakers |
|---|---:|---:|---:|
| §4/§5 baseline (used above) | +2.24 | −3.92 | 42 |
| **§22 ECAPA** | **+9.48** | **−6.51** | **102** |

That is roughly four times the separation, and it is the regime a deployment
built on §22 would actually be in. 40 replicates, marginal resampled over
speakers, as above.

| ρ | naive sum | **linear-logistic** | Gaussian latent | Latent − linear [95% CI] |
|---:|---:|---:|---:|---|
| 0.0 | 0.008 | **0.007** | 0.140 | +0.132 [+0.013, +0.306] |
| 0.2 | 0.023 | **0.011** | 0.201 | +0.190 [+0.035, +0.438] |
| 0.4 | 0.051 | **0.020** | 0.288 | +0.268 [+0.087, +0.546] |
| 0.6 | 0.097 | **0.038** | 0.389 | +0.351 [+0.152, +0.611] |
| 0.8 | 0.166 | **0.063** | 0.500 | +0.437 [+0.219, +0.686] |

**The conclusion does not merely survive, it sharpens by a factor of four.** The
dependence model's penalty was +0.019 to +0.110 on the weak marginal and is
+0.132 to +0.437 on the strong one, every interval still excluding zero. Whatever
was wrong with reporting the old figures, under-stating this was not it.

And under misspecification the same thing happens again, larger:

| ρ | naive sum | **linear-logistic** | Gaussian latent | Latent − linear [95% CI] |
|---:|---:|---:|---:|---|
| 0.0 | 0.029 | **0.015** | 0.220 | +0.205 [+0.073, +0.485] |
| 0.2 | 0.053 | **0.025** | 0.310 | +0.285 [+0.080, +0.545] |
| 0.4 | 0.085 | **0.035** | 0.392 | +0.357 [+0.168, +0.571] |
| 0.6 | 0.133 | **0.052** | 0.479 | +0.427 [+0.197, +0.667] |
| 0.8 | 0.196 | **0.073** | 0.575 | +0.502 [+0.229, +0.737] |

### Naive summation stops being the villain

The more interesting change is in the column nobody was watching. On the weak
marginal the naive sum cost 0.158 at ρ = 0 against the calibrated model's 0.082 —
roughly double, which is what made "the cost of not fitting anything" the
headline. On the strong marginal it costs **0.008 against 0.007**.

That is not a small difference in a small number; it is the effect disappearing.
Fitting weights buys almost nothing once the marginals separate well, because
there is very little left for a calibration to repair. Even at ρ = 0.8 the naive
sum reaches 0.166 where the *dependence model* reaches 0.500 — the method that
assumes the streams are independent and does not fit anything beats the method
built to model their dependence, by a factor of three, at the correlation where
the dependence correction should be most valuable.

**This inverts the practical advice this section has been giving.** The earlier
reading was that calibration matters more than dependence modelling. On a
marginal representative of the current system, calibration matters *less* than it
did and dependence modelling is actively harmful — so for a deployment built on
§22's extractor, the defensible fusion is the simplest one available.

### What this still does not establish

The generative parameters are unchanged and still mostly assumed: only the
acoustic marginal is real, and §13 gives reason to think the behavioural stream
is weaker than the 0.75 factor used here rather than stronger. Making that
marginal *more* realistic would widen the gap between streams, and nothing in
these two runs says what that does to the ordering.

Nor does a stronger marginal make the simulation less of a simulation. §11's
opening still governs: no figure here is a result, and the reason to run it on
the ECAPA marginal was to find out whether the *ordering* was an artefact of a
weak one. It was not.

### The "weaker streams" were never weaker, and it reached one arm only

> **A defect in this section's construction, found while trying to act on its own
> caveat.** The remaining item was that the behavioural stream is set at 0.75 of
> acoustic separation where §13's operating point says it should be worse than
> acoustic. Attempting to lower the number established that the number does not
> do what the section says it does.

`_ASSUMED_STRENGTH` multiplies the unmeasured streams' log-LRs by 0.75 and 0.50,
documented as making them "weaker than the measured one" and commented at the
multiplication as reducing separation. **Multiplying log-LRs by a positive
constant is monotonic.** The ranking of trials is untouched, so `C_llr_min` is
exactly unchanged and discrimination is identical at every setting:

| Scale | `C_llr` | `C_llr_min` |
|---:|---:|---:|
| 1.00 | 0.328 | **0.276** |
| 0.75 | 0.354 | **0.276** |
| 0.50 | 0.428 | **0.276** |
| 0.10 | 0.807 | **0.276** |

So all three simulated streams have **identical discriminability by
construction**, which is a far stronger assumption than the one the section
declared. What the constant actually varies is *confidence*: a scaled stream is
under-confident, which is a calibration defect.

**And a calibration defect reaches exactly one of the three arms.**
`LinearLogisticFusion` fits `w₀ + Σ wᵢlᵢ`, and `lᵢ → sᵢlᵢ` is the same model at
`wᵢ/sᵢ`, so refitting absorbs it. `GaussianLatentFusion` refits means and
covariances, and the Jacobian of a diagonal rescaling cancels between the
same-source and different-source densities. Only `NaiveIndependentFusion` adds
the values as given and has nothing to refit. Measured at 1,200 incidents:

| ρ | Strengths | naive sum | linear-logistic | Gaussian latent |
|---:|---|---:|---:|---:|
| 0.0 | 0.75 / 0.50 (as published) | 0.148 | **0.0766** | 0.105 |
| 0.0 | 1.00 / 1.00 | 0.130 | **0.0766** | 0.105 |
| 0.0 | 0.25 / 0.15 | 0.247 | **0.0766** | 0.104 |
| 0.6 | 0.75 / 0.50 (as published) | 0.416 | **0.2747** | 0.389 |
| 0.6 | 1.00 / 1.00 | 0.501 | **0.2747** | 0.389 |
| 0.6 | 0.25 / 0.15 | 0.374 | **0.2747** | 0.385 |

The linear column does not move at all — 0.0766 and 0.2747 to four decimals,
across a fourfold change in the constants. The latent column moves in the third
decimal, which is EM stopping rather than the property failing. The naive column
moves by up to a factor of two.

**Note the sign at ρ = 0.6.** Scaling the streams *down* makes the naive sum
**better** — 0.374 against 0.501 — because under-confidence partly offsets the
double-counting that dependence causes. The arbitrary constants were
accidentally compensating for the very effect this section exists to measure.

### What survives, and what does not

**The central finding survives, and survives cleanly.** Every comparison between
`gaussian_latent` and `linear_logistic` — the paired differences, all five
excluding zero, on both copulas and both marginals — is between two arms that are
*both* invariant to these constants. Nothing in that result depends on them, and
the fact that it holds at 0.75/0.50 and at 1.00/1.00 and at 0.25/0.15 is a
robustness check the section did not know it had run.

**Every statement about the naive sum is contaminated.** That includes the
framing this section led with — that what the first version called "the cost of
assuming independence" was mostly "the cost of not fitting anything". Part of
what it was measuring is the cost of being handed deliberately under-confident
inputs, which is neither.

**The comparison the section should have been making is between the two fitted
arms**, and that one was sound all along. The naive sum belongs in the table as a
floor, not as a baseline anything is measured against.

### What this does not fix

The constants are **kept**, not corrected, because every published figure above
was produced with them and changing them silently would make the section
unreproducible. The docstring and the comment now say what they do.

**Making the behavioural stream genuinely weaker is still undone**, and it is now
clear it needs a different mechanism: a stream with less discrimination, not a
stream with the same discrimination expressed less confidently. Reducing the
quantile-mapped separation — drawing that stream's marginal from a distribution
with more overlap — would do it. Nothing here measures what that does, and on the
evidence above it is the only version of the change that could move the result.

### A behavioural stream that is genuinely weaker, and the one cell it changes

The subsection above establishes that `_ASSUMED_STRENGTH` cannot make a stream
less informative. `weaken` can: it slides the same-source marginal toward the
different-source one by a fraction of the gap between their means, leaving both
spreads alone, so the two class distributions genuinely overlap more and the ROC
moves. Artefacts: `overstatement_weak_behavioural.json` and
`..._tcopula.json`.

**The setting is derived, not chosen.** §13's forensic operating point puts
authorship at `C_llr` 0.54 for 500-token transcripts, and Ishihara reports those
LRs as well calibrated, so `C_llr ≈ C_llr_min` and 0.54 can be read as
discrimination. Sweeping the parameter against the measured marginal:

| Discriminability | Behavioural stream's own `C_llr_min` |
|---:|---:|
| 1.00 (equal to acoustic) | 0.357 |
| 0.80 | 0.479 |
| **0.70** | **0.541** |
| 0.60 | 0.596 |

0.70 lands on §13's figure to three decimals, so that is the value used. The
behavioural stream is now materially worse than the acoustic one — 0.541 against
0.357 — which is what §13 said it should be and what the section had never
actually implemented.

| ρ | naive sum | **linear-logistic** | Gaussian latent | Latent − linear [95% CI] |
|---:|---:|---:|---:|---|
| 0.0 | 0.262 | **0.115** | 0.140 | +0.025 [+0.002, +0.066] |
| 0.2 | 0.358 | **0.190** | 0.236 | +0.046 [+0.012, +0.091] |
| 0.4 | 0.461 | **0.254** | 0.320 | +0.065 [+0.018, +0.113] |
| 0.6 | 0.575 | **0.306** | 0.386 | +0.080 [+0.017, +0.145] |
| 0.8 | 0.697 | **0.346** | 0.419 | **+0.073 [−0.013, +0.143]** |

**Four of five still exclude zero. The fifth does not, and it is the first cell
in this section's history where the dependence model is not significantly
worse.** At ρ = 0.8 with a properly weak behavioural stream the interval spans
zero, so at the highest correlation tested the two models become
indistinguishable on this evidence.

Under the t-copula the same cell recovers, barely — +0.073 [+0.005, +0.150], an
interval whose lower bound is 0.005 — and the other four are unchanged in
direction. So the ρ = 0.8 result is not robust in either direction: it excludes
zero under misspecification and includes it under correct specification, which
is a statement about how little separates the two models there rather than about
which is better.

### What this does and does not change

**The section's conclusion holds where it was ever load-bearing.** The dependence
model is significantly worse across ρ = 0 to 0.6 under both copulas and under
every marginal tried — the 42-speaker one, §22's ECAPA one, and now a properly
weakened behavioural stream. That is a robustness range the section did not have
before.

**But the claim can no longer be stated for every ρ without qualification.** The
earlier tables read "all five exclude zero" and that was true of the marginals
then in use; it is not true once the behavioural stream is given the strength §13
says it has. The honest form is: *worse at low and moderate dependence,
indistinguishable at ρ = 0.8*.

**Everything gets worse, which is the unsurprising part.** A weaker stream means
less information, and all three arms degrade — the naive sum from 0.158 to 0.262
at ρ = 0, the calibrated model from 0.082 to 0.115. Nothing about the ordering
follows from that; it is reported so the columns are not read as comparable to
the tables above.

**The temporal stream is still at 1.0.** Only the behavioural one was weakened,
because it is the only one §13 gives an operating point for. The temporal and
device streams have no published figure to calibrate against and are left equal
to acoustic, which is still optimistic and still stated rather than hidden.

### What this still does not establish (original)

> **Two items previously listed here have been addressed** and are struck rather
> than deleted, because both changed what the section can claim. *"Point
> estimates from one seed, with no intervals"* — now 40 replicates with the
> marginal resampled over speakers, and the intervals turned out to be wide
> enough that the marginal figures should never have been quoted to three
> decimals. *"Correct specification by construction"* — now tested against a
> t-copula, which widened the dependence model's penalty instead of narrowing it.

**The marginal is from the superseded 42-speaker model.**
`calibration_scores.npz` holds the §4/§5 baseline evaluation, not §9's pooled
system and certainly not §22's borrowed extractor. So the acoustic stream feeding
this simulation is materially weaker than the system's current best — same-source
mean +2.24 nats against different-source −3.92, where §22's extractor separates
far better. Whether the conclusion survives a stronger marginal is untested and
is now cheap to test.

**Only the acoustic marginal is real**, and §13 gives reason to think the
behavioural stream is weaker than the 0.75 factor assumed here — its forensic
operating point puts `C_llr` at 0.54 for 500-token transcripts, which is worse
than the acoustic stream rather than 0.75 of it.

**Dominated is not useless.** The dependence model beats the naive sum at every
level under both copulas. The finding is that a simpler model beats it too, not
that modelling dependence is wrong in principle.

**The true ρ is unknown.** The curve says where the models separate; it does not
say where this system sits on it.

---

## 12. Benchmarks: what these numbers mean against published work

Everything above this section reports measurements without situating them
against anything. That is a defect, not a stylistic choice: an unbenchmarked
error rate cannot distinguish "this architecture performs as it should and the
task is hard" from "this implementation is broken". This section supplies the
comparison, and one of the three benchmarks contradicts a conclusion drawn
earlier in this document.

### The duration curve is typical, so the implementation is sound

Published i-vector/PLDA systems on telephone speech degrade from roughly **5%
EER at 30 s to 18% at 3 s**. §9's pooled model gives **7.89% at 30 s and 16.87%
at 5 s**.

Same shape, same magnitude, slightly worse at the long end and slightly better
at the short end where the durations differ. This is the least glamorous
benchmark and the most reassuring one: the system degrades the way its
architecture is documented to degrade, so the findings in §5, §7 and §9 describe
the method rather than a fault in this build of it.

### The countermeasure is unremarkable, and so is its failure

| | EER |
|---|---:|
| ASVspoof 2019 LA, GMM baseline (published) | **8.09%** |
| §10, seen attacks | 16.41% |

Roughly twice the published baseline, with causes that are known rather than
mysterious: twelve training speakers against ASVspoof's full training partition,
32 mixture components against 512, and one of four attack families structurally
invisible to LFCC features. §13 revisits the first of those.

The generalisation gap is the more interesting comparison, because §10's +18.4
points looked alarming in isolation:

| System | In-domain EER | Out-of-domain EER |
|---|---:|---:|
| ResNet-OC → ASVspoof2015 (published) | 2.29% | 26.30% |
| ResNet-OC → VCC2020 (published) | 2.29% | **41.66%** |
| ASVspoof 2021 best baseline, dev → eval | 0.55% | 9.26% |
| §10, seen → unseen attack family | 16.41% | 34.77% |

**Our gap is entirely typical.** The field's documented worst case is nearly
+39 points from a system an order of magnitude better in domain. §10's finding
is a normal instance of a known failure, which strengthens rather than weakens
the argument for the gate's out-of-domain rule: this is what countermeasures do
everywhere, not something peculiar to these four attacks.

### The C_llr benchmark, and a correction to how it was first read

> **This subsection was rewritten.** Its first version stated that E3FS3 was
> "trained on 91 speakers" and concluded that architecture rather than corpus
> was the dominant constraint. That was wrong, and wrong in a way that inverted
> the conclusion. The error and its consequences are set out below rather than
> removed, because the wrong version was acted on.

E3FS3 is an open x-vector/ResNet forensic voice comparison system, validated
under conditions reflecting real casework.

| Duration | GSM 06.10 | μ-law + G.729a | μ-law + G.723.1 |
|---:|---:|---:|---:|
| 5 s | 0.330 | 0.341 | 0.376 |
| **30 s** | **0.090** | **0.097** | **0.085** |
| 120 s | 0.083 | 0.064 | 0.067 |

**What actually trained that system.** From the paper's methods: the ResNet was
trained on *"approximately 1M recordings total from approximately 6k speakers
from the VoxCeleb2 database"*. The 91 speakers — 125 for the female set — trained
**LDA and PLDA only**.

So the comparison is not 306 speakers against 91. It is:

| | Extractor | Back-end |
|---|---:|---:|
| This system | 306 speakers, ~13 h | 306 speakers |
| E3FS3 | **~6,000 speakers, ~2,400 h** | 91 speakers |

Roughly **twenty times** the speakers in the component doing the discriminative
work. The first version of this section read a summary of the back-end training
set as though it were the whole training resource, and attributed a gap produced
by 6,000 speakers to a choice of architecture.

**The enrolment differs too, by more than was conceded.** E3FS3's known-speaker
side is ~120 s of **net** speech per recording, and most speakers contribute two
or three sessions — several hundred seconds of multi-session enrolment against
our single recording of ~23.5 s net.

**And the comparison took the wrong cell.** Their best case-specific condition
was set against our best cell, which is the selection effect this project's own
evaluation code warns about. The like-for-like figure is the multi-laboratory
benchmark both systems can be placed on:

| | forensic_eval_01 |
|---|---:|
| E3FS3α | **C_llr 0.208** |
| Best system in the Speech Communication special issue | 0.208 |
| This system, matched `C_llr` at 12.2 kbit/s clean 30 s | **0.336** |

**A factor of 1.6, not 3** — on a harder benchmark, against a system with 20×
the extractor training data and an order of magnitude more enrolment speech.

### What this does and does not do to §9

§9 concluded that the corpus was the binding constraint, on the evidence that
more parameters hurt (§7) and more speakers helped (§9). **That conclusion
stands, and this benchmark supports it rather than undermining it.**

The first version of this section claimed the opposite — that architecture was
the larger constraint — and stamped scope-correction notes onto §7 and §9 saying
so. Those notes were wrong and have been withdrawn. Corrected, the benchmark
says: the system that reaches 0.208 did so with ~6,000 speakers behind its
extractor. Speaker count is the constraint in the extractor as well as in the
back-end, which is §9's mechanism operating one stage earlier than §9 measured
it.

### What follows

**Not "train an x-vector extractor" — borrow one.** An extractor trained on this
project's 306 speakers would very likely underperform the current i-vector
system; the published recipes use VoxCeleb-scale data with augmentation, and
6,000 speakers is not reachable here. The move that is reachable is the one
E3FS3 itself made: **take a publicly pre-trained embedding extractor** — ECAPA-TDNN
or ResNet trained on VoxCeleb2 — **and train only LDA and PLDA on the 306
speakers already on disk.**

That is a data-transfer strategy rather than an architecture change, and it is
the correct reading of the benchmark: it imports the 6,000 speakers this project
cannot collect, into the stage where §9 showed speaker count binds.

Two caveats remain and are not resolved by the correction. Their codecs are real
(GSM 06.10, G.729a, G.723.1) where ours is a parametric model of unvalidated
severity — see the channel caveat that governs every number in this document.
And their material is Australian English casework against our LibriSpeech
audiobooks.

---

## 13. Benchmarking the behavioural stream

§12 benchmarked the acoustic stream and the countermeasure. The behavioural
stream was built and tested but never situated against the authorship
attribution literature it draws its methods from. Doing so changes what should
be expected of it, and makes the defect recorded in the test suite worse rather
than better.

### The text length problem

Published work on authorship attribution puts the **minimum sample length at
2,500 to 5,000 words** for reliable attribution, varying by language and genre —
2,500 for Latin prose, around 5,000 for English, German, Polish and Hungarian
novels.

`BehaviouralComparator` refuses comparisons below **40 words**.

That threshold is between sixty and a hundred and twenty-five times below the
literature's floor. The comparator will therefore return confident-looking
likelihood ratios on transcripts far too short for its own method, and nothing
in the code says so.

Two qualifications, neither of which rescues it:

**The literary figure is not the forensic one.** Forensic text comparison
routinely operates on SMS messages far shorter than a novel, and does produce
usable likelihood ratios. So 40 words is aggressive rather than absurd. But the
forensic literature reports that n-gram methods need "a fairly large amount of
data" merely to reach **C_llr below 0.75** — which is to say, even a
well-resourced forensic authorship system produces *weak* evidence, weaker than
the acoustic stream measured in §9.

**Script structure is not authorship attribution.** The move inventory and its
ordering characterise the *operation*, which is a different and probably easier
task than identifying a person, and the length requirements above do not
obviously transfer to it. The benchmark bites on the **idiolect** component —
function words, disfluencies, character n-grams — which is exactly the component
those methods were built for.

### The forensic operating point, and the floor it now sets

> **Resolved.** The first version of this subsection had only the literary
> figure and called 40 words indefensible against it, which was right about the
> verdict and wrong about the comparison. The literary number is the wrong
> denominator: novels are not fraud calls. The search that should have happened
> then has now been done, and the register-matched forensic figure is below.

Ishihara (2017) is the closest published operating point to this material.[^ish]
It evaluates an LR-based forensic text comparison system on **predatory chatlog
messages** from 115 authors — conversational, turn-taking, transcribed informal
speech, which is the nearest published register to a transcribed fraud call —
at four sample sizes. `C_llr`, best configuration of each procedure:

| Tokens | MVKD features | Character *N*-grams | Token *N*-grams | **Fused** | Fused EER |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.68 | — | 0.97 | **0.54** | — |
| 1000 | 0.53 | — | 0.90 | **0.42** | 0.10 |
| 1500 | 0.35 | 0.41 | 0.65 | **0.15** | 0.05 |
| 2500 | 0.21 | 0.57 | 0.57 | **0.20** | 0.02 |

Three things follow, and only the first is about a threshold.

**500 tokens is the smallest size anyone has published a number for**, and the
number is weak: 0.54 fused, 0.68 for the best single procedure. That is worse
than §9's acoustic `C_llr_min` of 0.276 at 30 s. So a floor set at 500 admits
evidence this literature itself calls weak, and refuses everything below the
weakest published point. That is the defensible place to put it, and it is where
it now is.

**The paper's own optimum is 1,500 tokens**, not 500 — the fused system improves
from 0.42 to 0.15 between 1,000 and 1,500, the largest step in the table, and
then *deteriorates* to 0.20 at 2,500. A deployment able to supply 1,500 words
should say so; the parameter is exposed for that.

**More data is not monotonically better**, which is worth recording because it
is the opposite of what §9 found for speakers. The character *N*-gram procedure
degrades from 0.41 to 0.57 going from 1,500 to 2,500 tokens, and the fused
system with it.

A second and more recent source puts a lower bound on the same question.
Barlow, Nini and Manino (2026) evaluate authorship verification across fifteen
corpora at 100–9,500 tokens.[^bnm] On the Bolt SMS/chat corpus — 100 to 500
tokens, the shortest material in that study — they report strong performance
across all three of their approaches from **200 tokens** of questioned and known
data upward. On Twitter data the worst condition they report, at 500 tokens of
known data, still held `C_llr` at or below 0.39.

So the forensic range across these two is roughly **200 to 1,500 tokens**
depending on method and register, and 500 sits inside it rather than at either
end. It is not conservative and it is not aggressive; it is the smallest figure
the register-matched study measured.

[^ish]: Ishihara, S. (2017). "Strength of linguistic text evidence: A fused
forensic text comparison system." *Forensic Science International*.
doi:10.1016/j.forsciint.2017.06.040. Background database 38 authors; ELUB
applied because unrealistically strong LRs were observed, which is the same
correction §15 applies here.

[^bnm]: Barlow, S., Nini, A. and Manino, E. (2026). "Normalisation-Based
Likelihood Ratio Estimation for Forensic Authorship Verification."
arXiv:2607.09501.

**One caveat on the comparison.** Ishihara counts whitespace tokens; `n_words`
here counts words as `tokenise` finds them. The two differ by roughly the
punctuation rate, so the floor is accurate to within perhaps ten percent, which
is far finer than the quantity warrants.

### One floor was the error, not one number

Raising 40 to 500 across the board would have been wrong in the other direction.
The published requirements were measured for authorship attribution — which is
what the **idiolect** term does — and this section already records that script
structure is a different task with no comparable literature. A single threshold
must therefore be either too high for the script term or too low for the
idiolect term. At 40 it was the latter.

`BehaviouralComparator` now carries two:

| | Value | Basis |
|---|---:|---|
| `MIN_WORDS_SCRIPT` | 40 | **none, and marked as having none.** Unchanged, because no length requirement for move-sequence comparison was found and inventing one that looked derived would be worse than keeping an admitted guess. Below it no profile is built at all. |
| `MIN_WORDS_IDIOLECT` | 500 | Ishihara (2017), above. Below it the idiolect term is **withheld**, not computed and reported small. |

Withheld matters. Below the floor `idiolect_log_lr` is zero because nothing was
measured, which is a different statement from zero because the evidence was
neutral, and the two licence different readings of the same total. The
distinction is carried on the score and in the diagnostics
(`idiolect_was_withheld`) rather than left to be inferred.

**And withholding it created a defect that had to be guarded.**
`suggests_shared_operation_not_speaker` fires when the script term substantially
outruns the idiolect term. With the idiolect held at zero, *any* script evidence
above the trigger satisfies that ratio automatically — so the flag would have
reported "one operation, more than one operator" on every short transcript, a
delegation finding manufactured entirely by the absence of the evidence meant to
contradict it. On the test fixtures a transcript compared **with itself** has a
script term of 2.46 against a trigger of log 10 = 2.30, so the unguarded flag
asserted delegation about a text and a copy of itself. The flag now returns
false whenever the idiolect was withheld, and the self-comparison case is the
regression test.

### The literature makes the n-gram defect worse

The test suite records that character n-grams are counted as *script* evidence
while measuring the author, and that their magnitude is large enough to invert
the delegation signal.

The authorship attribution literature sharpens this considerably: **character
n-grams are repeatedly reported as the single best-performing feature for
authorship attribution.** They encode lexical patterns, word order tendencies,
punctuation and capitalisation habits, and they are robust to the errors that
break word-level features.

So the most author-discriminative feature available is sitting in the one
component of this stream that is supposed to survive a change of author. That is
not a marginal misplacement — it puts the strongest speaker signal into the
speaker-independent term, which is why it dominates and inverts the result.

### What should be expected of this stream

Setting expectations from the literature rather than from hope:

| | Expectation |
|---|---|
| Idiolect on 40-word transcripts | unusable; far below any published operating point — **now refused rather than scored** |
| Idiolect at 500 words | weak: 0.54 fused, 0.68 best single procedure, on register-matched chatlog data |
| Idiolect at 1,500 words | 0.15 fused — comparable to the acoustic stream, and the cited study's own optimum |
| Script structure | unbenchmarked; not an authorship task, and no comparable literature was found |

This has a consequence for §11. That simulation assigned the behavioural stream
0.75 of the acoustic stream's separation, chosen to be conservative because the
acoustic stream is the only one measured. Against these benchmarks, **0.75 is
probably optimistic**: forensic authorship at `C_llr` 0.75 is materially weaker
than acoustic at `C_llr_min` 0.276. Re-running the overstatement curve with a
weaker behavioural marginal is cheap and should be done before any figure from
§11 is quoted.

### What follows

1. **Raise `min_words` and document the basis.** Forty words is not defensible
   against any published operating point. The figure chosen should cite what it
   is derived from.

   > **Done.** `MIN_WORDS_IDIOLECT = 500`, from Ishihara (2017) on predatory
   > chatlog messages — see "The forensic operating point" above for the table
   > it comes from and for why the answer turned out to be *two* floors rather
   > than a larger single one. `MIN_WORDS_SCRIPT` stays at 40 and is now
   > explicitly marked as having no citation, because it still has none. The
   > change also required a guard: withholding the idiolect term makes the
   > delegation flag fire on anything with script evidence, including a
   > transcript compared with itself.
2. **Move character n-grams to the idiolect term**, or restrict them to
   script-bearing spans. The literature says plainly which component they belong
   to.

   > **Done.** The n-gram term is now summed into `idiolect_log_lr`. The
   > characterisation test that pinned the defect is kept and re-scoped: it
   > still measures which way the n-gram evidence runs — one operator over one
   > script, by a margin twenty times the sequence term — and two new tests
   > assert the placement and that the delegation flag now behaves. It is not a
   > clean separation and is not reported as one: a scripted transcript's
   > n-grams carry the script's fixed wording too, so some operation-level
   > evidence moves across with them. Restricting to script-bearing spans is the
   > better fix and still needs the labelled data this section says does not
   > exist. Placing them where the literature puts them is defensible in the
   > meantime; leaving the strongest authorship feature inside the
   > author-independent term was not.
3. **Report the two components separately in any output**, which the code
   already does — and never fuse the behavioural stream as a single number
   without stating which component carried it.

---

## 14. The intervals were biased, and correcting them changes §4 but not §9

Every interval in §§4–12 was a percentile bootstrap. That method assumes the
replicate distribution is centred on the estimate. For the statistic this
project decides on, it is not.

### Why `C_llr_min` needs a bias correction

`C_llr_min` fits its PAV transform on the very trials it scores. It is a
**resubstitution minimum**, and therefore optimistically biased — it reports the
best calibration achievable *on the data it was measured on*. A bootstrap
resample contains about 63% of the distinct speakers, so the bias is **larger**
in the replicates than in the point estimate, and the whole replicate
distribution sits low.

Measured on this project's own `amr12.2_clean@30` scores at B = 1000:

```
point estimate      0.3431
bootstrap median    0.3210
P(replicate < point)  0.632     →  BCa z0 = +0.337,  a = +0.110
```

Nearly two replicates in three fall below the point estimate. The percentile
method corrects for none of that, so it produces an interval shifted toward
zero — and the decision rule in §3 reads the *bounds*. A downward-shifted
interval withdraws falsifications that should stand and makes support easier to
claim than the evidence warrants. The second of those is the dangerous direction
for a forensic report.

`bootstrap_over_speakers` and `paired_bootstrap_over_speakers` now compute
bias-corrected and accelerated intervals, with a delete-one-speaker jackknife
for the acceleration term. `Estimate` carries `interval_method` and
`n_discarded`, so an interval's provenance and its discard rate travel with the
number rather than living in a log.

### The correction is larger where there are fewer speakers

That is the whole result, and it is what theory predicts: the resubstitution
bias shrinks as the effective sample grows.

**125-speaker model, 42 evaluation speakers** — the §4 sweep:

| Cell | Percentile | BCa | Verdict |
|---|---|---|---|
| 12.2 clean, 30 s | [0.214, 0.426] | [0.259, 0.475] | inconclusive |
| 12.2 clean, 5 s | [0.485, 0.644] | **[0.506, 0.672]** | **falsified** |

Both bounds move by about **+0.045**, and `clean@5s` crosses the falsification
threshold. §5 reported that cell as inconclusive; under a bias-corrected
interval it is falsified.

**306-speaker model, 102 evaluation speakers** — §9:

| Cell | Percentile | BCa | Verdict |
|---|---|---|---|
| clean, 30 s | [0.188, 0.342] | [0.215, 0.388] | inconclusive |
| clean, 15 s | [0.262, 0.418] | [0.288, 0.455] | inconclusive |
| clean, 5 s | [0.463, 0.598] | [0.483, 0.618] | inconclusive |
| babble 20 dB, 30 s | [0.207, 0.362] | [0.231, 0.400] | inconclusive |
| babble 20 dB, 15 s | [0.286, 0.435] | [0.311, 0.466] | inconclusive |
| babble 20 dB, 5 s | [0.445, 0.568] | [0.466, 0.594] | inconclusive |

Lower bounds move by about **+0.020** — less than half the shift at 42 speakers —
and **no verdict changes**. §9's withdrawal of the five-second falsification
survives the correction.

### A prediction that failed, and why it is worth recording

An external review of this document predicted the opposite: that BCa would move
§9's five-second lower bounds from 0.463 and 0.445 to roughly 0.507 and 0.489,
restoring the falsification and overturning §9's headline. That prediction came
from applying the **+0.045 shift measured at 42 speakers** to a result computed
on **102**.

It does not transfer. The bias correction is a function of the replicate
distribution, and that distribution is better behaved with more speakers. At 102
the shift is +0.020 and the claim stands.

The reasoning was right and the arithmetic was portable in the wrong way — which
is the same class of error as §12's first version, where a figure read out of
one context was carried into another where it did not hold. Both are recorded
rather than quietly fixed, because the pattern is more instructive than either
instance.

### Multiplicity, and a p-value that had to be made to agree with its interval

`viflap/evaluation/hypotheses.py` has defined and documented `holm_bonferroni`
for the whole life of this project — *"a thesis reporting seven tests without
correction is reporting that chance as a finding"* — and nothing called it.
Six paired comparisons on one dataset is a family, and §9 was reporting six
uncorrected verdicts.

Applying it required a p-value, which required a decision. The obvious bootstrap
p-value is the fraction of replicates on the far side of zero — but that is a
*percentile* statement, and pairing it with a BCa interval reports two different
inference procedures as though they agreed. They need not, and here they did
not: `babble20dB@15s` came back with an interval excluding zero (upper bound
−0.0003) beside a p-value of 0.063.

The p-value is now derived through the same BCa transformation, by inverting the
map to ask what nominal level places a bound exactly at zero. Interval and
p-value agree by construction, and share a helper so they cannot drift apart.
All six cells are now internally consistent.

§9's corrected result: **five of six exclude zero, four survive Holm, all six in
the same direction.** The conclusion is unchanged and better supported than the
uncorrected version was.

### The full sweep, recomputed

All 30 cells of §4 have now been rerun with BCa at B = 2000.

| | Percentile, B = 300 | BCa, B = 2000 |
|---|---:|---:|
| supported | 0 | 0 |
| **falsified** | **3** | **6** |
| inconclusive | 27 | 24 |

Three cells changed, all inconclusive → falsified, all at five seconds:

| Cell | Percentile | BCa |
|---|---|---|
| 12.2 kbit/s clean, 5 s | [0.485, 0.644] | **[0.509, 0.685]** |
| 12.2 kbit/s vehicle 20 dB, 5 s | [0.484, 0.643] | **[0.506, 0.684]** |
| 4.75 kbit/s clean, 5 s | [0.497, 0.679] | **[0.519, 0.732]** |

By duration, the corrected picture is cleaner than the original:

| Duration | Falsified | Inconclusive |
|---:|---:|---:|
| 30 s | 0 | 10 |
| 15 s | 0 | 10 |
| **5 s** | **6** | 4 |

**This is the one correction in this document that made a negative finding
stronger.** A downward-biased interval makes falsification harder to reach —
the lower bound has to clear 0.50 — so removing the bias pushed three cells
across a threshold they had been held back from. Everywhere else in this
document the corrections either strengthened a positive result (§7 to six of
six) or withdrew an overclaim (§5's calibration, §12's premise).

The substantive reading is unchanged in direction and firmer in support: the
acoustic stream is not merely unproven at five seconds, it is **falsified across
most of the channel grid** at that duration, while nothing at 15 or 30 seconds
is decided either way.

§7's capacity comparison has not been rerun under BCa, B = 2000 or Holm. Its
sixth cell was reported as missing significance "by a thousandth", which is
inside the Monte-Carlo error of a tail bound at B = 300 and should not have been
described that precisely.

---

## 15. The clip, not the calibrator, is what makes calibration look cheap

§5 concluded that matched calibration costs a mean of 0.054 bits, never more
than 0.078, and is therefore not the constraint. Every one of those figures is
measured **after** the empirical lower and upper bounds are applied. This section
reports what the bound is doing, which §5 did not, and the conclusion does not
survive it.

### How much of the reported evidence is a constant

`Calibrator.calibrate` clips its output to the ELUB bounds fitted on the
development set — the safeguard that stops the system reporting a likelihood
ratio more extreme than its data can support. Measured across the nine stored
cells:

| Cell | Clipped, all trials | Different-source | Same-source | `C_llr` unbounded → bounded |
|---|---:|---:|---:|---|
| 12.2 clean, 30 s | **60.6%** | 61.5% | 3.0% | 0.603 → **0.416** |
| 12.2 clean, 15 s | 54.4% | 55.2% | 3.3% | 0.548 → 0.458 |
| 12.2 clean, 5 s | 41.2% | 41.8% | 1.3% | 0.637 → 0.626 |
| 12.2 vehicle 5 dB, 30 s | 57.1% | 58.0% | 2.5% | 0.574 → 0.460 |
| 12.2 vehicle 5 dB, 15 s | 50.5% | 51.3% | 2.3% | 0.569 → 0.519 |
| 4.75 clean, 30 s | **63.1%** | 64.0% | 3.0% | 0.619 → 0.411 |
| 4.75 clean, 15 s | 55.3% | 56.2% | 2.8% | 0.556 → 0.457 |
| 4.75 clean, 5 s | 38.2% | 38.8% | 1.5% | 0.675 → 0.644 |

**At the best operating point the reported likelihood ratio for three trials in
five is a constant.**

### The clip is one-sided, and it is repairing the calibrator

The asymmetry is the whole story: 61.5% of different-source trials are clipped
against 3.0% of same-source. The bound is almost entirely capping how *negative*
the log-LR may go — which is precisely the direction an affine map fitted to
these scores gets wrong.

Decomposed at 12.2 kbit/s clean, 30 s:

| | bits |
|---|---:|
| `C_llr` unbounded | 0.603 |
| `C_llr` bounded (as reported) | 0.416 |
| `C_llr_min` (discrimination floor) | 0.343 |
| calibration loss **before** the clip | **0.260** |
| calibration loss **after** the clip | 0.073 |

The clip removes **0.187 of 0.260 bits — 72% of the calibration loss.** §5's
"0.054 bits, never more than 0.078" is the residue left after a safeguard has
repaired most of the problem.

The honest statement is the reverse of §5's: **the logistic calibration is badly
miscalibrated on this data, and ELUB is rescuing it.** Reporting the post-clip
number and concluding that calibration is cheap turns a safeguard into a
performance claim.

### What is and is not affected

**§5's conclusion that "matched calibration is not the constraint" is
withdrawn.** Pre-clip it costs 0.26 bits at the best cell, against a
discrimination floor of 0.343 — the same order as the discrimination problem,
not a rounding error beside it.

**The reported numbers stay correct as operational quantities.** §2's rule that
every `C_llr` in this document is the as-reported, bounded value is right and
should not change: bounded is what a deployment emits. The error is inferential,
not arithmetical — drawing a conclusion about *calibrator quality* from a number
that measures calibrator-plus-safeguard.

**§4's `C_llr_min` column is untouched.** Discrimination is computed before any
calibration and no clip enters it, so §§7, 9, 12 and 14 — all of which decide on
`C_llr_min` — are unaffected.

**§5's calibrator-family comparison is close to vacuous.** `empirical_bounds`
derives its bounds from a PAV fit, which depends only on the *rank order* of the
scores. A logistic map is affine-increasing and an isotonic map is monotone, so
both preserve the order of the raw PLDA scores and both produce **identical**
bounds — measured, at 12.2 clean 30 s, as log₁₀ [−2.347, 3.644] for each. That
three calibrator families land within 0.01 of one another post-clip is close to
a mathematical necessity rather than an empirical finding about calibration.

> **Now recorded in the artefact, and with a control that was not looked for.**
> The rerun of `compare_calibrators.py` stores each calibrator's bounds. The two
> order-preserving families agree to every digit printed, as the argument
> requires. The kernel-density calibrator does **not**: log₁₀ [−2.374, 3.611].
> That is the mechanism confirming itself rather than an anomaly — a ratio of
> two estimated densities need not be monotone in the score, so it is the one
> family of the three that can reorder trials, and it is the one whose bounds
> move. The identity is a consequence of order preservation, which is now
> visible in the output rather than argued in prose, and there is a unit test
> asserting it.
>
> The clipping asymmetry is likewise stored per calibrator and reproduces: at
> the best cell, 60.6% of all trials clipped, 61.5% of different-source against
> 3.0% of same-source.

### What follows

Report both columns. `evaluate_h1.py` already stores `c_llr_matched_unbounded`
per cell and §4 prints only the bounded one; printing both, with the clipped
fraction beside them, costs nothing and makes the number interpretable.

Then fix the calibrator rather than the reporting. A 0.26-bit pre-clip
calibration loss on a logistic map is large enough to be worth attacking
directly, and §5's own comparison shows isotonic regression already reaching
0.451 unbounded against logistic's 0.603 — a 0.15-bit gain that the clip
currently conceals by dragging both to nearly the same place.

---

## 16. The channel is a model of a codec, and it has never been validated

> **It has now. §20 reports the measurement this section asks for, and it
> corrects both of the findings below.** The heading stands as written because
> it was true when written and the correction is part of the record.

§1 records that the channel is a parametric CELP model rather than a reference
AMR-NB coder, because ffmpeg was unavailable. That is stated everywhere it
matters. What has never been done is the measurement that would say how much it
matters — and it turns out to matter more than the caveat implies.

### What the bitrate parameter actually controls

`ParametricCelpCodec` maps a nominal bitrate to exactly two quantities:

```
LSF_STEP_HZ_BY_BITRATE     the LSF quantiser step
PULSES_BY_BITRATE          the algebraic codebook pulse count
```

Nothing else varies. There is no bit budget: the adaptive and fixed codebook
gains and the pitch lag are carried as unquantised floats, and the pulse
positions are chosen from anywhere in the subframe rather than from the
interleaved algebraic tracks that produce a real ACELP coder's distortion.
**The labels "12.2 kbit/s" and "4.75 kbit/s" do not correspond to any quantity
of bits.** They name two points in a two-parameter family.

### Measured spectral distortion

> **The measurement this section calls for has now been made, and it corrects
> both findings below. See §20.** The reference coder run beside the model on
> identical input gives 3.28 dB where the model gives 6.63 — the model is about
> **twice** as harsh, not the six times the comparison against a 1 dB standards
> criterion implied, because that criterion describes an LSF quantiser in
> isolation rather than an envelope re-estimated from a decoded
> analysis-by-synthesis waveform. And the bitrate claim below is **withdrawn**:
> the figures in this table do not reproduce under the measurement script that
> now exists, whose code was written after this section and is the only version
> with tests behind it.

Log-spectral distance between LPC envelopes, six BembaSpeech recordings, 3,408
frames, comparing the codec output against a reference resampled to the same
8 kHz band so that the bandwidth change is not counted as coding distortion:

| Comparison | Mean spectral distortion | Frames > 2 dB |
|---|---:|---:|
| 12.20 kbit/s vs reference | **6.60 dB** | 99.9% |
| 4.75 kbit/s vs reference | **6.93 dB** | 100.0% |
| 12.20 vs 4.75 output | 2.68 dB | 81.8% |

Two things follow, and they point in opposite directions from the caveat in §1.

**The model is far harsher than a real coder.** The design criterion for
transparent LSF quantisation in the standards literature is around 1 dB average
spectral distortion with under 2% of frames beyond 2 dB. This model produces
6.6 dB with essentially every frame beyond 2 dB. Whatever it is doing to speech,
a real AMR-NB coder at 12.2 kbit/s does much less of it.

**The bitrate axis is a small perturbation on a large constant.** The difference
between the two nominal bitrates is **0.33 dB**, against 6.6 dB of
bitrate-independent distortion — about five percent of the total. That is the
mechanism behind §5's finding that bitrate costs +0.001 at 30 seconds: the knob
being varied moves the spectral envelope very little compared with what the rest
of the codec is already doing to it.

> **Withdrawn.** The 0.33 dB gap does not reproduce. `scripts/validate_channel.py`
> was written after this section and its measurement is the one with tests; run
> over BembaSpeech on the development machine it gives 8.02 dB at 12.2 and
> 9.74 dB at 4.75, a gap of **1.72 dB**, and over LibriSpeech on a runner
> 6.63 and 8.79, a gap of **2.16 dB**. The reference AMR-NB coder's own gap,
> measured identically, is **2.47 dB**. So the model's bitrate knob moves a
> comparable share of its distortion to the real coder's, and the reasoning
> built on it here — that §5's flat bitrate row is explained by a knob that
> barely moves — does not hold.
>
> The figures in the table above came from a measurement whose code was never
> committed; this section shipped with the first version of this document, and
> `validate_channel.py` did not exist yet. They cannot be reproduced, which is
> the reason they are withdrawn rather than merely superseded. §20 reports what
> the committed measurement says.

An external review measured the LSF quantiser *in isolation* and found it
transparent — 0.21 dB at the 12.2 setting — and concluded the model was too
gentle. Both observations hold: the quantiser is nearly transparent, and
everything downstream of it is not. The distortion is coming from the excitation
model, not from the parameter the bitrate label names.

### What this does to the results

**§5's bitrate row describes this parameterisation, not AMR-NB.** "Bitrate is
nearly free at 30 s" is a true statement about a knob that moves 5% of the
distortion. It is not evidence about what a real coder's bit allocation costs a
speaker recognition system, and it should not be quoted as though it were.

**The duration and noise findings are less exposed but not clean.** They are
measured *through* this channel, so their magnitudes are conditional on a
degradation harsher than the deployment target. If the model over-degrades, the
system's true performance on real AMR-NB is better than every number in this
document.

**§12's benchmark comparison is affected in a stateable direction.** E3FS3's
figures come from real GSM 06.10, G.729a and G.723.1 coders. If this channel is
harsher than those, part of the remaining gap between 0.336 and 0.208 is
channel severity rather than system quality — which would make the comparison
*more* favourable to this system than §12 concludes, not less.

**`C_llr_min` orderings are the most robust part.** §7 and §9 compare two models
through the *same* channel, so channel severity is common to both arms and
cancels in the paired difference. Those findings do not depend on the channel
being faithful.

### The measurement that has to happen

Build a real coder — `opencore-amrnb` compiles standalone, and the 3GPP
reference implementation is freely available — run both over the same hundred
recordings, and report:

1. log-spectral distance and segmental SNR, coder against coder;
2. the difference in `C_llr_min` between them at the reference condition.

Until that exists, the honest scope of every absolute figure in this document is
"through this parametric model", and the conditions should be labelled by what
they actually vary — LSF step and pulse count — rather than by bitrates that no
part of the implementation computes.

---

## 17. The duration effect is not an artefact of the normalisation window

Duration is the largest effect in this document — §5 reports +0.23 in
`C_llr_min` from 30 s to 5 s, and every later section treats it as the factor
that matters. It was confounded, and the confound sat in the front-end rather
than in the channel.

### The confound

`FrontEndConfig.sliding_cmvn_frames` is 300, three seconds of frames, and it is
a **fixed** window. What that window does depends on what it is given:

Measured on eight evaluation recordings through `amr12.2_clean`, which is what
the sweep actually feeds it:

| Duration | Speech frames (median, range) | Window as a share | What the operation is |
|---:|---:|---:|---|
| 30 s | 2,646 (2,496–2,841) | **11%** | a local estimate, tracking within-recording change |
| 15 s | 1,324 (1,231–1,405) | **23%** | still local |
| 5 s | 446 (349–454) | **67%** | most of the utterance at once |

At five seconds the sliding window has largely collapsed into utterance-level
normalisation — and utterance-level *variance* normalisation divides every
cepstral dimension by a deviation estimated over the whole recording, which
removes between-speaker variance along with the channel variance it is aimed at.

So the duration sweep varied two things at once. It removed speech, **and** it
changed what the front-end did to what was left. Nothing in the reported numbers
separates them, and the front-end change pushes in the same direction as the
duration change, so the effect could in principle have been substantially
front-end rather than duration.

### The control, and why it had to be retrained

The window travels inside the model archive, because it is part of the front-end
the model was fitted to. Evaluating an existing model under a different window
would therefore measure a train/test mismatch and not the window. Two control
models were trained instead, on the **same 306 speakers, the same split, the
same seed and the same channel conditions** as `acoustic_pooled.npz`, differing
in nothing but the window:

| Model | Window | What it does at every duration |
|---|---|---|
| `ivec-plda-d5023efe82508a33` | 300 frames | local at 30 s, global at 5 s — the confounded baseline |
| `ivec-plda-b2db0f6fcf6fbd4b` | utterance | **global** at every duration |
| `ivec-plda-311ddae15a2ea994` | 100 frames | **local** at every duration |

The two controls bracket the baseline rather than merely replacing it. One holds
the operation global everywhere, the other holds it local everywhere; neither
changes character with duration, and the baseline is the only arm that does.

### What is compared

`scripts/compare_cmvn.py`, at 12.2 kbit/s clean, on the pooled model's own 102
held-out speakers. The quantity that answers the question is a difference of two
differences — the 30 s→5 s gap under the control set against the same gap under
the baseline — so it needs a bootstrap that neither the plain nor the paired
version can express. `bootstrap_contrast_over_speakers` supplies it, and the two
existing functions are now wrappers over it so the BCa interval and its p-value
still cannot drift apart.

The trial list has to be common across **durations** as well as across models.
The front-end refused 12 of 518 recordings at 5 s — identically under both
models — and had those 12 been kept at 30 s the two ends of the gap would have
been computed over different populations. Everything below is therefore on the
506 recordings every model embedded at every duration, which is why the 30 s
baseline reads 0.274 here against 0.276 in §9.

### Result: 94% and 97% of the duration effect survives

`C_llr_min` per duration, both controls against the same baseline. Negative
means the control discriminates better.

| Duration | 300 frames | utterance | diff [95% CI] | p | 100 frames | diff [95% CI] | p |
|---:|---:|---:|---|---:|---:|---|---:|
| 30 s | 0.274 | **0.266** | −0.007 [−0.017, +0.004] | 0.205 | 0.288 | +0.015 [+0.004, +0.029] | 0.006 |
| 15 s | 0.351 | **0.331** | −0.020 [−0.034, −0.005] | 0.007 | 0.362 | +0.011 [−0.007, +0.028] | 0.215 |
| 5 s | 0.539 | **0.514** | −0.024 [−0.041, −0.008] | 0.002 | 0.545 | +0.007 [−0.010, +0.024] | 0.366 |

And the duration gaps themselves, which is the question:

| Gap | 300 frames | Control | Contrast [95% CI] | p | Surviving |
|---|---:|---:|---|---:|---:|
| 30 s → 15 s | +0.077 [+0.059, +0.109] | utterance +0.065 | −0.013 [−0.028, +0.002] | 0.080 | **84%** |
| 30 s → 15 s | +0.077 [+0.059, +0.109] | 100 frames +0.074 | −0.003 [−0.017, +0.010] | 0.494 | **96%** |
| 30 s → 5 s | +0.265 [+0.215, +0.309] | utterance +0.248 | −0.017 [−0.037, +0.001] | 0.079 | **94%** |
| 30 s → 5 s | +0.265 [+0.215, +0.309] | 100 frames +0.257 | −0.008 [−0.025, +0.011] | 0.369 | **97%** |

**The confound is real in direction and small in size.** Holding the front-end
duration-invariant shrinks the 30 s→5 s gap by 0.017 at most — six percent — and
no contrast excludes zero, at either duration, under either control, before or
after Holm-Bonferroni within each run.

That the two controls agree is what makes this more than one arm's result. They
fix the window in opposite directions and differ from each other in level by
0.022 at 30 s, yet both return 94–97% of the duration gap. The answer does not
depend on which way the window was held still.

§5's headline therefore stands, and on firmer ground than it did: the duration
effect is duration, not the normalisation window changing behaviour underneath
it. That was the outcome hoped for when the check was proposed, which is a
reason to state the arithmetic rather than the conclusion — 0.248 and 0.257 of
0.265, with intervals that both include no change at all.

### A second finding, which was not the one being looked for

The three arms order themselves cleanly, and in the same direction at every
duration:

| | 30 s | 15 s | 5 s |
|---|---:|---:|---:|
| utterance-level | **0.266** (EER 7.62%) | **0.331** (9.09%) | **0.514** (15.72%) |
| 300 frames | 0.274 (7.86%) | 0.351 (9.91%) | 0.539 (16.87%) |
| 100 frames | 0.288 (8.30%) | 0.362 (9.95%) | 0.545 (16.58%) |

**The more global the normalisation, the better this system discriminates.** The
sliding window is not merely neutral here; it is costing something, and a
shorter one costs more.

The mechanism is in the docstring that justifies it. The window is 300 frames
because *"AMR rate adaptation changes the effective channel within a single call
as radio conditions vary, so the utterance-level mean is an average over
conditions rather than an estimate of one"*. That is sound reasoning about real
telephony — and this channel does not do it. `DegradationCondition` carries one
`bitrate_kbps`, `apply_condition` codes the whole recording at it, and nothing
anywhere varies the rate within a recording. The sliding window is paying the
cost of a noisier estimate from fewer frames for a benefit the degradation model
cannot deliver.

The monotone ordering is the evidence for that reading. If there were
within-recording channel variation to track, a local estimate would recover
something and the ordering would not be monotone in how global the estimate is.
It is: every step toward a global estimate helps, at every duration.

That scopes the finding rather than settling it. On real AMR with rate
adaptation the sliding window may well earn its keep, and §16's unvalidated
channel is the reason this cannot be decided here — it is a direct prediction
the reference coder would test, since AMR rate adaptation is exactly what the
parametric model omits. What can be said is narrower and still worth saying:
**through this channel the sliding window is a small net loss, and the baseline
model in §§4–15 is very slightly worse than it needed to be.** The effect is
0.008 of `C_llr_min` at 30 s and nothing in this document turns on it.

### What this does not establish

**It is one condition.** 12.2 kbit/s clean. Whether the window interacts with
added noise — where the local estimate has more to track — is untested, and the
noise cells are where a sliding window would most plausibly pay.

**Both controls are still fixed windows.** Neither adapts to the material. The
comparison rules out the specific artefact of a window that changes character
with duration; it does not say what the best normalisation for this front-end
would be.

**The intervals here predate the trial-ownership correction of §18** and are
therefore the wider, pre-correction ones. That is the conservative direction for
a contrast asked to exclude zero, and none of them does, so the conclusion is
unaffected: a narrower interval could only have made the finding harder to
dismiss, not easier.

**The 5 s figures remain survivor figures.** 12 recordings were refused at 5 s
and the metric is computed over what remained, exactly as §6 describes. The
refusal was identical under all three models, so it does not bias the contrast,
but it still bounds what the absolute 5 s numbers mean.

---

## 18. The resampling units were not exchangeable, and it barely mattered

Every interval in this document comes from a bootstrap over speakers, and §2
states the reason: trials sharing a speaker are correlated, so the effective
sample size is the speaker count rather than the trial count. That argument
assumes speakers are exchangeable units. They were not.

### One speaker owned three thousand trials and another owned none

A different-source trial belongs to **two** speakers, and a cluster bootstrap
needs it to belong to one. `build_trials` gave it to the first recording's
speaker. That reads as arbitrary-but-harmless and is neither: `split_by_speaker`
returns recordings grouped by speaker and sorted by identifier, and the trial
loop pairs each recording with every *later* one. So the first speaker in the
sort owns a trial against every later recording in the set, and the last speaker
owns none at all.

Counted exactly on the 102-speaker evaluation partition — 518 recordings,
132,796 different-source trials:

| Attribution | Heaviest owner | Lightest | Speakers owning nothing | Kish effective sample |
|---|---:|---:|---:|---:|
| first recording's speaker | 3,072 | 24 | **1 of 102** | **72.2** |
| hashed unordered pair | 1,589 | 984 | 0 | **98.2** |

The same measurement on the stored 42-speaker scores of §4 gives 1,314 against
16, and a Kish effective sample of **31.1 of 42**.

A bootstrap drawing units that differ hundredfold in influence is not drawing
the units its interval claims. Whether the interval came out wide or narrow
depended substantially on whether a handful of early-sorting speakers were in
the resample.

### The fix, and what it cannot be

Nothing makes a two-speaker trial belong to one speaker correctly. What the
attribution can be is *symmetric*: the owner is now chosen by a SHA-256 of the
two recording identifiers taken as an **unordered** pair, so it does not depend
on which side of the comparison a recording fell on, and each speaker pair's
trials split roughly evenly between them. `hashlib` rather than the built-in
`hash`, which is salted per interpreter and would silently re-attribute every
trial on every run — changing the interval while leaving the point estimate
alone, which is the worst combination for anyone trying to reproduce a figure.

Kish effective sample size is now computed per cell and recorded in every
report, so the assumption is visible in the artefact instead of being an
argument in a docstring.

### Result: no verdict moves, and the widths move by under five percent

The six pooled cells of §9, rerun with nothing changed but the attribution:

| Cell | `C_llr_min` | Interval, first-recording rule | Interval, symmetric rule | Width |
|---|---:|---|---|---:|
| clean, 30 s | 0.2757 | [0.2150, 0.3883] | [0.2125, 0.3830] | −1.6% |
| clean, 15 s | 0.3493 | [0.2884, 0.4550] | [0.2884, 0.4478] | −4.4% |
| clean, 5 s | 0.5385 | [0.4833, 0.6180] | [0.4790, 0.6133] | −0.3% |
| babble 20 dB, 30 s | 0.2947 | [0.2307, 0.4002] | [0.2340, 0.4002] | −1.9% |
| babble 20 dB, 15 s | 0.3705 | [0.3105, 0.4662] | [0.3127, 0.4664] | −1.3% |
| babble 20 dB, 5 s | 0.5137 | [0.4661, 0.5937] | [0.4675, 0.6010] | **+4.6%** |

**Every point estimate and every EER is identical to twelve decimal places.**
That is by construction — ownership enters the resampling and nothing else — and
it is checked rather than asserted, because a change that moved a point estimate
would mean the attribution had reached somewhere it has no business being.

**No cell changes verdict.** All six were inconclusive and all six remain so.

**The effect is small and not uniformly in one direction.** Five cells narrow
and one widens. Restoring exchangeability is not the same as adding information:
it re-weights which speakers drive the replicate distribution, and where a
dominant speaker happened to be a stabilising one, removing that dominance
widens the interval instead.

### Recorded because the finding is a negative one

This was a real defect. The units a stated interval rests on were not the units
it claimed, one speaker in the evaluation set contributed nothing to the
resampling at all, and the effective sample was three quarters of the nominal
count. It is exactly the kind of thing that, found in someone else's method
section, would justify discounting their intervals.

And correcting it moved nothing that matters. The intervals shift by less than
0.008 in either bound and no verdict changes, which is consistent with the one
other measurement available here: a two-sided dyadic bootstrap, which handles
the second speaker properly rather than by attribution, was measured to move the
interval by about 0.004.

Both halves are worth stating. §§14 and 15 record corrections that changed
conclusions; this one records a correction that did not, and reporting only the
first kind would misrepresent how often careful checks come back negative. The
defect is fixed regardless — a design that happens not to bite is still a design
that should not be relied on to keep not biting as the corpus changes.

### What this does not fix

**The trial is still attributed rather than shared.** Weighting a trial one half
to each speaker, or re-forming trials among the drawn speakers, are the
principled treatments; the measurement above suggests neither would move a
reported figure here, which is why neither was built.

**The intervals elsewhere in this document predate it.** §§4–17 are the
first-recording attribution throughout. On the evidence of the table above they
are within about five percent of their corrected widths, and no verdict in the
document turns on a margin that small — but they have not each been rerun, and
the ones quoted above are the only ones measured both ways.

---

## 19. What the PLDA trainer was watching was not a likelihood

Every model in this document was trained by expectation-maximisation to a
stopping rule, and the quantity that rule read was not one EM is guaranteed to
improve.

### The quantity

The trainer accumulated the quadratic data term and nothing else:

```
log_likelihood += -0.5 * sum(residuals @ within_inverse * residuals)
```

No `log|W|`, no `log|B|`, no latent prior, no posterior-covariance trace. That is
neither the observed-data log-likelihood nor the evidence lower bound. It is a
piece of both, and nothing makes a piece of a monotone quantity monotone, so the
run halted wherever this happened to settle. Worse than the arbitrary stopping
point: the standard sanity check on an EM implementation — *does the likelihood
climb* — was unavailable, and it is the check that separates a correct M-step
from a plausible wrong one.

**The M-step itself is correct**, which is worth saying because it is the part
usually got wrong. The `Cov_s` term is present in the `W` update. Omitting it
treats each speaker's estimated position as if it were known exactly, inflating
`W`, deflating `B`, and understating the evidential value of a genuine
same-source pair.

### The marginal is exact and costs almost nothing

For a speaker with `n_s` recordings the observations are jointly Gaussian with
covariance `I ⊗ W + 11ᵀ ⊗ B`, and the determinant lemma with Woodbury reduces
both halves to quantities the E-step already forms:

```
log|Σ_s|   = n_s log|W| + log|B| - log|Cov_s|
xᵀ Σ_s⁻¹ x = Σᵢ xᵢᵀ W⁻¹ xᵢ - (Σᵢ xᵢ)ᵀ W⁻¹ Cov_s W⁻¹ (Σᵢ xᵢ)
```

so the exact marginal costs one extra determinant per *distinct recording
count*, not per speaker, and the posterior-covariance cache that already existed
serves both. A decrease now raises `ConvergenceError` rather than being logged,
and `n_iterations`, the final likelihood and whether the tolerance was actually
met are recorded on the model and reach `describe()`.

### The guard was checked against the defect it exists to catch

Asserting monotonicity is worth nothing if the assertion cannot fail. Dropping
the `Cov_s` term from the `W` update — the usual mistake — and tracking the same
likelihood:

| M-step | Smallest step in log-likelihood per recording |
|---|---:|
| as implemented | −0.000000 (monotone) |
| `Cov_s` dropped | **−0.139** |

Against a tolerance of 1e-6 relative, so the guard catches it by five orders of
magnitude.

### And then the guard fired on correct code

It failed immediately on the compact test configuration, by seven parts in a
hundred thousand — far too large to dismiss as floating point, and the first
reading was that the M-step had a defect after all.

It did not. `_stable_inverse` adds a trace-proportional ridge before inverting,
so the algorithm is exact EM for the **ridged** model and only approximate for
the unridged one, while the new likelihood was taking `log|B|` and `log|W|` of
the *unridged* matrices. Scoring one model with another model's sufficient
statistics carries no monotonicity guarantee and duly had none. The ridge is now
a named function both sides use.

This is recorded because the failure mode is instructive in both directions: a
correct check found a real inconsistency that had been invisible while nothing
was checking, and the inconsistency was in the *checking*, not in the thing
checked. Had the tolerance been set loosely enough to absorb it, both the
mismatch and any future genuine defect would have passed.

### What this does and does not change

**No result in this document changes.** The models on disk are unchanged files
and their scores are what they were.

**A retrain would not reproduce them exactly.** The stopping rule now reads a
different quantity, so a run may halt at a different iteration and produce a
different `model_id`. Any comparison against a figure here must be against the
stored archive rather than against a fresh training run — which is what the
content-derived model id is for.

**The iteration counts of the models in §§4–17 are unknown.** They were not
recorded. Models trained from here on carry them.

---

## 20. The channel model, against the coder it models

§16 named the measurement that had never been made — run both coders over the
same recordings and report the difference — and §1, §6 and §16 all scope every
absolute figure in this document on the fact that it had not been. It has now
been made. `data/reports/channel_validation.json` is the artefact and
`scripts/validate_channel.py` produced it.

### Getting an encoder, which took four attempts and is the useful part

The development machine has no ffmpeg carrying AMR-NB and cannot get one. A
GitHub runner can, and the job is in `.github/workflows/channel-validation.yml`.
Four things had to be true before it measured anything, and each was found by a
run that failed to:

**The Actions tab is unreachable from where the work is done.** `api.github.com`
is blocked on this network, so neither the button nor `gh workflow run` exists
here, while plain git over SSH does. The workflow now also triggers on a tag
matching `channel-validation-*`. Pushing one is the same deliberate act as
pressing the button, expressed in the only protocol available.

**So is the log, and so are artefacts.** A run that dies before measuring is
otherwise indistinguishable from a slow one. The job commits its report — and
commits one saying it failed, when it does — because `git fetch` is the only
status channel there is.

**`ffmpeg` is not enough.** The first run installed ffmpeg and correctly
reported `available: false`. Debian and Ubuntu build libavcodec twice and the
default package omits the patent-encumbered and GPLv3 codecs, opencore-amr among
them; `libavcodec-extra` is the flavour that carries it. Diagnosing that from
`ffmpeg_encoder: null` was not possible, so the report now carries the ffmpeg
path, version, configure line and every AMR row of the encoder and decoder
tables. The second run confirmed the guess in one shot.

**Six recordings were one voice.** The selection took the first six of a sorted
glob, and a corpus path puts one speaker's files consecutively, so a
"two-speaker sample" measured six utterances from a single chapter of speaker
374. Deterministic and unrepresentative: coding distortion varies with the
voice. Recordings are now taken round-robin across speakers, and the sample is
48 recordings across 16.

A fifth thing was found only after a real measurement existed to expose it, and
is recorded below with the caveats rather than here, because it is a defect in
the measurement rather than in getting one to run.

### Result

48 LibriSpeech recordings across 16 speakers, 30 s each, 47,090 frames, through
both coders. Log-spectral distance between LPC envelopes against the source
resampled to the same 8 kHz band, so the bandwidth reduction — a property of the
channel rather than of either coder — is not charged to either.

| Comparison | Rate | LSD (dB) | Frames > 2 dB | seg. SNR (dB) | Delay |
|---|---:|---:|---:|---:|---:|
| **reference AMR-NB** vs source | 12.20 | **3.22** | 87.6% | 2.18 | 39.3 |
| parametric model vs source | 12.20 | **6.79** | 99.3% | 3.98 | 0.0 |
| model vs reference | 12.20 | **6.93** | 99.9% | 1.59 | 0.0 |
| **reference AMR-NB** vs source | 4.75 | **5.71** | 99.8% | 1.27 | 39.3 |
| parametric model vs source | 4.75 | **8.57** | 100.0% | 1.71 | 0.0 |
| model vs reference | 4.75 | **7.78** | 100.0% | −0.15 | 0.0 |

The figures are stable in the sample size, which is worth stating because the
first successful run measured six recordings from six speakers and this one
measured eight times as many across sixteen: 3.28 → 3.22 and 6.63 → 6.79 at
12.2 kbit/s. Whatever these numbers are limited by, it is not how many
recordings went into them.

The third row of each pair is the one this measurement exists for, and it is
reported only now because the first version of it was misaligned — see the
caveats below, where the defect and its symptom are recorded.

**The model is about twice as harsh as the coder it models**, not six times.
§16 set its 6.6 dB against a standards criterion of roughly 1 dB and concluded
the model was far harsher than any real coder. That comparison was not
like-for-like: the 1 dB figure describes an LSF quantiser measured in isolation,
and what is measured here is an LPC envelope re-estimated from a decoded
analysis-by-synthesis waveform. Measured the same way, the reference coder gives
3.22 dB and puts 87.6% of frames beyond 2 dB. The ratio is **2.11 at
12.2 kbit/s and 1.50 at 4.75**. The direction of §16's finding stands and its
magnitude halves.

**The bitrate axis is not a token, and §16's claim that it was is withdrawn.**
The reference coder moves **2.48 dB** from 12.2 to 4.75 kbit/s; the model moves
**1.79 dB**, about seventy percent of it. Both are a fifth or more of the total
distortion rather than the five percent §16 reported. §16's 0.33 dB does not
reproduce — see the correction there — and the reasoning it supported, that §5's
flat bitrate row is explained by a knob that barely moves, does not survive it.

That leaves §5's bitrate row **better** supported than §16 allowed rather than
scoped away. The knob does move real spectral distortion, comparably to the real
coder's, and `C_llr_min` still barely responds at 30 seconds. §5's own
explanation — that with enough frames the i-vector averages away coarser
spectral quantisation, and at 5 seconds it cannot — is what is left standing.

**The two coders are not the same distortion at different strengths.** This is
the reading the coder-against-coder row supplies and neither of the other two
can. If the model were simply the reference plus more of the same, the distance
between them would be roughly the difference of their distances from the source
— 6.79 − 3.22 = 3.57 dB at 12.2 kbit/s. It is **6.93 dB**, slightly *larger*
than the model's own distance from the source, and close to the quadrature sum
of the two, √(6.79² + 3.22²) = 7.51.

That is the signature of two distortions that are close to independent rather
than aligned. The model is not a harsher AMR-NB; it is a different displacement
of the spectral envelope, of comparable size to the reference's and pointing
somewhere else. §16 reached the right suspicion by the wrong route in saying the
distortion comes from the excitation model rather than from the LSF quantiser
the bitrate label names — this is the evidence for it, because an excitation
substituted wholesale moves the envelope in a direction an analysis-by-synthesis
search never would.

It also sharpens what "pessimistic" can mean for the rest of the document. A
system trained and evaluated through a channel that is *differently* wrong,
rather than *more* wrong, need not be uniformly worse than one through the
reference. Whether it is remains the unmeasured half of §16's request.

### What is not yet reliable in this table, and is recorded rather than trimmed

**The coder-against-coder rows were wrong the first time they were measured**,
and they are the rows the whole measurement exists to produce. `estimate_delay`
searches non-negative lags only, on the physical ground that a codec delays its
output rather than anticipating its input. That holds for a coder against its
source and fails for one coder against the other: this model returns its output
aligned with the input while AMR delays by about 40 samples, so the model's
signal *leads* and the estimator cannot express it. Asked for a non-negative lag
it returns the best one available, which is noise — the first run of this table
reported a mean delay of 39 samples with a **maximum of 192** on a comparison
whose true offset is a constant −40.

That was visible only because the delay is recorded per comparison, which is
exactly what that field was put there for. Both coded signals are now aligned to
the band-matched source they share before being compared with each other, and
the corrected rows report a residual delay of **zero** on every recording, which
is the check that the alignment worked.

The size of the correction is worth recording. At 12.2 kbit/s the
model-against-reference figures moved from 7.97 dB and a segmental SNR of
**−1.87 dB** to 6.93 dB and **+1.59 dB**. A negative segmental SNR means the
error signal carried more energy than the speech, which is what five
milliseconds of offset looks like and is not what either coder does. The rows
against the source never had the problem — each coder is compared with its own
input, where the non-negative rule is correct — and they are unchanged.

**Segmental SNR is still the weaker of the two figures.** The reference reads
*lower* than the model against the source (2.18 against 3.98 dB) while
distorting the envelope half as much, and its 39-sample delay is removed only to
the nearest sample, so a residual sub-sample offset would depress a
phase-sensitive measure while leaving an envelope measure alone. Log-spectral
distance is the figure to take from this table; segmental SNR is corroborative
where the two agree, which after the alignment fix they now do.

**One recording per speaker per condition, and one corpus.** LibriSpeech read
speech, not telephony, and the material is the one caveat this measurement
shares with everything else in the document.

### What this does and does not do to the rest of the document

**Every absolute figure remains conditional on the parametric model, and is
still pessimistic** — by a factor of about two in spectral distortion, whose
translation into `C_llr_min` is unmeasured. That second half of §16's request
needs the reference coder and an evaluation together, and the machine that has
the models cannot run the coder while the machine that can run the coder does
not have the corpus. It is the obvious next use of the workflow.

**§12's benchmark gap narrows in a stateable direction.** E3FS3's figures come
from real GSM 06.10, G.729a and G.723.1. This channel is harsher than a real
narrowband coder by about a factor of two in envelope distortion, so part of the
0.336-against-0.208 gap is channel severity rather than system quality.

**The paired comparisons are untouched.** §7, §9 and §17 compare two models
through the same channel, so severity is common to both arms and cancels.

**And "12.2 kbit/s" still names no quantity of bits.** §16's first finding —
that the labels set an LSF step and a pulse count and nothing else — is about
the implementation rather than about the distortion, and nothing here bears on
it.

---

## 21. The training conditions were confounded with the speaker, and it was not the cause

§1 records a spectrum whose leading between-speaker variance is five to seven
times its second, across every model this project has trained and across all
three cepstral-normalisation front-ends. One dominant axis of between-speaker
variation is what a nuisance factor absorbed into the speaker subspace looks
like, and there was an obvious candidate. This section tests it, and the answer
is no.

### The confound

`assign_conditions` dealt the eight training channel conditions out by a global
permutation. That is exactly balanced corpus-wide, which is what it was written
for — independent draws would leave one condition over-represented by chance —
and it does not balance *within* a speaker.

That matters because of what PLDA estimates. A speaker's recordings are the only
evidence the model has about within-speaker variability, so whatever is common
to them is the speaker as far as the model can tell. Under a global permutation
each speaker draws their conditions at random from the balanced pool, so each
carries a mean channel offset of their own, and LDA and PLDA have no way to
distinguish that offset from the person.

Measured on the 306-speaker training partition, 1,539 recordings:

| | global permutation | stratified |
|---|---:|---:|
| speakers receiving a repeated condition | **75.2%** | **0%** |
| distinct conditions per speaker (of 5.03 reachable) | 3.93 | **5.03** |
| SD across speakers of their mean training bitrate | **1.42 kbit/s** | **0.66 kbit/s** |
| corpus-wide condition counts | 192–193 | 192–193 |

The last row is the point of the design and the reason the corpus-level summary
could never have detected the problem: both allocations are exactly balanced
across the corpus. Only the per-speaker view separates them.

The fix is a rotating balanced allocation — each speaker takes a contiguous
block of the condition cycle, the cycle running on across speaker boundaries, so
every speaker sees as many distinct conditions as they have recordings, no
condition attaches preferentially to the speakers that sort early, and the
corpus-wide counts are unchanged. The order within a speaker is then permuted,
or condition would track chapter order instead.

### Two models, and a control that matters more than it looks

Both trained on the same 306 speakers, the same split, the same seed and the
same code, differing in nothing but the allocation. The old model is included
because it is the one §§9–18 report.

| Model | Allocation | ψ₁ | ψ₂ | ψ₁/ψ₂ | dims above ψ = 0.1 |
|---|---|---:|---:|---:|---:|
| `ivec-plda-d5023efe82508a33` | global, pre-§19 stopping rule | 44.951 | 6.384 | 7.041 | 74 |
| `ivec-plda-88c22044f3488bf5` | global, current code | 44.951 | 6.384 | 7.041 | 74 |
| `ivec-plda-b31031b9e75e3d2c` | **stratified**, current code | **43.967** | 6.825 | **6.442** | 73 |

**The control is the first two rows.** §19 replaced the PLDA stopping rule with
the exact observed-data likelihood, so a retrain could have moved the spectrum
for that reason alone and any comparison against the stored model would have
confounded the two changes. It did not: the largest difference across all 100
dimensions is **0.00075**, and the two models differ only in the last digits.
Their model ids differ, because the id hashes every parameter and the run halted
at a different iteration — which is exactly what §19 predicted and is the reason
the control was run rather than assumed.

So the third row is attributable to the allocation alone.

### The confound is real, and it is not the explanation

**ψ₁ falls by 0.98, which is 2.2%.** The ratio falls from 7.04 to 6.44, and the
total between-speaker variance from 129.1 to 126.2. Halving the per-speaker
channel offset — 1.42 kbit/s to 0.66 — and eliminating repeated conditions
entirely removes about a fiftieth of the leading eigenvalue.

**The spike is therefore not the condition confound.** A ψ₁ of 44 against a ψ₂
of 6.4 is not a channel offset absorbed as speaker identity; whatever produces it
survives an allocation designed to remove exactly that. The hypothesis §1 raised
is refuted, and refuted by the measurement that was proposed to test it rather
than argued away.

The direction is right and the magnitude is not, which is the same shape as §18:
a real defect in the experimental design that turns out to move almost nothing.
Both are worth recording, because a method section that reports only the checks
that changed something misrepresents how often careful checks come back negative.

### What the spike might be instead

Not established, and listed so the next attempt does not start from scratch:

- **The corpus.** LibriSpeech chapters differ in microphone, room and recording
  date, and a speaker's chapters are far more alike than two speakers' are. That
  is a session effect rather than a channel effect, and the stratification here
  does nothing to it because sessions are not assignable.
- **Length normalisation.** The transform chain length-normalises before LDA,
  which concentrates i-vectors on a sphere and can leave one dominant radial
  direction that is not speaker-related.
- **The speaker population.** 306 speakers is not many, and the leading
  eigenvalue of a between-speaker scatter estimated from 306 draws in 100
  dimensions is upward-biased by construction. A ratio of 7 may be partly an
  estimation artefact rather than a structure in the data.

The last of these is testable without new material — the bias falls with the
speaker count, so the 125-speaker model should show a *larger* ratio than the
306-speaker one if estimation noise is what drives it. §1's table says 5.13
against 7.04, which is the wrong direction for that explanation and is the one
piece of evidence currently available against it.

### What this does not change

**No reported result moves.** The stratified model is a new artefact and nothing
in §§4–20 was computed with it. Whether it scores better is a separate question
this section does not answer: it compares subspaces, not `C_llr_min`, and a
paired evaluation over the 102 held-out speakers is what would settle it.

**The allocation is now the default anyway.** It is more defensible than the one
it replaces whether or not ψ₁ moved, and the old behaviour is retained behind
`--condition-allocation global` because every model in §§4–18 was trained under
it.

---

## 22. A borrowed extractor, and the first supported cells in this document

§12 ended with a specific instruction: **not "train an x-vector extractor" — borrow
one**, because the gap against a published forensic system was produced by ~6,000
VoxCeleb2 speakers behind its extractor, and 6,000 speakers is not collectable
here. This section is that move, carried out and measured.

| | |
|---|---|
| Extractor | ECAPA-TDNN, `speechbrain/spkrec-ecapa-voxceleb`, 192-dim |
| Its training data | VoxCeleb2, ~6,000 speakers, none of it ours |
| Back-end | length-norm → LDA/WCCN → two-covariance PLDA, **fitted here on 306 speakers** |
| Corpus, split, channel, trial rule, bootstrap | unchanged from §9 |
| Extraction cost | 330 minutes over 2,578 recordings |
| Scoring cost | 16 minutes |

Artefacts: `data/reports/neural_embeddings.npz`,
`data/reports/neural_extraction.json`, `data/reports/h1_neural.json`.

The i-vector column below is `data/reports/h1_pooled_ownership.json` — the §18
symmetric-ownership recomputation, not §9's original table — because
`score_neural.py` reuses `_different_source_owner` and `bootstrap_over_speakers`,
so both columns carry §14's BCa correction and §18's attribution rule. Comparing
against §9's percentile intervals would have credited the extractor with a change
of method.

### Result

`C_llr_min` with BCa intervals at B = 2000, on the same 102 held-out speakers.

| Condition | Dur. | i-vector (306 spk) | EER | ECAPA + same back-end | EER | H1 |
|---|---:|---|---:|---|---:|:---:|
| 12.2 kbit/s, clean | 30 s | 0.276 [0.212, 0.383] | 7.89% | **0.099 [0.031, 0.230]** | **2.47%** | **supported** |
| 12.2 kbit/s, clean | 15 s | 0.349 [0.288, 0.448] | 9.81% | **0.126 [0.064, 0.253]** | **2.83%** | **supported** |
| 12.2 kbit/s, clean | 5 s | 0.539 [0.479, 0.613] | 16.87% | 0.228 [0.169, 0.335] | 5.78% | inconclusive |
| 12.2 kbit/s, babble 20 dB | 30 s | 0.295 [0.234, 0.400] | 9.15% | **0.114 [0.041, 0.249]** | **2.59%** | **supported** |
| 12.2 kbit/s, babble 20 dB | 15 s | 0.370 [0.313, 0.466] | 10.95% | **0.156 [0.089, 0.280]** | **3.65%** | **supported** |
| 12.2 kbit/s, babble 20 dB | 5 s | 0.514 [0.467, 0.601] | 15.96% | 0.252 [0.197, 0.350] | 6.10% | inconclusive |

**Four cells reach `supported`.** Under the rule fixed in §3 — supported if the
*upper* bound of `C_llr_min` ≤ 0.30 — these are the first supported cells
anywhere in this document. §4 returned 0 supported of 28 evaluable cells; §9 moved
the system from *falsified at five seconds* to *inconclusive everywhere* without
reaching support; this reaches it in four of six.

At the best cell `C_llr_min` falls 0.276 → 0.099, a 64% relative reduction, and
EER 7.89% → 2.47%, a 69% relative reduction.

Calibration is also cheaper, which is a separate finding from discrimination:

| Condition | Dur. | i-vector matched `C_llr` | ECAPA matched `C_llr` | i-vector calib. loss | ECAPA calib. loss |
|---|---:|---:|---:|---:|---:|
| clean | 30 s | 0.336 | **0.138** | 0.060 | **0.039** |
| clean | 15 s | 0.412 | **0.158** | 0.062 | **0.032** |
| clean | 5 s | 0.582 | **0.264** | 0.043 | **0.035** |
| babble 20 dB | 30 s | 0.362 | **0.151** | 0.067 | **0.036** |
| babble 20 dB | 15 s | 0.430 | **0.193** | 0.059 | **0.037** |
| babble 20 dB | 5 s | 0.551 | **0.275** | 0.037 | **0.024** |

### §12's mechanism, confirmed one stage earlier

§9 established that speaker count binds in the back-end: 125 → 306 speakers
improved five of six cells. §12 argued the same mechanism operates in the
extractor, on the evidence that E3FS3's ResNet saw ~6,000 speakers while only its
LDA and PLDA saw 91.

This experiment holds the back-end's speaker count fixed at 306 and changes only
the extractor's, from 306 to ~6,000. Everything else — corpus, split, channel
conditions, trial rule, bootstrap, PLDA implementation — is the same code on the
same audio. **The improvement is therefore attributable to the extractor and to
essentially nothing else in this system**, which is the cleanest form §12's
prediction could have been tested in.

It is worth being precise about what was imported, because the experiment does
not separate two things. Not only training data: an i-vector extractor and an
ECAPA-TDNN differ in architecture, in objective and in augmentation as well as in
speaker count, and this design cannot apportion the gain between them. What it
does show is that the *reachable* move — buying 6,000 speakers with a checkpoint
download rather than a collection programme — delivers, and delivers more than any
change made to this system so far.

### The paired difference, which is what actually decides this

> **This subsection replaces a caveat.** The first version of §22 reported the
> two systems' marginal intervals side by side and said plainly that the
> comparison was not paired, so the direction was not established at a stated
> confidence. It now is. Artefact: `data/reports/h1_extractor_paired.json`.

Marginal intervals were never going to settle it. Both are wide because
*speakers differ from one another*, and both systems saw the same speakers — so
most of that width is common and cancels in a difference. §7 records that reading
marginal overlap as "no difference" manufactures a null; the converse is equally
untrue, and clean 30 s does overlap, [0.212, 0.383] against [0.031, 0.230].

Both scorers now persist per-trial scores with the **recording-id pair** behind
each trial, so the two systems' trials can be joined rather than assumed aligned.
Difference is `ECAPA − i-vector` on `C_llr_min`, so **negative favours ECAPA**.
BCa at B = 2000, resampling speakers, Holm over the six cells.

| Condition | Dur. | i-vector | ECAPA | Difference [95% CI] | p | Holm | Trials |
|---|---:|---:|---:|---|---:|:---:|---|
| clean | 30 s | 0.276 | 0.099 | **−0.176 [−0.233, −0.141]** | 0.0010 | **✓** | 133,645 identical |
| clean | 15 s | 0.349 | 0.126 | **−0.223 [−0.275, −0.187]** | 0.0010 | **✓** | 133,645 identical |
| clean | 5 s | 0.539 | 0.228 | **−0.310 [−0.341, −0.273]** | 0.0010 | **✓** | 127,519 identical |
| babble 20 dB | 30 s | 0.295 | 0.114 | **−0.180 [−0.224, −0.140]** | 0.0010 | **✓** | 133,645 identical |
| babble 20 dB | 15 s | 0.370 | 0.156 | **−0.215 [−0.260, −0.175]** | 0.0010 | **✓** | 133,645 identical |
| babble 20 dB | 5 s | 0.514 | 0.252 | **−0.262 [−0.302, −0.227]** | 0.0010 | **✓** | 98,595 identical |

**All six exclude zero, all six survive Holm, and all six rest on identical
trial sets.** Artefact: `data/reports/h1_extractor_paired_vad.json`.

> The first version of this table could pair only four. Its 5 s rows were
> computed on the *intersection* of two differing trial lists, because the
> neural extractor's speech gate was measuring wall-clock length — see the
> correction below. With the gate fixed the two systems refuse the same
> recordings and the intersection is the whole set. **Every difference is
> unchanged to three decimals**, which is the expected result rather than a
> lucky one: the intersection already was the correct set, so restricting to it
> was right for the wrong reason.

Set against §9, where 181 additional training speakers bought −0.104 at the best
cell and four of six cells survived Holm, borrowing the extractor buys −0.176 at
the same cell and **six of six survive**. The move §12 recommended is larger than
the corpus expansion that motivated it.

**The p-values are at the floor and should not be read as values.** 0.0010 is
2/B at B = 2000, the smallest a two-sided bootstrap p-value can be here, so every
cell is reporting "below this resampling's resolution" rather than a magnitude.
They are carried only so Holm has something to consume. The intervals are the
informative quantity.

**The five-second rows now rest on the same footing as the other four**, and
getting there measured something §6 had only asserted. Under the corrected gate
both systems refuse the same 12 recordings at 5 s clean and the same 73 at 5 s
babble — set identity, checked, not merely equal counts. Dropping those
recordings improves ECAPA's `C_llr_min` from 0.234 to 0.228 at 5 s clean and from
0.291 to 0.252 at 5 s babble, so **the refused recordings were indeed the harder
ones**, which is what §6 says refusal at short duration does and what nothing had
previously measured.

It also means the earlier figures for those two cells were computed over a set
including recordings the system should have declined, and the difference is
larger in babble (0.039) than in clean (0.006) — as it should be, since babble is
where net speech falls furthest below wall-clock duration.

### What this does not establish

**The five-second rows are not on the same trials.** The i-vector front-end
refused 12 recordings at 5 s clean and 73 at 5 s babble; ECAPA refused none. So
the i-vector figures there rest on 814/126,705 and 639/97,956 trials against
ECAPA's 849/132,796. §6 records that refusals at short duration are *not* random
with respect to difficulty — they remove the recordings carrying least speech,
which are the hardest — so the i-vector column at 5 s is scored on an easier
subset and the gap in those two rows is, if anything, understated.

**Zero refusals is not straightforwardly a virtue, and the reason for it was a
defect.**

> **This paragraph was wrong and is corrected in place.** It said the neural
> extractor "has no such notion and returns a 192-dimensional vector for any
> input", and concluded that an abstention mechanism would have to be built.
> That is not what the code did. `NeuralEmbeddingConfig.min_speech_seconds` has
> always existed, with the same default of 3.0 as the i-vector front-end and the
> same name — but it compared `signal.size / sample_rate`, **wall-clock
> length**, against a threshold named for speech. The i-vector front-end runs
> voice activity detection first and tests `speech_duration_seconds`. Same name,
> same default, different quantity.

So the honest statement is not that the neural system cannot abstain. It is that
it abstained on the wrong criterion, and at these durations that criterion could
never fire: every truncation is 5 s or longer of wall clock against a 3 s bar, so
zero refusals was arithmetic rather than confidence. The i-vector front-end
refused 12 recordings at 5 s clean and 73 at 5 s babble because those carried
under three seconds of *speech* inside five seconds of audio.

**The consequence was measured in this section before the cause was found.** It
is exactly why the 5 s cells could not be paired on identical trial sets — 6,126
and 35,050 trials dropped — and the extractor's own docstring named the purpose
the code was failing to serve: the gate is retained "so that both extractors
refuse the same recordings and the comparison between them stays paired".

The gate now measures net speech through the same detector, with tests that fail
on the old behaviour, and **the corpus has been re-extracted through it** (365
min). Artefacts: `neural_embeddings_vad.npz`, `neural_extraction_vad.json`,
`h1_neural_vad.json`, `h1_extractor_paired_vad.json`. Every figure in this
section is now from that run.

| Cell | Old wall-clock gate | Corrected gate | i-vector front-end |
|---|---:|---:|---:|
| 30 s, both conditions | 0 | **0** | 0 |
| 15 s, both conditions | 0 | **0** | 0 |
| evaluation, 5 s clean | 0 | **12** | 12 |
| evaluation, 5 s babble 20 dB | 0 | **73** | 73 |

**The two front-ends now refuse exactly the same recordings** — checked as set
identity, not inferred from equal counts. That is what the gate's docstring has
always claimed it was for, and it is now true.

Three things follow.

**The cells this section rests on are untouched, and provably so.** The training
vectors and all four 30 s and 15 s evaluation cells are **byte-identical** between
the two extractions, because nothing was refused there under either gate. The
back-end refits to the same model (ψ₁ 66.98, 35 iterations), and all four cells
reproduce to machine precision. The four `supported` verdicts and their four
paired differences never depended on the defect.

**All six cells are now pairable on identical trial sets**, where before only four
were. That is the concrete gain.

**And the paired differences did not move** — unchanged to three decimals in every
cell. That is the expected outcome rather than a fortunate one: the intersection
the first version paired on was already exactly the i-vector's trial set, so it
had been restricting to the right recordings for the wrong reason. The defect cost
a caveat, not a result.

**We did not control the extractor's training data.** The 102 evaluation speakers
are LibriVox volunteers and VoxCeleb2 is YouTube celebrity interview audio, so
overlap is implausible; but implausible is not verified, and it cannot be verified
here, because the checkpoint's training list is not ours. Every borrowed-extractor
result carries this caveat and so does this one.

**The channel is still the parametric model** validated in §20, not a real AMR-NB
coder, and §20 found the two distortions near-independent rather than one a
harsher version of the other. Nothing here changes that caveat's scope.

**Against published work, one comparison is available and one is not.** ECAPA's
matched `C_llr` of 0.138 at 12.2 kbit/s clean 30 s sits in the range of E3FS3's
case-specific conditions (0.085–0.097 at 30 s). §12 explicitly identified that
comparison as a selection effect — best cell against best case-specific condition
— and the correction it made should not now be quietly undone. The like-for-like
figure is `forensic_eval_01`, on which E3FS3α reaches 0.208 and **this system has
never been run at all**. The honest statement is that the gap §12 measured at a
factor of 1.6 has plainly narrowed, and that quantifying it requires running that
benchmark rather than reasoning across datasets.

### Back-end health, and a note on dimension

| | |
|---|---:|
| Transform dimension reaching PLDA | 192 |
| PLDA iterations to convergence | 35 |
| Converged | yes |
| ψ₁ | 66.98 |

LDA did not truncate: 192 is below the 305-dimension ceiling that 306 training
speakers impose, so `min(192, 305) = 192` and the embedding passes through whole.
That is a thinner estimate per dimension than the i-vector system's 100 from the
same 1,539 training recordings, and §19's convergence monitor raised nothing — but
"it converged" is not "it was well-determined", and a control at
`--lda-dimension 100` is reported below.

ψ₁ is 66.98. §1's ψ₁/ψ₂ *ratio* is the comparable quantity rather than ψ₁ itself
and is not computed here, so whether §21's ψ₁ spike survives a change of extractor
is a live question this run does not answer. It is now cheap to answer, since the
embeddings are on disk.

### Control: the result is not an artefact of the 192-dimensional transform

Rerun with `--lda-dimension 100`, forcing the transform to the same width the
i-vector system uses. Artefact: `data/reports/h1_neural_lda100.json`. Extraction
is a separate script, so this cost 16 minutes rather than another six hours —
which is the whole reason the two were split.

| Condition | Dur. | 192-dim | 100-dim | Δ | Verdict |
|---|---:|---:|---:|---:|:---:|
| clean | 30 s | 0.099 | 0.103 | +0.004 | supported, both |
| clean | 15 s | 0.126 | 0.135 | +0.009 | supported, both |
| clean | 5 s | 0.234 | 0.248 | +0.013 | inconclusive, both |
| babble 20 dB | 30 s | 0.114 | 0.118 | +0.003 | supported, both |
| babble 20 dB | 15 s | 0.156 | 0.165 | +0.009 | supported, both |
| babble 20 dB | 5 s | 0.291 | 0.304 | +0.013 | inconclusive, both |

**Every verdict is unchanged and four cells remain supported.** Truncating to 100
dimensions costs between 0.003 and 0.013 — real, consistently in one direction,
and an order of magnitude smaller than the 0.177 the extractor change bought at
the best cell. The cost grows as the cell gets harder, which is what discarding
genuine information looks like.

Two incidental readings, both worth having:

**The 192-dimensional fit was not unstable.** PLDA converged in 35 iterations at
192 dimensions and 6 at 100. A longer EM run is not an ill-conditioned one, and
§19's monitor — which asserts monotonicity of the exact observed-data
log-likelihood and raises on violation — stayed quiet in both. The concern that
prompted this control was reasonable and is answered.

**More dimensions help here, where in §7 more capacity hurt.** That is not a
contradiction. §7's larger model was truncated to 124 by a 125-speaker LDA
ceiling while its UBM had half the data per component; here 192 sits comfortably
below the 305-dimension ceiling that 306 speakers impose, nothing is truncated,
and the extra 92 dimensions are estimated from an extractor that never saw this
corpus. The constraint §7 identified is a property of estimating a subspace from
too few speakers, and borrowing the extractor is precisely what removes it.

### What follows

1. ~~**Run the paired difference test** on the four cells with identical trial
   sets.~~ **Done** — see "The paired difference" above. Six of six cells exclude
   zero and survive Holm. `TrialSet` now carries the recording-id pair behind
   each trial, both scorers take `--scores`, and `compare_extractors.py` joins on
   that key rather than on row index, which two differently ordered archives
   would have made silently wrong.
2. **Rerun §21's ψ₁ question against these embeddings.** The extractor changed and
   the corpus did not; if the spike is a LibriSpeech session effect it should
   survive, and if it is an i-vector length-normalisation artefact it should not.
3. **Place this system on `forensic_eval_01`**, which is the only way the §12
   benchmark becomes a measurement rather than an inference.
4. **The countermeasure is untouched by this.** §10's blindness to phase-only
   attacks is a property of LFCC features, and a better speaker embedding does
   nothing for a validity gate that cannot see the attack.

---

## 23. What produces the ψ₁ spike: two candidates refuted, one left standing

§1 recorded that the leading PLDA eigenvalue runs five to seven times the second
across every model this project has trained, and said plainly why that matters:
one dominant axis of between-speaker variation is what a nuisance factor absorbed
into the speaker subspace looks like, because a factor shared across a speaker's
recordings is indistinguishable from the speaker as far as PLDA is concerned. §21
refuted the most promising explanation — condition stratification moved ψ₁ only
44.951 → 43.967 — and left three candidates: **LibriSpeech session or environment
effects**, **length normalisation**, and **upward bias in a leading eigenvalue
estimated from a few hundred speakers**.

§22 supplies an instrument the earlier sections did not have. Artefact:
`data/reports/psi_spectrum.json`.

### The spike survives a complete change of extractor

ECAPA embeddings share the corpus, the channel, the split and the back-end with
the i-vector system, and share nothing of its front-end: no MFCCs, no GMM-UBM, no
total-variability subspace.

| Model | Transform dim. | ψ₁ | **ψ₁/ψ₂** |
|---|---:|---:|---:|
| `acoustic` (125 spk) | 100 | 50.761 | **5.127** |
| `acoustic_large` | 124 | 86.034 | **5.872** |
| `acoustic_pooled` (306 spk) | 100 | 44.951 | **7.041** |
| `acoustic_pooled_cmvn_utt` | 100 | 46.701 | **6.873** |
| `acoustic_pooled_cmvn100` | 100 | 41.951 | **6.584** |
| **ECAPA + same back-end** | 192 | 66.983 | **5.244** |

The borrowed extractor lands at 5.244, inside the i-vector range and close to
§1's 5.127. **Whatever produces the spike is not a property of the i-vector
representation.** That was never one of the three candidates, but it was the
obvious fourth, and it is now excluded.

What this does *not* clear is the back-end: length normalisation, LDA, WCCN and
the PLDA implementation are shared between the two systems, so a ratio that
survives is consistent with any of them. Hence the next arm.

### Length normalisation is refuted

The one shared component cheap enough to switch off.

| | ψ₁ | ψ₁/ψ₂ |
|---|---:|---:|
| ECAPA, length-normalised | 66.983 | **5.244** |
| ECAPA, **not** length-normalised | 74.973 | **5.308** |

The ratio moves by **0.064**, against a spread of 1.9 across the models in the
table above. Length normalisation is not what produces the spike, and the
candidate is withdrawn.

### Estimation bias is refuted, and the obvious reading of it is backwards

If the spike were upward bias in a leading eigenvalue estimated from few
speakers, then *fewer speakers should give a larger ratio*. §21 noted that the
125-speaker model shows a smaller one and called that the wrong direction, but
that was a comparison between two models differing in more than speaker count.
This is the controlled sweep: same embeddings, same back-end, transform dimension
pinned at 74 so the LDA ceiling does not move with the sample, five random draws
per count.

| Training speakers | ψ₁/ψ₂ [min, max over draws] | ψ₁ |
|---:|---|---:|
| 75 | **3.160** [2.904, 3.501] | 181.077 |
| 150 | **4.307** [4.021, 4.461] | 81.566 |
| 225 | **4.923** [4.472, 5.477] | 67.998 |
| 306 | **4.956** | 61.032 |

**The ratio rises with speaker count and the ranges at 75 and 150 do not
overlap.** That is the opposite of the estimation-bias prediction, so the
candidate is withdrawn.

The sweep also shows why the intuition behind it was reasonable, and where it
went wrong. **ψ₁ itself is very clearly upward-biased at small samples** — 181.1
at 75 speakers against 61.0 at 306, a factor of three. The bias is real. But the
quantity §1 tracks is the *ratio*, and ψ₂ is biased upward faster than ψ₁ is, so
the ratio moves the other way. A section reporting only ψ₁ would have concluded
the opposite of the truth, which is an argument for §1 having tracked the ratio
rather than the leading value.

### What is left

**LibriSpeech session and environment effects — the only candidate still
standing**, and the one consistent with every measurement above. It is a property
of the *corpus*, which both systems share, so it survives a change of extractor;
it is not a property of the back-end, so it survives switching length
normalisation off; and it does not shrink with more speakers, because adding
LibriVox readers adds more readers with the same structure rather than diluting
it.

The mechanism is concrete. LibriVox readers record themselves, each in one room
with one microphone. A recording's channel is therefore near-constant within a
reader and varies between readers — which is exactly the shape §1 describes as
indistinguishable from the speaker. Our same-source trials cross *chapters*, so
they cross recording days, but for a home-recorded reader they very often do not
cross a recording setup.

**This bears on §9 and §22 rather than only on §1.** If a per-reader channel is
being absorbed into the speaker subspace, then both systems are being helped by
something that will not be present in casework, where one person is heard through
different handsets, networks and rooms. It does not touch the *relative* findings
— §22's paired difference holds both systems to the same corpus and the same
confound — but it is a reason to expect the absolute figures, including §22's
four supported cells, to be optimistic against real telephony data.

Establishing that requires a corpus in which speaker and channel are crossed
rather than confounded. §8 records what is available for the target population,
and this is one more argument for the same missing resource.

### What this does not establish

**"Session effects" is now a residual, not a measurement.** It is what remains
after two candidates were refuted, and a residual is only as good as the list it
was drawn from. A fourth explanation nobody has thought of would sit exactly
where this one does.

**The two refutations are of the *ratio*, not of every claim about ψ₁.** ψ₁ is
demonstrably upward-biased at small speaker counts, and any future statement
about its absolute value has to carry that.

**No corpus-crossed control was run**, because none is available here. The
mechanism above is argued from how LibriVox is recorded, not measured.

---

## 24. The validity gate, reached at last, and made to work

Every test of the validity gate in this project has driven it through a
hand-built `ValidityAssessment`. That is not an oversight of testing but a
consequence of structure: `CompareIncidents._validity_absence` consults the
assessment only for streams where `is_gated_by_validity` holds, which is the
acoustic stream alone, and the synthetic corpus deliberately synthesises no
speech. With no acoustic payload the gate is unreachable.

`scripts/synthetic_acoustic.py` closes that by filling in the hook the corpus has
carried since it was written — `Operator.acoustic_speaker_id` — binding each
synthetic operator to a distinct held-out LibriSpeech speaker, hearing every
incident as one of that speaker's recordings through the 12.2 kbit/s channel, and
putting **the same degraded signal** to both the i-vector system and the trained
countermeasure. Artefact: `data/reports/synthetic_acoustic.json`.

What is real here is the audio, the channel, the extractor, the countermeasure
and the policy. What is invented is everything else about the incident, and the
binding between an operator and a speaker is arbitrary. So the acoustic stream is
a measurement and any fused figure remains a simulation; §11's boundary governs.

### The result

80 incidents, 18 operators, model `ivec-plda-d5023efe82508a33`, detector
`lfcc-gmm-b23145edfcacf976`, default `GatePolicy`.

| | |
|---|---:|
| Recordings judged | 80 |
| **Admitted** | **0** |
| Indeterminate | 79 |
| Excluded | 1 |

**The gate admits none of eighty genuine human recordings.** Not one.

### Why, and it is not the domain check

The obvious explanation is the out-of-domain rule — §10 established that the
countermeasure generalises poorly, and `INDETERMINATE` is what the policy returns
when too many frames fall outside the training domain. That is not what happened.
Measured over the same recordings:

| Quantity | Observed | Policy threshold | Fires? |
|---|---|---:|:---:|
| Out-of-domain fraction | 0.000–0.018 | > 0.25 | no |
| Dispersion ratio | 0.224–0.262 | > 2.5 | no |
| Countermeasure log-LR | **−2.33 to +1.60** | ≥ **+2.30** to admit | **never** |

Both domain checks pass comfortably — by an order of magnitude. The verdict is
decided by the threshold, and **the detector's scores on genuine speech never
reach it**. The maximum observed over eighty recordings is +1.60 against a bar of
+2.30; the median is −0.21. One recording fell below −2.30 and was excluded, so
the only confident verdict the gate issued on genuine audio was a wrong one.

### The mechanism is a channel mismatch, not a weak detector

> **The first version of this subsection got the diagnosis wrong**, and wrongly
> in the direction that would have stopped anyone fixing it. It blamed the
> detector's discrimination — 16.41% EER on twelve training speakers — and
> concluded that "asking it for ±2.3 is asking for a discrimination it does not
> have" and that "no policy setting changes that". That inference was
> plausible, consistent with §10 and §12, and untested. Testing it refutes it.

Score the *same recordings* before and after the channel:

| Genuine speech | log-LR min | median | max | Reaching +2.3 |
|---|---:|---:|---:|---:|
| **Clean** — as the detector was trained | +0.80 | **+2.76** | +4.93 | **23 of 40** |
| **12.2 kbit/s** — as it is deployed | −2.33 | **−0.23** | +1.08 | **0 of 40** |

**The detector reaches the threshold comfortably on clean audio and never on
coded audio.** Its median falls by 2.99 — from confidently genuine to slightly
spoofed — on identical recordings whose only difference is the coder. The
discrimination was never the problem.

The cause is that `train_countermeasure.py` trains on clean speech, and §1
records exactly why the acoustic stack does not:

> Training was multi-condition... Training on clean audio and evaluating on
> degraded audio would have measured a front-end mismatch rather than speaker
> discriminability.

That reasoning was applied to the i-vector system in §1 and never to the
countermeasure. §24 is the measurement of what it cost: an entire evidence
stream withheld from every comparison, because a detector was asked about audio
of a kind it had never been shown.

**This is a composition defect twice over.** The policy and the detector were
specified independently and never met — that part of the original diagnosis
stands. But the deeper error is that the *training condition* and the
*deployment condition* were specified independently too, in a project whose
first section is about why that must not happen.

### What this means for a deployment

The gate is **fail-safe in direction and inoperable in effect**. It withholds
acoustic evidence rather than admitting a spoof, which is the right way round —
but at this operating point it withholds *all* acoustic evidence, including from
the stream §22 just measured at `C_llr_min` 0.099. A deployment assembled from
these parts would run with its best-performing stream silently absent, reported
correctly as `INDETERMINATE` on every incident and therefore never wrong,
and never useful.

That is worth stating plainly because it is the failure mode this architecture is
otherwise good at avoiding. Nothing here is a silent error: the result carries
`rests_on_single_stream`, the outcome is a `StreamAbsent` with
`EXCLUDED_BY_VALIDITY_GATE` or `NO_DATA`, and the audit record names the policy.
The system reports exactly what it did. It simply does nothing useful with the
acoustic stream.

### What follows

1. **Do not fix this by lowering the threshold.** A band chosen to match a
   mismatched detector's output range would admit recordings on evidence of
   ±0.5, which is the number the policy exists to refuse. The threshold is not
   the defect and moving it would hide one.
2. **Train the countermeasure multi-condition**, which is the actual fix and is
   what §1 does for the acoustic stack. `train_countermeasure.py` now takes
   `--degrade`, putting every example — genuine and spoofed alike — through the
   same eight channel conditions the i-vector system trains on. Both classes,
   because degrading only the genuine side would teach the detector to recognise
   the coder rather than the synthesis. The result is in the next subsection.
3. **Report the gate's operating point alongside any acoustic result.** §22's
   0.099 is what the acoustic stream achieves *when it is admitted*, and this
   section measures how often that is: **71 of 80** once both defects are fixed,
   against none at the start.
4. **The composition test belongs in the suite**, not in a script that has to be
   remembered. `tests/integration/test_synthetic_acoustic.py` asserts that the
   gate is reached at all — that a countermeasure verdict derived from a real
   signal arrives on a real embedding — because the defect above is invisible to
   every test of either component alone.
5. **The phase-only blindness of §10 is untouched by any of this.** LFCCs are
   magnitude-only whatever they are trained on, and `phase_randomised` remains
   at chance. Multi-condition training fixes a channel mismatch; it does not
   give a magnitude feature access to phase.

### The fix works on the axis it targeted, and uncovers the next one

`train_countermeasure.py --degrade` retrains through the same eight conditions
the acoustic stack uses. Artefacts: `models/countermeasure_multicondition.npz`,
`data/reports/countermeasure_multicondition.json`,
`data/reports/synthetic_acoustic_multicondition.json`.

| On the same 80 recordings | Clean-trained | **Multi-condition** |
|---|---:|---:|
| Countermeasure log-LR, median | −0.21 | **+5.63** |
| Range | −2.33 to +1.60 | **+3.70 to +7.18** |
| **Reaching the +2.3 threshold** | **0 of 80** | **80 of 80** |
| Admitted | 0 | **3** |

**The channel mismatch is fixed and the diagnosis is confirmed.** Every one of
the eighty recordings now clears the score threshold, where none did before. The
median moves by 5.84. Nothing about the policy or the detector's architecture
changed; only the audio it was trained on.

**And the gate still admits only three of eighty**, because a second condition
was being masked by the first. With scores no longer failing, the *domain* check
takes over:

| | Clean-trained | Multi-condition | Policy |
|---|---:|---:|---:|
| Out-of-domain fraction, median | 0.011 | **0.355** | > 0.25 rejects |
| Recordings failing it | 0 of 40 | **38 of 40** | |
| Dispersion ratio, median | 0.236 | 1.000 | > 2.5 rejects |

The mechanism is in how the floor is calibrated. `out_of_domain_threshold` is the
**1st percentile of best-of-both frame likelihoods over the training set**, so by
construction about one percent of *training* frames fall below it. That is a
sound definition when training and deployment see the same distribution. Trained
across eight conditions, the pooled likelihood distribution is far broader than
any single condition's, and a recording from one condition — here
`amr12.2_clean` — lands systematically in its lower tail. Thirty-five percent of
its frames fall below a floor built to exclude one.

So the domain rule now fires on exactly the audio the detector was retrained to
handle. It is not wrong to fire: a frame below the floor genuinely is unlike the
training average. The floor is measuring the wrong thing — the spread of the
training mixture rather than the distance of this recording from it.

### Where that leaves the gate

Three of eighty admitted is not an operable system, and it is a different
failure from the first. It is worth being precise about what improved:

- **The detector now works through the channel.** That was the §24 finding and it
  is fixed.
- **The gate is no longer refusing on strength of evidence.** All eighty
  recordings are confidently genuine by the policy's own standard.
- **The gate is refusing on domain**, and doing so because the floor's
  calibration assumes a deployment distribution as broad as the training one.

The next move is to calibrate the out-of-domain floor **per condition**, or on
held-out material from the deployment condition, rather than on the pooled
training set. That is a change to how the threshold is derived and not to the
rule it feeds, and it is recorded in the handoff rather than attempted here.

### The floor was measuring the wrong thing, and the gate now works

Measuring the domain check per condition shows it running backwards:

| Condition | Out-of-domain fraction | Countermeasure log-LR |
|---|---:|---:|
| `amr12.2_vehicle10dB` | **0.362** | +5.98 |
| `amr12.2_clean` | **0.352** | +5.66 |
| `amr7.4_clean` | 0.247 | +4.60 |
| `amr4.75_vehicle15dB` | 0.238 | +3.58 |
| `amr4.75_clean` | 0.233 | +4.16 |
| `amr12.2_babble20dB` | 0.157 | +3.02 |
| `amr5.9_babble15dB` | 0.066 | +1.62 |
| `amr7.4_babble10dB` | 0.042 | +0.68 |

**The fraction tracks the log-LR.** The recordings the detector is most confident
are genuine are the ones it calls out of domain, and the cleanest conditions are
flagged hardest while the noisiest pass almost entirely. That is the opposite of
what a domain check is for.

The cause is in how the floor was derived. `out_of_domain_threshold` was the 1st
percentile of best-of-both frame likelihoods over the **pooled** training set —
sound for a single-condition model, wrong for a mixture. Clean speech has
peakier, less average features and therefore lower likelihood under a mixture
dominated by noisy material, so it lands in the pooled lower tail *despite being
exactly what the detector was trained on*. Thirty-five percent of its frames fall
below a floor built to exclude one percent.

**"Unlike anything in training" is a union, not a percentile of a blend.** A
recording typical of one trained condition is in domain even if it is atypical of
the average of all of them. The floor is now the minimum over per-condition
percentiles, so every trained condition passes at roughly its own rate and only
audio below all of them is flagged. Artefacts:
`models/countermeasure_union.npz`, `data/reports/countermeasure_union.json`,
`data/reports/synthetic_acoustic_union.json`.

### The gate, across all three configurations

| On the same 80 recordings | Clean-trained | Multi-condition, pooled floor | **Multi-condition, union floor** |
|---|---:|---:|---:|
| Countermeasure log-LR, median | −0.21 | +5.63 | **+5.63** |
| Reaching the +2.3 threshold | 0 of 80 | 80 of 80 | **80 of 80** |
| **Admitted** | **0** | **3** | **71** |
| Indeterminate | 79 | 77 | **9** |
| Excluded | 1 | 0 | **0** |

**The validity gate is operable.** 71 of 80 genuine recordings admitted, nine
held as indeterminate, none wrongly excluded — where the configuration this
section opened with admitted none and excluded one.

Two defects, found in order because the first masked the second, and neither
visible to any test of a component alone:

1. **A training-condition mismatch.** The detector was trained on clean audio and
   deployed on coded audio, which §1 forbids for the acoustic stack and nobody
   had applied to the countermeasure.
2. **A floor calibrated against a mixture** rather than against each of the
   conditions composing it, which only became visible once the first was fixed.

Neither was a policy error. `GatePolicy`'s ±2.3 band is unchanged throughout, and
the temptation to widen it at either stage would have hidden a real defect behind
a threshold that admitted things it should not.

**The nine remaining indeterminates are the check working.** They are recordings
whose frames genuinely sit below every trained condition's floor, and holding
them is what the rule is for. A gate admitting all eighty would be a gate that
had stopped checking.

### What this does not fix

**The detector is no better at its job.** Every EER is unchanged from the
previous subsection — 25.00% seen, 29.37% mean unseen — because the GMMs are
identical and only the threshold derivation changed. The gate now passes
recordings it was always confident about; it does not detect spoofs any better.

**`phase_randomised` is still near chance** at 41.53%, and §10's mechanism is
untouched: LFCCs are magnitude-only and no threshold fixes that.

**The admission rate is measured on genuine speech only.** How often this gate
*correctly excludes* a spoofed recording through the channel is a separate
experiment, and the EERs above suggest the answer is "not reliably".

**One channel condition at evaluation**, one policy, one detector, and a
simulated corpus. What is established is that the composition works, not that it
would work on Zambian casework.

### What the retrained detector costs, and why that is not an argument against it

| | Clean-trained | Multi-condition |
|---|---:|---:|
| Seen attacks, EER | **19.14%** | 25.00% |
| Unseen `lpc_noise` | 13.54% | 18.03% |
| Unseen `lpc_pulse` | 4.17% | 9.29% |
| Unseen `oversmoothed` | 28.13% | 48.63% |
| Unseen `phase_randomised` | 52.60% | 41.53% |

**Every figure except one gets worse, and the model is still the right one.**
Separating genuine speech from synthesis is harder through a coder than in clean
audio: the coder discards exactly the fine spectral detail an LFCC detector keys
on, and it does so for both classes. The clean-trained model's 19.14% is not a
better system, it is the same task made easier by evaluating on material the
deployment will never present. §24's whole point is that the flattering number
was measured on the wrong audio.

`phase_randomised` moving 52.60% → 41.53% should be read with suspicion rather
than pleasure. §10 established that LFCCs are magnitude-only and therefore
*structurally* blind to a phase-only attack, and nothing about multi-condition
training changes that. A shift of this size on a single held-out family, from a
figure that was at chance, is most likely the coder incidentally disturbing the
randomised phase in a way the magnitude spectrum registers — an artefact of the
channel, not detection of the attack. It is reported because it moved, and
flagged because the mechanism §10 identified has not gone away.

### What this does not establish

**One channel condition, one policy, one detector.** The clean 12.2 kbit/s cell
only. Whether a different bitrate or noise condition moves the score
distribution above the bar is untested, though the range observed makes it
implausible.

**The binding is arbitrary and the incidents are simulated**, so nothing here is
evidence about how often real Zambian fraud recordings would be admitted. What is
measured is that this detector, on genuine narrowband speech through this
channel, does not reach this policy's admit threshold.

---

## 25. The corpus was still the constraint: 562 speakers, and the i-vector system reaches supported on its own

§9 established that speaker count binds by moving from 125 to 306 and improving
five of six cells. §22 then showed the same mechanism operating in the extractor,
where a borrowed checkpoint trained on ~6,000 speakers reached four `supported`
cells. That left an obvious question §22 could not answer: **had the corpus route
run out, or had it simply been abandoned at 306?**

`train-clean-360` was fetched to 41% in §9 and the remainder was never taken. It
has now been completed.

| | Usable speakers | Recordings | Split |
|---|---:|---:|---|
| §9's pooled corpus | 510 | 2,578 | 306 / 102 / 102 |
| **Complete `train-clean-360`** | **936** | **4,697** | **562 / 187 / 187** |

921 speakers came down, of which 194 have a single session and cannot contribute
within-speaker variation; `scan_corpora` warns and excludes them. Fetched to a
**new root** so §§9, 22 and 23 continue to quote the corpus they were computed on.
Model `ivec-plda-369f609975761fc7`, trained at 128 components and rank 100 — the
same configuration as §9, deliberately unchanged.

### Result

Evaluated on its own 187 held-out speakers, verified disjoint from the model's
recorded 562 training speakers. Artefact: `data/reports/h1_expanded.json`.

| Condition | Dur. | §9, 306 spk (102 eval) | EER | **562 spk (187 eval)** | EER | H1 |
|---|---:|---|---:|---|---:|:---:|
| clean | 30 s | 0.276 [0.212, 0.383] | 7.89% | **0.157 [0.136, 0.189]** | **4.87%** | **supported** |
| clean | 15 s | 0.349 [0.288, 0.448] | 9.81% | **0.224 [0.202, 0.257]** | **7.09%** | **supported** |
| clean | 5 s | 0.539 [0.479, 0.613] | 16.87% | 0.451 [0.427, 0.487] | 14.69% | inconclusive |
| babble 20 dB | 30 s | 0.295 [0.234, 0.400] | 9.15% | **0.170 [0.150, 0.198]** | **5.47%** | **supported** |
| babble 20 dB | 15 s | 0.370 [0.313, 0.466] | 10.95% | **0.248 [0.224, 0.281]** | **7.93%** | **supported** |
| babble 20 dB | 5 s | 0.514 [0.467, 0.601] | 15.96% | 0.454 [0.425, 0.504] | 14.52% | inconclusive |

**Four of six cells reach `supported`, and this is the i-vector system.** No
borrowed checkpoint, no VoxCeleb2, no architecture change — the same GMM-UBM and
total-variability stack §4 ran on 125 speakers, given 562 instead of 306.

The trend across the three corpus sizes at the best cell:

| Training speakers | `C_llr_min` | EER |
|---:|---:|---:|
| 125 (§4) | 0.343 | 10.86% |
| 306 (§9) | 0.276 | 7.89% |
| **562** | **0.157** | **4.87%** |

**The corpus route had not run out.** §9's "no reason to think 306 is where the
returns stop" is now measured rather than asserted, and the gain from 306 → 562
(−0.119) is larger than the gain from 125 → 306 (−0.067), which is the opposite
of the diminishing return one might expect.

The intervals also narrow sharply — [0.136, 0.189] against [0.212, 0.383] — and
that is a separate effect worth separating: the evaluation partition grew from
102 speakers to 187, and the bootstrap resamples speakers. Kish effective sample
size is 179.8 of 187. Part of reaching `supported` is the system being better and
part is the *interval* being tighter, and the decision rule is on the upper bound,
so both contribute.

### What this cannot be compared against, and why that is a finding

**There is no clean paired comparison with the 306-speaker model.** Only **19
speakers** are held out by both, because the expanded model trained on 562 of the
936 and swept up most of the old model's evaluation set. §9 faced the same problem
at a milder scale — 23 of its 42 baseline evaluation speakers sat in the pooled
model's training set — and solved it by scoring both on the 35 held out by both.
Nineteen is too few: an interval over nineteen speakers describes those nineteen
people.

So the table above is a **standalone evaluation**, legitimate under §3's rule but
not a paired difference. It cannot be read as "−0.119 caused by speakers" with the
confidence §22's paired test carries, because the two rows also differ in *which*
187 or 102 people were scored.

**The general lesson is worth recording, because it will recur.** Expanding a
corpus destroys the ability to compare against models trained on the subset,
unless a fixed evaluation set is reserved *before* the expansion. Had 100
speakers been set aside at the outset and excluded from every training split,
every corpus-size comparison this project has made would be paired. They are not,
and each has had to be restricted after the fact to whatever overlap survived.

### What this does and does not do to §22

**It does not overturn §22, and the two are not in competition.** §22's paired
test compared extractors on identical trials with the back-end held fixed at 306
speakers, and −0.176 with six of six surviving Holm is a stronger form of evidence
than anything here. What §25 adds is that the *other* axis was also unexhausted.

The obvious next experiment is the combination — the borrowed extractor with a
562-speaker back-end — and it is cheap, because extraction is separate and the
embeddings for the old corpus already exist. It is the first item in the handoff.

**Neither result escapes §23.** Both are LibriSpeech, and §23's surviving
explanation for the ψ₁ spike is that each LibriVox reader records in one room with
one microphone, so a per-reader channel is confounded with the reader. More such
readers is more of the same confound, and there is a real possibility that part of
what 562 speakers buys is a better-estimated version of something that will not be
present in casework.

### What this does not establish

**One configuration, no capacity sweep.** 128 components and rank 100, chosen to
match §9 rather than to suit 562 speakers. §7 found more capacity hurt on 125
speakers and §9 noted the LDA ceiling had moved to 305; at 562 it moves to 561, so
rank 200 would now pass through untruncated. That is untested and is the cheapest
open experiment on this axis.

**The 5 s cells remain inconclusive**, and their upper bounds (0.487, 0.504) are
well clear of 0.30. The duration finding of §5 is untouched by corpus size.

**No transferred calibration, no wider channel sweep.** Two conditions and three
durations, matching §9's standalone table so the two can be read side by side.

---

## 26. Both axes at once: H1 supported in every cell, including five seconds

§22 changed the extractor with the back-end fixed at 306 speakers. §25 changed
the back-end with the extractor fixed. Neither had been done together, and there
was no reason to assume the two gains would compose — §7 is the standing reminder
that a plausible improvement can be worse in all six cells, and both of these
buy the same commodity, speakers, at different stages.

They compose, and more than additively at the hard end.

ECAPA embeddings over the expanded corpus (624 minutes of extraction), back-end
fitted on **562** speakers, evaluated on the same 187 held-out speakers as §25.
Artefacts: `data/reports/neural_embeddings_562.npz`,
`neural_extraction_562.json`, `h1_neural_562.json`.

| Condition | Dur. | i-vector, 306 | ECAPA, 306 (§22) | i-vector, 562 (§25) | **ECAPA, 562** | EER | H1 |
|---|---:|---:|---:|---:|---|---:|:---:|
| clean | 30 s | 0.276 | 0.099 | 0.157 | **0.033 [0.023, 0.049]** | **1.00%** | **supported** |
| clean | 15 s | 0.349 | 0.126 | 0.224 | **0.058 [0.046, 0.074]** | **1.60%** | **supported** |
| clean | 5 s | 0.539 | 0.228 | 0.451 | **0.165 [0.147, 0.188]** | **4.64%** | **supported** |
| babble 20 dB | 30 s | 0.295 | 0.114 | 0.170 | **0.046 [0.033, 0.066]** | **1.20%** | **supported** |
| babble 20 dB | 15 s | 0.370 | 0.156 | 0.248 | **0.078 [0.064, 0.098]** | **2.11%** | **supported** |
| babble 20 dB | 5 s | 0.514 | 0.252 | 0.454 | **0.193 [0.171, 0.227]** | **5.26%** | **supported** |

**Six of six cells reach `supported`, and that includes both five-second cells.**
No configuration in this document had ever taken a five-second cell past
`inconclusive` — §4 *falsified* three of them, and §9's withdrawal of that
falsification was the good news at the time.

At the best cell, `C_llr_min` runs 0.343 → 0.276 → 0.099 → **0.033** across the
project, and EER 10.86% → 7.89% → 2.47% → **1.00%**.

### And it is paired

The table above is a standalone evaluation, which §22's paired instrument was
built precisely to improve on. It applies here after all, and for a reason worth
stating: §25 and §26 used the **same corpus, the same seed and the same split**,
so they scored an identical trial set — 1,500 same-source and 435,613
different-source at 30 s and 15 s, and identical counts at 5 s too, which also
means the i-vector front-end and the corrected neural gate refused *the same
recordings*. Only the i-vector side lacked persisted scores, and that cost 85
minutes rather than the ten hours a re-extraction would have.

Difference is `ECAPA − i-vector` on `C_llr_min` at 562 speakers, so **negative
favours the borrowed extractor**. BCa at B = 2000, resampling speakers, Holm over
six cells. Artefact: `data/reports/h1_extractor_paired_562.json`.

| Condition | Dur. | i-vector | ECAPA | Difference [95% CI] | Holm | Trials |
|---|---:|---:|---:|---|:---:|---:|
| clean | 30 s | 0.157 | 0.033 | **−0.124 [−0.153, −0.103]** | **✓** | 437,113 |
| clean | 15 s | 0.224 | 0.058 | **−0.167 [−0.193, −0.145]** | **✓** | 437,113 |
| clean | 5 s | 0.451 | 0.165 | **−0.287 [−0.314, −0.263]** | **✓** | 421,370 |
| babble 20 dB | 30 s | 0.170 | 0.046 | **−0.124 [−0.150, −0.103]** | **✓** | 437,113 |
| babble 20 dB | 15 s | 0.248 | 0.078 | **−0.170 [−0.197, −0.145]** | **✓** | 437,113 |
| babble 20 dB | 5 s | 0.454 | 0.193 | **−0.261 [−0.299, −0.225]** | **✓** | 337,068 |

**Six of six exclude zero, six survive Holm, all on identical trial sets.** So
the extractor's contribution at 562 speakers is established at a stated
confidence, and §26 is no longer a standalone result.

### The extractor advantage shrinks where the corpus helped most

Set against §22, which ran the same comparison with the back-end at 306:

| Condition | Dur. | §22 difference (306 spk) | §26 difference (562 spk) | Change |
|---|---:|---:|---:|---:|
| clean | 30 s | −0.176 | −0.124 | +0.052 |
| clean | 15 s | −0.223 | −0.167 | +0.056 |
| clean | 5 s | −0.310 | −0.287 | +0.023 |
| babble 20 dB | 30 s | −0.180 | −0.124 | +0.056 |
| babble 20 dB | 15 s | −0.215 | −0.170 | +0.045 |
| babble 20 dB | 5 s | −0.262 | −0.261 | **+0.001** |

**The borrowed extractor is worth less once the back-end has more speakers, and
the shrinkage is concentrated at long duration.** At 30 s the advantage falls by
about 0.05; at 5 s babble it does not move at all (−0.262 → −0.261).

That is the same story §9 told from the other side. Doubling the back-end's
speakers recovers information the i-vector system could not previously reach —
but only where the channel is mild enough for the information to survive. At five
seconds in babble it recovers nothing, so the extractor's advantage there is
untouched: whatever ECAPA is doing at 5 s is something 256 extra back-end
speakers cannot substitute for.

**Neither §22's comparison nor this one is subsumed by the other.** §22 measures
the extractor at 306 speakers, this measures it at 562, and the difference between
the two differences is the interaction — measurable here only because both are
paired on identical trials. A single number for "what the extractor is worth"
would have concealed that it depends on what it is being added to.

### What is still not paired

**§22 against §26.** Their corpora differ, so their splits differ, so no trial set
is shared and the four-configuration progression in the table above remains four
models on **two** evaluation sets rather than one. `viflap/evaluation/reserved.py`
exists to fix that for everything computed after it, and nothing in this document
was. Re-running §22, §25 and §26 with `reserved_evaluation=` threaded through
would put all of them on one fixed set of 100 speakers; it costs a re-extraction
and is the first item in the handoff.

### The two gains compose, and the composition is worth reading carefully

Taking clean 30 s and treating the 306-speaker i-vector system as the baseline:

| | `C_llr_min` | Reduction from baseline |
|---|---:|---:|
| i-vector, 306 spk | 0.276 | — |
| i-vector, 562 spk (§25) | 0.157 | −0.119 |
| ECAPA, 306 spk (§22) | 0.099 | −0.177 |
| **ECAPA, 562 spk** | **0.033** | **−0.243** |

−0.119 and −0.177 separately, −0.243 together. That is **less than the sum**
(−0.296), which is what one should expect when both changes buy the same
underlying commodity: the second helping of speakers is worth less once the first
has been eaten. The interesting part is at the other end of the duration range.
At 5 s clean the two gains are −0.088 and −0.311 separately and **−0.374**
together, and at 5 s babble −0.060 and −0.262 give **−0.321** — proportionally
much more of the sum survives where the channel is harshest, which is the regime
§9 identified as the one the corpus could not fix.

### What this does not establish, and one thing it should not be read as

**Paired against §25, not against §22.** The extractor's contribution at 562
speakers is established above at a stated confidence, six of six surviving Holm
on identical trials. What is *not* paired is this configuration against §22's,
because their corpora differ and so no trial set is shared. The reserved
evaluation set added in this session fixes that for everything computed after it,
and **this run predates it**.

**Part of the interval narrowing is the evaluation set, not the system.** 187
speakers against 102, and the decision rule is on the upper bound. §25 makes the
same point and it applies here with the same force.

**This is not a claim about E3FS3 or about `forensic_eval_01`.** A matched
`C_llr` of 0.033 on this corpus sits below the 0.085–0.097 §12 quotes for E3FS3's
case-specific conditions, and that comparison is **not available to make**. §12
was rewritten once for precisely this error — best cell against best condition,
across datasets, with different enrolment. The like-for-like benchmark remains
`forensic_eval_01`, on which E3FS3α reaches 0.208 and this system has never been
run. Reporting 0.033 alongside their 0.208 as though the two were commensurable
would be the same mistake with a better number attached.

**§23 governs the absolute figures.** Both axes are LibriSpeech, and §23's
surviving explanation for the ψ₁ spike is that each LibriVox reader records in
one room with one microphone, so a per-reader channel is confounded with the
reader. Doubling the readers doubles the confound. A 1.00% EER on read audiobook
speech through a parametric channel model is not a forecast for Zambian
telephony, and the gap between them is the least measured quantity in this
document.

**And the validity gate still governs whether any of it is admitted.** §24
measures that at 71 of 80 once both of its defects are fixed. `C_llr_min` 0.033
is what the acoustic stream achieves when it is heard at all.

### A note on ψ₁, which moved the way §23 predicted

The 562-speaker back-end reports ψ₁ = **45.00** against the 306-speaker
back-end's 66.98 on the same extractor. §23's speaker sweep found ψ₁ strongly
upward-biased at small samples — 181.1 at 75 speakers against 61.0 at 306 — and
predicted it would keep falling. It did, on data §23 never saw. That is a
prediction confirmed rather than a coincidence noted, and it is the second
independent confirmation that the *leading eigenvalue* is a small-sample artefact
even though the *ratio* §1 tracks is not.

---

## 27. One evaluation set, fixed forever: the pinned split

Every section up to here carries an asterisk that §26 stated and could not
remove. §22 is paired at 306 speakers, §26 is paired at 562, and the two sit on
**different evaluation sets**, so the progression 0.343 → 0.276 → 0.099 → 0.033
is four models on three sets rather than four comparable measurements.

The cause is structural. `split_by_speaker` orders speakers by a seeded
permutation and takes fractions, so *adding* speakers reshuffles everyone: a
speaker held out at one corpus size is very likely training material at the next.
§9 compared 125 against 306 and kept 35 of its evaluation speakers. §25 compared
306 against 562 and kept **19** — too few to describe anyone but those nineteen.

`viflap/evaluation/reserved.py` names 100 speakers that go to the evaluation
partition of every split, at every corpus size, permanently, and
`split_by_speaker` now defaults to it rather than taking it as an opt-in. An
opt-in guarantee is one every caller has to remember, and forgetting is exactly
how §9 and §25 lost their comparability — nothing warned, because both splits
were internally valid either way.

This section is the first result computed under it. The whole chain was re-run:
extraction (585.3 min), i-vector training (73.4 min), `evaluate_h1`,
`score_neural`, and the pairing.

**The guarantee was checked rather than assumed.** Read back from
`neural_embeddings_pinned.npz`: all 100 reserved speakers are in the evaluation
partition, none leaked into train or development, and the three partitions are
pairwise disjoint. Split 562 / 187 / 187 over 2,827 / 933 / 937 recordings.
Reading it off the docstring would have established only that the docstring is
confident.

### The two systems, on one set

i-vector model `ivec-plda-0a3110cdcb3c542d`, 128 components, rank 100 — §9's
configuration unchanged. Neural back-end fitted on the same 562 training
speakers. Artefacts: `data/reports/h1_pinned.json`,
`data/reports/h1_neural_pinned.json`.

| Condition | Dur. | i-vector | EER | H1 | **ECAPA** | EER | H1 |
|---|---:|---|---:|:---:|---|---:|:---:|
| clean | 30 s | 0.135 [0.114, 0.164] | 4.10% | **supported** | **0.030 [0.022, 0.043]** | **0.93%** | **supported** |
| clean | 15 s | 0.208 [0.184, 0.248] | 6.12% | **supported** | **0.049 [0.037, 0.071]** | **1.45%** | **supported** |
| clean | 5 s | 0.424 [0.400, 0.464] | 13.72% | inconclusive | **0.160 [0.142, 0.186]** | **4.46%** | **supported** |
| babble 20 dB | 30 s | 0.157 [0.134, 0.188] | 4.74% | **supported** | **0.037 [0.029, 0.052]** | **1.20%** | **supported** |
| babble 20 dB | 15 s | 0.220 [0.195, 0.254] | 6.42% | **supported** | **0.066 [0.052, 0.086]** | **1.79%** | **supported** |
| babble 20 dB | 5 s | 0.402 [0.377, 0.446] | 12.64% | inconclusive | **0.181 [0.163, 0.210]** | **5.15%** | **supported** |

Four of six for the i-vector system, **six of six for ECAPA**, and 0.93% is the
first sub-one-percent equal error rate anywhere in this document.

### The pairing

Difference is `ECAPA − i-vector` on `C_llr_min`, so **negative favours the
borrowed extractor**. BCa at B = 2000, resampling speakers, Holm over six cells.
Artefact: `data/reports/h1_extractor_paired_pinned.json`.

| Condition | Dur. | i-vector | ECAPA | Difference [95% CI] | Holm | Trials |
|---|---:|---:|---:|---|:---:|---:|
| clean | 30 s | 0.135 | 0.030 | **−0.105 [−0.124, −0.088]** | **✓** | 438,049 |
| clean | 15 s | 0.208 | 0.049 | **−0.159 [−0.182, −0.137]** | **✓** | 438,049 |
| clean | 5 s | 0.424 | 0.160 | **−0.264 [−0.294, −0.237]** | **✓** | 424,129 |
| babble 20 dB | 30 s | 0.157 | 0.037 | **−0.120 [−0.144, −0.100]** | **✓** | 438,049 |
| babble 20 dB | 15 s | 0.220 | 0.066 | **−0.154 [−0.180, −0.131]** | **✓** | 438,049 |
| babble 20 dB | 5 s | 0.402 | 0.181 | **−0.221 [−0.256, −0.194]** | **✓** | 337,890 |

**Six of six exclude zero, six survive Holm, and all six sit on trial sets that
are identical between the two systems** — 187 owners in every cell. The two
front-ends refused the same recordings, which is visible before the pairing runs
at all: the trial counts in the two result tables match exactly, including the
two five-second cells where refusals occur at all (15 clean, 114 babble).

### What this buys, stated precisely

**Everything computed from here on is comparable to everything else computed
from here on.** Any future model — a different rank, a different extractor, a
larger corpus that is a superset of these 936 speakers — evaluates on these same
100 reserved speakers, and the paired instrument applies without restriction.
That is the whole return on the ten-hour re-extraction, and it is entirely
forward-looking.

**It does not retrospectively fix anything.** §22 is on the 306-speaker corpus,
which contains only 44 of the reserved hundred. §25 and §26 are on the
562-speaker corpus but under the *old* unpinned split. None of them is comparable
to this section, and re-running §22's chain to close that gap would cost a
further ~14 h for a 44-speaker evaluation set. **The recommendation is to treat
§§22–26 as the historical record and require the pinned set from here on**,
rather than to spend the compute.

### Three things this is not

**The i-vector column is not §25 improved.** 0.157 → 0.135 at the best cell is
two standalone evaluations on two different sets of 187 speakers, and the
movement could be entirely that the pinned speakers are easier. Nothing here
measures which. The same applies to ECAPA's 0.033 → 0.030 against §26. **Only
the within-section pairing above is a comparison**; the columns quoting earlier
sections are context, not contrast.

**Part of six-of-six is the evaluation set, not the system.** The decision rule
is on the *upper* bound, and 187 speakers give a tighter interval than 102. §25
made this point and it applies here with the same force — with the additional
wrinkle that this particular 187 has never been evaluated on before, so whether
it is harder or easier than the last one is unmeasured.

**§23 still governs the absolute figures.** Both axes are LibriSpeech, and §23's
surviving explanation for the ψ₁ spike is that each LibriVox reader records in
one room with one microphone, so a per-reader channel is confounded with the
reader. A 0.93% equal error rate on read audiobook speech through a parametric
channel model is not a forecast for Zambian telephony.

### A control that was not asked for

ψ₁ on the neural back-end is **45.38**, against §26's **45.00** on the same
corpus under a different split. The two share a corpus size and an extractor and
differ only in which 562 speakers trained them, and they land within 0.4 of each
other. §23 concluded the ψ₁ spike is corpus structure rather than any particular
sample of speakers, and this is a third observation consistent with that,
obtained incidentally. The i-vector back-end sits at **39.06**; the gap between
the two representations is not something this section measures.

### The extractor advantage against §26

Both §26 and this section pair the same two front-ends at 562 back-end speakers,
on different evaluation sets:

| Condition | Dur. | §26 (old split) | §27 (pinned) | Change |
|---|---:|---:|---:|---:|
| clean | 30 s | −0.124 | −0.105 | +0.019 |
| clean | 15 s | −0.167 | −0.159 | +0.008 |
| clean | 5 s | −0.287 | −0.264 | +0.023 |
| babble 20 dB | 30 s | −0.124 | −0.120 | +0.004 |
| babble 20 dB | 15 s | −0.170 | −0.154 | +0.016 |
| babble 20 dB | 5 s | −0.261 | −0.221 | +0.040 |

The Change column is the difference of the two rounded figures beside it, so
the row adds up as read; at full precision two of the six differ by 0.001
from that. Every cell moves the same way and by a small amount — the
extractor's advantage is 0.004 to 0.040 smaller on the pinned set. **This is not a measurement of
anything.** The two evaluation sets differ, so no paired instrument applies, and
six same-signed changes of this size are as consistent with the pinned speakers
being slightly easier for the i-vector system as with any effect on the
extractor. It is recorded because it is the table a reader would construct
anyway, and constructing it here with the caveat attached is better than leaving
it to be constructed without one.
