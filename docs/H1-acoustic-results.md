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
development speakers, applied to evaluation speakers. Same-source mean +1.98
against different-source −12.44. Draws are mapped onto that empirical
distribution by quantile, so its asymmetry and its tails survive; fitting a
normal instead would smooth away exactly the region an overstatement study lives
in.

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

| ρ | naive sum | **linear-logistic** (independent, calibrated) | Gaussian latent (dependence) | Band changes |
|---:|---:|---:|---:|---:|
| 0.0 | 0.148 | **0.077** | 0.105 | 92.6% |
| 0.2 | 0.230 | **0.147** | 0.200 | 76.6% |
| 0.4 | 0.317 | **0.213** | 0.294 | 58.5% |
| 0.6 | 0.415 | **0.275** | 0.388 | 42.4% |
| 0.8 | 0.538 | **0.335** | 0.491 | 63.4% |

Values are `C_llr`; lower is better.

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

### What this still does not establish

**Point estimates from one seed, with no intervals**, which remains inconsistent
with the rest of this document and with §14's treatment of the acoustic
intervals. Repeating across seeds is required before any figure here is quoted
as a result.

**Correct specification by construction.** The generative process is a Gaussian
latent factor and the dependence model is a Gaussian latent-factor model, so the
comparison is the best case for the correction — and it loses anyway. Testing
misspecification (generate under a t-copula, fit the Gaussian) would make the
result stronger, and would be the honest operational case.

**Only the acoustic marginal is real**, and §13 gives reason to think the
behavioural stream is weaker than the 0.75 factor assumed here.

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
| Idiolect on 40-word transcripts | unusable; far below any published operating point |
| Idiolect on several hundred words | weak — forensic n-gram work reports `C_llr < 0.75` as a *goal* |
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
2. **Move character n-grams to the idiolect term**, or restrict them to
   script-bearing spans. The literature says plainly which component they belong
   to.
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

| Coder | Rate | LSD (dB) | Frames > 2 dB | seg. SNR (dB) | Delay |
|---|---:|---:|---:|---:|---:|
| **reference AMR-NB** | 12.20 | **3.22** | 87.6% | 2.18 | 39.3 |
| parametric model | 12.20 | **6.79** | 99.3% | 3.99 | 0.0 |
| **reference AMR-NB** | 4.75 | **5.71** | 99.8% | 1.27 | 39.3 |
| parametric model | 4.75 | **8.57** | 100.0% | 1.71 | 0.0 |

The figures are stable in the sample size, which is worth stating because the
first successful run measured six recordings from six speakers and this one
measured eight times as many across sixteen: 3.28 → 3.22 and 6.63 → 6.79 at
12.2 kbit/s. Whatever these numbers are limited by, it is not how many
recordings went into them.

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

### What is not yet reliable in this table, and is recorded rather than trimmed

**The coder-against-coder rows are omitted, because they were misaligned.** They
are the rows the whole measurement exists to produce, and they were wrong.
`estimate_delay` searches non-negative lags only, on the physical ground that a
codec delays its output rather than anticipating its input. That holds for a
coder against its source and fails for one coder against the other: this model
returns its output aligned with the input while AMR delays by about 40 samples,
so the model's signal *leads* and the estimator cannot express it. Asked for a
non-negative lag it returns the best one available, which is noise — the run
above reports a mean delay of 39 samples with a **maximum of 192** on a
comparison whose true offset is a constant −40.

That was visible only because the delay is recorded per comparison, which is
exactly what that field was put there for. Both coded signals are now aligned to
the band-matched source they share before being compared with each other, and a
rerun supersedes those rows.

The two rows against the source are unaffected: each coder is compared with its
own input, where the non-negative rule is correct, and both report a delay
consistent to within a sample across all 48 recordings.

**Segmental SNR should not be read across coders yet.** The reference reads
*lower* than the model (2.18 against 3.99 dB) while distorting the envelope
half as much. Delay is estimated to the sample, and a residual sub-sample offset
depresses a phase-sensitive measure while leaving an envelope measure alone, so
the most likely reading is that the reference's 39-sample alignment is not being
removed exactly. Log-spectral distance is the figure to take from this table.

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
