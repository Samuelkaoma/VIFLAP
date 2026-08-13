# VIFLAP

**Calibrated multi-evidence case linkage for telephony-enabled fraud.**

VIFLAP evaluates how much the available evidence shifts the odds that two
reported incidents were conducted by the same actor, and reports that as a
calibrated likelihood ratio. It does not assert identity, and it has no
vocabulary for doing so.

This repository contains the reference implementation accompanying
[the doctoral research proposal](VIFLAP-Doctoral-Proposal.md). The proposal is
the primary document; this software exists to demonstrate that its framework is
computable and its methodology reproducible.

---

## The one thing to understand first

A likelihood ratio is **not** the probability that two incidents share an actor.
Converting one to the other requires prior odds, and those depend on the size and
composition of the population searched.

For a search against 100,000 enrolled incidents under a uniform prior:

```
likelihood ratio  = 1,000
prior odds        = 1 / 99,999          ≈ 1.0 × 10⁻⁵
posterior odds    = 1,000 × 1.0 × 10⁻⁵  ≈ 0.01
posterior         = 0.01 / 1.01         ≈ 0.99 %
```

**A thousand-to-one result from a national-scale search is about ninety-nine
percent likely to be wrong.** That calculation is the reason this system fuses
several weak evidence streams rather than comparing voices, and it is why every
result it produces carries its prior.

Fuse acoustic (1,000), behavioural (100) and transactional (50) evidence and the
posterior moves to roughly 98% — but only if the dependence between those streams
is modelled, because they share a cause and multiplying them counts it several
times.

---

## What is actually implemented

Every model below is trained from data. There are no placeholder scorers and no
constants standing in for measurements.

**Acoustic** — diagonal GMM-UBM by expectation-maximisation; i-vector total
variability with EM training and minimum-divergence re-estimation; length
normalisation, LDA and within-class covariance normalisation; two-covariance
PLDA with EM training and the exact closed-form log-likelihood ratio. The
i-vector posterior covariance is retained, giving a principled measure of how far
a short recording's representation was determined by the data rather than by the
prior.

**Signal processing** — framing and windowing, mel and linear filterbanks,
MFCC/LFCC with deltas, energy-and-flatness voice activity detection, linear
prediction with formant estimation by root-finding and prominence-based
selection, YIN pitch tracking, jitter/shimmer/HNR by pitch-synchronous
cross-correlation with Boersma's window correction, nasal segment detection with
antiformant estimation.

**Channel** — real AMR-NB via ffmpeg where available; otherwise an
analysis-by-synthesis CELP model (LPC, line-spectral-frequency quantisation,
adaptive codebook, algebraic fixed codebook) rather than a filter plus noise.
Spectrally shaped noise with SNR measured in the telephony passband. Packet loss
with pitch-period concealment.

**Spoofing countermeasure** — LFCC front-end with a two-class GMM, calibrated by
cross-validation, with an explicit out-of-domain indicator and first-class
cross-attack evaluation.

**Non-acoustic streams** — Dirichlet-multinomial and normal-inverse-gamma
marginal likelihood ratios. Rarity is handled by the model: sharing a cash-out
agent that handles 0.02% of volume counts for far more than sharing one that
handles 90%, without anyone choosing a weight.

**Calibration** — pool-adjacent-violators, linear logistic minimising `C_llr`
with an analytic gradient, kernel density ratio; `C_llr`, `C_llr_min`,
calibration loss, EER by exact interpolation; empirical bounds (ELUB) applied to
every reported value.

**Fusion** — naive summation (the prohibited baseline), linear logistic with
per-missingness-pattern models, a Gaussian latent model with exact marginalisation
over absent streams, and a Gaussian copula. Overstatement against naive summation
is measured on every comparison.

**Graph** — probabilistic and deterministic edges kept apart; path evidence as a
bound or a distribution, never a bare number; community detection with stability
assessed by resampling every edge from its own interval.

---

## What has been measured

One hypothesis has been tested on real speech. **H1** asks whether enough
speaker-discriminative information survives a narrowband telephony channel for
the acoustic stream to be worth including at all, and it was given its decision
rule before any data existed: supported if the *upper* bound of `C_llr_min`
reaches 0.30, falsified if the *lower* bound clears 0.50, inconclusive if the
interval spans both. The decision is made on the interval, never the point
estimate.

The current system trains on 306 LibriSpeech speakers and is evaluated on 102
held out from both it and the calibration set, through the parametric channel
model.

| | |
|---|---|
| Best cell — 12.2 kbit/s, clean, 30 s | `C_llr_min` **0.276** [0.215, 0.388], EER **7.89%** |
| 30-cell sweep, bias-corrected intervals | **0 supported, 6 falsified, 24 inconclusive** |
| Where the falsifications are | all six at 5 s — nothing at 15 s or 30 s is decided either way |

Four things that cost an experiment each:

**Capacity was never the constraint; the corpus was.** Doubling the model on the
same 125 speakers made all six tested cells significantly worse. Retraining the
*same* configuration on 306 speakers improved five of six, four of them surviving
Holm-Bonferroni. The between-speaker scatter of *S* speakers has rank *S* − 1, so
the training speaker count caps the discriminative subspace whatever the model
size — and the published system this is benchmarked against trained its extractor
on roughly 6,000 speakers.

**The countermeasure is structurally blind to one of its four attacks.** LFCCs
are computed from the magnitude spectrum, and phase randomisation preserves
magnitude spectra almost exactly. Twenty-five times the training speakers moved
that family from 50.00% to 52.60% EER, which is to say not at all. It needs a
phase-sensitive feature, not more data.

**A safeguard was being reported as a result.** Matched calibration looked cheap
at 0.054 bits — measured after the ELUB clip, which replaces the reported
likelihood ratio with a bound for 60.6% of trials. Before the clip it costs 0.26
bits, the same order as the discrimination floor. That conclusion is withdrawn in
place rather than deleted.

**The largest effect was checked for a confound and survived it.** The cepstral
normalisation window is fixed at 300 frames, which is 11% of a 30 s recording and
67% of a 5 s one — so the duration sweep varied the front-end alongside the
duration. Retrained with the window held duration-invariant, 94% of the 30 s → 5 s
gap remains and the contrast does not exclude zero.

The full write-up is in
[`docs/H1-acoustic-results.md`](docs/H1-acoustic-results.md), including a section
on everything these numbers do **not** establish and, where a conclusion has been
withdrawn, the wrong version with the reason it was wrong.

---

## Architecture

Five layers, dependencies pointing inward only. The rule is enforced by parsing
the import graph in `tests/architecture/`, not documented in this file.

```
viflap/
├── domain/          concepts and invariants — standard library only
├── analysis/        the science — numpy/scipy, no I/O
│   ├── dsp/         signal processing front-end
│   ├── channel/     controlled narrowband degradation
│   ├── speaker/     GMM-UBM → i-vector → LDA/WCCN → PLDA
│   ├── spoof/       countermeasure and validity gate
│   ├── patterns/    conjugate models for non-acoustic evidence
│   ├── behaviour/   idiolect and script structure
│   ├── calibration/ where a score becomes evidence
│   ├── fusion/      combination with dependence modelling
│   └── graph/       probabilistic linkage graph
├── evaluation/      speaker-disjoint splits, H1–H7 protocols, ablation
├── application/     use cases against ports
├── infrastructure/  adapters: audit chain, repositories, comparators
└── interfaces/      HTTP API and command line
```

### Three properties enforced by types rather than by discipline

**Absence is not neutrality.** A stream that produced nothing is a distinct type
carrying its reason. Substituting a likelihood ratio of one would assert that the
stream was computed and found the evidence equally probable under both
propositions — a fabricated observation.

**A likelihood ratio cannot exist without its prior.** The result type cannot be
constructed without a posterior, and a posterior cannot be constructed without an
explicit prior carrying its justification and the identity of whoever supplied
it. There is no default prior anywhere in the system.

**Uncalibrated numbers cannot be reported.** Every score passes through a fitted
calibrator at one architectural seam. A comparator built without one produces
absence, not numbers.

---

## Governance

These are capabilities the software lacks, not rules it is expected to follow.

- **No live path.** Nothing accepts a stream, a socket or a partial buffer.
- **No unbound operation.** Every operation takes a `CaseReference`, and that type
  cannot hold an invalid value.
- **Separation of duties.** Enrolment, query, export, audit and administration are
  distinct authorities; incompatible combinations cannot be assigned, because the
  `Principal` object refuses to exist. Each incompatible pair carries the specific
  risk it addresses.
- **Tamper-evident audit.** Hash-chained, append-only, fsync'd before
  acknowledgement, O(1) append. Refused and empty queries are recorded — they are
  what oversight most needs to see. The prior is recorded with every query.
- **Enforced retention.** Deletion is logged *before* it happens. Audit entries
  outlive the data they describe.
- **No vocabulary for identity.** Text crossing the boundary passes through a
  policy that rejects the language of identification, and a build-time check fails
  if that vocabulary appears in any emittable string or identifier. A field named
  `match_score` fails the build.

---

## Running it

Python 3.11+. The core needs only numpy and scipy — every model is implemented
against them, and no deep learning framework is required.

```bash
pip install -e ".[api,dev]"
```

Run the tests:

```bash
python -m pytest
```

575 tests: unit, property-based (hypothesis), integration, API contract, and
architecture. The architecture tests fail the build on a layering violation or on
identity vocabulary reaching emittable text.

Train the acoustic stack on real speech and run the H1 degradation sweep:

```bash
pip install -e ".[api,experiments,dev]"
# Stream a bounded subset of each partition; neither archive is kept.
python -m scripts.fetch_corpus
python -m scripts.fetch_corpus \
    --url https://www.openslr.org/resources/12/train-clean-360.tar.gz \
    --destination data/corpus/librispeech-360

python -m scripts.train_acoustic \
    --corpus data/corpus/librispeech --corpus data/corpus/librispeech-360 \
    --output models/acoustic_pooled.npz

python -m scripts.evaluate_h1 \
    --corpus data/corpus/librispeech --corpus data/corpus/librispeech-360 \
    --model models/acoustic_pooled.npz
```

**Pass every corpus root the model was trained on.** `--corpus` defaults to a
single partition, and a pooled model evaluated against one of them gets a split
that is internally disjoint, passes every check, and scores the model on speakers
it memorised. That has produced a spuriously good number here more than once.
Models now record their own training speakers so the check can be made against
what was actually trained on rather than against the split that happens to be in
hand.

Two paired comparisons answer questions the sweep cannot, by scoring two models
on identical trials so that the between-speaker variation cancels:

```bash
python -m scripts.compare_capacity --baseline models/acoustic.npz --variant models/acoustic_large.npz
python -m scripts.compare_cmvn --baseline models/acoustic_pooled.npz --variant models/acoustic_pooled_cmvn_utt.npz
```

Results are in [`docs/H1-acoustic-results.md`](docs/H1-acoustic-results.md).

Verify an audit chain without going through the service that wrote it:

```bash
viflap verify-audit ./data/audit/audit.jsonl
```

Evaluate a score file and get speaker-level intervals:

```bash
viflap evaluate scores.jsonl --resamples 1000
```

Start the API — note it is an application *factory*, so nothing is constructed at
import time:

```bash
uvicorn --factory viflap.interfaces.bootstrap:build_demonstration_container
```

The demonstration container has **untrained** calibrators, so every comparison it
performs returns absence for every stream. That is deliberate: a demonstration
deployment that produced plausible numbers would be indistinguishable in form
from an operational one and meaningless in content.

The investigator interface is in [`frontend/`](frontend/README.md).

---

## What this software is not

It is not a finding about telephony fraud. The acoustic stack has been trained
and evaluated on real speech — 510 usable LibriSpeech speakers across two
partitions, 306 of them training the current model, speaker-disjoint throughout —
and that is a genuine empirical result about *this system under simulated
narrowband degradation*. It is not a result about the target population. The
speech is read English recorded on good microphones; the target is
Bantu-language conversational telephony from callers attempting deception.

**The channel has never been validated, and it is harsher than the thing it
models.** ffmpeg with `libopencore-amrnb` was unavailable, so every figure comes
through the parametric CELP model. Measured against a matched-bandwidth
reference it introduces 6.6 dB of log-spectral distortion where the standards
literature designs for around 1 dB, and the nominal bitrate moves only 0.33 dB
of that — the labels "12.2 kbit/s" and "4.75 kbit/s" set an LSF quantiser step
and a pulse count, and no part of the implementation computes a bit budget.
Results obtained through this model and through a real coder must never be
pooled. Paired comparisons between two models are the robust part, since both
arms share the channel and it cancels in the difference.

Nothing about the other four evidence streams has been measured. H2, H3, H5 and
H7 remain untested, and no figure in this repository bears on them.

The corpus position for the target population is worse than the hour counts
suggest: roughly 390 hours of Zambian audio are available to the unsupervised
stages, which need no labels, against **twelve speakers** with recoverable
identity and more than one session — the quantity that actually determines
discrimination. BIG-C's `speaker_id` is a conversation-local participant index
rather than a person, and 37 of its 74 values carry both genders.

It is not a deployment. Production infrastructure, federation, real-time
operation and high availability are out of scope and stated as such in §12 of the
proposal.

It does not identify anyone. It reports how much the evidence shifts the odds,
with the prior that conditions it, the streams that produced it, what assuming
independence would have overstated, and the bounds of what the validation data
can support.
