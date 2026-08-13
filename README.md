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

348 tests: unit, property-based (hypothesis), integration, API contract, and
architecture. The architecture tests fail the build on a layering violation or on
identity vocabulary reaching emittable text.

Train the acoustic stack on real speech and run the H1 degradation sweep:

```bash
pip install -e ".[api,experiments,dev]"
python -m scripts.fetch_corpus            # streams LibriSpeech, keeps a bounded subset
python -m scripts.train_acoustic          # speaker-disjoint, multi-condition
python -m scripts.evaluate_h1             # the sweep, with speaker-level intervals
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

It is not a finding about telephony fraud. The acoustic stack has now been
trained and evaluated on real speech — 251 LibriSpeech speakers, speaker-disjoint
throughout, reported in [`docs/H1-acoustic-results.md`](docs/H1-acoustic-results.md)
— and that is a genuine empirical result about *this system under simulated
narrowband degradation*. It is not a result about the target population. The
speech is read English recorded on good microphones; the target is
Bantu-language conversational telephony from callers attempting deception. The
channel is the parametric CELP model rather than a real AMR-NB coder, because
ffmpeg with `libopencore-amrnb` was unavailable, and results obtained through the
two must never be pooled.

Nothing about the other four evidence streams has been measured. H2, H3, H5 and
H7 remain untested, and no figure in this repository bears on them.

It is not a deployment. Production infrastructure, federation, real-time
operation and high availability are out of scope and stated as such in §12 of the
proposal.

It does not identify anyone. It reports how much the evidence shifts the odds,
with the prior that conditions it, the streams that produced it, what assuming
independence would have overstated, and the bounds of what the validation data
can support.
