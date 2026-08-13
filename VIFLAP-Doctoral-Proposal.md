# VIFLAP

## Calibrated Multi-Evidence Case Linkage for Telephony-Enabled Fraud

**Doctoral Research Proposal**

---

> **On citations.** References in §16 are drawn from the author's working
> knowledge of the literature and **must be verified against the primary source
> before submission**. Years, venues and exact claim attributions are
> provisional. Legal instruments cited in §10 must be checked against the
> current consolidated statute text, and no claim in this document should be
> read as legal advice.
>
> **On the reference implementation.** §9 describes software written alongside
> this proposal. It exists to demonstrate that the framework in §6 is
> computable and to make the methodology reproducible. It is *not* an empirical
> result: no claim in this document rests on it, and where it has been exercised
> on synthetic data that fact is stated explicitly.

---

## 0. Abstract

Telephony-enabled fraud — vishing, mobile-money social engineering, SIM-swap
account takeover — is investigated today as a sequence of isolated incidents.
Each case is worked outward from its registered subscriber identity, and when
that identity is false or disposable, which is the common case, the
investigation terminates. The acoustic content of the recorded call, the
behavioural signature of the operator, and the topology of the resulting money
movement are, in current practice, discarded.

This proposal argues that the obstacle is not the absence of a voice-matching
capability. It is the absence of a principled method for **combining weakly
individuating evidence into a calibrated statement of uncertainty**. The
scientific literature is unambiguous that the human voice is not an
individuating biometric in the sense that a fingerprint or a DNA profile is; the
"voiceprint" analogy was discredited decades ago and its rehabilitation would be
a scientific error. Voice is a physiological–behavioural hybrid whose
discriminative power is moderate, degrades sharply under the narrowband speech
codecs that carry the evidence in question, and degrades further under
deliberate disguise.

VIFLAP takes that limitation as its premise rather than its obstacle. The
research question is not *whether two recordings share a speaker* but *how much
the totality of available evidence shifts the odds that two incidents share an
actor*, expressed as a calibrated likelihood ratio suitable for triage by a
human investigator and for scrutiny by a court.

The work develops four things. (i) A characterisation of speaker-discriminative
information surviving the African mobile telephony channel, including under
adversarial disguise. (ii) A fusion framework combining acoustic, behavioural,
temporal, transactional and device evidence with explicit treatment of the
conditional-dependence problem, together with a quantification of the confidence
overstatement that assuming independence produces. (iii) An evaluation
methodology grounded in forensic likelihood-ratio metrics rather than
classification accuracy, including empirical bounds on what a validation set can
support. (iv) A governance architecture in which constraints against misuse are
structural properties of the software rather than policy commitments.

The empirical contribution is anchored in a setting the speech literature has
largely ignored: multilingual, heavily code-switched Zambian telephony, where
speaker recognition systems trained on English and Mandarin corpora have no
established performance characterisation.

---

## 1. Introduction

### 1.1 The investigative failure

An investigator receiving a mobile-money fraud complaint in Lusaka today follows
a workflow that is essentially a single-hop lookup. The complainant supplies a
calling number. The number resolves — through mandatory SIM registration — to a
subscriber record. The record is fraudulent, or belongs to a person whose
identity documents were misused, or belongs to a SIM purchased through an agent
who did not enforce registration. The chain terminates. The recorded call, if it
exists at all, is treated as narrative context rather than as evidence carrying
extractable structure.

The consequence is not merely that individual cases fail. It is that the
*population structure* of the offending is invisible. Fraud of this kind is
overwhelmingly conducted by small, persistent, organised groups running scripted
operations at volume. Each of their incidents is investigated as though it were
sui generis. The single most valuable investigative fact — that incident 47 and
incident 2,113 were run by the same person — is never established, because
nothing in the workflow is designed to establish it.

### 1.2 Why "add voice recognition" is the wrong response

The intuitive response is to add speaker recognition and search for similar
voices. This proposal argues that response is technically naive and, deployed
without the framework developed here, actively dangerous.

The reasons are developed fully in §4 and §6, but stated briefly: the
discriminative power of a speaker embedding extracted from an 8 kHz, 12.2 kbit/s
AMR-coded telephone call is not sufficient to individuate a speaker within a
population of realistic size. A likelihood ratio of 1,000 — which would
represent strong performance under these conditions — applied against a
100,000-entry database with a uniform prior yields a posterior probability of
approximately one percent. A system that reports such a result as a positive
finding is manufacturing false accusations at scale, and doing so with the
institutional authority of a computer.

The failure mode is not hypothetical. It is the documented history of forensic
voice identification.

### 1.3 The central claim

> **Thesis.** No single trace recoverable from a fraudulent telephone call is
> individuating under realistic channel and adversarial conditions. Actionable
> investigative linkage is achievable through the *calibrated fusion* of
> multiple weakly individuating traces — acoustic, behavioural, temporal,
> transactional and device — provided that (a) the dependence structure between
> evidence streams is modelled rather than assumed away, (b) the output is a
> calibrated likelihood ratio rather than a decision, (c) the prior odds
> appropriate to a database search are applied explicitly rather than
> implicitly, and (d) reported likelihood ratios are bounded by what the
> validation data can support.

This claim is falsifiable. §5 states the hypotheses under which it fails, with
their falsification conditions fixed in advance.

### 1.4 What this work is not

Explicitly excluded from scope, and stated here because reviewers will otherwise
assume them:

- **Not an identification system.** VIFLAP produces ranked, explained
  investigative hypotheses. It does not assert identity, and no output is
  designed or represented as proof of identity.
- **Not a real-time interception system.** The architecture operates on lawfully
  obtained recordings associated with filed complaints. It has no live-tap
  capability, and §10.3 describes the architectural constraints that keep it
  that way.
- **Not an autonomous decision-maker.** No enforcement action is triggered by
  system output. Human review is a structural requirement, not a policy overlay.
- **Not a general surveillance platform.** §11 treats function creep as the
  primary threat to be engineered against.
- **Not a production deployment.** §12 draws the boundary between what is
  empirically evaluated and what is delivered as design and prototype.

---

## 2. Background

### 2.1 The structure of telephony-enabled fraud

Telephony-enabled fraud is distinguished from generic cybercrime by its reliance
on synchronous human interaction. The offender must speak. This is a
vulnerability, and it is the vulnerability this work exploits — not because
speech identifies, but because speech, unlike a spoofed header or a disposable
SIM, is produced by a physical human body executing a learned behavioural
routine, and is therefore *constrained*.

| Pattern | Mechanism | Trace left |
|---|---|---|
| Vishing / impersonation | Offender poses as bank, MNO, or agent staff | Audio, script structure, timing |
| Mobile-money social engineering | Victim induced to authorise or reverse a transfer | Audio, transaction graph |
| SIM-swap takeover | Offender obtains replacement SIM for victim's MSISDN | Registration records, device IMEI |
| Advance-fee / prize fraud | Victim pays a release fee | Audio, script, transaction graph |
| Agent-collusion cash-out | Insider agent launders proceeds | Transaction graph, location |

The critical observation is that these patterns are **scripted and repeated**.
An operator running the same pretext forty times a week produces forty
recordings of substantially the same speech act. This repetition is what makes
statistical linkage tractable; it is also what makes the behavioural evidence
stream informative independently of the acoustic one, and what makes the
behavioural stream survive an attack the acoustic stream cannot (§4.4).

### 2.2 Why the mobile-money context matters

In markets where mobile money is the dominant retail financial rail rather than a
supplementary one, the fraud economy is structurally different from card-centric
markets. Value moves between wallets in near-real-time, settlement is
effectively irreversible, cash-out occurs through a distributed agent network
with variable compliance discipline, and the victim population includes a large
segment with limited prior exposure to financial fraud typologies.

The investigative implication is that the transaction graph is unusually
information-rich. Wallet-to-wallet transfers, agent cash-out points, device
identifiers and SIM registration records exist in structured form, held by a
small number of operators. This is materially better raw material than a
card-fraud investigator typically has. It is, at present, not joined to anything.

It also has a property that the acoustic evidence lacks and that §6.9 exploits:
its elements have *known population frequencies*. Whether two incidents cashed
out through the same agent is evidence in proportion to how unusual that agent
is, and the operator's own records establish how unusual. This is the structure
of trace evidence — glass, fibres, DNA — and it admits the same treatment.

### 2.3 The Zambian setting as a research site

Zambia is proposed as the empirical setting for reasons that are scientific, not
merely logistical.

- **Linguistic diversity under-represented in the literature.** Bemba, Nyanja,
  Tonga, Lozi, Lunda, Luvale and Kaonde, alongside English, with pervasive
  code-switching *within utterances*. Speaker recognition systems are trained
  predominantly on English and Mandarin corpora. Their performance on
  code-switched Bantu-language telephony is, as far as the author can establish,
  uncharacterised. This is a genuine gap, not a localisation exercise.
- **Institutional separation.** The Zambia Police Service, the Financial
  Intelligence Centre, ZICTA and the mobile network operators hold complementary
  fragments of the same cases with no technical mechanism for correlating them.
- **Mandatory SIM registration.** Registration is enforced, which means the
  failure of registration to prevent fraud is empirically observable rather than
  speculative — a useful negative result in its own right.
- **An extant legal framework.** Data protection and cybercrime statutes exist
  (§10.2), meaning the governance question is "what does compliance require"
  rather than "is there any law at all".

---

## 3. Related Work

This section supports a specific argument: that four mature literatures each
solve a part of this problem, and that none addresses the combination, which is
where the difficulty actually lies.

### 3.1 Speaker recognition

The field's trajectory is well documented. Gaussian mixture models with
universal background model adaptation (Reynolds et al., 2000) gave way to joint
factor analysis and then to the i-vector total-variability representation (Dehak
et al., 2011), which established the paradigm of a fixed-dimensional utterance
embedding scored by a probabilistic linear discriminant analysis back-end. Deep
neural architectures replaced the generative front-end: x-vectors (Snyder et
al., 2018) demonstrated that a time-delay network with statistics pooling
trained discriminatively outperformed i-vectors, and ECAPA-TDNN (Desplanques et
al., 2020) added channel attention and multi-scale aggregation. Self-supervised
pre-training on large unlabelled speech corpora (e.g. WavLM, Chen et al., 2022)
has since provided representations that transfer well to speaker tasks with
limited labelled data — a property of direct relevance to a low-resource
language setting.

**What this literature optimises for.** Verification accuracy on benchmark
corpora, predominantly VoxCeleb, under conditions where the two utterances are
cooperative, unmodified, and drawn from wideband internet audio. Reported equal
error rates in the low single digits are conditioned on those assumptions.

**What it does not address.** The literature is largely silent on deliberate
disguise, treats channel mismatch as a nuisance to be normalised away rather
than as the primary operating condition, and — critically — produces *scores*
rather than *calibrated likelihood ratios*. A cosine similarity of 0.72 has no
interpretation without a reference population.

**A note on why this proposal builds on the older paradigm.** The reference
implementation (§9) uses the i-vector/PLDA stack rather than a neural embedding
extractor, and the choice is deliberate rather than conservative. Three reasons.
The generative back-end produces a quantity that is already a log-likelihood
ratio in form, so the path from score to calibrated evidence is short and
inspectable. The posterior covariance of the i-vector is available in closed
form, which gives a principled measure of how far a short recording's
representation was determined by the data rather than by the prior — a quantity
this application needs and a discriminative embedding does not provide. And the
whole stack is trainable on the data volume a low-resource setting realistically
affords. A neural extractor is retained as an alternative front-end to be
compared against this baseline (§8.6), not assumed superior.

### 3.2 Forensic voice comparison

A separate literature, with limited overlap in authorship or venue, addresses
exactly the interpretive question the speaker recognition literature leaves
open. Following the discrediting of spectrographic "voiceprint" identification
(National Research Council, 1979; and the broader critique in NRC, 2009), the
forensic phonetics community converged on the likelihood-ratio framework as the
logically correct expression of evidential value (Rose, 2002; Morrison, 2011;
Drygajlo et al., ENFSI methodological guidelines). Under this framework the
expert reports the ratio of the probability of the observed evidence under
same-origin and different-origin hypotheses, and explicitly does **not** report
the probability of the hypothesis, which requires a prior the expert does not
possess.

Associated with this framework is a metric apparatus largely unknown in the
mainstream speaker recognition literature: the log-likelihood-ratio cost
`C_llr` and its calibration-minimised variant `C_llr^min` (Brümmer & du Preez,
2006), the Tippett plot for displaying LR distributions across same- and
different-origin trials, pool-adjacent-violators calibration, and — more
recently — empirical bounds on the likelihood ratios a validation set can
support (Vergeer et al., 2016).

**What this literature does not address.** It is built around the single-case,
expert-witness paradigm: one questioned recording, one suspect recording, one
report. It has little to say about database search at scale, nothing about
fusion with non-acoustic evidence streams, and no engineering account of how to
operate at the volume an investigative unit requires.

### 3.3 Anti-spoofing and synthetic speech detection

The ASVspoof challenge series (2015, 2017, 2019, 2021) established the
evaluation infrastructure for detecting replay, voice conversion and
text-to-speech attacks against speaker verification, with the tandem detection
cost function providing a joint metric for the verification-plus-countermeasure
system. The consistent and important finding across editions is that
countermeasure performance degrades severely against attack types not
represented in training — generalisation to unseen synthesis methods remains
unsolved.

**Relevance and limitation.** For this work, spoof detection is not a security
control on an authentication gate; it is a **precondition on evidential
validity**. If a recording is synthetic, its acoustic evidence about a human
speaker is not weak — it is meaningless, and must be excluded from fusion rather
than down-weighted. The literature does not address this framing.

The generalisation failure is treated here not as a limitation to be noted but
as a condition to be *detected*. §7.2 describes an out-of-domain indicator: a
recording improbable under both the genuine and the spoofed model is one the
detector has no basis to judge, and the system returns "indeterminate" rather
than a confident verdict. Indeterminate does not admit acoustic evidence.

### 3.4 Graph analytics for financial crime

Fraud detection over transaction graphs is well developed, spanning community
detection (Blondel et al., 2008; Traag et al., 2019), centrality-based risk
propagation, and graph neural network approaches to node and edge
classification. The mature commercial and academic systems operate on
transactional and identity-attribute edges.

**What is absent.** These graphs contain no biometric or behavioural edge type.
The notion of an edge asserting "these two incidents were probably conducted by
the same voice, with likelihood ratio 340" does not appear in the literature, and
the graph algorithms are not formulated to propagate *uncertainty* along such
edges. Treating a probabilistic linkage edge as though it were a deterministic
transaction edge is an error that compounds across path traversals, and it is
silent: the algorithm returns communities that look exactly like the communities
it returns from real edges.

### 3.5 Evidence fusion

Score-level fusion by logistic regression trained to minimise `C_llr` is
established practice in multimodal biometrics and in the NIST evaluation
tradition (Brümmer et al.; the FoCal and BOSARIS toolkits). Bayesian networks
provide a more general formalism for combining dependent evidence, and copula
methods separate marginal behaviour from dependence structure.

**The unsolved part.** Fusion theory is clean when evidence streams are
conditionally independent given the hypothesis. In this application they
demonstrably are not — the same operator produces correlated acoustic, lexical,
temporal and transactional signatures — so naive log-LR summation overstates the
combined evidence, potentially by orders of magnitude. This is the single
hardest technical problem in the proposal and is treated formally in §6.6.

### 3.6 Gap matrix

| Capability | Speaker rec. | Forensic VC | Anti-spoof | Graph fraud | **VIFLAP** |
|---|---|---|---|---|---|
| Speaker discrimination | ✓ | ✓ | — | — | ✓ |
| Calibrated LR output | ✗ | ✓ | ✗ | ✗ | ✓ |
| Robust to narrowband codec | partial | partial | partial | n/a | ✓ (characterised) |
| Deliberate disguise modelled | ✗ | partial | ✗ | n/a | ✓ |
| Synthetic-speech gating | ✗ | ✗ | ✓ | ✗ | ✓ |
| Out-of-domain detection for the countermeasure | ✗ | ✗ | partial | ✗ | ✓ |
| Non-acoustic evidence fused | ✗ | ✗ | ✗ | ✓ | ✓ |
| Dependence structure modelled | ✗ | ✗ | ✗ | ✗ | ✓ |
| Database-search prior applied | ✗ | partial | ✗ | ✗ | ✓ |
| Reported LR bounded by validation support | ✗ | partial | ✗ | ✗ | ✓ |
| Uncertainty propagated in graph | n/a | n/a | n/a | ✗ | ✓ |
| Low-resource / code-switched | ✗ | ✗ | ✗ | n/a | ✓ |

The claim of this proposal is located in the rightmost column's lower half: not
in any individual capability, but in the rows that are empty across every
existing literature.

---

## 4. The Individuality Problem

*This section establishes the scientific premise on which the entire design
rests. It is placed before the research questions because those questions are
unintelligible without it.*

### 4.1 There is no voiceprint

The proposition that a voice carries an individuating signature analogous to a
fingerprint is false, and its historical propagation caused documented
miscarriages of justice. Spectrographic voice identification, introduced in the
1960s and marketed under the "voiceprint" label, was subjected to sustained
methodological criticism culminating in the National Research Council's 1979
assessment and reiterated in the 2009 NRC review of forensic science generally.
The core objection is that the analogy fails at the level of physics: a
fingerprint is a static morphological structure imprinted directly, whereas a
speech signal is the output of a *dynamical system under active neuromuscular
control*, which the speaker can modulate, which varies with health, affect,
fatigue and interlocutor, and which is transmitted through a channel that
discards much of the information the analysis depends upon.

This proposal adopts the term **speaker-discriminative information** and avoids
"voiceprint" entirely. The reference implementation enforces that avoidance
mechanically (§9.4).

### 4.2 What is anatomically constrained

Some acoustic properties are bounded by physical structures the speaker cannot
reshape. These are the most valuable, and the most resistant to disguise.

**Vocal tract resonance.** Modelling the tract as a uniform tube closed at the
glottis and open at the lips gives resonances at

```
F_n ≈ (2n − 1) · c / (4L)
```

with `c ≈ 35,000 cm/s` and adult tract length `L ≈ 17.5 cm`, placing F1 near
500 Hz. Tract length is an anatomical parameter and is the dominant determinant
of formant scaling. **However**, effective length is modulable: larynx raising
and lowering, and lip protrusion versus spreading, together permit a change on
the order of ±10–15%. The constraint is real but soft.

**Nasal cavity geometry.** This is the most disguise-resistant acoustic resource
available. The nasal passages and paranasal sinuses form a fixed, non-muscular
resonant cavity. During nasal consonants the tract is coupled to this cavity,
imposing characteristic formants and — diagnostically — *antiformants* (spectral
zeros) whose positions are determined by a morphology the speaker has no motor
pathway to alter. Forensic phonetic work on nasal segments (e.g. Amino & Arai)
supports their comparatively high speaker-specificity. A speaker attempting
disguise can change pitch, register, rate and accent; they cannot change the
shape of their sinuses.

Two qualifications belong here rather than in a footnote. Antiformants are
spectral *zeros*, and the linear-prediction models used throughout speech
analysis fit *poles*; estimating a zero from an all-pole model is indirect and
poorly conditioned on short band-limited frames. And the feature is defeated
entirely by nasal occlusion — deliberate or from a head cold — which is a
trivial-effort attack on the most valuable feature class (§4.4).

**Glottal source characteristics.** Vocal fold mass, length, tension and closure
pattern shape the excitation signal, observable through jitter, shimmer,
harmonics-to-noise ratio and open quotient obtained by inverse filtering. These
are partially under voluntary control, are sensitive to respiratory illness,
hydration, fatigue and age, and — as §4.5 develops — are the feature class the
channel damages most severely.

### 4.3 What is behaviourally stable

A second class of evidence derives not from anatomy but from over-learned motor
and linguistic habit: coarticulation patterns, segment-specific durations,
voice-onset timing, prosodic contour shapes, characteristic disfluency and
filled-pause behaviour, and idiolectal lexical choice.

These are consciously alterable in principle. In practice they are alterable
only under sustained attention, and attention is a depleting resource. An
offender executing a scripted social-engineering pretext is allocating cognitive
effort to the deception itself. The empirical prediction — testable, and stated
as H3 in §5 — is that behavioural markers **reassert themselves as call duration
increases**, and that disguise consistency degrades measurably over the course of
an interaction.

For a scripted, repetitive offender this class has a further property that
acoustic evidence lacks: it is **partially invariant to the speaker**. Script
structure, pretext sequencing and target-selection behaviour characterise the
*operation*, and therefore link incidents even across different operators within
the same organised group.

This proposal treats that as a first-class distinction rather than a remark. The
behavioural stream reports two components separately: **idiolect** (function-word
usage, disfluency inventory, code-switching behaviour), which is
speaker-specific and defeated by delegation, and **script structure** (move
inventory, move sequence, characteristic phrasing), which is operation-specific
and survives it. "These two incidents were run by the same operation but
probably not the same person" is a finding, and a single merged behavioural
score cannot express it.

### 4.4 What an adversary controls

An honest threat model must enumerate the adversary's capabilities:

| Technique | Effort | Expected effect on acoustic LR |
|---|---|---|
| Pitch shift (falsetto / creak) | Trivial | Severe degradation |
| Rate and loudness alteration | Trivial | Moderate |
| Accent / dialect adoption | Moderate | Severe on behavioural, moderate on acoustic |
| Larynx and lip manipulation | Moderate | Moderate — bounded by §4.2 |
| Nasal occlusion (physical) | Trivial | Severe on the most valuable feature class |
| Electronic pitch/formant shifting | Low | Severe |
| Voice conversion | Moderate | Evidence invalidated — must be gated, not weighted |
| Neural TTS | Moderate | Evidence invalidated — must be gated |
| Delegating calls to another person | Trivial | Acoustic evidence absent; behavioural script persists |

Two entries deserve emphasis. **Nasal occlusion** degrades precisely the feature
class §4.2 identifies as most valuable, at no cost to the offender. And
**delegation**: an organised group can defeat speaker-based linkage entirely by
rotating who speaks. The behavioural script and transactional streams are the
only defence, which is a further argument that the fusion framework is the
contribution and the acoustics are a component.

Published disguise studies report equal error rates degrading from low single
digits to the 20–30% range. Every claim in this proposal is conditioned on
disguise state, and the evaluation framework refuses to report a metric without
one.

### 4.5 What the channel destroys

The evidence arrives through a narrowband mobile speech codec. Adaptive
Multi-Rate narrowband operates at 4.75–12.2 kbit/s over an 8 kHz sampled,
approximately 300–3400 Hz passband signal. The consequences are specific and
severe:

- All energy above ~3.4 kHz is discarded. Fricative spectra, which carry
  substantial speaker-discriminative information, are largely destroyed.
- The codec is not a noisy channel. It is an **algebraic code-excited linear
  prediction** coder: it transmits a quantised spectral envelope, a pitch lag,
  and an *index* into a codebook of excitation vectors chosen to minimise a
  perceptually weighted error. The decoder then *constructs* a waveform. At
  12.2 kbit/s a 5 ms subframe is represented by ten pulses; at 4.75 kbit/s, by
  two.
- The consequence for the glottal source features of §4.2 is categorical rather
  than gradual. Jitter and shimmer are cycle-to-cycle variations in period and
  amplitude, typically below one percent — properties of exactly the fine
  structure the coder replaces. Measured on decoded low-rate speech, they
  substantially characterise the codebook rather than the larynx.
- Rate adaptation means bitrate varies *within a single call* with radio
  conditions, so channel conditions are non-stationary within an utterance.
- Handset transducer variation, ambient noise, packet loss concealment and
  transcoding across operator boundaries add further mismatch.

This has a methodological consequence that §8.1 acts on. Simulating the channel
by band-pass filtering and adding noise — the obvious approach — leaves the
excitation's temporal fine structure intact, merely buried. Glottal features
measured through such a simulation survive far better than they do in reality,
and any conclusion drawn about them would be optimistic in the direction that
overstates the evidence. The simulation used here is an analysis-by-synthesis
CELP model, and where a real AMR codec is available it is used in preference.

The literature's reported error rates are not obtained under these conditions.
Establishing what performance is actually attainable here is an empirical
contribution in itself (H1).

### 4.6 The consequence

Sections 4.1–4.5 establish that the acoustic evidence available in this
application is: non-individuating by nature, softly bounded by anatomy,
adversarially degradable, and channel-degraded before it is ever analysed.

The correct engineering response is not to seek a better voice comparator. It is
to **stop asking the voice to carry the inferential load alone**. Formally: to
treat each evidence stream as contributing a likelihood ratio of modest
magnitude, and to combine them in a framework that is honest about their
dependence, about the prior odds of the search being conducted, and about the
limits of what the validation data can support.

That framework is §6. It is the substance of this thesis.

---

## 5. Research Questions and Hypotheses

Each hypothesis is stated so that it can fail, with its falsification condition
given explicitly and fixed in advance. §8.5 describes the pre-registration
arrangement that makes "in advance" verifiable rather than asserted.

A decision rule that distinguishes only "supported" from "not supported" is
inadequate, and the evaluation framework does not use one. Each hypothesis
admits three outcomes: **supported**, **falsified**, and **inconclusive** — the
last being the honest verdict when the confidence interval spans both decision
thresholds. Reporting an underpowered experiment as "not supported" converts a
statement about the experiment into a statement about the world.

**H1 — Channel viability.**
Speaker embeddings extracted from AMR-NB-coded telephony audio retain sufficient
discriminative information to yield `C_llr^min ≤ 0.30` on same-condition trials
in the target language population.
*Supported if* the upper bound of the speaker-level confidence interval on
`C_llr^min` is at or below 0.30. *Falsified if* the lower bound exceeds 0.50, at
which point acoustic evidence contributes too little to justify inclusion and
the thesis proceeds on non-acoustic streams alone.

**H2 — Disguise resistance is feature-dependent.**
Under deliberate disguise, feature classes degrade unequally, and nasal-segment
and articulatory-timing features retain measurably more discriminative
information than global spectral or F0-derived features.
*Supported if* the resistant classes' `C_llr^min` interval lies below the fragile
classes' by more than 0.05. *Falsified if* the intervals overlap such that the
separation is consistent with zero — which would remove the basis for
disguise-aware feature weighting.

**H3 — Disguise decays with duration.**
Disguise consistency degrades over call duration; the discriminative information
recoverable from the final third of a disguised call materially exceeds that
from the first third.
*Supported if* the lower bound of the paired within-speaker difference exceeds
0.03. *Falsified if* the interval lies entirely below zero.

**H4 — Cross-lingual and code-switching penalty.**
Embedding extractors trained on English-dominant corpora incur a quantifiable
performance penalty on Zambian-language and intra-utterance code-switched
telephony, and this penalty is reducible by self-supervised adaptation on
unlabelled in-domain audio.
*Supported if* the penalty interval lies above 0.02. *Falsified if* it lies
below — **a valuable negative result**, since the transferability of these models
to Bantu-language telephony is currently unestablished in either direction.

**H5 — Fusion superiority.**
Calibrated fusion of acoustic, behavioural, temporal, transactional and device
evidence achieves lower `C_llr` than the best single stream, **and** differs from
naive conditional-independence summation.
*Supported if* both conditions hold with margins exceeding 0.02. Both are
required: beating the best single stream while performing no differently from
naive summation would mean the dependence modelling contributed nothing.
*Falsified if* fusion's interval does not lie below the best single stream's —
the single most damaging outcome for the thesis, and therefore the hypothesis
requiring the most rigorous test.

**H6 — Investigative utility.**
Investigators using calibrated LR-ranked linkage hypotheses identify true case
linkages at a higher rate, and act on false linkages at a lower rate, than
investigators using current workflow or uncalibrated similarity scores.
*Falsified if* no improvement, or — importantly — if calibrated output produces
*worse* decisions than raw scores due to misinterpretation of the LR, which
would be a significant human-factors finding worth reporting.

**H7 — Synthetic-speech gating.**
Synthetic and converted speech generated by methods **absent from training** can
be detected at a rate sufficient to gate acoustic evidence without unacceptable
loss of genuine evidence.
*Supported if* the upper bound of `C_llr` on held-out attack types is at or below
0.40. *Falsified if* the lower bound exceeds 1.0, i.e. the detector is worse than
uninformative on unseen attacks — in which case acoustic evidence must be
systematically discounted and the thesis must state that limitation.

Evaluation on seen attacks is reported for comparison but decides nothing. An
offender using a synthesiser the training set contains is not the offender this
system will meet.

### 5.1 Operationalisation

| H | Primary metric | Unit of resampling | Design | Decides |
|---|---|---|---|---|
| H1 | `C_llr^min` | speaker | bitrate × noise × duration sweep | acoustic stream inclusion |
| H2 | `C_llr^min` by feature class | speaker | disguise-condition ablation | disguise-aware weighting |
| H3 | paired ΔC_llr^min | speaker | within-call segmentation | duration-dependent weighting |
| H4 | ΔC_llr^min | speaker | cross-corpus, stratified by code-switch density | adaptation requirement |
| H5 | `C_llr` | operation | all-subset ablation | the central thesis |
| H6 | decision quality; false-linkage action rate | participant | 3-arm controlled study | interface design |
| H7 | `C_llr` on held-out attacks | speaker | cross-attack generalisation | gate operating point |

The unit of resampling in the third column is load-bearing and is discussed in
§8.4. For H5 it is the *operation* rather than the speaker, because a group
rotating its callers produces comparisons that share an operation without
sharing a speaker, and treating those as independent understates variance.

---

## 6. Theoretical Framework

### 6.1 The likelihood ratio

For a pair of incidents with observed evidence `E`, define the competing
propositions:

- `H_ss`: the incidents were conducted by the same actor
- `H_ds`: the incidents were conducted by different actors

The evidential value is

```
LR = p(E | H_ss) / p(E | H_ds)
```

The system reports `LR`. It does **not** report `p(H_ss | E)`, because that
requires a prior the system cannot legitimately supply and whose choice is a
matter for the investigator and ultimately the court. Conflating the two is the
**prosecutor's fallacy**, and preventing it is a design requirement of the
software and the user interface (§7.6, §9.4), not merely a caveat in
documentation.

Two notes on the propositions themselves. They are stated at the level of the
*activity* — did the same person conduct both incidents — rather than at the
source level of a single recording. This matters because the behavioural and
transactional streams speak to the operation, and an organised group that
rotates who speaks can defeat a source-level proposition while leaving an
activity-level one intact. And the pair is the unit of comparison, which makes
the relation symmetric: the evidential value of a pair cannot depend on which
incident the investigator happened to open first. The reference implementation
enforces that symmetry structurally.

### 6.2 The acoustic model, formally

The acoustic stream is developed here in full because it is the stream whose
limitations drive the whole design, and because the choice of estimator
determines what uncertainty can be quantified.

**Front-end.** A recording is reduced to a sequence of cepstral feature vectors
over voice-active frames. Voice activity detection is the highest-leverage
component of the front-end: every statistic downstream is computed over the
frames it admits, and silence in a coded telephone call is not silence but
network-generated comfort noise, which characterises the operator rather than
the speaker.

**Universal background model.** A diagonal-covariance Gaussian mixture with `C`
components, trained on pooled in-domain speech, describes "speech in general":

```
p(x) = Σ_c w_c N(x ; μ_c, Σ_c)
```

The diagonal restriction is not a loss of generality: a mixture of enough
diagonal Gaussians approximates any density, and trading covariance parameters
for components spends the same budget on quantities that can be estimated from
the data a low-resource setting affords.

**Sufficient statistics.** An utterance `u` of any length reduces to zeroth- and
first-order Baum-Welch statistics, the latter centred on the UBM means:

```
N_c(u) = Σ_t γ_tc            F_c(u) = Σ_t γ_tc (x_t − μ_c)
```

**Total variability.** The adapted mean supervector is modelled as a
low-dimensional offset from the UBM supervector `m`:

```
M(u) = m + T w(u),     w(u) ~ N(0, I)
```

with `T` of shape `(CD × R)`. The i-vector is the posterior mean of `w`:

```
L(u)  = I + Σ_c N_c(u) T_c^T Σ_c^{-1} T_c
ŵ(u)  = L(u)^{-1} Σ_c T_c^T Σ_c^{-1} F_c(u)
```

**The property this buys.** `L(u)` depends only on the zeroth-order statistics —
that is, on how much speech there was. A short recording gives small `N_c`, so
`L` approaches the identity and `ŵ` is shrunk toward the prior mean. The
posterior covariance `L^{-1}` is therefore a closed-form measure of *how much of
the representation was determined by the recording rather than by the model's
prior*. This proposal reports that quantity with every acoustic likelihood
ratio. It is the formal expression of "eleven seconds of speech supports less
than ninety seconds does", and it is not available from a discriminative
embedding extractor, which returns a point with no attached uncertainty.

**Back-end.** After length normalisation, linear discriminant projection and
within-class covariance normalisation, a two-covariance PLDA model treats an
observation as

```
x = μ + y + e,     y ~ N(0, B),  e ~ N(0, W)
```

with `y` the speaker's position, drawn once per speaker, and `e` everything that
varies between recordings of that speaker. Simultaneously diagonalising `B` and
`W` gives coordinates in which `W = I` and `B = diag(ψ)`, and the log-likelihood
ratio for two observations separates by dimension. For dimension `d` with
between-speaker variance `ψ` and observations `a`, `b`:

```
Σ_ss = [[ψ+1, ψ], [ψ, ψ+1]]          Σ_ds = [[ψ+1, 0], [0, ψ+1]]

log LR_d = ½[log|Σ_ds| − log|Σ_ss|] − ½ (a,b) Σ_ss^{-1} (a,b)^T
                                     + ½ (a,b) Σ_ds^{-1} (a,b)^T
```

which evaluates to

```
log LR_d = ½[2 log(ψ+1) − log(2ψ+1)]
         − ½ [(ψ+1)(a² + b²) − 2ψab] / (2ψ+1)
         + ½ (a² + b²) / (ψ+1)
```

Two limits check against intuition and both hold. As `ψ → 0` the dimension
carries no between-speaker variability and contributes exactly zero. As
`ψ → ∞` the contribution tends to `½ log(ψ/2) − (a−b)²/4`: large when the two
observations agree and strongly negative when they do not.

**What this does and does not give.** The output is a likelihood ratio *under
the model's assumptions* — Gaussian speaker and session distributions, and a
training population representative of the relevant population. Neither holds
exactly. It is therefore treated throughout as an **uncalibrated score** that
must pass through empirical calibration before it is reported as evidence.
Reporting a raw PLDA score as a likelihood ratio is the specific error the
forensic literature warns against, and the fact that the number is already
*shaped* like a likelihood ratio makes the error easier to commit, not harder.

### 6.3 Calibration, formally

Let `s` be a score from any stream. Calibration is the mapping `s ↦ log LR`
estimated on held-out trials with known ground truth. Three estimators are
considered.

**Linear logistic (FoCal).** `log LR = a·s + b`, with `(a, b)` minimising `C_llr`
directly. The objective is convex — `log(1 + e^{-x})` is convex and its argument
is affine in the parameters — so the optimum is unique and refitting on the same
data gives the same answer. That reproducibility is a requirement, not a
convenience: a result may have to be defended years after it was produced.

**Isotonic (pool-adjacent-violators).** Assumes only monotonicity. Follows
genuine curvature that an affine map cannot, and follows noise that an affine map
would smooth away. Its extrapolation beyond the observed score range is constant,
which is honest but means it cannot rank two results that both lie beyond that
range — a real limitation for a system whose strongest results matter most.

**Kernel density ratio.** Models `p(s | H_ss)` and `p(s | H_ds)` separately and
takes their ratio. The construction closest to the definition, and the right
choice where the two distributions differ in shape rather than by a location
shift — which is the common case for the non-acoustic streams, where a score is
often a count-based quantity with a spike at zero.

**A detail that is frequently got wrong.** Pool-adjacent-violators yields
posterior probabilities under the *empirical prior of the trial set*, which is
an artefact of how validation pairs were constructed — typically far more
different-source pairs, because pairs combine quadratically. Converting to a
likelihood ratio requires dividing that prior out:

```
LR = [p / (1 − p)] / [N_ss / N_ds]
```

Omitting the correction produces "likelihood ratios" that shift systematically
with trial-set composition, so the same system evaluated on two differently
balanced sets appears to have different evidential strength.

### 6.4 The database search problem

This is the most operationally consequential item in the framework and the one
most often omitted from systems of this type.

In a **verification** task — comparing a questioned recording against one named
suspect — the prior odds are supplied by the non-acoustic case circumstances.
In a **database search** — comparing against `N` enrolled entries — the prior
odds for any individual entry are approximately `1/(N−1)` under a uniform
assumption.

Worked example, `N = 100,000`, `LR = 1,000`:

```
prior odds     = 1 / 99,999           ≈ 1.0 × 10⁻⁵
posterior odds = 1,000 × 1.0 × 10⁻⁵   ≈ 0.01
posterior prob = 0.01 / 1.01          ≈ 0.99 %
```

**A thousand-to-one acoustic result, returned from a national-scale database
search, is approximately 99% likely to be wrong.** This single calculation is the
strongest available argument that voice-only linkage must not be deployed, and it
should be presented to every stakeholder who asks why the system does not simply
compare the voices.

Now fuse three streams — acoustic `LR = 1,000`, behavioural `LR = 100`,
transactional `LR = 50` — assuming for illustration that dependence correction
leaves the product substantially intact:

```
LR_total       = 5 × 10⁶
posterior odds = 5 × 10⁶ × 1.0 × 10⁻⁵ ≈ 50
posterior prob = 50 / 51              ≈ 98 %
```

The transition from 1% to 98% is the entire justification for the platform. It is
not achieved by better acoustics. It is achieved by combination.

Three cautions accompany this example, and all three are load-bearing. The
uniform prior is itself a modelling choice, and sensitivity to it is reported
with every result. The relevant population may in practice be far smaller
(offenders operating in a given corridor, in a given period) or the enrolled
population may be biased in ways that violate uniformity. And the "assuming
dependence correction leaves the product intact" clause is precisely what §6.6
shows to be false — the corrected figure is materially lower, and the difference
is a substantial number of percentage points on the posterior.

**A note on arithmetic.** These calculations are performed in log-odds
throughout. Bayes' rule is then addition, which is exact for any magnitude of
evidence and any size of database. The linear form loses precision for small
priors and overflows for strong evidence — which is exactly the corner a
national-scale search occupies.

### 6.5 Search-specific inflation

A search over `N` candidates produces `N` comparisons and therefore samples the
extreme tail of the score distribution by construction. Even a well-calibrated
system returns, as its top-ranked result, the most extreme *different-source*
comparison more often than a genuine linkage, whenever the prior is small enough.

This is distinct from the prior-odds problem of §6.4 and compounds with it. The
prior tells you how improbable any given candidate is a priori; the search size
tells you how many chances the system had to produce an extreme value. A
methodologically complete treatment reports both, and the interface presents the
number of comparisons performed alongside the ranking.

### 6.6 Fusion and the dependence problem

Naive combination assumes conditional independence:

```
log LR_total = Σ_i log LR_i
```

This assumption is **false in this application**, and the direction of the error
is dangerous: correlated evidence treated as independent overstates the combined
LR, producing overconfident linkage assertions against real people.

Formally, what is required is the joint likelihood ratio

```
LR = p(ℓ_1, …, ℓ_K | H_ss) / p(ℓ_1, …, ℓ_K | H_ds)
```

over the vector of per-stream log-likelihood ratios `ℓ`. Independence would give
the product of marginals; the correction is the ratio of the two dependence
structures. Three approaches are evaluated.

**(a) Discriminative fusion with dependence absorbed into weights.**
`log LR_fused = w_0 + Σ_i w_i ℓ_i`, with weights minimising `C_llr`. The weights
partially absorb systematic dependence: two streams that largely repeat each
other end up sharing the weight one alone would have had. Cheap, standard, and
provably insufficient where dependence is strong or varies with the strength of
the evidence — which is the regime that produces the results anyone acts on.

A subtlety that is easy to miss: a weight vector fitted where all five streams
were present encodes redundancy *in that configuration*. Applying it to a
comparison with two streams absent leaves the remaining weights discounted for
redundancy with evidence that is not there, systematically understating. Weights
must therefore be conditioned on the pattern of available streams.

**(b) Explicit latent-variable model.** Model `ℓ` as multivariate Gaussian under
each proposition, with a common-factor decomposition `Σ_ss = λλ^T + Ψ`
representing the shared cause — one person running one operation — explicitly.
The loadings `λ` say how strongly each stream responds to that common factor,
and the shared-variance fraction is a directly interpretable answer to "how much
of the apparent independence is illusory". Marginalisation over absent streams is
exact: the marginal of a Gaussian is a Gaussian on the corresponding sub-vector.

Two estimation details determine whether this works. Missing data in *training*
must be handled by expectation-maximisation including the conditional covariance
of the imputed entries; mean imputation drives every estimated correlation toward
zero, which is the error the model exists to correct, reintroduced during its own
fitting. And the covariance requires shrinkage: fifteen parameters per
proposition for five streams, estimated from a few hundred development
comparisons, gives a sample covariance whose inverse — which scoring uses — is
badly conditioned.

**(c) Copula-based dependence modelling.** Sklar's theorem factorises the joint
into marginals and a copula:

```
log LR = Σ_i [log f_ss,i(ℓ_i) − log f_ds,i(ℓ_i)] + log c_ss(u_ss) − log c_ds(u_ds)
```

Attractive because the marginals and the dependence structure are estimated
separately from different amounts of data. Two cautions. The marginal terms must
be computed from the *fitted marginal densities*, not taken to be the input
values — using the inputs assumes every stream is already perfectly calibrated,
and the difference is silently absorbed into the dependence correction, which
then measures stream miscalibration rather than dependence. And a Gaussian
copula has no tail dependence: it asserts that extreme values in two streams
become asymptotically independent. If the truth is that extreme acoustic and
extreme behavioural evidence tend to co-occur — which is what one operator
running one script would produce — it understates dependence exactly where the
evidence is strongest.

**The explicit deliverable.** Independently of which approach wins, the
**magnitude of the overstatement** incurred by naive summation on real data is
reported. This is a safety result about an entire class of deployed multimodal
forensic systems, and it holds whether or not the sophisticated correction is
worth its complexity. Exercised on synthetic data with a known dependence
structure, the reference implementation puts the median inflation at several
orders of magnitude, with the great majority of comparisons landing in a
different verbal strength band — but that figure is a property of the synthetic
construction and is stated here only to indicate the scale of the quantity being
measured, not as a finding.

### 6.7 Within-stream dependence

The same error occurs *inside* a stream, where the fusion layer cannot see it,
and it is worth naming because it is easy to introduce.

Character n-grams extracted with a sliding window share `n−1` characters with
each neighbour. Treating them as independent multinomial draws counts the same
characters repeatedly and inflates the likelihood ratio by roughly a factor of
`n` in the exponent — for a 4-gram over a few hundred characters, orders of
magnitude. The same applies to overlapping analysis frames, to transactions
sharing a counterparty, and to calls within one session.

The general principle: wherever observations are aggregated within a stream, the
effective sample size is not the count of observations. Each stream states how
it addresses this, and the ablation in §8.2 is what would reveal a stream whose
strength is an artefact of over-counting.

### 6.8 Calibration metrics and what they decompose

A system is **calibrated** when its stated likelihood ratios are numerically
warranted — when evidence assigned `LR = 100` is genuinely about a hundred times
more probable under `H_ss`. Discrimination without calibration is not merely
incomplete; it is misleading, because the number reported carries an implied
strength it does not possess.

The primary metric is the log-likelihood-ratio cost:

```
C_llr = ½ [ (1/N_ss) Σ_{i∈ss} log₂(1 + 1/LR_i)
          + (1/N_ds) Σ_{j∈ds} log₂(1 + LR_j) ]
```

Three properties earn it that position.

**It is strictly proper.** It is minimised only by reporting the likelihood
ratios the data actually supports. A system cannot improve its score by
exaggerating in the correct direction — which is precisely the failure mode that
matters when the output goes to a court.

**It has an absolute reference point.** `C_llr = 1` is the cost of reporting
`LR = 1` for everything, i.e. of contributing nothing. Above one the system is
*worse than useless*: an investigator acting on it decides less well than one
ignoring it. No accuracy-based metric has this property; a system can be 95%
accurate and have `C_llr` above one.

**It decomposes.** `C_llr^min`, obtained after optimal monotonic recalibration
via pool-adjacent-violators, isolates discrimination and is invariant to any
monotonic transformation of the scores. The remainder, `C_llr − C_llr^min`, is
calibration loss. The two failures are different — "the system cannot tell these
apart" versus "the system can tell them apart but is misrepresenting how
confidently" — and they have different remedies.

Reported alongside: equal error rate, **for comparability with the speaker
recognition literature only**, and Tippett plots showing cumulative LR
distributions for same- and different-source trials, which reveal the tail
behaviour that summary statistics hide.

**Accuracy, precision, recall and F1 are explicitly rejected as primary
metrics.** They require a decision threshold, and imposing a threshold inside the
system pre-empts a judgement that belongs to the investigator and the court.

### 6.9 Bounding what the data can support

A calibration model will emit a likelihood ratio of `10^12` from a validation set
of five thousand trials. That number asserts the evidence is a trillion times
more probable under one proposition than the other; the validation set contains
no observation remotely capable of supporting such a claim. It is an
extrapolation of the fitted functional form, and its magnitude is a property of
that form rather than of the evidence.

This is not an exotic failure. It is the ordinary behaviour of every parametric
calibration at the tails, and it is where the numbers that reach a court come
from.

The empirical bound (Vergeer et al., 2016) asks what the strongest defensible
likelihood ratio is given the data observed. The construction implemented here is
the "devil's advocate": augment the validation set with one hypothetical
counterexample at each extreme — a different-source trial scoring above
everything observed, and a same-source trial scoring below everything — and
recompute the optimal calibration. The strongest attainable likelihood ratio then
becomes finite and governed by the validation set's size. Intuitively: with `N`
same-source trials and no different-source trial ever scoring that high, the
strongest claim the data supports is of order `N` to one, because one more trial
could have been the counterexample.

Bounding here is clipping, and clipping is honest in a way it usually is not: it
does not discard information, because the information was never present. Both
figures are reported — the bounded value is what the data supports, the unbounded
value is what the model's algebra produced — so the size of the extrapolation is
visible rather than silently removed.

**Every likelihood ratio this system reports is bounded in this way.** It is
listed among the conditions of the central claim (§1.3) for that reason.

### 6.10 Non-acoustic evidence: rarity as a likelihood ratio

The non-acoustic streams admit a treatment the acoustic stream does not, and
using it is a substantive methodological choice rather than an implementation
detail.

The obvious approach to comparing two incidents on, say, their cash-out agents is
a set-overlap index combined with other similarities in a weighted sum, then
calibrated. Two things are wrong with it. It ignores **rarity**: two incidents
sharing an agent handling four transactions a month is strong evidence, two
sharing an agent handling four thousand is nearly none, and an overlap index
gives both the same value — a distinction no downstream calibration can recover
once the statistic has discarded it. And the weights are chosen rather than
estimated.

The approach taken instead asks the question directly. Under `H_ss` the two
incidents' observations are draws from **one** actor's distribution; under `H_ds`
they are draws from **two** independent actors' distributions. Both actors'
parameters are unknown, so integrate them out against a prior fitted to the
background population:

```
LR = p(x_A, x_B | one actor) / [ p(x_A | actor 1) · p(x_B | actor 2) ]
```

With a conjugate prior each term is closed-form. For categorical counts — agents,
cell sites, handset models, function words, script n-grams, discretised hour of
day — the Dirichlet-multinomial gives

```
log LR = logΓ(N_A + α) + logΓ(N_B + α) − logΓ(α) − logΓ(N_A + N_B + α)
       + Σ_c [ logΓ(n_Ac + n_Bc + α_c) + logΓ(α_c)
               − logΓ(n_Ac + α_c) − logΓ(n_Bc + α_c) ]
```

with `α_c` proportional to background frequency. Rarity is then handled by the
model rather than by a chosen weight: sharing a category the background rarely
produces contributes far more than sharing a common one, automatically. For
continuous measurements — log transaction amounts, call durations — the
normal-inverse-gamma model gives the analogous closed form.

Two properties follow that are worth stating. Evidence accumulates
**sub-linearly**: the second transaction through the same agent tells you less
than the first, which is correct and which a linear score does not reproduce.
And disagreement produces a *negative* log-likelihood ratio, correctly supporting
the different-source proposition, rather than merely a low similarity.

One consequence deserves emphasis because it is where the reported strength is
most sensitive: the background population is the denominator of every one of
these likelihood ratios, and its choice is a modelling decision. A background
estimated from one province's agent network says something different about a
shared agent than one estimated nationally. The background in force is recorded
with every result.

### 6.11 Propagating evidence through a graph

The linkage graph is a *consequence* of fusion, not a parallel system. Its edges
are of two kinds and conflating them is an error the standard algorithms make
silently.

Given `A—B` at `LR = 30` and `B—C` at `LR = 30`, the evidence that `A` and `C`
share an actor is emphatically not `900`. Three problems compound. The
propositions are different: each edge compares two incidents, and the chain
asserts something about a third pair that was never compared, through an
intermediate whose own attribution is uncertain. The edges are not independent:
they frequently rest on the same streams, models and channel conditions, so a
systematic bias is squared rather than accumulated. And evidential support is not
transitive — that is a property of equivalence relations, and this is not one.

Three treatments are offered with honestly different guarantees: a conservative
**weakest-link bound** assuming nothing beyond the edges being what they say (the
only one safe to report without qualification); analytic Gaussian propagation
with a stated inter-edge correlation; and Monte Carlo resampling with correlation
induced through shared evidence streams. All return a distribution or a bound,
never a bare number.

Community detection receives the same treatment. Detection is run repeatedly over
resampled edge weights, each draw taking every edge from its own uncertainty
interval, and a node's stability is the fraction of resamplings in which it lands
with the same partners. A community that survives resampling is a finding; one
that dissolves was an artefact of the particular values the evidence happened to
take. Adapting these algorithms to probabilistic edges — or bounding the error of
applying them naively — is a genuine research task, not an implementation detail.

---

## 7. System Design

The architecture is an *instantiation of §6*, and each component is justified by
the framework rather than by convention.

### 7.1 Evidence streams

| Stream | Extracted | Output |
|---|---|---|
| Acoustic | Speaker representation + disguise-robust feature subsets (§4.2) | `LR_acoustic` |
| Validity gate | Synthetic / converted speech detection | Admit / exclude / indeterminate |
| Behavioural | Idiolect (function words, disfluency, code-switching) and script structure, reported separately | `LR_behavioural` |
| Temporal | Call timing distribution, session cadence, duration distribution | `LR_temporal` |
| Transactional | Agent and counterparty rarity, value structure, velocity | `LR_transactional` |
| Device / network | IMEI, handset class, cell-site pattern, weighted by rarity | `LR_device` |

Each stream emits a calibrated likelihood ratio with an uncertainty interval and
the identity of the models that produced it. Streams that cannot be computed for
a given pair emit **nothing** — not a neutral value — and the fusion model
handles missingness by marginalisation rather than imputation. A neutral
substitute would assert that the stream was computed and found the evidence
equally probable under both propositions, which is a fabricated observation.

### 7.2 The validity gate

The synthetic-speech detector is architecturally distinct from the other streams
because it does not contribute evidence; it **conditions the admissibility** of
acoustic evidence. If a recording is judged synthetic, `LR_acoustic` is excluded
from fusion entirely rather than down-weighted, because a synthetic recording
carries no information about a human speaker's anatomy.

Three verdicts, and the third is the one that matters. **Admitted** and
**excluded** are confident calls. **Indeterminate** covers a score in the
uncertain band *and* — separately — a recording the detector has no basis to
judge because it falls outside its training domain. Indeterminate does not admit
acoustic evidence.

The out-of-domain condition is the concrete handling of the generalisation
failure documented across the ASVspoof series. A recording improbable under both
the genuine and the spoofed model is not evidence of either; reporting a
confident score in that regime is extrapolation presented as measurement. When
exercised against a synthesis method absent from training, the reference
implementation returns indeterminate on out-of-domain grounds rather than a
confident verdict — which is the designed behaviour and the honest one.

The operating point is a policy decision, expressed as an explicit and auditable
policy object stored with every verdict, so that a verdict can be re-derived if
policy changes without re-running the detector. It is set conservatively, on a
stated asymmetry: admitting synthetic speech puts fabricated evidence before a
court, whereas excluding genuine speech only weakens a case that other streams
may still support. Those costs are not comparable, so the threshold is not placed
where total error is minimised.

### 7.3 Fusion engine

Implements §6.6. Produces a single calibrated `LR` for an incident pair, together
with a per-stream decomposition, an uncertainty interval, the explicit prior-odds
context of the search that produced it, and the naive-summation comparator from
which the overstatement is computed.

The fused uncertainty interval is obtained by correlated resampling rather than
by summing the per-stream intervals. Summing them would assume the streams'
errors independent — the same assumption the fusion layer exists to correct,
reintroduced one level up, and in the same direction.

### 7.4 Graph layer

Implements §6.11. Nodes represent incidents, wallets, devices, subscriber records
and locations; edges are either deterministic facts of record or probabilistic
linkages carrying a likelihood ratio and an interval.

The central design property is a negative one: the graph object exposes **no**
method returning a path or a component computed over mixed edges. The operations
that would silently treat a likelihood ratio as a fact are absent, not merely
discouraged.

### 7.5 Evaluation layer

Speaker-disjoint splitting, speaker-level resampling, the H1–H7 protocols with
their decision rules, and exhaustive stream ablation. Discussed in §8.

### 7.6 Investigator interface

Interface design is a *safety-critical* component here, not presentation. Its
requirements derive directly from §6.1, §6.4 and §6.5:

- The prior odds of the search are displayed **adjacent to** every result, in the
  same visual block — not in a tooltip and not behind a disclosure control. The
  prior, the evidence and the posterior are laid out as one equation so that a
  future layout change cannot separate the posterior from the prior that produced
  it.
- Posterior probability is shown only with an explicitly supplied or case-derived
  prior, never a default. Where the prior dominates — the ordinary outcome of a
  large search — the interface says so in words.
- Per-stream contribution is always visible and never collapsed by default; a
  linkage driven entirely by one stream is visually distinguished from one
  supported by several, by a border treatment as well as by text.
- Absent streams are listed with their reason. A stream that silently disappears
  is indexed identically to one never attempted, and "the recording was judged
  synthetic" is a finding whereas "no transaction records exist" is a
  data-access problem.
- The number of comparisons the search performed is displayed with the ranking
  (§6.5), as is the number of candidates that could not be compared — which have
  *not* been excluded on the evidence.
- Language is constrained: the system reports evidential support with a magnitude
  and a direction, and is technically incapable of emitting the vocabulary of
  identity (§9.4).
- Every query, result and analyst action is logged immutably (§10.3).

H6 tests whether these choices actually improve decisions. It is entirely
possible that they do not — automation bias is well documented, and an interface
that presents caveats prominently may simply be one whose caveats are skipped
prominently — and that finding would be worth publishing.

---

## 8. Methodology

### 8.1 Data strategy

Data is the principal risk to this project and is addressed in tiers, ordered by
increasing access difficulty, so that the thesis remains viable if later tiers
fail.

**Tier 1 — Public corpora, channel-simulated.** VoxCeleb, LibriSpeech and
ASVspoof, passed through AMR-NB codec processing at multiple bitrates with
additive noise and packet-loss models. Establishes channel-degradation curves
(H1) with full reproducibility. *Requires no institutional access and is
sufficient for H1 and H7 in simulated form.*

Three methodological commitments attach to this tier, each addressing a way the
obvious approach would produce optimistic results.

*The codec is a codec.* Where a reference AMR-NB encoder is available it is used.
Where it is not, the fallback is an analysis-by-synthesis CELP model — LPC
analysis, line-spectral-frequency quantisation at rate-dependent resolution,
adaptive-codebook long-term prediction, and algebraic fixed-codebook excitation
with a rate-dependent pulse count — not a band-pass filter with added noise
(§4.5). Every degraded signal records which produced it, and results from the two
are not pooled.

*Noise is spectrally shaped.* Babble occupies the same band as the target and
follows the long-term average speech spectrum; vehicle noise sits largely below
the telephony passband. Signal-to-noise ratio is measured **within the passband**
so that a condition labelled "10 dB" means the same thing across noise types.

*Packet loss is concealed, not zeroed.* No real decoder hands the analysis a
silent gap; it extrapolates from the last good frame and mutes only after several
consecutive losses. Substituting silence creates sharp broadband transients at
every gap edge — which a countermeasure trained on such data would learn to
detect, meaning it would have learned to detect the simulation.

**Tier 2 — Purpose-collected disguise corpus.** Consented participants,
multilingual and code-switched, recording scripted and spontaneous speech under
controlled disguise conditions (natural, pitch-raised, pitch-lowered, accent
adoption, nasal occlusion), transmitted over real mobile networks rather than
simulated. Addresses H2, H3 and H4, and — subject to ethics approval and
participant consent for release — is proposed as a **standalone research
contribution**, since no comparable resource exists for these languages.

**Tier 3 — Operational data under formal agreement.** Real case recordings and
transaction records under written institutional agreement, ethics approval and a
defined legal basis. Necessary for H5 and H6 at operational fidelity.

**Contingency.** If Tier 3 access is not obtained, H5 and H6 are addressed on
realistically simulated case data with the limitation stated prominently, and the
thesis contribution narrows to the framework plus Tier 1–2 empirical results.
This is a viable doctorate. The proposal does not depend on institutional
cooperation that cannot be guaranteed in advance.

### 8.2 Experimental design

- **H1**: Controlled degradation sweep across bitrate × noise type × SNR ×
  packet-loss × duration. Reported as `C_llr^min` surfaces with speaker-level
  intervals, not point estimates.
- **H2**: Feature-class ablation under each disguise condition, with the
  resistant and fragile classes compared as an interval on their difference.
- **H3**: Within-call temporal segmentation; the contrast is *paired* within
  speaker, so the large between-speaker variance cancels rather than obscuring
  the effect.
- **H4**: Cross-corpus evaluation, English-trained versus in-domain-adapted,
  stratified by language and by code-switch density.
- **H5**: Exhaustive ablation over all `2^K − 1` non-empty stream subsets, with a
  model fitted independently per subset. Fitting once on all streams and zeroing
  the excluded ones would measure something else: the remaining weights would
  still be discounted for redundancy with absent evidence. The critical
  comparison is dependence-corrected fusion against naive summation, with the
  *overstatement magnitude* as a reported quantity.
- **H6**: Controlled study with investigator participants, three arms (current
  workflow / raw scores / calibrated LR), with true linkage known by
  construction.
- **H7**: Cross-attack generalisation — train on a subset of synthesis methods,
  evaluate on held-out unseen methods, reported per attack type rather than
  aggregated, so a method the detector fails on is identifiable individually.

The ablation reports each stream's **marginal contribution given the others**,
averaged over the subsets it could be added to, rather than its solo performance.
A stream with excellent solo performance and near-zero marginal contribution is
repeating what the others already said, and only the marginal figure reveals it.

### 8.3 Statistical power and sample size

A proposal that specifies analyses without specifying the sample sizes they need
is a proposal that will discover it was underpowered after the data is collected.
The following are planning figures, to be revised once Tier 2 pilot variance
estimates exist.

**The unit of analysis is the speaker, not the trial.** Trials sharing a speaker
are not independent, so the effective sample size is the number of speakers.
Every calculation below is in those terms.

**H1 (channel viability).** The quantity is an interval on `C_llr^min` narrow
enough to sit wholly below 0.30 or wholly above 0.50. Pilot work on comparable
narrowband corpora suggests a speaker-level standard error of roughly
`0.35/√S` for `S` speakers in the regime of interest. Requiring a half-width of
0.10 gives `S ≈ 48`; a half-width of 0.07, `S ≈ 100`. **Target: 100 speakers**,
each with at least four recordings so that within-speaker variability is
estimable.

**H2 (feature-class separation).** A difference of two dependent `C_llr^min`
estimates on the same speakers. The correlation between feature classes on the
same recordings is substantial — plausibly 0.6 — which *helps*: the variance of
the paired difference is `2σ²(1−ρ)`. To resolve a separation of 0.05 with a
half-width of 0.04 requires roughly **70 speakers per disguise condition**.

**H3 (duration decay).** Paired within speaker and within call, so the pairing is
tighter still. The effect size is unknown, which is the point of measuring it; a
minimum detectable paired difference of 0.03 at 80% power requires approximately
**60 speakers** under a within-pair correlation of 0.7. Calls must exceed roughly
90 seconds for the thirds to contain enough speech, which constrains recruitment
protocol rather than sample size.

**H5 (fusion superiority).** The unit is the **operation**, not the speaker: a
group rotating callers produces comparisons sharing an operation without sharing
a speaker. Resolving a `C_llr` improvement of 0.02 with a half-width of 0.015
requires on the order of **200 distinct operations** with multiple incidents
each. This is the most demanding requirement in the study and is the principal
reason Tier 3 access matters. Under the Tier 1–2 contingency the hypothesis is
addressed at reduced power on simulated case structure, and the reduction is
reported rather than absorbed.

**H6 (investigative utility).** A three-arm between-participants design detecting
a moderate effect (Cohen's *d* ≈ 0.6) at 80% power with α = 0.05, corrected for
three pairwise comparisons, requires approximately **45 participants per arm**,
i.e. **135 investigators**. This is likely infeasible. Two mitigations are
planned and both are stated as compromises: a within-participants design with
counterbalanced case order, which reduces the requirement to roughly 40
participants at the cost of carry-over effects; and pre-registration of the
primary outcome so that a null result is interpretable rather than ambiguous. If
neither yields adequate power, H6 is reported as an underpowered exploratory
study and labelled as such throughout.

**H7 (cross-attack generalisation).** Power here is driven by the number of
*attack types* held out, not by the number of recordings. With fewer than about
six distinct synthesis methods, a held-out estimate is dominated by which method
happened to be excluded. **Target: at least eight attack types**, evaluated
leave-one-out.

**Multiplicity.** Seven hypotheses tested at the five percent level give roughly
a 30% chance of at least one spurious result. Holm-Bonferroni correction is
applied across the confirmatory family, and the family is fixed in advance
(§8.5). Exploratory analyses are reported separately and labelled exploratory.

### 8.4 Statistical rigour

- **Speaker-disjoint splits throughout.** Any speaker in training may not appear
  in evaluation, for any stream. Recording-level splitting is the standard way
  leakage enters by accident, and the resulting error rates are often several
  times too good — good enough to be believed.
- **Resampling over speakers, never over trials.** Trials within a speaker share
  a vocal tract, a handset and a session. Resampling trials treats correlated
  observations as fresh information and produces intervals that are too narrow,
  sometimes by a factor of three. An interval that excludes the truth most of the
  time is worse than no interval, because a stated interval is believed. This
  extends past the acoustic stream: the same operator produces the behavioural,
  temporal and transactional evidence too.
- **Every performance figure carries its condition.** A `C_llr` reported without
  its bitrate, noise condition, disguise state and language is not a weak claim
  but an uninterpretable one, and the evaluation framework declines to produce
  one.
- **Three outcomes, not two.** Supported, falsified, and inconclusive. §5.
- **Multiple-comparison correction** across the confirmatory family.
- **Negative results reported with equal prominence.** A finding that fusion does
  not help, that no cross-lingual penalty exists, or that calibrated output
  confuses investigators, is a contribution.

### 8.5 Pre-registration

The analysis plans for **H5 and H6** are pre-registered before the corresponding
data is collected. These are the two hypotheses most vulnerable to post-hoc
specification search: H5 because the space of fusion models and stream subsets is
large enough that something will look significant, and H6 because human-factors
outcomes admit many defensible operationalisations.

The registration fixes, in advance: the primary metric; the decision thresholds
in §5; the unit of resampling; the composition of the confirmatory family for
multiplicity correction; the handling of excluded recordings; and the stopping
rule for data collection. Registration is with a public registry (OSF or
equivalent) with a timestamped record.

Deviations from the registered plan are reported as deviations, with their
rationale, rather than silently adopted. A deviation is not misconduct; an
unreported one is.

### 8.6 Baselines

Nothing is compared only against itself:

- Cosine scoring on off-the-shelf pretrained embeddings, uncalibrated — the
  status quo this work argues against.
- The same embeddings, calibrated — isolating how much of the improvement is
  calibration rather than representation.
- A neural embedding extractor (ECAPA-TDNN or a self-supervised front-end) in
  place of the i-vector representation, with the rest of the pipeline held fixed —
  isolating the contribution of the front-end.
- Current investigative workflow (identifier lookup only), for H6.
- Best single stream, for H5.
- Naive independent summation, for H5's dependence claim.

---

## 9. Reference Implementation

Software has been written alongside this proposal. Its purpose is to demonstrate
that the framework of §6 is computable, to make the methodology of §8
reproducible, and to establish that the governance constraints of §10 can be
architectural rather than procedural.

**It is not an empirical result.** No claim in this document rests on it. Where
it has been exercised, it has been exercised on synthetic data with known
structure, and any figure so obtained is a property of the synthetic construction
rather than a finding about telephony fraud.

### 9.1 What is implemented

The full pipeline of §6 and §7: DSP front-end (framing, filterbanks, voice
activity detection, linear prediction with formant estimation, YIN pitch
tracking, glottal source measures, nasal segment detection with antiformant
estimation); channel simulation (reference AMR where available, analysis-by-
synthesis CELP otherwise, spectrally shaped noise, concealed packet loss);
acoustic back-end (GMM-UBM, i-vector total variability, LDA/WCCN, two-covariance
PLDA); the LFCC-GMM countermeasure with its out-of-domain indicator and validity
gate; the conjugate marginal-likelihood models for the non-acoustic streams;
calibration (PAV, linear logistic, kernel density) with empirical bounding; the
four fusion models with overstatement measurement; the probabilistic graph with
its three propagation methods and stability-assessed community detection; the
evaluation framework with speaker-disjoint splitting, speaker-level resampling
and the H1–H7 protocols; and the governance architecture.

### 9.2 Design properties that are load-bearing

Three architectural choices exist to make classes of error *impossible* rather
than merely discouraged, and they are described here because they are part of
the methodological contribution.

**Absence is not neutrality.** A stream that produced nothing is represented by a
distinct type carrying its reason, not by a likelihood ratio of one. Code that
wants a stream's value must first establish which case it holds, so the error of
substituting neutral evidence for absent evidence cannot be made by omission.

**A likelihood ratio cannot exist without its prior.** The type representing a
result cannot be constructed without a posterior, and a posterior cannot be
constructed without an explicit prior carrying its justification and the identity
of whoever supplied it. There is no default prior anywhere in the system, and no
code path produces a result stripped of its search context.

**Uncalibrated numbers cannot be reported.** Every stream's score passes through
a fitted calibrator at one architectural seam, and a comparator constructed
without one produces absence rather than numbers. This is the rule "no
uncalibrated value is ever reported" expressed as structure rather than as
discipline.

### 9.3 Numerical choices with methodological consequences

All evidential quantities are carried as log-likelihood ratios and all Bayesian
updating is addition in log-odds. Fused evidence routinely exceeds the dynamic
range of a double in the linear domain, and a system storing linear ratios
converts strong evidence into infinity — silently, and on exactly the results
that matter.

Conversion to a linear ratio is deliberately fallible: a log-LR beyond the
representable range raises rather than saturating, because saturating would
present a calibration failure as an extremely strong result.

### 9.4 Enforced vocabulary

The system has no vocabulary for asserting identity. Text crossing the system
boundary passes through a policy that rejects the language of identification, and
a build-time check over the source tree fails if that vocabulary appears in any
emittable string or identifier. A field named `match_score`, or an empty state
reading "No matches found", fails the build.

The check is on word boundaries, not substrings, for a reason worth stating: a
policy that rejects "dispatch" and "rematch" is a policy developers route around,
and a control that is routinely disabled is weaker than a control with a known
gap.

### 9.5 Architectural layering, enforced

The system is layered — domain, analysis, application, infrastructure,
interfaces — with dependencies pointing inward only, and the layering is checked
mechanically by parsing the import graph rather than documented in a README. The
domain layer uses only the standard library, so the concepts a court would
recognise are not entangled with an array library's release schedule.

### 9.6 What running it demonstrates

Exercised on synthetic populations with known structure, the implementation
reproduces the worked examples of §6.4 exactly, recovers synthesised formants and
fundamental frequencies to within measurement tolerance, discriminates held-out
synthetic speakers, shows glottal-source reliability collapsing under CELP
coding as §4.5 predicts, returns "indeterminate" rather than a confident verdict
on a synthesis method absent from training, and shows dependence-corrected fusion
outperforming naive summation across a range of induced dependence.

These demonstrate that the framework computes and that the implementation behaves
as specified. **They demonstrate nothing about telephony fraud**, and are
reported here only as evidence that the methodology is executable.

---

## 10. Ethics, Law and Governance

### 10.1 The dual-use problem, stated plainly

A system that links individuals across a national telephony network by their
voice is a mass-surveillance capability. That is true regardless of the
intentions of its designer, the terms of its deployment agreement, or the
sincerity of its stated purpose. The same architecture that links a fraud
operator across forty incidents will link a journalist, an opposition organiser,
a union representative or a witness across forty calls.

A proposal that does not say this openly is not being careful; it is being
evasive, and a serious reviewer will notice. This work therefore treats **its own
misuse as the primary threat** (§11), and accepts constraints that reduce
capability in exchange for reducing misuse potential.

### 10.2 Legal basis

Deployment must be grounded in specific legal authority, not general public
interest. In the Zambian setting the relevant instruments include the Data
Protection Act (No. 3 of 2021) and the Cyber Security and Cyber Crimes Act (No. 2
of 2021), together with ZICTA's regulatory framework for subscriber registration
and the Financial Intelligence Centre's statutory reporting regime.

*These citations require verification against current consolidated text, and any
deployment requires formal legal opinion. Interpretation of statute is outside
the author's competence and will not be attempted in the thesis.*

Design principles derived from data protection law generally: purpose limitation
(evidence collected for one investigation is not reusable for another without
fresh authority), data minimisation, storage limitation with enforced retention
expiry, and a right to meaningful information about automated processing.

### 10.3 Technical constraints on misuse

Governance implemented in policy is governance that survives until the first
official who finds it inconvenient. The following are **architectural**
constraints — capabilities the system lacks rather than uses it is discouraged
from:

- **No live capability.** The system accepts only complete recordings associated
  with a filed complaint reference. There is no ingestion path for live audio and
  no bulk-import path. Nothing in the software accepts a stream, a socket or a
  partial buffer, and the absence is the control.
- **Case-bound queries.** Every operation requires a valid case reference, and
  the type representing one cannot hold an invalid value. Queries without one are
  not rejected by a check that could be removed; they cannot be constructed.
- **Immutable, independently held audit log.** Append-only, hash-chained, with
  read access held by an oversight body separate from the operating agency. The
  chain makes editing *detectable*, which is the strongest property software
  alone can provide; prevention requires custody the operator does not have, and
  that is a deployment arrangement this design supports rather than solves.
- **Queries are audited whether or not they succeeded.** A refused or empty query
  is exactly what an oversight body needs to see, and a log recording only
  successful operations records the least interesting half.
- **The prior is recorded with every query.** Reconstructing later what prior a
  historical result rested on is otherwise guesswork, and the prior is the
  difference between a likelihood ratio and a conclusion.
- **Enforced retention expiry.** Automatic deletion at stated limits, logged
  *before* it occurs — a record of a deletion that did not complete is
  recoverable, a deletion with no record is not. Audit entries outlive the data
  they describe; applying retention to the audit trail would let an operator
  remove the record of an access by waiting.
- **Separation of duties.** Enrolment, query, export, audit and administration
  are distinct authorities, and incompatible combinations cannot be assigned —
  the object representing a principal refuses to exist. Each incompatible pair is
  documented with the specific risk it addresses: a principal who can both enrol
  and query can plant a reference and then produce a linkage to it, and nothing
  in the audit trail would distinguish that from an investigation.
- **No identity assertion in output.** §9.4.

### 10.4 Data protection impact

A full data protection impact assessment is a deployment deliverable requiring
legal input. The following is the technical contribution to it.

**Processing.** Biometric data (voice), financial transaction records, device and
network identifiers, and location-adjacent data (cell site), concerning both
suspects and — unavoidably — victims and uninvolved third parties who appear in
call and transaction records.

**Necessity and proportionality.** The case for necessity rests on the
investigative failure of §1.1 and must be made per deployment, not assumed. The
proportionality argument is materially strengthened by the system's inability to
operate without a filed complaint reference, and by the absence of any bulk path.

**Risks to data subjects, and the mitigation each receives.**

| Risk | Mitigation | Residual |
|---|---|---|
| False investigative suspicion from an overstated LR | Calibration; empirical bounding (§6.9); dependence correction (§6.6); prior always displayed | Non-zero; measured by `C_llr` and reported |
| Disparate error rates across demographic groups | Disaggregated reporting (§10.5); deployment restriction where disparity is unacceptable | Depends on corpus composition |
| Function creep to non-fraud investigation | Case binding; purpose recorded per query; audit custody external | Mitigated technically, not eliminated |
| Retention beyond necessity | Enforced expiry with audited deletion | Depends on stated periods being correct |
| Re-identification from derived representations | i-vectors and profiles are subject to the same retention as source material | Derived data is still personal data |
| Third parties in call records | Data minimisation at ingestion; no enrolment of non-complaint material | Victims' voices are unavoidably processed |

**Rights.** A right to meaningful information about automated processing is
supported by the per-stream decomposition and the recorded model identities: a
result can be explained in terms of what each stream contributed and which
version of which model produced it. Rights of rectification and erasure interact
awkwardly with an append-only audit log, and the resolution — erasing the
evidence while retaining the record that it existed and was accessed — requires
legal confirmation.

### 10.5 Bias and equity

Speaker recognition performance varies across demographic groups, and the
literature documents disparities by sex, age and language background. In this
application, unequal error rates mean unequal exposure to false investigative
suspicion — a harm distributed along demographic lines.

Commitments: performance is disaggregated and reported by language, sex and age
band; disparity is treated as a primary result rather than a supplementary
analysis; and a finding that performance is unacceptably unequal for some group
is grounds for restricting deployment for that group, and will be stated as such.

A specific hazard in this setting deserves naming. The training corpus will
over-represent whichever languages and demographic groups are best represented in
the material that can be collected, and the population the system is deployed
against is not that population. Reporting overall performance while the corpus is
skewed reports the performance of the system on the people who were easiest to
record.

### 10.6 Human oversight

Human review is a structural requirement. However, "human in the loop" is a weak
guarantee, since automation bias is well documented — reviewers tend to ratify
algorithmic output rather than scrutinise it. H6 is designed partly to measure
whether the interface design in §7.6 actually resists automation bias or merely
claims to, and a finding that it does not would be reported.

---

## 11. Threat Model

Threats are enumerated in order of assessed severity, with the system itself at
the top.

| # | Threat | Actor | Mitigation |
|---|---|---|---|
| T1 | **Function creep to political surveillance** | Operating institution | Architectural constraints §10.3; external audit custody; no live path |
| T2 | **Prosecutor's fallacy in court** | Well-meaning prosecutor | LR-only output; prior inseparable from result at the type level; interface constraints §7.6; expert testimony protocol |
| T3 | Overconfidence from dependence error | System design flaw | §6.6 dependence modelling; overstatement quantified and published |
| T4 | Overconfidence from calibration extrapolation | System design flaw | §6.9 empirical bounding; extrapolation reported when it occurs |
| T5 | Adversarial disguise | Offender | Disguise-aware features (H2); multi-stream fusion |
| T6 | Synthetic voice injection | Offender | Validity gate §7.2; conservative operating point; out-of-domain detection |
| T7 | Speaker delegation within group | Organised group | Behavioural *script* component and transactional streams; idiolect/script decomposition surfaces the pattern |
| T8 | Poisoning of enrolled reference data | Insider | Separation of duties; enrolment audit; re-enrolment refused |
| T9 | Evidence exfiltration | Insider / external | Encryption; access control; export as a separate authority |
| T10 | Bias-driven disparate impact | Systemic | §10.5 disaggregated reporting |
| T11 | Automation bias in review | Investigator | Interface design §7.6; measured in H6 |
| T12 | Background population misspecification | System design flaw | Background recorded with every result (§6.10); sensitivity reported |

T1 and T2 are ranked above every technical attack deliberately. The most likely
serious harm from this system is not that a criminal defeats it; it is that it
works, and is then used for something else, or is believed more than it deserves.

---

## 12. Scope and Feasibility

The single greatest weakness of the antecedent draft was undifferentiated scope.
This section draws the boundary explicitly.

**Delivered and empirically evaluated (the thesis proper):**
§4 characterisation, §6 framework, acoustic and behavioural streams, fusion
engine, calibration and evaluation methodology, H1–H5 and H7.

**Delivered as design and reference implementation, evaluated by construction,
by synthetic exercise and by expert review, not by deployment:**
Graph layer, investigator interface, governance architecture. H6 evaluated at
prototype fidelity with investigator participants, subject to the power
constraints in §8.3.

**Explicitly out of scope, stated as future work:**
Production deployment, cross-border federation, real-time operation, multi-agency
integration, high-availability infrastructure, disaster recovery.

Stating that infrastructure engineering is out of scope is not a weakness. A
doctoral thesis is judged on its contribution to knowledge, and a Kubernetes
topology is not one. Reviewers penalise proposals that promise deployment they
cannot deliver far more than proposals that scope honestly.

### 12.1 Indicative sequencing

| Phase | Focus | Gate |
|---|---|---|
| 1 | Literature synthesis; framework formalisation; ethics approval; pre-registration of H5/H6 | Framework defensible; approval obtained; registration timestamped |
| 2 | Tier 1 data; channel characterisation | **H1 decided** — if falsified, thesis re-scopes to non-acoustic streams |
| 3 | Tier 2 corpus collection | Corpus viable and consented; pilot variance estimates revise §8.3 |
| 4 | Disguise and cross-lingual studies | H2, H3, H4 decided |
| 5 | Fusion and dependence modelling | **H5 decided** — the central hypothesis |
| 6 | Prototype and investigator study | H6 decided, or reported as underpowered |
| 7 | Synthesis and writing | — |

Phases 2 and 5 are decision gates with defined re-scoping consequences. A
proposal with gates is a proposal that has thought about failure.

---

## 13. Risks

| Risk | Likelihood | Impact | Response |
|---|---|---|---|
| Tier 3 data access refused | High | High | Tier 1–2 contingency (§8.1); thesis remains viable at reduced power, stated as such |
| H1 falsified — channel too destructive | Medium | High | Re-scope to behavioural/transactional fusion; the framework survives |
| H5 falsified — dependence eliminates fusion gain | Low–Medium | Critical | Publish as a significant negative result; the field needs to know |
| H6 underpowered | High | Medium | Within-participants design; pre-registration; report as exploratory rather than as null |
| Ethics approval delayed | Medium | Medium | Sequence Tier 1 work first; approval on critical path from Phase 1 |
| Disguise corpus recruitment shortfall | Medium | Medium | Reduce condition count; prioritise H2 over H3 |
| Institutional partner withdraws | Medium | Medium | No single-partner dependency in design |
| Scope expansion pressure from stakeholders | High | Medium | §12 boundary agreed in writing at outset |
| Corpus demographically skewed | High | High | Disaggregated reporting; deployment restriction (§10.5); stated as a limitation on generalisation |

The `H5 falsified` row deserves emphasis: if calibrated fusion does not
outperform the best single stream once dependence is properly modelled, that is a
*publishable and important* finding, because it would mean the entire class of
multimodal forensic fusion systems is resting on an unexamined assumption. The
thesis is designed so that its central hypothesis failing still produces a
contribution.

---

## 14. Expected Contributions

**Theoretical.** A formulation of investigative case linkage as calibrated
evidence combination under explicit dependence, explicit database-search priors,
and explicit bounds on what validation data can support — bridging the forensic
likelihood-ratio literature and the machine learning speaker recognition
literature, which currently do not communicate.

**Empirical.**
- Characterisation of speaker-discriminative information surviving African mobile
  telephony channels, under a channel model that reproduces what a CELP coder
  actually does to the glottal source.
- Systematic evaluation of feature-class degradation under deliberate disguise,
  including the duration-decay effect (H3), which the author believes is
  under-examined.
- First performance characterisation of speaker recognition on code-switched
  Zambian-language telephony.
- Quantification of the confidence overstatement produced by
  conditional-independence assumptions in forensic fusion — a safety result about
  an entire class of deployed systems.

**Artefact.** A multilingual, code-switched, disguise-condition telephony corpus,
released subject to consent and ethics approval, for a language group with no
existing resource.

**Methodological.** An evaluation protocol for investigative linkage systems
based on forensic calibration metrics rather than classification accuracy;
speaker-level rather than trial-level uncertainty quantification applied
consistently across non-acoustic streams; a treatment of rarity in non-acoustic
evidence as a marginal likelihood ratio under conjugate models rather than as a
similarity index; and an interface design pattern for communicating likelihood
ratios without inducing the prosecutor's fallacy.

**Governance.** An architecture in which constraints against misuse are
structural properties of the software — capabilities it lacks — rather than
policy commitments, together with a demonstration that such constraints can be
enforced mechanically and checked at build time.

---

## 15. Limitations

Stated in the proposal rather than discovered by examiners.

1. **Voice is not individuating, and no result in this work will make it so.**
   Every conclusion is conditional on population, channel and disguise state.
2. **Conditional dependence may be irreducible.** If the streams share too much
   common cause, fusion gains may be small regardless of modelling
   sophistication.
3. **Prior specification is a modelling choice**, and posterior conclusions are
   sensitive to it. Sensitivity will be reported, but the choice cannot be
   eliminated.
4. **Background population specification is likewise a modelling choice**, and
   it is the denominator of every non-acoustic likelihood ratio (§6.10).
5. **Simulated channel degradation is not real degradation.** Tier 1 results
   establish trends, not operational performance, and results obtained under the
   parametric codec model are not pooled with those from a reference codec.
6. **Antiformant estimation from an all-pole model is indirect**, and the nasal
   feature class — the most disguise-resistant available — is defeated entirely
   by nasal occlusion.
7. **Countermeasure generalisation to unseen synthesis is unsolved.** The system
   detects when it is out of its depth and returns indeterminate; it does not
   thereby detect the attack.
8. **The reference implementation is not an empirical result.** Its behaviour on
   synthetic data demonstrates that the framework computes, nothing more.
9. **Investigator study fidelity.** A controlled study with a modest participant
   pool cannot fully represent real investigative conditions, caseload pressure
   or institutional incentive, and §8.3 indicates it is likely to be
   underpowered.
10. **Generalisation beyond the study setting is unestablished.** Results on
    Zambian telephony do not transfer to other networks or language populations
    without re-characterisation.
11. **Legal interpretation is outside the author's competence** and is not
    attempted.
12. **The adversary adapts.** Any published characterisation of what survives
    disguise informs future disguise. This is an unavoidable tension in open
    security research and is acknowledged rather than resolved.

---

## 16. References

*Provisional — every entry requires verification against the primary source
before submission.*

**Speaker recognition**
- Reynolds, D. A., Quatieri, T. F., & Dunn, R. B. (2000). Speaker verification using adapted Gaussian mixture models. *Digital Signal Processing*.
- Dehak, N., Kenny, P., Dehak, R., Dumouchel, P., & Ouellet, P. (2011). Front-end factor analysis for speaker verification. *IEEE TASLP*.
- Kenny, P. (2005). Joint factor analysis of speaker and session variability: theory and algorithms. *CRIM technical report*.
- Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). X-vectors: robust DNN embeddings for speaker recognition. *ICASSP*.
- Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN. *Interspeech*.
- Chen, S., et al. (2022). WavLM: large-scale self-supervised pre-training for full stack speech processing. *IEEE JSTSP*.
- Nagrani, A., Chung, J. S., & Zisserman, A. (2017/2018). VoxCeleb. *Interspeech*.
- Hansen, J. H. L., & Hasan, T. (2015). Speaker recognition by machines and humans. *IEEE Signal Processing Magazine*.
- Garcia-Romero, D., & Espy-Wilson, C. Y. (2011). Analysis of i-vector length normalization in speaker recognition systems. *Interspeech*.
- Prince, S. J. D., & Elder, J. H. (2007). Probabilistic linear discriminant analysis for inferences about identity. *ICCV*.
- Sizov, A., Lee, K. A., & Kinnunen, T. (2014). Unifying probabilistic linear discriminant analysis variants in biometric authentication. *S+SSPR*.

**Forensic voice comparison and calibration**
- National Research Council (1979). *On the Theory and Practice of Voice Identification*.
- National Research Council (2009). *Strengthening Forensic Science in the United States: A Path Forward*.
- Rose, P. (2002). *Forensic Speaker Identification*. Taylor & Francis.
- Morrison, G. S. (2011). Measuring the validity and reliability of forensic likelihood-ratio systems. *Science & Justice*.
- Brümmer, N., & du Preez, J. (2006). Application-independent evaluation of speaker detection. *Computer Speech & Language*.
- Brümmer, N., & de Villiers, E. The BOSARIS Toolkit.
- Drygajlo, A., et al. ENFSI methodological guidelines for forensic speaker recognition.
- Amino, K., & Arai, T. Speaker-specific characteristics of nasal sounds. *Acoustical Science and Technology*.
- Vergeer, P., van Es, A., de Jongh, A., Alberink, I., & Stoel, R. (2016). Numerical likelihood ratios outputted by LR systems are often based on extrapolation. *Science & Justice*.
- Ramos, D., & Gonzalez-Rodriguez, J. (2013). Reliable support: measuring calibration of likelihood ratios. *Forensic Science International*.
- Meuwly, D., Ramos, D., & Haraksim, R. (2017). A guideline for the validation of likelihood ratio methods. *Forensic Science International*.

**Anti-spoofing**
- Wu, Z., et al. (2015). ASVspoof 2015. *Interspeech*.
- Todisco, M., et al. (2019). ASVspoof 2019. *Interspeech*.
- Yamagishi, J., et al. (2021). ASVspoof 2021.
- Kinnunen, T., et al. Tandem detection cost function.
- Sahidullah, M., Kinnunen, T., & Hanilçi, C. (2015). A comparison of features for synthetic speech detection. *Interspeech*.

**Graph analytics**
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.*
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden. *Scientific Reports*.
- Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm to detect community structures in large-scale networks. *Physical Review E*.

**Speech analysis and coding**
- 3GPP TS 26.090 — Adaptive Multi-Rate (AMR) speech codec: transcoding functions.
- 3GPP TS 26.071 — AMR speech codec: general description.
- de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *JASA*.
- Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. *IFA Proceedings*.
- Markel, J. D., & Gray, A. H. (1976). *Linear Prediction of Speech*. Springer.

**Statistics and authorship**
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*.
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publ. Inst. Statist. Univ. Paris*.
- Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*.

**Legal (Zambia — verify current consolidated text)**
- Data Protection Act No. 3 of 2021.
- Cyber Security and Cyber Crimes Act No. 2 of 2021.

---

## Appendix A — Notation

| Symbol | Meaning |
|---|---|
| `H_ss`, `H_ds` | Same-source / different-source propositions |
| `LR` | Likelihood ratio, `p(E\|H_ss) / p(E\|H_ds)` |
| `ℓ` | Log-likelihood ratio |
| `C_llr` | Log-likelihood-ratio cost |
| `C_llr^min` | `C_llr` after optimal monotonic recalibration (PAV) |
| `EER` | Equal error rate (reported for literature comparability only) |
| `N` | Enrolled population size in a database search |
| `S` | Number of speakers (the unit of resampling) |
| `K` | Number of evidence streams |
| `F_n` | *n*-th formant frequency |
| `L` | Vocal tract length |
| `C`, `D`, `R` | UBM components, feature dimension, i-vector rank |
| `T` | Total variability matrix |
| `B`, `W` | Between- and within-speaker covariance (PLDA) |
| `ψ` | Between-speaker variance in the diagonalised PLDA space |
| `α` | Dirichlet concentration parameters |
| `ELUB` | Empirical lower and upper bound on reportable LR |

---

## Appendix B — Writing Standard

Every substantive claim in this document should satisfy:

1. **Is it falsifiable?** If it cannot fail, it is not a finding.
2. **Is the uncertainty stated?** A number without an interval is an assertion.
3. **Is the resampling unit stated?** An interval from trial-level resampling and
   one from speaker-level resampling are not the same quantity.
4. **Is the condition stated?** Performance claims without channel, language and
   disguise conditions are meaningless.
5. **Is the direction stated with the magnitude?** A strength band without its
   direction inverts the finding while sounding confident.
6. **Would a hostile reviewer accept it?** Write for the examiner who wants to
   fail the thesis.
7. **Does it distinguish contribution from aspiration?** Future work is
   legitimate; future work presented as contribution is not.
8. **Does it distinguish demonstration from result?** Software running correctly
   on synthetic data is not an empirical finding.

Prohibited throughout: "voiceprint"; the vocabulary of identification as a system
output; "accuracy" as a primary metric; "AI-powered"; any claim of
identification.
