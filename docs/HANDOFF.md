# VIFLAP — handoff

Written to be read at the start of a fresh context window. Kept current: if you
finish something here, update this file in the same commit, because the next
session's first act is to trust it.

Structured after Anthropic's guidance for long-running agents — a progress file,
a status list, descriptive git history, and a restart script — and against the
failure modes it names: **premature completion**, **undocumented progress**,
**incomplete testing**, **context loss**. The last section is the standing
instruction for avoiding them.

---

## 1. Get your bearings (do this first, in order)

```bash
git -C . log --oneline -20
```

```bash
python -m pytest -q -p no:randomly
```

Then read `docs/H1-acoustic-results.md` (~4,050 lines, §§1–26). It is the
system of record. Every claim in it is either measured or marked withdrawn in
place; nothing is deleted.

Check what is running before starting anything heavy:

```bash
powershell -Command "Get-Process python | Select-Object Id,CPU,StartTime"
```

A long-running job may have survived a session restart even when the harness
lost its task record. Confirm with the command line before assuming it died:

```bash
powershell -Command "Get-CimInstance Win32_Process -Filter 'Name=\"python.exe\"' | Select-Object ProcessId,CommandLine"
```

---

## 2. Verified state

| | |
|---|---|
| Tests | **803, all passing** (~7 min) |
| ruff | clean (`viflap scripts tests`) |
| mypy | **63 errors on `viflap`** — pre-existing baseline, not a regression |
| Git | clean, all pushed, `main` |

Always scope mypy to `viflap`. `mypy .` pulls in tests/ and scripts/ and reports
a different number that is not the baseline.

**Repo is now PUBLIC.** Consequences: Actions minutes are unmetered, and Colab
or Kaggle can clone without a token.

```
origin = git@github-Personal:Samuelkaoma/VIFLAP.git
```

The `github-Personal` alias is mandatory. Plain `git@github.com` uses the wrong
identity. Pushes fail 1–4 times before succeeding; retry in a loop and check
`$LASTEXITCODE`, not `$?` — redirecting a native command's stderr in PowerShell
5.1 makes `$?` lie, which has already caused a push to be reported as failed
when it succeeded.

### Models on disk

```
acoustic.npz                    128/100, 125 spk, CMVN 300      §4/§5 baseline
acoustic_large.npz              256/200, 125 spk                §7
acoustic_pooled.npz             128/100, 306 spk                §9, §22 baseline
acoustic_expanded.npz           128/100, 562 spk                §25 — 4/6 supported
acoustic_pooled_cmvn_utt.npz    CMVN 0                          §17 control
acoustic_pooled_cmvn100.npz     CMVN 100                        §17 control
acoustic_pooled_global.npz      306 spk, global allocation      §21 control
acoustic_pooled_stratified.npz  306 spk, stratified allocation  §21
countermeasure_english.npz      300 English spk, CLEAN-trained  §10, §24
countermeasure_multicondition.npz  8 conditions, pooled floor  §24 — step 1
countermeasure_union.npz        8 conditions, union floor     §24 — THE ONE TO USE
pretrained/spkrec-ecapa-voxceleb/   ECAPA-TDNN, 89 MB, 192-dim  §22
```

Embeddings and reports, all under the gitignored `data/reports/`:

```
neural_embeddings_vad.npz  ECAPA, corrected gate — WHAT §22 QUOTES  §22
neural_extraction_vad.json 12 and 73 refusals at 5 s, 0 elsewhere    §22
h1_neural_vad.json         the §22 result table
h1_neural_vad_scores.npz   ECAPA per-trial scores (~16 min)
h1_extractor_paired_vad.json  §22's pairing, 6/6 identical, 6/6 Holm
h1_neural_562.npz / h1_neural_562.json   ECAPA on the 562 corpus  §26
h1_neural_562_scores.npz      ECAPA per-trial scores            §26
h1_expanded.json              i-vector on the 562 corpus        §25
h1_expanded_scores.npz        i-vector per-trial scores (~85 min)
h1_extractor_paired_562.json  §26's pairing, 6/6 identical, 6/6 Holm
h1_neural_lda100.json      the §22 dimension control
h1_pooled_ownership.json   the i-vector baseline §22 compares to
h1_pooled_scores.npz       i-vector per-trial scores (~43 min)
neural_embeddings.npz      SUPERSEDED — the old wall-clock gate
h1_neural.json             SUPERSEDED — quoted only by §22's correction
h1_extractor_paired.json   SUPERSEDED — the four-cell pairing
overstatement.json         §11, 40 replicates, Gaussian, 42-spk marginal
overstatement_tcopula.json §11, same, Student-t copula
overstatement_ecapa.json   §11 on §22's marginal — the strong-marginal arm
overstatement_ecapa_tcopula.json  §11, same, Student-t
overstatement_weak_behavioural.json  §11, behavioural at §13's operating point
overstatement_weak_behavioural_tcopula.json  §11, same, Student-t
psi_spectrum.json          §23, the ψ₁ candidates and the speaker sweep
synthetic_acoustic.json    §24, clean-trained: 0 of 80 admitted
synthetic_acoustic_multicondition.json  §24, 80/80 reach the bar, 3 admitted
synthetic_acoustic_union.json  §24, 71 of 80 admitted — the gate working
countermeasure_union.json  §24, the final detector's EERs
countermeasure_multicondition.json  §24, the retrained detector's EERs
afrispeech_survey.json     §8, zero Zambian speakers
```

The three archives marked SUPERSEDED are kept because §22's correction blocks
quote the before-and-after; **do not delete them and do not quote them as
current**. `neural_extraction.json` records zero refusals everywhere, which is a
property of the old wall-clock gate rather than of the corpus.
`h1_neural_rescored.json` and `h1_pooled_rescored.json` are byproducts of
generating score archives and are **not** what any section quotes.

**Current best: ECAPA embeddings with a 562-speaker back-end (§26).**
`C_llr_min` **0.033 [0.023, 0.049]**, EER **1.00%** at 12.2 kbit/s clean 30 s,
on 187 held-out speakers — **six of six cells supported**. Across the project
the best cell runs 0.343 → 0.276 → 0.099 → 0.033.

The four configurations, all on their own held-out speakers:

```
i-vector,  306 spk   0.276   §9 / §22 baseline
ECAPA,     306 spk   0.099   §22 — paired, 6/6 surviving Holm
i-vector,  562 spk   0.157   §25
ECAPA,     562 spk   0.033   §26 — current best
```

**§22 is the only paired result.** §25 and §26 are standalone evaluations under
§3's rule: their corpora differ from §22's, so the trial sets differ and no
paired instrument applies. Do not present the 0.343 → 0.033 progression as four
comparable measurements — they are four models on three different evaluation
sets.

Do not compare a new result against §9's table without checking which bootstrap
it used. §9 quotes percentile intervals, §14 recomputed them as BCa, and §18
recomputed those again under the symmetric ownership rule. `score_neural.py`
inherits the §14 and §18 corrections, so `h1_pooled_ownership.json` — not §9 — is
the like-for-like i-vector column.

---

## 3. Running now

**Nothing.** §26 is now **paired**: the extractor's contribution at 562 speakers
is −0.124 to −0.287 across six cells, all excluding zero, all surviving Holm, all
on identical trial sets.

That was possible without re-extraction because §25 and §26 used the same corpus,
seed and split, so they already shared a trial set — 437,113 trials at 30 s and
15 s, and identical counts at 5 s too, meaning both front-ends refused the same
recordings. Only the i-vector side lacked persisted scores; `evaluate_h1
--scores` supplied them in 85 minutes rather than the 10 hours a re-extraction
would have cost, and its point estimates reproduce §25 exactly.

The corrected-gate loop, for reference: re-extraction (365 min), scoring
(16 min) and pairing (25 min) all done, and §22 now reports the VAD-gated run
throughout with every table verified against its artefact.

```
neural_embeddings_vad.npz     the corrected-gate corpus  -- what §22 quotes
neural_extraction_vad.json    12 and 73 refusals at 5 s, 0 elsewhere
h1_neural_vad.json            the §22 result table
h1_neural_vad_scores.npz      per-trial scores for pairing
h1_extractor_paired_vad.json  6/6 on identical trial sets, 6/6 surviving Holm
```

The pre-registered control held exactly. Training vectors and all four 30 s and
15 s evaluation cells are **byte-identical** between the two extractions, the
back-end refits to the same model (ψ₁ 66.98, 35 iterations), and those four
cells reproduce to machine precision. Only the two 5 s cells moved.

The superseded archives — `neural_embeddings.npz`, `h1_neural.json`,
`h1_extractor_paired.json` — are kept because §22's correction blocks quote the
before-and-after. **Do not delete them and do not quote them as current.**

Per-trial score archives, the expensive inputs — **do not regenerate** unless
the trials themselves change:

```
h1_pooled_scores.npz      i-vector per-trial scores, evaluate_h1 --scores  (~43 min)
h1_neural_vad_scores.npz  ECAPA per-trial scores, score_neural --scores    (~16 min)
```

The `evaluate_h1` rerun that produced the first doubles as a control: all six
point estimates reproduce `h1_pooled_ownership.json` to five decimal places, so
adding score persistence changed nothing the script computes. Its intervals
differ in the third decimal (bootstrap noise between runs), so **§22 keeps
quoting `h1_pooled_ownership.json`**, not `h1_pooled_rescored.json`.

Timings for planning: `extract_neural` 330-365 min at ~6-7 s per recording per
three durations; `score_neural --resamples 2000` 16 min; `evaluate_h1` over
6 cells ~43 min; `compare_extractors` at B = 2000 about 25 min and
single-threaded — each resample runs a PAV over 133,645 trials for both systems,
plus a 102-fold jackknife.

---

## 4. Settled — do not re-litigate

Sections that reached a conclusion this project should not spend time
re-deriving. Each is measured, and several are negative results.

- **§7** More parameters on the same corpus: all 6 cells significantly worse,
  all survive Holm. Capacity was never the constraint.
- **§9** More speakers at the same parameters: 5/6 better, 4 survive Holm. The
  corpus was the constraint.
- **§10** The countermeasure is *structurally* blind to phase-only attacks. 25×
  the training speakers moved `phase_randomised` 50.00% → 52.60%. LFCCs are
  magnitude-only; more data will not help. Fix is a phase-sensitive feature.
- **§12** E3FS3's extractor saw ~6,000 VoxCeleb2 speakers; only LDA/PLDA saw 91.
  Like-for-like on forensic_eval_01: 0.208 vs our 0.336, a factor of 1.6.
- **§14** `C_llr_min` is a resubstitution PAV minimum and downward-biased. Both
  bootstraps are BCa with jackknife acceleration and a BCa-consistent p-value.
- **§15** ELUB clips 60.6% of trials at the best cell and removes 0.187 of 0.260
  bits of calibration loss. §5's "calibration is cheap" is **withdrawn**.
- **§17** The CMVN duration confound is real but small; 94–97% of the 30 s→5 s
  gap survives. §5's duration headline stands.
- **§18** Trial-ownership skew fixed by hashing the sorted recording-id pair.
  Kish ESS 72.2 → 98.2 of 102. No verdict changed. A real defect that did not
  matter.
- **§19** PLDA convergence monitor now tracks the exact observed-data
  log-likelihood. Monotonicity is asserted and raises `ConvergenceError`.
- **§20** **The channel is validated.** The parametric model is ~2× a real
  AMR-NB coder in log-spectral distance (6.79 vs 3.22 dB at 12.2; 8.57 vs 5.71
  at 4.75), not the ~6× §16 implied from a mismatched benchmark. Coder against
  coder is **6.93 dB** — larger than the model's own distance from the source —
  so the two distortions are **near-independent, not one a harsher version of
  the other**. §16's "bitrate knob moves 0.33 dB" is **withdrawn** as
  unreproducible; the real coder moves 2.48 dB and the model 1.79.
- **§22** **A borrowed ECAPA-TDNN extractor with the same 306-speaker back-end
  reaches `C_llr_min` 0.099 and EER 2.47% at 12.2 clean 30 s, against the
  i-vector system's 0.276 and 7.89% on the identical trial set. Four of six
  cells reach `supported` — the first anywhere in the document.** This is §12's
  prediction confirmed one stage earlier: only the extractor's speaker count
  changed, 306 → ~6,000. **Now paired**: differences −0.176 to −0.310, six of
  six excluding zero and surviving Holm, four of them on trial sets that are
  identical between the two systems. Against §9, where 181 extra training
  speakers bought −0.104 and four of six survived Holm, borrowing the extractor
  buys −0.176 at the same cell and six of six survive. Control at
  `--lda-dimension 100` says none of it is an artefact of the 192-dimensional
  transform. Read the two 5 s rows separately — they are paired on the
  intersection, which is the i-vector front-end's survivor subset and is
  measurably easier, so those two differences are understated.
- **§8 (AfriSpeech-200)** Surveyed and closed. 67,365 utterances from 2,463
  speakers, and **zero from Zambia** — not few, none. Bemba, Nyanja, Tonga and
  Lozi absent entirely; Chichewa has one speaker, nine minutes, and he is
  Malawian. It was never gated, which is the other half of the finding: the
  question sat open because nobody asked it, not because anything blocked it.
- **§13 (behavioural length)** `MIN_WORDS_IDIOLECT = 500` from Ishihara (2017)
  on predatory chatlog messages; `MIN_WORDS_SCRIPT` stays at 40 and is marked as
  having no citation, because it has none. Two floors, not one — the published
  requirements were measured for authorship attribution, which is only the
  idiolect half. Below the floor the idiolect term is **withheld**, and that
  required a guard: with idiolect pinned at zero the delegation flag fires on
  anything with script evidence, including a transcript compared with itself.
- **§24** **The validity gate: two defects, both fixed, and it now works.**
  Reached end to end for the first time by binding synthetic operators to real
  held-out LibriSpeech speakers. It admitted **0 of 80** genuine recordings.
  Cause one: the countermeasure was trained on clean audio and deployed on coded
  audio — §1's multi-condition rule, never applied here. The same recordings
  score a median log-LR of +2.76 clean and −0.23 through the coder. Retrained
  with `--degrade`, all 80 clear the +2.3 bar. Cause two, previously masked: the
  out-of-domain floor was the 1st percentile over the *pooled* mixture, so it
  fired hardest on the cleanest conditions (0.352) and barely on the noisiest
  (0.042) — clean speech is atypical of a blend dominated by noise. "Unlike
  anything in training" is a **union**; the floor is now the minimum over
  per-condition percentiles. Result: **71 of 80 admitted, 9 indeterminate, 0
  wrongly excluded.** `GatePolicy`'s ±2.3 band was never touched — widening it
  at either stage would have hidden a real defect.
- **§26** **Both axes at once: six of six cells `supported`, and paired.** ECAPA
  over the expanded corpus with a 562-speaker back-end reaches `C_llr_min`
  **0.033 [0.023, 0.049]**, EER **1.00%**, and takes both five-second cells past
  `inconclusive` for the first time — §4 had *falsified* three of them. The
  extractor's contribution is **paired against §25** on identical trials:
  −0.124 to −0.287, six of six excluding zero and surviving Holm.
  **The extractor is worth less once the back-end has more speakers**, and the
  shrinkage is concentrated at long duration — the advantage falls ~0.05 at 30 s
  and does not move at all at 5 s babble (−0.262 → −0.261), which is §9's
  "the corpus helps where the channel is mild" seen from the other side. ψ₁ fell
  66.98 → 45.00, confirming §23's small-sample-bias prediction on data it never
  saw. **Still not paired against §22** — different corpora, different splits.
- **§25** **The corpus route had not run out.** The complete `train-clean-360`
  gives 936 usable speakers against 510, split 562/187/187. The **i-vector
  system** — no borrowed checkpoint, same 128/100 configuration as §9 — reaches
  `C_llr_min` **0.157 [0.136, 0.189]**, EER **4.87%**, and **four of six cells
  supported**. The trend at the best cell runs 0.343 (125 spk) → 0.276 (306) →
  0.157 (562), and the 306→562 gain is *larger* than 125→306. **Not paired
  against §9** — only 19 speakers are held out by both — so it is a standalone
  evaluation, and part of reaching `supported` is the interval narrowing from
  187 evaluation speakers rather than the system improving.
- **§23** **The ψ₁ spike is corpus structure, by elimination.** It survives a
  complete change of extractor (ECAPA ratio 5.244, inside the i-vector range of
  5.13–7.04), so it is not the i-vector representation. Switching length
  normalisation off moves it 0.064 against a 1.9 spread across models, so that
  candidate is **refuted**. And the ratio *rises* with training speakers —
  3.160 at 75, 4.956 at 306, non-overlapping between 75 and 150 — which is the
  opposite of the estimation-bias prediction, so that one is **refuted** too.
  Note ψ₁ *itself* is strongly upward-biased at small samples (181 at 75 against
  61 at 306); it is ψ₂ that falls faster, which is why the ratio moves the other
  way and why tracking ψ₁ alone would have given the opposite conclusion.
  LibriVox per-reader recording environment is the only survivor, and it implies
  §9's and §22's absolute figures are optimistic against real telephony.
- **§22 (speech gate)** `NeuralEmbeddingConfig.min_speech_seconds` compared
  wall-clock length against a threshold named for speech, where the i-vector
  front-end runs VAD. Same name, same default, different quantity — and it is
  why §22's 5 s cells could not be paired. Fixed, with tests that fail on the
  old behaviour. §22's reported refusal counts predate the fix and say so; a
  probe puts the correction at 0% for 30 s and 15 s, so **the four supported
  cells and the four paired differences are untouched**.
- **§11** Now 40 replicates per level with the acoustic marginal resampled over
  speakers, plus a Student-t copula arm for misspecification. Two findings. The
  marginal intervals are **enormous** — the naive sum spans [0.032, 0.316] at
  ρ = 0 — and nearly all that width is the 42 speakers, not the 4,000 incidents,
  so the old three-decimal point estimates were never supportable. Differenced
  **within replicate**, §7-style, the Gaussian latent model is worse than the
  calibrated independence model at every level (+0.019 to +0.110, all five
  excluding zero). Misspecification **widened** that penalty rather than
  narrowing it. Also fixed: the section described its marginal as "+1.98 /
  −12.44", the *unbounded* means it had already announced it stopped using; the
  bounded ones are +2.24 / −3.92 nats and no reported figure moved.

  Re-run on §22's far stronger ECAPA marginal the ordering **sharpens by a
  factor of four** (+0.132 to +0.437). And the naive sum stops being the
  villain: it costs 0.008 against the calibrated model's 0.007 at ρ = 0, where
  on the weak marginal it cost double. On a marginal representative of the
  current system, calibration matters *less* than §11 originally said and
  dependence modelling is actively harmful — so the defensible fusion for a
  deployment built on §22 is the simplest one available. That inverts the
  section's practical advice and is the reason the re-run was worth doing.

  **`_ASSUMED_STRENGTH` was found not to weaken anything.** Scaling log-LRs is
  monotonic, so `C_llr_min` is exactly unchanged and all three streams had
  identical discrimination; the constant reached only the naive sum, and at
  ρ = 0.6 scaling *down* helped it because under-confidence offsets
  double-counting. `weaken` is the mechanism that does work — it slides the
  same-source marginal toward the different-source one — and at the value that
  reproduces §13's operating point (0.70, giving the behavioural stream
  `C_llr_min` 0.541 against 0.54) **four of five cells still exclude zero and
  the fifth does not**. At ρ = 0.8 the interval spans zero under the Gaussian
  copula and recovers only barely under the t-copula ([+0.005, +0.150]). The
  claim is now *worse at low and moderate dependence, indistinguishable at
  ρ = 0.8* — do not restate it as "all five".
- **§21** Condition stratification works as designed (speakers with a repeated
  condition 75.2% → 0%, per-speaker mean-bitrate SD 1.42 → 0.66 kbit/s) and
  **ψ₁ moved only 44.951 → 43.967**. The condition confound is **refuted** as
  the cause of the ψ₁ spike. Control: the global arm retrained under current
  code reproduces the stored model to within 0.00075 across all 100 dimensions,
  so §19's stopping-rule change is not confounding it.

### Also fixed this session

- `PldaModel.effective_dimension` tested `psi > 1e-6` and never fired. Now 0.1,
  named `INERT_PSI`, absolute rather than relative (in the diagonalised space
  `W = I`, so `psi` is a between-to-within variance ratio). Five models read
  60/100, 91/124, 74/100, 76/100, 72/100.
- `n_iterations`, `final_log_likelihood` and `converged` were on `PldaModel` and
  **dropped by `save()`**. Same shape as the `training_speakers` object-array
  defect. Now persisted, with a test that strips them back out to prove old
  archives still load.
- `calibrator_comparison.json` held unbounded values under a field the document
  reads as bounded. Rerun; reproduces §5 to 4 dp and every §15 clip fraction.
  Both scripts now reach the reported quantity through one `as_reported` helper.
- Character n-grams moved from the script term to the idiolect term (§13 item 6).

---

## 5. Open work, in priority order

1. **Put §22, §25 and §26 on one evaluation set.** Each is internally paired —
   §22 at 306 speakers, §26 at 562 — but they sit on **two** evaluation sets, so
   the 0.343 → 0.033 progression is four models on two sets rather than four
   comparable measurements. `viflap/evaluation/reserved.py` pins 100 speakers
   and `split_by_speaker` takes `reserved_evaluation=`, but **no script passes
   it yet**. Thread it through `extract_neural`, `train_acoustic` and
   `evaluate_h1`, then re-extract (~10 h). Everything after that is comparable
   to everything else forever; nothing before it is.

2. ~~**Reserve a fixed evaluation set before any further corpus growth.**~~
   **Done.** `viflap/evaluation/reserved.py` names 100 speakers and
   `split_by_speaker` takes `reserved_evaluation=`. Pass it on every future
   training and evaluation run, or the guarantee is worthless. Common evaluation
   speakers across this project's two corpora go 19 → 50; not 100 only because
   the older 510-speaker corpus lacks half of them. **Any corpus that is a
   superset of the current 936 shares all hundred.** Do not edit the list — the
   module docstring explains why.

3. **Sweep capacity at 562 speakers.** §7 found more capacity hurt at 125 and §9
   noted the LDA ceiling had moved to 305; at 562 it is 561, so rank 200 would
   now pass through untruncated for the first time. Untested and cheap.

4. ~~**Build an abstention mechanism for the neural extractor.**~~ **Largely
   done, and it was a repair rather than a build.** The claim behind this item —
   that ECAPA "has no notion" of a recording too short to compare — was wrong.
   The gate existed; it measured wall-clock length under a name that promised
   speech. It now runs the same VAD as the i-vector front-end and a probe says
   it refuses at comparable rates (item 1). What remains is the genuinely
   *neural* question that item never asked: net speech is a proxy, and an
   embedding-space confidence — nearest-neighbour density, or the norm before
   length normalisation — might abstain better than a duration rule. Nothing
   here measures whether it would.

### Blocked on the user

- **`forensic_eval_01` cannot be obtained without you.** This is the one thing
  that would turn §12's benchmark from an inference across datasets into a
  measurement, and §22 makes it worth doing: the matched `C_llr` of 0.138 now
  sits in the range of E3FS3's case-specific conditions (0.085–0.097), which is
  exactly the best-cell-against-best-condition comparison §12 was rewritten to
  stop making. The like-for-like figure is 0.208 and this system has never been
  run on it.

  Checked this session: the YorVoice catalogue entry
  (`catalogue.yorvoice.york.ac.uk/catalogue/XRBEJSG2`) is **unreachable — six
  attempts, all timing out**, so record it as down rather than blocked.
  `forensic-evaluation.net` **is** reachable and serves the special issue, but
  publishes no download: obtaining the data runs through contacting the curator
  and agreeing to the evaluation's rules. That is an email to send and terms to
  accept, and neither is mine to do on your behalf.

- ~~Common Voice / AfriSpeech-200 usable-speaker counts need `validated.tsv`,
  behind account creation and terms acceptance.~~ **Half of this was wrong.**
  AfriSpeech-200 reports `gated=False` and serves its manifests without
  authentication — three CSVs, 22 MB, no audio and nothing to agree to. It was
  never blocked and the question is now answered in §8: **zero Zambian
  speakers**, and one Chichewa speaker who is Malawian. Do not re-open it.

  Common Voice remains genuinely gated: the dataset is a loading script that
  fetches from Mozilla's CDN behind terms acceptance. **Do not create accounts
  or accept terms on their behalf** — that restriction stands regardless of what
  the user has agreed to, because account creation is something I must not do.
  If the user downloads `validated.tsv` themselves, processing it is ordinary
  work.
- Whether UNZA holds an LDC membership. Would unlock Switchboard/Fisher/NIST SRE
  and require rewriting §8.
- A Kaggle API token exists but is **not configured**, and is not needed: see §6.

---

## 6. Operational rules, all learned the hard way

**Network — retry six times before recording anything as unreachable.** This
link flaps badly. An earlier session declared `huggingface.co` and
`download.pytorch.org` blocked on the strength of two attempts each; both are
reachable, and that error moved the entire extractor import onto a remote runner
for no reason.

| Host | Reality |
|---|---|
| `github.com` over SSH via `github-Personal` | works, 1–4 attempts |
| `huggingface.co`, `download.pytorch.org` | **works** |
| `www.kaggle.com`, `kaggle.com/api/v1` | **works** |
| `pypi.org`, `files.pythonhosted.org` | works |
| `www.openslr.org` | works at **~40 Mbit/s**, but **died at 46% of a 23 GB stream** and could not reconnect in 40 tries |
| **`openslr.elda.org`** | **the mirror to use** — same files, honours `Range`, ~42 Mbit/s |
| `us.openslr.org`, `openslr.magicdatatech.com` | no response |
| **`api.github.com`** | **blocked** — the only one that has stayed blocked |

Because `api.github.com` is blocked there is no `gh`, no workflow dispatch and
no artefact download. `.github/workflows/channel-validation.yml` therefore
triggers on a tag matching `channel-validation-*`, and commits its report plus a
status file carrying the run id — a run that starts always leaves a trace.
**Runs are slow (20–40 min wall) and queue behind each other; re-tagging while
waiting only lengthens the queue.**

**ONE HEAVY JOB AT A TIME.** 12 GB RAM, 8 cores. Two at once drives free RAM
under 0.5 GB, the machine pages, and a job sits at ~6% CPU for hours. Heavy =
`evaluate_h1`, `compare_capacity`, `compare_cmvn`, `train_acoustic`,
`extract_neural`, `compare_calibrators`. Check free RAM and `Pages/sec` if a job
looks slow rather than assuming it is compute-bound.

**Timings on the pooled corpus.** `train_acoustic` ~35 min (26 degrade, 9
train). `compare_calibrators` ~44 min at B=2000. `extract_neural` ~4.5 s per
30 s recording. Degradation dominates and scales with cores.

**Never pipe a long job through `tail` or `Select-Object -Last`** — it buffers
until exit and destroys progress visibility. Use:

```
Start-Process python -ArgumentList ... -RedirectStandardOutput <log> -RedirectStandardError <log>.err -NoNewWindow -PassThru
```

then tail the file. Jobs started this way have survived a session restart;
harness-backgrounded Bash commands have not.

**Bash has no network. PowerShell does.** Anything that fetches must run under
PowerShell.

**PowerShell 5.1 specifics.** No `&&`. Here-strings are fragile with embedded
quotes — prefer `git commit -F <file>` or repeated `-m`. **Never** round-trip a
source file through `Get-Content`/`Set-Content`: it read a UTF-8 file as ANSI
and double-encoded every em dash in `evaluate_h1.py`. Use the Edit tool.

**Windows specifics.** SpeechBrain symlinks checkpoints out of the HuggingFace
cache by default, which needs a privilege an ordinary account lacks; it fails
*after* a successful download with `WinError 1314` and reads like a download
failure. Use `LocalStrategy.COPY` — already done in
`viflap/infrastructure/neural_extractor.py`.

**Multiprocessing needs `if __name__ == "__main__":`** on Windows. Without it
every pool worker re-imports and re-runs the module, spawning its own pool until
the pool collapses.

**`evaluate_h1 --corpus` defaults to `data/corpus/librispeech` alone.** The
pooled model trained on librispeech + librispeech-360; evaluating it without
both roots scores it on speakers it memorised, and produced a spuriously good
0.216 once.

**`--noise` does not accept `clean`** — it is implicit. Valid: white, babble,
vehicle, pink. To get exactly {clean, babble20} pass
`--noise babble --snrs 20.0`.

**Never use built-in `hash()`** for seeding or attribution. Salted per
interpreter. Use `hashlib`, as `_stable_seed` and `_different_source_owner` do.

**Reflection coefficients, not polynomial coefficients**, when interpolating or
smoothing LPC across time. Averaging polynomials does not preserve stability and
the failure is data-dependent.

**Save/load round-trips must be tested.** Two separate defects have now shipped
because no test saved a model and read it back.

**When a new check fires, suspect the check.** The PLDA monotonicity guard fired
twice on correct code — once because the likelihood used unridged matrices, once
because a test fixture was ill-conditioned. Both times the fix was elsewhere.
Loosening a tolerance to absorb it would have hidden the next real defect.

**Measure, do not estimate**, quantities the argument rests on. The CMVN window
shares were assumed 8%/85% and are 11%/67%.

**Round from the artefact, not from the terminal — and check it with a script.**
Two figures in §22's calibration table were wrong in the third decimal because
they were rounded off a four-decimal progress line instead of the JSON: 0.0625
printed became 0.063 where the stored 0.06247 rounds to 0.062. Both were caught
by re-parsing the committed markdown table and comparing every cell against the
artefact, which takes a few lines and should be done for any table this document
gains:

```bash
python -c "import json,re; doc=open('docs/H1-acoustic-results.md',encoding='utf-8').read(); ..."
```

The rule "read the numbers back from the artefact" is not satisfied by having
*seen* the artefact's numbers go past in a log. Parse the file.

---

## 7. Working standard

Add tests for anything new. Keep pytest and ruff green and mypy on `viflap` at
63. Benchmark every number against published work before drawing a conclusion
from it — a previous session produced twelve sections of measurements before
checking any against the literature, and doing so overturned two conclusions.

Where a claim is withdrawn, **mark it in place with the reason** rather than
deleting it. Several sections carry correction blocks and that pattern should
continue.

Report what is measured, not what is hoped — including when a real defect turns
out to change nothing (§18) and when a well-motivated hypothesis is refuted
(§21). A method section reporting only the checks that changed something
misrepresents how often careful checks come back negative.

Commit in coherent chunks with human-written messages, no AI trailers or
attribution, and push (retry the flaky link).

**Against the named failure modes.** Do not declare a section finished until its
artefact is on disk and its numbers have been read back from that artefact
rather than from the terminal. Do not mark work done that has not been run. If
you leave something half-finished, say so here before the context ends.
