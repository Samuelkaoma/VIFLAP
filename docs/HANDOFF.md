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

Then read `docs/H1-acoustic-results.md` (~2,400 lines, §§1–21). It is the
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
| Tests | **728, all passing** |
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
acoustic_pooled.npz             128/100, 306 spk                §9, system of record
acoustic_pooled_cmvn_utt.npz    CMVN 0                          §17 control
acoustic_pooled_cmvn100.npz     CMVN 100                        §17 control
acoustic_pooled_global.npz      306 spk, global allocation      §21 control
acoustic_pooled_stratified.npz  306 spk, stratified allocation  §21
pretrained/spkrec-ecapa-voxceleb/   ECAPA-TDNN, 89 MB, 192-dim
countermeasure_english.npz      300 English spk                 §10
pretrained/spkrec-ecapa-voxceleb/   ECAPA-TDNN, 89 MB, 192-dim  §22
```

Embeddings and reports, all under the gitignored `data/reports/`:

```
neural_embeddings.npz     ECAPA over the whole corpus, 6.5 MB   §22
neural_extraction.json    13 cells, 0 refusals anywhere         §22
h1_neural.json            the §22 result table
h1_neural_lda100.json     the §22 dimension control
h1_pooled_ownership.json  the i-vector baseline §22 compares to
overstatement.json        §11, 40 replicates, Gaussian copula
overstatement_tcopula.json §11, 40 replicates, Student-t copula
```

**Current best is now the borrowed ECAPA extractor with the same 306-speaker
back-end: `C_llr_min` 0.099 [0.031, 0.230], EER 2.47% at 12.2 kbit/s clean, 30 s,
102 evaluation speakers — the first `supported` cell in the document (§22).** The
pooled i-vector model is 0.276 / 7.89% on the identical trial set and remains the
reference implementation.

Do not compare a new result against §9's table without checking which bootstrap
it used. §9 quotes percentile intervals, §14 recomputed them as BCa, and §18
recomputed those again under the symmetric ownership rule. `score_neural.py`
inherits the §14 and §18 corrections, so `h1_pooled_ownership.json` — not §9 — is
the like-for-like i-vector column.

---

## 3. Running now

**Nothing.** `extract_neural` completed (330 min, 13 cells, 0 refusals) and
`score_neural` completed twice — once at the native 192 dimensions and once at
`--lda-dimension 100` as a control. §22 is written and its numbers were read
back off `h1_neural.json` rather than off the terminal.

Timings for planning: `extract_neural` 330 min end to end at ~6 s per recording
per three durations; `score_neural --resamples 2000` 16 min.

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
  changed, 306 → ~6,000. **The comparison is not paired**, so the direction is
  not established at a stated confidence; the marginal intervals do overlap
  slightly and §7 is the standing reminder that overlap decides nothing either
  way. Control at `--lda-dimension 100` says the result is not an artefact of
  the 192-dimensional transform.
- **§13 (behavioural length)** `MIN_WORDS_IDIOLECT = 500` from Ishihara (2017)
  on predatory chatlog messages; `MIN_WORDS_SCRIPT` stays at 40 and is marked as
  having no citation, because it has none. Two floors, not one — the published
  requirements were measured for authorship attribution, which is only the
  idiolect half. Below the floor the idiolect term is **withheld**, and that
  required a guard: with idiolect pinned at zero the delegation flag fires on
  anything with script evidence, including a transcript compared with itself.
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

1. **Make §22's comparison paired.** This is the largest claim in the document
   resting on marginal intervals, and §7 exists precisely to say that is not
   good enough. The trial sets are **identical** at 30 s and 15 s (849
   same-source, 132,796 different-source in both systems), so
   `paired_bootstrap_over_speakers` applies directly — the only obstacle is that
   neither `evaluate_h1.py` nor `score_neural.py` persists per-trial scores.
   Add that (`compare_calibrators.py` already does it, for the same reason),
   then run the four cells. Cheap: no re-extraction, no retraining.
   **Do not pair the 5 s cells** — the i-vector front-end refused 12 and 73
   recordings there and ECAPA refused none, so the trials do not correspond.

2. **Rerun §21's ψ₁ question on the ECAPA embeddings.** They are on disk, so
   this costs minutes. The extractor changed and the corpus did not: if the
   spike is a LibriSpeech session effect it should survive; if it is an i-vector
   length-normalisation artefact it should not. §22 reports ψ₁ = 66.98 but not
   the ψ₁/ψ₂ *ratio*, which is the comparable quantity.

3. **Re-run §11 against a marginal that is not superseded.** The intervals and
   the t-copula arm are done; what remains is that `calibration_scores.npz` is
   the **42-speaker** §4/§5 baseline evaluation, so the acoustic stream feeding
   the simulation is far weaker than §9's pooled model and much weaker than
   §22's borrowed extractor. Regenerating it from a current model is the
   remaining piece, and §13's benchmark says the behavioural strength factor
   (0.75 of acoustic) is optimistic too — its forensic operating point is
   `C_llr` 0.54 at 500 tokens, *worse* than acoustic rather than 75% of it. Both
   assumptions should move in the same pass.

4. **Put audio on the synthetic pipeline.** `scripts/synthetic_pipeline.py`
   drives four streams end to end with the real comparators, but not the
   acoustic one — so `_validity_absence` is never consulted and **the validity
   gate is still unexercised end to end**. `Operator.acoustic_speaker_id` is the
   hook: bind operators to real LibriSpeech speakers, embed, register
   `AcousticStreamComparator`. Heavy, because it needs the corpus and a model.

5. **Corpus expansion.** 510 usable speakers came from a fetch stopped at 41%.
   Completing `train-clean-360` reaches ~761; `train-other-500` adds ~1,100.
   Bandwidth-bound, not compute-bound. Note §22 weakens the *urgency*: the
   binding constraint was the extractor's speaker count and that has been
   bought rather than collected. More speakers now buy back-end quality only.

6. **Place the system on `forensic_eval_01`.** §12's benchmark is an inference
   across datasets until this is run, and §22 makes the gap worth measuring
   properly rather than narrating.

### Blocked on the user

- Common Voice / AfriSpeech-200 usable-speaker counts need `validated.tsv`,
  behind account creation and terms acceptance. **Do not create accounts or
  accept terms on their behalf.**
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
| `openslr.org` | works, ~5.4 Mbit/s, intermittent |
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
