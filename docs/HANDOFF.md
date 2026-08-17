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
| Tests | **671, all passing** |
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
```

Current best remains the pooled i-vector model: `C_llr_min` 0.276, EER 7.89% at
12.2 kbit/s clean, 30 s, 102 evaluation speakers.

---

## 3. Running now

**`python -m scripts.extract_neural`** (no arguments, so all defaults: both
corpora, durations 30/15/5, output `data/reports/neural_embeddings.npz`).

Progress log: `<scratchpad>/extract_neural.log`. At last check 1200/1539 of the
training partition at ~4.5 s/rec. Expect the training partition, then
development and evaluation at two conditions each. **Do not start another heavy
job while it runs** — see §6.

When it finishes:

```bash
python -m scripts.score_neural --resamples 2000
```

That fits LDA/WCCN/PLDA and scores the same 102 held-out speakers, writing
`data/reports/h1_neural.json`. It has been exercised end to end on a fabricated
archive; it has never seen real embeddings.

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

1. **Finish the neural extractor comparison.** Extraction is running. Then
   `score_neural.py`, then write §22. Two things to state in the write-up: the
   comparison is **not paired at the vector level** (both systems see the same
   audio but produce different vectors, so no paired difference test as in §7
   and §9), and 192 dimensions from 1,539 training embeddings is thinner than
   the i-vector system's 100 — if PLDA looks unstable, rerun with
   `--lda-dimension 100`, which costs seconds because extraction is separate.

2. **Wire the synthetic corpus into an end-to-end run.**
   `scripts/synthesise_incidents.py` generates operators, operations and
   incidents with the documented dependence structure. It is **not yet driving**
   `tests/integration/test_pipeline.py`, which still uses a scalar
   `SignatureComparator` test double. That wiring is the step that actually
   exercises fusion, the validity gate and the audit chain on realistic
   incidents. **Hard boundary: no figure from it is a result.** §11 already had
   to be withdrawn for treating simulation output as measurement.

3. **§11 has no intervals.** Point estimates from one seed in a document that
   insists on intervals everywhere. Repeat across seeds and bootstrap before
   quoting any figure. Also test misspecification — generate under a t-copula,
   fit the Gaussian.

4. **Behavioural `min_words = 40`** is indefensible against a literature floor
   of 2,500–5,000 words (§13). Raising it needs a published *forensic* operating
   point to cite; do the search rather than invent a number.

5. **Corpus expansion.** 510 usable speakers came from a fetch stopped at 41%.
   Completing `train-clean-360` reaches ~761; `train-other-500` adds ~1,100.
   Bandwidth-bound, not compute-bound.

6. **What produces the ψ₁ spike**, now that the confound is refuted. §21 lists
   three candidates: LibriSpeech session effects, length normalisation, and
   upward bias in a leading eigenvalue estimated from 306 speakers. The last is
   partly argued against already — the 125-speaker model shows a *smaller*
   ratio, the wrong direction for an estimation artefact.

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
