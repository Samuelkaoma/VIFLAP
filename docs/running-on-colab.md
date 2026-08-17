# Running parts of this somewhere else

The local development machine has 8 cores, 12 GB of RAM, a 5.4 Mbit/s link, and
no ffmpeg build carrying AMR-NB. Colab's free tier has roughly 2 cores, the same
12 GB, about 100 GB of scratch disk, a gigabit link, `apt-get`, and sometimes a
T4. It is not a faster machine — it is a differently shaped one, and the split
of work follows from the shape rather than from preference.

## What this network can and cannot reach

Measured rather than assumed, because it decides which jobs can run here at all.

**Retry before concluding anything is blocked.** This link flaps badly — a git
push takes one to four attempts routinely — and a single failed request means
nothing. An earlier version of this table listed `huggingface.co` and
`download.pytorch.org` as blocked on the strength of two attempts each, and both
are reachable within six. That error was load-bearing: it moved the pre-trained
extractor import off this machine and onto a runner for no reason. Anything
recorded here as unreachable should have been tried at least six times.

| Host | From the development machine |
|---|---|
| `github.com` over SSH via the `github-Personal` alias | works, 1–4 attempts |
| `raw.githubusercontent.com`, `github.com` HTML | works |
| `pypi.org`, `files.pythonhosted.org` | works |
| `huggingface.co`, incl. model resolve URLs | **works**, 1–2 attempts |
| `download.pytorch.org` | **works** |
| `www.kaggle.com`, `kaggle.com/api/v1` | **works**, 1–2 attempts |
| `openslr.org` | works, ~5.4 Mbit/s, intermittent |
| **`api.github.com`** | **blocked** — no `gh`, no workflow dispatch, no artefact download |

`api.github.com` is the one entry that has survived repeated retries across a
whole session, and it is the only one the workflow design has to work around.

Two consequences follow, and both shaped how the work is now organised.

**Workflows are triggered by a tag, not by the Actions tab.** `api.github.com`
is unreachable, so neither `gh workflow run` nor the button is available from
where the work is done. `channel-validation.yml` therefore also triggers on a
tag matching `channel-validation-*`, which is a deliberate act expressed in the
one protocol that does work:

```bash
git tag -a channel-validation-5 -m "why this run exists" && git push origin channel-validation-5
```

**The result comes back as a commit, and it has to.** Artefacts are fetched
through `api.github.com`, so an artefact cannot be retrieved from here at all.
The job commits its report to `main` instead, and — because a run that dies
before measuring anything would otherwise be indistinguishable from a slow one —
commits a report saying so when it fails. `git fetch` is the only status channel
there is.

**A pre-trained embedding extractor can be downloaded here after all**, which is
the correction above in its most consequential form. The SpeechBrain VoxCeleb2
checkpoint and the PyTorch wheel index are both reachable, so the whole of §12's
recommended move — import an extractor, retrain only LDA and PLDA — is a local
job on eight cores rather than something that has to be shipped anywhere. See
"Importing an extractor" below.

| | Local | Colab free |
|---|---:|---:|
| Cores | **8** | ~2 |
| RAM | 12 GB | 12 GB |
| Scratch disk | — | ~100 GB |
| Download | 5.4 Mbit/s | ~gigabit |
| AMR-NB encoder | no | **yes, via apt** |
| GPU | none | T4 when offered |

**Keep the degradation here.** The parametric coder costs about a quarter of a
second per second of audio and is pure Python, so it parallelises across
recordings and nothing else. 1,539 training recordings take 25 minutes on 7
workers locally and would take an hour and a half on two.

**Send the bandwidth- and GPU-bound work there.** Channel validation, corpus
expansion, and importing a pre-trained embedding extractor.

**Never upload the corpus.** `fetch_corpus.py` selects a bounded subset
deterministically — fixed chapters per speaker, fixed utterances per chapter,
taken in archive order — and `split_by_speaker` rebuilds the same three-way split
from the same corpus in every script. So a notebook can recreate the exact
experimental setup from a clone and a download, in minutes, without anything
crossing the local uplink. Models are 6-23 MB and move freely in either
direction; the corpus never has to move at all.

---

## Cell 1 — what the session actually gave you

Run this first. Colab's allocation varies between sessions and the answer
changes what is worth attempting.

```python
!nproc && free -g | head -2 && df -h /content | tail -1
!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU"
```

Two cores and no GPU is the common free-tier allocation and is enough for the
channel validation. A T4 is what makes the extractor import worth doing there.

## Cell 2 — the channel validation

This is the measurement §16 of the results document says has to happen and could
not be made locally. Every absolute figure in that document is currently scoped
"through this parametric model"; this replaces that with a number.

```python
!git clone -q https://github.com/Samuelkaoma/VIFLAP.git
%cd VIFLAP
!apt-get -qq install -y ffmpeg libopencore-amrnb-dev
!pip install -q -e ".[experiments]"
!ffmpeg -hide_banner -encoders | grep -i amr
```

That last line is the gate. It should list `libopencore_amrnb`; if it prints
nothing, the encoder is absent and everything below will correctly report
`available: false` and measure nothing.

```python
# Two speakers is ample: the comparison is coder against coder over identical
# input, so it needs representative speech rather than a corpus.
!python -m scripts.fetch_corpus --max-speakers 2 --chapters-per-speaker 2 --utterances-per-chapter 6
!python -m scripts.validate_channel \
    --audio data/corpus/librispeech --pattern '*.flac' \
    --max-recordings 6 --max-seconds 30 \
    --output channel_validation.json
```

The script prints a table and writes the JSON. Bring the JSON back — the printed
table is enough to read, but the file is the artefact the document should cite.

**On comparability with §16.** Those figures came from six BembaSpeech
recordings. This uses LibriSpeech, so the absolute distortion will differ a
little with the material. The number that matters is not affected: the
parametric-against-reference row compares two coders over the *same* input, and
that difference is what scopes the document.

If exact comparability with §16 is wanted, upload the six BembaSpeech files
(34 MB) and point `--audio` at them instead.

## Getting results back

The printed table can simply be read. For the JSON, either download it from the
file browser, or push it — but note that `data/` is gitignored on purpose, so a
report has to be force-added by name rather than by loosening the rule that
keeps evidence out of the history:

```python
!git add -f channel_validation.json && git -c user.email=you@example.com -c user.name=you commit -qm "Channel validation from Colab" && git push
```

Pushing needs a credential. Use Colab's secrets panel rather than pasting a
token into a cell, or just download the file.

---

## The headless alternative, which is the better one for this job

`.github/workflows/channel-validation.yml` does the same thing with no browser
and, crucially, **no credentials**: `actions/checkout` uses the automatic per-run
token, so a private repository needs no personal access token and nothing has to
be uploaded. Push a `channel-validation-*` tag; it installs the encoder, fetches
a deterministic sixteen-speaker sample, measures, commits the report back, and
uploads it as an artefact as well.

**`ffmpeg` alone is not enough, and this cost a run to find out.** Debian and
Ubuntu build libavcodec twice and the default package omits the
patent-encumbered and GPLv3 codecs, opencore-amr among them. The first run
installed `ffmpeg`, got a working ffmpeg that could not encode AMR-NB, and
correctly reported `available: false`. `libavcodec-extra` is the flavour that
carries it. Ubuntu 24.04's ffmpeg 6.1.1 then lists `libopencore_amrnb` as an
encoder, which is what the report now records so the next person does not have
to guess.

The result is the same result, not merely the same kind of one: the same Ubuntu
ffmpeg package and the same `libopencore-amrnb`, the parametric coder is seeded,
and the corpus selection is deterministic. Free runners are 2-core with a 6-hour
cap — useless for a sweep, ample for a measurement that takes minutes.

**A run takes far longer in wall time than it does in compute, so be patient
before concluding anything.** Measuring 48 recordings takes under three minutes;
a whole run took upwards of forty, and five tagged runs in a row appeared to
commit nothing at all. They had not failed — they were queued behind each other
and behind the corpus fetch, and every one of them eventually landed. The
temptation at that point is to keep re-tagging, which lengthens the queue and
makes the symptom worse.

Two things make the diagnosis possible rather than guessed at. The status file
carries the run id, so it differs on every run and is always committed, and the
`log_tail` field carries the last forty lines of the measurement's own output.
Between them, a committed status file means the job ran and says how it ended.

**The monthly minute budget is the other thing that can stop a run, and running
out of it *is* silent from here.** A private repository gets 2,000 free Actions
minutes a month on the Free plan and 3,000 on Pro or Team, with public
repositories unmetered on standard runners; Linux minutes count 1:1, Windows
2:1, macOS 10:1. Each run of this workflow costs roughly ten to twenty, mostly
fetching the corpus. There is no limit on how many *workflow files* a repository
may have, and no per-run cap beyond the six-hour job timeout — the budget is
minutes and storage, not workflows. When it is exhausted GitHub does not start
the run at all, and nothing is committed. That is the one case the status file
cannot cover, so if runs stop landing for good rather than slowly, the billing
page is where to look.

The report arrives as a commit rather than as a table to retype. That is
deliberate: carrying a figure by hand from one context into another is how §12
read a back-end training size as a whole training resource and how §14 ported a
bias shift measured at 42 speakers onto a result computed on 102. Both are
recorded in the results document. A number that arrives by commit is the number
that was measured.

Kaggle is the third option and the only one with a real CLI (`kaggle kernels
push`, `status`, `output`), which makes it the right home for the extractor
import: 4 cores, ~30 GB RAM, 30 GPU-hours a week, and drivable from a terminal
rather than a tab. It needs an account and an API token, which is a decision for
the user rather than something to be arranged on their behalf.

---

## Importing an extractor: what the shape of the machines actually implies

§12 concludes that the move worth making is to take a publicly pre-trained
VoxCeleb2 embedding extractor and retrain only LDA and PLDA on the 306 speakers
already on disk. Two things about that are commonly assumed and are wrong here.

**It does not need a GPU.** Training an extractor would; running one over 1,539
training and 1,039 held-out recordings is inference, and ECAPA-TDNN inference on
eight cores is a matter of hours rather than days. The GPU tiers matter for
turnaround, not for feasibility.

**And it does not need anything this machine cannot do.** The checkpoint is an
80 MB fetch from a host that is reachable, so every stage runs here:

| Stage | Where | Why |
|---|---|---|
| Fetch the checkpoint | **local** | reachable; one 80 MB download, cached |
| Degrade the corpus | **local** | pure-Python codec, 8 cores against 2–4 |
| Extract embeddings | **local** | CPU inference, hours not days |
| LDA / PLDA and evaluation | **local** | seconds on 192-dimensional vectors |

Nothing has to be shipped anywhere, the corpus never moves, and no 80 MB binary
needs to enter a repository that deliberately keeps evidence out of its history.
An earlier version of this document recommended sending the work to a runner and
having it commit the embeddings back; that recommendation existed only because
`huggingface.co` had been declared unreachable after two attempts. It is
withdrawn.

The remote tiers keep one genuine advantage — a GPU turns hours of extraction
into minutes — and that is a turnaround argument rather than a feasibility one.
It is worth reaching for if the import becomes iterative, and not before.
