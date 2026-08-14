# Running parts of this on Colab

The local development machine has 8 cores, 12 GB of RAM, a 5.4 Mbit/s link, and
no ffmpeg build carrying AMR-NB. Colab's free tier has roughly 2 cores, the same
12 GB, about 100 GB of scratch disk, a gigabit link, `apt-get`, and sometimes a
T4. It is not a faster machine — it is a differently shaped one, and the split
of work follows from the shape rather than from preference.

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
be uploaded. Trigger it from the Actions tab; it installs the encoder, fetches a
deterministic two-speaker sample, measures, commits the report back, and uploads
it as an artefact as well.

The result is the same result, not merely the same kind of one: the same Ubuntu
ffmpeg package and the same `libopencore-amrnb`, the parametric coder is seeded,
and the corpus selection is deterministic. Free runners are 2-core with a 6-hour
cap — useless for a sweep, ample for a measurement that takes minutes.

The report arrives as a commit rather than as a table to retype. That is
deliberate: carrying a figure by hand from one context into another is how §12
read a back-end training size as a whole training resource and how §14 ported a
bias shift measured at 42 speakers onto a result computed on 102. Both are
recorded in the results document. A number that arrives by commit is the number
that was measured.

Kaggle is the third option and the only one with a real CLI (`kaggle kernels
push`, `status`, `output`), which makes it the right home for the extractor
import: 4 cores, ~30 GB RAM, 30 GPU-hours a week, and drivable from a terminal
rather than a tab.
