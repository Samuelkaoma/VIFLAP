# VIFLAP investigator interface

Interface design is a safety-critical component of this system, not a
presentation layer. Its requirements come from section 7.5 of the research
proposal and each is structural.

## What the interface guarantees

**Prior odds are displayed with every result.** Adjacent to the likelihood
ratio, in the same grid — not in a tooltip and not behind a disclosure control.
The prior, the evidence and the posterior are laid out as one equation, so a
future layout change cannot separate the posterior from the prior that produced
it.

**Posterior probability appears only alongside the prior it was derived from.**
Where the prior dominates — the ordinary outcome of a large database search —
the interface says so in words rather than leaving it to be inferred from a
small number.

**Per-stream contribution is always visible.** Never collapsed by default: nobody
expands a panel to check whether the answer they have already read was sound. A
result resting on one stream is distinguished by a border treatment as well as
by a written warning, because the requirement is that the two cases be visually
distinguishable and that cannot depend on the reader having read the paragraph.

**Absent streams are listed with their reason.** A stream that silently
disappears is indexed identically to one that was never attempted, and the
difference matters: "the recording was judged synthetic" is a finding, "no
transaction records exist" is a data-access problem.

**The interface has no vocabulary for identity.** Every string rendered passes
through `src/safety/language.ts`, which throws on the vocabulary of identity.
This is a client-side mirror of the server's policy and it exists because the
*client* composes text too — labels, empty states, tooltips — and none of that
passes through the server. A button reading "Find matches" fails the test suite.

## What it deliberately does not do

It does not compose a strength band from a magnitude and a sign. A band without
its direction inverts the finding — a log10 LR of -4 is "very strong" support
for *different* sources — so the interface renders the sentence the server
produced.

It does not offer a threshold control that filters results by default. Imposing
a threshold inside the system pre-empts a judgement belonging to the
investigator.

## Running it

```bash
npm install
npm run dev        # http://127.0.0.1:3000, expects the API on :8000
npm test           # output-policy and formatting tests
npm run typecheck  # strict TypeScript, no implicit any, no unchecked indexing
```

Set `VITE_VIFLAP_API` to point at a different API origin.
