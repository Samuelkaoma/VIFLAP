/**
 * Client-side mirror of the server's output-language policy.
 *
 * The server already refuses to emit the vocabulary of identity, so this is not
 * the primary control. It exists because the *client* composes text too —
 * labels, tooltips, empty states, the text a user copies into a report — and
 * none of that passes through the server's policy.
 *
 * The failure this prevents is specific and likely. A developer adds a button
 * labelled "Find matches", or an empty state reading "No matches found", and the
 * interface now uses the exact vocabulary the entire system is arranged to
 * avoid. It would look like a UI detail and would be the most consequential text
 * on the screen, because it is what an investigator reads and repeats.
 */

/** Vocabulary that asserts identity or a conclusion. Mirrors the server's list. */
export const PROHIBITED_PHRASES: readonly string[] = [
  'match',
  'matches',
  'matched',
  'matching',
  'identified',
  'identification',
  'identify',
  'identifies',
  'voiceprint',
  'voice print',
  'voice fingerprint',
  'is the same person',
  'are the same person',
  'same individual',
  'positively identified',
  'conclusively',
  'proves',
  'proven',
  'guilty',
  'the suspect is',
];

/**
 * Word-boundary matcher.
 *
 * Boundaries, not substrings. Substring matching rejects "dispatch" and
 * "rematch", and an interface that rejects ordinary words trains developers to
 * disable the check — which defeats it more completely than a gap would.
 * Multi-word phrases tolerate arbitrary whitespace so a line break cannot evade
 * them.
 */
const PATTERN = new RegExp(
  `(?<![\\w-])(?:${PROHIBITED_PHRASES.map((phrase) =>
    phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/ /g, '\\s+'),
  ).join('|')})(?![\\w-])`,
  'gi',
);

export class OutputConstraintViolation extends Error {
  constructor(
    readonly phrase: string,
    readonly origin: string,
  ) {
    super(
      `Text from "${origin}" asserts identity ("${phrase}"). VIFLAP reports the ` +
        `weight of evidence and has no vocabulary for identity. Use "the evidence ` +
        `provides ... support for" or "linkage hypothesis" instead.`,
    );
    this.name = 'OutputConstraintViolation';
  }
}

/** Find every prohibited phrase in a string. */
export function findViolations(text: string): string[] {
  return Array.from(text.matchAll(PATTERN)).map((found) => found[0]);
}

/**
 * Return `text` unchanged, or throw.
 *
 * Throws rather than rewriting. Silently correcting the wording would leave in
 * place whatever code path formed a conclusion it had no basis to form, and the
 * next person would reintroduce it.
 */
export function assertPermitted(text: string, origin = 'ui'): string {
  // Destructured rather than indexed: under `noUncheckedIndexedAccess` an
  // element read is `string | undefined`, and narrowing on the binding is what
  // proves the throw path has a phrase to report. Testing `length > 0` does not
  // narrow the element type, so the compiler is right to reject it.
  const [first] = findViolations(text);
  if (first !== undefined) {
    throw new OutputConstraintViolation(first.toLowerCase(), origin);
  }
  return text;
}

/**
 * Verbal band for a base-10 log likelihood ratio.
 *
 * Deliberately returns only the band. There is no exported function producing a
 * band on its own as display text, because a band without its direction inverts
 * the finding: a log10 LR of -4 is "very strong" support for *different*
 * sources, and rendering "very strong" beside two incident numbers says the
 * opposite of what the evidence says.
 *
 * Use {@link describeEvidence} for anything a person will read.
 */
function bandFor(log10Lr: number): string {
  const magnitude = Math.abs(log10Lr);
  if (magnitude < 0.3) return 'provides no assistance';
  if (magnitude < 1) return 'provides weak support for';
  if (magnitude < 2) return 'provides moderate support for';
  if (magnitude < 3) return 'provides moderately strong support for';
  if (magnitude < 4) return 'provides strong support for';
  if (magnitude < 6) return 'provides very strong support for';
  return 'provides extremely strong support for';
}

/** A complete sentence stating strength *and* direction. */
export function describeEvidence(log10Lr: number): string {
  const band = bandFor(log10Lr);
  if (band === 'provides no assistance') {
    return 'The evidence provides no assistance in distinguishing the two propositions.';
  }
  const proposition =
    log10Lr > 0
      ? 'that the incidents were conducted by the same actor'
      : 'that the incidents were conducted by different actors';
  return `The evidence ${band} the proposition ${proposition}.`;
}

/**
 * Format a posterior probability for display.
 *
 * Small probabilities are rendered in full rather than rounded to "0.0%".
 * Rounding away the difference between one in ten thousand and one in ten
 * million erases the very quantity that makes a database search result
 * interpretable — and rounds the most common honest answer to something that
 * looks like an error.
 */
export function formatPosterior(probability: number): string {
  if (probability >= 0.01) return `${(probability * 100).toFixed(1)}%`;
  // 1e-5, not 1e-4. Three decimal places of a percentage carry information down
  // to 0.001%, which is a probability of 1e-5; switching to exponential at 1e-4
  // abandoned a whole decade the fixed form still renders exactly, so a
  // one-in-twenty-thousand posterior was shown in exponent notation while
  // "0.005%" — the form an investigator reads without decoding — was available.
  // Below 1e-5 the fixed form would round to "0.000%", which is the failure this
  // function exists to prevent, so that is where the exponential begins.
  if (probability >= 0.00001) return `${(probability * 100).toFixed(3)}%`;
  if (probability <= 0) return 'vanishingly small';
  return `${(probability * 100).toExponential(1)}%`;
}

/** Format prior odds as "1 in N", the form investigators reason in. */
export function formatPriorOdds(logOdds: number): string {
  const odds = Math.exp(logOdds);
  if (odds >= 1) return `${odds.toPrecision(3)} : 1`;
  return `1 in ${Math.round(1 / odds).toLocaleString()}`;
}
