/**
 * Tests for the client-side output policy.
 *
 * These check the property that matters most in the interface: that a developer
 * cannot ship a label reading "Find matches" without the build failing.
 */

import { describe, expect, it } from 'vitest';

import {
  OutputConstraintViolation,
  assertPermitted,
  describeEvidence,
  findViolations,
  formatPosterior,
  formatPriorOdds,
} from '../src/safety/language';

describe('output language policy', () => {
  it('permits the vocabulary of evidential weight', () => {
    const permitted = [
      'The evidence provides strong support for the linkage hypothesis.',
      'Likelihood ratio 10^3.2, prior odds 1 in 99,999.',
      'This result rests on a single evidence stream.',
    ];
    permitted.forEach((text) => expect(assertPermitted(text)).toBe(text));
  });

  it('refuses the vocabulary of identity', () => {
    const refused = [
      'We found a match',
      'Find matches',
      'No matches found',
      'The suspect was identified',
      'voiceprint analysis',
      'voice  print comparison',
      'These are the same individual',
      'MATCHED',
    ];
    refused.forEach((text) =>
      expect(() => assertPermitted(text)).toThrow(OutputConstraintViolation),
    );
  });

  it('matches on word boundaries rather than substrings', () => {
    // Rejecting ordinary English trains developers to disable the check, which
    // defeats it more thoroughly than a gap would.
    const permitted = [
      'Dispatched to the review queue.',
      'Rematch the filter parameters.',
      'Batch attached to the case file.',
    ];
    permitted.forEach((text) => expect(() => assertPermitted(text)).not.toThrow());
  });

  it('names the offending phrase and where it came from', () => {
    try {
      assertPermitted('We found a match', 'SearchView.emptyState');
      throw new Error('should have refused');
    } catch (error) {
      expect(error).toBeInstanceOf(OutputConstraintViolation);
      const violation = error as OutputConstraintViolation;
      expect(violation.phrase).toBe('match');
      expect(violation.origin).toBe('SearchView.emptyState');
      expect(violation.message).toContain('linkage hypothesis');
    }
  });

  it('finds every violation, not only the first', () => {
    expect(findViolations('a match and an identified voiceprint')).toHaveLength(3);
  });
});

describe('evidential description', () => {
  it('always states direction alongside strength', () => {
    // A band without its direction inverts the finding: 10^-4 is "very strong"
    // support for *different* sources.
    expect(describeEvidence(4)).toContain('same actor');
    expect(describeEvidence(-4)).toContain('different actors');
    expect(describeEvidence(4)).toContain('very strong');
    expect(describeEvidence(-4)).toContain('very strong');
  });

  it('reports no assistance near neutral', () => {
    expect(describeEvidence(0.1)).toContain('no assistance');
    expect(describeEvidence(-0.1)).toContain('no assistance');
  });

  it('produces policy-compliant text at every magnitude', () => {
    for (let log10 = -8; log10 <= 8; log10 += 0.25) {
      expect(() => assertPermitted(describeEvidence(log10))).not.toThrow();
    }
  });
});

describe('number formatting', () => {
  it('does not round small posteriors away to zero', () => {
    // The difference between one in ten thousand and one in ten million is the
    // quantity that makes a database search result interpretable.
    expect(formatPosterior(0.0000001)).not.toBe('0.0%');
    expect(formatPosterior(0.0000001)).toContain('e-');
    expect(formatPosterior(0.00005)).toBe('0.005%');
    expect(formatPosterior(0.98)).toBe('98.0%');
  });

  it('renders prior odds in the form investigators reason in', () => {
    expect(formatPriorOdds(-Math.log(99999))).toBe('1 in 99,999');
    expect(formatPriorOdds(0)).toBe('1.00 : 1');
  });
});
