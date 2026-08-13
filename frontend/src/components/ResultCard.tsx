/**
 * The result card.
 *
 * Interface design is a safety-critical component here, not presentation. The
 * layout below is arranged around four constraints from section 7.5 of the
 * research proposal, and each is structural rather than a matter of styling:
 *
 * 1. **The prior is displayed with every result.** Not in a tooltip, not behind
 *    a disclosure control — adjacent to the number it conditions. A likelihood
 *    ratio shown without its prior is the raw material of the prosecutor's
 *    fallacy.
 *
 * 2. **The posterior is shown only alongside the prior that produced it**, and
 *    when the prior dominates — the ordinary outcome of a large database search
 *    — that is stated in words rather than left to be inferred from a small
 *    number.
 *
 * 3. **Per-stream contribution is always visible**, and a result resting on one
 *    stream is visually distinguished from one supported by several. A single
 *    stream at 10^4 and four streams agreeing at 10^1 reach the same fused
 *    figure and warrant very different confidence.
 *
 * 4. **Language is constrained.** Every string rendered passes through the
 *    output policy, and the component uses the server's pre-rendered sentence
 *    rather than composing one from a strength band and a sign.
 */

import React from 'react';

import {
  assertPermitted,
  describeEvidence,
  formatPosterior,
  formatPriorOdds,
} from '../safety/language';
import type { ComparisonResult } from '../api/types';
import { StreamBreakdown } from './StreamBreakdown';
import { CaveatList } from './CaveatList';

interface Props {
  result: ComparisonResult;
  rank?: number;
}

export const ResultCard: React.FC<Props> = ({ result, rank }) => {
  const priorDominates =
    result.fused_log10_lr > 0 && result.posterior_log_odds < 0;
  const intervalSpansNeutral =
    result.interval_lower_log10 <= 0 && result.interval_upper_log10 >= 0;

  return (
    <article
      className={[
        'result-card',
        result.rests_on_single_stream ? 'result-card--single-stream' : '',
        intervalSpansNeutral ? 'result-card--indeterminate' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      aria-label={`Comparison of ${result.incident_a} and ${result.incident_b}`}
    >
      <header className="result-card__header">
        <div>
          {rank !== undefined && <span className="result-card__rank">#{rank}</span>}
          <h3 className="result-card__incidents">
            <span>{result.incident_a}</span>
            <span aria-hidden="true" className="result-card__link">
              &harr;
            </span>
            <span>{result.incident_b}</span>
          </h3>
          <p className="result-card__case">
            Authorised by case {result.case_reference}
          </p>
        </div>
        <div className="result-card__strength">
          <span className="result-card__lr-label">Likelihood ratio</span>
          <span className="result-card__lr-value">
            10<sup>{result.fused_log10_lr.toFixed(2)}</sup>
          </span>
          <span className="result-card__lr-interval">
            interval 10<sup>{result.interval_lower_log10.toFixed(2)}</sup> to 10
            <sup>{result.interval_upper_log10.toFixed(2)}</sup>
          </span>
        </div>
      </header>

      {/*
        The server's own sentence, not one composed here. Composing it client-side
        would mean combining a strength band with a sign, and getting that
        backwards renders "very strong support" for the opposite proposition
        while looking entirely correct.
      */}
      <p className="result-card__summary">
        {assertPermitted(result.verbal_summary, 'ResultCard.summary')}
      </p>

      {/*
        The prior and the posterior are one block. They are not separable in the
        markup, because a layout change that moved the posterior somewhere the
        prior is not would reintroduce precisely the error this arrangement
        exists to prevent.
      */}
      <section className="prior-posterior" aria-label="Prior odds and posterior">
        <div className="prior-posterior__cell">
          <span className="prior-posterior__label">Prior odds of this search</span>
          <span className="prior-posterior__value">
            {formatPriorOdds(result.prior.log_odds)}
          </span>
          <span className="prior-posterior__note">
            {result.prior.basis.replace(/_/g, ' ')}
            {result.prior.population_size !== null &&
              ` over ${result.prior.population_size.toLocaleString()} candidates`}
            {' — supplied by '}
            {result.prior.supplied_by}
          </span>
        </div>
        <div className="prior-posterior__operator" aria-hidden="true">
          &times;
        </div>
        <div className="prior-posterior__cell">
          <span className="prior-posterior__label">Evidence</span>
          <span className="prior-posterior__value">
            10<sup>{result.fused_log10_lr.toFixed(2)}</sup>
          </span>
          <span className="prior-posterior__note">
            {describeEvidence(result.fused_log10_lr)}
          </span>
        </div>
        <div className="prior-posterior__operator" aria-hidden="true">
          =
        </div>
        <div className="prior-posterior__cell prior-posterior__cell--posterior">
          <span className="prior-posterior__label">Posterior probability</span>
          <span className="prior-posterior__value">
            {formatPosterior(result.posterior_probability)}
          </span>
          {result.posterior_probability_lower !== null &&
            result.posterior_probability_upper !== null && (
              <span className="prior-posterior__note">
                interval {formatPosterior(result.posterior_probability_lower)} to{' '}
                {formatPosterior(result.posterior_probability_upper)}
              </span>
            )}
        </div>
      </section>

      <p className="prior-posterior__justification">
        {assertPermitted(result.prior.justification, 'ResultCard.priorJustification')}
      </p>

      {priorDominates && (
        <p className="alert alert--prior" role="status">
          <strong>The prior dominates this result.</strong> The evidence supports
          linkage, but against the prior odds of this search the proposition
          remains less probable than not. This is the ordinary outcome of a large
          search and is not a defect in the evidence.
        </p>
      )}

      {result.rests_on_single_stream && (
        <p className="alert alert--single-stream" role="status">
          <strong>This result rests on a single evidence stream.</strong> It does
          not survive a failure of that one model. A result supported by several
          streams at lower individual strength is more robust than this one at
          the same fused figure.
        </p>
      )}

      {result.acoustic_excluded_by_validity_gate && (
        <p className="alert alert--gated" role="status">
          <strong>Acoustic evidence was excluded.</strong> The recording was not
          admitted by the validity gate. It was removed from fusion entirely
          rather than given reduced weight, because a recording that is not human
          speech carries no information about a human speaker.
        </p>
      )}

      <StreamBreakdown streams={result.streams} />

      {result.independence_inflation_log10 !== null && (
        <p className="result-card__dependence">
          Treating the streams as independent would have placed this result{' '}
          <strong>
            {result.independence_inflation_log10 >= 0 ? '+' : ''}
            {result.independence_inflation_log10.toFixed(2)} orders of magnitude
          </strong>{' '}
          from its dependence-corrected value. Fusion method:{' '}
          {result.fusion_method.replace(/_/g, ' ')}.
        </p>
      )}

      <CaveatList caveats={result.caveats} />

      <footer className="result-card__footer">
        <span>Computed {new Date(result.computed_at).toLocaleString()}</span>
        <span className="result-card__model">Model {result.fusion_model_id}</span>
      </footer>
    </article>
  );
};
