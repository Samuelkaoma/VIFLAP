/**
 * Search results.
 *
 * The mandatory caveat is rendered *above* the ranking, not below it. Placing it
 * after the list means it is read after the reader has already formed a view
 * from the top entry, and the caveat's whole purpose is to condition how that
 * entry is read.
 *
 * The count of candidates that could not be compared is shown alongside the
 * count compared. Omitting it implies they were considered and excluded on the
 * evidence, which is a materially different claim from "their records did not
 * permit a comparison".
 */

import React from 'react';

import { assertPermitted, formatPriorOdds } from '../safety/language';
import type { SearchResults } from '../api/types';
import { ResultCard } from './ResultCard';

export const SearchView: React.FC<{ results: SearchResults }> = ({ results }) => (
  <div className="search-view">
    <section className="search-view__context" role="status">
      <h2 className="search-view__title">
        Linkage hypotheses for {results.probe_incident_id}
      </h2>
      <dl className="search-view__stats">
        <div>
          <dt>Candidates compared</dt>
          <dd>{results.n_candidates_compared.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Could not be compared</dt>
          <dd>{results.n_declined.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Prior odds applied</dt>
          <dd>{formatPriorOdds(results.prior.log_odds)}</dd>
        </div>
      </dl>
      <p className="search-view__caveat">
        {assertPermitted(results.mandatory_caveat, 'SearchView.caveat')}
      </p>
    </section>

    {results.results.length === 0 ? (
      <p className="search-view__empty">
        No candidate produced a comparison. This is not a finding that the probe
        is unrelated to the enrolled population: it means no candidate shared an
        evidence stream with it.
      </p>
    ) : (
      <ol className="search-view__results">
        {results.results.map((result, index) => (
          <li key={`${result.incident_a}|${result.incident_b}`}>
            <ResultCard result={result} rank={index + 1} />
          </li>
        ))}
      </ol>
    )}
  </div>
);
