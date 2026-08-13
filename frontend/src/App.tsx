/**
 * Application shell.
 *
 * The persistent banner is not decoration. Automation bias is well documented:
 * reviewers tend to ratify algorithmic output rather than scrutinise it, and the
 * banner is one of the interface choices hypothesis H6 exists to test. It states
 * what the system does and does not do, and it does not scroll away.
 */

import React, { useCallback, useMemo, useState } from 'react';

import { ApiRefusal, ViflapClient } from './api/client';
import type { Analyst, SearchResults } from './api/types';
import { SearchView } from './components/SearchView';

const BASE_URL = import.meta.env?.VITE_VIFLAP_API ?? 'http://127.0.0.1:8000';

export const App: React.FC = () => {
  const [analyst] = useState<Analyst>({ id: '', roles: ['investigator'] });
  const [caseReference, setCaseReference] = useState('');
  const [analystId, setAnalystId] = useState('');
  const [probe, setProbe] = useState('');
  const [results, setResults] = useState<SearchResults | null>(null);
  const [refusal, setRefusal] = useState<ApiRefusal | null>(null);
  const [busy, setBusy] = useState(false);

  const canSearch = Boolean(caseReference.trim() && analystId.trim() && probe.trim());

  const client = useMemo(() => {
    if (!caseReference.trim() || !analystId.trim()) return null;
    return new ViflapClient(
      BASE_URL,
      { ...analyst, id: analystId },
      caseReference,
    );
  }, [analyst, analystId, caseReference]);

  const runSearch = useCallback(async () => {
    if (!client) return;
    setBusy(true);
    setRefusal(null);
    setResults(null);
    try {
      setResults(await client.search(probe.trim()));
    } catch (error) {
      if (error instanceof ApiRefusal) setRefusal(error);
      else throw error;
    } finally {
      setBusy(false);
    }
  }, [client, probe]);

  return (
    <div className="app">
      <header className="app__banner" role="banner">
        <div className="app__identity">
          <h1>VIFLAP</h1>
          <p>Calibrated multi-evidence case linkage</p>
        </div>
        <p className="app__constraint">
          This system reports <strong>how much the evidence shifts the odds</strong>{' '}
          that two incidents share an actor. It does not assert identity, and it
          has no vocabulary for doing so. A likelihood ratio is not a probability:
          converting one requires the prior odds shown beside every result.
        </p>
      </header>

      <form
        className="query-bar"
        onSubmit={(event) => {
          event.preventDefault();
          void runSearch();
        }}
      >
        <label className="query-bar__field">
          <span>Case reference</span>
          <input
            value={caseReference}
            onChange={(event) => setCaseReference(event.target.value)}
            placeholder="ZP-2025-01847"
            required
            aria-describedby="case-help"
          />
          <small id="case-help">
            Required. Every operation is bound to a filed complaint.
          </small>
        </label>

        <label className="query-bar__field">
          <span>Analyst identifier</span>
          <input
            value={analystId}
            onChange={(event) => setAnalystId(event.target.value)}
            placeholder="inv-042"
            required
          />
        </label>

        <label className="query-bar__field">
          <span>Probe incident</span>
          <input
            value={probe}
            onChange={(event) => setProbe(event.target.value)}
            placeholder="ZP-2025-00104"
            required
          />
        </label>

        <button type="submit" disabled={!canSearch || busy}>
          {busy ? 'Evaluating evidence…' : 'Evaluate linkage hypotheses'}
        </button>
      </form>

      {refusal && (
        <div
          className={`alert ${refusal.isGovernance ? 'alert--governance' : 'alert--error'}`}
          role="alert"
        >
          <strong>{refusal.isGovernance ? 'Refused' : 'Could not complete'}:</strong>{' '}
          {refusal.detail.message}
          {refusal.detail.remedy && (
            <p className="alert__remedy">{refusal.detail.remedy}</p>
          )}
        </div>
      )}

      <main className="app__main">
        {results ? (
          <SearchView results={results} />
        ) : (
          !busy && (
            <p className="app__placeholder">
              Enter a case reference, your analyst identifier and a probe incident
              to evaluate linkage hypotheses against the enrolled population.
            </p>
          )
        )}
      </main>
    </div>
  );
};
