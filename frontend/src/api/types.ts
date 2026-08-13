/**
 * Types mirroring the API response schemas.
 *
 * `prior` is non-optional on every type carrying a likelihood ratio. That is
 * the client-side half of the server's guarantee: a component cannot destructure
 * a result and render its likelihood ratio without the prior being in scope,
 * because TypeScript will not let the object exist without one.
 */

export interface PriorContext {
  log_odds: number;
  odds_description: string;
  basis: string;
  justification: string;
  supplied_by: string;
  population_size: number | null;
}

export interface StreamContribution {
  stream: string;
  status: 'evidence' | 'absent';
  log10_lr: number | null;
  interval_lower_log10: number | null;
  interval_upper_log10: number | null;
  model_id: string | null;
  absence_reason: string | null;
  absence_detail: string | null;
  diagnostics: Record<string, number>;
}

export interface ComparisonResult {
  incident_a: string;
  incident_b: string;
  case_reference: string;

  fused_log10_lr: number;
  interval_lower_log10: number;
  interval_upper_log10: number;

  /** Pre-rendered by the server, stating strength and direction together. */
  verbal_summary: string;

  /** Never optional. A result without it is the raw material of a fallacy. */
  prior: PriorContext;
  posterior_log_odds: number;
  posterior_probability: number;
  posterior_probability_lower: number | null;
  posterior_probability_upper: number | null;

  streams: StreamContribution[];
  contributing_stream_count: number;
  rests_on_single_stream: boolean;
  acoustic_excluded_by_validity_gate: boolean;

  fusion_method: string;
  fusion_model_id: string;
  naive_log10_lr: number | null;
  independence_inflation_log10: number | null;

  /** Must be displayed. Derived server-side from the result's own structure. */
  caveats: string[];
  computed_at: string;
}

export interface SearchResults {
  probe_incident_id: string;
  case_reference: string;
  results: ComparisonResult[];
  n_candidates_compared: number;
  /** Candidates that could not be compared — not candidates ruled out. */
  n_declined: number;
  prior: PriorContext;
  mandatory_caveat: string;
  searched_at: string;
}

export interface ApiError {
  error_type: string;
  message: string;
  context: Record<string, string>;
  remedy: string;
}

export interface HealthStatus {
  status: string;
  enrolled_incidents: number;
  audit_chain_intact: boolean;
  audit_entries: number;
  case_reference_format: string;
}

export interface Analyst {
  id: string;
  roles: string[];
}
