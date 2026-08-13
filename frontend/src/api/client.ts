/**
 * Typed API client.
 *
 * Two behaviours here are safety properties rather than conveniences.
 *
 * **A case reference is a required constructor argument, not a per-call
 * option.** A client cannot be built without one, so there is no code path that
 * issues an unbound request. The server would refuse it anyway; making it
 * impossible to *compose* means the failure surfaces while writing the call
 * rather than at runtime in front of a user.
 *
 * **Server errors are surfaced with their remedy.** The API returns a structured
 * refusal explaining what to do differently, and a client that reduced it to
 * "Request failed" would discard the useful half. Governance refusals are the
 * common case and are usually actionable.
 */

import { assertPermitted } from '../safety/language';
import type {
  Analyst,
  ApiError,
  ComparisonResult,
  HealthStatus,
  SearchResults,
} from './types';

export class ApiRefusal extends Error {
  constructor(
    readonly status: number,
    readonly detail: ApiError,
  ) {
    super(detail.message);
    this.name = 'ApiRefusal';
  }

  /** Whether this refusal is a governance constraint rather than a fault. */
  get isGovernance(): boolean {
    return this.status === 403;
  }

  /** Whether the system declined because the data could not support an answer. */
  get isDeclined(): boolean {
    return this.status === 422;
  }
}

export class ViflapClient {
  constructor(
    private readonly baseUrl: string,
    private readonly analyst: Analyst,
    private readonly caseReference: string,
    /**
     * Bound to `globalThis`, and it must be.
     *
     * Held as a class property, `fetch` would otherwise be invoked with the
     * client as its receiver, and the browser rejects that with "Illegal
     * invocation" — `fetch` requires the global object as its `this`. The
     * failure appears only in a browser and only on the default: every test
     * injects its own implementation, so a suite can be entirely green while
     * no request the application makes can succeed.
     */
    private readonly fetchImpl: typeof fetch = globalThis.fetch.bind(globalThis),
  ) {
    if (!caseReference.trim()) {
      throw new Error(
        'A case reference is required. Every operation on evidence is bound to a ' +
          'filed complaint; there is no unbound path.',
      );
    }
  }

  private headers(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      'X-Analyst-Id': this.analyst.id,
      'X-Analyst-Roles': this.analyst.roles.join(','),
      'X-Case-Reference': this.caseReference,
    };
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers: { ...this.headers(), ...(init?.headers ?? {}) },
    });

    // `unknown`, not the `any` that `Response.json()` yields. The value is
    // whatever crossed the network, and typing it as `any` would let it be
    // destructured as a result without anything having checked that it is one —
    // including the `prior` the types exist to guarantee is present.
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      throw new ApiRefusal(
        response.status,
        (body as ApiError) ?? {
          error_type: 'TransportError',
          message: `The service returned ${response.status}.`,
          context: {},
          remedy: '',
        },
      );
    }
    return body as T;
  }

  async health(): Promise<HealthStatus> {
    return this.request<HealthStatus>('/health');
  }

  async compare(
    incidentA: string,
    incidentB: string,
    restriction?: { population: number; justification: string },
  ): Promise<ComparisonResult> {
    const params = new URLSearchParams({
      incident_a: incidentA,
      incident_b: incidentB,
    });
    if (restriction) {
      params.set('restricted_population', String(restriction.population));
      params.set('restriction_justification', restriction.justification);
    }
    const result = await this.request<ComparisonResult>(
      `/api/v1/comparisons?${params.toString()}`,
      { method: 'POST' },
    );
    return this.checkLanguage(result);
  }

  async search(
    probeIncidentId: string,
    maxResults = 25,
    restriction?: { population: number; justification: string },
  ): Promise<SearchResults> {
    const results = await this.request<SearchResults>('/api/v1/searches', {
      method: 'POST',
      body: JSON.stringify({
        probe_incident_id: probeIncidentId,
        case_reference: this.caseReference,
        max_results: maxResults,
        restricted_population: restriction?.population ?? null,
        restriction_justification: restriction?.justification ?? '',
      }),
    });
    assertPermitted(results.mandatory_caveat, 'api.search.caveat');
    results.results.forEach((result) => this.checkLanguage(result));
    return results;
  }

  /**
   * Verify that server-supplied text is policy-compliant before it is rendered.
   *
   * Redundant with the server's own check, and kept because the two run in
   * different processes. A server built from a different revision, or an
   * intermediary that rewrote a field, would be caught here rather than
   * displayed.
   */
  private checkLanguage(result: ComparisonResult): ComparisonResult {
    assertPermitted(result.verbal_summary, 'api.comparison.summary');
    result.caveats.forEach((caveat, index) =>
      assertPermitted(caveat, `api.comparison.caveat[${index}]`),
    );
    return result;
  }
}
