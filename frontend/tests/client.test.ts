/**
 * Tests for the API client.
 *
 * The first test exists because of a defect that reached a running browser: the
 * client held the global `fetch` as a class property, so it was invoked with the
 * client as its receiver and every request failed with "Illegal invocation".
 * The whole suite was green throughout, because every other test injects its own
 * fetch and never exercises the default. A test that injects a stub cannot catch
 * a bug in the thing the stub replaces, so this one does not inject.
 */

import { describe, expect, it, vi } from 'vitest';

import { ApiRefusal, ViflapClient } from '../src/api/client';
import { OutputConstraintViolation } from '../src/safety/language';
import type { Analyst, ComparisonResult } from '../src/api/types';

const ANALYST: Analyst = { id: 'inv-042', roles: ['investigator'] };

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

describe('default fetch binding', () => {
  it('calls the global fetch with a receiver the browser accepts', async () => {
    // A real function on globalThis that throws when its receiver is wrong,
    // exactly as the browser's fetch does.
    const original = globalThis.fetch;
    let sawGlobalReceiver = false;
    globalThis.fetch = function (this: unknown) {
      if (this !== globalThis && this !== undefined) {
        throw new TypeError("Failed to execute 'fetch' on 'Window': Illegal invocation");
      }
      sawGlobalReceiver = true;
      return Promise.resolve(jsonResponse({ status: 'ok' }));
    } as unknown as typeof fetch;

    try {
      const client = new ViflapClient('http://127.0.0.1:8000', ANALYST, 'ZP-2025-01847');
      await expect(client.health()).resolves.toEqual({ status: 'ok' });
      expect(sawGlobalReceiver).toBe(true);
    } finally {
      globalThis.fetch = original;
    }
  });
});

describe('governance and refusals', () => {
  it('requires a case reference to construct a client at all', () => {
    expect(() => new ViflapClient('http://x', ANALYST, '   ')).toThrow(
      /case reference is required/i,
    );
  });

  it('sends the analyst and case reference on every request', async () => {
    // Captured through a typed closure rather than read off a mock's call
    // record, which is untyped and would defeat the point of checking headers
    // whose names the server relies on.
    let captured: RequestInit | undefined;
    const capture = ((_url: string, init?: RequestInit) => {
      captured = init;
      return Promise.resolve(jsonResponse({ status: 'ok' }));
    }) as unknown as typeof fetch;

    const client = new ViflapClient('http://x', ANALYST, 'ZP-2025-01847', capture);
    await client.health();

    const headers = captured?.headers as Record<string, string>;
    expect(headers['X-Analyst-Id']).toBe('inv-042');
    expect(headers['X-Analyst-Roles']).toBe('investigator');
    expect(headers['X-Case-Reference']).toBe('ZP-2025-01847');
  });

  it('surfaces a governance refusal with its remedy rather than a bare failure', async () => {
    const detail = {
      error_type: 'AuthorityViolation',
      message: 'the principal does not hold the query authority',
      context: {},
      remedy: 'Request the investigator role.',
    };
    const spy = vi.fn().mockResolvedValue(jsonResponse(detail, 403));
    const client = new ViflapClient('http://x', ANALYST, 'ZP-2025-01847', spy);

    await expect(client.compare('a', 'b')).rejects.toMatchObject({
      name: 'ApiRefusal',
      detail: { remedy: 'Request the investigator role.' },
    });
    await client.compare('a', 'b').catch((error: unknown) => {
      expect(error).toBeInstanceOf(ApiRefusal);
      expect((error as ApiRefusal).isGovernance).toBe(true);
    });
  });
});

describe('server text is re-checked before it is rendered', () => {
  it('rejects a result whose summary asserts identity', async () => {
    // The server has its own policy; this is the client half, and it exists
    // because the two run in different processes. A server built from another
    // revision would otherwise have its wording rendered unchallenged.
    const result: Partial<ComparisonResult> = {
      verbal_summary: 'The recordings are a match.',
      caveats: [],
    };
    const spy = vi.fn().mockResolvedValue(jsonResponse(result));
    const client = new ViflapClient('http://x', ANALYST, 'ZP-2025-01847', spy);

    await expect(client.compare('a', 'b')).rejects.toBeInstanceOf(
      OutputConstraintViolation,
    );
  });

  it('rejects a caveat that asserts identity', async () => {
    const result: Partial<ComparisonResult> = {
      verbal_summary: 'The evidence provides weak support for the proposition.',
      caveats: ['The speaker was positively identified.'],
    };
    const spy = vi.fn().mockResolvedValue(jsonResponse(result));
    const client = new ViflapClient('http://x', ANALYST, 'ZP-2025-01847', spy);

    await expect(client.compare('a', 'b')).rejects.toBeInstanceOf(
      OutputConstraintViolation,
    );
  });
});
