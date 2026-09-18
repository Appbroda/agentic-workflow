import { describe, expect, it, vi, afterEach } from 'vitest';
import { z } from 'zod';
import { ApiClient } from '@/api/client';
import { ApiError } from '@/api/errors';

const config = { baseUrl: '', token: 'test-token', defaultTimeoutMs: 1_000 };
const schema = z.object({ ok: z.boolean() });

function respond(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe('ApiClient', () => {
  it('sends bearer auth and provider credentials as per-request headers', async () => {
    const fetchMock = vi.fn().mockResolvedValue(respond(200, { ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    await new ApiClient(config).request('/features', schema, {
      method: 'POST',
      body: { a: 1 },
      idempotencyKey: 'key-1',
      credentials: { openaiApiKey: 'sk-test', githubToken: 'ghp-test' },
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers.Authorization).toBe('Bearer test-token');
    expect(headers['Idempotency-Key']).toBe('key-1');
    // The server accepts these per request and never persists them; neither does the client.
    expect(headers['X-OpenAI-Api-Key']).toBe('sk-test');
    expect(headers['X-GitHub-Token']).toBe('ghp-test');
  });

  it('reads a runtime token after the API client has already been created', async () => {
    const runtime: { token: string | undefined } = { token: undefined };
    const fetchMock = vi.fn().mockResolvedValue(respond(200, { ok: true }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new ApiClient({ ...config, token: undefined, getToken: () => runtime.token });

    // AppProviders is mounted before TokenGate accepts a key. The request happens afterwards.
    runtime.token = 'entered-after-mount';
    await client.request('/features', schema);

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>).Authorization).toBe(
      'Bearer entered-after-mount',
    );
  });

  it('maps a 409 to a conflict rather than a generic failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(respond(409, { detail: 'feature is already running' })),
    );

    const error = await new ApiClient(config)
      .request('/features/x/cancel', schema, { method: 'POST' })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ApiError);
    // The platform decides which transitions are legal; a refusal is an outcome, not a fault,
    // and must not be retried.
    expect((error as ApiError).kind).toBe('conflict');
    expect((error as ApiError).isRetryable).toBe(false);
    expect((error as ApiError).detail).toBe('feature is already running');
  });

  it('flattens FastAPI 422 detail lists into one readable string', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        respond(422, { detail: [{ msg: 'field required' }, { msg: 'not a valid url' }] }),
      ),
    );

    const error = (await new ApiClient(config)
      .request('/features/start', schema, { method: 'POST' })
      .catch((caught: unknown) => caught)) as ApiError;

    expect(error.kind).toBe('validation');
    expect(error.detail).toBe('field required; not a valid url');
  });

  it('reports a response it cannot parse as a client-side schema failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(respond(200, { ok: 'yes' })));

    const error = (await new ApiClient(config)
      .request('/features', schema)
      .catch((caught: unknown) => caught)) as ApiError;

    // Not a server error: the server answered fine and this client did not understand it.
    expect(error.kind).toBe('schema');
  });

  it('turns a network failure into a retryable error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    const error = (await new ApiClient(config)
      .request('/features', schema)
      .catch((caught: unknown) => caught)) as ApiError;

    expect(error.kind).toBe('network');
    expect(error.isRetryable).toBe(true);
  });
});
