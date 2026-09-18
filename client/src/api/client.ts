import type { ZodTypeAny, output as ZodOutput } from 'zod';
import { ApiError, kindForStatus } from './errors';
import { clearToken, readToken } from './token';

/**
 * The single place this client talks to the platform API.
 *
 * Components never call `fetch`. Everything goes through here so that authentication, timeouts,
 * cancellation, error normalisation and response validation exist once rather than per call
 * site, and so a change to any of them is one edit.
 */

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
  signal?: AbortSignal;
  /** Provider credentials for live execution, forwarded per request and never stored. */
  credentials?: ProviderCredentials;
  idempotencyKey?: string;
  timeoutMs?: number;
}

/**
 * The platform accepts these as per-request headers and never persists them. This client holds
 * them in memory for the life of a submission for the same reason: they are not ours to keep.
 */
export interface ProviderCredentials {
  openaiApiKey?: string;
  /** One header per provider, never a generic one: a key sent to the wrong provider fails
   *  with an authentication error that names nothing useful. */
  anthropicApiKey?: string;
  githubToken?: string;
}

export interface ApiClientConfig {
  baseUrl: string;
  /** A personal bearer token, or the explicit shared-admin compatibility credential. */
  token: string | undefined;
  /** Runtime token lookup used by the shipped login page. */
  getToken?: () => string | undefined;
  defaultTimeoutMs: number;
  /**
   * Called when the platform answers `401`, before the error is thrown.
   *
   * Here rather than at each call site because this is the one place `fetch` happens, and a
   * session can expire during *any* request -- a poll, a stream reconnect, a background
   * refetch nobody is looking at. Handled per call site it would be handled in the loud
   * places and forgotten in the quiet ones, leaving a page that renders stale data and an
   * error box instead of a login screen.
   *
   * It clears the session and returns; it does not navigate. Deciding where the person goes
   * belongs to the application, not to the transport.
   */
  onUnauthorized?: () => void;
}

const DEFAULT_TIMEOUT_MS = 30_000;

export function readConfigFromEnv(): ApiClientConfig {
  const env = import.meta.env;
  return {
    // `/api` on this origin, which is what both the dev proxy and a same-origin production
    // deployment serve. Not the bare origin: the API's `/features/{id}` and this application's
    // own `/features/{id}` are the same URL, so served together the API wins and every page
    // load of a feature returns JSON instead of the app.
    baseUrl: env.VITE_API_BASE_URL ?? '/api',
    // Read at call time, not baked in: see `token.ts` for why a build-time token is the wrong
    // place for this one.
    // Do not snapshot the runtime token here. `AppProviders` is created before the login
    // page runs, so a snapshot stays empty even after somebody signs in.
    token: undefined,
    getToken: readToken,
    defaultTimeoutMs: DEFAULT_TIMEOUT_MS,
    // A session that has expired or been revoked stops being this browser's credential the
    // moment the platform says so. `SessionProvider` reacts to the cleared token by showing
    // the login screen; the client's job ends at forgetting the token.
    onUnauthorized: clearToken,
  };
}

export class ApiClient {
  constructor(private readonly config: ApiClientConfig) {}

  /**
   * The absolute URL this client would call for a path.
   *
   * Streams are read with a bare `fetch` -- they have to be, to hold the response open and
   * consume it incrementally -- so they cannot go through `request`. Exposing the URL rule
   * keeps that one piece shared rather than letting a second base-URL convention appear.
   */
  absoluteUrl(path: string): string {
    return new URL(
      `${this.config.baseUrl}${path}`,
      globalThis.location?.origin ?? 'http://localhost',
    ).toString();
  }

  /**
   * Read one binary response through the same authentication every other read uses.
   *
   * Separate from `request` rather than a flag on it, because everything after the fetch
   * differs: there is no schema to validate against and the body is a `Blob`. It exists for
   * exactly one caller -- a design preview, which the server renders on demand -- and it
   * exists at all because an `<img src>` cannot carry a bearer token, so the alternative
   * would be handing the browser a provider URL to fetch for itself.
   */
  async requestBlob(path: string, options: { signal?: AbortSignal } = {}): Promise<Blob> {
    const controller = new AbortController();
    const timeoutMs = this.config.defaultTimeoutMs;
    const timeout = setTimeout(
      () => controller.abort(new DOMException('timeout', 'TimeoutError')),
      timeoutMs,
    );
    const onCallerAbort = () => controller.abort(options.signal?.reason);
    options.signal?.addEventListener('abort', onCallerAbort, { once: true });

    let response: Response;
    try {
      response = await fetch(this.absoluteUrl(path), {
        method: 'GET',
        signal: controller.signal,
        headers: this.headers({
          hasBody: false,
          credentials: undefined,
          idempotencyKey: undefined,
        }),
      });
    } catch (cause) {
      throw this.abortOrNetworkError(cause, options.signal);
    } finally {
      clearTimeout(timeout);
      options.signal?.removeEventListener('abort', onCallerAbort);
    }
    if (!response.ok) throw await this.responseError(response);
    return response.blob();
  }

  // Generic over the schema rather than over a bare payload type: `z.infer` on a schema that
  // uses `.default()` otherwise resolves to the *input* shape, making required fields optional
  // at every call site.
  async request<S extends ZodTypeAny>(
    path: string,
    schema: S,
    options: RequestOptions = {},
  ): Promise<ZodOutput<S>> {
    const { method = 'GET', body, query, signal, credentials, idempotencyKey } = options;
    const timeoutMs = options.timeoutMs ?? this.config.defaultTimeoutMs;

    const url = new URL(this.absoluteUrl(path));
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }

    // One controller aborts on either the caller's signal or the timeout, so a hung request
    // cannot outlive the component that asked for it.
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(new DOMException('timeout', 'TimeoutError')), timeoutMs);
    const onCallerAbort = () => controller.abort(signal?.reason);
    signal?.addEventListener('abort', onCallerAbort, { once: true });

    let response: Response;
    try {
      response = await fetch(url.toString(), {
        method,
        signal: controller.signal,
        headers: this.headers({ hasBody: body !== undefined, credentials, idempotencyKey }),
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (cause) {
      throw this.abortOrNetworkError(cause, signal);
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', onCallerAbort);
    }

    if (!response.ok) throw await this.responseError(response);

    const raw: unknown = response.status === 204 ? {} : await response.json().catch(() => ({}));
    const parsed = schema.safeParse(raw);
    if (!parsed.success) {
      // A shape this client cannot read is a client defect. Say so rather than rendering
      // half-parsed data and failing somewhere further away from the cause.
      throw new ApiError('schema', 'Unexpected response shape', response.status, parsed.error.message);
    }
    return parsed.data;
  }

  /**
   * Send one `FormData` body, and read the JSON answer through the same schema check.
   *
   * A sibling of `request` rather than a flag on it, because `request` does two things this
   * cannot tolerate: it sets `Content-Type: application/json` whenever a body exists, and it
   * `JSON.stringify`s that body. A multipart upload needs the browser to set the header --
   * only the browser knows the boundary it generated -- so the funnel keeps its own rule and
   * this method states a different one, rather than the funnel growing a condition every
   * other call has to read past.
   *
   * Everything else is deliberately identical: the same URL rule, the same timeout and
   * cancellation, the same error normalisation, and the same response validation. A second
   * `fetch` that skipped any of those is the thing this client exists to prevent.
   */
  async upload<S extends ZodTypeAny>(
    path: string,
    schema: S,
    body: FormData,
    options: { signal?: AbortSignal; timeoutMs?: number } = {},
  ): Promise<ZodOutput<S>> {
    const { signal } = options;
    const timeoutMs = options.timeoutMs ?? this.config.defaultTimeoutMs;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(new DOMException('timeout', 'TimeoutError')), timeoutMs);
    const onCallerAbort = () => controller.abort(signal?.reason);
    signal?.addEventListener('abort', onCallerAbort, { once: true });

    let response: Response;
    try {
      response = await fetch(this.absoluteUrl(path), {
        method: 'POST',
        signal: controller.signal,
        // `hasBody: false` on purpose: it is what suppresses the JSON content type. The
        // browser writes `multipart/form-data` with its own boundary, and a value we wrote
        // would be a boundary nothing matches.
        headers: this.headers({ hasBody: false, credentials: undefined, idempotencyKey: undefined }),
        body,
      });
    } catch (cause) {
      throw this.abortOrNetworkError(cause, signal);
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', onCallerAbort);
    }

    if (!response.ok) throw await this.responseError(response);
    const raw: unknown = response.status === 204 ? {} : await response.json().catch(() => ({}));
    const parsed = schema.safeParse(raw);
    if (!parsed.success) {
      throw new ApiError('schema', 'Unexpected response shape', response.status, parsed.error.message);
    }
    return parsed.data;
  }

  /**
   * Fetch one authenticated binary body, for showing an image the platform holds.
   *
   * Not through `request`, which validates a JSON response against a schema; there is no
   * schema for bytes. It is still this client's method rather than a bare `fetch` at a
   * component, because the bearer token, the base URL rule, the timeout and the error
   * normalisation all have to be the same ones — and because every attachment read is
   * authenticated, so `<img src>` cannot be used and something has to do this.
   */
  async fetchBlob(path: string, options: { signal?: AbortSignal } = {}): Promise<Blob> {
    const { signal } = options;
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(new DOMException('timeout', 'TimeoutError')),
      this.config.defaultTimeoutMs,
    );
    const onCallerAbort = () => controller.abort(signal?.reason);
    signal?.addEventListener('abort', onCallerAbort, { once: true });

    let response: Response;
    try {
      response = await fetch(this.absoluteUrl(path), {
        signal: controller.signal,
        headers: this.headers({ hasBody: false, credentials: undefined, idempotencyKey: undefined }),
      });
    } catch (cause) {
      throw this.abortOrNetworkError(cause, signal);
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', onCallerAbort);
    }
    if (!response.ok) throw await this.responseError(response);
    return await response.blob();
  }

  private headers(input: {
    hasBody: boolean;
    credentials: ProviderCredentials | undefined;
    idempotencyKey: string | undefined;
  }): HeadersInit {
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (input.hasBody) headers['Content-Type'] = 'application/json';
    const token = this.config.getToken?.() ?? this.config.token;
    if (token) headers.Authorization = `Bearer ${token}`;
    if (input.idempotencyKey) headers['Idempotency-Key'] = input.idempotencyKey;
    if (input.credentials?.openaiApiKey) headers['X-OpenAI-Api-Key'] = input.credentials.openaiApiKey;
    if (input.credentials?.anthropicApiKey)
      headers['X-Anthropic-Api-Key'] = input.credentials.anthropicApiKey;
    if (input.credentials?.githubToken) headers['X-GitHub-Token'] = input.credentials.githubToken;
    return headers;
  }

  private abortOrNetworkError(cause: unknown, signal: AbortSignal | undefined): ApiError {
    if (signal?.aborted) return new ApiError('aborted', 'Request cancelled');
    if (cause instanceof DOMException && cause.name === 'TimeoutError') {
      return new ApiError('timeout', 'Request timed out');
    }
    return new ApiError('network', 'Network request failed', undefined, describe(cause));
  }

  private async responseError(response: Response): Promise<ApiError> {
    // FastAPI returns `{"detail": ...}`; `detail` is a list for 422. Both are safe to surface:
    // the server replaces unexpected exceptions with a fixed string before they reach here.
    const body: unknown = await response.json().catch(() => undefined);
    const detail = extractDetail(body);
    if (response.status === 401) this.config.onUnauthorized?.();
    return new ApiError(kindForStatus(response.status), detail ?? response.statusText, response.status, detail);
  }
}

function extractDetail(body: unknown): string | undefined {
  if (typeof body !== 'object' || body === null || !('detail' in body)) return undefined;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) =>
        typeof item === 'object' && item !== null && 'msg' in item
          ? String((item as { msg: unknown }).msg)
          : JSON.stringify(item),
      )
      .join('; ');
  }
  return undefined;
}

function describe(cause: unknown): string | undefined {
  return cause instanceof Error ? cause.message : undefined;
}
