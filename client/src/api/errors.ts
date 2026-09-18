/**
 * One normalised error shape for everything the API layer can fail with.
 *
 * The kinds are separated because the UI genuinely reacts differently to each: `conflict` is
 * the server refusing an illegal workflow transition and is a normal, explainable outcome
 * rather than a fault, and `schema` means the server answered in a shape this client does not
 * understand -- which is a client bug, not a user problem.
 */
export type ApiErrorKind =
  | 'network'
  | 'timeout'
  | 'aborted'
  | 'unauthorized'
  | 'forbidden'
  | 'not_found'
  | 'validation'
  | 'conflict'
  | 'server'
  | 'schema'
  | 'unknown';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | undefined;
  /** Server-supplied explanation, safe to show. Never a stack trace: the API does not send one. */
  readonly detail: string | undefined;

  constructor(kind: ApiErrorKind, message: string, status?: number, detail?: string) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }

  /** Whether retrying the identical request could plausibly succeed without anything changing. */
  get isRetryable(): boolean {
    return this.kind === 'network' || this.kind === 'timeout' || this.kind === 'server';
  }
}

const STATUS_KINDS: ReadonlyMap<number, ApiErrorKind> = new Map([
  [400, 'validation'],
  [401, 'unauthorized'],
  [403, 'forbidden'],
  [404, 'not_found'],
  [409, 'conflict'],
  [422, 'validation'],
]);

export function kindForStatus(status: number): ApiErrorKind {
  return STATUS_KINDS.get(status) ?? (status >= 500 ? 'server' : 'unknown');
}

/**
 * The server's own sentence when it refused something, and the generic one otherwise.
 *
 * `userMessage` deliberately says "The request was not valid" for every 422, because most of
 * them are a client sending a shape the server would not accept and the person reading has
 * nothing to do about it. Some 422s are the opposite: the server refusing a credential GitHub
 * answered no about, or a repository a token cannot reach. Those carry a sentence written for
 * the person, naming the remedy — and replacing it with "the request was not valid" throws
 * away the only useful part of the answer.
 *
 * Narrowed to `validation` on purpose. A 500's detail is a fixed string the server substitutes
 * for whatever went wrong, and a `schema` error's is about this client, so neither is somebody
 * else's remedy to read.
 */
export function refusalMessage(error: ApiError): string {
  return error.kind === 'validation' && error.detail ? error.detail : userMessage(error);
}

/** A short sentence for a person. Technical detail stays on `detail` for an expandable section. */
export function userMessage(error: ApiError): string {
  switch (error.kind) {
    case 'network':
      return 'Could not reach the platform API.';
    case 'timeout':
      return 'The platform API did not respond in time.';
    case 'aborted':
      return 'The request was cancelled.';
    case 'unauthorized':
      return 'The platform API rejected the configured credentials.';
    case 'forbidden':
      return 'The platform API refused this request.';
    case 'not_found':
      // Not "no longer exists": a mistyped feature id is far more common than a deleted one,
      // and telling somebody their feature has vanished when they simply got the id wrong
      // sends them looking for an incident.
      return 'The platform has nothing by that identifier.';
    case 'validation':
      return 'The request was not valid.';
    case 'conflict':
      // Worth its own wording: the platform decides which transitions are legal, and saying so
      // is more useful than "something went wrong".
      return 'The platform refused this action in the feature’s current state.';
    case 'schema':
      return 'The platform API returned data this client does not understand.';
    case 'server':
      return 'The platform API failed while handling this request.';
    default:
      return 'Something went wrong.';
  }
}
