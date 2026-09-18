/**
 * Server-sent events read with `fetch`, so they can carry an `Authorization` header.
 *
 * The browser's own `EventSource` cannot send headers. The usual way round that is to put the
 * token in the query string, which writes a credential into somewhere that gets logged by
 * every proxy in the path, kept in browser history and pasted into support tickets. So this
 * reads the same wire format off an ordinary authenticated response instead.
 *
 * What `EventSource` gives up in exchange is its automatic reconnect, which has to be written
 * here. That is no loss: its reconnect knows nothing about what the application missed while
 * it was away, and this one resumes from a cursor.
 */

/** One event off the wire, before anything has decided what it means. */
export interface StreamEvent {
  /** The `event:` name. Defaults to `message`, as the specification says. */
  event: string;
  /** The `data:` payload, already parsed. `{}` when a frame carried none. */
  data: unknown;
  /** The `id:` field, which is what a reconnect resumes from. */
  id: string | null;
}

export interface StreamOptions {
  signal?: AbortSignal;
  headers?: Record<string, string>;
  method?: 'GET' | 'POST';
  body?: unknown;
}

export class StreamError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'StreamError';
  }
}

/**
 * Open a stream and yield its events until it ends or the caller aborts.
 *
 * An async generator rather than a callback because ending a stream should be `break` or an
 * abort signal, not a subscription somebody has to remember to tear down. Leaving one running
 * after a component unmounted is how a page ends up applying events to a screen that is gone.
 */
export async function* readEventStream(
  url: string,
  options: StreamOptions = {},
): AsyncGenerator<StreamEvent, void, void> {
  const response = await fetch(url, {
    method: options.method ?? 'GET',
    headers: {
      Accept: 'text/event-stream',
      ...(options.body === undefined ? {} : { 'Content-Type': 'application/json' }),
      ...(options.headers ?? {}),
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  });

  if (!response.ok) {
    // A stream that failed before it started still has a status code, and it is the most
    // useful thing the caller will get -- 401 and 404 mean quite different things here.
    throw new StreamError(await describeFailure(response), response.status);
  }
  if (!response.body) {
    throw new StreamError('This browser cannot read a streamed response.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // `stream: true` matters: a multi-byte character can be split across two chunks, and
      // decoding each chunk independently turns it into a replacement character.
      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line. Anything after the last one is a partial frame
      // and stays in the buffer until the rest of it arrives.
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
        boundary = buffer.indexOf('\n\n');
      }
    }
  } finally {
    // Releasing the lock lets the connection be torn down when the caller stops reading.
    // Without it an abandoned stream holds the response open until the tab closes.
    reader.releaseLock();
    await response.body.cancel().catch(() => undefined);
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = 'message';
  let id: string | null = null;
  const data: string[] = [];

  for (const line of frame.split('\n')) {
    // A line beginning with a colon is a comment. The server sends one as a keep-alive, so
    // that a proxy does not drop a connection to a feature that is simply quiet.
    if (line.startsWith(':') || !line.trim()) continue;
    const separator = line.indexOf(':');
    const field = separator === -1 ? line : line.slice(0, separator);
    const value = separator === -1 ? '' : line.slice(separator + 1).replace(/^ /, '');
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
    else if (field === 'id') id = value;
  }

  if (data.length === 0 && event === 'message') return null;
  const raw = data.join('\n');
  let parsed: unknown = {};
  if (raw) {
    try {
      parsed = JSON.parse(raw);
    } catch {
      // The server sends JSON. Something that is not is a malformed frame, and passing the
      // raw text on lets a caller report that rather than crashing on a property access.
      parsed = { raw };
    }
  }
  return { event, data: parsed, id };
}

async function describeFailure(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => undefined);
  if (typeof body === 'object' && body !== null && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string') return detail;
  }
  return response.statusText || `Request failed with status ${response.status}`;
}
