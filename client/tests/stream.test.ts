import { describe, expect, it, vi } from 'vitest';
import { readEventStream, StreamError } from '@/api/stream';

/**
 * Reading server-sent events off an authenticated response.
 *
 * The framing is the contract with the server, and the reason this exists at all is that the
 * browser's own `EventSource` cannot send an `Authorization` header. So these tests check the
 * two things that follow from that: that the headers go, and that the wire format is read
 * exactly rather than approximately.
 */

/** Serve a body as a stream, in the chunks a network would actually deliver it in. */
function respondWith(chunks: string[], init: ResponseInit = {}): typeof fetch {
  return vi.fn(async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    return new Response(body, { status: 200, ...init });
  }) as unknown as typeof fetch;
}

async function collect(url: string, options?: Parameters<typeof readEventStream>[1]) {
  const events = [];
  for await (const event of readEventStream(url, options)) events.push(event);
  return events;
}

describe('reading an event stream', () => {
  it('reads each frame once, whatever the chunk boundaries are', async () => {
    // A frame split across two network chunks is the normal case, not an edge case.
    vi.stubGlobal(
      'fetch',
      respondWith([
        'event: delta\ndata: {"text":"Hel',
        'lo"}\n\nevent: delta\ndata: {"text":" world"}\n\n',
      ]),
    );

    const events = await collect('/stream');

    expect(events.map((item) => item.event)).toEqual(['delta', 'delta']);
    expect(events.map((item) => (item.data as { text: string }).text)).toEqual([
      'Hello',
      ' world',
    ]);
  });

  it('carries the id a reconnect resumes from', async () => {
    vi.stubGlobal('fetch', respondWith(['id: 7\nevent: event\ndata: {"id":7}\n\n']));

    const events = await collect('/stream');

    expect(events[0]?.id).toBe('7');
  });

  it('ignores the keep-alive comments a proxy needs', async () => {
    // The server sends these so a connection to a quiet feature is not dropped. They are not
    // events and must not reach the application as one.
    vi.stubGlobal(
      'fetch',
      respondWith([': keep-alive\n\n', 'event: event\ndata: {"id":1}\n\n', ': keep-alive\n\n']),
    );

    const events = await collect('/stream');

    expect(events).toHaveLength(1);
  });

  it('sends the authorization header, which is why this exists', async () => {
    const fetchMock = respondWith(['event: done\ndata: {}\n\n']);
    vi.stubGlobal('fetch', fetchMock);

    await collect('/stream', { headers: { Authorization: 'Bearer platform-token' } });

    const init = (fetchMock as unknown as ReturnType<typeof vi.fn>).mock.calls[0]?.[1];
    expect((init as RequestInit).headers).toMatchObject({
      Authorization: 'Bearer platform-token',
    });
  });

  it('never puts anything in the URL', async () => {
    const fetchMock = respondWith(['event: done\ndata: {}\n\n']);
    vi.stubGlobal('fetch', fetchMock);

    await collect('/features/f-1/events/stream?after=4', {
      headers: { Authorization: 'Bearer platform-token' },
    });

    const url = (fetchMock as unknown as ReturnType<typeof vi.fn>).mock.calls[0]?.[0];
    // The cursor belongs in the query string; the credential does not, and putting it there
    // is exactly what using `fetch` instead of `EventSource` avoids.
    expect(url).toBe('/features/f-1/events/stream?after=4');
    expect(String(url)).not.toContain('platform-token');
  });

  it('reports a refusal with the status it was refused with', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(JSON.stringify({ detail: 'Invalid authentication credentials.' }), {
            status: 401,
          }),
      ) as unknown as typeof fetch,
    );

    await expect(collect('/stream')).rejects.toBeInstanceOf(StreamError);
    await expect(collect('/stream')).rejects.toMatchObject({ status: 401 });
  });

  it('passes on a malformed frame rather than throwing on it', async () => {
    // A caller can report a stream that went wrong; it cannot recover from a crash inside
    // the reader.
    vi.stubGlobal('fetch', respondWith(['event: delta\ndata: not json\n\n']));

    const events = await collect('/stream');

    expect(events[0]?.event).toBe('delta');
    expect(events[0]?.data).toEqual({ raw: 'not json' });
  });

  it('stops reading when the caller aborts', async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit) => {
        if (init.signal?.aborted) throw new DOMException('aborted', 'AbortError');
        return new Response(new ReadableStream(), { status: 200 });
      }) as unknown as typeof fetch,
    );
    controller.abort();

    await expect(collect('/stream', { signal: controller.signal })).rejects.toThrow();
  });
});
