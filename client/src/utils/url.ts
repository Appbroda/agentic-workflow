/** Accept an external link only when it is plain HTTP(S) with no embedded credentials. */
export function safeHttpUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const parsed = new URL(value);
    return (parsed.protocol === 'http:' || parsed.protocol === 'https:')
      && !parsed.username
      && !parsed.password
      ? value
      : null;
  } catch {
    return null;
  }
}

/** Markdown may also link within this application, but never through a protocol-relative URL. */
export function safeMarkdownHref(value: string): boolean {
  if (value.startsWith('#')) return true;
  if (value.startsWith('/') && !value.startsWith('//') && !value.startsWith('/\\')) return true;
  return safeHttpUrl(value) !== null;
}
