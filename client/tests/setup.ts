import '@testing-library/jest-dom/vitest';

/**
 * jsdom@25 implements neither `URL.createObjectURL` nor `URL.revokeObjectURL`, so without this
 * polyfill every preview test silently exercised the *failure* branch: the query function threw
 * on the missing API, the fallback copy rendered, and an assertion on that copy passed while
 * the success path had never run anywhere.
 *
 * The polyfill records what it minted and what was revoked, so a test can assert the lifecycle
 * — an `<img>` actually rendered with a `blob:` src, and unmounting revoked it — rather than
 * only the absence of a crash.
 */
export const objectUrlRegistry = {
  created: [] as string[],
  revoked: [] as string[],
  reset(): void {
    this.created.length = 0;
    this.revoked.length = 0;
  },
};

let objectUrlSequence = 0;

if (typeof URL.createObjectURL !== 'function') {
  URL.createObjectURL = (_source: Blob | MediaSource): string => {
    objectUrlSequence += 1;
    const url = `blob:test/${objectUrlSequence}`;
    objectUrlRegistry.created.push(url);
    return url;
  };
}

if (typeof URL.revokeObjectURL !== 'function') {
  URL.revokeObjectURL = (url: string): void => {
    objectUrlRegistry.revoked.push(url);
  };
}
