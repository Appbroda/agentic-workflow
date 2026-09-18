import { describe, expect, it } from 'vitest';
import { reissueNote } from '@/features/feature-workspace/agent-work';

/**
 * The note for a call whose stream had to be asked again.
 *
 * A re-issue happens when the provider accepted the request and then said nothing inside the
 * first-event budget: the adapter closes that stream and issues the identical request again.
 * It costs one request, writes no journal row of its own, and the caller gets an ordinary
 * answer — so this note is the only place a person sees it happened.
 *
 * AB-Feature-215 is why it matters. Two calls went silent for the full 1800-second deadline
 * and one of them cost an attempt. The budget that now catches that in three minutes was
 * chosen by judgement, not measurement, so the thing an operator needs is the early warning:
 * "it re-issued once and then worked" is what says the budget is close to binding, long
 * before anything fails.
 */
describe('reissueNote', () => {
  it('says nothing at all for the healthy readings', () => {
    // Both render nothing, and they are *different* facts: `0` is a stream that spoke on the
    // first issue, `null` is a call nobody measured — a plain-POST transport, or a row older
    // than the field. A badge on every healthy row would be noise, so the distinction lives
    // in the data for whoever queries it rather than on the screen.
    expect(reissueNote(0)).toBeNull();
    expect(reissueNote(null)).toBeNull();
    // Defensive, not expected: a negative count is not a reading, and rendering
    // "re-issued -1 times" would be worse than saying nothing.
    expect(reissueNote(-1)).toBeNull();
  });

  it('counts in words at the small numbers the budget actually allows', () => {
    // Two is the ceiling `_MAX_STREAM_REISSUES` permits, so these are the only two counts a
    // single call can produce today. Spelled rather than numeric because they sit inline
    // beside "call 1" and a digit there reads as an identifier.
    expect(reissueNote(1)).toBe('re-issued once after a silent stream');
    expect(reissueNote(2)).toBe('re-issued twice after a silent stream');
  });

  it('falls back to a numeral rather than losing a count it did not expect', () => {
    // Reached only if the re-issue ceiling is raised, or if an attempt-level total is passed
    // in. Either way the number is the point, and a note that silently stopped counting past
    // two would hide exactly the escalation this exists to show.
    expect(reissueNote(3)).toBe('re-issued 3 times after a silent stream');
    expect(reissueNote(11)).toBe('re-issued 11 times after a silent stream');
  });
});
