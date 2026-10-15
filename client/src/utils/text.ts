/**
 * Acronyms the platform's identifiers contain, which sentence-casing would otherwise mangle.
 *
 * `prd` became "Prd" and `technical_prd` became "Technical prd" on every artifact heading and
 * tab. Restricted to words that really are acronyms here, so an ordinary word is never
 * shouted at somebody.
 */
const ACRONYMS: Record<string, string> = {
  prd: 'PRD',
  pr: 'PR',
  prs: 'PRs',
  api: 'API',
  url: 'URL',
  id: 'ID',
  sdk: 'SDK',
  sha: 'SHA',
  ui: 'UI',
};

/** Turn a backend identifier such as `integration_review` into `Integration review`. */
export function humanise(value: string): string {
  const words = value.replace(/[_-]/g, ' ').split(' ');
  return words
    .map((word, index) => {
      const acronym = ACRONYMS[word.toLowerCase()];
      if (acronym) return acronym;
      return index === 0 ? word.replace(/^./, (character) => character.toUpperCase()) : word;
    })
    .join(' ');
}

/** One clarification question's text, once it has been told apart into its two sides. */
export interface ContradictionQuestion {
  lede: string;
  sides: [string, string];
  ask: string;
}

/**
 * Split a contradiction question into its two sides, or return `null` for any other question.
 *
 * `contradiction_questions()` in `server/tools/requirement_reconciliation.py` composes this
 * question from one fixed template -- it is the only place that ever produces this exact
 * shape, and it always produces exactly two sides:
 *
 *   "These cannot both hold, so nothing is planned until you say which does. "
 *   f"First, {sides[0]}. Second, {sides[1]}. Which one holds, and what should "
 *   "the other one become?"
 *
 * Anchored on the connector phrases ("First, ", ". Second, ", ". Which one holds...") rather
 * than on the quote marks inside each side: a side's own content -- an answer, a requirement
 * description -- can itself legitimately contain a quoted word, but it is never going to
 * coincidentally contain this exact grammatical construction. If the server ever rewords the
 * template, this stops matching and the question falls back to rendering as plain text, which
 * is the same way an ordinary question (one this never matches at all) already renders.
 */
export function parseContradictionQuestion(text: string): ContradictionQuestion | null {
  const match =
    /^These cannot both hold, so nothing is planned until you say which does\. First, (.+?)\. Second, (.+?)\. (Which one holds, and what should the other one become\?)$/s.exec(
      text,
    );
  if (!match) return null;
  // All three groups are non-optional, so a match fills every one of them -- but they come
  // back off an array, which `noUncheckedIndexedAccess` widens to `string | undefined`
  // regardless of what the pattern guarantees.
  return {
    lede: 'These cannot both hold, so nothing is planned until you say which does.',
    sides: [match[1]!, match[2]!],
    ask: match[3]!,
  };
}

/** One side's own question and answer, once the quote marks bounding each have told them apart. */
export interface QuotedQuestionAndAnswer {
  question: string;
  answer: string;
}

/**
 * Pull a question and an answer out of one contradiction side, by position, never by wording.
 *
 * Every side `contradiction_questions()` composes quotes its question first and its answer
 * second -- `f'the answer to "{question}" says: "{answer}"'`, or, for a requirement's own
 * description, `f'requirement {id} says: "{description}" and is accepted on: "..."'`. The
 * English joining the two (`says:`, `and is accepted on:`) is free-form commentary and not a
 * stable anchor; the quote marks bounding each piece are the one thing the template guarantees.
 * The first two quoted spans are read as question, then answer; anything with fewer than two
 * returns `null` so the caller falls back to the side's own text rather than guessing at a
 * shape that is not there.
 */
export function quotedQuestionAndAnswer(side: string): QuotedQuestionAndAnswer | null {
  const matches = [...side.matchAll(/"([^"]*)"/g)];
  if (matches.length < 2) return null;
  return { question: matches[0]![1]!, answer: matches[1]![1]! };
}
