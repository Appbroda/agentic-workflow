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
