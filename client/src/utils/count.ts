/**
 * A count with its noun, singular or plural.
 *
 * Used everywhere a panel says how many of something it is showing. One implementation so
 * "1 repositories" cannot appear in the twentieth place somebody wrote it by hand.
 */
export function count(n: number, singular: string, plural = `${singular}s`): string {
  return `${n} ${n === 1 ? singular : plural}`;
}
