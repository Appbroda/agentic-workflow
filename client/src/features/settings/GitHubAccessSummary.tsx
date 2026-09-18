import { Badge } from '@/components/ui/Badge';
import type { GitHubAccess } from '@/schemas/feature';

/**
 * What GitHub said a token reaches, shown at the moment it was saved.
 *
 * A token that authenticates and can push to nothing used to be indistinguishable from a good
 * one until a run died at push time. GitHub answers both questions in the same breath, so the
 * second answer is put where the first one lands: on the save.
 *
 * The counts are the finding; the advisories are the server's sentences and are not repeated
 * here for the reason `CredentialCheckResult` gives — the distinction between "GitHub said no"
 * and "GitHub said something worth knowing" belongs to one authority, and this only renders it.
 * A token GitHub answered no about never reaches this component at all: it was refused, and
 * the form shows the refusal as an error instead.
 */
export function GitHubAccessSummary({ access }: { access: GitHubAccess }) {
  return (
    <div className="stack stack--tight" role="status">
      <span>
        <Badge tone={access.writable_count > 0 ? 'done' : 'neutral'}>
          {summary(access)}
        </Badge>
      </span>
      {access.advisories.map((advisory) => (
        <p className="muted" key={advisory}>
          {advisory}
        </p>
      ))}
    </div>
  );
}

/**
 * One sentence for what the token reaches.
 *
 * The writable count leads, because it is the number that decides whether anything can be
 * built. A listing that never happened says so rather than reporting zero of anything: those
 * are different answers, and only one of them is about the token.
 */
function summary(access: GitHubAccess): string {
  if (!access.repositories_listed) return 'GitHub accepted it; its repositories were not listed';
  const writable = `${access.writable_count} writable`;
  const of = `of ${access.repository_count} repositor${access.repository_count === 1 ? 'y' : 'ies'}`;
  return `GitHub accepted it — ${writable} ${of}${access.truncated ? ' (listing truncated)' : ''}`;
}
