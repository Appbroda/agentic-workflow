import { Badge } from '@/components/ui/Badge';
import type { CredentialCheck } from '@/schemas/feature';

/**
 * What a check said: whether this deployment can still open the key, and — when the provider
 * was asked — whether it still accepts it.
 *
 * Two answers rather than one, because they can disagree and the disagreement is the whole
 * point. A key can be perfectly readable and dead: run 190 lost two features to exactly that,
 * with a `usable` verdict on screen the entire time. So a refusal is shown as a refusal even
 * though `usable` is true beside it, and a provider that could not be reached is shown as
 * having said nothing — never as reassurance, which is the failure that let a dead credential
 * stay invisible for a day.
 *
 * The sentences are the server's. It owns the distinction, and two copies of that wording
 * would be one copy too many. Shared by the credentials panel and the Slack connection test,
 * which answer in the same verdict shape on purpose.
 */
export function CredentialCheckResult({ result }: { result: CredentialCheck }) {
  const verdict = result.verified ?? null;
  const refused = verdict === 'refused';
  return (
    <div className="stack stack--tight" role="status">
      {verdict ? (
        <span>
          <Badge tone={VERDICT_TONES[verdict]}>{VERDICT_LABELS[verdict]}</Badge>
        </span>
      ) : null}
      <p className={result.usable && !refused ? 'muted' : 'field__error'}>{result.detail}</p>
    </div>
  );
}

/**
 * The provider's verdict, in the tone vocabulary the rest of the application uses.
 *
 * `unknown` is neutral on purpose. It is the absence of an answer, so painting it as a
 * problem would cry wolf every time a provider was slow, and painting it as `done` would be
 * the lie this whole verdict exists to stop telling. Its word carries the meaning.
 */
const VERDICT_TONES: Record<'accepted' | 'refused' | 'unknown', 'done' | 'stopped' | 'neutral'> = {
  accepted: 'done',
  refused: 'stopped',
  unknown: 'neutral',
};

const VERDICT_LABELS: Record<'accepted' | 'refused' | 'unknown', string> = {
  accepted: 'Provider accepted it',
  refused: 'Provider refused it',
  unknown: 'Provider did not answer',
};
