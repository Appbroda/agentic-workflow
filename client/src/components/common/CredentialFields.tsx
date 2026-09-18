import { useId } from 'react';
import type { ProviderCredentials } from '@/api/client';

export type Credentials = ProviderCredentials;

/**
 * The one sentence a queued live action shows instead of credential inputs.
 *
 * Queued actions -- resume, retry, publish, revise, design verdicts -- run on a worker that
 * resolves the feature owner's stored keys; the server discards any header the request
 * carried. Rendering inputs there collected credentials nothing would ever send, which is
 * what every one of these dialogs did until the store's arrival made the fields dead.
 */
export function StoredCredentialsNote() {
  return (
    <p className="muted">
      Runs with the credentials stored in Settings. Nothing to type here.
    </p>
  );
}

/**
 * The only place provider credentials are typed in.
 *
 * They are held in the calling component's state for exactly as long as the request that needs
 * them, sent as request headers, and never written to storage. The server treats them the same
 * way. Only the *synchronous* surfaces render this -- chat, contract-change decisions, repair
 * approvals -- because those are the requests whose work runs inside them; a queued action's
 * worker resolves stored keys and discards headers, so its dialogs show
 * `StoredCredentialsNote` instead. Even here the fields are an override: left blank, the
 * request falls back to the keys stored in Settings.
 */
export function CredentialFields({
  value,
  onChange,
  note = 'These are sent as request headers for this request only, and are not stored.',
  // Only what the request actually uses. Chat needs a model key and no repository access, and
  // asking for a GitHub token there would be collecting a credential nothing sends. A feature
  // needs exactly one model key -- its own platform's -- and never the other provider's.
  need = ['openai', 'github'],
}: {
  value: Credentials;
  onChange: (next: Credentials) => void;
  note?: string;
  need?: ('openai' | 'anthropic' | 'github')[];
}) {
  const openaiId = useId();
  const anthropicId = useId();
  const githubId = useId();

  return (
    <div className="callout">
      <p className="muted">{note}</p>
      {need.includes('openai') ? (
        <>
          <label className="field__label" htmlFor={openaiId}>
            OpenAI API key
          </label>
          <input
            id={openaiId}
            type="password"
            autoComplete="off"
            onChange={(event) => onChange({ ...value, openaiApiKey: event.target.value })}
          />
        </>
      ) : null}
      {need.includes('anthropic') ? (
        <>
          <label className="field__label" htmlFor={anthropicId}>
            Anthropic API key
          </label>
          <input
            id={anthropicId}
            type="password"
            autoComplete="off"
            onChange={(event) => onChange({ ...value, anthropicApiKey: event.target.value })}
          />
        </>
      ) : null}
      {need.includes('github') ? (
        <>
          <label className="field__label" htmlFor={githubId}>
            GitHub token
          </label>
          <input
            id={githubId}
            type="password"
            autoComplete="off"
            onChange={(event) => onChange({ ...value, githubToken: event.target.value })}
          />
        </>
      ) : null}
    </div>
  );
}
