/**
 * How a model identifier is written for a person to read. One place, deliberately.
 *
 * This is presentation and nothing else. *Which* model ran, and whether one ran at all, is the
 * server's answer and arrives on the execution record; this only decides that `gpt-5.6-sol` is
 * written `GPT-5.6 Sol` rather than shouted as a slug halfway along an arrow. Scattering that
 * transformation across components is how two views come to spell one model differently, so
 * every caller uses this and the raw identifier stays available in technical details.
 *
 * Nothing here knows any model's name. The rules are about the *shape* of an identifier -- a
 * vendor prefix, a version, then variant words -- so a model this platform has never heard of
 * is still written sensibly, and a segment that fits no rule is passed through untouched rather
 * than guessed at. One consequence is worth stating: the rule is applied uniformly, so
 * `gpt-5.3-codex` is written `GPT-5.3 Codex` in the same way `gpt-5.6-sol` is written
 * `GPT-5.6 Sol`. Special-casing individual product names would mean holding a table of vendors'
 * marketing, which would be wrong the week after it was written.
 */

/** Leading segments that are acronyms rather than words, and how they are capitalised. */
const ACRONYMS: Record<string, string> = {
  ai: 'AI',
  gpt: 'GPT',
  llm: 'LLM',
};

/** Suffix words that are size or channel markers and are conventionally lower case. */
const KEPT_LOWER = new Set(['mini', 'nano', 'turbo', 'preview', 'latest']);

const HAS_DIGIT = /\d/;

/**
 * One hyphen-separated segment, written the way its own shape suggests.
 *
 * A segment containing a digit is left exactly as it is: the version in the identifier is what
 * the deployment configured, and `5.6` must not become `5.6.0`, `V5.6` or `5.6 `.
 */
function segment(part: string, index: number): string {
  const lower = part.toLowerCase();
  if (index === 0 && ACRONYMS[lower]) return ACRONYMS[lower];
  if (HAS_DIGIT.test(part)) return part;
  if (KEPT_LOWER.has(lower)) return lower;
  // Already mixed case: written that way on purpose, so it is left alone.
  if (part !== lower) return part;
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

/**
 * Write a backend-supplied model identifier for display.
 *
 * The vendor prefix stays joined to its version, because they are one token in every
 * identifier this has to render -- `GPT-5.6`, `GPT-4o`. Everything after becomes words.
 */
export function modelDisplayName(model: string | null | undefined): string | null {
  if (!model) return null;
  const trimmed = model.trim();
  if (!trimmed) return null;
  // A namespace-qualified identifier keeps its qualifier for the technical detail and shows
  // the model itself here.
  const tail = trimmed.split('/').at(-1) ?? trimmed;
  const parts = tail.split('-').filter(Boolean);
  if (parts.length === 0) return trimmed;
  const written = parts.map(segment);
  const [first, second] = written;
  if (written.length === 1 || second === undefined) return first ?? trimmed;
  // `gpt` + `5.6` is one token; `claude` + `opus` is two words.
  const prefixIsVersioned = !HAS_DIGIT.test(parts[0] ?? '') && HAS_DIGIT.test(parts[1] ?? '');
  const head = prefixIsVersioned ? `${first}-${second}` : (first ?? trimmed);
  const rest = written.slice(prefixIsVersioned ? 2 : 1);
  return [head, ...rest].join(' ');
}

/**
 * The compact handler label an edge carries: the model and, where one was asked for, the
 * effort. Never more than that -- everything else belongs in the drawer.
 *
 * The effort is written as `effort: high`, never as a bare word: `high` beside a model name
 * reads as a performance tier to anyone who knows the pinning rules, and a medium-tier
 * feature must not look like it violated them.
 */
export function modelLabel(
  model: string | null | undefined,
  reasoningEffort?: string | null,
): string | null {
  const name = modelDisplayName(model);
  if (!name) return null;
  return reasoningEffort ? `${name} · effort: ${reasoningEffort}` : name;
}

/**
 * How a provider identifier is written.
 *
 * Brand casing, not acronym casing: `openai` is written `OpenAI` because that is the vendor's
 * own spelling, and a general sentence-caser turns it into "Openai". Anything not listed is
 * capitalised, so a provider this platform has never seen is still written sensibly.
 */
const PROVIDER_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Google',
  azure: 'Azure',
};

export function providerDisplayName(provider: string | null | undefined): string | null {
  if (!provider) return null;
  const key = provider.trim().toLowerCase();
  if (!key) return null;
  return PROVIDER_NAMES[key] ?? key.charAt(0).toUpperCase() + key.slice(1);
}

/**
 * What a feature's performance tier is called where somebody reads it back.
 *
 * The same names the submission form offers, spelled here once so a feature's cost basis is
 * written the same way everywhere it appears. An unknown value is shown as itself rather
 * than mapped to a tier it might not be.
 */
const TIER_NAMES: Record<string, string> = {
  low: 'Economy',
  medium: 'Standard',
  high: 'Max',
  ultra: 'Ultra',
  // A user-authored model setup rather than an environment-backed preset. The feature's
  // Overview names the setup itself beside this word.
  custom: 'Custom',
};

export function performanceTierName(tier: string | null | undefined): string | null {
  if (!tier) return null;
  const key = tier.trim().toLowerCase();
  if (!key) return null;
  return TIER_NAMES[key] ?? key;
}

/**
 * What each model role is called where a person reads it, and the order the work happens in.
 *
 * The server's role names are the authority — an unlisted role is written as itself rather
 * than dropped, because a role added next month must still appear wherever roles are listed.
 * Spelled here once so the submission form and the settings table cannot come to call one
 * role two different things.
 */
export const MODEL_ROLE_LABELS: Record<string, string> = {
  reasoning: 'Reasoning',
  coding: 'Coding',
  review: 'Review',
  scoped_fix: 'Scoped fix',
};

export const MODEL_ROLE_ORDER = ['reasoning', 'coding', 'review', 'scoped_fix'] as const;

/**
 * Known roles first, in the order the work happens; anything the platform serves beyond them
 * follows, as itself. Never filters: a role this file has never heard of still appears.
 */
export function orderedModelRoles(roles: string[]): string[] {
  const known = MODEL_ROLE_ORDER.filter((role) => roles.includes(role));
  const unknown = roles.filter(
    (role) => !(MODEL_ROLE_ORDER as readonly string[]).includes(role),
  );
  return [...known, ...unknown];
}

/**
 * The model credential one feature's platform uses.
 *
 * Exactly one, never both: a feature submitted on Claude that was asked for an OpenAI key
 * would be collecting a credential none of its agents would ever reach for, and a feature
 * refused for a missing one would be refused over a key it does not use. An unknown platform
 * reads as `openai`, which is what every feature ran on before the choice existed.
 */
export function modelProviderFor(agentPlatform: string | undefined): 'openai' | 'anthropic' {
  return agentPlatform === 'anthropic' ? 'anthropic' : 'openai';
}
