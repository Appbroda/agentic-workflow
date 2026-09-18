import { z } from 'zod';
import type { StartFeatureInput } from '@/api/features';

/**
 * How prose points at an attached image, and what a marker may be called.
 *
 * The same two rules the server states in `server/api/prd_markers.py`, mirrored here for the
 * reason the repository-URL rules are mirrored: be told which field is wrong instead of
 * reading a 422 that names a JSON path. One place in this file, read by the field validator,
 * the slugger, the reference writer and the rename rewriter, so the four cannot drift from
 * each other or from the server.
 */
export const MARKER_PATTERN = /^[a-z0-9][a-z0-9-]{0,39}$/;

/** The reference syntax, as a matcher. `g` because renaming rewrites every occurrence. */
export const markerReferences = (marker: string) =>
  new RegExp(`\\[image:${marker}\\]`, 'g');

/** Every marker referenced anywhere in one piece of prose. */
export const MARKER_REFERENCE = /\[image:([a-z0-9][a-z0-9-]{0,39})\]/g;

/** The text that references one marker, so nothing builds the string by hand. */
export function imageReference(marker: string): string {
  return `[image:${marker}]`;
}

/**
 * Turn a filename into a candidate marker: lowercase, hyphenated, bounded.
 *
 * A starting point somebody edits, never a decision. An empty result — a file called `.png`,
 * or one named entirely in a script this slug rule cannot represent — falls back to a
 * positional name rather than to an invalid marker.
 */
export function markerFromFilename(filename: string, position: number): string {
  const stem = filename.replace(/\.[^.]+$/, '');
  const slug = stem
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40)
    .replace(/-+$/, '');
  return MARKER_PATTERN.test(slug) ? slug : `image-${position + 1}`;
}

/**
 * The submission form.
 *
 * What it requires is what `PRDSubmission` and `RepositorySpec` actually require: a title, a
 * problem statement, and at least one repository URL. Everything else is optional, because
 * everything else is work the platform exists to do -- the product-manager agent derives
 * goals, stories and requirements from the problem statement, and a repository's identity,
 * name and role are derived from its URL by the server.
 *
 * What is still validated here is validated because the server validates it too: a URL must
 * not carry credentials, and the same repository cannot be selected twice. Checking it here
 * means being told which field is wrong instead of reading a 422 that names a JSON path.
 *
 * Nothing here asks for an identifier any more. The repository id, display name and role are
 * all derived by the server from the URL, and the feature’s own id is allocated by the
 * server on submission — so there is no field for either.
 */

const nonEmpty = z.string().trim().min(1, 'Required');
const repositoryUrl = nonEmpty.url('Must be an HTTP(S) URL').superRefine((value, context) => {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    // `.url()` already owns the useful field error. Refinements still run after that failure.
    return;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    context.addIssue({ code: z.ZodIssueCode.custom, message: 'Must be an HTTP(S) URL' });
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'Do not put credentials, query parameters, or fragments in a repository URL',
    });
  }
});

const acceptanceCriteria = z
  .array(z.object({ value: nonEmpty }))
  .min(1, 'Add at least one acceptance criterion');

/**
 * What one pasted Figma URL says: which file, and which frames within it.
 *
 * Mirrors the server's `parse_design_url` for the reason the repository rules are mirrored:
 * being told which field is wrong beats reading a 422 that names a JSON path. The server is
 * still the authority — it re-parses every citation and refuses the same shapes — and this
 * derivation is never sent: only the URL the person pasted travels, and the server derives
 * the file key and node ids itself, so the two can never disagree about which frame was meant.
 */
export function parseDesignUrl(value: string): { fileKey: string; nodeIds: string[] } | null {
  let parsed: URL;
  try {
    parsed = new URL(value.trim());
  } catch {
    return null;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  if (parsed.username || parsed.password) return null;
  const host = parsed.hostname.toLowerCase();
  if (host !== 'figma.com' && !host.endsWith('.figma.com')) return null;
  const segments = parsed.pathname.split('/').filter(Boolean);
  // `/proto/`, `/board/` and `/slides/` carry a file key too and are deliberately excluded:
  // the platform reads design nodes, and a FigJam board would resolve to confident nonsense.
  if (segments.length < 2 || (segments[0] !== 'file' && segments[0] !== 'design')) return null;
  const fileKey = segments[1]!;
  if (!/^[A-Za-z0-9]{6,128}$/.test(fileKey)) return null;
  // A branch link (`/design/<parentKey>/branch/<branchKey>/<Name>`) carries the *parent* key
  // at segment 1, so reading it as a citation would snapshot the wrong file. Refused here the
  // way the server refuses it; four segments minimum, because a file literally named "branch"
  // produces `/design/<key>/branch` with no branch key and is an ordinary citation.
  if (segments.length >= 4 && segments[2] === 'branch') return null;
  // `1-23` is how a browser writes the id the API spells `1:23`; a node id contains no other
  // dash, so one replacement translates both forms and leaves an instance path intact.
  const nodeIds = (parsed.searchParams.getAll('node-id') ?? [])
    .flatMap((raw) => raw.split(','))
    .map((raw) => raw.trim().replace(/-/g, ':'))
    .filter((id) => id.length > 0);
  if (nodeIds.some((id) => !/^I?\d+:\d+(;\d+:\d+)*$/.test(id))) return null;
  return { fileKey, nodeIds: [...new Set(nodeIds)] };
}

/** The identity of one citation, for spotting the same frame cited twice. */
export function designReferenceKey(url: string): string | null {
  const parsed = parseDesignUrl(url);
  if (!parsed) return null;
  return parsed.nodeIds.length === 0
    ? `${parsed.fileKey}#whole-file`
    : `${parsed.fileKey}#${parsed.nodeIds.join(',')}`;
}

/** Whether a link points at a Figma branch, whose citation would resolve the parent file. */
export function isFigmaBranchUrl(value: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(value.trim());
  } catch {
    return false;
  }
  const host = parsed.hostname.toLowerCase();
  if (host !== 'figma.com' && !host.endsWith('.figma.com')) return false;
  const segments = parsed.pathname.split('/').filter(Boolean);
  if (segments.length < 4 || (segments[0] !== 'file' && segments[0] !== 'design')) return false;
  return segments[2] === 'branch';
}

const designUrl = nonEmpty.superRefine((value, context) => {
  // Named first, because "paste a design link" is wrong advice for somebody who did paste
  // one — of a branch. Mirrors the server's refusal, which is still the authority.
  if (isFigmaBranchUrl(value)) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message:
        'This is a branch link; branches are not supported yet. Paste a link to the main ' +
        'file, or merge the branch first.',
    });
    return;
  }
  if (parseDesignUrl(value) === null) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message:
        'Paste a Figma design link — figma.com/design/<key>/… (or /file/<key>/… from before ' +
        'the rename). A prototype, FigJam or Slides link is a different kind of document, and ' +
        'a link must not carry credentials.',
    });
  }
});

export const newFeatureSchema = z
  .object({
    execution_mode: z.enum(['mock', 'live']),
    agent_platform: z.enum(['openai', 'anthropic']),
    performance_tier: z.enum(['low', 'medium', 'high', 'ultra']),
    /**
     * A user-authored setup to run on instead of the (platform, tier) pairing. Empty means
     * none. When set, the submission carries `model_setup_id` alone: the server refuses a
     * request naming both a setup and an explicit pairing — two answers to one question.
     */
    model_setup_id: z.string(),
    title: nonEmpty,
    problem_statement: nonEmpty,
    // Optional throughout. An entry that exists is completed; an absent section is derived.
    goals: z.array(z.object({ value: nonEmpty })),
    user_stories: z.array(
      z.object({
        story_id: nonEmpty,
        persona: nonEmpty,
        need: nonEmpty,
        benefit: nonEmpty,
        acceptance_criteria: acceptanceCriteria,
      }),
    ),
    requirements: z.array(
      z.object({
        requirement_id: nonEmpty,
        description: nonEmpty,
        priority: z.enum(['must', 'should', 'could', 'wont']),
        acceptance_criteria: acceptanceCriteria,
      }),
    ),
    constraints: z.array(z.object({ value: z.string() })),
    out_of_scope: z.array(z.object({ value: z.string() })),
    stakeholders: z.array(z.object({ value: z.string() })),
    /**
     * Images already uploaded, and the marker each one answers to.
     *
     * Uploaded before submission rather than with it: an image is uploaded the moment it is
     * dropped, so somebody sees a thumbnail immediately and a five-megabyte file is not
     * re-sent every time the form is re-validated. Each entry therefore already has a
     * server-allocated id; the marker and the caption are the parts the form owns.
     */
    attachments: z.array(
      z.object({
        attachment_id: nonEmpty,
        marker: z
          .string()
          .regex(MARKER_PATTERN, 'Lowercase letters, digits and hyphens; up to 40 characters'),
        caption: z.string().max(500, 'Keep a caption to 500 characters'),
        /** Display only, never sent: the server has them and they are shown back from it. */
        filename: z.string(),
        byte_size: z.number(),
        media_type: z.string(),
      }),
    ),
    /**
     * The designs this request cites. Optional like everything else on this form, and the
     * key is omitted from the request entirely when the list is empty (see
     * `toStartFeatureInput`) so a citation-free submission sends exactly what it always did.
     */
    design_references: z.array(z.object({ url: designUrl, label: z.string() })),
    repositories: z
      .array(
        z.object({
          repository_url: repositoryUrl,
          default_branch: nonEmpty,
          required: z.boolean(),
          /**
           * Display only, never sent. A selected repository is shown as the name and label the
           * person saved it under; the server derives the name and the stable identity from
           * the URL, and the saved label is deliberately not sent as a `role` — a label
           * somebody typed must not be able to change how the planner orders the work.
           */
          label: z.string(),
          repository_type: z.string(),
          /** Which saved configuration this came from, or empty for a one-time entry. */
          configuration_id: z.string(),
        }),
      )
      .min(1, 'Select at least one repository')
      .refine((items) => items.some((item) => item.required), {
        message: 'At least one repository must be required',
      }),
  })
  .superRefine((values, context) => {
    addDuplicateIssues(
      values.user_stories.map((item) => item.story_id),
      'User story ID',
      ['user_stories'],
      'story_id',
      context,
    );
    addDuplicateIssues(
      values.requirements.map((item) => item.requirement_id),
      'Requirement ID',
      ['requirements'],
      'requirement_id',
      context,
    );
    // The same repository twice is a mistake whichever way it was selected.
    addDuplicateIssues(
      values.repositories.map((item) => item.repository_url.trim().replace(/\.git$/, '')),
      'Repository',
      ['repositories'],
      'repository_url',
      context,
      { ignoreBlank: true, message: 'This repository is already listed' },
    );
    // Two images answering to one marker means a reference points at neither. Checked here
    // as well as on the server, so it is said next to the field somebody is editing.
    addDuplicateIssues(
      values.attachments.map((item) => item.marker),
      'Marker',
      ['attachments'],
      'marker',
      context,
      { ignoreBlank: true, message: 'This marker is already used by another image' },
    );
    // A reference the prose makes and no image answers is what the server rejects, so the
    // form says so first -- and says which one, because "a reference is wrong" is not
    // actionable in a page of prose.
    const declared = new Set(values.attachments.map((item) => item.marker));
    const unknown = [...new Set(referencedMarkers(values))].filter(
      (marker) => !declared.has(marker),
    );
    if (unknown.length > 0) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['attachments'],
        message: `No image is attached for ${unknown
          .map((marker) => imageReference(marker))
          .join(', ')}`,
      });
    }
    // The same frame twice is ambiguous, and the ambiguity would be resolved silently: one
    // label would win and the other would vanish from the snapshot the engineer is judged
    // against. Compared on the derived identity rather than the text, so two spellings of one
    // link — `node-id=1-23` and `node-id=1%3A23` — are caught as the duplicate they are.
    addDuplicateIssues(
      values.design_references.map((item) => designReferenceKey(item.url) ?? ''),
      'Design',
      ['design_references'],
      'url',
      context,
      { ignoreBlank: true, message: 'This design is already cited' },
    );
  });

/**
 * Every marker the form's own text refers to, across every field a person writes in.
 *
 * The same set of fields the server scans (`PRDSubmission.prose`), because a reference
 * written in a requirement must not be silently literal text.
 */
export function referencedMarkers(values: NewFeatureValues): string[] {
  const texts = [
    values.title,
    values.problem_statement,
    ...values.goals.map((item) => item.value),
    ...values.constraints.map((item) => item.value),
    ...values.out_of_scope.map((item) => item.value),
    ...values.stakeholders.map((item) => item.value),
    ...values.user_stories.flatMap((story) => [
      story.persona,
      story.need,
      story.benefit,
      ...story.acceptance_criteria.map((item) => item.value),
    ]),
    ...values.requirements.flatMap((requirement) => [
      requirement.description,
      ...requirement.acceptance_criteria.map((item) => item.value),
    ]),
  ];
  return texts.flatMap((text) => [...text.matchAll(MARKER_REFERENCE)].map((match) => match[1]!));
}

function addDuplicateIssues(
  identifiers: string[],
  label: string,
  parent: string[],
  field: string,
  context: z.RefinementCtx,
  options: { ignoreBlank?: boolean; message?: string } = {},
) {
  const first = new Map<string, number>();
  identifiers.forEach((identifier, index) => {
    if (options.ignoreBlank && !identifier) return;
    const previous = first.get(identifier);
    if (previous === undefined) {
      first.set(identifier, index);
      return;
    }
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: options.message ?? `${label} must be unique`,
      path: [...parent, index, field],
    });
  });
}

export type NewFeatureValues = z.infer<typeof newFeatureSchema>;

export type NewRepositoryValues = NewFeatureValues['repositories'][number];

/**
 * A blank one-time repository row.
 *
 * `master` rather than `main`: it is what the repositories this platform is pointed at
 * actually use, and a default that is wrong for every one of them is not a default.
 */
export const EMPTY_REPOSITORY: NewRepositoryValues = {
  repository_url: '',
  default_branch: 'master',
  required: true,
  label: '',
  repository_type: '',
  configuration_id: '',
};

/**
 * Platforms temporarily withheld from new submissions. Their pairings render disabled in the
 * form with "(Not ready)" and a selection on one refuses to submit — never hidden, never
 * substituted, the same posture an unconfigured pairing gets. A product gate, not deployment
 * configuration: the pairings stay configured on the server and existing features keep
 * running. Remove an entry to re-open the platform; the first-visit default below and the
 * remembered-provider preselection both respect this set.
 */
export const PLATFORMS_NOT_READY: ReadonlySet<string> = new Set(['anthropic']);

export const DEFAULT_VALUES: NewFeatureValues = {
  // Live, because that is what somebody opening this page came to do: mock plans and reviews
  // a feature without touching anything, which is useful for trying the shape of one and is
  // not what a real submission wants. The mode sits on the form rather than behind the
  // advanced disclosure precisely because this default has consequences.
  execution_mode: 'live',
  // OpenAI while Claude sits behind the PLATFORMS_NOT_READY gate above: a first visit must
  // not open on a selection the form itself refuses. The form prefers the provider the
  // operator last used (see `preferredAgentPlatform`); this is the answer for a first
  // visit. If the resulting selection is unconfigured on this deployment the form says so
  // and refuses to submit -- it never re-aims the selection itself, which is what sent
  // AB-Feature-173 to the wrong platform. The choice is fixed for the feature's life,
  // which is why it sits on the form rather than behind the disclosure.
  agent_platform: 'openai',
  // Standard. The server's own default is `high` for API compatibility -- a caller that
  // never heard of tiers gets exactly today's behaviour -- so an untouched form must say
  // `medium` explicitly: the interface's recommendation is a deliberate statement, not an
  // inherited one.
  performance_tier: 'medium',
  // No setup selected. A setup is a deliberate per-visit choice, like the tier: it is never
  // remembered across visits, for the reason the tier is not.
  model_setup_id: '',
  title: '',
  problem_statement: '',
  goals: [],
  user_stories: [],
  requirements: [],
  constraints: [],
  out_of_scope: [],
  stakeholders: [],
  // Empty rather than one blank row: nothing about a feature requires a design, and a blank
  // row would read as a required field somebody has not filled in.
  design_references: [],
  // Empty rather than one blank row: repositories are chosen from the ones already saved,
  // and a blank row would read as a required field the person has not filled in.
  repositories: [],
  attachments: [],
};

/**
 * Which provider the operator last submitted on, so the form preselects Standard on it.
 *
 * Only the provider is remembered. The tier deliberately is not: Standard is the interface's
 * recommendation for every new feature, and remembering that somebody once picked Max would
 * quietly turn one expensive choice into a standing one.
 */
const LAST_PLATFORM_KEY = 'newFeature.lastAgentPlatform';

export function preferredAgentPlatform(): NewFeatureValues['agent_platform'] {
  try {
    const stored = window.localStorage.getItem(LAST_PLATFORM_KEY);
    // A remembered provider that has since been gated falls back to the default rather than
    // opening the form on a selection it will refuse. The memory itself is kept: re-opening
    // the platform restores the preference without the person doing anything.
    if ((stored === 'openai' || stored === 'anthropic') && !PLATFORMS_NOT_READY.has(stored)) {
      return stored;
    }
  } catch {
    // Storage can be unavailable (private windows, embedded contexts); the default stands.
  }
  return DEFAULT_VALUES.agent_platform;
}

export function rememberAgentPlatform(platform: NewFeatureValues['agent_platform']): void {
  try {
    window.localStorage.setItem(LAST_PLATFORM_KEY, platform);
  } catch {
    // Remembering is a convenience, not a requirement.
  }
}

const values = (items: { value: string }[]) =>
  items.map((item) => item.value.trim()).filter((item) => item.length > 0);

/**
 * The name the server will derive for a repository URL, shown back while typing.
 *
 * A preview, not a decision: it is never sent. The server derives identity itself from the
 * same URL, and duplicating that derivation into the request would make the client's guess
 * authoritative the moment the two disagreed.
 */
export function derivedRepositoryName(url: string): string | null {
  let path: string;
  try {
    path = new URL(url.trim()).pathname;
  } catch {
    return null;
  }
  const segments = path.replace(/\.git$/, '').split('/').filter(Boolean);
  return segments.at(-1) ?? null;
}

/** Flatten the form's `{value}` wrappers, which exist only because field arrays need object items. */
export function toStartFeatureInput(form: NewFeatureValues): StartFeatureInput {
  // Only the URL and the label travel. The file key and the node ids are the server's own
  // derivation from the same URL, and sending this client's copy would make its guess
  // authoritative the moment the two disagreed -- `derivedRepositoryName` above is the
  // precedent for the same decision.
  const designReferences = form.design_references
    .map((item) => ({ url: item.url.trim(), label: item.label.trim() }))
    .filter((item) => item.url.length > 0);
  return {
    execution_mode: form.execution_mode,
    // Exactly one selection travels. A submission never carries both a setup id and an
    // explicit pairing: the server refuses two answers to one question, and this is where
    // the client keeps that from ever being asked.
    ...(form.model_setup_id
      ? { model_setup_id: form.model_setup_id }
      : {
          agent_platform: form.agent_platform,
          performance_tier: form.performance_tier,
        }),
    prd: {
      title: form.title,
      problem_statement: form.problem_statement,
      goals: values(form.goals),
      user_stories: form.user_stories.map((story) => ({
        story_id: story.story_id,
        persona: story.persona,
        need: story.need,
        benefit: story.benefit,
        acceptance_criteria: values(story.acceptance_criteria),
      })),
      requirements: form.requirements.map((requirement) => ({
        requirement_id: requirement.requirement_id,
        description: requirement.description,
        priority: requirement.priority,
        acceptance_criteria: values(requirement.acceptance_criteria),
        dependencies: [],
      })),
      constraints: values(form.constraints),
      out_of_scope: values(form.out_of_scope),
      stakeholders: values(form.stakeholders),
      // Omitted entirely when there are none, rather than sent as an empty array. The key
      // is conditional because the server's own default is the empty list and a submission
      // that never mentions images should read, in a request log and in a fingerprint,
      // exactly as it did before images existed. The model-selection spread above is the
      // pattern.
      ...(form.attachments.length > 0
        ? {
            attachments: form.attachments.map((item) => ({
              attachment_id: item.attachment_id,
              marker: item.marker,
              caption: item.caption.trim(),
            })),
          }
        : {}),
      // Omitted entirely when nothing was cited, which is a change: this function used to
      // emit every PRD key unconditionally, empty arrays included. The conditional spread is
      // the one the model selection above already uses, and the reason is stronger here --
      // with the key absent, the request a citation-free submission sends is byte-identical
      // to the one this form has always sent, which is what makes "a feature that cites no
      // design behaves exactly as it does today" a fact about the wire and not a hope.
      ...(designReferences.length > 0 ? { design_references: designReferences } : {}),
    },
    // Three fields. Identity, display name and role are all derived by the server from the
    // URL, and the saved label is left out on purpose so an organisational choice cannot
    // reach the planner as a role.
    repositories: form.repositories.map((repository) => ({
      repository_url: repository.repository_url,
      default_branch: repository.default_branch,
      required: repository.required,
    })),
  };
}
