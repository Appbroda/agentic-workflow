import { useMemo, useRef, useState } from 'react';
import { useForm, useFieldArray, type Control, type UseFormRegister } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import { PageHeader, Panel } from '@/components/ui/Layout';
import { Badge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import {
  MODEL_ROLE_LABELS,
  modelDisplayName,
  orderedModelRoles,
  providerDisplayName,
} from '@/utils/model';
import { ErrorState, TableSkeleton } from '@/components/common/States';
import { IconPlus, IconSettings } from '@/components/ui/icons';
import type {
  AgentPlatformOption,
  SavedModelSetup,
  SavedRepository,
  SetupState,
} from '@/schemas/feature';
import { AttachmentsField } from './AttachmentsField';
import { Field, Fieldset, RemoveButton } from './fields';
import { StringListField } from './StringListField';
import {
  DEFAULT_VALUES,
  EMPTY_REPOSITORY,
  PLATFORMS_NOT_READY,
  derivedRepositoryName,
  newFeatureSchema,
  preferredAgentPlatform,
  rememberAgentPlatform,
  toStartFeatureInput,
  type NewFeatureValues,
} from './schema';

/**
 * Asking for a feature.
 *
 * Two questions once the prerequisites are in place: what is the problem, and which of your
 * repositories does it touch. The repositories are chosen from the ones already saved rather
 * than typed again, and the identifiers this form used to ask for — a feature id, and a
 * repository id, name and role per repository — are all the server's to derive.
 *
 * Before the form there are two gates, in this order: the provider keys the platform will use
 * on your behalf, then at least one saved repository. Both are real prerequisites rather than
 * validation. The work happens after this request has been answered, so it reads the stored
 * keys — there is no header left for it to use.
 */

/** `FileReader` rather than `Blob.text()`: the latter is not implemented everywhere. */
function readTextFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
    reader.onerror = () => reject(reader.error ?? new Error('could not read file'));
    reader.readAsText(file);
  });
}

export function NewFeatureForm() {
  const api = useApi();
  const setup = useQuery({
    queryKey: ['setup'],
    queryFn: ({ signal }) => api.getSetupState(signal),
    retry: false,
  });
  const saved = useQuery({
    queryKey: ['saved-repositories'],
    queryFn: ({ signal }) => api.listSavedRepositories(signal),
    retry: false,
  });
  // The caller's own model setups, offered beside the deployment's pairings. Best-effort: a
  // viewer is refused the list and a deployment may persist none, and neither takes the form
  // down — the pairings are still there.
  const ownSetups = useQuery({
    queryKey: ['model-setups'],
    queryFn: ({ signal }) => api.listModelSetups(signal),
    retry: false,
  });
  // Set when somebody chooses to name a repository for this one feature instead of saving it.
  // It only bypasses the repository prompt; the credential gate is not optional.
  const [oneOff, setOneOff] = useState(false);

  if (setup.isPending) {
    return (
      <div className="page">
        <TableSkeleton rows={4} label="Checking your setup…" />
      </div>
    );
  }
  if (setup.isError) {
    return (
      <div className="page">
        <ErrorState error={setup.error} onRetry={setup.refetch} />
      </div>
    );
  }

  if (!setup.data.credentials_ready) {
    return <ProviderSetupRequired setup={setup.data} />;
  }
  if (!setup.data.repositories_ready && !oneOff) {
    return <NoRepositoriesConfigured onUseOneOff={() => setOneOff(true)} />;
  }

  return (
    <FeatureForm
      savedRepositories={saved.data?.repositories ?? []}
      savedRepositoriesUnavailable={saved.isError}
      platforms={setup.data.agent_platforms}
      modelSetups={ownSetups.data?.setups ?? []}
    />
  );
}

/**
 * The first gate: the platform has no keys to work with.
 *
 * It names each provider and its state rather than saying "setup required", because the thing
 * somebody needs to know is which one is missing.
 */
function ProviderSetupRequired({ setup }: { setup: SetupState }) {
  return (
    <div className="page stack">
      <PageHeader
        title="Provider setup required"
        subtitle="Before creating a feature, configure the credentials the engineering platform uses on your behalf."
      />
      <Panel title="Credentials">
        <p className="muted">
          A feature is accepted immediately and built afterwards, so the platform needs keys it
          can read once your request has been answered. They are stored encrypted against your
          account and are never shown back.
        </p>
        <ul className="cards" aria-label="Required providers">
          {setup.providers.map((provider) => (
            <li className="card" key={provider.provider}>
              <div className="card__header">
                <strong>{provider.label}</strong>
                <Badge tone={provider.configured ? 'done' : 'attention'}>
                  {provider.configured ? 'Configured' : 'Not configured'}
                </Badge>
              </div>
            </li>
          ))}
        </ul>
        {!setup.credential_storage_available ? (
          <p className="muted">
            This deployment does not store provider credentials, so they cannot be configured
            here. Supply them per request instead.
          </p>
        ) : (
          <div className="form__actions">
            <Link className="button button--primary" to="/settings">
              <IconSettings />
              Configure credentials
            </Link>
          </div>
        )}
      </Panel>
    </div>
  );
}

/** The second gate: nothing saved to build in yet. */
function NoRepositoriesConfigured({ onUseOneOff }: { onUseOneOff: () => void }) {
  return (
    <div className="page stack">
      <PageHeader
        title="No repositories configured"
        subtitle="Save the repositories your team works with so they can be reused across features."
      />
      <Panel title="Repositories">
        <p className="muted">
          Saved repositories are chosen from a list when you create a feature, rather than
          retyped each time. A repository needs a URL, a default branch, and a label you choose.
        </p>
        <div className="form__actions">
          <Link className="button button--primary" to="/settings">
            <IconSettings />
            Configure repositories
          </Link>
          <button type="button" className="button button--quiet" onClick={onUseOneOff}>
            Use a one-time repository instead
          </button>
        </div>
      </Panel>
    </div>
  );
}

function FeatureForm({
  savedRepositories,
  savedRepositoriesUnavailable,
  platforms,
  modelSetups,
}: {
  savedRepositories: SavedRepository[];
  savedRepositoriesUnavailable: boolean;
  platforms: AgentPlatformOption[];
  modelSetups: SavedModelSetup[];
}) {
  const api = useApi();
  const navigate = useNavigate();
  const [loadedFile, setLoadedFile] = useState<string | null>(null);
  const submissionIdentity = useRef<{ fingerprint: string; key: string } | null>(null);

  const form = useForm<NewFeatureValues>({
    resolver: zodResolver(newFeatureSchema),
    // Standard, on whichever provider the operator last submitted with. The preselection is
    // only ever a starting point: if this deployment cannot run it, the selector says so and
    // the form refuses to submit -- it never re-aims the selection itself. A form that
    // quietly flipped an unconfigured choice at load is what sent AB-Feature-173 to the
    // wrong platform.
    defaultValues: { ...DEFAULT_VALUES, agent_platform: preferredAgentPlatform() },
    mode: 'onSubmit',
  });
  const { register, control, handleSubmit, formState, setValue, watch } = form;
  const executionMode = watch('execution_mode');
  const agentPlatform = watch('agent_platform');
  const performanceTier = watch('performance_tier');
  const modelSetupId = watch('model_setup_id');

  const selectedOption = platforms.find(
    (item) => item.platform === agentPlatform && item.performance_tier === performanceTier,
  );
  const selectedSetup = modelSetupId
    ? modelSetups.find((item) => item.setup_id === modelSetupId)
    : undefined;
  // A custom selection that is no longer usable — a credential deleted, a declaration changed,
  // the setup itself gone — blocks submission exactly as an unconfigured pairing does. The
  // form never substitutes another option.
  const selectionBlocked = modelSetupId
    ? selectedSetup === undefined || !selectedSetup.usable
    : !selectedOption?.configured || PLATFORMS_NOT_READY.has(agentPlatform);
  // Whether the current selection's reasoning model can be shown an image, published by the
  // server for exactly this: the vision refusal is surfaced where the selection is made,
  // not as a submit-time surprise. `undefined` while the selection resolves to nothing --
  // an unconfigured pairing already says why it cannot run, and adding a second reason
  // would be answering a question about a model nobody selected.
  const selectionReadsImages = modelSetupId
    ? selectedSetup?.vision_capable
    : selectedOption?.configured
      ? selectedOption.vision_capable
      : undefined;
  const selectionName = modelSetupId
    ? (selectedSetup?.name ?? 'This setup')
    : (selectedOption?.label ?? 'This selection');
  const attachmentCount = watch('attachments').length;

  const create = useMutation({
    mutationFn: (values: NewFeatureValues) => {
      const input = toStartFeatureInput(values);
      const fingerprint = JSON.stringify(input);
      // A timeout leaves creation ambiguous. Reusing the key for the same form data lets the
      // server replay the first result instead of creating a second queued feature. An edited
      // request gets a new identity, avoiding a false idempotency conflict.
      if (submissionIdentity.current?.fingerprint !== fingerprint) {
        submissionIdentity.current = { fingerprint, key: crypto.randomUUID() };
      }
      return api.createFeature(input, { idempotencyKey: submissionIdentity.current.key });
    },
    // Straight to the feature. It exists, it is queued, and it has a reference — there is
    // nothing to wait for on this page. The provider that just worked becomes the next
    // form's preselection.
    onSuccess: (feature, values) => {
      // Only a pairing selection is remembered: a setup is a deliberate per-visit choice,
      // and the remembered platform must be one the submission actually named.
      if (!values.model_setup_id) rememberAgentPlatform(values.agent_platform);
      navigate(`/features/${encodeURIComponent(feature.feature_id)}`);
    },
  });

  async function loadFromFile(file: File) {
    // The document becomes the problem statement verbatim. Nothing is parsed out of it:
    // guessing structure from prose would put words in the author's mouth, and the
    // product-manager agent reads the whole thing anyway.
    const text = await readTextFile(file);
    setValue('problem_statement', text.trim(), { shouldValidate: true });
    setLoadedFile(file.name);
  }

  return (
    <div className="page">
      <PageHeader
        title="Create feature"
        subtitle="Describe the problem and choose the repositories it touches. The platform writes the requirements, plans the work across those repositories, and opens a pull request in each."
      />
      <form
        className="form"
        onSubmit={handleSubmit((values) => {
          // Refuse rather than re-aim. Substituting a different (platform, tier) here is
          // exactly the silent fallback that ran a live feature on the wrong platform; the
          // selector already says why this one cannot run.
          if (selectionBlocked) return;
          // Refuse rather than send, on the same rule: the server would answer 422 and the
          // notice at the selector already says why. Never by dropping the images -- an
          // image the model will not see must not disappear on the way to it. Mock mode is
          // exempt exactly as the server exempts it: a mock feature selects no model, so
          // there is nothing whose capability could be wrong.
          if (attachmentCount > 0 && selectionReadsImages === false && executionMode !== 'mock') {
            return;
          }
          create.mutate(values);
        })}
        noValidate
      >
        {create.isError ? (
          <div className="state state--error" role="alert">
            <p className="state__title">
              {create.error instanceof ApiError ? userMessage(create.error) : 'Submission failed.'}
            </p>
            {create.error instanceof ApiError && create.error.detail ? (
              <details>
                <summary>Technical detail</summary>
                <pre>{create.error.detail}</pre>
              </details>
            ) : null}
          </div>
        ) : null}

        <Fieldset legend="What are we building?">
          {/* Read-only and unreserved. Nothing is allocated by opening this page: the number
              is assigned when the feature is created, by the database, so two submissions at
              the same moment cannot be given the same one. */}
          <div className="field">
            <span className="field__label">Feature ID</span>
            <p className="field__readonly subtle">Assigned automatically on submission</p>
          </div>

          <Field label="Title" error={formState.errors.title?.message}>
            {(id, describedBy) => (
              <input id={id} aria-describedby={describedBy} {...register('title')} />
            )}
          </Field>

          <Field
            label="Problem statement"
            error={formState.errors.problem_statement?.message}
            hint="What is wrong today, and what should be true instead. Markdown is fine, and as long as you like — the platform reads all of it."
          >
            {(id, describedBy) => (
              <textarea
                id={id}
                rows={8}
                aria-describedby={describedBy}
                placeholder="Operators can activate an ad unit from the console but cannot deactivate one…"
                {...register('problem_statement')}
              />
            )}
          </Field>

          <Field
            label="Or upload a PRD"
            hint="A .md or .txt document. It replaces the problem statement above and is read as written."
          >
            {(id) => (
              <div className="row">
                <input
                  id={id}
                  type="file"
                  accept=".md,.txt,text/markdown,text/plain"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void loadFromFile(file);
                  }}
                />
                {loadedFile ? <Badge tone="done">Loaded {loadedFile}</Badge> : null}
              </div>
            )}
          </Field>
        </Fieldset>

        {/* Directly after the problem statement, because the two are one act: somebody
            describing a screen and somebody showing one are doing the same thing, and a
            drop zone two sections below the prose it belongs to reads as an afterthought. */}
        <AttachmentsField
          control={control}
          form={form}
          visionCapable={selectionReadsImages}
          selectionLabel={selectionName}
          mockMode={executionMode === 'mock'}
        />

        <Repositories
          control={control}
          register={register}
          form={form}
          savedRepositories={savedRepositories}
          savedRepositoriesUnavailable={savedRepositoriesUnavailable}
        />

        {/* On the form rather than behind the disclosure. It defaults to live — which clones,
            writes and opens pull requests in the repositories above — and a default that
            consequential must be visible next to what it will act on, not two clicks away in
            a section named for things the platform works out for itself. */}
        <ExecutionMode register={register} mode={executionMode} />

        {/* Beside the mode, not inside the disclosure, and for the same reason: the
            disclosure is for what the platform works out for itself, and this is neither
            derived nor safe to default silently. It cannot be changed once the feature
            starts. */}
        <AgentPlatformField
          platform={agentPlatform}
          tier={performanceTier}
          platforms={platforms}
          modelSetups={modelSetups}
          modelSetupId={modelSetupId}
          blocked={selectionBlocked}
          onSelect={(platform, tier) => {
            setValue('model_setup_id', '');
            setValue('agent_platform', platform);
            setValue('performance_tier', tier);
          }}
          onSelectSetup={(setupId) => {
            setValue('model_setup_id', setupId);
          }}
        />

        <DesignReferences control={control} register={register} form={form} />
        <AdvancedDetails control={control} register={register} form={form} />

        <div className="form__actions">
          <button type="submit" className="button button--primary" disabled={create.isPending}>
            {create.isPending ? 'Creating…' : 'Create feature'}
          </button>
          {create.isPending ? (
            <span className="muted" role="status">
              Queueing the feature; you will be taken to it as soon as it exists.
            </span>
          ) : null}
        </div>
      </form>
    </div>
  );
}

/**
 * Mock or live, and what each one means.
 *
 * The consequence is stated rather than implied: live is the default, and somebody choosing
 * it should not have to already know that it clones the repositories, commits to them and
 * opens pull requests. Mock is offered beside it because trying the shape of a feature
 * without touching anything is a real thing to want.
 */
function ExecutionMode({
  register,
  mode,
}: {
  register: UseFormRegister<NewFeatureValues>;
  mode: NewFeatureValues['execution_mode'];
}) {
  return (
    <Fieldset legend="Execution">
      <Field
        label="Mode"
        hint={
          mode === 'live'
            ? 'Live clones each repository, writes to a new branch, and opens a draft pull request. It never merges and never deploys.'
            : 'Mock plans and reviews the whole feature without reaching a provider or touching a repository.'
        }
      >
        {(id, describedBy) => (
          <select id={id} aria-describedby={describedBy} {...register('execution_mode')}>
            <option value="live">Live</option>
            <option value="mock">Mock</option>
          </select>
        )}
      </Field>
    </Fieldset>
  );
}

/**
 * What each tier is for, stated where the choice is made rather than in documentation.
 *
 * The Economy warning is deliberate product copy: its label must carry its own scope, because
 * a cheap tier that retries more can cost more than an expensive tier that lands sooner.
 */
const TIER_NOTES: Record<NewFeatureValues['performance_tier'], string> = {
  low: 'Economy is for small, well-bounded changes — larger work will cost more here, not less, through retries.',
  medium: 'Standard is the recommended default for most features.',
  high: 'Max is for the hardest multi-repository work, at today’s maximum-effort rates.',
  ultra: 'Ultra runs the newest frontier model at full effort — above Max, priced accordingly.',
};

/**
 * The custom-setup sibling of the tier notes: honest scope, stated where the choice is made.
 * A setup is neither cheaper nor better by construction, and its author owns that trade.
 */
const CUSTOM_SETUP_NOTE =
  'A custom setup runs exactly the models its author pinned — it is neither cheaper nor better by construction, and its author owns that trade.';

/**
 * What each stage of the work is, one line, beside the model that runs it.
 *
 * The wording mirrors the server's own routing reasons; the roles and models themselves come
 * from the server. A role this file has never heard of still gets a row — written as itself,
 * with no description, rather than dropped.
 */
const STAGE_NOTES: Record<string, string> = {
  reasoning: 'Requirements, planning and analysis',
  coding: 'Implementation and substantive fixes',
  review: 'Independent review of the changes',
  scoped_fix: 'Small mechanical repairs',
};

/** One stage of the work, the model the current selection resolves it to, and its effort. */
type StageModelRow = {
  role: string;
  model: string;
  /** Set only for custom setups, which may pin each role to its own provider. */
  platform: string | null;
  /**
   * The reasoning effort this stage is sent at: a level name, `null` when the provider's own
   * default applies, and `undefined` when the source of this row said nothing about effort —
   * a server that predates the field. The three are rendered as three different things,
   * because "runs at the provider default" is a fact and "not reported" is the absence of one.
   */
  effort: string | null | undefined;
};

/**
 * The effort a stage is sent at, as a level name — never a guess.
 *
 * The level names are the deployment's own vocabulary (`low`, `high`, `xhigh`, `max`), so they
 * are set as typed rather than prettified: this is the word an operator would put in the
 * environment, and the settings table beside it shows the same one.
 */
function StageEffort({ effort }: { effort: string | null | undefined }) {
  if (effort === undefined) return <span className="subtle">Not reported</span>;
  if (effort === null) return <span className="subtle">Provider default</span>;
  return <code className="mono">{effort}</code>;
}

const STAGE_COLUMNS: Column<StageModelRow>[] = [
  {
    key: 'stage',
    header: 'Stage',
    render: (row) => (
      <span className="stack stack--tight">
        <strong>{MODEL_ROLE_LABELS[row.role] ?? row.role}</strong>
        {STAGE_NOTES[row.role] ? <span className="subtle">{STAGE_NOTES[row.role]}</span> : null}
      </span>
    ),
  },
  {
    key: 'model',
    header: 'Model',
    render: (row) => (
      <span className="stack stack--tight">
        <span title={row.model}>{modelDisplayName(row.model) ?? row.model}</span>
        {row.platform ? <span className="subtle">{providerDisplayName(row.platform)}</span> : null}
      </span>
    ),
  },
  {
    key: 'effort',
    header: 'Reasoning effort',
    render: (row) => <StageEffort effort={row.effort} />,
  },
];

/** One selectable value per (platform, tier) pairing, for a single `<select>`. */
function optionValue(platform: string, tier: string): string {
  return `${platform}:${tier}`;
}

/** The custom-setup form of the same value. Setup ids are server-minted and colon-free. */
function setupOptionValue(setupId: string): string {
  return `custom:${setupId}`;
}

/**
 * Which model provider runs this feature, at which performance tier, and what that resolves to.
 *
 * The hint says what the tier is for and that the choice is fixed once the feature starts;
 * below the control, a table says what a submitter cannot see for themselves — each stage of
 * the work and the model the selection resolves it to. The roles and model names come from
 * the server rather than from this file — a copy here would be a second source of truth for
 * deployment configuration, and it would be wrong the first time somebody changed the other
 * one.
 *
 * An unconfigured pairing stays visible and disabled rather than disappearing, with the reason
 * beside it. And the form never substitutes a selection: when the current choice cannot run on
 * this deployment it is rendered, disabled, with the reason — and submission is refused until
 * the person picks a configured option themselves. A form that quietly re-aimed the selection
 * at load is what ran AB-Feature-173 on the wrong platform.
 */
function AgentPlatformField({
  platform,
  tier,
  platforms,
  modelSetups,
  modelSetupId,
  blocked,
  onSelect,
  onSelectSetup,
}: {
  platform: NewFeatureValues['agent_platform'];
  tier: NewFeatureValues['performance_tier'];
  platforms: AgentPlatformOption[];
  modelSetups: SavedModelSetup[];
  modelSetupId: string;
  blocked: boolean;
  onSelect: (
    platform: NewFeatureValues['agent_platform'],
    tier: NewFeatureValues['performance_tier'],
  ) => void;
  onSelectSetup: (setupId: string) => void;
}) {
  const selectedSetup = modelSetupId
    ? modelSetups.find((item) => item.setup_id === modelSetupId)
    : undefined;
  const selectedValue = modelSetupId ? setupOptionValue(modelSetupId) : optionValue(platform, tier);
  const selected = platforms.find(
    (item) => item.platform === platform && item.performance_tier === tier,
  );
  // Custom setups may pin each role to its own provider, so those rows carry one; a
  // deployment pairing is one platform throughout, so its rows carry none.
  const setupRoles = selectedSetup?.roles ?? {};
  const pairingModels = selected?.configured ? selected.models : {};
  // Keyed like the models, and read the same way: a role the server named an effort for
  // carries it, a role it named `null` for runs at the provider's default, and a role it
  // said nothing about is left undefined rather than filled in as one of those two.
  const pairingEfforts = selected?.configured ? selected.reasoning_efforts : {};
  const stageRows: StageModelRow[] = modelSetupId
    ? orderedModelRoles(Object.keys(setupRoles)).map((role) => ({
        role,
        model: setupRoles[role]!.model,
        platform: setupRoles[role]!.platform ?? null,
        // A setup is stored as its author entered it, so a role with no effort is one they
        // left blank — the provider's default, not an unanswered question.
        effort: setupRoles[role]!.reasoning_effort ?? null,
      }))
    : orderedModelRoles(Object.keys(pairingModels)).map((role) => ({
        role,
        model: pairingModels[role]!,
        platform: null,
        effort: role in pairingEfforts ? pairingEfforts[role]! : undefined,
      }));

  return (
    <Fieldset legend="Agent platform">
      <Field
        label="Platform and performance tier"
        hint={[
          modelSetupId ? CUSTOM_SETUP_NOTE : TIER_NOTES[tier],
          'The platform and its tier cannot be changed after the feature starts.',
        ]
          .filter(Boolean)
          .join(' ')}
        error={blocked ? blockedReason(modelSetupId, selectedSetup, selected) : undefined}
      >
        {(id, describedBy) => (
          <select
            id={id}
            aria-describedby={describedBy}
            aria-invalid={blocked || undefined}
            value={selectedValue}
            onChange={(event) => {
              const value = event.target.value;
              if (value.startsWith('custom:')) {
                onSelectSetup(value.slice('custom:'.length));
                return;
              }
              const [nextPlatform, nextTier] = value.split(':');
              onSelect(
                nextPlatform as NewFeatureValues['agent_platform'],
                nextTier as NewFeatureValues['performance_tier'],
              );
            }}
          >
            {/* A selection this deployment does not even list — a remembered choice after a
                reconfiguration — is rendered as itself, disabled, rather than snapping the
                display to the first option while the form value says otherwise. */}
            {!modelSetupId && selected === undefined ? (
              <option value={selectedValue} disabled>
                {`${platform} — ${tier} — not configured on this deployment`}
              </option>
            ) : null}
            {modelSetupId && selectedSetup === undefined ? (
              <option value={selectedValue} disabled>
                This setup no longer exists — choose another option
              </option>
            ) : null}
            {platforms.map((item) => (
              <option
                key={optionValue(item.platform, item.performance_tier)}
                value={optionValue(item.platform, item.performance_tier)}
                disabled={!item.configured || PLATFORMS_NOT_READY.has(item.platform)}
              >
                {!item.configured
                  ? `${item.label} — not configured on this deployment`
                  : PLATFORMS_NOT_READY.has(item.platform)
                    ? `${item.label} (Not ready)`
                    : item.label}
              </option>
            ))}
            {modelSetups.length > 0 ? (
              <optgroup label="Your setups">
                {modelSetups.map((item) => (
                  // A setup that is not currently usable stays visible and disabled, with the
                  // reason — the same treatment an unconfigured pairing gets. The form never
                  // substitutes another option.
                  <option
                    key={setupOptionValue(item.setup_id)}
                    value={setupOptionValue(item.setup_id)}
                    disabled={!item.usable}
                  >
                    {item.usable ? item.name : `${item.name} — ${unusableSummary(item)}`}
                  </option>
                ))}
              </optgroup>
            ) : null}
          </select>
        )}
      </Field>
      {/* The stages this selection resolves and the model each one runs on. Rendered only
          when the selection actually resolves: an unconfigured pairing has no models to
          promise, and inventing a table for it would state a fallback as fact. */}
      {stageRows.length > 0 ? (
        <DataTable
          label="Stages, the models they run on and the effort each is sent at"
          compact
          columns={STAGE_COLUMNS}
          rows={stageRows}
          rowKey={(row) => row.role}
        />
      ) : null}
    </Fieldset>
  );
}

/** Why a custom option cannot run right now, short enough for an option label. */
function unusableSummary(setup: SavedModelSetup): string {
  if (setup.missing_credentials.length > 0) {
    return `missing ${listed(setup.missing_credentials)} credential${
      setup.missing_credentials.length > 1 ? 's' : ''
    }`;
  }
  return 'no longer valid on this deployment';
}

/** The refusal beside the control, in the server's sentence where one exists. */
function blockedReason(
  modelSetupId: string,
  selectedSetup: SavedModelSetup | undefined,
  selected: AgentPlatformOption | undefined,
): string {
  if (modelSetupId) {
    if (selectedSetup === undefined) {
      return 'This setup no longer exists. Choose another option — the form will not substitute one for you.';
    }
    // The server's own sentence, verbatim: this is where its author can act on it.
    if (selectedSetup.validation_error) return selectedSetup.validation_error;
    return `${selectedSetup.name} cannot run: ${unusableSummary(selectedSetup)}. Configure the credential in Settings, or choose another option — the form will not substitute one for you.`;
  }
  // A withheld platform outranks the configured/unconfigured distinction: the pairing may be
  // fully configured, and the honest reason it cannot be picked is the gate, not the config.
  if (selected !== undefined && PLATFORMS_NOT_READY.has(selected.platform)) {
    return `${selected.label} is not ready yet. Choose another option — the form will not substitute one for you.`;
  }
  return `${selected?.label ?? 'The selected option'} is not configured on this deployment. Choose a configured option — the form will not substitute one for you.`;
}

/** Join names the way a sentence does: "a", "a and b", "a, b and c". */
function listed(names: string[]): string {
  if (names.length <= 1) return names[0] ?? '';
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

type Section = {
  control: Control<NewFeatureValues>;
  register: UseFormRegister<NewFeatureValues>;
  form: ReturnType<typeof useForm<NewFeatureValues>>;
};

/**
 * The designs this feature is meant to look like.
 *
 * Optional, like everything else that is not a title, a problem statement and a repository.
 * A person asking for a screen has two documents -- what it must do, and what it must look
 * like -- and until there was somewhere to put the second one, a frontend workstream was
 * planned, built and reviewed against prose alone while the mock sat in a chat message.
 *
 * Only the URL and an optional label are asked for. The file key and the frame ids are the
 * server's derivation from the same URL, and what the design *applies to* is deliberately not
 * a field here: this platform does not know which repository renders UI, and the plan is where
 * that decision is made.
 */
function DesignReferences({ control, register, form }: Section) {
  const { fields, append, remove } = useFieldArray({ control, name: 'design_references' });
  const errors = form.formState.errors;
  const listError = errors.design_references?.message ?? errors.design_references?.root?.message;

  return (
    <Fieldset
      legend="Design"
      error={typeof listError === 'string' ? listError : undefined}
    >
      {fields.length === 0 ? (
        <p className="field__hint">
          None attached. Paste a Figma frame link and every role that builds or judges this
          feature is shown the design as part of the request.
        </p>
      ) : null}
      {fields.map((field, index) => (
        <div key={field.id} className="repeatable">
          <div className="repeatable__header">
            <strong>Design {index + 1}</strong>
            <RemoveButton onClick={() => remove(index)} label={`Remove design ${index + 1}`} />
          </div>
          <Field
            label="Figma link"
            hint="Right-click a frame in Figma → Copy link to selection. A link with no frame gets the file’s top-level frames."
            error={errors.design_references?.[index]?.url?.message}
          >
            {(id, describedBy) => (
              <input
                id={id}
                aria-describedby={describedBy}
                placeholder="https://www.figma.com/design/…?node-id=1-23"
                {...register(`design_references.${index}.url`)}
              />
            )}
          </Field>
          <Field
            label="What this is"
            hint="Optional — “empty state”, “mobile breakpoint”. It is shown beside the frame."
            error={errors.design_references?.[index]?.label?.message}
          >
            {(id, describedBy) => (
              <input
                id={id}
                aria-describedby={describedBy}
                {...register(`design_references.${index}.label`)}
              />
            )}
          </Field>
        </div>
      ))}
      <button
        type="button"
        className="button button--quiet"
        onClick={() => append({ url: '', label: '' })}
      >
        {fields.length === 0 ? 'Attach a design' : 'Add another design'}
      </button>
    </Fieldset>
  );
}

/**
 * Everything the platform will work out for itself, for the times somebody already knows.
 *
 * Closed by default and empty by default. A section left untouched sends nothing, so opening
 * this cannot accidentally commit an author to a half-written user story.
 */
function AdvancedDetails({ control, register, form }: Section) {
  const errors = form.formState.errors;
  // A validation error inside a closed section is invisible, which reads as a submit button
  // that does nothing. If anything in here failed, it opens.
  const hasError = Boolean(errors.goals || errors.user_stories || errors.requirements);

  return (
    <details className="advanced" open={hasError}>
      <summary className="advanced__summary">
        <span className="advanced__title">Advanced details</span>
        <span className="subtle">
          Goals, user stories and requirements. The platform derives these from the problem
          statement; anything you write here is used instead.
        </span>
      </summary>

      <div className="advanced__body">
        <StringListField
          control={control}
          name="goals"
          legend="Goals"
          itemLabel="Goal"
          errors={errors}
        />
        <UserStories control={control} register={register} form={form} />
        <Requirements control={control} register={register} form={form} />
        <Fieldset legend="Additional context">
          <StringListField
            control={control}
            name="constraints"
            legend="Constraints"
            itemLabel="Constraint"
            errors={errors}
          />
          <StringListField
            control={control}
            name="out_of_scope"
            legend="Out of scope"
            itemLabel="Exclusion"
            errors={errors}
          />
          <StringListField
            control={control}
            name="stakeholders"
            legend="Stakeholders"
            itemLabel="Stakeholder"
            errors={errors}
          />
        </Fieldset>
      </div>
    </details>
  );
}

function UserStories({ control, register, form }: Section) {
  const { fields, append, remove } = useFieldArray({ control, name: 'user_stories' });
  const error = form.formState.errors.user_stories?.message;
  return (
    <Fieldset legend="User stories" error={typeof error === 'string' ? error : undefined}>
      {fields.length === 0 ? (
        <p className="field__hint">None written. The platform will derive them.</p>
      ) : null}
      {fields.map((field, index) => (
        <div key={field.id} className="repeatable">
          <div className="repeatable__header">
            <strong>Story {index + 1}</strong>
            <RemoveButton onClick={() => remove(index)} label={`Remove story ${index + 1}`} />
          </div>
          <Field label="ID" error={form.formState.errors.user_stories?.[index]?.story_id?.message}>
            {(id) => <input id={id} {...register(`user_stories.${index}.story_id`)} />}
          </Field>
          <Field label="Persona" error={form.formState.errors.user_stories?.[index]?.persona?.message}>
            {(id) => <input id={id} {...register(`user_stories.${index}.persona`)} />}
          </Field>
          <Field label="Need" error={form.formState.errors.user_stories?.[index]?.need?.message}>
            {(id) => <input id={id} {...register(`user_stories.${index}.need`)} />}
          </Field>
          <Field label="Benefit" error={form.formState.errors.user_stories?.[index]?.benefit?.message}>
            {(id) => <input id={id} {...register(`user_stories.${index}.benefit`)} />}
          </Field>
          <NestedCriteria
            control={control}
            register={register}
            form={form}
            parent="user_stories"
            index={index}
          />
        </div>
      ))}
      <button
        type="button"
        className="button button--quiet"
        onClick={() =>
          append({
            story_id: `US-${fields.length + 1}`,
            persona: '',
            need: '',
            benefit: '',
            acceptance_criteria: [{ value: '' }],
          })
        }
      >
        Add story
      </button>
    </Fieldset>
  );
}

function Requirements({ control, register, form }: Section) {
  const { fields, append, remove } = useFieldArray({ control, name: 'requirements' });
  const error = form.formState.errors.requirements?.message;
  return (
    <Fieldset legend="Requirements" error={typeof error === 'string' ? error : undefined}>
      {fields.length === 0 ? (
        <p className="field__hint">None written. The platform will derive them.</p>
      ) : null}
      {fields.map((field, index) => (
        <div key={field.id} className="repeatable">
          <div className="repeatable__header">
            <strong>Requirement {index + 1}</strong>
            <RemoveButton
              onClick={() => remove(index)}
              label={`Remove requirement ${index + 1}`}
            />
          </div>
          <Field
            label="ID"
            error={form.formState.errors.requirements?.[index]?.requirement_id?.message}
          >
            {(id) => <input id={id} {...register(`requirements.${index}.requirement_id`)} />}
          </Field>
          <Field
            label="Description"
            error={form.formState.errors.requirements?.[index]?.description?.message}
          >
            {(id) => <textarea id={id} rows={3} {...register(`requirements.${index}.description`)} />}
          </Field>
          <Field label="Priority">
            {(id) => (
              <select id={id} {...register(`requirements.${index}.priority`)}>
                <option value="must">Must</option>
                <option value="should">Should</option>
                <option value="could">Could</option>
                <option value="wont">Won’t</option>
              </select>
            )}
          </Field>
          <NestedCriteria
            control={control}
            register={register}
            form={form}
            parent="requirements"
            index={index}
          />
        </div>
      ))}
      <button
        type="button"
        className="button button--quiet"
        onClick={() =>
          append({
            requirement_id: `REQ-${fields.length + 1}`,
            description: '',
            priority: 'must',
            acceptance_criteria: [{ value: '' }],
          })
        }
      >
        Add requirement
      </button>
    </Fieldset>
  );
}

function NestedCriteria({
  control,
  register,
  form,
  parent,
  index,
}: Section & { parent: 'user_stories' | 'requirements'; index: number }) {
  const name = `${parent}.${index}.acceptance_criteria` as const;
  const { fields, append, remove } = useFieldArray({ control, name });
  const branch = form.formState.errors[parent]?.[index]?.acceptance_criteria;
  const error = branch?.message ?? branch?.root?.message;
  return (
    <Fieldset legend="Acceptance criteria" error={typeof error === 'string' ? error : undefined}>
      <ul className="list-field">
        {fields.map((field, criterion) => (
          <li key={field.id} className="list-field__row">
            <input
              aria-label={`Acceptance criterion ${criterion + 1}`}
              {...register(`${name}.${criterion}.value` as const)}
            />
            {fields.length > 1 ? (
              <RemoveButton
                onClick={() => remove(criterion)}
                label={`Remove acceptance criterion ${criterion + 1}`}
              />
            ) : null}
          </li>
        ))}
      </ul>
      <button type="button" className="button button--quiet" onClick={() => append({ value: '' })}>
        Add criterion
      </button>
    </Fieldset>
  );
}

/**
 * Which repositories this feature touches, chosen from the ones already saved.
 *
 * Any number, and any mix of labels: nothing here assumes a frontend/backend pair, and two
 * repositories with the same label is an ordinary feature. A one-time repository is still
 * available for something not worth saving.
 */
function Repositories({
  control,
  register,
  form,
  savedRepositories,
  savedRepositoriesUnavailable,
}: Section & { savedRepositories: SavedRepository[]; savedRepositoriesUnavailable: boolean }) {
  const { fields, append, remove } = useFieldArray({ control, name: 'repositories' });
  const branch = form.formState.errors.repositories;
  const error = branch?.message ?? branch?.root?.message;
  const values = form.watch('repositories');
  const [addingOneOff, setAddingOneOff] = useState(false);

  const selectedUrls = useMemo(
    () => new Set((values ?? []).map((item) => item.repository_url)),
    [values],
  );

  return (
    <Fieldset legend="Repositories" error={typeof error === 'string' ? error : undefined}>
      {savedRepositories.length > 0 ? (
        <div className="field">
          <span className="field__label">Select repositories</span>
          {/* Toggles rather than a `<select multiple>`: the labels matter, the list is short,
              and a multi-select is the control people most often get wrong. */}
          <div className="repo-picker" role="group" aria-label="Saved repositories">
            {savedRepositories.map((repository) => {
              const selected = selectedUrls.has(repository.repository_url);
              return (
                <button
                  type="button"
                  key={repository.configuration_id}
                  aria-pressed={selected}
                  className={selected ? 'repo-option repo-option--on' : 'repo-option'}
                  onClick={() => {
                    if (selected) {
                      const index = (values ?? []).findIndex(
                        (item) => item.repository_url === repository.repository_url,
                      );
                      if (index >= 0) remove(index);
                      return;
                    }
                    append({
                      repository_url: repository.repository_url,
                      default_branch: repository.default_branch,
                      required: true,
                      label: repository.name,
                      repository_type: repository.repository_type,
                      configuration_id: repository.configuration_id,
                    });
                  }}
                >
                  <span className="repo-option__name">{repository.name}</span>
                  <span className="repo-option__type">{repository.repository_type}</span>
                </button>
              );
            })}
          </div>
        </div>
      ) : savedRepositoriesUnavailable ? (
        <p className="field__hint">
          This deployment does not save repositories, so name the ones this feature touches
          below.
        </p>
      ) : null}

      {fields.length > 0 ? (
        <div className="field">
          <span className="field__label">Selected</span>
          <ul className="repo-chips" aria-label="Selected repositories">
            {fields.map((field, index) => {
              const value = values?.[index];
              const name =
                value?.label || derivedRepositoryName(value?.repository_url ?? '') || 'Repository';
              return (
                <li key={field.id} className="repo-chip">
                  <span className="repo-chip__name">{name}</span>
                  {value?.repository_type ? (
                    <span className="repo-chip__meta">{value.repository_type}</span>
                  ) : null}
                  <span className="repo-chip__meta mono">{value?.default_branch}</span>
                  {/* Only offered where it can be true: with a single repository, making it
                      optional would leave the feature with no completion gate at all. */}
                  {fields.length > 1 ? (
                    <label className="repo-chip__required">
                      <input type="checkbox" {...register(`repositories.${index}.required`)} />
                      Required
                    </label>
                  ) : null}
                  <button
                    type="button"
                    className="repo-chip__remove"
                    aria-label={`Remove ${name}`}
                    onClick={() => remove(index)}
                  >
                    ×
                  </button>
                  {form.formState.errors.repositories?.[index]?.repository_url ? (
                    <span className="field__error">
                      {form.formState.errors.repositories[index]?.repository_url?.message}
                    </span>
                  ) : null}
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}

      {addingOneOff ? (
        <OneOffRepository
          onAdd={(repository) => {
            append(repository);
            setAddingOneOff(false);
          }}
          onCancel={() => setAddingOneOff(false)}
        />
      ) : (
        <button
          type="button"
          className="button button--quiet"
          onClick={() => setAddingOneOff(true)}
        >
          <IconPlus />
          Add one-time repository
        </button>
      )}
    </Fieldset>
  );
}

/**
 * A repository used for this feature and not kept.
 *
 * Deliberately not the primary path, and deliberately still here: a spike or a repository
 * somebody touches once should not have to become a saved configuration first. Saving it is
 * a separate act, in Settings, where the saved list lives.
 */
function OneOffRepository({
  onAdd,
  onCancel,
}: {
  onAdd: (repository: NewFeatureValues['repositories'][number]) => void;
  onCancel: () => void;
}) {
  const [url, setUrl] = useState('');
  const [defaultBranch, setDefaultBranch] = useState('master');
  const derived = derivedRepositoryName(url);

  return (
    <div className="repeatable">
      <div className="repeatable__header">
        <strong>One-time repository</strong>
        <span className="subtle">Used for this feature only, and not saved.</span>
      </div>
      <div className="field">
        <label className="field__label" htmlFor="one-off-url">
          URL
        </label>
        <input
          id="one-off-url"
          value={url}
          placeholder="https://github.com/owner/repository"
          onChange={(event) => setUrl(event.target.value)}
        />
        {derived ? (
          <p className="field__hint">
            Identified as <code className="mono">{derived}</code>
          </p>
        ) : null}
      </div>
      <div className="field">
        <label className="field__label" htmlFor="one-off-branch">
          Default branch
        </label>
        <input
          id="one-off-branch"
          value={defaultBranch}
          onChange={(event) => setDefaultBranch(event.target.value)}
        />
      </div>
      <div className="form__actions">
        <button
          type="button"
          className="button"
          disabled={!url.trim() || !defaultBranch.trim()}
          onClick={() =>
            onAdd({
              ...EMPTY_REPOSITORY,
              repository_url: url.trim(),
              default_branch: defaultBranch.trim(),
              label: derived ?? '',
              repository_type: 'One-time',
            })
          }
        >
          Add to this feature
        </button>
        <button type="button" className="button button--quiet" onClick={onCancel}>
          Cancel
        </button>
      </div>
      <p className="field__hint">
        To reuse it later, save it under <Link to="/settings">Settings → Repositories</Link>.
      </p>
    </div>
  );
}
