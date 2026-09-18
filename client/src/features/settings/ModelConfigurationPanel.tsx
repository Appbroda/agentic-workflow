import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import type { SaveModelSetupInput } from '@/api/features';
import { Async, EmptyState } from '@/components/common/States';
import { Badge } from '@/components/ui/Badge';
import { DataTable, type Column } from '@/components/ui/DataTable';
import { Drawer } from '@/components/ui/Drawer';
import { Panel, Segmented } from '@/components/ui/Layout';
import { CopyValue, DetailList, DetailRow } from '@/components/ui/Value';
import { Field } from '@/features/new-feature/fields';
import { MODEL_ROLE_LABELS, MODEL_ROLE_ORDER, modelDisplayName } from '@/utils/model';
import type {
  ModelRoleConfiguration,
  ModelSetup,
  SavedModelSetup,
} from '@/schemas/feature';

/**
 * Which model each stage of a feature runs on, and how hard it is asked to think.
 *
 * Two kinds of row share one table. The deployment's own pairings are read-only, and each row
 * says so — what they resolve to is deployment configuration, so the panel names the variable
 * each value comes from instead of pretending it could be edited here. The caller's own model
 * setups join them as `custom` rows: authored on this page, per role — platform, model, effort
 * and output bound — validated by the server with the same rules a tier passes, and refused
 * with the server's own sentence rather than a paraphrase.
 */
export function ModelConfigurationPanel() {
  return (
    <Panel
      title="Models"
      meta="What each stage of a feature runs on, as this deployment resolved it"
      flush
    >
      <ResolvedModelRoles />
    </Panel>
  );
}

/** The panel's body draws its own table edges, so a prose fallback supplies its own inset. */
const PADDED = { padding: 'var(--space-5)' };

function ResolvedModelRoles() {
  const api = useApi();
  const queryClient = useQueryClient();
  const configuration = useQuery({
    queryKey: ['model-configuration'],
    queryFn: ({ signal }) => api.getModelConfiguration(signal),
    // Deployment configuration does not change while somebody reads it, and a server that
    // predates this endpoint answers 404. Neither is worth retrying into an error box.
    retry: false,
  });
  // The raw authoring list rides beside the resolved table: it carries the values as entered
  // plus each setup's usability, which is what the edit form and the warnings need. Fetched
  // only when the server says this caller may author — a viewer would be refused.
  const ownSetups = useQuery({
    queryKey: ['model-setups'],
    queryFn: ({ signal }) => api.listModelSetups(signal),
    enabled: configuration.data?.editable === true,
    retry: false,
  });
  // Which pairing is on screen. Local state rather than a route: this is one panel inside
  // Settings, and a person switching between two presets is comparing them, not navigating.
  const [selected, setSelected] = useState<string | null>(null);
  // The authoring drawer: absent, a blank form, or an existing setup being edited.
  const [editing, setEditing] = useState<{ setup: SavedModelSetup | null } | null>(null);
  const remove = useMutation({
    mutationFn: (setupId: string) => api.deleteModelSetup(setupId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['model-configuration'] });
      await queryClient.invalidateQueries({ queryKey: ['model-setups'] });
    },
  });

  if (configuration.isError) {
    const error = configuration.error;
    if (error instanceof ApiError && error.status === 404) {
      return (
        <p className="muted" style={PADDED}>
          This deployment’s API does not publish its resolved model configuration. Its startup
          log still names it.
        </p>
      );
    }
    return (
      <p className="field__error" style={PADDED}>
        {userMessage(error as ApiError)}
      </p>
    );
  }

  return (
    <Async query={configuration}>
      {(data) => {
        const setups = data.setups;
        const editor = editing ? (
          <SetupEditor
            setup={editing.setup}
            suggestions={modelSuggestions(setups)}
            onClose={() => setEditing(null)}
            onSaved={async () => {
              await queryClient.invalidateQueries({ queryKey: ['model-configuration'] });
              await queryClient.invalidateQueries({ queryKey: ['model-setups'] });
            }}
          />
        ) : null;
        if (setups.length === 0) {
          return (
            <>
              <EmptyState
                title="No model configuration resolved"
                detail="This deployment has not resolved a model for any platform, so it cannot run a feature. The models are set in its environment."
              />
              {data.editable ? (
                <div className="toolbar" style={PADDED}>
                  <button
                    type="button"
                    className="button"
                    onClick={() => setEditing({ setup: null })}
                  >
                    New setup
                  </button>
                </div>
              ) : null}
              {editor}
            </>
          );
        }
        const current = setups.find((item) => setupKey(item) === selected) ?? setups[0]!;
        const ownRow =
          current.origin === 'custom'
            ? (ownSetups.data?.setups ?? []).find((item) => item.setup_id === current.setup_id)
            : undefined;
        return (
          <>
            <div className="toolbar">
              {setups.length > 1 ? (
                <Segmented
                  label="Model setup"
                  value={setupKey(current)}
                  options={setups.map((item) => ({
                    value: setupKey(item),
                    // The caller's own setups are grouped after the deployment's pairings by
                    // the server; the label marks them as theirs so the two kinds read apart.
                    label: item.origin === 'custom' ? `${item.label} — yours` : item.label,
                  }))}
                  onChange={setSelected}
                />
              ) : (
                // One pairing needs no control, but it still needs its name: the table below
                // is meaningless without knowing which platform and tier it describes.
                <strong>{current.label}</strong>
              )}
              <span className="toolbar__spacer" />
              {current.origin === 'custom' ? (
                <>
                  <button
                    type="button"
                    className="button button--small"
                    onClick={() => setEditing({ setup: ownRow ?? null })}
                    disabled={ownRow === undefined}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    className="button button--small button--quiet"
                    onClick={() => {
                      if (current.setup_id) remove.mutate(current.setup_id);
                      setSelected(null);
                    }}
                    disabled={remove.isPending}
                  >
                    Delete
                  </button>
                </>
              ) : null}
              {data.editable ? (
                <button
                  type="button"
                  className="button button--small"
                  onClick={() => setEditing({ setup: null })}
                >
                  New setup
                </button>
              ) : null}
              <span className="subtle">
                {/* Sourced from each row rather than from this component's belief: deployment
                    rows stay set in the environment however many custom rows exist. */}
                {current.origin === 'custom'
                  ? 'Your setup — authored here, pinned by features that select it.'
                  : data.editable
                    ? 'Set in the deployment’s environment. You can author your own setup.'
                    : 'Set in the deployment’s environment — not editable here'}
              </span>
            </div>
            {ownRow?.warnings.length ? (
              <p className="muted" style={PADDED}>
                <Badge tone="attention">Warning</Badge> {ownRow.warnings.join(' ')}
              </p>
            ) : null}
            <DataTable
              label={`Model roles — ${current.label}`}
              rows={current.roles}
              rowKey={(role) => role.role}
              columns={ROLE_COLUMNS}
              compact
              expandLabel="Why this role runs, and where it is configured"
              expand={(role) => <RoleProvenance role={role} origin={current.origin} />}
            />
            {editor}
          </>
        );
      }}
    </Async>
  );
}

/** One row's stable key: the pairing for deployment rows, the setup id for custom ones. */
function setupKey(setup: ModelSetup): string {
  return setup.origin === 'custom' && setup.setup_id
    ? `custom:${setup.setup_id}`
    : `${setup.platform}:${setup.performance_tier}`;
}

/** Every model identifier the deployment's rows resolve to, offered as typing suggestions. */
function modelSuggestions(setups: ModelSetup[]): string[] {
  return [
    ...new Set(
      setups
        .filter((item) => item.origin === 'deployment')
        .flatMap((item) => item.roles.map((role) => role.model)),
    ),
  ];
}

// The role vocabulary is shared with the submission form: one spelling per role, everywhere.
const ROLE_LABELS = MODEL_ROLE_LABELS;
const ROLE_ORDER = MODEL_ROLE_ORDER;

/** The effort vocabulary the platform accepts, plus the distinct "send nothing" choice. */
const REASONING_LEVELS = ['none', 'low', 'medium', 'high', 'xhigh', 'max'] as const;

const ROLE_COLUMNS: Column<ModelRoleConfiguration>[] = [
  {
    key: 'role',
    header: 'Role',
    render: (role) => (
      <span className="stack stack--tight">
        <strong>{ROLE_LABELS[role.role] ?? role.role}</strong>
        {/* A mixed custom setup pins each role to its own provider; deployment rows carry no
            per-role platform because the whole pairing is one. */}
        {role.platform ? <span className="subtle">{role.platform}</span> : null}
      </span>
    ),
  },
  {
    key: 'model',
    header: 'Model',
    // Written the way every other surface writes a model, with the identifier itself on the
    // element and one click away in the detail row — the configured value is the thing an
    // operator compares against their environment.
    render: (role) => <span title={role.model}>{modelDisplayName(role.model) ?? role.model}</span>,
  },
  {
    key: 'effort',
    header: 'Reasoning effort',
    render: (role) => <Effort role={role} />,
  },
  {
    key: 'max_tokens',
    header: 'Max output tokens',
    align: 'right',
    render: (role) =>
      // A bound the deployment never set is the provider's own, which is a different fact
      // from a bound of zero and must not be written as a number.
      role.max_tokens == null ? (
        <span className="subtle">provider default</span>
      ) : (
        role.max_tokens.toLocaleString()
      ),
  },
];

/**
 * The effort this role is asked for, and — when they differ — the one that was configured.
 *
 * The two differ when the deployment declared this model does not accept the configured
 * level, in which case nothing is sent and the provider's default applies. Showing only the
 * outcome would present that normalization as somebody's preference, and an operator
 * comparing this table against their environment would find a disagreement the page does not
 * explain. It is also the exact shape of the failure that killed feature 181: an effort
 * configured past what the model would take.
 */
function Effort({ role }: { role: ModelRoleConfiguration }) {
  if (!role.reasoning_effort) {
    return (
      <div className="stack stack--tight">
        <span className="subtle">provider default</span>
        {role.requested_reasoning_effort ? (
          <span className="subtle">
            <code className="mono">{role.requested_reasoning_effort}</code> is configured; this
            deployment declares the model does not accept it, so no effort is sent.
          </span>
        ) : null}
      </div>
    );
  }
  return <code className="mono">{role.reasoning_effort}</code>;
}

/** Why this role exists, the identifier it resolved to, and where each value comes from. */
function RoleProvenance({
  role,
  origin,
}: {
  role: ModelRoleConfiguration;
  origin: 'deployment' | 'custom';
}) {
  return (
    <div className="stack stack--tight">
      <p>{role.routing_reason}</p>
      <DetailList narrow>
        <DetailRow label="Model identifier">
          <CopyValue value={role.model} label="Copy model identifier" />
        </DetailRow>
        {origin === 'custom' ? (
          // Deliberately not a copyable variable name: a custom row has no environment
          // variable behind it, and offering one would send somebody grepping their
          // deployment for a string that is not there.
          <DetailRow label="Configured by">
            <span>authored in this setup</span>
          </DetailRow>
        ) : (
          <DetailRow label="Model variable">
            <CopyValue value={role.model_variable} label="Copy variable name" />
          </DetailRow>
        )}
        {origin === 'deployment' && role.reasoning_variable ? (
          <DetailRow label="Effort variable">
            <CopyValue value={role.reasoning_variable} label="Copy variable name" />
          </DetailRow>
        ) : null}
      </DetailList>
      {role.resolved_from_legacy_variable ? (
        // Worth saying where somebody can act on it: this role has no configuration of its
        // own and is following an older, broader variable, so changing that one moves two
        // roles at once.
        <p className="muted">
          <Badge tone="attention">Older variable</Badge> This role has no variable of its own
          set, so it follows <code className="mono">{role.model_variable}</code>. Setting its
          own gives it an independent model.
        </p>
      ) : null}
    </div>
  );
}

type RoleDraft = {
  platform: 'openai' | 'anthropic';
  model: string;
  /** '' means "provider default" — a distinct, chosen option, not an empty field. */
  reasoning_effort: string;
  max_tokens: string;
};

const BLANK_ROLE: RoleDraft = {
  platform: 'anthropic',
  model: '',
  reasoning_effort: '',
  max_tokens: '',
};

function draftFrom(setup: SavedModelSetup | null): Record<string, RoleDraft> {
  const draft: Record<string, RoleDraft> = {};
  for (const role of ROLE_ORDER) {
    const existing = setup?.roles[role];
    draft[role] = existing
      ? {
          platform: existing.platform === 'openai' ? 'openai' : 'anthropic',
          model: existing.model,
          reasoning_effort: existing.reasoning_effort ?? '',
          max_tokens: existing.max_tokens == null ? '' : String(existing.max_tokens),
        }
      : { ...BLANK_ROLE };
  }
  return draft;
}

/**
 * The authoring form: four role rows — platform, model, effort, output bound.
 *
 * The model is a free-text identifier with suggestions, never a closed list: there is no
 * model catalogue in this platform, models arrive from the environment, and a new model must
 * be usable the day the provider ships it. Save-time refusals are the server's own `detail`,
 * rendered verbatim — the AB-Feature-181 refusal is the whole point of the screen.
 */
function SetupEditor({
  setup,
  suggestions,
  onClose,
  onSaved,
}: {
  setup: SavedModelSetup | null;
  suggestions: string[];
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const api = useApi();
  const [name, setName] = useState(setup?.name ?? '');
  const [roles, setRoles] = useState<Record<string, RoleDraft>>(() => draftFrom(setup));
  const [refusal, setRefusal] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);

  const save = useMutation({
    mutationFn: (input: SaveModelSetupInput) =>
      setup ? api.updateModelSetup(setup.setup_id, input) : api.createModelSetup(input),
    onSuccess: async (saved) => {
      setRefusal(null);
      await onSaved();
      // The 181-shape warning is surfaced here, where the choice was made; a setup with no
      // warnings closes without ceremony.
      if (saved.warnings.length > 0) {
        setWarnings(saved.warnings);
      } else {
        onClose();
      }
    },
    onError: (error) => {
      // The server's sentence, verbatim. This is the one place a person meets the refusal,
      // and a paraphrase would soften it.
      setRefusal(
        error instanceof ApiError ? (error.detail ?? userMessage(error)) : String(error),
      );
    },
  });

  const submit = () => {
    const input: SaveModelSetupInput = { name, roles: {} };
    for (const role of ROLE_ORDER) {
      const draft = roles[role]!;
      input.roles[role] = {
        platform: draft.platform,
        model: draft.model,
        reasoning_effort: draft.reasoning_effort === '' ? null : draft.reasoning_effort,
        max_tokens:
          draft.platform === 'anthropic' && draft.max_tokens !== ''
            ? Number(draft.max_tokens)
            : null,
      };
    }
    save.mutate(input);
  };

  const platforms = [...new Set(ROLE_ORDER.map((role) => roles[role]!.platform))];

  return (
    <Drawer
      title={setup ? `Edit setup — ${setup.name}` : 'New model setup'}
      subtitle="Per role: the platform, the model, how hard it thinks, and its output bound"
      onDismiss={onClose}
    >
      {warnings.length > 0 ? (
        <div className="stack">
          <p>
            <Badge tone="done">Saved</Badge>
          </p>
          {warnings.map((warning) => (
            <p key={warning} className="muted">
              <Badge tone="attention">Warning</Badge> {warning}
            </p>
          ))}
          <div className="toolbar">
            <button type="button" className="button button--primary" onClick={onClose}>
              Done
            </button>
          </div>
        </div>
      ) : (
        <form
          className="form"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
          noValidate
        >
          <Field label="Setup name">
            {(id, describedBy) => (
              <input
                id={id}
                aria-describedby={describedBy}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Cheap review"
              />
            )}
          </Field>
          <datalist id="model-setup-suggestions">
            {suggestions.map((model) => (
              <option key={model} value={model} />
            ))}
          </datalist>
          {ROLE_ORDER.map((role) => (
            <RoleEditor
              key={role}
              role={role}
              draft={roles[role]!}
              onChange={(next) => setRoles((current) => ({ ...current, [role]: next }))}
            />
          ))}
          {refusal ? (
            <p className="field__error" role="alert">
              {refusal}
            </p>
          ) : null}
          <div className="toolbar">
            <CredentialChecks platforms={platforms} />
            <span className="toolbar__spacer" />
            <button type="button" className="button button--quiet" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="button button--primary" disabled={save.isPending}>
              {setup ? 'Save changes' : 'Save setup'}
            </button>
          </div>
        </form>
      )}
    </Drawer>
  );
}

function RoleEditor({
  role,
  draft,
  onChange,
}: {
  role: string;
  draft: RoleDraft;
  onChange: (next: RoleDraft) => void;
}) {
  return (
    <fieldset className="fieldset">
      <legend>{ROLE_LABELS[role] ?? role}</legend>
      <div className="row">
        <Field label="Platform">
          {(id, describedBy) => (
            <select
              id={id}
              aria-describedby={describedBy}
              value={draft.platform}
              onChange={(event) =>
                onChange({
                  ...draft,
                  platform: event.target.value === 'openai' ? 'openai' : 'anthropic',
                })
              }
            >
              <option value="anthropic">Claude</option>
              <option value="openai">OpenAI</option>
            </select>
          )}
        </Field>
        <Field label="Model identifier">
          {(id, describedBy) => (
            <input
              id={id}
              aria-describedby={describedBy}
              value={draft.model}
              list="model-setup-suggestions"
              onChange={(event) => onChange({ ...draft, model: event.target.value })}
              placeholder="any identifier the provider ships"
            />
          )}
        </Field>
        <Field label="Reasoning effort">
          {(id, describedBy) => (
            <select
              id={id}
              aria-describedby={describedBy}
              value={draft.reasoning_effort}
              onChange={(event) => onChange({ ...draft, reasoning_effort: event.target.value })}
            >
              <option value="">provider default</option>
              {REASONING_LEVELS.map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          )}
        </Field>
        {draft.platform === 'anthropic' ? (
          <Field
            label="Max output tokens"
            hint="Required: the Messages API rejects a request without a bound, and thinking spends the same bound the answer does."
          >
            {(id, describedBy) => (
              <input
                id={id}
                aria-describedby={describedBy}
                inputMode="numeric"
                value={draft.max_tokens}
                onChange={(event) => onChange({ ...draft, max_tokens: event.target.value })}
                placeholder="e.g. 64000"
              />
            )}
          </Field>
        ) : (
          // Absent with the reason stated where the choice is made: the Responses API is not
          // sent an output bound, so there is nothing to type.
          <p className="subtle">No output bound — the Responses API is not sent one.</p>
        )}
      </div>
    </fieldset>
  );
}

/**
 * Ask each named provider whether the stored key still works, on a button press.
 *
 * Verification stays in the authoring UI deliberately — a person pinning a second provider
 * for the first time is exactly who wants to press it — and it never gates a save: every way
 * of not getting an answer is `unknown`, and nothing refuses on `unknown`.
 */
function CredentialChecks({ platforms }: { platforms: string[] }) {
  const api = useApi();
  const [verdicts, setVerdicts] = useState<Record<string, string>>({});
  const check = useMutation({
    mutationFn: async (provider: string) => {
      const result = await api.checkCredential(provider, { verify: true });
      return { provider, detail: result.detail };
    },
    onSuccess: ({ provider, detail }) => {
      setVerdicts((current) => ({ ...current, [provider]: detail }));
    },
  });
  return (
    <span className="stack stack--tight">
      <span>
        {platforms.map((provider) => (
          <button
            key={provider}
            type="button"
            className="button button--small button--quiet"
            onClick={() => check.mutate(provider)}
            disabled={check.isPending}
          >
            {`Verify ${provider === 'anthropic' ? 'Claude' : 'OpenAI'} credential`}
          </button>
        ))}
      </span>
      {Object.entries(verdicts).map(([provider, detail]) => (
        <span key={provider} className="subtle">
          {detail}
        </span>
      ))}
    </span>
  );
}
