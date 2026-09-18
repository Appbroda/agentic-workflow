import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiContext } from '@/app/api-context';
import { SessionProvider } from '@/app/session';
import { SettingsView } from '@/features/settings/SettingsView';
import { PrdTab } from '@/features/feature-workspace/DocumentTab';
import { ArtifactViewer } from '@/components/artifacts/ArtifactViewer';
import { ARTIFACT_RENDERERS } from '@/components/artifacts/renderers';
import { artifactSchema } from '@/schemas/feature';
import type { DesignSource } from '@/schemas/feature';
import { stubApi } from './fixtures';
import { objectUrlRegistry } from './setup';
import capturedSnapshot from './fixtures/design-snapshot.json';
import capturedWithOmissions from './fixtures/design-snapshot.omissions.json';

/**
 * The design half of the console: where designs come from, and what was resolved.
 *
 * The renderer is checked against payloads captured from the real handler — the real start
 * path, the real resolution step, the shipped extraction over a live Figma capture, and the
 * real artifact endpoint's own response model. A hand-written double is how this client
 * silently lost a dozen fields once, and here it would hide the only thing the renderer can
 * actually get wrong: whether the names the server publishes are the names it reads.
 *
 * Both fixtures are loaded through `artifactSchema`, the same parse the workspace performs, so
 * a field the server renames or drops fails at parse rather than at an assertion written to
 * agree with it.
 */

const READY = {
  status: 'ok',
  build_revision: 'abcdef0123456789',
  workflow_schema_version: '1.0',
  runtime_compatible: true,
};

const BASE = {
  getReadiness: async () => READY,
  listCredentials: async () => ({
    credentials: [{ provider: 'figma', configured: true, hint: 'ab12' }],
  }),
};

/** An operator who may save the configuration; the server publishes the permission. */
const OPERATOR = {
  getMe: async () => ({
    actor_id: 'user-1',
    display_name: 'Alex',
    authentication: 'user_token',
    roles: ['operator'],
    permissions: ['design_source:manage'],
  }),
};

function configured(overrides: Partial<DesignSource> = {}): DesignSource {
  return {
    configured: true,
    enabled: true,
    token_owner_id: 'platform-admin',
    file_allowlist: ['28gd2JrZO28FCN9PCKM4qK'],
    file_allowlist_permits_any_file: false,
    status: 'active',
    status_reason: null,
    credential_configured: true,
    credential_hint: 'ab12',
    updated_by: 'platform-admin',
    updated_at: '2026-09-07T10:00:00Z',
    ...overrides,
  };
}

function renderSettings(api: Record<string, unknown>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ApiContext.Provider value={stubApi(api)}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <SessionProvider>
          <SettingsView />
          </SessionProvider>
        </MemoryRouter>
      </ApiContext.Provider>
    </QueryClientProvider>,
  );
}

function renderSnapshot(payload: unknown, api: Record<string, unknown> = {}) {
  const artifact = artifactSchema.parse(payload);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ApiContext.Provider value={stubApi(api)}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <SessionProvider>
          <Routes>
            <Route
              path="*"
              element={
                <ArtifactViewer
                  artifact={artifact}
                  renderers={ARTIFACT_RENDERERS}
                  featureId="feature-design-snapshot"
                />
              }
            />
          </Routes>
          </SessionProvider>
        </MemoryRouter>
      </ApiContext.Provider>
    </QueryClientProvider>,
  );
}

describe('the design source panel', () => {
  it('renders the configuration and never holds the token', async () => {
    const getDesignSource = vi.fn().mockResolvedValue(configured());
    const saveDesignSource = vi.fn().mockResolvedValue(configured());
    renderSettings({ ...BASE, ...OPERATOR, getDesignSource, saveDesignSource });
    const user = userEvent.setup();

    // The stored values populate the form, and the token is present only as its hint.
    const allowlist = await screen.findByLabelText('Permitted file keys');
    expect(allowlist).toHaveValue('28gd2JrZO28FCN9PCKM4qK');
    // The hint and only the hint: the credentials panel above renders the same four
    // characters for the same row, which is why this matches on the panel's own sentence.
    expect(screen.getByText(/Figma token: configured/)).toBeInTheDocument();
    expect(screen.getAllByText(/…ab12|ab12/).length).toBeGreaterThan(0);

    await user.clear(allowlist);
    await user.type(allowlist, 'VGULlnz44R0Ooe4FZKDxlhh4');
    await user.click(screen.getByRole('button', { name: 'Save design source' }));

    await waitFor(() => expect(saveDesignSource).toHaveBeenCalled());
    expect(saveDesignSource.mock.calls[0]?.[0]).toEqual({
      enabled: true,
      file_allowlist: ['VGULlnz44R0Ooe4FZKDxlhh4'],
      // Kept rather than re-defaulted: a different operator re-saving must not silently
      // re-point resolution at their own stored credential.
      token_owner_id: 'platform-admin',
    });
  });

  it('says which of the two things an empty allowlist means', async () => {
    renderSettings({
      ...BASE,
      ...OPERATOR,
      getDesignSource: async () =>
        configured({ file_allowlist: [], file_allowlist_permits_any_file: true }),
    });

    // The emptiness is consequential: for a shared token this is the wrong default, so the
    // panel says what it is rather than showing an empty list and leaving it to be inferred.
    expect(await screen.findByText('Any file this token can read')).toBeInTheDocument();
  });

  it('shows the platform’s own reason when the source has degraded', async () => {
    renderSettings({
      ...BASE,
      ...OPERATOR,
      getDesignSource: async () =>
        configured({
          status: 'degraded',
          status_reason: 'Figma refused the stored token. It has expired or been revoked.',
        }),
    });

    expect(await screen.findByText('Design resolution is paused')).toBeInTheDocument();
    // Verbatim from the server, which is where the wording is owned and tested.
    expect(screen.getByText(/Figma refused the stored token/)).toBeInTheDocument();
  });

  it('points at the credentials panel when no token is stored', async () => {
    renderSettings({
      ...BASE,
      ...OPERATOR,
      getDesignSource: async () =>
        configured({ credential_configured: false, credential_hint: '' }),
    });

    expect(await screen.findByText('No Figma token stored')).toBeInTheDocument();
    expect(screen.getByText(/refused at submission until it is/)).toBeInTheDocument();
  });

  it('re-reads the configuration after a check, because a refusal degrades it', async () => {
    const getDesignSource = vi
      .fn()
      .mockResolvedValueOnce(configured())
      .mockResolvedValue(configured({ status: 'degraded', status_reason: 'Figma said no.' }));
    const checkDesignSource = vi.fn().mockResolvedValue({
      provider: 'figma',
      configured: true,
      usable: true,
      verified: 'refused',
      detail: 'The stored credential is readable, but the provider refused it.',
    });
    renderSettings({ ...BASE, ...OPERATOR, getDesignSource, checkDesignSource });
    const user = userEvent.setup();

    await screen.findByLabelText('Permitted file keys');
    await user.click(screen.getByRole('button', { name: 'Test connection' }));

    expect(await screen.findByText(/the provider refused it/)).toBeInTheDocument();
    // The banner would otherwise show the state before the button was pressed.
    expect(await screen.findByText('Design resolution is paused')).toBeInTheDocument();
  });

  it('is not shown at all to somebody without the permission', async () => {
    // `design_source:manage` moved to administrators when workspaces became isolated. There
    // is one enabled design source for the whole deployment, so an operator holding this
    // could point everybody's citations at their own Figma account and choose which files
    // anybody may cite. It used to render read-only for them; now it does not render.
    renderSettings({
      ...BASE,
      getMe: async () => ({
        actor_id: 'user-2',
        display_name: 'Sam',
        authentication: 'user_token',
        roles: ['viewer'],
        permissions: [],
      }),
      getDesignSource: async () => configured(),
    });

    // The page rendered; the panel did not.
    expect(await screen.findByText('Sam')).toBeInTheDocument();
    expect(screen.queryByText('An operator saves this configuration.')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Permitted file keys')).not.toBeInTheDocument();
  });

  it('reads a 503 as a fact about the deployment rather than a failure', async () => {
    const { ApiError } = await import('@/api/errors');
    renderSettings({
      ...BASE,
      ...OPERATOR,
      getDesignSource: async () => {
        throw new ApiError('server', 'unavailable', 503);
      },
    });

    expect(
      await screen.findByText('This deployment does not resolve design references.'),
    ).toBeInTheDocument();
  });
});

describe('the design snapshot renderer', () => {
  it('renders every frame the real handler published, with its names and its text', async () => {
    renderSnapshot(capturedSnapshot, {
      getDesignPreview: async () => new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }),
    });

    // Three frames, from the live capture, quoted by the shipped extraction.
    expect(await screen.findByText('Frames (3)')).toBeInTheDocument();
    // The frame's own name from the design tree, and the path a reader uses to find it.
    expect(screen.getAllByText('#FClock').length).toBeGreaterThan(0);
    // The text a text node actually contains, which is one of the few design criteria a
    // reviewer reading a diff can check.
    expect(screen.getAllByText('10:10').length).toBeGreaterThan(0);
    // The provenance the preview is pinned to.
    expect(screen.getByText(/2315052197036992991/)).toBeInTheDocument();
    expect(screen.getByText('the names on the nodes')).toBeInTheDocument();
  });

  it('shows what the resolution could not show, as content and not a footnote', async () => {
    renderSnapshot(capturedWithOmissions, {
      getDesignPreview: async () => new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }),
    });

    // First-class: an unreported cap reads as "the whole design was considered".
    expect(
      await screen.findByText('Frames this snapshot does not contain'),
    ).toBeInTheDocument();
    expect(screen.getByText(/did not fit/)).toBeInTheDocument();
    // The bound that bit and the size it measured, both from the real payload.
    expect(screen.getByText(/over_per_node_character_bound/)).toBeInTheDocument();
    expect(screen.getByText(/8,994 characters/)).toBeInTheDocument();
    // And the one frame that did fit is still shown.
    expect(screen.getByText('Frames (1)')).toBeInTheDocument();
  });

  it('renders each preview as an image with a blob: src, and revokes it on unmount', async () => {
    // The assertion the earlier tests were missing: that the success path actually produces
    // an <img>. Before the setup polyfill, jsdom had no `URL.createObjectURL`, the query
    // function threw on it, and every "preview" test was exercising the failure branch.
    objectUrlRegistry.reset();
    const view = renderSnapshot(capturedSnapshot, {
      getDesignPreview: async () =>
        new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], { type: 'image/png' }),
    });

    const images = await screen.findAllByRole('img', { name: /^Preview of/ });
    expect(images).toHaveLength(3);
    for (const image of images) {
      expect(image.getAttribute('src')).toMatch(/^blob:/);
    }
    // The object URL belongs to the mounted component, so unmounting releases every one of
    // them — the react-query cache holds only the Blob.
    const created = [...objectUrlRegistry.created];
    expect(created.length).toBeGreaterThanOrEqual(3);
    view.unmount();
    for (const url of created) {
      expect(objectUrlRegistry.revoked).toContain(url);
    }
  });

  it('says a preview could not be rendered without implying the design is missing', async () => {
    const { ApiError } = await import('@/api/errors');
    renderSnapshot(capturedSnapshot, {
      getDesignPreview: async () => {
        // An expired render URL, which is what the server answers 504 for.
        throw new ApiError('server', 'gateway timeout', 504);
      },
    });

    expect(
      (await screen.findAllByText(/could not be rendered just now/)).length,
    ).toBeGreaterThan(0);
    // The text is what every judge was given, and it is still there.
    expect(screen.getAllByText('#FClock').length).toBeGreaterThan(0);
    expect(screen.getAllByText('10:10').length).toBeGreaterThan(0);
  });
});

describe('the design as a third document beside the two PRDs', () => {
  function renderPrdTab(doc: string, api: Record<string, unknown>) {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={queryClient}>
        <ApiContext.Provider value={stubApi(api)}>
          <MemoryRouter
            initialEntries={[`/features/feature-design-snapshot/prd${doc}`]}
            future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
          >
            <Routes>
              <Route
                path="/features/:featureId/prd"
                element={<PrdTab featureId="feature-design-snapshot" />}
              />
            </Routes>
          </MemoryRouter>
        </ApiContext.Provider>
      </QueryClientProvider>,
    );
  }

  const NO_ARTIFACTS = {
    getClarification: async () => ({
      feature_id: 'feature-design-snapshot',
      questions: [],
      previous_answers: {},
    }),
    getWorkstreams: async () => ({ feature_id: 'feature-design-snapshot', workstreams: [] }),
    listArtifacts: async () => ({ feature_id: 'feature-design-snapshot', artifacts: [] }),
  };

  it('offers the design beside the original and the technical PRD', async () => {
    renderPrdTab('', NO_ARTIFACTS);

    // Three readings of one request: what was asked for, what the platform understood, and
    // what it was meant to look like.
    const control = await screen.findByLabelText('Document');
    expect(control).toHaveTextContent('Original PRD');
    expect(control).toHaveTextContent('Technical PRD');
    expect(control).toHaveTextContent('Design');
  });

  it('renders “no design was cited” rather than disappearing', async () => {
    renderPrdTab('?doc=design', NO_ARTIFACTS);

    // The absence has to be legible: a feature nobody attached a mock to and a feature whose
    // mock the platform ignored would otherwise look identical.
    expect(await screen.findByText('No design was cited')).toBeInTheDocument();
    expect(screen.getByText(/worked from the prose above/)).toBeInTheDocument();
  });

  it('renders the snapshot the real handler published when there is one', async () => {
    renderPrdTab('?doc=design', {
      ...NO_ARTIFACTS,
      listArtifacts: async (_id: string, options?: { artifactType?: string }) => ({
        feature_id: 'feature-design-snapshot',
        artifacts:
          options?.artifactType === 'design_snapshot' ? [artifactSchema.parse(capturedSnapshot)] : [],
      }),
      getArtifact: async () => artifactSchema.parse(capturedSnapshot),
      getDesignPreview: async () => new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }),
    });

    // The ordinary artifact viewer, so the raw envelope stays one control away like every
    // other artifact — and the frames are the ones the live capture actually contains.
    expect(await screen.findByText('Frames (3)')).toBeInTheDocument();
    expect(screen.getAllByText('#FClock').length).toBeGreaterThan(0);
  });
});
