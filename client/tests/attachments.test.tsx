import { describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { AppProviders } from '@/app/providers';
import { NewFeaturePage } from '@/pages/NewFeaturePage';
import { PrdTab } from '@/features/feature-workspace/DocumentTab';
import type { FeatureApi } from '@/api/features';
import { ApiError } from '@/api/errors';
import { ONE_PIXEL_PNG, SETUP_READY, stubApi } from './fixtures';
import { markerFromFilename, toStartFeatureInput, DEFAULT_VALUES } from '@/features/new-feature/schema';

/**
 * Attaching a picture to a submission, and reading one back.
 *
 * The cycle these cover is upload, insert, rename, remove -- and rename is the one that
 * matters most, because it is the only edit that silently invalidates prose somebody already
 * wrote. A marker renamed without its references rewritten leaves the submission pointing at
 * a name that no longer exists, and the server refuses the whole thing naming a marker the
 * author does not remember typing.
 */

/** The suite's committed one-pixel PNG, so a `File` here is an actual image. */
const PNG_BYTES = ONE_PIXEL_PNG;

function pngFile(name: string, size = PNG_BYTES.byteLength): File {
  // Padded to a requested size where a test is about the caps, so the padding is only ever
  // the thing being measured and the leading bytes stay a real PNG header.
  const bytes =
    size <= PNG_BYTES.byteLength
      ? PNG_BYTES
      : new Uint8Array([...PNG_BYTES, ...new Uint8Array(size - PNG_BYTES.byteLength)]);
  return new File([bytes], name, { type: 'image/png' });
}

function uploaded(name: string, byteSize = PNG_BYTES.byteLength) {
  return {
    attachment_id: `attachment-${name}`,
    filename: name,
    media_type: 'image/png',
    byte_size: byteSize,
    sha256: 'a'.repeat(64),
  };
}

/** A setup state whose pairings all read images, so the notice is not the subject. */
const SEEING = {
  ...SETUP_READY,
  agent_platforms: SETUP_READY.agent_platforms.map((item) => ({ ...item, vision_capable: true })),
};

function renderForm(api: Partial<FeatureApi>) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <AppProviders api={stubApi({ getSetupState: async () => SEEING, ...api })} queryClient={queryClient}>
      <MemoryRouter
        initialEntries={['/features/new']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes>
          <Route path="/features/new" element={<NewFeaturePage />} />
          <Route path="/features/:featureId" element={<p>Workspace for feature</p>} />
        </Routes>
      </MemoryRouter>
    </AppProviders>,
  );
}

async function attach(user: ReturnType<typeof userEvent.setup>, ...files: File[]) {
  const picker = await screen.findByLabelText('Choose images');
  await user.upload(picker, files);
}

function svgFile(): File {
  return new File(['<svg xmlns="http://www.w3.org/2000/svg"/>'], 'mock.svg', {
    type: 'image/svg+xml',
  });
}

/** Drop files on the zone, which is the one path an unaccepted type can arrive by. */
async function drop(...files: File[]) {
  const zone = document.querySelector('.dropzone');
  if (zone === null) throw new Error('no drop zone rendered');
  await act(async () => {
    fireEvent.drop(zone, { dataTransfer: { files, types: ['Files'] } });
  });
}

describe('attaching an image to a submission', () => {
  it('uploads on drop, shows the file, and slugs a marker from its name', async () => {
    const user = userEvent.setup();
    const uploadAttachment = vi.fn(async (file: File) => uploaded(file.name));
    renderForm({ uploadAttachment });

    await attach(user, pngFile('Login Error State.png'));

    await waitFor(() => expect(uploadAttachment).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('Login Error State.png')).toBeInTheDocument();
    // Slugged from the filename, editable, and shown as the reference to write.
    const marker = screen.getByLabelText('Marker') as HTMLInputElement;
    expect(marker.value).toBe('login-error-state');
    expect(screen.getByText(/\[image:login-error-state\]/)).toBeInTheDocument();
  });

  it('writes the reference into the field that last had focus', async () => {
    const user = userEvent.setup();
    renderForm({ uploadAttachment: async (file: File) => uploaded(file.name) });

    await attach(user, pngFile('login.png'));
    await screen.findByText('login.png');
    const problem = screen.getByLabelText('Problem statement');
    await user.click(problem);
    await user.type(problem, 'Login fails silently. ');
    await user.click(screen.getByRole('button', { name: 'Insert reference' }));

    await waitFor(() =>
      expect((problem as HTMLTextAreaElement).value).toBe(
        'Login fails silently. [image:login]',
      ),
    );
  });

  it('rewrites the prose when a marker is renamed, and says how many references moved', async () => {
    const user = userEvent.setup();
    renderForm({ uploadAttachment: async (file: File) => uploaded(file.name) });

    await attach(user, pngFile('login.png'));
    await screen.findByText('login.png');
    const problem = screen.getByLabelText('Problem statement');
    await user.click(problem);
    // `paste` rather than `type`: user-event reads `[` as the start of a special-key
    // descriptor, and the whole point of this text is that it contains two of them.
    await user.paste('See [image:login] and again [image:login].');

    // Select-all-and-retype, which is how anybody edits a short field -- and the shape that
    // rewriting per keystroke turned into `[image:]`. The rewrite lands when the field is
    // finished, so it is committed by tabbing out of it.
    const marker = screen.getByLabelText('Marker');
    await user.clear(marker);
    await user.type(marker, 'login-error');
    await user.tab();

    await waitFor(() =>
      expect((problem as HTMLTextAreaElement).value).toBe(
        'See [image:login-error] and again [image:login-error].',
      ),
    );
    // And it says so: a silent rewrite of somebody's own prose is worse than no rewrite.
    expect(await screen.findByText(/rewrote 2 references/)).toBeInTheDocument();
  });

  it('deletes the upload when the row is removed, and warns if the prose still points at it', async () => {
    const user = userEvent.setup();
    const deleteAttachment = vi.fn(async () => undefined);
    renderForm({ uploadAttachment: async (file: File) => uploaded(file.name), deleteAttachment });

    await attach(user, pngFile('login.png'));
    await screen.findByText('login.png');
    const problem = screen.getByLabelText('Problem statement');
    await user.click(problem);
    await user.paste('See [image:login].');
    await user.click(screen.getByRole('button', { name: 'Remove' }));

    await waitFor(() => expect(deleteAttachment).toHaveBeenCalledWith('attachment-login.png'));
    expect(await screen.findByText(/still refers to \[image:login\]/)).toBeInTheDocument();
  });

  it('refuses a dropped SVG and an oversized file without uploading either', async () => {
    const uploadAttachment = vi.fn(async (file: File) => uploaded(file.name));
    renderForm({ uploadAttachment });
    await screen.findByLabelText('Choose images');

    // Dropped rather than picked, deliberately: both a real file picker and `user.upload`
    // filter on the input's `accept`, so a drop is how an unaccepted type actually reaches
    // the handler -- and is therefore the path the client-side check exists for.
    await drop(pngFile('huge.png', 6 * 1024 * 1024), svgFile());

    // Its own sentence, because "unsupported type" sends somebody to export another SVG.
    expect(await screen.findByText(/SVG is markup, not an image/)).toBeInTheDocument();
    expect(await screen.findByText(/larger than 5 MiB/)).toBeInTheDocument();
    expect(uploadAttachment).not.toHaveBeenCalled();
  });

  it('refuses a ninth image against the count the server published', async () => {
    const user = userEvent.setup();
    const uploadAttachment = vi.fn(async (file: File) => uploaded(file.name));
    renderForm({ uploadAttachment });

    await attach(user, ...Array.from({ length: 9 }, (_, index) => pngFile(`shot-${index}.png`)));

    expect(await screen.findByText(/at most 8 images/)).toBeInTheDocument();
    expect(uploadAttachment).toHaveBeenCalledTimes(8);
  });

  it('surfaces the server’s own refusal when an upload is rejected', async () => {
    const user = userEvent.setup();
    renderForm({
      uploadAttachment: async () => {
        throw new ApiError('validation', 'Rejected', 422, 'this file is not an image the platform accepts');
      },
    });

    await attach(user, pngFile('claims-to-be.png'));

    expect(
      await screen.findByText(/not an image the platform accepts/),
    ).toBeInTheDocument();
  });

  it('uploads a screenshot pasted while typing in a prose field, not only on the drop zone', async () => {
    const user = userEvent.setup();
    const uploadAttachment = vi.fn(async (file: File) => uploaded(file.name));
    renderForm({ uploadAttachment });

    // The paste lands where a paste actually lands: in the field somebody is writing in,
    // which never gives the drop zone focus. The listener is on the document.
    const problem = await screen.findByLabelText('Problem statement');
    await user.click(problem);
    await act(async () => {
      fireEvent.paste(problem, {
        clipboardData: { files: [pngFile('pasted-screenshot.png')], types: ['Files'] },
      });
    });

    await waitFor(() => expect(uploadAttachment).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('pasted-screenshot.png')).toBeInTheDocument();
  });

  it('leaves a text-only paste alone and uploads nothing', async () => {
    const user = userEvent.setup();
    const uploadAttachment = vi.fn(async (file: File) => uploaded(file.name));
    renderForm({ uploadAttachment });

    const problem = await screen.findByLabelText('Problem statement');
    await user.click(problem);
    await user.paste('Just a sentence, no screenshot.');

    expect((problem as HTMLTextAreaElement).value).toBe('Just a sentence, no screenshot.');
    expect(uploadAttachment).not.toHaveBeenCalled();
  });

  it('says at the selector when the chosen model cannot read the images', async () => {
    const user = userEvent.setup();
    renderForm({
      getSetupState: async () => ({
        ...SETUP_READY,
        agent_platforms: SETUP_READY.agent_platforms.map((item) => ({
          ...item,
          vision_capable: false,
        })),
      }),
      uploadAttachment: async (file: File) => uploaded(file.name),
    });

    // No images, no notice: the check is about images and nothing else.
    await screen.findByLabelText('Choose images');
    expect(screen.queryByText(/does not read images/)).not.toBeInTheDocument();

    await attach(user, pngFile('login.png'));

    expect(await screen.findByText(/does not read images/)).toBeInTheDocument();
  });

  it('clears the notice when the selection changes to one that reads them', async () => {
    const user = userEvent.setup();
    const mixed = {
      ...SETUP_READY,
      agent_platforms: SETUP_READY.agent_platforms.map((item) => ({
        ...item,
        vision_capable: item.performance_tier === 'high',
      })),
    };
    renderForm({
      getSetupState: async () => mixed,
      uploadAttachment: async (file: File) => uploaded(file.name),
    });

    await attach(user, pngFile('login.png'));
    expect(await screen.findByText(/does not read images/)).toBeInTheDocument();

    const selector = screen.getByLabelText('Platform and performance tier');
    await user.selectOptions(selector, 'openai:high');

    await waitFor(() => expect(screen.queryByText(/does not read images/)).not.toBeInTheDocument());
  });

  it('neither warns nor blocks a mock submission, which the server exempts from the check', async () => {
    const user = userEvent.setup();
    const createFeature = vi.fn().mockResolvedValue({ feature_id: 'feature-mock-images' });
    renderForm({
      createFeature,
      getSetupState: async () => ({
        ...SETUP_READY,
        agent_platforms: SETUP_READY.agent_platforms.map((item) => ({
          ...item,
          vision_capable: false,
        })),
      }),
      uploadAttachment: async (file: File) => uploaded(file.name),
    });

    await attach(user, pngFile('login.png'));
    expect(await screen.findByText(/does not read images/)).toBeInTheDocument();

    // Mock selects no model and reaches no provider, which is exactly why the server accepts
    // it: the notice saying "will be refused" would be wrong, so it goes.
    await user.selectOptions(screen.getByLabelText('Mode'), 'mock');
    await waitFor(() =>
      expect(screen.queryByText(/does not read images/)).not.toBeInTheDocument(),
    );

    // And the submit guard steps aside for the same reason the server's check does.
    await user.type(screen.getByLabelText('Title'), 'Deactivate ad units');
    await user.type(screen.getByLabelText('Problem statement'), 'Operators cannot deactivate.');
    await user.click(screen.getByRole('button', { name: /admanager_console-2\.0/ }));
    await user.click(screen.getByRole('button', { name: 'Create feature' }));

    await waitFor(() => expect(createFeature).toHaveBeenCalledTimes(1));
    const [input] = createFeature.mock.calls[0] as [{ execution_mode: string }];
    expect(input.execution_mode).toBe('mock');
  });
});

describe('the reference syntax the form and the server share', () => {
  it('slugs a filename into a marker, and falls back rather than producing an invalid one', () => {
    expect(markerFromFilename('Login Error.png', 0)).toBe('login-error');
    expect(markerFromFilename('CHECKOUT_v2.final.jpeg', 0)).toBe('checkout-v2-final');
    // Nothing a marker can be made of: a positional name, not an invalid marker.
    expect(markerFromFilename('.png', 3)).toBe('image-4');
    expect(markerFromFilename('日本語.png', 0)).toBe('image-1');
  });

  it('omits the attachments key entirely when there are none', () => {
    const withNone = toStartFeatureInput({ ...DEFAULT_VALUES, title: 't', problem_statement: 'p' });
    expect('attachments' in (withNone.prd as Record<string, unknown>)).toBe(false);

    const withOne = toStartFeatureInput({
      ...DEFAULT_VALUES,
      title: 't',
      problem_statement: 'p [image:login]',
      attachments: [
        {
          attachment_id: 'attachment-1',
          marker: 'login',
          caption: '  the login screen  ',
          filename: 'login.png',
          byte_size: 70,
          media_type: 'image/png',
        },
      ],
    });
    // Three fields travel, and the display-only ones do not.
    expect((withOne.prd as Record<string, unknown>).attachments).toEqual([
      { attachment_id: 'attachment-1', marker: 'login', caption: 'the login screen' },
    ]);
  });
});

describe('reading the images a submission attached', () => {
  const PRD_PAYLOAD = {
    artifact_type: 'prd',
    title: 'Login audit trail',
    problem_statement: 'Login fails with no explanation [image:login-error] and [image:missing].',
    goals: [],
    user_stories: [],
    requirements: [],
    constraints: [],
    out_of_scope: [],
    stakeholders: [],
    attachments: [
      {
        attachment_id: 'attachment-1',
        marker: 'login-error',
        caption: 'The error a user sees',
        filename: 'login-error.png',
        media_type: 'image/png',
        byte_size: 70,
        sha256: 'b'.repeat(64),
      },
    ],
  };

  /** The submitted PRD as the artifacts endpoint returns it, envelope and all. */
  function prdArtifact(featureId: string) {
    return {
      artifact_id: '001_prd.json',
      artifact_type: 'prd',
      schema_version: '1.0.0',
      workflow_id: featureId,
      producer: 'api',
      timestamp: '2026-09-07T12:00:00Z',
      metadata: { source: 'api' } as Record<string, unknown>,
      validation_status: 'valid',
      payload: PRD_PAYLOAD as Record<string, unknown>,
    };
  }

  function renderDocument(api: Partial<FeatureApi> = {}) {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <AppProviders
        api={stubApi({
          listArtifacts: async (featureId: string) => ({
            feature_id: featureId,
            artifacts: [prdArtifact(featureId)],
          }),
          getArtifact: async (featureId: string) => prdArtifact(featureId),
          getWorkstreams: async (featureId: string) => ({ feature_id: featureId, workstreams: [] }),
          getClarification: async (featureId: string) => ({
            feature_id: featureId,
            clarification_rounds: 0,
            max_clarification_rounds: 3,
            awaiting_answers: false,
            technical_prd_artifact_id: null,
            questions: [],
            previous_answers: {},
            design_conflicts: [],
          }),
          getAttachmentBlob: async () => new Blob([PNG_BYTES], { type: 'image/png' }),
          ...api,
        })}
        queryClient={queryClient}
      >
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <PrdTab featureId="feature-1" />
        </MemoryRouter>
      </AppProviders>,
    );
  }

  it('shows the strip with its captions and fetches each image with the token', async () => {
    const getAttachmentBlob = vi.fn(async () => new Blob([PNG_BYTES], { type: 'image/png' }));
    renderDocument({ getAttachmentBlob });

    expect(await screen.findByText('Screens and mock-ups')).toBeInTheDocument();
    expect(screen.getByText('The error a user sees')).toBeInTheDocument();
    await waitFor(() => expect(getAttachmentBlob).toHaveBeenCalledWith('attachment-1'));
    // Not an `<img src>` pointed at the API: every read is authenticated.
    await waitFor(() =>
      expect(screen.getByRole('img', { name: 'login-error.png' })).toBeInTheDocument(),
    );
  });

  it('renders a known marker as a chip and an unknown one as plain text', async () => {
    renderDocument();

    // Two references in the prose; only one has an image, and the other is left as written.
    expect(
      await screen.findByRole('button', { name: '[image:login-error]' }),
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '[image:missing]' })).not.toBeInTheDocument();
    expect(screen.getByText(/\[image:missing\]/)).toBeInTheDocument();
  });

  it('opens the full image in the dialog when a thumbnail is pressed', async () => {
    const user = userEvent.setup();
    renderDocument();

    await user.click(await screen.findByRole('button', { name: 'Open login-error.png' }));

    const dialog = await screen.findByRole('dialog', { name: 'login-error.png' });
    expect(within(dialog).getByText(/image\/png/)).toBeInTheDocument();
    // The hash is quoted: it is what the artifact records, and what survives a purge.
    expect(within(dialog).getByText(/sha256 bbbbbbbbbbbb/)).toBeInTheDocument();
  });
});
