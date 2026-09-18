import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useFieldArray, type Control, type UseFormReturn } from 'react-hook-form';
import { useQuery } from '@tanstack/react-query';
import { useApi } from '@/app/api-context';
import { ApiError, userMessage } from '@/api/errors';
import { Badge } from '@/components/ui/Badge';
// The same component the reading view uses. Every attachment read is authenticated, so a
// thumbnail is a fetch with a token rather than an `<img src>`, and that logic exists once.
import { AttachmentImage } from '@/components/artifacts/attachments';
import { Field, Fieldset } from './fields';
import {
  MARKER_PATTERN,
  imageReference,
  markerFromFilename,
  markerReferences,
  referencedMarkers,
  type NewFeatureValues,
} from './schema';

/**
 * Attaching a picture to a submission.
 *
 * A submitter can describe a screen; this is how they show one. Three ways in, because a
 * screenshot travels differently depending on where it came from: dropped from a folder,
 * chosen from a picker, or pasted straight out of a screenshot tool — which is how one
 * actually arrives most of the time, and the reason the paste handler exists at all.
 *
 * Each accepted file uploads immediately rather than at submit. Somebody sees a thumbnail
 * while they are still writing, a rejected file is rejected at once instead of after the
 * whole form is filled in, and five megabytes are not re-sent on every re-validation.
 *
 * The one edit that needs care is renaming a marker: prose already written points at the old
 * name, and a rename that left it alone would silently invalidate the submission. So a rename
 * rewrites every reference in the form's text fields, and says how many it rewrote.
 */

/**
 * The caps the shipped server enforces, used until it has answered with its own.
 *
 * A copy, and knowingly so: the alternative is a form with no numbers in its helper text
 * until a request returns. The server's answer replaces these the moment it arrives, and the
 * server is the only authority -- these numbers refuse nothing the server would accept.
 */
const DEFAULT_LIMITS = {
  max_attachment_bytes: 5 * 1024 * 1024,
  max_attachments_per_feature: 8,
  max_attachment_bytes_per_feature: 20 * 1024 * 1024,
  accepted_media_types: ['image/png', 'image/jpeg', 'image/webp'],
};

/** What the last-focused text field was, so `Insert reference` knows where to write. */
type Target = { name: string; element: HTMLInputElement | HTMLTextAreaElement } | null;

export function AttachmentsField({
  control,
  form,
  visionCapable,
  selectionLabel,
  mockMode,
}: {
  control: Control<NewFeatureValues>;
  form: UseFormReturn<NewFeatureValues>;
  /** Whether the selected pairing or setup reads images. `undefined` while unresolved. */
  visionCapable: boolean | undefined;
  /** What that selection is called, so the notice names it rather than describing it. */
  selectionLabel: string;
  /**
   * Whether the form is set to mock execution, which the server exempts from the vision
   * check: a mock feature selects no model and reaches no provider, so nothing would refuse
   * it and the "will be refused" notice would be wrong.
   */
  mockMode: boolean;
}) {
  const api = useApi();
  const { fields, append, remove } = useFieldArray({ control, name: 'attachments' });
  const [failures, setFailures] = useState<string[]>([]);
  const [notes, setNotes] = useState<string[]>([]);
  const [busy, setBusy] = useState(0);
  const [over, setOver] = useState(false);
  const picker = useRef<HTMLInputElement>(null);
  const target = useRef<Target>(null);
  // What a marker field held when it gained focus, which is what a rename is measured from.
  const focusedMarker = useRef('');

  // Read rather than restated, so the helper text and the client-side refusals are the
  // numbers the server will actually hold this submission to. Best effort: a server that
  // does not publish them leaves the schema's defaults, which are the shipped values.
  const limits = useQuery({
    queryKey: ['attachment-limits'],
    queryFn: ({ signal }) => api.getAttachmentLimits(signal),
    retry: false,
    staleTime: Infinity,
  });
  // Memoised because the upload handler closes over it: a fresh object literal on every
  // render would rebuild that callback on every keystroke in the form.
  const caps = useMemo(() => limits.data ?? DEFAULT_LIMITS, [limits.data]);

  // The whole form rather than the attachments alone, deliberately. The "Referenced" badge
  // on each row is a fact about the *prose*, so a subscription narrowed to `attachments`
  // would leave it saying whatever it said when the last image was added -- and this is a
  // handful of short strings, not a table.
  const values = form.watch();
  const attachments = values.attachments;
  const totalBytes = attachments.reduce((sum, item) => sum + item.byte_size, 0);
  const referenced = new Set(referencedMarkers(values));

  /** Remember the field somebody was last typing in, which is where a reference goes. */
  useEffect(() => {
    const remember = (event: FocusEvent) => {
      const element = event.target;
      if (!(element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement)) return;
      if (element.type === 'file' || !element.name) return;
      // Only the fields a marker may appear in. A reference written into a repository URL or
      // a marker box is not a reference, it is a typo this would have helped make.
      if (!/^(title|problem_statement|goals|user_stories|requirements|constraints|out_of_scope|stakeholders)/.test(element.name)) {
        return;
      }
      target.current = { name: element.name, element };
    };
    document.addEventListener('focusin', remember);
    return () => document.removeEventListener('focusin', remember);
  }, []);

  const accept = useCallback(
    async (files: File[]) => {
      setFailures([]);
      const refused: string[] = [];
      let count = attachments.length;
      let bytes = totalBytes;
      for (const file of files) {
        // Mirroring the server's order, and for the server's reasons: the type, then the
        // per-file size, then the caps across the set. Refused here so somebody is told
        // which file and why, instead of reading a 422 after filling in the whole form.
        if (!caps.accepted_media_types.includes(file.type)) {
          refused.push(
            file.type === 'image/svg+xml'
              ? `${file.name}: SVG is markup, not an image — export a PNG`
              : `${file.name}: only ${caps.accepted_media_types
                  .map((type) => type.replace('image/', '').toUpperCase())
                  .join(', ')} are accepted`,
          );
          continue;
        }
        if (file.size > caps.max_attachment_bytes) {
          refused.push(`${file.name}: larger than ${megabytes(caps.max_attachment_bytes)}`);
          continue;
        }
        if (count >= caps.max_attachments_per_feature) {
          refused.push(
            `${file.name}: a feature carries at most ${caps.max_attachments_per_feature} images`,
          );
          continue;
        }
        if (bytes + file.size > caps.max_attachment_bytes_per_feature) {
          refused.push(
            `${file.name}: a feature's images total at most ${megabytes(
              caps.max_attachment_bytes_per_feature,
            )}`,
          );
          continue;
        }
        setBusy((value) => value + 1);
        try {
          const uploaded = await api.uploadAttachment(file);
          append({
            attachment_id: uploaded.attachment_id,
            marker: uniqueMarker(
              markerFromFilename(uploaded.filename, count),
              form.getValues('attachments').map((item) => item.marker),
            ),
            caption: '',
            filename: uploaded.filename,
            byte_size: uploaded.byte_size,
            media_type: uploaded.media_type,
          });
          count += 1;
          bytes += uploaded.byte_size;
        } catch (error) {
          // The server's own sentence where there is one: it is the only text that says
          // whether this was the size, the type or the contents.
          refused.push(
            `${file.name}: ${
              error instanceof ApiError ? (error.detail ?? userMessage(error)) : 'upload failed'
            }`,
          );
        } finally {
          setBusy((value) => value - 1);
        }
      }
      if (refused.length > 0) setFailures(refused);
    },
    [api, append, attachments.length, caps, form, totalBytes],
  );

  /**
   * Accept a pasted screenshot wherever the paste lands, not only on the drop zone.
   *
   * Paste is the primary path -- a screenshot arrives straight out of the screenshot tool --
   * and it happens while the cursor is in a prose field, which is the one place a handler on
   * the drop zone never hears about. So the listener is on the document, like the focus
   * tracker above it, for as long as this field is mounted: any paste carrying image files
   * uploads them, and a paste of plain text is left entirely alone.
   */
  useEffect(() => {
    const pasted = (event: ClipboardEvent) => {
      const files = [...(event.clipboardData?.files ?? [])];
      if (files.length === 0) return;
      event.preventDefault();
      void accept(files);
    };
    document.addEventListener('paste', pasted);
    return () => document.removeEventListener('paste', pasted);
  }, [accept]);

  /** Write `[image:marker]` at the cursor of whichever field last had focus. */
  const insert = (marker: string) => {
    const current = target.current;
    const text = imageReference(marker);
    if (current === null) {
      setNotes([`Put the cursor in a text field, then insert ${text}`]);
      return;
    }
    const { element, name } = current;
    const start = element.selectionStart ?? element.value.length;
    const end = element.selectionEnd ?? start;
    const next = `${element.value.slice(0, start)}${text}${element.value.slice(end)}`;
    form.setValue(name as never, next as never, { shouldValidate: true, shouldDirty: true });
    setNotes([`Inserted ${text}`]);
    // Put the caret after what was written, so a second insert does not land inside the
    // first. Deferred because the value above lands on the next render.
    requestAnimationFrame(() => {
      element.focus();
      element.setSelectionRange(start + text.length, start + text.length);
    });
  };

  /**
   * Rename a marker, and rewrite the prose that pointed at the old one.
   *
   * The one edit that silently invalidates a submission otherwise: `[image:login]` written
   * three paragraphs up stops resolving the moment the marker becomes `login-error`, and the
   * server refuses the whole submission naming a marker somebody no longer remembers typing.
   *
   * The rewrite happens when the field is *finished*, not on every keystroke, and that is the
   * whole design of this pair of functions. Rewriting per keystroke turns select-all-and-retype
   * -- the ordinary way anybody edits a short field -- into a rename to the empty string
   * followed by twelve renames nothing can follow, and the references end up as `[image:]`.
   * So `editMarker` only moves the field, and `commitMarker` compares against the value the
   * field held when it was focused.
   */
  const editMarker = (index: number, next: string) => {
    // `setValue` on the one path rather than `useFieldArray.update`, which replaces the whole
    // row: replacing it re-registers the input and takes focus off it, so every keystroke
    // fired a fresh `onFocus` and the value a rename was measured from was always the
    // half-typed one. This is a text edit, not a row change.
    form.setValue(`attachments.${index}.marker`, next, { shouldValidate: true, shouldDirty: true });
  };

  const commitMarker = (index: number, previous: string) => {
    const item = form.getValues('attachments')[index];
    if (item === undefined) return;
    const next = item.marker;
    if (previous === next || !MARKER_PATTERN.test(previous) || !MARKER_PATTERN.test(next)) return;
    let rewritten = 0;
    for (const [name, value] of proseFields(form.getValues())) {
      if (!value.includes(imageReference(previous))) continue;
      rewritten += occurrences(value, imageReference(previous));
      form.setValue(
        name as never,
        value.replace(markerReferences(previous), imageReference(next)) as never,
        { shouldDirty: true, shouldValidate: true },
      );
    }
    setNotes(
      rewritten > 0
        ? [
            `Renamed to ${imageReference(next)} and rewrote ${rewritten} reference${
              rewritten === 1 ? '' : 's'
            } in your text`,
          ]
        : [],
    );
  };

  const removeAt = (index: number) => {
    const item = form.getValues('attachments')[index];
    if (item === undefined) return;
    // Read at click time, not from the render that drew this button: the prose may have
    // changed since, and the whole value of this warning is that it is current.
    const stillReferenced = new Set(referencedMarkers(form.getValues()));
    remove(index);
    // Deleted on the server too: an image nobody submitted must not sit in the database
    // waiting for the sweep. Best effort -- the sweep is the backstop, and a failed delete
    // must not stop somebody removing the row in front of them.
    void api.deleteAttachment(item.attachment_id).catch(() => undefined);
    setNotes(
      stillReferenced.has(item.marker)
        ? [
            `Removed ${item.filename}. Your text still refers to ${imageReference(
              item.marker,
            )} — remove that reference or the submission will be refused.`,
          ]
        : [`Removed ${item.filename}`],
    );
  };

  const full = attachments.length >= caps.max_attachments_per_feature;

  return (
    <Fieldset legend="Screens and mock-ups">
      <p className="field__hint">
        Show the platform what you are describing. Up to {caps.max_attachments_per_feature} images,{' '}
        {megabytes(caps.max_attachment_bytes)} each and {megabytes(caps.max_attachment_bytes_per_feature)} in
        total; PNG, JPEG or WebP. Paste a screenshot straight into this page, drop files here, or
        choose them. Refer to one in your text as <code>[image:marker]</code> — the product manager
        reads the picture and writes what it requires into the technical requirements.
      </p>

      {visionCapable === false && attachments.length > 0 && !mockMode ? (
        <p className="field__error" role="alert">
          {selectionLabel} does not read images, so these would not reach the model and the
          submission will be refused. Choose a platform, tier or setup whose reasoning model reads
          them, or remove the images.
        </p>
      ) : null}

      <div
        className={over ? 'dropzone dropzone--over' : 'dropzone'}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          void accept([...event.dataTransfer.files]);
        }}
      >
        <p className="muted">
          {full
            ? `That is ${caps.max_attachments_per_feature} images — remove one to add another.`
            : 'Drop images here, or paste a screenshot.'}
        </p>
        <button
          type="button"
          className="button button--small"
          disabled={full}
          onClick={() => picker.current?.click()}
        >
          Choose images
        </button>
        <input
          ref={picker}
          type="file"
          multiple
          className="sr-only"
          accept={caps.accepted_media_types.join(',')}
          aria-label="Choose images"
          onChange={(event) => {
            const chosen = [...(event.target.files ?? [])];
            event.target.value = '';
            void accept(chosen);
          }}
        />
        {busy > 0 ? (
          <span className="muted" role="status">
            Uploading {busy} image{busy === 1 ? '' : 's'}…
          </span>
        ) : null}
      </div>

      {failures.length > 0 ? (
        <div className="state state--error" role="alert">
          <p className="state__title">Some files were not attached.</p>
          <ul className="bullets">
            {failures.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {notes.length > 0 ? (
        <p className="field__hint" role="status">
          {notes.join(' ')}
        </p>
      ) : null}
      {form.formState.errors.attachments?.root?.message ? (
        <p className="field__error" role="alert">
          {form.formState.errors.attachments.root.message}
        </p>
      ) : null}
      {form.formState.errors.attachments?.message ? (
        <p className="field__error" role="alert">
          {form.formState.errors.attachments.message}
        </p>
      ) : null}

      {fields.length > 0 ? (
        <ul className="attachments">
          {fields.map((field, index) => {
            const item = attachments[index];
            if (item === undefined) return null;
            return (
              <li key={field.id} className="attachment card">
                <AttachmentImage
                  attachment={{
                    attachment_id: item.attachment_id,
                    marker: item.marker,
                    caption: item.caption,
                    filename: item.filename,
                    media_type: item.media_type,
                    byte_size: item.byte_size,
                    sha256: '',
                  }}
                  className="attachment__thumb"
                />
                <div className="stack stack--tight">
                  <div className="row row--between">
                    <strong>{item.filename}</strong>
                    <Badge outline>{kilobytes(item.byte_size)}</Badge>
                  </div>
                  <Field
                    label="Marker"
                    error={form.formState.errors.attachments?.[index]?.marker?.message}
                    hint={`Refer to this image in your text as ${imageReference(item.marker)}`}
                  >
                    {(id, describedBy) => (
                      <input
                        id={id}
                        aria-describedby={describedBy}
                        value={item.marker}
                        onChange={(event) => editMarker(index, event.target.value)}
                        onFocus={(event) => {
                          focusedMarker.current = event.target.value;
                        }}
                        onBlur={() => commitMarker(index, focusedMarker.current)}
                        onKeyDown={(event) => {
                          if (event.key !== 'Enter') return;
                          // Committed on Enter as well as on blur: somebody who types a new
                          // marker and presses Enter has finished, and a form that only
                          // rewrote on blur would leave them looking at stale references.
                          event.preventDefault();
                          commitMarker(index, focusedMarker.current);
                          focusedMarker.current = form.getValues('attachments')[index]?.marker ?? '';
                        }}
                      />
                    )}
                  </Field>
                  <Field
                    label="Caption"
                    error={form.formState.errors.attachments?.[index]?.caption?.message}
                  >
                    {(id, describedBy) => (
                      <input
                        id={id}
                        aria-describedby={describedBy}
                        placeholder="What this shows, in a sentence"
                        value={item.caption}
                        onChange={(event) =>
                          form.setValue(`attachments.${index}.caption`, event.target.value, {
                            shouldValidate: true,
                            shouldDirty: true,
                          })
                        }
                      />
                    )}
                  </Field>
                  <div className="row">
                    <button
                      type="button"
                      className="button button--small"
                      onClick={() => insert(item.marker)}
                    >
                      Insert reference
                    </button>
                    <button
                      type="button"
                      className="button button--quiet"
                      onClick={() => removeAt(index)}
                    >
                      Remove
                    </button>
                    {referenced.has(item.marker) ? (
                      <Badge tone="done">Referenced</Badge>
                    ) : (
                      <Badge outline>Not referenced — still shown to the model</Badge>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      ) : null}
    </Fieldset>
  );
}

/** Every form field a marker reference may legitimately appear in, with its current value. */
function proseFields(values: NewFeatureValues): [string, string][] {
  const entries: [string, string][] = [
    ['title', values.title],
    ['problem_statement', values.problem_statement],
  ];
  values.goals.forEach((item, index) => entries.push([`goals.${index}.value`, item.value]));
  values.constraints.forEach((item, index) =>
    entries.push([`constraints.${index}.value`, item.value]),
  );
  values.out_of_scope.forEach((item, index) =>
    entries.push([`out_of_scope.${index}.value`, item.value]),
  );
  values.stakeholders.forEach((item, index) =>
    entries.push([`stakeholders.${index}.value`, item.value]),
  );
  values.user_stories.forEach((story, index) => {
    entries.push([`user_stories.${index}.persona`, story.persona]);
    entries.push([`user_stories.${index}.need`, story.need]);
    entries.push([`user_stories.${index}.benefit`, story.benefit]);
    story.acceptance_criteria.forEach((item, position) =>
      entries.push([`user_stories.${index}.acceptance_criteria.${position}.value`, item.value]),
    );
  });
  values.requirements.forEach((requirement, index) => {
    entries.push([`requirements.${index}.description`, requirement.description]);
    requirement.acceptance_criteria.forEach((item, position) =>
      entries.push([`requirements.${index}.acceptance_criteria.${position}.value`, item.value]),
    );
  });
  return entries;
}

/** A marker not already taken, so a second `login.png` does not collide with the first. */
function uniqueMarker(candidate: string, taken: string[]): string {
  if (!taken.includes(candidate)) return candidate;
  for (let suffix = 2; suffix < 100; suffix += 1) {
    const next = `${candidate}-${suffix}`.slice(0, 40).replace(/-+$/, '');
    if (!taken.includes(next)) return next;
  }
  return candidate;
}

function occurrences(text: string, needle: string): number {
  return text.split(needle).length - 1;
}

function megabytes(bytes: number): string {
  return `${Math.round(bytes / (1024 * 1024))} MiB`;
}

function kilobytes(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
    : `${Math.max(1, Math.round(bytes / 1024))} KiB`;
}
