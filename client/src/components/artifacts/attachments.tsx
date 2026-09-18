import { useEffect, useState, type ReactNode } from 'react';
import { useApi } from '@/app/api-context';
import { Dialog } from '@/components/ui/Dialog';
import { Markdown } from '@/components/common/Markdown';
import {
  attachmentSize,
  thumbnailId,
  type RecordedAttachment,
} from './attachment-records';

/**
 * Reading the images a submission attached.
 *
 * Three pieces, and they are here rather than inside `DocumentTab` because the generic
 * artifact renderer needs the same strip: a PRD payload opened through the raw renderer must
 * show a list of what was attached, not a JSON array of hashes.
 *
 * `[image:marker]` in the rendered prose becomes a chip that scrolls to its thumbnail and
 * highlights it. An unknown marker renders as plain text: it cannot happen through the API --
 * the start endpoint refuses a dangling reference -- and a viewer is not the place to throw
 * over a payload somebody could have written directly into the database.
 */

/**
 * The strip of thumbnails under a submitted PRD, with the captions their authors wrote.
 *
 * Each thumbnail opens the full image in the existing `Dialog`. The element ids are what the
 * inline chips scroll to, so they are derived from the marker and not from a position: a
 * chip written for `login-error` must find `login-error` however the list is ordered.
 */
export function AttachmentStrip({ attachments }: { attachments: RecordedAttachment[] }) {
  const [open, setOpen] = useState<RecordedAttachment | null>(null);
  if (attachments.length === 0) return null;
  return (
    <>
      <ul className="attachment-strip">
        {attachments.map((item) => (
          <li key={item.attachment_id} className="attachment-strip__item">
            <button
              type="button"
              className="button button--quiet button--icon"
              id={thumbnailId(item.marker)}
              onClick={() => setOpen(item)}
              aria-label={`Open ${item.filename}`}
            >
              <AttachmentImage attachment={item} className="attachment__thumb" />
            </button>
            <code className="attachment-strip__caption">[image:{item.marker}]</code>
            {item.caption ? (
              <span className="attachment-strip__caption">{item.caption}</span>
            ) : null}
          </li>
        ))}
      </ul>
      {open ? (
        <Dialog title={open.filename} onDismiss={() => setOpen(null)}>
          <AttachmentImage attachment={open} className="dialog__image" />
          {open.caption ? <p className="prose">{open.caption}</p> : null}
          <p className="muted">
            {open.media_type} · {attachmentSize(open.byte_size)} · sha256 {open.sha256.slice(0, 12)}
          </p>
        </Dialog>
      ) : null}
    </>
  );
}

/**
 * Prose with `[image:marker]` rendered as a chip that finds its thumbnail.
 *
 * Split on the reference syntax and the pieces rendered separately, because the alternative
 * -- rewriting the markdown to inject a link -- would put a marker inside a markdown parser
 * and make the chip's behaviour depend on where in a sentence it happened to fall.
 */
export function ProseWithImageChips({
  value,
  markers,
}: {
  value: string;
  markers: Set<string>;
}) {
  const pieces: ReactNode[] = [];
  const pattern = /\[image:([a-z0-9][a-z0-9-]{0,39})\]/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(value)) !== null) {
    const marker = match[1]!;
    if (match.index > last) {
      pieces.push(<Markdown key={`text-${last}`}>{value.slice(last, match.index)}</Markdown>);
    }
    // An unknown marker is left exactly as written. It cannot arrive through the API, and a
    // chip that scrolled to nothing would be worse than the four words it replaced.
    pieces.push(
      markers.has(marker) ? (
        <ImageChip key={`chip-${match.index}`} marker={marker} />
      ) : (
        <span key={`chip-${match.index}`}>{match[0]}</span>
      ),
    );
    last = match.index + match[0].length;
  }
  if (last < value.length) {
    pieces.push(<Markdown key={`text-${last}`}>{value.slice(last)}</Markdown>);
  }
  return <>{pieces}</>;
}

/** One inline chip: scrolls its thumbnail into view and highlights it briefly. */
function ImageChip({ marker }: { marker: string }) {
  return (
    <button
      type="button"
      className="image-chip"
      onClick={() => {
        const target = document.getElementById(thumbnailId(marker));
        if (target === null) return;
        target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        const image = target.querySelector('img') ?? target;
        image.classList.add('attachment__thumb--found');
        // Removed on a timer rather than on blur: this is a "here it is", not a state
        // anybody should have to dismiss.
        window.setTimeout(() => image.classList.remove('attachment__thumb--found'), 2_000);
      }}
    >
      [image:{marker}]
    </button>
  );
}

/**
 * One attachment's bytes, fetched with this browser's token.
 *
 * Every attachment read is authenticated, so there is no URL an `<img src>` could follow --
 * which is the point, and is why this component exists rather than a plain `img`. The object
 * URL is revoked on unmount; leaked, it holds a multi-megabyte blob for the life of the page.
 *
 * A read that fails leaves an empty frame rather than an error: the two reasons it fails are
 * a purged attachment on a retired feature (410) and a permission the viewer does not have,
 * and neither is worth breaking a document over.
 */
export function AttachmentImage({
  attachment,
  className,
}: {
  attachment: RecordedAttachment;
  className: string;
}) {
  const api = useApi();
  const [source, setSource] = useState<string | null>(null);

  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    void api
      .getAttachmentBlob(attachment.attachment_id)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setSource(url);
      })
      // Swallowed on purpose: the two ways this fails are a purged attachment on a retired
      // feature and a permission this viewer does not hold, and the empty frame below says
      // as much as either would.
      .catch(() => undefined);
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [api, attachment.attachment_id]);

  if (source === null) {
    return <span className={`${className} attachment__thumb--empty`} aria-hidden="true" />;
  }
  return <img className={className} src={source} alt={attachment.filename} />;
}

