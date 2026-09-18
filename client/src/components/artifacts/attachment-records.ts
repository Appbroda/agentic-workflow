/**
 * What a PRD artifact records about an attached image, and how to read it out of a payload.
 *
 * Split from the components that render it because both a component and a plain function
 * need it -- and because an artifact payload arrives as `Record<string, unknown>` from the
 * API's schema check, so something has to narrow it once rather than at each use.
 */

/** One attachment as the PRD artifact records it. Never any bytes. */
export interface RecordedAttachment {
  attachment_id: string;
  marker: string;
  caption: string;
  filename: string;
  media_type: string;
  byte_size: number;
  sha256: string;
}

/**
 * Read the attachment records off an artifact payload, ignoring anything malformed.
 *
 * Tolerant on purpose. The API cannot produce a malformed entry -- the artifact is a strict
 * schema -- but this also renders payloads read straight out of the artifacts endpoint by a
 * reader looking at an older feature, and a viewer is not the place to throw.
 */
export function recordedAttachments(payload: Record<string, unknown>): RecordedAttachment[] {
  const raw = payload.attachments;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    if (typeof item !== 'object' || item === null) return [];
    const record = item as Record<string, unknown>;
    const id = record.attachment_id;
    const marker = record.marker;
    if (typeof id !== 'string' || typeof marker !== 'string') return [];
    return [
      {
        attachment_id: id,
        marker,
        caption: typeof record.caption === 'string' ? record.caption : '',
        filename: typeof record.filename === 'string' ? record.filename : marker,
        media_type: typeof record.media_type === 'string' ? record.media_type : 'image/png',
        byte_size: typeof record.byte_size === 'number' ? record.byte_size : 0,
        sha256: typeof record.sha256 === 'string' ? record.sha256 : '',
      },
    ];
  });
}

/** The element id an inline chip scrolls to, derived from the marker, never a position. */
export function thumbnailId(marker: string): string {
  return `attachment-thumb-${marker}`;
}

/** One size, phrased the way a person reads it. */
export function attachmentSize(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
    : `${Math.max(1, Math.round(bytes / 1024))} KiB`;
}
