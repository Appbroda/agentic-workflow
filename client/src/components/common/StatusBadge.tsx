import type { StatusWording } from '@/hooks/useStatusVocabulary';

/**
 * Status is conveyed by text as well as colour, so it survives a colour-blind reader and a
 * greyscale screenshot. The wording is the server's, never this component's.
 */
export function StatusBadge({ wording }: { wording: StatusWording }) {
  return (
    <span className={`badge badge--${wording.tone}`} title={wording.detail || undefined}>
      {wording.headline}
    </span>
  );
}
