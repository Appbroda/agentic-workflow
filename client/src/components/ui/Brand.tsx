import type { SVGProps } from 'react';

/**
 * The CRYN3T Systems identity, drawn rather than fetched.
 *
 * Everything here is inline SVG and CSS: no web font, no image asset, no icon package. The
 * mark is fourteen primitives and the backdrop is nineteen, which is less than the request
 * for a logo file would have cost -- and it inherits `currentColor` and the brand tokens, so
 * one theme definition covers light and dark and a future rebrand is this file plus the token
 * block in `styles.css`.
 *
 * The motif is the reference drawing's: a hexagon lattice, thin traces that terminate in a
 * node, and concentric circles -- an engineering diagram rather than a logo with a gradient
 * in it. The lockup stacks CRYN3T over SYSTEMS in wide tracking, which is what makes a system
 * sans read as a technical mark without shipping a typeface to do it.
 *
 * The product is CRYN3T Systems everywhere it is named. `BRAND_NAME` is the only place the
 * string lives, so it cannot drift between the title bar, the sidebar and the login page.
 */

export const BRAND_NAME = 'CRYN3T Systems';
export const BRAND_SHORT = 'CRYN3T';
export const BRAND_SUFFIX = 'SYSTEMS';

/**
 * The mark on its own: a hexagon containing a three-spoke node.
 *
 * Hidden from assistive technology in every use below, because in each of them a real word
 * sits beside it. A mark is never the only thing naming the product.
 */
export function BrandMark({ size = 24, ...props }: SVGProps<SVGSVGElement> & { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {/* The lattice cell. Pointy top and bottom, flat sides -- the reference's orientation. */}
      <path d="M16 2.6 27.6 9.3v13.4L16 29.4 4.4 22.7V9.3z" />
      {/* The node and its three spokes: the product is a system of parts that talk. */}
      <path d="M16 16V9.2M16 16l5.9 3.4M16 16l-5.9 3.4" strokeWidth={1.3} />
      <circle cx="16" cy="16" r="2.4" fill="currentColor" stroke="none" />
      <circle cx="16" cy="8.4" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="22.6" cy="20.2" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="9.4" cy="20.2" r="1.5" fill="currentColor" stroke="none" />
    </svg>
  );
}

/**
 * Mark plus wordmark.
 *
 * `stacked` is the two-line lockup the reference uses and the login page wants; the inline
 * form is for the sidebar and anywhere else a row of navigation sets the height. The whole
 * thing is one accessible name -- "CRYN3T Systems" -- rather than two fragments a screen
 * reader has to reassemble.
 */
export function BrandLockup({
  stacked = false,
  markSize,
  className,
}: {
  stacked?: boolean;
  markSize?: number;
  className?: string;
}) {
  const classes = ['brand', stacked ? 'brand--stacked' : '', className ?? '']
    .filter(Boolean)
    .join(' ');
  return (
    <span className={classes} aria-label={BRAND_NAME} role="img">
      <span className="brand__mark" aria-hidden="true">
        <BrandMark size={markSize ?? (stacked ? 40 : 24)} />
      </span>
      <span className="brand__words" aria-hidden="true">
        <span className="brand__name">{BRAND_SHORT}</span>
        <span className="brand__suffix">{BRAND_SUFFIX}</span>
      </span>
    </span>
  );
}

/**
 * The etched backdrop behind the login brand panel.
 *
 * Decoration, and declared as such: `aria-hidden`, no interactivity, and it is drawn in the
 * two brand line tokens so it fades into whichever theme is showing rather than being a grey
 * image that only works on one of them. It slices rather than scales, so the composition
 * keeps its proportions in a 390px column and in a 1920px panel.
 */
export function CircuitField({ className }: { className?: string }) {
  return (
    <svg
      className={['circuit', className ?? ''].filter(Boolean).join(' ')}
      viewBox="0 0 600 720"
      preserveAspectRatio="xMidYMid slice"
      fill="none"
      strokeLinecap="round"
      aria-hidden="true"
      focusable="false"
    >
      <g stroke="var(--brand-line)" strokeWidth={1.4}>
        {/* Concentric circles: the reference's two overlapping rings. */}
        <circle cx="212" cy="392" r="118" />
        <circle cx="212" cy="392" r="86" />
        <circle cx="398" cy="418" r="82" />
        <circle cx="398" cy="418" r="58" />
        {/* The hexagon lattice, running off the top-right corner. */}
        <path d="M470 118l52 30v60l-52 30-52-30v-60z" />
        <path d="M574 178l52 30v60l-52 30-52-30v-60z" />
        <path d="M470 238l52 30v60l-52 30-52-30v-60z" />
        {/* Traces. Each one turns at a right angle, the way a trace on a board does. */}
        <path d="M52 300h120l58-58h150" />
        <path d="M330 476l58 58v92" />
        <path d="M96 620v-96h140" />
        <path d="M470 330v78l-72 10" />
      </g>
      <g stroke="var(--brand-line-strong)" strokeWidth={1.6}>
        {/* The node: the one element in the field that is not a pale line, so the eye lands
            somewhere. Its stem runs down into the trace layer above. */}
        <circle cx="398" cy="176" r="26" />
        <path d="M398 202v96" />
        <path d="M386 286l12 12 12-12" />
        <circle cx="398" cy="176" r="9" fill="var(--brand-line-strong)" stroke="none" />
        {/* Two arrowheads, pointing where the traces go. */}
        <path d="M218 254l12-12 12 12" />
        <path d="M224 530l12 12 12-12" />
      </g>
    </svg>
  );
}
