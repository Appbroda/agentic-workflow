import type { SVGProps } from 'react';

/**
 * One icon family, drawn inline.
 *
 * All of them share a 16-unit box, a 1.5 stroke and round joins, so they sit together without
 * one looking heavier than the next. They are inline SVG rather than a dependency because the
 * set is small and a font or package would be more bytes than the twenty paths below.
 *
 * Every icon here accompanies a label. None of them is the only thing carrying a meaning.
 */

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 16, children, ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {children}
    </svg>
  );
}

export const IconOverview = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2" y="2" width="5" height="5" rx="1" />
    <rect x="9" y="2" width="5" height="5" rx="1" />
    <rect x="2" y="9" width="5" height="5" rx="1" />
    <rect x="9" y="9" width="5" height="5" rx="1" />
  </Icon>
);

export const IconFeatures = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2 4h12M2 8h12M2 12h7" />
  </Icon>
);

export const IconPlus = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 3v10M3 8h10" />
  </Icon>
);

export const IconAttention = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 2.5 14.5 13.5h-13z" />
    <path d="M8 6.5v3" />
    <circle cx="8" cy="11.6" r="0.5" fill="currentColor" />
  </Icon>
);

export const IconRunning = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="8" r="5.5" />
    <path d="M8 5v3.2l2 1.3" />
  </Icon>
);

export const IconDone = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="8" r="5.5" />
    <path d="m5.6 8.2 1.7 1.7 3.2-3.5" />
  </Icon>
);

export const IconStopped = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="8" r="5.5" />
    <path d="m6 6 4 4M10 6l-4 4" />
  </Icon>
);

export const IconRepository = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 2.5h8v11H4.8A1.3 1.3 0 0 1 3.5 12.2V3.8A1.3 1.3 0 0 1 4.8 2.5z" />
    <path d="M3.5 11.2h8.5" />
  </Icon>
);

export const IconSettings = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="8" r="2.2" />
    <path d="M8 1.8v1.6M8 12.6v1.6M14.2 8h-1.6M3.4 8H1.8M12.4 3.6l-1.1 1.1M4.7 11.3l-1.1 1.1M12.4 12.4l-1.1-1.1M4.7 4.7 3.6 3.6" />
  </Icon>
);

/** Two figures: the accounts page. Same 16-unit box and 1.5 stroke as the rest. */
export const IconPeople = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="6" cy="5.5" r="2.3" />
    <path d="M1.8 13.5c0-2.3 1.9-3.6 4.2-3.6s4.2 1.3 4.2 3.6" />
    <path d="M10.8 3.6a2.3 2.3 0 0 1 0 4.4M11.8 10.2c1.5.35 2.4 1.5 2.4 3.3" />
  </Icon>
);

export const IconChevronRight = (props: IconProps) => (
  <Icon {...props}>
    <path d="m6 3.5 5 4.5-5 4.5" />
  </Icon>
);

export const IconChevronDown = (props: IconProps) => (
  <Icon {...props}>
    <path d="m3.5 6 4.5 5 4.5-5" />
  </Icon>
);

export const IconChevronUp = (props: IconProps) => (
  <Icon {...props}>
    <path d="m3.5 10 4.5-5 4.5 5" />
  </Icon>
);

export const IconCopy = (props: IconProps) => (
  <Icon {...props}>
    <rect x="5.5" y="5.5" width="8" height="8" rx="1.2" />
    <path d="M10.5 3.2A1.2 1.2 0 0 0 9.3 2.5H3.7a1.2 1.2 0 0 0-1.2 1.2v5.6c0 .5.3.9.7 1.1" />
  </Icon>
);

export const IconCheck = (props: IconProps) => (
  <Icon {...props}>
    <path d="m3.5 8.5 3 3 6-7" />
  </Icon>
);

export const IconExternal = (props: IconProps) => (
  <Icon {...props}>
    <path d="M9 3h4v4" />
    <path d="M13 3 7.5 8.5" />
    <path d="M12 9.5v3a1 1 0 0 1-1 1H3.5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h3" />
  </Icon>
);

export const IconClose = (props: IconProps) => (
  <Icon {...props}>
    <path d="m4 4 8 8M12 4l-8 8" />
  </Icon>
);

export const IconSearch = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="7.2" cy="7.2" r="4.2" />
    <path d="m10.4 10.4 3 3" />
  </Icon>
);

export const IconChat = (props: IconProps) => (
  <Icon {...props}>
    <path d="M13.5 9.5a1.5 1.5 0 0 1-1.5 1.5H6l-3 2.5V4a1.5 1.5 0 0 1 1.5-1.5H12A1.5 1.5 0 0 1 13.5 4z" />
  </Icon>
);

export const IconSidebar = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2" y="3" width="12" height="10" rx="1.5" />
    <path d="M6.2 3v10" />
  </Icon>
);

export const IconPullRequest = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="4.2" cy="4" r="1.7" />
    <circle cx="4.2" cy="12" r="1.7" />
    <circle cx="11.8" cy="12" r="1.7" />
    <path d="M4.2 5.7v4.6M11.8 10.3V6.4A1.9 1.9 0 0 0 9.9 4.5H7.6" />
    <path d="m9 2.8-1.6 1.7L9 6.2" />
  </Icon>
);

export const IconAgent = (props: IconProps) => (
  <Icon {...props}>
    <rect x="3" y="5" width="10" height="8" rx="2" />
    <path d="M8 2.5V5" />
    <circle cx="6.2" cy="9" r="0.6" fill="currentColor" />
    <circle cx="9.8" cy="9" r="0.6" fill="currentColor" />
  </Icon>
);

export const IconValidation = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.5 4.5 5 7l-2.5 2.5" />
    <path d="M7.5 11.5h6" />
  </Icon>
);

export const IconArtifact = (props: IconProps) => (
  <Icon {...props}>
    <path d="M9 2H4.5A1.5 1.5 0 0 0 3 3.5v9A1.5 1.5 0 0 0 4.5 14h7a1.5 1.5 0 0 0 1.5-1.5V6z" />
    <path d="M9 2v4h4" />
  </Icon>
);

export const IconHistory = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.8 8a5.2 5.2 0 1 0 1.6-3.8" />
    <path d="M2.5 2.6v2.8h2.8" />
    <path d="M8 5.4V8l1.9 1.1" />
  </Icon>
);

export const IconDocument = (props: IconProps) => (
  <Icon {...props}>
    <rect x="3" y="2" width="10" height="12" rx="1.5" />
    <path d="M5.5 5.5h5M5.5 8h5M5.5 10.5h3" />
  </Icon>
);

/** Opens the navigation drawer on a phone. Named "Menu" by the button that carries it. */
export const IconMenu = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />
  </Icon>
);

/* The password reveal, in its two states. Two icons rather than one rotated, because the
   struck-through eye is the convention for "hidden" and a person recognises it faster than
   they read the button's label. */
export const IconEye = (props: IconProps) => (
  <Icon {...props}>
    <path d="M1.8 8S4 4.2 8 4.2 14.2 8 14.2 8 12 11.8 8 11.8 1.8 8 1.8 8z" />
    <circle cx="8" cy="8" r="1.8" />
  </Icon>
);

export const IconEyeOff = (props: IconProps) => (
  <Icon {...props}>
    <path d="M6.3 4.6A6.4 6.4 0 0 1 8 4.2c4 0 6.2 3.8 6.2 3.8a12 12 0 0 1-2 2.5" />
    <path d="M11.4 11.3A6.3 6.3 0 0 1 8 11.8C4 11.8 1.8 8 1.8 8a12.2 12.2 0 0 1 2.8-3.2" />
    <path d="M6.7 6.7a1.8 1.8 0 0 0 2.6 2.6" />
    <path d="m2.6 2.6 10.8 10.8" />
  </Icon>
);

/** Security, on the login page's brand panel. Accompanied by its sentence, never alone. */
export const IconShield = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 2.2l4.8 1.7v4A6.3 6.3 0 0 1 8 13.8 6.3 6.3 0 0 1 3.2 7.9v-4z" />
    <path d="m5.9 7.9 1.5 1.5 2.7-2.8" />
  </Icon>
);

/** An account, a person, a session. */
export const IconUser = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="5.6" r="2.6" />
    <path d="M3.2 13.4a4.8 4.8 0 0 1 9.6 0" />
  </Icon>
);

/**
 * The empty-state glyph: an open tray with nothing in it.
 *
 * Illustration rather than information -- the title beside it says what is missing -- so it is
 * hidden from assistive technology like every other icon here.
 */
export const IconEmpty = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.2 9.4 4 3.6a1 1 0 0 1 1-.7h6a1 1 0 0 1 1 .7l1.8 5.8" />
    <path d="M2.2 9.4h3l.8 1.6h4l.8-1.6h3v2.2a1.5 1.5 0 0 1-1.5 1.5h-8.6a1.5 1.5 0 0 1-1.5-1.5z" />
  </Icon>
);
