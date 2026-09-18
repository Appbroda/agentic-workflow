import { useCallback, useEffect, useRef, useState } from 'react';
import { NavLink, Navigate, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { LoginPage } from './LoginPage';
import { Breadcrumbs } from './Breadcrumbs';
import { useSession } from './session-context';
import { LoadingState } from '@/components/common/States';
import { BrandLockup, BrandMark } from '@/components/ui/Brand';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import {
  IconClose,
  IconFeatures,
  IconMenu,
  IconPeople,
  IconPlus,
  IconSettings,
  IconSidebar,
} from '@/components/ui/icons';

/**
 * The application shell: a persistent sidebar, a top bar carrying position and identity, and
 * the page.
 *
 * There are two things a person does here — look at features, or ask for a new one — so those
 * are the two navigation items, and Settings sits apart at the bottom because it is not part of
 * the daily loop. The sidebar previously also listed Needs attention, Running and Completed;
 * each was the Features page with one filter pre-applied, so they were four routes for one
 * screen. They are chips on that screen now, and the old URLs redirect to them.
 *
 * It has three shapes, and which one is showing is decided here rather than in the stylesheet:
 *
 *   - full, on a desktop;
 *   - a rail of icons, when somebody collapses it, or when the viewport is tablet-width and a
 *     14rem sidebar would be a third of the screen;
 *   - an off-canvas drawer on a phone, with a scrim, a close button and the focus moved into
 *     it, because at that width a permanent sidebar leaves no page.
 *
 * The rail used to be half-built: the grid narrowed to 3.25rem but `sidebar--collapsed` was
 * never put on the element, so the labels stayed in the markup and were clipped by the column
 * they no longer fit. The class is applied now, which is what the collapsed rules in
 * `styles.css` were always written for.
 */

const SIDEBAR_KEY = 'ui.sidebar.collapsed';

/** Matches `--bp-sm` and `--bp-md` in `styles.css`. Both places have to say the same width. */
const PHONE = '(max-width: 46rem)';
const NARROW = '(max-width: 62rem)';

export function AppLayout() {
  const [collapsed, setCollapsed] = useState(() => readCollapsed());
  const [drawerOpen, setDrawerOpen] = useState(false);
  const session = useSession();
  const location = useLocation();
  const isPhone = useMediaQuery(PHONE);
  const isNarrow = useMediaQuery(NARROW);

  // Above the session checks below, and they have to be: every hook this component uses must
  // run on every render, and the checks return early.
  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_KEY, collapsed ? '1' : '0');
    } catch {
      // A browser refusing storage is not a reason for the layout to fail.
    }
  }, [collapsed]);

  // Following a link inside the drawer closes it. Without this the page changes behind a
  // panel that is still covering it, and the only way out is the scrim.
  useEffect(() => {
    setDrawerOpen(false);
  }, [location.pathname, location.search]);

  // Escape closes it, the same key that closes every other overlay in this application.
  useEffect(() => {
    if (!drawerOpen) return;
    const dismiss = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setDrawerOpen(false);
    };
    document.addEventListener('keydown', dismiss);
    return () => document.removeEventListener('keydown', dismiss);
  }, [drawerOpen]);

  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  // Nothing of the application renders without a session -- not the sidebar, not the top
  // bar, not a breadcrumb. Previously the shell drew and only the page content was gated,
  // which meant a signed-out browser showed navigation for a deployment it could not read.
  if (!session.hasToken) return <LoginPage />;
  if (session.loading) return <LoadingState label="Signing in…" />;
  // A token this platform does not accept is not a session. `onUnauthorized` has already
  // cleared it, so this is the window between that and the re-render.
  if (!session.actor) return <LoginPage />;
  // A password somebody else chose is a handover credential, and the account holder replaces
  // it before doing anything else. A route rather than a modal, so a refresh does not bypass
  // it -- and it is checked here rather than in the router's route list because every path
  // has to be behind it, including ones added later.
  if (session.actor.must_change_password && location.pathname !== '/account/password') {
    return <Navigate to="/account/password" replace />;
  }

  // Never a rail on a phone: there the sidebar is a drawer, and a drawer showing six icons
  // and no words is worse than the full navigation it has room for.
  const rail = !isPhone && (collapsed || isNarrow);

  return (
    <div
      className={['shell', rail ? 'shell--collapsed' : '', isPhone ? 'shell--mobile' : '']
        .filter(Boolean)
        .join(' ')}
    >
      <Sidebar rail={rail} drawer={isPhone} open={drawerOpen} onClose={closeDrawer} />
      {isPhone && drawerOpen ? (
        <div className="nav-scrim" onClick={closeDrawer} aria-hidden="true" />
      ) : null}
      <div className="main">
        <header className="topbar">
          <NavigationControl
            isPhone={isPhone}
            isNarrow={isNarrow}
            collapsed={collapsed}
            drawerOpen={drawerOpen}
            onOpenDrawer={() => setDrawerOpen(true)}
            onToggleCollapsed={() => setCollapsed((current) => !current)}
          />
          {/* On a phone the sidebar is a closed drawer, so the mark is the only thing left
              saying which product this is. Decorative -- the drawer carries the name. */}
          {isPhone ? (
            <span className="topbar__mark" aria-hidden="true">
              <BrandMark size={20} />
            </span>
          ) : null}
          <Breadcrumbs />
          <div className="topbar__right">
            <UserMenu />
          </div>
        </header>
        <main className="main__content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

/**
 * The one control in the top bar that acts on the navigation, whichever shape it is in.
 *
 * On a phone it opens the drawer. On a desktop it collapses and expands the sidebar. At
 * tablet width it is absent, because the rail is forced there and a button offering to expand
 * it would do nothing when pressed.
 */
function NavigationControl({
  isPhone,
  isNarrow,
  collapsed,
  drawerOpen,
  onOpenDrawer,
  onToggleCollapsed,
}: {
  isPhone: boolean;
  isNarrow: boolean;
  collapsed: boolean;
  drawerOpen: boolean;
  onOpenDrawer: () => void;
  onToggleCollapsed: () => void;
}) {
  if (isPhone) {
    return (
      <button
        type="button"
        className="topbar__toggle"
        aria-label="Open navigation"
        aria-expanded={drawerOpen}
        aria-controls="app-sidebar"
        onClick={onOpenDrawer}
      >
        <IconMenu />
      </button>
    );
  }
  if (isNarrow) return null;
  return (
    <button
      type="button"
      className="topbar__toggle"
      aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
      aria-pressed={collapsed}
      onClick={onToggleCollapsed}
    >
      <IconSidebar />
    </button>
  );
}

function readCollapsed(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_KEY) === '1';
  } catch {
    return false;
  }
}

function Sidebar({
  rail,
  drawer,
  open,
  onClose,
}: {
  rail: boolean;
  drawer: boolean;
  open: boolean;
  onClose: () => void;
}) {
  const { may } = useSession();
  const closeButton = useRef<HTMLButtonElement>(null);

  // A slide-over takes the focus, or a keyboard user is still on the page behind it. The
  // close button is where it goes: it is the first control in the panel and pressing it
  // returns you to where you were.
  useEffect(() => {
    if (drawer && open) closeButton.current?.focus();
  }, [drawer, open]);

  return (
    <nav
      id="app-sidebar"
      className={['sidebar', rail ? 'sidebar--collapsed' : '', drawer && open ? 'sidebar--open' : '']
        .filter(Boolean)
        .join(' ')}
      aria-label="Main"
    >
      <div className="sidebar__brand">
        {/* Not a link. The brand is the product naming itself; Features is the page, and it
            is one row below with an icon and a word. */}
        <BrandLockup markSize={22} />
        {drawer ? (
          <button
            ref={closeButton}
            type="button"
            className="sidebar__close"
            aria-label="Close navigation"
            onClick={onClose}
          >
            <IconClose />
          </button>
        ) : null}
      </div>

      <div className="sidebar__group">
        <p className="sidebar__heading">Workspace</p>
        <SidebarLink to="/" end icon={<IconFeatures />} label="Features" />
        <SidebarLink to="/features/new" icon={<IconPlus />} label="New feature" />
      </div>

      <div className="sidebar__footer">
        <p className="sidebar__heading">Account</p>
        {/* Keyed on the named permission the server publishes, never on a role string:
            `ROLE_PERMISSIONS` is the one authority on what a role grants, and a client that
            tested `roles.includes('admin')` would be a second one. The page is reachable by
            typing the URL and says so politely; the endpoints behind it refuse. */}
        {may('user:manage') ? (
          <SidebarLink to="/users" icon={<IconPeople />} label="People" />
        ) : null}
        <SidebarLink to="/settings" icon={<IconSettings />} label="Settings" />
      </div>
    </nav>
  );
}

function SidebarLink({
  to,
  end,
  icon,
  label,
}: {
  to: string;
  end?: boolean;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      // The title is what a collapsed sidebar has instead of a label.
      title={label}
      className={({ isActive }) => (isActive ? 'sidebar__link sidebar__link--active' : 'sidebar__link')}
    >
      <span className="sidebar__icon" aria-hidden="true">
        {icon}
      </span>
      <span className="sidebar__label">{label}</span>
    </NavLink>
  );
}

/**
 * Who is signed in, with the two things you can do about it.
 *
 * It shows the name the authentication model gives and nothing else. It used to carry a "shared
 * key" badge beside the name, which told a normal user about an authentication mechanism they
 * cannot act on; how the request authenticated is now a line in Settings, where somebody who
 * cares can find it.
 */
function UserMenu() {
  const navigate = useNavigate();
  const { actor } = useSession();
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent) {
        if (event.key === 'Escape') setOpen(false);
        return;
      }
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', dismiss);
    document.addEventListener('keydown', dismiss);
    return () => {
      document.removeEventListener('mousedown', dismiss);
      document.removeEventListener('keydown', dismiss);
    };
  }, [open]);

  if (!actor) return null;
  const name = actor.display_name;

  return (
    <div className="usermenu" ref={container}>
      <button
        type="button"
        className="usermenu__trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="usermenu__avatar" aria-hidden="true">
          {initials(name)}
        </span>
        <span className="usermenu__name">{name}</span>
      </button>
      {open ? (
        <div className="usermenu__panel" role="menu">
          <p className="usermenu__identity">
            <strong>{name}</strong>
            {/* The email, not the account id. `subject` is what somebody recognises as
                theirs; `actor_id` is a machine string, and for the administrator it is
                permanently the literal `platform-admin`. */}
            <span className="subtle">{actor.subject || actor.actor_id}</span>
          </p>
          <button
            type="button"
            role="menuitem"
            className="usermenu__item"
            onClick={() => {
              setOpen(false);
              navigate('/settings');
            }}
          >
            Settings
          </button>
          <button
            type="button"
            role="menuitem"
            className="usermenu__item"
            onClick={() => {
              setOpen(false);
              navigate('/account/password');
            }}
          >
            Change password
          </button>
          <LogoutItem />
        </div>
      ) : null}
    </div>
  );
}

/**
 * Ends the session, and only the session.
 *
 * `signOut` revokes the token on the platform, forgets it here, and clears the query cache.
 * The cache clear is the part worth naming: every cached query was fetched as the identity
 * that is leaving, so without it the next person to sign in on this tab sees the previous
 * one's features until each query happens to refetch. That is a real cross-user leak in the
 * browser, and it lives in `signOut` rather than here so no second way to sign out can miss
 * it.
 *
 * There is no build-time-key branch any more. `VITE_PLATFORM_API_KEY` used to win over the
 * runtime credential and could not be cleared, which made signing out silently do nothing in
 * any environment that set it. It is retired.
 *
 * Saved provider credentials, repositories and model setups live on the server against the
 * account and are untouched; signing out is not a way to lose them.
 */
function LogoutItem() {
  const { signOut } = useSession();
  return (
    <button
      type="button"
      role="menuitem"
      className="usermenu__item usermenu__item--danger"
      onClick={() => void signOut()}
    >
      Sign out
    </button>
  );
}

/** Up to two letters from a display name, for the avatar. */
function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  const first = words.at(0);
  const last = words.at(-1);
  if (first === undefined || last === undefined) return '?';
  if (words.length === 1) return first.slice(0, 2).toUpperCase();
  return `${first.charAt(0)}${last.charAt(0)}`.toUpperCase();
}
