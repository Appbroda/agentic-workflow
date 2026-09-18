import { useCallback, useSyncExternalStore } from 'react';

/**
 * Whether a media query matches, as state.
 *
 * The shell has three shapes -- full sidebar, icon rail, off-canvas drawer -- and which one
 * it is has to be known in JavaScript, not only in CSS: the drawer needs a different button
 * in the top bar, a scrim, a close control and somewhere to put the focus. Deriving all of
 * that from a class the CSS sets would mean the markup and the layout disagreeing at exactly
 * the width where it matters.
 *
 * `useSyncExternalStore` rather than `useState` + an effect, so the first paint already knows
 * the answer instead of rendering the desktop shell and correcting itself. The server
 * snapshot is `false` because this application is client-rendered and never asked for one;
 * `matchMedia` is guarded because jsdom is not obliged to implement it, and a missing
 * `matchMedia` must mean "the wide layout", never a crash.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const list = window.matchMedia?.(query);
      if (!list) return () => {};
      list.addEventListener('change', onChange);
      return () => list.removeEventListener('change', onChange);
    },
    [query],
  );
  const read = useCallback(() => window.matchMedia?.(query).matches ?? false, [query]);
  return useSyncExternalStore(subscribe, read, () => false);
}
