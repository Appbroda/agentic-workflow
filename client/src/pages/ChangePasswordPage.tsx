import { useSession } from '@/app/session-context';
import { ChangePasswordView } from '@/features/settings/ChangePasswordView';

/**
 * The change-password screen, told whether this visit is a forced one.
 *
 * `must_change_password` is read here rather than passed down a route, because the same page
 * serves both cases: an account whose password was set for it is sent here and cannot leave,
 * and anybody may come here voluntarily from the account menu.
 */
export function ChangePasswordPage() {
  const { actor } = useSession();
  return <ChangePasswordView forced={Boolean(actor?.must_change_password)} />;
}
