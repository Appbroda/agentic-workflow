import { Link } from 'react-router-dom';

/**
 * A URL this application has no page for.
 *
 * Distinct from an artifact or feature the platform does not have, which is an API answer
 * shown in place. This is the address being wrong, so the only useful thing to offer is the
 * way back.
 */
export function NotFoundPage() {
  return (
    <div className="page">
      <section className="state">
        <h1>No such page</h1>
        <p className="state__detail">
          This address does not match anything in the control plane. It may have been mistyped,
          or it may be a link from an older version.
        </p>
        <Link className="button button--primary" to="/">
          Back to features
        </Link>
      </section>
    </div>
  );
}
