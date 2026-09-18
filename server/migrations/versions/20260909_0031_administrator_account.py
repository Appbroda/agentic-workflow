"""`platform-admin` becomes a person rather than a constructed identity.

Until now `platform-admin` was an `Actor` the code built when somebody presented the shared
platform key -- and, because it was the `actor_id` on that path, it is already the `owner_id`
on every stored provider credential, every repository configuration, every model setup and
every attachment this deployment holds. Migration 0030 gave it every existing feature too.
This revision gives that id a `platform_users` row, so the administrator can log in and the
ownership already recorded everywhere resolves to an account.

**No `owner_id` is rewritten here, or anywhere in this change.** The obvious alternative --
mint a fresh user id and `UPDATE` every `owner_id` -- destroys data, and not in a way anything
warns about:

    `owner_id` and `provider` are the additional authenticated data of the AES-GCM seal
    (`_associated_data` in `server/services/secrets.py`). Rewriting `owner_id` on
    `provider_credentials` without decrypting and re-sealing every row makes every stored
    credential permanently undecryptable. The rows look perfectly fine in SQL and fail at the
    next `resolve`. It also cannot be a pure SQL migration, because only the application
    holds the key.

The price of the chosen option is that the administrator's `user_id` is the machine string
`platform-admin` for ever. It is never displayed -- `display_name` is -- and `subject` carries
the email that login and `find_by_subject` match on. A cosmetic cost against a
data-destruction risk.

**No password is set here and no password appears in this file.** `password_hash` stays NULL,
which means "this account cannot password-login yet". The lifespan bootstrap in
`server/main.py` sets it from `BOOTSTRAP_ADMIN_PASSWORD` at deploy time, because the
migration is deterministic SQL that belongs in version control and a password is a secret
that must not.

Idempotent: re-running is a no-op. It refuses loudly, though, if the administrator's email
already belongs to a *different* `user_id` -- creating a second account for one person would
split their workspace in two and leave half of it unreachable.

Revision ID: 20260909_0031
Revises: 20260909_0030
"""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op

revision: str = "20260909_0031"
down_revision: str | None = "20260909_0030"
branch_labels: str | None = None
depends_on: str | None = None

# The id the shared platform key has always resolved to, and therefore the `owner_id` already
# on every credential, repository, setup, attachment and -- since 0030 -- feature. Spelled out
# rather than imported: a migration describes the schema as it was when it was written, and an
# import would let a later rename of the constant silently change what this revision did.
_ADMIN_USER_ID = "platform-admin"
_ADMIN_SUBJECT = "akhilesh@appbroda.com"
_ADMIN_DISPLAY_NAME = "Akhilesh Kumar Pandey"


def upgrade() -> None:
    """Create the administrator's account, or leave an existing one exactly as it is."""
    connection = op.get_bind()

    conflicting = connection.execute(
        sa.text("SELECT user_id FROM platform_users WHERE subject = :subject"),
        {"subject": _ADMIN_SUBJECT},
    ).scalar_one_or_none()
    if conflicting is not None and str(conflicting) != _ADMIN_USER_ID:
        # Fail rather than create a second account for one person. Their credentials, their
        # repositories and their features are filed under `platform-admin`; a second row for
        # the same email would log them into an empty workspace and leave the real one
        # reachable only by the shared platform key.
        msg = (
            f"{_ADMIN_SUBJECT} already belongs to user_id {conflicting!r}, not "
            f"{_ADMIN_USER_ID!r}. Decide which account is the administrator before applying "
            "this revision; creating a second one would split their workspace."
        )
        raise RuntimeError(msg)

    existing = connection.execute(
        sa.text("SELECT user_id FROM platform_users WHERE user_id = :user_id"),
        {"user_id": _ADMIN_USER_ID},
    ).scalar_one_or_none()
    if existing is not None:
        return

    # `now()` is written into the statement rather than bound, and `roles` is cast rather
    # than left to the driver's type inference. Both are asyncpg requirements this revision
    # was corrected for: a `sa.func.now()` passed as a bind parameter reaches asyncpg as an
    # object it refuses ("expected a datetime"), and a JSON string bound to a `json` column
    # without a cast is a text parameter the server will not coerce. Neither shows up on
    # SQLite, which is why this revision is verified against a real PostgreSQL.
    connection.execute(
        sa.text(
            "INSERT INTO platform_users ("
            "  user_id, subject, display_name, roles, disabled,"
            "  must_change_password, created_at, updated_at"
            ") VALUES ("
            "  :user_id, :subject, :display_name, CAST(:roles AS json), false,"
            "  true, now(), now()"
            ")"
        ),
        {
            "user_id": _ADMIN_USER_ID,
            "subject": _ADMIN_SUBJECT,
            "display_name": _ADMIN_DISPLAY_NAME,
            "roles": json.dumps(["admin"]),
        },
    )


def downgrade() -> None:
    """Remove the administrator's account, leaving everything it owns owned by the id.

    The row goes; nothing that names `platform-admin` changes. That is deliberate and it is
    what makes this reversible: the id keeps meaning what it meant before this revision --
    the identity behind the shared platform key -- and every credential, feature and
    repository filed under it still resolves the moment the row comes back.

    Any password set by the bootstrap is lost with the row. Rolling this back is not a way to
    undo login; see `docs/AUTHENTICATION_AND_WORKSPACES.md` for the supported direction.
    """
    op.get_bind().execute(
        sa.text("DELETE FROM platform_users WHERE user_id = :user_id"),
        {"user_id": _ADMIN_USER_ID},
    )
