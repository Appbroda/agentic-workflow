"""Where the platform's users and their tokens live.

A token is looked up by its digest, which is indexed and unique -- so authenticating a
request is one indexed read rather than a scan comparing candidates. That matters for more
than speed: a scan would make request time depend on how many tokens exist and where in the
table the match sits, which is exactly the kind of thing that leaks.

Passwords live here too, as a hash this module never derives itself: `api.passwords` owns the
KDF, and this module owns when a hash is written and which tokens a write invalidates. The
plaintext appears in exactly two places -- an argument to `set_password` and an argument to
`authenticate` -- and is never stored, logged or returned.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError

from api.identity import Actor, Role, hash_token, issue_token, token_is_usable
from api.passwords import dummy_hash, hash_password, verify_password
from storage.db import Database
from storage.models import (
    TOKEN_KIND_API,
    TOKEN_KIND_SESSION,
    PlatformApiTokenModel,
    PlatformUserModel,
)


def normalize_subject(subject: str) -> str:
    """Return the one spelling of a login identifier this platform stores and looks up by.

    Lowercased and trimmed, on both write and lookup, and applied here rather than trusted to
    callers. `Akhilesh@appbroda.com` and `akhilesh@appbroda.com` are one person, and without
    this the unique constraint would happily hold two accounts for them -- with separate
    credentials, separate features and no way to notice.
    """
    return subject.strip().lower()


@dataclass(frozen=True, slots=True)
class PlatformUser:
    """One identity the platform can distinguish."""

    user_id: str
    subject: str
    display_name: str
    roles: tuple[str, ...]
    disabled: bool
    created_at: datetime
    # Whether this account can password-login at all. A boolean rather than the hash: no
    # caller of this store has any use for the hash, and a projection that cannot carry it
    # cannot leak it into a response, a log line or a debugger's repr.
    has_password: bool = False
    must_change_password: bool = False
    last_login_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class IssuedToken:
    """A newly minted token, returned once and never retrievable again."""

    token_id: str
    user_id: str
    label: str
    token: str
    expires_at: datetime | None
    kind: str = TOKEN_KIND_API


@dataclass(frozen=True, slots=True)
class TokenSummary:
    """What can safely be shown about a token that already exists."""

    token_id: str
    user_id: str
    label: str
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None
    last_used_at: datetime | None
    kind: str = TOKEN_KIND_API


class UserDirectoryError(RuntimeError):
    """Raised when a user or token cannot be created or changed as asked."""


class SubjectAlreadyRegisteredError(UserDirectoryError):
    """Raised when a subject already belongs to an account.

    Its own type because the API answers it `409` and answers every other directory refusal
    `400`. Raised from the unique constraint rather than from a check before the insert: two
    simultaneous registrations both pass a pre-check and only the database can decide.
    """


class DatabaseUserDirectory:
    """Persist identities and their tokens beside everything else the platform owns."""

    def __init__(self, database: Database) -> None:
        """Bind the shared database."""
        self._database = database

    async def create_user(
        self,
        *,
        subject: str,
        display_name: str,
        roles: tuple[str, ...],
        password: str | None = None,
        must_change_password: bool = False,
    ) -> PlatformUser:
        """Add one identity, refusing a role this build does not define.

        An unknown role is refused at the boundary rather than stored and ignored later: a
        deployment that thinks it granted something it did not is worse than an error.

        A duplicate subject is decided by the unique constraint rather than by a read before
        the insert. Two administrators creating the same person at the same moment both pass
        a pre-check; only the database can refuse the second one.
        """
        unknown = [item for item in roles if item not in {role.value for role in Role}]
        if unknown:
            msg = f"unknown roles: {sorted(unknown)}"
            raise UserDirectoryError(msg)
        normalized = normalize_subject(subject)
        if not normalized or not display_name.strip():
            msg = "a user requires a subject and a display name"
            raise UserDirectoryError(msg)
        row = PlatformUserModel(
            user_id=f"user-{uuid4()}",
            subject=normalized,
            display_name=display_name,
            roles=list(roles),
            disabled=False,
            password_hash=None if password is None else hash_password(password),
            password_updated_at=None if password is None else datetime.now(UTC),
            must_change_password=must_change_password,
            created_at=datetime.now(UTC),
        )
        async with self._database.session() as session:
            session.add(row)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                msg = f"a user already exists for subject: {normalized}"
                raise SubjectAlreadyRegisteredError(msg) from error
            await session.refresh(row)
            return _as_user(row)

    async def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        roles: tuple[str, ...] | None = None,
        disabled: bool | None = None,
    ) -> PlatformUser:
        """Change what an identity is called, what it may do, or whether it works at all.

        Only the fields named are touched, so an administrator changing a role does not have
        to restate a display name and risk clobbering it. `subject` is deliberately not
        changeable here: it is the login identifier and the field an identity provider maps
        onto, and moving it silently re-points somebody's account at a different person.

        The last-administrator check is *not* here. It belongs to the caller that knows the
        whole population -- see `enabled_administrator_count` -- because the answer depends on
        every other row, not on this one.
        """
        if roles is not None:
            unknown = [item for item in roles if item not in {role.value for role in Role}]
            if unknown:
                msg = f"unknown roles: {sorted(unknown)}"
                raise UserDirectoryError(msg)
            if not roles:
                msg = "a user requires at least one role"
                raise UserDirectoryError(msg)
        values: dict[str, Any] = {}
        if display_name is not None:
            if not display_name.strip():
                msg = "a display name must not be empty"
                raise UserDirectoryError(msg)
            values["display_name"] = display_name
        if roles is not None:
            values["roles"] = list(roles)
        if disabled is not None:
            values["disabled"] = disabled
        if values:
            async with self._database.session() as session:
                await session.execute(
                    update(PlatformUserModel)
                    .where(PlatformUserModel.user_id == user_id)
                    .values(**values)
                )
                await session.commit()
        user = await self.get(user_id)
        if user is None:
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        return user

    async def enabled_administrator_count(self) -> int:
        """Return how many enabled accounts currently hold `admin`.

        Read by the route that changes a role or disables an account, so the deployment
        cannot be left with nobody who can hand out an account. Counted in Python rather than
        in SQL because `roles` is a JSON column and a JSON containment predicate would be one
        expression on PostgreSQL and a different one on SQLite -- and both would have to
        agree with `Actor.permissions`, which is the authority on what a role name grants.
        """
        statement = select(PlatformUserModel.roles).where(PlatformUserModel.disabled.is_(False))
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return sum(1 for roles in rows if Role.ADMIN.value in (roles or ()))

    async def administrator_ids(self) -> frozenset[str]:
        """Return every account id holding `admin`, enabled or not.

        Its own method rather than a `list_users` filter because `list_users` is bounded for
        a page and this must be complete: it is read by the Slack dispatcher to decide whose
        features may be delivered to the deployment's shared channel, and a truncated answer
        would silently withhold somebody's notifications.

        Disabled administrators are included deliberately. The question this answers is "is
        this feature's workspace the deployment's own", and a temporarily disabled
        administrator's historical features are still the deployment's -- excluding them would
        make a thread stop mid-run because somebody was turned off.
        """
        statement = select(PlatformUserModel.user_id, PlatformUserModel.roles)
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return frozenset(row.user_id for row in rows if Role.ADMIN.value in (row.roles or ()))

    async def find_by_subject(self, subject: str) -> PlatformUser | None:
        """Return the identity registered for one external subject.

        The subject is normalised on the way in, so a login typed with a capital letter finds
        the account that was created with a lowercase one.
        """
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(PlatformUserModel).where(
                        PlatformUserModel.subject == normalize_subject(subject)
                    )
                )
            ).scalar_one_or_none()
        return None if row is None else _as_user(row)

    async def authenticate(self, *, subject: str, password: str) -> PlatformUser | None:
        """Return the identity a password proves, or nothing.

        One `None` for an unknown subject, a wrong password, an account with no password and
        a disabled account -- the same discipline `resolve` keeps for tokens, and for the same
        reason: distinguishing them tells somebody which of their guesses was closest.

        The KDF runs in every one of those cases. Verifying against a throwaway hash when the
        account does not exist is what stops absence being a fast path, which would otherwise
        turn this endpoint into a way to enumerate which email addresses are real.
        """
        user_row = await self._row_for_subject(normalize_subject(subject))
        stored = None if user_row is None else user_row.password_hash
        matched = verify_password(password, stored if stored is not None else dummy_hash())
        if user_row is None or stored is None or user_row.disabled or not matched:
            return None
        return _as_user(user_row)

    async def set_password(
        self,
        user_id: str,
        *,
        password: str,
        must_change: bool = False,
        keep_token_id: str | None = None,
    ) -> PlatformUser:
        """Store a new password for one identity, and stop its existing sessions.

        Session tokens are revoked here rather than by the caller because the two facts are
        one decision: a password is changed *because* the old one may be known, and leaving
        the sessions it opened alive would make the change cosmetic. `api` tokens are left
        alone deliberately -- they are automation credentials an administrator issued
        separately, and killing somebody's deploy key because they rotated their password
        would be a surprising and unrelated outage.

        `keep_token_id` spares one session: the one the person changing their own password is
        holding, so the act of securing an account does not log them out of it. An
        administrator setting somebody else's password passes nothing, and every session that
        account had stops working -- which is the point of an administrator doing it.
        """
        encoded = hash_password(password)
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(PlatformUserModel)
                    .where(PlatformUserModel.user_id == user_id)
                    .values(
                        password_hash=encoded,
                        password_updated_at=datetime.now(UTC),
                        must_change_password=must_change,
                    )
                ),
            )
            if not result.rowcount:
                await session.rollback()
                msg = f"unknown user: {user_id}"
                raise UserDirectoryError(msg)
            await session.commit()
        await self.revoke_sessions(user_id, except_token_id=keep_token_id)
        user = await self.get(user_id)
        if user is None:  # pragma: no cover - the update above proved the row exists
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        return user

    async def revoke_sessions(self, user_id: str, *, except_token_id: str | None = None) -> int:
        """Stop this identity's login sessions, optionally sparing the one asking.

        Scoped to `kind='session'` in the WHERE clause. This is the operation the `kind`
        column exists for: "log out everywhere" and "you changed your password" must not
        also mean "your scripts stopped working".
        """
        statement = (
            update(PlatformApiTokenModel)
            .where(
                PlatformApiTokenModel.user_id == user_id,
                PlatformApiTokenModel.kind == TOKEN_KIND_SESSION,
                PlatformApiTokenModel.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(UTC))
        )
        if except_token_id is not None:
            statement = statement.where(PlatformApiTokenModel.token_id != except_token_id)
        async with self._database.session() as session:
            result = cast(CursorResult[Any], await session.execute(statement))
            await session.commit()
            return int(result.rowcount or 0)

    async def record_login(self, user_id: str) -> None:
        """Note that this identity logged in, for the administrator's user list.

        Best effort by design: a failure here must not refuse a login that succeeded. The
        same posture `resolve` takes with `last_used_at`, and for the same reason -- this is
        an operational convenience, not part of the authentication decision.
        """
        async with self._database.session() as session:
            await session.execute(
                update(PlatformUserModel)
                .where(PlatformUserModel.user_id == user_id)
                .values(last_login_at=datetime.now(UTC))
            )
            await session.commit()

    async def token_id_for(self, token: str) -> str | None:
        """Return the identifier of the token a request presented, or nothing.

        Needed by logout, which revokes the presenting credential, and by a password change,
        which spares it. The digest is what is stored, so this is one indexed read and the
        plaintext never travels further than this call.
        """
        digest = hash_token(token)
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(PlatformApiTokenModel.token_id).where(
                        PlatformApiTokenModel.token_hash == digest
                    )
                )
            ).scalar_one_or_none()
        return None if row is None else str(row)

    async def _row_for_subject(self, subject: str) -> PlatformUserModel | None:
        """Return the durable row for one normalised subject, hash included.

        Private, and the only thing in this module that reads `password_hash` out of the
        database. `PlatformUser` deliberately cannot carry it.
        """
        async with self._database.session() as session:
            return (
                await session.execute(
                    select(PlatformUserModel).where(PlatformUserModel.subject == subject)
                )
            ).scalar_one_or_none()

    async def get(self, user_id: str) -> PlatformUser | None:
        """Return one identity by its platform identifier."""
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(PlatformUserModel).where(PlatformUserModel.user_id == user_id)
                )
            ).scalar_one_or_none()
        return None if row is None else _as_user(row)

    async def list_users(self, *, limit: int = 100) -> list[PlatformUser]:
        """Return the identities this deployment knows about."""
        statement = select(PlatformUserModel).order_by(PlatformUserModel.created_at).limit(limit)
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [_as_user(row) for row in rows]

    async def set_disabled(self, user_id: str, *, disabled: bool) -> PlatformUser:
        """Turn an identity off without deleting the audit records that name it."""
        async with self._database.session() as session:
            await session.execute(
                update(PlatformUserModel)
                .where(PlatformUserModel.user_id == user_id)
                .values(disabled=disabled)
            )
            await session.commit()
        user = await self.get(user_id)
        if user is None:
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        return user

    async def issue_token(
        self,
        user_id: str,
        *,
        label: str,
        expires_at: datetime | None = None,
        kind: str = TOKEN_KIND_API,
    ) -> IssuedToken:
        """Mint a token for one identity and return it exactly once."""
        user = await self.get(user_id)
        if user is None:
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        if user.disabled:
            msg = "a disabled user cannot be given a token"
            raise UserDirectoryError(msg)
        token, digest = issue_token()
        row = PlatformApiTokenModel(
            token_id=f"token-{uuid4()}",
            user_id=user_id,
            token_hash=digest,
            label=label or "unnamed",
            kind=kind,
            created_at=datetime.now(UTC),
            expires_at=expires_at,
        )
        async with self._database.session() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
        return IssuedToken(
            token_id=row.token_id,
            user_id=user_id,
            label=row.label,
            token=token,
            expires_at=expires_at,
            kind=row.kind,
        )

    async def list_tokens(self, user_id: str) -> list[TokenSummary]:
        """Return what exists for one identity, never anything that could authenticate."""
        statement = (
            select(PlatformApiTokenModel)
            .where(PlatformApiTokenModel.user_id == user_id)
            .order_by(PlatformApiTokenModel.created_at.desc())
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [
            TokenSummary(
                token_id=row.token_id,
                user_id=row.user_id,
                label=row.label,
                created_at=_as_utc(row.created_at) or datetime.now(UTC),
                expires_at=_as_utc(row.expires_at),
                revoked_at=_as_utc(row.revoked_at),
                last_used_at=_as_utc(row.last_used_at),
                kind=row.kind,
            )
            for row in rows
        ]

    async def revoke_token(self, token_id: str) -> bool:
        """Stop a token authenticating anything, keeping the record that it existed."""
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(PlatformApiTokenModel)
                    .where(
                        PlatformApiTokenModel.token_id == token_id,
                        PlatformApiTokenModel.revoked_at.is_(None),
                    )
                    .values(revoked_at=datetime.now(UTC))
                ),
            )
            await session.commit()
            return bool(result.rowcount)

    async def resolve(self, token: str) -> Actor | None:
        """Return the identity a token proves, or nothing.

        Nothing is returned for an unknown token, an expired or revoked one, and a disabled
        user alike. The caller answers all of them identically: distinguishing them in a
        response would tell somebody which of their guesses was closest.
        """
        digest = hash_token(token)
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(PlatformApiTokenModel).where(PlatformApiTokenModel.token_hash == digest)
                )
            ).scalar_one_or_none()
            if row is None or not token_is_usable(
                expires_at=_as_utc(row.expires_at), revoked_at=_as_utc(row.revoked_at)
            ):
                return None
            user = (
                await session.execute(
                    select(PlatformUserModel).where(PlatformUserModel.user_id == row.user_id)
                )
            ).scalar_one_or_none()
            if user is None or user.disabled:
                return None
            # Recorded so an operator can tell which tokens are still in use before revoking
            # one. Best effort: a failure here must not refuse a request that authenticated.
            await session.execute(
                update(PlatformApiTokenModel)
                .where(PlatformApiTokenModel.token_id == row.token_id)
                .values(last_used_at=datetime.now(UTC))
            )
            await session.commit()
            return Actor(
                actor_id=user.user_id,
                display_name=user.display_name,
                authentication="user_token",
                roles=frozenset(user.roles or ()),
            )


class InMemoryUserDirectory:
    """The same contract without a database, for isolated tests and mock deployments.

    This is the implementation most tests authenticate against, so every method the durable
    store grows has to appear here with the same behaviour -- including the parts that are
    easy to leave out because they are about a database: the normalised subject, the
    session-only revocation, and the KDF running even when the account does not exist.
    """

    def __init__(self) -> None:
        """Start with nobody registered."""
        self._users: dict[str, PlatformUser] = {}
        self._tokens: dict[str, dict[str, Any]] = {}
        # Held apart from the user records for the reason `_as_user` projects the column away:
        # nothing that hands a `PlatformUser` to a caller can hand them a hash by accident.
        self._passwords: dict[str, str] = {}

    async def create_user(
        self,
        *,
        subject: str,
        display_name: str,
        roles: tuple[str, ...],
        password: str | None = None,
        must_change_password: bool = False,
    ) -> PlatformUser:
        """Add one identity, refusing an unknown role exactly as the durable store does."""
        unknown = [item for item in roles if item not in {role.value for role in Role}]
        if unknown:
            msg = f"unknown roles: {sorted(unknown)}"
            raise UserDirectoryError(msg)
        normalized = normalize_subject(subject)
        if not normalized or not display_name.strip():
            msg = "a user requires a subject and a display name"
            raise UserDirectoryError(msg)
        if any(item.subject == normalized for item in self._users.values()):
            msg = f"a user already exists for subject: {normalized}"
            raise SubjectAlreadyRegisteredError(msg)
        user = PlatformUser(
            user_id=f"user-{uuid4()}",
            subject=normalized,
            display_name=display_name,
            roles=tuple(roles),
            disabled=False,
            has_password=password is not None,
            must_change_password=must_change_password,
            created_at=datetime.now(UTC),
        )
        self._users[user.user_id] = user
        if password is not None:
            self._passwords[user.user_id] = hash_password(password)
        return user

    async def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        roles: tuple[str, ...] | None = None,
        disabled: bool | None = None,
    ) -> PlatformUser:
        """Change what an identity is called, what it may do, or whether it works."""
        user = self._users.get(user_id)
        if user is None:
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        if roles is not None:
            unknown = [item for item in roles if item not in {role.value for role in Role}]
            if unknown:
                msg = f"unknown roles: {sorted(unknown)}"
                raise UserDirectoryError(msg)
            if not roles:
                msg = "a user requires at least one role"
                raise UserDirectoryError(msg)
        if display_name is not None and not display_name.strip():
            msg = "a display name must not be empty"
            raise UserDirectoryError(msg)
        updated = replace(
            user,
            display_name=user.display_name if display_name is None else display_name,
            roles=user.roles if roles is None else tuple(roles),
            disabled=user.disabled if disabled is None else disabled,
        )
        self._users[user_id] = updated
        return updated

    async def enabled_administrator_count(self) -> int:
        """Return how many enabled accounts currently hold `admin`."""
        return sum(
            1
            for item in self._users.values()
            if not item.disabled and Role.ADMIN.value in item.roles
        )

    async def administrator_ids(self) -> frozenset[str]:
        """Return every account id holding `admin`, enabled or not."""
        return frozenset(
            item.user_id for item in self._users.values() if Role.ADMIN.value in item.roles
        )

    async def find_by_subject(self, subject: str) -> PlatformUser | None:
        """Return the identity for one subject, matched on its normalised spelling."""
        normalized = normalize_subject(subject)
        return next((item for item in self._users.values() if item.subject == normalized), None)

    async def authenticate(self, *, subject: str, password: str) -> PlatformUser | None:
        """Return the identity a password proves, or nothing, in constant-ish time."""
        user = await self.find_by_subject(subject)
        stored = None if user is None else self._passwords.get(user.user_id)
        matched = verify_password(password, stored if stored is not None else dummy_hash())
        if user is None or stored is None or user.disabled or not matched:
            return None
        return user

    async def set_password(
        self,
        user_id: str,
        *,
        password: str,
        must_change: bool = False,
        keep_token_id: str | None = None,
    ) -> PlatformUser:
        """Store a new password and stop this identity's existing sessions."""
        user = self._users.get(user_id)
        if user is None:
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        self._passwords[user_id] = hash_password(password)
        updated = replace(user, has_password=True, must_change_password=must_change)
        self._users[user_id] = updated
        await self.revoke_sessions(user_id, except_token_id=keep_token_id)
        return updated

    async def revoke_sessions(self, user_id: str, *, except_token_id: str | None = None) -> int:
        """Stop this identity's login sessions, sparing the one asking if named."""
        revoked = 0
        for item in self._tokens.values():
            if (
                item["user_id"] == user_id
                and item["kind"] == TOKEN_KIND_SESSION
                and item["revoked_at"] is None
                and item["token_id"] != except_token_id
            ):
                item["revoked_at"] = datetime.now(UTC)
                revoked += 1
        return revoked

    async def record_login(self, user_id: str) -> None:
        """Note that this identity logged in."""
        user = self._users.get(user_id)
        if user is not None:
            self._users[user_id] = replace(user, last_login_at=datetime.now(UTC))

    async def token_id_for(self, token: str) -> str | None:
        """Return the identifier of the token a request presented, or nothing."""
        record = self._tokens.get(hash_token(token))
        return None if record is None else str(record["token_id"])

    async def get(self, user_id: str) -> PlatformUser | None:
        """Return one identity."""
        return self._users.get(user_id)

    async def list_users(self, *, limit: int = 100) -> list[PlatformUser]:
        """Return every identity, oldest first."""
        return sorted(self._users.values(), key=lambda item: item.created_at)[:limit]

    async def set_disabled(self, user_id: str, *, disabled: bool) -> PlatformUser:
        """Turn an identity off."""
        user = self._users.get(user_id)
        if user is None:
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        updated = replace(user, disabled=disabled)
        self._users[user_id] = updated
        return updated

    async def issue_token(
        self,
        user_id: str,
        *,
        label: str,
        expires_at: datetime | None = None,
        kind: str = TOKEN_KIND_API,
    ) -> IssuedToken:
        """Mint a token for one identity."""
        user = self._users.get(user_id)
        if user is None:
            msg = f"unknown user: {user_id}"
            raise UserDirectoryError(msg)
        if user.disabled:
            msg = "a disabled user cannot be given a token"
            raise UserDirectoryError(msg)
        token, digest = issue_token()
        token_id = f"token-{uuid4()}"
        self._tokens[digest] = {
            "token_id": token_id,
            "user_id": user_id,
            "label": label or "unnamed",
            "kind": kind,
            "created_at": datetime.now(UTC),
            "expires_at": expires_at,
            "revoked_at": None,
            "last_used_at": None,
        }
        return IssuedToken(
            token_id=token_id,
            user_id=user_id,
            label=label or "unnamed",
            token=token,
            expires_at=expires_at,
            kind=kind,
        )

    async def list_tokens(self, user_id: str) -> list[TokenSummary]:
        """Return this identity's tokens without anything that could authenticate."""
        return [
            TokenSummary(
                token_id=item["token_id"],
                user_id=item["user_id"],
                label=item["label"],
                created_at=item["created_at"],
                expires_at=item["expires_at"],
                revoked_at=item["revoked_at"],
                last_used_at=item["last_used_at"],
                kind=item["kind"],
            )
            for item in self._tokens.values()
            if item["user_id"] == user_id
        ]

    async def revoke_token(self, token_id: str) -> bool:
        """Stop a token authenticating anything."""
        for item in self._tokens.values():
            if item["token_id"] == token_id and item["revoked_at"] is None:
                item["revoked_at"] = datetime.now(UTC)
                return True
        return False

    async def resolve(self, token: str) -> Actor | None:
        """Return the identity a token proves, or nothing."""
        record = self._tokens.get(hash_token(token))
        if record is None or not token_is_usable(
            expires_at=record["expires_at"], revoked_at=record["revoked_at"]
        ):
            return None
        user = self._users.get(record["user_id"])
        if user is None or user.disabled:
            return None
        record["last_used_at"] = datetime.now(UTC)
        return Actor(
            actor_id=user.user_id,
            display_name=user.display_name,
            authentication="user_token",
            roles=frozenset(user.roles),
        )


def _as_user(row: PlatformUserModel) -> PlatformUser:
    """Project a durable row onto the shape the API shares.

    `password_hash` becomes a boolean here and nowhere else. This is the seam that makes it
    impossible for a hash to reach a response: no caller of this store is ever handed one.
    """
    return PlatformUser(
        user_id=row.user_id,
        subject=row.subject,
        display_name=row.display_name,
        roles=tuple(row.roles or ()),
        disabled=bool(row.disabled),
        has_password=row.password_hash is not None,
        must_change_password=bool(row.must_change_password),
        last_login_at=_as_utc(row.last_login_at),
        created_at=_as_utc(row.created_at) or datetime.now(UTC),
    )


def _as_utc(value: datetime | None) -> datetime | None:
    """Treat a naive timestamp from a database without timezone support as UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


__all__ = [
    "DatabaseUserDirectory",
    "InMemoryUserDirectory",
    "IssuedToken",
    "PlatformUser",
    "SubjectAlreadyRegisteredError",
    "TokenSummary",
    "UserDirectoryError",
    "normalize_subject",
]
