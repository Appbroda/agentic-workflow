"""Keeping a provider credential without becoming the thing that leaks it.

The platform previously held no provider secrets at all, which is the strongest possible
position and was chosen on purpose. It cost something real: every action that needed a key
asked for it again, so nobody could resume a feature without being present with the key.

This gives the value back without giving up the property. What is stored is a sealed blob,
sealed with a key that lives in the deployment's environment rather than in the database, and
bound to its owner and provider so a row moved into somebody else's name will not open. What
is returned to a caller is never the secret -- only whether one is configured, and its last
four characters, which is enough for a person to recognise their own key and useless to
anybody else.

The interface is deliberately narrower than the implementation. A deployment that wants AWS
Secrets Manager, GCP Secret Manager or Vault implements ``SecretStore`` and changes nothing
else; the database implementation here is what a local or single-node deployment uses.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import delete, select, update
from sqlalchemy.engine import CursorResult

from storage.db import Database
from storage.models import ProviderCredentialModel

# The providers a workflow can be given a credential for. Closed on purpose: an unknown
# provider name would be stored, never resolved, and quietly do nothing.
#
# `slack` is the notification bot token, not a model or repository provider. It is in this
# tuple so it gets the same storage path and the same settings row as every other credential
# (`describe_all` iterates this tuple, so the console renders it for free) -- and it is
# deliberately NOT in `api.routes.CREDENTIAL_PROVIDERS`, which is the catalogue of what a
# feature requires: a deployment that does not use Slack must not report an unmet
# prerequisite and refuse every submission.
#
# `figma` is the design-source personal access token, and it is here for exactly the two
# reasons `slack` is: the storage path, and the settings row `describe_all` renders for free.
# It is deliberately NOT in `api.routes.CREDENTIAL_PROVIDERS` either, and for a stronger
# reason than Slack's -- a design citation is something one author attaches to one
# submission, so a deployment that has never heard of Figma must submit features exactly as
# it does today, with no unmet-prerequisite banner and no refused submission.
SUPPORTED_PROVIDERS = ("openai", "anthropic", "github", "slack", "figma")

# AES-GCM's standard nonce length. Never reused for a given key: a fresh one is generated on
# every write, including a replacement of an existing credential.
_NONCE_BYTES = 12
_KEY_BYTES = 32


class SecretStoreError(RuntimeError):
    """Raised when a secret cannot be sealed, opened, or stored as asked."""


class SecretStoreUnavailableError(SecretStoreError):
    """Raised when the deployment has not been given an encryption key.

    A distinct type because it is a deployment fact rather than a caller's mistake, and the
    API answers it with "unavailable" instead of "you did something wrong".
    """


@dataclass(slots=True)
class _SealedSecret:
    """One sealed value held in memory, with the same fields the durable row has.

    A typed record rather than a dictionary because the dictionary's values were genuinely
    heterogeneous, and every read of one needed a cast that asserted something the type
    system could not check.
    """

    ciphertext: bytes
    nonce: bytes
    hint: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None


@dataclass(frozen=True, slots=True)
class SecretDescriptor:
    """Everything about a stored secret that is safe to show somebody.

    Deliberately has no field that could hold the value. A descriptor cannot leak a
    credential by being logged, serialized into a response, or put in a timeline event,
    because there is nowhere in it for the credential to be.
    """

    provider: str
    owner_id: str
    configured: bool
    hint: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    last_used_at: datetime | None = None


class SecretStore(Protocol):
    """Store and resolve one owner's provider credentials."""

    async def put(
        self, *, owner_id: str, provider: str, secret: str, expires_at: datetime | None = None
    ) -> SecretDescriptor:
        """Seal a secret for one owner, replacing any it already had for that provider."""

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """Return the secret for use in this request only, or nothing when none is stored."""

    async def describe(self, *, owner_id: str, provider: str) -> SecretDescriptor:
        """Return whether a secret is configured, and never the secret."""

    async def describe_all(self, *, owner_id: str) -> list[SecretDescriptor]:
        """Return one descriptor per supported provider, configured or not."""

    async def delete(self, *, owner_id: str, provider: str) -> bool:
        """Remove a stored secret, reporting whether there was one."""


def load_encryption_key(raw: str | None) -> tuple[bytes, str]:
    """Read the deployment's encryption key and the version label to record with it.

    The key is a base64 32-byte value from the environment, and never from the database it
    protects -- a key stored beside the ciphertext protects nothing. Accepting a short or
    malformed key would produce a store that appears to work and encrypts weakly, so it is
    refused instead.
    """
    if raw is None or not raw.strip():
        msg = "no secret encryption key is configured"
        raise SecretStoreUnavailableError(msg)
    version, _, material = raw.partition(":")
    if not material:
        # Unversioned keys are accepted so a deployment does not have to invent a label
        # before it can start; rotation needs one, and this is where it gets added.
        material, version = version, "v1"
    try:
        key = base64.b64decode(material, validate=True)
    except (ValueError, TypeError) as error:
        msg = "the secret encryption key must be base64-encoded"
        raise SecretStoreUnavailableError(msg) from error
    if len(key) != _KEY_BYTES:
        msg = f"the secret encryption key must decode to {_KEY_BYTES} bytes"
        raise SecretStoreUnavailableError(msg)
    return key, version


def generate_encryption_key() -> str:
    """Return a new key in the form the deployment configures.

    Provided so nobody has to invent their own way of producing one, and so the documented
    way of producing one is the way that is tested.
    """
    return f"v1:{base64.b64encode(os.urandom(_KEY_BYTES)).decode('ascii')}"


class EncryptedDatabaseSecretStore:
    """Seal provider credentials with AES-GCM and keep the sealed blobs in PostgreSQL."""

    def __init__(
        self,
        database: Database,
        *,
        encryption_key: str | None,
        previous_encryption_keys: Sequence[str] = (),
    ) -> None:
        """Bind the database and a versioned key ring for online rotation."""
        self._database = database
        self._key, self._key_version = load_encryption_key(encryption_key)
        self._keyring = {self._key_version: self._key}
        for raw in previous_encryption_keys:
            key, version = load_encryption_key(raw)
            if version in self._keyring:
                msg = f"secret encryption key version is configured more than once: {version}"
                raise SecretStoreUnavailableError(msg)
            self._keyring[version] = key

    async def put(
        self, *, owner_id: str, provider: str, secret: str, expires_at: datetime | None = None
    ) -> SecretDescriptor:
        """Seal a secret for one owner, replacing whatever they had for that provider."""
        _require_supported(provider)
        if not secret.strip():
            msg = "a credential must not be empty"
            raise SecretStoreError(msg)
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(self._key).encrypt(
            nonce, secret.encode("utf-8"), _associated_data(owner_id, provider)
        )
        now = datetime.now(UTC)
        async with self._database.session() as session:
            existing = (
                await session.execute(
                    select(ProviderCredentialModel).where(
                        ProviderCredentialModel.owner_id == owner_id,
                        ProviderCredentialModel.provider == provider,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    ProviderCredentialModel(
                        credential_id=f"credential-{uuid4()}",
                        owner_id=owner_id,
                        provider=provider,
                        ciphertext=ciphertext,
                        nonce=nonce,
                        key_version=self._key_version,
                        hint=_hint(secret),
                        created_at=now,
                        updated_at=now,
                        expires_at=expires_at,
                    )
                )
            else:
                await session.execute(
                    update(ProviderCredentialModel)
                    .where(ProviderCredentialModel.credential_id == existing.credential_id)
                    .values(
                        ciphertext=ciphertext,
                        nonce=nonce,
                        key_version=self._key_version,
                        hint=_hint(secret),
                        updated_at=now,
                        expires_at=expires_at,
                        # A replacement has not been used yet, and saying otherwise would
                        # make a stale key look live.
                        last_used_at=None,
                    )
                )
            await session.commit()
        return await self.describe(owner_id=owner_id, provider=provider)

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """Open a stored secret for the life of one request.

        An expired credential resolves to nothing rather than being used: the platform would
        otherwise keep presenting a key its owner has already decided to retire.
        """
        _require_supported(provider)
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(ProviderCredentialModel).where(
                        ProviderCredentialModel.owner_id == owner_id,
                        ProviderCredentialModel.provider == provider,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            expires_at = _as_utc(row.expires_at)
            if expires_at is not None and expires_at <= datetime.now(UTC):
                return None
            key = self._keyring.get(row.key_version)
            if key is None:
                msg = (
                    f"stored credential uses unavailable encryption key version "
                    f"{row.key_version!r}; configure it as a previous key before resolving"
                )
                raise SecretStoreError(msg)
            try:
                plaintext = AESGCM(key).decrypt(
                    bytes(row.nonce), bytes(row.ciphertext), _associated_data(owner_id, provider)
                )
            except InvalidTag as error:
                # Either the deployment's key changed or the row was tampered with. Both mean
                # this value cannot be trusted, and neither is something to paper over by
                # returning nothing: the owner has to be told their key needs re-entering.
                msg = (
                    "a stored credential could not be opened with this deployment's "
                    "encryption key and must be re-entered"
                )
                raise SecretStoreError(msg) from error
            now = datetime.now(UTC)
            values: dict[str, Any] = {"last_used_at": now}
            if row.key_version != self._key_version:
                # Lazy rotation keeps plaintext request-local and makes normal use migrate
                # rows without a bulk decrypt job or downtime.
                nonce = os.urandom(_NONCE_BYTES)
                values.update(
                    ciphertext=AESGCM(self._key).encrypt(
                        nonce, plaintext, _associated_data(owner_id, provider)
                    ),
                    nonce=nonce,
                    key_version=self._key_version,
                    updated_at=now,
                )
            await session.execute(
                update(ProviderCredentialModel)
                .where(ProviderCredentialModel.credential_id == row.credential_id)
                .values(**values)
            )
            await session.commit()
            return plaintext.decode("utf-8")

    async def describe(self, *, owner_id: str, provider: str) -> SecretDescriptor:
        """Return whether a secret is configured, and never the secret."""
        _require_supported(provider)
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(ProviderCredentialModel).where(
                        ProviderCredentialModel.owner_id == owner_id,
                        ProviderCredentialModel.provider == provider,
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            return SecretDescriptor(provider=provider, owner_id=owner_id, configured=False)
        return SecretDescriptor(
            provider=provider,
            owner_id=owner_id,
            configured=True,
            hint=row.hint,
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
            expires_at=_as_utc(row.expires_at),
            last_used_at=_as_utc(row.last_used_at),
        )

    async def describe_all(self, *, owner_id: str) -> list[SecretDescriptor]:
        """Return one descriptor per supported provider, so a settings page is complete."""
        return [
            await self.describe(owner_id=owner_id, provider=provider)
            for provider in SUPPORTED_PROVIDERS
        ]

    async def delete(self, *, owner_id: str, provider: str) -> bool:
        """Remove a stored secret, reporting whether there was one."""
        _require_supported(provider)
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    delete(ProviderCredentialModel).where(
                        ProviderCredentialModel.owner_id == owner_id,
                        ProviderCredentialModel.provider == provider,
                    )
                ),
            )
            await session.commit()
            return bool(result.rowcount)


class InMemorySecretStore:
    """The same contract without a database, for isolated tests and mock deployments.

    Still encrypts. A test double that stored plaintext would let a test pass while proving
    nothing about the property the real store exists to have.
    """

    def __init__(self, *, encryption_key: str | None = None) -> None:
        """Seal with a supplied key, or a generated one when a test does not care."""
        self._key, self._key_version = load_encryption_key(
            encryption_key or generate_encryption_key()
        )
        self._rows: dict[tuple[str, str], _SealedSecret] = {}

    async def put(
        self, *, owner_id: str, provider: str, secret: str, expires_at: datetime | None = None
    ) -> SecretDescriptor:
        """Seal a secret for one owner."""
        _require_supported(provider)
        if not secret.strip():
            msg = "a credential must not be empty"
            raise SecretStoreError(msg)
        nonce = os.urandom(_NONCE_BYTES)
        now = datetime.now(UTC)
        existing = self._rows.get((owner_id, provider))
        self._rows[(owner_id, provider)] = _SealedSecret(
            ciphertext=AESGCM(self._key).encrypt(
                nonce, secret.encode("utf-8"), _associated_data(owner_id, provider)
            ),
            nonce=nonce,
            hint=_hint(secret),
            created_at=existing.created_at if existing else now,
            updated_at=now,
            expires_at=expires_at,
            last_used_at=None,
        )
        return await self.describe(owner_id=owner_id, provider=provider)

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """Open a stored secret for the life of one request."""
        _require_supported(provider)
        row = self._rows.get((owner_id, provider))
        if row is None:
            return None
        if row.expires_at is not None and row.expires_at <= datetime.now(UTC):
            return None
        try:
            plaintext = AESGCM(self._key).decrypt(
                row.nonce, row.ciphertext, _associated_data(owner_id, provider)
            )
        except InvalidTag as error:
            msg = (
                "a stored credential could not be opened with this deployment's encryption "
                "key and must be re-entered"
            )
            raise SecretStoreError(msg) from error
        row.last_used_at = datetime.now(UTC)
        return plaintext.decode("utf-8")

    async def describe(self, *, owner_id: str, provider: str) -> SecretDescriptor:
        """Return whether a secret is configured, and never the secret."""
        _require_supported(provider)
        row = self._rows.get((owner_id, provider))
        if row is None:
            return SecretDescriptor(provider=provider, owner_id=owner_id, configured=False)
        return SecretDescriptor(
            provider=provider,
            owner_id=owner_id,
            configured=True,
            hint=row.hint,
            created_at=row.created_at,
            updated_at=row.updated_at,
            expires_at=row.expires_at,
            last_used_at=row.last_used_at,
        )

    async def describe_all(self, *, owner_id: str) -> list[SecretDescriptor]:
        """Return one descriptor per supported provider."""
        return [
            await self.describe(owner_id=owner_id, provider=provider)
            for provider in SUPPORTED_PROVIDERS
        ]

    async def delete(self, *, owner_id: str, provider: str) -> bool:
        """Remove a stored secret, reporting whether there was one."""
        _require_supported(provider)
        return self._rows.pop((owner_id, provider), None) is not None


def _require_supported(provider: str) -> None:
    """Refuse a provider nothing would ever resolve."""
    if provider not in SUPPORTED_PROVIDERS:
        msg = f"unsupported credential provider: {provider}"
        raise SecretStoreError(msg)


def _associated_data(owner_id: str, provider: str) -> bytes:
    """Bind a sealed value to who it belongs to and what it is for.

    AES-GCM authenticates this alongside the ciphertext, so a row copied into another
    owner's name fails to decrypt rather than handing that owner somebody else's key.
    """
    return f"{owner_id}:{provider}".encode()


def _hint(secret: str) -> str:
    """Return the last four characters, which identify a key only to whoever owns it."""
    return secret[-4:] if len(secret) >= 4 else "****"


def _as_utc(value: datetime | None) -> datetime | None:
    """Treat a naive timestamp from a database without timezone support as UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


__all__ = [
    "SUPPORTED_PROVIDERS",
    "EncryptedDatabaseSecretStore",
    "InMemorySecretStore",
    "SecretDescriptor",
    "SecretStore",
    "SecretStoreError",
    "SecretStoreUnavailableError",
    "generate_encryption_key",
    "load_encryption_key",
]
