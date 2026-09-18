"""The repositories somebody works with, saved so they are not retyped every feature.

Every feature submission used to ask for a URL and a branch per repository, and the pilot
repositories are the same three every time. This is that list, kept against the person who
saved it.

It is organisational metadata and nothing more. The identity a workflow uses is still the
`repository_id` derived from the URL when a feature is submitted, and the `type` here is a
label the person chose -- it does not tell reconnaissance or the planner anything, and both go
on deciding what a repository actually is from the checkout. That separation is deliberate: a
saved label must not be able to make the platform believe something about a repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from storage.db import Database
from storage.models import RepositoryConfigurationModel

# Labels offered in the interface. Not a closed set -- a custom label is accepted, because a
# fixed list of somebody else's repository kinds is exactly the assumption this platform
# avoids everywhere else. It is bounded in length only.
SUGGESTED_REPOSITORY_TYPES = (
    "Frontend",
    "Backend",
    "Service",
    "Worker",
    "SDK",
    "Library",
    "Infrastructure",
    "Mobile",
    "Database",
    "Other",
)

_MAX_TYPE_LENGTH = 64
_MAX_BRANCH_LENGTH = 256


class RepositoryConfigurationError(ValueError):
    """A saved repository was rejected, with a message meant for whoever typed it."""


@dataclass(frozen=True, slots=True)
class RepositoryConfiguration:
    """One saved repository, as it is shown back."""

    configuration_id: str
    owner_id: str
    name: str
    repository_url: str
    default_branch: str
    repository_type: str
    created_at: datetime
    updated_at: datetime


class RepositoryConfigurationDirectory(Protocol):
    """Read and write one identity's saved repositories."""

    async def list_for_owner(self, owner_id: str) -> list[RepositoryConfiguration]:
        """Return every repository this identity has saved, by name."""

    async def create(
        self,
        *,
        owner_id: str,
        repository_url: str,
        default_branch: str,
        repository_type: str,
    ) -> RepositoryConfiguration:
        """Save one repository, deriving its name and generating its identifier."""

    async def update(
        self,
        configuration_id: str,
        *,
        owner_id: str,
        repository_url: str,
        default_branch: str,
        repository_type: str,
    ) -> RepositoryConfiguration:
        """Replace one saved repository's fields, keeping its identifier."""

    async def delete(self, configuration_id: str, *, owner_id: str) -> bool:
        """Forget one saved repository, and report whether there was one."""


class DatabaseRepositoryConfigurationDirectory:
    """Saved repositories in PostgreSQL, scoped to their owner on every read and write."""

    def __init__(self, database: Database) -> None:
        """Bind the database these configurations live in."""
        self._database = database

    async def list_for_owner(self, owner_id: str) -> list[RepositoryConfiguration]:
        """Return this identity's saved repositories, ordered by the name they show as."""
        statement = (
            select(RepositoryConfigurationModel)
            .where(RepositoryConfigurationModel.owner_id == owner_id)
            .order_by(RepositoryConfigurationModel.name, RepositoryConfigurationModel.created_at)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [_configuration(row) for row in rows]

    async def create(
        self,
        *,
        owner_id: str,
        repository_url: str,
        default_branch: str,
        repository_type: str,
    ) -> RepositoryConfiguration:
        """Save one repository. The identifier is the server's; the caller never supplies one."""
        url = normalise_repository_url(repository_url)
        model = RepositoryConfigurationModel(
            configuration_id=f"repo-{uuid4()}",
            owner_id=owner_id,
            name=derive_repository_name(url),
            repository_url=url,
            default_branch=_branch(default_branch),
            repository_type=_repository_type(repository_type),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        async with self._database.session() as session:
            session.add(model)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                msg = "this repository is already saved"
                raise RepositoryConfigurationError(msg) from error
        return _configuration(model)

    async def update(
        self,
        configuration_id: str,
        *,
        owner_id: str,
        repository_url: str,
        default_branch: str,
        repository_type: str,
    ) -> RepositoryConfiguration:
        """Replace the fields of one saved repository, re-deriving its name from the URL."""
        url = normalise_repository_url(repository_url)
        async with self._database.session() as session:
            model = await session.get(RepositoryConfigurationModel, configuration_id)
            # Scoped by owner as well as by id: an identifier is not an authorisation, and
            # guessing one must not reach somebody else's list.
            if model is None or model.owner_id != owner_id:
                msg = f"saved repository not found: {configuration_id}"
                raise RepositoryConfigurationError(msg)
            model.repository_url = url
            model.name = derive_repository_name(url)
            model.default_branch = _branch(default_branch)
            model.repository_type = _repository_type(repository_type)
            model.updated_at = datetime.now(UTC)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                msg = "this repository is already saved"
                raise RepositoryConfigurationError(msg) from error
            return _configuration(model)

    async def delete(self, configuration_id: str, *, owner_id: str) -> bool:
        """Forget one saved repository. Features already created from it are untouched."""
        async with self._database.session() as session:
            model = await session.get(RepositoryConfigurationModel, configuration_id)
            # Scoped by owner as well as by id, for the same reason `update` is: an identifier
            # somebody guessed must not reach another person's list.
            if model is None or model.owner_id != owner_id:
                return False
            await session.delete(model)
            await session.commit()
        return True


class InMemoryRepositoryConfigurationDirectory:
    """The same contract without a database, for isolated applications."""

    def __init__(self) -> None:
        """Start with nothing saved."""
        self._rows: dict[str, RepositoryConfiguration] = {}

    async def list_for_owner(self, owner_id: str) -> list[RepositoryConfiguration]:
        """Return this identity's saved repositories, ordered by name."""
        return sorted(
            (item for item in self._rows.values() if item.owner_id == owner_id),
            key=lambda item: (item.name, item.created_at),
        )

    async def create(
        self,
        *,
        owner_id: str,
        repository_url: str,
        default_branch: str,
        repository_type: str,
    ) -> RepositoryConfiguration:
        """Save one repository, refusing a duplicate URL for the same owner."""
        url = normalise_repository_url(repository_url)
        if any(
            item.owner_id == owner_id and item.repository_url == url for item in self._rows.values()
        ):
            msg = "this repository is already saved"
            raise RepositoryConfigurationError(msg)
        now = datetime.now(UTC)
        configuration = RepositoryConfiguration(
            configuration_id=f"repo-{uuid4()}",
            owner_id=owner_id,
            name=derive_repository_name(url),
            repository_url=url,
            default_branch=_branch(default_branch),
            repository_type=_repository_type(repository_type),
            created_at=now,
            updated_at=now,
        )
        self._rows[configuration.configuration_id] = configuration
        return configuration

    async def update(
        self,
        configuration_id: str,
        *,
        owner_id: str,
        repository_url: str,
        default_branch: str,
        repository_type: str,
    ) -> RepositoryConfiguration:
        """Replace one saved repository's fields, keeping its identifier."""
        existing = self._rows.get(configuration_id)
        if existing is None or existing.owner_id != owner_id:
            msg = f"saved repository not found: {configuration_id}"
            raise RepositoryConfigurationError(msg)
        url = normalise_repository_url(repository_url)
        if any(
            item.owner_id == owner_id
            and item.repository_url == url
            and item.configuration_id != configuration_id
            for item in self._rows.values()
        ):
            msg = "this repository is already saved"
            raise RepositoryConfigurationError(msg)
        updated = RepositoryConfiguration(
            configuration_id=configuration_id,
            owner_id=owner_id,
            name=derive_repository_name(url),
            repository_url=url,
            default_branch=_branch(default_branch),
            repository_type=_repository_type(repository_type),
            created_at=existing.created_at,
            updated_at=datetime.now(UTC),
        )
        self._rows[configuration_id] = updated
        return updated

    async def delete(self, configuration_id: str, *, owner_id: str) -> bool:
        """Forget one saved repository, and report whether there was one."""
        existing = self._rows.get(configuration_id)
        if existing is None or existing.owner_id != owner_id:
            return False
        del self._rows[configuration_id]
        return True


def normalise_repository_url(value: str) -> str:
    """Validate a repository URL and return it in the one form the platform stores.

    The same rules `RepositorySpec` enforces, applied here so a repository is rejected when it
    is saved rather than when a feature built from it fails validation. `.git` is stripped and
    the trailing slash removed so the same repository saved twice in two spellings is one row.
    """
    url = value.strip()
    if not url:
        msg = "a repository URL is required"
        raise RepositoryConfigurationError(msg)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        msg = "a repository URL must be http:// or https://"
        raise RepositoryConfigurationError(msg)
    if not parsed.netloc:
        msg = "a repository URL must name a host"
        raise RepositoryConfigurationError(msg)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        msg = "a repository URL must not contain credentials, query parameters, or fragments"
        raise RepositoryConfigurationError(msg)
    path = parsed.path.rstrip("/").removesuffix(".git")
    if len([segment for segment in path.split("/") if segment]) < 1:
        msg = "a repository URL must name a repository, for example https://host/owner/name"
        raise RepositoryConfigurationError(msg)
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def derive_repository_name(url: str) -> str:
    """Return the last path segment of a normalised URL, which is the repository's name.

    Derived rather than asked for. Somebody who has just pasted a URL has already said what
    the repository is called, and a name they can disagree with the URL about is a second
    source of truth.
    """
    segments = [segment for segment in urlsplit(url).path.split("/") if segment]
    return segments[-1] if segments else urlsplit(url).netloc


def _branch(value: str) -> str:
    """Validate the default branch, which is the one field with no sensible derivation."""
    branch = value.strip()
    if not branch:
        msg = "a default branch is required"
        raise RepositoryConfigurationError(msg)
    if len(branch) > _MAX_BRANCH_LENGTH:
        msg = f"a default branch must be at most {_MAX_BRANCH_LENGTH} characters"
        raise RepositoryConfigurationError(msg)
    return branch


def _repository_type(value: str) -> str:
    """Accept any short label, including one not in the suggested set."""
    label = value.strip() or "Other"
    if len(label) > _MAX_TYPE_LENGTH:
        msg = f"a repository type must be at most {_MAX_TYPE_LENGTH} characters"
        raise RepositoryConfigurationError(msg)
    return label


def _configuration(model: RepositoryConfigurationModel) -> RepositoryConfiguration:
    """Project one row onto the shape the API returns."""
    return RepositoryConfiguration(
        configuration_id=model.configuration_id,
        owner_id=model.owner_id,
        name=model.name,
        repository_url=model.repository_url,
        default_branch=model.default_branch,
        repository_type=model.repository_type,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


__all__ = [
    "SUGGESTED_REPOSITORY_TYPES",
    "DatabaseRepositoryConfigurationDirectory",
    "InMemoryRepositoryConfigurationDirectory",
    "RepositoryConfiguration",
    "RepositoryConfigurationDirectory",
    "RepositoryConfigurationError",
    "derive_repository_name",
    "normalise_repository_url",
]
