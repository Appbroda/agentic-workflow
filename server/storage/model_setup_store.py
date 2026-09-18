"""The model setups somebody authored: a persisted role-map per owner, raw as entered.

A setup is a user-defined performance tier -- per role, the platform, model, effort and
output bound -- kept against the person who authored it, like their credentials and saved
repositories. This store keeps the values exactly as entered: the list a form re-populates
itself from needs what the person typed, and a normalized copy is how a refusal quietly turns
into a rewrite.

Validation deliberately does not live here. The one predicate
(`configs.model_roles.validate_model_setup`) is called by the routes at save and by the start
path at acceptance, so the store cannot grow a second, weaker copy of the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from storage.db import Database
from storage.models import ModelSetupModel

# How many setups one identity may keep. Generous -- twenty is far more personal presets than
# anybody compares in a sitting -- because the bound exists to stop unbounded growth in a
# user-writable table, not to ration configurations. Saved repositories shipped without one
# and the lesson recorded there is that a cap is cheap now and awkward later.
MAX_SETUPS_PER_OWNER = 20

_MAX_NAME_LENGTH = 256


class ModelSetupError(ValueError):
    """A model setup was rejected by the store, with a message meant for whoever typed it."""


@dataclass(frozen=True, slots=True)
class ModelSetup:
    """One saved setup, exactly as it is shown back."""

    setup_id: str
    owner_id: str
    name: str
    roles: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ModelSetupDirectory(Protocol):
    """Read and write one identity's model setups."""

    async def list_for_owner(self, owner_id: str) -> list[ModelSetup]:
        """Return every setup this identity has saved, by name."""

    async def get(self, setup_id: str, *, owner_id: str) -> ModelSetup | None:
        """Return one setup, or nothing when it does not exist or is not this identity's."""

    async def find(self, setup_id: str) -> ModelSetup | None:
        """Return one setup by id alone, for the edited-since verdict on a pinned feature.

        Deliberately unscoped: the caller publishes only an equality against a snapshot the
        feature already carries, never the row itself. Every path that shows or edits a setup
        goes through the owner-scoped reads above.
        """

    async def create(self, *, owner_id: str, name: str, roles: dict[str, Any]) -> ModelSetup:
        """Save one setup, generating its identifier."""

    async def update(
        self, setup_id: str, *, owner_id: str, name: str, roles: dict[str, Any]
    ) -> ModelSetup:
        """Replace one setup's fields, keeping its identifier."""

    async def delete(self, setup_id: str, *, owner_id: str) -> bool:
        """Forget one setup, and report whether there was one.

        Features already created from it are untouched: they carry their own snapshot.
        """


class DatabaseModelSetupDirectory:
    """Model setups in PostgreSQL, scoped to their owner on every read and write."""

    def __init__(self, database: Database) -> None:
        """Bind the database these setups live in."""
        self._database = database

    async def list_for_owner(self, owner_id: str) -> list[ModelSetup]:
        """Return this identity's setups, ordered by the name they show as."""
        statement = (
            select(ModelSetupModel)
            .where(ModelSetupModel.owner_id == owner_id)
            .order_by(ModelSetupModel.name, ModelSetupModel.created_at)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [_setup(row) for row in rows]

    async def get(self, setup_id: str, *, owner_id: str) -> ModelSetup | None:
        """Return one setup this identity owns, or nothing.

        Scoped by owner as well as id: an identifier is not an authorisation, and the caller
        answers "does not exist" and "is not yours" identically on purpose.
        """
        async with self._database.session() as session:
            model = await session.get(ModelSetupModel, setup_id)
            if model is None or model.owner_id != owner_id:
                return None
            return _setup(model)

    async def find(self, setup_id: str) -> ModelSetup | None:
        """Return one setup by id alone; see the protocol for why this read is unscoped."""
        async with self._database.session() as session:
            model = await session.get(ModelSetupModel, setup_id)
            return None if model is None else _setup(model)

    async def create(self, *, owner_id: str, name: str, roles: dict[str, Any]) -> ModelSetup:
        """Save one setup. The identifier is the server's; the caller never supplies one."""
        label = _name(name)
        async with self._database.session() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(ModelSetupModel)
                .where(ModelSetupModel.owner_id == owner_id)
            )
            if (count or 0) >= MAX_SETUPS_PER_OWNER:
                msg = (
                    f"you already keep {MAX_SETUPS_PER_OWNER} model setups; delete one "
                    "before saving another"
                )
                raise ModelSetupError(msg)
            model = ModelSetupModel(
                setup_id=f"setup-{uuid4()}",
                owner_id=owner_id,
                name=label,
                roles=dict(roles),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            session.add(model)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                msg = f"a model setup named {label!r} already exists"
                raise ModelSetupError(msg) from error
            return _setup(model)

    async def update(
        self, setup_id: str, *, owner_id: str, name: str, roles: dict[str, Any]
    ) -> ModelSetup:
        """Replace the fields of one saved setup. Running features keep their snapshots."""
        label = _name(name)
        async with self._database.session() as session:
            model = await session.get(ModelSetupModel, setup_id)
            if model is None or model.owner_id != owner_id:
                msg = f"model setup not found: {setup_id}"
                raise ModelSetupError(msg)
            model.name = label
            model.roles = dict(roles)
            model.updated_at = datetime.now(UTC)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                msg = f"a model setup named {label!r} already exists"
                raise ModelSetupError(msg) from error
            return _setup(model)

    async def delete(self, setup_id: str, *, owner_id: str) -> bool:
        """Forget one setup. Features already created from it keep their snapshots."""
        async with self._database.session() as session:
            model = await session.get(ModelSetupModel, setup_id)
            if model is None or model.owner_id != owner_id:
                return False
            await session.delete(model)
            await session.commit()
        return True


class InMemoryModelSetupDirectory:
    """The same contract without a database, for isolated applications."""

    def __init__(self) -> None:
        """Start with nothing saved."""
        self._rows: dict[str, ModelSetup] = {}

    async def list_for_owner(self, owner_id: str) -> list[ModelSetup]:
        """Return this identity's setups, ordered by name."""
        return sorted(
            (item for item in self._rows.values() if item.owner_id == owner_id),
            key=lambda item: (item.name, item.created_at),
        )

    async def get(self, setup_id: str, *, owner_id: str) -> ModelSetup | None:
        """Return one setup this identity owns, or nothing."""
        existing = self._rows.get(setup_id)
        if existing is None or existing.owner_id != owner_id:
            return None
        return existing

    async def find(self, setup_id: str) -> ModelSetup | None:
        """Return one setup by id alone; see the protocol for why this read is unscoped."""
        return self._rows.get(setup_id)

    async def create(self, *, owner_id: str, name: str, roles: dict[str, Any]) -> ModelSetup:
        """Save one setup, refusing a duplicate name for the same owner."""
        label = _name(name)
        mine = [item for item in self._rows.values() if item.owner_id == owner_id]
        if len(mine) >= MAX_SETUPS_PER_OWNER:
            msg = (
                f"you already keep {MAX_SETUPS_PER_OWNER} model setups; delete one before "
                "saving another"
            )
            raise ModelSetupError(msg)
        if any(item.name == label for item in mine):
            msg = f"a model setup named {label!r} already exists"
            raise ModelSetupError(msg)
        now = datetime.now(UTC)
        setup = ModelSetup(
            setup_id=f"setup-{uuid4()}",
            owner_id=owner_id,
            name=label,
            roles=dict(roles),
            created_at=now,
            updated_at=now,
        )
        self._rows[setup.setup_id] = setup
        return setup

    async def update(
        self, setup_id: str, *, owner_id: str, name: str, roles: dict[str, Any]
    ) -> ModelSetup:
        """Replace one setup's fields, keeping its identifier."""
        existing = self._rows.get(setup_id)
        if existing is None or existing.owner_id != owner_id:
            msg = f"model setup not found: {setup_id}"
            raise ModelSetupError(msg)
        label = _name(name)
        if any(
            item.owner_id == owner_id and item.name == label and item.setup_id != setup_id
            for item in self._rows.values()
        ):
            msg = f"a model setup named {label!r} already exists"
            raise ModelSetupError(msg)
        updated = ModelSetup(
            setup_id=setup_id,
            owner_id=owner_id,
            name=label,
            roles=dict(roles),
            created_at=existing.created_at,
            updated_at=datetime.now(UTC),
        )
        self._rows[setup_id] = updated
        return updated

    async def delete(self, setup_id: str, *, owner_id: str) -> bool:
        """Forget one setup, and report whether there was one."""
        existing = self._rows.get(setup_id)
        if existing is None or existing.owner_id != owner_id:
            return False
        del self._rows[setup_id]
        return True


def _name(value: str) -> str:
    """Validate the label, which is the one field with no derivation."""
    label = value.strip()
    if not label:
        msg = "a model setup needs a name"
        raise ModelSetupError(msg)
    if len(label) > _MAX_NAME_LENGTH:
        msg = f"a model setup name must be at most {_MAX_NAME_LENGTH} characters"
        raise ModelSetupError(msg)
    return label


def _setup(model: ModelSetupModel) -> ModelSetup:
    """Project one row onto the shape the API returns."""
    return ModelSetup(
        setup_id=model.setup_id,
        owner_id=model.owner_id,
        name=model.name,
        roles=dict(model.roles),
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


__all__ = [
    "MAX_SETUPS_PER_OWNER",
    "DatabaseModelSetupDirectory",
    "InMemoryModelSetupDirectory",
    "ModelSetup",
    "ModelSetupDirectory",
    "ModelSetupError",
]
