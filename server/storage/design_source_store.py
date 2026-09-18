"""Where this deployment's designs come from: one configuration, read by one resolver.

Modelled field for field on `slack_store.DatabaseSlackConfigurationDirectory`, because a
design source is the same kind of thing a Slack workspace is: a deployment-level statement
about an outside service, plus a credential that lives in `provider_credentials` and never
here. Like that one, this dataclass is safe to show -- it has no field a token could occupy.

Two deliberate differences from the precedent.

**The status vocabulary is enforced on write.** `slack_store.CONFIGURATION_STATUSES` is
declared and exported and then never checked: only scopes and verbosity are validated, and
statuses are written as bare literals at four call sites. A vocabulary nothing enforces is a
comment. Every write path here goes through `_require_status`, so `degraded` cannot become
`degrade` in one branch and stay unnoticed until a console filter quietly matches nothing.

**`file_allowlist` is a control, and its emptiness is stated rather than implied.** A citation
is a URL somebody pastes; the resolver opens it with this deployment's token. An empty
allowlist therefore means *any file that token can read*, which is the right default for a
single-team deployment and the wrong one for a shared token. So the column exists, and the
response says which of the two a reader is looking at instead of leaving them to infer it
from an empty array.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

# One spelling of "is this a Figma file key", shared with the citation parser. An allowlist
# entry is compared against the `file_key` a citation was normalized to, so the two must agree
# by construction rather than by two regexes that were written to match.
from artifacts.design_references import DESIGN_FILE_KEY
from storage.db import Database
from storage.models import DesignSourceConfigurationModel

# The configuration's own lifecycle vocabulary, and unlike the Slack precedent it is checked.
# `active` is a saved, enabled source; `disabled` is a saved source somebody turned off, which
# is a state and not an absent row; `degraded` is what a refused check writes, with the reason
# beside it in the platform's own words.
DESIGN_SOURCE_STATUSES = ("active", "degraded", "disabled")

# How many files one deployment may allowlist, and how long one entry may be. Bounds rather
# than none because this is a request body, and unbounded is how a settings form becomes a
# way to write a megabyte into a configuration row.
MAX_ALLOWLIST_ENTRIES = 64


class DesignSourceConfigurationError(ValueError):
    """Raised when a design source cannot be saved as asked."""


@dataclass(frozen=True, slots=True)
class DesignSourceConfiguration:
    """The stored configuration, safe to show: it has no field a token could occupy."""

    configuration_id: str
    enabled: bool
    token_owner_id: str
    # The file keys citations are permitted against. Empty means any file the configured
    # token can read -- see the module docstring; the emptiness is reported, never implied.
    file_allowlist: tuple[str, ...]
    status: str
    status_reason: str | None
    updated_by: str
    created_at: datetime
    updated_at: datetime

    @property
    def permits_any_file(self) -> bool:
        """Whether this configuration constrains which files may be cited at all."""
        return not self.file_allowlist

    def permits(self, file_key: str) -> bool:
        """Whether a citation against this file key is allowed by the configuration."""
        return self.permits_any_file or file_key in self.file_allowlist


class DesignSourceConfigurationDirectory(Protocol):
    """Read and write the deployment's one design source configuration."""

    async def get(self) -> DesignSourceConfiguration | None:
        """Return the configuration, or nothing when none was ever saved."""

    async def save(
        self,
        *,
        enabled: bool,
        token_owner_id: str,
        file_allowlist: tuple[str, ...],
        updated_by: str,
    ) -> DesignSourceConfiguration:
        """Create or replace the single configuration row. A save re-activates it."""

    async def mark_degraded(self, configuration_id: str, *, reason: str) -> bool:
        """Degrade the configuration, reporting whether this call made the transition."""


class DatabaseDesignSourceConfigurationDirectory:
    """The deployment's configuration row, kept in PostgreSQL."""

    def __init__(self, database: Database) -> None:
        """Bind the database this directory reads and writes."""
        self._database = database

    async def get(self) -> DesignSourceConfiguration | None:
        """Return the configuration. The table is effectively singleton; read the newest."""
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(DesignSourceConfigurationModel)
                    .order_by(DesignSourceConfigurationModel.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        return _configuration_from(row) if row is not None else None

    async def save(
        self,
        *,
        enabled: bool,
        token_owner_id: str,
        file_allowlist: tuple[str, ...],
        updated_by: str,
    ) -> DesignSourceConfiguration:
        """Create or replace the single row.

        A save is an operator statement, so it clears any degraded status: re-saving the
        configuration is exactly the remedy the status banner asks for, and it is the one
        place that clears it -- a check that both degraded and recovered the row would be two
        writers of one fact.
        """
        allowlist = require_allowlist(file_allowlist)
        status = _require_status("active" if enabled else "disabled")
        now = datetime.now(UTC)
        async with self._database.session() as session:
            existing = (
                await session.execute(
                    select(DesignSourceConfigurationModel)
                    .order_by(DesignSourceConfigurationModel.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing is None:
                existing = DesignSourceConfigurationModel(
                    configuration_id=f"design-source-{uuid4()}",
                    created_at=now,
                )
                session.add(existing)
            existing.enabled = enabled
            existing.token_owner_id = token_owner_id
            existing.file_allowlist = list(allowlist)
            existing.updated_by = updated_by
            existing.status = status
            existing.status_reason = None
            existing.updated_at = now
            await session.commit()
            configuration = _configuration_from(existing)
        return configuration

    async def mark_degraded(self, configuration_id: str, *, reason: str) -> bool:
        """Degrade once: only the write that made the transition reports True."""
        status = _require_status("degraded")
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(DesignSourceConfigurationModel)
                    .where(
                        DesignSourceConfigurationModel.configuration_id == configuration_id,
                        DesignSourceConfigurationModel.status != status,
                    )
                    .values(status=status, status_reason=reason)
                ),
            )
            await session.commit()
            return bool(result.rowcount)


class InMemoryDesignSourceConfigurationDirectory:
    """The same contract without a database, for isolated applications and tests."""

    def __init__(self) -> None:
        """Start with no configuration, which is what a fresh deployment has."""
        self._row: DesignSourceConfiguration | None = None

    async def get(self) -> DesignSourceConfiguration | None:
        """Return the configuration, or nothing when none was ever saved."""
        return self._row

    async def save(
        self,
        *,
        enabled: bool,
        token_owner_id: str,
        file_allowlist: tuple[str, ...],
        updated_by: str,
    ) -> DesignSourceConfiguration:
        """Create or replace the single configuration."""
        allowlist = require_allowlist(file_allowlist)
        status = _require_status("active" if enabled else "disabled")
        now = datetime.now(UTC)
        previous = self._row
        self._row = DesignSourceConfiguration(
            configuration_id=(
                previous.configuration_id if previous else f"design-source-{uuid4()}"
            ),
            enabled=enabled,
            token_owner_id=token_owner_id,
            file_allowlist=allowlist,
            status=status,
            status_reason=None,
            updated_by=updated_by,
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        return self._row

    async def mark_degraded(self, configuration_id: str, *, reason: str) -> bool:
        """Degrade once."""
        status = _require_status("degraded")
        row = self._row
        if row is None or row.configuration_id != configuration_id or row.status == status:
            return False
        self._row = replace(row, status=status, status_reason=reason)
        return True


def require_allowlist(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return the allowlist as it will be stored, or refuse it with a readable reason.

    Refuses a pasted URL rather than storing one. An allowlist entry is compared against the
    `file_key` a citation was normalized to, so a stored URL would match nothing, ever -- the
    control would be on, appear configured, and refuse every citation. That is the "control
    that lies" the Slack configuration request's docstring refuses.
    """
    cleaned: list[str] = []
    for value in values:
        entry = value.strip()
        if not entry:
            continue
        if not DESIGN_FILE_KEY.fullmatch(entry):
            msg = (
                f"'{entry}' is not a Figma file key. Paste the key from the URL "
                "(the segment after /design/ or /file/), not the whole URL"
            )
            raise DesignSourceConfigurationError(msg)
        if entry not in cleaned:
            cleaned.append(entry)
    if len(cleaned) > MAX_ALLOWLIST_ENTRIES:
        msg = f"a design source may allowlist at most {MAX_ALLOWLIST_ENTRIES} files"
        raise DesignSourceConfigurationError(msg)
    return tuple(cleaned)


def _require_status(value: str) -> str:
    """Refuse a status nothing would ever match.

    The check the Slack precedent declares and never makes. Its `CONFIGURATION_STATUSES` is
    exported and then bypassed by four bare-literal writes, so a typo there would reach the
    column and the console filter that reads it would silently match nothing.
    """
    if value not in DESIGN_SOURCE_STATUSES:
        msg = f"unknown design source status: {value}"
        raise DesignSourceConfigurationError(msg)
    return value


def _configuration_from(row: DesignSourceConfigurationModel) -> DesignSourceConfiguration:
    """Read one configuration row into the frozen shape callers hold.

    Deliberately does not re-validate the stored status. A row already written must stay
    readable; enforcement belongs on the way in, where it can still refuse.
    """
    stored = row.file_allowlist
    return DesignSourceConfiguration(
        configuration_id=row.configuration_id,
        enabled=row.enabled,
        token_owner_id=row.token_owner_id,
        file_allowlist=tuple(str(item) for item in stored) if stored else (),
        status=row.status,
        status_reason=row.status_reason,
        updated_by=row.updated_by,
        created_at=_require_utc(row.created_at),
        updated_at=_require_utc(row.updated_at),
    )


def _require_utc(value: datetime) -> datetime:
    """Treat a naive timestamp from a database without timezone support as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


__all__ = [
    "DESIGN_FILE_KEY",
    "DESIGN_SOURCE_STATUSES",
    "MAX_ALLOWLIST_ENTRIES",
    "DatabaseDesignSourceConfigurationDirectory",
    "DesignSourceConfiguration",
    "DesignSourceConfigurationDirectory",
    "DesignSourceConfigurationError",
    "InMemoryDesignSourceConfigurationDirectory",
    "require_allowlist",
]
