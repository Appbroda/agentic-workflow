"""A repository's checks run with the environment the repository publishes.

Validation commands often need configuration before they will run at all. Both DAM
repositories are the case in point: `next build` and `jest` load `next.config.ts`, which
imports a t3-env module that throws unless five `NEXT_PUBLIC_*` variables are set. The
platform supplied one hardcoded variable -- `CI=true` for tests -- so such a repository failed
its own build on an untouched checkout and was reported as an unrepairable defect.

Nothing here invents a value. The platform cannot derive somebody's API URL, and a placeholder
of its own choosing would be a decision about their project dressed up as a default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.process_runner import repository_subprocess_environment
from tools.repository_environment import (
    declared_validation_environment,
    example_environment_files,
)


def test_the_repository_declares_what_its_checks_need(tmp_path: Path) -> None:
    """The shape a real example file is written in, parsed."""
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "# Public API surface",
                "NEXT_PUBLIC_API_URL=https://api.example.com",
                'NEXT_PUBLIC_S3_REGION="ap-south-1"',
                "NEXT_PUBLIC_S3_ROOT_FOLDER='assets'",
                "export NEXT_PUBLIC_S3_TEMP_BUCKET=tmp-bucket",
                "",
                "  # indented comment",
                "EMPTY_ON_PURPOSE=",
            ]
        ),
        encoding="utf-8",
    )

    assert declared_validation_environment(tmp_path) == {
        "NEXT_PUBLIC_API_URL": "https://api.example.com",
        # One pair of surrounding quotes removed, both styles.
        "NEXT_PUBLIC_S3_REGION": "ap-south-1",
        "NEXT_PUBLIC_S3_ROOT_FOLDER": "assets",
        "NEXT_PUBLIC_S3_TEMP_BUCKET": "tmp-bucket",
        # Kept: the repository wrote it, and an empty value is a value.
        "EMPTY_ON_PURPOSE": "",
    }


def test_a_real_secret_file_is_never_read(tmp_path: Path) -> None:
    """The mistake this must not make, asserted against the repository's own secret rule.

    `.env`, `.env.local`, `.env.test` and `.env.defaults` are key material by
    `is_credential_shaped_path` -- the same predicate the reviewer, the engineer's context rule
    and the commit gate use. Reading one here would put a real credential into every validation
    subprocess and from there into its output.
    """
    for name in (".env", ".env.local", ".env.test", ".env.defaults", ".env.production"):
        (tmp_path / name).write_text(f"SECRET_FROM_{name.upper()}=value\n", encoding="utf-8")

    assert example_environment_files(tmp_path) == []
    assert declared_validation_environment(tmp_path) == {}


def test_a_repository_that_declares_nothing_is_unchanged(tmp_path: Path) -> None:
    """Every repository the platform already validates publishes no such file."""
    assert declared_validation_environment(tmp_path) == {}
    assert repository_subprocess_environment(
        {"CI": "true"}, declared=declared_validation_environment(tmp_path)
    ) == repository_subprocess_environment({"CI": "true"})


def test_prose_and_interpolation_are_skipped_not_evaluated(tmp_path: Path) -> None:
    """An example file is documentation, so anything that is not a plain assignment is left.

    `${OTHER}` in particular stays literal: expanding it would either leak the platform's own
    environment into the value or invent one.
    """
    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "Copy this file to .env before running the app.",
                "API_URL=${BASE}/v1",
                "1INVALID=x",
                "has spaces=x",
                "VALID=ok",
            ]
        ),
        encoding="utf-8",
    )

    assert declared_validation_environment(tmp_path) == {
        "API_URL": "${BASE}/v1",
        "VALID": "ok",
    }


@pytest.mark.parametrize(
    "name",
    [
        # Platform-controlled: redirecting any of these is the attack, not a configuration.
        "PATH",
        "GIT_TERMINAL_PROMPT",
        "PLATFORM_GIT_TOKEN",
        "NPM_CONFIG_USERCONFIG",
        "COREPACK_ENABLE_AUTO_PIN",
        # Secret-shaped: a value like this does not belong in a committed example file, and
        # honouring one would make this reader a channel for what it exists to avoid.
        "OPENAI_API_KEY",
        "AWS_SECRET_PASSWORD",
        "DATABASE_URL",
        "GITHUB_TOKEN",
    ],
)
def test_a_declared_value_can_neither_displace_nor_impersonate_platform_state(
    tmp_path: Path, name: str
) -> None:
    """What the repository may not set, whatever its example file says."""
    (tmp_path / ".env.example").write_text(f"{name}=repository-supplied\n", encoding="utf-8")

    environment = repository_subprocess_environment(
        {"CI": "true"}, declared=declared_validation_environment(tmp_path)
    )

    assert environment.get(name) != "repository-supplied"


def test_the_platforms_own_flags_win(tmp_path: Path) -> None:
    """`CI` decides whether a test runner returns at all, so it is not the repository's call.

    A watch-mode runner without it never exits, which is a hang rather than a verdict.
    """
    (tmp_path / ".env.example").write_text("CI=false\nFORCE_COLOR=9\n", encoding="utf-8")

    environment = repository_subprocess_environment(
        {"CI": "true"}, declared=declared_validation_environment(tmp_path)
    )

    assert environment["CI"] == "true"


def test_a_declared_value_reaches_the_subprocess(tmp_path: Path) -> None:
    """The point of all of the above."""
    (tmp_path / ".env.example").write_text(
        "NEXT_PUBLIC_API_URL=https://api.example.com\n", encoding="utf-8"
    )

    environment = repository_subprocess_environment(
        {"CI": "true"}, declared=declared_validation_environment(tmp_path)
    )

    assert environment["NEXT_PUBLIC_API_URL"] == "https://api.example.com"
    assert environment["CI"] == "true"
    assert "PATH" in environment


def test_an_oversized_or_unreadable_file_is_ignored(tmp_path: Path) -> None:
    """Bounded, because this reads a file the repository controls."""
    (tmp_path / ".env.example").write_text("A=1\n" + "B=x\n" * 100_000, encoding="utf-8")

    assert declared_validation_environment(tmp_path) == {}

    (tmp_path / ".env.example").write_bytes(b"\xff\xfe\x00binary")

    assert declared_validation_environment(tmp_path) == {}
