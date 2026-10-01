"""Interruptible, bounded subprocess execution for live repository operations."""

from __future__ import annotations

import asyncio
import math
import os
import re
import signal
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from services.cancellation import CancellationToken

_INHERITED_ENVIRONMENT_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SYSTEMROOT",
    "WINDIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
_REPOSITORY_OVERRIDE_KEYS = frozenset({"CI", "FORCE_COLOR", "NO_COLOR"})
_EXPLICIT_ENVIRONMENT_KEYS = frozenset(
    {
        *_INHERITED_ENVIRONMENT_KEYS,
        *_REPOSITORY_OVERRIDE_KEYS,
        # Set by `sanitized_subprocess_environment` for Corepack and Browserslist, so they
        # have to be spellable here too -- this allowlist is a second, independent check at
        # the spawn boundary, and a key the sanitizer sets but this rejects fails every
        # subprocess.
        "BROWSERSLIST_IGNORE_OLD_DATA",
        "COREPACK_ENABLE_AUTO_PIN",
        "COREPACK_ENABLE_DOWNLOAD_PROMPT",
        "GIT_ASKPASS",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_TERMINAL_PROMPT",
        "NPM_CONFIG_USERCONFIG",
        "PIP_CONFIG_FILE",
        "PLATFORM_GIT_TOKEN",
    }
)
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_ENVIRONMENT_MARKERS = (
    "API_KEY",
    "AUTHORIZATION",
    "CREDENTIAL",
    "DATABASE_URL",
    "GITHUB_TOKEN",
    "OPENAI",
    "PASSWORD",
    "REDIS_URL",
    "SECRET",
    "TOKEN",
)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """A bounded subprocess result that distinguishes timeout from cancellation."""

    command: tuple[str, ...]
    return_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    cancelled: bool = False
    output_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        """Return true only for a completed, successful, non-cancelled process."""
        return not self.timed_out and not self.cancelled and self.return_code == 0


class ProcessRunner(Protocol):
    """Run a fixed executable command while observing cancellation and timeout concurrently."""

    async def run(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
        cancellation_token: CancellationToken,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Run without a shell and terminate the complete process group when required."""


class AsyncioProcessRunner:
    """Portable `asyncio.create_subprocess_exec` implementation with process-group cleanup."""

    def __init__(
        self,
        *,
        termination_grace_seconds: float = 2.0,
        output_limit_bytes: int = 64 * 1024,
    ) -> None:
        if termination_grace_seconds <= 0:
            msg = "termination_grace_seconds must be positive"
            raise ValueError(msg)
        if output_limit_bytes < 1024:
            msg = "output_limit_bytes must be at least 1024"
            raise ValueError(msg)
        self._termination_grace_seconds = termination_grace_seconds
        self._output_limit_bytes = output_limit_bytes

    async def run(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
        cancellation_token: CancellationToken,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Run a validated argv command and stop its process group for timeout or cancellation."""
        if not command or any(not part for part in command):
            msg = "command must contain non-empty executable arguments"
            raise ValueError(msg)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            msg = "timeout_seconds must be finite and positive"
            raise ValueError(msg)
        await cancellation_token.raise_if_cancelled()
        started_at = time.monotonic()
        # ``env=None`` means "inherit everything" to the operating system. Repository
        # package scripts, validators, formatters, and Git hooks are untrusted code, so the
        # safe default is instead a deliberately small execution environment. A caller that
        # needs a request-scoped credential (only the remote Git boundary) must pass the
        # complete environment explicitly; it is never merged with the API process.
        process_environment = (
            sanitized_subprocess_environment()
            if environment is None
            else _validated_explicit_environment(environment)
        )
        if os.name != "nt":
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=process_environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=process_environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(_capture_stream(process.stdout, self._output_limit_bytes))
        stderr_task = asyncio.create_task(_capture_stream(process.stderr, self._output_limit_bytes))
        process_task = asyncio.create_task(process.wait())
        cancellation_task = asyncio.create_task(cancellation_token.wait_cancelled())
        timed_out = False
        cancelled = False
        try:
            done, _ = await asyncio.wait(
                {process_task, cancellation_task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if process_task not in done:
                if cancellation_task in done:
                    cancelled = True
                else:
                    timed_out = True
                await _terminate_process_group(
                    process, grace_seconds=self._termination_grace_seconds
                )
                await process_task
        except asyncio.CancelledError:
            await _terminate_process_group(process, grace_seconds=self._termination_grace_seconds)
            raise
        finally:
            cancellation_task.cancel()
            await asyncio.gather(cancellation_task, return_exceptions=True)
        stdout, stdout_truncated = await stdout_task
        stderr, stderr_truncated = await stderr_task
        return ProcessResult(
            command=tuple(command),
            return_code=None if (timed_out or cancelled) else process.returncode,
            stdout=_redact_output(stdout),
            stderr=_redact_output(stderr),
            duration_seconds=time.monotonic() - started_at,
            timed_out=timed_out,
            cancelled=cancelled,
            output_truncated=stdout_truncated or stderr_truncated,
        )


async def _capture_stream(
    stream: asyncio.StreamReader, output_limit_bytes: int
) -> tuple[str, bool]:
    """Drain a pipe while retaining only a bounded prefix so children cannot block on output."""
    output = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = output_limit_bytes - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return output.decode("utf-8", errors="replace"), truncated


async def _terminate_process_group(
    process: asyncio.subprocess.Process, *, grace_seconds: float
) -> None:
    """Terminate all descendants, then kill only if the grace period expires."""
    if process.returncode is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except TimeoutError:
        pass
    if process.returncode is None:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            process.kill()
    await process.wait()


_SOURCE_PROVIDER_SECRET = re.compile(
    r"\b(?:"
    r"(?:AKIA|ASIA)[0-9A-Z]{16}|"
    r"sk[-_][A-Za-z0-9_-]{8,}|"
    r"gh[pousr]_[A-Za-z0-9_-]{8,}|"
    r"github_pat_[A-Za-z0-9_-]{8,}|"
    r"glpat-[A-Za-z0-9_-]{8,}|"
    r"npm_[A-Za-z0-9_-]{8,}|"
    r"xox[baprs]-[A-Za-z0-9_-]{8,}"
    r")\b"
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s]+"),
    _SOURCE_PROVIDER_SECRET,
    re.compile(
        r"(?i)\b((?:aws_)?(?:access_key_id|secret_access_key|session_token)"
        r"\s*[=:]\s*)[^\s]+"
    ),
    re.compile(
        r"(?i)\b((?:openai_api_key|github_token|database_url|redis_url|"
        r"platform_git_token|password|secret|token)\s*[=:]\s*)[^\s]+"
    ),
    re.compile(r"(?i)(?:postgres(?:ql)?|redis(?:s)?)://[^\s]+"),
)
_SOURCE_BEARER_LITERAL = re.compile(r"(?i)(authorization:\s*bearer\s+)([A-Za-z0-9._-]{8,})")
_SOURCE_CREDENTIAL_URI = re.compile(r"(?i)(?:https?|postgres(?:ql)?|redis(?:s)?)://[^\s\"']+")
_SOURCE_SENSITIVE_LITERAL = re.compile(
    r"(?i)(?<![A-Za-z0-9_])((?:[\"'])?([A-Za-z][A-Za-z0-9_.-]{0,80})"
    r"(?:[\"'])?\]?\s*[=:]\s*)([\"'])([^\"'\r\n]+)([\"'])"
)
_SOURCE_SENSITIVE_KEY_SUFFIXES = (
    "accesskey",
    "accesskeyid",
    "apikey",
    "credential",
    "credentials",
    "databaseurl",
    "password",
    "privatekey",
    "redisurl",
    "secret",
    "token",
)
_SOURCE_PLACEHOLDER_MARKERS = frozenset(
    {
        "changeme",
        "dummy",
        "example",
        "fake",
        "placeholder",
        "replace-me",
        "sample",
        "test",
        "your-api-key",
        "your-password",
        "your-secret",
        "your-token",
    }
)


def _redact_output(value: str) -> str:
    """Avoid propagating token-like process output into review artifacts or logs."""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]" if pattern.groups else "[REDACTED]", redacted)
    # Pattern matching cannot enumerate provider token formats. Remove the actual values of
    # every credential-bearing API environment entry as a second line of defence. Repository
    # output is still never persisted as a trusted diagnostic by callers of this module.
    return _redact_known_environment_secrets(redacted)


def _redact_known_environment_secrets(value: str) -> str:
    """Remove actual credential-bearing process values without guessing source semantics."""
    redacted = value
    for key, secret in os.environ.items():
        if (
            secret
            and len(secret) >= 8
            and any(marker in key.upper() for marker in _SENSITIVE_ENVIRONMENT_MARKERS)
        ):
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def redact_output(value: str) -> str:
    """Redact token-like text from process output that will be shown to an operator."""
    return _redact_output(value)


def redact_source_credentials(value: str) -> str:
    """Redact credential literals while preserving ordinary authentication source code.

    Repository code legitimately assigns dynamic references to names such as ``token`` and
    ``password``. Treating every such assignment as a secret hid the code from review and
    caused an unrecoverable retry loop. This narrower policy withholds provider-shaped tokens,
    credential URLs, sensitive quoted literals, and actual secrets present in this process.
    """
    redacted = _SOURCE_PROVIDER_SECRET.sub("[REDACTED]", value)

    def replace_bearer(match: re.Match[str]) -> str:
        token = match.group(2)
        if not _is_high_confidence_source_secret(token):
            return match.group(0)
        return f"{match.group(1)}[REDACTED]"

    def replace_uri(match: re.Match[str]) -> str:
        uri = match.group(0)
        try:
            parsed = urlsplit(uri)
        except ValueError:
            return uri
        credentials = tuple(item for item in (parsed.username, parsed.password) if item is not None)
        return (
            "[REDACTED]"
            if any(_is_high_confidence_source_secret(item) for item in credentials)
            else uri
        )

    def replace_literal(match: re.Match[str]) -> str:
        normalized_key = re.sub(r"[^a-z0-9]", "", match.group(2).lower())
        opening = match.group(3)
        closing = match.group(5)
        literal = match.group(4)
        if (
            not normalized_key.endswith(_SOURCE_SENSITIVE_KEY_SUFFIXES)
            or opening != closing
            or not _is_high_confidence_source_secret(literal)
        ):
            return match.group(0)
        return f"{match.group(1)}{opening}[REDACTED]{closing}"

    redacted = _SOURCE_BEARER_LITERAL.sub(replace_bearer, redacted)
    redacted = _SOURCE_CREDENTIAL_URI.sub(replace_uri, redacted)
    redacted = _SOURCE_SENSITIVE_LITERAL.sub(replace_literal, redacted)
    return _redact_known_environment_secrets(redacted)


def _is_high_confidence_source_secret(value: str) -> bool:
    """Distinguish credential literals from common test and documentation placeholders."""
    candidate = value.strip()
    if not candidate or candidate == "[REDACTED]":
        return False
    lowered = re.sub(
        r"[_.\s]+",
        "-",
        candidate.lower().strip("<>{}[]()'\""),
    )
    if _SOURCE_PROVIDER_SECRET.search(candidate):
        return True
    if any(
        lowered == marker
        or any(
            lowered.startswith(f"{marker}{separator}") or lowered.endswith(f"{separator}{marker}")
            for separator in ("-", "_", ".")
        )
        for marker in _SOURCE_PLACEHOLDER_MARKERS
    ):
        return False
    if len(candidate) < 20 or any(character.isspace() for character in candidate):
        return False
    frequencies = Counter(candidate)
    entropy_per_character = -sum(
        (count / len(candidate)) * math.log2(count / len(candidate))
        for count in frequencies.values()
    )
    return entropy_per_character >= 3.0 and entropy_per_character * len(candidate) >= 60.0


def sanitized_subprocess_environment() -> dict[str, str]:
    """Return the minimal non-secret environment inherited by untrusted checkout code."""
    environment = {
        key: value
        for key in _INHERITED_ENVIRONMENT_KEYS
        if (value := os.environ.get(key)) is not None
    }
    path_entries = [
        entry
        for entry in environment.get("PATH", os.defpath).split(os.pathsep)
        if entry and Path(entry).is_absolute()
    ]
    environment["PATH"] = os.pathsep.join(path_entries) or os.defpath
    # Do not load host/user Git, pip, or npm configuration. Such files commonly contain
    # private registry credentials even when their paths are not themselves secret.
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "NPM_CONFIG_USERCONFIG": os.devnull,
            "PIP_CONFIG_FILE": os.devnull,
            # Corepack provisions the package-manager version a repository declares. Both
            # flags are load-bearing rather than tidiness:
            #
            # Without the first, a version that is not yet cached asks on stdin whether to
            # download it. Nothing here answers stdin, so the install would hang until its
            # timeout and report as a slow repository.
            #
            # Without the second, Corepack writes the version it resolved back into the
            # repository's package.json. That is a modification to a file the feature never
            # touched, arriving in the checkout before the engineer is handed it -- so it
            # would land in the diff, in the review, and in the pull request.
            "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0",
            "COREPACK_ENABLE_AUTO_PIN": "0",
            # `browserslist` (pulled in by webpack, babel-preset-env, postcss, react-scripts,
            # and jest) checks its bundled `caniuse-lite` release date on every run and warns
            # to stderr once that data is more than a few months old -- "Browserslist:
            # caniuse-lite is outdated. Please run: npx browserslist@latest --update-db." It
            # is a reminder about the *installed dependency's* own age, not a finding about
            # this feature's change, and the platform never runs an interactive `npx` update
            # on a repository's behalf -- so every repository whose lockfile predates this
            # check by a few months would show it on every install/lint/test/build,
            # indefinitely, regardless of which feature is running. This is the environment
            # variable `browserslist` itself documents for suppressing exactly that check; it
            # does not change which browsers are targeted or anything else about the build.
            "BROWSERSLIST_IGNORE_OLD_DATA": "1",
        }
    )
    return environment


def repository_subprocess_environment(
    overrides: Mapping[str, str] | None = None,
    *,
    declared: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply only execution flags that are safe for repository-controlled subprocesses.

    `overrides` are the platform's own flags and stay on a strict allowlist: they are the
    platform speaking, and an unrecognised one is a programming error worth refusing.

    `declared` are values the repository published for running its own checks, read from a
    committed example environment file. They cannot be allowlisted -- their whole purpose is
    to carry names only the repository knows -- so they are filtered instead, and they are
    merged *under* the platform's flags so nothing the repository writes can displace one.
    """
    supplied = dict(overrides or {})
    unsupported = sorted(set(supplied) - _REPOSITORY_OVERRIDE_KEYS)
    if unsupported:
        msg = f"repository subprocess environment contains unsupported keys: {unsupported}"
        raise ValueError(msg)
    return {
        **sanitized_subprocess_environment(),
        **_admissible_declared_values(declared),
        **supplied,
    }


def _admissible_declared_values(declared: Mapping[str, str] | None) -> dict[str, str]:
    """Keep repository-declared values that cannot displace or impersonate platform state.

    Three exclusions, each for its own reason. A name the platform sets itself is refused so a
    committed file cannot redirect `PATH`, silence `GIT_TERMINAL_PROMPT`, or hand a subprocess
    a `PLATFORM_GIT_TOKEN`. A name that looks like a secret is refused because a value shaped
    like a credential does not belong in an example file, and honouring one would make this
    reader a channel for exactly what it was written to avoid. A name that is not a valid
    environment variable is refused rather than repaired.
    """
    if not declared:
        return {}
    return {name: value for name, value in declared.items() if _is_repository_declared_name(name)}


def _is_repository_declared_name(name: str) -> bool:
    """Whether a repository may set this environment variable for its own checks.

    A repository's configuration names cannot be enumerated in advance -- carrying names only
    the repository knows is the entire point -- so this is a filter rather than an allowlist,
    and it excludes the two things that must never come from a checked-in file: a name the
    platform sets itself, and a name shaped like a secret.
    """
    return (
        _ENVIRONMENT_NAME.fullmatch(name) is not None
        and name not in _EXPLICIT_ENVIRONMENT_KEYS
        and not any(marker in name.upper() for marker in _SENSITIVE_ENVIRONMENT_MARKERS)
    )


def _validated_explicit_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Reject accidental full-process environments at the final spawn boundary.

    The third and last of the environment checks, and the one that runs immediately before
    `execve`. It exists to catch a caller that passed `os.environ` by mistake and to stop a
    secret reaching a repository-controlled subprocess, so both of those remain refused.

    What it can no longer do is name every acceptable key. A repository declares the variables
    its own checks need, in a file it commits, and those names are unknowable here -- so a name
    outside the platform's own allowlist is admitted when it is a valid variable that neither
    impersonates platform state nor looks like a credential. That is the same predicate the
    values passed through on the way in, applied again at the boundary rather than trusted from
    it, because a dict carries no evidence of where it has been.
    """
    supplied = {str(key): str(value) for key, value in environment.items()}
    unsupported = sorted(
        key
        for key in supplied
        if key not in _EXPLICIT_ENVIRONMENT_KEYS
        and re.fullmatch(r"GIT_CONFIG_(?:KEY|VALUE)_\d+", key) is None
        and not _is_repository_declared_name(key)
    )
    if unsupported:
        msg = f"subprocess environment contains unsupported keys: {unsupported}"
        raise ValueError(msg)
    return supplied


def subprocess_result_code(
    *,
    return_code: int | None,
    timed_out: bool = False,
    cancelled: bool = False,
    prefix: str = "SUBPROCESS",
) -> str:
    """Describe a child outcome using platform-owned fields, never child-controlled output."""
    normalized = re.sub(r"[^A-Z0-9]+", "_", prefix.upper()).strip("_") or "SUBPROCESS"
    if cancelled:
        return f"{normalized}_CANCELLED"
    if timed_out:
        return f"{normalized}_TIMED_OUT"
    if return_code == 0:
        return f"{normalized}_SUCCEEDED"
    if return_code is None:
        return f"{normalized}_FAILED_NO_EXIT_CODE"
    return f"{normalized}_FAILED_EXIT_{return_code}"


__all__ = [
    "AsyncioProcessRunner",
    "ProcessResult",
    "ProcessRunner",
    "redact_output",
    "redact_source_credentials",
    "repository_subprocess_environment",
    "sanitized_subprocess_environment",
    "subprocess_result_code",
]
