"""Workspace-bound UTF-8 file operations with traversal and symlink protection."""

from __future__ import annotations

import os
import re
import stat
import tempfile
from contextlib import suppress
from pathlib import Path, PurePath
from typing import Protocol

type PathLike = str | Path
DEFAULT_MAX_FILE_BYTES = 1_048_576

# Directories whose contents are installed, generated, or version-control internals, and so
# describe no repository's own conventions.
IGNORED_CONTEXT_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
    }
)
_CREDENTIAL_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_CREDENTIAL_STEMS = frozenset({"id_ed25519", "id_rsa"})
# A name saying "credential" or "secret" only means key material when it also carries a data
# format. `secretRotation.ts` is a module about secrets; `secrets.yaml` is the secrets.
_CREDENTIAL_DATA_SUFFIXES = frozenset({"", ".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"})
_TEMPLATE_MARKERS = frozenset({"dist", "example", "sample", "template"})
# Bounded because callers hand over text they have already read for another purpose, and an
# unbounded scan over every context candidate is a denial-of-service surface. A key at byte
# 400,000 of a minified bundle is deliberately out of scope.
_KEY_MATERIAL_SCAN_CHARACTERS = 8_192
_KEY_MATERIAL_PATTERNS = (
    # Every PEM private-key flavour: RSA, EC, OPENSSH, ENCRYPTED, and PGP's `... BLOCK`.
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"),
    # A Google service-account key, which no filename rule can see.
    re.compile(r'"type"\s*:\s*"service_account"'),
)


def is_credential_shaped_path(path: PurePath) -> bool:
    """Report whether a checkout path names key material, judging by its filename alone.

    This is the single answer to "is this key material". Engineer context, reviewer evidence
    and the autonomous commit gate all ask it, because copies of the rule drift apart
    silently and the direction they drift is a credential escaping. What legitimately differs
    between those callers is the *policy* applied to a positive answer -- context drops the
    file, review redacts it and records the limitation, because there a credential-shaped
    path the change itself touched is evidence rather than a leak, and a commit refuses
    outright, because a change that thinks it needs to commit a key is wrong about something
    larger and staging the rest of it silently would hide that -- never the answer.

    A filename is only half the question. `imls-1566290608647-c1739c763650.json` is a live
    service-account key and is indistinguishable from `package.json` by name, so no rule
    here can catch it; `carries_key_material` answers the other half from the bytes.
    """
    name = path.name.lower()
    if _is_environment_file(name):
        return not _is_explicit_template(name)
    return (
        name in _CREDENTIAL_STEMS
        or path.suffix.lower() in _CREDENTIAL_SUFFIXES
        or (
            any(marker in name for marker in ("credential", "secret"))
            and path.suffix.lower() in _CREDENTIAL_DATA_SUFFIXES
            and not _is_explicit_template(name)
        )
    )


def carries_key_material(content: str) -> bool:
    """Report whether a file's own bytes carry key material, whatever the file is called.

    Deliberately two unambiguous structural markers and no entropy heuristic. A general
    secret scanner produces false positives, and a false positive here silently withholds a
    file the Engineer needed, which is the failure that made an attempt guess at the shape
    of a module it was never shown. New markers are cheap to add; a scoring model is not.

    Documentation quoting a PEM header in a fenced example matches and is withheld. That is
    the deliberate choice: withholding a README is a much smaller harm than sending a key.
    """
    head = content[:_KEY_MATERIAL_SCAN_CHARACTERS]
    return any(pattern.search(head) for pattern in _KEY_MATERIAL_PATTERNS)


def read_key_material_scan_prefix(path: PathLike) -> str:
    """Read exactly the prefix `carries_key_material` inspects, and no more.

    Callers holding a file's text for another purpose pass it straight to
    `carries_key_material`. A caller reading a file *solely* to ask the question -- the
    commit gate -- must neither read a whole file to do it nor choose its own bound. Two
    bounds are two rules, and drifting rules about key material is the defect this module
    exists to close.

    Decode-tolerant, because "can this be decoded as UTF-8" was never the question. Both
    markers are ASCII, so a byte scan finds them exactly where a string scan does, and a file
    that is simply not text carries neither. Letting `UnicodeDecodeError` escape here meant
    the commit gate refused every binary a repository could contain -- an image, a font, an
    icon, a compiled asset -- with a message saying the file could not be read, which was
    false: it reads perfectly. Undecodable bytes become replacement characters and are
    inspected like any others; a PEM inside a `.txt` still matches.

    Raises whatever the read raises. Every caller so far fails closed on that, but they fail
    closed differently -- context drops the file, a commit refuses outright -- so the policy
    stays with the caller. Only *undecodable* stops being a failure: a permissions error, a
    vanished file and a device error all still raise.
    """
    with Path(path).open("r", encoding="utf-8", errors="replace") as source:
        return source.read(_KEY_MATERIAL_SCAN_CHARACTERS)


def _is_environment_file(name: str) -> bool:
    """Recognize an environment file however the repository chose to spell it.

    `.env`, `.env.production` and `sendgrid.env` are the same thing. Matching the prefix
    alone sent `sendgrid.env`'s live API key to a provider on five engineer calls.
    """
    return name == ".env" or name.startswith(".env.") or name.endswith(".env")


def _is_explicit_template(name: str) -> bool:
    """Permit a credential-shaped name only when a component of it says template.

    `.env.example` and `secrets.sample.yaml` are committed deliberately and hold no key.
    """
    parts = set(name.replace("-", ".").replace("_", ".").split("."))
    return bool(parts & _TEMPLATE_MARKERS)


def is_model_safe_context_path(path: PurePath) -> bool:
    """Report whether a checkout path may be placed in a model request as repository context.

    Path half only, and it defers to `is_credential_shaped_path` for the credential question
    so there stays exactly one of those. A caller that has the file's bytes must also ask
    `carries_key_material`; a path that passes here is not thereby safe to send.
    """
    return not (
        any(part.lower() in IGNORED_CONTEXT_DIRECTORIES for part in path.parts)
        or is_credential_shaped_path(path)
    )


class WorkspacePathError(ValueError):
    """Raised when an operation attempts to access a path outside its workspace."""


class PatchApplicationError(ValueError):
    """Raised when a requested textual patch cannot be applied unambiguously."""


class FileTool(Protocol):
    """Protocol for file operations constrained to one configured workspace."""

    def read_file(self, relative_path: PathLike) -> str:
        """Read a UTF-8 file from the workspace."""

    def write_file(self, relative_path: PathLike, content: str) -> Path:
        """Atomically write a UTF-8 file within the workspace."""

    def patch_file(
        self,
        relative_path: PathLike,
        old_text: str,
        new_text: str,
        *,
        expected_replacements: int = 1,
    ) -> Path:
        """Replace a known text fragment within the workspace."""

    def list_files(
        self, relative_directory: PathLike = ".", *, pattern: str = "**/*"
    ) -> list[Path]:
        """List regular files within the workspace."""


class WorkspaceFileTools:
    """A reusable file-tool implementation bound to an immutable workspace root."""

    def __init__(
        self, workspace_root: PathLike, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    ) -> None:
        """Resolve and validate the workspace root once for subsequent operations."""
        self.workspace_root = resolve_workspace_root(workspace_root)
        self.max_file_bytes = _validate_max_file_bytes(max_file_bytes)

    def read_file(self, relative_path: PathLike) -> str:
        """Read a UTF-8 file from the configured workspace."""
        return read_file(self.workspace_root, relative_path, max_file_bytes=self.max_file_bytes)

    def write_file(self, relative_path: PathLike, content: str) -> Path:
        """Atomically write a UTF-8 file in the configured workspace."""
        return write_file(
            self.workspace_root,
            relative_path,
            content,
            max_file_bytes=self.max_file_bytes,
        )

    def patch_file(
        self,
        relative_path: PathLike,
        old_text: str,
        new_text: str,
        *,
        expected_replacements: int = 1,
    ) -> Path:
        """Apply an exact textual patch in the configured workspace."""
        return patch_file(
            self.workspace_root,
            relative_path,
            old_text,
            new_text,
            expected_replacements=expected_replacements,
            max_file_bytes=self.max_file_bytes,
        )

    def list_files(
        self, relative_directory: PathLike = ".", *, pattern: str = "**/*"
    ) -> list[Path]:
        """List regular files in the configured workspace."""
        return list_files(self.workspace_root, relative_directory, pattern=pattern)


def resolve_workspace_root(workspace_root: PathLike) -> Path:
    """Return an existing absolute directory that is safe to use as a workspace root."""
    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        msg = f"workspace root is not a directory: {root}"
        raise NotADirectoryError(msg)
    return root


def resolve_workspace_path(workspace_root: PathLike, path: PathLike) -> Path:
    """Resolve a path and prove that it remains inside the configured workspace root."""
    root = resolve_workspace_root(workspace_root)
    requested_path = Path(path).expanduser()
    candidate = requested_path if requested_path.is_absolute() else root / requested_path
    resolved_path = candidate.resolve(strict=False)
    try:
        resolved_path.relative_to(root)
    except ValueError as error:
        msg = f"path escapes workspace root: {path}"
        raise WorkspacePathError(msg) from error
    return resolved_path


def read_file(
    workspace_root: PathLike,
    relative_path: PathLike,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> str:
    """Read a regular UTF-8 file after resolving it inside the workspace root."""
    file_path = resolve_workspace_path(workspace_root, relative_path)
    if not file_path.is_file():
        msg = f"path is not a regular file: {relative_path}"
        raise FileNotFoundError(msg)
    if file_path.stat().st_size > _validate_max_file_bytes(max_file_bytes):
        msg = f"file exceeds configured size limit: {relative_path}"
        raise ValueError(msg)
    return file_path.read_text(encoding="utf-8")


def write_file(
    workspace_root: PathLike,
    relative_path: PathLike,
    content: str,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Path:
    """Atomically write UTF-8 content and return its workspace-relative path."""
    if not isinstance(content, str):
        msg = "content must be a string"
        raise TypeError(msg)
    encoded_content = content.encode("utf-8")
    if len(encoded_content) > _validate_max_file_bytes(max_file_bytes):
        msg = f"file content exceeds configured size limit: {relative_path}"
        raise ValueError(msg)

    root = resolve_workspace_root(workspace_root)
    target_path = resolve_workspace_path(root, relative_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = resolve_workspace_path(root, target_path)
    if target_path.exists() and not target_path.is_file():
        msg = f"path is not a regular file: {relative_path}"
        raise IsADirectoryError(msg)

    existing_mode = stat.S_IMODE(target_path.stat().st_mode) if target_path.exists() else 0o600
    descriptor, temporary_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        text=False,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, target_path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_path)
        raise

    return target_path.relative_to(root)


def _validate_max_file_bytes(max_file_bytes: int) -> int:
    """Require a positive byte limit before reading or writing untrusted repository files."""
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes < 1
    ):
        msg = "max_file_bytes must be a positive integer"
        raise ValueError(msg)
    return max_file_bytes


def patch_file(
    workspace_root: PathLike,
    relative_path: PathLike,
    old_text: str,
    new_text: str,
    *,
    expected_replacements: int = 1,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Path:
    """Apply an exact, bounded textual replacement through the atomic write path."""
    if not old_text:
        msg = "old_text must not be empty"
        raise PatchApplicationError(msg)
    if expected_replacements < 1:
        msg = "expected_replacements must be at least one"
        raise PatchApplicationError(msg)

    current_content = read_file(workspace_root, relative_path, max_file_bytes=max_file_bytes)
    actual_replacements = current_content.count(old_text)
    if actual_replacements != expected_replacements:
        msg = (
            f"expected {expected_replacements} replacement(s), found {actual_replacements} "
            f"in {relative_path}"
        )
        raise PatchApplicationError(msg)
    updated_content = current_content.replace(old_text, new_text, expected_replacements)
    return write_file(
        workspace_root,
        relative_path,
        updated_content,
        max_file_bytes=max_file_bytes,
    )


def list_files(
    workspace_root: PathLike,
    relative_directory: PathLike = ".",
    *,
    pattern: str = "**/*",
) -> list[Path]:
    """List sorted workspace-relative regular files matching a safe glob pattern."""
    _validate_glob_pattern(pattern)
    root = resolve_workspace_root(workspace_root)
    directory = resolve_workspace_path(root, relative_directory)
    if not directory.is_dir():
        msg = f"path is not a directory: {relative_directory}"
        raise NotADirectoryError(msg)

    files: list[Path] = []
    for candidate in directory.glob(pattern):
        try:
            resolved_candidate = resolve_workspace_path(root, candidate)
        except WorkspacePathError:
            continue
        if resolved_candidate.is_file():
            files.append(resolved_candidate.relative_to(root))
    return sorted(set(files), key=lambda path: path.as_posix())


def _validate_glob_pattern(pattern: str) -> None:
    """Reject glob patterns that could target an absolute path or parent directory."""
    if not pattern:
        msg = "glob pattern must not be empty"
        raise ValueError(msg)
    pure_pattern = PurePath(pattern)
    if pure_pattern.is_absolute() or ".." in pure_pattern.parts:
        msg = "glob pattern must stay within the workspace"
        raise WorkspacePathError(msg)
