"""Turning a password somebody chose into something safe to store, and back into a decision.

Its own module rather than a pair of functions beside `hash_token`, deliberately. That
function is a plain sha256, and its docstring explains why: a token is 256 bits of randomness
and there is nothing cheaper for an attacker to guess than the token itself, so a work factor
would be latency for nothing. A password a person chose is the opposite case -- a small
guessable space -- and the two must never look interchangeable to somebody reaching for one.

The KDF is scrypt from `cryptography`, which this platform already depends on: AES-GCM comes
from it, so a memory-hard KDF arrives with no new dependency, no C extension to screen, and
nothing new to audit for what it logs. `argon2id` is the marginally better primitive and is
the alternative if a reviewer prefers it; it is a new dependency for a function called once
per login.

Parameters travel in the stored string -- `scrypt$n$r$p$salt$hash` -- so raising the work
factor later rewrites rows and needs no migration, and a row sealed under the old parameters
still verifies while it waits to be rewritten.
"""

from __future__ import annotations

import secrets
from base64 import b64decode, b64encode
from dataclasses import dataclass

from cryptography.exceptions import InvalidKey
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

# The work factor a new password is sealed under. `n=2**15` is ~32 MiB of memory per
# derivation, which is the point of a memory-hard KDF: it costs an attacker's parallel
# hardware the same memory it costs this process, and it costs this process one login.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_LENGTH = 32
SALT_BYTES = 16

# The algorithm label the stored string starts with. A second algorithm added later gets its
# own label and its own branch in `verify_password`; nothing has to guess from the shape.
SCRYPT_LABEL = "scrypt"

# Shortest password this platform accepts. No composition rules: they push people to
# `Password1!`, and the threat here is credential stuffing rather than a cracking rig, which
# length answers and punctuation does not.
MIN_PASSWORD_LENGTH = 12
# Longest, so a request cannot make the KDF the denial of service. A megabyte of password is
# a megabyte of scrypt input, and the caller chooses the length.
MAX_PASSWORD_LENGTH = 1024


class PasswordPolicyError(ValueError):
    """Raised when a password cannot be accepted, with a sentence saying why.

    The message is safe to return to the caller who typed it: it describes the rule, never
    the value. It must never be used for a *verification* failure -- see `verify_password`,
    which answers a boolean precisely so that no message can distinguish which guess was
    closest.
    """


@dataclass(frozen=True, slots=True)
class _Encoded:
    """One parsed stored password hash."""

    n: int
    r: int
    p: int
    salt: bytes
    digest: bytes


def require_acceptable_password(password: str) -> None:
    """Refuse a password this platform will not store, before any work is done on it.

    Length only, and both ends of it. The lower bound is the security rule; the upper bound
    is a resource rule, and it is checked here rather than in the KDF because the whole point
    is not to hand a megabyte to scrypt.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        msg = f"a password must be at least {MIN_PASSWORD_LENGTH} characters"
        raise PasswordPolicyError(msg)
    if len(password) > MAX_PASSWORD_LENGTH:
        msg = f"a password must be at most {MAX_PASSWORD_LENGTH} characters"
        raise PasswordPolicyError(msg)


def hash_password(password: str) -> str:
    """Return the storable form of one password, with a fresh salt and the parameters used.

    The policy is checked here rather than trusted to the caller: every path that sets a
    password -- login-time change, an administrator handing one out, the startup bootstrap --
    goes through this function, and a rule enforced in three routes is a rule missing from
    the fourth.
    """
    require_acceptable_password(password)
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return "$".join(
        (
            SCRYPT_LABEL,
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            b64encode(salt).decode("ascii"),
            b64encode(digest).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str | None) -> bool:
    """Return whether one password matches one stored hash, and nothing else.

    A boolean rather than an exception with a reason, on purpose. An unparseable hash, a hash
    sealed by an algorithm this build does not know, an absent hash and a wrong password are
    all one answer here, because the caller answers all of them with the same `401` and
    anything finer would tell somebody which of their guesses was closest.

    `None` is accepted and answers `False`: an account that cannot password-login is not an
    error at this layer, and making the caller special-case it is how a `None` hash ends up
    matching an empty password.
    """
    if encoded is None:
        return False
    parsed = _parse(encoded)
    if parsed is None:
        return False
    try:
        Scrypt(
            salt=parsed.salt,
            length=len(parsed.digest),
            n=parsed.n,
            r=parsed.r,
            p=parsed.p,
        ).verify(password.encode("utf-8"), parsed.digest)
    except (InvalidKey, ValueError):
        # `InvalidKey` is the wrong password. `ValueError` is scrypt refusing the stored
        # parameters -- a row written by a build that allowed something this one does not.
        # Neither is a fact worth distinguishing to whoever is guessing.
        return False
    return True


def dummy_hash() -> str:
    """Return a hash of a value nobody knows, for verifying against when no user was found.

    The login route runs the KDF against this when the email matches nothing, so that "no
    such account" costs the same time as "wrong password". Without it, absence is a fast path
    and the timing tells an attacker which of their addresses are real -- which is the same
    discipline `DatabaseUserDirectory.resolve` already keeps by returning one `None` for
    every kind of failure.

    Built from fresh randomness on every call rather than from a constant, so the value is
    never a literal in this repository and nothing can be tempted to compare against it.
    """
    return hash_password(secrets.token_urlsafe(32))


def _derive(password: str, *, salt: bytes, n: int, r: int, p: int) -> bytes:
    """Run the KDF once."""
    return Scrypt(salt=salt, length=SCRYPT_LENGTH, n=n, r=r, p=p).derive(password.encode("utf-8"))


def _parse(encoded: str) -> _Encoded | None:
    """Return the parts of a stored hash, or nothing if this build cannot read it."""
    parts = encoded.split("$")
    if len(parts) != 6 or parts[0] != SCRYPT_LABEL:
        return None
    try:
        return _Encoded(
            n=int(parts[1]),
            r=int(parts[2]),
            p=int(parts[3]),
            salt=b64decode(parts[4], validate=True),
            digest=b64decode(parts[5], validate=True),
        )
    except ValueError:
        return None


__all__ = [
    "MAX_PASSWORD_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "PasswordPolicyError",
    "dummy_hash",
    "hash_password",
    "require_acceptable_password",
    "verify_password",
]
