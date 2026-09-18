"""Name the hosts a changed lockfile resolves from that its committed baseline never used.

53- made an oversized changed lockfile reachable for review as deterministic facts -- an
orphan-change flag, a numstat, a streamed hash -- and said plainly that those facts are not a
defence. A swapped registry host or a substituted integrity hash is a real supply-chain
vector, and neither a size nor a line count nor a digest of the tampered bytes can see one:
the digest changes because the file changed, which is exactly what a legitimate install also
does. The reviewer is told the lockfile moved; it is not told where the packages now come
from.

This is that recorded follow-up, in the same shape as the entry it joins. The question is
deliberately narrow and answerable without reading the file: which hosts does the changed
lockfile resolve packages from that the checkout's own committed lockfile did not already
resolve from? A host the repository already used is not a finding -- it is the status quo. A
host that arrived with this change is a fact worth stating. What to make of it belongs to the
reviewer and to the human reading the review; nothing here classifies, blocks, or forces a
verdict, and no limitation is raised.

Three properties are load-bearing, and each is why a piece of this looks the way it does:

* **Memory-bounded.** The whole reason this evidence path exists is a file past the 1 MiB
  executor cap, so the file is never loaded: it is streamed in windows with an overlap large
  enough that no resolution token can straddle a boundary. The baseline side is never loaded
  either -- `ProcessRunner` retains only a 64 KiB prefix of any subprocess output, so asking
  Git for a 1.3 MB blob would silently return a fraction of it and every host past the cut
  would read as new. Each candidate host is instead *probed* against the baseline blob with a
  `git grep` that answers by exit code and prints nothing.
* **Deterministic.** Same bytes in, byte-identical field out: sorted host names, no
  timestamps, no ordering that depends on the filesystem. `_review_input_signature` is the
  idempotency key of the journaled review operation, so a field that varied between builds
  over one checkout would make crash recovery re-call the review model.
* **Honestly incomplete.** An unrecognised format, a missing baseline or a read failure
  produce a stated reason and a `None` host list -- never an empty one. An empty list means
  "compared, and nothing new"; the difference matters more here than anywhere, because the
  reassuring answer is the dangerous one to guess at.

Repository-agnostic: which files are lockfiles and which manager wrote each one comes from
`dependency_sync`'s own registry, the single place in this platform that knows a lockfile is
machine-generated output. A manager added there is understood here as soon as this module
learns where that manager records a resolution, and until it does the answer is
`unrecognized_format` rather than silence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.cancellation import CancellationToken
from services.process_runner import ProcessRunner, sanitized_subprocess_environment
from tools.dependency_sync import generated_lockfile_manager

# One window at a time, with an overlap prepended to the next: a resolution token split
# across a read boundary is still matched, because the overlap is orders of magnitude longer
# than the longest URL any package manager writes. Peak memory is one window plus one overlap
# regardless of how large the lockfile is, which is the entire point.
_SCAN_WINDOW_BYTES = 1024 * 1024
_SCAN_OVERLAP_BYTES = 8192
# How many leading bytes decide whether the file's syntax is the one its name claims.
_FORMAT_PROBE_BYTES = 64
# A ceiling on distinct hosts, for two reasons at once: it bounds what is held in memory, and
# it bounds how many baseline probes run. Real lockfiles resolve from one or two hosts; a
# handful more where Git or GitHub Packages dependencies are declared. Reaching this many is
# itself remarkable, and when it happens the field says the list was cut rather than
# presenting a truncated list as the whole answer.
_MAX_DISTINCT_HOSTS = 24
_GIT_TIMEOUT_SECONDS = 60.0

# Where each manager's own lockfile records the place a package's bytes came from. Anchored on
# those keys and no others, because "resolved hosts" means the hosts this file resolves *from*
# -- a homepage, a funding link or a repository URL is not a download, and reporting one as a
# new resolution host would be a false alarm on a lockfile that did nothing unusual.
_RESOLUTION_KEYS: dict[str, re.Pattern[bytes]] = {
    # npm, every `lockfileVersion` from 1 to 3: `"resolved": "https://registry.npmjs.org/..."`.
    "npm": re.compile(rb'"resolved"\s*:\s*"([^"\\\s]+)"'),
    # yarn classic writes `resolved "https://..."`; berry writes `resolved: "https://..."`, and
    # omits it entirely for packages that came from the configured default registry.
    "yarn": re.compile(rb'\bresolved:?[ \t]+"([^"\\\s]+)"'),
    # pnpm carries a `tarball` only where the resolution is not the default registry, and a
    # top-level `registry` in its v5/v6 shapes.
    "pnpm": re.compile(rb'\b(?:tarball|registry)\s*:\s*[\'"]?([^\s,\'"}\]]+)'),
    # uv: `url = "https://files.pythonhosted.org/..."` per artifact, plus
    # `source = { registry = "https://pypi.org/simple" }` per package.
    "uv": re.compile(rb'\b(?:url|registry)\s*=\s*[\'"]([^\'"\\\s]+)[\'"]'),
}
# Whether this manager's lockfile is a JSON object. The name says which manager wrote the
# file; this says whether the bytes agree, so a file *named* `package-lock.json` that holds
# something else is reported as an unrecognised format instead of scanned with a pattern that
# cannot match it and reported as clean.
_FORMAT_IS_JSON: dict[str, bool] = {"npm": True, "yarn": False, "pnpm": False, "uv": False}
# The authority of a resolution URL, with any userinfo dropped before the host is read. Not an
# oversight and not merely tidiness: a registry URL in a lockfile can carry a token, and this
# field is the one thing about the file that reaches the model. A host cannot leak a
# credential; `user:password@host` would.
_RESOLUTION_URL = re.compile(
    rb"^[A-Za-z][A-Za-z0-9+.\-]*://(?:[^/@\s]*@)?(\[[^\]\s]+\]|[^/?#\s:]+)"
)
# A hostname, an IPv4 literal, or a bracketed IPv6 literal -- and nothing else. Anything the
# authority pattern extracted that fails this is a malformed URL rather than a host, and is
# not reported as one.
_HOSTNAME = re.compile(r"^(?:\[[0-9A-Fa-f:.]+\]|[a-z0-9](?:[a-z0-9.\-]*[a-z0-9])?)$")

_UNRECOGNIZED_FORMAT = "unrecognized_format"
_BASELINE_UNAVAILABLE = "baseline_unavailable"
_BASELINE_UNREADABLE = "baseline_unreadable"
_UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class LockfileHostScan:
    """What a changed lockfile resolves from, and how much of that question was answerable.

    `new_hosts` is ``None`` whenever `scanned` is false, and the two are never mixed: a caller
    reading an empty tuple is reading "compared against the baseline, nothing new", which is a
    claim this platform is willing to make. A reason without a list is the honest shape of
    every case where it is not.
    """

    lockfile_format: str | None
    scanned: bool
    reason: str | None
    distinct_hosts: int | None
    new_hosts: tuple[str, ...] | None
    hosts_truncated: bool
    baseline_present: bool | None

    def __post_init__(self) -> None:
        """Refuse the two shapes a reader could misinterpret, rather than documenting them.

        A reason without a scan is the honest form and a scan without a reason is the answered
        form; either field alone leaves the other's meaning to a convention, and the whole
        value of this type is that "no new hosts" and "no answer" cannot be confused.
        """
        if self.scanned == (self.reason is not None):
            msg = "a lockfile host scan states a reason when and only when it was not performed"
            raise ValueError(msg)
        if self.scanned != (self.new_hosts is not None):
            msg = "a lockfile host scan lists new hosts when and only when it was performed"
            raise ValueError(msg)

    def as_evidence(self) -> dict[str, Any]:
        """Render the scan as the evidence entry's own field, in a stable key order."""
        return {
            "format": self.lockfile_format,
            "scanned": self.scanned,
            "reason": self.reason,
            "baseline_present": self.baseline_present,
            "distinct_hosts": self.distinct_hosts,
            "new_hosts": list(self.new_hosts) if self.new_hosts is not None else None,
            "hosts_truncated": self.hosts_truncated,
        }

    def sentence(self) -> str:
        """Say the finding in one line, for the narrative the reviewer reads beside it.

        The field is also serialised into the model's input verbatim, so this is deliberately
        the same fact rather than an interpretation of it: what changed, measured against
        what, and -- where the answer is not available -- which of the reasons it is.
        """
        # A reason is present exactly when the scan was not performed, which `__post_init__`
        # enforces, so branching on it is the same question as `not self.scanned`.
        if self.reason is not None:
            return (
                f"resolved hosts new to this change: not scanned ({_REASON_WORDING[self.reason]})"
            )
        # Stated wherever a list is stated, because a list cut at the ceiling and presented
        # without saying so reads as the whole answer.
        cut = f", and further hosts were not listed ({_MAX_DISTINCT_HOSTS} is the ceiling)"
        cut = cut if self.hosts_truncated else ""
        if self.baseline_present is False:
            named = ", ".join(self.new_hosts or ()) or "none"
            return (
                f"resolved hosts new to this change: {named} -- this lockfile did not exist at "
                "the revision this change branched from, so every host it resolves from is new"
                f"{cut}"
            )
        if not self.new_hosts:
            return (
                "resolved hosts new to this change: none -- all "
                f"{self.distinct_hosts} host(s) this lockfile resolves from are already used by "
                f"the committed lockfile at the revision this change branched from{cut}"
            )
        return (
            f"resolved hosts new to this change: {', '.join(self.new_hosts)} -- not used by the "
            f"committed lockfile at the revision this change branched from{cut}"
        )


_REASON_WORDING = {
    _UNRECOGNIZED_FORMAT: "this lockfile's format is not one this platform knows how to read "
    "resolutions from",
    _BASELINE_UNAVAILABLE: "the revision this change branched from could not be resolved, so "
    "there is nothing to compare against",
    _BASELINE_UNREADABLE: "the committed lockfile at the revision this change branched from "
    "could not be searched",
    _UNREADABLE: "the file could not be read",
}


async def scan_resolved_hosts(
    *,
    workspace: Path,
    safe_path: Path,
    relative_path: str,
    baseline_revision: str | None,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> LockfileHostScan:
    """Report which hosts this changed lockfile resolves from that its baseline did not.

    `baseline_revision` is the lineage base -- where the working branch left the default
    branch -- and ``None`` where the reviewer could not resolve one. Passing ``HEAD`` instead
    would be worse than passing nothing: an earlier approved attempt on this same working
    branch has already committed its work there, so a host that attempt introduced would read
    as one the repository always used. That is the trap 51-A exists for, and the answer here is
    the same as it is there -- an unresolvable baseline makes the question unanswerable, not
    answered.
    """
    lockfile_format = generated_lockfile_manager(relative_path)
    pattern = _RESOLUTION_KEYS.get(lockfile_format or "")
    if lockfile_format is None or pattern is None:
        return _unscanned(lockfile_format, _UNRECOGNIZED_FORMAT)
    try:
        if not _syntax_matches_format(safe_path, lockfile_format):
            return _unscanned(lockfile_format, _UNRECOGNIZED_FORMAT)
        hosts, saw_resolution_url, hosts_truncated = _streamed_hosts(safe_path, pattern)
    except OSError:
        return _unscanned(lockfile_format, _UNREADABLE)
    if saw_resolution_url and not hosts:
        # The file resolves packages from somewhere and this module could not tell where. The
        # reassuring answer would be an empty list, which is why it is not the answer given.
        return _unscanned(lockfile_format, _UNRECOGNIZED_FORMAT)
    if not hosts:
        # Nothing is resolved over the network at all -- a pnpm or berry lockfile whose every
        # package came from the configured default registry looks exactly like this. There are
        # no hosts, so there are no new ones, and no baseline is needed to say so.
        return LockfileHostScan(
            lockfile_format=lockfile_format,
            scanned=True,
            reason=None,
            distinct_hosts=0,
            new_hosts=(),
            hosts_truncated=False,
            baseline_present=None,
        )
    if baseline_revision is None or not (workspace / ".git").exists():
        return _unscanned(lockfile_format, _BASELINE_UNAVAILABLE, distinct_hosts=len(hosts))
    baseline_present = await _baseline_holds_path(
        workspace=workspace,
        relative_path=relative_path,
        baseline_revision=baseline_revision,
        process_runner=process_runner,
        cancellation_token=cancellation_token,
    )
    if baseline_present is None:
        return _unscanned(lockfile_format, _BASELINE_UNREADABLE, distinct_hosts=len(hosts))
    if not baseline_present:
        # The install created a lockfile where the checkout had none. Every host is new, and
        # `baseline_present` is what stops that reading as an accusation.
        return LockfileHostScan(
            lockfile_format=lockfile_format,
            scanned=True,
            reason=None,
            distinct_hosts=len(hosts),
            new_hosts=tuple(sorted(hosts)),
            hosts_truncated=hosts_truncated,
            baseline_present=False,
        )
    new_hosts: list[str] = []
    for host in sorted(hosts):
        used = await _baseline_uses_host(
            host,
            workspace=workspace,
            relative_path=relative_path,
            baseline_revision=baseline_revision,
            process_runner=process_runner,
            cancellation_token=cancellation_token,
        )
        if used is None:
            return _unscanned(lockfile_format, _BASELINE_UNREADABLE, distinct_hosts=len(hosts))
        if not used:
            new_hosts.append(host)
    return LockfileHostScan(
        lockfile_format=lockfile_format,
        scanned=True,
        reason=None,
        distinct_hosts=len(hosts),
        new_hosts=tuple(new_hosts),
        hosts_truncated=hosts_truncated,
        baseline_present=True,
    )


def _unscanned(
    lockfile_format: str | None, reason: str, *, distinct_hosts: int | None = None
) -> LockfileHostScan:
    """Return a scan that states why it has no answer, rather than an answer of "nothing"."""
    return LockfileHostScan(
        lockfile_format=lockfile_format,
        scanned=False,
        reason=reason,
        distinct_hosts=distinct_hosts,
        new_hosts=None,
        hosts_truncated=False,
        baseline_present=None,
    )


def _syntax_matches_format(path: Path, lockfile_format: str) -> bool:
    """Check the file's leading bytes against the syntax its manager's lockfile has.

    The name alone is not the format. `dependency_sync`'s registry says which manager wrote a
    file of this name, and that is the right authority for the question it answers, but a file
    whose bytes are not what that manager writes would be scanned with a pattern that cannot
    match and reported as resolving from nothing. One probe of the first bytes is enough to
    tell a JSON object from a text or YAML or TOML document, which is the distinction that
    separates the formats this module knows.
    """
    with path.open("rb") as source:
        leading = source.read(_FORMAT_PROBE_BYTES).lstrip()
    return leading.startswith(b"{") is _FORMAT_IS_JSON[lockfile_format]


def _streamed_hosts(path: Path, pattern: re.Pattern[bytes]) -> tuple[set[str], bool, bool]:
    """Collect the distinct hosts this lockfile resolves from without holding the file.

    Returns the hosts, whether any resolution-shaped URL was seen at all, and whether the host
    ceiling was reached. The second is what separates "resolves from nothing over the network"
    -- a real and unremarkable state for a pnpm or berry lockfile -- from "resolves from
    somewhere this module could not read", which must not be reported as the first.
    """
    hosts: set[str] = set()
    saw_resolution_url = False
    truncated = False
    overlap = b""
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_SCAN_WINDOW_BYTES), b""):
            window = overlap + chunk
            if b"://" in window:
                saw_resolution_url = True
            for match in pattern.finditer(window):
                host = _resolution_host(match.group(1))
                if host is None or host in hosts:
                    continue
                if len(hosts) >= _MAX_DISTINCT_HOSTS:
                    truncated = True
                    continue
                hosts.add(host)
            # Re-scanned with the next window, which is harmless: hosts are a set, so a match
            # found twice is recorded once.
            overlap = window[-_SCAN_OVERLAP_BYTES:]
    return hosts, saw_resolution_url, truncated


def _resolution_host(value: bytes) -> str | None:
    """Return the lowercased host a resolution points at, or ``None`` if it points nowhere.

    ``None`` covers every resolution that is not a network download: npm's `file:` and `link:`
    values, berry's `pkg@npm:1.2.3` protocol descriptors, a workspace reference. None of those
    name a host, and none of them are a supply-chain question.
    """
    match = _RESOLUTION_URL.match(value)
    if match is None:
        return None
    try:
        host = match.group(1).decode("ascii").lower()
    except UnicodeDecodeError:
        return None
    return host if _HOSTNAME.match(host) else None


async def _baseline_holds_path(
    *,
    workspace: Path,
    relative_path: str,
    baseline_revision: str,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> bool | None:
    """Report whether the baseline revision holds this path at all, or ``None`` if Git cannot.

    Asked separately because `git grep` cannot answer it: a pathspec that matches nothing in
    the revision and a pattern that matches nothing in the file both exit 1, so without this
    every host in a newly created lockfile would be reported as new with no explanation of
    why.

    Asked with `ls-tree` rather than `cat-file -e` because the exit codes have to separate
    three outcomes and `cat-file` collapses two of them: it exits 128 both for a path the
    revision does not hold and for a revision it cannot resolve. `ls-tree` succeeds either
    way and answers by whether it named anything, which leaves a non-zero exit meaning only
    what it should -- Git could not answer. Its output is one path, so nothing here depends on
    a subprocess returning bytes it may have had to truncate.
    """
    result = await process_runner.run(
        ("git", "ls-tree", "--name-only", baseline_revision, "--", relative_path),
        workspace,
        _GIT_TIMEOUT_SECONDS,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not result.succeeded:
        return None
    return bool(result.stdout.strip())


async def _baseline_uses_host(
    host: str,
    *,
    workspace: Path,
    relative_path: str,
    baseline_revision: str,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
) -> bool | None:
    """Report whether the baseline's own lockfile already resolved from this host.

    One bounded question per candidate host, answered by exit code with no output at all,
    because the blob being searched is the same file whose size started all of this: asking
    Git to print it would return only the 64 KiB prefix `ProcessRunner` retains, and every
    host past the cut would then be reported as new. `git grep` searches the whole blob inside
    Git and says only yes or no.

    The needles are fixed strings rather than a pattern, and each is anchored on both sides:
    `//` before the host and a delimiter after it. That is what makes the answer exact in the
    direction that matters -- `//evil.test/` is not a substring of `//not-evil.test/` or of
    `//evil.test.attacker.example/`, so a host cannot be waved through as already-used by
    a longer name that merely contains it.
    """
    result = await process_runner.run(
        (
            "git",
            "grep",
            "--quiet",
            "--fixed-strings",
            "--ignore-case",
            "--text",
            *_host_needles(host),
            baseline_revision,
            "--",
            relative_path,
        ),
        workspace,
        _GIT_TIMEOUT_SECONDS,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if result.timed_out or result.cancelled or result.return_code is None:
        return None
    return {0: True, 1: False}.get(result.return_code)


def _host_needles(host: str) -> Iterable[str]:
    """Every literal spelling of "resolved from this host" that a lockfile uses.

    A path follows the host in every resolution URL a package manager writes, so `//host/` is
    the ordinary case; the rest cover a port, and a host at the very end of a quoted value.
    `git grep` ORs its `-e` patterns, so all four are one search of one blob.
    """
    return [
        argument
        for delimiter in ("/", ":", '"', "'")
        for argument in ("-e", f"//{host}{delimiter}")
    ]


__all__ = ["LockfileHostScan", "scan_resolved_hosts"]
