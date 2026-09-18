"""Where a changed lockfile now resolves packages from, and what it did not resolve from before.

53- made an oversized changed lockfile reviewable as size, line count and digest, and said in
so many words that those facts are not a defence against a tampered one: a substituted
registry host changes the digest exactly as a legitimate install does. These are the tests for
the scan that answers the remaining question.

Every parsing test that can be run against a real lockfile is run against one -- this
repository's own `client/package-lock.json` and `server/uv.lock`, several hundred resolved
packages each. A hand-written lockfile proves only that the pattern matches the shape whoever
wrote the test had in mind, which is the failure mode this file exists to avoid.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from services.cancellation import MockCancellationToken
from services.process_runner import AsyncioProcessRunner, ProcessResult
from tests.fixtures import commit_all, init_git_repository, start_working_branch
from tools.lockfile_hosts import (
    _MAX_DISTINCT_HOSTS,
    _SCAN_WINDOW_BYTES,
    LockfileHostScan,
    scan_resolved_hosts,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Two lockfiles this repository commits and its own toolchain wrote. Referenced rather than
# copied into the fixtures: a saved copy is a snapshot of one manager version, and the point of
# testing against a real payload is that nobody chose which fields it carries.
_REAL_NPM_LOCKFILE = _REPOSITORY_ROOT / "client" / "package-lock.json"
_REAL_UV_LOCKFILE = _REPOSITORY_ROOT / "server" / "uv.lock"


async def _scan(
    root: Path,
    relative_path: str,
    *,
    baseline_revision: str | None,
) -> LockfileHostScan:
    """Scan one lockfile in a real checkout with a real Git, as the reviewer does."""
    return await scan_resolved_hosts(
        workspace=root,
        safe_path=root / relative_path,
        relative_path=relative_path,
        baseline_revision=baseline_revision,
        process_runner=AsyncioProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )


def _checkout_holding(root: Path, files: dict[str, str]) -> tuple[Path, str]:
    """Commit these files as the baseline and branch, returning the checkout and that revision.

    The returned revision is the lineage base -- where the working branch left the default
    branch -- because that, and never `HEAD`, is what a lockfile change is measured against.
    """
    init_git_repository(root)
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    commit_all(root, "baseline")
    return root, start_working_branch(root, "workflow/workflow-1")


def _npm_lockfile(resolutions: dict[str, str]) -> str:
    """Return a `package-lock.json` resolving each named package from the given URL."""
    packages: dict[str, Any] = {"": {"name": "service", "version": "1.0.0"}}
    for name, url in resolutions.items():
        packages[f"node_modules/{name}"] = {
            "version": "1.0.0",
            "resolved": url,
            "integrity": f"sha512-{'0' * 86}==",
        }
    document = {"name": "service", "lockfileVersion": 3, "packages": packages}
    return f"{json.dumps(document, indent=2)}\n"


def _registry_resolutions(host: str, count: int, *, prefix: str = "pkg") -> dict[str, str]:
    """Return `count` resolutions all pointing at one host."""
    return {
        f"{prefix}-{index:04d}": f"https://{host}/{prefix}-{index:04d}/-/{prefix}-{index:04d}-1.0.0.tgz"
        for index in range(count)
    }


# --------------------------------------------------------------- real payloads, real shapes


async def test_the_hosts_of_this_repository_s_own_npm_lockfile_are_read_from_it(
    tmp_path: Path,
) -> None:
    """A real `package-lock.json` resolves from exactly the registry it says it does.

    This is the payload test. `client/package-lock.json` holds several hundred resolved
    packages written by npm itself, with the `integrity`, `dev`, `license`, `engines` and
    `peerDependencies` fields a hand-written fixture would not have thought to include, and
    the answer is one host because that is the truthful answer for this checkout.
    """
    assert _REAL_NPM_LOCKFILE.is_file(), f"{_REAL_NPM_LOCKFILE} is a committed file of this repo"
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    shutil.copyfile(_REAL_NPM_LOCKFILE, root / "package-lock.json")

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.lockfile_format == "npm"
    assert scan.scanned is True
    assert scan.reason is None
    assert scan.distinct_hosts == 1
    # Untracked at the baseline, so the honest answer is that every host is new *and why*.
    assert scan.baseline_present is False
    assert scan.new_hosts == ("registry.npmjs.org",)


async def test_the_hosts_of_this_repository_s_own_uv_lockfile_are_read_from_it(
    tmp_path: Path,
) -> None:
    """A real `uv.lock` resolves from the index and the file host, in TOML rather than JSON.

    The second real payload, and a different syntax: uv records `source = { registry = ... }`
    per package and a `url` per wheel and sdist. Two hosts, both genuinely in the file, and
    npm's pattern would have found neither -- which is why the format decides the pattern.
    """
    assert _REAL_UV_LOCKFILE.is_file(), f"{_REAL_UV_LOCKFILE} is a committed file of this repo"
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    shutil.copyfile(_REAL_UV_LOCKFILE, root / "uv.lock")

    scan = await _scan(root, "uv.lock", baseline_revision=baseline)

    assert scan.lockfile_format == "uv"
    assert scan.scanned is True
    assert scan.new_hosts == ("files.pythonhosted.org", "pypi.org")


async def test_a_swapped_host_in_a_real_lockfile_is_the_one_thing_reported(
    tmp_path: Path,
) -> None:
    """The tampered variant: one resolution repointed, and only that host comes back.

    The whole point of the field. Both lockfiles are the real npm payload; the change between
    them is a single `resolved` URL whose host was substituted, which is what a supply-chain
    attack on a machine-generated file looks like -- no manifest edit, no new package, a digest
    that changes exactly as a legitimate install's would, and one line out of thousands.

    The registry the repository already used is not reported. A host is listed because it is
    new here, not because it is unfamiliar, and stating the status quo as a finding is how a
    scanner earns its way into being ignored.
    """
    original = _REAL_NPM_LOCKFILE.read_text(encoding="utf-8")
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package-lock.json": original})
    tampered = original.replace(
        "https://registry.npmjs.org/lru-cache/",
        "https://registry.substituted.test/lru-cache/",
        1,
    )
    assert tampered != original, "the real payload must still hold the entry being repointed"
    (root / "package-lock.json").write_text(tampered, encoding="utf-8")

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.scanned is True
    assert scan.baseline_present is True
    assert scan.distinct_hosts == 2
    assert scan.new_hosts == ("registry.substituted.test",)
    assert scan.hosts_truncated is False
    assert "registry.substituted.test" in scan.sentence()
    assert "registry.npmjs.org" not in scan.sentence()


async def test_an_ordinary_install_on_the_registry_the_repository_already_uses_reports_nothing(
    tmp_path: Path,
) -> None:
    """A real dependency change adds packages, not hosts, and comes back empty.

    The false-alarm case, and the reason the baseline is compared against at all rather than
    the changed file being matched against a list of blessed registries. Adding packages is
    what an install does; if that read as a finding the field would fire on every
    dependency-affecting change in the platform's history.
    """
    original = _REAL_NPM_LOCKFILE.read_text(encoding="utf-8")
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package-lock.json": original})
    document = json.loads(original)
    document["packages"].update(
        {
            f"node_modules/{name}": {"version": "1.0.0", "resolved": url}
            for name, url in _registry_resolutions("registry.npmjs.org", 40, prefix="added").items()
        }
    )
    (root / "package-lock.json").write_text(f"{json.dumps(document, indent=2)}\n", encoding="utf-8")

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.scanned is True
    assert scan.baseline_present is True
    assert scan.distinct_hosts == 1
    assert scan.new_hosts == ()
    assert "none" in scan.sentence()


# ------------------------------------------------------- the comparison, and how it is exact


async def test_a_host_the_baseline_merely_contains_is_not_treated_as_a_host_it_used(
    tmp_path: Path,
) -> None:
    """`registry.npmjs.org.attacker.test` is not `registry.npmjs.org`, in either direction.

    The way a substring comparison would be quietly wrong, and the reason each needle is
    anchored on both sides. A lookalike host that ends in the real registry's name would be
    waved through by a search for the registry alone, and a legitimate subdomain of it would
    be reported as new by a search anchored only on the left.
    """
    root, baseline = _checkout_holding(
        tmp_path / "checkout",
        {"package-lock.json": _npm_lockfile(_registry_resolutions("registry.npmjs.org", 3))},
    )
    (root / "package-lock.json").write_text(
        _npm_lockfile(
            {
                **_registry_resolutions("registry.npmjs.org", 3),
                "lookalike": "https://registry.npmjs.org.attacker.test/lookalike/-/l-1.0.0.tgz",
                "prefixed": "https://not-registry.npmjs.org/prefixed/-/p-1.0.0.tgz",
            }
        ),
        encoding="utf-8",
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.new_hosts == ("not-registry.npmjs.org", "registry.npmjs.org.attacker.test")


async def test_a_host_the_baseline_used_on_a_port_is_still_a_host_it_used(tmp_path: Path) -> None:
    """A port is not a different host: `//host:8443/` in the baseline answers for `host`."""
    root, baseline = _checkout_holding(
        tmp_path / "checkout",
        {
            "package-lock.json": _npm_lockfile(
                {"internal": "https://mirror.internal.test:8443/internal/-/internal-1.0.0.tgz"}
            )
        },
    )
    (root / "package-lock.json").write_text(
        _npm_lockfile(
            {
                "internal": "https://mirror.internal.test:8443/internal/-/internal-1.0.0.tgz",
                "second": "https://mirror.internal.test/second/-/second-1.0.0.tgz",
            }
        ),
        encoding="utf-8",
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.distinct_hosts == 1
    assert scan.new_hosts == ()


async def test_a_credential_in_a_resolution_url_never_reaches_the_field(tmp_path: Path) -> None:
    """The host is reported; the userinfo in front of it is dropped before anything is built.

    A registry URL in a lockfile can carry a token, and this field is the one thing about a
    file the reviewer never sees that does reach the model. Reporting the authority verbatim
    would have made the supply-chain scanner into a credential leak.
    """
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package-lock.json": "{}\n"})
    (root / "package-lock.json").write_text(
        _npm_lockfile(
            {"private": "https://deploy:s3cret-token@registry.private.test/private/-/p-1.0.0.tgz"}
        ),
        encoding="utf-8",
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.new_hosts == ("registry.private.test",)
    rendered = json.dumps(scan.as_evidence()) + scan.sentence()
    assert "s3cret-token" not in rendered
    assert "deploy" not in rendered


async def test_a_lockfile_the_baseline_never_held_says_so_rather_than_accusing_it(
    tmp_path: Path,
) -> None:
    """An install that created a lockfile where there was none has every host new, and says why.

    `baseline_present` is what stops that being read as an accusation. Every host really is
    new -- there was nothing to already use one -- and a reviewer told only "3 new hosts" about
    a first-ever lockfile would be looking for an attack in an ordinary bootstrap.
    """
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    (root / "package-lock.json").write_text(
        _npm_lockfile(_registry_resolutions("registry.npmjs.org", 2)), encoding="utf-8"
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.scanned is True
    assert scan.baseline_present is False
    assert scan.new_hosts == ("registry.npmjs.org",)
    assert "did not exist at the revision this change branched from" in scan.sentence()


async def test_without_a_resolvable_branch_point_the_question_is_unanswered_not_answered(
    tmp_path: Path,
) -> None:
    """No lineage base, no comparison -- and emphatically not an empty list of new hosts.

    51-A's rule, and it bites harder here than it does for a line count. `HEAD` on a working
    branch already holds what an earlier approved attempt committed, so measuring against it
    would report a host that attempt introduced as one the repository always used: the
    reassuring answer, produced by the one input that cannot support it.
    """
    root, _baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    (root / "package-lock.json").write_text(
        _npm_lockfile(_registry_resolutions("registry.npmjs.org", 2)), encoding="utf-8"
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=None)

    assert scan.scanned is False
    assert scan.reason == "baseline_unavailable"
    assert scan.new_hosts is None
    # What was measurable is still stated: the file resolves from one host, and the thing that
    # could not be established is only whether that host is new.
    assert scan.distinct_hosts == 1
    assert "not scanned" in scan.sentence()


async def test_a_git_that_cannot_answer_the_comparison_is_not_a_clean_result(
    tmp_path: Path,
) -> None:
    """Git failing to search the baseline blob reports a reason, never an empty host list."""

    class FailingGit:
        """Answer the blob-existence probe, then refuse to search the blob it named."""

        async def run(self, command: Any, *_args: Any, **_kwargs: Any) -> ProcessResult:
            received = tuple(command)
            searching = "grep" in received
            return ProcessResult(
                command=received,
                return_code=129 if searching else 0,
                stdout="" if searching else "package-lock.json\n",
                stderr="fatal: unable to read tree" if searching else "",
                duration_seconds=0.0,
            )

    root, baseline = _checkout_holding(tmp_path / "checkout", {"package-lock.json": "{}\n"})
    (root / "package-lock.json").write_text(
        _npm_lockfile(_registry_resolutions("registry.npmjs.org", 2)), encoding="utf-8"
    )

    scan = await scan_resolved_hosts(
        workspace=root,
        safe_path=root / "package-lock.json",
        relative_path="package-lock.json",
        baseline_revision=baseline,
        process_runner=FailingGit(),
        cancellation_token=MockCancellationToken(),
    )

    assert scan.scanned is False
    assert scan.reason == "baseline_unreadable"
    assert scan.new_hosts is None


# ------------------------------------------------------- the formats, and the honest refusal


@pytest.mark.parametrize(
    ("relative_path", "content", "expected"),
    [
        pytest.param(
            "yarn.lock",
            '# yarn lockfile v1\n\n\nlru-cache@^10.0.0:\n  version "10.4.3"\n'
            '  resolved "https://registry.yarnpkg.com/lru-cache/-/lru-cache-10.4.3.tgz#hash"\n'
            f"  integrity sha512-{'0' * 86}==\n",
            ("registry.yarnpkg.com",),
            id="yarn-classic",
        ),
        pytest.param(
            "yarn.lock",
            '# This file is generated by running "yarn install"\n\n__metadata:\n  version: 8\n\n'
            '"lru-cache@npm:10.4.3":\n  version: 10.4.3\n'
            '  resolution: "lru-cache@npm:10.4.3"\n'
            '  resolved: "https://npm.pkg.github.com/lru-cache/-/lru-cache-10.4.3.tgz"\n',
            ("npm.pkg.github.com",),
            id="yarn-berry",
        ),
        pytest.param(
            "pnpm-lock.yaml",
            "lockfileVersion: '6.0'\n\npackages:\n\n  /lru-cache@10.4.3:\n"
            "    resolution: {tarball: https://mirror.internal.test/lru-cache-10.4.3.tgz}\n",
            ("mirror.internal.test",),
            id="pnpm-tarball",
        ),
        pytest.param(
            "package-lock.json",
            json.dumps(
                {
                    "lockfileVersion": 1,
                    "dependencies": {
                        "lru-cache": {
                            "version": "10.4.3",
                            "resolved": "https://registry.npmjs.org/lru-cache/-/l-10.4.3.tgz",
                        }
                    },
                },
                indent=2,
            ),
            ("registry.npmjs.org",),
            id="npm-lockfile-version-1",
        ),
    ],
)
async def test_each_manager_s_own_way_of_recording_a_resolution_is_read(
    tmp_path: Path, relative_path: str, content: str, expected: tuple[str, ...]
) -> None:
    """Four managers, four syntaxes, one question. The format decides where to look."""
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    (root / relative_path).write_text(content, encoding="utf-8")

    scan = await _scan(root, relative_path, baseline_revision=baseline)

    assert scan.scanned is True
    assert scan.new_hosts == expected


async def test_a_lockfile_that_downloads_nothing_reports_no_hosts_and_means_it(
    tmp_path: Path,
) -> None:
    """A pnpm lockfile whose every package came from the default registry has no hosts.

    Not the same fact as an unreadable one, and the distinction is the whole reason a URL is
    counted separately from a host: this file resolves nothing over the network, so there are
    no hosts to be new, and no baseline is needed to say so.
    """
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    (root / "pnpm-lock.yaml").write_text(
        "lockfileVersion: '9.0'\n\npackages:\n\n  lru-cache@10.4.3:\n"
        f"    resolution: {{integrity: sha512-{'0' * 86}==}}\n",
        encoding="utf-8",
    )

    scan = await _scan(root, "pnpm-lock.yaml", baseline_revision=baseline)

    assert scan.scanned is True
    assert scan.distinct_hosts == 0
    assert scan.new_hosts == ()


async def test_a_lockfile_that_resolves_from_somewhere_unreadable_is_not_reported_as_clean(
    tmp_path: Path,
) -> None:
    """URLs in the file and none of them where this platform looks: not scanned, and why.

    The failure this whole module is most likely to have one day, when a manager changes where
    it records a resolution. The tempting answer is an empty host list -- every pattern
    matched nothing, so nothing is new -- and it is the one answer that would be a false
    assurance about a file nobody read.
    """
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    (root / "package-lock.json").write_text(
        json.dumps(
            {
                "lockfileVersion": 4,
                "packages": {
                    "node_modules/lru-cache": {
                        "version": "10.4.3",
                        "downloadedFrom": "https://registry.substituted.test/lru-cache.tgz",
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.scanned is False
    assert scan.reason == "unrecognized_format"
    assert scan.new_hosts is None


async def test_a_file_whose_bytes_are_not_the_syntax_its_name_claims_is_not_scanned(
    tmp_path: Path,
) -> None:
    """The name says npm wrote it and the bytes say otherwise, so nothing is claimed about it.

    `dependency_sync`'s registry is the right authority for which manager a filename belongs
    to, and it is a name-and-directory question by design -- it reads nothing. That leaves one
    gap this closes: a file called `package-lock.json` holding something that is not JSON would
    be searched with npm's pattern, match nothing, and be reported as resolving from nowhere.
    """
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    (root / "package-lock.json").write_text(
        '# yarn lockfile v1\nlru-cache@^10:\n  resolved "https://registry.substituted.test/l.tgz"\n',
        encoding="utf-8",
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.lockfile_format == "npm"
    assert scan.scanned is False
    assert scan.reason == "unrecognized_format"


async def test_a_path_that_is_not_a_lockfile_at_all_is_reported_as_unrecognized(
    tmp_path: Path,
) -> None:
    """Nothing about a source file is claimed either, and no format is invented for it."""
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    (root / "composer.lock").write_text('{"packages": []}\n', encoding="utf-8")

    scan = await _scan(root, "composer.lock", baseline_revision=baseline)

    assert scan.lockfile_format is None
    assert scan.scanned is False
    assert scan.reason == "unrecognized_format"


async def test_a_lockfile_that_cannot_be_read_reports_the_read_failure(tmp_path: Path) -> None:
    """A missing file is a stated read failure, not an absence of hosts."""
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.scanned is False
    assert scan.reason == "unreadable"
    assert scan.new_hosts is None


# ----------------------------------------------------------------- bounded, and still correct


async def test_a_resolution_straddling_a_read_window_is_still_found(tmp_path: Path) -> None:
    """The file is never loaded, and the host on the seam between two windows is still reported.

    The observable consequence of streaming: a lockfile past the read cap is consumed a window
    at a time, so the one place a naive chunked scan loses a match is a token that begins in
    one window and ends in the next. The lockfile here is built so that the substituted
    resolution lands exactly across that seam.
    """
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    substituted = "https://registry.substituted.test/spans-the-seam/-/spans-the-seam-1.0.0.tgz"
    document = _npm_lockfile(
        {**_registry_resolutions("registry.npmjs.org", 1), "spans-the-seam": substituted}
    )
    token = f'"resolved": "{substituted}"'
    # Insert whitespace ahead of the substituted resolution -- legal anywhere in JSON -- until
    # that resolution's own midpoint sits exactly on the window boundary, so half of it is read
    # in one window and half in the next.
    shift = _SCAN_WINDOW_BYTES - document.index(token) - len(token) // 2
    assert shift > 0, "the padding has to come before the resolution being straddled"
    opening = document.index("{") + 1
    seamed = f"{document[:opening]}{' ' * shift}{document[opening:]}"
    assert len(seamed) > _SCAN_WINDOW_BYTES, "the lockfile must be read in more than one window"
    (root / "package-lock.json").write_text(seamed, encoding="utf-8")

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.scanned is True
    assert scan.new_hosts == ("registry.npmjs.org", "registry.substituted.test")


async def test_more_distinct_hosts_than_the_ceiling_holds_are_counted_not_hidden(
    tmp_path: Path,
) -> None:
    """The list is cut and the entry says it was cut, because a silent cut reads as the answer."""
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package.json": "{}\n"})
    over = _MAX_DISTINCT_HOSTS + 6
    (root / "package-lock.json").write_text(
        _npm_lockfile(
            {
                f"pkg-{index:03d}": f"https://mirror-{index:03d}.substituted.test/p/-/p-1.0.0.tgz"
                for index in range(over)
            }
        ),
        encoding="utf-8",
    )

    scan = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert scan.hosts_truncated is True
    assert scan.distinct_hosts == _MAX_DISTINCT_HOSTS
    assert scan.new_hosts is not None
    assert len(scan.new_hosts) == _MAX_DISTINCT_HOSTS
    assert "further hosts were not listed" in scan.sentence()


async def test_the_same_bytes_scan_to_the_same_field_every_time(tmp_path: Path) -> None:
    """Byte-identical output over one checkout: the field is part of the review's idempotency key.

    `_review_input_signature` keys the journaled `RUN_REVIEWER` operation, so evidence that
    varied between builds would make crash recovery re-call the review model instead of
    reusing the answer it already paid for. Sorted host names and nothing measured are what
    make this hold.
    """
    original = _REAL_NPM_LOCKFILE.read_text(encoding="utf-8")
    root, baseline = _checkout_holding(tmp_path / "checkout", {"package-lock.json": original})
    (root / "package-lock.json").write_text(
        original.replace(
            "https://registry.npmjs.org/lru-cache/",
            "https://registry.substituted.test/lru-cache/",
            1,
        ),
        encoding="utf-8",
    )

    first = await _scan(root, "package-lock.json", baseline_revision=baseline)
    second = await _scan(root, "package-lock.json", baseline_revision=baseline)

    assert first == second
    assert first.as_evidence() == second.as_evidence()
    assert list(first.as_evidence()) == [
        "format",
        "scanned",
        "reason",
        "baseline_present",
        "distinct_hosts",
        "new_hosts",
        "hosts_truncated",
    ]
