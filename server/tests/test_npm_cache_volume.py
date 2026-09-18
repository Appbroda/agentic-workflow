"""68-: the npm cache survives a deploy, and it is mounted where npm actually looks.

The budget in the commit before this one makes a cold install survivable; this makes it rare.
`~/.npm` was 256 MB inside the api container's own filesystem, and the service mounted exactly
one volume, so every `docker compose up --force-recreate` threw the cache away. Attempt 0 of
AB-Feature-203 and 204 were the first installs after a recreate, and they are the 224-378s rows
in the evidence table. Neither change substitutes for the other: 204's attempt 3 had a warm
cache and still exceeded 120s while the other lane ran a 170s build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# What `npm config get cache` answers inside the api container. Read there rather than assumed
# from `~/.npm`, and read twice: once directly, and once through `AsyncioProcessRunner` with
# `repository_subprocess_environment()`, which is the environment the install subprocess
# actually gets and which withholds HOME -- so npm resolves the home directory from the passwd
# entry for uid 999 rather than from an inherited variable. Both answered this path.
#
# T7 exists because a volume at a path npm does not use is worse than no volume: it looks like
# the fix and changes nothing.
_NPM_CACHE_PATH = "/home/platform/.npm"

# Every service that runs the platform image and therefore installs a repository's
# dependencies. `api-dev` shares the cache rather than holding its own: same installs, same
# repositories, and npm's cache is content-addressed and safe for concurrent use.
_INSTALLING_SERVICES = ("api", "api-dev")


def _compose() -> dict[str, Any]:
    """The deployment as Compose parses it."""
    text = (_REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document


def test_the_npm_cache_volume_is_mounted_where_npm_actually_looks() -> None:
    """Each service that installs dependencies mounts a named volume at npm's cache path."""
    compose = _compose()

    for service in _INSTALLING_SERVICES:
        mounts = compose["services"][service]["volumes"]
        cache_mounts = [mount for mount in mounts if mount.split(":")[1] == _NPM_CACHE_PATH]
        assert len(cache_mounts) == 1, f"{service} mounts {mounts}"
        source = cache_mounts[0].split(":")[0]
        # A named volume, not a host bind: the cache has to survive a recreate without
        # depending on a path that happens to exist on one operator's machine.
        assert not source.startswith(("/", ".", "~")), source
        assert source in compose["volumes"], f"{source} is not declared"
        # The volume this change adds, never a replacement for the one already there.
        assert "workspace_data:/workspaces" in mounts


def test_both_installing_services_share_one_cache() -> None:
    """One cache, so the second service does not reintroduce the cold mode for itself."""
    compose = _compose()

    sources = {
        mount.split(":")[0]
        for service in _INSTALLING_SERVICES
        for mount in compose["services"][service]["volumes"]
        if mount.split(":")[1] == _NPM_CACHE_PATH
    }

    assert len(sources) == 1, sources


def test_the_image_owns_the_cache_directory_so_a_fresh_volume_is_writable() -> None:
    """Docker seeds a fresh named volume from the image, ownership included.

    Proven against the built image rather than reasoned about. Mounting a bare named volume at
    `/home/platform/.npm`, a path the image did not contain, produced a `root:root` directory
    that uid 999 could not write -- so `npm ci` would have failed outright, which is strictly
    worse than the slow install this volume exists to prevent. Creating the directory owned by
    `platform` first made the same fresh volume writable. The spec asked only for the Compose
    entry; on its own it would have broken every install.
    """
    dockerfile = (_REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    statements = [
        line.strip() for line in dockerfile.splitlines() if not line.strip().startswith("#")
    ]

    creation = next(
        (index for index, line in enumerate(statements) if _NPM_CACHE_PATH in line), None
    )
    switch_to_platform = next(
        (index for index, line in enumerate(statements) if line == "USER platform"), None
    )
    assert creation is not None, "the image never creates npm's cache directory"
    assert switch_to_platform is not None
    # While still root, or the chown cannot be made.
    assert creation < switch_to_platform
    assert "mkdir -p /home/platform/.npm" in statements[creation]
    assert "chown platform:platform /home/platform/.npm" in statements[creation]
