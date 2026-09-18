# syntax=docker/dockerfile:1

ARG BUILD_REVISION=unknown

# Keep dependency resolution in a dedicated uv-backed build stage so the runtime
# image contains only the application virtual environment and required binaries.
FROM ghcr.io/astral-sh/uv:0.10.4 AS uv

# The web control plane is built here and copied into the runtime image, so one origin serves
# both it and the API. That is what lets the browser call the API without CORS -- see
# `_serve_web_client` in server/main.py for why the application lives under /ui.
FROM node:22-slim AS webclient
WORKDIR /client
# The lockfile alone first, so a source change does not reinstall the dependency tree.
COPY client/package.json client/package-lock.json /client/
RUN npm ci
COPY client/ /client/
RUN npm run build


FROM python:3.12-slim-bookworm AS builder

ARG BUILD_REVISION

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends --yes git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# The Python project lives under server/; the build context stays the repository root
# because the identity check below reads .git from it.
COPY server/pyproject.toml server/uv.lock /app/server/
WORKDIR /app/server
RUN uv sync --frozen --no-dev --no-install-project

WORKDIR /app
COPY . .
RUN test "${BUILD_REVISION}" != "unknown" \
    && test "$(git rev-parse --verify HEAD)" = "${BUILD_REVISION}" \
    && test -z "$(git status --porcelain --untracked-files=all)" \
    && rm -rf .git .claude .codex-prompts logs server/tests
WORKDIR /app/server
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim-bookworm AS runtime

# A repository's package manager is chosen by its checked-in lockfile, not by the platform,
# so every manager the platform is willing to select has to exist here. Installing them
# explicitly rather than through Corepack keeps the image self-contained at build time.
# Node comes from NodeSource rather than the distribution, because a repository declares
# the runtime it needs and the platform has to satisfy it. Debian's Node 18 lacks globals
# that a repository's own dependencies assume, so its test suite failed here for a reason
# that had nothing to do with the change being made.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates curl gnupg git \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install --no-install-recommends --yes nodejs \
    && npm install --global --no-audit --no-fund pnpm@9.15.4 yarn@1.22.22 \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system platform \
    && useradd --system --gid platform --create-home platform

# npm's cache directory, and it is the mount point for the `npm_cache` volume declared in
# docker-compose.yml. The path is what `npm config get cache` answers inside this image --
# asked there rather than assumed from `~/.npm`, and asked through the environment the install
# subprocess actually gets, which withholds HOME so npm resolves the home directory from the
# passwd entry for uid 999. Both answers were `/home/platform/.npm`.
#
# Created here, while still root, and chowned, because Docker seeds a fresh named volume from
# whatever the image holds at the mount point, ownership included. Without this line the volume
# arrives as an empty root:root directory that uid 999 cannot write, and `npm ci` then fails
# outright -- strictly worse than the slow install the volume exists to prevent.
RUN mkdir -p /home/platform/.npm && chown platform:platform /home/platform/.npm

# Corepack's cache, mount point for the `corepack_cache` volume, and it exists for the same
# two reasons as the directory above. A repository that declares `packageManager` has its
# exact version provisioned here rather than being asked to match the pnpm and yarn pinned
# globally above -- which is the whole point, since one image cannot carry every version
# every repository pins, and AB-Feature-222 was refused for declaring npm@9.8.1.
#
# The path is what Corepack reports inside this image, asked through the same environment the
# install subprocess gets, which withholds HOME so the home directory comes from the passwd
# entry: /home/platform/.cache/node/corepack. Measured at 37 MB for one npm version, and a
# fresh named volume inherits the mount point's ownership, so both the chown and the volume
# are load-bearing exactly as they are for `.npm`.
RUN mkdir -p /home/platform/.cache/node/corepack \
    && chown -R platform:platform /home/platform/.cache

# Supplied at build time so a running container can be matched to the commit it was
# built from: `docker build --build-arg BUILD_REVISION=$(git rev-parse HEAD)`.
ARG BUILD_REVISION
LABEL org.opencontainers.image.revision=${BUILD_REVISION}

WORKDIR /app/server
ENV BUILD_REVISION=${BUILD_REVISION} \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/server \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder --chown=platform:platform /opt/venv /opt/venv
COPY --from=uv /uv /uvx /bin/
COPY --from=builder --chown=platform:platform /app /app
COPY --from=webclient --chown=platform:platform /client/dist /app/client/dist

USER platform
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-access-log"]
