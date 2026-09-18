#!/usr/bin/env bash
# One-command deploy for this platform's compose stack (74- Part B).
#
# A correct deploy needs BUILD_REVISION and WORKFLOW_SCHEMA_VERSION exported before
# `docker compose build && up`; forgetting either leaves the x-runtime-identity
# expectations at "unknown" and the api crash-loops on runtime_identity (it happened on
# 2026-09-04). This script computes both from their single sources, refuses a dirty tree
# before the image build refuses it less legibly, runs the known-good compose sequence,
# and then verifies the deploy by asking the running container what revision it actually
# carries -- the image bakes BUILD_REVISION at build time, so equality with HEAD proves
# the container came from this build and not a stale image.
set -euo pipefail

fail() {
  echo "deploy: error: $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v git >/dev/null 2>&1 || fail "git is required"
command -v docker >/dev/null 2>&1 || fail "docker is required"

# The image build asserts the same thing (Dockerfile: revision == checkout HEAD and a
# clean tree) but fails there as a bare non-zero exit minutes in. Same rule, better
# message, before any build starts.
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  fail "the working tree is dirty; commit or stash everything first -- the image build \
refuses a dirty tree, and a deploy must be attributable to exactly one commit"
fi

BUILD_REVISION="$(git rev-parse HEAD)"

# The schema version's single source is the assignment in server/workflow_schema.py.
# Parsed, never hardcoded: a hardcoded copy here would be wrong the day the schema moves.
SCHEMA_FILE="server/workflow_schema.py"
[ -f "$SCHEMA_FILE" ] || fail "$SCHEMA_FILE not found; run this from a platform checkout"
WORKFLOW_SCHEMA_VERSION="$(sed -n 's/^WORKFLOW_SCHEMA_VERSION = "\([^"]*\)"$/\1/p' "$SCHEMA_FILE")"
case "$WORKFLOW_SCHEMA_VERSION" in
  "")
    fail "could not read WORKFLOW_SCHEMA_VERSION from $SCHEMA_FILE"
    ;;
  *$'\n'*)
    fail "$SCHEMA_FILE declares WORKFLOW_SCHEMA_VERSION more than once"
    ;;
esac

# Shell exports override the .env file in compose interpolation, which is what makes the
# hand-export step obsolete rather than duplicated.
export BUILD_REVISION WORKFLOW_SCHEMA_VERSION

# Every build leaves roughly 1.5GB of layer cache behind, and this script is run once per
# change. Thirteen deploys in one session took the daemon's filesystem from comfortable to
# 100% full, and the failure surfaced as the platform's own workspace-capacity preflight:
# "707715072 bytes free, but 2147483648 bytes are required" against `/workspaces` -- which
# was 16GB of a 97GB-used, 103GB filesystem shared with `/`. Anyone reading that message
# goes and deletes feature checkouts, which is the wrong volume and does not help.
#
# Pruned before the build rather than after, so a deploy that needs the space has it, and
# `--keep-storage` so an incremental rebuild is still incremental: this trims the tail, it
# does not throw the cache away.
echo "deploy: trimming build cache over ${BUILD_CACHE_KEEP_BYTES:-20GB}"
docker builder prune --force --keep-storage "${BUILD_CACHE_KEEP_BYTES:-20GB}" >/dev/null || {
  echo "deploy: warning: build-cache trim failed; continuing" >&2
}

echo "deploy: building revision ${BUILD_REVISION} (workflow schema ${WORKFLOW_SCHEMA_VERSION})"
docker compose build
docker compose up -d --force-recreate api

# The api healthcheck polls /readyz (start_period 15s, 5 retries at 10s); 180s is
# comfortable headroom over the point where compose itself would call the container
# unhealthy.
echo "deploy: waiting for the api to report healthy"
DEADLINE=$((SECONDS + 180))
STATUS="unknown"
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  CONTAINER="$(docker compose ps -q api)"
  if [ -n "$CONTAINER" ]; then
    STATUS="$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo unknown)"
    [ "$STATUS" = "healthy" ] && break
  fi
  sleep 5
done
if [ "$STATUS" != "healthy" ]; then
  echo "deploy: last api log lines:" >&2
  docker compose logs --tail 20 api >&2 || true
  fail "the api did not report healthy within 180s (last status: ${STATUS})"
fi

verify_revision() {
  # $1 = service, remaining args = extra compose flags (e.g. --profile dev).
  local service="$1"
  shift
  local deployed
  deployed="$(docker compose "$@" exec -T "$service" printenv BUILD_REVISION | tr -d '\r')"
  if [ "$deployed" != "$BUILD_REVISION" ]; then
    fail "the running ${service} container reports BUILD_REVISION=${deployed} but HEAD is \
${BUILD_REVISION}; the deploy did not take (stale image, or the build was skipped)"
  fi
}

verify_revision api

# The declaration-class MODEL_* settings, in the same posture (item 37): compose's
# environment is an explicit allowlist, so a declaration can exist on the host, parse
# cleanly at startup, and never reach the container -- 211's death. Any non-empty
# declaration whose in-container value differs fails the deploy naming the variable.
"$REPO_ROOT/scripts/verify_declarations.sh" api

# api-dev shares the x-runtime-identity expectations, so a running dev container holds the
# stale identity after a deploy and would fail the same way. It is behind the `dev` compose
# profile, so it is recreated only when it is actually running -- a deploy must not start a
# dev profile nobody asked for.
if [ -n "$(docker compose --profile dev ps -q --status running api-dev 2>/dev/null || true)" ]; then
  echo "deploy: api-dev is running; recreating it with the same identity"
  docker compose --profile dev up -d --force-recreate api-dev
  verify_revision api-dev --profile dev
  "$REPO_ROOT/scripts/verify_declarations.sh" api-dev --profile dev
fi

echo "deploy: done"
echo "deploy:   revision:                ${BUILD_REVISION}"
echo "deploy:   workflow schema version: ${WORKFLOW_SCHEMA_VERSION}"
