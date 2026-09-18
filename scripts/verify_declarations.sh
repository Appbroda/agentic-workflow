#!/usr/bin/env bash
# Verify the declaration-class MODEL_* settings crossed the compose boundary (item 37, 80-).
#
# docker-compose.yml's environment is an explicit allowlist, so a setting can exist on the
# host, parse cleanly at startup (empty means "nothing declared"), and never reach the
# container -- which is how AB-Feature-211 died on the exact 400 its own fix was for. This
# takes the same verify-what-actually-ran posture deploy.sh takes for BUILD_REVISION: for
# each declaration whose effective host value is non-empty, ask the running container what
# it actually received and fail the deploy naming any variable that did not cross.
#
# Usage: verify_declarations.sh <service> [extra docker compose flags...]
# Run from the repository root (deploy.sh already is). The effective host value follows
# compose interpolation precedence: a shell export overrides the .env file. Values are
# compared byte-exactly; on success only the verified variable NAMES are printed -- the
# list may one day sit beside secrets.
set -euo pipefail

# Every declaration-class setting, and the list is not maintained by hand alone:
# test_deploy_script.py derives the same set from Settings and fails if this one is missing a
# member. MODEL_VISION_CAPABLE was added by 89- and spent a deploy absent from here, which is
# the boundary failure this script exists to catch, arriving through the script itself.
DECLARATION_VARIABLES=(
  MODEL_REASONING_UNSUPPORTED
  MODEL_TEMPERATURE_UNSUPPORTED
  MODEL_MAX_OUTPUT_TOKENS
  MODEL_ROUTER_MIN_CONFIDENCE
  MODEL_CONTEXT_WINDOW_TOKENS
  MODEL_VISION_CAPABLE
)

[ "$#" -ge 1 ] || { echo "usage: verify_declarations.sh <service> [compose flags...]" >&2; exit 2; }
SERVICE="$1"
shift

env_file_value() {
  # Print the .env value of $1, tolerating '=' inside the value (JSON declarations carry
  # them); the last assignment wins, matching how compose reads the file.
  local name="$1" line value=""
  [ -f .env ] || { printf ''; return; }
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      "$name="*) value="${line#*=}" ;;
    esac
  done < .env
  printf '%s' "$value"
}

verified=()
for name in "${DECLARATION_VARIABLES[@]}"; do
  if [ -n "${!name+x}" ]; then
    # Compose interpolation reads the shell first; comparing against .env when a shell
    # export overrides it would fail a deploy whose container is actually correct.
    expected="${!name}"
  else
    expected="$(env_file_value "$name")"
  fi
  [ -n "$expected" ] || continue
  deployed="$(docker compose "$@" exec -T "$SERVICE" printenv "$name" 2>/dev/null | tr -d '\r' || true)"
  if [ "$deployed" != "$expected" ]; then
    echo "deploy: error: ${name} did not cross the compose boundary into ${SERVICE}: the \
declaration exists on the host but the running container carries something else (add it to \
the ${SERVICE} environment allowlist in docker-compose.yml and redeploy)" >&2
    exit 1
  fi
  verified+=("$name")
done

if [ "${#verified[@]}" -gt 0 ]; then
  echo "deploy: declarations verified in ${SERVICE}: ${verified[*]}"
else
  echo "deploy: no non-empty declaration-class settings to verify for ${SERVICE}"
fi
