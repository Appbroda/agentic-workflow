"""scripts/deploy.sh: the refusal and identity logic, exercised against a recording stub.

The script exists because a correct deploy needed BUILD_REVISION and
WORKFLOW_SCHEMA_VERSION hand-exported, and forgetting either crash-loops the api on
runtime_identity (2026-09-04). What is testable here is everything short of a Docker
daemon: the dirty-tree refusal fires before any docker call, the identity values are
computed from their single sources and exported into the compose invocations, an
unparseable schema file refuses, and a container reporting the wrong revision fails the
deploy loudly. The real compose path (build, health wait, api-dev recreation) is the
documented manual verification in 74-.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy.sh"
VERIFY_DECLARATIONS_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "verify_declarations.sh"
)


def _declaration_variables() -> tuple[str, ...]:
    """The names in the script's own DECLARATION_VARIABLES array.

    Read out of the script rather than restated, so the hermetic environment below cannot
    fall behind the list it is supposed to neutralize.
    """
    source = VERIFY_DECLARATIONS_SCRIPT.read_text(encoding="utf-8")
    body = source.split("DECLARATION_VARIABLES=(", 1)[1].split(")", 1)[0]
    return tuple(name for name in body.split() if name.startswith("MODEL_"))


# A `docker` that records every invocation and the exported identity, then answers the
# script's questions the way a healthy freshly-deployed stack would: the api container
# exists, reports healthy, and echoes back the BUILD_REVISION the build was given --
# unless DOCKER_STUB_DEPLOYED_REVISION forces the stale-image answer.
_DOCKER_STUB = """#!/usr/bin/env bash
{
  printf 'argv: %s\\n' "$*"
  printf 'env BUILD_REVISION=%s\\n' "${BUILD_REVISION:-}"
  printf 'env WORKFLOW_SCHEMA_VERSION=%s\\n' "${WORKFLOW_SCHEMA_VERSION:-}"
} >> "${DOCKER_STUB_LOG}"
case "$*" in
  "compose ps -q api")
    echo stub-api-container
    ;;
  "inspect -f {{.State.Health.Status}} stub-api-container")
    echo healthy
    ;;
  "compose exec -T api printenv BUILD_REVISION")
    echo "${DOCKER_STUB_DEPLOYED_REVISION:-${BUILD_REVISION:-}}"
    ;;
  "compose exec -T api printenv MODEL_CONTEXT_WINDOW_TOKENS")
    # What the container actually received, controlled by the test; empty (with printenv's
    # non-zero exit) is the allowlist-miss shape item 37 is about.
    if [ -n "${DOCKER_STUB_MODEL_CONTEXT_WINDOW_TOKENS:-}" ]; then
      printf '%s\\n' "${DOCKER_STUB_MODEL_CONTEXT_WINDOW_TOKENS}"
    else
      exit 1
    fi
    ;;
esac
exit 0
"""


def _git(repo: Path, *args: str) -> str:
    """Run one git command in the temp checkout and return its stdout."""
    completed = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _checkout(
    tmp_path: Path,
    *,
    schema: str = 'WORKFLOW_SCHEMA_VERSION = "9.9-test"\n',
    env_file: str | None = None,
) -> Path:
    """A committed temp repository shaped like a platform checkout, with the real scripts.

    ``env_file`` writes an untracked-but-ignored ``.env``, the way a real deploy checkout
    carries one: ignored so the dirty-tree refusal does not fire on it.
    """
    repo = tmp_path / "checkout"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(DEPLOY_SCRIPT, repo / "scripts" / "deploy.sh")
    (repo / "scripts" / "deploy.sh").chmod(0o755)
    shutil.copy(VERIFY_DECLARATIONS_SCRIPT, repo / "scripts" / "verify_declarations.sh")
    (repo / "scripts" / "verify_declarations.sh").chmod(0o755)
    (repo / "server").mkdir()
    (repo / "server" / "workflow_schema.py").write_text(schema, encoding="utf-8")
    (repo / ".gitignore").write_text(".env\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "deploy-test@example.invalid")
    _git(repo, "config", "user.name", "Deploy Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    if env_file is not None:
        (repo / ".env").write_text(env_file, encoding="utf-8")
    return repo


def _run_deploy(
    repo: Path,
    tmp_path: Path,
    *,
    deployed_revision: str | None = None,
    stub_container_declaration: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the script with the stub docker first on PATH; return the result and stub log."""
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(_DOCKER_STUB, encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "docker-stub.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log),
    }
    env.pop("BUILD_REVISION", None)
    env.pop("WORKFLOW_SCHEMA_VERSION", None)
    # Hermetic against the developer's own shell: the declaration check reads the shell
    # first, so an exported declaration would leak into every checkout-only scenario.
    for name in _declaration_variables():
        env.pop(name, None)
    if stub_container_declaration is not None:
        env["DOCKER_STUB_MODEL_CONTEXT_WINDOW_TOKENS"] = stub_container_declaration
    if deployed_revision is not None:
        env["DOCKER_STUB_DEPLOYED_REVISION"] = deployed_revision
    completed = subprocess.run(
        ["bash", str(repo / "scripts" / "deploy.sh")],
        # Deliberately not the checkout: the script must resolve its repository from its
        # own location, so a deploy works from any working directory.
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return completed, log.read_text(encoding="utf-8") if log.exists() else ""


def test_a_dirty_tree_is_refused_before_any_docker_call(tmp_path: Path) -> None:
    """The Dockerfile refuses the same tree minutes later; the script says why up front."""
    repo = _checkout(tmp_path)
    (repo / "uncommitted.txt").write_text("not yet a commit\n", encoding="utf-8")

    completed, stub_log = _run_deploy(repo, tmp_path)

    assert completed.returncode != 0
    assert "dirty" in completed.stderr
    assert stub_log == ""


def test_the_identity_is_computed_and_exported_from_its_single_sources(tmp_path: Path) -> None:
    """BUILD_REVISION is HEAD and the schema version is parsed, both visible to compose."""
    repo = _checkout(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    completed, stub_log = _run_deploy(repo, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert f"env BUILD_REVISION={head}" in stub_log
    assert "env WORKFLOW_SCHEMA_VERSION=9.9-test" in stub_log
    assert "argv: compose build" in stub_log
    assert "argv: compose up -d --force-recreate api" in stub_log
    assert head in completed.stdout
    assert "9.9-test" in completed.stdout


def test_an_unparseable_schema_version_refuses_before_compose_runs(tmp_path: Path) -> None:
    """No assignment line means no value to deploy with -- never a hardcoded fallback."""
    repo = _checkout(tmp_path, schema="# the assignment is gone\n")

    completed, stub_log = _run_deploy(repo, tmp_path)

    assert completed.returncode != 0
    assert "WORKFLOW_SCHEMA_VERSION" in completed.stderr
    assert "argv: compose build" not in stub_log


def test_a_duplicated_schema_assignment_refuses_rather_than_guessing(tmp_path: Path) -> None:
    """Two assignments is two candidate identities; the script must not pick one."""
    repo = _checkout(
        tmp_path,
        schema='WORKFLOW_SCHEMA_VERSION = "1.0"\nWORKFLOW_SCHEMA_VERSION = "2.0"\n',
    )

    completed, stub_log = _run_deploy(repo, tmp_path)

    assert completed.returncode != 0
    assert "more than once" in completed.stderr
    assert "argv: compose build" not in stub_log


def test_a_container_reporting_the_wrong_revision_fails_the_deploy_loudly(tmp_path: Path) -> None:
    """The verify step compares what actually runs to HEAD, and a stale image fails."""
    repo = _checkout(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    completed, _ = _run_deploy(repo, tmp_path, deployed_revision="deadbeefdeadbeef")

    assert completed.returncode != 0
    assert "deadbeefdeadbeef" in completed.stderr
    assert head in completed.stderr


# A declaration whose value carries spaces, braces and an inner `=`-free JSON shape: the
# byte-exact comparison and the `.env` parsing both have to survive it.
_DECLARED_WINDOWS = '{"gpt-6-astra": 400000, "claude-fable-5": 200000}'


def test_a_declaration_that_did_not_cross_the_boundary_fails_the_deploy_by_name(
    tmp_path: Path,
) -> None:
    """T11 (80-, item 37): the allowlist miss becomes a deploy failure, not a live fatality.

    The `.env` declares a context-window table, the stub container does not carry it --
    exactly AB-Feature-211's shape, where the setting existed on the host, parsed cleanly at
    startup, and the run died on the very 400 the fix was for. The deploy must fail naming
    the variable that did not cross.
    """
    repo = _checkout(tmp_path, env_file=f"MODEL_CONTEXT_WINDOW_TOKENS={_DECLARED_WINDOWS}\n")

    completed, _ = _run_deploy(repo, tmp_path)

    assert completed.returncode != 0
    assert "MODEL_CONTEXT_WINDOW_TOKENS" in completed.stderr
    assert "did not cross the compose boundary" in completed.stderr


def test_a_declaration_that_crossed_is_verified_and_named_without_its_value(
    tmp_path: Path,
) -> None:
    """T11's other half: all declarations crossing succeeds and prints the verified names.

    Names only -- the success line may one day sit beside secrets, so the value must not
    appear in the deploy output.
    """
    repo = _checkout(tmp_path, env_file=f"MODEL_CONTEXT_WINDOW_TOKENS={_DECLARED_WINDOWS}\n")

    completed, _ = _run_deploy(repo, tmp_path, stub_container_declaration=_DECLARED_WINDOWS)

    assert completed.returncode == 0, completed.stderr
    assert "declarations verified in api: MODEL_CONTEXT_WINDOW_TOKENS" in completed.stdout
    assert _DECLARED_WINDOWS not in completed.stdout


def test_a_checkout_with_no_declarations_still_deploys(tmp_path: Path) -> None:
    """An empty declaration set is not an error: there is nothing to verify."""
    repo = _checkout(tmp_path)

    completed, _ = _run_deploy(repo, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "no non-empty declaration-class settings" in completed.stdout


def test_the_script_verifies_every_declaration_class_setting() -> None:
    """A new declaration cannot be added to Settings and forgotten here.

    The script's whole purpose is that a declaration can exist on the host, parse cleanly,
    and never reach the container -- so a declaration it does not know about is exactly the
    silence it exists to break. `MODEL_VISION_CAPABLE` was added by 89- and shipped a deploy
    absent from the array, verified by nothing; the list is derived on both sides here so the
    next one fails this test instead of a live run.
    """
    from configs.settings import Settings

    declared = {name.upper() for name in Settings.model_fields if name.startswith("model_")}

    assert set(_declaration_variables()) == declared
