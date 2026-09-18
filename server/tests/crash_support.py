"""Doubles and process control for the crash-window suite.

Audit risk P1-8, second half. Nothing in this suite ever killed a process. Every recovery
path was verified by calling a recovery function directly with a hand-built
`ExternalOperation`, which is the coverage that already existed and did not catch anything.

What a crash test needs is a real process boundary, and these are the parts that make one:

* `CrashingJournal` ends its own process, with `SIGKILL`, at one precise journal write. The
  side effect has happened; the journal has not recorded it. That is the window every
  reconcile callback in the platform exists for, and it is the one no test could reach.
* `PersistentGitHubDouble` holds pull requests, labels, reviewers and comments in a file, so
  the provider's state outlives the process that talked to it -- which is the whole point.
  It also counts creations, because "a second pull request was not opened" is a statement
  about the provider, not about the journal.
* `CountingModelDouble` returns a real, deterministic coding response and records every
  invocation in a file. The invocation count is the load-bearing assertion for boundary 2.
* `install_index_lock_killer` uses a real Git clean filter to have `git add` die while it holds
  `.git/index.lock`. That is exactly what this platform manufactures when it terminates a
  process group, and the file left behind is Git's, not this module's.

Doubles hold state; they do not count calls, except where a call count *is* the property
under test. See overview section 4.4.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

from adapters.github_adapter import MockGitHubService, PullRequestDetails
from adapters.llm_adapter import ImageInput, LLMResponse
from services.cancellation import CancellationToken
from services.process_runner import AsyncioProcessRunner, ProcessResult
from state.external_operations import ExternalOperationType
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal

SERVER_ROOT = Path(__file__).resolve().parent.parent

# Where in one operation's life the process dies.
BEFORE_JOURNAL_SUCCESS = "before_journal_success"
AFTER_JOURNAL_SUCCESS = "after_journal_success"


def die_now() -> NoReturn:
    """End this process the way a killed worker ends: no unwinding, no cleanup, no flush.

    `SIGKILL` rather than an exception on purpose. An exception is caught by the handler in
    `ExternalOperationExecutor.run`, which records a failure -- so the journal would be left
    in a state a crash never produces, and the test would be asserting against a shape that
    does not occur.
    """
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("unreachable: SIGKILL does not return")  # pragma: no cover


class CrashingJournal(ExternalOperationJournal):
    """The real journal, which ends its process at one named write.

    Subclassed rather than wrapped so every other method is the production one: the
    operation this kills is journaled, claimed, heartbeated and read exactly as it would be
    in the deployment, and only the single write the boundary names behaves differently.
    """

    def __init__(
        self,
        database: Database,
        *,
        crash_on: ExternalOperationType,
        crash_at: str = BEFORE_JOURNAL_SUCCESS,
        **options: Any,
    ) -> None:
        """Bind the operation type and the moment this process should stop existing."""
        super().__init__(database, **options)
        self._crash_on = crash_on
        self._crash_at = crash_at

    async def record_result(
        self,
        operation_id: str,
        *,
        external_reference: str | None,
        result_payload: dict[str, Any] | None,
    ) -> Any:
        """Die immediately before, or immediately after, committing this operation's success."""
        operation = await self.get(operation_id)
        if operation.operation_type is self._crash_on and self._crash_at == BEFORE_JOURNAL_SUCCESS:
            die_now()
        completed = await super().record_result(
            operation_id, external_reference=external_reference, result_payload=result_payload
        )
        if operation.operation_type is self._crash_on and self._crash_at == AFTER_JOURNAL_SUCCESS:
            die_now()
        return completed


class PersistentGitHubDouble(MockGitHubService):
    """A GitHub double whose repositories, branches, pull requests and labels outlive a crash.

    `MockGitHubService` already holds real state rather than counting calls; what it cannot
    do is survive the process. Persisting to a file is what lets a fresh process ask the
    provider what the dead one did -- which is precisely what a reconcile callback does.

    `creations` is the one deliberate call count here, and it is load-bearing: "no second
    pull request was opened" is a claim about the provider having been asked, not about the
    number of rows the journal holds.
    """

    def __init__(self, state_path: Path, *, crash_after_create: bool = False) -> None:
        """Restore whatever a previous process left behind, and optionally die after creating."""
        super().__init__()
        self._state_path = state_path
        self._crash_after_create = crash_after_create
        self.creations = 0
        self._load()

    def _load(self) -> None:
        """Rehydrate the provider's state from the file the last process wrote."""
        if not self._state_path.is_file():
            return
        stored = json.loads(self._state_path.read_text(encoding="utf-8"))
        self.creations = int(stored.get("creations", 0))
        self._next_pull_request_number = int(stored.get("next_number", 1))
        for entry in stored.get("pull_requests", []):
            details = PullRequestDetails(**entry["details"])
            key = (details.repository, details.number)
            self.pull_requests[key] = details
            self.labels[key] = list(entry.get("labels", []))
            self.reviewers[key] = list(entry.get("reviewers", []))
            self.comments[key] = list(entry.get("comments", []))

    def _save(self) -> None:
        """Write the provider's state durably, because the next reader is a different process."""
        payload = {
            "creations": self.creations,
            "next_number": self._next_pull_request_number,
            "pull_requests": [
                {
                    "details": asdict(details),
                    "labels": self.labels.get(key, []),
                    "reviewers": self.reviewers.get(key, []),
                    "comments": self.comments.get(key, []),
                }
                for key, details in self.pull_requests.items()
            ],
        }
        handle = os.open(self._state_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(handle, json.dumps(payload).encode("utf-8"))
            os.fsync(handle)
        finally:
            os.close(handle)

    def create_pull_request(self, repository: str, **arguments: Any) -> PullRequestDetails:
        """Open a pull request, record that it happened, and then optionally stop existing."""
        self.creations += 1
        details = super().create_pull_request(repository, **arguments)
        self._save()
        if self._crash_after_create:
            die_now()
        return details

    def add_labels(self, repository: str, pull_request_number: int, labels: Any) -> None:
        """Attach labels and persist them for the process that reads this next."""
        super().add_labels(repository, pull_request_number, labels)
        self._save()

    def request_reviewers(self, repository: str, pull_request_number: int, reviewers: Any) -> None:
        """Record review requests and persist them."""
        super().request_reviewers(repository, pull_request_number, reviewers)
        self._save()

    def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        """Record a comment and persist it."""
        super().add_comment(repository, pull_request_number, body)
        self._save()


class RecordingRealProcessRunner:
    """The real process runner, with a record of every command it was asked to run.

    Not a double: every command actually executes. The record exists because "no second push
    was issued" is a claim about what was run, and the remote SHA alone cannot distinguish a
    push that was skipped from one that was repeated with the same result.
    """

    def __init__(self) -> None:
        """Start with an empty record and the production runner underneath."""
        self._runner = AsyncioProcessRunner()
        self.commands: list[tuple[str, ...]] = []

    async def run(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
        cancellation_token: CancellationToken,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Record the command, then run it for real."""
        self.commands.append(tuple(command))
        return await self._runner.run(
            command, cwd, timeout_seconds, cancellation_token, environment
        )

    def ran(self, *prefix: str) -> int:
        """How many recorded commands start with these arguments."""
        return len([item for item in self.commands if item[: len(prefix)] == prefix])


class SucceedingProcessRunner:
    """Answer every planned command with exit zero, in the process that is about to die.

    The publication boundary needs a child workstream to reach commit and push, and the
    fixture repositories' package managers are neither installed nor network-reachable in
    the default tier. Git is untouched: the executor's Git service runs real `git`
    regardless of this runner, which is what makes the commit and the push real effects.
    """

    async def run(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
        cancellation_token: CancellationToken,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Return success without executing anything."""
        del cwd, timeout_seconds, cancellation_token, environment
        return ProcessResult(
            command=tuple(command),
            return_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.01,
        )


class DyingModelDouble:
    """A model whose every call is the process's last: record it durably, then `SIGKILL`.

    This is what a worker killed *inside* a pre-coding model call looks like from the
    database: the call's journal row is claimed and RUNNING, and nothing ever comes back to
    finish it. The file-backed count is what lets the surviving test prove the dead
    process made exactly one call -- and that the resumed one made its own rather than
    replaying a result out of the abandoned row.
    """

    def __init__(self, call_log: Path) -> None:
        """Bind the file that outlives this process."""
        self._call_log = call_log

    @property
    def invocations(self) -> int:
        """How many calls any process recorded before dying."""
        if not self._call_log.is_file():
            return 0
        recorded = self._call_log.read_text(encoding="utf-8").splitlines()
        return len([line for line in recorded if line])

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> NoReturn:
        """Record the call, then end the process without unwinding."""
        del instructions, input_text
        handle = os.open(self._call_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(handle, b"invoked\n")
            os.fsync(handle)
        finally:
            os.close(handle)
        die_now()


class CountingModelDouble:
    """A deterministic coding model that records every invocation in a file.

    Returns a real response the adapter parses and writes to disk, so the workspace after a
    crash holds actual bytes rather than a marker. The log is a file because the assertion
    that matters -- "the coding model was not called again" -- spans two processes.
    """

    def __init__(self, payload: dict[str, Any], call_log: Path) -> None:
        """Bind the response this model always gives and the file that counts the asking."""
        self._payload = payload
        self._call_log = call_log

    @property
    def invocations(self) -> int:
        """How many times any process has asked this model for a response."""
        if not self._call_log.is_file():
            return 0
        recorded = self._call_log.read_text(encoding="utf-8").splitlines()
        return len([line for line in recorded if line])

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> LLMResponse:
        """Record the call durably, then return the same deterministic coding response."""
        del instructions, input_text
        handle = os.open(self._call_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(handle, b"invoked\n")
            os.fsync(handle)
        finally:
            os.close(handle)
        return LLMResponse(
            output_text=json.dumps(self._payload),
            model="crash-suite-model",
            response_id="response-crash-suite",
            input_tokens=0,
            output_tokens=0,
        )


def install_index_lock_killer(workspace: Path, *, path: str, worker_pid: int) -> None:
    """Arrange for the next `git add` of one path to be killed while it holds the index lock.

    A Git clean filter runs while `git add` holds `.git/index.lock`, so killing from inside
    it leaves the lock behind -- which is exactly what this platform does to itself when it
    terminates a process group on cancellation or worker death. Doing it through real Git
    matters: the lock file is Git's, written and abandoned by Git, not a file this test made
    up to look like one.

    Two kills, because the platform's own process runner puts every subprocess in its own
    process group so it can terminate it wholesale. `kill 0` therefore reaches Git and this
    filter and stops there; the worker that was waiting on Git has to be named. Between them
    they reproduce the state a killed worker leaves: no worker, and a locked checkout.

    Only ever installed by the process that is about to die, and it names that process. A
    caller in the test process would be arranging for the test suite to kill itself.
    """
    killer = workspace / ".git" / "crash-suite-killer.sh"
    killer.write_text(f"#!/bin/sh\nkill -KILL {worker_pid}\nkill -KILL 0\n", encoding="utf-8")
    killer.chmod(0o755)
    subprocess.run(
        ("git", "config", "filter.crashsuite.clean", str(killer)),
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    (workspace / ".gitattributes").write_text(f"{path} filter=crashsuite\n", encoding="utf-8")


def remove_index_lock_killer(workspace: Path) -> None:
    """Take the filter back out, so the recovering attempt runs against ordinary Git."""
    subprocess.run(
        ("git", "config", "--unset", "filter.crashsuite.clean"),
        cwd=workspace,
        check=False,
        capture_output=True,
    )
    (workspace / ".gitattributes").unlink(missing_ok=True)


def run_crash_child(scenario: str, spec: dict[str, Any], *, timeout: float = 240.0) -> None:
    """Run one boundary's work in a real child process and require that it was killed.

    `start_new_session` puts the child in its own process group. Boundary 6 kills a process
    group on purpose, and without this that group would be the one running the test suite.

    The assertion on the return code is not decoration. If the child exits normally the work
    completed, no crash window was opened, and everything the test goes on to assert about
    recovery would be asserting about a run that never needed recovering.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "tests.crash_child", scenario, json.dumps(spec)],
        cwd=SERVER_ROOT,
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != -signal.SIGKILL:
        msg = (
            f"the {scenario} child was expected to be killed at its boundary but exited "
            f"{completed.returncode}.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        raise AssertionError(msg)


__all__ = [
    "AFTER_JOURNAL_SUCCESS",
    "BEFORE_JOURNAL_SUCCESS",
    "SERVER_ROOT",
    "CountingModelDouble",
    "CrashingJournal",
    "PersistentGitHubDouble",
    "RecordingRealProcessRunner",
    "die_now",
    "install_index_lock_killer",
    "remove_index_lock_killer",
    "run_crash_child",
]
