"""The external-operation journal reaches the screen as a read-only endpoint.

Every operational question of the 185-194 verification cycle -- "is 186 stuck", "which call
is this", "how old is the heartbeat" -- was answered by hand-querying `external_operations`
while the UI said "Running". These tests pin the endpoint that replaces the hand query:
one repository's rows, newest first, bounded, each carrying its stage name -- and nothing
from `safe_metadata`, because the journal's screen must not gain a second, unscreened
channel through a read route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from main import create_app
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from state.external_operations import (
    ExternalOperation,
    ExternalOperationType,
    child_attempt_of,
    stream_reissues_of,
)
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal, OperationResult
from tests.support import settle
from tests.test_feature_api import feature_payload

pytestmark = pytest.mark.asyncio

# A value that exists nowhere but in safe_metadata. If any response byte carries it, the
# endpoint has opened the unscreened channel this spec forbids.
_METADATA_CANARY = "SAFE-METADATA-CANARY-9c41"

_HEADERS = {"Authorization": "Bearer operations-test-key", "Idempotency-Key": "operations-001"}


async def _journal(tmp_path: Path) -> ExternalOperationJournal:
    """A durable journal of this test's own, the shape the deployment composes."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")
    await database.create_schema()
    return ExternalOperationJournal(database)


async def _record(
    journal: ExternalOperationJournal,
    *,
    repository_id: str,
    operation_type: ExternalOperationType,
    step: str,
    child_attempt: int | None = None,
) -> str:
    """One journal row for the fixture lineage, carrying the canary in its safe metadata.

    ``child_attempt`` is omitted by default, which is how every row written before the stamp
    existed reads -- the case the endpoint has to answer for without inventing a number.
    """
    operation = await journal.create_operation(
        workflow_id="feature-login",
        feature_id="feature-login",
        child_workflow_id=f"feature-login-{repository_id}",
        repository_id=repository_id,
        operation_type=operation_type,
        idempotency_key=f"feature-login:{repository_id}:{operation_type.value}:{step}",
        input_fingerprint=f"fingerprint-{step}",
        max_attempts=3,
        safe_metadata={
            "logical_step": step,
            "canary": _METADATA_CANARY,
            **({"child_attempt": child_attempt} if child_attempt is not None else {}),
        },
    )
    return operation.operation_id


async def _lineage(journal: ExternalOperationJournal) -> None:
    """Fixture a backend lineage across setup, coding, validation, review and publication."""
    for operation_type, step in (
        (ExternalOperationType.CLONE_REPOSITORY, "setup"),
        (ExternalOperationType.RUN_CODING_EXECUTOR, "coding"),
        (ExternalOperationType.RUN_LINTER, "validation"),
        (ExternalOperationType.RUN_REVIEWER, "review"),
        (ExternalOperationType.CREATE_PULL_REQUEST, "publication"),
    ):
        operation_id = await _record(
            journal, repository_id="backend", operation_type=operation_type, step=step
        )
        if operation_type is ExternalOperationType.RUN_REVIEWER:
            # The lineage's live row: claimed and running, with a real heartbeat.
            await journal.claim_attempt(operation_id)
        elif operation_type is ExternalOperationType.CLONE_REPOSITORY:
            await journal.claim_attempt(operation_id)
            await journal.record_result(operation_id, external_reference=None, result_payload=None)


async def test_the_endpoint_returns_the_journals_rows_with_their_recorded_shapes(
    tmp_path: Path,
) -> None:
    """1a: the recorded rows come back newest first, staged, and only for their repository."""
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        await settle(app)
        await _lineage(journal)

        backend = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )
        frontend = await client.get(
            "/features/feature-login/workstreams/frontend/operations", headers=_HEADERS
        )
        unknown_feature = await client.get(
            "/features/no-such-feature/workstreams/backend/operations", headers=_HEADERS
        )
        unauthenticated = await client.get("/features/feature-login/workstreams/backend/operations")

    assert backend.status_code == 200, backend.text
    body = backend.json()
    assert body["feature_id"] == "feature-login"
    assert body["repository_id"] == "backend"
    rows: list[dict[str, Any]] = body["operations"]
    # Newest first: the lineage was recorded setup -> publication, so it reads back reversed.
    assert [item["operation_type"] for item in rows] == [
        "create_pull_request",
        "run_reviewer",
        "run_linter",
        "run_coding_executor",
        "clone_repository",
    ]
    assert [item["stage"] for item in rows] == [
        "publication",
        "review",
        "validation",
        "coding",
        "setup",
    ]
    by_type = {item["operation_type"]: item for item in rows}
    running = by_type["run_reviewer"]
    assert running["status"] == "running"
    assert running["attempt"] == 1
    assert running["max_attempts"] == 3
    assert running["started_at"] is not None
    assert running["heartbeat_at"] is not None
    assert running["completed_at"] is None
    assert running["error_code"] is None
    finished = by_type["clone_repository"]
    assert finished["status"] == "succeeded"
    assert finished["completed_at"] is not None
    pending = by_type["run_coding_executor"]
    assert pending["status"] == "pending"
    assert pending["attempt"] == 0
    assert pending["started_at"] is None

    # A repository that has recorded nothing answers an empty list, never 404: "nothing has
    # happened yet" is an ordinary state of a queued workstream.
    assert frontend.status_code == 200
    assert frontend.json()["operations"] == []

    # The feature itself still has to exist, and the read is authenticated like every other
    # `/features/*` route.
    assert unknown_feature.status_code == 404
    assert unauthenticated.status_code in {401, 403}


async def test_nothing_from_safe_metadata_appears_in_any_response_byte(tmp_path: Path) -> None:
    """1b: the journal's screened metadata never travels; the row shape is the whole answer."""
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        await settle(app)
        await _lineage(journal)

        response = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )

    assert response.status_code == 200
    assert _METADATA_CANARY.encode() not in response.content
    # Byte-level, then shape-level: no row carries a metadata field at all, so a future
    # serializer change cannot reopen the channel by renaming it.
    for row in response.json()["operations"]:
        assert "safe_metadata" not in row
        assert set(row) == {
            "operation_id",
            "operation_type",
            "stage",
            "status",
            "attempt",
            "max_attempts",
            "child_attempt",
            "started_at",
            "heartbeat_at",
            "completed_at",
            "error_code",
            "repeat",
            # Read from `result_payload`, never from the screened metadata -- what a call
            # reported about itself on the way out, like the model beside it.
            "stream_reissues",
        }
    # The fixture rows carry no attempt stamp, and the endpoint says so rather than assuming
    # the current attempt: a row whose attempt nobody recorded belongs to no attempt.
    assert {row["child_attempt"] for row in response.json()["operations"]} == {None}


async def test_the_executor_stamps_the_child_attempt_on_every_row_it_creates(
    tmp_path: Path,
) -> None:
    """23: the attempt is on the row, so a retry that never re-clones is still divisible.

    Run 197 BE: three attempts rendered as one, because the only confident divider was a
    fresh `clone_repository` and a retry-edits-in-place retry never clones. Stamped through
    the real executor rather than by writing metadata by hand -- the point of the change is
    that the executor does it for every operation type without each caller remembering to.
    """
    journal = await _journal(tmp_path)

    async def effect() -> tuple[str, OperationResult]:
        return "done", OperationResult()

    for attempt in (0, 1):
        executor = ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id="feature-login",
                feature_id="feature-login",
                child_workflow_id="feature-login-backend",
                repository_id="backend",
                child_attempt=attempt,
            ),
        )
        # No clone on the second attempt: exactly the shape that used to be indivisible.
        steps = (
            (ExternalOperationType.CLONE_REPOSITORY, "clone"),
            (ExternalOperationType.RUN_CODING_EXECUTOR, "coding_and_file_changes"),
        )
        for operation_type, step in steps if attempt == 0 else steps[1:]:
            await executor.run(
                operation_type=operation_type,
                logical_step=step,
                # The attempt is part of the input here only so the two attempts get distinct
                # idempotency keys, as a real retry's changed instructions do. The stamp
                # itself is never in the key: keying on it would give the same work a new
                # identity on every retry, which is what the reuse machinery exists to stop.
                safe_input={"workspace_path": "/w", "attempt_input": attempt},
                action=effect,
            )

    rows = await journal.list_operations_for_repository("feature-login", "backend")
    assert [(str(row.operation_type), child_attempt_of(row)) for row in rows] == [
        ("run_coding_executor", 1),
        ("run_coding_executor", 0),
        ("clone_repository", 0),
    ]
    # A bare integer, and the journal's credential screen accepted it: it is persisted and
    # read back as an `int`, not as a string that happens to look like one.
    stamps = [row.safe_metadata["child_attempt"] for row in rows]
    assert all(type(value) is int for value in stamps)
    # The stamp is a label on the row, not part of its identity. Two attempts of the same
    # logical step are two rows, and neither key mentions the attempt.
    assert len({row.idempotency_key for row in rows}) == 3
    assert not any("child_attempt" in row.idempotency_key for row in rows)


async def test_the_endpoint_publishes_the_attempt_stamp_and_nothing_else_beside_it(
    tmp_path: Path,
) -> None:
    """23: one typed integer out of `safe_metadata`, and the channel stays that narrow."""
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        await settle(app)
        # Attempt 1's coding row, attempt 0's clone, and an unstamped row from before the
        # stamp existed -- the three cases a reader has to tell apart.
        await _record(
            journal,
            repository_id="backend",
            operation_type=ExternalOperationType.CLONE_REPOSITORY,
            step="clone",
            child_attempt=0,
        )
        await _record(
            journal,
            repository_id="backend",
            operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
            step="coding_and_file_changes",
            child_attempt=1,
        )
        await _record(
            journal,
            repository_id="backend",
            operation_type=ExternalOperationType.RUN_LINTER,
            step="historical",
        )

        response = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )

    assert response.status_code == 200, response.text
    rows = response.json()["operations"]
    assert [(row["operation_type"], row["child_attempt"]) for row in rows] == [
        ("run_linter", None),
        ("run_coding_executor", 1),
        ("clone_repository", 0),
    ]
    # The one field, and only it: the canary sharing the same metadata dictionary still never
    # travels, so publishing the stamp did not open the passthrough.
    assert _METADATA_CANARY.encode() not in response.content
    assert all("logical_step" not in row for row in rows)


async def test_a_metadata_stamp_that_is_not_an_attempt_number_is_published_as_nothing(
    tmp_path: Path,
) -> None:
    """23: the reader is typed, so a row carrying junk under the key reads as unstamped.

    `True` is the case worth naming: Python counts it as an integer, so an untyped read would
    match it against a retry counter of 1 and file the row under the wrong attempt.
    """
    journal = await _journal(tmp_path)
    for index, value in enumerate(("1", True, -1, 1.0, None)):
        operation = await journal.create_operation(
            workflow_id="feature-login",
            feature_id="feature-login",
            child_workflow_id="feature-login-backend",
            repository_id="backend",
            operation_type=ExternalOperationType.RUN_TESTS,
            idempotency_key=f"junk-{index}",
            input_fingerprint=f"fingerprint-{index}",
            safe_metadata={"logical_step": "validation", "child_attempt": value},
        )
        assert child_attempt_of(operation) is None


async def test_the_response_carries_where_each_finished_attempt_ended(tmp_path: Path) -> None:
    """65 A.2: the ending rides the same polled response, so the drawer costs no request.

    An isolated application runs the whole feature through the mock orchestrator, so the
    artifacts this reads -- the per-attempt result, its completion and its review -- are the
    ones a real run writes rather than fixtures written to match the assembler. What is
    asserted is the block's presence and shape on the response the drawer polls; which
    ending each record kind produces is pinned in `test_attempt_endings.py`.
    """
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        await settle(app)
        response = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["attempt"] for item in body["attempts"]] == [0]
    ending = body["attempts"][0]
    assert set(ending) == {
        "attempt",
        "ended_by",
        "stage",
        "detail",
        "workspace",
        "self_review_outcome",
        "self_review_corrected_files",
        "source_repair_passes",
        # The in-attempt passes are unjournaled, so the re-issues they spent can only reach a
        # reader here -- assembled from the completion artifact, like everything else in this
        # block, and never from the screened metadata.
        "stream_reissues",
    }
    # The feature delivered, so its one attempt is the one that delivered -- and the stage
    # the ending names is a stage name the rows above are also stamped with.
    assert ending["ended_by"] in {"approved", "published"}
    assert ending["stage"] == "publication"
    # The metadata channel is still shut. The block is assembled from artifacts, and nothing
    # from `safe_metadata` travels with it.
    assert _METADATA_CANARY.encode() not in response.content


async def test_a_repository_with_no_finished_attempt_answers_an_empty_block(
    tmp_path: Path,
) -> None:
    """65 A.2: nothing to say is said as nothing, never as an ending nobody recorded."""
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        # Deliberately not settled: no attempt has run, so no attempt has ended.
        response = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )

    assert response.status_code == 200, response.text
    assert response.json()["attempts"] == []


async def test_a_repeated_row_says_why_it_repeats_from_the_columns_not_the_metadata(
    tmp_path: Path,
) -> None:
    """65 A.3: 201-a3's shape -- two linter rows and two test rows, each honest.

    The linter ran at the revision the attempt started from and again after an in-attempt
    repair changed the tree: same command, new revision. The two test rows ran at one
    revision under two different commands: the change's own scoped test beside the full
    suite (47-C). Rendered as bare duplicates, both look like a bug.

    Read from `repository_revision` and `command_fingerprint`, which are columns. The same
    information is in `logical_step`, which lives in `safe_metadata` and never travels --
    and the canary assertion below is what keeps that true.
    """
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    lint = "f64699799f15fac712deb6b9a10379be87e25b7a1dcb6a1238f96e930fc20491"
    suite = "897d0436a9fd300b8bf2323bcc0463eb2bf039c0c2196f857f5610357376383d"
    scoped = "890b8bbfb347839a67ef5df2787a1acd440f1549d9430e07f3c540ecf2898dbb"
    before = "f137767074f6df67cca20f8244e3f0df1451561f07c1428a73fede65d1a70d07"
    after = "9e773a4d6a1cded6153d6652388d1a34727b22bf1460a941dbc816ee055134d4"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        await settle(app)
        for step, operation_type, revision, fingerprint in (
            ("lint-before", ExternalOperationType.RUN_LINTER, before, lint),
            ("lint-after", ExternalOperationType.RUN_LINTER, after, lint),
            ("test-scoped", ExternalOperationType.RUN_TESTS, after, scoped),
            ("test-suite", ExternalOperationType.RUN_TESTS, after, suite),
            # One `create_commit` per attempt, so it is not a repeat of anything -- and two
            # coding rows, which carry neither column and so have nothing to be told apart by.
            ("commit", ExternalOperationType.CREATE_COMMIT, None, None),
            ("coding-1", ExternalOperationType.RUN_CODING_EXECUTOR, None, None),
            ("coding-2", ExternalOperationType.RUN_CODING_EXECUTOR, None, None),
        ):
            operation = await journal.create_operation(
                workflow_id="feature-login",
                feature_id="feature-login",
                child_workflow_id="feature-login-backend",
                repository_id="backend",
                operation_type=operation_type,
                idempotency_key=f"feature-login:backend:{step}",
                input_fingerprint=f"fingerprint-{step}",
                repository_revision=revision,
                command_fingerprint=fingerprint,
                safe_metadata={
                    "logical_step": step,
                    "canary": _METADATA_CANARY,
                    "child_attempt": 3,
                },
            )
            assert operation.operation_id

        response = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )

    assert response.status_code == 200, response.text
    by_step: dict[str, list[Any]] = {}
    for row in response.json()["operations"]:
        by_step.setdefault(row["operation_type"], []).append(row["repeat"])
    # Newest first, so the later linter run comes first: it is the re-run, and the first run
    # of the command is not described as a repeat of itself.
    assert by_step["run_linter"] == [{"kind": "new_revision", "detail": after[:8]}, None]
    # Two commands at one revision: each is told apart by its own fingerprint, and neither is
    # called a re-run of the other.
    assert by_step["run_tests"] == [
        {"kind": "different_command", "detail": suite[:8]},
        {"kind": "different_command", "detail": scoped[:8]},
    ]
    # Nothing to distinguish them, so the answer is only that this is another run of the same
    # step -- never a fabricated reason, and never an ordinal, which the response's own row
    # budget would silently change the meaning of.
    assert by_step["run_coding_executor"] == [
        {"kind": "same_step", "detail": None},
        {"kind": "same_step", "detail": None},
    ]
    # One row of its type in the attempt: not a repeat at all.
    assert by_step["create_commit"] == [None]
    # The guard that made this design read columns rather than parse `logical_step`.
    assert _METADATA_CANARY.encode() not in response.content
    assert all("logical_step" not in row for row in response.json()["operations"])


async def test_the_baseline_checks_are_setup_and_the_change_s_validation_is_not(
    tmp_path: Path,
) -> None:
    """67 T1/T2/T3: one type, two phases, and the row says which it was.

    The platform runs the repository's own commands twice: once on the untouched checkout to
    prove it can pass them at all, and once on what the attempt wrote. Same commands, same
    tool, same `run_linter`/`run_tests`/`run_build` types -- so a stage mapped from the type
    alone filed the repository's measurement under the change's stage. AB-Feature-201's
    backend attempt 0 was stopped at coding by the self-review gate, never validated its
    change, and showed three ticked validation rows anyway.
    """
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        await settle(app)
        # The order the runtime writes them in: install, the baseline suite, the coding call,
        # then the attempt's own validation of what it wrote.
        for operation_type, step in (
            (ExternalOperationType.INSTALL_DEPENDENCIES, "repository_preflight:dependency_install"),
            (ExternalOperationType.RUN_LINTER, "baseline_validation:lint:806ff601:f6469979"),
            (ExternalOperationType.RUN_TESTS, "baseline_validation:test:806ff601:897d0436"),
            (ExternalOperationType.RUN_BUILD, "baseline_validation:build:806ff601:16be003c"),
            (ExternalOperationType.RUN_CODING_EXECUTOR, "coding_and_file_changes"),
            (ExternalOperationType.RUN_LINTER, "validation:lint:c0ffee01:f6469979"),
        ):
            await _record(
                journal,
                repository_id="backend",
                operation_type=operation_type,
                step=step,
                child_attempt=0,
            )

        response = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )

    assert response.status_code == 200, response.text
    rows = response.json()["operations"]
    staged = [(row["operation_type"], row["stage"]) for row in reversed(rows)]
    assert staged == [
        ("install_dependencies", "setup"),
        # T1: the baseline suite is the platform preparing a checkout, beside the install it
        # shares a revision with -- not this attempt's validation.
        ("run_linter", "setup"),
        ("run_tests", "setup"),
        ("run_build", "setup"),
        ("run_coding_executor", "coding"),
        # T2: the attempt's own run is untouched, which is the case the widening must not
        # swallow.
        ("run_linter", "validation"),
    ]
    # T3: one attempt holding both, and no row counted twice.
    assert len(rows) == len({row["operation_id"] for row in rows})
    # And the step itself never reaches the response: the stage is what the client reads.
    assert "baseline_validation" not in response.text


async def test_a_row_written_before_the_baseline_stamp_keeps_todays_stage(
    tmp_path: Path,
) -> None:
    """67 T5: history is not re-labelled from a heuristic that cannot tell the two apart.

    A validation row whose revision matches the preflight install's ran on the untouched
    checkout -- unless the coding call wrote nothing, in which case the attempt's own run
    matches the same rule. So an unstamped row keeps the attribution it has always had, and
    the fix reaches new runs only.
    """
    app = create_app(platform_api_key="operations-test-key")
    journal = await _journal(tmp_path)
    app.state.operation_journal = journal
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=_HEADERS, json=feature_payload())
        assert started.status_code == 201, started.text
        await settle(app)
        await _record(
            journal,
            repository_id="backend",
            operation_type=ExternalOperationType.RUN_LINTER,
            step="validation:lint:806ff601:f6469979",
            child_attempt=0,
        )

        response = await client.get(
            "/features/feature-login/workstreams/backend/operations", headers=_HEADERS
        )

    assert response.status_code == 200, response.text
    assert [row["stage"] for row in response.json()["operations"]] == ["validation"]


def _row_with_payload(payload: object) -> ExternalOperation:
    """One operation carrying just the payload under test.

    ``model_construct`` on purpose: `stream_reissues_of` is a pure reader over one field, and
    filling a dozen unrelated required columns to exercise it would test the constructor.
    """
    return ExternalOperation.model_construct(result_payload=payload)


async def test_the_re_issue_count_is_read_from_the_payload_and_never_guessed() -> None:
    """Both shapes a journaled call writes, and every value that is not a count.

    Two seams journal this. The coding executor writes its facts flat; a planning call routed
    through `llm_call_record` nests them under `execution`. One reader answers both, because
    "did this call's stream stall" is one question regardless of which seam recorded it.
    """
    # The coding executor's flat payload, and the nested block a planning call records.
    assert stream_reissues_of(_row_with_payload({"model": "m", "stream_reissues": 2})) == 2
    assert stream_reissues_of(_row_with_payload({"execution": {"stream_reissues": 1}})) == 1

    # Zero is an answer and survives: streams that spoke first time are the reassuring
    # reading, and a surface that cannot see them cannot tell a healthy budget from an
    # unmeasured one.
    assert stream_reissues_of(_row_with_payload({"stream_reissues": 0})) == 0

    # Everything that is not a count reads as no measurement. `True` is rejected explicitly
    # -- Python counts it as an integer, and it would otherwise render as one re-issue.
    assert stream_reissues_of(_row_with_payload({"stream_reissues": True})) is None
    assert stream_reissues_of(_row_with_payload({"stream_reissues": -1})) is None
    assert stream_reissues_of(_row_with_payload({"stream_reissues": "2"})) is None
    assert stream_reissues_of(_row_with_payload({"execution": "not a block"})) is None
    assert stream_reissues_of(_row_with_payload({"model": "m"})) is None
    assert stream_reissues_of(_row_with_payload(None)) is None
