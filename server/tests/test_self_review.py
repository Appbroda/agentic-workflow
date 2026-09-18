"""The bounded self-review: its boundary, its parsing, and how it gates one attempt.

47- Part E. The Engineer reviews its own finished work exactly once, corrects what is
bounded, and hands back what is not -- gating `completion_status` and nothing further. The
realistic end-to-end scenarios (E1-E3) live in the realrepo tier; what is here is the
boundary's own contract: strict parsing with one bounded re-ask, degradation to exactly the
pre-Part-E behaviour on every failure of the review itself, the named substantive escape,
and the key-material rule applied to the review's input like everywhere else (F1).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import ImageInput, LLMResponse, MockCodingExecutor
from agents.engineer.agent import EngineerAgent
from artifacts.schemas import CodeCompletionArtifact, FileChange
from prompts.prompt_loader import PromptLoader
from tests.test_agents import (
    agent_state,
    code_completion_artifact,
    review_artifact,
    task_plan_artifact,
)
from tools.implementation_completeness import classify_file_change
from tools.self_review import (
    SELF_REVIEW_DIAGNOSTIC_PREFIX,
    SELF_REVIEW_DIRECTIVE,
    SELF_REVIEW_SUBSTANTIVE_OUTCOME,
    CodingRoleSelfReviewer,
    NullSelfReviewer,
    SelfReviewAssessment,
    SelfReviewFinding,
    evidence_blind,
    self_review_diagnostic,
)
from tools.source_formatting import SourceValidationError


class ScriptedTextClient:
    """Return queued raw response texts, so malformed output is expressible byte for byte."""

    def __init__(self, outputs: list[str]) -> None:
        """Queue one output per expected call."""
        self._outputs = list(outputs)
        self.calls: list[tuple[str, str]] = []

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
        """Pop the next scripted text and record what was asked."""
        self.calls.append((instructions, input_text))
        if not self._outputs:
            msg = "scripted client exhausted"
            raise RuntimeError(msg)
        return LLMResponse(
            response_id=f"review-{len(self.calls)}",
            model="scripted-review-model",
            output_text=self._outputs.pop(0),
            input_tokens=1,
            output_tokens=1,
        )


def _review_payload(findings: list[dict[str, Any]]) -> str:
    """One schema-valid review response with these findings."""
    return json.dumps(
        {
            "summary": "Reviewed the finished work against the assignment.",
            "requirement_coverage": [
                {
                    "requirement_id": "requirement-1",
                    "status": "partial" if findings else "implemented",
                    "evidence": "src/service.py implements the endpoint.",
                }
            ],
            "findings": findings,
        }
    )


class StubSelfReviewer:
    """Return a canned assessment, counting how many passes the agent actually asked for."""

    def __init__(self, assessment: SelfReviewAssessment | None) -> None:
        """Queue the single outcome this reviewer reports."""
        self._assessment = assessment
        self.calls: list[tuple[str, str]] = []

    async def review(self, *, instructions: str, input_text: str) -> SelfReviewAssessment | None:
        """Record the ask and return the canned verdict."""
        self.calls.append((instructions, input_text))
        return self._assessment


def _assessment(*findings: SelfReviewFinding) -> SelfReviewAssessment:
    """Build a minimal assessment carrying these findings."""
    return SelfReviewAssessment(
        findings=tuple(findings),
        requirement_coverage=(
            {"requirement_id": "requirement-1", "status": "partial", "evidence": "seen"},
        ),
        summary="reviewed",
        response_id="review-1",
        model="stub-review-model",
        execution={"provider": "", "model": "stub-review-model"},
    )


class QueuedCodingExecutor:
    """Run a different scripted write per call, so the correction pass is steerable."""

    def __init__(self, updates: list[dict[str, str]]) -> None:
        """Queue one file-update set per expected coding call."""
        self._executors = [MockCodingExecutor(file_updates=item) for item in updates]
        self.calls: list[str] = []

    async def execute(self, **kwargs: Any) -> Any:
        """Delegate to the next queued deterministic executor."""
        self.calls.append(kwargs["instructions"])
        return await self._executors.pop(0).execute(**kwargs)


# --------------------------------------------------------------------------------------
# The boundary: strict parsing, one bounded re-ask, degradation on its own failure
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_clean_response_parses_into_findings_and_coverage() -> None:
    """The structured verdict comes back typed, with the execution that produced it."""
    client = ScriptedTextClient(
        [
            _review_payload(
                [
                    {
                        "description": "The update branch is missing.",
                        "classification": "localized",
                        "paths": ["src/service.py"],
                        "requirement_id": "requirement-1",
                    }
                ]
            )
        ]
    )

    assessment = await CodingRoleSelfReviewer(client).review(
        instructions="implement the change", input_text="{}"
    )

    assert assessment is not None
    assert len(assessment.findings) == 1
    finding = assessment.findings[0]
    assert finding.classification == "localized"
    assert not finding.substantive
    assert finding.paths == ("src/service.py",)
    assert assessment.requirement_coverage[0]["status"] == "partial"
    assert assessment.model == "scripted-review-model"
    # The single pass carried the directive appended to the engineer's own instructions.
    assert len(client.calls) == 1
    assert client.calls[0][0].startswith("implement the change")
    assert "Classify every finding" in client.calls[0][0]


@pytest.mark.asyncio
async def test_a_fenced_response_is_still_read() -> None:
    """A model that wraps its JSON in a Markdown fence has still answered the question."""
    client = ScriptedTextClient([f"```json\n{_review_payload([])}\n```"])

    assessment = await CodingRoleSelfReviewer(client).review(
        instructions="implement", input_text="{}"
    )

    assert assessment is not None
    assert assessment.findings == ()


@pytest.mark.asyncio
async def test_a_malformed_response_is_re_asked_once_then_read() -> None:
    """The same bounded repair the coding executor gives a malformed response."""
    client = ScriptedTextClient(["this is prose, not a review", _review_payload([])])

    assessment = await CodingRoleSelfReviewer(client).review(
        instructions="implement", input_text="{}"
    )

    assert assessment is not None
    assert len(client.calls) == 2
    assert "could not be read" in client.calls[1][0]


@pytest.mark.asyncio
async def test_two_unreadable_responses_degrade_to_no_review() -> None:
    """A review that cannot be had is None -- never an exception the attempt inherits."""
    client = ScriptedTextClient(["not json", '{"findings": "not a list"}'])

    assessment = await CodingRoleSelfReviewer(client).review(
        instructions="implement", input_text="{}"
    )

    assert assessment is None
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_an_unknown_classification_is_not_guessed_at() -> None:
    """A verdict this gate cannot act on is re-asked, not coerced into one it can."""
    invalid = json.dumps(
        {
            "summary": "reviewed",
            "requirement_coverage": [],
            "findings": [{"description": "gap", "classification": "cosmetic"}],
        }
    )
    client = ScriptedTextClient([invalid, invalid])

    assessment = await CodingRoleSelfReviewer(client).review(
        instructions="implement", input_text="{}"
    )

    assert assessment is None
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_a_model_error_degrades_to_no_review() -> None:
    """The boundary's own failure is absorbed; the attempt proceeds exactly as before."""

    class FailingClient:
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
            del instructions, input_text
            msg = "provider unavailable"
            raise RuntimeError(msg)

    assessment = await CodingRoleSelfReviewer(FailingClient()).review(
        instructions="implement", input_text="{}"
    )

    assert assessment is None


@pytest.mark.asyncio
async def test_the_null_reviewer_reviews_nothing() -> None:
    """A composition with no configured boundary runs no review."""
    assert (await NullSelfReviewer().review(instructions="implement", input_text="{}")) is None


def test_the_diagnostic_names_what_the_finding_names() -> None:
    """The sentence carries the prefix, the classification, the paths and the requirement."""
    diagnostic = self_review_diagnostic(
        SelfReviewFinding(
            description="The update branch is missing.",
            classification="substantive",
            paths=("src/service.py",),
            requirement_id="requirement-1",
        )
    )

    assert diagnostic.startswith(SELF_REVIEW_DIAGNOSTIC_PREFIX)
    assert "(substantive)" in diagnostic
    assert "src/service.py" in diagnostic
    assert "requirement-1" in diagnostic


# --------------------------------------------------------------------------------------
# The gate on one attempt: what `completed` now means, and every way it degrades
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_composed_reviewer_leaves_the_completion_without_a_record(
    tmp_path: Path,
) -> None:
    """No boundary, no record: pre-Part-E compositions produce byte-identical metadata."""
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    assert completion.completion_status == "completed"
    assert "self_review" not in completion.metadata


@pytest.mark.asyncio
async def test_a_review_that_could_not_run_is_recorded_and_the_attempt_proceeds(
    tmp_path: Path,
) -> None:
    """Composed-and-unavailable is a fact about the attempt, distinct from not composed."""
    reviewer = StubSelfReviewer(None)
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        self_reviewer=reviewer,
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    assert completion.completion_status == "completed"
    assert completion.metadata["self_review"] == {"ran": False, "outcome": "unavailable"}
    assert len(reviewer.calls) == 1


@pytest.mark.asyncio
async def test_a_clean_review_completes_and_is_recorded(tmp_path: Path) -> None:
    """No findings means ready -- and the record says the review ran and found nothing."""
    reviewer = StubSelfReviewer(_assessment())
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        self_reviewer=reviewer,
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    assert completion.completion_status == "completed"
    record = completion.metadata["self_review"]
    assert record["ran"] is True
    assert record["outcome"] == "clean"
    assert record["findings"] == []
    assert record["corrections_applied"] == []
    assert len(reviewer.calls) == 1


@pytest.mark.asyncio
async def test_a_localized_finding_is_corrected_once_and_the_attempt_completes(
    tmp_path: Path,
) -> None:
    """One correction pass, re-validated, recorded -- and exactly one review pass (E3)."""
    reviewer = StubSelfReviewer(
        _assessment(
            SelfReviewFinding(
                description="The update branch is missing.",
                classification="localized",
                paths=("src/service.py",),
            )
        )
    )
    executor = QueuedCodingExecutor(
        [
            {"src/service.py": "value = 1\n"},
            {"src/service.py": "value = 1\nupdated = True\n"},
        ]
    )
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        self_reviewer=reviewer,
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    assert completion.completion_status == "completed"
    record = completion.metadata["self_review"]
    assert record["outcome"] == "corrected"
    assert record["corrections_applied"] == ["src/service.py"]
    assert record["correction_response_id"] == "mock-response"
    # The effect: the corrected bytes are what the workspace holds.
    content = (tmp_path / "src/service.py").read_text(encoding="utf-8")
    assert "updated = True" in content
    # Exactly one review pass, however the corrections went: no review-of-a-review.
    assert len(reviewer.calls) == 1
    # Two coding calls: the implementation and the one correction.
    assert len(executor.calls) == 2
    assert SELF_REVIEW_DIAGNOSTIC_PREFIX in executor.calls[1]


@pytest.mark.asyncio
async def test_a_substantive_finding_leaves_the_attempt_with_the_named_outcome(
    tmp_path: Path,
) -> None:
    """No correction is attempted; the escape is named and the record survives the raise."""
    reviewer = StubSelfReviewer(
        _assessment(
            SelfReviewFinding(
                description="The contract requires a different response shape.",
                classification="substantive",
            )
        )
    )
    executor = QueuedCodingExecutor([{"src/service.py": "value = 1\n"}])

    with pytest.raises(SourceValidationError) as rejected:
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=executor,
            self_reviewer=reviewer,
        ).run(agent_state(tmp_path, [task_plan_artifact()]))

    error = rejected.value
    assert error.terminal_outcome == SELF_REVIEW_SUBSTANTIVE_OUTCOME
    assert any(item.startswith(SELF_REVIEW_DIAGNOSTIC_PREFIX) for item in error.diagnostics)
    # One review pass, one coding call: nothing tried to correct a substantive problem.
    assert len(reviewer.calls) == 1
    assert len(executor.calls) == 1
    completion = error.code_completion
    assert isinstance(completion, CodeCompletionArtifact)
    assert completion.completion_status == "failed"
    assert completion.metadata["terminal_outcome"] == SELF_REVIEW_SUBSTANTIVE_OUTCOME
    assert completion.metadata["self_review"]["outcome"] == "substantive_problem"
    # Nothing about the *source* was rejected: the pre-commit allowance has no claim here.
    assert completion.metadata["source_validation_rejected"] is False


@pytest.mark.asyncio
async def test_a_correction_that_writes_nothing_leaves_the_findings_unresolved(
    tmp_path: Path,
) -> None:
    """`completed` means the findings were resolved; a no-op correction may not claim it."""
    reviewer = StubSelfReviewer(
        _assessment(
            SelfReviewFinding(
                description="The empty case is unhandled.", classification="localized"
            )
        )
    )
    executor = QueuedCodingExecutor([{"src/service.py": "value = 1\n"}, {}])

    with pytest.raises(SourceValidationError) as rejected:
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=executor,
            self_reviewer=reviewer,
        ).run(agent_state(tmp_path, [task_plan_artifact()]))

    error = rejected.value
    assert error.terminal_outcome == SELF_REVIEW_SUBSTANTIVE_OUTCOME
    completion = error.code_completion
    assert isinstance(completion, CodeCompletionArtifact)
    assert completion.metadata["self_review"]["outcome"] == "corrections_failed"
    assert len(reviewer.calls) == 1


@pytest.mark.asyncio
async def test_key_material_never_reaches_the_review_input(tmp_path: Path) -> None:
    """F1: the one key-material rule applies to the review's input, and withholding is named."""
    secret = "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----\n"
    client = ScriptedTextClient([_review_payload([])])
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={"src/service.py": "value = 1\n", "config/deploy.pem": secret}
        ),
        self_reviewer=CodingRoleSelfReviewer(client),
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    record = completion.metadata["self_review"]
    assert record["outcome"] == "clean"
    # The bytes never left the workspace; the path is still named, so "the review never saw
    # the file" stays answerable from the artifact.
    review_input = client.calls[0][1]
    assert "PRIVATE KEY" not in review_input
    assert "config/deploy.pem" in record["withheld_paths"]
    # The safe file was shown.
    assert "src/service.py" in review_input


# --------------------------------------------------------------------------------------
# 76-: the review sees the completion view whole, or says so -- and blindness is not
# a finding. Fixture shapes follow AB-Feature-207's preserved artifacts.
# --------------------------------------------------------------------------------------

# The verbatim substantive finding that killed AB-Feature-207 attempt 4: every path it
# names as absent is in the same completion artifact's own file_changes.
_207_FINDING = (
    "The quoted changed_files do not contain the bulk route, authorization and upload "
    "middleware, CSV classification and issue collection, duplicate checks, GAM batch "
    "integration, Mongoose transaction, synchronous controller, safe error contract, or "
    "required tests. Those requirements cannot be considered implemented from the "
    "supplied change set."
)
_207_PATHS = (
    "server/routes/app/app.route.js",
    "server/controllers/app/app.controller.js",
    "server/middlewares/bulkAppUpload.js",
    "server/validation/bulkCreateApps.validation.js",
    "server/services/app/app.service.js",
    "server/models/app.model.js",
    "server/gamUtils/app/create_apps.py",
    "server/tests/bulk-create-apps.test.js",
)


def _prior_completion(paths: list[str]) -> CodeCompletionArtifact:
    """A prior attempt's completion whose file_changes carry this cumulative candidate."""
    return code_completion_artifact().model_copy(
        update={
            "file_changes": [
                FileChange(path=path, change_type="added", description="Prior attempt write.")
                for path in paths
            ]
        }
    )


def _write(workspace: Path, path: str, content: str) -> None:
    """Put one prior-attempt file into the workspace the lineage claims it from."""
    target = workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_evidence_blindness_is_narrow() -> None:
    """Only a substantive finding judging nothing but declared-unseen files is blind."""

    def substantive(*paths: str) -> SelfReviewFinding:
        return SelfReviewFinding(description="gap", classification="substantive", paths=paths)

    unseen = {"a.py", "b.py"}
    assert evidence_blind(substantive("a.py"), unseen)
    assert evidence_blind(substantive("a.py", "b.py"), unseen)
    # One genuinely seen path keeps the judgement alive.
    assert not evidence_blind(substantive("a.py", "seen.py"), unseen)
    # Pathless stands: the directive is the only defense there, and the record shows it.
    assert not evidence_blind(substantive(), unseen)
    # Unwritten is not unseen: "this file was never created" is the true finding.
    assert not evidence_blind(substantive("never_written.py"), unseen)
    # Localized findings are never demoted; their correction reads the workspace itself.
    assert not evidence_blind(
        SelfReviewFinding(description="gap", classification="localized", paths=("a.py",)),
        unseen,
    )


def test_the_directive_teaches_the_three_states() -> None:
    """The prompt half of 76- C: quoted, withheld-with-reason, truncated -- all named."""
    assert "withheld_paths" in SELF_REVIEW_DIRECTIVE
    assert "withheld_reasons" in SELF_REVIEW_DIRECTIVE
    assert "truncation marker" in SELF_REVIEW_DIRECTIVE
    assert "never report a withheld file" in SELF_REVIEW_DIRECTIVE


def test_every_drop_is_named_with_its_reason(tmp_path: Path) -> None:
    """The budget mechanics declare each not-fully-shown file in the shared vocabulary."""
    agent = EngineerAgent(
        prompt_loader=PromptLoader(), coding_executor=MockCodingExecutor(file_updates={})
    )
    _write(tmp_path, "config/deploy.pem", "not even a key")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    _write(tmp_path, "huge.txt", "x" * 2_000_001)
    _write(
        tmp_path,
        "secrets_lookalike.js",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----\n",
    )
    # Cut at the per-file cap: quoted with the flag and the in-line marker, never withheld.
    _write(tmp_path, "long.py", "y" * 13_000)
    # Five fillers fit beside the cut file; the sixth exhausts the total budget.
    for index in range(6):
        _write(tmp_path, f"filler_{index}.py", "z" * 11_900)

    contents, withheld = agent._self_review_files(  # noqa: SLF001 - the unit under test
        tmp_path,
        [
            "config/deploy.pem",
            "blob.bin",
            "huge.txt",
            "secrets_lookalike.js",
            "long.py",
            *[f"filler_{index}.py" for index in range(6)],
        ],
    )

    assert withheld["config/deploy.pem"] == "sensitive_path"
    assert withheld["blob.bin"] == "unreadable"
    assert withheld["huge.txt"] == "unreadable"
    assert withheld["secrets_lookalike.js"] == "key_material"
    assert withheld["filler_4.py"] == "evidence_budget"
    assert withheld["filler_5.py"] == "evidence_budget"
    quoted = {path: (content, truncated) for path, content, truncated in contents}
    assert "long.py" in quoted
    assert quoted["long.py"][1] is True
    assert quoted["long.py"][0].endswith("[truncated for size: review the visible portion only]")
    for index in range(4):
        assert quoted[f"filler_{index}.py"][1] is False


@pytest.mark.asyncio
async def test_the_review_is_quoted_the_completion_view_not_the_attempts_keyhole(
    tmp_path: Path,
) -> None:
    """A remediation attempt's self-review judges the candidate the artifact claims (76- A)."""
    _write(tmp_path, "src/routes.py", "route = 1\n")
    _write(tmp_path, "src/middleware.py", "middleware = 1\n")
    # A lineage entry whose file a reset discarded contributes nothing.
    prior = _prior_completion(["src/routes.py", "src/middleware.py", "src/discarded.py"])
    state = agent_state(tmp_path, [task_plan_artifact(), prior, review_artifact()])
    state["retry_count"] = 1
    client = ScriptedTextClient([_review_payload([])])

    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 2\n"}),
        self_reviewer=CodingRoleSelfReviewer(client),
    ).run(state)

    completion = update["artifacts"][0]
    review_input = json.loads(client.calls[0][1])
    # This attempt's own write first, then the prior lineage; the discarded file nowhere.
    assert review_input["changed_paths"][0] == "src/service.py"
    assert set(review_input["changed_paths"]) == {
        "src/service.py",
        "src/routes.py",
        "src/middleware.py",
    }
    # The judged set is the candidate the completion artifact reports.
    assert set(review_input["changed_paths"]) == {change.path for change in completion.file_changes}
    assert {entry["path"] for entry in review_input["changed_files"]} == set(
        review_input["changed_paths"]
    )
    assert review_input["withheld_paths"] == []
    assert review_input["withheld_reasons"] == {}


@pytest.mark.asyncio
async def test_a_finding_blind_to_declared_drops_is_a_limitation_not_a_rejection(
    tmp_path: Path,
) -> None:
    """The 207 replay (76- D): the verbatim blind finding is demoted, twice, and no attempt dies.

    Attempt one: the blind substantive finding rides beside a genuine localized one --
    the localized correction runs, the attempt completes, and the substantive sentence
    exists nowhere an engineer, a ledger, or a repeat rule could read it. Attempt two:
    the same blind finding alone -- the record is clean-with-limitations. Two consecutive
    blind attempts therefore put no sentence anywhere a convergence guard could count,
    which is the charter's 49-C clause answered one mechanism earlier.
    """
    fillers = [f"server/fillers/filler_{index}.js" for index in range(5)]
    for path in fillers:
        _write(tmp_path, path, "f" * 11_900)
    for path in _207_PATHS:
        _write(tmp_path, path, "n" * 11_900)
    prior = _prior_completion([*fillers, *_207_PATHS])
    state = agent_state(tmp_path, [task_plan_artifact(), prior, review_artifact()])
    state["retry_count"] = 1
    blind_finding = SelfReviewFinding(
        description=_207_FINDING, classification="substantive", paths=_207_PATHS
    )
    localized_finding = SelfReviewFinding(
        description=(
            "appDetailsSchema accepts the hard-coded value 'Google Play Store' in addition "
            "to the three configured STORES values."
        ),
        classification="localized",
        paths=("server/validation/app.validation.js",),
    )
    reviewer = StubSelfReviewer(_assessment(blind_finding, localized_finding))
    executor = QueuedCodingExecutor(
        [
            {"server/validation/app.validation.js": "schema = 1\n"},
            {"server/validation/app.validation.js": "schema = 2\n"},
        ]
    )

    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        self_reviewer=reviewer,
    ).run(state)

    completion = update["artifacts"][0]
    assert completion.completion_status == "completed"
    record = completion.metadata["self_review"]
    assert record["outcome"] == "corrected"
    # The blind finding is a declared limitation, with every named path's reason.
    assert [item["description"] for item in record["evidence_blind_findings"]] == [_207_FINDING]
    for path in _207_PATHS:
        assert record["withheld_reasons"][path] == "evidence_budget"
    # The genuine localized finding was corrected; the blind sentence reached no prompt.
    assert record["corrections_applied"] == ["server/validation/app.validation.js"]
    assert "(localized)" in executor.calls[1]
    assert "(substantive)" not in executor.calls[1]
    assert _207_FINDING not in executor.calls[1]

    # Attempt two: the same deterministic blindness, alone, on the grown lineage.
    second_state = agent_state(tmp_path, [task_plan_artifact(), completion, review_artifact()])
    second_state["retry_count"] = 2
    second_update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=QueuedCodingExecutor(
            [{"server/validation/app.validation.js": "schema = 3\n"}]
        ),
        self_reviewer=StubSelfReviewer(_assessment(blind_finding)),
    ).run(second_state)

    second_completion = second_update["artifacts"][0]
    assert second_completion.completion_status == "completed"
    second_record = second_completion.metadata["self_review"]
    # `inconclusive`, not `clean` (85- D): every changed production file was withheld from
    # this pass's own evidence, so its silence is a fact about the budget and not about the
    # work. It is still the `clean` code path -- the attempt completed, and the demotion
    # this test exists for is unchanged.
    assert second_record["outcome"] == "inconclusive"
    # The production subset only -- the suite among 207's paths is not blindness about the
    # work, and the fillers were shown.
    assert second_record["inconclusive_paths"] == sorted(
        path for path in _207_PATHS if classify_file_change(path) == "production"
    )
    assert "server/tests/bulk-create-apps.test.js" not in second_record["inconclusive_paths"]
    assert [item["description"] for item in second_record["evidence_blind_findings"]] == [
        _207_FINDING
    ]


@pytest.mark.asyncio
async def test_a_substantive_finding_about_an_unwritten_file_still_rejects(
    tmp_path: Path,
) -> None:
    """Unwritten is not unseen: a file no attempt created is a judgement, not blindness."""
    reviewer = StubSelfReviewer(
        _assessment(
            SelfReviewFinding(
                description="The route module was never created.",
                classification="substantive",
                paths=("src/never_written.py",),
            )
        )
    )

    with pytest.raises(SourceValidationError) as rejected:
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
            self_reviewer=reviewer,
        ).run(agent_state(tmp_path, [task_plan_artifact()]))

    assert rejected.value.terminal_outcome == SELF_REVIEW_SUBSTANTIVE_OUTCOME


@pytest.mark.asyncio
async def test_a_localized_finding_on_a_withheld_file_is_still_corrected(
    tmp_path: Path,
) -> None:
    """Localized findings are never demoted: the corrector reads the workspace itself."""
    _write(tmp_path, "config/deploy.pem", "value")
    prior = _prior_completion(["config/deploy.pem"])
    state = agent_state(tmp_path, [task_plan_artifact(), prior, review_artifact()])
    state["retry_count"] = 1
    reviewer = StubSelfReviewer(
        _assessment(
            SelfReviewFinding(
                description="The deploy configuration misses the rollback entry.",
                classification="localized",
                paths=("config/deploy.pem",),
            )
        )
    )
    executor = QueuedCodingExecutor(
        [
            {"src/service.py": "value = 1\n"},
            {"src/service.py": "value = 1\nrollback = True\n"},
        ]
    )

    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        self_reviewer=reviewer,
    ).run(state)

    completion = update["artifacts"][0]
    record = completion.metadata["self_review"]
    assert record["outcome"] == "corrected"
    assert record["withheld_reasons"]["config/deploy.pem"] == "sensitive_path"
    assert "evidence_blind_findings" not in record


# --------------------------------------------------------------------------------------
# 85- D: a self-review that could not see the file does not say "clean"
# --------------------------------------------------------------------------------------

# One character over `_SELF_REVIEW_FILE_MAX_CHARACTERS`, so the file is quoted as a window
# and named in `truncated_paths` -- 218's frontend attempt 2 exactly.
_OVERSIZED = "n" * 12_001


@pytest.mark.asyncio
async def test_a_blind_pass_over_a_changed_production_file_is_inconclusive(
    tmp_path: Path,
) -> None:
    """AB-Feature-218's frontend attempt 2 -- approved, pushed, two production blockers.

    It recorded `truncated_paths: ["src/pages/AllApps.js"]` -- the file the whole attempt
    was about -- and returned `outcome: "clean"`, `findings: []`. The backend's attempt 4
    did the same, on a summary that said in its own words "some service internals inferred
    from tests because the quoted bulk service file is truncated".

    76- stopped a blind FINDING from counting as a finding. Nothing stopped a blind PASS
    from counting as a pass, and a verdict may not be stronger than its evidence in either
    direction.
    """
    blinded = "src/pages/AllApps.js"

    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={blinded: _OVERSIZED}),
        self_reviewer=StubSelfReviewer(_assessment()),
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    record = completion.metadata["self_review"]

    assert record["truncated_paths"] == [blinded]
    assert record["outcome"] == "inconclusive"
    assert record["inconclusive_paths"] == [blinded]
    # It withholds an assurance and never adds a demand: the attempt completes exactly as a
    # `clean` one does, with no finding, no correction and nothing for a guard to count.
    assert completion.completion_status == "completed"
    assert record["findings"] == []
    assert record["corrections_applied"] == []
    assert "evidence_blind_findings" not in record


@pytest.mark.asyncio
async def test_a_pass_that_saw_everything_is_still_clean(tmp_path: Path) -> None:
    """No finding and nothing unseen is byte-identical to today."""
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        self_reviewer=StubSelfReviewer(_assessment()),
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    record = update["artifacts"][0].metadata["self_review"]

    assert record["truncated_paths"] == []
    assert record["outcome"] == "clean"
    assert "inconclusive_paths" not in record


@pytest.mark.asyncio
async def test_only_a_changed_production_file_makes_a_pass_inconclusive(tmp_path: Path) -> None:
    """A truncated documentation file is not blindness about the work.

    The intersection is with the change's own PRODUCTION files, exactly as the item states.
    Widening it to every unseen path would make this a noise generator, which is the one
    thing an assurance-withholding rule must not become -- 218's own README was rewritten on
    four separate attempts.
    """
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"README.md": _OVERSIZED}),
        self_reviewer=StubSelfReviewer(_assessment()),
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    record = update["artifacts"][0].metadata["self_review"]

    assert record["truncated_paths"] == ["README.md"]
    assert record["outcome"] == "clean"


@pytest.mark.asyncio
async def test_an_inconclusive_outcome_reaches_no_blocking_issue_or_ledger(
    tmp_path: Path,
) -> None:
    """The non-negotiable constraint: `inconclusive` takes exactly the `clean` code path.

    An `inconclusive` self-review that could fail an attempt would be the AB-Feature-207
    defect with the sign flipped, and 76- exists because that already happened once. So no
    diagnostic is composed, nothing is raised, and the attempt's own record carries no
    sentence a convergence guard, a resolved-issue ledger or a retry strategy could read.
    """
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={"server/services/app/bulkApp.service.js": _OVERSIZED}
        ),
        self_reviewer=StubSelfReviewer(_assessment()),
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    record = completion.metadata["self_review"]

    assert record["outcome"] == "inconclusive"
    assert completion.completion_status == "completed"
    # Nothing that ends an attempt, and nothing a later rule counts.
    assert completion.metadata.get("blocking_issues") in (None, [])
    assert completion.metadata.get("retry_strategy") in (None, {})
    assert "self_review_rejected" not in completion.metadata
    assert SELF_REVIEW_DIAGNOSTIC_PREFIX not in json.dumps(completion.metadata)
