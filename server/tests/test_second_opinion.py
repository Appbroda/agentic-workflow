"""The second opinion chooses its own evidence (75-).

AB-Feature-206's approved PRs hid one blocker and two majors that fresh-context reviewers --
free to read the checkout, told nothing about the requirement or the retry history -- found in
minutes. The coder, its self-review, and the reviewer share context selection, so they share
blind spots; a fourth platform-chosen bundle would too. The pass tested here runs only for an
approved channel-touching or large change, lets the second reviewer ask for the files it wants
(one ask, one read), merges what it finds into the primary review under the existing
authorities, and records the escaped-defect count the live matrix reads.

Repo-agnostic like everything else: the one channel package named below is read out of the
per-ecosystem tables in `tools.channel_packages`, never spelled out.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import ImageInput, LLMResponse
from agents.reviewer.agent import (
    _SECOND_OPINION_MIN_CHARACTERS,
    _SECOND_OPINION_TEMPLATE,
    ReviewerAgent,
    _second_opinion_payload,
)
from agents.shared.contracts import AgentArtifactError
from artifacts.schemas import FileChange, ReviewArtifact
from prompts.prompt_loader import PromptLoader
from services.feature_runtime import _resolve_review_outcome
from tests.test_agents import (
    StaticLLMClient,
    StaticValidationTool,
    agent_state,
    code_completion_artifact,
    domain_payload,
    review_artifact,
    task_plan_artifact,
    technical_prd_artifact,
    validation_result,
)
from tests.test_bounded_review_scope import _finding, _review
from tests.test_overruled_review_demand import _settled
from tools.channel_packages import CHANNEL_ECOSYSTEMS
from tools.review_fix_classification import fingerprint_for_text

_NODE = next(item for item in CHANNEL_ECOSYSTEMS if ".js" in item.source_suffixes)
_NODE_PACKAGE = sorted(name for name in _NODE.packages if re.fullmatch(r"[a-z]\w*", name))[0]

_PLAIN_SOURCE = "export const format = (value) => String(value);\n"


def _channel_source() -> str:
    """The 206 shape: changed code importing a channel package bare."""
    return (
        f"import client from '{_NODE_PACKAGE}';\n"
        "export const upload = (payload) => client.post('/upload', payload);\n"
    )


class SequencedLLMClient:
    """A protocol double whose Nth call returns the Nth configured payload, then refuses."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
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
        """Record the inputs and return the next configured structured response."""
        index = len(self.calls)
        self.calls.append((instructions, input_text))
        if index >= len(self._payloads):
            msg = f"unexpected model call number {index + 1}"
            raise AssertionError(msg)
        return LLMResponse(
            response_id=f"mock-response-{index}",
            model="mock-review-model",
            output_text=json.dumps(self._payloads[index]),
            input_tokens=10,
            output_tokens=5,
        )


def _reviewer(client: Any) -> ReviewerAgent:
    """One reviewer with passing validation, the way the agent unit tests build one."""
    return ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    )


def _write(workspace: Path, files: dict[str, str]) -> Any:
    """Write the changed files into the workspace and report them as the change."""
    for relative_path, content in files.items():
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return code_completion_artifact().model_copy(
        update={
            "file_changes": [
                FileChange(path=path, change_type="added", description=f"Add {path}")
                for path in files
            ]
        }
    )


def _state(workspace: Path, completion: Any, extra: list[Any] | None = None) -> Any:
    return agent_state(
        workspace,
        [technical_prd_artifact(), task_plan_artifact(), completion, *(extra or [])],
    )


def _so_response(
    *,
    requested: list[str] | None = None,
    findings: list[dict[str, Any]] | None = None,
    summary: str = "Examined the change and the seams it stands on.",
) -> dict[str, Any]:
    return {
        "requested_paths": requested or [],
        "findings": findings or [],
        "summary": summary,
    }


def _so_finding(
    finding_id: str, *, severity: str = "high", description: str | None = None
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "severity": severity,
        "title": f"Escaped defect {finding_id}",
        "description": description or f"The retry path can reach the effect in {finding_id} twice.",
        "recommendation": f"Guard {finding_id} with an idempotency key.",
        "file_path": "src/features/upload.js",
        "line_number": 2,
        "finding_category": "code_quality",
    }


# --------------------------------------------------------------------------------------
# T1-T4 -- the gate
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ungated_change_pays_nothing_and_still_records_the_gate(tmp_path: Path) -> None:
    """Small, channel-free, approved: one model call, and the measured numbers on record."""
    completion = _write(tmp_path, {"src/util/format.js": _PLAIN_SOURCE})
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await _reviewer(client).run(_state(tmp_path, completion))

    assert len(client.calls) == 1
    review = update["artifacts"][0]
    record = review.metadata["second_opinion"]
    assert record["ran"] is False
    assert record["reason"] == "gate"
    assert record["gate"]["channel_packages"] == []
    assert record["gate"]["changed_file_count"] == 1
    assert record["gate"]["changed_source_characters"] == len(_PLAIN_SOURCE)
    assert review.verdict == "approved"


@pytest.mark.asyncio
async def test_the_channel_arm_fires_from_the_one_scan(tmp_path: Path) -> None:
    """A channel-touching approved change runs the pass; the gate reads the seam scan."""
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    client = SequencedLLMClient([domain_payload(review_artifact()), _so_response()])

    update = await _reviewer(client).run(_state(tmp_path, completion))

    assert len(client.calls) == 2
    review = update["artifacts"][0]
    record = review.metadata["second_opinion"]
    assert record["ran"] is True
    assert record["gate"]["reason"] == "channel"
    # The same computation, not a re-scan: the gate's packages are the artifact's seam facts.
    assert record["gate"]["channel_packages"] == review.metadata["channel_seam_packages"]
    assert record["escaped_blocking"] == 0
    assert record["escaped_advisory"] == 0
    assert record["prompt_template"] == _SECOND_OPINION_TEMPLATE
    # A clean second opinion changes nothing (T8).
    assert review.verdict == "approved"


@pytest.mark.asyncio
async def test_the_size_arm_fires_without_a_channel(tmp_path: Path) -> None:
    """Bulk alone gates the pass: many fully-shown files crossing the character threshold."""
    line = "export const value_%04d = 'x';\n"
    body = "".join(line % index for index in range(600))
    files = {f"src/generated/part{index}.js": body for index in range(4)}
    assert sum(len(content) for content in files.values()) >= _SECOND_OPINION_MIN_CHARACTERS
    completion = _write(tmp_path, files)
    client = SequencedLLMClient([domain_payload(review_artifact()), _so_response()])

    update = await _reviewer(client).run(_state(tmp_path, completion))

    record = update["artifacts"][0].metadata["second_opinion"]
    assert record["ran"] is True
    assert record["gate"]["reason"] == "size"
    assert record["gate"]["channel_packages"] == []


@pytest.mark.asyncio
async def test_a_round_already_going_back_gets_no_second_opinion(tmp_path: Path) -> None:
    """A rejection whose blockers all cite a file is on its way to a retry that can act on it.

    The next attempt, the retry planner and a person reading the diff can each judge a finding
    that names its file. Nothing is gained by a second read, so the pass stays off.
    """
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    rejected = domain_payload(review_artifact())
    rejected["verdict"] = "changes_requested"
    rejected["findings"] = [_so_finding("FILED-1", severity="high")]
    client = StaticLLMClient(rejected)

    update = await _reviewer(client).run(_state(tmp_path, completion))

    assert len(client.calls) == 1
    record = update["artifacts"][0].metadata["second_opinion"]
    assert record["ran"] is False
    assert record["reason"] == "verdict"


@pytest.mark.asyncio
async def test_a_blocker_that_names_no_file_is_read_a_second_time(tmp_path: Path) -> None:
    """The one shape nothing downstream can check, and the one nobody ever looked at twice.

    A blocking `validation_failure` that cites no file cannot be judged against a diff: it can
    only be judged against the validation plan, and the demand it makes may be one no attempt
    is able to satisfy. AB-Feature-216 was stopped by two such findings that were both false;
    AB-Feature-225 by one the plan had made impossible. The pass that picks its own evidence
    was gated to approvals, so in both cases the false blocker was never re-read.

    It runs to measure, never to overturn -- the rejection stands either way.
    """
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    primary = domain_payload(review_artifact())
    primary["verdict"] = "changes_requested"
    primary["findings"] = [
        {
            **_so_finding("PLAYWRIGHT-NOT-RUN", severity="high"),
            "file_path": None,
            "line_number": None,
            "finding_category": "validation_failure",
        }
    ]
    client = SequencedLLMClient([primary, _so_response(findings=[])])

    update = await _reviewer(client).run(_state(tmp_path, completion))

    review = update["artifacts"][0]
    record = review.metadata["second_opinion"]
    assert record["ran"] is True
    assert record["gate"]["reason"] == "unfiled_blocker"
    assert record["gate"]["unfiled_blocking_finding_ids"] == ["PLAYWRIGHT-NOT-RUN"]
    # The independent read did not reproduce it, and that is now on the record.
    assert record["unsubstantiated_blocking_finding_ids"] == ["PLAYWRIGHT-NOT-RUN"]
    # And the primary's judgement is untouched: measuring is not overruling.
    assert review.verdict == "changes_requested"
    assert "PLAYWRIGHT-NOT-RUN" in [item.finding_id for item in review.findings]


# --------------------------------------------------------------------------------------
# T5 -- the brief withholds the requirement, the report, and the history
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_second_opinion_never_sees_requirement_report_or_history(
    tmp_path: Path,
) -> None:
    """The withheld documents are the shared blind spot; sentinels prove the separation."""
    requirement_sentinel = "REQUIREMENT-SENTINEL-8f3a"
    report_sentinel = "REPORT-SENTINEL-2c9d"
    history_sentinel = "HISTORY-SENTINEL-6b1e"
    prd = technical_prd_artifact().model_copy(
        update={"solution_summary": f"Use strict artifacts. {requirement_sentinel}"}
    )
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()}).model_copy(
        update={"summary": f"Implemented the endpoint. {report_sentinel}"}
    )
    prior = review_artifact().model_copy(
        update={
            "verdict": "changes_requested",
            "findings": [
                _finding("PRIOR-1").model_copy(
                    update={"description": f"A prior blocking demand. {history_sentinel}"}
                )
            ],
        }
    )
    client = SequencedLLMClient([domain_payload(review_artifact()), _so_response()])

    await _reviewer(client).run(
        agent_state(tmp_path, [prd, task_plan_artifact(), completion, prior])
    )

    assert len(client.calls) == 2
    primary_instructions, primary_input = client.calls[0]
    assert requirement_sentinel in primary_instructions + primary_input
    assert history_sentinel in primary_input
    second_instructions, second_input = client.calls[1]
    for sentinel in (requirement_sentinel, report_sentinel, history_sentinel):
        assert sentinel not in second_instructions
        assert sentinel not in second_input
    # What it does get: the change's own source and the checkout inventory.
    second_payload = json.loads(second_input)
    assert second_payload["turn"] == "ask"
    shown_paths = [item["path"] for item in second_payload["change_evidence"]["files"]]
    assert "src/features/upload.js" in shown_paths
    assert "src/features/upload.js" in second_payload["repository_inventory"]["paths"]


# --------------------------------------------------------------------------------------
# T6 -- ask-then-read: bounded, declared, and exactly one read round
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requested_files_are_shown_or_declared_withheld(tmp_path: Path) -> None:
    """Every asked-for path gets an answer: content, or the reason it was withheld."""
    (tmp_path / ".env").write_text("API_TOKEN=live-value\n", encoding="utf-8")
    (tmp_path / "src" / "notes").mkdir(parents=True)
    (tmp_path / "src" / "notes" / "deploy.txt").write_text(
        "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "lib" / "helper.js").write_text(_PLAIN_SOURCE, encoding="utf-8")
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    requested = ["src/lib/helper.js", ".env", "src/notes/deploy.txt", "missing/nope.js"]
    client = SequencedLLMClient(
        [
            domain_payload(review_artifact()),
            _so_response(requested=requested),
            # The read turn asks for more: it is ignored -- one read round, no third fetch.
            _so_response(requested=["src/lib/again.js"]),
        ]
    )

    update = await _reviewer(client).run(_state(tmp_path, completion))

    assert len(client.calls) == 3
    read_payload = json.loads(client.calls[2][1])
    assert read_payload["turn"] == "read"
    entries = {item["path"]: item for item in read_payload["requested_files"]}
    assert entries["src/lib/helper.js"]["content"] == _PLAIN_SOURCE
    assert entries[".env"]["withheld_reason"] == "sensitive_path"
    assert entries["src/notes/deploy.txt"]["withheld_reason"] == "key_material"
    assert entries["missing/nope.js"]["withheld_reason"] == "outside_inventory"
    assert "live-value" not in client.calls[2][1]
    assert "BEGIN RSA" not in client.calls[2][1]
    record = update["artifacts"][0].metadata["second_opinion"]
    assert record["requested_paths"] == requested
    assert record["shown_paths"] == ["src/lib/helper.js"]
    assert {item["path"]: item["reason"] for item in record["withheld"]} == {
        ".env": "sensitive_path",
        "src/notes/deploy.txt": "key_material",
        "missing/nope.js": "outside_inventory",
    }


# --------------------------------------------------------------------------------------
# T7 + T10 -- the merge, the demotion, and the round-trip replay depends on
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_escaped_blocker_demotes_the_verdict_and_is_counted(tmp_path: Path) -> None:
    """Merged findings are the escaped-defect measurement; a blocker ends the approval."""
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    primary = domain_payload(review_artifact())
    duplicate_description = "The response omits the pagination cursor on the last page."
    primary["findings"] = [
        {**_so_finding("LOW-1", severity="low", description=duplicate_description)}
    ]
    client = SequencedLLMClient(
        [
            primary,
            _so_response(
                findings=[
                    _so_finding("SEAM-1", severity="high"),
                    _so_finding("EDGE-1", severity="medium"),
                    # A restatement of what the primary already saw: dropped, counted.
                    _so_finding("DUP-1", severity="medium", description=duplicate_description),
                ]
            ),
        ]
    )

    update = await _reviewer(client).run(_state(tmp_path, completion))

    review = update["artifacts"][0]
    assert review.verdict == "changes_requested"
    finding_ids = [item.finding_id for item in review.findings]
    assert "SO-SEAM-1" in finding_ids
    assert "SO-EDGE-1" in finding_ids
    assert "SO-DUP-1" not in finding_ids
    record = review.metadata["second_opinion"]
    assert record["escaped_blocking"] == 1
    assert record["escaped_advisory"] == 1
    assert record["duplicate_count"] == 1
    assert record["escaped_finding_ids"] == ["SO-SEAM-1", "SO-EDGE-1"]
    # Merged findings are model judgement, never measurement: the deterministic census
    # must not adopt them (75- guard 7).
    assert not any(item.startswith("SO-") for item in review.metadata["deterministic_finding_ids"])
    # T10: the journaled payload is this artifact's dump; replay revalidates it through the
    # same schema, so the merged artifact must round-trip with its validators.
    restored = ReviewArtifact.model_validate_json(json.dumps(review.model_dump(mode="json")))
    assert restored.verdict == "changes_requested"
    assert [item.finding_id for item in restored.findings] == finding_ids
    assert restored.metadata["second_opinion"]["escaped_blocking"] == 1


# --------------------------------------------------------------------------------------
# T9 -- a malfunction degrades, never dies (77-, item 34)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_malformed_second_opinion_degrades_after_its_one_repair(tmp_path: Path) -> None:
    """Two malformed responses cost the platform its second opinion, never the attempt.

    The record is not silent: `ran: false` with `malformed_response` reports exactly the
    coverage the pass had -- none -- and the review stands on the primary verdict alone.
    AB-Feature-208's frontend attempt 1 died `platform_defect` to the old behaviour.
    """
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    malformed = {"requested_paths": "not-a-list"}
    client = SequencedLLMClient([domain_payload(review_artifact()), malformed, malformed])

    update = await _reviewer(client).run(_state(tmp_path, completion))

    assert len(client.calls) == 3
    review = update["artifacts"][0]
    assert review.verdict == "approved"
    record = review.metadata["second_opinion"]
    assert record["ran"] is False
    assert record["reason"] == "malformed_response"
    assert record["gate"]["reason"] == "channel"
    # Both spent calls are on the audit trail even though the pass did not run.
    assert record["response_ids"] == ["mock-response-1", "mock-response-2"]


@pytest.mark.asyncio
async def test_a_schema_rejected_finding_degrades_instead_of_killing_the_attempt(
    tmp_path: Path,
) -> None:
    """The 208 shape end to end: a finding pydantic rejects, twice, and the review survives.

    `ReviewFinding.file_path` pipes through the workspace-bound path policy, so an absolute
    path raises `ValidationError` from the constructor -- the exact exception that escaped
    `_second_opinion_pass` and failed the workstream as `platform_defect`.
    """
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    bad = _so_response(findings=[{**_so_finding("SEAM-1"), "file_path": "/etc/passwd"}])
    client = SequencedLLMClient([domain_payload(review_artifact()), bad, bad])

    update = await _reviewer(client).run(_state(tmp_path, completion))

    review = update["artifacts"][0]
    assert review.verdict == "approved"
    assert review.metadata["second_opinion"]["ran"] is False
    assert review.metadata["second_opinion"]["reason"] == "malformed_response"
    # The repair prompt named the finding and the field instead of dumping a pydantic trace.
    repair_instructions = client.calls[2][0]
    assert "second-opinion finding 0" in repair_instructions
    assert "file_path" in repair_instructions


@pytest.mark.asyncio
async def test_a_schema_rejected_finding_is_repaired_when_the_repair_is_valid(
    tmp_path: Path,
) -> None:
    """The one-shot repair still fires through the new wrap, and its result is merged."""
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    bad = _so_response(findings=[{**_so_finding("SEAM-1"), "file_path": "/etc/passwd"}])
    good = _so_response(findings=[_so_finding("SEAM-1")])
    client = SequencedLLMClient([domain_payload(review_artifact()), bad, good])

    update = await _reviewer(client).run(_state(tmp_path, completion))

    review = update["artifacts"][0]
    record = review.metadata["second_opinion"]
    assert record["ran"] is True
    assert record["escaped_blocking"] == 1
    assert "SO-SEAM-1" in [item.finding_id for item in review.findings]


@pytest.mark.asyncio
async def test_the_read_turn_degrades_the_same_way(tmp_path: Path) -> None:
    """A malfunction after the ask turn degrades too; partial coverage is not coverage."""
    (tmp_path / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "lib" / "helper.js").write_text(_PLAIN_SOURCE, encoding="utf-8")
    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    bad = _so_response(findings=[{**_so_finding("SEAM-1"), "file_path": "/etc/passwd"}])
    client = SequencedLLMClient(
        [
            domain_payload(review_artifact()),
            _so_response(requested=["src/lib/helper.js"]),
            bad,
            bad,
        ]
    )

    update = await _reviewer(client).run(_state(tmp_path, completion))

    record = update["artifacts"][0].metadata["second_opinion"]
    assert record["ran"] is False
    assert record["reason"] == "malformed_response"
    assert len(record["response_ids"]) == 3


@pytest.mark.asyncio
async def test_a_provider_fault_is_not_converted_into_a_malformed_response(
    tmp_path: Path,
) -> None:
    """Transport and provider errors keep their own classification; only parse shapes degrade."""

    class FaultingClient(SequencedLLMClient):
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
            if len(self.calls) >= 1:
                self.calls.append((instructions, input_text))
                msg = "provider unavailable"
                raise ConnectionError(msg)
            return await super().respond(instructions=instructions, input_text=input_text)

    completion = _write(tmp_path, {"src/features/upload.js": _channel_source()})
    client = FaultingClient([domain_payload(review_artifact())])

    with pytest.raises(ConnectionError):
        await _reviewer(client).run(_state(tmp_path, completion))


def test_the_payload_names_the_finding_and_field_without_echoing_the_value() -> None:
    """Per-finding validation failures re-raise as the malformed-response type, safely."""
    payload = _so_response(findings=[{**_so_finding("SEAM-1"), "file_path": "/etc/passwd"}])

    with pytest.raises(AgentArtifactError) as raised:
        _second_opinion_payload(payload)

    message = str(raised.value)
    assert "second-opinion finding 0" in message
    assert "file_path" in message
    assert "/etc/passwd" not in message


# --------------------------------------------------------------------------------------
# T11 -- the loop treats a merged finding as the review's own
# --------------------------------------------------------------------------------------


def test_a_merged_finding_blocks_and_demotes_like_a_primary_one() -> None:
    """Same division, same 73- demotion: `SO-` is display identity, not an authority."""
    demand = "The bulk endpoint must hold a lock across the read-modify-write."
    finding = _finding("SO-SEAM-1").model_copy(update={"description": demand})

    blocking = _resolve_review_outcome(
        _review(findings=[finding]), bounded=False, publishable=True, settled=[]
    )
    assert blocking.blocking_issues == [demand]

    demoted = _resolve_review_outcome(
        _review(findings=[finding]),
        bounded=False,
        publishable=True,
        settled=[_settled(demand=demand)],
    )
    assert demoted.blocking_issues == []
    assert [item.finding_id for item in demoted.overruled_findings] == ["SO-SEAM-1"]
    assert demoted.overruled_findings[0].fingerprint == fingerprint_for_text(demand)
