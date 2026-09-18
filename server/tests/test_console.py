"""Coverage for the operator console page and its plain-language status vocabulary."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from api.console import (
    CHILD_STATUS_VOCABULARY,
    FEATURE_STATUS_VOCABULARY,
    status_vocabulary,
)
from main import create_app
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    """Serve the console from an application with authentication configured."""
    return TestClient(create_app(platform_api_key="test-platform-key"))


def test_every_feature_status_has_wording_a_non_engineer_can_act_on() -> None:
    """A lifecycle state with no entry would surface the raw enum to a product manager."""
    missing = set(FeatureWorkflowStatus) - set(FEATURE_STATUS_VOCABULARY)

    assert not missing, (
        f"feature states with no console wording: {sorted(item.value for item in missing)}"
    )
    for status, explanation in FEATURE_STATUS_VOCABULARY.items():
        assert explanation.headline.strip(), status
        assert explanation.detail.strip(), status
        assert explanation.next_step.strip(), status


def test_every_workstream_status_has_wording_a_non_engineer_can_act_on() -> None:
    """The same guarantee for the per-repository states shown under a feature."""
    missing = set(ChildWorkflowStatus) - set(CHILD_STATUS_VOCABULARY)

    assert not missing, (
        f"workstream states with no console wording: {sorted(item.value for item in missing)}"
    )
    for status, explanation in CHILD_STATUS_VOCABULARY.items():
        assert explanation.headline.strip(), status
        assert explanation.detail.strip(), status
        assert explanation.next_step.strip(), status


def test_the_partly_done_state_says_the_pull_requests_are_real() -> None:
    """`failed_requires_human` is the state that most misleads: work often did land."""
    explanation = FEATURE_STATUS_VOCABULARY[FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN]

    assert "pull request" in explanation.detail.lower()
    assert "pull request" in explanation.next_step.lower()
    assert explanation.tone == "attention"


def test_the_console_page_is_served_without_platform_authentication(client: TestClient) -> None:
    """A browser cannot send a Bearer header when navigating, so the page itself is public."""
    response = client.get("/console")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Feature console</title>" in response.text


def test_the_console_page_carries_no_credentials_and_no_remote_resources(
    client: TestClient,
) -> None:
    """The page is markup only. It must not embed a key, nor reach a third-party host."""
    page = client.get("/console").text

    assert "test-platform-key" not in page
    # A remote script or stylesheet would put the operator's pasted tokens one XSS away from
    # a third party, and would stop the console working on an isolated network.
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", page)
    # Credentials are deliberately kept in tab memory only. Matched as use rather than as a
    # word, so the comment in the page explaining the rule does not trip its own test.
    assert not re.search(r"\b(localStorage|sessionStorage)\s*[.\[]", page)
    assert not re.search(r"\bdocument\s*\.\s*cookie", page)


def test_the_vocabulary_endpoint_matches_the_module(client: TestClient) -> None:
    """The page must not invent wording; it reads exactly what the module defines."""
    response = client.get("/console/status-vocabulary")

    assert response.status_code == 200
    assert response.json() == status_vocabulary()


def test_the_console_does_not_expose_feature_data_without_a_platform_key(
    client: TestClient,
) -> None:
    """Serving the page publicly must not make the data behind it public too."""
    response = client.get("/features/feature-does-not-exist")

    assert response.status_code == 401


def test_no_status_promises_something_a_feature_in_it_might_not_have() -> None:
    """One status covers several situations, so its wording must be true of all of them.

    `failed_requires_human` is reached both by a feature whose repositories mostly finished
    and by one where every repository stopped. It used to say "Everything that passed its own
    review has a real draft pull request open" and "The pull requests below are real and worth
    reviewing" -- which feature -083 displayed directly above a pull-request tab reading "No
    pull requests yet".
    """
    promises = ("the pull requests below", "partly done", "everything that passed")
    for status, explanation in FEATURE_STATUS_VOCABULARY.items():
        wording = f"{explanation.headline} {explanation.detail} {explanation.next_step}".lower()
        for promise in promises:
            assert promise not in wording, (
                f"{status.value} asserts {promise!r}, which is not true of every feature "
                "that reaches this status"
            )
