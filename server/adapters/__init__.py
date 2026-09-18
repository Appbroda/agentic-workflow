"""Protocol-backed external service adapters for GitHub, Git, LLMs, and persistence."""

from adapters.git_adapter import (
    GitAdapterError,
    GitPushResult,
    GitPythonService,
    GitSafetyError,
    GitService,
    MockGitService,
)
from adapters.github_adapter import (
    GitHubAdapterError,
    GitHubService,
    MockGitHubService,
    PullRequestDetails,
    PyGithubService,
)
from adapters.llm_adapter import (
    CodingExecutionResult,
    CodingExecutor,
    LLMAdapterError,
    LLMClient,
    LLMResponse,
    MockCodingExecutor,
    OpenAILLMClient,
    ResponsesCodingExecutor,
)

__all__ = [
    "CodingExecutionResult",
    "CodingExecutor",
    "GitAdapterError",
    "GitHubAdapterError",
    "GitHubService",
    "GitPushResult",
    "GitPythonService",
    "GitSafetyError",
    "GitService",
    "LLMAdapterError",
    "LLMClient",
    "LLMResponse",
    "MockCodingExecutor",
    "MockGitHubService",
    "MockGitService",
    "OpenAILLMClient",
    "PullRequestDetails",
    "PyGithubService",
    "ResponsesCodingExecutor",
]
