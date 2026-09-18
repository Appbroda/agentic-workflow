"""Strict, versioned Jinja prompt-template rendering for workflow agents."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

DEFAULT_PROMPT_DIRECTORY = Path(__file__).resolve().parent


class PromptTemplateError(ValueError):
    """Raised when a requested template name is unsafe or unavailable."""


class PromptLoader:
    """Load Jinja templates solely from one configured prompt directory."""

    def __init__(self, prompt_directory: Path = DEFAULT_PROMPT_DIRECTORY) -> None:
        """Resolve the prompt directory and configure strict template rendering."""
        self.prompt_directory = prompt_directory.resolve(strict=True)
        if not self.prompt_directory.is_dir():
            msg = f"prompt directory is not a directory: {self.prompt_directory}"
            raise NotADirectoryError(msg)
        self.environment = Environment(
            loader=FileSystemLoader(self.prompt_directory),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )

    def render(self, template_name: str, **context: Any) -> str:
        """Render one versioned prompt template with explicitly supplied context values."""
        _validate_template_name(template_name)
        try:
            template = self.environment.get_template(template_name)
        except TemplateNotFound as error:
            msg = f"prompt template does not exist: {template_name}"
            raise PromptTemplateError(msg) from error
        return template.render(**context)

    def list_templates(self) -> list[str]:
        """Return all available Jinja templates in deterministic order."""
        return sorted(self.environment.list_templates(extensions=["jinja2"]))


def _validate_template_name(template_name: str) -> None:
    """Reject template paths that are absolute, empty, or traverse parent directories."""
    if not template_name or "\\" in template_name:
        msg = "template name must be a non-empty relative POSIX path"
        raise PromptTemplateError(msg)
    path = PurePosixPath(template_name)
    if path.is_absolute() or ".." in path.parts:
        msg = "template name must stay within the prompt directory"
        raise PromptTemplateError(msg)
