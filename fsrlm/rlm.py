"""
RLM - Recursive Language Model abstraction.

This is the main interface for fsRLM. Use it like you would use an LLM,
but internally it spins up a full agent loop with recursive scripting.

Usage:
    from fsrlm import RLM

    # Simple string prompt
    rlm = RLM()
    result = rlm.invoke("Your complex prompt here...")

    # Or with messages (like Anthropic API)
    result = rlm.invoke(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Analyze this data..."},
    ])

    print(result.answer)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Callable, Any, Union, TypedDict, List, Dict
from dataclasses import dataclass, field

from .workspace import Workspace, WorkspaceConfig
from .runner import AgentRunner, RunnerConfig


# Type definitions for messages (matches Anthropic API)
class TextBlock(TypedDict, total=False):
    type: str  # "text"
    text: str


class ImageBlock(TypedDict, total=False):
    type: str  # "image"
    source: dict


ContentBlock = Union[str, TextBlock, ImageBlock, dict]


class Message(TypedDict, total=False):
    role: str  # "user" | "assistant"
    content: Union[str, list[ContentBlock]]


Messages = list[Message]


@dataclass
class RLMResult:
    """Result from an RLM invocation."""

    answer: Optional[str]
    """The final answer from output/answer.md, or None if not produced."""

    metrics: dict
    """Metrics from the run (subcalls, tokens, cache hits, etc.)."""

    errors: list[dict]
    """Any errors logged during the run."""

    artifacts: list[dict]
    """Artifacts collected during processing (intermediate results, extracted data)."""

    success: bool
    """Whether the run produced an answer."""

    workspace_path: Optional[Path] = None
    """Path to workspace (if preserve_workspace=True)."""


@dataclass
class RLMConfig:
    """Configuration for the RLM."""

    # Subcall settings (for recursive Claude calls in scripts)
    max_tokens_per_subcall: int = 1000
    max_depth: int = 5
    max_scripts: int = 20
    max_subcalls_per_script: int = 25
    submodel: str = "claude-haiku-4-20250414"

    # Root agent settings
    root_model: str = "claude-sonnet-4-20250514"
    max_turns: int = 50

    # Bedrock settings
    use_bedrock: bool = False
    bedrock_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    bedrock_submodel: str = "us.anthropic.claude-haiku-4-20250414-v1:0"

    # Workspace settings
    workspace_dir: Optional[Path] = None  # None = temp directory
    preserve_workspace: bool = False  # Keep workspace after run

    # Caching
    cache_responses: bool = True


class RLM:
    """
    Recursive Language Model abstraction.

    Invoke it like an LLM:
        rlm = RLM()
        result = rlm.invoke("Complex task...")

    Internally it:
    1. Creates/clears a structured workspace
    2. Writes your prompt to input/prompt.md
    3. Runs the Claude Agent SDK
    4. Agent writes scripts that make recursive Claude calls
    5. Returns the answer from output/answer.md
    """

    def __init__(
        self,
        config: Optional[RLMConfig] = None,
        on_message: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize the RLM.

        Args:
            config: RLM configuration
            on_message: Callback for streaming agent messages
        """
        self.config = config or RLMConfig()
        self.on_message = on_message
        self._workspace: Optional[Workspace] = None

    def _ensure_workspace(self) -> Workspace:
        """Get or create the workspace."""
        if self._workspace is None or not self._workspace.root.exists():
            workspace_config = WorkspaceConfig(
                max_tokens_per_call=self.config.max_tokens_per_subcall,
                max_depth=self.config.max_depth,
                max_scripts=self.config.max_scripts,
                max_subcalls_per_script=self.config.max_subcalls_per_script,
                submodel=self.config.submodel,
                cache_responses=self.config.cache_responses,
                use_bedrock=self.config.use_bedrock,
                bedrock_submodel=self.config.bedrock_submodel,
            )
            self._workspace = Workspace.create(
                root=self.config.workspace_dir,
                config=workspace_config,
            )
        return self._workspace

    def invoke(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Messages] = None,
        system: Optional[str] = None,
        attachments: Optional[dict[str, bytes]] = None,
    ) -> RLMResult:
        """
        Invoke the RLM on a prompt or conversation.

        Can be called with either a simple prompt string or a messages list
        (like the Anthropic API).

        Args:
            prompt: Simple string prompt (mutually exclusive with messages)
            messages: List of message dicts with 'role' and 'content' keys,
                      following Anthropic API format.
            system: Optional system prompt (used with messages)
            attachments: Optional dict of filename -> bytes for attachments

        Returns:
            RLMResult with answer, metrics, errors, artifacts

        Examples:
            # Simple prompt
            result = rlm.invoke("Analyze this data...")

            # With messages
            result = rlm.invoke(messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "What is 2+2?"},
            ])

            # With system prompt
            result = rlm.invoke(
                messages=[{"role": "user", "content": "Explain recursion"}],
                system="You are a computer science tutor."
            )
        """
        if prompt is None and messages is None:
            raise ValueError("Must provide either 'prompt' or 'messages'")
        if prompt is not None and messages is not None:
            raise ValueError("Cannot provide both 'prompt' and 'messages'")

        # Set up workspace
        workspace = self._ensure_workspace()
        workspace.clear()  # Clear between invocations

        # Write input
        if messages is not None:
            workspace.set_messages(messages, system=system)
        else:
            workspace.set_prompt(prompt)

        # Add attachments
        if attachments:
            for name, content in attachments.items():
                workspace.add_attachment(name, content)

        # Set up runner
        runner_config = RunnerConfig(
            model=self.config.root_model,
            max_turns=self.config.max_turns,
            use_bedrock=self.config.use_bedrock,
            bedrock_model=self.config.bedrock_model,
        )
        runner = AgentRunner(
            workspace_path=workspace.root,
            config=runner_config,
            on_message=self.on_message,
        )

        # Run
        result = runner.run()

        # Build result
        rlm_result = RLMResult(
            answer=result.get("answer"),
            metrics=result.get("metrics", {}),
            errors=result.get("errors", []),
            artifacts=workspace.get_artifacts(),
            success=result.get("answer") is not None,
            workspace_path=workspace.root if self.config.preserve_workspace else None,
        )

        # Cleanup if not preserving
        if not self.config.preserve_workspace and workspace._is_temp:
            workspace.destroy()
            self._workspace = None

        return rlm_result

    def __call__(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Messages] = None,
        **kwargs,
    ) -> RLMResult:
        """Allow calling RLM instance directly."""
        return self.invoke(prompt, messages=messages, **kwargs)


# Convenience functions
def invoke(
    prompt: Optional[str] = None,
    *,
    messages: Optional[Messages] = None,
    system: Optional[str] = None,
    max_tokens: int = 1000,
    max_depth: int = 5,
    model: str = "claude-sonnet-4-20250514",
    submodel: str = "claude-haiku-4-20250414",
) -> RLMResult:
    """
    One-shot RLM invocation.

    Examples:
        from fsrlm import invoke

        # Simple prompt
        result = invoke("Complex analysis task...")
        print(result.answer)

        # With messages (like Anthropic API)
        result = invoke(messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "What is 2+2?"},
        ])

        # With system prompt
        result = invoke(
            messages=[{"role": "user", "content": "Explain recursion"}],
            system="You are a helpful tutor."
        )
    """
    config = RLMConfig(
        max_tokens_per_subcall=max_tokens,
        max_depth=max_depth,
        root_model=model,
        submodel=submodel,
    )
    rlm = RLM(config=config)
    return rlm.invoke(prompt, messages=messages, system=system)
