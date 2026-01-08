"""
Agent SDK runner for ccRLM.

This module wraps the Claude Agent SDK to run the root agent
in the workspace with proper configuration.
"""

from __future__ import annotations

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Callable, Any, Tuple
from dataclasses import dataclass, field

# Lazy import - SDK only needed at runtime
_sdk_imported = False
query = None
ClaudeAgentOptions = None


def _ensure_sdk():
    """Import SDK on first use."""
    global _sdk_imported, query, ClaudeAgentOptions
    if _sdk_imported:
        return
    try:
        from claude_agent_sdk import (
            query as _query,
            ClaudeAgentOptions as _ClaudeAgentOptions,
        )
        query = _query
        ClaudeAgentOptions = _ClaudeAgentOptions
        _sdk_imported = True
    except ImportError:
        raise ImportError(
            "claude-agent-sdk required: pip install claude-agent-sdk\n"
            "Also ensure Claude Code CLI is installed: npm install -g @anthropic-ai/claude-code"
        )


# Threshold for including full content vs referencing file
INLINE_THRESHOLD_CHARS = 8000  # ~2k tokens


@dataclass
class RunnerConfig:
    """Configuration for the Agent SDK runner."""

    # Model settings (use Bedrock inference profile format if use_bedrock=True)
    model: str = "claude-sonnet-4-20250514"

    # Bedrock settings
    use_bedrock: bool = False
    bedrock_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Permission mode - acceptEdits auto-approves file operations
    permission_mode: str = "acceptEdits"

    # Tools to allow
    allowed_tools: list[str] = field(default_factory=lambda: [
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Grep",
        "Glob",
        "Skill",  # Enable skills
    ])

    # Load project settings (CLAUDE.md, skills)
    setting_sources: list[str] = field(default_factory=lambda: ["project"])

    # Sandbox settings for bash execution
    sandbox_enabled: bool = True
    auto_allow_bash_if_sandboxed: bool = True

    # Timeouts
    max_turns: int = 50
    timeout_seconds: int = 600  # 10 minutes


class AgentRunner:
    """
    Runs the Claude Agent SDK in a workspace.

    This is the "root agent" in the RLM architecture. It operates
    on the workspace filesystem, writing/running scripts that make
    recursive Claude calls.
    """

    def __init__(
        self,
        workspace_path: Path,
        config: Optional[RunnerConfig] = None,
        on_message: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize the runner.

        Args:
            workspace_path: Path to the workspace directory
            config: Runner configuration
            on_message: Callback for streaming messages
        """
        self.workspace = Path(workspace_path)
        self.config = config or RunnerConfig()
        self.on_message = on_message

    def _build_system_prompt_suffix(self) -> str:
        """Build the RLM-specific system prompt addition."""
        return """
You are RLM-FS: an agent that solves tasks by operating on a workspace filesystem.

You have access to a structured workspace with tools for recursive processing:
- tools/llm_client.py: Call Claude from scripts (cached, logged)
- tools/chunking.py: Split large text into chunks

Workspace layout:
- input/: User's request (prompt.md) and attachments
- state/: Working state (job.json config, notes.md, evidence.jsonl)
- cache/: Response cache and indexes
- scratch/scripts/: Your generated scripts
- output/: Final deliverables (answer.md)

Rules:
- For large inputs, prefer scripts over loading everything into context
- Store intermediate results in files (state/, cache/)
- Write your final answer to output/answer.md
- Check state/job.json for budget limits (max tokens, max scripts, etc.)
"""

    def _load_input_content(self) -> Tuple[Optional[str], Optional[dict], int]:
        """
        Load input content from workspace.

        Returns:
            (prompt_text, messages_dict, total_chars)
        """
        prompt_path = self.workspace / "input" / "prompt.md"
        messages_path = self.workspace / "input" / "messages.json"

        prompt_text = None
        messages_dict = None
        total_chars = 0

        if messages_path.exists():
            with open(messages_path) as f:
                messages_dict = json.load(f)
            # Estimate size
            total_chars = len(json.dumps(messages_dict))
        elif prompt_path.exists():
            prompt_text = prompt_path.read_text()
            total_chars = len(prompt_text)

        return prompt_text, messages_dict, total_chars

    def _format_messages_for_prompt(self, messages_dict: dict) -> str:
        """Format messages dict into a readable prompt."""
        lines = []

        if messages_dict.get("system"):
            lines.append(f"[System: {messages_dict['system']}]\n")

        for msg in messages_dict.get("messages", []):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Handle content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)

            lines.append(f"[{role}]\n{content}\n")

        return "\n".join(lines)

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt that kicks off the agent."""
        prompt_text, messages_dict, total_chars = self._load_input_content()

        # If content is small enough, include it directly
        if total_chars <= INLINE_THRESHOLD_CHARS:
            if messages_dict:
                conversation = self._format_messages_for_prompt(messages_dict)
                return f"""Here is the user's conversation/request:

---
{conversation}
---

Complete this request. Write your final answer to output/answer.md.

If this requires complex processing, you can:
- Write scripts to scratch/scripts/
- Use tools/llm_client.py for sub-queries
- Store intermediate results in state/

For simple requests, just solve it directly."""

            elif prompt_text:
                return f"""Here is the user's request:

---
{prompt_text}
---

Complete this request. Write your final answer to output/answer.md.

If this requires complex processing, you can:
- Write scripts to scratch/scripts/
- Use tools/llm_client.py for sub-queries
- Store intermediate results in state/

For simple requests, just solve it directly."""

        # Content is large - reference the file instead
        size_kb = total_chars // 1024

        if messages_dict:
            last_user_msg = None
            for msg in reversed(messages_dict.get("messages", [])):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                last_user_msg = block.get("text", "")[:500]
                                break
                    else:
                        last_user_msg = content[:500]
                    break

            preview = f'Last user message: "{last_user_msg}..."' if last_user_msg else ""

            return f"""The user's conversation is in input/messages.json ({size_kb}KB, too large to include here).
{preview}

The conversation is also rendered as markdown in input/prompt.md.

To process this:
1. Read state/job.json for configuration and budget limits
2. Use tools/chunking.py to split the content
3. Use tools/llm_client.py for sub-queries on chunks
4. Store intermediate results in state/evidence.jsonl
5. Write your final answer to output/answer.md

Begin by understanding what the user is asking for."""

        else:
            # Large prompt - show preview
            preview = prompt_text[:500] if prompt_text else ""

            return f"""The user's request is in input/prompt.md ({size_kb}KB, too large to include here).

Preview:
---
{preview}...
---

To process this large input:
1. Read state/job.json for configuration and budget limits
2. Use tools/chunking.py to split the content
3. Use tools/llm_client.py for sub-queries on chunks
4. Store intermediate results in state/evidence.jsonl
5. Write your final answer to output/answer.md

Begin by scanning the input to understand its structure."""

    async def run_async(self) -> dict:
        """
        Run the agent asynchronously.

        Returns:
            Dict with 'answer' (str or None), 'metrics' (dict), 'errors' (list)
        """
        _ensure_sdk()  # Import SDK on first use

        # Set up environment for Bedrock if enabled
        if self.config.use_bedrock:
            os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
            model = self.config.bedrock_model
        else:
            model = self.config.model

        # Build options
        options = ClaudeAgentOptions(
            cwd=str(self.workspace),
            allowed_tools=self.config.allowed_tools,
            system_prompt=self._build_system_prompt_suffix(),
            max_turns=self.config.max_turns,
            permission_mode=self.config.permission_mode,
        )

        # Run the agent
        messages = []
        async for message in query(
            prompt=self._build_initial_prompt(),
            options=options,
        ):
            messages.append(message)
            if self.on_message:
                self.on_message(message)

        # Collect results
        answer_path = self.workspace / "output" / "answer.md"
        answer = answer_path.read_text() if answer_path.exists() else None

        metrics_path = self.workspace / "state" / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)

        errors_path = self.workspace / "state" / "errors.jsonl"
        errors = []
        if errors_path.exists():
            import json
            with open(errors_path) as f:
                for line in f:
                    if line.strip():
                        errors.append(json.loads(line))

        return {
            "answer": answer,
            "metrics": metrics,
            "errors": errors,
            "messages": messages,
        }

    def run(self) -> dict:
        """
        Run the agent synchronously.

        Returns:
            Dict with 'answer', 'metrics', 'errors', 'messages'
        """
        return asyncio.run(self.run_async())


def run_in_workspace(
    workspace_path: Path,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 50,
    on_message: Optional[Callable[[dict], None]] = None,
) -> dict:
    """
    Convenience function to run the agent in a workspace.

    Args:
        workspace_path: Path to workspace
        model: Model to use
        max_turns: Max conversation turns
        on_message: Streaming callback

    Returns:
        Result dict with answer, metrics, errors
    """
    config = RunnerConfig(model=model, max_turns=max_turns)
    runner = AgentRunner(workspace_path, config, on_message)
    return runner.run()
