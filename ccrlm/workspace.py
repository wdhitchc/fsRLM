"""
Workspace management for ccRLM.

Handles creation, scaffolding, and cleanup of the structured workspace
that serves as the "environment" for the RLM agent.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class WorkspaceConfig:
    """Configuration for workspace behavior."""
    max_tokens_per_call: int = 1000
    max_depth: int = 5
    max_scripts: int = 20
    max_subcalls_per_script: int = 25
    submodel: str = "claude-haiku-4-20250414"
    cache_responses: bool = True
    # Bedrock settings
    use_bedrock: bool = False
    bedrock_submodel: str = "us.anthropic.claude-haiku-4-20250414-v1:0"


@dataclass
class Workspace:
    """
    Manages the structured workspace filesystem for RLM operations.

    The workspace layout:
        input/           - User prompt and attachments
        state/           - Job manifest, notes, evidence
        cache/           - LLM response cache, indexes
        scratch/         - Generated scripts, temp files
        output/          - Final answer artifacts
        tools/           - Shared utilities (llm_client, chunking)
        .claude/         - CLAUDE.md and skills
    """

    root: Path
    config: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    _is_temp: bool = field(default=False, repr=False)

    # Standard subdirectories
    DIRS = [
        "input",
        "input/attachments",
        "state",
        "cache/llm",
        "cache/indexes",
        "scratch/scripts",
        "scratch/tmp",
        "output",
        "tools",
        ".claude/skills/rlm-scripting",
    ]

    @classmethod
    def create(
        cls,
        root: Optional[Path] = None,
        config: Optional[WorkspaceConfig] = None,
    ) -> "Workspace":
        """
        Create a new workspace.

        Args:
            root: Path for workspace. If None, creates a temp directory.
            config: Workspace configuration. Uses defaults if None.

        Returns:
            Initialized Workspace instance.
        """
        config = config or WorkspaceConfig()
        is_temp = root is None

        if is_temp:
            root = Path(tempfile.mkdtemp(prefix="ccrlm_"))
        else:
            root = Path(root)
            root.mkdir(parents=True, exist_ok=True)

        workspace = cls(root=root, config=config, _is_temp=is_temp)
        workspace._scaffold()
        return workspace

    def _scaffold(self) -> None:
        """Create the directory structure and initial files."""
        # Create directories
        for dir_path in self.DIRS:
            (self.root / dir_path).mkdir(parents=True, exist_ok=True)

        # Write job.json manifest
        self._write_job_manifest()

        # Copy tool files
        self._install_tools()

        # Write CLAUDE.md and skill
        self._install_claude_config()

    def _write_job_manifest(self) -> None:
        """Write the job.json manifest file."""
        manifest = {
            "prompt_path": "input/prompt.md",
            "messages_path": "input/messages.json",
            "attachments_dir": "input/attachments",
            "answer_path": "output/answer.md",
            "answer_json_path": "output/answer.json",
            "evidence_path": "state/evidence.jsonl",
            "notes_path": "state/notes.md",
            "errors_path": "state/errors.jsonl",
            "metrics_path": "state/metrics.json",
            "cache_dir": "cache/llm",
            "script_dir": "scratch/scripts",
            "config": {
                "max_tokens_per_call": self.config.max_tokens_per_call,
                "max_depth": self.config.max_depth,
                "max_scripts": self.config.max_scripts,
                "max_subcalls_per_script": self.config.max_subcalls_per_script,
                "submodel": self.config.submodel,
                "cache_responses": self.config.cache_responses,
                "use_bedrock": self.config.use_bedrock,
                "bedrock_submodel": self.config.bedrock_submodel,
            },
            "rules": {
                "prefer_scripts_over_reading_entire_prompt": True,
                "store_large_outputs_in_files": True,
                "log_errors_and_continue": True,
            },
        }

        with open(self.root / "state" / "job.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Initialize empty metrics
        metrics = {
            "scripts_created": 0,
            "subcalls_made": 0,
            "tokens_used": 0,
            "cache_hits": 0,
            "errors": 0,
        }
        with open(self.root / "state" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def _install_tools(self) -> None:
        """Copy the tool files into the workspace."""
        # Get the package directory to find tool templates
        package_dir = Path(__file__).parent.parent
        tools_src = package_dir / "tools"
        tools_dst = self.root / "tools"

        # Copy each tool file if source exists
        for tool_file in ["llm_client.py", "chunking.py", "README.md"]:
            src = tools_src / tool_file
            if src.exists():
                shutil.copy(src, tools_dst / tool_file)

    def _install_claude_config(self) -> None:
        """Write CLAUDE.md and skill files."""
        package_dir = Path(__file__).parent.parent

        # Copy CLAUDE.md
        claude_md_src = package_dir / "workspace_template" / ".claude" / "CLAUDE.md"
        if claude_md_src.exists():
            shutil.copy(claude_md_src, self.root / ".claude" / "CLAUDE.md")

        # Copy skill
        skill_src = package_dir / "workspace_template" / ".claude" / "skills" / "rlm-scripting" / "SKILL.md"
        if skill_src.exists():
            shutil.copy(skill_src, self.root / ".claude" / "skills" / "rlm-scripting" / "SKILL.md")

    def set_prompt(self, prompt: str) -> None:
        """Write the user prompt to input/prompt.md."""
        with open(self.root / "input" / "prompt.md", "w") as f:
            f.write(prompt)

    def set_messages(self, messages: List[Dict], system: Optional[str] = None) -> None:
        """
        Write conversation messages to the workspace.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Follows Anthropic API format.
            system: Optional system prompt.

        The messages are written to:
        - input/messages.json (structured, for scripts)
        - input/prompt.md (rendered as markdown, for reading)
        """
        # Write structured JSON
        data = {
            "system": system,
            "messages": messages,
        }
        with open(self.root / "input" / "messages.json", "w") as f:
            json.dump(data, f, indent=2)

        # Also render as readable markdown
        md_lines = []
        if system:
            md_lines.append("## System Prompt\n")
            md_lines.append(system)
            md_lines.append("\n")

        md_lines.append("## Conversation\n")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle content blocks (text, images, etc.)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            text_parts.append("[image]")
                        else:
                            text_parts.append(f"[{block.get('type', 'unknown')}]")
                    else:
                        text_parts.append(str(block))
                content = "\n".join(text_parts)

            md_lines.append(f"### {role.upper()}\n")
            md_lines.append(content)
            md_lines.append("\n")

        with open(self.root / "input" / "prompt.md", "w") as f:
            f.write("\n".join(md_lines))

    def get_messages(self) -> Optional[Dict]:
        """Read messages if they exist."""
        messages_path = self.root / "input" / "messages.json"
        if messages_path.exists():
            with open(messages_path) as f:
                return json.load(f)
        return None

    def add_attachment(self, name: str, content: bytes) -> Path:
        """Add an attachment file."""
        path = self.root / "input" / "attachments" / name
        with open(path, "wb") as f:
            f.write(content)
        return path

    def get_answer(self) -> Optional[str]:
        """Read the final answer if it exists."""
        answer_path = self.root / "output" / "answer.md"
        if answer_path.exists():
            return answer_path.read_text()
        return None

    def get_metrics(self) -> dict:
        """Read the current metrics."""
        metrics_path = self.root / "state" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}

    def get_errors(self) -> list[dict]:
        """Read logged errors."""
        errors_path = self.root / "state" / "errors.jsonl"
        errors = []
        if errors_path.exists():
            with open(errors_path) as f:
                for line in f:
                    if line.strip():
                        errors.append(json.loads(line))
        return errors

    def get_evidence(self) -> list[dict]:
        """Read collected evidence."""
        evidence_path = self.root / "state" / "evidence.jsonl"
        evidence = []
        if evidence_path.exists():
            with open(evidence_path) as f:
                for line in f:
                    if line.strip():
                        evidence.append(json.loads(line))
        return evidence

    def clear(self) -> None:
        """Clear the workspace for reuse, preserving structure."""
        # Clear content directories
        for dir_name in ["input", "state", "cache", "scratch", "output"]:
            dir_path = self.root / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)

        # Re-scaffold
        self._scaffold()

    def destroy(self) -> None:
        """Completely remove the workspace."""
        if self.root.exists():
            shutil.rmtree(self.root)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._is_temp:
            self.destroy()
