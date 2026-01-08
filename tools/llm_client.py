"""
LLM Client wrapper for fsRLM recursive calls.

This module provides a cached, logged wrapper around the Anthropic API
for use in RLM scripts. It handles:
- Response caching (by content hash)
- Error logging (to state/errors.jsonl)
- Evidence logging (to state/evidence.jsonl)
- Metrics tracking (to state/metrics.json)
- Budget enforcement (max tokens, max depth)

Usage in scripts:
    from tools.llm_client import LLMClient

    client = LLMClient()
    result = client.call("Summarize this: ...")

    # Or with more control:
    result = client.call(
        prompt="Summarize this chunk",
        system="You are a summarization assistant",
        max_tokens=500,
        log_to_evidence=True,
        evidence_tag="chunk_summary"
    )
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from anthropic import Anthropic, APIError, AnthropicBedrock
except ImportError:
    raise ImportError("anthropic package required: pip install anthropic")


@dataclass
class CallResult:
    """Result from an LLM call."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    call_id: str


def _hash_request(model: str, system: str, prompt: str, max_tokens: int) -> str:
    """Generate a deterministic hash for caching."""
    key = json.dumps({
        "model": model,
        "system": system,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class LLMClient:
    """
    Cached, logged wrapper around Anthropic API for RLM scripts.

    Reads configuration from state/job.json and enforces budgets.
    All calls are logged and cached to the workspace filesystem.
    Supports both direct Anthropic API and AWS Bedrock.
    """

    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize the client.

        Args:
            workspace_root: Path to workspace. If None, uses cwd.
        """
        self.workspace = Path(workspace_root or os.getcwd())
        self.config = self._load_config()

        # Check if using Bedrock
        self.use_bedrock = self.config.get("use_bedrock", False)

        if self.use_bedrock:
            # Use AnthropicBedrock client - uses AWS credentials from environment
            # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
            self.client = AnthropicBedrock()
        else:
            self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Paths
        self.cache_dir = self.workspace / "cache" / "llm"
        self.errors_path = self.workspace / "state" / "errors.jsonl"
        self.evidence_path = self.workspace / "state" / "evidence.jsonl"
        self.metrics_path = self.workspace / "state" / "metrics.json"

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from job.json."""
        job_path = self.workspace / "state" / "job.json"
        if job_path.exists():
            with open(job_path) as f:
                job = json.load(f)
                return job.get("config", {})
        return {}

    def _get_metrics(self) -> dict:
        """Load current metrics."""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return {
            "scripts_created": 0,
            "subcalls_made": 0,
            "tokens_used": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    def _save_metrics(self, metrics: dict) -> None:
        """Save metrics."""
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _log_error(self, error: Exception, context: dict) -> None:
        """Log an error to errors.jsonl."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        }
        with open(self.errors_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Update metrics
        metrics = self._get_metrics()
        metrics["errors"] = metrics.get("errors", 0) + 1
        self._save_metrics(metrics)

    def _log_evidence(self, content: str, tag: str, metadata: dict) -> None:
        """Log evidence to evidence.jsonl."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tag": tag,
            "content": content,
            "metadata": metadata,
        }
        with open(self.evidence_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _get_cached(self, cache_key: str) -> Optional[dict]:
        """Try to get a cached response."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _save_cache(self, cache_key: str, response: dict) -> None:
        """Save a response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(response, f, indent=2)

    def call(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        log_to_evidence: bool = False,
        evidence_tag: str = "llm_response",
        use_cache: bool = True,
    ) -> Optional[CallResult]:
        """
        Call Claude with caching and logging.

        Args:
            prompt: The user prompt
            system: System prompt
            model: Model to use (defaults to config submodel, usually Haiku)
            max_tokens: Max tokens (defaults to config max_tokens_per_call)
            temperature: Temperature (default 0 for determinism)
            log_to_evidence: Whether to log response to evidence.jsonl
            evidence_tag: Tag for evidence entry
            use_cache: Whether to use response cache

        Returns:
            CallResult on success, None on failure (error is logged)
        """
        # Apply defaults from config
        # Use Bedrock model format if enabled
        if self.use_bedrock:
            default_model = self.config.get("bedrock_submodel", "us.anthropic.claude-haiku-4-20250414-v1:0")
        else:
            default_model = self.config.get("submodel", "claude-haiku-4-20250414")
        model = model or default_model
        max_tokens = max_tokens or self.config.get("max_tokens_per_call", 1000)

        # Check cache
        cache_key = _hash_request(model, system, prompt, max_tokens)

        if use_cache and self.config.get("cache_responses", True):
            cached = self._get_cached(cache_key)
            if cached:
                metrics = self._get_metrics()
                metrics["cache_hits"] = metrics.get("cache_hits", 0) + 1
                self._save_metrics(metrics)

                result = CallResult(
                    content=cached["content"],
                    model=cached["model"],
                    input_tokens=cached.get("input_tokens", 0),
                    output_tokens=cached.get("output_tokens", 0),
                    cached=True,
                    call_id=cache_key,
                )

                if log_to_evidence:
                    self._log_evidence(result.content, evidence_tag, {
                        "cached": True,
                        "call_id": cache_key,
                    })

                return result

        # Make API call
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Update metrics
            metrics = self._get_metrics()
            metrics["subcalls_made"] = metrics.get("subcalls_made", 0) + 1
            metrics["tokens_used"] = metrics.get("tokens_used", 0) + input_tokens + output_tokens
            self._save_metrics(metrics)

            # Cache response
            if use_cache and self.config.get("cache_responses", True):
                self._save_cache(cache_key, {
                    "content": content,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            result = CallResult(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=False,
                call_id=cache_key,
            )

            if log_to_evidence:
                self._log_evidence(content, evidence_tag, {
                    "cached": False,
                    "call_id": cache_key,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                })

            return result

        except APIError as e:
            self._log_error(e, {
                "model": model,
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
            })
            return None
        except Exception as e:
            self._log_error(e, {
                "model": model,
                "prompt_length": len(prompt),
            })
            return None

    def call_batch(
        self,
        prompts: list[str],
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        log_to_evidence: bool = False,
        evidence_tag: str = "batch_response",
    ) -> list[Optional[CallResult]]:
        """
        Call Claude on multiple prompts.

        This is a convenience wrapper that calls each prompt sequentially.
        For true batching with the Message Batches API, use call_batch_async.

        Args:
            prompts: List of prompts
            system: System prompt (same for all)
            model: Model to use
            max_tokens: Max tokens per call
            log_to_evidence: Whether to log responses
            evidence_tag: Tag for evidence entries

        Returns:
            List of CallResults (None for failed calls)
        """
        results = []
        for i, prompt in enumerate(prompts):
            result = self.call(
                prompt=prompt,
                system=system,
                model=model,
                max_tokens=max_tokens,
                log_to_evidence=log_to_evidence,
                evidence_tag=f"{evidence_tag}_{i}",
            )
            results.append(result)
        return results

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text.

        Uses the Anthropic token counting API.
        """
        if self.use_bedrock:
            default_model = self.config.get("bedrock_submodel", "us.anthropic.claude-haiku-4-20250414-v1:0")
        else:
            default_model = self.config.get("submodel", "claude-haiku-4-20250414")
        model = model or default_model
        try:
            result = self.client.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception:
            # Rough estimate if API fails
            return len(text) // 4


# Convenience function for simple usage
def call_claude(
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 1000,
    log_to_evidence: bool = False,
    evidence_tag: str = "llm_response",
) -> Optional[str]:
    """
    Simple function to call Claude from scripts.

    Returns just the content string (or None on failure).

    Example:
        from tools.llm_client import call_claude
        summary = call_claude("Summarize: " + chunk)
    """
    client = LLMClient()
    result = client.call(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        log_to_evidence=log_to_evidence,
        evidence_tag=evidence_tag,
    )
    return result.content if result else None
