#!/usr/bin/env python3
"""
OOLONG Benchmark Runner for fsRLM.

OOLONG (Out-of-Length Ordered Natural Generation) is a benchmark for
evaluating long-context reasoning and aggregation capabilities.

Paper: https://arxiv.org/abs/2511.02817
Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth

Usage:
    uv run python -m benchmarks.oolong --split synth --limit 10
    uv run python -m benchmarks.oolong --split real --task trec_coarse
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "Benchmark dependencies required: uv pip install -e '.[benchmarks]'"
    )

from fsrlm import RLM, RLMConfig, RLMResult


@dataclass
class OolongTask:
    """A single OOLONG benchmark task."""
    task_id: str
    dataset_name: str
    query: str
    context: str  # The long context (can be huge)
    expected_answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from running a single benchmark task."""
    task_id: str
    success: bool
    predicted_answer: Optional[str]
    expected_answer: Any
    score: float
    latency_seconds: float
    metrics: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class BenchmarkRun:
    """Summary of a complete benchmark run."""
    benchmark_name: str
    split: str
    timestamp: str
    total_tasks: int
    completed_tasks: int
    avg_score: float
    avg_latency: float
    total_subcalls: int
    total_tokens: int
    results: List[BenchmarkResult]
    config: Dict[str, Any]


def load_oolong_synth(
    task_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[OolongTask]:
    """
    Load OOLONG-synth benchmark tasks.

    Args:
        task_name: Specific dataset to load (e.g., 'trec', 'agnews', 'metaphors')
                   If None, loads all datasets.
        limit: Maximum number of tasks to load.

    Returns:
        List of OolongTask objects.
    """
    print("Loading OOLONG-synth dataset from HuggingFace...")

    # Load the dataset
    dataset = load_dataset("oolongbench/oolong-synth", split="test")

    tasks = []
    for i, item in enumerate(dataset):
        # Filter by dataset name if specified
        if task_name and item.get("dataset") != task_name:
            continue

        # Extract answer (can be a list)
        answer = item.get("answer")
        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]

        task = OolongTask(
            task_id=f"synth_{item.get('id', i)}",
            dataset_name=item.get("dataset", "unknown"),
            query=item.get("question", ""),
            context=item.get("context_window_text", ""),
            expected_answer=answer,
            metadata={
                "context_len": item.get("context_len"),
                "task_group": item.get("task_group"),
                "task": item.get("task"),
                "answer_type": item.get("answer_type"),
                "num_labels": item.get("num_labels"),
            }
        )
        tasks.append(task)

        if limit and len(tasks) >= limit:
            break

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def load_oolong_real(
    config_name: str = "dnd",
    limit: Optional[int] = None,
) -> List[OolongTask]:
    """
    Load OOLONG-real benchmark tasks.

    Args:
        config_name: Dataset config ('dnd' or 'toy_dnd')
        limit: Maximum number of tasks to load.

    Returns:
        List of OolongTask objects.
    """
    print(f"Loading OOLONG-real ({config_name}) dataset from HuggingFace...")

    dataset = load_dataset("oolongbench/oolong-real", config_name, split="test")

    tasks = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        task = OolongTask(
            task_id=f"real_{item.get('id', i)}",
            dataset_name=f"oolong-real-{config_name}",
            query=item.get("question", ""),
            context=item.get("context_window_text", ""),
            expected_answer=item.get("answer"),
            metadata={
                "question_type": item.get("question_type"),
                "campaign": item.get("campaign"),
                "episodes": item.get("episodes"),
            }
        )
        tasks.append(task)

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def score_answer(predicted: str, expected: Any, dataset_name: str) -> float:
    """
    Score a predicted answer against the expected answer.

    OOLONG uses different scoring depending on answer type:
    - Numerical: 0.75^|y - ŷ|
    - Exact match: 1.0 if match, 0.0 otherwise
    """
    if predicted is None:
        return 0.0

    predicted = str(predicted).strip().lower()

    # Try numerical scoring
    try:
        pred_num = float(predicted)
        exp_num = float(expected)
        diff = abs(pred_num - exp_num)
        return 0.75 ** diff
    except (ValueError, TypeError):
        pass

    # Exact match
    expected_str = str(expected).strip().lower()
    return 1.0 if predicted == expected_str else 0.0


def format_task_prompt(task: OolongTask) -> str:
    """Format a task into a prompt for the RLM."""
    return f"""You are given a long context and a query. Analyze the context to answer the query.

## Context

{task.context}

## Query

{task.query}

## Instructions

1. Carefully analyze the context above
2. Count, aggregate, or reason as needed to answer the query
3. Provide your final answer as a single value (number or short text)
4. Write ONLY the answer, nothing else

## Answer"""


def run_benchmark(
    tasks: List[OolongTask],
    config: Optional[RLMConfig] = None,
    verbose: bool = True,
) -> BenchmarkRun:
    """
    Run the benchmark on a list of tasks.

    Args:
        tasks: List of OolongTask objects
        config: RLMConfig to use (uses defaults if None)
        verbose: Whether to show progress

    Returns:
        BenchmarkRun with results
    """
    config = config or RLMConfig(
        max_tokens_per_subcall=800,
        max_depth=5,
        max_scripts=15,
        max_subcalls_per_script=30,
        submodel="claude-haiku-4-20250414",
        preserve_workspace=False,
    )

    rlm = RLM(config=config)
    results: List[BenchmarkResult] = []

    iterator = tqdm(tasks, desc="Running benchmark") if verbose else tasks

    for task in iterator:
        start_time = time.time()
        error = None
        predicted = None
        score = 0.0
        metrics = {}

        try:
            prompt = format_task_prompt(task)
            result = rlm.invoke(prompt)

            predicted = result.answer.strip() if result.answer else None
            metrics = result.metrics
            score = score_answer(predicted, task.expected_answer, task.dataset_name)

        except Exception as e:
            error = str(e)

        latency = time.time() - start_time

        bench_result = BenchmarkResult(
            task_id=task.task_id,
            success=error is None and predicted is not None,
            predicted_answer=predicted,
            expected_answer=task.expected_answer,
            score=score,
            latency_seconds=latency,
            metrics=metrics,
            error=error,
        )
        results.append(bench_result)

        if verbose:
            status = "✓" if score > 0.5 else "✗"
            tqdm.write(f"  {status} {task.task_id}: score={score:.2f}, latency={latency:.1f}s")

    # Compute summary statistics
    completed = [r for r in results if r.success]
    avg_score = sum(r.score for r in results) / len(results) if results else 0
    avg_latency = sum(r.latency_seconds for r in results) / len(results) if results else 0
    total_subcalls = sum(r.metrics.get("subcalls_made", 0) for r in results)
    total_tokens = sum(r.metrics.get("tokens_used", 0) for r in results)

    run = BenchmarkRun(
        benchmark_name="OOLONG",
        split=tasks[0].dataset_name if tasks else "unknown",
        timestamp=datetime.utcnow().isoformat(),
        total_tasks=len(tasks),
        completed_tasks=len(completed),
        avg_score=avg_score,
        avg_latency=avg_latency,
        total_subcalls=total_subcalls,
        total_tokens=total_tokens,
        results=results,
        config=asdict(config),
    )

    return run


def save_results(run: BenchmarkRun, output_dir: Path) -> Path:
    """Save benchmark results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"oolong_{run.split}_{run.timestamp.replace(':', '-')}.json"
    output_path = output_dir / filename

    # Convert to dict for JSON serialization
    data = {
        "benchmark_name": run.benchmark_name,
        "split": run.split,
        "timestamp": run.timestamp,
        "total_tasks": run.total_tasks,
        "completed_tasks": run.completed_tasks,
        "avg_score": run.avg_score,
        "avg_latency": run.avg_latency,
        "total_subcalls": run.total_subcalls,
        "total_tokens": run.total_tokens,
        "config": run.config,
        "results": [asdict(r) for r in run.results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def print_summary(run: BenchmarkRun) -> None:
    """Print a summary of the benchmark run."""
    print("\n" + "=" * 60)
    print(f"OOLONG Benchmark Results - {run.split}")
    print("=" * 60)
    print(f"Tasks:      {run.completed_tasks}/{run.total_tasks} completed")
    print(f"Avg Score:  {run.avg_score:.3f}")
    print(f"Avg Latency: {run.avg_latency:.1f}s per task")
    print(f"Subcalls:   {run.total_subcalls} total")
    print(f"Tokens:     {run.total_tokens:,} total")
    print("=" * 60)

    # Score distribution
    scores = [r.score for r in run.results]
    perfect = sum(1 for s in scores if s == 1.0)
    good = sum(1 for s in scores if 0.5 <= s < 1.0)
    poor = sum(1 for s in scores if s < 0.5)

    print(f"Score Distribution:")
    print(f"  Perfect (1.0):  {perfect}")
    print(f"  Good (≥0.5):    {good}")
    print(f"  Poor (<0.5):    {poor}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run OOLONG benchmark on fsRLM")
    parser.add_argument(
        "--split",
        choices=["synth", "real"],
        default="synth",
        help="Which OOLONG split to run (default: synth)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task name for synth split (e.g., trec_coarse)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Max tokens per subcall (default: 800)",
    )
    parser.add_argument(
        "--max-scripts",
        type=int,
        default=15,
        help="Max scripts per task (default: 15)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Load tasks
    if args.split == "synth":
        tasks = load_oolong_synth(task_name=args.task, limit=args.limit)
    else:
        tasks = load_oolong_real(limit=args.limit)

    if not tasks:
        print("No tasks loaded!")
        return

    # Configure RLM
    config = RLMConfig(
        max_tokens_per_subcall=args.max_tokens,
        max_scripts=args.max_scripts,
        max_subcalls_per_script=30,
        submodel="claude-haiku-4-20250414",
        preserve_workspace=False,
    )

    # Run benchmark
    run = run_benchmark(tasks, config=config, verbose=not args.quiet)

    # Save and print results
    output_path = save_results(run, args.output_dir)
    print_summary(run)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
