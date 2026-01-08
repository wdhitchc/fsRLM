---
name: rlm-scripting
description: Conventions for writing/running scripts, storing artifacts, and calling Claude via the anthropic Python SDK (recursive calls).
---

## Overview

This skill teaches you how to write Python scripts that:
1. Process large inputs by chunking
2. Make recursive Claude API calls
3. Store results properly in the workspace
4. Handle errors gracefully

## Script Conventions

### Naming
Write scripts to `scratch/scripts/` with incrementing prefixes:
```
001_scan.py       # Initial scan/index
002_chunk.py      # Chunk the input
003_summarize.py  # Summarize chunks
004_synthesize.py # Combine results
005_verify.py     # Verify answer
```

### Structure
Every script MUST:
1. Write main output to `state/` or `cache/`
2. Print ONLY a short summary to stdout
3. Log errors to `state/errors.jsonl` (via llm_client)

### Example Script Template

```python
#!/usr/bin/env python3
"""
Script: 001_scan.py
Purpose: Scan input/prompt.md and build an index
Output: cache/indexes/prompt_index.json
"""

import json
from pathlib import Path

# Use provided tools
from tools.chunking import Chunker

def main():
    chunker = Chunker()

    # Build heading index
    index = chunker.build_index("input/prompt.md", index_type="headings")

    # Print SHORT summary only
    print(f"Indexed {len(index['entries'])} headings")
    print(f"Output: cache/indexes/prompt_headings_index.json")

if __name__ == "__main__":
    main()
```

## Calling Claude from Scripts

Use the provided `tools/llm_client.py` wrapper. It handles caching, error logging, and metrics.

### Simple Usage

```python
from tools.llm_client import call_claude

# Basic call
result = call_claude("Summarize this: " + chunk_text)
if result:
    print(f"Summary: {result[:100]}...")
```

### Full Client Usage

```python
from tools.llm_client import LLMClient

client = LLMClient()

result = client.call(
    prompt="Extract key facts from this text: " + chunk,
    system="You are a fact extraction assistant. Output JSON.",
    max_tokens=500,
    log_to_evidence=True,      # Saves to state/evidence.jsonl
    evidence_tag="facts",
    use_cache=True,            # Uses cache/llm/
)

if result:
    print(f"Extracted facts, tokens: {result.input_tokens}+{result.output_tokens}")
else:
    print("Call failed - check state/errors.jsonl")
```

### Batch Processing

```python
from tools.llm_client import LLMClient
from tools.chunking import chunk_prompt

client = LLMClient()
chunks = chunk_prompt(max_tokens=2000)

# Process each chunk
for chunk in chunks:
    result = client.call(
        prompt=f"Summarize:\n\n{chunk.content}",
        log_to_evidence=True,
        evidence_tag=f"summary_chunk_{chunk.index}",
    )
    # Result is cached - re-running won't re-call API
```

## Where to Store Things

| What | Where | Format |
|------|-------|--------|
| Working notes | `state/notes.md` | Markdown |
| Extracted facts | `state/evidence.jsonl` | Line-delimited JSON |
| Chunk manifests | `cache/indexes/` | JSON |
| API responses | `cache/llm/` | JSON (auto by llm_client) |
| Intermediate data | `scratch/tmp/` | Any |
| Final answer | `output/answer.md` | Markdown |

## Chunking Large Input

Use `tools/chunking.py` to split large text:

```python
from tools.chunking import Chunker, chunk_prompt

# Quick: chunk the main prompt
chunks = chunk_prompt(max_tokens=2000)

# Or with more control
chunker = Chunker()
chunks = chunker.chunk_file(
    "input/prompt.md",
    method="tokens",      # or "lines", "paragraphs"
    max_tokens=2000,
    overlap=100,
)

# Get specific lines without loading whole file
section = chunker.get_lines("input/prompt.md", start=100, end=150)
```

## Error Handling

Errors are automatically logged to `state/errors.jsonl` by llm_client. To handle in scripts:

```python
from tools.llm_client import LLMClient

client = LLMClient()
result = client.call(prompt="...")

if result is None:
    # Error was logged, continue with fallback or skip
    print("Call failed, skipping chunk")
else:
    # Process result
    process(result.content)
```

## Output Discipline

**DO:**
```python
# Write to file
with open("state/analysis.json", "w") as f:
    json.dump(data, f)
print("Analysis saved to state/analysis.json (42 entries)")
```

**DON'T:**
```python
# Don't dump large content to stdout
print(large_text)  # BAD - clogs agent context
print(json.dumps(big_data, indent=2))  # BAD
```

## Typical Workflow

1. **Scan**: Build index of input
   ```
   001_scan.py -> cache/indexes/prompt_index.json
   ```

2. **Plan**: Write notes about approach
   ```
   -> state/notes.md
   ```

3. **Chunk**: Split input for processing
   ```
   002_chunk.py -> cache/indexes/prompt_chunks.json
   ```

4. **Extract**: Process chunks with Claude calls
   ```
   003_extract.py -> state/evidence.jsonl
   ```

5. **Synthesize**: Combine evidence into answer
   ```
   004_synthesize.py -> output/answer.md
   ```

6. **Verify**: Check answer completeness
   ```
   005_verify.py -> prints pass/fail
   ```

## Budget Awareness

Check limits in `state/job.json`:
```python
import json
with open("state/job.json") as f:
    config = json.load(f)["config"]

max_calls = config["max_subcalls_per_script"]  # e.g., 25
max_tokens = config["max_tokens_per_call"]     # e.g., 1000
```

Monitor usage in `state/metrics.json`:
```python
with open("state/metrics.json") as f:
    metrics = json.load(f)
print(f"Subcalls so far: {metrics['subcalls_made']}")
print(f"Total tokens: {metrics['tokens_used']}")
```
