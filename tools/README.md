# RLM Tools

Shared utilities for RLM scripts. These tools handle common operations like calling Claude and chunking large text.

## llm_client.py

Cached, logged wrapper around the Anthropic API.

### Quick Usage

```python
from tools.llm_client import call_claude

# Simple call
result = call_claude("Summarize this text: ...")

# With evidence logging
result = call_claude(
    "Extract key facts from: ...",
    log_to_evidence=True,
    evidence_tag="facts_extraction"
)
```

### Full Client Usage

```python
from tools.llm_client import LLMClient

client = LLMClient()

# Single call with all options
result = client.call(
    prompt="Your prompt here",
    system="You are a helpful assistant",
    model="claude-haiku-4-20250414",  # default from config
    max_tokens=1000,
    temperature=0.0,
    log_to_evidence=True,
    evidence_tag="my_tag",
    use_cache=True,
)

if result:
    print(result.content)
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
    print(f"Cached: {result.cached}")

# Batch calls
results = client.call_batch(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    system="Summarize briefly",
    log_to_evidence=True,
)
```

### Features

- **Caching**: Responses cached by content hash in `cache/llm/`
- **Error logging**: Failures logged to `state/errors.jsonl`
- **Evidence logging**: Optional logging to `state/evidence.jsonl`
- **Metrics tracking**: Usage tracked in `state/metrics.json`
- **Config-driven**: Reads defaults from `state/job.json`

## chunking.py

Utilities for splitting large text into chunks.

### Quick Usage

```python
from tools.chunking import chunk_prompt, get_prompt_lines

# Chunk the main prompt file
chunks = chunk_prompt(max_tokens=2000)
for chunk in chunks:
    process(chunk.content)

# Get specific lines
section = get_prompt_lines(100, 150)
```

### Full Chunker Usage

```python
from tools.chunking import Chunker

chunker = Chunker()

# Chunk by tokens (default)
chunks = chunker.chunk_file(
    "input/prompt.md",
    method="tokens",
    max_tokens=2000,
    overlap=100,
)

# Chunk by lines
chunks = chunker.chunk_file(
    "input/prompt.md",
    method="lines",
    lines_per_chunk=100,
    overlap=5,
)

# Chunk by paragraphs
chunks = chunker.chunk_file(
    "input/prompt.md",
    method="paragraphs",
)

# Build heading index
index = chunker.build_index("input/prompt.md", index_type="headings")
```

### Chunk Object

Each chunk has:
- `content`: The text content
- `index`: Chunk number (0-indexed)
- `start_line`, `end_line`: Line range (1-indexed)
- `start_byte`, `end_byte`: Byte range
- `token_estimate`: Approximate token count
- `source_file`: Original file path
- `chunk_id`: Unique identifier

### Manifests

When chunking a file, a manifest is saved to `cache/indexes/<filename>_chunks.json` with:
- Source file info
- Chunking method and parameters
- All chunk metadata

## Best Practices

1. **Always log to evidence** when extracting important information
2. **Use caching** to avoid redundant API calls
3. **Check errors** periodically via `state/errors.jsonl`
4. **Print minimal stdout** - write large outputs to files instead
5. **Use manifests** to track what's been processed
