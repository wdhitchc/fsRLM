# ccRLM - Filesystem-Based Recursive Language Model

A filesystem-based implementation of [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601) using the Claude Agent SDK.

> **RLMs** enable language models to process arbitrarily long inputs by treating prompts as an external environment, allowing the model to programmatically examine, decompose, and recursively call itself over snippets of the input.

## What is ccRLM?

ccRLM implements the RLM paradigm with a **filesystem-as-working-memory** approach:

- **Traditional RLM**: Python REPL with `extra_data` variable + `llm_batch()` function
- **ccRLM**: Structured workspace filesystem + Claude Agent SDK + Python scripts

```
┌─────────────────────────────────────────────────────────────┐
│  rlm.invoke(messages=[...])                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Workspace Filesystem (working memory)                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ input/prompt.md        # Your input (can be huge)       ││
│  │ state/evidence.jsonl   # Extracted facts from chunks    ││
│  │ cache/llm/             # Cached recursive call results  ││
│  │ scratch/scripts/       # Agent-generated processing     ││
│  │ output/answer.md       # Final answer                   ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│              Agent writes & runs Python scripts              │
│                     that call Claude (Haiku)                 │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ tools/llm_client.py    # Cached Claude API wrapper      ││
│  │ tools/chunking.py      # Text chunking utilities        ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Why Filesystem-Based?

| Approach | Pros | Cons |
|----------|------|------|
| **REPL-based** (original RLM) | Direct variable access, fast | State lost on crash, hard to debug |
| **Filesystem-based** (ccRLM) | Persistent, debuggable, observable | Slightly more overhead |

The filesystem approach gives you:
- **Observability**: Watch the agent's reasoning unfold in real files
- **Persistence**: Crash recovery, pause/resume capability
- **Debugging**: Inspect `state/evidence.jsonl`, replay scripts
- **Caching**: Response cache survives across runs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ccrlm
cd ccrlm

# Install with uv (recommended)
uv venv && uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"

# Ensure Claude Code CLI is installed
npm install -g @anthropic-ai/claude-code
```

## Quick Start

```python
from ccrlm import RLM

rlm = RLM()

# Simple prompt
result = rlm.invoke("Analyze this data and find patterns...")
print(result.answer)

# Or with messages (Anthropic API format)
result = rlm.invoke(messages=[
    {"role": "user", "content": "Here's a large dataset..."},
    {"role": "assistant", "content": "I see the data. What should I analyze?"},
    {"role": "user", "content": "Find all anomalies and explain them."},
])

# With system prompt
result = rlm.invoke(
    messages=[{"role": "user", "content": "Review this codebase..."}],
    system="You are a security expert. Be thorough."
)
```

## Configuration

```python
from ccrlm import RLM, RLMConfig

config = RLMConfig(
    # Recursive subcalls (scripts calling Claude)
    max_tokens_per_subcall=1000,
    max_depth=5,
    max_scripts=20,
    max_subcalls_per_script=25,
    submodel="claude-haiku-4-20250414",  # Fast & cheap for chunks

    # Root agent
    root_model="claude-sonnet-4-20250514",
    max_turns=50,

    # Debugging
    preserve_workspace=True,  # Keep files after run
    cache_responses=True,
)

rlm = RLM(config=config)
```

## How It Works

1. **Input**: Your prompt/messages are written to `input/prompt.md`
2. **Root Agent**: Claude (via Agent SDK) reads the input and decides how to process it
3. **Scripts**: For large inputs, the agent writes Python scripts to `scratch/scripts/`
4. **Recursive Calls**: Scripts use `tools/llm_client.py` to call Claude (Haiku) on chunks
5. **Evidence**: Results accumulate in `state/evidence.jsonl`
6. **Answer**: Final answer is written to `output/answer.md`

### Small Input (< 8KB)
The agent sees the full input inline and can answer directly.

### Large Input (> 8KB)
The agent receives a preview and instructions to:
1. Build an index of the input structure
2. Chunk the content for processing
3. Make recursive Claude calls on each chunk
4. Synthesize the results into a final answer

## Benchmarks

ccRLM includes integration with the [OOLONG benchmark](https://huggingface.co/datasets/oolongbench/oolong-synth) used in the original RLM paper.

```bash
# Run OOLONG benchmark
uv run python -m benchmarks.oolong --split synth --limit 10

# Compare with baseline
uv run python -m benchmarks.compare
```

See [benchmarks/README.md](benchmarks/README.md) for details.

## Comparison with Original RLM

| Feature | Original RLM | ccRLM |
|---------|--------------|-------|
| Environment | Python REPL | Filesystem + Agent SDK |
| State storage | `extra_data` variable | `input/prompt.md` file |
| Recursive calls | `llm_batch()` function | `tools/llm_client.py` |
| Final answer | `answer` variable | `output/answer.md` file |
| Sandboxing | Docker/Modal | Agent SDK sandbox |
| Debugging | Trajectory logs | Real files you can inspect |

## Roadmap

- [x] Core RLM implementation with filesystem workspace
- [x] Claude Agent SDK integration
- [x] Messages API support (Anthropic format)
- [x] Response caching and metrics
- [ ] OOLONG benchmark integration
- [ ] LangChain/LangGraph agent backend (alternative to Agent SDK)
- [ ] Streaming output support
- [ ] Multi-agent collaboration
- [ ] Web UI for workspace visualization

## Architecture

```
ccrlm/
├── ccrlm/
│   ├── __init__.py      # Public API
│   ├── rlm.py           # Main RLM class
│   ├── runner.py        # Agent SDK wrapper
│   └── workspace.py     # Workspace lifecycle
├── tools/
│   ├── llm_client.py    # Cached Claude API for scripts
│   ├── chunking.py      # Text chunking utilities
│   └── README.md        # Usage docs for agent
├── workspace_template/
│   └── .claude/
│       ├── CLAUDE.md    # Agent instructions
│       └── skills/      # RLM scripting skill
├── benchmarks/
│   ├── oolong.py        # OOLONG benchmark runner
│   └── data/            # Downloaded datasets
└── examples/
    └── basic_usage.py   # Usage examples
```

## Related Work

- [Recursive Language Models (Paper)](https://arxiv.org/abs/2512.24601) - Zhang & Khattab, MIT
- [Official RLM Implementation](https://github.com/alexzhang13/rlm) - REPL-based approach
- [Prime Intellect RLMEnv](https://github.com/PrimeIntellect-ai/verifiers) - RL training environment
- [OOLONG Benchmark](https://huggingface.co/datasets/oolongbench/oolong-synth) - Long-context evaluation

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
