# fsRLM Architecture

This document explains how fsRLM works, the complete prompting flow, and how it maps to the original RLM paper.

## The Original RLM Paper

The [Recursive Language Model (RLM) paper](https://arxiv.org/abs/2512.24601) by Zhang & Khattab introduces a key insight:

> **Prompt as Environment**: Instead of treating the prompt as static input that must fit in the context window, treat it as an external environment that the model can programmatically explore.

### Original RLM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Prompt (can be arbitrarily large)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Python REPL Environment                                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ context = "<the full prompt>"  # In-memory variable     ││
│  │                                                          ││
│  │ def llm_query(prompt):         # Recursive calls        ││
│  │     return call_llm(prompt)    # ~500K char capacity    ││
│  │                                                          ││
│  │ print()                        # View REPL output       ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│              LLM writes and executes Python code             │
│                              │                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ # Model-generated code:                                  ││
│  │ chunks = context.split('\n\n')                           ││
│  │ summaries = [llm_query(f"Summarize: {c}") for c in chunks]│
│  │ final_answer = synthesize(summaries)                     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Final Answer (returned from REPL execution)                │
└─────────────────────────────────────────────────────────────┘
```

**Key Components (from the paper):**
- `context`: In-memory variable containing "extremely important information about your query"
- `llm_query()`: Function for recursive LLM calls (handles ~500K chars)
- `print()`: View REPL code output for continued reasoning
- Python REPL: Executes model-generated code that peeks into, decomposes, and processes the context

## fsRLM: Filesystem-Based Implementation

fsRLM implements the same concept, but uses the **filesystem as working memory** instead of Python variables:

```
┌─────────────────────────────────────────────────────────────┐
│  rlm.invoke(prompt="...", messages=[...])                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Workspace Filesystem (working memory)                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ input/prompt.md        ←── context variable equivalent   ││
│  │ state/evidence.jsonl   ←── intermediate results         ││
│  │ cache/llm/             ←── cached recursive calls       ││
│  │ scratch/scripts/       ←── model-generated code         ││
│  │ output/answer.md       ←── final answer destination     ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│              Claude Agent SDK runs the agent                 │
│                     (reads CLAUDE.md + skills)               │
│                              │                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ tools/llm_client.py    ←── llm_query() equivalent       ││
│  │ tools/chunking.py      ←── text processing utilities    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RLMResult(answer=..., metrics=..., evidence=...)           │
└─────────────────────────────────────────────────────────────┘
```

### Mapping Table

| Original RLM | fsRLM | Purpose |
|--------------|-------|---------|
| `context` variable | `input/prompt.md` file | Stores the (potentially huge) input |
| Return value | `output/answer.md` file | Final answer destination |
| `llm_query()` function | `tools/llm_client.py` | Makes recursive LLM calls |
| Python REPL | Claude Agent SDK | Executes model-generated code |
| In-memory state | `state/` directory | Working memory |
| (none) | `cache/llm/` | Response caching (fsRLM addition) |
| (none) | `.claude/CLAUDE.md` | Agent instructions |
| (none) | `.claude/skills/` | Specialized behaviors |

## Key Design Decisions

### 1. Pre-built Tools (Don't Reinvent the Wheel)

In the original RLM, the model must write all processing code from scratch each time. fsRLM provides **pre-built tools** in the `tools/` directory:

**`tools/llm_client.py`** - Claude API wrapper with:
- Response caching (don't re-call for same prompt)
- Automatic error logging to `state/errors.jsonl`
- Evidence logging to `state/evidence.jsonl`
- Metrics tracking (token counts, call counts)
- Budget enforcement from `job.json` config

**`tools/chunking.py`** - Text processing utilities:
- Token-aware chunking with overlap
- Line/paragraph/heading-based splitting
- Index building for large documents
- Random access to file sections without loading entire file

**Why?** The agent doesn't waste turns reimplementing caching, error handling, or chunking logic. It can focus on the actual task.

```python
# Agent just imports and uses - no boilerplate needed
from tools.llm_client import LLMClient
from tools.chunking import chunk_prompt

client = LLMClient()
for chunk in chunk_prompt(max_tokens=2000):
    result = client.call(f"Summarize: {chunk.content}")
    # Caching, logging, metrics all handled automatically
```

### 2. Skills (Teaching the Agent How to Script)

The `.claude/skills/rlm-scripting/SKILL.md` teaches the agent **how to write scripts for this specific environment**:

- **Naming conventions**: `001_scan.py`, `002_extract.py`, etc.
- **Output discipline**: Write to files, print only summaries
- **Where to store what**: evidence → `state/evidence.jsonl`, cache → `cache/`, etc.
- **How to use the tools**: Example code for `LLMClient`, `Chunker`
- **Typical workflow**: scan → chunk → extract → synthesize → verify
- **Budget awareness**: How to check limits in `job.json`

**Why?** Without this guidance, the agent might:
- Print huge outputs to stdout (clogs context)
- Not use caching (wastes API calls)
- Store files in wrong locations
- Miss budget constraints

The skill acts as a **tutorial** the agent can reference, ensuring consistent, efficient behavior.

### 3. Filesystem-Based Intermediate State (vs REPL Variables)

This is a fundamental difference from the original RLM:

**Original RLM (REPL variables):**
```python
# Intermediate results live in Python memory
chunks = context.split('\n\n')
summaries = []
for chunk in chunks:
    summaries.append(llm_query(f"Summarize: {chunk}"))
# If the process crashes here, summaries is lost
final = synthesize(summaries)
```

**fsRLM (filesystem):**
```python
# Intermediate results written to files
for i, chunk in enumerate(chunks):
    result = client.call(f"Summarize: {chunk}")
    # Result automatically logged to state/evidence.jsonl
    # AND cached to cache/llm/<hash>.json

# If process crashes, evidence.jsonl persists
# Re-running uses cache - no repeated API calls
```

**Key benefits of filesystem-based state:**

| Aspect | REPL Variables | Filesystem |
|--------|---------------|------------|
| Crash recovery | Lost | Preserved in files |
| Debugging | Print statements | Inspect actual files |
| Caching | Must implement | Built into llm_client |
| Observability | Limited | Watch files in real-time |
| Auditability | Trajectory logs | Full evidence trail |
| Pause/resume | Complex | Natural (stop/start) |

**Evidence as structured data:**

Instead of summaries sitting in a Python list, they're appended to `state/evidence.jsonl`:

```json
{"timestamp": "2024-01-15T10:30:00", "tag": "summary_chunk_0", "content": "...", "metadata": {...}}
{"timestamp": "2024-01-15T10:30:05", "tag": "summary_chunk_1", "content": "...", "metadata": {...}}
{"timestamp": "2024-01-15T10:30:10", "tag": "summary_chunk_2", "content": "...", "metadata": {...}}
```

This creates an **audit trail** of all extracted information, tagged and timestamped, that:
- Survives crashes
- Can be inspected mid-run
- Enables the synthesis step to read all evidence
- Provides transparency into what the agent learned

### 4. Standard Unix Tools for Exploration

In the original RLM, the model interacts with `context` through Python string operations. In fsRLM, the agent has access to **standard Unix tools** via the Agent SDK's Bash/Read/Grep/Glob tools:

**Original RLM (Python only):**
```python
# Must load into memory or write custom parsing
first_1000_chars = context[:1000]
lines_with_error = [l for l in context.split('\n') if 'error' in l.lower()]
```

**fsRLM (Unix tools + Python):**
```bash
# Peek at first 50 lines without loading file
head -50 input/prompt.md

# Find all lines mentioning "error"
grep -i "error" input/prompt.md

# Count sections
grep -c "^## " input/prompt.md

# Get lines 100-150
sed -n '100,150p' input/prompt.md

# Check file size before deciding strategy
wc -l input/prompt.md
```

**Why this matters:**

The agent can **explore** the input before committing to a processing strategy:

1. **Quick inspection**: `head`, `tail`, `wc` to understand size/structure
2. **Search without loading**: `grep` to find relevant sections
3. **Selective reading**: `sed` to extract specific line ranges
4. **Pattern matching**: `grep -E` for regex searches

This enables **intelligent chunking** - the agent can:
- Count sections (`grep -c "^## "`)
- Find boundaries (`grep -n "^## "` for line numbers)
- Sample content (`head -100`, `tail -100`)
- Search for keywords before deciding what to process

```bash
# Agent's exploration before writing processing script:
$ wc -l input/prompt.md
15847 input/prompt.md

$ grep -c "^## " input/prompt.md
47

$ head -20 input/prompt.md
# Security Audit Report
## Executive Summary
...

# Now agent knows: 15K lines, 47 sections, it's a security report
# Can write targeted extraction script
```

This is a natural fit because **filesystems already have rich tooling** for exploration. The Agent SDK exposes Read, Grep, Glob, and Bash - giving the agent the same tools a human developer would use.

## Complete Flow

### Step 1: User Invokes fsRLM

```python
from fsrlm import RLM, RLMConfig

config = RLMConfig(
    max_tokens_per_subcall=1000,  # Budget for recursive calls
    max_depth=5,
    submodel="claude-haiku-4-20250414",  # Fast model for subcalls
)

rlm = RLM(config=config)
result = rlm.invoke("Analyze this 500-page document and find all security vulnerabilities...")
```

### Step 2: Workspace Creation (`workspace.py`)

The `Workspace` class creates a structured directory:

```
/tmp/fsrlm_abc123/
├── input/
│   ├── prompt.md           # User's input written here
│   └── attachments/
├── state/
│   ├── job.json            # Configuration + budget limits
│   ├── notes.md            # Agent's working notes
│   ├── evidence.jsonl      # Extracted facts (line-delimited JSON)
│   ├── errors.jsonl        # Error log
│   └── metrics.json        # Token/call counts
├── cache/
│   ├── llm/                # Cached API responses
│   └── indexes/            # Chunk manifests
├── scratch/
│   ├── scripts/            # Agent-generated Python scripts
│   └── tmp/
├── output/
│   └── answer.md           # Final answer goes here
├── tools/
│   ├── llm_client.py       # Claude API wrapper
│   ├── chunking.py         # Text chunking utilities
│   └── README.md
└── .claude/
    ├── CLAUDE.md           # Agent instructions
    └── skills/
        └── rlm-scripting/
            └── SKILL.md    # Script-writing guidance
```

**job.json contents:**
```json
{
  "prompt_path": "input/prompt.md",
  "answer_path": "output/answer.md",
  "config": {
    "max_tokens_per_call": 1000,
    "max_depth": 5,
    "max_scripts": 20,
    "max_subcalls_per_script": 25,
    "submodel": "claude-haiku-4-20250414",
    "cache_responses": true
  }
}
```

### Step 3: Agent Runner Setup (`runner.py`)

The `AgentRunner` configures the Claude Agent SDK:

```python
options = ClaudeAgentOptions(
    cwd=str(workspace_path),           # Working directory
    allowed_tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob", "Skill"],
    system_prompt=RLM_SYSTEM_PROMPT,   # Additional instructions
    max_turns=50,
    permission_mode="acceptEdits",     # Auto-approve file operations
    setting_sources=["project"],       # Load .claude/CLAUDE.md and skills
)
```

### Step 4: Initial Prompt Construction (`runner.py`)

The runner builds an initial prompt based on input size:

**For small inputs (<8KB)** - content is inlined:
```
Here is the user's request:

---
<full prompt content>
---

Complete this request. Write your final answer to output/answer.md.

If this requires complex processing, you can:
- Write scripts to scratch/scripts/
- Use tools/llm_client.py for sub-queries
- Store intermediate results in state/

For simple requests, just solve it directly.
```

**For large inputs (>8KB)** - only a preview is shown:
```
The user's request is in input/prompt.md (125KB, too large to include here).

Preview:
---
<first 500 characters>...
---

To process this large input:
1. Read state/job.json for configuration and budget limits
2. Use tools/chunking.py to split the content
3. Use tools/llm_client.py for sub-queries on chunks
4. Store intermediate results in state/evidence.jsonl
5. Write your final answer to output/answer.md

Begin by scanning the input to understand its structure.
```

### Step 5: Agent Execution (Claude Agent SDK)

The agent runs with these inputs:

1. **System Prompt** (from runner):
```
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
```

2. **CLAUDE.md** (loaded via `setting_sources=["project"]`):
   - Workspace layout explanation
   - Mandatory rules (never read huge files into context, use scripts)
   - Budget constraints
   - Success criteria

3. **Skills** (loaded via `setting_sources=["project"]`):
   - `rlm-scripting`: How to write scripts, use llm_client, handle chunking

### Step 6: Agent Behavior

**For simple tasks**, the agent answers directly:
```
Agent reads input/prompt.md
Agent writes answer to output/answer.md
Done.
```

**For complex tasks**, the agent writes scripts:

```python
# Agent creates: scratch/scripts/001_scan.py
#!/usr/bin/env python3
"""Scan input and build index."""
from tools.chunking import Chunker

chunker = Chunker()
index = chunker.build_index("input/prompt.md", index_type="headings")
print(f"Found {len(index['entries'])} sections")
```

```python
# Agent creates: scratch/scripts/002_extract.py
#!/usr/bin/env python3
"""Extract facts from each section."""
from tools.llm_client import LLMClient
from tools.chunking import chunk_prompt

client = LLMClient()
chunks = chunk_prompt(max_tokens=2000)

for chunk in chunks:
    result = client.call(
        prompt=f"Extract key facts:\n\n{chunk.content}",
        log_to_evidence=True,
        evidence_tag=f"facts_chunk_{chunk.index}",
    )
    if result:
        print(f"Processed chunk {chunk.index}")
```

```python
# Agent creates: scratch/scripts/003_synthesize.py
#!/usr/bin/env python3
"""Synthesize evidence into final answer."""
import json

# Read all evidence
evidence = []
with open("state/evidence.jsonl") as f:
    for line in f:
        evidence.append(json.loads(line))

# Write final answer
with open("output/answer.md", "w") as f:
    f.write("# Analysis Results\n\n")
    for item in evidence:
        f.write(f"- {item['content']}\n")

print("Answer written to output/answer.md")
```

### Step 7: Recursive Calls (`tools/llm_client.py`)

When scripts call `client.call()`, the LLMClient:

1. **Checks cache** (`cache/llm/<hash>.json`)
2. **Makes API call** to the submodel (Haiku by default)
3. **Caches response** for future runs
4. **Updates metrics** (`state/metrics.json`)
5. **Logs errors** (`state/errors.jsonl`)
6. **Optionally logs evidence** (`state/evidence.jsonl`)

```python
# What happens inside client.call():
cache_key = hash(model + system + prompt + max_tokens)
if cache_key in cache:
    return cached_response  # No API call!

response = anthropic.messages.create(
    model="claude-haiku-4-20250414",  # Fast/cheap
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}],
)

save_to_cache(cache_key, response)
update_metrics(tokens_used)
if log_to_evidence:
    append_to_evidence(response, tag)

return response
```

### Step 8: Result Collection (`runner.py`)

After the agent finishes:

```python
# Read answer
answer = (workspace / "output" / "answer.md").read_text()

# Read metrics
metrics = json.load(open(workspace / "state" / "metrics.json"))

# Read errors
errors = [json.loads(line) for line in open(workspace / "state" / "errors.jsonl")]

# Read evidence
evidence = [json.loads(line) for line in open(workspace / "state" / "evidence.jsonl")]

return RLMResult(
    answer=answer,
    metrics=metrics,
    errors=errors,
    evidence=evidence,
    success=answer is not None,
)
```

## Why Filesystem-Based?

| Aspect | REPL-based (Original) | Filesystem-based (fsRLM) |
|--------|----------------------|--------------------------|
| State persistence | Lost on crash | Survives crashes |
| Debugging | Trajectory logs | Real files you can inspect |
| Caching | Must implement | Built-in with file cache |
| Observability | Limited | Watch files in real-time |
| Pause/resume | Difficult | Natural (just stop/start) |
| Tool ecosystem | Custom | Standard file tools |

## Prompt Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ USER                                                                │
│   rlm.invoke("Analyze 500 pages...")                               │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ WORKSPACE CREATION                                                  │
│   - Create temp directory structure                                 │
│   - Write prompt to input/prompt.md                                 │
│   - Copy tools (llm_client.py, chunking.py)                        │
│   - Copy .claude/CLAUDE.md and skills                              │
│   - Write config to state/job.json                                 │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ AGENT SDK INITIALIZATION                                            │
│   ClaudeAgentOptions(                                               │
│     cwd = workspace_path,                                          │
│     setting_sources = ["project"],  ← Loads CLAUDE.md + skills     │
│     system_prompt = RLM instructions,                              │
│     allowed_tools = [Bash, Read, Write, Edit, ...],                │
│   )                                                                 │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ INITIAL PROMPT TO AGENT                                             │
│   "The user's request is in input/prompt.md (125KB)..."            │
│   "Use tools/llm_client.py for sub-queries..."                     │
│   "Write final answer to output/answer.md"                         │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ AGENT EXECUTION LOOP                                                │
│                                                                     │
│   Agent reads: .claude/CLAUDE.md (workspace rules)                 │
│   Agent reads: state/job.json (budget limits)                      │
│   Agent reads: input/prompt.md (preview or chunks)                 │
│                                                                     │
│   Agent writes: scratch/scripts/001_scan.py                        │
│   Agent runs:   python scratch/scripts/001_scan.py                 │
│                                                                     │
│   Agent writes: scratch/scripts/002_extract.py                     │
│   Agent runs:   python scratch/scripts/002_extract.py              │
│       └── Script calls: tools/llm_client.py                        │
│           └── LLM calls Haiku for each chunk                       │
│           └── Results cached to cache/llm/                         │
│           └── Evidence logged to state/evidence.jsonl              │
│                                                                     │
│   Agent writes: scratch/scripts/003_synthesize.py                  │
│   Agent runs:   python scratch/scripts/003_synthesize.py           │
│       └── Reads state/evidence.jsonl                               │
│       └── Writes output/answer.md                                  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ RESULT COLLECTION                                                   │
│   - Read output/answer.md → answer                                 │
│   - Read state/metrics.json → metrics                              │
│   - Read state/errors.jsonl → errors                               │
│   - Read state/evidence.jsonl → evidence                           │
│                                                                     │
│   Return RLMResult(answer, metrics, errors, evidence)              │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ USER                                                                │
│   print(result.answer)                                             │
│   print(result.metrics)  # {"subcalls_made": 47, "tokens": 52000}  │
└────────────────────────────────────────────────────────────────────┘
```

## Key Insight: The Agent SDK is the "REPL"

In the original RLM paper, the Python REPL:
- Receives code from the LLM
- Executes it
- Returns results
- Repeats until `answer` is set

In fsRLM, the Claude Agent SDK:
- Receives tool calls from the LLM (Write, Bash, Read, etc.)
- Executes them in the workspace
- Returns results
- Repeats until `output/answer.md` exists

The Agent SDK effectively **is** the REPL, but with:
- File operations instead of variable assignments
- Bash execution instead of `exec()`
- Structured workspace instead of namespace

## Bedrock Support

fsRLM supports AWS Bedrock as an alternative to direct Anthropic API:

```python
config = RLMConfig(
    use_bedrock=True,
    bedrock_model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",      # Root agent
    bedrock_submodel="us.anthropic.claude-haiku-4-20250414-v1:0",      # Subcalls
)
```

When Bedrock is enabled:
1. Runner sets `CLAUDE_CODE_USE_BEDROCK=1` environment variable
2. LLMClient creates `AnthropicBedrock()` client instead of `Anthropic()`
3. Both use inference profile model format

## Summary

fsRLM faithfully implements the RLM paper's core insight—treating prompts as an explorable environment—while adding practical benefits:

1. **Same concept**: Prompt as environment, recursive decomposition
2. **Different implementation**: Filesystem instead of REPL variables
3. **Added benefits**: Persistence, caching, observability, debugging
4. **Same result**: Arbitrarily long inputs processed through recursive calls

The filesystem IS the working memory. The Agent SDK IS the REPL. The scripts ARE the model-generated code. `input/prompt.md` IS the `context` variable. `output/answer.md` IS the final answer.
