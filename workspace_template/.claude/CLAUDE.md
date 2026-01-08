# RLM-FS Project Rules

You are operating in an RLM (Recursive Language Model) workspace. The filesystem IS your working memory.

## Core Principle

**Prompt as environment**: Instead of loading large text into your context window, operate on it through the filesystem using scripts.

## Workspace Layout

```
input/
  prompt.md          # The user's request (may be HUGE)
  attachments/       # Optional files
state/
  job.json           # Configuration and paths
  notes.md           # Your working notes
  evidence.jsonl     # Extracted facts and summaries
  errors.jsonl       # Error log (scripts log here)
  metrics.json       # Token/call counts
cache/
  llm/               # Cached API responses
  indexes/           # Chunk manifests, indexes
scratch/
  scripts/           # Your generated scripts
  tmp/               # Temporary files
output/
  answer.md          # YOUR FINAL DELIVERABLE
tools/
  llm_client.py      # Claude API wrapper with caching
  chunking.py        # Text chunking utilities
```

## Mandatory Rules

1. **Read `state/job.json` first** - it contains your configuration and budget limits
2. **Never read all of `input/prompt.md` into context** if it's large - use scripts to process it
3. **Store intermediate results in files** - use `state/` and `cache/`
4. **Scripts go in `scratch/scripts/`** - name them with incrementing prefixes: `001_scan.py`, `002_extract.py`
5. **Final answer goes in `output/answer.md`** - this is your deliverable
6. **Never write secrets to disk** - API keys come from environment variables only

## Budget Constraints

Check `state/job.json` for:
- `max_tokens_per_call`: Token limit per subcall
- `max_depth`: Max recursion depth
- `max_scripts`: Max scripts you can create
- `max_subcalls_per_script`: Max Claude calls per script

Stay within these limits.

## Output Discipline

- Scripts should print ONLY short summaries to stdout
- Large outputs go to files, then print the filepath
- Check `state/errors.jsonl` if scripts fail

## Success Criteria

Your job is done when:
1. `output/answer.md` exists and answers the user's request
2. No TODOs or placeholders in the answer
3. You stayed within budget (check `state/metrics.json`)
