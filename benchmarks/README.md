# fsRLM Benchmarks

Benchmarks for evaluating fsRLM on long-context reasoning tasks.

## OOLONG Benchmark

[OOLONG](https://arxiv.org/abs/2511.02817) (Out-of-Length Ordered Natural Generation) evaluates long-context reasoning and aggregation capabilities.

### Dataset

- **OOLONG-synth**: Synthetic tasks from classification datasets (TREC, AGNews, etc.)
- **OOLONG-real**: Real-world conversational data reasoning

Both are available on HuggingFace:
- `oolongbench/oolong-synth`
- `oolongbench/oolong-real`

### Installation

```bash
# Install benchmark dependencies
uv pip install -e ".[benchmarks]"
```

### Usage

```bash
# Run on OOLONG-synth (default)
uv run python -m benchmarks.oolong --limit 10

# Run specific task
uv run python -m benchmarks.oolong --split synth --task trec_coarse --limit 20

# Run on OOLONG-real
uv run python -m benchmarks.oolong --split real --limit 10

# Full benchmark (warning: expensive!)
uv run python -m benchmarks.oolong --split synth
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--split` | Which split: `synth` or `real` | `synth` |
| `--task` | Specific task name (synth only) | all |
| `--limit` | Max tasks to run | unlimited |
| `--max-tokens` | Tokens per subcall | 800 |
| `--max-scripts` | Scripts per task | 15 |
| `--output-dir` | Results directory | `benchmarks/results` |
| `--quiet` | Suppress progress | false |

### Scoring

OOLONG uses task-specific scoring:
- **Numerical answers**: `0.75^|predicted - expected|`
- **Categorical answers**: Exact match (1.0 or 0.0)

### Results

Results are saved to `benchmarks/results/` as JSON files containing:
- Per-task scores and predictions
- Aggregate metrics (avg score, latency, token usage)
- Configuration used

### Comparison with Paper Results

The RLM paper reports these results on OOLONG:

| Model | Direct | Summarize | RLM |
|-------|--------|-----------|-----|
| GPT-5 | 44.0% | 47.3% | **56.5%** |
| GPT-5-mini | 38.2% | 41.1% | **49.8%** |

fsRLM aims to achieve similar gains using a filesystem-based approach.

## Adding New Benchmarks

To add a new benchmark:

1. Create `benchmarks/your_benchmark.py`
2. Implement task loading and scoring
3. Use the `RLM` class to run tasks
4. Save results in consistent format

See `oolong.py` for reference implementation.
