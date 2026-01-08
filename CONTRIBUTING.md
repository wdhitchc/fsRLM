# Contributing to fsRLM

Thanks for your interest in contributing to fsRLM!

## Project Status: Early Development

**This project is very new and rapidly evolving.** Expect:

- Significant architectural changes
- Breaking API updates
- Incomplete features
- Rough edges and bugs

I'm actively iterating on the core implementation. Feel free to:
- **Open issues** - Bug reports, questions, and feature ideas are all welcome
- **Submit PRs** - Contributions appreciated, but be aware things may shift underneath you
- **Experiment** - Try it out and share what works or doesn't

This is a work-in-progress, not a stable release. The goal is to get the core concepts right first, then stabilize.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/fsrlm`
3. Install dependencies: `uv venv && uv pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Create virtual environment
uv venv

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run ruff format .
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Type hints are encouraged
- Docstrings for public functions

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Submit PR with clear description

## Areas for Contribution

### High Priority
- [ ] OOLONG benchmark improvements
- [ ] Additional long-context benchmarks
- [ ] Performance optimizations
- [ ] Better error handling and recovery

### Future Features
- [ ] LangChain/LangGraph backend integration
- [ ] Streaming output support
- [ ] Web UI for workspace visualization
- [ ] Multi-agent collaboration

## Reporting Issues

Please include:
- Python version
- fsRLM version
- Minimal reproduction steps
- Expected vs actual behavior

## Questions?

Open a discussion or issue on GitHub.
