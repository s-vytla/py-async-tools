# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python async tools library targeting Python 3.12+.

## Commands

All commands are available via `./run.sh`:

```bash
./run.sh install_dev    # Install dev dependencies and set up pre-commit hooks
./run.sh lint           # Run ruff linter
./run.sh format         # Format code with ruff
./run.sh type_check     # Run mypy type checking
./run.sh test           # Run tests
./run.sh test_cov       # Run tests with coverage report
./run.sh clean          # Remove generated files and caches
```

Run a single test file:
```bash
pytest tests/unit/test_example.py -v
```

Run tests matching a pattern:
```bash
pytest -k "test_name_pattern" -v
```

## Project Structure

- `src/async_tools/` - Main package source code
- `tests/unit/` - Unit tests (fast, isolated)
- `tests/integration/` - Integration tests (may require external resources)

## Code Quality

- **Linting/Formatting**: Ruff (line length 88, double quotes)
- **Type Checking**: Mypy in strict mode
- **Pre-commit**: Runs ruff and mypy on commit

## Test Markers

```bash
pytest -m unit         # Run unit tests only
pytest -m integration  # Run integration tests only
pytest -m slow         # Run slow tests
pytest -m smoke        # Run smoke tests
```
