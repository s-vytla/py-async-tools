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

## Architecture

### Module Organization

The library is organized into subdirectories by functionality:

```
src/async_tools/
├── __init__.py              # Public API exports
├── retry.py                 # Retry decorator utility
├── circuit_breaker/         # Circuit breaker pattern (subdirectory)
│   ├── __init__.py          # Circuit breaker exports
│   └── circuit_breaker.py   # Circuit breaker implementation
└── rate_limit/              # Rate limiting utilities (subdirectory)
    ├── __init__.py          # Rate limit exports
    ├── base.py              # Base classes and protocols
    ├── token_bucket.py      # Token bucket implementation
    ├── semaphore_limiter.py # Semaphore wrapper
    └── per_key.py           # Per-key limiter with cleanup
```

### Resilience Utilities Design Pattern

All resilience utilities (rate limiters and circuit breaker) follow a unified architecture:

1. **Base Class Pattern**: `RateLimiterBase` (in `base.py`) provides decorator and context manager patterns. Subclasses only implement:
   - `acquire()` - Core rate limiting logic (required)
   - `release()` - Resource cleanup (optional, defaults to no-op)

2. **Three Usage Patterns** (all limiters support):
   ```python
   # Context manager (canonical)
   async with limiter:
       await operation()

   # Decorator
   @limiter
   async def my_function():
       await operation()

   # Direct acquire/release
   await limiter.acquire()
   try:
       await operation()
   finally:
       await limiter.release()
   ```

3. **Protocol-Based**: `AsyncRateLimiter` protocol defines the interface for structural typing and duck typing.

### Rate Limiter Types

- **TokenBucketRateLimiter**: Rate-based limiting (e.g., 10 ops/sec with burst capacity). Uses `asyncio.Lock` for concurrency safety and monotonic time from event loop.

- **SemaphoreRateLimiter**: Concurrency limiting (e.g., max 5 concurrent ops). Thin wrapper around `asyncio.Semaphore`. Unlike token bucket, requires explicit `release()`.

- **PerKeyRateLimiter**: Generic wrapper that creates independent limiter instances per key (e.g., per-user limits). Features:
  - Factory pattern - accepts any limiter type
  - Automatic cleanup of idle limiters via background task
  - Decorator with key extraction: `@limiter.decorator(key_func=lambda user_id: user_id)`
  - Must call `await limiter.close()` or use as async context manager

### Circuit Breaker

- **CircuitBreaker**: Prevents cascading failures by detecting repeated failures and temporarily blocking operations. Implements a state machine:
  - **CLOSED**: Normal operation, requests pass through. Failures are counted.
  - **OPEN**: Too many failures occurred. All requests blocked immediately with `CircuitBreakerOpenError`.
  - **HALF_OPEN**: Testing recovery after timeout. Limited requests allowed through.

  State transitions:
  - CLOSED → OPEN: When `failure_count >= failure_threshold`
  - OPEN → HALF_OPEN: After `timeout` seconds
  - HALF_OPEN → CLOSED: On first success
  - HALF_OPEN → OPEN: On any failure

  Key implementation details:
  - Inherits from `RateLimiterBase` to reuse decorator/context manager patterns
  - Overrides `__aexit__()` to track success/failure based on exceptions
  - Uses `record_success()` and `record_failure()` for manual tracking
  - State transitions are atomic using `asyncio.Lock`

### Testing Patterns

Tests use mocking for deterministic timing:
- Mock `asyncio.sleep` to control sleep behavior
- Mock `asyncio.get_event_loop().time()` to control time progression
- Use `pytest.mark.asyncio` for async tests
- Use `pytest.mark.unit` for test categorization

Example from tests:
```python
async def mock_sleep(duration):
    nonlocal current_time
    current_time += duration

with patch("asyncio.sleep", side_effect=mock_sleep):
    # Test rate limiting behavior deterministically
```

### Type Safety

- Full type hints with strict mypy checking
- Generic types for `PerKeyRateLimiter[KeyType, LimiterType]`
- Test files may use `# mypy: disable-error-code="var-annotated"` to silence verbose test variable annotations
- For tests with dynamic state changes, use `# mypy: disable-error-code="comparison-overlap,unreachable"` to silence literal type narrowing issues

### Async Considerations

- Always use `asyncio.Lock` not `threading.Lock` for async concurrency
- Use `asyncio.get_event_loop().time()` for monotonic time
- Background tasks should be cancellable and handle `asyncio.CancelledError`
- Context managers should be exception-safe (resources released even on error)
