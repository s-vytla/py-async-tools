"""Integration tests for retry decorator with realistic failure scenarios."""

import asyncio
from collections.abc import Callable

import pytest

from async_tools.retry import retry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_with_flaky_network() -> None:
    """Simulate network calls that fail then succeed."""
    call_count = 0

    @retry(max_attempts=5, initial_delay=0.1, exponential_base=2.0)
    async def flaky_network_call() -> str:
        nonlocal call_count
        call_count += 1
        # Simulate slow network
        await asyncio.sleep(0.05)
        # Fail first 3 times
        if call_count <= 3:
            raise ConnectionError("Network unreachable")
        return "success"

    start = asyncio.get_event_loop().time()
    result = await flaky_network_call()
    elapsed = asyncio.get_event_loop().time() - start

    assert result == "success"
    assert call_count == 4
    # Should take at least 0.1 + 0.2 = 0.3s for retries + operation time
    assert elapsed >= 0.3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_exhaustion_permanent_failure() -> None:
    """Test retry gives up after max_attempts with permanent failures."""
    call_count = 0

    @retry(max_attempts=3, initial_delay=0.05, exponential_base=2.0)
    async def always_fails() -> None:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.02)
        raise TimeoutError("Service timeout")

    start = asyncio.get_event_loop().time()

    with pytest.raises(TimeoutError, match="Service timeout"):
        await always_fails()

    elapsed = asyncio.get_event_loop().time() - start

    assert call_count == 3
    # Should take at least 0.05 + 0.1 = 0.15s for retries
    assert elapsed >= 0.15
    # Should be less than 0.5s (with operation overhead)
    assert elapsed < 0.5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_callback_logging() -> None:
    """Test retry callback receives correct parameters during retries."""
    retry_log: list[tuple[type[Exception], int, float]] = []

    def log_retry(exc: Exception, attempt: int, delay: float) -> None:
        retry_log.append((type(exc), attempt, delay))

    @retry(
        max_attempts=4,
        initial_delay=0.1,
        exponential_base=2.0,
        on_retry=log_retry,
    )
    async def flaky_service() -> str:
        await asyncio.sleep(0.02)
        if len(retry_log) < 2:
            raise ValueError("Service error")
        return "recovered"

    result = await flaky_service()

    assert result == "recovered"
    assert len(retry_log) == 2

    # Verify callback parameters
    assert retry_log[0] == (ValueError, 2, 0.1)
    assert retry_log[1] == (ValueError, 3, 0.2)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_with_different_exception_types() -> None:
    """Test retry only retries on specified exception types."""
    call_count = 0

    @retry(
        max_attempts=5,
        initial_delay=0.05,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def selective_retry() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.02)

        if call_count == 1:
            raise ConnectionError("Connection failed")
        elif call_count == 2:
            raise TimeoutError("Request timeout")
        elif call_count == 3:
            # This should not be retried
            raise ValueError("Invalid response")

        return "success"

    with pytest.raises(ValueError, match="Invalid response"):
        await selective_retry()

    # Should have attempted 3 times before hitting ValueError
    assert call_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_retry_max_delay_cap() -> None:
    """Test that retry respects max_delay cap with real timing."""
    call_count = 0
    retry_times: list[float] = []

    @retry(
        max_attempts=5,
        initial_delay=0.1,
        exponential_base=2.0,
        max_delay=0.2,
    )
    async def capped_delay() -> None:
        nonlocal call_count
        call_count += 1
        retry_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.01)
        raise RuntimeError("Still failing")

    start = asyncio.get_event_loop().time()

    with pytest.raises(RuntimeError, match="Still failing"):
        await capped_delay()

    elapsed = asyncio.get_event_loop().time() - start

    assert call_count == 5

    # Calculate actual delays between retries
    if len(retry_times) >= 2:
        # First retry should be ~0.1s
        delay1 = retry_times[1] - retry_times[0]
        assert 0.08 <= delay1 <= 0.15

        # Second retry should be capped at ~0.2s (not 0.4s)
        if len(retry_times) >= 3:
            delay2 = retry_times[2] - retry_times[1]
            assert 0.18 <= delay2 <= 0.25


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_with_async_context() -> None:
    """Test retry works correctly with async context managers."""
    call_count = 0

    class FlakyResource:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        async def __aenter__(self) -> "FlakyResource":
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.02)
            if call_count <= 2:
                raise ConnectionError("Resource unavailable")
            self.entered = True
            return self

        async def __aexit__(self, *args: object) -> None:
            self.exited = True
            await asyncio.sleep(0.01)

    @retry(max_attempts=4, initial_delay=0.05)
    async def use_resource() -> str:
        async with FlakyResource() as resource:
            assert resource.entered
            return "success"

    result = await use_resource()

    assert result == "success"
    assert call_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_retries() -> None:
    """Test multiple concurrent operations with retry."""
    call_counts: dict[int, int] = {}

    @retry(max_attempts=3, initial_delay=0.05, exponential_base=1.5)
    async def operation(op_id: int) -> str:
        if op_id not in call_counts:
            call_counts[op_id] = 0
        call_counts[op_id] += 1

        await asyncio.sleep(0.02)

        # Fail on first attempt for odd IDs
        if op_id % 2 == 1 and call_counts[op_id] == 1:
            raise ConnectionError(f"Operation {op_id} failed")

        return f"op_{op_id}_success"

    start = asyncio.get_event_loop().time()

    # Launch 10 concurrent operations
    tasks = [operation(i) for i in range(10)]
    completed_results = await asyncio.gather(*tasks)

    elapsed = asyncio.get_event_loop().time() - start

    # All operations should eventually succeed
    assert len(completed_results) == 10
    assert all("success" in r for r in completed_results)

    # Odd IDs should have retried once
    for i in range(1, 10, 2):
        assert call_counts[i] == 2

    # Even IDs should succeed on first try
    for i in range(0, 10, 2):
        assert call_counts[i] == 1

    # Should complete relatively quickly (concurrent, not sequential)
    # Even with retries, concurrent execution should be faster than 1 second
    assert elapsed < 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_preserves_exception_chain() -> None:
    """Test that retry preserves the exception chain on final failure."""

    @retry(max_attempts=2, initial_delay=0.05)
    async def fails_with_chain() -> None:
        await asyncio.sleep(0.01)
        try:
            raise ValueError("Original error")
        except ValueError as e:
            raise ConnectionError("Wrapped error") from e

    with pytest.raises(ConnectionError, match="Wrapped error") as exc_info:
        await fails_with_chain()

    # Verify exception chain is preserved
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "Original error"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_with_successful_first_attempt() -> None:
    """Test that successful operations complete quickly without delays."""
    call_count = 0

    @retry(max_attempts=5, initial_delay=1.0)
    async def immediate_success() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return "success"

    start = asyncio.get_event_loop().time()
    result = await immediate_success()
    elapsed = asyncio.get_event_loop().time() - start

    assert result == "success"
    assert call_count == 1
    # Should complete quickly without retry delays
    assert elapsed < 0.1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_zero_delay() -> None:
    """Test retry with zero delay retries immediately."""
    call_count = 0

    @retry(max_attempts=3, initial_delay=0.0)
    async def fast_retry() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        if call_count < 3:
            raise ConnectionError("Retry")
        return "success"

    start = asyncio.get_event_loop().time()
    result = await fast_retry()
    elapsed = asyncio.get_event_loop().time() - start

    assert result == "success"
    assert call_count == 3
    # Should complete very quickly with zero delay
    assert elapsed < 0.1
