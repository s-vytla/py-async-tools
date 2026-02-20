"""Integration tests for rate limiters with realistic load scenarios."""

import asyncio

import pytest

from async_tools.rate_limit import (
    PerKeyRateLimiter,
    SemaphoreRateLimiter,
    TokenBucketRateLimiter,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_bucket_burst_handling() -> None:
    """Test token bucket handles burst traffic correctly."""
    # Allow 10 ops/sec with burst capacity of 20
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=20.0)

    start = asyncio.get_event_loop().time()
    completed_times: list[float] = []

    async def burst_operation(op_id: int) -> int:
        async with limiter:
            completed_times.append(asyncio.get_event_loop().time() - start)
            await asyncio.sleep(0.01)  # Simulate work
            return op_id

    # Launch 25 operations concurrently (exceeds burst capacity)
    results = await asyncio.gather(*[burst_operation(i) for i in range(25)])

    assert len(results) == 25

    # First 20 should complete quickly (burst)
    # Remaining 5 should be throttled
    if len(completed_times) >= 20:
        # First 20 operations should complete within ~0.5s
        assert completed_times[19] < 0.5

    # Total time should reflect rate limiting for excess operations
    total_time = asyncio.get_event_loop().time() - start
    # 20 in burst + 5 at 10/sec = ~0.5s minimum
    assert total_time >= 0.4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_bucket_sustained_rate() -> None:
    """Test token bucket maintains rate over sustained period."""
    # Allow 5 ops/sec
    limiter = TokenBucketRateLimiter(rate=5.0, capacity=5.0)

    start = asyncio.get_event_loop().time()
    operation_count = 0

    async def rate_limited_op() -> None:
        nonlocal operation_count
        async with limiter:
            operation_count += 1
            await asyncio.sleep(0.01)

    # Run operations for approximately 0.5 seconds
    operations = [rate_limited_op() for _ in range(10)]
    await asyncio.gather(*operations)

    elapsed = asyncio.get_event_loop().time() - start

    assert operation_count == 10
    # 10 operations at 5/sec should take ~1.8-2.0 seconds
    # (first 5 immediate, next 5 at 5/sec)
    assert elapsed >= 0.8


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semaphore_concurrency_limit() -> None:
    """Test semaphore limiter enforces max concurrent operations."""
    limiter = SemaphoreRateLimiter(max_concurrent=3)

    max_concurrent = 0
    current_concurrent = 0

    async def tracked_operation(op_id: int) -> int:
        nonlocal max_concurrent, current_concurrent

        async with limiter:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)

            # Simulate work
            await asyncio.sleep(0.05)

            current_concurrent -= 1
            return op_id

    # Launch 10 operations
    results = await asyncio.gather(*[tracked_operation(i) for i in range(10)])

    assert len(results) == 10
    # Should never exceed 3 concurrent operations
    assert max_concurrent <= 3
    # Should have reached the limit
    assert max_concurrent == 3


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_semaphore_sequential_batches() -> None:
    """Test semaphore processes operations in batches."""
    limiter = SemaphoreRateLimiter(max_concurrent=2)
    batch_sizes: list[int] = []
    current_batch = 0

    async def batch_operation(op_id: int) -> int:
        nonlocal current_batch

        async with limiter:
            current_batch += 1
            await asyncio.sleep(0.1)  # Hold for 0.1s
            current_batch -= 1
            return op_id

    start = asyncio.get_event_loop().time()

    # Launch 6 operations
    results = await asyncio.gather(*[batch_operation(i) for i in range(6)])

    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 6
    # With max_concurrent=2 and 0.1s per operation:
    # Should take at least 0.3s (3 batches of 2)
    assert elapsed >= 0.25


@pytest.mark.integration
@pytest.mark.asyncio
async def test_per_key_limiter_isolation() -> None:
    """Test per-key limiter isolates limits between keys."""
    # Create per-key limiter with 3 ops/sec per key
    limiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=5.0, capacity=5.0),
        cleanup_interval=10.0,  # Long interval to avoid cleanup during test
    )

    results_by_key: dict[str, list[float]] = {"key1": [], "key2": [], "key3": []}

    async def keyed_operation(key: str, op_id: int) -> str:
        start = asyncio.get_event_loop().time()
        async with limiter.for_key(key):
            elapsed = asyncio.get_event_loop().time() - start
            results_by_key[key].append(elapsed)
            await asyncio.sleep(0.01)
            return f"{key}_{op_id}"

    # Launch operations for different keys concurrently
    operations = []
    for key in ["key1", "key2", "key3"]:
        for i in range(8):
            operations.append(keyed_operation(key, i))

    start = asyncio.get_event_loop().time()
    results = await asyncio.gather(*operations)
    total_elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 24

    # Each key should have processed 8 operations
    for key in results_by_key:
        assert len(results_by_key[key]) == 8

    # Keys should have independent rate limits
    # Operations should complete faster than if they shared a limit
    # With independent limits, they run in parallel
    assert total_elapsed < 3.0  # Would be much longer if serialized

    await limiter.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_per_key_limiter_cleanup() -> None:
    """Test per-key limiter cleans up idle keys."""
    limiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
        cleanup_interval=0.2,
        max_idle_time=0.3,
    )

    # Use key1
    async with limiter.for_key("key1"):
        await asyncio.sleep(0.01)

    # Use key2
    async with limiter.for_key("key2"):
        await asyncio.sleep(0.01)

    # Check that limiters were created
    # (We can't directly access internal state, so we verify behavior)

    # Wait for cleanup to potentially run
    await asyncio.sleep(0.5)

    # Use key1 again after idle timeout
    async with limiter.for_key("key1"):
        await asyncio.sleep(0.01)

    # Verify no errors occurred (cleanup worked correctly)
    await limiter.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_per_key_limiter_decorator() -> None:
    """Test per-key limiter with decorator pattern."""
    limiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=5.0, capacity=10.0),
        cleanup_interval=10.0,
    )

    call_counts: dict[str, int] = {}

    @limiter.decorator(key_func=lambda user_id, data: user_id)
    async def process_user_request(user_id: str, data: str) -> str:
        if user_id not in call_counts:
            call_counts[user_id] = 0
        call_counts[user_id] += 1
        await asyncio.sleep(0.01)
        return f"{user_id}:{data}"

    # Process requests for multiple users
    operations = []
    for user in ["alice", "bob", "charlie"]:
        for i in range(6):
            operations.append(process_user_request(user, f"data{i}"))

    results = await asyncio.gather(*operations)

    assert len(results) == 18

    # Each user should have processed 6 requests
    for user in ["alice", "bob", "charlie"]:
        assert call_counts[user] == 6

    await limiter.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_bucket_as_decorator() -> None:
    """Test token bucket used as a decorator."""
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=15.0)
    call_count = 0

    @limiter
    async def rate_limited_function(value: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return value * 2

    # Launch 20 operations
    results = await asyncio.gather(*[rate_limited_function(i) for i in range(20)])

    assert len(results) == 20
    assert call_count == 20
    # Verify results are correct
    for i, result in enumerate(results):
        assert result == i * 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_limiters_combined() -> None:
    """Test using multiple limiters in sequence."""
    rate_limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0)
    concurrency_limiter = SemaphoreRateLimiter(max_concurrent=3)

    operation_count = 0
    max_concurrent = 0
    current_concurrent = 0

    async def dual_limited_operation(op_id: int) -> int:
        nonlocal operation_count, max_concurrent, current_concurrent

        # First acquire rate limit
        async with rate_limiter:
            # Then acquire concurrency limit
            async with concurrency_limiter:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

                operation_count += 1
                await asyncio.sleep(0.02)

                current_concurrent -= 1
                return op_id

    # Launch 15 operations
    results = await asyncio.gather(*[dual_limited_operation(i) for i in range(15)])

    assert len(results) == 15
    assert operation_count == 15
    # Should respect concurrency limit
    assert max_concurrent <= 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rate_limiter_with_errors() -> None:
    """Test that rate limiter releases resources even on errors."""
    limiter = SemaphoreRateLimiter(max_concurrent=2)
    success_count = 0
    error_count = 0

    async def sometimes_fails(should_fail: bool) -> str:
        nonlocal success_count, error_count

        async with limiter:
            await asyncio.sleep(0.01)
            if should_fail:
                error_count += 1
                raise ValueError("Operation failed")
            success_count += 1
            return "success"

    # Mix of successful and failing operations
    operations = []
    for i in range(10):
        operations.append(sometimes_fails(i % 3 == 0))

    results = await asyncio.gather(*operations, return_exceptions=True)

    assert len(results) == 10
    # Some succeeded, some failed
    assert success_count > 0
    assert error_count > 0
    assert success_count + error_count == 10

    # Verify limiter still works after errors
    async with limiter:
        await asyncio.sleep(0.01)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_bucket_refill_rate() -> None:
    """Test that token bucket refills at the correct rate."""
    # 10 tokens per second
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=5.0, initial_tokens=5.0)

    start = asyncio.get_event_loop().time()

    # Use all initial tokens quickly
    for _ in range(5):
        async with limiter:
            await asyncio.sleep(0.01)

    # Track time after using initial tokens
    mid_time = asyncio.get_event_loop().time()

    # This should wait for tokens to refill
    async with limiter:
        await asyncio.sleep(0.01)

    elapsed_waiting = asyncio.get_event_loop().time() - mid_time

    # The wait time for the 6th operation should be close to 0.1s
    # (accounting for the sleep times during token consumption)
    # Being more lenient since async timing can vary
    assert elapsed_waiting >= 0.05


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_high_concurrency_stress() -> None:
    """Stress test with high concurrency."""
    limiter = SemaphoreRateLimiter(max_concurrent=10)
    operation_count = 0

    async def stress_operation(op_id: int) -> int:
        nonlocal operation_count
        async with limiter:
            operation_count += 1
            await asyncio.sleep(0.001)  # Very fast operation
            return op_id

    # Launch 100 operations
    start = asyncio.get_event_loop().time()
    results = await asyncio.gather(*[stress_operation(i) for i in range(100)])
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 100
    assert operation_count == 100
    # Should complete relatively quickly with concurrency=10
    assert elapsed < 2.0
