"""Tests for semaphore-based rate limiter."""

import asyncio

import pytest

from async_tools.rate_limit.semaphore_limiter import SemaphoreRateLimiter


@pytest.mark.unit
@pytest.mark.asyncio
class TestSemaphoreRateLimiter:
    """Test semaphore-based rate limiter."""

    async def test_basic_acquire_release(self):
        """Test basic acquire and release."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)

        await limiter.acquire()
        await limiter.release()

    async def test_context_manager(self):
        """Test context manager usage."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)
        call_count = 0

        async with limiter:
            call_count += 1

        assert call_count == 1

    async def test_context_manager_with_exception(self):
        """Test context manager releases on exception."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)
        call_count = 0

        with pytest.raises(ValueError):
            async with limiter:
                call_count += 1
                raise ValueError("test error")

        assert call_count == 1

        # Should be able to acquire again after exception
        async with limiter:
            call_count += 1

        assert call_count == 2

    async def test_decorator(self):
        """Test decorator usage."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)
        call_count = 0

        @limiter
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await test_func()
        assert result == "result"
        assert call_count == 1

    async def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)

        @limiter
        async def test_func():
            """Test docstring."""
            pass

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test docstring."

    async def test_decorator_with_arguments(self):
        """Test decorator with function arguments."""
        limiter = SemaphoreRateLimiter(max_concurrent=2)

        @limiter
        async def test_func(x: int, y: int) -> int:
            return x + y

        result = await test_func(5, 3)
        assert result == 8

    async def test_invalid_max_concurrent(self):
        """Test that invalid max_concurrent raises ValueError."""
        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            SemaphoreRateLimiter(max_concurrent=0)

        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            SemaphoreRateLimiter(max_concurrent=-1)

    async def test_single_concurrency_limit(self):
        """Test that only one operation runs at a time with max_concurrent=1."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)

        running = []
        max_concurrent_seen = 0

        async def operation(i):
            nonlocal max_concurrent_seen
            async with limiter:
                running.append(i)
                max_concurrent_seen = max(max_concurrent_seen, len(running))
                await asyncio.sleep(0.01)  # Small delay
                running.remove(i)

        # Run 10 operations
        await asyncio.gather(*[operation(i) for i in range(10)])

        # Should never have more than 1 concurrent
        assert max_concurrent_seen == 1

    async def test_multiple_concurrency_limit(self):
        """Test that exact number of operations run concurrently."""
        max_concurrent = 5
        limiter = SemaphoreRateLimiter(max_concurrent=max_concurrent)

        running = []
        max_concurrent_seen = 0

        async def operation(i):
            nonlocal max_concurrent_seen
            async with limiter:
                running.append(i)
                max_concurrent_seen = max(max_concurrent_seen, len(running))
                await asyncio.sleep(0.01)  # Small delay
                running.remove(i)

        # Run 20 operations
        await asyncio.gather(*[operation(i) for i in range(20)])

        # Should see exactly max_concurrent operations at once
        assert max_concurrent_seen == max_concurrent

    async def test_operations_queue_properly(self):
        """Test that operations queue when limit is reached."""
        limiter = SemaphoreRateLimiter(max_concurrent=2)

        started = []
        completed = []

        async def operation(i):
            async with limiter:
                started.append(i)
                await asyncio.sleep(0.02)  # Hold the slot
                completed.append(i)

        # Start 5 operations
        await asyncio.gather(*[operation(i) for i in range(5)])

        # All should complete
        assert len(started) == 5
        assert len(completed) == 5
        assert set(started) == set(range(5))
        assert set(completed) == set(range(5))

    async def test_release_enables_new_operations(self):
        """Test that releasing allows queued operations to proceed."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)

        operation_started = asyncio.Event()
        can_complete = asyncio.Event()

        async def blocking_operation():
            async with limiter:
                operation_started.set()
                await can_complete.wait()

        async def queued_operation():
            async with limiter:
                return "completed"

        # Start blocking operation
        task1 = asyncio.create_task(blocking_operation())
        await operation_started.wait()

        # Start queued operation (will wait)
        task2 = asyncio.create_task(queued_operation())
        await asyncio.sleep(0.01)  # Give it time to queue

        # Task2 should not be done yet
        assert not task2.done()

        # Allow first operation to complete
        can_complete.set()
        await task1

        # Now task2 should complete
        result = await task2
        assert result == "completed"

    async def test_high_concurrency(self):
        """Test high concurrency with many operations."""
        limiter = SemaphoreRateLimiter(max_concurrent=10)

        completed = []

        async def operation(i):
            async with limiter:
                completed.append(i)
                await asyncio.sleep(0.001)

        # Run 100 operations
        await asyncio.gather(*[operation(i) for i in range(100)])

        assert len(completed) == 100
        assert set(completed) == set(range(100))

    async def test_concurrent_acquire_and_release(self):
        """Test concurrent acquire and release operations."""
        limiter = SemaphoreRateLimiter(max_concurrent=5)

        running = []
        max_concurrent = 0

        async def operation(i):
            nonlocal max_concurrent
            await limiter.acquire()
            try:
                running.append(i)
                max_concurrent = max(max_concurrent, len(running))
                await asyncio.sleep(0.01)
                running.remove(i)
            finally:
                await limiter.release()

        # Run 50 operations
        await asyncio.gather(*[operation(i) for i in range(50)])

        assert max_concurrent == 5

    async def test_semaphore_fairness(self):
        """Test that semaphore queuing is fair."""
        limiter = SemaphoreRateLimiter(max_concurrent=1)

        order = []

        async def operation(i):
            async with limiter:
                order.append(i)
                await asyncio.sleep(0.001)

        # Start operations in sequence
        tasks = [asyncio.create_task(operation(i)) for i in range(10)]
        await asyncio.gather(*tasks)

        # All operations should complete
        assert len(order) == 10
        assert set(order) == set(range(10))

    async def test_large_burst(self):
        """Test handling large burst with controlled concurrency."""
        limiter = SemaphoreRateLimiter(max_concurrent=20)

        running = []
        max_concurrent_seen = 0

        async def operation(i):
            nonlocal max_concurrent_seen
            async with limiter:
                running.append(i)
                max_concurrent_seen = max(max_concurrent_seen, len(running))
                await asyncio.sleep(0.001)
                running.remove(i)

        # Large burst of 1000 operations
        await asyncio.gather(*[operation(i) for i in range(1000)])

        # Should limit to exactly 20 concurrent
        assert max_concurrent_seen <= 20

    async def test_multiple_limiters_independent(self):
        """Test that multiple limiter instances are independent."""
        limiter1 = SemaphoreRateLimiter(max_concurrent=2)
        limiter2 = SemaphoreRateLimiter(max_concurrent=2)

        running1 = []
        running2 = []

        async def operation1(i):
            async with limiter1:
                running1.append(i)
                await asyncio.sleep(0.02)
                running1.remove(i)

        async def operation2(i):
            async with limiter2:
                running2.append(i)
                await asyncio.sleep(0.02)
                running2.remove(i)

        # Run operations on both limiters concurrently
        await asyncio.gather(
            *[operation1(i) for i in range(10)],
            *[operation2(i) for i in range(10)],
        )

        # Both should have processed all operations independently
        # (not limiting each other)

    async def test_reusable_after_exception(self):
        """Test limiter is reusable after exception in operation."""
        limiter = SemaphoreRateLimiter(max_concurrent=2)

        call_count = 0

        @limiter
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("test error")

        @limiter
        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        # First operation fails
        with pytest.raises(ValueError):
            await failing_operation()

        # Should still be able to use limiter
        result = await successful_operation()
        assert result == "success"
        assert call_count == 2
