"""Tests for token bucket rate limiter."""

import asyncio
from unittest.mock import patch

import pytest

from async_tools.rate_limit.token_bucket import TokenBucketRateLimiter


@pytest.mark.unit
@pytest.mark.asyncio
class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter."""

    async def test_basic_acquire(self):
        """Test basic token acquisition."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0)

        # Should not block with initial tokens
        await limiter.acquire()

    async def test_context_manager(self):
        """Test context manager usage."""
        limiter = TokenBucketRateLimiter(rate=10.0)
        call_count = 0

        async with limiter:
            call_count += 1

        assert call_count == 1

    async def test_context_manager_with_exception(self):
        """Test context manager handles exceptions properly."""
        limiter = TokenBucketRateLimiter(rate=10.0)
        call_count = 0

        with pytest.raises(ValueError):
            async with limiter:
                call_count += 1
                raise ValueError("test error")

        assert call_count == 1

    async def test_decorator(self):
        """Test decorator usage."""
        limiter = TokenBucketRateLimiter(rate=10.0)
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
        limiter = TokenBucketRateLimiter(rate=10.0)

        @limiter
        async def test_func():
            """Test docstring."""
            pass

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test docstring."

    async def test_decorator_with_arguments(self):
        """Test decorator with function arguments."""
        limiter = TokenBucketRateLimiter(rate=10.0)

        @limiter
        async def test_func(x: int, y: int) -> int:
            return x + y

        result = await test_func(5, 3)
        assert result == 8

    async def test_invalid_rate(self):
        """Test that invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="rate must be > 0"):
            TokenBucketRateLimiter(rate=0.0)

        with pytest.raises(ValueError, match="rate must be > 0"):
            TokenBucketRateLimiter(rate=-1.0)

    async def test_invalid_capacity(self):
        """Test that invalid capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be > 0"):
            TokenBucketRateLimiter(rate=10.0, capacity=0.0)

        with pytest.raises(ValueError, match="capacity must be > 0"):
            TokenBucketRateLimiter(rate=10.0, capacity=-1.0)

    async def test_invalid_tokens_per_operation(self):
        """Test that invalid tokens_per_operation raises ValueError."""
        with pytest.raises(ValueError, match="tokens_per_operation must be > 0"):
            TokenBucketRateLimiter(rate=10.0, tokens_per_operation=0.0)

        with pytest.raises(ValueError, match="tokens_per_operation must be > 0"):
            TokenBucketRateLimiter(rate=10.0, tokens_per_operation=-1.0)

    async def test_invalid_initial_tokens(self):
        """Test that invalid initial_tokens raises ValueError."""
        with pytest.raises(ValueError, match="initial_tokens must be >= 0"):
            TokenBucketRateLimiter(rate=10.0, initial_tokens=-1.0)

    async def test_token_refill_rate(self):
        """Test that tokens refill at the correct rate."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0, initial_tokens=0.0)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            # Start with 0 tokens
            limiter._last_refill = 0.0
            limiter._tokens = 0.0

            # Advance time by 0.5 seconds = 5 tokens
            current_time = 0.5

            # Manually refill to test
            await limiter._refill_tokens()
            assert limiter._tokens == pytest.approx(5.0)

            # Advance time by another 0.3 seconds = 3 more tokens
            current_time = 0.8
            await limiter._refill_tokens()
            assert limiter._tokens == pytest.approx(8.0)

    async def test_burst_capacity(self):
        """Test that capacity limits token accumulation."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=5.0, initial_tokens=5.0)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            limiter._last_refill = 0.0
            limiter._tokens = 5.0

            # Advance time by 2 seconds = 20 tokens, but capacity is 5
            current_time = 2.0
            await limiter._refill_tokens()
            assert limiter._tokens == pytest.approx(5.0)

    async def test_wait_for_tokens(self):
        """Test that acquire waits when tokens are insufficient."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0, initial_tokens=0.0)

        current_time = 0.0
        sleep_calls = []

        def mock_time():
            return current_time

        async def mock_sleep(duration):
            nonlocal current_time
            sleep_calls.append(duration)
            current_time += duration

        with (
            patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            limiter._last_refill = 0.0
            limiter._tokens = 0.0

            await limiter.acquire()

            # Should have slept to wait for 1 token at rate 10/sec = 0.1 seconds
            assert len(sleep_calls) > 0
            assert sum(sleep_calls) == pytest.approx(0.1)

    async def test_multiple_operations_consume_tokens(self):
        """Test that multiple operations consume tokens correctly."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0, initial_tokens=10.0)

        current_time = 0.0

        def mock_time():
            return current_time

        async def mock_sleep(duration):
            nonlocal current_time
            current_time += duration

        with (
            patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            limiter._last_refill = 0.0
            limiter._tokens = 10.0

            # Consume 3 tokens
            await limiter.acquire()
            await limiter.acquire()
            await limiter.acquire()

            assert limiter._tokens == pytest.approx(7.0)

    async def test_custom_tokens_per_operation(self):
        """Test using custom tokens per operation."""
        limiter = TokenBucketRateLimiter(
            rate=10.0,
            capacity=10.0,
            initial_tokens=10.0,
            tokens_per_operation=2.5,
        )

        current_time = 0.0

        def mock_time():
            return current_time

        async def mock_sleep(duration):
            nonlocal current_time
            current_time += duration

        with (
            patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            limiter._last_refill = 0.0
            limiter._tokens = 10.0

            # Consume 2.5 tokens
            await limiter.acquire()
            assert limiter._tokens == pytest.approx(7.5)

            # Consume another 2.5 tokens
            await limiter.acquire()
            assert limiter._tokens == pytest.approx(5.0)

    async def test_concurrent_operations(self):
        """Test concurrent operations are serialized by the lock."""
        limiter = TokenBucketRateLimiter(
            rate=100.0, capacity=100.0, initial_tokens=100.0
        )

        completed = []

        async def operation(i):
            async with limiter:
                completed.append(i)

        # Run 10 operations concurrently
        await asyncio.gather(*[operation(i) for i in range(10)])

        # All should complete
        assert len(completed) == 10
        assert set(completed) == set(range(10))

    async def test_high_concurrency(self):
        """Test high concurrency with many operations."""
        limiter = TokenBucketRateLimiter(
            rate=1000.0, capacity=1000.0, initial_tokens=1000.0
        )

        completed = []

        async def operation(i):
            async with limiter:
                completed.append(i)

        # Run 100 operations concurrently
        await asyncio.gather(*[operation(i) for i in range(100)])

        assert len(completed) == 100

    async def test_rate_limiting_behavior(self):
        """Test that rate limiting actually limits rate."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=2.0, initial_tokens=2.0)

        current_time = 0.0
        total_sleep_time = 0.0

        def mock_time():
            return current_time

        async def mock_sleep(duration):
            nonlocal current_time, total_sleep_time
            current_time += duration
            total_sleep_time += duration

        with (
            patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            limiter._last_refill = 0.0
            limiter._tokens = 2.0

            # First 2 operations use initial tokens (no sleep)
            await limiter.acquire()
            await limiter.acquire()

            # Next operations should require waiting
            # Need 1 token at rate 10/sec = 0.1 seconds each
            await limiter.acquire()  # Should sleep ~0.1s
            await limiter.acquire()  # Should sleep ~0.1s
            await limiter.acquire()  # Should sleep ~0.1s

            # Total sleep should be approximately 0.3 seconds (3 waits * 0.1s)
            assert total_sleep_time == pytest.approx(0.3, abs=0.01)

    async def test_continuous_token_accumulation(self):
        """Test that tokens accumulate continuously over time."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0, initial_tokens=0.0)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            limiter._last_refill = 0.0
            limiter._tokens = 0.0

            # Check tokens at different time points
            current_time = 0.25
            await limiter._refill_tokens()
            assert limiter._tokens == pytest.approx(2.5)

            current_time = 0.75
            await limiter._refill_tokens()
            assert limiter._tokens == pytest.approx(7.5)

            current_time = 1.0
            await limiter._refill_tokens()
            assert limiter._tokens == pytest.approx(10.0)

    async def test_fractional_rate(self):
        """Test rate limiter with fractional rate."""
        limiter = TokenBucketRateLimiter(rate=0.5, capacity=1.0, initial_tokens=1.0)

        current_time = 0.0

        def mock_time():
            return current_time

        async def mock_sleep(duration):
            nonlocal current_time
            current_time += duration

        with (
            patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            limiter._last_refill = 0.0
            limiter._tokens = 1.0

            # First operation uses initial token
            await limiter.acquire()
            assert limiter._tokens == pytest.approx(0.0)

            # Second operation needs to wait 2 seconds (1 token at 0.5/sec)
            await limiter.acquire()
            assert current_time == pytest.approx(2.0)

    async def test_fairness_under_load(self):
        """Test that operations are handled fairly under concurrent load."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=5.0, initial_tokens=5.0)

        order = []

        async def operation(i):
            async with limiter:
                order.append(i)
                # Small delay to ensure operations don't complete instantly
                await asyncio.sleep(0.001)

        # Launch operations with slight delays
        tasks = []
        for i in range(10):
            tasks.append(asyncio.create_task(operation(i)))
            await asyncio.sleep(0.0001)  # Tiny delay to establish order

        await asyncio.gather(*tasks)

        # All operations should complete
        assert len(order) == 10
        # Order should roughly preserve submission order due to lock FIFO
        # (though exact order isn't guaranteed with asyncio)
        assert set(order) == set(range(10))

    async def test_zero_initial_tokens(self):
        """Test limiter starting with zero tokens."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0, initial_tokens=0.0)

        current_time = 0.0

        def mock_time():
            return current_time

        async def mock_sleep(duration):
            nonlocal current_time
            current_time += duration

        with (
            patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            limiter._last_refill = 0.0
            limiter._tokens = 0.0

            # Should wait for tokens
            await limiter.acquire()
            assert current_time > 0

    async def test_large_burst(self):
        """Test handling large burst of operations."""
        limiter = TokenBucketRateLimiter(
            rate=1000.0, capacity=1000.0, initial_tokens=1000.0
        )

        completed = []

        async def operation(i):
            async with limiter:
                completed.append(i)

        # Large burst of 1000 operations
        await asyncio.gather(*[operation(i) for i in range(1000)])

        assert len(completed) == 1000
        assert set(completed) == set(range(1000))
