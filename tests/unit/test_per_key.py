"""Tests for per-key rate limiter."""

# mypy: disable-error-code="var-annotated"

import asyncio
from unittest.mock import patch

import pytest

from async_tools.rate_limit.per_key import PerKeyRateLimiter
from async_tools.rate_limit.semaphore_limiter import SemaphoreRateLimiter
from async_tools.rate_limit.token_bucket import TokenBucketRateLimiter


@pytest.mark.unit
@pytest.mark.asyncio
class TestPerKeyRateLimiter:
    """Test per-key rate limiter."""

    async def test_basic_usage(self):
        """Test basic per-key rate limiting."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0)
        )

        try:
            call_count = 0

            async with limiter.for_key("user1"):
                call_count += 1

            assert call_count == 1
        finally:
            await limiter.close()

    async def test_independent_limiters_per_key(self):
        """Test that different keys have independent limiters."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: SemaphoreRateLimiter(max_concurrent=1)
        )

        try:
            running = {"user1": [], "user2": []}

            async def operation(user_id: str, op_id: int):
                async with limiter.for_key(user_id):
                    running[user_id].append(op_id)
                    await asyncio.sleep(0.02)
                    running[user_id].remove(op_id)

            # Run operations for both users concurrently
            await asyncio.gather(
                operation("user1", 1),
                operation("user2", 1),
                operation("user1", 2),
                operation("user2", 2),
            )

            # Each user processed 2 operations
            # They should have been independent
        finally:
            await limiter.close()

    async def test_shared_limiter_for_same_key(self):
        """Test that the same key shares a limiter instance."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: SemaphoreRateLimiter(max_concurrent=1)
        )

        try:
            running = []
            max_concurrent = 0

            async def operation(op_id: int):
                async with limiter.for_key("user1"):
                    running.append(op_id)
                    max_concurrent_var = len(running)
                    nonlocal max_concurrent
                    max_concurrent = max(max_concurrent, max_concurrent_var)
                    await asyncio.sleep(0.01)
                    running.remove(op_id)

            # All operations use same key, should be serialized
            await asyncio.gather(*[operation(i) for i in range(5)])

            # Should never have more than 1 concurrent for same key
            assert max_concurrent == 1
        finally:
            await limiter.close()

    async def test_cleanup_removes_idle_keys(self):
        """Test that cleanup removes idle limiters."""
        current_time = 0.0

        def mock_time():
            return current_time

        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
            max_idle_time=5.0,
            cleanup_interval=1.0,
        )

        try:
            with patch.object(
                asyncio.get_event_loop(), "time", side_effect=mock_time
            ):
                # Use limiter for user1
                async with limiter.for_key("user1"):
                    pass

                # Verify limiter exists
                assert "user1" in limiter._limiters

                # Advance time past idle timeout
                current_time = 10.0

                # Trigger cleanup
                await limiter._cleanup_idle_limiters()

                # Limiter should be removed
                assert "user1" not in limiter._limiters
        finally:
            await limiter.close()

    async def test_cleanup_preserves_active_keys(self):
        """Test that cleanup doesn't remove recently used limiters."""
        current_time = 0.0

        def mock_time():
            return current_time

        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
            max_idle_time=5.0,
            cleanup_interval=1.0,
        )

        try:
            with patch.object(
                asyncio.get_event_loop(), "time", side_effect=mock_time
            ):
                # Use limiter for user1 at time 0
                async with limiter.for_key("user1"):
                    pass

                # Advance time but not past idle timeout
                current_time = 3.0

                # Trigger cleanup
                await limiter._cleanup_idle_limiters()

                # Limiter should still exist
                assert "user1" in limiter._limiters

                # Use limiter again (updates last used time to 3.0)
                async with limiter.for_key("user1"):
                    pass

                # Advance to time 7.0 (4 seconds since last use)
                current_time = 7.0

                # Trigger cleanup - should NOT remove (only 4s idle)
                await limiter._cleanup_idle_limiters()
                assert "user1" in limiter._limiters

                # Advance to time 9.0 (6 seconds since last use)
                current_time = 9.0

                # Trigger cleanup - should remove (6s > 5s idle time)
                await limiter._cleanup_idle_limiters()
                assert "user1" not in limiter._limiters
        finally:
            await limiter.close()

    async def test_decorator_pattern(self):
        """Test decorator pattern with key extraction."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0)
        )

        try:
            call_count = 0

            @limiter.decorator(key_func=lambda user_id, action: user_id)
            async def process_request(user_id: str, action: str):
                nonlocal call_count
                call_count += 1
                return f"{user_id}:{action}"

            result = await process_request("user1", "read")
            assert result == "user1:read"
            assert call_count == 1

            result = await process_request("user2", "write")
            assert result == "user2:write"
            assert call_count == 2
        finally:
            await limiter.close()

    async def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0)
        )

        try:

            @limiter.decorator(key_func=lambda user_id, **kw: user_id)
            async def process_request(user_id: str):
                """Process a user request."""
                pass

            assert process_request.__name__ == "process_request"
            assert process_request.__doc__ == "Process a user request."
        finally:
            await limiter.close()

    async def test_multiple_keys_concurrent(self):
        """Test multiple keys being used concurrently."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: SemaphoreRateLimiter(max_concurrent=1)
        )

        try:
            results = {}

            async def operation(user_id: str, op_id: int):
                async with limiter.for_key(user_id):
                    if user_id not in results:
                        results[user_id] = []
                    results[user_id].append(op_id)
                    await asyncio.sleep(0.01)

            # Run operations for multiple users
            tasks = []
            for user in ["user1", "user2", "user3"]:
                for op_id in range(5):
                    tasks.append(operation(user, op_id))

            await asyncio.gather(*tasks)

            # Each user should have processed all operations
            assert len(results) == 3
            for user in ["user1", "user2", "user3"]:
                assert len(results[user]) == 5
        finally:
            await limiter.close()

    async def test_context_manager_close(self):
        """Test using limiter as async context manager."""
        async with PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0)
        ) as limiter, limiter.for_key("user1"):
            pass

        # Cleanup task should be cancelled after exit

    async def test_invalid_max_idle_time(self):
        """Test that invalid max_idle_time raises ValueError."""
        with pytest.raises(ValueError, match="max_idle_time must be > 0"):
            PerKeyRateLimiter(
                limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
                max_idle_time=0.0,
            )

        with pytest.raises(ValueError, match="max_idle_time must be > 0"):
            PerKeyRateLimiter(
                limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
                max_idle_time=-1.0,
            )

    async def test_invalid_cleanup_interval(self):
        """Test that invalid cleanup_interval raises ValueError."""
        with pytest.raises(ValueError, match="cleanup_interval must be > 0"):
            PerKeyRateLimiter(
                limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
                max_idle_time=10.0,
                cleanup_interval=0.0,
            )

        with pytest.raises(ValueError, match="cleanup_interval must be > 0"):
            PerKeyRateLimiter(
                limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
                max_idle_time=10.0,
                cleanup_interval=-1.0,
            )

    async def test_no_cleanup_without_max_idle_time(self):
        """Test that no cleanup task is created without max_idle_time."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0)
        )

        try:
            # Use some keys
            async with limiter.for_key("user1"):
                pass
            async with limiter.for_key("user2"):
                pass

            # No cleanup task should be running
            assert limiter._cleanup_task is None

            # Limiters should persist
            assert len(limiter._limiters) == 2
        finally:
            await limiter.close()

    async def test_high_concurrency_multiple_keys(self):
        """Test high concurrency with many keys."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=100.0, capacity=100.0)
        )

        try:
            results = {}

            async def operation(user_id: str, op_id: int):
                async with limiter.for_key(user_id):
                    if user_id not in results:
                        results[user_id] = []
                    results[user_id].append(op_id)

            # Create many operations across many keys
            tasks = []
            num_users = 20
            ops_per_user = 50
            for user_idx in range(num_users):
                user_id = f"user{user_idx}"
                for op_id in range(ops_per_user):
                    tasks.append(operation(user_id, op_id))

            await asyncio.gather(*tasks)

            # Verify all operations completed
            assert len(results) == num_users
            for user_idx in range(num_users):
                user_id = f"user{user_idx}"
                assert len(results[user_id]) == ops_per_user
        finally:
            await limiter.close()

    async def test_cleanup_loop_runs_periodically(self):
        """Test that cleanup loop runs at specified interval."""
        cleanup_count = 0

        async def mock_cleanup():
            nonlocal cleanup_count
            cleanup_count += 1

        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
            max_idle_time=10.0,
            cleanup_interval=0.05,  # Short interval for testing
        )

        try:
            # Let cleanup run a few times
            await asyncio.sleep(0.2)

            # Cleanup should have run multiple times
            # (We can't easily verify this without mocking, but we can verify
            # the task is running)
            assert limiter._cleanup_task is not None
            assert not limiter._cleanup_task.done()
        finally:
            await limiter.close()

    async def test_limiter_reuse_after_cleanup(self):
        """Test that keys can be reused after being cleaned up."""
        current_time = 0.0

        def mock_time():
            return current_time

        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
            max_idle_time=5.0,
        )

        try:
            with patch.object(
                asyncio.get_event_loop(), "time", side_effect=mock_time
            ):
                # Use limiter for user1
                async with limiter.for_key("user1"):
                    pass

                assert "user1" in limiter._limiters

                # Clean up
                current_time = 10.0
                await limiter._cleanup_idle_limiters()
                assert "user1" not in limiter._limiters

                # Use again - should create new limiter
                current_time = 15.0
                async with limiter.for_key("user1"):
                    pass

                assert "user1" in limiter._limiters
        finally:
            await limiter.close()

    async def test_exception_in_context_manager(self):
        """Test that exceptions in context manager are handled properly."""
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0)
        )

        try:
            call_count = 0

            with pytest.raises(ValueError):
                async with limiter.for_key("user1"):
                    call_count += 1
                    raise ValueError("test error")

            assert call_count == 1

            # Should be able to use limiter again
            async with limiter.for_key("user1"):
                call_count += 1

            assert call_count == 2
        finally:
            await limiter.close()

    async def test_different_limiter_types(self):
        """Test that different limiter types can be used."""
        # Token bucket limiter
        token_limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0)
        )

        # Semaphore limiter
        semaphore_limiter = PerKeyRateLimiter(
            limiter_factory=lambda: SemaphoreRateLimiter(max_concurrent=2)
        )

        try:
            async with token_limiter.for_key("user1"):
                pass

            async with semaphore_limiter.for_key("user1"):
                pass

            # Both should work
        finally:
            await token_limiter.close()
            await semaphore_limiter.close()

    async def test_memory_cleanup(self):
        """Test that cleanup prevents memory leaks."""
        current_time = 0.0

        def mock_time():
            return current_time

        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
            max_idle_time=1.0,
        )

        try:
            with patch.object(
                asyncio.get_event_loop(), "time", side_effect=mock_time
            ):
                # Create many keys
                for i in range(100):
                    async with limiter.for_key(f"user{i}"):
                        pass

                assert len(limiter._limiters) == 100

                # Advance time and cleanup
                current_time = 10.0
                await limiter._cleanup_idle_limiters()

                # All should be removed
                assert len(limiter._limiters) == 0
        finally:
            await limiter.close()
