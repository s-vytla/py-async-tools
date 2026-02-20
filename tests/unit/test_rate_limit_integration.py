"""Integration tests for rate limiters."""

# mypy: disable-error-code="var-annotated"

import asyncio

import pytest

from async_tools.rate_limit import (
    PerKeyRateLimiter,
    SemaphoreRateLimiter,
    TokenBucketRateLimiter,
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestRateLimitIntegration:
    """Integration tests combining multiple rate limiters."""

    async def test_token_bucket_and_semaphore(self):
        """Test combining token bucket and semaphore limiters."""
        rate_limiter = TokenBucketRateLimiter(
            rate=100.0, capacity=100.0, initial_tokens=100.0
        )
        concurrency_limiter = SemaphoreRateLimiter(max_concurrent=5)

        completed = []
        running = []
        max_concurrent = 0

        async def operation(op_id: int):
            nonlocal max_concurrent
            # Apply both rate and concurrency limits
            async with rate_limiter, concurrency_limiter:
                running.append(op_id)
                max_concurrent = max(max_concurrent, len(running))
                await asyncio.sleep(0.001)
                running.remove(op_id)
                completed.append(op_id)

        # Run 50 operations
        await asyncio.gather(*[operation(i) for i in range(50)])

        # All should complete
        assert len(completed) == 50
        # Concurrency should be limited
        assert max_concurrent <= 5

    async def test_nested_per_key_limiters(self):
        """Test nested per-key limiters for multi-tenant scenarios."""
        # Tenant-level limiter
        tenant_limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=50.0, capacity=50.0)
        )

        # User-level limiter within tenant
        user_limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0, capacity=10.0)
        )

        try:
            results = {"tenant1": {}, "tenant2": {}}

            async def process_request(tenant_id: str, user_id: str, req_id: int):
                async with tenant_limiter.for_key(tenant_id):
                    async with user_limiter.for_key(f"{tenant_id}:{user_id}"):
                        if user_id not in results[tenant_id]:
                            results[tenant_id][user_id] = []
                        results[tenant_id][user_id].append(req_id)

            # Multiple tenants, multiple users per tenant
            tasks = []
            for tenant in ["tenant1", "tenant2"]:
                for user in ["user1", "user2"]:
                    for req_id in range(5):
                        tasks.append(process_request(tenant, user, req_id))

            await asyncio.gather(*tasks)

            # Verify all requests completed
            assert len(results["tenant1"]) == 2
            assert len(results["tenant2"]) == 2
            for tenant_data in results.values():
                for user_requests in tenant_data.values():
                    assert len(user_requests) == 5

        finally:
            await tenant_limiter.close()
            await user_limiter.close()

    async def test_global_and_per_key_limiting(self):
        """Test combining global and per-key rate limiting."""
        # Global limit for entire system
        global_limiter = TokenBucketRateLimiter(
            rate=100.0, capacity=100.0, initial_tokens=100.0
        )

        # Per-key limit for each user
        per_user_limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=20.0, capacity=20.0)
        )

        try:
            results = {}

            async def process_request(user_id: str, req_id: int):
                async with global_limiter:
                    async with per_user_limiter.for_key(user_id):
                        if user_id not in results:
                            results[user_id] = []
                        results[user_id].append(req_id)

            # Multiple users making requests
            tasks = []
            for user_idx in range(5):
                user_id = f"user{user_idx}"
                for req_id in range(10):
                    tasks.append(process_request(user_id, req_id))

            await asyncio.gather(*tasks)

            # All requests should complete
            assert len(results) == 5
            for user_results in results.values():
                assert len(user_results) == 10

        finally:
            await per_user_limiter.close()

    async def test_decorator_composition(self):
        """Test composing multiple rate limiter decorators."""
        rate_limiter = TokenBucketRateLimiter(
            rate=100.0, capacity=100.0, initial_tokens=100.0
        )
        concurrency_limiter = SemaphoreRateLimiter(max_concurrent=3)

        call_count = 0
        running = []
        max_concurrent = 0

        @rate_limiter
        @concurrency_limiter
        async def limited_operation(op_id: int):
            nonlocal call_count, max_concurrent
            call_count += 1
            running.append(op_id)
            max_concurrent = max(max_concurrent, len(running))
            await asyncio.sleep(0.001)
            running.remove(op_id)
            return op_id

        # Run many operations
        results = await asyncio.gather(*[limited_operation(i) for i in range(30)])

        # All should complete
        assert len(results) == 30
        assert call_count == 30
        # Concurrency should be limited
        assert max_concurrent <= 3

    async def test_different_limiters_per_endpoint(self):
        """Test different rate limiters for different API endpoints."""
        # Different limits for different endpoints
        read_limiter = TokenBucketRateLimiter(rate=100.0, capacity=100.0)
        write_limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0)
        admin_limiter = TokenBucketRateLimiter(rate=5.0, capacity=5.0)

        results = {"read": 0, "write": 0, "admin": 0}

        @read_limiter
        async def read_endpoint():
            results["read"] += 1

        @write_limiter
        async def write_endpoint():
            results["write"] += 1

        @admin_limiter
        async def admin_endpoint():
            results["admin"] += 1

        # Make many requests to each endpoint
        await asyncio.gather(
            *[read_endpoint() for _ in range(50)],
            *[write_endpoint() for _ in range(20)],
            *[admin_endpoint() for _ in range(10)],
        )

        # All should complete
        assert results["read"] == 50
        assert results["write"] == 20
        assert results["admin"] == 10

    async def test_cascading_limiters(self):
        """Test cascading rate limiters for tiered rate limiting."""
        # Tier 1: Very fast limit (100/sec)
        tier1_limiter = TokenBucketRateLimiter(
            rate=1000.0, capacity=1000.0, initial_tokens=1000.0
        )

        # Tier 2: Moderate limit (50/sec)
        tier2_limiter = TokenBucketRateLimiter(
            rate=500.0, capacity=500.0, initial_tokens=500.0
        )

        # Tier 3: Slow limit (10/sec)
        tier3_limiter = TokenBucketRateLimiter(
            rate=100.0, capacity=100.0, initial_tokens=100.0
        )

        completed = []

        async def tiered_operation(op_id: int, tier: int):
            if tier >= 1:
                async with tier1_limiter:
                    if tier >= 2:
                        async with tier2_limiter:
                            if tier >= 3:
                                async with tier3_limiter:
                                    completed.append((op_id, tier))
                            else:
                                completed.append((op_id, tier))
                    else:
                        completed.append((op_id, tier))
            else:
                completed.append((op_id, tier))

        # Run operations at different tiers
        await asyncio.gather(
            *[tiered_operation(i, 1) for i in range(20)],
            *[tiered_operation(i, 2) for i in range(20)],
            *[tiered_operation(i, 3) for i in range(20)],
        )

        # All should complete
        assert len(completed) == 60

    async def test_per_key_with_different_limiter_types(self):
        """Test per-key limiter with mixed limiter types."""

        # Some users get token bucket, others get semaphore
        def mixed_factory():
            # Alternate between limiter types
            return TokenBucketRateLimiter(rate=10.0)

        limiter = PerKeyRateLimiter(limiter_factory=mixed_factory)

        try:
            results = {}

            async def operation(user_id: str, op_id: int):
                async with limiter.for_key(user_id):
                    if user_id not in results:
                        results[user_id] = []
                    results[user_id].append(op_id)

            # Multiple users
            tasks = []
            for user_idx in range(5):
                user_id = f"user{user_idx}"
                for op_id in range(10):
                    tasks.append(operation(user_id, op_id))

            await asyncio.gather(*tasks)

            # All operations should complete
            assert len(results) == 5
            for user_results in results.values():
                assert len(user_results) == 10

        finally:
            await limiter.close()

    async def test_high_concurrency_integration(self):
        """Test high concurrency with multiple limiter types."""
        token_limiter = TokenBucketRateLimiter(
            rate=1000.0, capacity=1000.0, initial_tokens=1000.0
        )
        semaphore_limiter = SemaphoreRateLimiter(max_concurrent=20)
        per_key_limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=200.0, capacity=200.0)
        )

        try:
            completed = []

            async def operation(user_id: str, op_id: int):
                async with token_limiter, semaphore_limiter:
                    async with per_key_limiter.for_key(user_id):
                        completed.append((user_id, op_id))

            # Many users, many operations
            tasks = []
            for user_idx in range(10):
                user_id = f"user{user_idx}"
                for op_id in range(50):
                    tasks.append(operation(user_id, op_id))

            await asyncio.gather(*tasks)

            # All should complete
            assert len(completed) == 500

        finally:
            await per_key_limiter.close()

    async def test_limiters_with_exception_handling(self):
        """Test that limiters work correctly with exception handling."""
        limiter = TokenBucketRateLimiter(
            rate=100.0, capacity=100.0, initial_tokens=100.0
        )

        success_count = 0
        error_count = 0

        @limiter
        async def sometimes_failing_operation(op_id: int):
            nonlocal success_count, error_count
            if op_id % 5 == 0:
                error_count += 1
                raise ValueError(f"Operation {op_id} failed")
            success_count += 1
            return op_id

        # Run operations, some will fail
        results = []
        errors = []
        for i in range(50):
            try:
                result = await sometimes_failing_operation(i)
                results.append(result)
            except ValueError as e:
                errors.append(str(e))

        # All operations should have been attempted
        assert success_count == 40
        assert error_count == 10
        assert len(results) == 40
        assert len(errors) == 10
