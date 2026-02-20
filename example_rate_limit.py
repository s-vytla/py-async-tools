"""Example usage of rate limiters."""
# mypy: ignore-errors

import asyncio
import time

from async_tools.rate_limit import (
    PerKeyRateLimiter,
    SemaphoreRateLimiter,
    TokenBucketRateLimiter,
)


async def example_token_bucket()->None:
    """Example: API rate limiting with token bucket."""
    print("\n=== Token Bucket Example ===")
    print("Rate limiting API calls to 5 per second with burst of 10")

    # Allow 5 operations per second, burst up to 10
    limiter = TokenBucketRateLimiter(rate=5.0, capacity=10.0)

    @limiter
    async def api_call(request_id: int)->None:
        print(f"  API call {request_id} executing")
        # Simulate API work
        await asyncio.sleep(0.01)

    start = time.time()

    # Make 20 API calls - first 10 burst, then rate limited
    await asyncio.gather(*[api_call(i) for i in range(20)])

    elapsed = time.time() - start
    print(f"Completed 20 calls in {elapsed:.2f} seconds")
    print(f"Effective rate: {20 / elapsed:.1f} calls/second")


async def example_semaphore()->None:
    """Example: Database connection pooling with semaphore."""
    print("\n=== Semaphore Example ===")
    print("Limiting concurrent database connections to 3")

    # Limit to 3 concurrent database connections
    limiter = SemaphoreRateLimiter(max_concurrent=3)

    active_connections = 0
    max_concurrent_seen = 0

    @limiter
    async def database_query(query_id: int)->None:
        nonlocal active_connections, max_concurrent_seen
        active_connections += 1
        max_concurrent_seen = max(max_concurrent_seen, active_connections)
        print(
            f"  Query {query_id} executing (active connections: {active_connections})"
        )
        # Simulate database work
        await asyncio.sleep(0.05)
        active_connections -= 1

    start = time.time()

    # Make 15 database queries
    await asyncio.gather(*[database_query(i) for i in range(15)])

    elapsed = time.time() - start
    print(f"Completed 15 queries in {elapsed:.2f} seconds")
    print(f"Max concurrent connections: {max_concurrent_seen}")


async def example_per_key()->None:
    """Example: Per-user rate limiting."""
    print("\n=== Per-Key Example ===")
    print("Rate limiting per user (10 requests/sec per user)")

    # Each user gets their own rate limiter
    limiter: PerKeyRateLimiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0, capacity=10.0),
        max_idle_time=300.0,  # Clean up after 5 minutes idle
    )

    try:

        @limiter.decorator(key_func=lambda user_id, request_id: user_id)
        async def process_user_request(user_id: str, request_id: int) -> None:
            print(f"  User {user_id} - Request {request_id}")
            await asyncio.sleep(0.01)

        # Multiple users making requests concurrently~
        tasks = []
        for user in ["alice", "bob", "charlie"]:
            for req_id in range(5):
                tasks.append(process_user_request(user, req_id))

        start = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print(f"Completed 15 requests (3 users x 5 requests) in {elapsed:.2f}s")
        print("Each user rate limited independently")

    finally:
        await limiter.close()


async def example_combined():
    """Example: Combining global and per-key rate limiting."""
    print("\n=== Combined Limiters Example ===")
    print("Global limit: 20/sec, Per-user limit: 10/sec")

    # Global rate limit for entire application
    global_limiter = TokenBucketRateLimiter(rate=20.0, capacity=20.0)

    # Per-user rate limit
    per_user_limiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0, capacity=10.0)
    )

    try:

        async def process_request(user_id: str, request_id: int):
            # Apply both global and per-user limits
            async with global_limiter, per_user_limiter.for_key(user_id):
                print(f"  {user_id} - Request {request_id}")
                await asyncio.sleep(0.01)

        # Two users making many requests
        tasks = []
        for user in ["alice", "bob"]:
            for req_id in range(15):
                tasks.append(process_request(user, req_id))

        start = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print(f"Completed 30 requests in {elapsed:.2f}s")
        print("Respected both global and per-user rate limits")

    finally:
        await per_user_limiter.close()


async def example_error_handling():
    """Example: Error handling with rate limiters."""
    print("\n=== Error Handling Example ===")

    limiter = TokenBucketRateLimiter(rate=10.0)

    @limiter
    async def failing_operation(op_id: int):
        if op_id % 3 == 0:
            raise ValueError(f"Operation {op_id} failed")
        print(f"  Operation {op_id} succeeded")

    # Rate limiter properly releases on exceptions
    for i in range(5):
        try:
            await failing_operation(i)
        except ValueError as e:
            print(f"  Caught: {e}")

    print("Rate limiter remained functional after exceptions")


async def example_context_manager():
    """Example: Manual control with context manager."""
    print("\n=== Context Manager Example ===")

    limiter = TokenBucketRateLimiter(rate=5.0)

    async def operation_with_setup(op_id: int):
        # Setup work (not rate limited)
        print(f"  Operation {op_id} - setup")

        # Critical section (rate limited)
        async with limiter:
            print(f"  Operation {op_id} - executing")
            await asyncio.sleep(0.01)

        # Cleanup work (not rate limited)
        print(f"  Operation {op_id} - cleanup")

    await asyncio.gather(*[operation_with_setup(i) for i in range(3)])


async def main():
    """Run all examples."""
    print("Rate Limiter Examples")
    print("=" * 50)

    await example_token_bucket()
    await example_semaphore()
    await example_per_key()
    await example_combined()
    await example_error_handling()
    await example_context_manager()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
