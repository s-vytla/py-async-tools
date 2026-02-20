"""Semaphore-based rate limiter for concurrency control."""

import asyncio

from async_tools.rate_limit.base import RateLimiterBase


class SemaphoreRateLimiter(RateLimiterBase):
    """Semaphore-based rate limiter for controlling concurrency.

    This limiter restricts the number of operations that can run concurrently.
    Unlike token bucket, which limits rate over time, this limits simultaneous
    operations.

    Args:
        max_concurrent: Maximum number of concurrent operations (must be >= 1)

    Example:
        # Allow maximum 5 concurrent operations
        limiter = SemaphoreRateLimiter(max_concurrent=5)

        async with limiter:
            await database_query()

        # Or as a decorator
        @limiter
        async def db_operation():
            await database_query()
    """

    def __init__(self, max_concurrent: int):
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def acquire(self) -> None:
        """Acquire a slot in the concurrency limit.

        Blocks until a slot is available.
        """
        await self._semaphore.acquire()

    async def release(self) -> None:
        """Release a slot, allowing another operation to proceed."""
        self._semaphore.release()
