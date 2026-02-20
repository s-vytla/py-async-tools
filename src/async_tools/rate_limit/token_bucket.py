"""Token bucket rate limiter implementation."""

import asyncio

from async_tools.rate_limit.base import RateLimiterBase


class TokenBucketRateLimiter(RateLimiterBase):
    """Token bucket rate limiter with burst support.

    This limiter adds tokens at a constant rate and allows operations to consume tokens.
    If tokens are available, operations proceed immediately. Otherwise, they wait until
    enough tokens accumulate.

    The bucket has a maximum capacity, allowing for burst behavior where multiple
    operations can proceed rapidly if tokens have accumulated.

    Args:
        rate: Number of operations allowed per second (must be > 0)
        capacity: Maximum number of tokens the bucket can hold.
                 Defaults to rate (1 second worth of tokens)
        initial_tokens: Number of tokens to start with.
                       Defaults to capacity (full bucket)
        tokens_per_operation: Number of tokens consumed per operation.
                             Defaults to 1.0

    Example:
        # Allow 10 operations per second with burst of 20
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=20.0)

        async with limiter:
            await api_call()

        # Or as a decorator
        @limiter
        async def rate_limited_function():
            await api_call()
    """

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        initial_tokens: float | None = None,
        tokens_per_operation: float = 1.0,
    ):
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        if tokens_per_operation <= 0:
            raise ValueError(
                f"tokens_per_operation must be > 0, got {tokens_per_operation}"
            )

        self._rate = rate
        self._capacity = capacity if capacity is not None else rate
        if self._capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {self._capacity}")

        self._tokens_per_op = tokens_per_operation
        self._tokens = initial_tokens if initial_tokens is not None else self._capacity
        if self._tokens < 0:
            raise ValueError(f"initial_tokens must be >= 0, got {self._tokens}")

        # Use asyncio.Lock for async concurrency safety
        self._lock = asyncio.Lock()
        # Use monotonic time from event loop
        self._last_refill = asyncio.get_event_loop().time()

    async def acquire(self) -> None:
        """Acquire permission by consuming tokens.

        Blocks until enough tokens are available.
        """
        async with self._lock:
            await self._refill_tokens()

            # Wait until we have enough tokens
            while self._tokens < self._tokens_per_op:
                # Calculate how long to wait for enough tokens
                tokens_needed = self._tokens_per_op - self._tokens
                wait_time = tokens_needed / self._rate

                # Release lock during sleep to allow other operations to check
                # Actually, we need to keep the lock to maintain consistency
                await asyncio.sleep(wait_time)
                await self._refill_tokens()

            # Consume tokens
            self._tokens -= self._tokens_per_op

    async def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time.

        This method should only be called while holding self._lock.
        """
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self._rate
        self._tokens = min(self._tokens + tokens_to_add, self._capacity)
        self._last_refill = now
