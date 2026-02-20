"""Per-key rate limiter with automatic cleanup."""

import asyncio
import contextlib
import functools
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from async_tools.rate_limit.base import AsyncRateLimiter


KeyType = TypeVar("KeyType")
LimiterType = TypeVar("LimiterType", bound=AsyncRateLimiter)
F = TypeVar("F", bound=Callable[..., Any])


class PerKeyLimiterContext(Generic[LimiterType]):
    """Context manager for a specific key's limiter.

    This class provides the async context manager interface for operations
    on a specific key.
    """

    def __init__(self, limiter: LimiterType):
        self._limiter = limiter

    async def __aenter__(self) -> None:
        """Enter the async context manager by acquiring the key's limiter."""
        await self._limiter.acquire()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager by releasing the key's limiter."""
        # Release if the limiter has a release method
        if hasattr(self._limiter, "__aexit__"):
            await self._limiter.__aexit__(exc_type, exc_val, exc_tb)


class PerKeyRateLimiter(Generic[KeyType, LimiterType]):
    """Per-key rate limiter with automatic cleanup.

    Manages independent rate limiter instances for different keys. Each key gets
    its own limiter instance created by the factory function. Optionally cleans up
    idle limiters after a specified timeout.

    Args:
        limiter_factory: Callable that returns a new limiter instance
        max_idle_time: Seconds before removing idle limiters (None = no cleanup)
        cleanup_interval: How often to run cleanup in seconds (default 60.0)

    Example:
        # Create per-user rate limiter
        limiter = PerKeyRateLimiter(
            limiter_factory=lambda: TokenBucketRateLimiter(rate=10.0),
            max_idle_time=300.0  # Remove after 5 minutes idle
        )

        # Use with context manager
        async with limiter.for_key(user_id):
            await process_request()

        # Use with decorator
        @limiter.decorator(key_func=lambda user_id, **kw: user_id)
        async def process_user_request(user_id: str):
            await do_work()

        # Clean up when done
        await limiter.close()
    """

    def __init__(
        self,
        limiter_factory: Callable[[], LimiterType],
        max_idle_time: float | None = None,
        cleanup_interval: float = 60.0,
    ):
        if max_idle_time is not None and max_idle_time <= 0:
            raise ValueError(f"max_idle_time must be > 0, got {max_idle_time}")
        if cleanup_interval <= 0:
            raise ValueError(f"cleanup_interval must be > 0, got {cleanup_interval}")

        self._factory = limiter_factory
        self._max_idle_time = max_idle_time
        self._cleanup_interval = cleanup_interval

        # Store (limiter, last_used_time) for each key
        self._limiters: dict[KeyType, tuple[LimiterType, float]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

        # Start cleanup task if max_idle_time is set
        if self._max_idle_time is not None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def for_key(self, key: KeyType) -> PerKeyLimiterContext[LimiterType]:
        """Get a context manager for the specified key's limiter.

        Args:
            key: The key to get a limiter for

        Returns:
            Context manager that acquires/releases the key's limiter

        Example:
            async with limiter.for_key("user123"):
                await process_request()
        """

        # We need to return a context manager that will get or create
        # the limiter and update the last used time
        class KeyContextManager:
            def __init__(
                self, parent: "PerKeyRateLimiter[KeyType, LimiterType]", key: KeyType
            ):
                self._parent = parent
                self._key = key
                self._limiter: LimiterType | None = None

            async def __aenter__(self) -> None:
                self._limiter = await self._parent._get_or_create_limiter(self._key)
                await self._limiter.acquire()

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self._limiter is not None:
                    # Update last used time
                    async with self._parent._lock:
                        if self._key in self._parent._limiters:
                            now = asyncio.get_event_loop().time()
                            self._parent._limiters[self._key] = (
                                self._limiter,
                                now,
                            )
                    # Release the limiter
                    if hasattr(self._limiter, "__aexit__"):
                        await self._limiter.__aexit__(exc_type, exc_val, exc_tb)

        return KeyContextManager(self, key)  # type: ignore

    def decorator(self, key_func: Callable[..., KeyType]) -> Callable[[F], F]:
        """Create a decorator that extracts the key from function arguments.

        Args:
            key_func: Function that extracts the key from the decorated
                     function's arguments

        Returns:
            Decorator that applies per-key rate limiting

        Example:
            @limiter.decorator(key_func=lambda user_id, **kw: user_id)
            async def process_user_request(user_id: str, action: str):
                await do_work()
        """

        def decorator_wrapper(func: F) -> F:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = key_func(*args, **kwargs)
                async with self.for_key(key):
                    return await func(*args, **kwargs)

            return cast("F", wrapper)

        return decorator_wrapper

    async def _get_or_create_limiter(self, key: KeyType) -> LimiterType:
        """Get or create a limiter for the specified key.

        Updates the last used timestamp.

        Args:
            key: The key to get a limiter for

        Returns:
            The limiter instance for this key
        """
        async with self._lock:
            now = asyncio.get_event_loop().time()

            if key in self._limiters:
                limiter, _ = self._limiters[key]
                self._limiters[key] = (limiter, now)
                return limiter

            # Create new limiter
            limiter = self._factory()
            self._limiters[key] = (limiter, now)
            return limiter

    async def _cleanup_loop(self) -> None:
        """Background task that removes idle limiters.

        Runs periodically and removes limiters that haven't been used
        for longer than max_idle_time.
        """
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_idle_limiters()
        except asyncio.CancelledError:
            # Task was cancelled, exit cleanly
            pass

    async def _cleanup_idle_limiters(self) -> None:
        """Remove limiters that have been idle for too long."""
        if self._max_idle_time is None:
            return

        async with self._lock:
            now = asyncio.get_event_loop().time()
            keys_to_remove = []

            for key, (_limiter, last_used) in self._limiters.items():
                idle_time = now - last_used
                if idle_time > self._max_idle_time:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._limiters[key]

    async def close(self) -> None:
        """Close the limiter and cancel the cleanup task.

        Should be called when the limiter is no longer needed to ensure
        proper cleanup of resources.

        Example:
            limiter = PerKeyRateLimiter(...)
            try:
                # Use limiter
                pass
            finally:
                await limiter.close()
        """
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    async def __aenter__(self) -> "PerKeyRateLimiter[KeyType, LimiterType]":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and clean up."""
        await self.close()
