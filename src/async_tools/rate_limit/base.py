"""Base classes and protocols for rate limiters."""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast


F = TypeVar("F", bound=Callable[..., Any])


class AsyncRateLimiter(Protocol):
    """Protocol defining the interface for async rate limiters.

    All rate limiters should implement this protocol to ensure
    they can be used interchangeably.
    """

    async def acquire(self) -> None:
        """Acquire permission to proceed.

        This method will block until the rate limit allows the operation.
        """
        ...

    async def __aenter__(self) -> None:
        """Enter the async context manager."""
        ...

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Exit the async context manager."""
        ...


class RateLimiterBase(ABC):
    """Abstract base class for rate limiters.

    Provides default implementations for context manager and decorator patterns.
    Subclasses only need to implement the core acquire() and optionally
    release() methods.

    The context manager pattern is the canonical usage:
        async with limiter:
            await operation()

    The decorator pattern is a convenience wrapper:
        @limiter
        async def my_function():
            await operation()
    """

    @abstractmethod
    async def acquire(self) -> None:
        """Acquire permission to proceed.

        This method will block until the rate limit allows the operation.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    async def release(self) -> None:
        """Release any held resources.

        Default implementation is a no-op. Subclasses that need to release
        resources (like semaphores) should override this method.
        """
        pass

    async def __aenter__(self) -> None:
        """Enter the async context manager by acquiring permission."""
        await self.acquire()

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Exit the async context manager by releasing resources."""
        await self.release()

    def __call__(self, func: F) -> F:
        """Decorate an async function to rate limit its execution.

        Args:
            func: The async function to decorate

        Returns:
            The decorated function that respects the rate limit
        """
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func).__name__}")

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)

        return cast(F, wrapper)
