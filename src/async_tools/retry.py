"""Async retry decorator with exponential backoff.

This module provides a decorator for retrying async functions with configurable
exponential backoff, exception filtering, and custom callbacks.
"""

import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]
]:
    """Decorator that retries an async function with exponential backoff.

    This decorator catches specified exceptions and retries the decorated function
    with exponentially increasing delays between attempts. It's useful for handling
    transient failures such as network issues or temporary service unavailability.

    Args:
        max_attempts: Maximum number of attempts (including the first try).
            Must be at least 1. Default is 3.
        initial_delay: Initial delay in seconds before the first retry.
            Must be non-negative. Default is 1.0.
        max_delay: Maximum delay in seconds between retries. This caps the
            exponential growth. Must be non-negative. Default is 60.0.
        exponential_base: Base for exponential backoff calculation.
            Must be greater than or equal to 1.0. Default is 2.0.
        exceptions: Tuple of exception types to catch and retry on.
            Only these exceptions will trigger a retry; other exceptions
            will propagate immediately. Default is (Exception,).
        on_retry: Optional callback invoked before each retry. Called with
            (exception, attempt_number, delay) where attempt_number is the
            number of the upcoming attempt (2, 3, ...) and delay is the
            sleep duration in seconds.

    Returns:
        A decorator that wraps async functions with retry logic.

    Raises:
        ValueError: If max_attempts < 1, initial_delay < 0, max_delay < 0,
            or exponential_base < 1.0.
        The original exception: After all retry attempts are exhausted.

    Examples:
        Basic usage with default settings:

        >>> @retry()
        ... async def fetch_data():
        ...     response = await http_client.get("https://api.example.com")
        ...     return response.json()

        Custom retry configuration:

        >>> @retry(
        ...     max_attempts=5,
        ...     initial_delay=0.5,
        ...     max_delay=30.0,
        ...     exceptions=(ConnectionError, TimeoutError),
        ... )
        ... async def connect_to_service():
        ...     return await service.connect()

        With retry callback for logging:

        >>> def log_retry(exc: Exception, attempt: int, delay: float) -> None:
        ...     print(f"Retry {attempt} after {delay}s due to {type(exc).__name__}")
        ...
        >>> @retry(on_retry=log_retry)
        ... async def flaky_operation():
        ...     return await external_api.call()

    The delay between retries follows the formula:
        delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)

    For example, with default settings (initial_delay=1.0, exponential_base=2.0):
        - Retry 1: 1.0 seconds
        - Retry 2: 2.0 seconds
        - Retry 3: 4.0 seconds
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if initial_delay < 0:
        raise ValueError("initial_delay must be non-negative")
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")
    if exponential_base < 1.0:
        raise ValueError("exponential_base must be at least 1.0")

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc

                    if attempt >= max_attempts:
                        # No more attempts left, re-raise the exception
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )

                    # Invoke callback if provided
                    if on_retry is not None:
                        on_retry(exc, attempt + 1, delay)

                    # Wait before next retry
                    await asyncio.sleep(delay)

            # This should never be reached, but mypy requires it
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator
