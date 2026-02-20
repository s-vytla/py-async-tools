"""Circuit breaker implementation for preventing cascading failures."""

import asyncio
from enum import Enum
from typing import Any

from async_tools.rate_limit.base import RateLimiterBase


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open.

    This indicates that too many failures have occurred and the circuit
    breaker is temporarily blocking all requests to give the failing
    system time to recover.
    """

    pass


class CircuitBreaker(RateLimiterBase):
    """Circuit breaker for preventing cascading failures.

    The circuit breaker monitors failures and automatically transitions between
    three states:

    - CLOSED: Normal operation. Requests pass through. Failures are counted.
    - OPEN: Too many failures occurred. All requests are immediately blocked.
    - HALF_OPEN: Testing recovery. Limited requests are allowed through.

    State transitions:
    - CLOSED → OPEN: When failure_count >= failure_threshold
    - OPEN → HALF_OPEN: After timeout seconds
    - HALF_OPEN → CLOSED: On first success
    - HALF_OPEN → OPEN: On any failure

    Args:
        failure_threshold: Number of failures before opening circuit (default: 5)
        timeout: Seconds to wait in OPEN state before testing recovery (default: 60.0)
        half_open_max_calls: Max concurrent requests in HALF_OPEN state (default: 1)

    Example:
        # Context manager (canonical)
        breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)
        try:
            async with breaker:
                await unreliable_api_call()
        except CircuitBreakerOpenError:
            # Circuit is open, too many recent failures
            pass

        # Decorator
        @breaker
        async def my_function():
            await unreliable_api_call()

        # Manual tracking (for fine-grained control)
        await breaker.acquire()
        try:
            result = await unreliable_api_call()
            await breaker.record_success()
        except Exception as e:
            await breaker.record_failure()
            raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        if failure_threshold <= 0:
            raise ValueError(f"failure_threshold must be > 0, got {failure_threshold}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        if half_open_max_calls <= 0:
            raise ValueError(
                f"half_open_max_calls must be > 0, got {half_open_max_calls}"
            )

        self._failure_threshold = failure_threshold
        self._timeout = timeout
        self._half_open_max_calls = half_open_max_calls

        # State management
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._opened_at: float | None = None

        # Async safety
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to proceed.

        Checks circuit state and blocks if circuit is OPEN.

        Raises:
            CircuitBreakerOpenError: If the circuit is OPEN
        """
        async with self._lock:
            await self._update_state()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. "
                    f"Failures: {self._failure_count}/{self._failure_threshold}"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is HALF_OPEN and at capacity. "
                        f"Current calls: {self._half_open_calls}/"
                        f"{self._half_open_max_calls}"
                    )
                self._half_open_calls += 1

    async def _update_state(self) -> None:
        """Update circuit state based on time and failure count.

        This method should only be called while holding self._lock.
        """
        if self._state == CircuitState.OPEN:
            # Check if we should transition to HALF_OPEN
            now = asyncio.get_event_loop().time()
            if self._opened_at is not None:
                elapsed = now - self._opened_at
                if elapsed >= self._timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0

    async def record_success(self) -> None:
        """Record a successful operation.

        In CLOSED state: resets failure count.
        In HALF_OPEN state: transitions back to CLOSED.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
            elif self._state == CircuitState.HALF_OPEN:
                # Success in HALF_OPEN means recovery confirmed
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._opened_at = None

    async def record_failure(self) -> None:
        """Record a failed operation.

        In CLOSED state: increments failure count, may open circuit.
        In HALF_OPEN state: transitions back to OPEN.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    # Open the circuit
                    self._state = CircuitState.OPEN
                    self._opened_at = asyncio.get_event_loop().time()
            elif self._state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN means still not recovered
                self._state = CircuitState.OPEN
                self._opened_at = asyncio.get_event_loop().time()
                self._half_open_calls = 0

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager by recording success or failure.

        Unlike the base class, this tracks whether the wrapped operation
        succeeded or failed based on whether an exception occurred.
        """
        if exc_type is not None:
            # Exception occurred, record as failure
            await self.record_failure()
        else:
            # No exception, record as success
            await self.record_success()

    @property
    def state(self) -> CircuitState:
        """Get the current circuit state.

        Note: This is a snapshot and may change immediately after reading.
        """
        return self._state

    @property
    def failure_count(self) -> int:
        """Get the current failure count.

        Note: This is a snapshot and may change immediately after reading.
        """
        return self._failure_count
