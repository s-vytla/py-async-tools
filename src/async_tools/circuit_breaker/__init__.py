"""Circuit breaker pattern for preventing cascading failures."""

from async_tools.circuit_breaker.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


__all__ = ["CircuitBreaker", "CircuitBreakerOpenError", "CircuitState"]
