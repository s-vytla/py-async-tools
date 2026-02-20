# mypy: disable-error-code="var-annotated,comparison-overlap,unreachable"
"""Tests for circuit breaker."""

import asyncio
from unittest.mock import patch

import pytest

from async_tools.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCircuitBreaker:
    """Test circuit breaker."""

    async def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    async def test_context_manager_success(self):
        """Test context manager with successful operation."""
        breaker = CircuitBreaker(failure_threshold=3)

        async with breaker:
            pass  # Successful operation

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    async def test_context_manager_failure(self):
        """Test context manager with failed operation."""
        breaker = CircuitBreaker(failure_threshold=3)

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 1

    async def test_transition_to_open_on_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        # First 2 failures should keep circuit closed
        for _ in range(2):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 2

        # Third failure should open circuit
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    async def test_open_circuit_blocks_requests(self):
        """Test that OPEN circuit blocks all requests."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

        assert breaker.state == CircuitState.OPEN

        # Next request should be blocked immediately
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.acquire()

    async def test_transition_to_half_open_after_timeout(self):
        """Test that circuit transitions to HALF_OPEN after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            breaker._opened_at = 0.0

            # Open the circuit
            for _ in range(2):
                with pytest.raises(ValueError):
                    async with breaker:
                        raise ValueError("test error")

            assert breaker.state == CircuitState.OPEN

            # Before timeout, still OPEN
            current_time = 0.5
            with pytest.raises(CircuitBreakerOpenError):
                await breaker.acquire()
            assert breaker.state == CircuitState.OPEN

            # After timeout, should transition to HALF_OPEN
            current_time = 1.0
            await breaker.acquire()
            assert breaker.state == CircuitState.HALF_OPEN

    async def test_half_open_success_closes_circuit(self):
        """Test that success in HALF_OPEN state closes the circuit."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            # Open the circuit
            for _ in range(2):
                with pytest.raises(ValueError):
                    async with breaker:
                        raise ValueError("test error")

            assert breaker.state == CircuitState.OPEN
            assert breaker.failure_count == 2

            # Wait for timeout
            current_time = 1.0

            # Successful operation in HALF_OPEN should close circuit
            async with breaker:
                pass  # Success

            assert breaker.state == CircuitState.CLOSED
            assert breaker.failure_count == 0

    async def test_half_open_failure_reopens_circuit(self):
        """Test that failure in HALF_OPEN state reopens the circuit."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            # Open the circuit
            for _ in range(2):
                with pytest.raises(ValueError):
                    async with breaker:
                        raise ValueError("test error")

            assert breaker.state == CircuitState.OPEN

            # Wait for timeout to enter HALF_OPEN
            current_time = 1.0

            # Failed operation in HALF_OPEN should reopen circuit
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

            assert breaker.state == CircuitState.OPEN
            # opened_at should be updated to current time
            assert breaker._opened_at == pytest.approx(1.0)

    async def test_half_open_max_calls_limit(self):
        """Test that HALF_OPEN state limits concurrent calls."""
        breaker = CircuitBreaker(
            failure_threshold=2, timeout=1.0, half_open_max_calls=2
        )

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            # Open the circuit
            for _ in range(2):
                with pytest.raises(ValueError):
                    async with breaker:
                        raise ValueError("test error")

            # Wait for timeout
            current_time = 1.0

            # First two calls should be allowed
            await breaker.acquire()
            assert breaker._half_open_calls == 1

            await breaker.acquire()
            assert breaker._half_open_calls == 2

            # Third call should be blocked
            with pytest.raises(CircuitBreakerOpenError, match="at capacity"):
                await breaker.acquire()

    async def test_decorator_pattern(self):
        """Test decorator pattern."""
        breaker = CircuitBreaker(failure_threshold=2)

        call_count = 0

        @breaker
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("test error")
            return "success"

        # First two calls fail
        with pytest.raises(ValueError):
            await test_func()

        with pytest.raises(ValueError):
            await test_func()

        assert breaker.state == CircuitState.OPEN

        # Third call blocked by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            await test_func()

    async def test_manual_success_tracking(self):
        """Test manual success tracking."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Manual failure tracking
        await breaker.acquire()
        await breaker.record_failure()

        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

        # Manual success tracking resets count
        await breaker.acquire()
        await breaker.record_success()

        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    async def test_manual_failure_tracking(self):
        """Test manual failure tracking."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Manually record failures
        await breaker.acquire()
        await breaker.record_failure()

        await breaker.acquire()
        await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 2

    async def test_success_resets_failure_count(self):
        """Test that success resets failure count in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=5)

        # Record some failures
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

        assert breaker.failure_count == 3

        # Success should reset count
        async with breaker:
            pass  # Success

        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    async def test_invalid_failure_threshold(self):
        """Test that invalid failure_threshold raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be > 0"):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be > 0"):
            CircuitBreaker(failure_threshold=-1)

    async def test_invalid_timeout(self):
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            CircuitBreaker(timeout=0.0)

        with pytest.raises(ValueError, match="timeout must be > 0"):
            CircuitBreaker(timeout=-1.0)

    async def test_invalid_half_open_max_calls(self):
        """Test that invalid half_open_max_calls raises ValueError."""
        with pytest.raises(ValueError, match="half_open_max_calls must be > 0"):
            CircuitBreaker(half_open_max_calls=0)

        with pytest.raises(ValueError, match="half_open_max_calls must be > 0"):
            CircuitBreaker(half_open_max_calls=-1)

    async def test_concurrent_operations(self):
        """Test that circuit breaker handles concurrent operations safely."""
        breaker = CircuitBreaker(failure_threshold=10)

        results = []

        async def operation(i, should_fail):
            try:
                async with breaker:
                    if should_fail:
                        raise ValueError(f"error {i}")
                    results.append(f"success {i}")
            except (ValueError, CircuitBreakerOpenError):
                results.append(f"failed {i}")

        # Run mix of successful and failing operations
        tasks = []
        for i in range(5):
            tasks.append(operation(i, should_fail=(i % 2 == 0)))

        await asyncio.gather(*tasks)

        # All operations should complete (not all successful)
        assert len(results) == 5

    async def test_multiple_timeout_cycles(self):
        """Test multiple open/half-open/open cycles."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            # Cycle 1: Open circuit
            for _ in range(2):
                with pytest.raises(ValueError):
                    async with breaker:
                        raise ValueError("test error")

            assert breaker.state == CircuitState.OPEN

            # Wait and enter HALF_OPEN, then fail to reopen circuit
            current_time = 1.0

            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

            assert breaker.state == CircuitState.OPEN
            # opened_at should be updated to current time
            assert breaker._opened_at == pytest.approx(1.0)

            # Cycle 2: Wait again and succeed
            current_time = 2.0
            async with breaker:
                pass  # Success

            assert breaker.state == CircuitState.CLOSED
            assert breaker.failure_count == 0

    async def test_state_property_snapshot(self):
        """Test that state property returns a snapshot."""
        breaker = CircuitBreaker(failure_threshold=2)

        initial_state = breaker.state
        assert initial_state == CircuitState.CLOSED

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

        new_state = breaker.state
        assert new_state == CircuitState.OPEN
        assert initial_state != new_state

    async def test_failure_count_property_snapshot(self):
        """Test that failure_count property returns a snapshot."""
        breaker = CircuitBreaker(failure_threshold=5)

        initial_count = breaker.failure_count
        assert initial_count == 0

        # Record a failure
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        new_count = breaker.failure_count
        assert new_count == 1
        assert initial_count != new_count

    async def test_exception_propagation(self):
        """Test that exceptions are properly propagated."""
        breaker = CircuitBreaker(failure_threshold=5)

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            async with breaker:
                raise CustomError("custom error")

        assert breaker.failure_count == 1

    async def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        breaker = CircuitBreaker()

        @breaker
        async def test_func():
            """Test docstring."""
            pass

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test docstring."

    async def test_decorator_with_arguments(self):
        """Test decorator with function arguments."""
        breaker = CircuitBreaker(failure_threshold=5)

        @breaker
        async def test_func(x: int, y: int) -> int:
            return x + y

        result = await test_func(5, 3)
        assert result == 8
        assert breaker.failure_count == 0

    async def test_open_state_error_message(self):
        """Test that CircuitBreakerOpenError has informative message."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

        # Check error message
        with pytest.raises(CircuitBreakerOpenError, match="Circuit breaker is OPEN"):
            await breaker.acquire()

    async def test_half_open_capacity_error_message(self):
        """Test that HALF_OPEN capacity error has informative message."""
        breaker = CircuitBreaker(
            failure_threshold=2, timeout=1.0, half_open_max_calls=1
        )

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            # Open the circuit
            for _ in range(2):
                with pytest.raises(ValueError):
                    async with breaker:
                        raise ValueError("test error")

            # Wait for timeout
            current_time = 1.0

            # First call allowed
            await breaker.acquire()

            # Second call blocked with informative message
            with pytest.raises(
                CircuitBreakerOpenError, match="HALF_OPEN and at capacity"
            ):
                await breaker.acquire()

    async def test_rapid_state_transitions(self):
        """Test rapid state transitions work correctly."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)

        current_time = 0.0

        async def mock_sleep(duration):
            nonlocal current_time
            current_time += duration

        def mock_time():
            return current_time

        with (
            patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            # Fail -> OPEN
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")
            assert breaker.state == CircuitState.OPEN

            # Wait -> HALF_OPEN
            await asyncio.sleep(0.1)
            await breaker.acquire()
            assert breaker.state == CircuitState.HALF_OPEN

            # Success -> CLOSED
            await breaker.record_success()
            assert breaker.state == CircuitState.CLOSED

            # Fail -> OPEN again
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")
            assert breaker.state == CircuitState.OPEN

    async def test_high_failure_threshold(self):
        """Test circuit breaker with high failure threshold."""
        breaker = CircuitBreaker(failure_threshold=100)

        # Record many failures
        for _ in range(99):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 99

        # 100th failure should open circuit
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 100

    async def test_fractional_timeout(self):
        """Test circuit breaker with fractional timeout."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.5)

        current_time = 0.0

        def mock_time():
            return current_time

        with patch.object(asyncio.get_event_loop(), "time", side_effect=mock_time):
            # Open circuit
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("test error")

            assert breaker.state == CircuitState.OPEN

            # Before timeout
            current_time = 0.4
            with pytest.raises(CircuitBreakerOpenError):
                await breaker.acquire()

            # After timeout
            current_time = 0.5
            await breaker.acquire()
            assert breaker.state == CircuitState.HALF_OPEN
