"""Integration tests for circuit breaker with realistic failure scenarios."""

import asyncio

import pytest

from async_tools.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_opens_after_failures() -> None:
    """Test circuit breaker opens after reaching failure threshold."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
    failure_count = 0

    async def failing_operation() -> None:
        nonlocal failure_count
        failure_count += 1
        await asyncio.sleep(0.02)
        raise ConnectionError("Service unavailable")

    # First 3 failures should go through
    for i in range(3):
        try:
            async with breaker:
                await failing_operation()
        except ConnectionError:
            pass
        assert breaker.state == (CircuitState.OPEN if i == 2 else CircuitState.CLOSED)

    assert breaker.state == CircuitState.OPEN
    assert failure_count == 3

    # Circuit is now open, should fail immediately
    with pytest.raises(CircuitBreakerOpenError):
        async with breaker:
            await failing_operation()

    # Failure count should still be 3 (operation didn't run)
    assert failure_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_circuit_recovers_after_timeout() -> None:
    """Test circuit transitions to half-open after timeout and recovers."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.3)
    call_count = 0

    async def flaky_operation() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.02)
        # Fail first 2 times, then succeed
        if call_count <= 2:
            raise ConnectionError("Service down")
        return "success"

    # Cause 2 failures to open the circuit
    for _ in range(2):
        try:
            async with breaker:
                await flaky_operation()
        except ConnectionError:
            pass

    assert breaker.state == CircuitState.OPEN
    assert call_count == 2

    # Circuit is open, immediate failure
    with pytest.raises(CircuitBreakerOpenError):
        async with breaker:
            await flaky_operation()

    assert call_count == 2  # Operation didn't run

    # Wait for timeout to transition to half-open
    await asyncio.sleep(0.4)

    # Next call should go through (half-open)
    result = await breaker.acquire()
    assert result is None  # acquire() succeeded

    # Manually test the operation and record success
    result_str = await flaky_operation()
    await breaker.record_success()

    assert result_str == "success"
    assert breaker.state == CircuitState.CLOSED
    assert call_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_circuit_half_open_failure_reopens() -> None:
    """Test that failure in half-open state reopens the circuit."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.2)
    call_count = 0

    async def still_failing() -> None:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        raise ConnectionError("Still down")

    # Open the circuit with 2 failures
    for _ in range(2):
        try:
            async with breaker:
                await still_failing()
        except ConnectionError:
            pass

    assert breaker.state == CircuitState.OPEN
    initial_call_count = call_count

    # Wait for half-open transition
    await asyncio.sleep(0.25)

    # Half-open state allows one call through
    try:
        async with breaker:
            await still_failing()
    except ConnectionError:
        pass

    # Circuit should be open again
    assert breaker.state == CircuitState.OPEN
    assert call_count == initial_call_count + 1

    # Subsequent calls should fail immediately
    with pytest.raises(CircuitBreakerOpenError):
        async with breaker:
            await still_failing()

    assert call_count == initial_call_count + 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_resets_on_success() -> None:
    """Test that successes reset the failure count in closed state."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
    operation_count = 0

    async def sometimes_fails(should_fail: bool) -> str:
        nonlocal operation_count
        operation_count += 1
        await asyncio.sleep(0.01)
        if should_fail:
            raise ConnectionError("Failed")
        return "success"

    # Fail once
    try:
        async with breaker:
            await sometimes_fails(True)
    except ConnectionError:
        pass

    assert breaker.failure_count == 1
    assert breaker.state == CircuitState.CLOSED

    # Success should reset counter
    async with breaker:
        result = await sometimes_fails(False)

    assert result == "success"
    assert breaker.failure_count == 0
    assert breaker.state == CircuitState.CLOSED

    # Fail twice more (under threshold)
    for _ in range(2):
        try:
            async with breaker:
                await sometimes_fails(True)
        except ConnectionError:
            pass

    assert breaker.failure_count == 2
    assert breaker.state == CircuitState.CLOSED

    # One more success should reset
    async with breaker:
        result = await sometimes_fails(False)

    assert breaker.failure_count == 0
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests_with_circuit_breaker() -> None:
    """Test circuit breaker with multiple concurrent requests."""
    breaker = CircuitBreaker(failure_threshold=5, timeout=0.5)
    success_count = 0
    failure_count = 0

    async def concurrent_operation(op_id: int) -> str:
        nonlocal success_count, failure_count
        await asyncio.sleep(0.01)

        # Operations 0-4 succeed, 5-9 fail
        if op_id < 5:
            success_count += 1
            return f"success_{op_id}"
        else:
            failure_count += 1
            raise ConnectionError(f"Failed {op_id}")

    results: list[str | None] = []
    errors: list[Exception] = []

    # Launch 10 concurrent operations
    async def run_operation(op_id: int) -> None:
        try:
            async with breaker:
                result = await concurrent_operation(op_id)
                results.append(result)
        except (ConnectionError, CircuitBreakerOpenError) as e:
            errors.append(e)
            results.append(None)

    await asyncio.gather(*[run_operation(i) for i in range(10)])

    # Some should succeed, some fail
    assert success_count >= 5
    assert len([r for r in results if r is not None]) >= 5

    # Circuit may or may not be open depending on execution order
    # At least we verify no crashes with concurrent access


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_breaker_as_decorator() -> None:
    """Test circuit breaker used as a decorator."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.5)
    call_count = 0

    @breaker
    async def decorated_operation(should_fail: bool) -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        if should_fail:
            raise ConnectionError("Operation failed")
        return "success"

    # Cause 2 failures to open circuit
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await decorated_operation(True)

    assert breaker.state == CircuitState.OPEN
    assert call_count == 2

    # Circuit is open, should raise CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError):
        await decorated_operation(False)

    # Operation didn't run
    assert call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_multiple_recovery_cycles() -> None:
    """Test circuit breaker through multiple failure and recovery cycles."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.2)
    cycle_count = 0

    async def cycling_operation(should_fail: bool) -> str:
        nonlocal cycle_count
        await asyncio.sleep(0.01)
        if should_fail:
            raise ConnectionError(f"Cycle {cycle_count} failed")
        return f"Cycle {cycle_count} success"

    # Cycle 1: Fail and open
    for _ in range(2):
        try:
            async with breaker:
                await cycling_operation(True)
        except ConnectionError:
            pass

    assert breaker.state == CircuitState.OPEN
    cycle_count += 1

    # Wait for recovery
    await asyncio.sleep(0.25)

    # Recover
    async with breaker:
        result = await cycling_operation(False)

    assert "success" in result
    assert breaker.state == CircuitState.CLOSED
    cycle_count += 1

    # Cycle 2: Fail and open again
    for _ in range(2):
        try:
            async with breaker:
                await cycling_operation(True)
        except ConnectionError:
            pass

    assert breaker.state == CircuitState.OPEN
    cycle_count += 1

    # Wait and recover again
    await asyncio.sleep(0.25)

    async with breaker:
        result = await cycling_operation(False)

    assert "success" in result
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_breaker_with_manual_tracking() -> None:
    """Test manual success/failure tracking for fine-grained control."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=0.5)

    async def complex_operation() -> str:
        await asyncio.sleep(0.02)
        # Simulate partial operation that needs manual tracking
        return "partial_result"

    # Manual tracking allows custom success/failure logic
    # Record 2 failures first (below threshold)
    for i in range(2):
        await breaker.acquire()
        result = await complex_operation()
        await breaker.record_failure()

    assert breaker.failure_count == 2
    assert breaker.state == CircuitState.CLOSED

    # Then record a success to reset
    await breaker.acquire()
    result = await complex_operation()
    await breaker.record_success()

    assert breaker.failure_count == 0
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_circuit_breaker_timing_precision() -> None:
    """Test that circuit breaker timeout is relatively accurate."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.3)

    async def failing_op() -> None:
        await asyncio.sleep(0.01)
        raise ConnectionError("Failed")

    # Open the circuit
    for _ in range(2):
        try:
            async with breaker:
                await failing_op()
        except ConnectionError:
            pass

    assert breaker.state == CircuitState.OPEN
    open_time = asyncio.get_event_loop().time()

    # Wait just under timeout
    await asyncio.sleep(0.25)

    # Should still be blocked
    with pytest.raises(CircuitBreakerOpenError):
        async with breaker:
            await failing_op()

    # Wait past timeout
    await asyncio.sleep(0.1)
    elapsed = asyncio.get_event_loop().time() - open_time

    # Should now allow calls (half-open)
    # We just check that acquire succeeds
    await breaker.acquire()
    await breaker.record_success()  # Transition to closed

    assert breaker.state == CircuitState.CLOSED
    assert elapsed >= 0.3
