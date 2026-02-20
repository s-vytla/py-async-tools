"""Integration tests combining multiple resilience utilities."""

import asyncio

import pytest

from async_tools.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState
from async_tools.rate_limit import (
    PerKeyRateLimiter,
    SemaphoreRateLimiter,
    TokenBucketRateLimiter,
)
from async_tools.retry import retry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_with_circuit_breaker() -> None:
    """Combine retry and circuit breaker for flaky service."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=0.5)
    call_count = 0

    @retry(max_attempts=5, initial_delay=0.05, exponential_base=2.0)
    async def flaky_service_call() -> str:
        nonlocal call_count
        call_count += 1

        async with breaker:
            await asyncio.sleep(0.01)
            # Fail first 2 times, then succeed
            if call_count <= 2:
                raise ConnectionError("Service temporarily down")
            return "success"

    result = await flaky_service_call()

    assert result == "success"
    assert call_count == 3
    # Circuit should still be closed (didn't reach threshold)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0  # Reset on success


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_circuit_breaker_prevents_retry_waste() -> None:
    """Circuit breaker stops retry from wasting attempts."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.3)
    retry_attempts = 0

    @retry(max_attempts=10, initial_delay=0.05, exceptions=(ConnectionError,))
    async def failing_service() -> None:
        nonlocal retry_attempts
        retry_attempts += 1

        async with breaker:
            await asyncio.sleep(0.01)
            raise ConnectionError("Service down")

    # Retry doesn't catch CircuitBreakerOpenError, so it will propagate
    with pytest.raises((ConnectionError, CircuitBreakerOpenError)):
        await failing_service()

    # The circuit opens after 2 failures
    # Then subsequent attempts hit CircuitBreakerOpenError which isn't retried
    # So we should have 2 failures + 1 attempt that hits open circuit
    assert 2 <= retry_attempts <= 3
    assert breaker.state == CircuitState.OPEN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rate_limiter_with_retry() -> None:
    """Combine rate limiter with retry for rate-limited API."""
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0)
    call_count = 0

    @retry(max_attempts=3, initial_delay=0.05, exceptions=(ConnectionError,))
    async def rate_limited_api_call(request_id: int) -> str:
        nonlocal call_count

        async with limiter:
            call_count += 1
            await asyncio.sleep(0.01)

            # Simulate occasional failures
            if call_count % 5 == 0:
                raise ConnectionError("Temporary network issue")

            return f"response_{request_id}"

    # Make multiple API calls
    results = await asyncio.gather(*[rate_limited_api_call(i) for i in range(15)])

    assert len(results) == 15
    # Some requests were retried due to failures
    assert call_count >= 15


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_breaker_with_rate_limiter() -> None:
    """Combine circuit breaker with rate limiter for overloaded service."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=0.5)
    limiter = SemaphoreRateLimiter(max_concurrent=2)

    success_count = 0
    failure_count = 0

    async def protected_service_call(should_fail: bool) -> str:
        nonlocal success_count, failure_count

        # Rate limit first
        async with limiter:
            # Then circuit breaker
            async with breaker:
                await asyncio.sleep(0.02)

                if should_fail:
                    failure_count += 1
                    raise ConnectionError("Service error")

                success_count += 1
                return "success"

    # First few succeed
    for _ in range(2):
        result = await protected_service_call(False)
        assert result == "success"

    # Then failures occur
    for _ in range(3):
        try:
            await protected_service_call(True)
        except ConnectionError:
            pass

    # Circuit should be open now
    assert breaker.state == CircuitState.OPEN

    # Subsequent calls fail immediately (not rate limited)
    with pytest.raises(CircuitBreakerOpenError):
        await protected_service_call(False)

    assert success_count == 2
    assert failure_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_per_key_limiter_with_retry() -> None:
    """Combine per-key rate limiter with retry for multi-user API."""
    limiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=5.0, capacity=5.0),
        cleanup_interval=10.0,
    )

    call_counts: dict[str, int] = {}

    @retry(max_attempts=3, initial_delay=0.05)
    @limiter.decorator(key_func=lambda user_id, _: user_id)
    async def user_api_call(user_id: str, request: str) -> str:
        if user_id not in call_counts:
            call_counts[user_id] = 0
        call_counts[user_id] += 1

        await asyncio.sleep(0.01)

        # Simulate occasional failures
        if call_counts[user_id] % 7 == 1:
            raise ConnectionError("Network error")

        return f"{user_id}:{request}"

    # Make requests for multiple users
    operations = []
    for user in ["alice", "bob"]:
        for i in range(10):
            operations.append(user_api_call(user, f"req{i}"))

    results = await asyncio.gather(*operations)

    assert len(results) == 20
    # Each user's requests were rate limited independently
    for user in ["alice", "bob"]:
        user_results = [r for r in results if r.startswith(user)]
        assert len(user_results) == 10

    await limiter.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_triple_protection_stack() -> None:
    """Test all three utilities together: retry + circuit breaker + rate limiter."""
    breaker = CircuitBreaker(failure_threshold=4, timeout=0.5)
    limiter = SemaphoreRateLimiter(max_concurrent=2)

    operation_count = 0
    success_count = 0

    @retry(max_attempts=3, initial_delay=0.05, exceptions=(ConnectionError,))
    async def triple_protected_operation(op_id: int) -> str:
        nonlocal operation_count, success_count

        # Rate limit first
        async with limiter:
            # Then circuit breaker
            async with breaker:
                operation_count += 1
                await asyncio.sleep(0.02)

                # Succeed on all operations
                success_count += 1
                return f"result_{op_id}"

    # Run operations successfully
    results = await asyncio.gather(*[triple_protected_operation(i) for i in range(10)])

    assert len(results) == 10
    assert success_count == 10
    # Circuit should remain closed
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_recovery_with_all_utilities() -> None:
    """Test service recovery with combined utilities."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.3)
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0)

    phase = "failing"
    call_count = 0

    @retry(max_attempts=5, initial_delay=0.05, exponential_base=1.5)
    async def recovering_service() -> str:
        nonlocal call_count, phase
        call_count += 1

        async with limiter:
            async with breaker:
                await asyncio.sleep(0.01)

                if phase == "failing":
                    raise ConnectionError("Service down")
                else:
                    return "recovered"

    # Phase 1: Service fails, circuit opens
    for _ in range(2):
        try:
            await recovering_service()
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    assert breaker.state == CircuitState.OPEN

    # Phase 2: Wait for circuit to transition to half-open
    await asyncio.sleep(0.35)

    # Phase 3: Service recovers
    phase = "recovered"

    # This should succeed and close the circuit
    result = await recovering_service()

    assert result == "recovered"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_users_with_protection() -> None:
    """Test multiple concurrent users with per-key protection."""
    per_key_limiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=5.0, capacity=5.0),
        cleanup_interval=10.0,
    )
    global_limiter = SemaphoreRateLimiter(max_concurrent=5)

    results_by_user: dict[str, list[str]] = {
        f"user{i}": [] for i in range(5)
    }

    async def user_request(user_id: str, request_id: int) -> str:
        # Global concurrency limit
        async with global_limiter:
            # Per-user rate limit
            async with per_key_limiter.for_key(user_id):
                await asyncio.sleep(0.01)
                result = f"{user_id}_req{request_id}"
                results_by_user[user_id].append(result)
                return result

    # Each user makes 8 requests
    operations = []
    for user_num in range(5):
        user_id = f"user{user_num}"
        for req_id in range(8):
            operations.append(user_request(user_id, req_id))

    start = asyncio.get_event_loop().time()
    results = await asyncio.gather(*operations)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(results) == 40

    # Each user should have completed all requests
    for user_id in results_by_user:
        assert len(results_by_user[user_id]) == 8

    # Should complete in reasonable time with concurrent processing
    assert elapsed < 5.0

    await per_key_limiter.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_breaker_isolation_per_key() -> None:
    """Test that per-key limiters can be combined with per-key circuit breakers."""
    # Each user gets their own circuit breaker
    breakers: dict[str, CircuitBreaker] = {
        "alice": CircuitBreaker(failure_threshold=2, timeout=0.5),
        "bob": CircuitBreaker(failure_threshold=2, timeout=0.5),
    }

    limiter = PerKeyRateLimiter(
        limiter_factory=lambda: TokenBucketRateLimiter(rate=5.0, capacity=5.0),
        cleanup_interval=10.0,
    )

    async def user_operation(user_id: str, should_fail: bool) -> str:
        breaker = breakers[user_id]

        async with limiter.for_key(user_id):
            async with breaker:
                await asyncio.sleep(0.01)
                if should_fail:
                    raise ConnectionError(f"{user_id} operation failed")
                return f"{user_id}_success"

    # Alice's operations fail
    for _ in range(2):
        try:
            await user_operation("alice", True)
        except ConnectionError:
            pass

    # Alice's circuit should be open
    assert breakers["alice"].state == CircuitState.OPEN

    # Bob's operations succeed
    result = await user_operation("bob", False)
    assert result == "bob_success"

    # Bob's circuit should still be closed
    assert breakers["bob"].state == CircuitState.CLOSED

    # Alice's further operations fail immediately
    with pytest.raises(CircuitBreakerOpenError):
        await user_operation("alice", False)

    await limiter.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_respects_circuit_breaker_state() -> None:
    """Test that retry stops when circuit breaker opens."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.5)
    retry_count = 0

    @retry(max_attempts=10, initial_delay=0.05, exceptions=(ConnectionError,))
    async def smart_retry() -> None:
        nonlocal retry_count
        retry_count += 1

        async with breaker:
            await asyncio.sleep(0.01)
            raise ConnectionError("Persistent failure")

    with pytest.raises((ConnectionError, CircuitBreakerOpenError)):
        await smart_retry()

    # Should have stopped retrying when circuit opened
    # Circuit opens after 2 failures, then CircuitBreakerOpenError is raised
    # which isn't in the retry exceptions list, so it propagates immediately
    assert 2 <= retry_count <= 3
    assert breaker.state == CircuitState.OPEN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_layered_protection_error_handling() -> None:
    """Test that errors propagate correctly through multiple protection layers."""
    breaker = CircuitBreaker(failure_threshold=5, timeout=1.0)
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=10.0)

    @retry(max_attempts=2, initial_delay=0.05)
    async def protected_operation(error_type: str) -> str:
        async with limiter:
            async with breaker:
                await asyncio.sleep(0.01)

                if error_type == "connection":
                    raise ConnectionError("Network error")
                elif error_type == "value":
                    raise ValueError("Invalid input")
                else:
                    return "success"

    # ConnectionError should propagate and be retried
    with pytest.raises(ConnectionError):
        await protected_operation("connection")

    # ValueError should propagate immediately (not in retry exceptions)
    with pytest.raises(ValueError):
        await protected_operation("value")

    # Success should work
    result = await protected_operation("success")
    assert result == "success"
