"""Unit tests for the async retry decorator."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from async_tools.retry import retry


@pytest.mark.unit  # Custom label (used for grouping tests)
@pytest.mark.asyncio  # Required for async tests
class TestRetryDecorator:
    """Test suite for the retry decorator."""

    async def test_success_on_first_attempt(self) -> None:
        """Function succeeds on first attempt, no retries needed."""
        call_count = 0

        @retry()
        async def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count == 1

    async def test_success_after_retries(self) -> None:
        """Function fails initially but succeeds on retry."""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 3

    async def test_max_attempts_exhausted(self) -> None:
        """Function fails all attempts, exception propagates."""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        async def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError, match="Permanent failure"):
            await always_fails()

        assert call_count == 3

    async def test_exponential_backoff_timing(self) -> None:
        """Verify correct exponential backoff delay calculations."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            @retry(max_attempts=4, initial_delay=1.0, exponential_base=2.0)
            async def always_fails() -> None:
                raise ValueError("Fail")

            with pytest.raises(ValueError, match="Fail"):
                await always_fails()

            # Check sleep was called with correct delays
            # Attempt 1 fails -> sleep 1.0 (1.0 * 2^0)
            # Attempt 2 fails -> sleep 2.0 (1.0 * 2^1)
            # Attempt 3 fails -> sleep 4.0 (1.0 * 2^2)
            # Attempt 4 fails -> no more retries
            assert mock_sleep.call_count == 3
            mock_sleep.assert_has_calls([call(1.0), call(2.0), call(4.0)])

    async def test_max_delay_cap(self) -> None:
        """Exponential backoff respects max_delay limit."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            @retry(
                max_attempts=5,
                initial_delay=1.0,
                exponential_base=2.0,
                max_delay=3.0,
            )
            async def always_fails() -> None:
                raise ValueError("Fail")

            with pytest.raises(ValueError, match="Fail"):
                await always_fails()

            # Delays should be capped at max_delay=3.0
            # Attempt 1 fails -> sleep 1.0 (1.0 * 2^0)
            # Attempt 2 fails -> sleep 2.0 (1.0 * 2^1)
            # Attempt 3 fails -> sleep 3.0 (min(4.0, 3.0))
            # Attempt 4 fails -> sleep 3.0 (min(8.0, 3.0))
            # Attempt 5 fails -> no more retries
            assert mock_sleep.call_count == 4
            mock_sleep.assert_has_calls([call(1.0), call(2.0), call(3.0), call(3.0)])

    async def test_exception_filtering(self) -> None:
        """Only specified exceptions trigger retry, others propagate."""
        call_count = 0

        @retry(
            max_attempts=3,
            initial_delay=0.01,
            exceptions=(ValueError, ConnectionError),
        )
        async def selective_retry() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable")
            elif call_count == 2:
                raise TypeError("Not retryable")
            return "success"

        # TypeError is not in the exceptions list, so it should propagate immediately
        with pytest.raises(TypeError, match="Not retryable"):
            await selective_retry()

        # Should have tried twice: first ValueError (retry), then TypeError (fail)
        assert call_count == 2

    async def test_custom_callback_invocation(self) -> None:
        """Callback is called with correct parameters before each retry."""
        callback_calls: list[tuple[Exception, int, float]] = []

        def track_retry(exc: Exception, attempt: int, delay: float) -> None:
            callback_calls.append((exc, attempt, delay))

        @retry(
            max_attempts=3,
            initial_delay=1.0,
            exponential_base=2.0,
            on_retry=track_retry,
        )
        async def flaky_func() -> None:
            raise ValueError("Fail")

        with pytest.raises(ValueError, match="Fail"):
            await flaky_func()

        # Should have 2 callback calls (before retry 2 and retry 3)
        assert len(callback_calls) == 2

        # Check first callback: before attempt 2, delay should be 1.0
        exc1, attempt1, delay1 = callback_calls[0]
        assert isinstance(exc1, ValueError)
        assert attempt1 == 2
        assert delay1 == 1.0

        # Check second callback: before attempt 3, delay should be 2.0
        exc2, attempt2, delay2 = callback_calls[1]
        assert isinstance(exc2, ValueError)
        assert attempt2 == 3
        assert delay2 == 2.0

    async def test_preserve_function_metadata(self) -> None:
        """Decorator preserves function name, docstring, and other metadata."""

        @retry()
        async def documented_function() -> str:
            """This is a test function with documentation."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert (
            documented_function.__doc__ == "This is a test function with documentation."
        )

    async def test_function_with_arguments(self) -> None:
        """Retry works correctly with functions that have arguments."""
        call_count = 0

        @retry(max_attempts=2, initial_delay=0.01)
        async def func_with_args(x: int, y: str, *, z: bool = False) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry")
            return f"{x}-{y}-{z}"

        result = await func_with_args(42, "test", z=True)

        assert result == "42-test-True"
        assert call_count == 2

    async def test_return_type_preservation(self) -> None:
        """Decorator preserves return type correctly."""

        @retry()
        async def returns_int() -> int:
            return 42

        @retry()
        async def returns_dict() -> dict[str, int]:
            return {"key": 123}

        int_result = await returns_int()
        dict_result = await returns_dict()

        assert isinstance(int_result, int)
        assert int_result == 42
        assert isinstance(dict_result, dict)
        assert dict_result == {"key": 123}

    async def test_no_retry_on_success(self) -> None:
        """No sleep occurs when function succeeds on first attempt."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            @retry(max_attempts=3)
            async def immediate_success() -> str:
                return "success"

            result = await immediate_success()

            assert result == "success"
            mock_sleep.assert_not_called()

    async def test_callback_not_called_on_success(self) -> None:
        """Callback is not invoked when function succeeds without retries."""
        callback_mock = MagicMock()

        @retry(max_attempts=3, on_retry=callback_mock)
        async def immediate_success() -> str:
            return "success"

        result = await immediate_success()

        assert result == "success"
        callback_mock.assert_not_called()

    async def test_single_attempt_no_retry(self) -> None:
        """With max_attempts=1, function is called once and exception propagates."""
        call_count = 0

        @retry(max_attempts=1)
        async def single_attempt() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Fail")

        with pytest.raises(ValueError, match="Fail"):
            await single_attempt()

        assert call_count == 1

    async def test_different_exponential_base(self) -> None:
        """Exponential backoff works with different base values."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            @retry(max_attempts=4, initial_delay=1.0, exponential_base=3.0)
            async def always_fails() -> None:
                raise ValueError("Fail")

            with pytest.raises(ValueError, match="Fail"):
                await always_fails()

            # With base 3.0: 1.0, 3.0, 9.0
            assert mock_sleep.call_count == 3
            mock_sleep.assert_has_calls([call(1.0), call(3.0), call(9.0)])

    async def test_zero_initial_delay(self) -> None:
        """Retry works with zero initial delay."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

            @retry(max_attempts=3, initial_delay=0.0, exponential_base=2.0)
            async def always_fails() -> None:
                raise ValueError("Fail")

            with pytest.raises(ValueError, match="Fail"):
                await always_fails()

            # All delays should be 0.0
            assert mock_sleep.call_count == 2
            mock_sleep.assert_has_calls([call(0.0), call(0.0)])

    async def test_exception_specific_to_type(self) -> None:
        """Only exact exception types in the tuple are caught."""
        call_count = 0

        # Define custom exception hierarchy
        class CustomError(Exception):
            pass

        class SpecificError(CustomError):
            pass

        @retry(max_attempts=3, initial_delay=0.01, exceptions=(SpecificError,))
        async def specific_exception() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SpecificError("Caught")
            else:
                raise CustomError("Not caught")

        # SpecificError triggers retry, but CustomError propagates
        with pytest.raises(CustomError, match="Not caught"):
            await specific_exception()

        assert call_count == 2


@pytest.mark.unit
class TestRetryValidation:
    """Test parameter validation for the retry decorator."""

    def test_invalid_max_attempts(self) -> None:
        """max_attempts < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            retry(max_attempts=0)

    def test_invalid_initial_delay(self) -> None:
        """Negative initial_delay raises ValueError."""
        with pytest.raises(ValueError, match="initial_delay must be non-negative"):
            retry(initial_delay=-1.0)

    def test_invalid_max_delay(self) -> None:
        """Negative max_delay raises ValueError."""
        with pytest.raises(ValueError, match="max_delay must be non-negative"):
            retry(max_delay=-1.0)

    def test_invalid_exponential_base(self) -> None:
        """exponential_base < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"exponential_base must be at least 1\.0"):
            retry(exponential_base=0.5)

    def test_valid_edge_case_parameters(self) -> None:
        """Edge case valid parameters don't raise errors."""
        # These should not raise
        retry(max_attempts=1)  # Minimum attempts
        retry(initial_delay=0.0)  # Zero delay
        retry(max_delay=0.0)  # Zero max delay
        retry(exponential_base=1.0)  # Base of 1 (no growth)
