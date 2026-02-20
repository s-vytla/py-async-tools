"""Example usage of the retry decorator."""

import asyncio

from async_tools import retry


# Example 1: Basic retry with default settings
@retry()
async def flaky_api_call() -> str:
    """Simulates a flaky API call that succeeds on the 2nd attempt."""
    print("Attempting API call...")
    if not hasattr(flaky_api_call, "attempt_count"):
        flaky_api_call.attempt_count = 0
    flaky_api_call.attempt_count += 1

    if flaky_api_call.attempt_count < 2:
        raise ConnectionError("Network error")
    return "Success!"


# Example 2: Custom retry with logging callback
def log_retry(exc: Exception, attempt: int, delay: float) -> None:
    """Log retry attempts."""
    print(
        f"  Retry #{attempt - 1} after {delay:.1f}s due to {type(exc).__name__}: {exc}"
    )


@retry(
    max_attempts=4,
    initial_delay=0.5,
    exponential_base=2.0,
    exceptions=(ValueError, ConnectionError),
    on_retry=log_retry,
)
async def unreliable_service() -> str:
    """Simulates an unreliable service."""
    print("Calling unreliable service...")
    if not hasattr(unreliable_service, "call_count"):
        unreliable_service.call_count = 0
    unreliable_service.call_count += 1

    if unreliable_service.call_count < 3:
        raise ValueError(
            f"Service unavailable (attempt {unreliable_service.call_count})"
        )
    return f"Service responded after {unreliable_service.call_count} attempts"


# Example 3: Retry that exhausts all attempts
@retry(max_attempts=3, initial_delay=0.2)
async def always_fails() -> None:
    """Function that always fails."""
    print("Attempting operation...")
    raise RuntimeError("This operation always fails")


async def main() -> None:
    """Run examples."""
    print("=" * 60)
    print("Example 1: Basic retry (succeeds on 2nd attempt)")
    print("=" * 60)
    result = await flaky_api_call()
    print(f"Result: {result}\n")

    print("=" * 60)
    print("Example 2: Retry with logging callback")
    print("=" * 60)
    result = await unreliable_service()
    print(f"Result: {result}\n")

    print("=" * 60)
    print("Example 3: Exhausting all retry attempts")
    print("=" * 60)
    try:
        await always_fails()
    except RuntimeError as e:
        print(f"Failed after all retries: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
