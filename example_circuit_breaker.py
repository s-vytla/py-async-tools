"""Example demonstrating circuit breaker usage."""

import asyncio
import random

from async_tools.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError


# Simulated unreliable API that fails randomly
class UnreliableAPI:
    """Simulates an unreliable external API."""

    def __init__(self, failure_rate: float = 0.7):
        self.failure_rate = failure_rate
        self.call_count = 0

    async def call(self) -> str:
        """Make an API call that may fail."""
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate network delay

        if random.random() < self.failure_rate:
            raise Exception(f"API call #{self.call_count} failed")

        return f"API call #{self.call_count} succeeded"


async def demo_context_manager() -> None:
    """Demonstrate context manager pattern."""
    print("\n=== Context Manager Pattern ===")

    breaker = CircuitBreaker(failure_threshold=3, timeout=2.0)
    api = UnreliableAPI(failure_rate=0.8)  # 80% failure rate

    for i in range(10):
        try:
            async with breaker:
                result = await api.call()
                print(f"✓ Call {i + 1}: {result}")
        except CircuitBreakerOpenError:
            print(
                f"✗ Call {i + 1}: Circuit breaker is OPEN "
                f"(failures: {breaker.failure_count})"
            )
        except Exception as e:
            print(f"✗ Call {i + 1}: {e}")

        await asyncio.sleep(0.2)

        # Show current state
        print(f"  State: {breaker.state.value}, Failures: {breaker.failure_count}")


async def demo_decorator() -> None:
    """Demonstrate decorator pattern."""
    print("\n=== Decorator Pattern ===")

    breaker = CircuitBreaker(failure_threshold=2, timeout=1.5)
    api = UnreliableAPI(failure_rate=0.7)

    @breaker
    async def make_api_call() -> str:
        """Decorated function that calls the unreliable API."""
        return await api.call()

    for i in range(8):
        try:
            result = await make_api_call()
            print(f"✓ Call {i + 1}: {result}")
        except CircuitBreakerOpenError:
            print(
                f"✗ Call {i + 1}: Circuit breaker is OPEN "
                f"(state: {breaker.state.value})"
            )
        except Exception as e:
            print(f"✗ Call {i + 1}: {e}")

        await asyncio.sleep(0.3)


async def demo_manual_tracking() -> None:
    """Demonstrate manual tracking pattern."""
    print("\n=== Manual Tracking Pattern ===")

    breaker = CircuitBreaker(failure_threshold=3, timeout=2.0)
    api = UnreliableAPI(failure_rate=0.6)

    for i in range(8):
        try:
            # Manually acquire permission
            await breaker.acquire()

            # Attempt operation
            result = await api.call()
            print(f"✓ Call {i + 1}: {result}")

            # Manually record success
            await breaker.record_success()

        except CircuitBreakerOpenError:
            print(
                f"✗ Call {i + 1}: Circuit breaker is OPEN "
                f"(failures: {breaker.failure_count})"
            )
        except Exception as e:
            print(f"✗ Call {i + 1}: {e}")

            # Manually record failure
            await breaker.record_failure()

        await asyncio.sleep(0.2)


async def demo_recovery() -> None:
    """Demonstrate circuit recovery after failures."""
    print("\n=== Recovery Demo ===")

    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0, half_open_max_calls=2)

    # Create API that fails initially, then recovers
    api = UnreliableAPI(failure_rate=1.0)  # Start with 100% failure

    print("Phase 1: Triggering failures...")
    for i in range(5):
        try:
            async with breaker:
                result = await api.call()
                print(f"✓ Call {i + 1}: {result}")
        except CircuitBreakerOpenError:
            print(f"✗ Call {i + 1}: Circuit is OPEN")
        except Exception as e:
            print(f"✗ Call {i + 1}: {e}")

        print(f"  State: {breaker.state.value}, Failures: {breaker.failure_count}")
        await asyncio.sleep(0.1)

    print("\nPhase 2: Waiting for timeout...")
    await asyncio.sleep(1.1)  # Wait for timeout

    print("\nPhase 3: Testing recovery (API now reliable)...")
    api.failure_rate = 0.0  # API has recovered

    for i in range(5):
        try:
            async with breaker:
                result = await api.call()
                print(f"✓ Call {i + 1}: {result}")
        except CircuitBreakerOpenError:
            print(f"✗ Call {i + 1}: Circuit is still OPEN")
        except Exception as e:
            print(f"✗ Call {i + 1}: {e}")

        print(f"  State: {breaker.state.value}, Failures: {breaker.failure_count}")
        await asyncio.sleep(0.1)


async def main() -> None:
    """Run all examples."""
    print("Circuit Breaker Examples")
    print("=" * 50)

    # Set random seed for reproducible demo
    random.seed(42)

    await demo_context_manager()
    await asyncio.sleep(1)

    await demo_decorator()
    await asyncio.sleep(1)

    await demo_manual_tracking()
    await asyncio.sleep(1)

    await demo_recovery()


if __name__ == "__main__":
    asyncio.run(main())
