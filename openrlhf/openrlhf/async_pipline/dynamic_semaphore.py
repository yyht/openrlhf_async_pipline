
import os
import asyncio

class DynamicSemaphore:
    def __init__(self, initial_max):
        self.max_concurrent = initial_max
        self.current_used = 0
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()

    async def acquire(self):
        async with self._condition:
            await self._condition.wait_for(lambda: self.current_used < self.max_concurrent)
            self.current_used += 1

    async def release(self):
        async with self._condition:
            self.current_used -= 1
            self._condition.notify_all()

    def set_max(self, new_max):
        # This should be thread-safe if called from async context
        async def _adjust():
            async with self._condition:
                old_max = self.max_concurrent
                self.max_concurrent = new_max
                # If increasing capacity, notify waiting tasks
                if new_max > old_max:
                    self._condition.notify_all()
        # Schedule the adjustment in the event loop
        asyncio.create_task(_adjust())


class StatsCollector:
    def __init__(self):
        self.success = 0
        self.failures = 0
        self.total_time = 0.0
        self.lock = asyncio.Lock()

    async def record_success(self, elapsed):
        async with self.lock:
            self.success += 1
            self.total_time += elapsed

    async def record_failure(self):
        async with self.lock:
            self.failures += 1

    async def get_stats(self):
        async with self.lock:
            total = self.success + self.failures
            success_rate = self.success / total if total > 0 else 1.0
            avg_time = self.total_time / self.success if self.success > 0 else 0
            return success_rate, avg_time

    async def reset(self):
        async with self.lock:
            self.success = 0
            self.failures = 0
            self.total_time = 0.0