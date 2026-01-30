import asyncio
import logging
import threading
from typing import Callable, Any, TypeVar, Coroutine
from ratelimitqueue import RateLimitQueue

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GlobalRateLimiter:
    """
    A global rate limiter that ensures operations are executed at a specific rate.
    Uses ratelimitqueue.RateLimitQueue for the heavy lifting.

    Since RateLimitQueue is thread-based/blocking, we run a worker in a separate thread
    to avoid blocking the asyncio event loop.
    """

    def __init__(self, calls: int = 1, period: float = 1.0):
        self.queue = RateLimitQueue(calls=calls, per=period)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info(f"GlobalRateLimiter started with {calls} calls per {period}s")

    async def execute(self, func: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """
        Schedule a function to be executed within the rate limit.
        Returns the result of the function.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.queue.put((func, future, loop))
        return await future

    def _worker(self):
        """Worker thread that pulls from the RateLimitQueue and executes tasks."""
        while not self._stop_event.is_set():
            try:
                # get() blocks based on rate limit
                # We don't use timeout here to avoid the skip issue
                # instead we depend on the queue blocking
                item = self.queue.get(block=True)
                task_fn, future, loop = item

                # Execute the task
                async def run_task(f=task_fn, fut=future, l=loop):
                    try:
                        result = await f()
                        l.call_soon_threadsafe(fut.set_result, result)
                    except Exception as e:
                        l.call_soon_threadsafe(fut.set_exception, e)

                asyncio.run_coroutine_threadsafe(run_task(), loop)
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error in GlobalRateLimiter worker: {e}")
                time.sleep(0.1)

    def stop(self):
        """Stop the background worker."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
