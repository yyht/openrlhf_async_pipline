

from loguru import logger as loggeru
import time
class Timer:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        loggeru.opt(depth=1).info(f"{self.message}, time cost: {time.time() - self.start_time:.2f}s")
