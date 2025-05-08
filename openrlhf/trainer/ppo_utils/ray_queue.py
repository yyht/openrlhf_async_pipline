
"""A queue implemented by Ray Actor."""
import asyncio
from copy import deepcopy
from typing import List

import ray
from openrlhf.utils.logging_utils import init_logger
from openrlhf.async_pipline.show_timer import Timer

logger = init_logger(__name__)

@ray.remote
class QueueActor:
    """An asyncio.Queue based queue actor."""

    def __init__(self, capacity=1024) -> None:
        self.capacity = capacity
        self.queue = asyncio.Queue(self.capacity)

    def length(self) -> int:
        """The length of the queue."""
        return self.queue.qsize()

    async def put_batch(self, batch) -> None:
        """Put batch of experience."""
        old_queue_size = self.queue.qsize()
        async with Timer("##ASYNC-PUT-QUEUE##"):
            await self.queue.put(batch)
        logger.info(f"Put batch of size {len(batch)} into queue.")
        new_queue_size = self.queue.qsize()
        if new_queue_size > old_queue_size:
            logger.info({
                'INFO': f"##STATUS-FOR-PUT-QUEUE##",
                "VALUE": f"Queue size increased from {old_queue_size} to {new_queue_size}."
            })
            return True
        else:
            return False

    async def get_batch(self, batch_size: int) -> List:
        """Get batch of experience."""
        batch = []
        old_queue_size = self.queue.qsize()
        while True:
            async with Timer("##ASYNC-GET-QUEUE##"):
                experience = await self.queue.get()
            batch.append(experience)
            if len(batch) >= batch_size:
                break
        new_queue_size = self.queue.qsize()
        logger.info({
                'INFO': f"##STATUS-FOR-GET-QUEUE##",
                "VALUE": f"Queue size decreased from {new_queue_size} to {old_queue_size}."
        })
        return batch
