

"""A queue implemented by Ray Actor."""
import asyncio
from copy import deepcopy
from typing import List

import ray
from openrlhf.utils.logging_utils import init_logger
from openrlhf.async_pipline.show_timer import Timer
import asyncio
from collections import defaultdict

import socket, asyncio
import ray, random
import uvicorn, torch
from vllm import LLM, SamplingParams
from typing import Optional, Any, List, Dict, Tuple
from typing import List, Dict, Union, Any
import os, sys, uuid
from fastapi.responses import JSONResponse, Response, StreamingResponse
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304_vllm083'))

from openrlhf.async_pipline.rollout_output_base import GenerateOutput, Output, generate_output_to_dict

import ray
# 初始化 Ray（本地运行）
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

from typing import Optional, Any, List, Dict, Tuple
from pydantic import BaseModel
class GenerateRequest(BaseModel):
    category: int = 0
    value: dict = {}
    batch_size: int = 1
    def to_json(self):
        return {
            "category": self.category,
            "value": self.value,
            "batch_size": self.batch_size
        }

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

logger = init_logger(__name__)

@ray.remote
class DictQueueActor:
    """An asyncio.Queue based queue actor."""
    def __init__(self, capacity=1024) -> None:
        self.capacity = capacity
        self.queue_dict = defaultdict(lambda: asyncio.Queue(self.capacity))

        # async-server
        self.server = None
        self.server_ready = asyncio.Event()
        self.port = None
        self.address = ray._private.services.get_node_ip_address()

        self.lock = asyncio.Lock()
        asyncio.create_task(self._start_fastapi_server())

    def category_length(self, category) -> int:
        """The length of the queue."""
        if category in self.queue_dict:
            return self.queue_dict[category].qsize()
        else:
            return 0

    def length(self) -> int:
        """The length of the queue."""
        return sum(self.queue_dict.values())

    async def get_valid_category(self, request: GenerateRequest):
        batch_size = request.batch_size
        valid_category = []
        for category in self.queue_dict:
            if self.queue_dict[key].qsize() >= batch_size:
                valid_category.append(key)
        return valid_category

    async def enqueue(self, request: GenerateRequest):
        """Put batch of experience."""
        category = request.category
        item = request.value
        old_queue_size = self.queue_dict[category].qsize()
        async with Timer("##ASYNC-PUT-QUEUE##"):
            await self.queue_dict[category].put(item)
        new_queue_size = self.queue_dict[category].qsize()
        if new_queue_size > old_queue_size:
            logger.info({
                'INFO': f"##STATUS-FOR-PUT-QUEUE##",
                "VALUE": f"Queue size increased from {old_queue_size} to {new_queue_size}."
            })
            return True
        else:
            return False

    async def dequeue(self, request: GenerateRequest):
        """Get batch of experience."""
        category = request.category
        batch_size = request.batch_size
        batch = []
        old_queue_size = self.queue_dict[category].qsize()
        while True:
            async with Timer("##ASYNC-GET-QUEUE##"):
                experience = await self.queue_dict[category].get()
            batch.append(experience)
            if len(batch) >= batch_size:
                break
        return batch

    async def health(self):
        return {"status": "ok"}

    async def get_server_address(self):
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    async def _start_fastapi_server(self):
        import fastapi, uvicorn, asyncio
        app = fastapi.FastAPI()
        app.router.add_api_route("/health", self.health, methods=["GET"])
        app.router.add_api_route("/enqueue", self.enqueue, methods=["POST"])
        app.router.add_api_route("/dequeue", self.dequeue, methods=["POST"])
        app.router.add_api_route("/get_server_address", self.get_server_address, methods=["POST"])

        await asyncio.sleep(random.uniform(0, 3))
        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port)
        self.server = uvicorn.Server(config)  # 保存实例
        self.server_ready.set()
        await self.server.serve()



queue_actor = DictQueueActor.remote()
ref = queue_actor.get_server_address.remote()
ip_addresses = ray.get(ref)

print(ip_addresses, '===ip_addresses===')


import requests, time
from timeout_decorator import timeout

@timeout(1, use_signals=False)
def health_check(url):
    for _ in range(2):
        status = requests.get(f'http://{url}/health')
        print(status.json(), '==statue==')

health_check(ip_addresses)

# d_dict = GenerateRequest(
#     category=0,
#     value=[1,2,3],
#     batch_size=1
# )

# ray.get(queue_actor.enqueue.remote(d_dict))
# ray.get(queue_actor.enqueue.remote(d_dict))
# print(ray.get(queue_actor.dequeue.remote(d_dict)))

headers = {
        "Content-Type": "application/json",
    }

output = GenerateOutput(
            outputs=[Output(
                token_ids=[1,2,3],
                action_mask=[1,1,1],
                text='你好',
                stop_reason='stop',
                finish_reason='stop',
                env_exec_times=1,
                reward_info={}
            )],
            prompt_token_ids=[1,2,3],
            request_id='0',
            label={},
            prompt='0',
            request_rank=1
    )

d_dict = GenerateRequest(
    category=0,
    value=generate_output_to_dict(output),
    batch_size=1
)
def async_post(url, d_dict):
    import json
    response = requests.post(f'http://{url}/enqueue', json=d_dict.to_json(), headers=headers)
    print(response.json(), '===response===')
    return response.json()



ref = queue_actor.get_server_address.remote()
ip_addresses = ray.get(ref)
print(async_post(ip_addresses, d_dict))

print(ray.get(queue_actor.dequeue.remote(d_dict)))