

# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys, uuid
import atexit
import logging
import time
from typing import Optional
import asyncio
import os
import httpx
import ray
from typing import Dict, List, Optional, Tuple
import logging

import os
import queue
from collections import defaultdict
import numpy as np
from typing import Any, List

import torch
from torch import nn

import requests
from vllm import LLM, SamplingParams

import ray
# 初始化 Ray（本地运行）
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304_vllm083'))

from openrlhf.async_pipline.vllm_serve import GenerateRequest
from openrlhf.async_pipline.process_single_request import process_single_request
# from openrlhf.async_pipline.vllm_client import VLLMClient
from env.vllm_client_debug import VLLMClient
import ray, os

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 10))
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

logger = logging.getLogger(__name__)

async def _async_process(url, start_idx, batch, **kwargs):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    process_fn = kwargs.get('process_fn', process_single_request)

    async with httpx.AsyncClient() as client:
        tasks = []
        for offset, request in enumerate(batch):
            idx = start_idx + offset
            # 为每个请求创建异步任务
            task = process_fn(
                client=client,
                semaphore=semaphore,
                url=url,
                headers=headers,
                idx=idx,
                request=request,
                **kwargs
            )
            tasks.append(task)
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
    return results

@ray.remote
def process_batch_requests(url: str, start_idx: int, batch: List[Dict], **kwargs) -> List[Tuple[int, Optional[str]]]:
    """Ray 远程任务：处理一个 batch 的请求（同步包装异步逻辑），使用信号量控制并发"""
    return asyncio.run(_async_process(url, start_idx, batch, **kwargs))


def flatten_responses(responses):
    flat_results = [item for batch in responses for item in batch]

    # 按 idx 排序
    flat_results.sort(key=lambda x: x[0])
    results = [r[1] for r in flat_results]
    return results


# @ray.remote
class LLMClientRayActor(VLLMClient):
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self, host: str = "0.0.0.0", server_port: int = 8000, group_port: int = 51216, connection_timeout: float = 0.0,
        batch_size: int = 2,
        **kwargs
    ):

        self.is_master_actor = kwargs.pop("master_actor")
        # init-vllm-client
        super().__init__(host, server_port, group_port, connection_timeout, self.is_master_actor)

        self.execute_url = f"http://{self.host}:{self.server_port}/generate/"
        
        # copy from vllm-engine
        self.batch_size = batch_size
        self.results = []

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        # self.responses = {}
        self.response_queues = defaultdict(queue.Queue)
        self.requests_of_ids = {}

        # Update model weights
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained("/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/", 
                                torch_dtype=torch.bfloat16).to("cuda")

    def build_requests(self, prompts, prompt_ids, sampling_params):
        request_list = []
        for prompt, prompt_id in zip(prompts, prompt_ids):
            request = GenerateRequest(
                prompts=[prompt],
                prompt_token_ids=prompt_id,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                stop=sampling_params.stop,
                uuids=str(uuid.uuid4())
            )
            request_list.append(request)
        return request_list

    def _create_batches(self, data_list: List[Dict]) -> List[Tuple[int, List[Dict]]]:
        """将数据分成 batch，返回 [(start_idx, batch), ...]"""
        batches = []
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]
            batches.append((i, batch))
        return batches

    def add_requests(self, actor_rank, *, sampling_params, prompt_token_ids, prompts=None, **kwargs):
        self.requests[actor_rank] = prompts
        self.requests_of_ids[actor_rank] = prompt_token_ids
        self.actor_counter += 1
        
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors
            assert len(self.requests_of_ids) == self.num_actors
            num_requests = []
            requests = []
            requests_of_ids = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)
            for request_rank, request_ids in self.requests_of_ids.items():
                requests_of_ids.extend(request_ids)
        
            assert len(requests_of_ids) == len(requests)

            if len(requests) > 0:
                request_list = self.build_requests(requests, requests_of_ids, sampling_params)
                # _create_batches
                batch_inputs = self._create_batches(request_list)

                responses_ray = []
                for start_idx, batch in batch_inputs:
                    responses_ray.append(process_batch_requests.remote(self.execute_url, start_idx, batch, **kwargs))
                
                responses = ray.get(responses_ray)
                
                offset = 0
                self.responses = {}
                for actor_rank, num in num_requests:
                    self.response_queues[actor_rank].put(responses[offset : offset + num])
                    offset += num

                self.actor_counter = 0
                self.requests = {}
                self.requests_of_ids = {}
                
    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        # return self.responses.pop(actor_rank)
        responses = self.response_queues[actor_rank].get()
        return responses

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        pass

    def update_weight(self, name, dtype, shape, weights, empty_cache=False):
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def update_weight_cuda_ipc(self, name, dtype, shape, weights, ipc_handles, empty_cache=False):
        self.update_weight(name, dtype, shape, weights, empty_cache)

    def sleep(self, level=1):
        pass

    def wake_up(self):
        pass

if __name__ == "__main__":
    client = LLMClientRayActor(host='10.39.14.25',
                            port=8000,
                            group_port=51216,
                            num_actors=1,
                            master_actor=True)  # 每个 worker 处理 3 条

    client1 = LLMClientRayActor(host='10.39.14.25',
                            port=8000,
                            group_port=51216,
                            num_actors=1,
                            master_actor=False)  # 每个 worker 处理 3 条

    refs = []

    client.add_requests(
        actor_rank=0,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["\n\n"],
        ),
        prompt_token_ids=[[14990, 11, 847, 829, 374, 28047, 1466, 13],
                        [14990, 11, 847, 829, 374, 28047, 1466, 13]],
        prompts=["hello, my name is sarah.",
                "hello, my name is sarah."])

    all_outputs = client.get_responses(0)

    print(all_outputs, '==client==')

    for (name, param) in client.model.named_parameters():
        dtype, shape = str(param.data.dtype), tuple(param.data.shape)
        client.update_weight(name, dtype, shape, param.data)

    client1.add_requests(
        actor_rank=0,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,
            stop=["\n\n"],
        ),
        prompt_token_ids=[[14990, 11, 847, 829, 374, 28047, 1466, 13],
                        [14990, 11, 847, 829, 374, 28047, 1466, 13]],
        prompts=["hello, my name is sarah.",
                "hello, my name is sarah."])

    all_outputs = client1.get_responses(0)

    print(all_outputs, '==client1==')

    


    

    
    


    

    

        
                



    