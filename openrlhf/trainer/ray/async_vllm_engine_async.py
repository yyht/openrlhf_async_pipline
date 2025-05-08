
import socket, asyncio
import ray, random
import uvicorn, torch
from vllm import LLM, SamplingParams
from typing import Optional, Any, List, Dict, Tuple
from typing import List, Dict, Union, Any
import os, sys, uuid
from fastapi.responses import JSONResponse, Response, StreamingResponse

from openrlhf.async_pipline.process_request import GenerateRequest, process_batch_requests
import sys, os, json

from openrlhf.trainer.ray.vllm_engine import (
    LLMRayActor,
    get_all_env_variables,
    batch_vllm_engine_call,
)

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os
import queue
from collections import defaultdict
import numpy as np
from typing import Any, List

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM
from asyncio import Queue

from .utils import get_bundle_indices, ray_noset_visible_devices
from openrlhf.utils.logging_utils import init_logger
logger = init_logger(__name__)

from openrlhf.env.env_config import ENV_GENERATE_CONFIG
from openrlhf.async_pipline.show_timer import Timer

import os
env_method = os.getenv('GENERATE_METHOD', '')
GENERATE_FUNC = ENV_GENERATE_CONFIG.get(env_method, None)

logger.info({
    'ENV_METHOD': env_method,
    'GENERATE_FUNC': GENERATE_FUNC
})

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 32))  # 最大并发数，可根据需要调整
logger.info({
    'INFO': "##MAX_CONCURRENT##",
    'VALUE': MAX_CONCURRENT
})


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

@ray.remote
class AsyncLLMRayAsyncActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        noset_visible_devices = kwargs.pop("noset_visible_devices")
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating CUDA_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.2"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        # self.responses = {}
        self.response_queues = defaultdict(queue.Queue)
        self.requests_of_ids = {}
        self.requests_labels = {}

        self.actor_counter_dict = {}
        self.response_queues_dict = {}

        import vllm

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism or vllm.__version__ == "0.8.3":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)

        # async-server
        self.port = None
        self.address = ray._private.services.get_node_ip_address()

        self.batch_size = int(kwargs.get('batch_size', 32))

        self.worker_num = int(kwargs.get('worker_num', 4))
        self.max_queue_size = int(kwargs.get('max_queue_size', 1024))
        self.request_queue: Queue = Queue(maxsize=self.max_queue_size)
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.workers = []

        self.max_batch_size = int(kwargs.get('max_batch_size', 32))  # 单个批次最大请求数
        self.max_wait_time = float(kwargs.get('max_wait_time', 1e-1))    # 批次等待时间（秒）
        
        # 优先级队列存储元组 (priority, insertion_order, request_data)
        self.priority_queue = []
        self.queue_index = 0
        self.max_retries = 5
        self.retry_delay = 0.1
    
    async def generate_async_server(self, prompts, sampling_params, request_id):
        # Send the request to the LLM engine.
        import asyncio
        async with asyncio.Semaphore(MAX_CONCURRENT):
            stream = self.async_llm.generate(
                request_id=str(request_id),
                prompt=prompts[0],
                sampling_params=sampling_params,
            )

        # Consume the stream until the request is finished.
        async for request_output in stream:
            final_output = request_output
        output = [{
                'outputs':[
                    {
                        "text": final_output.outputs[0].text,
                        "token_ids": final_output.outputs[0].token_ids,
                        "stop_reason": final_output.outputs[0].stop_reason,
                        "finish_reason": final_output.outputs[0].finish_reason,
                    }
                ],
                "prompt_token_ids": final_output.prompt_token_ids
            }]
        return output

    async def async_llm_generate(self, request: GenerateRequest):
        # 实际生成逻辑
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            include_stop_str_in_output=request.include_stop_str_in_output,
            stop=request.stop,
            skip_special_tokens=False
        )

        request_id = str(uuid.uuid4())
        response = await self.generate_async_server(request.prompts, sampling_params, request_id)
        return response

    def build_requests(self, prompts, prompt_ids, sampling_params, labels=None, requests_ranks=None):
        request_list = []
        for idx, (prompt, prompt_id) in enumerate(zip(prompts, prompt_ids)):
            if labels is not None:
                if labels[idx] is not None:
                    label_dict = json.loads(labels[idx])
                    uuid_str = label_dict.get('uuid', str(uuid.uuid4()))
                    env_func = label_dict.get('env_func', 'math_tir_async')
                else:
                    env_func = 'math_tir_async'
                    uuid_str = str(uuid.uuid4())
                    label_dict = {
                        'uuid': uuid_str,
                        'env_func': env_func
                    }
            else:
                env_func = 'math_tir_async'
                uuid_str = str(uuid.uuid4())
                label_dict = {
                    'uuid': uuid_str,
                    'env_func': env_func
                }
            
            if requests_ranks is not None:
                request_rank = requests_ranks[idx]
            else:
                request_rank = 0
            request = GenerateRequest(
                prompts=[prompt],
                prompt_token_ids=prompt_id,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                stop=sampling_params.stop,
                uuids=uuid_str+f'####idx:{idx}',
                env_func=env_func,
                label=json.dumps(label_dict, ensure_ascii=False),
                request_rank=request_rank
            )
            request_list.append(request)
        return request_list

    def group_requests(self, data_list: List[Dict]):
        requests_dict = {}
        for data in data_list:
            env_func = data.env_func
            if env_func not in requests_dict:
                requests_dict[env_func] = []
            requests_dict[env_func].append(data)
        return requests_dict

    def _create_batches(self, data_list: Union[List[Dict[Any, Any]], Dict[Any, List[Any]]]) -> List[Tuple[int, List[Dict]]]:
        """将数据分成 batch，返回 [(start_idx, batch), ...]"""
        batches = []
        if isinstance(data_list, list):
            for i in range(0, len(data_list), self.batch_size):
                batch = data_list[i:i + self.batch_size]
                batches.append((i, batch))
            if i + self.batch_size < len(data_list) - 1:
                batches.append((i+1, data_list[i + self.batch_size:]))
        elif isinstance(data_list, dict):
            for env_func in data_list:
                for i in range(0, len(data_list[env_func]), self.batch_size):
                    batch = data_list[env_func][i:i + self.batch_size]
                    batches.append((i, batch))
                if i + self.batch_size < len(data_list[env_func]) - 1:
                    batches.append((i+1, data_list[env_func][i + self.batch_size:]))
        else:
            raise ValueError("data_list must be a list or dict")
        return batches

    async def add_env_pipline_requests(self, prompt_idx, actor_rank, *, sampling_params, prompt_token_ids, 
                prompts=None, tokenizer=None, labels=None):
        """
        Save the requests from actors and generate responses when all actors have sent their requests
        """
        if prompt_idx not in self.requests:
            self.requests[prompt_idx] = {}
        if prompt_idx not in self.requests_of_ids:
            self.requests_of_ids[prompt_idx] = {}
        if prompt_idx not in self.requests_labels:
            self.requests_labels[prompt_idx] = {}
        if prompt_idx not in self.actor_counter_dict:
            self.actor_counter_dict[prompt_idx] = 0
        self.requests[prompt_idx][actor_rank] = prompts
        self.requests_of_ids[prompt_idx][actor_rank] = prompt_token_ids
        self.requests_labels[prompt_idx][actor_rank] = labels
        self.actor_counter_dict[prompt_idx] += 1
        if self.actor_counter_dict[prompt_idx] == self.num_actors:
            assert len(self.requests[prompt_idx]) == self.num_actors
            assert len(self.requests_of_ids[prompt_idx]) == self.num_actors
            assert len(self.requests_labels[prompt_idx]) == self.num_actors
            num_requests = []
            requests = []
            requests_of_ids = []
            requests_labels = []
            requests_ranks = []
            for actor_rank, request in self.requests[prompt_idx].items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)
                requests_ranks.extend([actor_rank]*len(request))
            for request_rank, request_ids in self.requests_of_ids[prompt_idx].items():
                requests_of_ids.extend(request_ids)
            for request_rank, request_label in self.requests_labels[prompt_idx].items():
                requests_labels.extend(request_label)
            
            assert len(requests_of_ids) == len(requests)
            assert len(requests_labels) == len(requests)
            assert len(requests_ranks) == len(requests)

            logger.info({
                'INFO': f'##BEGIN-TO-ROLLOUT FOR PROMPT_IDX {prompt_idx} and RANK {actor_rank}##',
            })

            if len(requests_of_ids) > 0:
                all_requests = self.build_requests(prompts=requests, prompt_ids=requests_of_ids, 
                                                    sampling_params=sampling_params, labels=requests_labels,
                                                    requests_ranks=requests_ranks)
                if labels is not None:
                    all_requests = self.group_requests(all_requests)
                batches = self._create_batches(all_requests)
                responses_ray = []
                for start_idx, batch in batches:
                    env_func = batch[0].env_func
                    responses_ray.append(process_batch_requests(self.async_llm_generate, start_idx, batch, env_func=env_func, tokenizer=tokenizer))

                async with Timer("##ASYNC-ROLLOUT-WHOLE-ROLLOUT##"):
                    results_raw = await asyncio.gather(*responses_ray)
                flat_results = []
                for result_raw in results_raw:
                    successful_results, failed_results = result_raw
                    for item in successful_results:
                        flat_results.append(item)
                responses = [result[1][1] for result in flat_results]
                responses.sort(key=lambda x: int(x.request_id.split('####idx:')[-1]))
                logger.info({
                    'INFO': f'##END-TO-ROLLOUT FOR PROMPT_IDX {prompt_idx} and RANK {actor_rank}##',
                })
            else:
                responses = []

            offset = 0
            
            if prompt_idx not in self.response_queues_dict:
                self.response_queues_dict[prompt_idx] = defaultdict(queue.Queue)
            for actor_rank, num in num_requests:
                if self.response_queues_dict[prompt_idx][actor_rank].qsize() > 0:
                    _ = self.response_queues_dict[prompt_idx][actor_rank].get()
                self.response_queues_dict[prompt_idx][actor_rank].put(responses[offset : offset + num])
                offset += num

            self.actor_counter_dict[prompt_idx] = 0
            self.requests[prompt_idx] = {}
            self.requests_of_ids[prompt_idx] = {}
            self.requests_labels[prompt_idx] = {}

    def get_pipline_responses(self, prompt_idx, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        # return self.responses.pop(actor_rank)
        if prompt_idx not in self.response_queues_dict:
            logger.info({
                'INFO': '##PROMPT_IDX IS NOT FOUND##',
                'VALUE': f'PROMPT_IDX {prompt_idx} NOT FOUND FOR RANK {actor_rank}',
            })
            return []
        return self.response_queues_dict[prompt_idx][actor_rank].get()
        

    async def add_env_requests(self, actor_rank, *, sampling_params, prompt_token_ids, 
                prompts=None, tokenizer=None, labels=None):
        """
        Save the requests from actors and generate responses when all actors have sent their requests
        """
        self.requests[actor_rank] = prompts
        self.requests_of_ids[actor_rank] = prompt_token_ids
        self.requests_labels[actor_rank] = labels
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors
            assert len(self.requests_of_ids) == self.num_actors
            assert len(self.requests_labels) == self.num_actors
            num_requests = []
            requests = []
            requests_of_ids = []
            requests_labels = []
            requests_ranks = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)
                requests_ranks.append(actor_rank)
            for request_rank, request_ids in self.requests_of_ids.items():
                requests_of_ids.extend(request_ids)
            for request_rank, request_label in self.requests_labels.items():
                requests_labels.extend(request_label)
            
            assert len(requests_of_ids) == len(requests)
            assert len(requests_labels) == len(requests)

            logger.info({
                'INFO': '##BEGIN-TO-ROLLOUT##'
            })

            if len(requests_of_ids) > 0:
                all_requests = self.build_requests(prompts=requests, prompt_ids=requests_of_ids, 
                                                    sampling_params=sampling_params, labels=requests_labels,
                                                    requests_ranks=requests_ranks)
                if labels is not None:
                    all_requests = self.group_requests(all_requests)
                batches = self._create_batches(all_requests)
                responses_ray = []
                for start_idx, batch in batches:
                    env_func = batch[0].env_func
                    responses_ray.append(process_batch_requests(self.async_llm_generate, start_idx, batch, env_func=env_func, tokenizer=tokenizer))

                async with Timer("##ASYNC-ROLLOUT-WHOLE-ROLLOUT##"):
                    results_raw = await asyncio.gather(*responses_ray)
                flat_results = []
                for result_raw in results_raw:
                    successful_results, failed_results = result_raw
                    for item in successful_results:
                        flat_results.append(item)
                responses = [result[1][1] for result in flat_results]
                responses.sort(key=lambda x: int(x.request_id.split('####idx:')[-1]))
            else:
                responses = []

            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                # self.responses[actor_rank] = responses[offset : offset + num]
                self.response_queues[actor_rank].put(responses[offset : offset + num])
                offset += num

            self.actor_counter = 0
            self.requests = {}
            self.requests_of_ids = {}
            self.requests_labels = {}

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.async_llm.engine.model_executor.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.async_llm.engine.model_executor.collective_rpc(
            "update_weight", args=(name, dtype, shape, empty_cache)
        )

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.async_llm.engine.model_executor.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    def reset_prefix_cache(self):
        self.async_llm.engine.reset_prefix_cache()

    async def sleep(self, level: int = 1):
        await self.async_llm.sleep(level=level)

    async def wake_up(self):
        await self.async_llm.wake_up()

    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        # return self.responses.pop(actor_rank)
        return self.response_queues[actor_rank].get()


def create_async_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
):
    import vllm

    assert vllm.__version__ >= "0.7.0", "OpenRLHF only supports vllm >= 0.7.0"

    vllm_engines = []
    num_gpus = int(tensor_parallel_size == 1)
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    for i in range(num_engines):
        bundle_indices = None
        scheduling_strategy = None

        # Hybrid engine
        if shared_pg is not None:
            assert vllm.__version__ >= "0.7.2", "Only vllm >= 0.7.2 supports hybrid engine"

            if tensor_parallel_size > 1:
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=i * tensor_parallel_size
                )
                bundle_indices = np.arange(i * tensor_parallel_size, (i + 1) * tensor_parallel_size).tolist()
            else:
                num_gpus = 0.2
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=i
                )
        # Distributed RLHF
        elif tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        vllm_engines.append(
            AsyncLLMRayAsyncActor.options(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                # seed=seed + i,
                seed=seed,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                num_actors=num_actors,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices if shared_pg else None,
                enable_sleep_mode=vllm_enable_sleep,
                noset_visible_devices=ray_noset_visible_devices(),
            )
        )
    
    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)