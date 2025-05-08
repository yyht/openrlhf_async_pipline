import socket, asyncio
import ray, random
import uvicorn, torch
from vllm import LLM, SamplingParams
from typing import Optional, Any, List, Dict, Tuple
from typing import List, Dict, Union, Any
import os, sys, uuid, json

from openrlhf.async_pipline.process_request import GenerateRequest, process_batch_requests
import sys, os

sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304_vllm083'))

from env.env_config import ENV_GENERATE_CONFIG

os.environ['COMPILE_SERVER'] = 'http://10.39.17.106:10003'
os.environ['REMOTE_RM_URL'] = 'http://10.39.17.106:10007'
os.environ['VLLM_USE_V1'] = '0'
# os.environ['DEBUG_FLAG'] = 'yes'
from env.math.math_tir_process_single_request import math_tir_generate_async
from env.math.math_tir import math_tir_generate as math_tir_generate_offline

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from starlette.requests import Request
from vllm.entrypoints.openai.protocol import (CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse)
from starlette.responses import JSONResponse, StreamingResponse 
from asyncio import Queue                       
from openrlhf.async_pipline.process_request import default_generate                  

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

import ray
# 初始化 Ray（本地运行）
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logger.warning("Request queue is full. Waiting for space...")

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


from openrlhf.async_pipline.rollout_output_base import GenerateOutput, Output
import threading
import queue

@ray.remote(num_gpus=1)
class ServerController:
    def __init__(self, tensor_parallel_size=1):
        self.server = None
        self.server_ready = asyncio.Event()
        self.port = None
        self.address = ray._private.services.get_node_ip_address()

        self.batch_size = 16

        self.max_retries = 5
        self.retry_delay = 0.1

        # Update model weights

        # self.llm = LLM(
        #     # model='/cpfs/user/chenhao/outputs/qwen25_7B_reinforce_baseline_zero_tir_fix_boxed_lr1e-6_warmup0.0_kl0.0_zero_tir_0325_nginx_prefetch_fix_env_mask/_actor/global_step400/ckpt/pytorch_model.bin/',
        #     model='/cpfs/user/maixinji/Open-Reasoner-Zero/orz_ckpt/orz_7b_ppo_agent_5_fliter/iter500/policy',
        #     tensor_parallel_size=1,
        #     gpu_memory_utilization=0.9,
        #     dtype='auto',
        #     # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        #     # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        #     # This is particularly useful here because we generate completions from the same prompts.
        #     # enable_prefix_caching=True,
        #     max_model_len=8192,
        # )

        import vllm
        engine_args = vllm.AsyncEngineArgs(
            # model='/cpfs/user/chenhao/outputs/qwen25_7B_reinforce_baseline_zero_tir_fix_boxed_lr1e-6_warmup0.0_kl0.0_zero_tir_0325_nginx_prefetch_fix_env_mask/_actor/global_step400/ckpt/pytorch_model.bin/',
            model='/cpfs/user/maixinji/Open-Reasoner-Zero/orz_ckpt/orz_7b_ppo_agent_5_fliter/iter500/policy',
            enforce_eager=False,
            tensor_parallel_size=tensor_parallel_size,
            seed=42,
            max_model_len=8192,
            enable_prefix_caching=False,
            dtype='auto',
            trust_remote_code=True,
            task="generate",
            disable_log_requests=True,
            gpu_memory_utilization=0.4,
            enable_chunked_prefill=False,
            # max_num_batched_tokens=256, # you can further set this parameter to reduce the vllm peak memory usage
        )
        self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)

        self.worker_num = 4
        self.max_queue_size = 1024
        self.request_queue: Queue = Queue(maxsize=self.max_queue_size)
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.workers = []

        self.max_batch_size = 16  # 单个批次最大请求数
        self.max_wait_time = 1e-1    # 批次等待时间（秒）
        
        # 优先级队列存储元组 (priority, insertion_order, request_data)
        self.priority_queue = []
        self.queue_index = 0
        
        self.lock = asyncio.Lock()

        asyncio.create_task(self._start_fastapi_server())

        self.request_id = 0

    async def start(self):
        """启动工作线程"""
        self._running = True
        for _ in range(self.worker_num):
            self.workers.append(asyncio.create_task(self._worker_loop()))
        print('==Succeeded in starting==')

    async def stop(self):
        """停止服务并清空队列"""
        self._running = False
        await self.request_queue.join()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)

    async def generate_sync(self, prompts, sampling_params):
        outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params
            )
        return outputs

    async def generate_async_server(self, prompts, sampling_params, request_id):
        # Send the request to the LLM engine.
        async with asyncio.Semaphore(128):
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
            yield output
        # output = [{
        #         'outputs':[
        #             {
        #                 "text": final_output.outputs[0].text,
        #                 "token_ids": final_output.outputs[0].token_ids,
        #                 "stop_reason": final_output.outputs[0].stop_reason,
        #                 "finish_reason": final_output.outputs[0].finish_reason,
        #             }
        #         ],
        #         "prompt_token_ids": final_output.prompt_token_ids
        #     }]
        # return output
            
                # if request_output.finished:
                #     # Bypass the original full prompt.
                #     # request_output.prompt = request.prompt
                #     output = [{
                #         'outputs':[
                #             {
                #                 "text": request_output.outputs[0].text,
                #                 "token_ids": request_output.outputs[0].token_ids,
                #                 "stop_reason": request_output.outputs[0].stop_reason,
                #                 "finish_reason": request_output.outputs[0].finish_reason,
                #             }
                #         ]
                #     }]
                #     return output

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
            stop=request.stop
        )
        response = await self.generate_async_server(request.prompts, sampling_params, id_generator(10))
        return response

    async def _worker_loop(self):
        """工作协程循环"""
        while self._running:
            try:
                request_id, request, future = await self.request_queue.get()
                
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
                    stop=request.stop
                )
                response = await self.generate_async_server(request.prompts, sampling_params, id_generator(10))
                
                # response = await self.generate_async(request.prompts, sampling_params)
                
                future.set_result(response)
                self.pending_requests.pop(request_id, None)
                
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                self.request_queue.task_done()

    # async def _worker_loop(self):
    #     """工作协程循环"""
    #     while self._running:
    #         try:
    #             batch = []
    #             batch_requests = []
    #             batch_futures = []

    #             # 组批逻辑
    #             async with self.lock:
    #                 start_time = asyncio.get_running_loop().time()
    #                 while len(batch) < self.max_batch_size:
    #                     elapsed_time = asyncio.get_running_loop().time() - start_time
    #                     if elapsed_time >= self.max_wait_time and batch:
    #                         break
    #                     try:
    #                         request_id, request, future = self.request_queue.get_nowait()
    #                         batch.append(request.prompts)
    #                         batch_requests.append(request)
    #                         batch_futures.append(future)
    #                     except Exception:
    #                         if elapsed_time >= self.max_wait_time:
    #                             break
    #                         await asyncio.sleep(0.001)

    #             if batch:
    #                 flat_prompts = [sublist[0] for sublist in batch]
    #                 # 假设所有请求的采样参数相同，取第一个请求的参数
    #                 sampling_params = [SamplingParams(
    #                     n=request.n,
    #                     repetition_penalty=request.repetition_penalty,
    #                     temperature=request.temperature,
    #                     top_p=request.top_p,
    #                     top_k=request.top_k,
    #                     min_p=request.min_p,
    #                     max_tokens=request.max_tokens,
    #                     include_stop_str_in_output=request.include_stop_str_in_output,
    #                     stop=request.stop
    #                 ) for request in batch_requests]

    #                 # 实际生成逻辑
    #                 responses = await self.generate_sync(flat_prompts, sampling_params)

    #                 # 拆分结果
    #                 index = 0
    #                 for i, request in enumerate(batch_requests):
    #                     num_prompts = len(request.prompts)
    #                     response = responses[index:index + num_prompts]
    #                     batch_futures[i].set_result(response)
    #                     self.pending_requests.pop(batch_requests[i].uuids, None)
    #                     index += num_prompts

    #         except Exception as e:
    #             for future in batch_futures:
    #                 if not future.done():
    #                     future.set_exception(e)
    #         finally:
    #             for _ in batch_requests:
    #                 self.request_queue.task_done()

    # async def _worker_loop(self):
    #     """工作协程循环"""
    #     while self._running:
    #         try:
    #             batch = []
    #             batch_requests = []
    #             batch_futures = []

    #             start_time = asyncio.get_running_loop().time()
    #             while len(batch) < self.max_batch_size:
    #                 elapsed_time = asyncio.get_running_loop().time() - start_time
    #                 if elapsed_time >= self.max_wait_time and batch:
    #                     break
    #                 try:
    #                     # 非阻塞获取请求
    #                     request_id, request, future = self.request_queue.get_nowait()
    #                     # 展开每个请求的prompts并为每个prompt创建参数
    #                     for prompt in request.prompts:
    #                         batch.append(prompt)
    #                         batch_requests.append(request)
    #                         batch_futures.append(future)
    #                 except asyncio.QueueEmpty:
    #                     if elapsed_time >= self.max_wait_time:
    #                         break
    #                     await asyncio.sleep(0.001)
    #                 except Exception as e:
    #                     break

    #             if batch:
    #                 # 构造每个prompt对应的参数
    #                 sampling_params_list = []
    #                 for req in batch_requests:
    #                     sp = SamplingParams(
    #                         n=req.n,
    #                         repetition_penalty=req.repetition_penalty,
    #                         temperature=req.temperature,
    #                         top_p=req.top_p,
    #                         top_k=req.top_k,
    #                         min_p=req.min_p,
    #                         max_tokens=req.max_tokens,
    #                         include_stop_str_in_output=req.include_stop_str_in_output,
    #                         stop=req.stop
    #                     )
    #                     # 每个prompt使用相同的请求参数
    #                     sampling_params_list.extend([sp] * len(req.prompts))

    #                 # 异步生成
    #                 responses = await self.generate_sync(batch, sampling_params_list)

    #                 # 分配结果到对应的future
    #                 index = 0
    #                 for future, req in zip(batch_futures, batch_requests):
    #                     num_prompts = len(req.prompts)
    #                     result = responses[index:index + num_prompts]
    #                     future.set_result(result)
    #                     index += num_prompts
    #                     self.pending_requests.pop(req.uuids, None)

    #         except Exception as e:
    #             for future in batch_futures:
    #                 if not future.done():
    #                     future.set_exception(e)
    #         finally:
    #             for _ in batch_requests:
    #                 self.request_queue.task_done()

    # async def _worker_loop(self):
    #     """工作协程循环"""
    #     while self._running:
    #         try:
    #             batch = []
    #             batch_requests = []
    #             batch_futures = []

    #             start_time = asyncio.get_running_loop().time()
    #             while len(batch) < self.max_batch_size:
    #                 elapsed_time = asyncio.get_running_loop().time() - start_time
    #                 if elapsed_time >= self.max_wait_time and batch:
    #                     break
    #                 try:
    #                     # 非阻塞获取请求
    #                     request_id, request, future = self.request_queue.get_nowait()
    #                     # 展开每个请求的prompts并为每个prompt创建参数
    #                     for prompt in request.prompts:
    #                         batch.append(prompt)
    #                         batch_requests.append(request)
    #                         batch_futures.append(future)
    #                 except asyncio.QueueEmpty:
    #                     if elapsed_time >= self.max_wait_time:
    #                         break
    #                     await asyncio.sleep(0.001)
    #                 except Exception as e:
    #                     break

    #             if batch:
    #                 # 构造每个prompt对应的参数
    #                 sampling_params_list = []
    #                 for req in batch_requests:
    #                     sp = SamplingParams(
    #                         n=req.n,
    #                         repetition_penalty=req.repetition_penalty,
    #                         temperature=req.temperature,
    #                         top_p=req.top_p,
    #                         top_k=req.top_k,
    #                         min_p=req.min_p,
    #                         max_tokens=req.max_tokens,
    #                         include_stop_str_in_output=req.include_stop_str_in_output,
    #                         stop=req.stop
    #                     )
    #                     # 每个prompt使用相同的请求参数
    #                     sampling_params_list.extend([sp] * len(req.prompts))

    #                 # 增加重试机制
    #                 last_exception = None
    #                 for retry in range(self.max_retries + 1):
    #                     try:
    #                         # 异步生成
    #                         responses = await self.generate_sync(batch, sampling_params_list)
    #                         last_exception = None
    #                         break
    #                     except Exception as e:
    #                         last_exception = e
    #                         if retry < self.max_retries and self._running:
    #                             await asyncio.sleep(self.retry_delay)

    #                 if last_exception is not None:
    #                     # 直接丢掉这个请求，为每个未完成的 future 设置异常
    #                     for future in batch_futures:
    #                         if not future.done():
    #                             future.set_exception(last_exception)
    #                 else:
    #                     # 分配结果到对应的future
    #                     index = 0
    #                     for future, req in zip(batch_futures, batch_requests):
    #                         num_prompts = len(req.prompts)
    #                         result = responses[index:index + num_prompts]
    #                         future.set_result(result)
    #                         index += num_prompts
    #                         self.pending_requests.pop(req.uuids, None)

    #         except Exception as e:
    #             for future in batch_futures:
    #                 if not future.done():
    #                     future.set_exception(e)
    #         finally:
    #             for _ in batch_requests:
    #                 self.request_queue.task_done()

    async def async_generate(self, request: GenerateRequest):
        """异步生成接口"""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = request.uuids

        # 添加回调自动清理pending_requests
        future.add_done_callback(
            lambda _: self.pending_requests.pop(request_id, None)
        )

        self.pending_requests[request_id] = future

        try:
            # 非阻塞等待队列空间（队列满时自动等待）
            await self.request_queue.put((request_id, request, future))
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise

        try:
            return await future
        except asyncio.CancelledError:
            # 处理任务取消
            if not future.done():
                future.cancel()
            raise

    # async def async_generate(self, request: GenerateRequest):
    #     """异步生成接口"""
    #     # if self.request_queue.qsize() >= self.max_queue_size:
    #     #     raise RuntimeError("Request queue is full")

    #     # while self.request_queue.full():
    #     #     logging.warning("Request queue is full. Waiting for space...")
    #     #     await asyncio.sleep(0.1)
        
    #     # 创建异步Future
    #     loop = asyncio.get_running_loop()
    #     future = loop.create_future()
    #     request_id = request.uuids
        
    #     # 将请求存入等待队列
    #     self.pending_requests[request_id] = future
    #     await self.request_queue.put((request_id, request, future))
        
    #     try:
    #         return await future
    #     except Exception as e:
    #         print(f"Error in async_generate: {e}")
    #         if not future.done():
    #             future.set_exception(e)
    #         self.pending_requests.pop(request_id, None)
    #         raise
    #     # finally:
    #     #     # 确保异常时清理资源
    #     #     self.pending_requests.pop(request_id, None)
    
    async def _start_fastapi_server(self):
        import fastapi
        app = fastapi.FastAPI()
        app.router.add_api_route("/health", self.health, methods=["GET"])
        app.router.add_api_route("/async_generate", self.async_generate, methods=["POST"])

        await asyncio.sleep(random.uniform(0, 3))
        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port)
        self.server = uvicorn.Server(config)  # 保存实例
        self.server_ready.set()
        await self.start()
        await self.server.serve()

    async def health(self):
        return 1

    # async def restart_server(self):
    #     if self.server:
    #         await self.server.shutdown()
    #         await asyncio.sleep(0.5)  # 确保关闭完成
    #         self.server = None
    #     self.server_ready.clear()
    #     asyncio.create_task(self._start_fastapi_server())

    async def get_server_address(self):
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    async def get_server_port(self):
        await self.server_ready.wait()
        return self.port

    def build_requests(self, prompts, prompt_ids, sampling_params, labels=None):
        request_list = []
        for idx, (prompt, prompt_id) in enumerate(zip(prompts, prompt_ids)):
            if labels is not None:
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
            request = GenerateRequest(
                prompts=[prompt],
                prompt_token_ids=prompt_id,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                stop=sampling_params.stop,
                uuids=uuid_str+f'####idx:{idx}',
                env_func=env_func,
                label=json.dumps(label_dict, ensure_ascii=False)
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

    async def add_requests(self, url, prompts, prompt_ids, sampling_params, labels=None, **kwargs):
        requests = self.build_requests(prompts, prompt_ids, sampling_params, labels=labels)
        if labels is not None:
            requests = self.group_requests(requests)
        batches = self._create_batches(requests)

        ip_port = await self.get_server_address()
        url = f'http://{ip_port}/async_generate'

        responses_ray = []
        for start_idx, batch in batches:
            env_func = batch[0].env_func
            responses_ray.append(process_batch_requests(self.async_llm_generate, start_idx, batch, env_func=env_func, tokenizer=kwargs.get('tokenizer', None)))
                        
        results_raw = await asyncio.gather(*responses_ray)
        flat_results = []
        for result_raw in results_raw:
            successful_results, failed_results = result_raw
            for item in successful_results:
                flat_results.append(item)
        # flat_results = [item for batch in results_raw for item in batch]
        falt_responses = [result[1][1] for result in flat_results]

        # 按 idx 排序
        falt_responses.sort(key=lambda x: int(x.request_id.split('####idx:')[-1]))
        # return [result[1] for result in flat_results]
        return falt_responses

        # flat_results = [item for batch in results_raw for item in batch]

        # # 按 idx 排序
        # flat_results.sort(key=lambda x: x[0])
        # return [result[1] for result in flat_results]

    def add_offline_requests(self, prompts, prompt_ids, sampling_params, **kwargs):
        tokenizer = kwargs.get('tokenizer', None)
        responses = math_tir_generate_offline(self.llm, sampling_params, prompt_ids, tokenizer, prompts=prompts)
        return responses

def distributed_rlhf(tensor_parallel_size):
    bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
    pg = placement_group(bundles)
    ray.get(pg.ready())

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
    )
    return scheduling_strategy

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

class RayActor():
    def __init__(self, tensor_parallel_size=1):
        num_gpus = int(tensor_parallel_size==1)
        bundle_indices = None
        scheduling_strategy = None
        if tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )
        self.async_llm_workers = [ServerController.options(
                num_cpus=0,
                num_gpus=tensor_parallel_size,
                # scheduling_strategy=scheduling_strategy,
            ).remote()]
        self.server_addresses = ray.get([worker.get_server_address.remote() for worker in self.async_llm_workers])

ray_actor = RayActor(tensor_parallel_size=1)
addresses = [ray_actor.server_addresses]


print(ray.get([worker.get_server_port.remote() for worker in ray_actor.async_llm_workers]))


import requests, time
from timeout_decorator import timeout

@timeout(1, use_signals=False)
def health_check(url):
    for _ in range(2):
        status = requests.get(f'http://{url}/health')
        print(status, '==statue==')

def async_post(url, prompts):
    request = GenerateRequest(
                prompts=[prompts],
                prompt_token_ids=[0],
                max_tokens=1024,
                temperature=0.1,
                stop=[],
                model='default',
                uuids=str(uuid.uuid4())
            )
    # d_dict = {
    #     'model': 'default',
    #     'prompt': prompts,
    #     'frequency_penalty': 1.0,
    #     'skip_special_tokens': False,
    #     'max_tokens': 1024,
    #     'temperature': 0.7
    # }
    response = requests.post(f'http://{url}/async_generate', json=request.to_json())
    print(response.json(), '===response===')
    return response.json()

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/", use_fast=True)

template = 'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. In your reasoning process, you can use python code to solve your problem. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag. \nThis is the problem: {query}.\nAssistant: <think>'
prompt = 'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. In your reasoning process, you can use python code to solve your problem. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag. \nThis is the problem: 1. The monotonic decreasing interval of the function $y=\\log _{\\frac{1}{5}}|x-2|$ is ( ).\nA. $(-\\infty, 2)$\nB. $(-\\infty,-2) \\cup(2,+\\infty)$\nC. $(2,+\\infty)$\nD. $(0,2) \\cup(2,+\\infty)$ .\nAssistant: <think>'
prompt = 'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:Given \\( x_{1}, x_{2}, x_{3} \\in [0, 12] \\),\n\\[ x_{1} x_{2} x_{3} = \\left(\\left(12 - x_{1}\\right)\\left(12 - x_{2}\\right)\\left(12 - x_{3}\\right)\\right)^{2}. \\]\n\nFind the maximum value of \\( f = x_{1} x_{2} x_{3} \\).\nAssistant: <think>'
prompt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. In your reasoning process, you can use python code to solve your problem. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag. \nThis is the problem: Find the number of permutations  $x_1, x_2, x_3, x_4, x_5$  of numbers  $1, 2, 3, 4, 5$  such that the sum of five products  $$ x_1x_2x_3 + x_2x_3x_4 + x_3x_4x_5 + x_4x_5x_1 + x_5x_1x_2 $$  is divisible by  $3$ .\nAssistant: <think>"
prompt_token_ids = tokenizer(prompt)['input_ids']

total_time = 0

for _ in range(1):
    print('123')
    time.sleep(1.0)
    health_check(addresses[0][0])
    time.sleep(10.0)
    resp = async_post(addresses[0][0], '你是谁')
    resp1 = async_post(addresses[0][0], '你是谁')
    print(resp, resp1)
    # time.sleep(1.0)
    # health_check(addresses[0][0])
    # print(async_post(addresses[0][0], '你是谁'), '===after-restart-server===')

    ip_port = addresses[0][0]

    url = f'http://{ip_port}/async_generate'

    import time
    start = time.time()

    sampling_params = SamplingParams(temperature=0.0, 
                             top_p=1.0,
                             max_tokens=4096,
                             stop=["User:", " User:", "Human:", "Assistant:", " Assistant:", "</answer>"],
                             include_stop_str_in_output=True,
                             min_tokens=32,
                             n=1)
    
    refs = ray_actor.async_llm_workers[0].add_requests.remote(url, [prompt]*128, 
    [prompt_token_ids]*128, sampling_params, 
    # process_fn=default_generate,
    env_func='math_tir_async',
    tokenizer=tokenizer)

    outputs = ray.get(refs)

    # import threading
    # import queue

    # replay_queue = queue.Queue()

    # def make_exp(dup=2):
    
    #     refs = ray_actor.async_llm_workers[0].add_requests.remote(url, [prompt]*dup, 
    #     [prompt_token_ids]*dup, sampling_params, process_fn=math_tir_generate_async,
    #     tokenizer=tokenizer)

    #     outputs = ray.get(refs)
    #     return outputs

    # replay_queue.put(make_exp(2))

    # def async_sampler(generator):
    #     outputs = []
    #     for item in generator:
    #         if item is None:
    #             break
    #         outputs.append(item)
    #     replay_queue.put(outputs)
    # # Get the generator function which will yield results as they complete
    # gen_seq_generator = self.rollout_wg.generate_sequences_async(prompts=sample_batch)
    # thread = threading.Thread(target=async_sampler, args=(make_exp, replay_queue))
    # thread.start()


#     for _ in range(64):


    outputs_dict = []
    for output in outputs:
        outputs_dict.append(output._asdict())

    total_time += time.time()-start

    print(len(outputs), '===http===', time.time()-start)
    json.dump(outputs_dict, open('/cpfs/debug/outputs_latest.json', 'w'))

    refs = []
    ttt = ray.get(refs)
    print(ttt)

# print(total_time/5)