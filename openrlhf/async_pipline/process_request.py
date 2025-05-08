import logging
import asyncio
import ray
import logging
import asyncio
import os, json
import httpx
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
import aiohttp
import os, sys
from openrlhf.async_pipline.show_timer import Timer

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 10))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 1000000000))
TASK_MAX_CONCURRENT = int(os.getenv("TASK_MAX_CONCURRENT", 32))  # 最大并发数，可根据需要调整
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

from openrlhf.env.env_config import ENV_GENERATE_CONFIG
from openrlhf.async_pipline.show_timer import Timer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info({
    'INFO': "##TASK_MAX_CONCURRENT##",
    'VALUE': TASK_MAX_CONCURRENT
})


from pydantic import BaseModel
class GenerateRequest(BaseModel):
    prompts: list[str]
    prompt_token_ids: Optional[list[int]] = None
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    min_tokens: int = 1
    guided_decoding_regex: Optional[str] = None
    stop: Optional[list[str]] = None
    include_stop_str_in_output: bool = True
    uuids: Optional[str] = None
    model: str = 'default'
    env_func: Optional[str] = 'default'
    output_text: Optional[str] = ''
    iterative_num: Optional[int] = 0
    env_exec_times: Optional[int] = 0
    label: Optional[str] = json.dumps({})
    request_rank: Optional[int] = 0
    

    def to_json(self):
        return {
            "prompts": self.prompts,
            "prompt_token_ids": self.prompt_token_ids,
            "n": self.n,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p":self.min_p,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "guided_decoding_regex": self.guided_decoding_regex,
            "stop": self.stop,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "uuids": self.uuids,
            "model": self.model,
            "output_text": self.output_text,
            "iterative_num": self.iterative_num,
            "env_exec_times": self.env_exec_times,
            "label": self.label,
            "request_rank": self.request_rank
        }

def flatten_responses(responses):
    flat_results = [item for batch in responses for item in batch]

    # 按 idx 排序
    flat_results.sort(key=lambda x: x[0])
    results = [r[1] for r in flat_results]
    return results

from typing import Generic, TypeVar, Union, NamedTuple
from openrlhf.async_pipline.rollout_output_base import Output, GenerateOutput
REMOTE_SERVER = os.getenv('REMOTE_RM_URL', '')
from openrlhf.utils.remote_rm_utils import request_api_wrapper
from openrlhf.async_pipline.base_request import process_single_request

async def default_generate(url, headers, idx, request, tokenizer, **kwargs):
    prompts = request.prompts
    assert len(request.prompts) == 1


    idx, output = await process_single_request(url, headers, idx, request, **kwargs)
    if output is None:
        return None

    stop_reason = output[0]['outputs'][0]['stop_reason']
    finish_reason = output[0]['outputs'][0]['finish_reason']
    new_prompt = request.prompts[0]
    output_text = output[0]['outputs'][0]['text']

    label = json.loads(request.label)
    reward_data = {"query": [new_prompt+output_text], 
                    "prompts": [new_prompt], 
                    "labels": [request.label], 
                    "templates": label.get('template', 'ZERO_TIR'),
                    "stop_reason": [stop_reason], 
                    "finish_reason": [finish_reason],
                    "use_model_reward": label.get('use_model_reward', 'yes')}
    reward_info = await request_api_wrapper(REMOTE_SERVER, reward_data)

    token_ids = list(output[0]['outputs'][0]['token_ids'])
    action_masks = [1] * len(token_ids)

    output = GenerateOutput(
            outputs=[Output(
                token_ids=list(output[0]['outputs'][0]['token_ids']),
                action_mask=action_masks+[1] if tokenizer.eos_token_id not in token_ids else action_masks,
                text=output[0]['outputs'][0]['text'],
                stop_reason=output[0]['outputs'][0]['stop_reason'],
                finish_reason=output[0]['outputs'][0]['finish_reason'],
                env_exec_times=0,
                reward_info=reward_info
            )],
            prompt_token_ids=list(output[0]['prompt_token_ids']),
            request_id=request.uuids,
            label=label,
            prompt=request.prompts[0],
            request_rank=request.request_rank
    )
    return (idx, output)

async def _async_process(url, start_idx, batch, **kwargs):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    env_func = kwargs.get('env_func', None)
    process_fn = ENV_GENERATE_CONFIG.get(env_func, None)
    if process_fn is None:
        process_fn = default_generate

    results = [None] * len(batch)
    retries = [0] * len(batch)

    # async def run_task(i):
    #     async with Timer("##ASYNC PROCESS-PROCESS-FN-PROCESS##"):
    #         async with asyncio.Semaphore(TASK_MAX_CONCURRENT):  # 信号量控制并发
    #             idx = start_idx + i
    #             return await process_fn(
    #                 url=url,
    #                 headers=headers,
    #                 idx=idx,
    #                 request=batch[i],
    #                 **kwargs
    #             )

    while True:
        tasks = []
        task_indices = []
        for i, (result, retry) in enumerate(zip(results, retries)):
            if result is None and retry < MAX_RETRIES:
                idx = start_idx + i
                task = process_fn(
                    url=url,
                    headers=headers,
                    idx=idx,
                    request=batch[i],
                    **kwargs
                )
                # task = run_task(i)
                tasks.append(task)
                task_indices.append(i)

        if not tasks:
            break

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for task_index, task_result in zip(task_indices, task_results):
            # error-retry
            if isinstance(task_result, Exception) or task_result is None:
                retries[task_index] += 1
                logger.info(f"Unexpected error, please check: {task_result}\nTask {start_idx + task_index} failed on attempt {retries[task_index]}: {task_result}")
            # multiturn-intractive-error and restart from the succeeded last-turn
            elif isinstance(task_result, GenerateRequest):
                retries[task_index] += 1
                batch[task_index] = task_result
                logger.info(f"Task {start_idx + task_index} failed on attempt {retries[task_index]}: {task_result} and continue to generate")
            # task is done successfully
            else:
                results[task_index] = task_result

    successful_results = [(start_idx + i, result) for i, result in enumerate(results) if result is not None]
    failed_results = [(start_idx + i, result) for i, result in enumerate(results) if result is None]

    return successful_results, failed_results

async def _async_process_queue(url, start_idx, batch, **kwargs):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    env_func = kwargs.get('env_func', None)
    process_fn = ENV_GENERATE_CONFIG.get(env_func, None)
    if process_fn is None:
        process_fn = default_generate
    
    concurrency = kwargs.get('concurrency', 32)  # 并发worker数量
    queue = asyncio.Queue(maxsize=1000)  # 防止队列无限增长
    results = {}
    lock = asyncio.Lock()  # 用于结果字典的线程安全访问

    # 初始化队列
    for i, request in enumerate(batch):
        await queue.put({
            "orig_idx": start_idx + i,
            "request": request,
            "retries": 0,
            "queue_idx": i  # 在批量中的原始位置
        })

    async def worker():
        while True:
            try:
                task = await asyncio.wait_for(queue.get(), timeout=1)
                current_idx = task["orig_idx"]
                request = task["request"]
                retries = task["retries"]
                queue_idx = task["queue_idx"]

                try:
                    async with Timer("##ASYNC PROCESS-PROCESS-FN-PROCESS##"):
                        async with asyncio.Semaphore(TASK_MAX_CONCURRENT):  # 信号量控制并发
                                result = await process_fn(
                                    url=url,
                                    headers=headers,
                                    idx=current_idx,
                                    request=request,
                                    **kwargs
                                )
                except Exception as e:
                    logger.info(f"Unexpected error, please check: {e}\nTask {current_idx} failed, Unexpected error, please check: {e}")
                    result = e

                # 处理结果
                async with lock:
                    if isinstance(result, Exception) or result is None:
                        if retries < MAX_RETRIES - 1:
                            new_task = {
                                "orig_idx": current_idx,
                                "request": request,
                                "retries": retries + 1,
                                "queue_idx": queue_idx
                            }
                            await queue.put(new_task)
                            logger.info(f"Task {current_idx} failed, retrying ({retries+1}/{MAX_RETRIES})")
                        else:
                            results[queue_idx] = None
                            logger.error(f"Task {current_idx} failed after {MAX_RETRIES} attempts")
                    elif isinstance(result, GenerateRequest):
                        new_task = {
                            "orig_idx": current_idx,
                            "request": result,  # 更新请求
                            "retries": retries,
                            "queue_idx": queue_idx
                        }
                        await queue.put(new_task)
                        logger.info(f"Task {current_idx} requires continuation")
                    else:
                        results[queue_idx] = result

                queue.task_done()
            except asyncio.TimeoutError:
                break  # 队列为空超时，结束worker

    # # 创建worker池
    # workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    # await queue.join()  # 等待所有任务完成

    # 动态调整并发度（示例）
    workers = []
    for _ in range(min(concurrency, len(batch))):  # 避免创建多余worker
        worker_task = asyncio.create_task(worker())
        workers.append(worker_task)

    await queue.join()

    # 清理worker
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    # 整理结果
    successful = []
    failed = []
    for i in range(len(batch)):
        result = results.get(i)
        if result is not None:
            successful.append((start_idx + i, result))
        else:
            failed.append((start_idx + i, None))
    
    return successful, failed


async def _async_process_hybrid(url, start_idx, batch, **kwargs):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    env_func = kwargs.get('env_func', None)
    process_fn = ENV_GENERATE_CONFIG.get(env_func, None)
    if process_fn is None:
        process_fn = default_generate
    
    concurrency = kwargs.get('concurrency', 32)
    
    # 创建双队列系统
    fast_queue = asyncio.Queue(maxsize=1000)  # 高优先级队列（新任务）
    retry_queue = asyncio.Queue()           # 普通优先级队列（重试任务）
    results = {}
    lock = asyncio.Lock()

    # 初始化高优先级队列
    for i, request in enumerate(batch):
        await fast_queue.put({
            "orig_idx": start_idx + i,
            "request": request,
            "retries": 0,
            "queue_idx": i,
            "priority": 1  # 1=高优先级
        })

    async def worker():
        while True:
            try:
                # 优先获取高优先级任务，非阻塞方式
                try:
                    task = fast_queue.get_nowait()
                except asyncio.QueueEmpty:
                    # 没有高优先级任务时获取普通任务
                    task = await retry_queue.get()
                
                current_idx = task["orig_idx"]
                request = task["request"]
                retries = task["retries"]
                queue_idx = task["queue_idx"]

                try:
                    result = await process_fn(
                        url=url,
                        headers=headers,
                        idx=current_idx,
                        request=request,
                        **kwargs
                    )
                except Exception as e:
                    result = e

                async with lock:
                    if isinstance(result, Exception) or result is None:
                        if retries < MAX_RETRIES - 1:
                            new_task = {
                                "orig_idx": current_idx,
                                "request": request,
                                "retries": retries + 1,
                                "queue_idx": queue_idx,
                                "priority": 2  # 重试任务低优先级
                            }
                            # 根据重试次数选择队列
                            if retries + 1 < 2:  # 前两次重试仍用快速队列
                                await fast_queue.put(new_task)
                            else:
                                await retry_queue.put(new_task)
                            logger.info(f"Task {current_idx} retry {retries+1}")
                        else:
                            results[queue_idx] = None
                    elif isinstance(result, GenerateRequest):
                        # 多轮交互任务保持高优先级
                        new_task = {
                            "orig_idx": current_idx,
                            "request": result,
                            "retries": retries,
                            "queue_idx": queue_idx,
                            "priority": 1
                        }
                        await fast_queue.put(new_task)
                    else:
                        results[queue_idx] = result

                # 标记任务完成
                if task["priority"] == 1:
                    fast_queue.task_done()
                else:
                    retry_queue.task_done()

            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)  # 避免CPU空转

    # 创建固定数量的worker
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(fast_queue.join(), retry_queue.join())

    # 清理worker
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    # 整理结果（同原始方案）
    successful = []
    failed = []
    for i in range(len(batch)):
        if (result := results.get(i)) is not None:
            successful.append((start_idx + i, result))
        else:
            failed.append((start_idx + i, None))
    
    return successful, failed

async def process_batch_requests(url: str, start_idx: int, batch: List[Dict], **kwargs) -> List[Tuple[int, Optional[str]]]:
    """Ray 远程任务：处理一个 batch 的请求（同步包装异步逻辑），使用信号量控制并发"""
    return await (_async_process_queue(url, start_idx, batch, **kwargs))

@ray.remote
def process_batch_requests_ray(url: str, start_idx: int, batch: List[Dict], **kwargs) -> List[Tuple[int, Optional[str]]]:
    """Ray 远程任务：处理一个 batch 的请求（同步包装异步逻辑），使用信号量控制并发"""
    return  asyncio.run(_async_process(url, start_idx, batch, **kwargs))