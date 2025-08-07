

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import socket, asyncio
import ray, random
import uvicorn, torch
from vllm import LLM, SamplingParams
from typing import Optional, Any, List, Dict, Tuple
from typing import List, Dict, Union, Any
import os, sys, uuid

import logging
import asyncio
import ray
import logging
import asyncio
import os
import httpx
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
import aiohttp
from asyncio import Queue

from openrlhf.async_pipline.show_timer import Timer

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

from pydantic import BaseModel
class HttpRequest(BaseModel):
    prompt: str
    query: str
    uuids: Optional[str] = None
    http_method: str = 'health'
    url: str = 'http://0.0.0.0:8000'
    max_retries: int = 10
    request_timeout: float = 10.0
    max_concurrent: int = 10
    authorization: Optional[str] = "EMPTY"
    uuid_str: str = ""
    output_key: Optional[str] = None

    def to_json(self):
        return {
            "prompt": self.prompt,
            "query": self.query,
            "uuids": self.uuids,
            "http_method": self.http_method,
            "url": self.url,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
            "max_concurrent": self.max_concurrent,
            "authorization": self.authorization,
            "uuid_str": self.uuid_str,
            "output_key": self.output_key
        }

async def process_single_request(request, **kwargs):
    """处理单个请求（含重试逻辑和信号量控制）"""
    headers = {
        "Content-Type": "application/json"
    }
    task_url = f"{request.url}/{request.http_method}"
    for attempt in range(1, request.max_retries + 1):
        try:
            async with Timer("##ASYNC PROCESS-SINGLE-PROCESS##"):
                async with httpx.AsyncClient(timeout=None) as client:
                    async with asyncio.Semaphore(request.max_concurrent):  # 信号量控制并发
                        # 构造请求负载
                        payload = request.to_json()
                        # 发送请求
                        response = await client.post(
                            task_url,
                            headers=headers,
                            json=payload,
                            timeout=request.request_timeout
                        )
                        response.raise_for_status()
                        data = response.json()
                        return (request.uuids, data)
        except Exception as e:
            logger.warning(f"[{request.uuids}] Attempt {attempt} failed: {e}")
            if attempt == request.max_retries:
                logger.error(f"[{request.uuids}] Failed after {request.max_retries} attempts.")
                return (request.uuids, None)
            await asyncio.sleep(attempt * 1.2)  # 指数退避
    return (request.uuids, None)

import ray
@ray.remote(num_cpus=1)
class ServerController:
    def __init__(self):

        self.server = None
        self.server_ready = asyncio.Event()
        self.port = None
        try:
            self.address = ray._private.services.get_node_ip_address()
        except:
            self.address = '0.0.0.0'

        self.batch_size = 16

        self.worker_num = 10
        self.max_queue_size = 1024
        self.request_queue: Queue = Queue(maxsize=self.max_queue_size)
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.workers = []

        self.max_batch_size = 16  # 单个批次最大请求数
        self.max_wait_time = 1e-1 # 批次等待时间（秒）
        
        # 优先级队列存储元组 (priority, insertion_order, request_data)
        self.priority_queue = []
        self.queue_index = 0
        
        self.lock = asyncio.Lock()
        self.server_start = False

    def start_server(self):
        if not self.server_start:
            asyncio.create_task(self._start_fastapi_server())
            self.server_start = True

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

    async def _worker_loop(self):
        """工作协程循环"""
        while self._running:
            try:
                request_id, request, future = await self.request_queue.get()
                
                response = process_single_request(request)
                
                future.set_result(response)
                self.pending_requests.pop(request_id, None)
                
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                self.request_queue.task_done()

    async def async_infer(self, request: HttpRequest):
        """异步生成接口"""
        # if self.request_queue.qsize() >= self.max_queue_size:
        #     raise RuntimeError("Request queue is full")

        while self.request_queue.full():
            logging.warning("Request queue is full. Waiting for space...")
            await asyncio.sleep(0.1)
        
        # 创建异步Future
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = request.uuids
        
        # 将请求存入等待队列
        self.pending_requests[request_id] = future
        await self.request_queue.put((request_id, request, future))
        
        try:
            return await future
        except Exception as e:
            print(f"Error in async_generate: {e}")
            if not future.done():
                future.set_exception(e)
            self.pending_requests.pop(request_id, None)
            raise
        # finally:
        #     # 确保异常时清理资源
        #     self.pending_requests.pop(request_id, None)

    async def get_server_address(self):
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    async def get_server_port(self):
        await self.server_ready.wait()
        return self.port

    async def health(self):
        return 1
    
    async def _start_fastapi_server(self):
        import fastapi
        app = fastapi.FastAPI()
        app.router.add_api_route("/health", self.health, methods=["GET"])
        app.router.add_api_route("/async_infer", self.async_infer, methods=["POST"])

        await asyncio.sleep(random.uniform(0, 3))
        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port)
        self.server = uvicorn.Server(config)  # 保存实例
        self.server_ready.set()
        await self.start()
        await self.server.serve()






