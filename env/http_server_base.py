

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import socket, asyncio
import ray, random
import uvicorn, torch
from typing import Optional, Any, List, Dict, Tuple
from typing import List, Dict, Union, Any
import os, sys, uuid
import queue
import threading

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
import requests

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

session = requests.Session()

def process_single_request(request, **kwargs):
    """处理单个请求（含重试逻辑和信号量控制）"""
    headers = {
        "Content-Type": "application/json"
    }
    task_url = f"{request.url}/{request.http_method}"
    for attempt in range(1, request.max_retries + 1):
        try:
            # 构造请求负载
            payload = request.to_json()
            # 发送请求
            response = session.post(
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
            time.sleep(attempt * 1.2)  # 指数退避
    return (request.uuids, None)

class ServerController:
    def __init__(self):

        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.producer_done = threading.Event()
        self.consumer_threads = []

    def _produce_tasks(self, request_list):
        """同步生成任务到队列"""
        for request in request_list:
            self.request_queue.put(request)
        self.producer_done.set()

    def _process_task(self, request: HttpRequest):
        """处理单个提示词并返回结果"""
        for _ in range(10):
            try:
                response = process_single_request(request)
                return response
            except Exception as e:
                time.sleep(1.0)
                print(f"Request failed: {e}")
                return None

    def _consumer_loop(self):
        """消费者线程的主循环"""
        while True:
            # 检查生产者是否完成且队列已空
            if self.producer_done.is_set() and self.request_queue.empty():
                break
            
            try:
                request = self.request_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            try:
                result = self._process_task(request)
                self.result_queue.put(result)
            finally:
                self.request_queue.task_done()

    def process_requests(self, request_list: List[Any], num_requests: int, num_consumers: int):
        """
        处理请求的主入口
        :param num_requests: 总请求数量
        :param num_consumers: 消费者线程数量
        """
        # 同步生成任务（无生产者线程开销）
        self._produce_tasks(request_list)
        
        # 启动消费者线程
        self.consumer_threads = [
            threading.Thread(target=self._consumer_loop)
            for _ in range(num_consumers)
        ]
        for thread in self.consumer_threads:
            thread.start()
        
        # 等待所有任务处理完成
        self.request_queue.join()
        
        # 确保所有消费者线程退出
        for thread in self.consumer_threads:
            thread.join()

    def get_specific_results(self, request_id):
        """获取处理结果列表"""
        results = []
        while not self.result_queue.empty():
            result = self.result_queue.get()
            if result[0] == request_id:
                return result[1]
        return None

    def get_all_results(self):
        results = []
        while not self.result_queue.empty():
            result = self.result_queue.get()
            results.append(result)
        return results


if __name__ == "__main__":
    import string
    import random
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))
    
    requests_list = []
    for _ in range(10):
        uuid_str = id_generator(6)
        new_request = HttpRequest(
            prompt='print(123)',
            query='print(123)',
            url='http://10.39.17.106:10003',
            uuids=uuid_str,
            uuid_str=uuid_str,
            http_method='compile_python'
        )
        requests_list.append(new_request)
    
    pipeline = ServerController()
    
    import time
    start = time.time()
    
    pipeline.process_requests(requests_list, 5, 5)
    results = pipeline.get_all_results()
    target_result = [result for result in results if result[0] == requests_list[-1].uuids]

    # for i, r in enumerate(results):
    #     print(f"Result {i+1}: {r}")
        
    print(time.time()-start, '====', target_result)

    requests_list = []
    for _ in range(10):
        uuid_str = id_generator(6)
        new_request = HttpRequest(
            prompt='print(123)',
            query='print(123)',
            url='http://10.39.17.106:10003',
            uuids=uuid_str,
            uuid_str=uuid_str,
            http_method='compile_python'
        )
        requests_list.append(new_request)

    pipeline.process_requests(requests_list, 5, 5)
    results = pipeline.get_all_results()
    target_result = [result for result in results if result[0] == requests_list[-1].uuids]

    # for i, r in enumerate(results):
    #     print(f"Result {i+1}: {r}")
        
    print(time.time()-start, '====', target_result)








