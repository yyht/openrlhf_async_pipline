
import os, sys, uuid

import logging
import asyncio
import ray
import logging
import asyncio
import os
import httpx

from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
import aiohttp
import aiohttp
from aiohttp import ClientTimeout
from asyncio import Queue

from openrlhf.async_pipline.show_timer import Timer

import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from pydantic import BaseModel
class HttpRequest(BaseModel):
    prompt: str = ''
    query: str = ''
    response: str = ''
    uuids: Optional[str] = ''
    http_method: str = 'health'
    url: str = 'http://0.0.0.0:8000'
    max_retries: int = 10
    request_timeout: float = 10.0
    max_concurrent: int = 10
    authorization: Optional[str] = "EMPTY"
    uuid_str: str = ""
    output_key: Optional[str] = ''
    gold_ans: str = ''
    meta: Optional[dict] = {}

    def to_json(self):
        return {
            "prompt": self.prompt,
            "query": self.query,
            "response": self.response,
            "uuids": self.uuids,
            "http_method": self.http_method,
            "url": self.url,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
            "max_concurrent": self.max_concurrent,
            "authorization": self.authorization,
            "uuid_str": self.uuid_str,
            "output_key": self.output_key,
            "gold_ans": self.gold_ans,
            "meta": self.meta
        }

async def process_single_request(request, **kwargs):
    """处理单个请求（含重试逻辑和信号量控制）"""
    headers = {
        "Content-Type": "application/json"
    }
    task_url = f"{request.url}/{request.http_method}"

    connector = aiohttp.TCPConnector(
        keepalive_timeout=30,  # 空闲连接保持时间(秒)
        limit=100              # 总连接池大小
    )

    semaphore = asyncio.Semaphore(request.max_concurrent)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=ClientTimeout(total=180)) as session:
        for attempt in range(1, request.max_retries + 1):
            try:
                async with Timer("##ASYNC PROCESS-SINGLE-PROCESS##"):
                    async with semaphore:  # 信号量控制并发
                        # 构造请求负载
                        payload = request.to_json()
                        # 发送请求
                        response = await session.post(
                            task_url,
                            headers=headers,
                            json=payload,
                            timeout=request.request_timeout
                        )
                        response.raise_for_status()
                        data = await response.json()
                        return (request.uuids, data)
            except Exception as e:
                logger.warning(f"[{request.uuids}] Attempt {attempt} failed: {e} of {task_url}")
                if attempt == request.max_retries:
                    logger.error(f"[{request.uuids}] Failed after {request.max_retries} attempts of {task_url}.")
                    return (request.uuids, None)
                await asyncio.sleep(attempt * 1.2)  # 指数退避
    return (request.uuids, None)