import os
import sys
import uuid
import logging
import asyncio
import httpx

from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
from pydantic import BaseModel
from openrlhf.async_pipline.show_timer import Timer  # 假设路径正确

# 配置日志
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


async def process_single_request(request: HttpRequest, **kwargs):
    """
    使用 httpx 处理单个请求（含重试逻辑和信号量控制）
    """
    headers = {
        "Content-Type": "application/json"
    }
    if request.authorization:
        headers["Authorization"] = request.authorization

    task_url = f"{request.url}/{request.http_method}"

    # 创建客户端配置（连接池 + 超时）
    transport = httpx.AsyncHTTPTransport(
        limits=httpx.Limits(
            max_connections=100,           # 总连接数
            max_keepalive_connections=30,  # 长连接数
            keepalive_expiry=30.0          # 保持时间
        )
    )

    timeout = httpx.Timeout(timeout=180.0)  # 整体超时（与 session 级别一致）

    # 控制最大并发请求数
    semaphore = asyncio.Semaphore(request.max_concurrent)

    async with httpx.AsyncClient(transport=transport, timeout=timeout, headers=headers) as client:
        for attempt in range(1, request.max_retries + 1):
            try:
                async with Timer("##ASYNC PROCESS-SINGLE-PROCESS##"):
                    async with semaphore:  # 控制并发量
                        payload = request.to_json()

                        # 发送 POST 请求
                        response = await client.post(
                            task_url,
                            json=payload,
                            timeout=httpx.Timeout(request.request_timeout)
                        )
                        response.raise_for_status()
                        data = response.json()
                        return request.uuids, data

            except Exception as e:
                logger.warning(f"[{request.uuids}] Attempt {attempt} failed: {e} | URL: {task_url}")
                if attempt == request.max_retries:
                    logger.error(f"[{request.uuids}] Failed after {request.max_retries} attempts: {task_url}")
                    return request.uuids, None

                # 指数退避：等待 attempt * 1.2 秒
                await asyncio.sleep(attempt * 1.2)

        # 理论上不会走到这里，保险返回
        return request.uuids, None