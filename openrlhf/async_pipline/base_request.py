
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
import aiohttp
from aiohttp import ClientTimeout
import os, sys
from openrlhf.async_pipline.show_timer import Timer

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 10))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 1000000000))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 32))  # 最大并发数，可根据需要调整
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info({
    'INFO': "##MAX_CONCURRENT##",
    'VALUE': MAX_CONCURRENT
})

async def process_single_request(url, headers, idx, request, **kwargs):
    if isinstance(url, str):
        result = await process_single_request_server(url, headers, idx, request, **kwargs)
        return result
    else:
        result = await process_single_request_model(url, headers, idx, request, **kwargs)
        return result

async def process_single_request_server(url, headers, idx, request, **kwargs):
    """处理单个请求（含重试逻辑和信号量控制）"""
    connector = aiohttp.TCPConnector(
        keepalive_timeout=30,  # 空闲连接保持时间(秒)
        limit=100              # 总连接池大小
    )

    async with aiohttp.ClientSession(
        connector=connector, timeout=None) as session:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # async with asyncio.Semaphore(MAX_CONCURRENT):  # 信号量控制并发
                    async with Timer("##ASYNC-ROLLOUT-PROCESS##"):
                        # 构造请求负载
                        payload = request.to_json()
                        # 发送请求
                        response = await session.post(
                            url,
                            headers=headers,
                            json=payload,
                            timeout=REQUEST_TIMEOUT
                        )
                        response.raise_for_status()
                        data = await response.json()
                        return (idx, data)
            except Exception as e:
                logger.warning(f"[{idx}] Attempt {attempt} failed: {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"[{idx}] Failed after {MAX_RETRIES} attempts.")
                    return (idx, None)
                await asyncio.sleep(attempt * 1.2)  # 指数退避
    return (idx, None)

async def process_single_request_model(model, headers, idx, request, **kwargs):
    """处理单个请求（含重试逻辑和信号量控制）"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # async with asyncio.Semaphore(MAX_CONCURRENT):  # 信号量控制并发
                async with Timer("##ASYNC-ROLLOUT-PROCESS##"):
                    # 构造请求负载
                    # 发送请求
                    response = await model(request)
                    if isinstance(response[0], dict):
                        return (idx, response)
                    else:
                        data = [{
                        'outputs':[
                                {
                                    "text": response[0].outputs[0].text,
                                    "token_ids": response[0].outputs[0].token_ids,
                                    "stop_reason": response[0].outputs[0].stop_reason,
                                    "finish_reason": response[0].outputs[0].finish_reason,
                                }
                            ],
                            "prompt_token_ids": response[0].prompt_token_ids
                        }]
                        return (idx, data)
        except Exception as e:
            logger.warning(f"[{idx}] Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"[{idx}] Failed after {MAX_RETRIES} attempts.")
                return (idx, None)
            await asyncio.sleep(attempt * 1.2)  # 指数退避
    return (idx, None)
