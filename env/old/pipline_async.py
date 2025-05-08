import asyncio
import httpx
import os
from typing import List, Dict, Optional
import logging

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304_vllm083'))
from env.vllm_serve import GenerateRequest
from typing import Any, List
from env.vllm_client import VLLMClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncPipeline:
    def __init__(self, OPENAI_BASE_URL, OPENAI_API_KEY):
        # 从环境变量获取配置
        self.base_url = OPENAI_BASE_URL
        self.api_key = OPENAI_API_KEY
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", 30))
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", 10))  # 并发限制

        # 结果保持顺序
        self.results = []

    async def _process_single(self, client: httpx.AsyncClient, idx: int, request: Dict, semaphore: asyncio.Semaphore):
        """处理单个请求，有重试机制，带并发控制"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": PROMPT.format_map({
                    'question': request['question'],
                    'output': request['predict_answer'],
                    'answer': request['answer']
                })}
            ],
            "temperature": 0,
            "max_tokens": 8192
        }

        async with semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload, timeout=self.request_timeout)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return idx, content
                except Exception as e:
                    logger.warning(f"[{idx}] Attempt {attempt} failed: {e}")
                    if attempt == self.max_retries:
                        logger.error(f"[{idx}] Failed after {self.max_retries} attempts.")
                        return idx, None
                    await asyncio.sleep(1.5 * attempt)  # 简单 backoff

    async def process_requests_async(self, data_dict_list: List[Dict]):
        """主处理流程（异步）"""
        self.results = [None] * len(data_dict_list)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async with httpx.AsyncClient() as client:
            tasks = [
                self._process_single(client, idx, data_dict, semaphore)
                for idx, data_dict in enumerate(data_dict_list)
            ]

            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                self.results[idx] = result

    def run(self, data_dict_list: List[Dict]):
        """外部统一调用入口"""
        if not data_dict_list:
            logger.warning("Empty input list.")
            return []

        try:
            asyncio.run(self.process_requests_async(data_dict_list))
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
        return self.get_ordered_results()

    def get_ordered_results(self) -> List[str]:
        """返回按照输入顺序排序的结果"""
        return self.results