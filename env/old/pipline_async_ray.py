import asyncio
import os
import httpx
import ray
from typing import Dict, List, Optional, Tuple
import logging

# 初始化 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 Ray（本地运行）
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

PROMPT = """请根据以下问答对，判断模型回答的正确性。
问题：{question}
模型预测回答：{output}
参考答案：{answer}
请给出评价："""

# 环境变量配置
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://10.39.17.106:10005/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))

MAX_CONCURRENT = 10  # 最大并发数，可根据需要调整

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def _async_process(start_idx, batch):
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        tasks = []
        for offset, request in enumerate(batch):
            idx = start_idx + offset
            # 为每个请求创建异步任务
            task = process_single_request(
                client=client,
                semaphore=semaphore,
                url=url,
                headers=headers,
                idx=idx,
                request=request
            )
            tasks.append(task)
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
    return results

async def process_single_request(client, semaphore, url, headers, idx, request):
    """处理单个请求（含重试逻辑和信号量控制）"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:  # 信号量控制并发
                # 构造请求负载
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
                # 发送请求
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return (idx, content)
        except Exception as e:
            logger.warning(f"[{idx}] Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"[{idx}] Failed after {MAX_RETRIES} attempts.")
                return (idx, None)
            await asyncio.sleep(attempt * 1.2)  # 指数退避
    return (idx, None)

@ray.remote
def process_batch_requests(start_idx: int, batch: List[Dict]) -> List[Tuple[int, Optional[str]]]:
    """Ray 远程任务：处理一个 batch 的请求（同步包装异步逻辑），使用信号量控制并发"""
    return asyncio.run(_async_process(start_idx, batch))

@ray.remote
class RayAsyncBatchPipeline:
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.results = []

    def _create_batches(self, data_list: List[Dict]) -> List[Tuple[int, List[Dict]]]:
        """将数据分成 batch，返回 [(start_idx, batch), ...]"""
        batches = []
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]
            batches.append((i, batch))
        return batches

    def run(self, data_list: List[Dict]) -> List[str]:
        """主调度函数：分发 Ray 批处理任务"""
        if not data_list:
            logger.warning("Empty input list.")
            return []

        batch_inputs = self._create_batches(data_list)
        futures = [
            process_batch_requests.remote(start_idx, batch)
            for start_idx, batch in batch_inputs
        ]

        # 收集所有结果
        results_raw = ray.get(futures)
        flat_results = [item for batch in results_raw for item in batch]

        # 按 idx 排序
        flat_results.sort(key=lambda x: x[0])
        self.results = [r[1] for r in flat_results]
        return self.results


if __name__ == "__main__":
    data_dict = {
        'question': "Two cards are chosen at random from a standard 52-card deck. What is the probability that the first card is a spade and the second card is a king?",
        'predict_answer': '\\boxed{\\frac{17}{884}}',
        'answer': '\\frac{1}{52}'
    }
    
    pipeline = RayAsyncBatchPipeline.remote()  # 每个 worker 处理 3 条
    
    import time
    start = time.time()
    
    results = pipeline.run.remote(data_list=[data_dict]*10)
    r_refs = [results]
    results_values = ray.get(results)

    print(time.time() - start, '===time===')

    for i, res in enumerate(results_values):
        print(f"[{i}] {res}")