import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.default_reward_remote_fn import default_remote_fn

logger = init_logger(__name__)

session = requests.Session()
import numpy as np
import random, os, time
import aiohttp
import asyncio, httpx, json

STRUCTURED_REWARD = os.getenv('STRUCTURED_REWARD', 'NONE')
TEMPLATE_NAME = os.getenv('TEMPLATE_NAME', 'NONE')
RANK = int(os.getenv('RANK', '1000'))
USE_MODEL_REWARD = os.getenv('USE_MODEL_REWARD', 'no')

from openrlhf.utils.load_balancer import LoadBalancer
url_list = os.getenv('REMOTE_RM_URL', '').split('#')
if len(url_list) > 10:
    balancer = LoadBalancer(url_list)
else:
    balancer = None

logger.info({
    'INFO': '##USE_MODEL_REWARD##',
    'VALUE': USE_MODEL_REWARD
})

import asyncio
from openrlhf.async_pipline.show_timer import Timer

try:
    from aiolimiter import AsyncLimiter
    CONCURRENT = int(os.getenv('CONCURRENT', '100'))
    WINDOW = int(os.getenv('WINDOW', '30'))
    rate_limit = AsyncLimiter(CONCURRENT, WINDOW)
except:
    rate_limit = None

async def request_api_wrapper(url, data, score_key="rewards", try_max_times=10):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    url_list = url.split('#')
    
    for try_idx in range(try_max_times):
        seed = random.randint(0, 100000000+1000*RANK)
        random.seed(seed)
        url_base = random.randint(0, (1000+(try_idx+RANK)*100)*(10000+RANK))
        url_idx = url_base+(try_idx+1)*random.randint(1, 10000)
        try:
            # time.sleep(sleep_time)
            if len(url_list) > 10:
                response = balancer.send_request(data, headers, 
                                    current_index=url_idx, method='/get_reward',
                                    timeout=180)
            else:
                # response = session.post(url=url_list[url_idx%len(url_list)]+'/get_reward', 
                #                         json=data, headers=headers, timeout=180)
                # # response = session.post(url=url, json=data, headers=headers, timeout=180)
                # response.raise_for_status()  # Raise an HTTPError for bad responses
                # response = response.json()
                async with Timer("##ASYNC REMOTE-REWARD##"):
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with asyncio.Semaphore(1):  # 信号量控制并发
                            response = await client.post(url=url_list[url_idx%len(url_list)]+'/get_reward', json=data, headers=headers, timeout=180)
                            response.raise_for_status()  # Raise an HTTPError for bad responses
                            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info({
                'INFO': "RequestException",
                "VALUE": f"Request error, please check: {e}",
                "CODE": json.dumps(data, ensure_ascii=False)
            })
        except Exception as e:
            logger.info({
                'INFO': "UnexpectedException",
                "VALUE": f"Unexpected error, please check: {e}",
                "CODE": json.dumps(data, ensure_ascii=False)
            })
        await asyncio.sleep(1.0)  # 异步休眠
    response = default_remote_fn(data)
    return response.get(score_key)

if rate_limit:
    async def do_coroutine(url, data, score_key):
        async with rate_limit:
            # this section is *at most* going to entered 100 times
            # in a 30 second period.
            scores = await request_api_wrapper(url=url, data=data, score_key=score_key)
            return scores

async def remote_rm_fn(api_url, queries, prompts, labels, stop_reason=None, finish_reason=None, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    if rate_limit:
        scores = await do_coroutine(api_url, {"query": queries, "prompts": prompts, 
                    "labels": labels, "templates": TEMPLATE_NAME,
                    "stop_reason": stop_reason, "finish_reason": finish_reason,
                    "use_model_reward": USE_MODEL_REWARD,
                    "labels": labels}, score_key)
    else:
        scores = await request_api_wrapper(api_url, {"query": queries, "prompts": prompts, 
                        "labels": labels, "templates": TEMPLATE_NAME,
                        "stop_reason": stop_reason, "finish_reason": finish_reason,
                        "use_model_reward": USE_MODEL_REWARD,
                        "labels": labels}, score_key)
    if STRUCTURED_REWARD == "NONE":
        return torch.tensor(scores)
    else:
        return scores


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels, stop_reason=None, finish_reason=None, score_key="rewards"):
    import asyncio
    task = remote_rm_fn(api_url, queries, prompts, labels, stop_reason, finish_reason, score_key)
    res = asyncio.run(task)
    return res 

if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
