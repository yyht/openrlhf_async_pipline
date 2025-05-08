
import time
import ray
import requests
import torch
import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger
import numpy as np
import random, os, time
import aiohttp
import asyncio

logger = init_logger(__name__)

session = requests.Session()

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

CONCURRENT = int(os.getenv('CONCURRENT', '100'))
WINDOW = int(os.getenv('WINDOW', '30'))
CONCURRENCY = int(os.getenv('CONCURRENCY', '100'))

logger.info({
    'INFO': '##CONCURRENT##',
    'CONCURRENT': CONCURRENT,
    'WINDOW': WINDOW,
    'CONCURRENCY': CONCURRENCY,
})

from aiolimiter import AsyncLimiter

from openrlhf.utils.remote_rm_utils import request_api_wrapper, remote_rm_fn

@ray.remote(concurrency_groups={"io": CONCURRENCY})
class RemoteRMActor(object):

    @ray.method(concurrency_group="io")
    async def remote_rm_fn(self, api_url, queries, prompts, labels, stop_reason=None, finish_reason=None, score_key="rewards"):
        rate_limit = AsyncLimiter(CONCURRENT, WINDOW)
        data = {"query": queries, "prompts": prompts, 
                "labels": labels, "templates": TEMPLATE_NAME,
                "stop_reason": stop_reason, "finish_reason": finish_reason,
                "use_model_reward": USE_MODEL_REWARD}
        async with rate_limit:
            scores = await request_api_wrapper(url=api_url, data=data, score_key=score_key)
        return scores