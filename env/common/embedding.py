

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

import asyncio, random
from openrlhf.async_pipline.show_timer import Timer
import numpy as np
import random, os, time
import aiohttp
import asyncio, httpx, json
from env.common.http_async_interface import process_single_request, HttpRequest

import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDING_MODEL_SERVER = os.getenv('EMBEDDING_MODEL_SERVER', '')

logger.info({
    'INFO': '##EMBEDDING_MODEL_SERVER##',
    'VALUE': EMBEDDING_MODEL_SERVER
})

import openai
client = openai.Client(
            base_url=f"{EMBEDDING_MODEL_SERVER}/v1", 
            api_key="EMPTY")

async def embedding_server(prompt, response, label, **generation_kwargs):

    uuids = label['uuid']
    max_retries = 3

    for attempt in range(1, generation_kwargs.get('max_retries', 1) + 1):
        try:
            async with Timer("##ASYNC PROCESS-EMBEDDING##"):
                async with asyncio.Semaphore(generation_kwargs.get('max_concurrent', 1)):  # 信号量控制并发
                    # 构造请求负载
                    responses = client.embeddings.create(
                        input=[response],
                        model='embed',
                    )
                    embeddings = responses.data[0].embedding
                    return embeddings
        except Exception as e:
            logger.warning(f"[{uuids}] Attempt {attempt} failed: {e} of {EMBEDDING_MODEL_SERVER}")
            if attempt == max_retries:
                logger.error(f"[{uuids}] Failed after {max_retries} attempts  of {EMBEDDING_MODEL_SERVER}.")
                return None
            await asyncio.sleep(attempt * 1.2)  # 指数退避

        
