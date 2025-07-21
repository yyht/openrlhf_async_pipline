

import os, sys
import ray
import os, copy, httpx
import uuid, time
import asyncio

import sys, os
import logging
import asyncio, httpx
import aiohttp
from aiohttp import ClientTimeout
import os, ray
from openrlhf.async_pipline.show_timer import Timer

import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

COMPILE_SERVER_PORT = os.getenv('COMPILE_SERVER_PORT', '')
DEBUG_FLAG = os.getenv('DEBUG_FLAG', '')
NGINX_IP_FILE = os.getenv('NGINX_IP_FILE', '')

logger.info({
    'INFO': 'NGINX_IP_FILE',
    "VALUE": NGINX_IP_FILE,
    "COMPILE_SERVER_PORT": COMPILE_SERVER_PORT
})

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 8))  # 最大并发数，可根据需要调整
MULTITURN_CODE_MERGE = os.getenv("MULTITURN_CODE_MERGE", 'NONE')
CODE_CONCURRENT = int(os.getenv("CODE_CONCURRENT", 2))  # 最大并发数，可根据需要调整

logger.info({
    'INFO': '##CODE-INFO##',
    "MAX_CONCURRENT": MAX_CONCURRENT,
    'MULTITURN_CODE_MERGE': MULTITURN_CODE_MERGE,
    "CODE_CONCURRENT": CODE_CONCURRENT
})

async def remote_compile(code4exec, uuid_str, try_max_times=10, score_key='exec_result', **kwargs):

    headers = {
        "Content-Type": "application/json",
    }

    multiturn_codes = kwargs.get('multiturn_codes', '')
    request_id = f"{time.time_ns()}-{uuid.uuid4()}"

    data = {
        'query': multiturn_codes+'\n'+code4exec,
        'uuid_str': request_id,
    }

    ip_list = []
    with open(NGINX_IP_FILE) as frobj:
        for line in frobj:
            ip_list.append(line.strip())

    COMPILE_SERVER = f"{ip_list[0]}:{COMPILE_SERVER_PORT}"

    # time.sleep(10*random.random())
    # await asyncio.sleep(10)  # 异步休眠
    # time.sleep(10.0)

    connector = aiohttp.TCPConnector(
        keepalive_timeout=30,  # 空闲连接保持时间(秒)
        limit=100              # 总连接池大小
    )

    output_dict = {
        'query': code4exec,
        'uuid': uuid_str,
        'exec_result': "RunTime ERROR",
        'error': 1
    }

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=ClientTimeout(total=180)) as session:
        for try_idx in range(try_max_times):
            url = COMPILE_SERVER+'/compile_python'
            try:
                # 检查服务器状态
                if try_idx > 0:
                    try:
                        async with asyncio.Semaphore(CODE_CONCURRENT):  # 信号量控制并发
                            async with Timer("##ASYNC CODE-COMPILE-HEALTH-CHECK##"):
                                health_check_url = COMPILE_SERVER + '/health'
                                health_response = await session.get(health_check_url, timeout=10)
                                health_response.raise_for_status()
                    except requests.RequestException as e:
                        logger.info(f"Server is not healthy: {e}")
                        # 指数退避策略
                        wait_time = 2 ** try_idx
                        logger.info(f"Waiting for {wait_time} seconds before retrying...")
                        await asyncio.sleep(min([wait_time, 300]))  # 异步休眠
                        continue
                async with asyncio.Semaphore(CODE_CONCURRENT):  # 信号量控制并发
                    async with Timer("##ASYNC CODE-COMPILE##"):
                        response = await session.post(url=url, json=data, headers=headers, timeout=180)
                        response.raise_for_status()  # Raise an HTTPError for bad responses
                        response = await response.json()
                        # return uuid_str, response.get(score_key)
                        assert response['uuid'] == request_id
                        return uuid_str, response
            except requests.RequestException as e:
                logger.info({
                    'INFO': "RequestException",
                    "VALUE": f"{try_idx}-th Request error, please check: {e}",
                    "CODE": code4exec
                })
            except Exception as e:
                logger.info({
                    'INFO': "UnexpectedException",
                    "VALUE": f"{try_idx}-th Unexpected error, please check: {e}",
                    "CODE": code4exec
                })
            # 指数退避策略
            wait_time = 2 ** try_idx
            await asyncio.sleep(min([wait_time, 300]))  # 异步休眠
    return uuid_str, output_dict