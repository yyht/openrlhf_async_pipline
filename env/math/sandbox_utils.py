import os
import uuid
import time
import asyncio
import logging
import httpx
from typing import Dict, Any

from openrlhf.async_pipline.show_timer import Timer
from env.math.code_package import BASE_IMPORTS

# 日志配置
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 环境变量
COMPILE_SERVER_PORT = os.getenv('COMPILE_SERVER_PORT', '8080')
DEBUG_FLAG = os.getenv('DEBUG_FLAG', '')
NGINX_IP_FILE = os.getenv('NGINX_IP_FILE', '')

REMOTE_SERVER = os.getenv('REMOTE_SERVER', None)

logger.info({
    'INFO': 'NGINX_IP_FILE',
    "VALUE": NGINX_IP_FILE,
    "COMPILE_SERVER_PORT": COMPILE_SERVER_PORT,
    "REMOTE_SERVER": REMOTE_SERVER
})

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 8))
MULTITURN_CODE_MERGE = os.getenv("MULTITURN_CODE_MERGE", 'NONE')
CODE_CONCURRENT = int(os.getenv("CODE_CONCURRENT", 2))

logger.info({
    'INFO': '##CODE-INFO##',
    "MAX_CONCURRENT": MAX_CONCURRENT,
    'MULTITURN_CODE_MERGE': MULTITURN_CODE_MERGE,
    "CODE_CONCURRENT": CODE_CONCURRENT
})

# 正则清理错误信息
import re

def parse_response(response: Dict[str, Any]):
    output = ''
    error = 0

    run_result = response.get('run_result', {})
    status = run_result.get('status')
    response_status = response.get('status')

    if status == 'Finished':
        if response_status == 'Success':
            output = run_result.get('stdout', '')
        else:
            stderr = run_result.get('stderr', '')
            output = re.sub(r'File ".*?", ', '', stderr)
            output = output.replace('^', '')
    else:
        output = status

    if response_status == 'Failed' or status != 'Finished':
        error = 1
    return output, error

async def remote_compile(
    code4exec: str,
    uuid_str: str,
    try_max_times: int = 10,
    score_key: str = 'exec_result',
    **kwargs
) -> tuple[str, dict]:
    headers = {"Content-Type": "application/json"}
    multiturn_codes = kwargs.get('multiturn_codes', '')

    # 读取 IP 列表
    ip_list = []
    with open(NGINX_IP_FILE) as f:
        ip_list = [line.strip() for line in f if line.strip()]
    if not ip_list:
        raise FileNotFoundError("Empty IP list")

    COMPILE_SERVER = f"{ip_list[0]}:{COMPILE_SERVER_PORT}"
    if REMOTE_SERVER:
        COMPILE_SERVER = REMOTE_SERVER

    output_dict = {
        'query': code4exec,
        'uuid': uuid_str,
        'exec_result': "RunTime ERROR",
        'error': 0
    }

    # 全局限流信号量
    semaphore = asyncio.Semaphore(CODE_CONCURRENT)

    # 重试循环
    for try_idx in range(try_max_times):
        request_id = f"{time.time_ns()}-{uuid.uuid4()}"
        if REMOTE_SERVER:
            headers['X-Request-ID'] = request_id

        data = {
            'compile_timeout': 10,
            'run_timeout': 20,
            'code': code4exec,
            'request_id': request_id,
            'language': "python"
        }

        # # === 健康检查（仅重试时）===
        # if try_idx > 0:
        #     await asyncio.sleep(0.1)  # 小延迟，避免资源竞争

        #     health_client_timeout = httpx.Timeout(timeout=10.0, connect=5.0)
        #     async with httpx.AsyncClient(timeout=health_client_timeout) as health_client:
        #         try:
        #             async with Timer("##ASYNC CODE-COMPILE-HEALTH-CHECK##"):
        #                 health_url = f"{COMPILE_SERVER}/health"
        #                 resp = await health_client.get(health_url)
        #                 resp.raise_for_status()
        #             logger.info(f"Health check passed on retry {try_idx}")
        #         except Exception as e:
        #             logger.warning(f"Health check failed on retry {try_idx}: {e}")
        #             wait_time = min(2 ** try_idx, 300)
        #             await asyncio.sleep(wait_time)
        #             continue  # 跳过本次执行，进入下一次重试

        # === 执行代码请求 ===
        client_timeout = httpx.Timeout(
            connect=10.0,
            read=40.0,
            write=20.0,
            pool=10.0
        )

        async with semaphore:  # 控制并发
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                try:
                    url = f"{COMPILE_SERVER}/run_code"
                    async with Timer("##ASYNC CODE-COMPILE##"):
                        response = await client.post(
                            url,
                            json=data,
                            headers=headers
                        )
                        response.raise_for_status()
                        response_data = response.json()

                        # 验证 request_id
                        if response_data.get('request_id') != request_id:
                            raise ValueError(
                                f"Request ID mismatch: expected {request_id}, got {response_data.get('request_id')}"
                            )

                        output, error = parse_response(response_data)
                        output_dict['exec_result'] = output
                        output_dict['error'] = error
                        return uuid_str, output_dict

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    logger.warning(f"Network/Timeout error on try {try_idx}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error on try {try_idx}: {str(e)}", exc_info=True)

        # 指数退避
        wait_time = min(2 ** try_idx, 300)
        await asyncio.sleep(wait_time)

    # 所有重试失败
    return uuid_str, output_dict