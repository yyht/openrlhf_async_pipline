

import os, sys
import ray
import os, copy, httpx
import uuid, time
import asyncio

import sys, os

import requests
from env.math.code_exec import run_code
from env.math.extract_code import extract_code
from collections import OrderedDict
import re, uuid
from vllm import SamplingParams
import random

import logging
import asyncio, httpx
import os, ray

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 1))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

session = requests.Session()

import random
code_pattern = re.compile(r"```python.*?```", re.DOTALL)
RANK = int(os.getenv('RANK', '1000'))
ENV_ITER_NUM = int(os.getenv('ENV_ITER_NUM', '2'))

COMPILE_SERVER = os.getenv('COMPILE_SERVER', '')
DEBUG_FLAG = os.getenv('DEBUG_FLAG', '')
REMOTE_SERVER = os.getenv('REMOTE_RM_URL', '')

logger.info({
    'INFO': 'COMPILE_SERVER',
    "VALUE": COMPILE_SERVER,
    'ENV_ITER_NUM': ENV_ITER_NUM,
    "REMOTE_SERVER": REMOTE_SERVER
})

from typing import Generic, TypeVar, Union, NamedTuple
from typing import Optional, Any, List, Dict, Tuple
from openrlhf.async_pipline.base_request import process_single_request
from openrlhf.utils.remote_rm_utils import request_api_wrapper
from openrlhf.async_pipline.show_timer import Timer
from openrlhf.async_pipline.rollout_output_base import Output, GenerateOutput
from openrlhf.async_pipline.utils import make_async

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 8))  # 最大并发数，可根据需要调整


async def remote_compile(code4exec, uuid_str, try_max_times=10, score_key='exec_result'):

    headers = {
        "Content-Type": "application/json",
    }

    data = {
        'query': code4exec,
        'uuid_str': uuid_str,
    }

    # time.sleep(10*random.random())
    # await asyncio.sleep(10)  # 异步休眠
    # time.sleep(10.0)

    for try_idx in range(try_max_times):
        url = COMPILE_SERVER+'/compile_python'
        try:
            # 检查服务器状态
            if try_idx > 0:
                try:
                    async with Timer("##ASYNC CODE-COMPILE-HEALTH-CHECK##"):
                        async with httpx.AsyncClient(timeout=None) as client:
                            async with asyncio.Semaphore(1):  # 信号量控制并发
                                health_check_url = COMPILE_SERVER + '/health'
                                health_response = await client.get(health_check_url, timeout=10)
                                health_response.raise_for_status()
                except requests.RequestException as e:
                    logger.info(f"Server is not healthy: {e}")
                    # 指数退避策略
                    wait_time = 2 ** try_idx
                    logger.info(f"Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(min([wait_time, 300]))  # 异步休眠
                    continue
            async with Timer("##ASYNC CODE-COMPILE##"):
                async with httpx.AsyncClient(timeout=None) as client:
                    async with asyncio.Semaphore(1):  # 信号量控制并发
                        response = await client.post(url=url, json=data, headers=headers, timeout=180)
                        response.raise_for_status()  # Raise an HTTPError for bad responses
                        response = response.json()
                        return uuid_str, response.get(score_key)
        except requests.RequestException as e:
            logger.info({
                'INFO': "RequestException",
                "VALUE": f"Request error, please check: {e}",
                "CODE": code4exec
            })
        except Exception as e:
            logger.info({
                'INFO': "UnexpectedException",
                "VALUE": f"Unexpected error, please check: {e}",
                "CODE": code4exec
            })
        # 指数退避策略
        wait_time = 2 ** try_idx
        await asyncio.sleep(min([wait_time, 300]))  # 异步休眠
    return uuid_str, "RunTime ERROR"


async def math_tir_generate_async(url, headers, idx, request, tokenizer, **kwargs):

    prompts = request.prompts

    assert len(request.prompts) == 1
    
    output_token_ids = []
    action_masks = []
    output_text = ''
    stop_reason = ''
    finish_reason = ''
    request_id = ''
    env_exec_times = 0

    prompt_token_ids = request.prompt_token_ids
    
    is_terminated = False
    max_tokens = request.max_tokens

    turn = 0
    new_request = copy.copy(request)

    iterative_num = 0
    ITERA_NUM = 2 * ENV_ITER_NUM # state_num * iter_num

    new_prompt = copy.copy(prompts[0])
    all_request_id = request.uuids

    if new_request.iterative_num == 0:
        new_request.stop += ['```python']
        new_request.stop = list(set(new_request.stop))

    if new_request.iterative_num > 0:
        iterative_num = request.iterative_num
        output_text = request.output_text
        output_token_ids = tokenizer(new_request.output_text)['input_ids']
        action_masks = [1]*len(output_token_ids)
        env_exec_times = request.env_exec_times
        logger.info("##Recontinue for env-interactive##")

    while not is_terminated:

        logger.info({
            'ITER-INFO': iterative_num
        })

        if iterative_num >= ITERA_NUM:
            for stop_token in ['```', '```\n', '```\n\n']:
                if stop_token in new_request.stop:
                    new_request.stop.remove(stop_token)
            if '```python' in new_request.stop:
                new_request.stop.remove('```python')
        left_max_tokens = max_tokens - len(output_token_ids)
        if left_max_tokens > 0:
            new_request.max_tokens = left_max_tokens
        else:
            new_request.max_tokens = 1024

        new_request.prompts = [new_prompt+output_text]
        
        idx, output = await process_single_request(url, headers, idx, new_request, **kwargs)
        
        if output is None:
            if iterative_num == 0:
                logger.info("##OUTPUT IS NONE##")
                return output
            else:
                logger.info(f"##OUTPUT IS NONE AND iterative_num: {iterative_num}##")
                new_request.prompts = [new_prompt]
                new_request.output_text = output_text
                new_request.iterative_num = iterative_num
                new_request.env_exec_times = env_exec_times
                new_request.max_tokens = max_tokens
                return new_request

        text = output[0]['outputs'][0]['text']
        token_ids = list(output[0]['outputs'][0]['token_ids'])
        action_mask = [1] * len(token_ids)
        stop_reason = output[0]['outputs'][0]['stop_reason']
        finish_reason = output[0]['outputs'][0]['finish_reason']
        
        if output[0]['outputs'][0]['stop_reason'] in ['```python']:
            new_request.stop += ['```', '```\n', '```\n\n']
            new_request.stop = list(set(new_request.stop))

            if '```python' in new_request.stop:
                new_request.stop.remove('```python')
            
            output_text += text

            if DEBUG_FLAG == 'yes':
                logger.info({
                    'STAGE': 'code-gen',
                    'prompt': prompts,
                    'output': text,
                    'params': new_request.stop
                })
            
        elif output[0]['outputs'][0]['stop_reason'] in ['```', '```\n', '```\n\n']:
            new_request.stop += ['```python']
            new_request.stop = list(set(new_request.stop))
            for stop_token in ['```', '```\n', '```\n\n']:
                if stop_token in new_request.stop:
                    new_request.stop.remove(stop_token)

            code_text = re.findall(code_pattern, f"```python\n{text}")
            
            if DEBUG_FLAG == 'yes':
                logger.info({
                    'STAGE': 'detect-code-exec',
                    'code_text': code_text,
                    'output': text,
                    'params': new_request.stop
                })
            
            if code_text:
                code_text = code_text[0]
                code4exec = extract_code(code_text)

                if DEBUG_FLAG == 'yes':
                    logger.info({
                        'STAGE': 'code-exec',
                        'code_text': code4exec,
                        'output': text,
                        'params': new_request.stop
                    })
                
                if code4exec:

                    if not COMPILE_SERVER:
                        try:
                            result = run_code(code4exec)
                        except Exception as e:
                            result = str(e)
                    else:
                        try:
                            _, result = await remote_compile(code4exec, all_request_id, try_max_times=MAX_RETRIES)
                        except:
                            result = 'TimeOut Error'

                    # be careful about the output pattern, make stop-token be complete
                    code_output = f"""\n\n\n```output\n{result}\n```\n\n\n"""
                    # code_output = f"""\n{result}"""

                    if DEBUG_FLAG == 'yes':
                        logger.info({
                            'STAGE': 'code-exec-output',
                            'code_text': code_output,
                            'output': text,
                            'params': new_request.stop
                        })

                    code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                    # tokenizer_outputs = await make_async(tokenizer)(code_output, add_special_tokens=False)
                    # code_output_ids = tokenizer_outputs['input_ids']

                    if len(code_output_ids) > 512:
                        code_output = """The output of the code is too long, please check your code."""
                        code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                        # tokenizer_outputs = await make_async(tokenizer)(code_output, add_special_tokens=False)
                        # code_output_ids = tokenizer_outputs['input_ids']

                    token_ids.extend(code_output_ids)
                    action_mask.extend([0]*len(code_output_ids))
                    env_exec_times += 1
                else:
                    code_output = ''

                output_text += (text+code_output)
            else:
                is_terminated = True
                output_text += text
        else:
            is_terminated = True
            output_text += text

        output_token_ids.extend(token_ids)
        action_masks.extend(action_mask)

        if iterative_num >= ITERA_NUM:
            break
        if len(output_token_ids) > max_tokens:
            break
        iterative_num += 1
    
    label = json.loads(new_request.label)
    reward_data = {"query": [new_prompt+output_text], 
                    "prompts": [new_prompt], 
                    "labels": [new_request.label], 
                    "templates": label.get('template', 'ZERO_TIR'),
                    "stop_reason": [stop_reason], 
                    "finish_reason": [finish_reason],
                    "use_model_reward": label.get('use_model_reward', 'yes')}
    reward_info = await request_api_wrapper(REMOTE_SERVER, reward_data)

    output = GenerateOutput(
            outputs=[Output(
                token_ids=output_token_ids,
                action_mask=action_masks+[1] if tokenizer.eos_token_id not in output_token_ids else action_masks,
                text=output_text,
                stop_reason=stop_reason,
                finish_reason=finish_reason,
                env_exec_times=env_exec_times,
                reward_info=reward_info
            )],
            prompt_token_ids=prompt_token_ids,
            request_id=all_request_id,
            label=label,
            prompt=request.prompts[0],
            request_rank=request.request_rank
    )
    return (idx, output)
