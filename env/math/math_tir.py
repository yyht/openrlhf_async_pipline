

import ray
import os, copy
import uuid, time
import asyncio
import aiohttp
import uuid
import logging
import queue
from threading import Thread
import sys, os
# sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_0304'))

import requests
from env.math.code_exec import run_code
from env.math.extract_code import extract_code
from collections import OrderedDict
import re, uuid
from vllm import SamplingParams
from openrlhf.async_pipline.show_timer import Timer
import asyncio
import os
import httpx

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import logging, json
import asyncio, httpx
import os, ray
from tqdm import tqdm

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 500))
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

session = requests.Session()

import random
code_pattern = re.compile(r"```python.*?```", re.DOTALL)
RANK = int(os.getenv('RANK', '1000'))
ENV_ITER_NUM = int(os.getenv('ENV_ITER_NUM', '2'))

COMPILE_SERVER = os.getenv('COMPILE_SERVER', '')
DEBUG_FLAG = os.getenv('DEBUG_FLAG', '')

logger.info({
    'INFO': 'COMPILE_SERVER',
    "VALUE": COMPILE_SERVER,
    'ENV_ITER_NUM': ENV_ITER_NUM
})

from typing import Generic, TypeVar, Union, NamedTuple
from typing import Optional, Any, List, Dict, Tuple
import queue
from threading import Thread
from openrlhf.async_pipline.rollout_output_base import Output, GenerateOutput


# def remote_compile(code4exec, try_max_times=10, score_key='exec_result'):

#     headers = {
#         "Content-Type": "application/json",
#     }

#     data = {
#         'query': code4exec,
#         'uuid_str': str(uuid.uuid4()),
#     }

#     # time.sleep(1)

#     for try_idx in range(try_max_times):
#         url = COMPILE_SERVER+'/compile_python'
#         try:
#             response = session.post(url=url, json=data, headers=headers, timeout=180)
#             response.raise_for_status()  # Raise an HTTPError for bad responses
#             response = response.json()
#             return response.get(score_key)
#         except requests.RequestException as e:
#             logger.info(f"Request error, please check: {e}")
#         except Exception as e:
#             logger.info(f"Unexpected error, please check: {e}")
#         time.sleep(1)

def remote_compile(code4exec, uuid_str, try_max_times=10, score_key='exec_result'):
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        'query': code4exec,
        'uuid_str': uuid_str,
    }

    # time.sleep(10*random.random())
    # time.sleep(1.0)

    for try_idx in range(try_max_times):
        url = COMPILE_SERVER + '/compile_python'
        try:
            # 检查服务器状态
            if try_idx > 0:
                try:
                    health_check_url = COMPILE_SERVER + '/health'
                    health_response = session.get(health_check_url, timeout=10)
                    health_response.raise_for_status()
                except requests.RequestException as e:
                    logger.info(f"Server is not healthy: {e}")
                    # 指数退避策略
                    wait_time = 2 ** try_idx
                    logger.info(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(min([wait_time, 300]))  # 异步休眠
                    continue

            response = session.post(url=url, json=data, headers=headers, timeout=300)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            # logger.info(f"Response received: {response}")
            return uuid_str, response.get(score_key)
        except requests.RequestException as e:
            # 指数退避策略
            wait_time = 2 ** try_idx
            logger.info({
                'INFO': "RequestException",
                "VALUE": f"Request error, please check: {e}",
                "CODE": code4exec
            })
            time.sleep(min([wait_time, 300]))  # 异步休眠
        except Exception as e:
            # 指数退避策略
            wait_time = 2 ** try_idx
            logger.info({
                'INFO': "UnexpectedException",
                "VALUE": f"Unexpected error, please check: {e}",
                "CODE": code4exec
            })
            time.sleep(min([wait_time, 300]))  # 异步休眠
    return uuid_str, None


def math_tir_generate(llm, sampling_params, prompt_token_ids, tokenizer, prompts=None):
    
    output_token_ids = [[] for idx in range(len(prompts))]
    action_masks = [[] for idx in range(len(prompts))]
    all_text = ['' for idx in range(len(prompts))]
    all_stop_reason = ['' for idx in range(len(prompts))]
    all_finish_reason = ['' for idx in range(len(prompts))]
    all_request_id = ['' for idx in range(len(prompts))]
    all_env_exec_times = [0 for idx in range(len(prompts))]
    if prompt_token_ids is None:
        all_prompt_token_ids = []
        for prompt in prompts:
            input_ids = tokenizer(prompt)['input_ids']
            all_prompt_token_ids.append(input_ids)
    else:
        all_prompt_token_ids = prompt_token_ids

    id2uuid = OrderedDict()
    uuid2id = OrderedDict()
    uuid2data = OrderedDict()

    for idx, prompt in enumerate(prompts):
        uuid_num = str(uuid.uuid4())
        id2uuid[idx] = uuid_num
        uuid2id[uuid_num] = idx
        uuid2data[uuid_num] = prompt

    is_all_terminated = [False for _ in range(len(prompts))]
    is_terminated = sum(is_all_terminated) == len(is_all_terminated)
    idx_list = list(range(len(prompts)))

    max_tokens = sampling_params.max_tokens

    turn = 0
    new_sampling_params = copy.copy(sampling_params)
    new_sampling_params.stop += ['```python']
    new_sampling_params.stop = list(set(new_sampling_params.stop))

    sampling_params_list = [copy.copy(new_sampling_params) for _ in range(len(prompts))]
    new_prompts = prompts.copy()

    iterative_num = 0
    ITERA_NUM = 2 * ENV_ITER_NUM # state_num * iter_num

    while not is_terminated:

        logger.info({
            'ITER-INFO': iterative_num
        })

        new_sampling_params_list = []
        for idx in idx_list:
            new_sampling_params = sampling_params_list[idx]
            if iterative_num >= ITERA_NUM:
                for stop_token in ['```', '```\n', '```\n\n']:
                    if stop_token in new_sampling_params.stop:
                        new_sampling_params.stop.remove(stop_token)
                if '```python' in new_sampling_params.stop:
                    new_sampling_params.stop.remove('```python')
            left_max_tokens = max_tokens - len(output_token_ids[idx])
            if left_max_tokens > 0:
                new_sampling_params.max_tokens = left_max_tokens
            else:
                new_sampling_params.max_tokens = 1024
            new_sampling_params_list.append(new_sampling_params)

        # print(len(new_sampling_params_list), '===', len(new_prompts))

        outputs = llm.generate(sampling_params=new_sampling_params_list, prompts=new_prompts)
        if iterative_num == 0:
            for idx, output in enumerate(outputs):
                all_request_id[idx] = output.request_id

        left_idx = []
        left_prompts = []

        for index, (prompt, output, prompt_idx) in tqdm(enumerate(zip(new_prompts, outputs, idx_list))):
            text = output.outputs[0].text
            token_ids = list(output.outputs[0].token_ids)
            action_mask = [1] * len(token_ids)
            all_stop_reason[prompt_idx] = output.outputs[0].stop_reason
            all_finish_reason[prompt_idx] = output.outputs[0].finish_reason
            
            if output.outputs[0].stop_reason in ['```python']:
                new_sampling_params = sampling_params_list[prompt_idx]
                new_sampling_params.stop += ['```', '```\n', '```\n\n']
                new_sampling_params.stop = list(set(new_sampling_params.stop))

                if '```python' in new_sampling_params.stop:
                    new_sampling_params.stop.remove('```python')
                
                left_idx.append(prompt_idx)
                left_prompts.append(prompt+text)
                all_text[prompt_idx] += text

                if DEBUG_FLAG == 'yes':
                    logger.info({
                        'STAGE': 'code-gen',
                        'prompt': prompt,
                        'output': text,
                        'params': new_sampling_params
                    })
                
            elif output.outputs[0].stop_reason in ['```', '```\n', '```\n\n']:
                new_sampling_params = sampling_params_list[prompt_idx]
                new_sampling_params.stop += ['```python']
                new_sampling_params.stop = list(set(new_sampling_params.stop))
                for stop_token in ['```', '```\n', '```\n\n']:
                    if stop_token in new_sampling_params.stop:
                        new_sampling_params.stop.remove(stop_token)

                code_text = re.findall(code_pattern, f"```python\n{text}")
                
                if DEBUG_FLAG == 'yes':
                    logger.info({
                        'STAGE': 'detect-code-exec',
                        'code_text': code_text,
                        'output': text,
                        'params': new_sampling_params
                    })
                
                if code_text:
                    code_text = code_text[0]
                    code4exec = extract_code(code_text)

                    if DEBUG_FLAG == 'yes':
                        logger.info({
                            'STAGE': 'code-exec',
                            'code_text': code4exec,
                            'output': text,
                            'params': new_sampling_params
                        })
                    
                    if code4exec:

                        uuid_str = id2uuid[prompt_idx]

                        if not COMPILE_SERVER:
                            try:
                                result = run_code(code4exec)
                            except Exception as e:
                                result = str(e)
                        else:
                            try:
                                # result = remote_compile(code4exec)
                                _, result = remote_compile(code4exec, uuid_str, try_max_times=MAX_RETRIES)
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
                                'params': new_sampling_params
                            })

                        code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                        if len(code_output_ids) > 512:
                            code_output = """The output of the code is too long, please check your code."""
                            code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                        
                        token_ids.extend(code_output_ids)
                        action_mask.extend([0]*len(code_output_ids))
                        all_env_exec_times[prompt_idx] += 1
                    else:
                        code_output = ''

                    left_idx.append(prompt_idx)
                    left_prompts.append(prompt+text+code_output)
                    all_text[prompt_idx] += (text+code_output)
                else:
                    is_all_terminated[prompt_idx] = True
                    all_text[prompt_idx] += text
            else:
                is_all_terminated[prompt_idx] = True
                all_text[prompt_idx] += text

            output_token_ids[prompt_idx].extend(token_ids)
            action_masks[prompt_idx].extend(action_mask)

        is_terminated = sum(is_all_terminated) == len(is_all_terminated)
        new_prompts = left_prompts
        idx_list = left_idx

        assert len(new_prompts) == len(idx_list)

        if iterative_num >= ITERA_NUM:
            break
        iterative_num += 1
    
    outputs = []
    for (output_token_id, action_mask, 
        output_text, stop_reason, 
        finish_reason, prompt_token_id, 
        request_id, env_exec_times) in zip(output_token_ids, 
                            action_masks, all_text, 
                            all_stop_reason, all_finish_reason,
                            all_prompt_token_ids, all_request_id, all_env_exec_times):
        assert len(output_token_id) == len(action_mask)
        tmp = GenerateOutput(
            outputs=[Output(
                token_ids=output_token_id,
                action_mask=action_mask+[1] if tokenizer.eos_token_id not in output_token_id else action_mask,
                text=output_text,
                stop_reason=stop_reason,
                finish_reason=finish_reason,
                env_exec_times=env_exec_times
            )],
            prompt_token_ids=prompt_token_id,
            request_id=request_id
        )
        outputs.append(tmp)
    del sampling_params_list, idx_list, new_prompts
    return outputs
