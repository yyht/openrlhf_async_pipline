

import os, sys
import ray
import os, copy, httpx
import uuid, time
import asyncio

import sys, os

import requests
from env.math.code_exec import run_code
from env.math.extract_code import extract_code
from env.math.code_merge import multiturn_code_merge
from collections import OrderedDict
import re, uuid
from vllm import SamplingParams
import random

import logging
import asyncio, httpx
import aiohttp
from aiohttp import ClientTimeout
import os, ray

MAX_RETRIES = int(os.getenv("MAX_COMPILE_RETRIES", 3))
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

COMPILE_SERVER_PORT = os.getenv('COMPILE_SERVER_PORT', '')
DEBUG_FLAG = os.getenv('DEBUG_FLAG', '')
NGINX_IP_FILE = os.getenv('NGINX_IP_FILE', '')

logger.info({
    'INFO': 'NGINX_IP_FILE',
    "VALUE": NGINX_IP_FILE,
    'ENV_ITER_NUM': ENV_ITER_NUM,
    "COMPILE_SERVER_PORT": COMPILE_SERVER_PORT
})

from typing import Generic, TypeVar, Union, NamedTuple
from typing import Optional, Any, List, Dict, Tuple
from openrlhf.async_pipline.base_request import process_single_request
from openrlhf.utils.remote_rm_utils import request_api_wrapper
from openrlhf.async_pipline.show_timer import Timer
from openrlhf.async_pipline.rollout_output_base import Output, GenerateOutput
from openrlhf.async_pipline.utils import make_async
from openrlhf.env.reward_config import REWARD_CONFIG

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 8))  # 最大并发数，可根据需要调整
MULTITURN_CODE_MERGE = os.getenv("MULTITURN_CODE_MERGE", 'NONE')
CODE_CONCURRENT = int(os.getenv("CODE_CONCURRENT", 2))  # 最大并发数，可根据需要调整

logger.info({
    'INFO': '##CODE-INFO##',
    "MAX_CONCURRENT": MAX_CONCURRENT,
    'MULTITURN_CODE_MERGE': MULTITURN_CODE_MERGE,
    "CODE_CONCURRENT": CODE_CONCURRENT
})

from env.math.remote_utils import remote_compile
from env.math.sandbox_utils import remote_compile as sandbox_compile


async def math_tir_generate_async(url, headers, idx, request, tokenizer, **kwargs):

    ip_list = []
    with open(NGINX_IP_FILE) as frobj:
        for line in frobj:
            ip_list.append(line.strip())

    COMPILE_SERVER = f"{ip_list[0]}:{COMPILE_SERVER_PORT}"

    prompts = request.prompts

    assert len(request.prompts) == 1
    
    output_token_ids = []
    action_masks = []
    output_text = ''
    stop_reason = ''
    finish_reason = ''
    request_id = ''
    env_exec_times = 0

    if request.enable_vllm_is_correction:
        rollout_log_probs = []
    else:
        rollout_log_probs = None

    prompt_token_ids = request.prompt_token_ids
    
    is_terminated = False
    max_tokens = request.max_tokens

    turn = 0
    new_request = copy.copy(request)

    iterative_num = 0
    ITERA_NUM = 2 * ENV_ITER_NUM # state_num * iter_num

    new_prompt = copy.copy(prompts[0])
    all_request_id = request.uuids

    # initial-code_snippets:
    code_snippets = []

    if new_request.iterative_num == 0:
        new_request.stop += ['```python']
        new_request.stop = list(set(new_request.stop))

    if new_request.iterative_num > 0:
        iterative_num = request.iterative_num
        output_text = request.output_text
        output_token_ids = request.output_token_ids
        action_masks = [1]*len(output_token_ids)
        env_exec_times = request.env_exec_times
        logger.info("##Recontinue for env-interactive##")
        code_snippets = request.code_snippets
        rollout_log_probs = request.rollout_log_probs

    code_exection_errors = []
    code_exection_nums = []

    while not is_terminated:

        logger.info({
            'ITER-INFO': iterative_num,
            'LEN_OF_OUTPUT_TOKEN': len(output_token_ids)
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
            finish_reason = 'length'
            logger.info({
                'INFO': '##LENGTH-STOP##',
                'OUTPUT_VALUE': len(output_token_ids),
            })
            break

        new_request.prompts = [new_prompt+output_text]
        new_request.prompt_token_ids = prompt_token_ids + output_token_ids
        if len(new_request.prompt_token_ids) >= new_request.max_length:
            finish_reason = 'length'
            logger.info({
                'INFO': '##LENGTH-STOP##',
                'PROMPT_VALUE': len(prompt_token_ids),
                'OUTPUT_VALUE': len(output_token_ids),
            })
            break
        
        idx, output = await process_single_request(url, headers, idx, new_request, **kwargs)
        
        if output is None:
            if iterative_num == 0:
                logger.info("##OUTPUT IS NONE##")
                return output
            else:
                logger.info(f"##OUTPUT IS NONE AND iterative_num: {iterative_num}##")
                new_request.prompts = [new_prompt]
                new_request.output_text = output_text
                new_request.output_token_ids = output_token_ids
                new_request.iterative_num = iterative_num
                new_request.env_exec_times = env_exec_times
                new_request.max_tokens = max_tokens
                new_request.code_snippets = code_snippets
                new_request.rollout_log_probs = rollout_log_probs
                return new_request

        text = output[0]['outputs'][0]['text']
        token_ids = list(output[0]['outputs'][0]['token_ids'])
        action_mask = [1] * len(token_ids)
        stop_reason = output[0]['outputs'][0]['stop_reason']
        finish_reason = output[0]['outputs'][0]['finish_reason']

        # Calculate rollout log probs
        if request.enable_vllm_is_correction:
            # action tokens logprobs
            current_rollout_log_probs = []
            for i, token_id in enumerate(token_ids):
                log_prob_list = output[0]['outputs'][0]['log_probs']
                current_rollout_log_probs.append(log_prob_list[i][token_id].logprob)
        else:
            current_rollout_log_probs = None
        
        if output[0]['outputs'][0]['stop_reason'] in ['```python']:
            iterative_num += 1
            new_request.stop += ['```', '```\n', '```\n\n']
            new_request.stop = list(set(new_request.stop))

            if '```python' in new_request.stop:
                new_request.stop.remove('```python')
            
            # output_text += text
            next_turn_output_text = text

            if DEBUG_FLAG == 'yes':
                logger.info({
                    'STAGE': 'code-gen',
                    'prompt': prompts,
                    'output': text,
                    'params': new_request.stop
                })
            
        elif output[0]['outputs'][0]['stop_reason'] in ['```', '```\n', '```\n\n']:
            iterative_num += 1
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

                    if MULTITURN_CODE_MERGE == 'YES':
                        # merge-muliturn-code
                        multiturn_codes = multiturn_code_merge(code_snippets)
                    else:
                        multiturn_codes = ''
                    
                    try:
                        _, output_dict = await sandbox_compile(code4exec, all_request_id, try_max_times=MAX_RETRIES, multiturn_codes=multiturn_codes)
                        # each-time, we only keep the latest code4exec
                        output_dict['query'] = code4exec
                    except Exception as e:
                        output_dict = {
                            'query': code4exec,
                            'uuid': all_request_id,
                            'exec_result': "TimeOut Error",
                            'error': 1
                        }
                        logger.info({
                            'INFO': "##UnexpectedException##",
                            "VALUE": f"{e}",
                        })

                    code_result = output_dict['exec_result']
                    if MULTITURN_CODE_MERGE == 'YES':
                        code_snippets.append(output_dict)

                    code_exection_nums.append(1)
                    if output_dict['error'] == 1:
                        code_exection_errors.append(1)
                    else:
                        code_exection_errors.append(0)

                    # be careful about the output pattern, make stop-token be complete
                    code_output_prefix = f"""\n```output\n"""
                    code_output_suffix = f"""\n```\n\n\n"""
                    code_output = f"""{code_output_prefix}{code_result}{code_output_suffix}"""

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

                    if len(code_output_ids) > 256:
                        logger.info({
                            'INFO': f'##LARGE-OUTPUT-EXEC-OUTPUT##{len(code_output_ids)}',
                        })
                        # code_result_ids = tokenizer(code_result, add_special_tokens=False)['input_ids']
                        # truncated_code_result = tokenizer.decode(code_result_ids[:512])
                        overlong_code_result = """The output of the code execution is too long, please check your code."""
                        code_output = f"""{code_output_prefix}{overlong_code_result}{code_output_suffix}"""
                        code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                    
                    token_ids.extend(code_output_ids)
                    action_mask.extend([0]*len(code_output_ids))

                    # Calculate rollout log probs
                    if request.enable_vllm_is_correction:
                        # action tokens logprobs
                        current_rollout_log_probs.extend([0.0] * (len(code_output_ids)))

                    env_exec_times += 1
                else:
                    code_output = ''

                # output_text += (text+code_output)
                next_turn_output_text = (text+code_output)
            else:
                is_terminated = True
                # output_text += text
                next_turn_output_text = text
        else:
            is_terminated = True
            # output_text += text
            next_turn_output_text = text

        # if len(output_token_ids)+len(token_ids) > max_tokens:
        #     finish_reason = 'length'
        #     logger.info({
        #         'INFO': '##LENGTH-STOP##',
        #         'OUTPUT_VALUE': len(output_token_ids),
        #         'NEXT_OUTPUT_VALUE': len(token_ids)
        #     })
        #     break

        output_token_ids.extend(token_ids)
        action_masks.extend(action_mask)

        if request.enable_vllm_is_correction:
            rollout_log_probs.extend(current_rollout_log_probs)

            assert len(token_ids) == len(rollout_log_probs)
        
        output_text += next_turn_output_text

        assert len(output_token_ids) == len(action_masks)


    label = json.loads(new_request.label)

    if kwargs.get('use_reward', True):
        reward_info = await REWARD_CONFIG[label.get('task', 'math').lower()](request.prompts[0], output_text, label, finish_reason, tokenizer.pad_token, 
                                code_exection_nums=code_exection_nums,
                                code_exection_errors=code_exection_errors,
                                max_tokens=max_tokens,
                                stop_tokens=new_request.stop)
    else:
        reward_info = {}

    if not request.enable_vllm_is_correction:
        if tokenizer.eos_token_id not in output_token_ids:
           output_token_ids += [tokenizer.eos_token_id]
           action_masks += [1]

    output = GenerateOutput(
            outputs=[Output(
                # token_ids=output_token_ids+[tokenizer.eos_token_id] if tokenizer.eos_token_id not in output_token_ids else output_token_ids,
                # action_mask=action_masks+[1] if tokenizer.eos_token_id not in output_token_ids else action_masks,
                token_ids=output_token_ids,
                action_mask=action_masks,
                text=output_text,
                stop_reason=stop_reason,
                finish_reason=finish_reason,
                env_exec_times=env_exec_times,
                reward_info=reward_info,
                log_probs=rollout_log_probs
            )],
            prompt_token_ids=prompt_token_ids,
            request_id=all_request_id,
            label=label,
            prompt=request.prompts[0],
            request_rank=request.request_rank
    )
    return (idx, output)
