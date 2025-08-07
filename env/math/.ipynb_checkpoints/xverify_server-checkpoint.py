
import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', 'YOUR_PATH'))

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

XVERIFY_MATH_MODEL_SERVER_PORT = os.getenv('XVERIFY_MATH_MODEL_SERVER_PORT', '')
NGINX_IP_FILE = os.getenv('NGINX_IP_FILE', '')

GLOBAL_IP_LIST = []
with open(NGINX_IP_FILE) as frobj:
    for line in frobj:
        GLOBAL_IP_LIST.append(line.strip())

import openai
XVERIFY_MATH_MODEL_SERVER = f"{GLOBAL_IP_LIST[0]}:{XVERIFY_MATH_MODEL_SERVER_PORT}/v1"
client = openai.Client(
            base_url=XVERIFY_MATH_MODEL_SERVER, 
            api_key="EMPTY")

logger.info({
    'INFO': '##XVERIFY_MATH_MODEL_SERVER##',
    'VALUE': NGINX_IP_FILE,
    'PORT': XVERIFY_MATH_MODEL_SERVER_PORT,
    'SERVER': XVERIFY_MATH_MODEL_SERVER
})

import re
strict_pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)

PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''

import os
MATH_FAIL_SCORE = float(os.getenv('MATH_FAIL_SCORE', -10000))

LABEL_MAPPING = {
    'Incorrect': 0.0,
    'Correct': 1.0,
    'NONE': float(MATH_FAIL_SCORE)
}

async def xverify_server(prompt, response, label, max_retries=10, **generation_kwargs):

    global client
    global XVERIFY_MATH_MODEL_SERVER

    template = label.get('template', 'ZERO_TIR')
    gold_ans = label['gold_ans']
    uuids = label['uuid']
    max_retries = 3

    ip_list = []
    with open(NGINX_IP_FILE) as frobj:
        for line in frobj:
            ip_list.append(line.strip())
    
    NEW_XVERIFY_MATH_MODEL_SERVER = f"{ip_list[0]}:{XVERIFY_MATH_MODEL_SERVER_PORT}/v1"
    if NEW_XVERIFY_MATH_MODEL_SERVER != XVERIFY_MATH_MODEL_SERVER:
        client.base_url = NEW_XVERIFY_MATH_MODEL_SERVER
        XVERIFY_MATH_MODEL_SERVER = NEW_XVERIFY_MATH_MODEL_SERVER

    if template in ['ZERO_TIR', 'ZERO_V2']:
        matches = re.findall(strict_pattern, response)
        if len(matches) > 0:
            boxed_answer = matches[-1]
        else:
            boxed_answer = ''
    else:
        boxed_answer = ''
    
    if boxed_answer and gold_ans:
        for attempt in range(1, generation_kwargs.get('max_retries', 1) + 1):
            try:
                async with Timer("##ASYNC PROCESS-XVERIFY##"):
                    async with asyncio.Semaphore(generation_kwargs.get('max_concurrent', 2)):  # 信号量控制并发
                        # 构造请求负载
                        payload = PROMPT.format_map({
                            'question': prompt,
                            'output': boxed_answer,
                            'answer':gold_ans
                        })
                        response = client.chat.completions.create(
                            model="default",
                            messages=[
                                {"role": "system", "content": 'You are a helpful AI assistant.'},
                                {"role": "user", "content": payload},
                            ],
                            temperature=generation_kwargs.get('temperature', 0.0),
                            max_tokens=generation_kwargs.get('max_tokens', 8192),
                            timeout=5
                        )
                        content = response.choices[0].message.content
                        assert content in ['Correct', 'Incorrect', 'NONE']
                        return LABEL_MAPPING[content]
            except Exception as e:
                logger.warning(f"[{uuids}] Attempt {attempt} failed: {e} of {XVERIFY_MATH_MODEL_SERVER}")
                if attempt == max_retries:
                    logger.error(f"[{uuids}] Failed after {max_retries} attempts  of {XVERIFY_MATH_MODEL_SERVER}.")
                    return None
                await asyncio.sleep(attempt * 1.2)  # 指数退避
    else:
        return  None

        
