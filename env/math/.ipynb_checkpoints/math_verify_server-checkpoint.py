
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

MATH_VERIFY_SERVER_PORT = os.getenv('MATH_VERIFY_SERVER_PORT', '')
NGINX_IP_FILE = os.getenv('NGINX_IP_FILE', '')
MATH_SERVER = os.getenv('MATH_SERVER', None)

logger.info({
    'INFO': '##MATH-VERIFY-DERVER##',
    'VALUE': NGINX_IP_FILE,
    'PORT': MATH_VERIFY_SERVER_PORT,
    'MATH_VERIFIER_SERVER': MATH_SERVER
})

def extract_answer(pred_str):
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
        return pred
    else:
        return None

async def math_verify_server(prompt, response, label, ouput_key):

    ip_list = []
    with open(NGINX_IP_FILE) as frobj:
        for line in frobj:
            ip_list.append(line.strip())

    MATH_VERIFY_SERVER = f"{ip_list[0]}:{MATH_VERIFY_SERVER_PORT}"
    http_method = 'math_verify'
    if MATH_SERVER:
        MATH_VERIFY_SERVER = MATH_SERVER
        http_method = 'verify'

    resp_ans = extract_answer(response)
    if resp_ans:
        request = HttpRequest(
            prompt=prompt,
            response=f"\\boxed{{{resp_ans}}}",
            meta=label,
            gold_ans=label['gold_ans'],
            uuid_str=label['uuid'],
            request_timeout=10,
            max_concurrent=2,
            max_retries=3,
            url=MATH_VERIFY_SERVER, # http://{ip}:{port}
            http_method=http_method,
            gold=f"\\boxed{{{label['gold_ans']}}}",
            answer=f"\\boxed{{{resp_ans}}}"
        )

        uuids, response_dict = await process_single_request(request)
        if response_dict is not None:
            return float(response_dict[ouput_key])
        else:
            return 0.0
    else:
        return 0.0

