# -*- coding: utf-8 -*-

import numpy as np
import numpy as np
import transformers
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoTokenizer

import json
import ssl
from typing import AsyncGenerator
# import torch
import argparse
import transformers
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from func_timeout import func_set_timeout
import argparse
import json
import pdb
import jsonlines
from timeout_decorator import timeout
# import torch
import os
import logging
from tqdm import tqdm

import gc

# initialize + freeze gc before starting the server  
gc.collect(2)  

_, gen1, gen2 = gc.get_threshold()  
gc.set_threshold(50000, gen1, gen2)

from auto_gc_timer import func
func()
 
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import os, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TIMEOUT_KEEP_ALIVE = 5  # seconds.

TEMP_FOLDER = os.getenv('TEMP_FOLDER', '')
if TEMP_FOLDER:
    fwobj = open(TEMP_FOLDER, 'a+')
    logger.info('####TEMP_FOLDER####')
    logger.info(TEMP_FOLDER)

app = FastAPI()

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from functools import lru_cache
@lru_cache
def math_verify_reward_function(solution_str, ground_truth):
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0

async def math_verify_reward_function_async(solution_str, ground_truth):
    return math_verify_reward_function(solution_str, ground_truth)

# from qwen_rm.sync2async import make_async
# from concurrent.futures.thread import ThreadPoolExecutor
# math_verify_executor = ThreadPoolExecutor(max_workers=1)

# math_verify_reward_function_async = make_async(math_verify_reward_function, executor=math_verify_executor)
    
@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/math_verify")
async def math_verify_server(request: Request) -> Response:
    request_dict = await request.json()
    response = request_dict.pop("response") 
    prompt = request_dict.pop("prompt") 
    gold_ans = request_dict.pop("gold_ans")
    
    try:
        score = await math_verify_reward_function_async(response, gold_ans)
    except Exception as e:
        score = 0.0
        logger.info({
            'ERROR_INFO': '######ERROR######',
            'ERROR': str(e),
            'gold_ans': gold_ans,
            'response': response
        })
    
    output_dict = {
        'score': score
    }
    
    return JSONResponse(output_dict)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    
    parser.add_argument("--log-level", type=str, default="debug")
    args = parser.parse_args()

    app.root_path = args.root_path
    # uvicorn.run(app,
    #             host=args.host,
    #             port=args.port,
    #             log_level=args.log_level,
    #             timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    #             ssl_keyfile=args.ssl_keyfile,
    #             ssl_certfile=args.ssl_certfile,
    #             ssl_ca_certs=args.ssl_ca_certs,
    #             ssl_cert_reqs=args.ssl_cert_reqs)