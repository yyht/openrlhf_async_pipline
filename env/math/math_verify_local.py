

import asyncio
import math_verify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import asyncio, random
from openrlhf.async_pipline.show_timer import Timer
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


semaphore = asyncio.Semaphore(100)  # 限制最多5个并发

async def math_verify_async(prompt, response, label, ouput_key):
    resp_ans = extract_answer(response)
    async with semaphore:  # 限制并发数量
        async with Timer("##ASYNC MATH-VERIFY##"):
            if resp_ans:
                try:
                    score = await asyncio.wait_for(
                        asyncio.to_thread(
                            math_verify_reward_function,
                            f"\\boxed{{{resp_ans}}}",
                            label['gold_ans']
                        ),
                        timeout=10.0
                    )
                except:
                    score = 0.0
            else:
                score = 0.0
    return score