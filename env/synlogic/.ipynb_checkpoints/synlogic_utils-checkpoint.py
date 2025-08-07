

import sys, os, asyncio, re
sys.path.append(os.getenv('SYNLOGIC_PATH', 'YOUR_PATH'))
sys.path.append(os.getenv('OPENRLHF_PATH', 'YOUR_PATH'))

from task2verifier import verifier_classes
from openrlhf.async_pipline.utils import make_async
from concurrent.futures.thread import ThreadPoolExecutor
common_executor = ThreadPoolExecutor(max_workers=1)

verifier_async_dict = {}
for key in verifier_classes:
    verifier_async_dict[key] = make_async(verifier_classes[key]().verify, executor=common_executor)

import asyncio
from functools import wraps

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"函数 {func.__name__} 执行超时，超时时间: {seconds} 秒")
        return wrapper
    return decorator

import json
from pydantic import BaseModel
from typing import Optional, Any, List, Dict, Tuple
class Data(BaseModel):
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """
    question: str
    answer: Any
    difficulty: int = 1
    metadata: dict = None

@timeout(10)
async def synlogic_score_fn(prompt, response, label, **kwargs):
    
    data = Data(
        question=prompt,
        answer=label['answer'],
        difficulty=label['difficulty'],
        metadata=label['metadata']
    )

    try:
        result = await verifier_async_dict[label['data_source']](data, response)
        result = float(result)
    except:
        result = None
    return result
    
