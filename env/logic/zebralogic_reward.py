

import sys, os, asyncio, re
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

from env.common.utils import make_async
from concurrent.futures.thread import ThreadPoolExecutor
from env.logic.zebralogic_score import compute_score
from env.common.common_score import COMMON_METRIC
from openrlhf.async_pipline.show_timer import Timer

common_executor = ThreadPoolExecutor(max_workers=1)

compute_score_async = make_async(
    compute_score, executor=common_executor)

from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained(os.getenv('PRETRAIN', '/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-Coder-7B-local/'))
from env.common.zero_template_format_score import zero_pattern, zero_format_pattern
async def zebralogic_score(prompt, response, label, 
        finish_reason, pad_token, **kwargs):
    default = {
        'rewards': [0.0],
        'length_rewards': [0.0],
        'answer_rewards': [0.0],
        'reflection': [0.0],
        'new_idea': [0.0],
        'code_rewards': [0.0],
        'code_correct': [0.0],
        'rule_eval_fails': [0.0],
        'model_eval_fails': [0.0],
        'more_boxed': [0.0],
        'no_boxed': [0.0],
        "format_answer": [0.0],
        "finish_reason": [0.0],
        'truncated': [0.0],
        'other': [0.0],
        "repeatness": [0.0],
        "code_count": [0.0],
        'model_rewards': [0.0],
        'rule_rewards': [0.0],
        'repetition_penalty': [0.0],
        'diversity_score': [0.0]
    }

    # remove pad-token from response
    response = response.replace(pad_token, '')
    default['length_rewards'] = [len(tokenizer(response)['input_ids'])]

    async with Timer("##ASYNC SCORE TIMING WITH ZEBRA-LOGIC##"):
        try:
            results = await asyncio.gather(
                COMMON_METRIC['code_rewards'](response),
                COMMON_METRIC['repeatness'](response),
                COMMON_METRIC['pattern_count'](response),
                COMMON_METRIC['repetition_penalty']([response], 3, -1.0),
                compute_score_async(response, label['gold_ans'], format_reward=1.0, answer_reward=1.0, tokenizer=tokenizer),
                return_exceptions=True  # 默认False，有异常会立即抛出
            )
        except:
            results = None
    
    if results is not None:
        code_rewards = results[0]
        repeatness_score = results[1]
        pattern_count = results[2]
        repetition_penalty = results[3]
        logic_scores = results[4]

        logic_score = logic_scores[0]
        logic_gold_score = logic_scores[1]
        logic_answer_score = logic_scores[2]

        # if repeatness_score > 0.05 or finish_reason != 'stop' or repetition_penalty > 0.5:
        #     logic_gold_score = 0.0
        #     logic_score -= logic_answer_score

        default['answer_rewards'] = [logic_gold_score]
        default['rewards'] = [logic_score]
        default['repetition_penalty'] = [float(-repetition_penalty[0])]

        default['reflection'] = [pattern_count['reflection']]
        default['new_idea'] = [pattern_count['new_idea']]
        default['code_rewards'] = [float(code_rewards)]

        default['model_rewards'] = [0.0]
        default['rule_rewards'] = [logic_gold_score]
        
        boxed_list = re.findall(zero_pattern, response)
        format_list = re.findall(zero_format_pattern, response)
        if len(boxed_list) >= 2:
            default['more_boxed'] = [1.0]
        if len(boxed_list) == 0:
            default['no_boxed'] = [1.0]

        default['finish_reason'] = [float(finish_reason=='stop')]
        default['truncated'] = [float(finish_reason=='length')]
        default['other'] = [float(finish_reason not in ['length', 'stop'])]
        default['code_count'] = [float(response.count('```python'))]
        return default
    else:
        default['rule_eval_fails'] = [1.0]
        default['model_eval_fails'] = [1.0]
        return default



