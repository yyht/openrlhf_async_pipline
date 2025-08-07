
import sys, os, asyncio, re
sys.path.append(os.getenv('OPENRLHF_PATH', 'YOUR_PATH'))

from env.common.common_score import COMMON_METRIC
from env.common.zero_template_format_score import zero_pattern, zero_format_pattern
from env.synlogic.synlogic_utils import synlogic_score_fn
from openrlhf.async_pipline.show_timer import Timer
from env.math.extract_code import code_pattern, output_pattern
from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained(os.getenv('PRETRAIN', 'YOUR_PATH'))
import re
from env.common.universal_output import UNIVERSAL_FORMAT

CONDITIONAL_TOOL_REWARD = float(os.getenv('CONDITIONAL_TOOL_REWARD', '0'))
QUALITY_REWARD = float(os.getenv('QUALITY_REWARD', '0'))
OVER_LONG_REWARD = float(os.getenv('OVER_LONG_REWARD', '0'))
async def synlogic_score(prompt, response, label, 
        finish_reason, pad_token, **kwargs):
    max_tokens = kwargs.get('max_tokens', 8192)
    max_cache = 512

    default = UNIVERSAL_FORMAT.copy()

    # remove pad-token from response
    response = response.replace(pad_token, '')
    default['length_rewards'] = [len(tokenizer(response)['input_ids'])]

    is_valid, final_reward = await COMMON_METRIC['think_answer_pattern'](prompt, response, use_format_reward=OVER_LONG_REWARD)
    overlong_score = await COMMON_METRIC['soft_overlong_punishment'](
                                default['length_rewards'][0], 
                                max_tokens,
                                max_cache)
    default['overlong_punishment'] = [overlong_score]
    
    if OVER_LONG_REWARD > 0:
        final_reward += overlong_score

    stop_tokens = kwargs.get('stop_tokens', [])
    is_stop = False
    for token in stop_tokens:
        if token in response[-20:]:
            is_stop = True
            break
    if not is_stop and finish_reason not in  ['length']:
        finish_reason = 'truncated'
        logger.info({
            'INFO': '##truncated##',
            'VALUE': f"stop_tokens: {stop_tokens}, suffix: {response[-20:]}"
        })

    if is_valid and finish_reason == 'stop':
        resp = response.split('</think>')[-1]
        async with Timer("##ASYNC SCORE TIMING WITH SYN-LOGIC##"):
            try:
                results = await asyncio.gather(
                    COMMON_METRIC['code_rewards'](response),
                    COMMON_METRIC['repeatness'](response),
                    COMMON_METRIC['pattern_count'](response),
                    COMMON_METRIC['repetition_penalty']([response], 3, -1.0),
                    synlogic_score_fn(prompt, resp, label),
                    return_exceptions=True  # 默认False，有异常会立即抛出
                )
            except:
                results = None
    else:
        async with Timer("##ASYNC SCORE TIMING SYN-LOGIC##"):
            try:
                results = await asyncio.gather(
                    COMMON_METRIC['code_rewards'](response),
                    COMMON_METRIC['repeatness'](response),
                    COMMON_METRIC['pattern_count'](response),
                    COMMON_METRIC['repetition_penalty']([response], 3, -1.0),
                    return_exceptions=True  # 默认False，有异常会立即抛出
                )
            except:
                results = None
    
    if is_valid and finish_reason == 'stop':
        if results is not None:
            code_rewards = results[0]
            repeatness_score = results[1]
            pattern_count = results[2]
            repetition_penalty = results[3]
            rule_reward = results[4]

            if rule_reward is None:
                default['rule_eval_fails'] = [1.0]

            code_count = float(len(re.findall(code_pattern, response)))
            output_count = float(len(re.findall(output_pattern, response)))

            if rule_reward is not None:
                rule_rewards = rule_reward
            else:
                rule_rewards = 0
            # print(rule_rewards, '===rule_rewards===', label)
            answer_rewards = float(rule_rewards)
            if QUALITY_REWARD > 0:
                if answer_rewards > 0:
                    if repeatness_score > label.get('repeatness_threshold', 0.1):
                        final_reward -= label.get('repeatness_penalty', 0.1)

            # if embedding is not None:
            #     default['embedding'] = [embedding]

            default['answer_rewards'] = [float(answer_rewards)]
            default['rewards'] = [float(answer_rewards) + float(final_reward)]
            default['reflection'] = [float(pattern_count['reflection'])]
            default['new_idea'] = [float(pattern_count['new_idea'])]
            default['code_rewards'] = [float(code_rewards)]
            default['rule_rewards'] = [float(rule_rewards)]
            default['repetition_penalty'] = [float(-repetition_penalty[0])]

            if answer_rewards > 0 and code_count > 0 and output_count > 0:
                default['rewards'][0] += float(CONDITIONAL_TOOL_REWARD)

            if code_rewards > 0 and  answer_rewards > 0:
                default['code_correct'] = [1.0]
            default['repeatness'] = [float(repeatness_score)]
            if is_valid:
                default['format_answer'] = [0.0]
            else:
                default['format_answer'] = [1.0]
            
            default['finish_reason'] = [float(finish_reason=='stop')]
            default['truncated'] = [float(finish_reason=='length')]
            default['other'] = [float(finish_reason not in ['length', 'stop'])]
            default['code_count'] = [float(response.count('```python'))]
            return default
        else:
            default['rule_eval_fails'] = [1.0]
            default['model_eval_fails'] = [1.0]
            return default
    else:
        if results is not None:
            code_rewards = results[0]
            repeatness_score = results[1]
            pattern_count = results[2]
            repetition_penalty = results[3]
            # embedding = results[4]

            answer_rewards = 0.0
            model_rewards = 0.0
            rule_rewards = 0.0

            # if embedding is not None:
            #     default['embedding'] = [embedding]

            # if QUALITY_REWARD > 0:
            #     if repeatness_score > label.get('repeatness_threshold', 0.1):
            #         final_reward -= label.get('repeatness_penalty', 0.1)
            
            default['repeatness'] = [float(repeatness_score)]
            default['answer_rewards'] = [float(answer_rewards)]
            default['rewards'] = [float(answer_rewards) + float(final_reward)]
            default['reflection'] = [float(pattern_count['reflection'])]
            default['new_idea'] = [float(pattern_count['new_idea'])]
            default['code_rewards'] = [float(code_rewards)]
            default['model_rewards'] = [float(model_rewards)]
            default['rule_rewards'] = [float(rule_rewards)]
            default['format_answer'] = [1.0]
            default['repetition_penalty'] = [float(-repetition_penalty[0])]
            
            default['finish_reason'] = [float(finish_reason=='stop')]
            default['truncated'] = [float(finish_reason=='length')]
            default['other'] = [float(finish_reason not in ['length', 'stop'])]
            default['code_count'] = [float(response.count('```python'))]
            return default
        else:
            default['rule_eval_fails'] = [1.0]
            default['model_eval_fails'] = [1.0]
            return default
    



    