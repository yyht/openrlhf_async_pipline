import sys, os, asyncio, re
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

from env.common.common_score import COMMON_METRIC
from env.common.zero_template_format_score import zero_pattern, zero_format_pattern, code_pattern_score
from env.math.extract_code import code_pattern, output_pattern
from env.math.math_verify_server import math_verify_server
from env.math.xverify_server import xverify_server
from env.common.embedding import embedding_server
from openrlhf.async_pipline.show_timer import Timer
from env.common.universal_output import UNIVERSAL_FORMAT

import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONDITIONAL_TOOL_REWARD = float(os.getenv('CONDITIONAL_TOOL_REWARD', '0'))
PATTERN_REWARD = float(os.getenv('PATTERN_REWARD', '0'))
QUALITY_REWARD = float(os.getenv('QUALITY_REWARD', '0'))
CODE_EXECTION_REWARD = float(os.getenv('CODE_EXECTION_REWARD', '0'))
OVER_LONG_REWARD = float(os.getenv('OVER_LONG_REWARD', '0'))
USE_FORMAT_REWARD = os.getenv('USE_FORMAT_REWARD', 'no')
USE_SHORTCUT_REWARD = float(os.getenv('USE_SHORTCUT_REWARD', '0'))

logger.info({
    'INFO': '##CONDITIONAL_TOOL_REWARD##',
    'VALUE': CONDITIONAL_TOOL_REWARD,
    'PATTERN_VALUE': PATTERN_REWARD,
    'QUALITY_REWARD': QUALITY_REWARD,
    'OVER_LONG_REWARD': OVER_LONG_REWARD,
    'CODE_EXECTION_REWARD': CODE_EXECTION_REWARD,
    'USE_FORMAT_REWARD': USE_FORMAT_REWARD,
    'USE_SHORTCUT_REWARD': USE_SHORTCUT_REWARD
})

from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained(os.getenv('PRETRAIN', '/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-Coder-7B-local/'))

async def math_score(prompt, response, label, 
        finish_reason, pad_token, **kwargs):

    # remove pad-token from response
    response = response.replace(pad_token, '')
    code_exection_nums = kwargs.get('code_exection_nums', [0.0])
    code_exection_errors = kwargs.get('code_exection_errors', [0.0])

    max_tokens = kwargs.get('max_tokens', 8192)
    max_cache = 512

    code_exection_all_failed = float(sum(code_exection_errors)==sum(code_exection_nums))

    default = UNIVERSAL_FORMAT.copy()
    default['code_exection_errors'] = [float(sum(code_exection_errors)/(1e-10+sum(code_exection_nums)))]
    default['code_exection_all_failed'] = [code_exection_all_failed]
    if code_exection_errors:
        default['code_exection_last_failed'] = [float(code_exection_errors[-1])]
    else:
        default['code_exection_last_failed'] = [0.0]
    default['code_exection_times'] = [float(sum(code_exection_nums))]

    default['length_rewards'] = [len(tokenizer(response)['input_ids'])]

    is_valid, final_reward = await COMMON_METRIC[label['template']](prompt, response, use_format_reward=USE_FORMAT_REWARD)
    overlong_score = await COMMON_METRIC['soft_overlong_punishment'](
                                default['length_rewards'][0], 
                                max_tokens,
                                max_cache)
    default['overlong_punishment'] = [overlong_score]

    code_exection_penalty = 0.0
    if sum(code_exection_errors) / (sum(code_exection_nums)+1e-10) > 0.9 and sum(code_exection_nums) > 1:
        code_exection_penalty = -0.1

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
        async with Timer("##ASYNC SCORE TI MING WITH MATH##"):
            try:
                results = await asyncio.gather(
                    COMMON_METRIC['code_rewards'](response),
                    COMMON_METRIC['repeatness'](response),
                    COMMON_METRIC['pattern_count'](response),
                    COMMON_METRIC['repetition_penalty']([response], 3, -1.0),
                    math_verify_server(prompt, resp, label, 'score'),
                    xverify_server(prompt, resp, label, max_retries=3, temperature=0.0, max_tokens=8192),
                    # embedding_server(prompt, response, label, max_retries=3),
                    return_exceptions=True  # 默认False，有异常会立即抛出
                )
            except:
                results = None
    else:
        async with Timer("##ASYNC SCORE TIMING WITH MATH##"):
            try:
                results = await asyncio.gather(
                    COMMON_METRIC['code_rewards'](response),
                    COMMON_METRIC['repeatness'](response),
                    COMMON_METRIC['pattern_count'](response),
                    COMMON_METRIC['repetition_penalty']([response], 3, -1.0),
                    # embedding_server(prompt, response, label, max_retries=3),
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
            model_reward = results[5]
            # embedding = results[6]

            if rule_reward is None:
                default['rule_eval_fails'] = [1.0]
            if model_reward is None:
                default['model_eval_fails'] = [1.0]

            code_count = float(len(re.findall(code_pattern, response)))
            output_count = float(len(re.findall(output_pattern, response)))

            if rule_reward is not None:
                rule_rewards = rule_reward
            else:
                rule_rewards = 0
            if model_reward is not None:
                model_rewards = model_reward
            else:
                model_rewards = 0
            
            answer_rewards = max([float(rule_rewards), 
                                float(model_rewards)])
            if answer_rewards > 0.1:
                if CODE_EXECTION_REWARD > 0:
                    final_reward += code_exection_penalty
            if USE_SHORTCUT_REWARD > 0:
                if answer_rewards > 0.1:
                    # if code_exection_errors:
                    #     if code_exection_errors[-1] == 1:
                    #         answer_rewards = 0.0
                    if repeatness_score > 0.1:
                        answer_rewards = 0.0
            # if answer_rewards > 0.1:
            #     if repeatness_score > 0.05:
            #         answer_rewards = 0

            # if answer_rewards > 0:
            #     if CODE_EXECTION_REWARD > 0:
            #         code_score = code_pattern_score(response)
            #         final_reward += code_score

            # if embedding is not None:
            #     default['embedding'] = [embedding]

            default['answer_rewards'] = [float(answer_rewards)]
            default['rewards'] = [float(answer_rewards) + float(final_reward)]
            default['reflection'] = [float(pattern_count['reflection'])]
            default['new_idea'] = [float(pattern_count['new_idea'])]
            default['code_rewards'] = [float(code_rewards)]
            default['model_rewards'] = [float(model_rewards)]
            default['rule_rewards'] = [float(rule_rewards)]
            default['repetition_penalty'] = [float(-repetition_penalty[0])]

            if answer_rewards > 0 and code_count > 0 and output_count > 0 and code_exection_all_failed < 1:
                default['rewards'][0] += float(CONDITIONAL_TOOL_REWARD)

            if answer_rewards > 0:
                if float(pattern_count['new_idea']) > 0.5:
                    default['rewards'][0] += float(PATTERN_REWARD)

            if code_rewards > 0 and  answer_rewards > 0:
                default['code_correct'] = [1.0]
            default['repeatness'] = [float(repeatness_score)]
            if is_valid:
                default['format_answer'] = [0.0]
            else:
                default['format_answer'] = [1.0]
            
            boxed_list = re.findall(zero_pattern, response)
            format_list = re.findall(zero_format_pattern, response)
            if len(boxed_list) >= 2:
                default['more_boxed'] = [1.0]
            if len(boxed_list) == 0:
                default['no_boxed'] = [1.0]
            
            default['finish_reason'] = [float(finish_reason=='stop')]
            default['truncated'] = [float(finish_reason=='length')]
            default['other'] = [float(finish_reason not in ['length', 'stop'])]
            default['code_count'] = [float(code_count)]
            default['output_count'] = [float(output_count)]
            default['partial_code_count'] = [float(response.count('```python\n'))]
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

            code_count = float(len(re.findall(code_pattern, response)))
            output_count = float(len(re.findall(output_pattern, response)))

            # if embedding is not None:
            #     default['embedding'] = [embedding]
            # if QUALITY_REWARD > 0:
            #     if repeatness_score > label.get('repeatness_threshold', 0.05):
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

            boxed_list = re.findall(zero_pattern, response)
            format_list = re.findall(zero_format_pattern, response)
            if len(boxed_list) >= 2:
                default['more_boxed'] = [1.0]
            if len(boxed_list) == 0:
                default['no_boxed'] = [1.0]
            
            default['finish_reason'] = [float(finish_reason=='stop')]
            default['truncated'] = [float(finish_reason=='length')]
            default['other'] = [float(finish_reason not in ['length', 'stop'])]
            default['code_count'] = [float(code_count)]
            default['output_count'] = [float(output_count)]
            default['partial_code_count'] = [float(response.count('```python\n'))]
            return default
        else:
            default['rule_eval_fails'] = [1.0]
            default['model_eval_fails'] = [1.0]
            return default
    



    
    


