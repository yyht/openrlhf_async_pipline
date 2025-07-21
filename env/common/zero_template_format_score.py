

import re
zero_pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
zero_format_pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
code_output_pattern = re.compile(r"```python.*?```output", re.DOTALL)
code_pattern = re.compile(r"```python\n.*?```", re.DOTALL)
from transformers import AutoTokenizer
import os
import logging, json
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tokenizer = AutoTokenizer.from_pretrained(os.getenv('PRETRAIN', '/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B-local/'))

USE_TIR_FORMAT_LOOSE = float(os.getenv('USE_TIR_FORMAT_LOOSE', '0'))

logger.info({
    'INFO': '##FORMAT-ENV##',
    'VALUE': USE_TIR_FORMAT_LOOSE
})

from functools import lru_cache
@lru_cache
def zero_template_format_score_fn(prompt, response, use_format_reward='no'):
    format_matches = re.findall(zero_format_pattern, response)
    matches = re.findall(zero_pattern, response)
    is_valid = False
    final_reward = 0.0
    if len(format_matches) > 0:
        is_valid = False
        final_reward = 0.0
    else:
        if use_format_reward == 'yes':
            final_reward = -3.0
        else:
            final_reward = 0.0
    if len(format_matches) > 0:
        if len(matches) > 0:
            is_valid = True
            final_reward = 0.0
        else:
            if use_format_reward == 'yes':
                final_reward = -2.0
            else:
                final_reward = 0.0
    if is_valid:
        for tag in ['</think>', '<answer>', '</answer>']:
            count = response.count(tag)
            if count != 1:
                is_valid = False
                if use_format_reward == 'yes':
                    final_reward = -1.0
                else:
                    final_reward = 0.0
                break
    if is_valid:
        left_str = ''.join(response.split('</answer>')[1:])
        left_str_len = len(tokenizer(left_str)['input_ids'])
        if left_str_len > 5:
            is_valid = False
            if use_format_reward == 'yes':
                final_reward = -0.5
            else:
                final_reward = 0.0
    return is_valid, final_reward

@lru_cache
def zero_tir_template_format_score_fn(prompt, response, use_format_reward='no'):
    format_matches = re.findall(zero_format_pattern, response)
    matches = re.findall(zero_pattern, response)
    is_valid = False
    final_reward = 0.0
    if len(format_matches) > 0:
        is_valid = False
        final_reward = 0.0
    else:
        if use_format_reward == 'yes':
            final_reward = -3.0
        else:
            final_reward = 0.0
    if len(format_matches) > 0:
        if len(matches) > 0:
            is_valid = True
            final_reward = 0.0
        else:
            if use_format_reward == 'yes':
                final_reward = -2.0
            else:
                final_reward = 0.0
    if is_valid:
        for tag in ['</think>', '<answer>', '</answer>']:
            if USE_TIR_FORMAT_LOOSE > 0:
                if tag in ['</think>']:
                    continue
            count = response.count(tag)
            if count != 1:
                is_valid = False
                if use_format_reward == 'yes':
                    final_reward = -1.0
                else:
                    final_reward = 0.0
                break
    if is_valid:
        left_str = ''.join(response.split('</answer>')[1:])
        left_str_len = len(tokenizer(left_str)['input_ids'])
        if left_str_len > 5:
            is_valid = False
            if use_format_reward == 'yes':
                final_reward = 0.5
            else:
                final_reward = 0.0
    # if is_valid:
    #     code_output_matches = re.findall(code_output_pattern, response)
    #     code_matches = re.findall(code_pattern, response)
    #     if len(code_matches) >= 1:
    #         if len(code_output_matches) != len(code_matches):
    #             final_reward -= 0.2
                # is_valid = False
                # if use_format_reward == 'yes':
                #     final_reward -= 0.5
    return is_valid, final_reward


@lru_cache
def think_answer_pattern_score(prompt, response, use_format_reward='no'):
    format_matches = re.findall(zero_format_pattern, response)
    is_valid = False
    if len(format_matches) > 0:
        is_valid = True
        final_reward = 0.0
    else:
        if use_format_reward == 'yes':
            final_reward = -2.0
        else:
            final_reward = 0.0
    if is_valid:
        for tag in ['</think>', '<answer>', '</answer>']:
            count = response.count(tag)
            if count != 1:
                is_valid = False
                if use_format_reward == 'yes':
                    final_reward = -1.0
                else:
                    final_reward = 0.0
                break
    if is_valid:
        left_str = ''.join(response.split('</answer>')[1:])
        left_str_len = len(tokenizer(left_str)['input_ids'])
        if left_str_len > 5:
            is_valid = False
            if use_format_reward == 'yes':
                final_reward = -0.5
            else:
                final_reward = 0.0
    return is_valid, final_reward

def code_pattern_score(response):
    code_output_matches = re.findall(code_output_pattern, response)
    code_matches = re.findall(code_pattern, response)
    score = 0.0
    if len(code_matches) >= 1:
        if len(code_output_matches) != len(code_matches):
            score = -0.1
    return score

    
