



import re
from typing import Dict, Tuple, Optional

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
from typing import Dict, Tuple, Optional

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys, os
sys.path.append(os.getenv('OPENRLHF_PATH', '/cpfs/user/chenhao/debug/OpenRLHF_082'))

from env.logic.base import extract_solution, validate_response_structure
from functools import lru_cache

@lru_cache
def extract_answer(pred_str, data_name, use_last_number=True):
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
    
@lru_cache
def validate_response_structure_local(answer_text, 
                                      processed_str,
                                      tokenizer):
    format_correct = validate_response_structure(processed_str, tokenizer)
    if '\\boxed{' not in answer_text:
        format_correct = False
    return format_correct

@lru_cache
def compute_score(solution_str, 
                 ground_truth,
                 format_reward=1,
                 answer_reward=1.0,
                 tokenizer=None) : 
    
    answer_text, processed_str = extract_solution(solution_str)
    format_correct = validate_response_structure_local(
        answer_text, 
        processed_str,
        tokenizer)
    
    format_score = format_reward if format_correct else -abs(format_reward)
    
    gold_score = 0.0
    answer_score = 0.0
    
    if format_correct:
        boxed_answer_text = extract_answer(answer_text, 
                                           'math', False)
        boxed_answer_text_tmp = boxed_answer_text.lower().replace(' ', '')
        ground_truth_tmp = ground_truth.lower().replace(' ', '')
        if boxed_answer_text_tmp == ground_truth_tmp:
            answer_score = 1.0
            gold_score = 1.0
        else:
            answer_score = 0.0
    else:
        answer_score = 0.0
        
    total_score = format_score + answer_score
    
    return total_score, gold_score, answer_score
    
