


from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from timeout_decorator import timeout
import re

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

import re
pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
def answer_extraction_v2(response):
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        boxed_answer = matches[-1]
        pred_ans = extract_answer(boxed_answer, 'math', False)
        return pred_ans
    return ''

@timeout(10, use_signals=False)
def my_verify(gold, pred):
    return float(verify(gold, pred))

from functools import lru_cache
@lru_cache
def infer_rm(prompt, response, gold_ans, is_debug=True, template=''):
    if isinstance(prompt, str):
        prompt = [prompt]
        response = [response]
        gold_ans = [gold_ans]
    
    timeout_cnt = 0
    rewards = []
    for (prompt_, response_, 
         gold_ans_) in zip(prompt, response, gold_ans):
        gold_parsed = parse('\\boxed{'+gold_ans_+'}', 
        extraction_mode="first_match", 
        extraction_config=[LatexExtractionConfig()])
        gold_ans_ = extract_answer('\\boxed{'+gold_ans_+'}', 
                                   'math', 
                                   use_last_number=False)
        if template in ['ZERO_V2', 'ZERO', 'ZERO_TIR']:
            pred_ans = answer_extraction_v2(response_)
        else:
            pred_ans = extract_answer(response_, 'math',
                        use_last_number=False)
        if not pred_ans:
            tmp = {
                'prompt': prompt_,
                'response': response_,
                'gold_ans': gold_ans_,
                'pred_ans': 'none',
                'score': NO_BOXED_SCORE
            }
        else:
            pred_parsed = parse(
                response_,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            pred_ans = extract_answer(response_, 
                                      'math', 
                                use_last_number=False)
            if len(gold_parsed) != 0 and len(pred_parsed) != 0:
                tmp = {
                    'prompt': prompt_,
                    'response': response_,
                    'gold_ans': gold_ans_,
                    'pred_ans': pred_ans
                }
                try:
                    score = my_verify(gold_parsed, 
                                         pred_parsed)
                except Exception as e:
                    score = MATH_FAIL_SCORE
                    logger.info({
                        'INFO': 'RULEEVALFAIL',
                        'INPUT': json.dumps(tmp),
                        'e': e,
                        'type': type(e)
                    })
                tmp['score'] = score
            else:
                tmp = {
                    'prompt': prompt_,
                    'response': response_,
                    'score': -1.0,
                    'pred_ans': pred_ans,
                    'gold_ans': gold_ans_
                }
        rewards.append(float(tmp['score']))
    if is_debug:
        show = {
            'score': float(tmp['score']),
            'pred_ans': tmp['pred_ans'],
            'gold_ans': tmp['gold_ans'],
            'prompt': tmp['prompt'],
            'response': tmp['response'],
            'info': '###rule_score###'
        }
        logger.info(json.dumps(show, ensure_ascii=False))
    return rewards